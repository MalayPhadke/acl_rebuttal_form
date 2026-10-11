import datetime
import json
import os
import random
import threading
import time

import gspread
import pandas as pd
import streamlit as st
from google.oauth2.service_account import Credentials


# --- Configuration ---
def _find_file(filename, start_dir=None):
    """Find a deployment asset near this app, allowing a few parent levels."""
    if start_dir is None:
        start_dir = os.path.dirname(os.path.abspath(__file__))
    candidate = start_dir
    for _ in range(4):
        path = os.path.join(candidate, filename)
        if os.path.isfile(path):
            return path
        candidate = os.path.dirname(candidate)
    return os.path.join(start_dir, filename)


def _find_dir(dirname, start_dir=None):
    """Find a deployment directory near this app."""
    if start_dir is None:
        start_dir = os.path.dirname(os.path.abspath(__file__))
    candidate = start_dir
    for _ in range(4):
        path = os.path.join(candidate, dirname)
        if os.path.isdir(path):
            return path
        candidate = os.path.dirname(candidate)
    return os.path.join(start_dir, dirname)


TRAJECTORIES_CSV = _find_file("trajectory_gt_488.csv")
IMAGES_BASE_DIR = _find_dir("trajectory_gt_488_images")
TOTAL_TRAJECTORIES = 488

SHEET_RESPONSE_METADATA = "trajectory_gt_metadata"
SHEET_RESPONSE_SIMPLE = "trajectory_gt_responses"

GSHEETS_SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]


# --- Session State ---
DEFAULT_SESSION_STATE = {
    "page": "instructions",
    "trials": [],
    "current_trial_index": 0,
    "user_id": "",
    "results": [],
    "start_time": None,
    "completed_trajectory_ids": set(),
    "resumed_count": 0,
}

for key, value in DEFAULT_SESSION_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value.copy() if isinstance(value, (list, set, dict)) else value


# --- Google Sheets ---
def _build_gspread_creds():
    """Build credentials from Streamlit secrets on the main thread."""
    secrets = dict(st.secrets["connections"]["gsheets"])
    spreadsheet_url = secrets.pop("spreadsheet")
    creds = Credentials.from_service_account_info(secrets, scopes=GSHEETS_SCOPES)
    return creds, spreadsheet_url


def _bg_append_row(creds, spreadsheet_url, worksheet_name, data_dict):
    """Append one result row without calling Streamlit from the worker thread."""
    try:
        client = gspread.authorize(creds)
        spreadsheet = client.open_by_url(spreadsheet_url)
        try:
            worksheet = spreadsheet.worksheet(worksheet_name)
        except gspread.exceptions.WorksheetNotFound:
            worksheet = spreadsheet.add_worksheet(
                title=worksheet_name,
                rows=1000,
                cols=max(len(data_dict), 1),
            )
            worksheet.append_row(list(data_dict.keys()), value_input_option="RAW")
        worksheet.append_row(
            [str(value) for value in data_dict.values()],
            value_input_option="RAW",
        )
    except Exception as exc:
        print(f"[Background save error] {worksheet_name}: {exc}")


def _bg_write_result(creds, spreadsheet_url, save_data):
    safe_uid = save_data["safe_uid"]
    _bg_append_row(
        creds,
        spreadsheet_url,
        f"{SHEET_RESPONSE_METADATA}_{safe_uid}",
        save_data["metadata"],
    )
    _bg_append_row(
        creds,
        spreadsheet_url,
        f"{SHEET_RESPONSE_SIMPLE}_{safe_uid}",
        save_data["simple"],
    )


def _fetch_completed_trajectories(creds, spreadsheet_url, safe_uid):
    """Return trajectory IDs already submitted by this participant."""
    try:
        client = gspread.authorize(creds)
        spreadsheet = client.open_by_url(spreadsheet_url)
        worksheet_name = f"{SHEET_RESPONSE_SIMPLE}_{safe_uid}"
        try:
            worksheet = spreadsheet.worksheet(worksheet_name)
        except gspread.exceptions.WorksheetNotFound:
            return set()
        return {
            str(record.get("trajectory_id", "")).strip()
            for record in worksheet.get_all_records()
            if str(record.get("trajectory_id", "")).strip()
        }
    except Exception as exc:
        print(f"[Progress fetch error] {exc}")
        return set()


# --- Dataset Loading ---
def parse_frame_list(value):
    """Parse a JSON frame-number list from the packaged trajectory CSV."""
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return []
    if not isinstance(parsed, list):
        return []
    return [int(frame) for frame in parsed]


def clean_cell(row, column):
    value = row.get(column, "")
    return "" if pd.isna(value) else str(value).strip()


def resolve_frame_image_path(trajectory_id, frame_number):
    """Resolve mixed four/six-digit PNG/JPEG frame naming conventions."""
    trajectory_dir = os.path.join(IMAGES_BASE_DIR, trajectory_id)
    for extension in (".jpg", ".jpeg", ".png"):
        for filename in (
            f"frame_{frame_number:06d}{extension}",
            f"frame_{frame_number:04d}{extension}",
            f"frame_{frame_number}{extension}",
        ):
            path = os.path.join(trajectory_dir, filename)
            if os.path.isfile(path):
                return path
    return os.path.join(trajectory_dir, f"frame_{frame_number:04d}.png")


def load_trajectories(seed=None):
    """Load all 488 trajectories and deterministically shuffle their pages."""
    if not os.path.isfile(TRAJECTORIES_CSV):
        st.error(f"Trajectory CSV not found: {TRAJECTORIES_CSV}")
        return []

    try:
        data = pd.read_csv(TRAJECTORIES_CSV)
    except Exception as exc:
        st.error(f"Failed to read {TRAJECTORIES_CSV}: {exc}")
        return []

    required = {
        "trajectory_id",
        "question",
        "upper_frames",
        "lower_frames",
        "excluded_frames",
    }
    missing = required.difference(data.columns)
    if missing:
        st.error("CSV is missing required columns: " + ", ".join(sorted(missing)))
        return []

    trajectories = []
    for _, row in data.iterrows():
        trajectory_id = clean_cell(row, "trajectory_id")
        upper_frames = parse_frame_list(row["upper_frames"])
        lower_frames = parse_frame_list(row["lower_frames"])
        trajectories.append(
            {
                "trajectory_id": trajectory_id,
                "round": clean_cell(row, "round"),
                "source": clean_cell(row, "source"),
                "source_datapoint_id": clean_cell(row, "source_datapoint_id"),
                "source_video_path": clean_cell(row, "source_video_path"),
                "question": clean_cell(row, "question"),
                "frames": parse_frame_list(row["frames"]),
                "center_frames": parse_frame_list(row["center_frames"]),
                "excluded_frames": parse_frame_list(row["excluded_frames"]),
                "upper_frames": upper_frames,
                "lower_frames": lower_frames,
                "upper_image_paths": [
                    resolve_frame_image_path(trajectory_id, frame)
                    for frame in upper_frames
                ],
                "lower_image_paths": [
                    resolve_frame_image_path(trajectory_id, frame)
                    for frame in lower_frames
                ],
                "reference_gt_answer": clean_cell(row, "gt_answer"),
                "category": clean_cell(row, "primary_category"),
            }
        )

    rng = random.Random(seed)
    rng.shuffle(trajectories)
    return trajectories


# --- Navigation and Saving ---
def scroll_to_top(label=""):
    st.components.v1.html(
        f"""
        <!-- {label} -->
        <script>
            function scrollToTop() {{
                var container = window.parent.document.querySelector('[data-testid="stAppViewContainer"]');
                if (container) container.scrollTo({{top: 0, behavior: 'instant'}});
                var main = window.parent.document.querySelector('.main');
                if (main) main.scrollTop = 0;
                window.parent.document.documentElement.scrollTop = 0;
                window.parent.document.body.scrollTop = 0;
                window.parent.scrollTo(0, 0);
            }}
            scrollToTop();
            setTimeout(scrollToTop, 100);
            setTimeout(scrollToTop, 300);
            setTimeout(scrollToTop, 600);
        </script>
        """,
        height=0,
    )


def prepare_result(trial, upper_same_gt, upper_gt, lower_same_gt, lower_gt, comment):
    reaction_time = time.time() - st.session_state.start_time
    user_id = st.session_state.user_id
    safe_uid = "".join(char for char in user_id if char.isalnum() or char in "._-")

    metadata = {
        "user_id": user_id,
        "trajectory_id": trial["trajectory_id"],
        "round": trial["round"],
        "source": trial["source"],
        "source_datapoint_id": trial["source_datapoint_id"],
        "source_video_path": trial["source_video_path"],
        "question": trial["question"],
        "category": trial["category"],
        "all_frames": json.dumps(trial["frames"]),
        "excluded_max_ambiguity_frames": json.dumps(trial["excluded_frames"]),
        "upper_frames": json.dumps(trial["upper_frames"]),
        "lower_frames": json.dumps(trial["lower_frames"]),
        "reference_gt_answer": trial["reference_gt_answer"],
        "upper_all_same_gt": "" if upper_same_gt is None else upper_same_gt,
        "upper_gt": upper_gt or "",
        "lower_all_same_gt": "" if lower_same_gt is None else lower_same_gt,
        "lower_gt": lower_gt or "",
        "comment": comment,
        "reaction_time_ms": int(reaction_time * 1000),
        "timestamp": datetime.datetime.now().isoformat(),
    }
    simple = {
        "user_id": user_id,
        "trajectory_id": trial["trajectory_id"],
        "upper_all_same_gt": "" if upper_same_gt is None else upper_same_gt,
        "upper_gt": upper_gt or "",
        "lower_all_same_gt": "" if lower_same_gt is None else lower_same_gt,
        "lower_gt": lower_gt or "",
        "comment": comment,
        "reaction_time_ms": metadata["reaction_time_ms"],
    }
    st.session_state.results.append(metadata)
    return {"safe_uid": safe_uid, "metadata": metadata, "simple": simple}


def record_response(trial, upper_same_gt, upper_gt, lower_same_gt, lower_gt, comment):
    save_data = prepare_result(
        trial,
        upper_same_gt,
        upper_gt,
        lower_same_gt,
        lower_gt,
        comment,
    )
    creds, spreadsheet_url = _build_gspread_creds()
    threading.Thread(
        target=_bg_write_result,
        args=(creds, spreadsheet_url, save_data),
        daemon=True,
    ).start()

    st.session_state.completed_trajectory_ids.add(trial["trajectory_id"])
    st.session_state.current_trial_index += 1
    st.session_state.start_time = None
    st.session_state.page = "countdown"
    st.rerun()


# --- Pages ---
def instructions_page():
    scroll_to_top("instructions")
    st.title("Trajectory Ground-Truth Consistency Experiment")
    st.write(
        """
        ### Instructions

        This experiment contains **488 trajectory pages**. Each page shows one
        yes/no question and two ordered rows of video frames.

        1. The **upper row** is the left side of the trajectory: ambiguity increases from left to right.
        2. The **lower row** is the right side of the trajectory: ambiguity decreases from left to right.
        3. The frame or interval with **maximum ambiguity is omitted** and will not be shown.
        4. For each available row:
            - Check **“Do all images in this row have the same GT?”** when they do.
            - Select the row's ground truth as **Yes** or **No**.
        5. Use the optional comments box to explain inconsistent or unclear rows.
        6. Each submitted page is saved automatically. Re-enter the same Participant ID to resume.
        """
    )

    st.markdown("---")
    user_id_input = st.text_input(
        "Enter your Participant ID:",
        value=st.session_state.user_id,
    )
    start_clicked = st.button(
        "▶ Start / Resume Experiment",
        use_container_width=True,
        type="primary",
    )

    if not start_clicked:
        return
    if not user_id_input.strip():
        st.error("Please enter a valid Participant ID to continue.")
        return

    user_id = user_id_input.strip()
    safe_uid = "".join(char for char in user_id if char.isalnum() or char in "._-")
    st.session_state.user_id = user_id
    st.session_state.trials = load_trajectories(seed=user_id)

    if len(st.session_state.trials) != TOTAL_TRAJECTORIES:
        st.error(
            f"Expected {TOTAL_TRAJECTORIES} trajectories but loaded "
            f"{len(st.session_state.trials)}. Check trajectory_gt_488.csv and "
            "trajectory_gt_488_images."
        )
        return

    with st.spinner("Checking for existing progress…"):
        try:
            creds, spreadsheet_url = _build_gspread_creds()
            completed = _fetch_completed_trajectories(
                creds,
                spreadsheet_url,
                safe_uid,
            )
        except Exception as exc:
            st.warning(f"Could not check progress; starting fresh: {exc}")
            completed = set()

    valid_ids = {trial["trajectory_id"] for trial in st.session_state.trials}
    completed = completed.intersection(valid_ids)
    st.session_state.completed_trajectory_ids = completed
    st.session_state.resumed_count = len(completed)
    st.session_state.current_trial_index = 0
    st.session_state.results = []
    st.session_state.start_time = None
    st.session_state.page = "done" if len(completed) >= TOTAL_TRAJECTORIES else "countdown"
    st.rerun()


def countdown_page():
    placeholder = st.empty()
    for number in (3, 2, 1):
        with placeholder.container():
            st.markdown(
                f"""
                <style>
                    [data-testid="stSidebar"], [data-testid="stHeader"], [data-testid="stFooter"] {{
                        display: none !important;
                    }}
                </style>
                <div class="countdown-overlay">
                    <span>{number}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        time.sleep(1)
    st.session_state.page = "experiment"
    st.rerun()


def render_image_row(trial, row_name, direction_text, frames, image_paths):
    """Render one trajectory branch and return its two response values."""
    st.markdown(
        f'<div class="row-heading"><b>{row_name} row</b> — {direction_text}</div>',
        unsafe_allow_html=True,
    )

    if not frames:
        st.info(
            f"No {row_name.lower()}-row images are available because maximum "
            "ambiguity occurs at the trajectory boundary."
        )
        return None, None, False

    columns = st.columns(len(frames), gap="small")
    for position, (column, frame, image_path) in enumerate(
        zip(columns, frames, image_paths),
        start=1,
    ):
        with column:
            if os.path.isfile(image_path):
                st.image(image_path, use_container_width=True)
            else:
                st.warning("Missing image")
            st.caption(f"Image {position} · frame {frame}")

    key_prefix = f"{trial['trajectory_id']}_{row_name.lower()}"
    check_col, gt_col = st.columns([3, 2])
    with check_col:
        all_same_gt = st.checkbox(
            f"Do all {row_name.lower()} row images have the same GT?",
            key=f"{key_prefix}_same_gt",
            help="Check the box for Yes; leave it unchecked for No.",
        )
    with gt_col:
        ground_truth = st.radio(
            f"What is the GT for the {row_name.lower()} row?",
            options=("Yes", "No"),
            index=None,
            horizontal=True,
            key=f"{key_prefix}_gt",
        )
    return all_same_gt, ground_truth.lower() if ground_truth else None, True


def experiment_page():
    trials = st.session_state.trials
    total = len(trials)
    index = st.session_state.current_trial_index
    completed = st.session_state.completed_trajectory_ids

    while index < total and trials[index]["trajectory_id"] in completed:
        index += 1
        st.session_state.current_trial_index = index

    if index >= total:
        st.session_state.page = "done"
        st.rerun()
        return

    trial = trials[index]
    scroll_to_top(f"trajectory_{trial['trajectory_id']}")

    st.markdown(
        f"<div class='page-counter'><b>Trajectory page {index + 1} of {total}</b>"
        f" · ID {trial['trajectory_id']}</div>",
        unsafe_allow_html=True,
    )
    st.progress(index / total)

    if st.session_state.resumed_count > 0:
        st.info(
            f"Resumed session — {st.session_state.resumed_count} of {total} "
            "trajectory pages were already completed."
        )

    st.markdown(
        f'<h4 class="question-heading">Question: {trial["question"]}</h4>',
        unsafe_allow_html=True,
    )

    if st.session_state.start_time is None:
        st.session_state.start_time = time.time()

    upper_same_gt, upper_gt, upper_available = render_image_row(
        trial,
        "Upper",
        "increasing ambiguity →",
        trial["upper_frames"],
        trial["upper_image_paths"],
    )

    st.markdown('<hr class="row-divider">', unsafe_allow_html=True)

    lower_same_gt, lower_gt, lower_available = render_image_row(
        trial,
        "Lower",
        "decreasing ambiguity →",
        trial["lower_frames"],
        trial["lower_image_paths"],
    )

    st.markdown('<hr class="row-divider">', unsafe_allow_html=True)
    comment = st.text_area(
        "Comments (optional):",
        key=f"{trial['trajectory_id']}_comment",
        height=80,
        placeholder="Note any GT change, ambiguity, or issue with this trajectory.",
    )

    _, submit_col, _ = st.columns([2, 3, 2])
    with submit_col:
        submit_clicked = st.button(
            "Submit & Next →",
            use_container_width=True,
            type="primary",
            key=f"submit_{trial['trajectory_id']}",
        )

    if submit_clicked:
        missing_answers = []
        if upper_available and upper_gt is None:
            missing_answers.append("upper-row GT")
        if lower_available and lower_gt is None:
            missing_answers.append("lower-row GT")
        if missing_answers:
            st.error("Please select: " + " and ".join(missing_answers) + ".")
        else:
            record_response(
                trial,
                upper_same_gt,
                upper_gt,
                lower_same_gt,
                lower_gt,
                comment.strip(),
            )


def done_page():
    scroll_to_top("done")
    st.title("Experiment Complete")
    st.success("Thank you for completing all 488 trajectory pages!")

    if st.session_state.results:
        results = pd.DataFrame(st.session_state.results)
        st.write(f"Responses collected in this session: {len(results)}")
        st.download_button(
            label="Download This Session's Results CSV",
            data=results.to_csv(index=False),
            file_name=f"trajectory_gt_results_{st.session_state.user_id}.csv",
            mime="text/csv",
        )

    if st.button("Return to Start"):
        for key, value in DEFAULT_SESSION_STATE.items():
            st.session_state[key] = value.copy() if isinstance(value, (list, set, dict)) else value
        st.rerun()


def main():
    st.set_page_config(
        page_title="Trajectory GT Consistency Experiment",
        layout="wide",
    )
    st.markdown(
        """
        <style>
            .block-container {
                padding-top: 0.5rem;
                padding-bottom: 1rem;
                max-width: 1600px;
            }
            header, #MainMenu, footer {
                visibility: hidden;
                height: 0;
            }
            .page-counter {
                margin-top: -8px;
                font-size: 14px;
            }
            .question-heading {
                text-align: center;
                color: #2c3e50;
                font-size: 19px;
                margin: 8px 0 10px 0;
            }
            .row-heading {
                font-size: 14px;
                margin: 3px 0 4px 0;
                color: #2c3e50;
            }
            .row-divider {
                border: none;
                border-top: 1px solid #bdc3c7;
                margin: 7px 0;
            }
            div[data-testid="stImage"] img {
                max-height: 185px;
                object-fit: contain;
            }
            div[data-testid="stCaptionContainer"] {
                text-align: center;
                font-size: 10px;
            }
            .countdown-overlay {
                position: fixed;
                top: 0;
                left: 0;
                width: 100vw;
                height: 100vh;
                background: white;
                z-index: 999999;
                display: flex;
                justify-content: center;
                align-items: center;
            }
            .countdown-overlay span {
                font-size: 140px;
                font-weight: bold;
                color: #4A90D9;
                margin-top: -10vh;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    page = st.session_state.page
    if page == "instructions":
        instructions_page()
    elif page == "countdown":
        countdown_page()
    elif page == "experiment":
        experiment_page()
    elif page == "done":
        done_page()


if __name__ == "__main__":
    main()
