"""Shared Streamlit implementation for the five 90-item GT forms."""

from __future__ import annotations

import datetime
import html
import json
import os
import random
import threading
import time

import gspread
import pandas as pd
import streamlit as st
from google.oauth2.service_account import Credentials


APP_DIR = os.path.dirname(os.path.abspath(__file__))
FORM_SIZE = 90
EXAMPLE_IMAGES_DIR = os.path.join(
    APP_DIR, "trajectory_gt_450_example_images", "original_0093"
)
GSHEETS_SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

FORM_ID = None
TRAJECTORIES_CSV = None
IMAGES_BASE_DIR = None
SHEET_RESPONSE_METADATA = None
SHEET_RESPONSE_SIMPLE = None

DEFAULT_SESSION_STATE = {
    "page": "instructions",
    "trials": [],
    "current_trial_index": 0,
    "user_id": "",
    "results": [],
    "start_time": None,
    "completed_trajectory_ids": set(),
    "resumed_count": 0,
    "last_save_warning": "",
}


def configure(form_id):
    global FORM_ID, TRAJECTORIES_CSV, IMAGES_BASE_DIR
    global SHEET_RESPONSE_METADATA, SHEET_RESPONSE_SIMPLE
    if form_id not in range(1, 6):
        raise ValueError("form_id must be from 1 to 5")
    FORM_ID = form_id
    TRAJECTORIES_CSV = os.path.join(
        APP_DIR, f"trajectory_gt_450_form_{form_id}.csv"
    )
    IMAGES_BASE_DIR = os.path.join(
        APP_DIR, f"trajectory_gt_450_form_{form_id}_images"
    )
    SHEET_RESPONSE_METADATA = f"traj450_f{form_id}_metadata"
    SHEET_RESPONSE_SIMPLE = f"traj450_f{form_id}_responses"


def initialize_state():
    for key, value in DEFAULT_SESSION_STATE.items():
        if key not in st.session_state:
            st.session_state[key] = (
                value.copy() if isinstance(value, (list, set, dict)) else value
            )


def safe_user_id(user_id):
    cleaned = "".join(
        character
        for character in user_id
        if character.isalnum() or character in "._-"
    )
    return cleaned[:48] or "participant"


def build_gspread_creds():
    secrets = dict(st.secrets["connections"]["gsheets"])
    spreadsheet_url = secrets.pop("spreadsheet")
    credentials = Credentials.from_service_account_info(
        secrets, scopes=GSHEETS_SCOPES
    )
    return credentials, spreadsheet_url


def background_append_row(credentials, spreadsheet_url, worksheet_name, data):
    try:
        client = gspread.authorize(credentials)
        spreadsheet = client.open_by_url(spreadsheet_url)
        try:
            worksheet = spreadsheet.worksheet(worksheet_name)
            headers = worksheet.row_values(1)
            missing = [key for key in data if key not in headers]
            if missing:
                required_columns = len(headers) + len(missing)
                if required_columns > worksheet.col_count:
                    worksheet.add_cols(required_columns - worksheet.col_count)
                for header in missing:
                    headers.append(header)
                    worksheet.update_cell(1, len(headers), header)
        except gspread.exceptions.WorksheetNotFound:
            worksheet = spreadsheet.add_worksheet(
                title=worksheet_name, rows=1000, cols=max(len(data), 1)
            )
            headers = list(data)
            worksheet.append_row(headers, value_input_option="RAW")
        worksheet.append_row(
            [str(data.get(header, "")) for header in headers],
            value_input_option="RAW",
        )
    except Exception as exc:
        print(f"[Form {FORM_ID} background save error] {exc}")


def background_write_result(credentials, spreadsheet_url, save_data):
    safe_uid = save_data["safe_uid"]
    background_append_row(
        credentials,
        spreadsheet_url,
        f"{SHEET_RESPONSE_METADATA}_{safe_uid}",
        save_data["metadata"],
    )
    background_append_row(
        credentials,
        spreadsheet_url,
        f"{SHEET_RESPONSE_SIMPLE}_{safe_uid}",
        save_data["simple"],
    )


def fetch_completed(credentials, spreadsheet_url, safe_uid):
    try:
        client = gspread.authorize(credentials)
        spreadsheet = client.open_by_url(spreadsheet_url)
        try:
            worksheet = spreadsheet.worksheet(
                f"{SHEET_RESPONSE_SIMPLE}_{safe_uid}"
            )
        except gspread.exceptions.WorksheetNotFound:
            return set()
        return {
            str(record.get("trajectory_id", "")).strip()
            for record in worksheet.get_all_records()
            if str(record.get("trajectory_id", "")).strip()
        }
    except Exception as exc:
        print(f"[Form {FORM_ID} progress fetch error] {exc}")
        return set()


def parse_json_list(value):
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return []
    return parsed if isinstance(parsed, list) else []


def clean_cell(row, column):
    value = row.get(column, "")
    return "" if pd.isna(value) else str(value).strip()


def resolve_frame_image_path(trajectory_id, frame_number):
    directory = os.path.join(IMAGES_BASE_DIR, trajectory_id)
    for extension in (".png", ".jpg", ".jpeg"):
        for filename in (
            f"frame_{frame_number:04d}{extension}",
            f"frame_{frame_number:06d}{extension}",
            f"frame_{frame_number}{extension}",
        ):
            path = os.path.join(directory, filename)
            if os.path.isfile(path):
                return path
    return os.path.join(directory, f"frame_{frame_number:04d}.png")


def load_trajectories(seed):
    if not os.path.isfile(TRAJECTORIES_CSV):
        st.error(f"Form CSV not found: {TRAJECTORIES_CSV}")
        return []
    try:
        data = pd.read_csv(TRAJECTORIES_CSV, dtype=str, keep_default_na=False)
    except Exception as exc:
        st.error(f"Failed to read form data: {exc}")
        return []

    required = {
        "trajectory_id",
        "form_id",
        "question",
        "upper_frames",
        "lower_frames",
        "excluded_frames",
        "reference_frame_gt",
    }
    missing = required.difference(data.columns)
    if missing:
        st.error("CSV is missing: " + ", ".join(sorted(missing)))
        return []
    if len(data) != FORM_SIZE or set(data["form_id"]) != {str(FORM_ID)}:
        st.error(f"Form {FORM_ID} must contain exactly {FORM_SIZE} assigned rows.")
        return []

    trials = []
    for row in data.to_dict("records"):
        trajectory_id = clean_cell(row, "trajectory_id")
        upper_frames = [int(value) for value in parse_json_list(row["upper_frames"])]
        lower_frames = [int(value) for value in parse_json_list(row["lower_frames"])]
        trials.append(
            {
                "trajectory_id": trajectory_id,
                "form_id": FORM_ID,
                "round": clean_cell(row, "round"),
                "source": clean_cell(row, "source"),
                "source_datapoint_id": clean_cell(row, "source_datapoint_id"),
                "source_video_path": clean_cell(row, "source_video_path"),
                "question": clean_cell(row, "question"),
                "frames": parse_json_list(row["frames"]),
                "center_frames": parse_json_list(row["center_frames"]),
                "excluded_frames": parse_json_list(row["excluded_frames"]),
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
                "reference_frame_gt": parse_json_list(row["reference_frame_gt"]),
                "reference_upper_gt": parse_json_list(row["reference_upper_gt"]),
                "reference_lower_gt": parse_json_list(row["reference_lower_gt"]),
                "reference_upper_all_same": clean_cell(
                    row, "reference_upper_all_same"
                ),
                "reference_lower_all_same": clean_cell(
                    row, "reference_lower_all_same"
                ),
                "reference_gt_answer": clean_cell(row, "reference_gt_answer"),
                "category": clean_cell(row, "primary_category"),
            }
        )
    random.Random(seed).shuffle(trials)
    return trials


def scroll_to_top(label=""):
    st.iframe(
        f"""
        <!-- {label} -->
        <script>
            function topNow() {{
                const container = window.parent.document.querySelector(
                    '[data-testid="stAppViewContainer"]'
                );
                if (container) container.scrollTo({{top: 0, behavior: 'instant'}});
                window.parent.scrollTo(0, 0);
            }}
            topNow(); setTimeout(topNow, 100); setTimeout(topNow, 300);
        </script>
        """,
        height=1,
    )


def render_example_row(row_name, direction, frames, all_same, ground_truth):
    st.markdown(
        f'<div class="row-heading"><b>{row_name} example</b> — {direction}</div>',
        unsafe_allow_html=True,
    )
    columns = st.columns(len(frames), gap="small")
    for position, (column, frame) in enumerate(zip(columns, frames), start=1):
        image_path = resolve_frame_image_path("original_0093", frame)
        # The example has a separate shared asset directory rather than a form
        # directory, so resolve it explicitly.
        image_path = os.path.join(EXAMPLE_IMAGES_DIR, os.path.basename(image_path))
        with column:
            if os.path.isfile(image_path):
                st.image(image_path, width="stretch")
            else:
                st.warning("Example image missing")
            st.caption(f"Image {position}")
    st.success(
        f"Correct annotation: All images have the same GT = **{all_same}** · "
        f"Row GT = **{ground_truth}**"
    )


def worked_example():
    st.markdown("---")
    st.subheader("Worked example")
    st.markdown(
        "Question: **Are there 2 people?**  "
        "  \nRead each row from left to right and judge the visible evidence "
        "in every image."
    )
    render_example_row(
        "Upper",
        "read left to right →",
        [538, 159, 165, 170],
        "Yes",
        "Yes",
    )
    render_example_row(
        "Lower",
        "read left to right →",
        [651, 864, 892, 962],
        "Yes",
        "No",
    )
    st.markdown(
        "**What this shows:** all images within each row agree, but the shared "
        "GT changes from **Yes** in the upper row to **No** in the lower row. "
        "Evaluate the two rows independently. If even one image in a row has a "
        "different GT, answer **No** to the consistency question and explain "
        "the change in the comments box."
    )
    st.markdown("---")


def prepare_result(trial, upper_same, upper_gt, lower_same, lower_gt, comment):
    reaction_time = time.time() - st.session_state.start_time
    metadata = {
        "user_id": st.session_state.user_id,
        "form_id": FORM_ID,
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
        "reference_frame_gt": json.dumps(trial["reference_frame_gt"]),
        "reference_upper_gt": json.dumps(trial["reference_upper_gt"]),
        "reference_lower_gt": json.dumps(trial["reference_lower_gt"]),
        "reference_upper_all_same": trial["reference_upper_all_same"],
        "reference_lower_all_same": trial["reference_lower_all_same"],
        "reference_gt_answer": trial["reference_gt_answer"],
        "upper_all_same_gt": upper_same or "",
        "upper_gt": upper_gt or "",
        "lower_all_same_gt": lower_same or "",
        "lower_gt": lower_gt or "",
        "comment": comment,
        "reaction_time_ms": int(reaction_time * 1000),
        "timestamp": datetime.datetime.now().isoformat(),
    }
    simple = {
        "user_id": st.session_state.user_id,
        "form_id": FORM_ID,
        "trajectory_id": trial["trajectory_id"],
        "upper_all_same_gt": upper_same or "",
        "upper_gt": upper_gt or "",
        "lower_all_same_gt": lower_same or "",
        "lower_gt": lower_gt or "",
        "comment": comment,
        "reaction_time_ms": metadata["reaction_time_ms"],
    }
    st.session_state.results.append(metadata)
    return {
        "safe_uid": safe_user_id(st.session_state.user_id),
        "metadata": metadata,
        "simple": simple,
    }


def record_response(trial, upper_same, upper_gt, lower_same, lower_gt, comment):
    save_data = prepare_result(
        trial, upper_same, upper_gt, lower_same, lower_gt, comment
    )
    try:
        credentials, spreadsheet_url = build_gspread_creds()
        threading.Thread(
            target=background_write_result,
            args=(credentials, spreadsheet_url, save_data),
            daemon=True,
        ).start()
    except Exception as exc:
        st.session_state.last_save_warning = str(exc)
    st.session_state.completed_trajectory_ids.add(trial["trajectory_id"])
    st.session_state.current_trial_index += 1
    st.session_state.start_time = None
    st.session_state.page = "countdown"
    st.rerun()


def instructions_page():
    scroll_to_top("instructions")
    st.title(f"Trajectory Ground-Truth Verification · Form {FORM_ID}")
    st.markdown(
        f"""
        This form contains **{FORM_SIZE} trajectory pages**. Each page shows one
        yes/no question and two ordered rows of video frames.

        1. Read both the **upper row** and **lower row** from left to right.
        2. For each available row, answer whether every image has the same
           ground truth, then select the row's ground truth as **Yes** or **No**.
        3. Use the optional comment box for GT changes or data issues.
        4. Responses are saved after every page. Use the same Participant ID to resume.
        """
    )
    worked_example()
    user_id = st.text_input("Participant ID", value=st.session_state.user_id)
    if not st.button("Start / resume Form " + str(FORM_ID), type="primary"):
        return
    if not user_id.strip():
        st.error("Enter a Participant ID before starting.")
        return

    st.session_state.user_id = user_id.strip()
    st.session_state.trials = load_trajectories(
        seed=f"trajectory450|form{FORM_ID}|{user_id.strip()}"
    )
    if len(st.session_state.trials) != FORM_SIZE:
        return

    with st.spinner("Checking for saved progress…"):
        try:
            credentials, spreadsheet_url = build_gspread_creds()
            completed = fetch_completed(
                credentials, spreadsheet_url, safe_user_id(user_id.strip())
            )
        except Exception as exc:
            st.warning(f"Could not check saved progress; starting locally: {exc}")
            completed = set()
    valid_ids = {trial["trajectory_id"] for trial in st.session_state.trials}
    completed = completed.intersection(valid_ids)
    st.session_state.completed_trajectory_ids = completed
    st.session_state.resumed_count = len(completed)
    st.session_state.current_trial_index = 0
    st.session_state.results = []
    st.session_state.start_time = None
    st.session_state.page = "done" if len(completed) == FORM_SIZE else "countdown"
    st.rerun()


def countdown_page():
    placeholder = st.empty()
    for number in (3, 2, 1):
        with placeholder.container():
            st.markdown(
                f'<div class="countdown"><span>{number}</span></div>',
                unsafe_allow_html=True,
            )
        time.sleep(1)
    st.session_state.page = "experiment"
    st.rerun()


def render_image_row(trial, row_name, direction, frames, image_paths):
    st.markdown(
        f'<div class="row-heading"><b>{row_name} row</b> — {direction}</div>',
        unsafe_allow_html=True,
    )
    if not frames:
        st.info(f"No {row_name.lower()}-row images are available for this page.")
        return None, None, False

    columns = st.columns(len(frames), gap="small")
    for position, (column, frame, image_path) in enumerate(
        zip(columns, frames, image_paths), start=1
    ):
        with column:
            if os.path.isfile(image_path):
                st.image(image_path, width="stretch")
            else:
                st.warning("Missing image")
            st.caption(f"Image {position} · frame {frame}")

    prefix = f"{trial['trajectory_id']}_{row_name.lower()}"
    same_column, gt_column = st.columns([3, 2])
    with same_column:
        all_same = st.radio(
            f"Do all {row_name.lower()} row images have the same GT?",
            options=("Yes", "No"),
            index=None,
            horizontal=True,
            key=f"{prefix}_same",
        )
    with gt_column:
        ground_truth = st.radio(
            f"What is the GT for the {row_name.lower()} row?",
            options=("Yes", "No"),
            index=None,
            horizontal=True,
            key=f"{prefix}_gt",
        )
    return (
        all_same.lower() if all_same else None,
        ground_truth.lower() if ground_truth else None,
        True,
    )


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
    scroll_to_top(trial["trajectory_id"])
    st.markdown(
        f"<div class='page-counter'><b>Form {FORM_ID}</b> · "
        f"Trajectory {len(completed) + 1} of {total} · "
        f"ID {html.escape(trial['trajectory_id'])}</div>",
        unsafe_allow_html=True,
    )
    st.progress(min(len(completed) / total, 1.0))
    if st.session_state.resumed_count:
        st.info(
            f"Resumed session — {st.session_state.resumed_count} pages already completed."
        )
        st.session_state.resumed_count = 0
    st.markdown(
        f'<h4 class="question-heading">Question: '
        f'{html.escape(trial["question"])}</h4>',
        unsafe_allow_html=True,
    )
    if st.session_state.start_time is None:
        st.session_state.start_time = time.time()

    upper_same, upper_gt, upper_available = render_image_row(
        trial,
        "Upper",
        "read left to right →",
        trial["upper_frames"],
        trial["upper_image_paths"],
    )
    st.markdown('<hr class="row-divider">', unsafe_allow_html=True)
    lower_same, lower_gt, lower_available = render_image_row(
        trial,
        "Lower",
        "read left to right →",
        trial["lower_frames"],
        trial["lower_image_paths"],
    )
    st.markdown('<hr class="row-divider">', unsafe_allow_html=True)
    comment = st.text_area(
        "Comments (optional)",
        key=f"{trial['trajectory_id']}_comment",
        height=80,
        placeholder="Note any GT change or data issue.",
    )
    _, submit_column, _ = st.columns([2, 3, 2])
    with submit_column:
        submitted = st.button(
            "Submit & next →",
            type="primary",
            width="stretch",
            key=f"submit_{trial['trajectory_id']}",
        )
    if submitted:
        missing = []
        if upper_available and upper_same is None:
            missing.append("upper-row consistency")
        if upper_available and upper_gt is None:
            missing.append("upper-row GT")
        if lower_available and lower_same is None:
            missing.append("lower-row consistency")
        if lower_available and lower_gt is None:
            missing.append("lower-row GT")
        if missing:
            st.error("Please select: " + ", ".join(missing) + ".")
        else:
            record_response(
                trial,
                upper_same,
                upper_gt,
                lower_same,
                lower_gt,
                comment.strip(),
            )


def done_page():
    scroll_to_top("done")
    st.title(f"Form {FORM_ID} complete")
    st.success("Thank you for completing the ground-truth verification form.")
    if st.session_state.last_save_warning:
        st.warning(
            "At least one response could not be sent to Google Sheets. "
            "Download this session's CSV and send it to the researcher."
        )
    if st.session_state.results:
        results = pd.DataFrame(st.session_state.results)
        st.download_button(
            "Download this session's responses",
            data=results.to_csv(index=False),
            file_name=(
                f"trajectory_gt_450_form{FORM_ID}_"
                f"{safe_user_id(st.session_state.user_id)}.csv"
            ),
            mime="text/csv",
        )
    if st.button("Return to start"):
        for key, value in DEFAULT_SESSION_STATE.items():
            st.session_state[key] = (
                value.copy() if isinstance(value, (list, set, dict)) else value
            )
        st.rerun()


def main(form_id):
    configure(form_id)
    st.set_page_config(
        page_title=f"Trajectory GT Verification · Form {form_id}", layout="wide"
    )
    initialize_state()
    st.markdown(
        """
        <style>
            .block-container { padding-top: .5rem; max-width: 1600px; }
            header, #MainMenu, footer { visibility: hidden; height: 0; }
            .page-counter { margin-top: -8px; font-size: 14px; }
            .question-heading { text-align: center; color: #2c3e50;
                font-size: 19px; margin: 8px 0 10px; }
            .row-heading { font-size: 14px; margin: 3px 0 4px; color: #2c3e50; }
            .row-divider { border: none; border-top: 1px solid #bdc3c7;
                margin: 7px 0; }
            div[data-testid="stImage"] img { max-height: 185px; object-fit: contain; }
            div[data-testid="stCaptionContainer"] { text-align: center;
                font-size: 10px; }
            .countdown { position: fixed; inset: 0; background: white;
                z-index: 999999; display: flex; align-items: center;
                justify-content: center; }
            .countdown span { font-size: 140px; font-weight: bold; color: #4A90D9; }
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
    else:
        st.session_state.page = "instructions"
        st.rerun()
