"""Randomized single-image annotation study for 10 sampled trajectories."""

from __future__ import annotations

import datetime
import hashlib
import html
import os
import random
import threading
import time
from collections import Counter, defaultdict

import gspread
import pandas as pd
import streamlit as st
from google.oauth2.service_account import Credentials


APP_DIR = os.path.dirname(os.path.abspath(__file__))
TRIALS_CSV = os.path.join(APP_DIR, "unordered_annotation_100.csv")
TOTAL_TRIALS = 100
SHEET_RESPONSE_METADATA = "unordered100_metadata"
SHEET_RESPONSE_SIMPLE = "unordered100_responses"
GSHEETS_SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

DEFAULT_STATE = {
    "page": "welcome",
    "trials": [],
    "current_trial_index": 0,
    "user_id": "",
    "results": [],
    "start_time": None,
    "completed_trial_ids": set(),
    "resumed_count": 0,
    "last_save_warning": "",
}


def initialize_state():
    for key, value in DEFAULT_STATE.items():
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
                required = len(headers) + len(missing)
                if required > worksheet.col_count:
                    worksheet.add_cols(required - worksheet.col_count)
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
        print(f"[Unordered-study background save error] {exc}")


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
            str(record.get("trial_uid", "")).strip()
            for record in worksheet.get_all_records()
            if str(record.get("trial_uid", "")).strip()
        }
    except Exception as exc:
        print(f"[Unordered-study progress fetch error] {exc}")
        return set()


def no_adjacent_trajectory_order(trials, seed):
    """Randomize images while forbidding consecutive images from one trajectory."""
    rng = random.Random(seed)
    grouped = defaultdict(list)
    for trial in trials:
        grouped[trial["trajectory_id"]].append(trial)
    for group in grouped.values():
        rng.shuffle(group)

    ordered = []
    last_trajectory = None
    while len(ordered) < len(trials):
        candidates = [
            trajectory_id
            for trajectory_id, group in grouped.items()
            if group and trajectory_id != last_trajectory
        ]
        if not candidates:
            raise RuntimeError("Unable to construct a non-adjacent trajectory order")
        maximum_remaining = max(len(grouped[item]) for item in candidates)
        candidates = [
            item for item in candidates if len(grouped[item]) == maximum_remaining
        ]
        chosen = rng.choice(candidates)
        ordered.append(grouped[chosen].pop())
        last_trajectory = chosen
    return ordered


def load_trials(seed):
    if not os.path.isfile(TRIALS_CSV):
        st.error(f"Trial CSV not found: {TRIALS_CSV}")
        return []
    try:
        data = pd.read_csv(TRIALS_CSV, dtype=str, keep_default_na=False)
    except Exception as exc:
        st.error(f"Failed to load trials: {exc}")
        return []
    required = {
        "trial_uid",
        "trajectory_id",
        "question",
        "frame_number",
        "reference_gt",
        "primary_category",
        "image_path",
    }
    missing = required.difference(data.columns)
    if missing:
        st.error("Trial CSV is missing: " + ", ".join(sorted(missing)))
        return []
    if len(data) != TOTAL_TRIALS:
        st.error(f"Expected {TOTAL_TRIALS} image trials, found {len(data)}.")
        return []
    trajectory_counts = data["trajectory_id"].value_counts().to_dict()
    if len(trajectory_counts) != 10 or set(trajectory_counts.values()) != {10}:
        st.error("The cohort must contain 10 images from each of 10 trajectories.")
        return []

    trials = []
    for row in data.to_dict("records"):
        image_path = row["image_path"]
        if not os.path.isabs(image_path):
            image_path = os.path.join(APP_DIR, image_path)
        row["image_path"] = image_path
        trials.append(row)
    ordered = no_adjacent_trajectory_order(trials, seed)
    if any(
        left["trajectory_id"] == right["trajectory_id"]
        for left, right in zip(ordered, ordered[1:])
    ):
        raise AssertionError("Adjacent trials share a trajectory")
    return ordered


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


def welcome_page():
    scroll_to_top("welcome")
    st.title("Randomized Single-Image Annotation Study")
    st.write(
        "You will answer 100 visual questions. Each trial contains one image, "
        "one Yes/No question, and a certainty score. Images are randomized and "
        "are never presented as trajectories or sequences."
    )
    if st.button("Continue to instructions →", type="primary"):
        st.session_state.page = "instructions"
        st.rerun()


def instructions_page():
    scroll_to_top("instructions")
    st.title("Instructions")
    st.markdown(
        """
        For each trial:

        1. Read the **Yes/No question** and inspect the single image.
        2. Select **Yes** or **No** using only evidence visible in that image.
        3. Select your **certainty from 1 to 10**:
           - **1** = completely uncertain / the image does not support an answer
           - **5** = moderately certain
           - **10** = completely certain
        4. Submit to continue. You cannot return to earlier trials.

        Images from the same underlying video are deliberately separated and
        shown without their trajectory context. Do not use outside tools.
        """
    )
    st.markdown("---")
    user_id = st.text_input("Participant ID", value=st.session_state.user_id)
    if not st.button("Start / resume study", type="primary"):
        if st.button("← Back"):
            st.session_state.page = "welcome"
            st.rerun()
        return
    if not user_id.strip():
        st.error("Enter a Participant ID before starting.")
        return

    user_id = user_id.strip()
    digest = hashlib.sha256(f"unordered100|{user_id}".encode()).hexdigest()
    trials = load_trials(seed=int(digest[:16], 16))
    if len(trials) != TOTAL_TRIALS:
        return
    st.session_state.user_id = user_id
    st.session_state.trials = trials
    st.session_state.results = []
    st.session_state.start_time = None

    with st.spinner("Checking for saved progress…"):
        try:
            credentials, spreadsheet_url = build_gspread_creds()
            completed = fetch_completed(
                credentials, spreadsheet_url, safe_user_id(user_id)
            )
        except Exception as exc:
            st.warning(f"Could not check saved progress; starting locally: {exc}")
            completed = set()
    valid_ids = {trial["trial_uid"] for trial in trials}
    completed = completed.intersection(valid_ids)
    st.session_state.completed_trial_ids = completed
    st.session_state.resumed_count = len(completed)
    st.session_state.current_trial_index = 0
    st.session_state.page = "done" if len(completed) == TOTAL_TRIALS else "experiment"
    st.rerun()


def prepare_result(trial, answer, certainty):
    reaction_time = time.time() - st.session_state.start_time
    answer = answer.lower()
    metadata = {
        "user_id": st.session_state.user_id,
        "trial_uid": trial["trial_uid"],
        "trajectory_id": trial["trajectory_id"],
        "round": trial.get("round", ""),
        "source": trial.get("source", ""),
        "question": trial["question"],
        "frame_number": trial["frame_number"],
        "frame_position": trial.get("frame_position", ""),
        "category": trial["primary_category"],
        "human_answer": answer,
        "certainty": int(certainty),
        "reference_gt": trial["reference_gt"],
        "human_correct": answer == trial["reference_gt"].lower(),
        "image_filename": os.path.basename(trial["image_path"]),
        "reaction_time_ms": int(reaction_time * 1000),
        "timestamp": datetime.datetime.now().isoformat(),
    }
    simple = {
        "user_id": st.session_state.user_id,
        "trial_uid": trial["trial_uid"],
        "human_answer": answer,
        "certainty": int(certainty),
        "reaction_time_ms": metadata["reaction_time_ms"],
    }
    st.session_state.results.append(metadata)
    return {
        "safe_uid": safe_user_id(st.session_state.user_id),
        "metadata": metadata,
        "simple": simple,
    }


def record_response(trial, answer, certainty):
    save_data = prepare_result(trial, answer, certainty)
    try:
        credentials, spreadsheet_url = build_gspread_creds()
        threading.Thread(
            target=background_write_result,
            args=(credentials, spreadsheet_url, save_data),
            daemon=True,
        ).start()
    except Exception as exc:
        st.session_state.last_save_warning = str(exc)
    st.session_state.completed_trial_ids.add(trial["trial_uid"])
    st.session_state.current_trial_index += 1
    st.session_state.start_time = None
    st.rerun()


def experiment_page():
    trials = st.session_state.trials
    total = len(trials)
    index = st.session_state.current_trial_index
    completed = st.session_state.completed_trial_ids
    while index < total and trials[index]["trial_uid"] in completed:
        index += 1
        st.session_state.current_trial_index = index
    if index >= total:
        st.session_state.page = "done"
        st.rerun()
        return

    trial = trials[index]
    scroll_to_top(f"trial_{index}")
    st.caption(f"Question {len(completed) + 1} of {total}")
    st.progress(min(len(completed) / total, 1.0))
    if st.session_state.resumed_count:
        st.info(
            f"Resumed session — {st.session_state.resumed_count} responses already saved."
        )
        st.session_state.resumed_count = 0
    st.markdown(
        f'<div class="question">{html.escape(trial["question"])}</div>',
        unsafe_allow_html=True,
    )
    if not os.path.isfile(trial["image_path"]):
        st.error("This trial's image is missing. Please contact the researcher.")
        return
    _, image_column, _ = st.columns([1, 3, 1])
    with image_column:
        st.image(trial["image_path"], width="stretch")
    if st.session_state.start_time is None:
        st.session_state.start_time = time.time()

    with st.form(f"response_{trial['trial_uid']}"):
        answer = st.radio(
            "Your answer",
            options=("Yes", "No"),
            index=None,
            horizontal=True,
        )
        certainty = st.radio(
            "How certain are you?",
            options=list(range(1, 11)),
            index=None,
            horizontal=True,
            help="1 = completely uncertain; 10 = completely certain",
        )
        st.caption("1 = completely uncertain · 10 = completely certain")
        submitted = st.form_submit_button(
            "Submit & next →", type="primary", width="stretch"
        )
    if submitted:
        if answer is None:
            st.error("Select Yes or No.")
        elif certainty is None:
            st.error("Select a certainty score from 1 to 10.")
        else:
            record_response(trial, answer, certainty)


def done_page():
    scroll_to_top("done")
    st.title("Study complete")
    st.success("Thank you for completing all 100 single-image annotations.")
    if st.session_state.last_save_warning:
        st.warning(
            "At least one response could not be sent to Google Sheets. "
            "Download the CSV and send it to the researcher."
        )
    if st.session_state.results:
        results = pd.DataFrame(st.session_state.results)
        st.download_button(
            "Download this session's responses",
            data=results.to_csv(index=False),
            file_name=(
                f"unordered_annotation_100_{safe_user_id(st.session_state.user_id)}.csv"
            ),
            mime="text/csv",
        )
    if st.button("Return to start"):
        for key, value in DEFAULT_STATE.items():
            st.session_state[key] = (
                value.copy() if isinstance(value, (list, set, dict)) else value
            )
        st.rerun()


def main():
    st.set_page_config(
        page_title="Randomized Single-Image Annotation Study", layout="wide"
    )
    initialize_state()
    st.markdown(
        """
        <style>
            .block-container { max-width: 1100px; padding-top: 1.2rem; }
            header, #MainMenu, footer { visibility: hidden; height: 0; }
            .question { text-align: center; color: #1f2937; font-size: 1.4rem;
                font-weight: 650; margin: .6rem 0 1rem; }
            div[data-testid="stImage"] img { max-height: 560px; object-fit: contain; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    page = st.session_state.page
    if page == "welcome":
        welcome_page()
    elif page == "instructions":
        instructions_page()
    elif page == "experiment":
        experiment_page()
    elif page == "done":
        done_page()
    else:
        st.session_state.page = "welcome"
        st.rerun()


if __name__ == "__main__":
    main()
