"""Blinded human validation form for a balanced VQAv2 model cohort."""

from __future__ import annotations

import datetime
import hashlib
import html
import os
import random
import re
import threading
import time

import gspread
import pandas as pd
import streamlit as st
from google.oauth2.service_account import Credentials


APP_DIR = os.path.dirname(os.path.abspath(__file__))
QUESTIONS_CSV = os.path.join(APP_DIR, "vqav2_validation_100.csv")
SHEET_RESPONSE_METADATA = "vqav2_response_metadata"
SHEET_RESPONSE_SIMPLE = "vqav2_response_simple"
QUESTIONS_PER_PART = 50
MODELS = ("qwenvl", "gemma3", "llava", "phi4", "pixtral")

GSHEETS_SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]


def _initialize_state():
    defaults = {
        "page": "welcome",
        "trials": [],
        "current_trial_index": 0,
        "user_id": "",
        "results": [],
        "start_time": None,
        "selected_part": None,
        "completed_trial_ids": set(),
        "resumed_count": 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _safe_user_id(user_id):
    # Google Sheets limits worksheet titles to 100 characters.
    cleaned = "".join(char for char in user_id if char.isalnum() or char in "._-")
    return cleaned[:48] or "participant"


def _build_gspread_creds():
    secrets = dict(st.secrets["connections"]["gsheets"])
    spreadsheet_url = secrets.pop("spreadsheet")
    creds = Credentials.from_service_account_info(secrets, scopes=GSHEETS_SCOPES)
    return creds, spreadsheet_url


def _bg_append_row(creds, spreadsheet_url, worksheet_name, data_dict):
    """Append a result without calling Streamlit APIs from the worker thread."""
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
            [str(value) for value in data_dict.values()], value_input_option="RAW"
        )
    except Exception as exc:
        print(f"[VQAv2 background save error] {worksheet_name}: {exc}")


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


def _fetch_completed_trials(creds, spreadsheet_url, safe_uid, part):
    try:
        client = gspread.authorize(creds)
        spreadsheet = client.open_by_url(spreadsheet_url)
        try:
            worksheet = spreadsheet.worksheet(
                f"{SHEET_RESPONSE_SIMPLE}_{safe_uid}"
            )
        except gspread.exceptions.WorksheetNotFound:
            return set()

        completed = set()
        for record in worksheet.get_all_records():
            if str(record.get("part", "")) == str(part):
                trial_uid = str(record.get("trial_uid", "")).strip()
                if trial_uid:
                    completed.add(trial_uid)
        return completed
    except Exception as exc:
        print(f"[VQAv2 resume check error] {exc}")
        return set()


def _normalize_answer(answer):
    answer = str(answer).strip().lower()
    answer = re.sub(r"[^\w\s]", " ", answer)
    answer = re.sub(r"\b(a|an|the)\b", " ", answer)
    return re.sub(r"\s+", " ", answer).strip()


def load_trials(part, seed):
    if not os.path.exists(QUESTIONS_CSV):
        st.error(f"Trial CSV not found: {QUESTIONS_CSV}")
        return []

    try:
        frame = pd.read_csv(QUESTIONS_CSV, dtype=str, keep_default_na=False)
    except Exception as exc:
        st.error(f"Could not read the VQAv2 trial CSV: {exc}")
        return []

    required = {
        "trial_uid",
        "part",
        "selection_group",
        "target_model",
        "allocation_model",
        "source_correctness",
        "question_id",
        "question",
        "reference_answer",
        "image_path",
    }
    for model in MODELS:
        required.add(f"{model}_prediction")
        required.add(f"{model}_correctness")
    missing = required.difference(frame.columns)
    if missing:
        st.error("Trial CSV is missing: " + ", ".join(sorted(missing)))
        return []

    frame = frame[frame["part"] == str(part)]
    if len(frame) != QUESTIONS_PER_PART:
        st.error(
            f"Part {part} should contain {QUESTIONS_PER_PART} trials, "
            f"but the CSV contains {len(frame)}."
        )
        return []

    group_counts = frame["selection_group"].value_counts().to_dict()
    expected_groups = {
        "unanimous_incorrect": 5,
        "model_specific_incorrect": 20,
        "unanimous_correct": 25,
    }
    target_counts = (
        frame[frame["selection_group"] == "model_specific_incorrect"]
        ["target_model"]
        .value_counts()
        .to_dict()
    )
    labels_ok = (
        (frame["source_correctness"] == "0").sum() == 25
        and (frame["source_correctness"] == "1").sum() == 25
    )
    targets_ok = target_counts == {model: 4 for model in MODELS}
    if group_counts != expected_groups or not labels_ok or not targets_ok:
        st.error(
            f"Part {part} does not match the required 25/25 balanced cohort. "
            "Rebuild it before collecting responses."
        )
        return []

    trials = []
    for row in frame.to_dict("records"):
        image_path = row["image_path"]
        if not os.path.isabs(image_path):
            image_path = os.path.join(APP_DIR, image_path)
        row["image_path"] = image_path
        trials.append(row)

    random.Random(seed).shuffle(trials)
    return trials


def scroll_to_top(label=""):
    st.iframe(
        f"""
        <!-- {label} -->
        <script>
            function scrollToTop() {{
                const container = window.parent.document.querySelector(
                    '[data-testid="stAppViewContainer"]'
                );
                if (container) container.scrollTo({{top: 0, behavior: 'instant'}});
                window.parent.scrollTo(0, 0);
            }}
            scrollToTop();
            setTimeout(scrollToTop, 100);
            setTimeout(scrollToTop, 300);
        </script>
        """,
        height=1,
    )


def welcome_page():
    scroll_to_top("welcome")
    st.title("VQAv2 Answer Validation Study")
    st.subheader("Welcome")
    st.write(
        "In this study, you will answer visual questions using only the image "
        "shown and report how certain you are about each answer."
    )
    st.info(
        "The study has two independent parts of 50 questions each. Complete "
        "the part assigned to you; you may return later using the same "
        "Participant ID to resume."
    )
    st.write(
        "Please work independently, do not use outside tools, and base every "
        "answer only on visible evidence in the image."
    )
    if st.button("Continue to instructions →", type="primary"):
        st.session_state.page = "instructions"
        st.rerun()


def instructions_page():
    scroll_to_top("instructions")
    st.title("Instructions")
    st.markdown(
        """
        For every trial:

        1. Read the **question** and inspect the **image** carefully.
        2. Enter a **short, direct answer**. Use a word or brief phrase rather
           than a full sentence whenever possible.
        3. Select your **certainty from 1 to 10**:
           - **1** = completely uncertain / the image does not support an answer
           - **5** = moderately certain
           - **10** = completely certain
        4. Submit the response to continue. There is no back button, so review
           both fields before submitting.

        Some questions may be ambiguous or hard to answer. Give your best short
        answer and use the certainty score to reflect that uncertainty. Do not
        assume facts that are not visible in the image. If the question truly
        cannot be answered from the image, enter **cannot determine** and choose
        a low certainty score.
        """
    )
    st.warning(
        "Do not refresh or close the page immediately after submitting a "
        "response. Progress is saved after every question."
    )

    st.markdown("---")
    user_id_input = st.text_input(
        "Participant ID", value=st.session_state.user_id,
        help="Use the same ID if you need to resume this part later.",
    )
    st.markdown("#### Select your assigned part")
    left, right = st.columns(2)
    with left:
        part_1 = st.button(
            "Start / resume Part 1 (50 questions)",
            type="primary",
            width="stretch",
        )
    with right:
        part_2 = st.button(
            "Start / resume Part 2 (50 questions)",
            type="primary",
            width="stretch",
        )

    if not (part_1 or part_2):
        if st.button("← Back"):
            st.session_state.page = "welcome"
            st.rerun()
        return

    if not user_id_input.strip():
        st.error("Enter a Participant ID before starting.")
        return

    part = 1 if part_1 else 2
    user_id = user_id_input.strip()
    # hashlib makes the per-part order stable across Python processes.
    digest = hashlib.sha256(f"{user_id}|vqav2|{part}".encode()).hexdigest()
    trials = load_trials(part, seed=int(digest[:16], 16))
    if not trials:
        return

    safe_uid = _safe_user_id(user_id)
    st.session_state.user_id = user_id
    st.session_state.selected_part = part
    st.session_state.trials = trials
    st.session_state.results = []

    with st.spinner("Checking for saved progress…"):
        try:
            creds, spreadsheet_url = _build_gspread_creds()
            completed = _fetch_completed_trials(
                creds, spreadsheet_url, safe_uid, part
            )
        except Exception as exc:
            st.warning(f"Could not check saved progress; starting locally: {exc}")
            completed = set()

    st.session_state.completed_trial_ids = completed
    st.session_state.resumed_count = len(completed)
    st.session_state.current_trial_index = 0
    st.session_state.start_time = None
    st.session_state.page = "countdown"
    st.rerun()


def countdown_page():
    placeholder = st.empty()
    for number in (3, 2, 1):
        with placeholder.container():
            st.markdown(
                f"""
                <style>
                    [data-testid="stSidebar"], [data-testid="stHeader"],
                    [data-testid="stFooter"] {{ display: none !important; }}
                </style>
                <div class="countdown"><span>{number}</span></div>
                """,
                unsafe_allow_html=True,
            )
        time.sleep(1)
    st.session_state.page = "experiment"
    st.rerun()


def _prepare_result(trial, human_answer, certainty, reaction_time):
    user_id = st.session_state.user_id
    normalized_answer = _normalize_answer(human_answer)
    normalized_reference = _normalize_answer(trial["reference_answer"])
    metadata = {
        "user_id": user_id,
        "trial_uid": trial["trial_uid"],
        "part": st.session_state.selected_part,
        "question_id": trial["question_id"],
        "image_id": trial.get("image_id", ""),
        "question": trial["question"],
        "human_answer": human_answer.strip(),
        "human_answer_normalized": normalized_answer,
        "certainty": int(certainty),
        "selection_group": trial["selection_group"],
        "target_model": trial.get("target_model", ""),
        "allocation_model": trial.get("allocation_model", ""),
        "reference_answer": trial["reference_answer"],
        "source_correctness": trial["source_correctness"],
        "human_exact_match_reference": normalized_answer == normalized_reference,
        "source_split": trial.get("source_split", ""),
        "image_filename": os.path.basename(trial["image_path"]),
        "reaction_time_ms": int(reaction_time * 1000),
        "timestamp": datetime.datetime.now().isoformat(),
    }
    for model in MODELS:
        metadata[f"{model}_prediction"] = trial[f"{model}_prediction"]
        metadata[f"{model}_correctness"] = trial[f"{model}_correctness"]

    simple = {
        "user_id": user_id,
        "trial_uid": trial["trial_uid"],
        "part": st.session_state.selected_part,
        "selection_group": trial["selection_group"],
        "target_model": trial.get("target_model", ""),
        "source_correctness": trial["source_correctness"],
        "human_answer": human_answer.strip(),
        "certainty": int(certainty),
        "reaction_time_ms": metadata["reaction_time_ms"],
    }
    st.session_state.results.append(metadata)
    return {
        "safe_uid": _safe_user_id(user_id),
        "metadata": metadata,
        "simple": simple,
    }


def record_response(trial, human_answer, certainty):
    reaction_time = time.time() - st.session_state.start_time
    save_data = _prepare_result(trial, human_answer, certainty, reaction_time)
    try:
        creds, spreadsheet_url = _build_gspread_creds()
        threading.Thread(
            target=_bg_write_result,
            args=(creds, spreadsheet_url, save_data),
            daemon=True,
        ).start()
    except Exception as exc:
        # Keep the in-session result downloadable even if Sheets is unavailable.
        st.session_state["last_save_warning"] = str(exc)

    st.session_state.completed_trial_ids.add(trial["trial_uid"])
    st.session_state.current_trial_index += 1
    st.session_state.start_time = None
    st.session_state.page = "countdown"
    st.rerun()


def experiment_page():
    total = len(st.session_state.trials)
    index = st.session_state.current_trial_index
    completed = st.session_state.completed_trial_ids

    while index < total and st.session_state.trials[index]["trial_uid"] in completed:
        index += 1
        st.session_state.current_trial_index = index

    if index >= total:
        st.session_state.page = "done"
        st.rerun()
        return

    scroll_to_top(f"trial_{index}")
    trial = st.session_state.trials[index]
    answered = len(completed)
    st.caption(
        f"Part {st.session_state.selected_part} · Question {answered + 1} of {total}"
    )
    st.progress(min(answered / total, 1.0))

    if st.session_state.resumed_count:
        st.info(
            f"Resumed: {st.session_state.resumed_count} saved response(s) found."
        )
        st.session_state.resumed_count = 0

    st.markdown(
        f'<div class="question">{html.escape(trial["question"])}</div>',
        unsafe_allow_html=True,
    )
    image_path = trial["image_path"]
    if os.path.exists(image_path):
        left, center, right = st.columns([1, 3, 1])
        with center:
            st.image(image_path, width="stretch")
    else:
        st.error("The image for this question is unavailable. Please contact the researcher.")
        return

    if st.session_state.start_time is None:
        st.session_state.start_time = time.time()

    with st.form(f"response_{trial['trial_uid']}", clear_on_submit=False):
        human_answer = st.text_input(
            "Your answer",
            placeholder="Enter a short, direct answer",
            max_chars=200,
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
        if not human_answer.strip():
            st.error("Enter an answer before submitting.")
        elif certainty is None:
            st.error("Select a certainty score from 1 to 10.")
        else:
            record_response(trial, human_answer, certainty)


def done_page():
    scroll_to_top("done")
    part = st.session_state.selected_part
    st.title("Part complete")
    st.success(f"Thank you for completing Part {part}.")

    warning = st.session_state.get("last_save_warning")
    if warning:
        st.warning(
            "At least one response could not be sent to Google Sheets in this "
            "session. Please download the CSV below and send it to the researcher."
        )

    if st.session_state.results:
        results = pd.DataFrame(st.session_state.results)
        st.download_button(
            "Download this session's responses",
            data=results.to_csv(index=False),
            file_name=(
                f"vqav2_part{part}_{_safe_user_id(st.session_state.user_id)}.csv"
            ),
            mime="text/csv",
        )

    if st.button("Return to start"):
        for key in (
            "trials",
            "results",
            "completed_trial_ids",
        ):
            st.session_state[key] = [] if key != "completed_trial_ids" else set()
        st.session_state.current_trial_index = 0
        st.session_state.start_time = None
        st.session_state.selected_part = None
        st.session_state.page = "welcome"
        st.rerun()


def main():
    st.set_page_config(
        page_title="VQAv2 Answer Validation Study",
        layout="wide",
    )
    _initialize_state()
    st.markdown(
        """
        <style>
            .block-container { max-width: 1100px; padding-top: 1.5rem; }
            header, #MainMenu, footer { visibility: hidden; height: 0; }
            .question {
                text-align: center;
                color: #1f2937;
                font-size: 1.45rem;
                font-weight: 650;
                margin: 0.7rem 0 1rem;
            }
            .countdown {
                position: fixed; inset: 0; z-index: 999999;
                display: flex; align-items: center; justify-content: center;
                background: white;
            }
            .countdown span { font-size: 9rem; font-weight: 700; color: #2563eb; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    page = st.session_state.page
    if page == "welcome":
        welcome_page()
    elif page == "instructions":
        instructions_page()
    elif page == "countdown":
        countdown_page()
    elif page == "experiment":
        experiment_page()
    elif page == "done":
        done_page()
    else:
        st.session_state.page = "welcome"
        st.rerun()


if __name__ == "__main__":
    main()
