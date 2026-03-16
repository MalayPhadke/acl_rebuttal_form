import datetime
import os
import random
import threading
import time
from collections import defaultdict

import gspread
import pandas as pd
import streamlit as st
from google.oauth2.service_account import Credentials


# --- Configuration ---
IMAGE_CSV = "images.csv"
INPUT_FOLDERS = [
    "NExTVideo_0038_3277823769",
    "NExTVideo_1000_3824469712",
    "NExTVideo_1106_6016405951",
]
SUBSAMPLE_SIZE = None  # Use all rows

# Google Sheet worksheet names (same pattern as original app)
SHEET_RESPONSE_METADATA = "response_metadata"
SHEET_RESPONSE_SIMPLE = "response_simple"


# --- Session State Initialization ---
if "page" not in st.session_state:
    st.session_state.page = "instructions"
if "trials" not in st.session_state:
    st.session_state.trials = []
if "current_trial_index" not in st.session_state:
    st.session_state.current_trial_index = 0
if "user_id" not in st.session_state:
    st.session_state.user_id = ""
if "results" not in st.session_state:
    st.session_state.results = []
if "start_time" not in st.session_state:
    st.session_state.start_time = None
if "countdown_num" not in st.session_state:
    st.session_state.countdown_num = 0


GSHEETS_SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]


def resolve_image_path(csv_path, input_folders):
    """Resolve image path from CSV against known folder roots."""
    csv_path = str(csv_path).strip()

    if os.path.exists(csv_path):
        return csv_path

    candidate = os.path.join(os.getcwd(), csv_path)
    if os.path.exists(candidate):
        return candidate

    normalized = csv_path.replace("\\", "/")
    parts = [p for p in normalized.split("/") if p]

    for root in input_folders:
        root = os.path.abspath(root)
        root_name = os.path.basename(root.rstrip("/"))

        if root_name in parts:
            idx = parts.index(root_name)
            suffix = parts[idx + 1 :]
            mapped = os.path.join(root, *suffix) if suffix else root
            if os.path.exists(mapped):
                return mapped

        basename_guess = os.path.join(root, os.path.basename(csv_path))
        if os.path.exists(basename_guess):
            return basename_guess

    return candidate


def shuffle_no_adjacent_category(trials):
    """Shuffle trials so adjacent items do not share the same category when possible."""
    if not trials:
        return []

    grouped = defaultdict(list)
    for trial in trials:
        category = str(trial.get("category", "unknown")).strip().lower() or "unknown"
        grouped[category].append(trial)

    for category in grouped:
        random.shuffle(grouped[category])

    ordered = []
    last_category = None
    total = len(trials)

    for _ in range(total):
        candidates = [
            (category, items)
            for category, items in grouped.items()
            if items and category != last_category
        ]

        if not candidates:
            candidates = [(category, items) for category, items in grouped.items() if items]

        if not candidates:
            break

        max_count = max(len(items) for _, items in candidates)
        top_categories = [category for category, items in candidates if len(items) == max_count]
        chosen_category = random.choice(top_categories)

        ordered.append(grouped[chosen_category].pop())
        last_category = chosen_category

    return ordered


def _build_gspread_creds():
    """Build gspread credentials from Streamlit secrets. Call from MAIN thread only."""
    secrets = dict(st.secrets["connections"]["gsheets"])
    spreadsheet_url = secrets.pop("spreadsheet")
    creds = Credentials.from_service_account_info(secrets, scopes=GSHEETS_SCOPES)
    return creds, spreadsheet_url


def _bg_append_row(creds, spreadsheet_url, worksheet_name, data_dict):
    """Append a single row to a Google Sheet worksheet. Thread-safe, NO st.* calls."""
    try:
        client = gspread.authorize(creds)
        spreadsheet = client.open_by_url(spreadsheet_url)
        try:
            ws = spreadsheet.worksheet(worksheet_name)
        except gspread.exceptions.WorksheetNotFound:
            ws = spreadsheet.add_worksheet(title=worksheet_name, rows=1000, cols=max(len(data_dict), 1))
            ws.append_row(list(data_dict.keys()), value_input_option="RAW")
        ws.append_row([str(v) for v in data_dict.values()], value_input_option="RAW")
    except Exception as exc:
        print(f"[BG Save Error] {worksheet_name}: {exc}")


def _bg_write_result(creds, spreadsheet_url, save_data):
    """Background thread target: write result to Google Sheets."""
    safe_uid = save_data["safe_uid"]
    _bg_append_row(
        creds,
        spreadsheet_url,
        f"{SHEET_RESPONSE_METADATA}_{safe_uid}",
        save_data["message_result"],
    )
    _bg_append_row(
        creds,
        spreadsheet_url,
        f"{SHEET_RESPONSE_SIMPLE}_{safe_uid}",
        save_data["simple_result"],
    )


def load_image_data():
    """Load image-only trials from images.csv."""
    if not os.path.exists(IMAGE_CSV):
        st.error(f"CSV not found: {IMAGE_CSV}")
        return []

    try:
        df = pd.read_csv(
            IMAGE_CSV,
            header=None,
            names=["image_path", "question", "gt", "category"],
        )
    except Exception as exc:
        st.error(f"Failed to read {IMAGE_CSV}: {exc}")
        return []

    if not df.empty and str(df.iloc[0]["image_path"]).lower() in {
        "image_path",
        "path",
        "filename",
    }:
        df = df.iloc[1:].reset_index(drop=True)

    trials = []
    for idx, row in df.iterrows():
        image_path = resolve_image_path(row["image_path"], INPUT_FOLDERS)
        trials.append(
            {
                "id": idx + 1,
                "type": "image",
                "filename": image_path,
                "question": str(row["question"]).strip(),
                "gt": str(row["gt"]).strip().lower(),
                "category": str(row["category"]).strip() if "category" in df.columns else "",
            }
        )

    trials = shuffle_no_adjacent_category(trials)

    if SUBSAMPLE_SIZE:
        trials = trials[:SUBSAMPLE_SIZE]

    return trials


def prepare_result(trial, response, confidence, reaction_time):
    """Prepare result payload and local result row in same shape as original app."""
    gt = trial.get("gt", "").lower()
    resp = response.lower()
    correct = gt == resp

    user_id = st.session_state.user_id
    safe_uid = "".join(x for x in user_id if x.isalnum() or x in "._-")

    message_result = {
        "user_id": user_id,
        "trial_id": trial["id"],
        "type": trial.get("type", "image"),
        "filename": os.path.basename(trial["filename"]) if trial.get("filename") else "unknown",
        "question": trial.get("question", ""),
        "response": response,
        "confidence": confidence,
        "correct": correct,
        "gt": trial.get("gt", ""),
        "reaction_time_ms": int(reaction_time * 1000),
        "timestamp": datetime.datetime.now().isoformat(),
    }

    exclude_keys = {"filename", "question", "type", "id", "gt", "is_checker"}
    for key, value in trial.items():
        if key not in message_result and key not in exclude_keys:
            message_result[key] = value

    st.session_state.results.append(message_result)

    simple_result = {
        "user_id": user_id,
        "trial_id": message_result["trial_id"],
        "response": message_result["response"],
        "confidence": message_result["confidence"],
        "gt": message_result["gt"],
        "correct": message_result["correct"],
        "reaction_time_ms": message_result["reaction_time_ms"],
    }

    return {
        "safe_uid": safe_uid,
        "message_result": message_result,
        "simple_result": simple_result,
    }


def record_response(trial, answer, confidence):
    end_time = time.time()
    reaction_time = end_time - st.session_state.start_time
    response = answer.strip().lower()
    save_data = prepare_result(trial, response, confidence, reaction_time)
    creds, spreadsheet_url = _build_gspread_creds()
    threading.Thread(
        target=_bg_write_result,
        args=(creds, spreadsheet_url, save_data),
        daemon=True,
    ).start()

    st.session_state.current_trial_index += 1
    st.session_state.start_time = None
    st.session_state.page = "countdown"
    st.session_state.countdown_num = 3
    st.rerun()


def instructions_page():
    st.title("Reaction Time Experiment (Images Only)")
    st.write(
        """
        ### Instructions

        1. You will see a series of **images** and a question for each image.
        2. Rate your **certainty** in your answer on a scale of 1 to 5:
           - **1**: Highly Uncertain
           - **2**: Somewhat Uncertain
           - **3**: Neutral
           - **4**: Somewhat Certain
           - **5**: Highly Certain
        3. Answer **YES** or **NO** by clicking the buttons as quickly and accurately as possible.
        4. A 3-second countdown appears between trials.
        """
    )

    user_id_input = st.text_input("Enter your Participant ID:", value=st.session_state.user_id)

    if st.button("Start Experiment"):
        if not user_id_input.strip():
            st.error("Please enter a valid Participant ID to continue.")
            return

        st.session_state.user_id = user_id_input.strip()
        st.session_state.trials = load_image_data()
        st.session_state.current_trial_index = 0
        st.session_state.results = []

        if not st.session_state.trials:
            st.error("No image trials found. Please check images.csv and input folders.")
            return

        st.session_state.page = "countdown"
        st.session_state.countdown_num = 3
        st.rerun()


def countdown_page():
    num = st.session_state.countdown_num
    if num <= 0:
        st.session_state.page = "experiment"
        st.rerun()
        return

    st.markdown(
        f"<div style='display:flex; justify-content:center; align-items:center; height:80vh;'>"
        f"<span style='font-size:140px; font-weight:bold; color:#4A90D9;'>{num}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )
    time.sleep(1)
    st.session_state.countdown_num -= 1
    st.rerun()


def experiment_page():
    total = len(st.session_state.trials)
    idx = st.session_state.current_trial_index

    if idx >= total:
        st.session_state.page = "done"
        st.rerun()
        return

    trial = st.session_state.trials[idx]

    st.progress(idx / total)
    st.write(f"Trial {idx + 1} of {total}")
    st.markdown("---")

    # Prompt text above image
    st.markdown(f'How certainly the question "**{trial["question"]}**" can be answered given the below image? Also answer the question with Yes or No.')

    image_path = trial.get("filename")
    if image_path and os.path.exists(image_path):
        col1, col2, col3 = st.columns([1, 6, 1])
        with col2:
            st.image(image_path, caption="", use_container_width=True)
    else:
        st.warning(f"Image not found: {image_path}")

    if st.session_state.start_time is None:
        st.session_state.start_time = time.time()

    st.markdown("---")
    st.write("**Certainty Score**")

    # Initialize certainty in session state if not present or if trial changed
    state_key = f"cert_choice_{idx}"
    if state_key not in st.session_state:
        st.session_state[state_key] = 3  # Default to Neutral

    confidence_map = {
        1: "Highly Uncertain",
        2: "Somewhat Uncertain",
        3: "Neutral",
        4: "Somewhat Certain",
        5: "Highly Certain"
    }

    # Certainty Cells (Buttons)
    cols = st.columns(5)
    for val in range(1, 6):
        with cols[val-1]:
            # Highlight selected button
            btn_type = "primary" if st.session_state[state_key] == val else "secondary"
            if st.button(confidence_map[val], key=f"btn_{idx}_{val}", use_container_width=True, type=btn_type):
                st.session_state[state_key] = val
                st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        yes_clicked = st.button("YES", use_container_width=True, key=f"yes_{idx}")
    with col2:
        no_clicked = st.button("NO", use_container_width=True, key=f"no_{idx}")

    confidence = st.session_state[state_key]

    if yes_clicked:
        record_response(trial, "yes", confidence)
    elif no_clicked:
        record_response(trial, "no", confidence)


def done_page():
    st.title("Experiment Complete")
    st.success("Thank you for participating!")

    if st.session_state.results:
        results_df = pd.DataFrame(st.session_state.results)
        st.write(f"Collected responses: {len(results_df)}")
        st.download_button(
            label="Download Results CSV",
            data=results_df.to_csv(index=False),
            file_name=f"results_{st.session_state.user_id}.csv",
            mime="text/csv",
        )

    if st.button("Restart"):
        st.session_state.page = "instructions"
        st.session_state.trials = []
        st.session_state.current_trial_index = 0
        st.session_state.results = []
        st.session_state.start_time = None
        st.session_state.countdown_num = 0
        st.rerun()


def main():
    st.set_page_config(page_title="Reaction Time Experiment (Images Only)")

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