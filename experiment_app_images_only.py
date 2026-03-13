import datetime
import os
import random
import time

import pandas as pd
import streamlit as st


# --- Configuration ---
IMAGE_CSV = "images.csv"
INPUT_FOLDERS = [
    "/home/leaplab/acl_rebuttal_form/NExTVideo_0038_3277823769",
    "/home/leaplab/acl_rebuttal_form/NExTVideo_1000_3824469712",
    "/home/leaplab/acl_rebuttal_form/NExTVideo_1106_6016405951",
]
SUBSAMPLE_SIZE = None  # Use all rows


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

    random.shuffle(trials)

    if SUBSAMPLE_SIZE:
        trials = trials[:SUBSAMPLE_SIZE]

    return trials


def record_response(trial, answer):
    end_time = time.time()
    reaction_time = end_time - st.session_state.start_time

    response = answer.strip().lower()
    gt = trial.get("gt", "").strip().lower()

    st.session_state.results.append(
        {
            "user_id": st.session_state.user_id,
            "trial_id": trial["id"],
            "filename": trial["filename"],
            "question": trial["question"],
            "category": trial.get("category", ""),
            "response": response,
            "gt": gt,
            "correct": response == gt,
            "reaction_time_ms": int(reaction_time * 1000),
            "timestamp": datetime.datetime.now().isoformat(),
        }
    )

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
        2. Answer **YES** or **NO** as quickly and accurately as possible.
        3. A 3-second countdown appears between trials.
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

    image_path = trial.get("filename")
    if image_path and os.path.exists(image_path):
        col1, col2, col3 = st.columns([1, 6, 1])
        with col2:
            st.image(image_path, caption="", use_container_width=True)
    else:
        st.warning(f"Image not found: {image_path}")

    st.markdown(f"### {trial['question']}")

    if st.session_state.start_time is None:
        st.session_state.start_time = time.time()

    col1, col2 = st.columns(2)
    with col1:
        if st.button("YES", use_container_width=True, key=f"yes_{idx}"):
            record_response(trial, "yes")
    with col2:
        if st.button("NO", use_container_width=True, key=f"no_{idx}"):
            record_response(trial, "no")


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