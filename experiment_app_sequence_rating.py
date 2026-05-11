import ast
import datetime
import os
import random
import re
import threading
import time
from collections import defaultdict

import gspread
import pandas as pd
import streamlit as st
from google.oauth2.service_account import Credentials


# --- Configuration ---
# QUESTIONS_CSV = "list_questions.csv"
# IMAGES_BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")

QUESTIONS_CSV_1_2 = "list_questions_kalash_new.csv"
IMAGES_BASE_DIR_1_2 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "video_frames_kalash_new")

# QUESTIONS_CSV_3_4 = "/home/debarpanb1/Videoframe_extract/list_questions_saksham.csv"
# IMAGES_BASE_DIR_3_4 = "/home/debarpanb1/Videoframe_extract/video_frames_saksham_filtered"

# Google Sheet worksheet names
SHEET_RESPONSE_METADATA = "seq_response_metadata"
SHEET_RESPONSE_SIMPLE = "seq_response_simple"

# TOTAL_QUESTIONS = 112
# HALF = TOTAL_QUESTIONS // 2  # 56 per part
TOTAL_QUESTIONS = 200
QUESTIONS_PER_PART = 50


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
if "selected_part" not in st.session_state:
    st.session_state.selected_part = None


GSHEETS_SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]


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
            ws = spreadsheet.add_worksheet(
                title=worksheet_name, rows=1000, cols=max(len(data_dict), 1)
            )
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


def parse_frames_list(frames_str):
    """Parse the frames column which is a string like '[1840, 1845, ...]' into a list of ints."""
    frames_str = str(frames_str).strip()
    try:
        parsed = ast.literal_eval(frames_str)
        if isinstance(parsed, list):
            return [int(f) for f in parsed]
    except (ValueError, SyntaxError):
        pass
    # Fallback: extract numbers with regex
    nums = re.findall(r'\d+', frames_str)
    return [int(n) for n in nums]


def video_path_to_folder(video_path, images_dir=None):
    """Convert video_path like '.../NExTVideo/0089/3066966990.mp4' to folder name 'NExTVideo_0089_3066966990'."""
    video_path = str(video_path).strip()
    
    # Check if the exact basename without extension exists in images_dir
    basename = os.path.basename(video_path)
    for ext in ['.mp4', '.webm']:
        if basename.endswith(ext):
            basename = basename[:-len(ext)]
            break
            
    if images_dir and os.path.exists(os.path.join(images_dir, basename)):
        return basename
        
    # Extract components: look for NExTVideo/XXXX/YYYYYYYY.mp4
    parts = video_path.replace("\\", "/").split("/")
    # Find 'NExTVideo' in parts
    for i, part in enumerate(parts):
        if part == "NExTVideo" and i + 2 < len(parts):
            folder_id = parts[i + 1]
            video_file = parts[i + 2].replace(".mp4", "")
            return f"NExTVideo_{folder_id}_{video_file}"
            
    # Fallback: try to parse from the last segments
    parent = os.path.basename(os.path.dirname(video_path))
    grandparent = os.path.basename(os.path.dirname(os.path.dirname(video_path)))
    return f"{grandparent}_{parent}_{basename}"


def resolve_frame_image_path(video_path, frame_num, images_dir):
    """Resolve the path to a frame image given a video path and frame number."""
    folder_name = video_path_to_folder(video_path, images_dir)
    
    for ext in ['.jpg', '.png']:
        frame_filename = f"frame_{frame_num:04d}{ext}"
        full_path = os.path.join(images_dir, folder_name, frame_filename)
        if os.path.exists(full_path):
            return full_path
        # Try without zero-padding
        frame_filename_nopad = f"frame_{frame_num}{ext}"
        full_path_nopad = os.path.join(images_dir, folder_name, frame_filename_nopad)
        if os.path.exists(full_path_nopad):
            return full_path_nopad
            
    return os.path.join(images_dir, folder_name, f"frame_{frame_num:04d}.png")  # Return the expected path even if not found


def load_questions_data(part):
    """Load questions from list_questions.csv and split into parts."""
    # csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), QUESTIONS_CSV)
    
    if part in [1, 2]:
        csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), QUESTIONS_CSV_1_2)
        images_dir = IMAGES_BASE_DIR_1_2
    # elif part in [3, 4]:
    #     csv_path = QUESTIONS_CSV_3_4
    #     images_dir = IMAGES_BASE_DIR_3_4
    else:
        return []

    if not os.path.exists(csv_path):
        st.error(f"CSV not found: {csv_path}")
        return []

    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        st.error(f"Failed to read {csv_path}: {exc}")
        return []

    trials = []
    for idx, row in df.iterrows():
        frames = parse_frames_list(row["frames"])
        image_paths = [resolve_frame_image_path(row["video_path"], f, images_dir) for f in frames]

        trials.append(
            {
                "id": idx + 1,
                "video_path": str(row["video_path"]).strip(),
                "question": str(row["question"]).strip(),
                "frames": frames,
                "image_paths": image_paths,
                "center_frames": str(row.get("center_frames", "")).strip(),
                "gt_answer": str(row.get("gt_answer", "")).strip(),
                "category": str(row.get("primary_category", "")).strip(),
            }
        )

    # Split into parts
    # if part == 1:
    #     trials = trials[:HALF]
    # elif part == 2:
    #     trials = trials[HALF:]
    
    if part in [1, 3]:
        trials = trials[:QUESTIONS_PER_PART]
    elif part in [2, 4]:
        trials = trials[QUESTIONS_PER_PART:QUESTIONS_PER_PART*2]

    # Shuffle so adjacent trials don't share the same category
    trials = shuffle_no_adjacent_category(trials)

    return trials


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
        top_categories = [
            category for category, items in candidates if len(items) == max_count
        ]
        chosen_category = random.choice(top_categories)

        ordered.append(grouped[chosen_category].pop())
        last_category = chosen_category

    return ordered


def scroll_to_top(label=""):
    """Inject JS to scroll Streamlit's main container to top (works on mobile)."""
    st.components.v1.html(
        f"""
        <!-- {label} -->
        <script>
            function scrollToTop() {{
                // Streamlit's actual scrollable container
                var container = window.parent.document.querySelector('[data-testid="stAppViewContainer"]');
                if (container) {{
                    container.scrollTo({{top: 0, behavior: 'instant'}});
                }}
                // Fallbacks
                var main = window.parent.document.querySelector('.main');
                if (main) main.scrollTop = 0;
                window.parent.document.documentElement.scrollTop = 0;
                window.parent.document.body.scrollTop = 0;
                window.parent.scrollTo(0, 0);
            }}
            // Fire at multiple delays to catch slow mobile renders
            scrollToTop();
            setTimeout(scrollToTop, 100);
            setTimeout(scrollToTop, 300);
            setTimeout(scrollToTop, 600);
        </script>
        """,
        height=0,
    )


def prepare_result(trial, ratings, good_datapoint, comment, reaction_time):
    """Prepare result payload for saving."""
    user_id = st.session_state.user_id
    safe_uid = "".join(x for x in user_id if x.isalnum() or x in "._-")

    message_result = {
        "user_id": user_id,
        "trial_id": trial["id"],
        "part": st.session_state.selected_part,
        "video_path": trial.get("video_path", ""),
        "question": trial.get("question", ""),
        "category": trial.get("category", ""),
        "gt_answer": trial.get("gt_answer", ""),
        "frames": str(trial.get("frames", [])),
    }

    # Add individual frame ratings
    for i, rating in enumerate(ratings):
        frame_num = trial["frames"][i] if i < len(trial["frames"]) else i
        message_result[f"frame_{frame_num}_rating"] = rating

    message_result["ratings_list"] = str(ratings)
    message_result["good_datapoint"] = good_datapoint
    message_result["comment"] = comment
    message_result["reaction_time_ms"] = int(reaction_time * 1000)
    message_result["timestamp"] = datetime.datetime.now().isoformat()

    st.session_state.results.append(message_result)

    simple_result = {
        "user_id": user_id,
        "trial_id": message_result["trial_id"],
        "part": st.session_state.selected_part,
        "question": message_result["question"],
        "ratings_list": message_result["ratings_list"],
        "good_datapoint": message_result["good_datapoint"],
        "comment": message_result["comment"],
        "reaction_time_ms": message_result["reaction_time_ms"],
    }

    return {
        "safe_uid": safe_uid,
        "message_result": message_result,
        "simple_result": simple_result,
    }


def record_response(trial, ratings, good_datapoint, comment):
    """Record a response and move to next trial."""
    end_time = time.time()
    reaction_time = end_time - st.session_state.start_time
    save_data = prepare_result(trial, ratings, good_datapoint, comment, reaction_time)
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
    scroll_to_top(label="instructions")
    st.title("Image Sequence Rating Experiment")
    st.write(
        """
        ### Instructions

        1. You will see a series of **questions**, each accompanied by a sequence of **10 images** (video frames).
        2. For each image in the sequence, rate on a scale of **1 to 10** how well the question can be answered from that specific image:
            - **1** = The question **cannot be answered at all** from this image
            - **10** = The question can be answered with **very high certainty** from this image
        3. After rating all 10 images, answer whether this is a **good datapoint** (Yes / No).
            - If you select **No**, please provide a brief explanation in the comment box.
        4. A 3-second countdown appears between trials.
        5. The experiment is split into **two parts** (56 questions each). Select which part to complete below.
        """
    )

    st.markdown("---")
    st.write("### Examples")
    st.write("Question: **Is the child standing on the chair?**")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.image("frame_0110.png", width="stretch")
        st.write("**Rating: 1–2** (Cannot determine from this image)")
    with col2:
        st.image("frame_0226.png", width="stretch")
        st.write("**Rating: 4–5** (Somewhat answerable)")
    with col3:
        st.image("frame_0440.png", width="stretch")
        st.write("**Rating: 9–10** (Very high certainty)")
    st.markdown("---")

    user_id_input = st.text_input(
        "Enter your Participant ID:", value=st.session_state.user_id
    )

    st.markdown("### Select Part")
    col_p1, col_p2 = st.columns(2)
    # col_p1, col_p2, col_p3, col_p4 = st.columns(4)
    with col_p1:
        part1_clicked = st.button(
            f"▶ Part 1 (1–50)",
            use_container_width=True,
            type="primary",
            key="btn_part1",
        )
    with col_p2:
        part2_clicked = st.button(
            f"▶ Part 2 (51–100)",
            use_container_width=True,
            type="primary",
            key="btn_part2",
        )
    # with col_p3:
    #     part3_clicked = st.button(
    #         f"▶ Part 3 (101–150)",
    #         use_container_width=True,
    #         type="primary",
    #         key="btn_part3",
    #     )
    # with col_p4:
    #     part4_clicked = st.button(
    #         f"▶ Part 4 (151–200)",
    #         use_container_width=True,
    #         type="primary",
    #         key="btn_part4",
    #     )
    
    part3_clicked = False
    part4_clicked = False

    if part1_clicked or part2_clicked or part3_clicked or part4_clicked:
        if not user_id_input.strip():
            st.error("Please enter a valid Participant ID to continue.")
            return

        if part1_clicked: selected_part = 1
        elif part2_clicked: selected_part = 2
        elif part3_clicked: selected_part = 3
        elif part4_clicked: selected_part = 4
        
        st.session_state.user_id = user_id_input.strip()
        st.session_state.selected_part = selected_part
        st.session_state.trials = load_questions_data(selected_part)
        st.session_state.current_trial_index = 0
        st.session_state.results = []

        if not st.session_state.trials:
            st.error(
                "No trials found. Please check list_questions.csv and the images folder."
            )
            return

        st.session_state.page = "countdown"
        st.session_state.countdown_num = 3
        st.rerun()


def countdown_page():
    # Full-screen overlay to ensure nothing else is visible
    placeholder = st.empty()

    for num in [3, 2, 1]:
        with placeholder.container():
            st.markdown(
                f"""
                <style>
                    /* Force hide absolute everything else */
                    #root > div:nth-child(1) > div.withScreencast > div > div > div > section {{
                        overflow: hidden !important;
                    }}
                    [data-testid="stSidebar"], [data-testid="stHeader"], [data-testid="stFooter"] {{
                        display: none !important;
                    }}
                </style>
                <div style="position: fixed; top: 0; left: 0; width: 100vw; height: 100vh; background-color: white; z-index: 999999; display: flex; justify-content: center; align-items: center; flex-direction: column;">
                    <span style="font-size: 140px; font-weight: bold; color: #4A90D9; margin-top: -10vh;">{num}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        time.sleep(1)

    st.session_state.page = "experiment"
    st.rerun()


def experiment_page():
    total = len(st.session_state.trials)
    idx = st.session_state.current_trial_index
    scroll_to_top(label=f"exp_{idx}")

    if idx >= total:
        st.session_state.page = "done"
        st.rerun()
        return

    trial = st.session_state.trials[idx]
    part = st.session_state.selected_part

    st.markdown(f"<div style='margin-top: -20px;'><p><b>Part {part}</b> — Question {idx + 1} of {total}</p></div>", unsafe_allow_html=True)
    st.progress(idx / total)

    # Question heading
    st.markdown(
        f'<h4 style="text-align: center; color: #2c3e50; font-size: 16px; margin-top: 5px;">Question: {trial["question"]}</h4>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<p style="text-align: center; color: #7f8c8d; font-size: 12px; margin-top: -5px;">Category: {trial["category"]}</p>',
        unsafe_allow_html=True,
    )

    if st.session_state.start_time is None:
        st.session_state.start_time = time.time()

    # Display 10 images in a 5+5 grid with rating inputs
    st.markdown(
        "<p style='font-size: 12px; margin-bottom: 0px; margin-top: -10px;'><b>Rate each image (1–10):</b> 1 = cannot answer, 10 = very high certainty</p>",
        unsafe_allow_html=True,
    )

    ratings = [None] * len(trial["frames"])
    row_layouts = [5, 5]

    idx_counter = 0
    for row_num, num_cols in enumerate(row_layouts):
        cols = st.columns(5)
        for col_idx in range(num_cols):
            if idx_counter >= len(trial["frames"]):
                break

            frame_num = trial["frames"][idx_counter]
            img_path = trial["image_paths"][idx_counter]

            with cols[col_idx]:
                if os.path.exists(img_path):
                    st.image(img_path, use_container_width=True)
                else:
                    st.warning("Missing")

                rating = st.number_input(
                    f"Img {idx_counter + 1}",
                    min_value=1,
                    max_value=10,
                    value=5,
                    step=1,
                    key=f"rating_{idx}_{idx_counter}",
                    label_visibility="collapsed",
                )

                ratings[idx_counter] = rating

            idx_counter += 1

    st.markdown('<hr style="border: none; border-top: 1px solid #bdc3c7; margin: 5px 0;">', unsafe_allow_html=True)

    good_dp_key = f"good_dp_{idx}"
    comment_key = f"comment_{idx}"

    if good_dp_key not in st.session_state:
        st.session_state[good_dp_key] = None

    # Put Yes, No, Submit on the exact same row!
    col_label, col_yes, col_no, col_submit = st.columns([2, 1, 1, 2])
    with col_label:
        st.markdown('<p style="font-weight: bold; margin-top: 10px;">Is this a good datapoint?</p>', unsafe_allow_html=True)
    with col_yes:
        yes_type = "primary" if st.session_state[good_dp_key] == "yes" else "secondary"
        if st.button("Yes", key=f"gdp_yes_{idx}", use_container_width=True, type=yes_type):
            st.session_state[good_dp_key] = "yes"
            st.rerun()
    with col_no:
        no_type = "primary" if st.session_state[good_dp_key] == "no" else "secondary"
        if st.button("No", key=f"gdp_no_{idx}", use_container_width=True, type=no_type):
            st.session_state[good_dp_key] = "no"
            st.rerun()

    comment = ""
    if st.session_state[good_dp_key] == "no":
        comment = st.text_area(
            "Explain why this is not a good datapoint:",
            key=comment_key,
            height=68,
        )

    with col_submit:
        if st.button("Submit & Next →", use_container_width=True, type="primary", key=f"submit_{idx}"):
            if st.session_state[good_dp_key] is None:
                st.error("⚠️ Please answer whether this is a **good datapoint** (Yes/No) before submitting.")
            elif st.session_state[good_dp_key] == "no" and not comment.strip():
                st.error("⚠️ Please provide a comment explaining why this is **not** a good datapoint.")
            else:
                record_response(
                    trial,
                    ratings,
                    st.session_state[good_dp_key],
                    comment.strip() if comment else "",
                )


def done_page():
    scroll_to_top(label="done")
    st.title("Experiment Complete")
    part = st.session_state.selected_part
    st.success(f"Thank you for completing Part {part}!")

    if st.session_state.results:
        results_df = pd.DataFrame(st.session_state.results)
        st.write(f"Collected responses: {len(results_df)}")
        st.download_button(
            label="Download Results CSV",
            data=results_df.to_csv(index=False),
            file_name=f"results_part{part}_{st.session_state.user_id}.csv",
            mime="text/csv",
        )

    if st.button("Restart"):
        st.session_state.page = "instructions"
        st.session_state.trials = []
        st.session_state.current_trial_index = 0
        st.session_state.results = []
        st.session_state.start_time = None
        st.session_state.countdown_num = 0
        st.session_state.selected_part = None
        st.rerun()


def main():
    st.set_page_config(
        page_title="Image Sequence Rating Experiment",
        layout="wide",
    )
    
    st.markdown(
        """
        <style>
            .block-container {
                padding-top: 0rem;
                padding-bottom: 0rem;
                max-width: 100%;
            }
            header {visibility: hidden; height: 0px;}
            #MainMenu {visibility: hidden; height: 0px;}
            footer {visibility: hidden; height: 0px;}
            
            /* Squeeze all stMarkdown/stText margins */
            p, h4 {
                margin-bottom: 0rem !important;
                padding-bottom: 0rem !important;
            }
            
            /* Compact number inputs */
            .stNumberInput {
                padding-top: 0px !important;
                padding-bottom: 0px !important;
            }
            .stNumberInput label {
                font-size: 11px !important;
                margin-bottom: 0px !important;
            }
            
            /* Reduce gaps between rows */
            div[data-testid="stVerticalBlock"] > div {
                padding-top: 0rem !important;
                padding-bottom: 0rem !important;
                gap: 0rem !important;
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
