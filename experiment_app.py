import streamlit as st
import pandas as pd
import os
import time
import random
import datetime
import wave
import contextlib
import threading
import gspread
from google.oauth2.service_account import Credentials

# --- Configuration ---
# Specific CSV paths to load
DATA_FILES = [
    "lr_sweep/image_matrix_complex_complex.csv",
    "distractor_sweep/distractor_matrix_basic_basic.csv",
    "size_sweep/size_matrix_basic_basic.csv",
    "distractor_task/metadata.csv",
    "volume_task/metadata.csv",
    "temporal_loc_task/metadata.csv"
]
EXPLAINER_CSV = "explainers/explainers.csv"
CHECKER_CSV = "checkers/checkers.csv"
CHECKER_POSITIONS = [30, 31]  # 0-indexed positions to insert checker trials
SUBSAMPLE_SIZE = None # Use all data

# Google Sheet worksheet names (each maps to a former CSV)
SHEET_RESPONSE_METADATA = "response_metadata"
SHEET_RESPONSE_SIMPLE = "response_simple"
SHEET_RESPONSE_CHECKER = "response_checker"

# --- Session State Initialization ---
if "page" not in st.session_state:
    st.session_state.page = "instructions"
if "trials" not in st.session_state:
    st.session_state.trials = []
if "current_trial_index" not in st.session_state:
    st.session_state.current_trial_index = 0
if "user_id" not in st.session_state:
    st.session_state.user_id = f"user_{int(time.time())}"
if "results" not in st.session_state:
    st.session_state.results = []
if "start_time" not in st.session_state:
    st.session_state.start_time = None
if "explainers" not in st.session_state:
    st.session_state.explainers = []
if "current_explainer_index" not in st.session_state:
    st.session_state.current_explainer_index = 0
if "explainer_audio_done" not in st.session_state:
    st.session_state.explainer_audio_done = False
if "countdown_num" not in st.session_state:
    st.session_state.countdown_num = 0
if "audio_wait_done" not in st.session_state:
    st.session_state.audio_wait_done = False

# --- Helper Functions ---

def get_audio_duration(filepath):
    try:
        with contextlib.closing(wave.open(filepath, 'r')) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            duration = frames / float(rate)
            return duration
    except Exception as e:
        print(f"Error getting duration for {filepath}: {e}")
        return 5.0 # Fallback

def robust_shuffle(trials):
    """
    Shuffles trials such that no two trials from the same source task (CSV) are adjacent.
    """
    if not trials:
        return []
    
    # Group by task_source
    groups = {}
    for t in trials:
        src = t.get('task_source', 'unknown')
        if src not in groups:
            groups[src] = []
        groups[src].append(t)
    
    # Shuffle each group internally
    for k in groups:
        random.shuffle(groups[k])
        
    # Interleave
    result = []
    # Create a list of (key, list)
    pool = list(groups.items())
    
    last_source = None
    
    # While we have items left
    total_items = sum(len(g) for g in groups.values())
    
    # Simple heuristic: try to pick a random available source != last_source
    # If stuck, pick any available.
    
    for _ in range(total_items):
        # Candidates excluding last_source if possible
        candidates = [k for k, v in groups.items() if len(v) > 0 and k != last_source]
        
        if not candidates:
            # Must pick from last_source (forced)
            candidates = [k for k, v in groups.items() if len(v) > 0]
            
        if not candidates:
            break
            
        choice_key = random.choice(candidates)
        result.append(groups[choice_key].pop())
        last_source = choice_key
        
    return result

def resolve_media_path(file_path, csv_dir):
    """Resolve a media file path, trying absolute then fallback locations."""
    file_path = str(file_path).strip()
    if os.path.exists(file_path):
        return file_path
    basename = os.path.basename(file_path)
    candidates = [
        os.path.join(csv_dir, basename),
        os.path.join(csv_dir, "images", basename),
        os.path.join(csv_dir, "..", basename),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return file_path  # Keep original if not found

def detect_media_type(filename):
    """Detect whether a file is image or audio based on extension."""
    ext = os.path.splitext(str(filename).strip())[1].lower()
    if ext in ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp'):
        return "image"
    elif ext in ('.wav', '.mp3', '.ogg', '.flac'):
        return "audio"
    return None

def load_explainers():
    """Loads explainer trials from the explainers CSV. Audio explainers come first."""
    explainers = []
    filepath = EXPLAINER_CSV
    if not os.path.exists(filepath):
        st.warning(f"Explainer file not found: {filepath}")
        return explainers
    try:
        df = pd.read_csv(filepath)
        # Normalize: column might be 'path' or 'image_path' or 'audio_path'
        if 'path' in df.columns:
            df['filename'] = df['path']
        elif 'image_path' in df.columns:
            df['filename'] = df['image_path']
        elif 'audio_path' in df.columns:
            df['filename'] = df['audio_path']
        
        csv_dir = os.path.dirname(filepath)
        for _, row in df.iterrows():
            found_path = resolve_media_path(row['filename'], csv_dir)
            media_type = detect_media_type(found_path)
            if media_type is None:
                media_type = "image"  # fallback
            duration = 0
            if media_type == "audio" and os.path.exists(found_path):
                duration = get_audio_duration(found_path)
            explainers.append({
                "id": row['id'],
                "type": media_type,
                "filename": found_path,
                "question": row['question'],
                "gt": str(row['gt']).strip(),
                "duration": duration,
            })
    except Exception as e:
        st.error(f"Error loading explainers: {e}")
    # Sort: audio explainers first, then image
    explainers.sort(key=lambda x: 0 if x["type"] == "audio" else 1)
    return explainers

def load_checkers():
    """Loads checker trials from the checkers CSV."""
    checkers = []
    filepath = CHECKER_CSV
    if not os.path.exists(filepath):
        st.warning(f"Checker file not found: {filepath}")
        return checkers
    try:
        df = pd.read_csv(filepath)
        if 'image_path' in df.columns:
            df['filename'] = df['image_path']
        elif 'audio_path' in df.columns:
            df['filename'] = df['audio_path']
        elif 'path' in df.columns:
            df['filename'] = df['path']

        csv_dir = os.path.dirname(filepath)
        for _, row in df.iterrows():
            found_path = resolve_media_path(row['filename'], csv_dir)
            media_type = detect_media_type(found_path)
            if media_type is None:
                media_type = "image"
            duration = 0
            if media_type == "audio" and os.path.exists(found_path):
                duration = get_audio_duration(found_path)
            trial_data = row.to_dict()
            trial_data.update({
                "id": row['id'],
                "type": media_type,
                "filename": found_path,
                "question": row['question'],
                "gt": str(row['gt']).strip(),
                "task_source": "checker",
                "duration": duration,
                "is_checker": True,
            })
            checkers.append(trial_data)
    except Exception as e:
        st.error(f"Error loading checkers: {e}")
    return checkers

def load_data():
    """Loads image and audio data from specific CSVs."""
    trials = []
    
    # helper to process a single CSV
    def process_csv(filepath):
        try:
            if not os.path.exists(filepath):
                st.warning(f"File not found: {filepath}")
                return

            # Special handling for volume_task malformed CSV
            if "volume_task/metadata.csv" in filepath:
                try:
                    lines = []
                    with open(filepath, 'r') as f:
                        header = f.readline().strip().split(',')
                        lines.append(header)
                        for line in f:
                            parts = line.strip().split(',')
                            if len(parts) >= 3 and '/home/' in parts[0]:
                                p0 = parts[0]
                                idx = p0.find('/home/')
                                if idx != -1:
                                    id_val = p0[:idx]
                                    path_val = p0[idx:]
                                    new_line = [id_val, path_val] + parts[1:]
                                    lines.append(new_line)
                                else:
                                    lines.append(parts)
                            else:
                                lines.append(parts)
                    df = pd.DataFrame(lines[1:], columns=header)
                except Exception as e:
                    st.error(f"Failed custom parse for volume_task: {e}")
                    return
            else:
                df = pd.read_csv(filepath)
            
            # Normalize columns
            if 'image_path' in df.columns:
                df['filename'] = df['image_path']
                media_type = "image"
            elif 'audio_path' in df.columns:
                df['filename'] = df['audio_path']
                media_type = "audio"
            else:
                st.warning(f"Unknown media type in {filepath}")
                return
                
            if 'answer' in df.columns:
                df['gt'] = df['answer']
                
            required = {'id', 'filename', 'question', 'gt'}
            if not required.issubset(df.columns):
                st.warning(f"Skipping {filepath}: Missing columns {required - set(df.columns)}")
                return
            
            task_source = os.path.basename(filepath)
            csv_dir = os.path.dirname(filepath)

            for _, row in df.iterrows():
                found_path = resolve_media_path(row['filename'], csv_dir)

                trial_data = row.to_dict()
                
                duration = 0
                if media_type == "audio" and os.path.exists(found_path):
                    duration = get_audio_duration(found_path)

                trial_data.update({
                    "id": row['id'],
                    "type": media_type,
                    "filename": found_path,
                    "question": row['question'],
                    "gt": str(row['gt']).strip(),
                    "task_source": task_source,
                    "duration": duration,
                    "is_checker": False,
                })
                trials.append(trial_data)
        except Exception as e:
            st.error(f"Error reading {filepath}: {e}")

    for csv_file in DATA_FILES:
        process_csv(csv_file)

    trials = robust_shuffle(trials)
    
    if SUBSAMPLE_SIZE:
        trials = trials[:SUBSAMPLE_SIZE]

    # Insert checker trials at specified positions
    checker_trials = load_checkers()
    for i, pos in enumerate(CHECKER_POSITIONS):
        if i < len(checker_trials):
            insert_at = min(pos, len(trials))
            trials.insert(insert_at, checker_trials[i])
        
    return trials

# --- Google Sheets Background Save (thread-safe, no st.* calls) ---

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
            # Create the worksheet with a header row
            ws = spreadsheet.add_worksheet(title=worksheet_name, rows=1000, cols=max(len(data_dict), 1))
            ws.append_row(list(data_dict.keys()), value_input_option='RAW')
        # Append the data row
        ws.append_row([str(v) for v in data_dict.values()], value_input_option='RAW')
    except Exception as e:
        print(f"[BG Save Error] {worksheet_name}: {e}")

def _bg_write_result(creds, spreadsheet_url, save_data):
    """Background thread target: write result to Google Sheets."""
    safe_uid = save_data["safe_uid"]
    if save_data["is_checker"]:
        _bg_append_row(creds, spreadsheet_url, f"{SHEET_RESPONSE_CHECKER}_{safe_uid}", save_data["simple_result"])
    else:
        _bg_append_row(creds, spreadsheet_url, f"{SHEET_RESPONSE_METADATA}_{safe_uid}", save_data["message_result"])
        _bg_append_row(creds, spreadsheet_url, f"{SHEET_RESPONSE_SIMPLE}_{safe_uid}", save_data["simple_result"])

def prepare_result(trial, response, reaction_time):
    """Prepare result data (fast, no API calls). Returns save payload."""
    gt = trial.get("gt", "").lower()
    resp = response.lower()
    correct = (gt == resp)
    
    user_id = st.session_state.user_id
    safe_uid = "".join(x for x in user_id if x.isalnum() or x in "._-")
    
    message_result = {
        "user_id": user_id,
        "trial_id": trial["id"],
        "type": trial["type"],
        "filename": os.path.basename(trial["filename"]) if trial["filename"] else "unknown",
        "question": trial["question"],
        "response": response,
        "correct": correct,
        "gt": trial["gt"],
        "reaction_time_ms": int(reaction_time * 1000),
        "timestamp": datetime.datetime.now().isoformat()
    }
    
    # Merge all other trial data (metadata from CSV) excluding internal keys
    exclude_keys = {"filename", "question", "type", "id", "gt", "is_checker"} 
    for k, v in trial.items():
        if k not in message_result and k not in exclude_keys:
            message_result[k] = v
            
    st.session_state.results.append(message_result)

    # Simple result format (shared by normal + checker)
    simple_result = {
        "user_id": user_id,
        "trial_id": message_result["trial_id"],
        "response": message_result["response"],
        "gt": message_result["gt"],
        "correct": message_result["correct"],
        "reaction_time_ms": message_result["reaction_time_ms"]
    }

    return {
        "safe_uid": safe_uid,
        "message_result": message_result,
        "simple_result": simple_result,
        "is_checker": trial.get("is_checker", False),
    }

# --- Pages ---

def instructions_page():
    st.title("Reaction Time Experiment")
    st.write("""
    ### Instructions
    
    1. You will be presented with a series of questions based on **Images** or **Audio**.
    2. Read the question and listen to audio or analyse the image and answer **YES** or **NO** based on them.
    3. Between each question there will be a 3 second countdown.
    4. You will be given some examples at the start to understand the task.
    
    Click 'Start' when you are ready.
    """)
    
    # User ID Input
    default_id = "" if st.session_state.user_id.startswith("user_") else st.session_state.user_id
    user_id_input = st.text_input("Enter your Participant ID:", value=default_id)

    if st.button("Start Experiment"):
        if not user_id_input.strip():
            st.error("Please enter a valid Participant ID to continue.")
        else:
            st.session_state.user_id = user_id_input.strip()
            st.session_state.explainers = load_explainers()
            st.session_state.current_explainer_index = 0
            if st.session_state.explainers:
                st.session_state.page = "explainer"
            else:
                # No explainers, go straight to experiment
                st.session_state.trials = load_data()
                if not st.session_state.trials:
                    st.error("No trials found! Please check data files.")
                    return
                st.session_state.page = "experiment"
                st.session_state.current_trial_index = 0
                st.session_state.start_time = None
            st.rerun()

def experiment_page():
    if st.session_state.current_trial_index >= len(st.session_state.trials):
        st.session_state.page = "done"
        st.rerun()
        return

    trial = st.session_state.trials[st.session_state.current_trial_index]

    # Determine if audio is still playing (hasn't finished yet)
    is_waiting_for_audio = (trial["type"] == "audio" and st.session_state.start_time is None
                            and not st.session_state.audio_wait_done)

    # --- Render trial content (always shown) ---
    st.progress((st.session_state.current_trial_index) / len(st.session_state.trials))
    st.write(f"Trial {st.session_state.current_trial_index + 1} of {len(st.session_state.trials)}")
    st.markdown("---")

    if trial["type"] == "image":
        if st.session_state.start_time is None:
            st.session_state.start_time = time.time()
        image_path = trial.get("filename")
        if image_path and os.path.exists(image_path):
            col1, col2, col3 = st.columns([1, 6, 1])
            with col2:
                st.image(image_path, caption="", use_container_width=True)
        else:
            st.warning(f"Image not found: {image_path}")
    elif trial["type"] == "audio":
        audio_path = trial.get("filename")
        if audio_path and os.path.exists(audio_path):
            st.audio(audio_path, format="audio/wav", autoplay=is_waiting_for_audio)
        else:
            st.warning(f"Audio not found: {audio_path}")

    st.markdown(f"### {trial['question']}")

    # Show buttons — disabled while audio is playing, enabled otherwise
    buttons_container = st.empty()

    def render_buttons(disabled=False):
        with buttons_container.container():
            col1, col2 = st.columns(2)
            with col1:
                if st.button("YES", use_container_width=True, disabled=disabled,
                             key=f"btn_yes_{st.session_state.current_trial_index}_{disabled}"):
                    handle_response(trial, "YES")
            with col2:
                if st.button("NO", use_container_width=True, disabled=disabled,
                             key=f"btn_no_{st.session_state.current_trial_index}_{disabled}"):
                    handle_response(trial, "NO")

    render_buttons(disabled=is_waiting_for_audio)

    # If audio is playing, wait for it to finish then enable buttons
    if is_waiting_for_audio:
        duration = trial.get("duration", 5.0)
        with st.spinner(f"Playing audio ({duration:.1f}s)..."):
            time.sleep(duration)
        # Audio finished — start reaction time and enable buttons
        st.session_state.audio_wait_done = True
        st.session_state.start_time = time.time()
        render_buttons(disabled=False)

def handle_response(trial, answer):
    end_time = time.time()
    rt = end_time - st.session_state.start_time
    # Prepare result data (fast — no API calls)
    save_data = prepare_result(trial, answer, rt)
    # Launch background thread to write to Google Sheets
    # (reads secrets in main thread, writes in background)
    creds, spreadsheet_url = _build_gspread_creds()
    threading.Thread(
        target=_bg_write_result,
        args=(creds, spreadsheet_url, save_data),
        daemon=True,
    ).start()
    # Immediately transition to countdown (no waiting for API)
    st.session_state.current_trial_index += 1
    st.session_state.start_time = None
    st.session_state.audio_wait_done = False
    st.session_state.page = "countdown"
    st.session_state.countdown_num = 3
    st.rerun()

def countdown_page():
    """Shows a single countdown number, sleeps, then moves to next number or trial."""
    num = st.session_state.countdown_num
    if num <= 0:
        st.session_state.page = "experiment"
        st.rerun()
        return
    # Render ONLY the number — nothing else
    st.markdown(
        f"<div style='display:flex; justify-content:center; align-items:center; height:80vh;'>"
        f"<span style='font-size:140px; font-weight:bold; color:#4A90D9;'>{num}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )
    time.sleep(1)
    st.session_state.countdown_num -= 1
    st.rerun()

def explainer_audio_playing_page():
    """Dedicated page for playing the explainer audio. Only shows audio + spinner."""
    explainers = st.session_state.explainers
    idx = st.session_state.current_explainer_index
    trial = explainers[idx]

    st.title("📖 Example Trial — Audio")
    st.write(f"Example {idx + 1} of {len(explainers)}")
    st.markdown("---")
    audio_path = trial.get("filename")
    if audio_path and os.path.exists(audio_path):
        st.audio(audio_path, format="audio/wav", autoplay=True)
        duration = trial.get("duration", 5.0)
        with st.spinner(f"🔊 Playing audio... ({duration:.1f}s)"):
            time.sleep(duration)
    else:
        st.warning(f"Audio not found: {audio_path}")
    # Done — transition to explainer page to show full content
    st.session_state.explainer_audio_done = True
    st.session_state.page = "explainer"
    st.rerun()

def explainer_page():
    """Shows explainer trials (audio first, then image) with correct answer and reasoning."""
    explainers = st.session_state.explainers
    idx = st.session_state.current_explainer_index

    if idx >= len(explainers):
        st.session_state.page = "ready"
        st.rerun()
        return

    trial = explainers[idx]

    # --- AUDIO EXPLAINER ---
    if trial["type"] == "audio":
        if not st.session_state.explainer_audio_done:
            # Redirect to dedicated audio-playing page
            st.session_state.page = "explainer_audio_playing"
            st.rerun()
            return
        # Phase 2: Audio finished — show everything together
        st.title("📖 Example Trial — Audio")
        st.write(f"Example {idx + 1} of {len(explainers)} — This shows you how audio questions work.")
        st.markdown("---")
        audio_path = trial.get("filename")
        if audio_path and os.path.exists(audio_path):
            st.audio(audio_path, format="audio/wav", autoplay=False)
        answer = trial['gt'].upper()
        st.markdown(f"### ❓ {trial['question']}")
        st.markdown(f"### ✅ Correct Answer: **{answer}**")
        st.info(f"Listen to the audio clip. The answer is **{answer}** "
                f"because you can determine the temporal order of the sounds "
                f"by listening to which sound occurs first in the recording.")
        st.markdown("---")
        if st.button("Next →", use_container_width=True, key=f"explainer_next_{idx}"):
            st.session_state.current_explainer_index += 1
            st.session_state.explainer_audio_done = False
            st.rerun()

    # --- IMAGE EXPLAINER ---
    elif trial["type"] == "image":
        st.title("📖 Example Trial — Image")
        st.write(f"Example {idx + 1} of {len(explainers)} — This shows you how image questions work.")
        st.markdown("---")
        image_path = trial.get("filename")
        if image_path and os.path.exists(image_path):
            col1, col2, col3 = st.columns([1, 6, 1])
            with col2:
                st.image(image_path, caption="", use_container_width=True)
        else:
            st.warning(f"Image not found: {image_path}")
        answer = trial['gt'].upper()
        st.markdown(f"### ❓ {trial['question']}")
        st.markdown(f"### ✅ Correct Answer: **{answer}**")
        st.info(f"Look at the image carefully. The answer is **{answer}** "
                f"because you can visually verify the spatial relationship "
                f"described in the question by examining the positions of the objects.")
        st.markdown("---")
        if st.button("Next →", use_container_width=True, key=f"explainer_next_{idx}"):
            st.session_state.current_explainer_index += 1
            st.rerun()

def ready_page():
    """Transition page between explainers and experiment."""
    st.markdown(
        "<div style='display:flex; flex-direction:column; justify-content:center; "
        "align-items:center; height:60vh;'>"
        "<h1 style='font-size:60px; margin-bottom:20px;'>🚀 Ready to Start!</h1>"
        "<p style='font-size:20px; color:#888;'>The experiment will now begin.</p>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("---")
    if st.button("Begin Experiment →", use_container_width=True, key="begin_experiment"):
        st.session_state.trials = load_data()
        if not st.session_state.trials:
            st.error("No trials found! Please check data files.")
            return
        st.session_state.page = "countdown"
        st.session_state.countdown_num = 3
        st.session_state.current_trial_index = 0
        st.session_state.start_time = None
        st.rerun()

def done_page():
    st.title("Experiment Complete")
    st.success("Thank you for participating!")
    st.write("Your data has been saved.")
    
    if st.button("Restart"):
        st.session_state.page = "instructions"
        st.session_state.results = []
        st.session_state.trials = []
        st.session_state.explainers = []
        st.session_state.current_explainer_index = 0
        st.session_state.explainer_audio_done = False
        st.session_state.countdown_num = 0
        st.session_state.audio_wait_done = False
        st.rerun()

# --- Main App ---

def main():
    st.set_page_config(page_title="Reaction Time Experiment")

    # Single top-level container that clears previous page content on each run.
    app_root = st.empty()

    with app_root.container():
        page = st.session_state.page
        if page == "instructions":
            instructions_page()
        elif page == "explainer":
            explainer_page()
        elif page == "explainer_audio_playing":
            explainer_audio_playing_page()
        elif page == "ready":
            ready_page()
        elif page == "countdown":
            countdown_page()
        elif page == "experiment":
            experiment_page()
        elif page == "done":
            done_page()

if __name__ == "__main__":
    main()
