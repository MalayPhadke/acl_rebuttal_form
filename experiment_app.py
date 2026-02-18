import streamlit as st
import pandas as pd
import os
import time
import random
import datetime
import wave
import contextlib

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
RESPONSES_DIR = "responses" # Directory to save per-user responses
SUBSAMPLE_SIZE = None # Use all data

# Ensure responses directory exists
os.makedirs(RESPONSES_DIR, exist_ok=True)

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
                # Read manually or with engine python
                try:
                    # Attempt to fix the known issue: id and path concatenated
                    lines = []
                    with open(filepath, 'r') as f:
                        header = f.readline().strip().split(',')
                        lines.append(header)
                        for line in f:
                            # Expected: id, audio_path, question, answer
                            # Actual line 2: clean_step...wav,Is...,no
                            # We split by first comma, then check if first part has /home/
                            parts = line.strip().split(',')
                            if len(parts) >= 3 and '/home/' in parts[0]:
                                # Split the first part at /home/
                                p0 = parts[0]
                                idx = p0.find('/home/')
                                if idx != -1:
                                    id_val = p0[:idx]
                                    path_val = p0[idx:]
                                    # reassemble: id, path, question, answer
                                    new_line = [id_val, path_val] + parts[1:]
                                    lines.append(new_line)
                                else:
                                    lines.append(parts)
                            else:
                                lines.append(parts)
                    
                    # Create DF from list of lists
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
                
            # Normalize Ground Truth column
            if 'answer' in df.columns:
                df['gt'] = df['answer']
                
            required = {'id', 'filename', 'question', 'gt'}
            if not required.issubset(df.columns):
                st.warning(f"Skipping {filepath}: Missing columns {required - set(df.columns)}")
                return
            
            task_source = os.path.basename(filepath)

            for _, row in df.iterrows():
                # Path resolution
                file_path = str(row['filename']).strip()
                
                # Check if absolute path exists
                found_path = file_path if os.path.exists(file_path) else None
                
                if not found_path:
                    # Fallback resolution
                    csv_dir = os.path.dirname(filepath)
                    basename = os.path.basename(file_path)
                    candidates = [
                        os.path.join(csv_dir, basename),
                        os.path.join(csv_dir, "images", basename), # legacy
                        os.path.join(csv_dir, "..", basename),
                    ]
                    for c in candidates:
                        if os.path.exists(c):
                            found_path = c
                            break
                    if not found_path:
                         # Keep original if not found
                         found_path = file_path

                # Create trial object
                trial_data = row.to_dict()
                
                duration = 0
                if media_type == "audio" and found_path and os.path.exists(found_path):
                    duration = get_audio_duration(found_path)

                trial_data.update({
                    "id": row['id'],
                    "type": media_type,
                    "filename": found_path,
                    "question": row['question'],
                    "gt": str(row['gt']).strip(),
                    "task_source": task_source,
                    "duration": duration
                })
                trials.append(trial_data)
        except Exception as e:
            st.error(f"Error reading {filepath}: {e}")

    for csv_file in DATA_FILES:
        process_csv(csv_file)

    trials = robust_shuffle(trials)
    
    if SUBSAMPLE_SIZE:
        trials = trials[:SUBSAMPLE_SIZE]
        
    return trials

def save_result(trial, response, reaction_time):
    """Saves a single trial result to User-Specific CSVs."""
    
    # Calculate correctness
    gt = trial.get("gt", "").lower()
    resp = response.lower()
    correct = (gt == resp)
    
    user_id = st.session_state.user_id
    
    # Define filenames based on user_id
    # We sanitize user_id just in case
    safe_uid = "".join(x for x in user_id if x.isalnum() or x in "._-")
    full_csv_path = os.path.join(RESPONSES_DIR, f"response_metadata_{safe_uid}.csv")
    simple_csv_path = os.path.join(RESPONSES_DIR, f"response_simple_{safe_uid}.csv")
    
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
    exclude_keys = {"filename", "question", "type", "id", "gt"} 
    for k, v in trial.items():
        if k not in message_result and k not in exclude_keys:
            message_result[k] = v
            
    # Append to local list
    st.session_state.results.append(message_result)
    
    # Append to Full Metadata CSV
    df = pd.DataFrame([message_result])
    
    if os.path.exists(full_csv_path):
        try:
            existing_df = pd.read_csv(full_csv_path)
            updated_df = pd.concat([existing_df, df], ignore_index=True)
            updated_df.to_csv(full_csv_path, index=False)
        except:
             df.to_csv(full_csv_path, mode='a', header=False, index=False)
    else:
        df.to_csv(full_csv_path, index=False)

    # Append to Simple CSV
    simple_result = {
        "user_id": message_result["user_id"],
        "trial_id": message_result["trial_id"],
        "response": message_result["response"],
        "gt": message_result["gt"],
        "correct": message_result["correct"],
        "reaction_time_ms": message_result["reaction_time_ms"]
    }
    df_simple = pd.DataFrame([simple_result])
    
    if os.path.exists(simple_csv_path):
        try:
            existing_simple = pd.read_csv(simple_csv_path)
            updated_simple = pd.concat([existing_simple, df_simple], ignore_index=True)
            updated_simple.to_csv(simple_csv_path, index=False)
        except:
             df_simple.to_csv(simple_csv_path, mode='a', header=False, index=False)
    else:
        df_simple.to_csv(simple_csv_path, index=False)

# --- Pages ---

def instructions_page():
    st.title("Reaction Time Experiment")
    st.write("""
    ### Instructions
    
    1. You will be presented with a series of questions based on **Images** or **Audio**.
    2. Read the question (and listen to audio if applicable) and answer **YES** or **NO** as quickly as possible.
    3. Your reaction time and correctness will be recorded.
    
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
            st.session_state.trials = load_data()
            if not st.session_state.trials:
                st.error("No trials found! Please check data files.")
            else:
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
    
    # Progress
    st.progress((st.session_state.current_trial_index) / len(st.session_state.trials))
    st.write(f"Trial {st.session_state.current_trial_index + 1} of {len(st.session_state.trials)}")

    # Display Stimulus
    st.markdown("---")
    
    # Render Media
    # Determine if we are in "waiting for audio" state
    is_waiting_for_audio = (trial["type"] == "audio" and st.session_state.start_time is None)
    
    # Render Media
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
             if is_waiting_for_audio:
                 st.session_state.start_time = time.time()
                 is_waiting_for_audio = False
    
    st.markdown(f"### {trial['question']}")
    
    # Buttons Container
    buttons_container = st.empty()
    
    # helper to render buttons
    def render_buttons(disabled=False):
        with buttons_container.container():
            col1, col2 = st.columns(2)
            with col1:
                # We need unique keys if we render multiple times to avoid duplicate ID error? 
                # Actually, if we use the same key, Streamlit might complain if rendered twice in one run.
                # But here we render, sleep (UI updates), then render again (overwrite).
                # Streamlit overwrites element with same ID? 
                # Buttons with same label have distinct IDs?
                # To be safe, we don't need keys if we are replacing the container content.
                if st.button("YES", use_container_width=True, disabled=disabled, key=f"btn_yes_{st.session_state.current_trial_index}_{disabled}"):
                     handle_response(trial, "YES")
            with col2:
                if st.button("NO", use_container_width=True, disabled=disabled, key=f"btn_no_{st.session_state.current_trial_index}_{disabled}"):
                     handle_response(trial, "NO")

    # Initial Render
    render_buttons(disabled=is_waiting_for_audio)

    # Handle the wait logic
    if is_waiting_for_audio:
        duration = trial.get("duration", 5.0)
        
        # We assume the audio started playing when st.audio was rendered above.
        # We block here.
        with st.spinner(f"Playing audio ({duration:.1f}s)..."):
            time.sleep(duration)
        
        # Audio finished
        st.session_state.start_time = time.time()
        
        # Re-render buttons as ENABLED immediately
        render_buttons(disabled=False)

def handle_response(trial, answer):
    end_time = time.time()
    rt = end_time - st.session_state.start_time
    
    save_result(trial, answer, rt)
    
    st.session_state.current_trial_index += 1
    st.session_state.start_time = None # Reset for next trial
    st.rerun()

def done_page():
    st.title("Experiment Complete")
    st.success("Thank you for participating!")
    st.write("Your data has been saved.")
    
    if st.button("Restart"):
        st.session_state.page = "instructions"
        st.session_state.results = []
        st.session_state.trials = []
        st.rerun()
        st.rerun()

# --- Main App ---

def main():
    st.set_page_config(page_title="Reaction Time Experiment")
    
    if st.session_state.page == "instructions":
        instructions_page()
    elif st.session_state.page == "experiment":
        experiment_page()
    elif st.session_state.page == "done":
        done_page()

if __name__ == "__main__":
    main()
