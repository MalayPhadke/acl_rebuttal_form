import streamlit as st
import pandas as pd
import os
import time
import random
import datetime

# --- Configuration ---
DATA_DIR_IMAGES = "images"
DATA_DIR_AUDIO = "audio"
RESULTS_FILE = "reaction_times.csv"
SUBSAMPLE_SIZE = 2 # Set to None to use all data

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

def find_files(directory, pattern):
    import glob
    return glob.glob(os.path.join(directory, "**", pattern), recursive=True)

def load_data():
    """Loads image and audio data from all CSVs found in data directories."""
    trials = []
    
    # helper to process a single CSV
    def process_csv(filepath, media_type):
        try:
            df = pd.read_csv(filepath)
            # Normalize columns
            # The user's CSV has: id, image_path, question, gt, variation, gt_ab
            # We need: id, filename (path), question, type
            
            # Check for known columns
            if 'image_path' in df.columns:
                df['filename'] = df['image_path']
            elif 'audio_path' in df.columns:
                df['filename'] = df['audio_path']
                
            required = {'id', 'filename', 'question'}
            if not required.issubset(df.columns):
                st.warning(f"Skipping {filepath}: Missing columns {required - set(df.columns)}")
                return

            for _, row in df.iterrows():
                # Fix Path: 
                # value is like /home/.../lr_sweep/complex_complex/images/01.png
                # We assume the file is relative to the CSV location OR 
                # we search for the filename in the current directory tree?
                # Best guess: The structure locally mirrors the CSV structure?
                # User said: "checkout ive added the image folders... this corresponds to the dataset csv pairs"
                # Let's try to find the file relative to the CSV's directory or the main data dir.
                
                # Method 1: Try to find file by basename in the same dir as CSV or 'images' subdir of CSV dir
                csv_dir = os.path.dirname(filepath)
                basename = os.path.basename(row['filename'])
                
                # Check possible locations
                candidates = [
                    os.path.join(csv_dir, basename),
                    os.path.join(csv_dir, "images", basename),
                    os.path.join(DATA_DIR_IMAGES, basename), # fallback to root
                ]
                
                # Also try to resolve the relative path from the "images" folder if it matches
                # e.g. path in csv: .../lr_sweep/...
                # local: images/lr_sweep/...
                
                found_path = None
                for c in candidates:
                    if os.path.exists(c):
                        found_path = c
                        break
                
                # If still not found, try to search by basename in the whole data dir (slow but robust)
                if not found_path:
                   # Simplistic fallback: assume it is wherever the csv is
                   found_path = os.path.join(csv_dir, basename)

                # Create trial object with all raw data first
                trial_data = row.to_dict()
                
                # Override/Add specific internal keys
                trial_data.update({
                    "id": row['id'],
                    "type": media_type,
                    "filename": found_path,
                    "question": row['question'],
                    "gt": row.get('gt', None)
                })
                trials.append(trial_data)
        except Exception as e:
            st.error(f"Error reading {filepath}: {e}")

    # Load Images
    image_csvs = find_files(DATA_DIR_IMAGES, "*.csv")
    for csv_file in image_csvs:
        process_csv(csv_file, "image")

    # Load Audio
    audio_csvs = find_files(DATA_DIR_AUDIO, "*.csv")
    for csv_file in audio_csvs:
        process_csv(csv_file, "audio")

    random.shuffle(trials)
    
    if SUBSAMPLE_SIZE:
        trials = trials[:SUBSAMPLE_SIZE]
        
    return trials

def save_result(trial, response, reaction_time):
    """Saves a single trial result to CSV."""
    result = {
        "user_id": st.session_state.user_id,
        "trial_id": trial["id"],
        "type": trial["type"],
        # Save just the basename for cleaner data
        "filename": os.path.basename(trial["filename"]) if trial["filename"] else "unknown",
        "question": trial["question"],
        "response": response,
        "reaction_time_ms": int(reaction_time * 1000),
        "timestamp": datetime.datetime.now().isoformat()
    }
    
    # Merge all other trial data (metadata from CSV) excluding internal keys if needed, 
    # or just simple merge and let specific keys override if they duplicate.
    # We want to keep 'variation', 'gt_ab', etc.
    # We exclude large objects or internal processing keys if any.
    exclude_keys = {"filename", "question", "type", "id"} 
    for k, v in trial.items():
        if k not in result and k not in exclude_keys:
            result[k] = v
            
    # Append to local list
    st.session_state.results.append(result)
    
    # Append to CSV file immediately
    df = pd.DataFrame([result])
    # Reorder columns to put standard ones first if possible, but pandas might just append
    # If file exists, we must ensure columns match or use mode='a' with caution if new columns appear.
    # For robust appending with varying columns, it's safer to read-append-write or ensure superset.
    # But for speed/simplicity in this session:
    
    if os.path.exists(RESULTS_FILE):
        try:
            # Load existing to get columns, or just append and let pandas handle schema change (it might not in 'a' mode without header)
            # Actually, to_csv mode='a' with header=False assumes columns match positions! 
            # This is DANGEROUS if we add new columns dynamically.
            # safe approach: read, concat, write. (Slower but safer)
            existing_df = pd.read_csv(RESULTS_FILE)
            updated_df = pd.concat([existing_df, df], ignore_index=True)
            updated_df.to_csv(RESULTS_FILE, index=False)
        except:
             # Fallback if read fails
             df.to_csv(RESULTS_FILE, mode='a', header=False, index=False)
    else:
        df.to_csv(RESULTS_FILE, index=False)

# --- Pages ---

def instructions_page():
    st.title("Reaction Time Experiment")
    st.write("""
    ### Instructions
    
    1. You will be presented with a series of questions based on **Images** or **Audio**.
    2. Read the question and answer **YES** or **NO** as quickly as possible.
    3. Your reaction time will be recorded.
    
    Click 'Start' when you are ready.
    """)
    
    # User ID Input
    # If the current user_id looks like an auto-generated one (starts with user_), default to empty to prompt input
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
                st.session_state.start_time = None # Will be set when experiment page renders
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
    
    # Ensure timer starts/resets for this trial if not already
    # We rely on st.session_state.start_time being set EITHER by checking 'if None' 
    # OR by the previous step. 
    # Ideally, we set it RIGHT NOW if it's not set, but Streamlit reruns.
    # To be precise, we want the time when the MEDIA is rendered.
    # A simple approx is time.time() at top of render, but let's be consistent.
    if st.session_state.start_time is None:
         st.session_state.start_time = time.time()

    if trial["type"] == "image":
        image_path = trial["filename"]
        if image_path and os.path.exists(image_path):
            col1, col2, col3 = st.columns([1, 6, 1])
            with col2:
                st.image(image_path, caption="Look at this image", use_container_width=True)
        else:
            st.warning(f"Image not found: {image_path}")
            
    elif trial["type"] == "audio":
        audio_path = trial["filename"]
        if audio_path and os.path.exists(audio_path):
            st.audio(audio_path, format="audio/wav")
        else:
            st.warning(f"Audio not found: {audio_path}")
    
    st.markdown(f"### {trial['question']}")
    
    # Buttons
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("YES", use_container_width=True):
            handle_response(trial, "YES")
            
    with col2:
        if st.button("NO", use_container_width=True):
            handle_response(trial, "NO")

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
