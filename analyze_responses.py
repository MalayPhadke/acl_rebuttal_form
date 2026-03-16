import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
import re

def extract_frame_number(filename):
    match = re.search(r'frame_(\d+)', filename)
    if match:
        return match.group(1)
    return None

def parse_list(s):
    if not isinstance(s, str): return s
    s = s.strip()
    if s.startswith('[') and s.endswith(']'): s = s[1:-1]
    return [item.strip() for item in s.split(',') if item.strip()]

def main():
    # 1. Load list_questions.csv
    list_questions_path = '/home/leaplab/acl_rebuttal_form/list_questions.csv'
    df_questions = pd.read_csv(list_questions_path)
    
    # Precompute maps
    shifted_map = {} # (question, frame_number) -> shifted_index
    center_frame_map = {} # question -> center_frame_str
    frames_order_map = {} # question -> list of frame_numbers
    
    for _, row in df_questions.iterrows():
        q_text = row['question'].strip()
        frames = parse_list(row['frames'])
        center_frames = parse_list(row['center_frames'])
        
        frames = [str(f).zfill(4) if isinstance(f, int) else str(f) for f in frames]
        center_f = str(center_frames[0]).zfill(4) if center_frames else None
        
        center_frame_map[q_text] = center_f
        frames_order_map[q_text] = frames
        
        if center_f in frames:
            ref_idx = frames.index(center_f)
            for i, f in enumerate(frames):
                shifted_map[(q_text, f)] = i - ref_idx

    # 2. Load all response metadata
    files = glob.glob('/home/leaplab/acl_rebuttal_form/form sheet - response_metadata_*.csv')
    df_responses = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df_responses['frame_num'] = df_responses['filename'].apply(extract_frame_number)
    df_responses['Result'] = df_responses['correct'].map({True: 'Correct', False: 'Incorrect'})
    df_responses['shifted_index'] = df_responses.apply(lambda r: shifted_map.get((r['question'].strip(), r['frame_num'])), axis=1)

    # Ensure category field is consistent (lowercase)
    df_responses['category'] = df_responses['category'].str.lower()
    categories = ['geometric', 'compositional', 'semantic']
    
    # 3. Plotting Setup
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    # Global styling
    sns.set_theme(style="whitegrid")
    
    # Subplots 1-3: Individual Categories
    for i, cat in enumerate(categories):
        ax = axes[i]
        cat_df = df_responses[df_responses['category'] == cat].copy()
        
        if cat_df.empty:
            ax.set_title(f"{cat.capitalize()} (No Data)")
            continue
            
        # Get the question text for this category to find the frame order
        q_text = cat_df['question'].iloc[0].strip()
        f_order = frames_order_map.get(q_text, [])
        center_f = center_frame_map.get(q_text)
        
        # Map frame numbers to a simple integer index (0, 1, 2...) for plotting
        f_to_idx = {f: idx for idx, f in enumerate(f_order)}
        cat_df['plot_index'] = cat_df['frame_num'].map(f_to_idx)
        
        # Calculate mean RT per frame
        mean_rt = cat_df.groupby('plot_index')['reaction_time_ms'].mean().reset_index()
        
        # Plot mean line
        sns.lineplot(data=mean_rt, x='plot_index', y='reaction_time_ms', ax=ax, 
                     color='black', linewidth=2, label='Mean RT', marker='o')
        
        # Plot individual dots (Correct/Incorrect)
        # We manually scatter them to ensure blue/red coloring
        for res, color in [('Correct', 'blue'), ('Incorrect', 'red')]:
            subset = cat_df[cat_df['Result'] == res]
            ax.scatter(subset['plot_index'], subset['reaction_time_ms'], 
                       color=color, alpha=0.6, s=40, label=f"{res} Response")
            
        # Mark Center Frame
        if center_f in f_to_idx:
            center_idx = f_to_idx[center_f]
            ax.axvline(x=center_idx, color='green', linestyle='--', linewidth=2, label='Center Frame')
            
        ax.set_title(f"Reaction Time: {cat.capitalize()}", fontsize=14)
        ax.set_xticks(range(len(f_order)))
        ax.set_xticklabels(f_order, rotation=45)
        ax.set_xlabel("Frame Number")
        ax.set_ylabel("RT (ms)")
        ax.legend()

    # Subplot 4: Mean of all categories with reindexing
    ax_all = axes[3]
    df_reindexed = df_responses.dropna(subset=['shifted_index'])
    
    # Group by shifted_index and category for separate lines, then a total mean
    for cat in categories:
        cat_data = df_reindexed[df_reindexed['category'] == cat]
        if not cat_data.empty:
            mean_cat = cat_data.groupby('shifted_index')['reaction_time_ms'].mean().reset_index()
            sns.lineplot(data=mean_cat, x='shifted_index', y='reaction_time_ms', ax=ax_all, 
                         label=f"{cat.capitalize()}", alpha=0.5, linestyle='--')
            
    # Total mean line
    mean_all = df_reindexed.groupby('shifted_index')['reaction_time_ms'].mean().reset_index()
    sns.lineplot(data=mean_all, x='shifted_index', y='reaction_time_ms', ax=ax_all, 
                 color='black', linewidth=3, marker='s', label='ALL Categories (Mean)')
    
    # Individual trial dots for the reindexed plot?
    # User said "remvoe the scatter plot" but also "mark correct/incorrect".
    # I'll add them here too but maybe smaller.
    for res, color in [('Correct', 'blue'), ('Incorrect', 'red')]:
        subset = df_reindexed[df_reindexed['Result'] == res]
        ax_all.scatter(subset['shifted_index'], subset['reaction_time_ms'], 
                       color=color, alpha=0.3, s=20)

    # Mark Center Frame at 0
    ax_all.axvline(x=0, color='green', linestyle='--', linewidth=2, label='Center Frame (0)')
    
    ax_all.set_title("Aligned Average (Shifted Index)", fontsize=14)
    ax_all.set_xlabel("Relative Frame Index (Center = 0)")
    ax_all.set_ylabel("RT (ms)")
    ax_all.legend()

    plt.tight_layout()
    output_path = '/home/leaplab/acl_rebuttal_form/reaction_time_analysis_grid.png'
    plt.savefig(output_path, dpi=300)
    print(f"Analysis plot saved to {output_path}")

if __name__ == "__main__":
    main()
