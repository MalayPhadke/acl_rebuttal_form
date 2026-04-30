import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
import re
import numpy as np

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
    list_questions_path = '/home/leaplab/Downloads/acl_rebuttal_form/list_questions.csv'
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
    files = glob.glob('/home/leaplab/Downloads/acl_rebuttal_form/form sheet - response_metadata_*.csv')
    df_responses = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df_responses['frame_num'] = df_responses['filename'].apply(extract_frame_number)
    df_responses['Result'] = df_responses['correct'].map({True: 'Correct', False: 'Incorrect'})
    df_responses['shifted_index'] = df_responses.apply(lambda r: shifted_map.get((r['question'].strip(), r['frame_num'])), axis=1)

    # Min-max normalization for confidence (1-5 -> 0-1)
    df_responses['confidence_norm'] = (df_responses['confidence'] - 1) / 4

    # Ensure category field is consistent (lowercase)
    df_responses['category'] = df_responses['category'].str.lower()
    categories = ['geometric', 'compositional', 'semantic']
    
    # 3. New Plotting Setup: Uncertainty / Accuracy / ECE Analysis Grid
    # Following the style of the reference script with shaded regions (min/max and mean +/- std)
    
    # Uncertainty is 1 - confidence_norm
    df_responses['uncertainty'] = df_responses['confidence_norm']
    # Ensure correct is numeric (0/1)
    df_responses['correct_val'] = df_responses['correct'].astype(float)
    
    df_reindexed = df_responses.dropna(subset=['shifted_index']).copy()
    
    def compute_ece(confidences, accuracies, n_bins=10):
        confidences = np.array(confidences)
        accuracies = np.array(accuracies)
        if not len(confidences): return np.nan
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        n = len(confidences)
        for i in range(n_bins):
            if i == 0:
                mask = (confidences >= bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])
            else:
                mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])
            bin_size = np.sum(mask)
            if bin_size > 0:
                avg_conf = np.mean(confidences[mask])
                avg_acc = np.mean(accuracies[mask])
                ece += (bin_size / n) * abs(avg_acc - avg_conf)
        return float(ece)

    def get_metric_curves(df_segment):
        # We compute metrics per question (sequence) at each relative index to get a distribution
        questions = df_segment['question'].unique()
        u_curves = []
        a_curves = []
        e_curves = []
        counts = {} # rel_index -> total samples
        
        for q in questions:
            q_df = df_segment[df_segment['question'] == q]
            u_c, a_c, e_c = {}, {}, {}
            for rel_idx, group in q_df.groupby('shifted_index'):
                u_c[rel_idx] = group['uncertainty'].mean()
                a_c[rel_idx] = group['correct_val'].mean()
                e_c[rel_idx] = compute_ece(group['confidence_norm'], group['correct_val'])
                counts[rel_idx] = counts.get(rel_idx, 0) + len(group)
            u_curves.append(u_c)
            a_curves.append(a_c)
            e_curves.append(e_c)
        return u_curves, a_curves, e_curves, counts

    def plot_shaded_row(axes_row, curves_list, counts, color_theme, metric_label, min_samples=5, show_shading=True, break_at_zero=False):
        # curves_list: list of dicts {rel_idx: value}
        # metric_label: for y-axis
        all_steps = sorted(set().union(*[set(c.keys()) for c in curves_list]))
        # Filter steps by minimum sample count
        valid_steps = [s for s in all_steps if counts.get(s, 0) >= min_samples]
        
        if not valid_steps:
            axes_row.text(0.5, 0.5, "No Data", ha='center', va='center', transform=axes_row.transAxes)
            return

        means, mins, maxs, stds = [], [], [], []
        for s in valid_steps:
            vals = [c[s] for c in curves_list if s in c and not np.isnan(c[s])]
            if vals:
                arr = np.array(vals)
                means.append(np.mean(arr))
                mins.append(np.min(arr))
                maxs.append(np.max(arr))
                stds.append(np.std(arr) / np.sqrt(len(arr)))
            else:
                means.append(np.nan); mins.append(np.nan); maxs.append(np.nan); stds.append(0)

        means, mins, maxs, stds = map(np.array, [means, mins, maxs, stds])
        
        # Plotting
        if show_shading:
            # Shading represents the standard error of the mean across across different sequences (questions)
            axes_row.fill_between(valid_steps, means - stds, means + stds, color=color_theme, alpha=0.25)
        
        # Filter out NaNs
        mask = np.isfinite(means)
        valid_steps_arr = np.array(valid_steps)
        
        if break_at_zero:
            # Plot negative and positive segments separately to create a gap at the center frame (0)
            neg_mask = (valid_steps_arr < 0) & mask
            pos_mask = (valid_steps_arr > 0) & mask
            zero_mask = (valid_steps_arr == 0) & mask
            
            axes_row.plot(valid_steps_arr[neg_mask], means[neg_mask], marker='o', linewidth=2, markersize=4, color=color_theme)
            axes_row.plot(valid_steps_arr[pos_mask], means[pos_mask], marker='o', linewidth=2, markersize=4, color=color_theme)
            
            # Draw the point at 0 if it exists, but do not connect it to the lines
            if np.any(zero_mask):
                axes_row.plot(valid_steps_arr[zero_mask], means[zero_mask], marker='o', markersize=4, color=color_theme, linestyle='None')
        else:
            # Standard continuous plot connecting all valid points
            axes_row.plot(valid_steps_arr[mask], means[mask], marker='o', linewidth=2, markersize=4, color=color_theme)
        
        axes_row.axvline(x=0, color='red', linestyle='--', alpha=0.7)
        axes_row.grid(True, alpha=0.3)
        
        # Add sample count text (optional but helpful)
        if len(maxs) > 0 and np.any(np.isfinite(maxs)) and np.any(np.isfinite(mins)):
            offset = 0.05 * (np.nanmax(maxs) - np.nanmin(mins))
        else:
            offset = 0.05
            
        for i, s in enumerate(valid_steps):
            if np.isfinite(means[i]):
                axes_row.text(s, mins[i] - offset if metric_label == 'Uncertainty' else maxs[i] + offset, 
                              str(int(counts[s])), ha='center', va='center', fontsize=6, alpha=0.7)

    plot_cats = categories + ['all']
    metrics = ['Uncertainty', 'Accuracy', 'ECE']
    colors = {'Uncertainty': 'tab:red', 'Accuracy': 'tab:green', 'ECE': 'tab:blue'}
    
    fig, axes = plt.subplots(len(metrics), len(plot_cats), figsize=(5 * len(plot_cats), 4 * len(metrics)), sharex=True)
    
    for col_idx, cat in enumerate(plot_cats):
        cat_df = df_reindexed if cat == 'all' else df_reindexed[df_reindexed['category'] == cat]
        
        if cat_df.empty:
            for row_idx in range(len(metrics)):
                axes[row_idx, col_idx].text(0.5, 0.5, "No Data", ha='center', va='center')
            continue
            
        u_curves, a_curves, e_curves, counts = get_metric_curves(cat_df)
        
        # Uncertainty
        plot_shaded_row(axes[0, col_idx], u_curves, counts, colors['Uncertainty'], 'Uncertainty', show_shading=True)
        # Accuracy
        plot_shaded_row(axes[1, col_idx], a_curves, counts, colors['Accuracy'], 'Accuracy', show_shading=False, break_at_zero=True)
        # ECE
        plot_shaded_row(axes[2, col_idx], e_curves, counts, colors['ECE'], 'ECE', show_shading=False, break_at_zero=True)
        
        # Titles and Labels
        axes[0, col_idx].set_title(f"{cat.capitalize()}", fontsize=16, fontweight='bold')
        for row_idx in range(len(metrics)):
            if col_idx == 0:
                axes[row_idx, col_idx].set_ylabel(metrics[row_idx], fontsize=14, fontweight='bold')
            if row_idx == len(metrics) - 1:
                axes[row_idx, col_idx].set_xlabel("Relative Frame Index (Center=0)", fontsize=12)
            if metrics[row_idx] in ['Uncertainty', 'Accuracy', 'ECE']:
                axes[row_idx, col_idx].set_ylim(-0.05, 1.05)

    plt.suptitle("Human Response Analysis: Uncertainty, Accuracy, and Calibration", fontsize=20, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    output_path = '/home/leaplab/Downloads/acl_rebuttal_form/confidence_analysis_grid.png'
    plt.savefig(output_path, dpi=300)
    print(f"Analysis plot saved to {output_path}")

    # 4. Separate Horizontal Plot for 'All Categories'
    fig_all, axes_all = plt.subplots(1, 3, figsize=(18, 5))
    
    # Get metrics for ALL data
    u_curves, a_curves, e_curves, counts = get_metric_curves(df_reindexed)
    
    # Plot each metric horizontally
    plot_shaded_row(axes_all[0], u_curves, counts, colors['Uncertainty'], 'Uncertainty', show_shading=True, break_at_zero=False)
    plot_shaded_row(axes_all[1], a_curves, counts, colors['Accuracy'], 'Accuracy', show_shading=False, break_at_zero=True)
    plot_shaded_row(axes_all[2], e_curves, counts, colors['ECE'], 'ECE', show_shading=False, break_at_zero=True)
    
    # Styling for the standalone plot
    for i, metric in enumerate(metrics):
        axes_all[i].set_title(f"{metric}", fontsize=16, fontweight='bold')
        axes_all[i].set_xlabel("Relative Frame Index (Center=0)", fontsize=12)
        axes_all[i].set_ylabel(metric, fontsize=14, fontweight='bold')
        if metric in ['Uncertainty', 'Accuracy', 'ECE']:
            axes_all[i].set_ylim(-0.05, 1.05)
            
    plt.tight_layout()
    output_path_all = '/home/leaplab/Downloads/acl_rebuttal_form/all_categories_metrics.png'
    plt.savefig(output_path_all, dpi=300)
    print(f"All Categories standalone plot saved to {output_path_all}")

if __name__ == "__main__":
    main()
