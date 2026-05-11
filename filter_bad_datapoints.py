import pandas as pd
import os

base_dir = "/home/debarpanb1/Videoframe_extract/acl_rebuttal_form"
files = [
    "form sheet - seq_response_metadata_debarpan.csv",
    "form sheet - seq_response_metadata_Kalash.csv",
    "form sheet - seq_response_metadata_sahil.csv",
    "form sheet - seq_response_metadata_SakshamMaitri.csv"
]

bad_responses = {}

for f in files:
    path = os.path.join(base_dir, f)
    if os.path.exists(path):
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            dp = str(row.get('good_datapoint', '')).strip().lower()
            if dp == 'no':
                vp = str(row['video_path']).strip()
                q = str(row['question']).strip()
                uid = str(row['user_id']).strip()
                comment = str(row.get('comment', '')).strip()
                
                key = (vp, q)
                if key not in bad_responses:
                    bad_responses[key] = {'users': [], 'comments': []}
                
                bad_responses[key]['users'].append(uid)
                if comment:
                    bad_responses[key]['comments'].append(f"{uid}: {comment}")

list_q_path = os.path.join(base_dir, "list_questions.csv")
list_q_df = pd.read_csv(list_q_path)

filtered_rows = []

for _, row in list_q_df.iterrows():
    vp = str(row['video_path']).strip()
    q = str(row['question']).strip()
    key = (vp, q)
    
    if key in bad_responses:
        new_row = row.to_dict()
        new_row['annotator_said_no'] = ", ".join(bad_responses[key]['users'])
        new_row['comments'] = " | ".join(bad_responses[key]['comments'])
        filtered_rows.append(new_row)

filtered_df = pd.DataFrame(filtered_rows)
out_path = os.path.join(base_dir, "list_questions_bad_datapoints.csv")
filtered_df.to_csv(out_path, index=False)

print(f"Filtered {len(filtered_df)} bad datapoints and saved to {out_path}")
