import pandas as pd
import numpy as np
import glob
import os

"""This code will collect each C&C related sample from all datasets, which after there will be approximately 57k C&C samples.
It will also collect DDos, Benign, PortScan and Okiru samples from specific files to have a balanced dataset."""

"""file names:
CTU-IoT-Malware-Capture-34-1 (Mirai) -> 34.1.labeled
CTU-IoT-Malware-Capture-43-1 (Mirai) -> 43.1.labeled
CTU-IoT-Malware-Capture-44-1 (Mirai) -> 44.1.labeled
CTU-IoT-Malware-Capture-49-1 (Mirai) -> 49.1.labeled
CTU-IoT-Malware-Capture-52-1 (Mirai) -> 52.1.labeled
CTU-IoT-Malware-Capture-20-1 (Torii) -> 20.1.labeled
CTU-IoT-Malware-Capture-21-1 (Torii) -> 21.1.labeled
CTU-IoT-Malware-Capture-42-1 (Trojan) -> 42.1.labeled
CTU-IoT-Malware-Capture-60-1 (Gagfyt) -> 60.1.labeled
CTU-IoT-Malware-Capture-17-1 (Kenjiro) -> 17.1.labeled
CTU-IoT-Malware-Capture-36-1 (Okiru) -> 36.1.labeled
CTU-IoT-Malware-Capture-33-1 (Kenjiro) -> 33.1.labeled
CTU-IoT-Malware-Capture-8-1 (Hakai) -> 8.1.labeled
CTU-IoT-Malware-Capture-35-1 (Mirai) -> 35.1.labeled
CTU-IoT-Malware-Capture-48-1 (Mirai) -> 48.1.labeled


"""
# C&C-related labels to map to 'C&C'
cnc_labels = [
    "C&C",
    "C&C-FileDownload",
    "C&C-Torii",
    "C&C-HeartBeat",
    "C&C-HeartBeat-FileDownload",
    "C&C-PartOfAHorizontalPortScan",
    "C&C-HeartBeat-Attack"
]

LABEL_MAP = {lbl: "C&C" for lbl in cnc_labels}
LABEL_MAP.update({
    "Benign": "Benign",
    "DDoS": "DDoS",
    "PartOfAHorizontalPortScan": "PortScan",
    "Okiru": "Okiru"
})

TARGET_SAMPLES = 50000

def balanced_sample(df, label, n):
    subset = df[df['label'] == label]
    if len(subset) > n:
        return subset.sample(n=n, random_state=42)
    else:
        return subset

def parse_log(filename, selected_cols, rename_map):
    """Parse a .labeled file and return a cleaned DataFrame with selected/renamed columns."""
    with open(filename, 'r') as f:
        lines = f.readlines()
    header_line_idx = next(i for i, line in enumerate(lines) if line.startswith("#fields"))
    columns = lines[header_line_idx].strip().split("\t")[1:]
    data_rows = []
    for line in lines[header_line_idx + 1:]:
        if line.startswith("#") or line.strip() == "":
            continue
        parts = line.strip().split('\t')
        if len(parts) != len(columns):
            continue  # skip malformed lines
        row = dict(zip(columns, parts))
        # map label if possible
        orig_label = row.get("label", "")
        row['label'] = LABEL_MAP.get(orig_label, orig_label)
        data_rows.append(row)
    df = pd.DataFrame(data_rows)
    # Select and rename columns early for efficiency
    available_cols = [col for col in selected_cols if col in df.columns]
    df = df[available_cols]
    df = df.rename(columns=rename_map)
    return df

def clean_df(df):
    """Clean a single parsed dataframe: normalize missing values and coerce numeric/categorical types."""
    # Replace dash placeholders with NaN
    df = df.replace("-", np.nan)
    # Numeric columns to coerce
    num_cols = ["src_port", "dst_port", "duration", "src_bytes", "dst_bytes", "src_pkts", "dst_pkts"]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    # Categorical columns to fill
    cat_cols = ["proto", "state"]
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].fillna("unknown")
    return df

# Find all .labeled files in the current directory (adjust path as needed)
all_files = glob.glob("*.labeled")

# Columns to select and rename mapping
selected_cols = [
    "id.orig_p", "id.resp_p", "proto", "duration",
    "orig_bytes", "resp_bytes", "orig_pkts", "resp_pkts",
    "conn_state", "label"
]
rename_map = {
    "id.orig_p": "src_port",
    "id.resp_p": "dst_port",
    "orig_bytes": "src_bytes",
    "resp_bytes": "dst_bytes",
    "orig_pkts": "src_pkts",
    "resp_pkts": "dst_pkts",
    "conn_state": "state"
}

# For storing all rows
rows = []

for filename in all_files:
    print(f"Processing {filename}...")
    df = parse_log(filename, selected_cols, rename_map)
    # Clean this file's dataframe before sampling to keep memory usage low and ensure
    # balanced sampling works on typed data.
    df = clean_df(df)

    base = os.path.basename(filename)
    # Always collect all C&C samples
    rows.append(df[df['label'] == "C&C"])
    # Special handling for CTU-IoT-Malware-Capture-43-1 (Mirai)
    if "43-1" in base:
        for label in ["Benign", "DDoS", "PortScan"]:
            sampled = balanced_sample(df, label, TARGET_SAMPLES)
            rows.append(sampled)
    # Special handling for CTU-IoT-Malware-Capture-36-1 (Okiru)
    if "36.1" in base:
        sampled = balanced_sample(df, "Okiru", TARGET_SAMPLES)
        rows.append(sampled)

# Concatenate everything
df_all = pd.concat(rows, ignore_index=True)
# Save
df_all.to_csv("balanced_multiclass.csv", index=False)
print("✓ Saved balanced multiclass dataset to balanced_multiclass.csv")
print("Label distribution:")
print(df_all['label'].value_counts())