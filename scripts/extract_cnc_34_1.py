"""Stream `34.1.labeled`, detect label/detailed-label like the main script, and write only rows labeled/mapped to 'C&C'.
Saves CSV to 34.1_CnC_only.csv (same selected columns as the main script).
"""
import re
import csv
import os

IN = '/home/tr/IOT-SECURITY-ML-MODEL/34.1.labeled'
OUT = '/home/tr/IOT-SECURITY-ML-MODEL/34.1_CnC_only.csv'
LABEL_MAP = {
    # include common C&C variants; expand as needed
    'C&C': 'C&C',
    'C&C-FileDownload': 'C&C',
    'C&C-Torii': 'C&C',
    'C&C-HeartBeat': 'C&C',
    'C&C-HeartBeat-FileDownload': 'C&C',
    'C&C-PartOfAHorizontalPortScan': 'C&C',
    'C&C-HeartBeat-Attack': 'C&C'
}

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

# read header
raw_cols = None
with open(IN, 'r') as fh:
    for line in fh:
        if line.startswith('#fields'):
            raw_cols = line.strip().split('\t')[1:]
            break
    if raw_cols is None:
        raise RuntimeError('no #fields in file')

# build logical columns handling combined tokens
logical_columns = []
combined_indices = {}
for i, rc in enumerate(raw_cols):
    parts = re.split(r"\s+", rc.strip())
    if len(parts) > 1:
        combined_indices[i] = parts
        logical_columns.extend(parts)
    else:
        logical_columns.append(parts[0])

# prepare writer
out_cols = [c for c in selected_cols if c in logical_columns]
out_header = [rename_map.get(c,c) for c in out_cols]

with open(OUT, 'w', newline='') as outf:
    writer = csv.writer(outf)
    writer.writerow(out_header)
    # stream read lines
    with open(IN, 'r') as fh:
        for line in fh:
            if line.startswith('#') or line.strip() == '':
                continue
            parts = line.rstrip('\n').split('\t')
            if combined_indices and len(parts) == len(raw_cols):
                new_parts = []
                for idx, val in enumerate(parts):
                    if idx in combined_indices:
                        vals = re.split(r"\s+", val.strip())
                        expected = len(combined_indices[idx])
                        if len(vals) < expected:
                            vals += [''] * (expected - len(vals))
                        new_parts.extend(vals[:expected])
                    else:
                        new_parts.append(val)
                parts = new_parts
            if len(parts) != len(logical_columns):
                continue
            row = dict(zip(logical_columns, parts))
            det_label = row.get('det_label') or row.get('detailed-label') or row.get('detailed_label')
            chosen = None
            if det_label and det_label != '-':
                chosen = det_label
            else:
                chosen = row.get('label', '')
            mapped = LABEL_MAP.get(chosen, chosen)
            if mapped == 'C&C':
                out_row = [row.get(c, '') for c in out_cols]
                # rename mapping not strictly needed for CSV headers already set
                writer.writerow(out_row)

print('Wrote', OUT)
