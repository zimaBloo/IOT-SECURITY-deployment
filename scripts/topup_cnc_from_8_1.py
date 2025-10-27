"""Stream 8.1.labeled, collect C&C rows (prefer detailed label), and reservoir-sample up to the number needed to bring total C&C in balanced_multiclass.csv to 15000.
Appends sampled rows (with label set to 'C&C') to balanced_multiclass.csv.
"""
import re
import csv
import random
from pathlib import Path

IN = '/home/tr/IOT-SECURITY-ML-MODEL/8.1.labeled'
BAL = Path('/home/tr/IOT-SECURITY-ML-MODEL/balanced_multiclass.csv')

random.seed(42)

LABEL_MAP = {
    'C&C': 'C&C',
    'C&C-FileDownload': 'C&C',
    'C&C-Torii': 'C&C',
    'C&C-HeartBeat': 'C&C',
    'C&C-HeartBeat-FileDownload': 'C&C',
    'C&C-PartOfAHorizontalPortScan': 'C&C',
    'C&C-HeartBeat-Attack': 'C&C'
}

def build_logical_columns(path):
    raw_cols = None
    with open(path,'r') as fh:
        for line in fh:
            if line.startswith('#fields'):
                raw_cols = line.strip().split('\t')[1:]
                break
    if raw_cols is None:
        raise RuntimeError('no #fields')
    logical_columns = []
    combined_indices = {}
    for i, rc in enumerate(raw_cols):
        parts = re.split(r"\s+", rc.strip())
        if len(parts) > 1:
            combined_indices[i] = parts
            logical_columns.extend(parts)
        else:
            logical_columns.append(parts[0])
    return raw_cols, logical_columns, combined_indices

# determine how many more C&C we need
if not BAL.exists():
    raise RuntimeError('balanced_multiclass.csv not found; run main collection first')

import pandas as pd
bal = pd.read_csv(BAL)
current = int(bal['label'].value_counts().get('C&C', 0))
need = 15000 - current
print('Current C&C:', current, 'Need:', need)
if need <= 0:
    print('No topping up needed')
    raise SystemExit(0)

raw_cols, logical_columns, combined_indices = build_logical_columns(IN)

# identify selected columns to match balanced csv if possible
selected_cols = [c for c in ['id.orig_p','id.resp_p','proto','duration','orig_bytes','resp_bytes','orig_pkts','resp_pkts','conn_state','label'] if c in logical_columns]

# use reservoir sampling to pick exactly `need` rows from the stream of C&C rows
reservoir = []
count = 0
with open(IN,'r') as fh:
    for line in fh:
        if line.startswith('#') or line.strip()=='':
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
        if det_label and det_label != '-':
            chosen = det_label
        else:
            chosen = row.get('label','')
        mapped = LABEL_MAP.get(chosen, chosen)
        if mapped != 'C&C':
            continue
        # this is a C&C row; consider for reservoir
        count += 1
        if len(reservoir) < need:
            reservoir.append(row)
        else:
            # replace with decreasing probability
            j = random.randrange(count)
            if j < need:
                reservoir[j] = row

print('Found', count, 'C&C rows in file; sampling', len(reservoir))
if len(reservoir) == 0:
    print('No samples to append')
    raise SystemExit(0)

# prepare rows to append matching balanced columns
bal_cols = bal.columns.tolist()
append_rows = []
for r in reservoir:
    out = {c: r.get(c, '') for c in bal_cols}
    out['label'] = 'C&C'
    append_rows.append(out)

# append to balanced_multiclass.csv
import csv
with open(BAL,'a', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=bal_cols)
    for row in append_rows:
        writer.writerow(row)

print('Appended', len(append_rows), 'rows to balanced_multiclass.csv')

# print new counts
bal2 = pd.read_csv(BAL)
print('New C&C count:', int(bal2['label'].value_counts().get('C&C',0)))
