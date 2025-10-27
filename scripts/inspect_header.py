import pandas as pd
import re

path = '/home/tr/IOT-SECURITY-ML-MODEL/43.1.labeled'
raw_cols = None
with open(path, 'r') as fh:
    for line in fh:
        if line.startswith('#fields'):
            raw_cols = line.strip().split('\t')[1:]
            cols = []
            for rc in raw_cols:
                parts = re.split(r"\s+", rc.strip())
                for p in parts:
                    if p:
                        cols.append(p)
            break
if raw_cols is None:
    raise RuntimeError('no #fields')

print('raw_cols =', raw_cols)
print('cols =', cols)

# read a small number of rows with these columns
try:
    df = pd.read_csv(path, sep='\t', comment='#', names=cols, header=None, nrows=20)
except Exception as e:
    print('read_csv failed:', e)
    raise

print('\nParsed columns:')
print(df.columns.tolist())
print('\nData sample:')
print(df.head().to_string(index=False))

candidates = ['det_label','detailed-label','detailed_label','label']
for c in candidates:
    print('\n-- Checking column:', c)
    if c in df.columns:
        print('unique values (sample up to 10):', pd.Series(df[c].unique()).astype(str).tolist()[:10])
    else:
        print('not present')

# Also print any oddly long column names
long_cols = [c for c in df.columns if ' ' in c or '\\t' in c or len(c) > 30]
print('\nColumns with spaces or long names:')
print(long_cols)
