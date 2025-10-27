"""Visualize `balanced_multiclass.csv`.
Produces:
- label_counts.png (bar)
- numeric_histograms.png (grid of histograms for numeric cols)
- correlation_heatmap.png (pearson corr heatmap)
- pairplot_sample.png (seaborn pairplot for small sample)

Usage: python3 scripts/visualize_balanced.py
"""
import os
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
"""Visualize `balanced_multiclass.csv`.
Produces:
- label_counts.png (bar)
- numeric_histograms.png (grid of histograms for numeric cols)
- correlation_heatmap.png (pearson corr heatmap)
- pairplot_sample.png (seaborn pairplot for small sample, optional)

Usage: python3 scripts/visualize_balanced.py
"""
import os
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    _HAS_SEABORN = True
except Exception:
    sns = None
    _HAS_SEABORN = False
import numpy as np

DATA = Path('balanced_multiclass.csv')
OUT_DIR = Path('plots')
OUT_DIR.mkdir(exist_ok=True)

if not DATA.exists():
    raise SystemExit('balanced_multiclass.csv not found. Run collecting script first.')

print('Loading data...')
df = pd.read_csv(DATA)
print('Rows:', len(df))

# Basic label counts
label_counts = df['label'].value_counts()
plt.figure(figsize=(8,6))
if _HAS_SEABORN:
    sns.barplot(x=label_counts.index, y=label_counts.values, palette='tab10')
else:
    plt.bar(label_counts.index, label_counts.values)
plt.ylabel('count')
plt.xlabel('label')
plt.title('Label distribution')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(OUT_DIR / 'label_counts.png')
plt.close()
print('Saved label_counts.png')

# Numeric histograms for key numeric cols
num_cols = []
for c in ['duration','src_bytes','dst_bytes','src_pkts','dst_pkts','src_port','dst_port']:
    if c in df.columns:
        num_cols.append(c)

if num_cols:
    n = len(num_cols)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols*5, rows*3))
    # normalize axes iterable
    if isinstance(axes, (list, np.ndarray)):
        axes_flat = list(np.array(axes).flatten())
    else:
        axes_flat = [axes]
    for i, c in enumerate(num_cols):
        ax = axes_flat[i]
        data = pd.to_numeric(df[c], errors='coerce').dropna()
        if _HAS_SEABORN:
            sns.histplot(np.log1p(data), bins=50, ax=ax)
        else:
            ax.hist(np.log1p(data), bins=50)
        ax.set_title(c + ' (log1p)')
    # remove unused axes
    for j in range(i+1, len(axes_flat)):
        try:
            fig.delaxes(axes_flat[j])
        except Exception:
            pass
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'numeric_histograms.png')
    plt.close()
    print('Saved numeric_histograms.png')

# Correlation heatmap for numeric cols
if num_cols:
    corr = df[num_cols].apply(pd.to_numeric, errors='coerce').corr()
    plt.figure(figsize=(6,5))
    if _HAS_SEABORN:
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='vlag', center=0)
    else:
        plt.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
        plt.colorbar()
        plt.xticks(range(len(corr.columns)), corr.columns, rotation=45)
        plt.yticks(range(len(corr.index)), corr.index)
    plt.title('Correlation (numeric cols)')
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'correlation_heatmap.png')
    plt.close()
    print('Saved correlation_heatmap.png')

# Pairplot on a small sample (if seaborn available)
sample_size = min(1000, len(df))
if _HAS_SEABORN and sample_size > 1 and len(num_cols) >= 2:
    sample = df[num_cols].dropna().sample(sample_size, random_state=42)
    try:
        g = sns.pairplot(sample[num_cols].apply(pd.to_numeric, errors='coerce'))
        g.fig.suptitle('Pairplot (sample)')
        g.fig.tight_layout()
        g.fig.savefig(OUT_DIR / 'pairplot_sample.png')
        plt.close()
        print('Saved pairplot_sample.png')
    except Exception as e:
        print('Pairplot failed:', e)
else:
    if not _HAS_SEABORN:
        print('Seaborn not installed; skipping pairplot.')

print('All plots saved in', OUT_DIR)
