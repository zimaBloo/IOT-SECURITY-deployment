path = '/home/tr/IOT-SECURITY-ML-MODEL/43.1.labeled'
count = 0
print('Inspecting raw lines (first 20 non-comment):')
with open(path,'r') as fh:
    for line in fh:
        if line.startswith('#'):
            continue
        if line.strip()=='':
            continue
        parts = line.rstrip('\n').split('\t')
        print(f'line {count}: parts={len(parts)} last5={parts[-5:]}')
        count += 1
        if count >= 20:
            break

# Also show the exact characters of last token of first few lines
print('\nInspecting last token raw repr for first 10 non-comment lines:')
count = 0
with open(path,'r') as fh:
    for line in fh:
        if line.startswith('#'):
            continue
        if line.strip()=='':
            continue
        parts = line.rstrip('\n').split('\t')
        last = parts[-1] if parts else ''
        print(f'line {count} last_token_repr={repr(last)}')
        count += 1
        if count >= 10:
            break
