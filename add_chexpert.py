import pandas as pd
import os
import shutil

BASE = os.path.expanduser('~/Lung-Disease-Diagnostic-AI')
CHEXPERT_CSV = os.path.join(BASE, 'data/raw/archive-2/train.csv')
IMG_DIR = os.path.join(BASE, 'data/raw/archive-2/train')
DST_CANCER = os.path.join(BASE, 'data/processed_large/Lung_Cancer')
DST_NORMAL = os.path.join(BASE, 'data/processed_large/Normal')

df = pd.read_csv(CHEXPERT_CSV)

# For CheXpert‑small, lung cancer proxy is 'Lung Lesion' == 1
cancer = df[df['Lung Lesion'] == 1]
print(f"Found {len(cancer)} images with Lung Lesion")
for idx, row in cancer.iterrows():
    src = os.path.join(IMG_DIR, row['Path'])
    if os.path.exists(src):
        shutil.copy(src, DST_CANCER)

# Normal: No Finding == 1
normal = df[df['No Finding'] == 1]
sample = normal.sample(n=min(10000, len(normal)), random_state=42)
print(f"Copying {len(sample)} Normal images")
for idx, row in sample.iterrows():
    src = os.path.join(IMG_DIR, row['Path'])
    if os.path.exists(src):
        shutil.copy(src, DST_NORMAL)

print("Done.")
