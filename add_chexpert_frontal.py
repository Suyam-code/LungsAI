import pandas as pd
import os
import shutil

BASE = os.path.expanduser('~/Lung-Disease-Diagnostic-AI')
CHEXPERT_CSV = os.path.join(BASE, 'data/raw/archive-2/train.csv')
IMG_DIR = os.path.join(BASE, 'data/raw/archive-2/train')
DST_CANCER = os.path.join(BASE, 'data/processed_large/Lung_Cancer')
DST_NORMAL = os.path.join(BASE, 'data/processed_large/Normal')

df = pd.read_csv(CHEXPERT_CSV)
# Keep only frontal images
df = df[df['Frontal/Lateral'] == 'Frontal']

# Lung Cancer (Mass or Nodule)
cancer = df[(df['Mass'] == 1) | (df['Nodule'] == 1)]
print(f"Found {len(cancer)} frontal Lung Cancer images")
for idx, row in cancer.iterrows():
    src = os.path.join(IMG_DIR, row['Path'])
    if os.path.exists(src):
        shutil.copy(src, DST_CANCER)

# Normal (No Finding) – take up to 10,000 (but we'll balance later)
normal = df[df['No Finding'] == 1]
sample = normal.sample(n=min(10000, len(normal)), random_state=42)
print(f"Copying {len(sample)} frontal Normal images")
for idx, row in sample.iterrows():
    src = os.path.join(IMG_DIR, row['Path'])
    if os.path.exists(src):
        shutil.copy(src, DST_NORMAL)

print("Done.")
