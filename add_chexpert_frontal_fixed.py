import pandas as pd
import os
import shutil

BASE = os.path.expanduser('~/Lung-Disease-Diagnostic-AI')
CSV_PATH = os.path.join(BASE, 'data/raw/archive-2/train.csv')
IMG_DIR = os.path.join(BASE, 'data/raw/archive-2/train')
DST_CANCER = os.path.join(BASE, 'data/processed_large/Lung_Cancer')
DST_NORMAL = os.path.join(BASE, 'data/processed_large/Normal')

df = pd.read_csv(CSV_PATH)

# Keep only frontal images
df = df[df['Frontal/Lateral'] == 'Frontal']

# Lung Cancer: use 'Lung Lesion' column (1 = present)
cancer = df[df['Lung Lesion'] == 1]
print(f"Found {len(cancer)} frontal Lung Lesion images")
for idx, row in cancer.iterrows():
    src = os.path.join(IMG_DIR, row['Path'])
    if os.path.exists(src):
        shutil.copy(src, DST_CANCER)

# Normal: 'No Finding' = 1
normal = df[df['No Finding'] == 1]
# Take up to 10,000 random normal images (to avoid huge numbers)
sample = normal.sample(n=min(10000, len(normal)), random_state=42)
print(f"Copying {len(sample)} frontal Normal images")
for idx, row in sample.iterrows():
    src = os.path.join(IMG_DIR, row['Path'])
    if os.path.exists(src):
        shutil.copy(src, DST_NORMAL)

print("Done.")
