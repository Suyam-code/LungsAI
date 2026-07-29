import pandas as pd
import os

BASE = os.path.expanduser('~/Lung-Disease-Diagnostic-AI')
CSV = os.path.join(BASE, 'data/raw/archive-2/train.csv')
df = pd.read_csv(CSV)

# Get set of lateral image basenames
lateral_basenames = set()
for path in df[df['Frontal/Lateral'] == 'Lateral']['Path']:
    lateral_basenames.add(os.path.basename(path))

print(f"Unique lateral image basenames: {len(lateral_basenames)}")

# Delete from Lung_Cancer and Normal
folders = [
    os.path.join(BASE, 'data/processed_large/Lung_Cancer'),
    os.path.join(BASE, 'data/processed_large/Normal')
]

removed = 0
for folder in folders:
    if not os.path.exists(folder):
        continue
    for file in os.listdir(folder):
        if file in lateral_basenames:
            os.remove(os.path.join(folder, file))
            removed += 1
print(f"Removed {removed} lateral images.")
