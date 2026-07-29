import pandas as pd
import shutil
import os

# Paths
csv_path = "data/raw/nih_resized/Data_Entry_2017.csv"
base_dir = "data/raw/nih_resized/images-224/images-224"
output_dir = "data/processed_large/Lung_Cancer"

os.makedirs(output_dir, exist_ok=True)

# Load CSV
df = pd.read_csv(csv_path)

# Filter lung cancer
lung_df = df[df["Finding Labels"].str.contains("Mass|Nodule", na=False)]

# 🔥 Create set ONCE (fast lookup)
lung_set = set(lung_df["Image Index"])

print("Total Lung Cancer Images Found:", len(lung_set))

count = 0
limit = 5000

# Walk through all images
for file in os.listdir(base_dir):
    if file in lung_set:
        src = os.path.join(base_dir, file)
        dst = os.path.join(output_dir, file)

        shutil.copy(src, dst)
        count += 1

        if count % 100 == 0:
            print(f"Copied {count} images")

        if count >= limit:
            break

print("✅ Done! Total copied:", count)