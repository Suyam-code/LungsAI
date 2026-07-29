"""
Build the v2 dataset: NIH ChestX-ray14 only, three mutually exclusive classes,
split by PATIENT so no patient appears in more than one split.

This script exists because of two flaws found in v1:

  1. v1 split the data randomly by file. NIH has multiple scans per patient
     (up to 30), so the same patient landed in both train and validation and the
     model could score well by memorising anatomy it had already seen.
  2. v1 drew each class from a different source dataset, so the model could
     learn "which dataset is this" instead of "what is wrong with the patient".

v2 fixes both: every image comes from NIH, and splits are patient-disjoint.

Output layout (ready for flow_from_directory):

    data/nih_3class/
      train/{No_Finding,Pneumonia,Mass_Nodule}/
      val/{...}/
      test/{...}/

Run from the project root:  python src/build_dataset_v2.py
"""

import csv
import os
import random
import shutil
from collections import defaultdict

# ---------------------------------------------------------------- config

SEED = 42

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NIH_DIR = os.path.join(BASE_DIR, "data", "raw", "nih_resized")
CSV_PATH = os.path.join(NIH_DIR, "Data_Entry_2017.csv")
IMAGES_DIR = os.path.join(NIH_DIR, "images-224", "images-224")
TEST_LIST = os.path.join(NIH_DIR, "test_list_NIH.txt")

OUT_DIR = os.path.join(BASE_DIR, "data", "nih_3class")
CLASSES = ["No_Finding", "Pneumonia", "Mass_Nodule"]

# Fraction of training PATIENTS held out for validation.
VAL_PATIENT_FRACTION = 0.15


# ---------------------------------------------------------------- labelling

def assign_class(finding_labels: str):
    """
    Map NIH's multi-label finding string to one of our three classes, or None.

    NIH images can carry several findings at once (e.g. "Mass|Effusion"). To keep
    the problem a clean single-label one, an image is only used when it belongs
    unambiguously to exactly one of our classes:

      No_Finding   -> the label is exactly "No Finding"
      Mass_Nodule  -> contains Mass or Nodule, and NOT Pneumonia
      Pneumonia    -> contains Pneumonia, and NOT Mass or Nodule

    Everything else (other pathologies, or images that are both) is dropped.
    Note: "Mass" and "Nodule" are radiological findings, not a cancer diagnosis -
    that is deliberately reflected in the class name.
    """
    labels = set(finding_labels.split("|"))

    if labels == {"No Finding"}:
        return "No_Finding"

    has_mass_nodule = bool(labels & {"Mass", "Nodule"})
    has_pneumonia = "Pneumonia" in labels

    if has_mass_nodule and not has_pneumonia:
        return "Mass_Nodule"
    if has_pneumonia and not has_mass_nodule:
        return "Pneumonia"
    return None


# ---------------------------------------------------------------- helpers

def balance(records, rng):
    """
    Downsample every class to the size of the smallest one.

    Balancing makes the metrics directly interpretable: with three equal classes
    chance accuracy is 33%, so any score above that is real signal rather than an
    artefact of one class dominating.
    """
    by_class = defaultdict(list)
    for rec in records:
        by_class[rec["cls"]].append(rec)

    target = min(len(v) for v in by_class.values())

    out = []
    for cls in CLASSES:
        items = by_class[cls]
        rng.shuffle(items)
        out.extend(items[:target])
    return out, target


def summarise(name, records):
    counts = defaultdict(int)
    patients = defaultdict(set)
    for r in records:
        counts[r["cls"]] += 1
        patients[r["cls"]].add(r["patient"])
    total_patients = len({r["patient"] for r in records})
    print(f"\n{name}: {len(records)} images, {total_patients} patients")
    for cls in CLASSES:
        print(f"   {cls:12s} {counts[cls]:>5d} images  ({len(patients[cls])} patients)")


# ---------------------------------------------------------------- main

def main():
    rng = random.Random(SEED)

    # ---- 1. read labels ------------------------------------------------
    records = []
    with open(CSV_PATH) as f:
        for row in csv.DictReader(f):
            cls = assign_class(row["Finding Labels"])
            if cls is None:
                continue
            records.append(
                {
                    "image": row["Image Index"],
                    "cls": cls,
                    "patient": row["Patient ID"],
                }
            )
    print(f"Usable images after class assignment: {len(records)}")

    # ---- 2. official NIH test split ------------------------------------
    # NIH publishes a test list that is patient-disjoint from the training data.
    # Using their split rather than inventing one means the held-out set is
    # trustworthy by construction.
    with open(TEST_LIST) as f:
        official_test = {line.strip() for line in f if line.strip()}

    trainval_pool = [r for r in records if r["image"] not in official_test]
    test_pool = [r for r in records if r["image"] in official_test]
    print(f"train+val pool: {len(trainval_pool)}   official test pool: {len(test_pool)}")

    # ---- 3. patient-grouped train/val split ----------------------------
    # Split on PATIENTS, not images. This is the fix for the v1 leakage bug:
    # every scan belonging to a patient moves together into one split.
    patients = sorted({r["patient"] for r in trainval_pool})
    rng.shuffle(patients)
    n_val = int(len(patients) * VAL_PATIENT_FRACTION)
    val_patients = set(patients[:n_val])

    train_records = [r for r in trainval_pool if r["patient"] not in val_patients]
    val_records = [r for r in trainval_pool if r["patient"] in val_patients]

    # ---- 4. balance each split -----------------------------------------
    train_records, n_train = balance(train_records, rng)
    val_records, n_val_each = balance(val_records, rng)
    test_records, n_test = balance(test_pool, rng)

    splits = {"train": train_records, "val": val_records, "test": test_records}
    for name, recs in splits.items():
        summarise(name, recs)

    # ---- 5. verify no patient crosses splits ---------------------------
    # This is the assertion the whole rebuild exists for. If it ever fails, the
    # results are not trustworthy and the run should be thrown away.
    patient_sets = {name: {r["patient"] for r in recs} for name, recs in splits.items()}
    print("\n--- patient overlap check ---")
    ok = True
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        overlap = patient_sets[a] & patient_sets[b]
        print(f"{a} ∩ {b}: {len(overlap)} patients")
        if overlap:
            ok = False
    if not ok:
        raise SystemExit("FAILED: patients appear in more than one split.")
    print("PASSED: all splits are patient-disjoint.")

    # ---- 6. write files -------------------------------------------------
    if os.path.exists(OUT_DIR):
        shutil.rmtree(OUT_DIR)

    manifest_rows = []
    missing = 0
    for split, recs in splits.items():
        for cls in CLASSES:
            os.makedirs(os.path.join(OUT_DIR, split, cls), exist_ok=True)
        for r in recs:
            src = os.path.join(IMAGES_DIR, r["image"])
            if not os.path.exists(src):
                missing += 1
                continue
            dst = os.path.join(OUT_DIR, split, r["cls"], r["image"])
            shutil.copyfile(src, dst)
            manifest_rows.append([split, r["cls"], r["patient"], r["image"]])

    with open(os.path.join(OUT_DIR, "manifest.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "class", "patient_id", "image"])
        w.writerows(manifest_rows)

    print(f"\nWrote {len(manifest_rows)} images to {OUT_DIR}")
    if missing:
        print(f"WARNING: {missing} images listed in the CSV were not found on disk.")
    print("Manifest: data/nih_3class/manifest.csv")


if __name__ == "__main__":
    main()
