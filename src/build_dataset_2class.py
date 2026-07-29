"""
Build a binary NIH dataset: No_Finding vs Mass_Nodule.

Purpose of this experiment
--------------------------
v2, v3 and v4 all scored 41-44% on the 3-class task despite spanning two
architectures and both frozen and fine-tuned backbones. That convergence
suggested the limiting factor was the DATA (689 training images per class, with
noisy NLP-mined labels), not the model.

To test that properly, the task must stay fixed while only the data volume
changes. Comparing the 3-class result to a 2-class result would prove nothing,
because chance accuracy moves from 33% to 50%.

So this script builds the SAME binary task at different training sizes:

    python src/build_dataset_2class.py --per-class 689     -> small
    python src/build_dataset_2class.py --per-class 5000    -> large

Critically, the validation and test sets are built to a FIXED size and from a
fixed random seed, so they are identical across both runs. Only the training set
changes size. Any difference in test accuracy is then attributable to training
data volume alone.

Output:  data/nih_2class_<per_class>/{train,val,test}/{No_Finding,Mass_Nodule}/
"""

import argparse
import csv
import os
import random
import shutil
from collections import defaultdict

SEED = 42

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NIH_DIR = os.path.join(BASE_DIR, "data", "raw", "nih_resized")
CSV_PATH = os.path.join(NIH_DIR, "Data_Entry_2017.csv")
IMAGES_DIR = os.path.join(NIH_DIR, "images-224", "images-224")
TEST_LIST = os.path.join(NIH_DIR, "test_list_NIH.txt")

CLASSES = ["No_Finding", "Mass_Nodule"]

# Fixed across every run so the two experiments are compared on identical data.
VAL_PER_CLASS = 400
TEST_PER_CLASS = 1000
VAL_PATIENT_FRACTION = 0.15


def assign_class(finding_labels: str):
    """
    Binary labelling. Pneumonia cases are excluded entirely rather than folded
    into either class, so the two groups stay clean and unambiguous.
    """
    labels = set(finding_labels.split("|"))
    if labels == {"No Finding"}:
        return "No_Finding"
    if labels & {"Mass", "Nodule"} and "Pneumonia" not in labels:
        return "Mass_Nodule"
    return None


def take_balanced(records, per_class, rng):
    """
    Take the same number of items from every class.

    The cap is the GLOBAL minimum across classes, not a per-class minimum. Taking
    min(per_class, len(items)) independently per class would silently produce an
    imbalanced set whenever one class runs out first - e.g. asking for 11,000
    when No_Finding has 43,005 available but Mass_Nodule only has 6,880 would
    yield 11,000 vs 6,880. Balance has to hold for the comparison across
    training sizes to mean anything, and for chance accuracy to stay at 50%.
    """
    by_class = defaultdict(list)
    for r in records:
        by_class[r["cls"]].append(r)

    available = {cls: len(by_class[cls]) for cls in CLASSES}
    n = min(per_class, min(available.values()))

    if n < per_class:
        print(f"  NOTE: requested {per_class}/class but only {n} available "
              f"(limited by {min(available, key=available.get)}); "
              f"using {n} for every class to keep the split balanced.")

    out = []
    for cls in CLASSES:
        items = by_class[cls]
        rng.shuffle(items)
        out.extend(items[:n])
    return out, {cls: n for cls in CLASSES}


def summarise(name, records):
    counts = defaultdict(int)
    pats = defaultdict(set)
    for r in records:
        counts[r["cls"]] += 1
        pats[r["cls"]].add(r["patient"])
    print(f"\n{name}: {len(records)} images, {len({r['patient'] for r in records})} patients")
    for cls in CLASSES:
        print(f"   {cls:12s} {counts[cls]:>6d} images  ({len(pats[cls])} patients)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, required=True,
                    help="training images per class (e.g. 689 or 5000)")
    args = ap.parse_args()

    out_dir = os.path.join(BASE_DIR, "data", f"nih_2class_{args.per_class}")
    rng = random.Random(SEED)

    # ---- read labels ----------------------------------------------------
    records = []
    with open(CSV_PATH) as f:
        for row in csv.DictReader(f):
            cls = assign_class(row["Finding Labels"])
            if cls:
                records.append({"image": row["Image Index"], "cls": cls,
                                "patient": row["Patient ID"]})
    print(f"Usable images: {len(records)}")

    # ---- official NIH test split (patient-disjoint by construction) ------
    with open(TEST_LIST) as f:
        official_test = {l.strip() for l in f if l.strip()}
    trainval_pool = [r for r in records if r["image"] not in official_test]
    test_pool = [r for r in records if r["image"] in official_test]

    # ---- patient-grouped train/val split --------------------------------
    patients = sorted({r["patient"] for r in trainval_pool})
    rng.shuffle(patients)
    val_patients = set(patients[:int(len(patients) * VAL_PATIENT_FRACTION)])

    train_pool = [r for r in trainval_pool if r["patient"] not in val_patients]
    val_pool = [r for r in trainval_pool if r["patient"] in val_patients]

    # ---- sample -----------------------------------------------------------
    # A fresh RNG with the same seed for val and test guarantees those two sets
    # are byte-for-byte identical across runs with different --per-class values.
    train_records, train_n = take_balanced(train_pool, args.per_class, rng)
    val_records, val_n = take_balanced(val_pool, VAL_PER_CLASS, random.Random(SEED))
    test_records, test_n = take_balanced(test_pool, TEST_PER_CLASS, random.Random(SEED))

    splits = {"train": train_records, "val": val_records, "test": test_records}
    for name, recs in splits.items():
        summarise(name, recs)

    # ---- patient-disjointness assertion ----------------------------------
    ps = {n: {r["patient"] for r in recs} for n, recs in splits.items()}
    print("\n--- patient overlap check ---")
    ok = True
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        n = len(ps[a] & ps[b])
        print(f"{a} ∩ {b}: {n} patients")
        ok = ok and n == 0
    if not ok:
        raise SystemExit("FAILED: patients appear in more than one split.")
    print("PASSED: all splits are patient-disjoint.")

    # ---- write ------------------------------------------------------------
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)

    written, missing = 0, 0
    rows = []
    for split, recs in splits.items():
        for cls in CLASSES:
            os.makedirs(os.path.join(out_dir, split, cls), exist_ok=True)
        for r in recs:
            src = os.path.join(IMAGES_DIR, r["image"])
            if not os.path.exists(src):
                missing += 1
                continue
            shutil.copyfile(src, os.path.join(out_dir, split, r["cls"], r["image"]))
            rows.append([split, r["cls"], r["patient"], r["image"]])
            written += 1

    with open(os.path.join(out_dir, "manifest.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "class", "patient_id", "image"])
        w.writerows(rows)

    print(f"\nWrote {written} images to {out_dir}")
    if missing:
        print(f"WARNING: {missing} images not found on disk.")


if __name__ == "__main__":
    main()
