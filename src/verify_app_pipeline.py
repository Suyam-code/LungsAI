"""
Verify that the app's inference path reproduces the recorded test metrics.

Why this exists: streamlit_app.py rebuilds preprocessing by hand (PIL resize ->
preprocess_input) rather than going through Keras' ImageDataGenerator. If that
path drifts from the one used in training - wrong scaling, wrong channel order,
wrong class order - nothing crashes. Predictions just get quietly worse, and a
degraded model looks identical to a working one from the outside.

This script runs the app's exact classify() logic over the held-out test set and
compares the resulting accuracy to the number recorded at training time. If they
match, the deployed path is sound.

Run from the project root:  python src/verify_app_pipeline.py
"""

import json
import os
import sys

import numpy as np
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from streamlit_app import CLASSES, classify  # noqa: E402  (uses the app's own code)

TEST_DIR = os.path.join(BASE_DIR, "data", "nih_2class_6880", "test")
METRICS = os.path.join(BASE_DIR, "models", "binary_6880_metrics.json")

# Sampling rather than scoring all 2000 keeps this quick; 300 images gives a
# standard error of ~2.8 points, which is tight enough to catch a broken
# pipeline (which would collapse accuracy toward 50%).
PER_CLASS = 150


def main():
    with open(METRICS) as f:
        recorded = json.load(f)["test_accuracy"]

    correct = total = 0
    for true_cls in CLASSES:
        folder = os.path.join(TEST_DIR, true_cls)
        files = sorted(os.listdir(folder))[:PER_CLASS]
        for name in files:
            conf = classify(Image.open(os.path.join(folder, name)))
            if max(conf, key=conf.get) == true_cls:
                correct += 1
            total += 1
        print(f"  scored {len(files)} {true_cls} images")

    acc = correct / total
    se = np.sqrt(acc * (1 - acc) / total)

    print(f"\napp pipeline accuracy : {acc * 100:.2f}%  (+/- {se * 100:.2f}% SE, n={total})")
    print(f"recorded at training  : {recorded * 100:.2f}%")
    print(f"difference            : {abs(acc - recorded) * 100:.2f} points")

    # Two SE of the sample plus a little slack for the smaller sample size.
    if abs(acc - recorded) < 3 * se:
        print("\nPASS - the app's inference path matches training.")
    else:
        print("\nFAIL - the app's preprocessing has drifted from training. "
              "Check scaling (preprocess_input vs /255), channel order, and "
              "that CLASSES is in the same order as training.")
        sys.exit(1)


if __name__ == "__main__":
    main()
