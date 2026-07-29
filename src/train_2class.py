"""
Train the binary No_Finding vs Mass_Nodule model at a given training-set size.

This is the data-volume experiment. Architecture, augmentation, schedule,
validation set and test set are all held constant; the only thing that changes
between runs is how many training images per class are available.

    python src/train_2class.py --per-class 689
    python src/train_2class.py --per-class 5000

Architecture is MobileNetV2 with a frozen backbone - the v2 configuration. It is
used here because it was statistically tied with the larger DenseNet121 on the
3-class task and trains far faster, so it is the sensible control.

Metrics reported:
  * accuracy, with its standard error (chance = 50% for a balanced binary task)
  * ROC-AUC, which is threshold-independent and the standard measure for binary
    medical classification - accuracy alone hides how well the model ranks cases
  * full confusion matrix and per-class precision/recall
"""

import argparse
import json
import os

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

SEED = 42
tf.keras.utils.set_random_seed(SEED)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 30


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, required=True)
    args = ap.parse_args()

    data_dir = os.path.join(BASE_DIR, "data", f"nih_2class_{args.per_class}")
    model_out = os.path.join(BASE_DIR, "models", f"binary_{args.per_class}.keras")
    metrics_out = os.path.join(BASE_DIR, "models", f"binary_{args.per_class}_metrics.json")

    if not os.path.isdir(data_dir):
        raise SystemExit(
            f"{data_dir} not found. Run:\n"
            f"  python src/build_dataset_2class.py --per-class {args.per_class}"
        )

    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=10, zoom_range=0.1,
        width_shift_range=0.05, height_shift_range=0.05,
        horizontal_flip=True,
    )
    eval_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_gen = train_datagen.flow_from_directory(
        os.path.join(data_dir, "train"), target_size=IMG_SIZE,
        batch_size=BATCH_SIZE, class_mode="categorical", shuffle=True, seed=SEED)
    val_gen = eval_datagen.flow_from_directory(
        os.path.join(data_dir, "val"), target_size=IMG_SIZE,
        batch_size=BATCH_SIZE, class_mode="categorical", shuffle=False)
    test_gen = eval_datagen.flow_from_directory(
        os.path.join(data_dir, "test"), target_size=IMG_SIZE,
        batch_size=BATCH_SIZE, class_mode="categorical", shuffle=False)

    class_names = list(train_gen.class_indices.keys())
    print("Class mapping:", train_gen.class_indices)

    base = MobileNetV2(weights="imagenet", include_top=False, input_shape=(*IMG_SIZE, 3))
    base.trainable = False

    model = models.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(len(class_names), activation="softmax"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                  loss="categorical_crossentropy", metrics=["accuracy"])

    model.fit(
        train_gen, validation_data=val_gen, epochs=EPOCHS,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=5, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", patience=3, factor=0.3),
        ],
    )

    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    model.save(model_out)

    # ---------------- evaluation ------------------------------------------
    print("\n" + "=" * 62)
    print(f"TEST EVALUATION - binary, {args.per_class} train images/class")
    print("=" * 62)

    probs = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(probs, axis=1)
    y_true = test_gen.classes

    print("\n" + classification_report(y_true, y_pred, target_names=class_names, digits=3))

    cm = confusion_matrix(y_true, y_pred)
    print("Confusion matrix (rows = true, cols = predicted)")
    print(f"{'':14s}" + "".join(f"{c:>14s}" for c in class_names))
    for name, row in zip(class_names, cm):
        print(f"{name:14s}" + "".join(f"{v:>14d}" for v in row))

    acc = float((y_pred == y_true).mean())
    n = len(y_true)
    se = float(np.sqrt(acc * (1 - acc) / n))

    # AUC on the probability of the positive class. Unlike accuracy this does
    # not depend on the 0.5 decision threshold, so it measures how well the
    # model ranks cases rather than how well one arbitrary cut-off performs.
    pos_index = class_names.index("Mass_Nodule")
    auc = float(roc_auc_score((y_true == pos_index).astype(int), probs[:, pos_index]))

    print(f"\nTest accuracy: {acc * 100:.2f}%  (+/- {se * 100:.2f}% SE, n={n})")
    print(f"ROC-AUC:       {auc:.3f}   (0.5 = no better than chance)")
    print(f"chance accuracy for balanced binary: 50.00%")

    with open(metrics_out, "w") as f:
        json.dump({
            "train_images_per_class": args.per_class,
            "test_accuracy": acc,
            "standard_error": se,
            "roc_auc": auc,
            "chance_baseline": 0.5,
            "class_names": class_names,
            "classification_report": classification_report(
                y_true, y_pred, target_names=class_names, digits=3, output_dict=True),
            "confusion_matrix": cm.tolist(),
            "n_test_images": int(n),
            "notes": ("Binary No_Finding vs Mass_Nodule, NIH only, patient-disjoint "
                      "splits. Validation and test sets identical across training "
                      "sizes, so only training volume varies."),
        }, f, indent=2)
    print(f"Metrics written to {metrics_out}")


if __name__ == "__main__":
    main()
