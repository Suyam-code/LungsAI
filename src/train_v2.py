"""
Train and honestly evaluate the v2 model.

Differences from v1, and why each one matters:

  * Data comes from build_dataset_v2.py, where splits are grouped by PATIENT.
    v1 split randomly by file, so the same patient appeared in both train and
    validation and the reported accuracy was inflated.
  * Every class comes from NIH ChestX-ray14. v1 drew each class from a different
    dataset, so the model could learn "which dataset is this" instead of the
    pathology.
  * Preprocessing uses MobileNetV2's own preprocess_input (scales to [-1, 1]),
    which is what the pretrained ImageNet weights expect. v1 used a plain
    1/255 rescale, which does not match how the backbone was trained.
  * Evaluation reports a confusion matrix and per-class precision/recall on an
    untouched test set - not one aggregate number on the validation split.

Run from the project root:  python src/train_v2.py
"""

import json
import os

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

SEED = 42
tf.keras.utils.set_random_seed(SEED)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "nih_3class")
MODEL_OUT = os.path.join(BASE_DIR, "models", "v2_nih_3class.keras")
METRICS_OUT = os.path.join(BASE_DIR, "models", "v2_metrics.json")

CLASSES = ["Mass_Nodule", "No_Finding", "Pneumonia"]  # alphabetical: matches Keras
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 25


def make_generators():
    """
    Build the three data generators.

    Augmentation is applied to training data only - never to validation or test,
    because those must reflect the real distribution to give an honest score.
    """
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=10,
        zoom_range=0.1,
        width_shift_range=0.05,
        height_shift_range=0.05,
        horizontal_flip=True,
    )
    # No augmentation here - only the same preprocessing the model expects.
    eval_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_gen = train_datagen.flow_from_directory(
        os.path.join(DATA_DIR, "train"),
        target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode="categorical", shuffle=True, seed=SEED,
    )
    val_gen = eval_datagen.flow_from_directory(
        os.path.join(DATA_DIR, "val"),
        target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode="categorical", shuffle=False,
    )
    test_gen = eval_datagen.flow_from_directory(
        os.path.join(DATA_DIR, "test"),
        target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode="categorical", shuffle=False,
    )
    return train_gen, val_gen, test_gen


def build_model(num_classes):
    """
    MobileNetV2 as a frozen feature extractor plus a small classifier head.

    The backbone stays frozen because the training set is small (689 images per
    class). Fine-tuning millions of parameters on that little data would overfit
    almost immediately; training only the head keeps the parameter count low.
    """
    base = MobileNetV2(weights="imagenet", include_top=False, input_shape=(*IMG_SIZE, 3))
    base.trainable = False

    model = models.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    train_gen, val_gen, test_gen = make_generators()
    class_names = list(train_gen.class_indices.keys())
    print("Class mapping:", train_gen.class_indices)

    model = build_model(len(class_names))

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", patience=3, factor=0.3
        ),
    ]

    # The classes are already balanced by build_dataset_v2.py, so no class
    # weighting is needed - equal counts mean equal influence.
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    model.save(MODEL_OUT)
    print(f"\nSaved model to {MODEL_OUT}")

    # ---------------- honest evaluation on the untouched test set ----------
    print("\n" + "=" * 60)
    print("TEST SET EVALUATION (patient-disjoint, never seen during training)")
    print("=" * 60)

    probs = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(probs, axis=1)
    y_true = test_gen.classes

    report = classification_report(
        y_true, y_pred, target_names=class_names, digits=3, output_dict=True
    )
    print("\n" + classification_report(y_true, y_pred, target_names=class_names, digits=3))

    cm = confusion_matrix(y_true, y_pred)
    print("Confusion matrix (rows = true, cols = predicted)")
    print(f"{'':14s}" + "".join(f"{c:>14s}" for c in class_names))
    for name, row in zip(class_names, cm):
        print(f"{name:14s}" + "".join(f"{v:>14d}" for v in row))

    acc = float((y_pred == y_true).mean())
    # Three balanced classes, so random guessing scores ~33.3%. Stating the
    # baseline stops the headline number being read out of context.
    print(f"\nTest accuracy: {acc * 100:.2f}%   (chance = 33.3% with 3 balanced classes)")

    with open(METRICS_OUT, "w") as f:
        json.dump(
            {
                "test_accuracy": acc,
                "chance_baseline": 1 / len(class_names),
                "class_names": class_names,
                "classification_report": report,
                "confusion_matrix": cm.tolist(),
                "n_test_images": int(len(y_true)),
                "notes": "Splits are patient-disjoint; all data from NIH ChestX-ray14.",
            },
            f,
            indent=2,
        )
    print(f"Metrics written to {METRICS_OUT}")


if __name__ == "__main__":
    main()
