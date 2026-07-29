"""
v3: two-stage training with fine-tuning of the MobileNetV2 backbone.

Why fine-tune at all? In v2 the training and validation accuracy were almost
identical (0.48 vs 0.47), which is the signature of UNDERFITTING, not
overfitting. The model was not memorising - it simply could not extract enough
signal. The likely cause is that the frozen ImageNet features were learned on
colour photographs of everyday objects, and grayscale chest radiographs are a
very different domain. The fix for underfitting is more capacity and adaptation,
not more regularisation, so here the top of the backbone is allowed to adapt.

Two stages, in this order for a reason:

  Stage 1 - backbone frozen, train only the head.
      The head starts with random weights. If the backbone were unfrozen from
      the start, the large, noisy gradients coming from that random head would
      flow back and damage the pretrained features before they were ever useful.
      Warming up the head first keeps those gradients small.

  Stage 2 - unfreeze the top layers, train everything at a much lower rate.
      A low learning rate (1e-5) is essential: the pretrained weights are
      already good, and we want to nudge them, not overwrite them.

The evaluation is identical to train_v2.py so the two runs are directly
comparable. v2's model file is left untouched so the baseline is preserved.

Run from the project root:  python src/train_v3_finetune.py
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
MODEL_OUT = os.path.join(BASE_DIR, "models", "v3_finetuned.keras")
METRICS_OUT = os.path.join(BASE_DIR, "models", "v3_metrics.json")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

STAGE1_EPOCHS = 10      # head warm-up
STAGE2_EPOCHS = 25      # fine-tuning
UNFREEZE_FROM = 100     # MobileNetV2 has 154 layers; unfreeze the top ~54


def make_generators():
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=10,
        zoom_range=0.1,
        width_shift_range=0.05,
        height_shift_range=0.05,
        horizontal_flip=True,
    )
    eval_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_gen = train_datagen.flow_from_directory(
        os.path.join(DATA_DIR, "train"), target_size=IMG_SIZE,
        batch_size=BATCH_SIZE, class_mode="categorical", shuffle=True, seed=SEED,
    )
    val_gen = eval_datagen.flow_from_directory(
        os.path.join(DATA_DIR, "val"), target_size=IMG_SIZE,
        batch_size=BATCH_SIZE, class_mode="categorical", shuffle=False,
    )
    test_gen = eval_datagen.flow_from_directory(
        os.path.join(DATA_DIR, "test"), target_size=IMG_SIZE,
        batch_size=BATCH_SIZE, class_mode="categorical", shuffle=False,
    )
    return train_gen, val_gen, test_gen


def evaluate(model, test_gen, class_names, tag):
    print("\n" + "=" * 62)
    print(f"TEST SET EVALUATION [{tag}] (patient-disjoint, never seen in training)")
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
    print(f"\nTest accuracy: {acc * 100:.2f}%   (chance = 33.3%)")
    print(f"v2 frozen-backbone baseline was 42.41% - compare against that.")

    report = classification_report(
        y_true, y_pred, target_names=class_names, digits=3, output_dict=True
    )
    return acc, report, cm


def main():
    train_gen, val_gen, test_gen = make_generators()
    class_names = list(train_gen.class_indices.keys())
    print("Class mapping:", train_gen.class_indices)

    # ---------------- build ------------------------------------------------
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

    # ---------------- stage 1: warm up the head ----------------------------
    print("\n### STAGE 1: frozen backbone, training head only ###")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(
        train_gen, validation_data=val_gen, epochs=STAGE1_EPOCHS,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=4, restore_best_weights=True
            )
        ],
    )

    # ---------------- stage 2: fine-tune the top of the backbone -----------
    print(f"\n### STAGE 2: unfreezing layers from index {UNFREEZE_FROM} ###")
    base.trainable = True

    for layer in base.layers[:UNFREEZE_FROM]:
        layer.trainable = False

    # Keep every BatchNormalization layer in the backbone frozen (inference
    # mode). Their running mean/variance came from ImageNet's large batches;
    # updating them here with batches of 32 medical images would produce noisy
    # statistics and can destabilise or destroy the pretrained features. This is
    # the standard practice when fine-tuning, and it is easy to get wrong.
    frozen_bn = 0
    for layer in base.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
            frozen_bn += 1

    trainable = sum(1 for l in base.layers if l.trainable)
    print(f"Backbone layers: {len(base.layers)}, trainable: {trainable}, "
          f"BatchNorm kept frozen: {frozen_bn}")

    # Much lower learning rate: nudge the pretrained weights, do not overwrite.
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(
        train_gen, validation_data=val_gen, epochs=STAGE2_EPOCHS,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=5, restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", patience=3, factor=0.3
            ),
        ],
    )

    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    model.save(MODEL_OUT)
    print(f"\nSaved model to {MODEL_OUT}")

    acc, report, cm = evaluate(model, test_gen, class_names, "v3 fine-tuned")

    with open(METRICS_OUT, "w") as f:
        json.dump(
            {
                "test_accuracy": acc,
                "chance_baseline": 1 / len(class_names),
                "v2_frozen_baseline": 0.4241,
                "class_names": class_names,
                "classification_report": report,
                "confusion_matrix": cm.tolist(),
                "n_test_images": int(len(test_gen.classes)),
                "notes": (
                    "Two-stage: head warm-up then fine-tuning of the top "
                    f"{154 - UNFREEZE_FROM} MobileNetV2 layers at lr=1e-5, "
                    "BatchNorm layers kept frozen. Splits patient-disjoint, "
                    "all data from NIH ChestX-ray14."
                ),
            },
            f,
            indent=2,
        )
    print(f"Metrics written to {METRICS_OUT}")


if __name__ == "__main__":
    main()
