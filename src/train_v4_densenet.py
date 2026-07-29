"""
v4: DenseNet121 on the same patient-disjoint 3-class NIH split.

Why this experiment:

  DenseNet121 is the architecture used by CheXNet, the reference model for
  NIH ChestX-ray14, so it is the natural architecture to try on this dataset.
  It is also a useful CONTROL. v2 (frozen MobileNetV2) scored 42.41% and v3
  (fine-tuned MobileNetV2) scored 40.96% - statistically indistinguishable,
  which suggested the bottleneck is the data rather than the model. If a second,
  architecturally different network also lands near 42%, that conclusion becomes
  much harder to argue with. If DenseNet121 does noticeably better, the
  architecture mattered after all and the earlier conclusion was wrong.

  Either outcome is informative, which is what makes it worth running.

Everything else is held constant on purpose - same splits, same augmentation,
same evaluation - so the architecture is the only variable that changed.

Run from the project root:  python src/train_v4_densenet.py
"""

import json
import os

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import layers, models
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

SEED = 42
tf.keras.utils.set_random_seed(SEED)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "nih_3class")
MODEL_OUT = os.path.join(BASE_DIR, "models", "v4_densenet121.keras")
METRICS_OUT = os.path.join(BASE_DIR, "models", "v4_metrics.json")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
STAGE1_EPOCHS = 12      # head warm-up, backbone frozen
STAGE2_EPOCHS = 20      # fine-tune the last dense block
UNFREEZE_FROM = 313     # DenseNet121 has 427 layers; this frees conv5_* (last block)

# Results from the earlier runs, for direct comparison in the output.
PRIOR = {"v2_mobilenet_frozen": 0.4241, "v3_mobilenet_finetuned": 0.4096}


def make_generators():
    """Identical to the MobileNetV2 runs apart from DenseNet's preprocess_input."""
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


def main():
    train_gen, val_gen, test_gen = make_generators()
    class_names = list(train_gen.class_indices.keys())
    print("Class mapping:", train_gen.class_indices)

    base = DenseNet121(weights="imagenet", include_top=False, input_shape=(*IMG_SIZE, 3))
    base.trainable = False
    print(f"DenseNet121 loaded: {len(base.layers)} layers")

    model = models.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(len(class_names), activation="softmax"),
    ])

    # ---- stage 1: warm up the randomly initialised head ------------------
    print("\n### STAGE 1: frozen backbone, head only ###")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="categorical_crossentropy", metrics=["accuracy"],
    )
    model.fit(
        train_gen, validation_data=val_gen, epochs=STAGE1_EPOCHS,
        callbacks=[tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=4, restore_best_weights=True)],
    )

    # ---- stage 2: fine-tune the final dense block ------------------------
    print(f"\n### STAGE 2: unfreezing from layer {UNFREEZE_FROM} ###")
    base.trainable = True
    for layer in base.layers[:UNFREEZE_FROM]:
        layer.trainable = False

    # Same reasoning as v3: BatchNorm running statistics come from ImageNet's
    # large batches. Updating them on batches of 32 X-rays makes them noisy and
    # can destabilise the pretrained features, so they stay in inference mode.
    frozen_bn = 0
    for layer in base.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
            frozen_bn += 1

    print(f"Trainable backbone layers: {sum(1 for l in base.layers if l.trainable)}, "
          f"BatchNorm frozen: {frozen_bn}")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss="categorical_crossentropy", metrics=["accuracy"],
    )
    model.fit(
        train_gen, validation_data=val_gen, epochs=STAGE2_EPOCHS,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=5, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", patience=3, factor=0.3),
        ],
    )

    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    model.save(MODEL_OUT)
    print(f"\nSaved model to {MODEL_OUT}")

    # ---- evaluation ------------------------------------------------------
    print("\n" + "=" * 62)
    print("TEST SET EVALUATION [v4 DenseNet121] (patient-disjoint)")
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
    # Standard error of a proportion, so the comparison against the earlier runs
    # is made on statistical grounds rather than by eyeballing the numbers.
    se = float(np.sqrt(acc * (1 - acc) / n))

    print(f"\nTest accuracy: {acc * 100:.2f}%  (+/- {se * 100:.2f}% SE, n={n})")
    print(f"chance                     : 33.33%")
    for k, v in PRIOR.items():
        print(f"{k:27s}: {v * 100:.2f}%")
    print("\nDifferences smaller than roughly 2 standard errors (~2.7 points) are "
          "not meaningful at this sample size.")

    with open(METRICS_OUT, "w") as f:
        json.dump(
            {
                "test_accuracy": acc,
                "standard_error": se,
                "chance_baseline": 1 / len(class_names),
                "prior_runs": PRIOR,
                "class_names": class_names,
                "classification_report": classification_report(
                    y_true, y_pred, target_names=class_names,
                    digits=3, output_dict=True),
                "confusion_matrix": cm.tolist(),
                "n_test_images": int(n),
                "notes": (
                    "DenseNet121 (CheXNet architecture) control experiment. "
                    "Same patient-disjoint NIH 3-class splits, same augmentation "
                    "and evaluation as v2/v3 - architecture is the only variable."
                ),
            },
            f, indent=2,
        )
    print(f"Metrics written to {METRICS_OUT}")


if __name__ == "__main__":
    main()
