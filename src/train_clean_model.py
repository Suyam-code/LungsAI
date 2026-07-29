import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

data_dir = "data/processed_clean"

train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=10,
    zoom_range=0.1,
    horizontal_flip=True
)

train_gen = train_datagen.flow_from_directory(
    data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    subset="training",
    shuffle=True
)

val_gen = train_datagen.flow_from_directory(
    data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)

print("Class indices:", train_gen.class_indices)

base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(4, activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        patience=2,
        factor=0.3
    ),
    tf.keras.callbacks.ModelCheckpoint(
        "models/final_clean_model.keras",
        monitor="val_accuracy",
        save_best_only=True,
        mode="max"
    )
]

# Increase penalty for missing Lung_Cancer
# Expected mapping:
# {'COVID': 0, 'Lung_Cancer': 1, 'Normal': 2, 'Pneumonia': 3}
class_weights = {
    0: 1.0,
    1: 1.8,
    2: 1.0,
    3: 1.0
}

print("Training with class weights:", class_weights)

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=12,
    callbacks=callbacks,
    class_weight=class_weights
)

print("Best model saved to models/final_clean_model.keras")