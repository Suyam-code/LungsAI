from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import numpy as np
import os

input_dir = "data/processed_clean/Pneumonia"
target_count = 5000

datagen = ImageDataGenerator(
    rotation_range=8,
    width_shift_range=0.05,
    height_shift_range=0.05,
    zoom_range=0.05,
    horizontal_flip=True,
    fill_mode="nearest"
)

files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
count = len(files)

print("Current count:", count)

i = 0
while count < target_count:
    img_name = files[i % len(files)]
    img_path = os.path.join(input_dir, img_name)

    img = load_img(img_path, target_size=(224, 224))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)

    for _ in datagen.flow(
        x,
        batch_size=1,
        save_to_dir=input_dir,
        save_prefix="aug",
        save_format="png"
    ):
        count += 1
        if count % 100 == 0:
            print("Generated:", count)
        break

    i += 1

print("Done. Final count:", count)
