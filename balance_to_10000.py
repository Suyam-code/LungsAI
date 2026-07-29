import os
import random
from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

BASE = os.path.expanduser('~/Lung-Disease-Diagnostic-AI/data/processed_large')
TARGET = 10000

def downsample_class(path, target):
    files = [f for f in os.listdir(path) if f.lower().endswith(('.png','.jpg','.jpeg'))]
    if len(files) > target:
        to_remove = random.sample(files, len(files) - target)
        for f in to_remove:
            os.remove(os.path.join(path, f))
        print(f"{os.path.basename(path)}: reduced to {len(os.listdir(path))}")
    else:
        print(f"{os.path.basename(path)}: already at {len(files)}")

def upsample_class(path, target):
    files = [f for f in os.listdir(path) if f.lower().endswith(('.png','.jpg','.jpeg'))]
    current = len(files)
    if current >= target:
        print(f"{os.path.basename(path)}: already has {current} images")
        return
    needed = target - current
    print(f"{os.path.basename(path)}: generating {needed} augmented images...")

    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    generated = 0
    while generated < needed:
        img_name = random.choice(files)
        img_path = os.path.join(path, img_name)
        img = Image.open(img_path).convert('RGB')
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        flow = datagen.flow(img_array, batch_size=1)
        aug = next(flow)[0]
        aug = (aug * 255).astype(np.uint8)
        new_img = Image.fromarray(aug)
        new_name = f"aug_{generated}_{img_name}"
        new_img.save(os.path.join(path, new_name))
        generated += 1
        if generated % 500 == 0:
            print(f"  Generated {generated}/{needed}")

    print(f"{os.path.basename(path)}: now has {len(os.listdir(path))} images")

for cls in ['COVID', 'Lung_Cancer', 'Normal', 'Pneumonia']:
    class_path = os.path.join(BASE, cls)
    if not os.path.exists(class_path):
        print(f"Warning: {class_path} not found")
        continue
    if cls in ['Lung_Cancer', 'Normal']:
        downsample_class(class_path, TARGET)
    else:
        upsample_class(class_path, TARGET)
