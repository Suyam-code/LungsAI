import tensorflow as tf
import numpy as np
import os
from PIL import Image

BASE_DIR = os.path.expanduser('~/Lung-Disease-Diagnostic-AI')
MODEL_PATH = os.path.join(BASE_DIR, 'models/main_diagnostic_model_balanced.h5')
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

img_path = input("Enter the full path to a lung cancer X‑ray image: ")

try:
    img = Image.open(img_path).convert('RGB').resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array, verbose=0)[0]
    print("\nRaw predictions (indices 0–3):", preds)
    print("Argmax index:", np.argmax(preds))
    classes = ['COVID', 'Lung_Cancer', 'Normal', 'Pneumonia']
    print("Predicted class:", classes[np.argmax(preds)])
except Exception as e:
    print(f"Error: {e}")
