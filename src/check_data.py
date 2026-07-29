import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np

BASE_DIR = os.path.expanduser('~/Lung-Disease-Diagnostic-AI')
DATA_DIR = os.path.join(BASE_DIR, 'data/processed_large')

datagen = ImageDataGenerator(rescale=1./255)
gen = datagen.flow_from_directory(DATA_DIR, target_size=(224,224), batch_size=32)

x, y = next(gen)
print("Batch shape:", x.shape)
print("Labels shape:", y.shape)
print("First 5 labels (one-hot):", y[:5])
print("First 5 label indices:", np.argmax(y[:5], axis=1))
print("Image min/max:", x.min(), x.max())
print("Image mean:", x.mean())
