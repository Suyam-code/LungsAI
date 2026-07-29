import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

BASE_DIR = os.path.expanduser('~/Lung-Disease-Diagnostic-AI')
DATA_DIR = os.path.join(BASE_DIR, 'data/processed_large')

datagen = ImageDataGenerator(rescale=1./255)
gen = datagen.flow_from_directory(DATA_DIR, target_size=(224,224), batch_size=9, shuffle=True)

x, y = next(gen)
plt.figure(figsize=(10,10))
for i in range(9):
    plt.subplot(3,3,i+1)
    plt.imshow(x[i])
    label_idx = np.argmax(y[i])
    label = list(gen.class_indices.keys())[label_idx]
    plt.title(label)
    plt.axis('off')
plt.tight_layout()
plt.show()
