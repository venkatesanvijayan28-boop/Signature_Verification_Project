import os
import cv2
import numpy as np
import keras
from keras import layers
import random

# -----------------------------
# Custom layer
# -----------------------------
class AbsoluteDifference(layers.Layer):
    def call(self, inputs):
        x1, x2 = inputs
        return keras.ops.abs(x1 - x2)

# -----------------------------
# Config
# -----------------------------
IMG_SIZE = 224
MODEL_PATH = "siamese_signature_verification.keras"

genuine_dir = "dataset/genuine"
forged_dir = "dataset/forged"

# -----------------------------
# Load model
# -----------------------------
model = keras.models.load_model(
    MODEL_PATH,
    custom_objects={"AbsoluteDifference": AbsoluteDifference}
)

# -----------------------------
# Preprocess
# -----------------------------
def preprocess(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)
    return img

# -----------------------------
# Accuracy test
# -----------------------------
genuine_imgs = os.listdir(genuine_dir)
forged_imgs = os.listdir(forged_dir)

total = 50
correct = 0

for i in range(total):
    # 50% genuine, 50% forged
    if i % 2 == 0:
        img1 = preprocess(os.path.join(genuine_dir, random.choice(genuine_imgs)))
        img2 = preprocess(os.path.join(genuine_dir, random.choice(genuine_imgs)))
        true_label = 1
    else:
        img1 = preprocess(os.path.join(genuine_dir, random.choice(genuine_imgs)))
        img2 = preprocess(os.path.join(forged_dir, random.choice(forged_imgs)))
        true_label = 0

    score = model.predict([img1, img2], verbose=0)[0][0]
    predicted = 1 if score >= 0.5 else 0

    if predicted == true_label:
        correct += 1

accuracy = (correct / total) * 100
print(f"Test Accuracy : {accuracy:.2f}%")
