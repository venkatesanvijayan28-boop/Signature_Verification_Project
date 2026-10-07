
import os
import cv2
import numpy as np
import keras
from keras import layers

# -----------------------------
# Custom Layer (same as training)
# -----------------------------
class AbsoluteDifference(layers.Layer):
    def call(self, inputs):
        x1, x2 = inputs
        return keras.ops.abs(x1 - x2)

# -----------------------------
# Configuration
# -----------------------------
IMG_SIZE = 224
MODEL_PATH = "siamese_signature_verification.keras"

# -----------------------------
# Load trained model
# -----------------------------
model = keras.models.load_model(
    MODEL_PATH,
    custom_objects={"AbsoluteDifference": AbsoluteDifference}
)

# -----------------------------
# Image preprocessing
# -----------------------------
def preprocess(img_path):
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to read image: {img_path}")

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)
    return img

# -----------------------------
# 🔴 MANUAL IMAGE SELECTION (REPLACED HERE)
# -----------------------------
ref_img_path = "dataset/genuine/v.JPEG"
test_img_path = "dataset/forged/v1.JPEG"

# -----------------------------
# Preprocess images
# -----------------------------
img1 = preprocess(ref_img_path)
img2 = preprocess(test_img_path)

# -----------------------------
# Predict
# -----------------------------
score = model.predict([img1, img2])[0][0]

if score >= 0.5:
    print("Result     : Genuine")
    print("Confidence :", round(score * 100, 2), "%")
else:
    print("Result     : Forged")
    print("Confidence :", round((1 - score) * 100, 2), "%")
