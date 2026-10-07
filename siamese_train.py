import os
import cv2
import random
import numpy as np
import keras
from keras import layers, Model
import tensorflow as tf

# -----------------------------
# Fix randomness (reproducible)
# -----------------------------
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

# -----------------------------
# Custom Distance Layer (SAFE)
# -----------------------------
class AbsoluteDifference(layers.Layer):
    def call(self, inputs):
        x1, x2 = inputs
        return keras.ops.abs(x1 - x2)

    def compute_output_shape(self, input_shape):
        return input_shape[0]


# -----------------------------
# Configuration
# -----------------------------
IMG_SIZE = 224
DATASET_PATH = "dataset"

genuine_dir = os.path.join(DATASET_PATH, "genuine")
forged_dir = os.path.join(DATASET_PATH, "forged")


# -----------------------------
# Image preprocessing (SAFE)
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
    return img


# -----------------------------
# Create image pairs (BALANCED)
# -----------------------------
def create_pairs():
    pairs = []
    labels = []

    genuine_imgs = [
        os.path.join(genuine_dir, f)
        for f in sorted(os.listdir(genuine_dir))
    ]
    forged_imgs = [
        os.path.join(forged_dir, f)
        for f in sorted(os.listdir(forged_dir))
    ]

    # Genuine–Genuine pairs (label = 1)
    for _ in range(300):
        img1 = preprocess(random.choice(genuine_imgs))
        img2 = preprocess(random.choice(genuine_imgs))
        pairs.append([img1, img2])
        labels.append(1)

    # Genuine–Forged pairs (label = 0)
    for _ in range(300):
        img1 = preprocess(random.choice(genuine_imgs))
        img2 = preprocess(random.choice(forged_imgs))
        pairs.append([img1, img2])
        labels.append(0)

    return np.array(pairs), np.array(labels)


# -----------------------------
# Load & prepare data
# -----------------------------
pairs, labels = create_pairs()
X1, X2 = pairs[:, 0], pairs[:, 1]
y = labels

# Shuffle data
indices = np.arange(len(y))
np.random.shuffle(indices)
X1, X2, y = X1[indices], X2[indices], y[indices]

# Train / Validation split
split = int(0.8 * len(y))
X1_train, X1_val = X1[:split], X1[split:]
X2_train, X2_val = X2[:split], X2[split:]
y_train, y_val = y[:split], y[split:]


# -----------------------------
# Base CNN Network (LIGHTWEIGHT)
# -----------------------------
def build_base_network():
    inp = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1))

    x = layers.Conv2D(32, (3, 3), activation="relu")(inp)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(64, (3, 3), activation="relu")(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dense(64, activation="relu")(x)

    return Model(inp, x)


base_network = build_base_network()


# -----------------------------
# Siamese Network (NO Lambda)
# -----------------------------
input_a = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1))
input_b = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1))

feat_a = base_network(input_a)
feat_b = base_network(input_b)

diff = AbsoluteDifference()([feat_a, feat_b])
output = layers.Dense(1, activation="sigmoid")(diff)

model = Model([input_a, input_b], output)

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# -----------------------------
# Train model
# -----------------------------
model.fit(
    [X1_train, X2_train],
    y_train,
    validation_data=([X1_val, X2_val], y_val),
    epochs=30,
    batch_size=8,
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=3,
            restore_best_weights=True,
            mode="max"
        )
    ]
)


# -----------------------------
# Save model (MODERN FORMAT)
# -----------------------------
model.save("siamese_signature_verification.keras")

print("✅ Training completed successfully and model saved.")
