import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import keras
from keras import layers
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

# -----------------------------
# Custom Layer
# -----------------------------
class AbsoluteDifference(layers.Layer):
    def call(self, inputs):
        x1, x2 = inputs
        return keras.ops.abs(x1 - x2)

# -----------------------------
# Paths
# -----------------------------
MODEL_PATH = "siamese_signature_verification.keras"
GENUINE_DIR = "dataset/genuine"
FORGED_DIR = "dataset/forged"
IMG_SIZE = 224

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

# =====================================================
# 1️⃣ VISUAL EDA – SAMPLE IMAGES
# =====================================================
def show_samples(folder, title):
    images = os.listdir(folder)[:5]
    plt.figure(figsize=(10, 3))
    for i, img_name in enumerate(images):
        img = cv2.imread(os.path.join(folder, img_name), cv2.IMREAD_GRAYSCALE)
        plt.subplot(1, 5, i+1)
        plt.imshow(img, cmap="gray")
        plt.axis("off")
    plt.suptitle(title)
    plt.show()

show_samples(GENUINE_DIR, "Sample Genuine Signatures")
show_samples(FORGED_DIR, "Sample Forged Signatures")

# =====================================================
# 2️⃣ DATASET DISTRIBUTION
# =====================================================
labels = ["Genuine", "Forged"]
counts = [len(os.listdir(GENUINE_DIR)), len(os.listdir(FORGED_DIR))]

plt.figure(figsize=(5,4))
plt.bar(labels, counts)
plt.title("Dataset Distribution")
plt.ylabel("Number of Images")
plt.show()

# =====================================================
# 3️⃣ PIXEL INTENSITY HISTOGRAM
# =====================================================
def pixel_histogram(folder, label):
    pixels = []
    for img_name in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, img_name), cv2.IMREAD_GRAYSCALE)
        pixels.extend(img.flatten())
    plt.hist(pixels, bins=50, alpha=0.5, label=label)

plt.figure(figsize=(7,4))
pixel_histogram(GENUINE_DIR, "Genuine")
pixel_histogram(FORGED_DIR, "Forged")
plt.legend()
plt.title("Pixel Intensity Distribution")
plt.show()

# =====================================================
# 4️⃣ MODEL METRICS (CONFUSION MATRIX, ROC, ETC.)
# =====================================================
y_true = []
y_pred = []
y_score = []

genuine_imgs = os.listdir(GENUINE_DIR)
forged_imgs = os.listdir(FORGED_DIR)

# Genuine-Genuine → label 1
for img in genuine_imgs[:30]:
    img1 = preprocess(os.path.join(GENUINE_DIR, img))
    img2 = preprocess(os.path.join(GENUINE_DIR, img))
    score = model.predict([img1, img2], verbose=0)[0][0]
    y_true.append(1)
    y_pred.append(1 if score >= 0.5 else 0)
    y_score.append(score)

# Genuine-Forged → label 0
for img in forged_imgs[:30]:
    img1 = preprocess(os.path.join(GENUINE_DIR, genuine_imgs[0]))
    img2 = preprocess(os.path.join(FORGED_DIR, img))
    score = model.predict([img1, img2], verbose=0)[0][0]
    y_true.append(0)
    y_pred.append(1 if score >= 0.5 else 0)
    y_score.append(score)

# =====================================================
# 5️⃣ CONFUSION MATRIX
# =====================================================
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Forged", "Genuine"],
            yticklabels=["Forged", "Genuine"])
plt.title("Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.show()

# =====================================================
# 6️⃣ ROC CURVE
# =====================================================
fpr, tpr, _ = roc_curve(y_true, y_score)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0,1], [0,1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# =====================================================
# 7️⃣ PRECISION, RECALL, F1-SCORE
# =====================================================
report = classification_report(y_true, y_pred, output_dict=True)
metrics = ["precision", "recall", "f1-score"]
values = [
    report["1"][m] for m in metrics
]

plt.figure(figsize=(5,4))
plt.bar(metrics, values)
plt.ylim(0,1)
plt.title("Model Performance Metrics (Genuine Class)")
plt.show()
