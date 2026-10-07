import os
import cv2
import numpy as np

# Paths
DATASET_PATH = "dataset"
GENUINE_PATH = os.path.join(DATASET_PATH, "genuine")
FORGED_PATH = os.path.join(DATASET_PATH, "forged")

IMG_SIZE = 224  # Image size for CNN

X = []  # images
y = []  # labels


def preprocess_image(image_path):
    # Read image
    img = cv2.imread(image_path)

    # Convert to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Resize image
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    # Normalize pixel values
    img = img / 255.0

    return img


# Load Genuine Images (Label = 1)
for file in os.listdir(GENUINE_PATH):
    img_path = os.path.join(GENUINE_PATH, file)
    try:
        img = preprocess_image(img_path)
        X.append(img)
        y.append(1)   # Genuine = 1
    except:
        print("Error loading image:", img_path)

# Load Forged Images (Label = 0)
for file in os.listdir(FORGED_PATH):
    img_path = os.path.join(FORGED_PATH, file)
    try:
        img = preprocess_image(img_path)
        X.append(img)
        y.append(0)   # Forged = 0
    except:
        print("Error loading image:", img_path)

# Convert to NumPy arrays
X = np.array(X)
y = np.array(y)

# Add channel dimension (CNN needs this)
X = X.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

# Save preprocessed data
np.save("X_data.npy", X)
np.save("y_labels.npy", y)

print("Preprocessing completed!")
print("Total images:", len(X))
print("Genuine:", np.sum(y == 1))
print("Forged:", np.sum(y == 0))
