import cv2
import numpy as np
import keras
from keras import layers
import streamlit as st
import tempfile
import os

# =============================
# Custom Layer
# =============================
class AbsoluteDifference(layers.Layer):
    def call(self, inputs):
        x1, x2 = inputs
        return keras.ops.abs(x1 - x2)

# =============================
# Config & Load Model
# =============================
IMG_SIZE = 224

model = keras.models.load_model(
    "siamese_signature_verification.keras",
    custom_objects={"AbsoluteDifference": AbsoluteDifference}
)

# =============================
# Preprocess Function
# =============================
def preprocess(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)
    return img

# =============================
# Session State
# =============================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# =============================
# LOGIN PAGE
# =============================
def login_page():
    st.title("   Login Page")
    st.subheader("Signature Verification System")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == "venkatesan" and password == "venkat@28":
            st.session_state.logged_in = True
            st.success("Login successful!")
            st.rerun()   # ✅ FIXED
        else:
            st.error("Invalid username or password")

# =============================
# MAIN APPLICATION PAGE
# =============================
def main_app():
    st.title("  Signature Verification System")
    st.write(
        "Upload a **reference signature** and a **test signature** "
        "to verify authenticity."
    )

    # Logout
    if st.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()   # ✅ FIXED

    ref_file = st.file_uploader(
        "Upload Reference Signature (Genuine)",
        type=["png", "jpg", "jpeg"]
    )

    test_file = st.file_uploader(
        "Upload Test Signature",
        type=["png", "jpg", "jpeg"]
    )

    if st.button("Verify Signature"):
        if ref_file and test_file:
            # Save uploaded files temporarily
            with tempfile.NamedTemporaryFile(delete=False) as ref_temp:
                ref_temp.write(ref_file.read())
                ref_path = ref_temp.name

            with tempfile.NamedTemporaryFile(delete=False) as test_temp:
                test_temp.write(test_file.read())
                test_path = test_temp.name

            # Preprocess
            img1 = preprocess(ref_path)
            img2 = preprocess(test_path)

            # Predict
            score = model.predict([img1, img2])[0][0]

            if score >= 0.5:
                st.success("✅ Result: Genuine")
                confidence = round(score * 100, 2)
            else:
                st.error("❌ Result: Forged")
                confidence = round((1 - score) * 100, 2)

            st.write(f"**Confidence:** {confidence} %")

            # Cleanup
            os.remove(ref_path)
            os.remove(test_path)

        else:
            st.warning("⚠️ Please upload both images.")

# =============================
# APP FLOW
# =============================
st.set_page_config(page_title="Signature Verification", layout="centered")

if st.session_state.logged_in:
    main_app()
else:
    login_page()
