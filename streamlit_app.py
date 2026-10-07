import os
import tempfile
import urllib.request
import cv2
import numpy as np
import requests
import keras
from keras import layers
import streamlit as st

# =============================
# Streamlit Page Configuration
# =============================
st.set_page_config(page_title="Signature Verification", layout="centered")

# =============================
# Custom Layer
# =============================
class AbsoluteDifference(layers.Layer):
    def call(self, inputs):
        x1, x2 = inputs
        return keras.ops.abs(x1 - x2)

# =============================
# Model Loading & Caching Logic
# =============================
IMG_SIZE = 224
MODEL_FILENAME = "siamese_signature_verification.keras"
DEFAULT_MODEL_URL = os.environ.get(
    "MODEL_URL",
    "https://huggingface.co/Venkatesanv/signature-verification/resolve/main/siamese_signature_verification.keras"
)

def download_model(target_path: str, url: str):
    """Downloads model using requests streaming with urllib.request fallback."""
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    temp_target = target_path + ".download"
    try:
        response = requests.get(url, stream=True, timeout=120)
        response.raise_for_status()
        with open(temp_target, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    except Exception:
        # Fallback to urllib.request
        urllib.request.urlretrieve(url, temp_target)

    if os.path.exists(temp_target) and os.path.getsize(temp_target) > 0:
        os.replace(temp_target, target_path)
    else:
        raise RuntimeError("Model download resulted in an empty or corrupted file.")

@st.cache_resource(show_spinner="Loading Siamese Signature Verification Model...")
def load_signature_model():
    # 1. Local development: use existing file directly if available
    if os.path.exists(MODEL_FILENAME):
        model_path = MODEL_FILENAME
    else:
        # 2. Production/Cloud: use cached temporary location
        cache_dir = os.path.join(tempfile.gettempdir(), "signature_verification_cache")
        cached_model_path = os.path.join(cache_dir, MODEL_FILENAME)

        if not os.path.exists(cached_model_path) or os.path.getsize(cached_model_path) == 0:
            model_url = None
            try:
                model_url = st.secrets.get("MODEL_URL", None)
            except Exception:
                model_url = None

            if not model_url:
                model_url = os.environ.get("MODEL_URL", DEFAULT_MODEL_URL)

            with st.spinner("Downloading model weights from Hugging Face..."):
                try:
                    download_model(cached_model_path, model_url)
                except Exception as exc:
                    st.error(
                        f"Failed to download model from `{model_url}`: {exc}. "
                        "Please verify your `MODEL_URL` secret or environment variable."
                    )
                    raise exc

        model_path = cached_model_path

    return keras.models.load_model(
        model_path,
        custom_objects={"AbsoluteDifference": AbsoluteDifference}
    )

model = load_signature_model()

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
if st.session_state.logged_in:
    main_app()
else:
    login_page()
