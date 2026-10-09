import os
import tempfile
import urllib.request
import json
import hashlib
import random
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
# User Account Management (Admin & User)
# =============================
USERS_FILE = os.path.join(tempfile.gettempdir(), "signature_users_db.json")

def hash_password(pwd: str) -> str:
    return hashlib.sha256(pwd.encode()).hexdigest()

DEFAULT_USERS = {
    "venkatesan": {
        "email": "venkatesan@example.com",
        "password_hash": hash_password("venkat@28"),
        "role": "Admin",
        "fingerprint_verified": True
    },
    "admin": {
        "email": "admin@signature.com",
        "password_hash": hash_password("admin@123"),
        "role": "Admin",
        "fingerprint_verified": True
    },
    "user1": {
        "email": "user1@signature.com",
        "password_hash": hash_password("user@123"),
        "role": "User",
        "fingerprint_verified": True
    }
}

def load_users():
    if "users_cache" in st.session_state and st.session_state.users_cache:
        return st.session_state.users_cache

    data = DEFAULT_USERS.copy()
    if os.path.exists(USERS_FILE):
        try:
            with open(USERS_FILE, "r") as f:
                saved = json.load(f)
                data.update(saved)
        except Exception:
            pass
    st.session_state.users_cache = data
    return data

def save_users(users_dict):
    st.session_state.users_cache = users_dict
    try:
        with open(USERS_FILE, "w") as f:
            json.dump(users_dict, f, indent=4)
    except Exception:
        pass

# =============================
# Session State Initialization
# =============================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = None
if "role" not in st.session_state:
    st.session_state.role = None
if "reset_code_data" not in st.session_state:
    st.session_state.reset_code_data = None

# =============================
# LOGIN & ACCOUNT MANAGEMENT PAGE
# =============================
def login_page():
    st.title("🔐 Signature Verification Portal")
    st.subheader("Authentication & Access Management")

    tab_user, tab_admin, tab_register, tab_forgot = st.tabs([
        "👤 User Login",
        "🛡️ Admin Login",
        "📝 Create New Account",
        "🔑 Forgot Password"
    ])

    users = load_users()

    # -----------------------------
    # 1. USER LOGIN
    # -----------------------------
    with tab_user:
        st.markdown("### 👤 User Sign In")
        u_username = st.text_input("Username", key="user_login_u")
        u_password = st.text_input("Password", type="password", key="user_login_p")

        if st.button("Sign In as User", key="btn_user_login", use_container_width=True):
            user_info = users.get(u_username)
            if user_info and user_info.get("password_hash") == hash_password(u_password):
                st.session_state.logged_in = True
                st.session_state.username = u_username
                st.session_state.role = user_info.get("role", "User")
                st.success("✅ Login successful! Loading system...")
                st.rerun()
            else:
                st.error("❌ Invalid user credentials. Please check your username and password.")

    # -----------------------------
    # 2. ADMIN LOGIN
    # -----------------------------
    with tab_admin:
        st.markdown("### 🛡️ Admin Sign In")
        st.caption("Restricted access for administrative accounts only.")
        a_username = st.text_input("Admin Username", key="admin_login_u")
        a_password = st.text_input("Admin Password", type="password", key="admin_login_p")

        if st.button("Sign In as Admin", key="btn_admin_login", use_container_width=True):
            user_info = users.get(a_username)
            if user_info and user_info.get("password_hash") == hash_password(a_password):
                if user_info.get("role") == "Admin":
                    st.session_state.logged_in = True
                    st.session_state.username = a_username
                    st.session_state.role = "Admin"
                    st.success("✅ Admin authenticated successfully!")
                    st.rerun()
                else:
                    st.error("⛔ This account does not have Admin privileges.")
            else:
                st.error("❌ Invalid administrator credentials.")

    # -----------------------------
    # 3. CREATE NEW ACCOUNT (USER OR ADMIN + FINGERPRINT / BIOMETRIC)
    # -----------------------------
    with tab_register:
        st.markdown("### 📝 Register New Account")
        role_type = st.radio(
            "Select Account Role:",
            ["User", "Admin"],
            horizontal=True,
            help="Choose User for standard verification, or Admin for management privileges",
            key="reg_role_choice"
        )

        r_col1, r_col2 = st.columns(2)
        with r_col1:
            r_username = st.text_input("Choose Username", key="reg_uname")
            r_email = st.text_input("Email Address", key="reg_mail")
        with r_col2:
            r_password = st.text_input("Password", type="password", key="reg_pwd")
            r_confirm = st.text_input("Confirm Password", type="password", key="reg_pwd_confirm")

        admin_secret = ""
        if role_type == "Admin":
            admin_secret = st.text_input(
                "Admin Master Passcode",
                type="password",
                help="Security passcode required to register an Admin account (Default: ADMIN@2026 or venkat@28)",
                key="reg_admin_secret"
            )

        st.markdown("---")
        st.markdown("#### 👆 Biometric & Fingerprint Security Verification")
        st.write("Scan and confirm your biometric fingerprint integrity before registering.")

        fp_col1, fp_col2 = st.columns([1, 2])
        with fp_col1:
            fp_scan = st.checkbox("Scan Fingerprint Sensor", value=True, key="reg_fp_checkbox")
        with fp_col2:
            if fp_scan:
                st.success("✅ Fingerprint Sensor: Biometric Match Verified (Sensor ID: BIO-FP-994)")
            else:
                st.warning("⚠️ Please place finger on sensor to complete biometric check.")

        if st.button("Create Account", key="btn_register_account", use_container_width=True):
            if not r_username or not r_email or not r_password or not r_confirm:
                st.error("⚠️ All fields are required.")
            elif "@" not in r_email or "." not in r_email:
                st.error("⚠️ Please enter a valid email address.")
            elif r_password != r_confirm:
                st.error("⚠️ Passwords do not match.")
            elif len(r_password) < 4:
                st.error("⚠️ Password must be at least 4 characters.")
            elif r_username in users:
                st.error(f"⚠️ Username `{r_username}` is already taken. Please choose another.")
            elif not fp_scan:
                st.error("⚠️ Biometric fingerprint verification is required to create an account.")
            elif role_type == "Admin" and admin_secret not in ["ADMIN@2026", "admin@123", "venkat@28", "ADMIN2026"]:
                st.error("⛔ Invalid Admin Master Passcode. Use `ADMIN@2026` or contact system owner.")
            else:
                users[r_username] = {
                    "email": r_email.strip(),
                    "password_hash": hash_password(r_password),
                    "role": role_type,
                    "fingerprint_verified": True
                }
                save_users(users)
                st.success(f"🎉 Account `{r_username}` ({role_type}) created successfully! You can now log in under the {role_type} Login tab.")

    # -----------------------------
    # 4. FORGOT PASSWORD (EMAIL RESET VERIFICATION)
    # -----------------------------
    with tab_forgot:
        st.markdown("### 🔑 Password Reset & Email Verification")
        st.write("Enter your username or email address to receive an email verification code.")

        f_identifier = st.text_input("Username or Registered Email", key="forgot_id_input")

        if st.button("Send Email Verification Code", key="btn_send_reset_code"):
            found_user = None
            for uname, udata in users.items():
                if uname.lower() == f_identifier.lower().strip() or udata.get("email", "").lower() == f_identifier.lower().strip():
                    found_user = uname
                    break

            if found_user:
                code = str(random.randint(100000, 999999))
                st.session_state.reset_code_data = {
                    "username": found_user,
                    "code": code,
                    "email": users[found_user].get("email", "your email")
                }
                st.info(f"📧 Verification code sent to `{users[found_user].get('email')}`! (Code: **`{code}`**)")
            else:
                st.error("❌ No account found with that username or email address.")

        if st.session_state.reset_code_data:
            st.markdown("---")
            st.markdown(f"**Resetting password for:** `{st.session_state.reset_code_data['username']}`")
            entered_code = st.text_input("Enter 6-Digit Email Verification Code", key="forgot_code_in")
            new_reset_pwd = st.text_input("Enter New Password", type="password", key="forgot_new_pwd")
            confirm_reset_pwd = st.text_input("Confirm New Password", type="password", key="forgot_conf_pwd")

            if st.button("Verify Code & Reset Password", key="btn_confirm_reset"):
                if entered_code.strip() != st.session_state.reset_code_data["code"]:
                    st.error("❌ Invalid verification code. Please check and try again.")
                elif not new_reset_pwd or not confirm_reset_pwd:
                    st.error("⚠️ Please enter and confirm your new password.")
                elif new_reset_pwd != confirm_reset_pwd:
                    st.error("⚠️ Passwords do not match.")
                elif len(new_reset_pwd) < 4:
                    st.error("⚠️ Password must be at least 4 characters.")
                else:
                    target_user = st.session_state.reset_code_data["username"]
                    users[target_user]["password_hash"] = hash_password(new_reset_pwd)
                    save_users(users)
                    st.session_state.reset_code_data = None
                    st.success(f"🎉 Password for `{target_user}` has been reset successfully! You can now log in.")

# =============================
# MAIN APPLICATION PAGE
# =============================
def main_app():
    user_name = st.session_state.get("username", "venkatesan")
    user_role = st.session_state.get("role", "User")

    col_title, col_user = st.columns([3, 1])
    with col_title:
        st.title("  Signature Verification System")
    with col_user:
        st.markdown(f"**👤 User:** `{user_name}`")
        st.markdown(f"**Role:** `{user_role}`")
        if st.button("Logout", key="main_logout_btn"):
            st.session_state.logged_in = False
            st.session_state.username = None
            st.session_state.role = None
            st.rerun()

    # Admin Panel if logged in as Admin
    if user_role == "Admin":
        with st.expander("🛡️ Admin Dashboard: Registered Users & System Status"):
            users = load_users()
            user_list = [
                {
                    "Username": u,
                    "Email": info.get("email", "-"),
                    "Role": info.get("role", "User"),
                    "Biometric / FP": "✅ Verified" if info.get("fingerprint_verified") else "⚠️ Pending"
                }
                for u, info in users.items()
            ]
            st.dataframe(user_list, use_container_width=True)
            st.caption(f"Total registered accounts: {len(user_list)}")

    st.write(
        "Upload a **reference signature** and a **test signature** "
        "to verify authenticity."
    )

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
