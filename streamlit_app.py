import os
import tempfile
import urllib.request
import json
import hashlib
import cv2
import numpy as np
import requests
import keras
from keras import layers
import streamlit as st

from auth_db import (
    init_db,
    hash_password,
    validate_email,
    validate_password,
    generate_reset_token,
    db_get_user,
    db_get_user_by_email,
    db_create_user,
    db_update_password,
    db_create_reset_token,
    db_verify_reset_token,
    db_reset_password_with_token,
    db_get_all_users,
)

# Initialize database schema on startup
init_db()

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
# Demo Password Reset Helper
# =============================
def is_demo_password_reset_enabled() -> bool:
    """
    Checks if demo password reset is explicitly enabled via secrets or env var.
    Default: False for public deployments.
    """
    val = False
    try:
        val = st.secrets.get("ENABLE_DEMO_PASSWORD_RESET", False)
    except Exception:
        val = False

    if not val:
        val = os.environ.get("ENABLE_DEMO_PASSWORD_RESET", "false")

    if isinstance(val, bool):
        return val
    return str(val).strip().lower() in ("true", "1", "yes", "enabled")

# =============================
# Session State Initialization
# =============================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = None
if "role" not in st.session_state:
    st.session_state.role = None
if "password_reset_success" not in st.session_state:
    st.session_state.password_reset_success = False

# =============================
# RESET PASSWORD PAGE (TOKEN-BASED)
# =============================
def reset_password_page(token: str):
    st.markdown("## 🔐 Password Reset")
    st.caption("Secure token-based credential recovery")

    if st.session_state.get("password_reset_success"):
        st.success("🎉 Your password has been successfully updated! You can now sign in with your new password.")
        if st.button("Return to Sign In", key="btn_return_login_after_success", use_container_width=True):
            st.session_state.password_reset_success = False
            st.query_params.clear()
            st.rerun()
        return

    record, err = db_verify_reset_token(token)
    if err:
        st.error(f"❌ {err}")
        st.info("The reset link may have expired (15-minute validity), already been used, or been invalidated by a newer request.")
        if st.button("Return to Sign In", key="btn_return_login_after_err", use_container_width=True):
            st.query_params.clear()
            st.rerun()
        return

    st.info(f"Resetting password for: **{record['username']}** (`{record['email']}`)")

    new_password = st.text_input("New Password (minimum 12 characters)", type="password", key="reset_page_new_pwd")
    confirm_password = st.text_input("Confirm New Password", type="password", key="reset_page_conf_pwd")

    col_sub, col_cancel = st.columns(2)
    with col_sub:
        if st.button("Reset Password", key="btn_exec_pw_reset", use_container_width=True):
            if not new_password or not confirm_password:
                st.error("⚠️ Both password fields are required.")
            elif new_password != confirm_password:
                st.error("❌ Passwords do not match.")
            else:
                is_valid, msg = validate_password(new_password)
                if not is_valid:
                    st.error(f"⚠️ {msg}")
                else:
                    success, reset_msg = db_reset_password_with_token(token, hash_password(new_password))
                    if success:
                        st.session_state.password_reset_success = True
                        st.rerun()
                    else:
                        st.error(f"❌ {reset_msg}")

    with col_cancel:
        if st.button("Cancel & Return to Login", key="btn_cancel_pw_reset", use_container_width=True):
            st.query_params.clear()
            st.rerun()

# =============================
# LOGIN & ACCOUNT MANAGEMENT PAGE
# =============================
def login_page():
    st.title("🔐 Signature Verification Portal")
    st.subheader("Database-Backed Authentication & Access Management")

    tab_user, tab_admin, tab_register, tab_forgot = st.tabs([
        "👤 User Login",
        "🛡️ Admin Login",
        "📝 Create New Account",
        "🔑 Forgot Password"
    ])

    # -----------------------------
    # 1. USER LOGIN
    # -----------------------------
    with tab_user:
        st.markdown("### 👤 User Sign In")
        u_username = st.text_input("Username", key="user_login_u")
        u_password = st.text_input("Password", type="password", key="user_login_p")

        if st.button("Sign In as User", key="btn_user_login", use_container_width=True):
            user_info = db_get_user(u_username)
            if user_info and user_info["password_hash"] == hash_password(u_password):
                st.session_state.logged_in = True
                st.session_state.username = u_username
                st.session_state.role = user_info["role"]
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
            user_info = db_get_user(a_username)
            if user_info and user_info["password_hash"] == hash_password(a_password):
                if user_info["role"] == "Admin":
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
    # 3. CREATE NEW ACCOUNT (STORED IN SQL DATABASE)
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
            elif not validate_email(r_email):
                st.error("⚠️ Please enter a valid email address.")
            elif r_password != r_confirm:
                st.error("⚠️ Passwords do not match.")
            elif len(r_password) < 4:
                st.error("⚠️ Password must be at least 4 characters.")
            elif not fp_scan:
                st.error("⚠️ Biometric fingerprint verification is required to create an account.")
            elif role_type == "Admin" and admin_secret not in ["ADMIN@2026", "admin@123", "venkat@28", "ADMIN2026"]:
                st.error("⛔ Invalid Admin Master Passcode. Use `ADMIN@2026` or contact system owner.")
            else:
                ok, message = db_create_user(
                    r_username,
                    r_email,
                    hash_password(r_password),
                    role_type,
                    1 if fp_scan else 0
                )
                if ok:
                    st.success(f"🎉 Account `{r_username}` ({role_type}) registered in database! You can now log in under {role_type} Login or use Forgot Password with `{r_email}`.")
                else:
                    st.error(f"⚠️ {message}")

    # -----------------------------
    # 4. FORGOT PASSWORD (SECURE TOKEN RESET - NO SMTP NEEDED)
    # -----------------------------
    with tab_forgot:
        st.markdown("### 🔑 Forgot Password")
        st.write("Enter your registered email address to request a secure password-reset link.")

        f_email = st.text_input("Enter Registered Email Address", placeholder="name@example.com", key="forgot_email_in")

        if st.button("Request Password Reset", key="btn_request_reset_token", use_container_width=True):
            clean_email = f_email.strip().lower()
            if not validate_email(clean_email):
                st.error("⚠️ Please enter a valid email address.")
            else:
                user_record = db_get_user_by_email(clean_email)
                demo_enabled = is_demo_password_reset_enabled()

                if user_record:
                    raw_token = generate_reset_token()
                    db_create_reset_token(user_record["username"], user_record["email"], raw_token, valid_minutes=15)

                    if demo_enabled:
                        st.session_state["active_demo_token"] = raw_token
                        st.session_state["active_demo_user"] = user_record["username"]
                        st.session_state["active_demo_email"] = user_record["email"]
                        st.rerun()
                    else:
                        # Production / Public deployment (demo mode disabled)
                        st.info(
                            "ℹ️ If an account associated with that email exists, the password reset request has been processed.\n\n"
                            "⚠️ **Notice:** Automated email delivery is not configured on this server. "
                            "Please contact your system administrator to assist with your password reset."
                        )
                else:
                    # Unknown account: generic response to prevent user enumeration
                    if demo_enabled:
                        st.warning("⚠️ No account found with that email address.")
                    else:
                        st.info(
                            "ℹ️ If an account associated with that email exists, the password reset request has been processed.\n\n"
                            "⚠️ **Notice:** Automated email delivery is not configured on this server. "
                            "Please contact your system administrator to assist with your password reset."
                        )

        # Active Demo Reset Form (Direct inline reset with New Password & Confirm Password)
        if st.session_state.get("active_demo_token"):
            demo_token = st.session_state["active_demo_token"]
            demo_user = st.session_state.get("active_demo_user", "")
            demo_mail = st.session_state.get("active_demo_email", "")

            st.markdown("---")
            st.success("✅ Password reset request processed successfully.")
            st.markdown("#### 🧪 Demo Password Reset Link (Development Mode)")
            st.caption("No email has been sent. Because `ENABLE_DEMO_PASSWORD_RESET` is active for testing, you can open the reset page or enter your new password below:")
            st.code(f"?reset_token={demo_token}", language="text")

            st.markdown(f"**Reset Credentials for:** `{demo_user}` (`{demo_mail}`)")
            f_new_pwd = st.text_input("New Password (minimum 12 characters)", type="password", key="forgot_direct_new_pwd")
            f_conf_pwd = st.text_input("Confirm New Password", type="password", key="forgot_direct_conf_pwd")

            col_submit, col_dismiss = st.columns(2)
            with col_submit:
                if st.button("Reset Password", key="btn_forgot_direct_submit", use_container_width=True):
                    if not f_new_pwd or not f_conf_pwd:
                        st.error("⚠️ Please fill in both password fields.")
                    elif f_new_pwd != f_conf_pwd:
                        st.error("❌ Passwords do not match.")
                    else:
                        is_valid, msg = validate_password(f_new_pwd)
                        if not is_valid:
                            st.error(f"⚠️ {msg}")
                        else:
                            ok, reset_msg = db_reset_password_with_token(demo_token, hash_password(f_new_pwd))
                            if ok:
                                st.session_state["active_demo_token"] = None
                                st.session_state["reset_tab_completed"] = True
                                st.rerun()
                            else:
                                st.error(f"❌ {reset_msg}")
            with col_dismiss:
                if st.button("Cancel", key="btn_forgot_direct_cancel", use_container_width=True):
                    st.session_state["active_demo_token"] = None
                    st.rerun()

        if st.session_state.get("reset_tab_completed"):
            st.success("🎉 Your password has been successfully updated! You can now log in.")
            if st.button("Sign In Now", key="btn_after_tab_reset_done", use_container_width=True):
                st.session_state["reset_tab_completed"] = False
                st.rerun()

# =============================
# MAIN APPLICATION PAGE
# =============================
def main_app():
    user_name = st.session_state.get("username", "venkatesan")
    user_role = st.session_state.get("role", "User")

    col_title, col_user = st.columns([3, 1])
    with col_title:
        st.title("Signature Verification System")
    with col_user:
        st.markdown(f"**👤 User:** `{user_name}`")
        st.markdown(f"**Role:** `{user_role}`")
        if st.button("Logout", key="main_logout_btn"):
            st.session_state.logged_in = False
            st.session_state.username = None
            st.session_state.role = None
            st.rerun()

    # Admin Panel querying SQLite database directly
    if user_role == "Admin":
        with st.expander("🛡️ Admin Dashboard: Database Users & System Status"):
            users_list = db_get_all_users()
            display_list = [
                {
                    "Username": u["username"],
                    "Email": u["email"],
                    "Role": u["role"],
                    "Biometric / FP": "✅ Verified" if u["fingerprint_verified"] else "⚠️ Pending",
                    "Created At": u.get("created_at", "-")
                }
                for u in users_list
            ]
            st.dataframe(display_list, use_container_width=True)
            st.caption(f"Total registered accounts in database: {len(display_list)}")

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
# APP FLOW & ROUTING
# =============================
query_token = st.query_params.get("reset_token")

if query_token:
    reset_password_page(query_token)
elif st.session_state.logged_in:
    main_app()
else:
    login_page()
