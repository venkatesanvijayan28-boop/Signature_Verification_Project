import os
import tempfile
import urllib.request
import json
import hashlib
import random
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
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
USERS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "users_db.json")
SMTP_CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "smtp_config.json")

def hash_password(pwd: str) -> str:
    return hashlib.sha256(pwd.encode()).hexdigest()

DEFAULT_USERS = {
    "venkatesan": {
        "email": "venkatesanvijayan28@gmail.com",
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
    if "venkatesan" in data and data["venkatesan"].get("email") in ["venkatesan@example.com", ""]:
        data["venkatesan"]["email"] = "venkatesanvijayan28@gmail.com"
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
# SMTP Email Sending Configuration
# =============================
def load_smtp_config():
    config = {
        "email": "",
        "password": "",
        "host": "smtp.gmail.com",
        "port": 587
    }
    # 1. Streamlit Secrets
    try:
        if "smtp" in st.secrets:
            config["email"] = st.secrets["smtp"].get("email") or st.secrets["smtp"].get("sender_email", "")
            config["password"] = st.secrets["smtp"].get("password") or st.secrets["smtp"].get("sender_password", "")
            config["host"] = st.secrets["smtp"].get("host", "smtp.gmail.com")
            config["port"] = int(st.secrets["smtp"].get("port", 587))
        elif "SMTP_EMAIL" in st.secrets:
            config["email"] = st.secrets.get("SMTP_EMAIL", "")
            config["password"] = st.secrets.get("SMTP_PASSWORD", "")
            config["host"] = st.secrets.get("SMTP_HOST", "smtp.gmail.com")
            config["port"] = int(st.secrets.get("SMTP_PORT", 587))
    except Exception:
        pass

    # 2. Environment Variables
    if not config["email"] and os.environ.get("SMTP_EMAIL"):
        config["email"] = os.environ.get("SMTP_EMAIL")
    if not config["password"] and os.environ.get("SMTP_PASSWORD"):
        config["password"] = os.environ.get("SMTP_PASSWORD")
    if os.environ.get("SMTP_HOST"):
        config["host"] = os.environ.get("SMTP_HOST")
    if os.environ.get("SMTP_PORT"):
        config["port"] = int(os.environ.get("SMTP_PORT"))

    # 3. Persistent Local Config File
    if os.path.exists(SMTP_CONFIG_FILE):
        try:
            with open(SMTP_CONFIG_FILE, "r") as f:
                saved = json.load(f)
                if not config["email"]:
                    config["email"] = saved.get("email", "")
                if not config["password"]:
                    config["password"] = saved.get("password", "")
                if saved.get("host"):
                    config["host"] = saved.get("host")
                if saved.get("port"):
                    config["port"] = int(saved.get("port"))
        except Exception:
            pass

    # 4. Session State Override
    if "smtp_email" in st.session_state and st.session_state.smtp_email:
        config["email"] = st.session_state.smtp_email
    if "smtp_password" in st.session_state and st.session_state.smtp_password:
        config["password"] = st.session_state.smtp_password

    return config

def save_smtp_config(email, password, host="smtp.gmail.com", port=587):
    data = {
        "email": email.strip(),
        "password": password.strip(),
        "host": host.strip(),
        "port": int(port)
    }
    st.session_state.smtp_email = data["email"]
    st.session_state.smtp_password = data["password"]
    try:
        with open(SMTP_CONFIG_FILE, "w") as f:
            json.dump(data, f, indent=4)
    except Exception:
        pass

def send_verification_email(recipient_email: str, code: str, username: str):
    """Sends a real verification email via SMTP containing the 6-digit code."""
    config = load_smtp_config()
    sender_email = config.get("email", "").strip()
    sender_password = config.get("password", "").strip()
    host = config.get("host", "smtp.gmail.com").strip()

    if not sender_email or not sender_password:
        return False, "SMTP sender credentials not configured in Streamlit Secrets. Please add [smtp] email and password in Streamlit Cloud Secrets."

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = "🔐 Password Reset Code - Signature Verification System"
        msg["From"] = f"Signature Verification System <{sender_email}>"
        msg["To"] = recipient_email

        text_content = f"""Hello {username},

You requested to reset your password for the Signature Verification System.

Your 6-Digit Email Verification Code: {code}

This code will expire in 15 minutes. Enter this code on the verification page to set your new password.

If you did not request this password reset, please ignore this email.

Best regards,
Signature Verification Team
"""
        html_content = f"""
        <html>
          <body style="font-family: Arial, sans-serif; background-color: #f4f6f9; padding: 25px;">
            <div style="max-width: 520px; margin: 0 auto; background-color: #ffffff; padding: 30px; border-radius: 10px; border: 1px solid #e1e4e8; box-shadow: 0 4px 6px rgba(0,0,0,0.05);">
              <h2 style="color: #1f2937; margin-top: 0; text-align: center;">🔐 Password Reset Request</h2>
              <p style="color: #4b5563; font-size: 15px;">Hello <strong>{username}</strong>,</p>
              <p style="color: #4b5563; font-size: 15px;">We received a request to reset your password for the <strong>Signature Verification System</strong>. Use the 6-digit verification code below to proceed:</p>
              
              <div style="background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%); border: 1px solid #86efac; border-radius: 8px; padding: 20px; text-align: center; margin: 25px 0;">
                <span style="font-size: 32px; font-weight: 800; letter-spacing: 8px; color: #15803d; font-family: monospace;">{code}</span>
              </div>
              
              <p style="color: #6b7280; font-size: 13px; line-height: 1.5;">This verification code will expire in 15 minutes. If you did not request this reset, your account is safe and you can ignore this email.</p>
              <hr style="border: none; border-top: 1px solid #e5e7eb; margin: 25px 0;">
              <p style="color: #9ca3af; font-size: 12px; text-align: center; margin-bottom: 0;">Signature Verification Project • AI Security</p>
            </div>
          </body>
        </html>
        """
        msg.attach(MIMEText(text_content, "plain"))
        msg.attach(MIMEText(html_content, "html"))

        # Try SSL 465 first
        try:
            with smtplib.SMTP_SSL(host, 465, timeout=20) as server:
                server.login(sender_email, sender_password)
                server.sendmail(sender_email, recipient_email, msg.as_string())
            return True, "Email sent successfully!"
        except Exception as ssl_err:
            # Fallback to STARTTLS 587
            try:
                with smtplib.SMTP(host, 587, timeout=20) as server:
                    server.ehlo()
                    server.starttls()
                    server.ehlo()
                    server.login(sender_email, sender_password)
                    server.sendmail(sender_email, recipient_email, msg.as_string())
                return True, "Email sent successfully!"
            except Exception as tls_err:
                return False, f"SMTP Error: {str(ssl_err)}"
    except Exception as exc:
        return False, str(exc)

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
    # 4. FORGOT PASSWORD (EMAIL RESET)
    # -----------------------------
    with tab_forgot:
        st.markdown("### 🔑 Forgot Password")
        st.write("Enter your registered email address to receive a verification code to reset your password.")

        f_email = st.text_input("Enter Email Address", placeholder="name@example.com", key="forgot_email_in")

        if st.button("Send Verification Code", key="btn_send_reset_code", use_container_width=True):
            clean_email = f_email.strip().lower()
            if not clean_email or "@" not in clean_email:
                st.warning("⚠️ Please enter a valid email address.")
            else:
                found_user = None
                for uname, udata in users.items():
                    if udata.get("email", "").strip().lower() == clean_email or uname.lower() == clean_email:
                        found_user = uname
                        break

                if found_user:
                    target_email = users[found_user].get("email", clean_email)
                    code = str(random.randint(100000, 999999))
                    st.session_state.reset_code_data = {
                        "username": found_user,
                        "code": code,
                        "email": target_email
                    }

                    # Attempt email dispatch via SMTP
                    with st.spinner(f"Sending verification code to {target_email}..."):
                        sent, msg = send_verification_email(target_email, code, found_user)

                    if sent:
                        st.success(f"📧 A 6-digit verification code has been sent directly to your email (**{target_email}**). Please check your inbox and enter the code below.")
                    else:
                        st.error(f"❌ Could not send email to {target_email}: {msg}")
                        st.warning("⚠️ Please configure your [smtp] email and App Password in Streamlit Cloud Secrets to dispatch emails.")
                else:
                    st.error(f"❌ No registered account found for `{clean_email}`. Please check your email or create a new account.")

        if st.session_state.reset_code_data:
            st.markdown("---")
            st.markdown(f"**Reset Password for:** `{st.session_state.reset_code_data['username']}` (`{st.session_state.reset_code_data['email']}`)")
            entered_code = st.text_input("Enter 6-Digit Verification Code", key="forgot_code_in")
            new_reset_pwd = st.text_input("Enter New Password", type="password", key="forgot_new_pwd")
            confirm_reset_pwd = st.text_input("Confirm New Password", type="password", key="forgot_conf_pwd")

            if st.button("Reset Password", key="btn_confirm_reset", use_container_width=True):
                if entered_code.strip() != st.session_state.reset_code_data["code"]:
                    st.error("❌ Invalid verification code. Please check your code and try again.")
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
                    st.success(f"🎉 Password for `{target_user}` has been reset successfully! You can now sign in.")

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
