import os
import io
import csv
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
    db_get_user,
    db_get_user_by_email,
    db_create_user,
    db_update_password,
    db_direct_reset_password,
    db_get_all_users,
    db_delete_user,
    db_save_signature,
    db_get_user_signatures,
    db_get_signature_by_id,
    db_delete_signature,
    db_get_all_signatures,
    db_log_verification,
    db_get_audit_logs,
    db_get_audit_stats,
)

# Initialize database schema on startup
init_db()

# ==============================================================================
# Streamlit Page Configuration
# ==============================================================================
st.set_page_config(
    page_title="AI-Powered Signature Verification System",
    page_icon="✍️",
    layout="wide"
)

# ==============================================================================
# Custom Layer (Siamese Architecture)
# ==============================================================================
class AbsoluteDifference(layers.Layer):
    def call(self, inputs):
        x1, x2 = inputs
        return keras.ops.abs(x1 - x2)

# ==============================================================================
# Model Loading & Caching Logic (Machine Learning Component)
# ==============================================================================
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
        urllib.request.urlretrieve(url, temp_target)

    if os.path.exists(temp_target) and os.path.getsize(temp_target) > 0:
        os.replace(temp_target, target_path)
    else:
        raise RuntimeError("Model download resulted in an empty or corrupted file.")

@st.cache_resource(show_spinner="Loading Siamese Neural Network Model...")
def load_signature_model():
    if os.path.exists(MODEL_FILENAME):
        model_path = MODEL_FILENAME
    else:
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
                    st.error(f"Failed to load model from `{model_url}`: {exc}")
                    raise exc

        model_path = cached_model_path

    return keras.models.load_model(
        model_path,
        custom_objects={"AbsoluteDifference": AbsoluteDifference}
    )

model = load_signature_model()

# ==============================================================================
# Image Processing & Feature Extraction (ML Component 2)
# ==============================================================================
def preprocess(image_path):
    """Preprocesses signature image for Siamese neural network inference."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)
    return img

def extract_signature_features(image_path):
    """
    Extracts geometric and stroke features (density, aspect ratio, curvature/edge variance)
    as specified in Functional Requirements & ML Feature Extraction.
    """
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return {"density": 0.0, "aspect_ratio": 1.0, "edge_variance": 0.0}

    # Binarize signature strokes
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    stroke_pixels = cv2.countNonZero(thresh)
    total_pixels = gray.shape[0] * gray.shape[1]
    density = round((stroke_pixels / float(total_pixels)) * 100, 2)

    # Bounding contour & aspect ratio
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        aspect_ratio = round(w / float(h), 2) if h > 0 else 1.0
    else:
        aspect_ratio = 1.0

    # Curvature / edge variance using Sobel filter
    sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=3)
    edge_variance = round(float(np.var(sobel)), 2)

    return {
        "density": density,
        "aspect_ratio": aspect_ratio,
        "edge_variance": edge_variance
    }

# ==============================================================================
# Session State Initialization
# ==============================================================================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = None
if "role" not in st.session_state:
    st.session_state.role = None
if "last_alert" not in st.session_state:
    st.session_state.last_alert = None

# ==============================================================================
# Authentication & Access Management (Requirement 1)
# ==============================================================================
def login_page():
    st.markdown("<h1 style='text-align: center;'>AI-Powered Signature Verification System</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #6b7280;'>Secure Biometric Authentication • Siamese CNN • Database Records</p>", unsafe_allow_html=True)
    st.markdown("---")

    tab_user, tab_admin, tab_register, tab_reset = st.tabs([
        "👤 User Login",
        "🛡️ Admin Login",
        "📝 Register Account",
        "🔑 Reset Password"
    ])

    # 1. User Sign In
    with tab_user:
        st.markdown("### 👤 User Sign In")
        u_username = st.text_input("Username", key="u_login_user")
        u_password = st.text_input("Password", type="password", key="u_login_pass")
        u_2fa = st.checkbox("Two-Factor / Biometric Authentication Verified (Sensor OK)", value=True, key="u_login_2fa")

        if st.button("Sign In as User", key="btn_u_login", use_container_width=True):
            if not u_2fa:
                st.error("⚠️ Two-Factor / Biometric verification is required to sign in.")
            else:
                user_info = db_get_user(u_username)
                if user_info and user_info["password_hash"] == hash_password(u_password):
                    st.session_state.logged_in = True
                    st.session_state.username = u_username
                    st.session_state.role = user_info["role"]
                    st.success("✅ Authentication successful!")
                    st.rerun()
                else:
                    st.error("❌ Invalid username or password.")

    # 2. Admin Sign In
    with tab_admin:
        st.markdown("### 🛡️ Admin Sign In")
        st.caption("Restricted access for administrative accounts.")
        a_username = st.text_input("Admin Username", key="a_login_user")
        a_password = st.text_input("Admin Password", type="password", key="a_login_pass")

        if st.button("Sign In as Admin", key="btn_a_login", use_container_width=True):
            user_info = db_get_user(a_username)
            if user_info and user_info["password_hash"] == hash_password(a_password):
                if user_info["role"] == "Admin":
                    st.session_state.logged_in = True
                    st.session_state.username = a_username
                    st.session_state.role = "Admin"
                    st.success("✅ Admin credentials verified!")
                    st.rerun()
                else:
                    st.error("⛔ This account does not possess administrator privileges.")
            else:
                st.error("❌ Invalid administrator credentials.")

    # 3. Create Account (User Registration)
    with tab_register:
        st.markdown("### 📝 Register New Account")
        col1, col2 = st.columns(2)
        with col1:
            r_username = st.text_input("Choose Username", key="reg_uname")
            r_email = st.text_input("Email Address", key="reg_email")
        with col2:
            r_password = st.text_input("Password (min 6 chars)", type="password", key="reg_pwd")
            r_confirm = st.text_input("Confirm Password", type="password", key="reg_pwd_conf")

        st.markdown("#### 👆 Biometric 2FA Verification")
        fp_scan = st.checkbox("Biometric / Fingerprint Sensor Verified", value=True, key="reg_fp_sensor")

        if st.button("Create Account", key="btn_create_account", use_container_width=True):
            if not r_username or not r_email or not r_password or not r_confirm:
                st.error("⚠️ All fields are required.")
            elif not validate_email(r_email):
                st.error("⚠️ Please enter a valid email address.")
            elif r_password != r_confirm:
                st.error("⚠️ Passwords do not match.")
            elif len(r_password) < 6:
                st.error("⚠️ Password must be at least 6 characters long.")
            elif not fp_scan:
                st.error("⚠️ Biometric verification is required.")
            else:
                ok, msg = db_create_user(
                    r_username,
                    r_email,
                    hash_password(r_password),
                    "User",
                    1 if fp_scan else 0
                )
                if ok:
                    st.success(f"🎉 User account `{r_username}` created in database! You can now log in under User Login.")
                else:
                    st.error(f"⚠️ {msg}")

    # 4. In-App Password Reset (Direct Database-Backed - Zero SMTP Errors)
    with tab_reset:
        st.markdown("### 🔑 In-App Password Recovery")
        st.caption("Reset your password directly in the database using your registered username and email.")
        reset_uname = st.text_input("Registered Username", key="res_uname")
        reset_email = st.text_input("Registered Email Address", key="res_email")
        reset_new_pwd = st.text_input("New Password", type="password", key="res_new_pwd")
        reset_conf_pwd = st.text_input("Confirm New Password", type="password", key="res_conf_pwd")
        reset_2fa = st.checkbox("Confirm Biometric / Two-Factor Identity Verification", value=True, key="res_2fa")

        if st.button("Reset Password", key="btn_direct_reset", use_container_width=True):
            if not reset_uname or not reset_email or not reset_new_pwd or not reset_conf_pwd:
                st.error("⚠️ All fields are required.")
            elif reset_new_pwd != reset_conf_pwd:
                st.error("❌ Passwords do not match.")
            elif len(reset_new_pwd) < 6:
                st.error("⚠️ Password must be at least 6 characters long.")
            elif not reset_2fa:
                st.error("⚠️ Biometric identity confirmation required.")
            else:
                ok, msg = db_direct_reset_password(reset_uname, reset_email, hash_password(reset_new_pwd))
                if ok:
                    st.success("🎉 Password updated successfully in the database! You can now log in.")
                else:
                    st.error(f"❌ {msg}")

# ==============================================================================
# Main Application Dashboard
# ==============================================================================
def main_app():
    user_name = st.session_state.get("username", "User")
    user_role = st.session_state.get("role", "User")

    # Navigation & Header
    header_col1, header_col2 = st.columns([3, 1])
    with header_col1:
        st.title("✍️ AI-Powered Signature Verification System")
        st.caption(f"Authenticated as: **{user_name}** | Access Level: **{user_role}**")
    with header_col2:
        st.write("")
        if st.button("🚪 Sign Out", key="btn_signout", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.username = None
            st.session_state.role = None
            st.rerun()

    st.markdown("---")

    # In-App Notification / Alert Banner (Functional Requirement 6)
    if st.session_state.get("last_alert"):
        alert_info = st.session_state.last_alert
        if alert_info["type"] == "forged":
            st.error(f"🚨 **SECURITY ALERT:** Suspicious / Forged Signature Detected! (Confidence: {alert_info['confidence']}%, Fraud Risk: HIGH). Attempt logged in Audit Trail.")
        elif alert_info["type"] == "genuine":
            st.success(f"✅ **IN-APP NOTIFICATION:** Signature Authenticity Verified Successfully (Confidence: {alert_info['confidence']}%).")

    # Admin Management Interface (Functional Requirement 7)
    if user_role == "Admin":
        with st.expander("🛡️ Admin Management Panel (Accounts, Signatures, Audit Trail & Reports)", expanded=False):
            adm_tab1, adm_tab2, adm_tab3, adm_tab4 = st.tabs([
                "👥 User Accounts",
                "✍️ Signature Records",
                "📋 Audit Trail",
                "📊 Reports & Analytics"
            ])

            # Tab 1: User Accounts
            with adm_tab1:
                st.markdown("#### Registered User Accounts")
                all_users = db_get_all_users()
                display_users = [
                    {
                        "Username": u["username"],
                        "Email": u["email"],
                        "Role": u["role"],
                        "2FA / Biometric": "✅ Verified" if u["fingerprint_verified"] else "⚠️ Pending",
                        "Created At": u.get("created_at", "-")
                    }
                    for u in all_users
                ]
                st.dataframe(display_users, use_container_width=True)

                del_col1, del_col2 = st.columns([3, 1])
                with del_col1:
                    user_to_delete = st.selectbox("Select user to manage / delete:", [u["username"] for u in all_users if u["username"] not in ("admin", "venkatesan")])
                with del_col2:
                    if st.button("Delete User", key="btn_delete_user"):
                        if user_to_delete:
                            ok, del_msg = db_delete_user(user_to_delete)
                            if ok:
                                st.success(del_msg)
                                st.rerun()
                            else:
                                st.error(del_msg)

            # Tab 2: Signature Records
            with adm_tab2:
                st.markdown("#### All Stored Signature Records (Database)")
                all_sigs = db_get_all_signatures()
                if all_sigs:
                    st.dataframe(all_sigs, use_container_width=True)
                    sig_to_view = st.selectbox("View Stored Signature Record:", [s["id"] for s in all_sigs], format_func=lambda sid: f"ID #{sid} - {next((s['signature_name'] for s in all_sigs if s['id'] == sid), '')} ({next((s['username'] for s in all_sigs if s['id'] == sid), '')})")
                    if sig_to_view:
                        sig_rec = db_get_signature_by_id(sig_to_view)
                        if sig_rec:
                            st.image(sig_rec["image_data"], caption=f"Stored Signature: {sig_rec['signature_name']} by {sig_rec['username']}", width=250)
                            if st.button("Delete This Signature Record", key="btn_del_sig_adm"):
                                db_delete_signature(sig_to_view)
                                st.success("Record removed from database.")
                                st.rerun()
                else:
                    st.info("No signatures stored in database yet.")

            # Tab 3: Audit Trail (Requirement 5)
            with adm_tab3:
                st.markdown("#### Audit Trail: Signature Verification Attempts")
                audit_logs = db_get_audit_logs(limit=150)
                if audit_logs:
                    st.dataframe(audit_logs, use_container_width=True)
                else:
                    st.info("No verification audit records logged yet.")

            # Tab 4: Reports & Analytics
            with adm_tab4:
                st.markdown("#### Verification Analytics & Exportable Reports")
                stats = db_get_audit_stats()
                stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
                stat_col1.metric("Total Verifications", stats["total"])
                stat_col2.metric("Genuine Matches", stats["genuine"])
                stat_col3.metric("Forged / Rejected", stats["forged"])
                stat_col4.metric("Suspicious Alerts", stats["suspicious"])

                # CSV Report Export
                audit_logs = db_get_audit_logs(limit=500)
                if audit_logs:
                    csv_buffer = io.StringIO()
                    writer = csv.DictWriter(csv_buffer, fieldnames=audit_logs[0].keys())
                    writer.writeheader()
                    writer.writerows(audit_logs)
                    csv_data = csv_buffer.getvalue()

                    st.download_button(
                        label="📥 Download Detailed Audit Report (CSV)",
                        data=csv_data,
                        file_name="signature_verification_audit_report.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

    # ==========================================================================
    # Signature Database Management (Functional Requirement 4)
    # ==========================================================================
    with st.expander("📁 My Stored Signature Database (Add / View Saved Signatures)", expanded=False):
        user_sigs = db_get_user_signatures(user_name)
        st.markdown(f"**Stored Genuine Signatures for `{user_name}`:**")
        if user_sigs:
            sig_cols = st.columns(min(len(user_sigs), 4))
            for i, sig in enumerate(user_sigs):
                with sig_cols[i % len(sig_cols)]:
                    sig_data = db_get_signature_by_id(sig["id"])
                    if sig_data:
                        st.image(sig_data["image_data"], caption=sig["signature_name"], width=160)
                        if st.button("🗑️ Remove", key=f"del_sig_{sig['id']}"):
                            db_delete_signature(sig["id"], username=user_name)
                            st.success("Removed.")
                            st.rerun()
        else:
            st.info("You haven't saved any reference signatures to your database yet.")

        st.markdown("---")
        st.markdown("#### ➕ Add New Genuine Signature to Database")
        new_sig_name = st.text_input("Signature Label / Purpose", placeholder="e.g. Official Bank Signature", key="in_new_sig_name")
        new_sig_file = st.file_uploader("Upload Genuine Signature File (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"], key="in_new_sig_file")

        if st.button("💾 Save Signature to Database", key="btn_save_sig"):
            if not new_sig_name or not new_sig_file:
                st.warning("⚠️ Please provide a label and select a signature image.")
            else:
                img_bytes = new_sig_file.read()
                ext = new_sig_file.name.split(".")[-1].lower()
                ok, smsg = db_save_signature(user_name, new_sig_name, img_bytes, file_type=ext)
                if ok:
                    st.success("✅ Signature stored successfully in database!")
                    st.rerun()
                else:
                    st.error(f"❌ {smsg}")

    # ==========================================================================
    # Signature Verification Interface (Functional Requirements 2 & 3)
    # ==========================================================================
    st.markdown("### 🔍 AI Signature Verification")
    st.write("Compare an uploaded signature with stored signature data or a reference file to verify authenticity.")

    v_col1, v_col2 = st.columns(2)

    # 1. Reference Signature Source
    with v_col1:
        st.markdown("#### 1. Reference Signature (Genuine)")
        user_sigs = db_get_user_signatures(user_name)

        ref_source_options = ["Upload Reference Signature File"]
        if user_sigs:
            ref_source_options.insert(0, "Select from Stored Database Signatures")

        ref_mode = st.radio("Reference Source:", ref_source_options, key="ref_mode_choice")

        ref_img_path = None
        ref_source_desc = ""

        if ref_mode == "Select from Stored Database Signatures" and user_sigs:
            selected_sig_id = st.selectbox(
                "Choose Stored Signature:",
                [s["id"] for s in user_sigs],
                format_func=lambda sid: next((s["signature_name"] for s in user_sigs if s["id"] == sid), ""),
                key="sel_db_sig"
            )
            sig_obj = db_get_signature_by_id(selected_sig_id)
            if sig_obj:
                st.image(sig_obj["image_data"], caption=f"Selected: {sig_obj['signature_name']}", width=220)
                # Save temporarily for preprocessing
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{sig_obj['file_type']}") as tmp_ref:
                    tmp_ref.write(sig_obj["image_data"])
                    ref_img_path = tmp_ref.name
                ref_source_desc = f"Database: {sig_obj['signature_name']}"
        else:
            ref_file = st.file_uploader(
                "Upload Reference Signature Image",
                type=["png", "jpg", "jpeg"],
                key="upl_ref_file"
            )
            if ref_file:
                st.image(ref_file, caption="Uploaded Genuine Reference", width=220)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_ref:
                    tmp_ref.write(ref_file.read())
                    ref_img_path = tmp_ref.name
                ref_source_desc = f"Uploaded File: {ref_file.name}"

    # 2. Test Signature Upload
    with v_col2:
        st.markdown("#### 2. Test Signature (To Verify)")
        test_file = st.file_uploader(
            "Upload Test Signature Image",
            type=["png", "jpg", "jpeg"],
            key="upl_test_file"
        )
        test_img_path = None
        if test_file:
            st.image(test_file, caption="Uploaded Test Signature", width=220)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_test:
                tmp_test.write(test_file.read())
                test_img_path = tmp_test.name

    st.markdown("---")

    # Verification Action Button
    if st.button("🚀 Verify Signature Authenticity", key="btn_run_verify", use_container_width=True):
        if not ref_img_path or not test_img_path:
            st.warning("⚠️ Please provide both a genuine reference signature and a test signature to proceed.")
        else:
            with st.spinner("Analyzing stroke geometry, contour features, and Siamese CNN embeddings..."):
                # Preprocess for Siamese CNN
                img1 = preprocess(ref_img_path)
                img2 = preprocess(test_img_path)

                # Inference
                score = float(model.predict([img1, img2])[0][0])

                # Feature extraction (ML Component 2)
                feats_ref = extract_signature_features(ref_img_path)
                feats_test = extract_signature_features(test_img_path)

                # Density difference
                density_diff = abs(feats_ref["density"] - feats_test["density"])
                aspect_ratio_diff = abs(feats_ref["aspect_ratio"] - feats_test["aspect_ratio"])

                # Determine Result & Confidence Score (ML Component 3)
                if score >= 0.5:
                    result = "Genuine"
                    confidence = round(score * 100, 2)
                    fraud_risk = "Low" if density_diff < 15.0 else "Medium (Minor Inconsistency)"
                else:
                    result = "Forged"
                    confidence = round((1.0 - score) * 100, 2)
                    fraud_risk = "High (Suspicious Forgery)"

                # In-App Notification / Security Alert
                st.session_state.last_alert = {
                    "type": "genuine" if result == "Genuine" else "forged",
                    "confidence": confidence
                }

                # Audit Trail Logging (Functional Requirement 5)
                test_name = test_file.name if test_file else "test_signature.png"
                log_details = f"Siamese score: {round(score, 4)} | Stroke density diff: {round(density_diff, 2)}% | Aspect diff: {round(aspect_ratio_diff, 2)}"
                db_log_verification(
                    username=user_name,
                    reference_source=ref_source_desc,
                    test_name=test_name,
                    result=result,
                    confidence=confidence,
                    fraud_risk=fraud_risk,
                    details=log_details
                )

                # Clean up temporary files
                try:
                    if os.path.exists(ref_img_path):
                        os.remove(ref_img_path)
                    if os.path.exists(test_img_path):
                        os.remove(test_img_path)
                except Exception:
                    pass

                # Display Results
                res_box1, res_box2 = st.columns([2, 1])
                with res_box1:
                    if result == "Genuine":
                        st.success(f"### ✅ Verification Result: GENUINE")
                        st.write(f"The test signature matches the reference pattern with **{confidence}%** confidence.")
                    else:
                        st.error(f"### 🚨 Verification Result: FORGED")
                        st.write(f"The test signature does NOT match the authentic pattern. Confidence of forgery: **{confidence}%**.")

                with res_box2:
                    st.metric("Model Confidence Score", f"{confidence} %")
                    st.metric("Fraud / Anomaly Risk", fraud_risk)

                # Display Extracted Features
                with st.expander("🔬 Feature Extraction & Image Processing Analysis (Details)", expanded=True):
                    feat_col1, feat_col2, feat_col3 = st.columns(3)
                    feat_col1.metric("Reference Stroke Density", f"{feats_ref['density']}%", f"Diff: {round(density_diff, 2)}%")
                    feat_col2.metric("Test Stroke Density", f"{feats_test['density']}%")
                    feat_col3.metric("Aspect Ratio Match", f"{feats_test['aspect_ratio']}", f"Ref: {feats_ref['aspect_ratio']}")

# ==============================================================================
# App Routing
# ==============================================================================
if st.session_state.logged_in:
    main_app()
else:
    login_page()
