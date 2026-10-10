import os
import re
import hashlib
import secrets
import sqlite3
from datetime import datetime, timezone, timedelta

try:
    from supabase import create_client, Client
    HAS_SUPABASE = True
except ImportError:
    HAS_SUPABASE = False

DB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "signature_system.db")

# Default Supabase Credentials (configurable via st.secrets or environment variables)
DEFAULT_SUPABASE_URL = "https://lxudtklogmuoypkpaovc.supabase.co"
DEFAULT_SUPABASE_KEY = "sb_publishable_odJu-vWFQ9Ujjgk-CUTBLA_frCe3Qv5"

def get_supabase_client():
    """Initializes and returns a Supabase client using secrets, environment vars, or defaults."""
    if not HAS_SUPABASE:
        return None
    url = None
    key = None
    try:
        import streamlit as st
        url = st.secrets.get("SUPABASE_URL") or st.secrets.get("NEXT_PUBLIC_SUPABASE_URL")
        key = st.secrets.get("SUPABASE_KEY") or st.secrets.get("NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY") or st.secrets.get("SUPABASE_ANON_KEY")
    except Exception:
        pass
    if not url:
        url = os.environ.get("SUPABASE_URL") or os.environ.get("NEXT_PUBLIC_SUPABASE_URL", DEFAULT_SUPABASE_URL)
    if not key:
        key = os.environ.get("SUPABASE_KEY") or os.environ.get("NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY", DEFAULT_SUPABASE_KEY)
    try:
        return create_client(url, key)
    except Exception:
        return None

def supabase_sync_user(email: str, password: str, username: str, role: str, fingerprint_verified: int = 1):
    """Auto-syncs user creation to Supabase Auth."""
    client = get_supabase_client()
    if not client:
        return False, "Supabase client not initialized."
    try:
        res = client.auth.sign_up({
            "email": email.strip(),
            "password": password,
            "options": {
                "data": {
                    "username": username.strip(),
                    "role": role,
                    "fingerprint_verified": fingerprint_verified
                }
            }
        })
        return True, "User synced to Supabase Auth."
    except Exception as exc:
        return False, str(exc)

def supabase_send_reset_email(email: str):
    """Sends a password-reset verification email to the user's inbox using Supabase's built-in mailer."""
    client = get_supabase_client()
    if not client:
        return False, "Supabase client not available."
    try:
        client.auth.reset_password_for_email(email.strip())
        return True, "Supabase password reset email dispatched."
    except Exception as exc:
        return False, str(exc)

def supabase_verify_otp_and_reset(email: str, token: str, new_password: str):
    """Verifies recovery OTP token and updates password in Supabase."""
    client = get_supabase_client()
    if not client:
        return False, "Supabase client not available."
    try:
        # Verify recovery OTP/token
        verify_res = client.auth.verify_otp({
            "email": email.strip(),
            "token": token.strip(),
            "type": "recovery"
        })
        # Update user password in Supabase
        update_res = client.auth.update_user({"password": new_password})
        return True, "Password updated in Supabase successfully."
    except Exception as exc:
        return False, str(exc)

def get_db_connection(db_path: str = None):
    target = db_path or DB_FILE
    conn = sqlite3.connect(target, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def hash_password(pwd: str) -> str:
    """Computes SHA-256 hash for secure password storage."""
    return hashlib.sha256(pwd.encode("utf-8")).hexdigest()

def hash_token(raw_token: str) -> str:
    """Computes SHA-256 hash of a raw reset token."""
    return hashlib.sha256(raw_token.encode("utf-8")).hexdigest()

def validate_email(email: str) -> bool:
    """Validates basic email address structure."""
    if not email or not isinstance(email, str):
        return False
    pattern = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
    return bool(re.match(pattern, email.strip()))

def validate_password(pwd: str) -> tuple:
    """Validates password constraints (minimum 12 characters)."""
    if not pwd or len(pwd) < 12:
        return False, "Password must be at least 12 characters long."
    return True, ""

def generate_reset_token() -> str:
    """Generates a cryptographically secure, random, URL-safe token."""
    return secrets.token_urlsafe(32)

def init_db(db_path: str = None):
    """Initializes SQLite tables for users and password reset tokens."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    # Users table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            email TEXT NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL,
            fingerprint_verified INTEGER DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Secure Password Reset Tokens table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS password_reset_tokens (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            email TEXT NOT NULL,
            token_hash TEXT NOT NULL UNIQUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            expires_at TIMESTAMP NOT NULL,
            is_used INTEGER DEFAULT 0,
            is_revoked INTEGER DEFAULT 0
        )
    """)
    conn.commit()

    # Seed default accounts if users table is empty
    cursor.execute("SELECT COUNT(*) FROM users")
    if cursor.fetchone()[0] == 0:
        cursor.execute(
            "INSERT INTO users (username, email, password_hash, role, fingerprint_verified) VALUES (?, ?, ?, ?, ?)",
            ("venkatesan", "venkatesanvijayan28@gmail.com", hash_password("venkat@28"), "Admin", 1)
        )
        cursor.execute(
            "INSERT INTO users (username, email, password_hash, role, fingerprint_verified) VALUES (?, ?, ?, ?, ?)",
            ("admin", "admin@signature.com", hash_password("admin@123"), "Admin", 1)
        )
        cursor.execute(
            "INSERT INTO users (username, email, password_hash, role, fingerprint_verified) VALUES (?, ?, ?, ?, ?)",
            ("user1", "user1@signature.com", hash_password("user@123"), "User", 1)
        )
        conn.commit()
    else:
        # Keep default admin email synchronized
        cursor.execute(
            "UPDATE users SET email = ? WHERE username = ? AND email != ?",
            ("venkatesanvijayan28@gmail.com", "venkatesan", "venkatesanvijayan28@gmail.com")
        )
        conn.commit()

    conn.close()

def db_get_user(username: str, db_path: str = None):
    """Fetches user record by username."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username = ?", (username.strip(),))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None

def db_get_user_by_email(email: str, db_path: str = None):
    """Fetches user record by email (case-insensitive)."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE LOWER(email) = LOWER(?)", (email.strip(),))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None

def db_get_user_by_email_or_username(identifier: str, db_path: str = None):
    """Fetches user record by either email or username."""
    clean_id = identifier.strip()
    user = db_get_user_by_email(clean_id, db_path)
    if not user:
        user = db_get_user(clean_id, db_path)
    return user

def db_create_user(username: str, email: str, password_hash: str, role: str, fingerprint_verified: int = 1, raw_password: str = None, db_path: str = None):
    """Creates a new user record in the database and auto-syncs with Supabase."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO users (username, email, password_hash, role, fingerprint_verified) VALUES (?, ?, ?, ?, ?)",
            (username.strip(), email.strip(), password_hash, role, fingerprint_verified)
        )
        conn.commit()
        # Auto-sync to Supabase if raw password provided
        if raw_password:
            try:
                supabase_sync_user(email, raw_password, username, role, fingerprint_verified)
            except Exception:
                pass
        return True, "User registered successfully."
    except sqlite3.IntegrityError:
        return False, f"Username '{username}' already exists in database."
    finally:
        conn.close()

def db_update_password(username: str, new_password_hash: str, db_path: str = None):
    """Updates password hash for a specific user."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("UPDATE users SET password_hash = ? WHERE username = ?", (new_password_hash, username.strip()))
    conn.commit()
    conn.close()

def db_create_reset_token(username: str, email: str, raw_token: str, valid_minutes: int = 15, db_path: str = None):
    """Stores a hashed single-use token and invalidates previous tokens for this user."""
    thash = hash_token(raw_token)
    now_utc = datetime.now(timezone.utc)
    expires_at = (now_utc + timedelta(minutes=valid_minutes)).isoformat()

    conn = get_db_connection(db_path)
    try:
        with conn:
            cursor = conn.cursor()
            # Invalidate prior outstanding tokens for this user
            cursor.execute(
                "UPDATE password_reset_tokens SET is_revoked = 1 WHERE username = ? AND is_used = 0 AND is_revoked = 0",
                (username.strip(),)
            )
            # Insert the new token hash
            cursor.execute(
                "INSERT INTO password_reset_tokens (username, email, token_hash, expires_at, is_used, is_revoked) VALUES (?, ?, ?, ?, 0, 0)",
                (username.strip(), email.strip(), thash, expires_at)
            )
        return True, expires_at
    except Exception as exc:
        return False, str(exc)
    finally:
        conn.close()

def db_verify_reset_token(raw_token: str, db_path: str = None):
    """Validates raw token against stored hash, ensuring it is unexpired, unused, and unrevoked."""
    if not raw_token or not isinstance(raw_token, str):
        return None, "Reset token is missing or empty."

    thash = hash_token(raw_token)
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM password_reset_tokens WHERE token_hash = ?",
        (thash,)
    )
    row = cursor.fetchone()
    conn.close()

    if not row:
        return None, "Invalid reset token."

    record = dict(row)
    if record["is_used"]:
        return None, "This password reset token has already been used."

    if record["is_revoked"]:
        return None, "This password reset token has been revoked or superseded by a newer request."

    try:
        exp_dt = datetime.fromisoformat(record["expires_at"])
        if exp_dt.tzinfo is None:
            exp_dt = exp_dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None, "Corrupted token expiration timestamp."

    if datetime.now(timezone.utc) > exp_dt:
        return None, "This password reset token has expired (valid for 15 minutes)."

    return record, None

def db_reset_password_with_token(raw_token: str, new_password_hash: str, db_path: str = None):
    """Atomically updates password, marks token used, and invalidates other tokens."""
    if not raw_token or not isinstance(raw_token, str):
        return False, "Missing reset token."

    thash = hash_token(raw_token)
    conn = get_db_connection(db_path)
    try:
        with conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM password_reset_tokens WHERE token_hash = ? AND is_used = 0 AND is_revoked = 0",
                (thash,)
            )
            row = cursor.fetchone()
            if not row:
                return False, "Invalid, expired, or revoked reset token."

            record = dict(row)
            try:
                exp_dt = datetime.fromisoformat(record["expires_at"])
                if exp_dt.tzinfo is None:
                    exp_dt = exp_dt.replace(tzinfo=timezone.utc)
            except Exception:
                return False, "Corrupted token expiration timestamp."

            if datetime.now(timezone.utc) > exp_dt:
                return False, "This password reset token has expired."

            username = record["username"]
            # 1. Update user password
            cursor.execute("UPDATE users SET password_hash = ? WHERE username = ?", (new_password_hash, username))
            # 2. Mark this token as used
            cursor.execute("UPDATE password_reset_tokens SET is_used = 1 WHERE id = ?", (record["id"],))
            # 3. Invalidate any remaining outstanding tokens for this user
            cursor.execute(
                "UPDATE password_reset_tokens SET is_revoked = 1 WHERE username = ? AND id != ?",
                (username, record["id"])
            )
        return True, "Password reset successfully."
    except Exception as exc:
        return False, str(exc)
    finally:
        conn.close()

def db_get_all_users(db_path: str = None):
    """Retrieves all registered users for admin dashboard."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT username, email, role, fingerprint_verified, created_at FROM users ORDER BY created_at ASC")
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]
