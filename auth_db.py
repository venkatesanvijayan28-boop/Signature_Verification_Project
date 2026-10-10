import os
import re
import hashlib
import secrets
import sqlite3
import urllib.parse
from datetime import datetime, timezone, timedelta

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    HAS_PSYCOPG2 = True
except ImportError:
    HAS_PSYCOPG2 = False

try:
    from supabase import create_client, Client
    HAS_SUPABASE = True
except ImportError:
    HAS_SUPABASE = False

DB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "signature_system.db")

# =============================
# Supabase & PostgreSQL Credentials
# =============================
DEFAULT_SUPABASE_URL = "https://lxudtklogmuoypkpaovc.supabase.co"
DEFAULT_SUPABASE_KEY = "sb_publishable_odJu-vWFQ9Ujjgk-CUTBLA_frCe3Qv5"

DEFAULT_PG_PASSWORD = "$$Venkat@28$$"
ENCODED_PG_PASSWORD = urllib.parse.quote_plus(DEFAULT_PG_PASSWORD)
DEFAULT_POSTGRES_URL = f"postgresql://postgres:{ENCODED_PG_PASSWORD}@db.lxudtklogmuoypkpaovc.supabase.co:5432/postgres"

def get_postgres_url() -> str:
    """Returns the PostgreSQL connection string with password properly encoded."""
    url = None
    try:
        import streamlit as st
        url = st.secrets.get("DATABASE_URL") or st.secrets.get("POSTGRES_URL")
    except Exception:
        pass
    if not url:
        url = os.environ.get("DATABASE_URL") or os.environ.get("POSTGRES_URL", DEFAULT_POSTGRES_URL)
    return url

def get_postgres_connection():
    """Attempts direct connection to Supabase PostgreSQL database."""
    if not HAS_PSYCOPG2:
        return None
    url = get_postgres_url()
    try:
        conn = psycopg2.connect(url, cursor_factory=RealDictCursor, connect_timeout=3)
        return conn
    except Exception:
        return None

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
        client.auth.verify_otp({
            "email": email.strip(),
            "token": token.strip(),
            "type": "recovery"
        })
        client.auth.update_user({"password": new_password})
        return True, "Password updated in Supabase successfully."
    except Exception as exc:
        return False, str(exc)

# =============================
# Local SQLite & Helper Methods
# =============================
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
    """Initializes tables in both Supabase PostgreSQL and local SQLite."""
    # 1. Initialize SQLite
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

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
        cursor.execute(
            "UPDATE users SET email = ? WHERE username = ? AND email != ?",
            ("venkatesanvijayan28@gmail.com", "venkatesan", "venkatesanvijayan28@gmail.com")
        )
        conn.commit()

    conn.close()

    # 2. Try initializing Supabase PostgreSQL if accessible
    if not db_path:
        pg_conn = get_postgres_connection()
        if pg_conn:
            try:
                with pg_conn:
                    cur = pg_conn.cursor()
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS users (
                            username TEXT PRIMARY KEY,
                            email TEXT NOT NULL,
                            password_hash TEXT NOT NULL,
                            role TEXT NOT NULL,
                            fingerprint_verified INTEGER DEFAULT 1,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        );
                    """)
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS password_reset_tokens (
                            id SERIAL PRIMARY KEY,
                            username TEXT NOT NULL,
                            email TEXT NOT NULL,
                            token_hash TEXT NOT NULL UNIQUE,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            expires_at TIMESTAMP NOT NULL,
                            is_used INTEGER DEFAULT 0,
                            is_revoked INTEGER DEFAULT 0
                        );
                    """)
                    cur.execute("SELECT COUNT(*) FROM users;")
                    res = cur.fetchone()
                    if res and res.get('count', 0) == 0:
                        cur.execute(
                            "INSERT INTO users (username, email, password_hash, role, fingerprint_verified) VALUES (%s, %s, %s, %s, %s) ON CONFLICT DO NOTHING;",
                            ("venkatesan", "venkatesanvijayan28@gmail.com", hash_password("venkat@28"), "Admin", 1)
                        )
                pg_conn.close()
            except Exception:
                pass

def db_get_user(username: str, db_path: str = None):
    """Fetches user record by username from PostgreSQL or SQLite."""
    if not db_path:
        pg_conn = get_postgres_connection()
        if pg_conn:
            try:
                cur = pg_conn.cursor()
                cur.execute("SELECT * FROM users WHERE username = %s;", (username.strip(),))
                row = cur.fetchone()
                pg_conn.close()
                if row:
                    return dict(row)
            except Exception:
                pass

    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username = ?", (username.strip(),))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None

def db_get_user_by_email(email: str, db_path: str = None):
    """Fetches user record by email (case-insensitive) from PostgreSQL or SQLite."""
    if not db_path:
        pg_conn = get_postgres_connection()
        if pg_conn:
            try:
                cur = pg_conn.cursor()
                cur.execute("SELECT * FROM users WHERE LOWER(email) = LOWER(%s);", (email.strip(),))
                row = cur.fetchone()
                pg_conn.close()
                if row:
                    return dict(row)
            except Exception:
                pass

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
    """Creates user in PostgreSQL, SQLite, and auto-syncs with Supabase Auth."""
    # 1. Sync to PostgreSQL if available
    if not db_path:
        pg_conn = get_postgres_connection()
        if pg_conn:
            try:
                with pg_conn:
                    cur = pg_conn.cursor()
                    cur.execute(
                        "INSERT INTO users (username, email, password_hash, role, fingerprint_verified) VALUES (%s, %s, %s, %s, %s);",
                        (username.strip(), email.strip(), password_hash, role, fingerprint_verified)
                    )
                pg_conn.close()
            except Exception:
                pass

    # 2. Save in SQLite
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO users (username, email, password_hash, role, fingerprint_verified) VALUES (?, ?, ?, ?, ?)",
            (username.strip(), email.strip(), password_hash, role, fingerprint_verified)
        )
        conn.commit()

        # 3. Auto-sync to Supabase Auth if raw password provided
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
    """Updates password hash in both PostgreSQL and SQLite."""
    if not db_path:
        pg_conn = get_postgres_connection()
        if pg_conn:
            try:
                with pg_conn:
                    cur = pg_conn.cursor()
                    cur.execute("UPDATE users SET password_hash = %s WHERE username = %s;", (new_password_hash, username.strip()))
                pg_conn.close()
            except Exception:
                pass

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
            cursor.execute(
                "UPDATE password_reset_tokens SET is_revoked = 1 WHERE username = ? AND is_used = 0 AND is_revoked = 0",
                (username.strip(),)
            )
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
            # 1. Update user password in SQLite
            cursor.execute("UPDATE users SET password_hash = ? WHERE username = ?", (new_password_hash, username))
            # 2. Mark this token as used
            cursor.execute("UPDATE password_reset_tokens SET is_used = 1 WHERE id = ?", (record["id"],))
            # 3. Invalidate any remaining outstanding tokens for this user
            cursor.execute(
                "UPDATE password_reset_tokens SET is_revoked = 1 WHERE username = ? AND id != ?",
                (username, record["id"])
            )

        # Also sync password update to PostgreSQL if connected
        if not db_path:
            pg_conn = get_postgres_connection()
            if pg_conn:
                try:
                    with pg_conn:
                        cur = pg_conn.cursor()
                        cur.execute("UPDATE users SET password_hash = %s WHERE username = %s;", (new_password_hash, username))
                    pg_conn.close()
                except Exception:
                    pass

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
