import os
import re
import hashlib
import secrets
import sqlite3
from datetime import datetime, timezone, timedelta

DB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "signature_system.db")

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

def validate_password(pwd: str, min_length: int = 12) -> tuple:
    """Validates password constraints (default minimum 12 characters)."""
    if not pwd or len(pwd) < min_length:
        return False, f"Password must be at least {min_length} characters long."
    return True, ""

def generate_reset_token() -> str:
    """Generates a cryptographically secure, random, URL-safe token."""
    return secrets.token_urlsafe(32)

def init_db(db_path: str = None):
    """Initializes SQLite tables for users, signatures, audit trail, and security tokens."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    # 1. Users table (Functional Requirement 1)
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

    # 2. Signature Database table (Functional Requirement 4)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS signatures (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            signature_name TEXT NOT NULL,
            image_data BLOB NOT NULL,
            file_type TEXT NOT NULL,
            uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (username) REFERENCES users(username)
        )
    """)

    # 3. Audit Trail table (Functional Requirement 5)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS audit_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            reference_source TEXT NOT NULL,
            test_signature_name TEXT NOT NULL,
            result TEXT NOT NULL,
            confidence REAL NOT NULL,
            fraud_risk TEXT NOT NULL,
            details TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # 4. Secure Password Reset Tokens table
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

# -----------------------------
# User Account Queries
# -----------------------------
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

def db_create_user(username: str, email: str, password_hash: str, role: str, fingerprint_verified: int = 1, db_path: str = None):
    """Creates a new user record in the database."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO users (username, email, password_hash, role, fingerprint_verified) VALUES (?, ?, ?, ?, ?)",
            (username.strip(), email.strip(), password_hash, role, fingerprint_verified)
        )
        conn.commit()
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

def db_direct_reset_password(username: str, email: str, new_password_hash: str, db_path: str = None):
    """Direct in-app password reset matching registered username and email."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username = ? AND LOWER(email) = LOWER(?)", (username.strip(), email.strip()))
    row = cursor.fetchone()
    if not row:
        conn.close()
        return False, "Username and registered email do not match any existing record."
    cursor.execute("UPDATE users SET password_hash = ? WHERE username = ?", (new_password_hash, username.strip()))
    conn.commit()
    conn.close()
    return True, "Password reset successfully!"

def db_get_all_users(db_path: str = None):
    """Retrieves all registered users for admin dashboard."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT username, email, role, fingerprint_verified, created_at FROM users ORDER BY created_at ASC")
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]

def db_delete_user(username: str, db_path: str = None):
    """Deletes a user and their associated signatures."""
    if username in ("venkatesan", "admin"):
        return False, "Protected system administrator account cannot be deleted."
    conn = get_db_connection(db_path)
    try:
        with conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM signatures WHERE username = ?", (username,))
            cursor.execute("DELETE FROM users WHERE username = ?", (username,))
        return True, f"User '{username}' and associated data removed."
    except Exception as exc:
        return False, str(exc)
    finally:
        conn.close()

# -----------------------------
# Signature Database Management (Requirement 4)
# -----------------------------
def db_save_signature(username: str, signature_name: str, image_bytes: bytes, file_type: str = "png", db_path: str = None):
    """Saves a genuine reference signature to the user's signature database."""
    conn = get_db_connection(db_path)
    try:
        with conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO signatures (username, signature_name, image_data, file_type) VALUES (?, ?, ?, ?)",
                (username.strip(), signature_name.strip(), sqlite3.Binary(image_bytes), file_type.lower())
            )
        return True, "Signature stored successfully in database."
    except Exception as exc:
        return False, str(exc)
    finally:
        conn.close()

def db_get_user_signatures(username: str, db_path: str = None):
    """Fetches all stored signatures for a given user."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT id, username, signature_name, file_type, uploaded_at FROM signatures WHERE username = ? ORDER BY uploaded_at DESC", (username.strip(),))
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]

def db_get_signature_by_id(signature_id: int, db_path: str = None):
    """Fetches complete signature record including binary image data."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM signatures WHERE id = ?", (signature_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None

def db_delete_signature(signature_id: int, username: str = None, db_path: str = None):
    """Deletes a stored signature."""
    conn = get_db_connection(db_path)
    try:
        with conn:
            cursor = conn.cursor()
            if username:
                cursor.execute("DELETE FROM signatures WHERE id = ? AND username = ?", (signature_id, username))
            else:
                cursor.execute("DELETE FROM signatures WHERE id = ?", (signature_id,))
        return True, "Signature removed from database."
    except Exception as exc:
        return False, str(exc)
    finally:
        conn.close()

def db_get_all_signatures(db_path: str = None):
    """Retrieves all signature records across all users for admin view."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT id, username, signature_name, file_type, uploaded_at FROM signatures ORDER BY uploaded_at DESC")
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]

# -----------------------------
# Audit Trail Logging (Requirement 5)
# -----------------------------
def db_log_verification(username: str, reference_source: str, test_name: str, result: str, confidence: float, fraud_risk: str = "Low", details: str = "", db_path: str = None):
    """Logs a signature verification attempt into the audit trail."""
    conn = get_db_connection(db_path)
    try:
        with conn:
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO audit_logs (username, reference_source, test_signature_name, result, confidence, fraud_risk, details)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (username, reference_source, test_name, result, confidence, fraud_risk, details)
            )
        return True
    except Exception:
        return False
    finally:
        conn.close()

def db_get_audit_logs(limit: int = 200, db_path: str = None):
    """Retrieves audit trail records for admin review."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM audit_logs ORDER BY timestamp DESC LIMIT ?", (limit,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]

def db_get_audit_stats(db_path: str = None):
    """Calculates audit trail statistics for admin overview."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM audit_logs")
    total = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM audit_logs WHERE result = 'Genuine'")
    genuine = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM audit_logs WHERE result = 'Forged'")
    forged = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM audit_logs WHERE fraud_risk = 'High' OR result = 'Forged'")
    suspicious = cursor.fetchone()[0]
    conn.close()
    return {
        "total": total,
        "genuine": genuine,
        "forged": forged,
        "suspicious": suspicious
    }

# -----------------------------
# Token-Based Password Reset Helpers (Backwards compatibility)
# -----------------------------
def db_create_reset_token(username: str, email: str, raw_token: str, valid_minutes: int = 15, db_path: str = None):
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
    if not raw_token or not isinstance(raw_token, str):
        return None, "Reset token is missing or empty."
    thash = hash_token(raw_token)
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM password_reset_tokens WHERE token_hash = ?", (thash,))
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
            cursor.execute("UPDATE users SET password_hash = ? WHERE username = ?", (new_password_hash, username))
            cursor.execute("UPDATE password_reset_tokens SET is_used = 1 WHERE id = ?", (record["id"],))
            cursor.execute("UPDATE password_reset_tokens SET is_revoked = 1 WHERE username = ? AND id != ?", (username, record["id"]))
        return True, "Password reset successfully."
    except Exception as exc:
        return False, str(exc)
    finally:
        conn.close()
