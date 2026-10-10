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
    from psycopg2 import sql
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
# Custom Database Exceptions
# =============================
class DatabaseError(Exception):
    """Base exception for authentication database operations."""
    pass

class DatabaseConnectionError(DatabaseError):
    """Raised when unable to connect to the authoritative database."""
    pass

# =============================
# Configuration Getters (Secrets / Environment ONLY)
# =============================
def get_database_url() -> str | None:
    """
    Reads DATABASE_URL exclusively from Streamlit secrets or environment variables.
    Never uses hardcoded credentials or passwords.
    """
    url = None
    try:
        import streamlit as st
        if hasattr(st, "secrets") and "DATABASE_URL" in st.secrets:
            val = st.secrets["DATABASE_URL"]
            if val and str(val).strip():
                url = str(val).strip()
    except Exception:
        pass

    if not url:
        val = os.environ.get("DATABASE_URL")
        if val and val.strip():
            url = val.strip()

    return url

def mask_connection_url(url: str | None) -> str:
    """Masks credentials in a database URL for safe logging/display."""
    if not url:
        return "<UNSET>"
    try:
        parsed = urllib.parse.urlsplit(url)
        if parsed.password:
            netloc = f"{parsed.username}:******@{parsed.hostname}"
            if parsed.port:
                netloc += f":{parsed.port}"
            return urllib.parse.urlunsplit((parsed.scheme, netloc, parsed.path, parsed.query, parsed.fragment))
        return url
    except Exception:
        return "<CONFIGURED_URL>"

def get_supabase_config() -> tuple[str | None, str | None]:
    """
    Reads Supabase URL and Key exclusively from Streamlit secrets or environment variables.
    """
    url = None
    key = None
    try:
        import streamlit as st
        if hasattr(st, "secrets"):
            url = st.secrets.get("SUPABASE_URL") or st.secrets.get("NEXT_PUBLIC_SUPABASE_URL")
            key = (
                st.secrets.get("SUPABASE_KEY")
                or st.secrets.get("NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY")
                or st.secrets.get("SUPABASE_ANON_KEY")
            )
    except Exception:
        pass

    if not url:
        url = os.environ.get("SUPABASE_URL") or os.environ.get("NEXT_PUBLIC_SUPABASE_URL")
    if not key:
        key = (
            os.environ.get("SUPABASE_KEY")
            or os.environ.get("NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY")
            or os.environ.get("SUPABASE_ANON_KEY")
        )

    return (url.strip() if url else None, key.strip() if key else None)

def is_postgres_authoritative() -> bool:
    """
    Returns True when DATABASE_URL is configured, making PostgreSQL authoritative.
    When True, the application MUST fail fast if PostgreSQL fails and NOT silently fall back.
    """
    return bool(get_database_url())

# =============================
# Database Connection Factories
# =============================
def get_postgres_connection():
    """
    Connects to authoritative PostgreSQL database.
    Raises DatabaseConnectionError on failure.
    """
    url = get_database_url()
    if not url:
        raise DatabaseConnectionError("DATABASE_URL is not configured in secrets or environment.")
    if not HAS_PSYCOPG2:
        raise DatabaseConnectionError("psycopg2 is not installed in the Python runtime.")
    try:
        conn = psycopg2.connect(url, cursor_factory=RealDictCursor, connect_timeout=10)
        return conn
    except Exception as exc:
        raise DatabaseConnectionError(f"PostgreSQL connection to {mask_connection_url(url)} failed: {exc}") from exc

def get_sqlite_connection(db_path: str = None):
    """Connects to local SQLite database (used for local development/testing without PostgreSQL)."""
    target = db_path or DB_FILE
    conn = sqlite3.connect(target, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

# Alias for backwards compatibility with tests
get_db_connection = get_sqlite_connection

def get_supabase_client():
    """Initializes and returns a Supabase client using configured secrets or env vars."""
    if not HAS_SUPABASE:
        return None
    url, key = get_supabase_config()
    if url and key:
        try:
            return create_client(url, key)
        except Exception:
            return None
    return None

# =============================
# Password & Token Helpers
# =============================
def hash_password(pwd: str) -> str:
    """Computes SHA-256 hash for password storage."""
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

def validate_password(pwd: str) -> tuple[bool, str]:
    """Validates password constraints (minimum 12 characters)."""
    if not pwd or len(pwd) < 12:
        return False, "Password must be at least 12 characters long."
    return True, ""

def generate_reset_token() -> str:
    """Generates a cryptographically secure, random, URL-safe token."""
    return secrets.token_urlsafe(32)

# =============================
# Schema Initialization
# =============================
def init_postgres_db(conn=None):
    """Initializes schema in PostgreSQL database with parameterized queries and transaction safety."""
    close_at_end = False
    if conn is None:
        conn = get_postgres_connection()
        close_at_end = True

    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                        username TEXT PRIMARY KEY,
                        email TEXT NOT NULL UNIQUE,
                        password_hash TEXT NOT NULL,
                        role TEXT NOT NULL,
                        fingerprint_verified INTEGER DEFAULT 1,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS password_reset_tokens (
                        id SERIAL PRIMARY KEY,
                        username TEXT NOT NULL,
                        email TEXT NOT NULL,
                        token_hash TEXT NOT NULL UNIQUE,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                        expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
                        is_used INTEGER DEFAULT 0,
                        is_revoked INTEGER DEFAULT 0
                    );
                """)
                cur.execute("SELECT COUNT(*) AS cnt FROM users;")
                res = cur.fetchone()
                if res and res["cnt"] == 0:
                    # Seed default accounts
                    cur.execute(
                        """
                        INSERT INTO users (username, email, password_hash, role, fingerprint_verified)
                        VALUES (%s, %s, %s, %s, %s)
                        ON CONFLICT (username) DO NOTHING;
                        """,
                        ("venkatesan", "venkatesanvijayan28@gmail.com", hash_password("venkat@28"), "Admin", 1)
                    )
                    cur.execute(
                        """
                        INSERT INTO users (username, email, password_hash, role, fingerprint_verified)
                        VALUES (%s, %s, %s, %s, %s)
                        ON CONFLICT (username) DO NOTHING;
                        """,
                        ("admin", "admin@signature.com", hash_password("admin@123"), "Admin", 1)
                    )
                    cur.execute(
                        """
                        INSERT INTO users (username, email, password_hash, role, fingerprint_verified)
                        VALUES (%s, %s, %s, %s, %s)
                        ON CONFLICT (username) DO NOTHING;
                        """,
                        ("user1", "user1@signature.com", hash_password("user@123"), "User", 1)
                    )
    finally:
        if close_at_end:
            conn.close()

def init_sqlite_db(db_path: str = None):
    """Initializes schema in SQLite database (local / test environment)."""
    conn = get_sqlite_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                email TEXT NOT NULL UNIQUE,
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
    finally:
        conn.close()

def init_db(db_path: str = None):
    """
    Initializes the database.
    If db_path is specified, explicitly targets SQLite (for isolated unit tests).
    If DATABASE_URL is configured, authoritative PostgreSQL is initialized (fails fast on error).
    Otherwise, defaults to local SQLite.
    """
    if db_path:
        init_sqlite_db(db_path)
    elif is_postgres_authoritative():
        init_postgres_db()
    else:
        init_sqlite_db()

# =============================
# Supabase Auth Synchronization
# =============================
def supabase_sync_user(email: str, password: str, username: str, role: str, fingerprint_verified: int = 1):
    """Auto-syncs user creation to Supabase Auth if Supabase client is configured."""
    client = get_supabase_client()
    if not client:
        return False, "Supabase client not configured."
    try:
        client.auth.sign_up({
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
    """Sends a password-reset verification email to the user's inbox via Supabase if configured."""
    client = get_supabase_client()
    if not client:
        return False, "Supabase client not configured."
    try:
        client.auth.reset_password_for_email(email.strip())
        return True, "Supabase password reset email dispatched."
    except Exception as exc:
        return False, str(exc)

def supabase_verify_otp_and_reset(email: str, token: str, new_password: str, db_path: str = None):
    """
    Verifies recovery OTP token, updates password in Supabase Auth,
    and synchronizes the updated password to the authoritative database to ensure consistency.
    """
    client = get_supabase_client()
    if not client:
        return False, "Supabase client not configured."
    try:
        client.auth.verify_otp({
            "email": email.strip(),
            "token": token.strip(),
            "type": "recovery"
        })
        client.auth.update_user({"password": new_password})

        # Ensure the authoritative database is synchronized with the new password
        user = db_get_user_by_email(email.strip(), db_path=db_path)
        if user:
            db_update_password(user["username"], hash_password(new_password), db_path=db_path)

        return True, "Password updated in Supabase Auth and authoritative database."
    except Exception as exc:
        return False, str(exc)

# =============================
# Authoritative Data Operations
# =============================
def db_get_user(username: str, db_path: str = None) -> dict | None:
    """Fetches user record by username from the authoritative database."""
    if not db_path and is_postgres_authoritative():
        conn = get_postgres_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM users WHERE username = %s;", (username.strip(),))
                row = cur.fetchone()
                return dict(row) if row else None
        finally:
            conn.close()

    conn = get_sqlite_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username = ?", (username.strip(),))
        row = cursor.fetchone()
        return dict(row) if row else None
    finally:
        conn.close()

def db_get_user_by_email(email: str, db_path: str = None) -> dict | None:
    """Fetches user record by email (case-insensitive) from the authoritative database."""
    clean_email = email.strip()
    if not db_path and is_postgres_authoritative():
        conn = get_postgres_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM users WHERE LOWER(email) = LOWER(%s);", (clean_email,))
                row = cur.fetchone()
                return dict(row) if row else None
        finally:
            conn.close()

    conn = get_sqlite_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE LOWER(email) = LOWER(?)", (clean_email,))
        row = cursor.fetchone()
        return dict(row) if row else None
    finally:
        conn.close()

def db_get_user_by_email_or_username(identifier: str, db_path: str = None) -> dict | None:
    """Fetches user record by either email or username."""
    clean_id = identifier.strip()
    if not db_path and is_postgres_authoritative():
        conn = get_postgres_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT * FROM users WHERE LOWER(email) = LOWER(%s) OR username = %s LIMIT 1;",
                    (clean_id, clean_id)
                )
                row = cur.fetchone()
                return dict(row) if row else None
        finally:
            conn.close()

    user = db_get_user_by_email(clean_id, db_path)
    if not user:
        user = db_get_user(clean_id, db_path)
    return user

def db_create_user(
    username: str,
    email: str,
    password_hash: str,
    role: str,
    fingerprint_verified: int = 1,
    raw_password: str = None,
    db_path: str = None
) -> tuple[bool, str]:
    """
    Creates a new user record in the authoritative database.
    Does not silently fall back to another database when authoritative fails.
    """
    if not db_path and is_postgres_authoritative():
        conn = get_postgres_connection()
        try:
            with conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO users (username, email, password_hash, role, fingerprint_verified)
                        VALUES (%s, %s, %s, %s, %s);
                        """,
                        (username.strip(), email.strip(), password_hash, role, fingerprint_verified)
                    )
            if raw_password:
                try:
                    supabase_sync_user(email, raw_password, username, role, fingerprint_verified)
                except Exception:
                    pass
            return True, "User registered successfully."
        except Exception as exc:
            err_str = str(exc).lower()
            if "unique constraint" in err_str or "already exists" in err_str:
                return False, f"Username '{username}' or email '{email}' already registered."
            raise DatabaseError(f"PostgreSQL user registration failed: {exc}") from exc
        finally:
            conn.close()

    conn = get_sqlite_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO users (username, email, password_hash, role, fingerprint_verified) VALUES (?, ?, ?, ?, ?)",
            (username.strip(), email.strip(), password_hash, role, fingerprint_verified)
        )
        conn.commit()
        if raw_password:
            try:
                supabase_sync_user(email, raw_password, username, role, fingerprint_verified)
            except Exception:
                pass
        return True, "User registered successfully."
    except sqlite3.IntegrityError:
        return False, f"Username '{username}' or email '{email}' already exists."
    finally:
        conn.close()

def db_update_password(username: str, new_password_hash: str, db_path: str = None) -> bool:
    """Updates password hash for a specific user in the authoritative database."""
    if not db_path and is_postgres_authoritative():
        conn = get_postgres_connection()
        try:
            with conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "UPDATE users SET password_hash = %s WHERE username = %s;",
                        (new_password_hash, username.strip())
                    )
            return True
        finally:
            conn.close()

    conn = get_sqlite_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("UPDATE users SET password_hash = ? WHERE username = ?", (new_password_hash, username.strip()))
        conn.commit()
        return True
    finally:
        conn.close()

def db_create_reset_token(username: str, email: str, raw_token: str, valid_minutes: int = 15, db_path: str = None) -> tuple[bool, str]:
    """Stores a hashed single-use token and invalidates previous tokens for this user."""
    thash = hash_token(raw_token)
    now_utc = datetime.now(timezone.utc)
    expires_at = now_utc + timedelta(minutes=valid_minutes)
    expires_at_iso = expires_at.isoformat()

    if not db_path and is_postgres_authoritative():
        conn = get_postgres_connection()
        try:
            with conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        UPDATE password_reset_tokens
                        SET is_revoked = 1
                        WHERE username = %s AND is_used = 0 AND is_revoked = 0;
                        """,
                        (username.strip(),)
                    )
                    cur.execute(
                        """
                        INSERT INTO password_reset_tokens (username, email, token_hash, expires_at, is_used, is_revoked)
                        VALUES (%s, %s, %s, %s, 0, 0);
                        """,
                        (username.strip(), email.strip(), thash, expires_at)
                    )
            return True, expires_at_iso
        finally:
            conn.close()

    conn = get_sqlite_connection(db_path)
    try:
        with conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE password_reset_tokens SET is_revoked = 1 WHERE username = ? AND is_used = 0 AND is_revoked = 0",
                (username.strip(),)
            )
            cursor.execute(
                "INSERT INTO password_reset_tokens (username, email, token_hash, expires_at, is_used, is_revoked) VALUES (?, ?, ?, ?, 0, 0)",
                (username.strip(), email.strip(), thash, expires_at_iso)
            )
        return True, expires_at_iso
    finally:
        conn.close()

def db_verify_reset_token(raw_token: str, db_path: str = None) -> tuple[dict | None, str | None]:
    """Validates raw token against stored hash, ensuring it is unexpired, unused, and unrevoked."""
    if not raw_token or not isinstance(raw_token, str):
        return None, "Reset token is missing or empty."

    thash = hash_token(raw_token)
    row = None

    if not db_path and is_postgres_authoritative():
        conn = get_postgres_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM password_reset_tokens WHERE token_hash = %s;", (thash,))
                row = cur.fetchone()
        finally:
            conn.close()
    else:
        conn = get_sqlite_connection(db_path)
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM password_reset_tokens WHERE token_hash = ?", (thash,))
            row = cursor.fetchone()
        finally:
            conn.close()

    if not row:
        return None, "Invalid reset token."

    record = dict(row)
    if record["is_used"]:
        return None, "This password reset token has already been used."

    if record["is_revoked"]:
        return None, "This password reset token has been revoked or superseded by a newer request."

    raw_exp = record["expires_at"]
    if isinstance(raw_exp, datetime):
        exp_dt = raw_exp
    else:
        try:
            exp_dt = datetime.fromisoformat(str(raw_exp))
        except Exception:
            return None, "Corrupted token expiration timestamp."

    if exp_dt.tzinfo is None:
        exp_dt = exp_dt.replace(tzinfo=timezone.utc)

    if datetime.now(timezone.utc) > exp_dt:
        return None, "This password reset token has expired (valid for 15 minutes)."

    return record, None

def db_reset_password_with_token(raw_token: str, new_password_hash: str, db_path: str = None) -> tuple[bool, str]:
    """Atomically updates password, marks token used, and invalidates other tokens."""
    if not raw_token or not isinstance(raw_token, str):
        return False, "Missing reset token."

    thash = hash_token(raw_token)

    if not db_path and is_postgres_authoritative():
        conn = get_postgres_connection()
        try:
            with conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT * FROM password_reset_tokens WHERE token_hash = %s AND is_used = 0 AND is_revoked = 0;",
                        (thash,)
                    )
                    row = cur.fetchone()
                    if not row:
                        return False, "Invalid, expired, or revoked reset token."

                    record = dict(row)
                    raw_exp = record["expires_at"]
                    exp_dt = raw_exp if isinstance(raw_exp, datetime) else datetime.fromisoformat(str(raw_exp))
                    if exp_dt.tzinfo is None:
                        exp_dt = exp_dt.replace(tzinfo=timezone.utc)

                    if datetime.now(timezone.utc) > exp_dt:
                        return False, "This password reset token has expired."

                    username = record["username"]
                    # 1. Update user password
                    cur.execute("UPDATE users SET password_hash = %s WHERE username = %s;", (new_password_hash, username))
                    # 2. Mark this token as used
                    cur.execute("UPDATE password_reset_tokens SET is_used = 1 WHERE id = %s;", (record["id"],))
                    # 3. Invalidate remaining outstanding tokens for user
                    cur.execute(
                        "UPDATE password_reset_tokens SET is_revoked = 1 WHERE username = %s AND id != %s;",
                        (username, record["id"])
                    )
            return True, "Password reset successfully in PostgreSQL."
        finally:
            conn.close()

    conn = get_sqlite_connection(db_path)
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
            raw_exp = record["expires_at"]
            exp_dt = raw_exp if isinstance(raw_exp, datetime) else datetime.fromisoformat(str(raw_exp))
            if exp_dt.tzinfo is None:
                exp_dt = exp_dt.replace(tzinfo=timezone.utc)

            if datetime.now(timezone.utc) > exp_dt:
                return False, "This password reset token has expired."

            username = record["username"]
            cursor.execute("UPDATE users SET password_hash = ? WHERE username = ?", (new_password_hash, username))
            cursor.execute("UPDATE password_reset_tokens SET is_used = 1 WHERE id = ?", (record["id"],))
            cursor.execute(
                "UPDATE password_reset_tokens SET is_revoked = 1 WHERE username = ? AND id != ?",
                (username, record["id"])
            )
        return True, "Password reset successfully in SQLite."
    finally:
        conn.close()

def db_get_all_users(db_path: str = None) -> list[dict]:
    """Retrieves all registered users for admin dashboard from authoritative database."""
    if not db_path and is_postgres_authoritative():
        conn = get_postgres_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT username, email, role, fingerprint_verified, created_at FROM users ORDER BY created_at ASC;")
                rows = cur.fetchall()
                return [dict(r) for r in rows]
        finally:
            conn.close()

    conn = get_sqlite_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT username, email, role, fingerprint_verified, created_at FROM users ORDER BY created_at ASC")
        rows = cursor.fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()
