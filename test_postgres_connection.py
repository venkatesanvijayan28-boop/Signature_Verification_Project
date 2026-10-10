import os
import sys
import uuid
from datetime import datetime, timezone

from auth_db import (
    get_database_url,
    mask_connection_url,
    get_postgres_connection,
    init_postgres_db,
    hash_password,
    generate_reset_token,
    db_get_user,
    db_get_user_by_email,
    db_create_user,
    db_update_password,
    db_create_reset_token,
    db_verify_reset_token,
    db_reset_password_with_token,
    DatabaseConnectionError,
    HAS_PSYCOPG2
)

def run_postgres_audit():
    print("=" * 65)
    print("   PostgreSQL Connectivity & Integration Audit")
    print("=" * 65)

    if not HAS_PSYCOPG2:
        print("[FAIL] psycopg2 is not installed. Please install psycopg2-binary.")
        return 1

    db_url = get_database_url()
    if not db_url:
        print("[SKIP] DATABASE_URL is not set in environment or Streamlit secrets.")
        print("To run live database integration tests against Supabase PostgreSQL, set:")
        print("   $env:DATABASE_URL=\"postgresql://postgres:[PASSWORD]@[HOST]:[PORT]/postgres\"")
        print("=" * 65)
        return 0

    masked_url = mask_connection_url(db_url)
    print(f"Target Database URL: {masked_url}")
    print("Attempting live connection...")

    try:
        conn = get_postgres_connection()
        print("[PASS] Successfully established live TCP/TLS connection to PostgreSQL!")
    except DatabaseConnectionError as exc:
        print(f"[FAIL] Could not connect to PostgreSQL: {exc}")
        print("=" * 65)
        return 1

    try:
        # 1. Verify PostgreSQL version
        with conn.cursor() as cur:
            cur.execute("SELECT version();")
            pg_version = cur.fetchone()["version"]
            print(f"[PASS] Live PostgreSQL Engine: {pg_version[:60]}...")

        # 2. Schema verification & initialization
        print("Verifying schema and tables...")
        init_postgres_db(conn)

        with conn.cursor() as cur:
            cur.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                  AND table_name IN ('users', 'password_reset_tokens');
            """)
            tables = [row["table_name"] for row in cur.fetchall()]
            print(f"[PASS] Verified tables in public schema: {tables}")
            if "users" not in tables or "password_reset_tokens" not in tables:
                print("[FAIL] Expected tables ('users', 'password_reset_tokens') not found.")
                return 1

        # 3. Live CRUD Integration Test with temporary user
        test_uid = uuid.uuid4().hex[:8]
        test_uname = f"audit_user_{test_uid}"
        test_email = f"audit_{test_uid}@example.com"
        test_pwd = "InitialStrongPassword2026!#"

        print(f"Testing live registration for: {test_uname} ({test_email})...")
        ok, msg = db_create_user(test_uname, test_email, hash_password(test_pwd), "User", 1)
        if not ok:
            print(f"[FAIL] Registration failed: {msg}")
            return 1
        print("[PASS] User record created in PostgreSQL.")

        # Query verification
        user_by_uname = db_get_user(test_uname)
        user_by_mail = db_get_user_by_email(test_email)
        if not user_by_uname or not user_by_mail:
            print("[FAIL] User lookup failed after insertion.")
            return 1
        print("[PASS] Verified user lookup by username and email.")

        # Password update verification
        new_pwd = "UpdatedStrongPassword2026!#"
        db_update_password(test_uname, hash_password(new_pwd))
        updated_user = db_get_user(test_uname)
        if updated_user["password_hash"] != hash_password(new_pwd):
            print("[FAIL] Password update was not persisted.")
            return 1
        print("[PASS] Verified direct password update in PostgreSQL.")

        # Reset token workflow
        token = generate_reset_token()
        db_create_reset_token(test_uname, test_email, token, valid_minutes=15)
        rec, err = db_verify_reset_token(token)
        if err or not rec:
            print(f"[FAIL] Token verification failed: {err}")
            return 1
        print("[PASS] Verified single-use reset token issuance and hash lookup.")

        # Reset password with token
        final_pwd = "FinalStrongPassword2026!#"
        reset_ok, reset_msg = db_reset_password_with_token(token, hash_password(final_pwd))
        if not reset_ok:
            print(f"[FAIL] Token-based reset failed: {reset_msg}")
            return 1
        print("[PASS] Verified token-based password reset in PostgreSQL.")

        # Verify token cannot be reused
        rec_after, err_after = db_verify_reset_token(token)
        if rec_after or "already been used" not in (err_after or "").lower():
            print("[FAIL] Re-use of consumed token was not rejected.")
            return 1
        print("[PASS] Verified used token cannot be reused.")

        # 4. Cleanup test data
        with conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM password_reset_tokens WHERE username = %s;", (test_uname,))
                cur.execute("DELETE FROM users WHERE username = %s;", (test_uname,))
        print("[PASS] Cleaned up temporary audit records.")

        print("=" * 65)
        print("   ALL POSTGRESQL LIVE INTEGRATION CHECKS PASSED")
        print("=" * 65)
        return 0

    except Exception as exc:
        print(f"[FAIL] Unexpected error during PostgreSQL audit: {exc}")
        print("=" * 65)
        return 1
    finally:
        conn.close()

if __name__ == "__main__":
    sys.exit(run_postgres_audit())
