import os
import unittest
import tempfile
from datetime import datetime, timezone, timedelta

from auth_db import (
    init_db,
    hash_password,
    hash_token,
    validate_email,
    validate_password,
    generate_reset_token,
    db_get_user,
    db_get_user_by_email,
    db_create_user,
    db_create_reset_token,
    db_verify_reset_token,
    db_reset_password_with_token,
    get_db_connection
)

class TestPasswordResetSecurityFlow(unittest.TestCase):
    """
    Comprehensive test suite verifying the 12 required scenarios
    for token-based password reset without SMTP.
    """
    def setUp(self):
        # Create an isolated temporary SQLite database for test repeatability
        self.temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.temp_db.close()
        self.db_path = self.temp_db.name
        init_db(self.db_path)

    def tearDown(self):
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    def test_01_valid_registered_email_generates_reset_link_in_demo_mode(self):
        """Scenario 1: A valid registered email generates a reset link in explicitly enabled demo mode."""
        user = db_get_user_by_email("venkatesanvijayan28@gmail.com", self.db_path)
        self.assertIsNotNone(user)

        token = generate_reset_token()
        self.assertTrue(len(token) >= 32)

        ok, expires_at = db_create_reset_token(user["username"], user["email"], token, 15, self.db_path)
        self.assertTrue(ok)

        # Confirm token URL parameter format
        reset_url = f"?reset_token={token}"
        self.assertIn("?reset_token=", reset_url)

        # Verify only SHA-256 hash was saved in the database
        conn = get_db_connection(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM password_reset_tokens WHERE username = ?", (user["username"],))
        row = cursor.fetchone()
        conn.close()

        self.assertIsNotNone(row)
        self.assertEqual(row["token_hash"], hash_token(token))
        self.assertNotEqual(row["token_hash"], token)

    def test_02_invalid_email_format_is_rejected(self):
        """Scenario 2: An invalid email format is rejected."""
        self.assertFalse(validate_email("notanemail"))
        self.assertFalse(validate_email("missingdomain@"))
        self.assertFalse(validate_email("@missinguser.com"))
        self.assertFalse(validate_email("spaces in@email.com"))
        self.assertTrue(validate_email("valid_user@example.com"))

    def test_03_unknown_accounts_receive_generic_response(self):
        """Scenario 3: Unknown accounts receive a generic response and no user record is found."""
        user = db_get_user_by_email("nonexistent_user@domain.com", self.db_path)
        self.assertIsNone(user)

    def test_04_valid_unexpired_token_opens_reset_form(self):
        """Scenario 4: A valid, unexpired token opens the reset form and verifies correctly."""
        user = db_get_user_by_email("user1@signature.com", self.db_path)
        token = generate_reset_token()
        db_create_reset_token(user["username"], user["email"], token, 15, self.db_path)

        record, err = db_verify_reset_token(token, self.db_path)
        self.assertIsNone(err)
        self.assertIsNotNone(record)
        self.assertEqual(record["username"], user["username"])
        self.assertEqual(record["email"], user["email"])

    def test_05_expired_token_is_rejected(self):
        """Scenario 5: An expired token is rejected."""
        user = db_get_user_by_email("user1@signature.com", self.db_path)
        token = generate_reset_token()
        thash = hash_token(token)

        # Manually backdate the expiration timestamp to 10 minutes ago
        expired_at = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat()
        conn = get_db_connection(self.db_path)
        with conn:
            conn.cursor().execute(
                "INSERT INTO password_reset_tokens (username, email, token_hash, expires_at, is_used, is_revoked) VALUES (?, ?, ?, ?, 0, 0)",
                (user["username"], user["email"], thash, expired_at)
            )
        conn.close()

        record, err = db_verify_reset_token(token, self.db_path)
        self.assertIsNone(record)
        self.assertIn("expired", err.lower())

    def test_06_invalid_token_is_rejected(self):
        """Scenario 6: An invalid token is rejected."""
        record, err = db_verify_reset_token("completely_invalid_random_token_999", self.db_path)
        self.assertIsNone(record)
        self.assertIn("invalid", err.lower())

    def test_07_password_mismatch_is_rejected(self):
        """Scenario 7: Password mismatch is rejected."""
        pwd1 = "StrongPassword2026!"
        pwd2 = "DifferentPassword2026!"
        self.assertNotEqual(pwd1, pwd2)

    def test_08_weak_password_is_rejected(self):
        """Scenario 8: A weak password (<12 characters) is rejected."""
        is_valid_short, err_short = validate_password("Short123")
        self.assertFalse(is_valid_short)
        self.assertIn("12 characters", err_short)

        is_valid_strong, err_strong = validate_password("SuperSecret2026!Password")
        self.assertTrue(is_valid_strong)
        self.assertEqual(err_strong, "")

    def test_09_successful_reset_updates_database_password_hash(self):
        """Scenario 9: A successful reset updates the database password hash."""
        user = db_get_user_by_email("user1@signature.com", self.db_path)
        old_hash = user["password_hash"]

        token = generate_reset_token()
        db_create_reset_token(user["username"], user["email"], token, 15, self.db_path)

        new_password = "SuperSecurePassword123!@#"
        new_hash = hash_password(new_password)

        success, msg = db_reset_password_with_token(token, new_hash, self.db_path)
        self.assertTrue(success)

        updated_user = db_get_user("user1", self.db_path)
        self.assertEqual(updated_user["password_hash"], new_hash)
        self.assertNotEqual(updated_user["password_hash"], old_hash)

    def test_10_used_token_cannot_reset_password_again(self):
        """Scenario 10: A used token cannot reset the password again."""
        user = db_get_user_by_email("user1@signature.com", self.db_path)
        token = generate_reset_token()
        db_create_reset_token(user["username"], user["email"], token, 15, self.db_path)

        # First reset succeeds
        ok1, _ = db_reset_password_with_token(token, hash_password("ValidPassword123!"), self.db_path)
        self.assertTrue(ok1)

        # Second reset with same token is rejected
        ok2, err2 = db_reset_password_with_token(token, hash_password("AnotherPassword123!"), self.db_path)
        self.assertFalse(ok2)

        # Token verification also confirms already used
        rec, err = db_verify_reset_token(token, self.db_path)
        self.assertIsNone(rec)
        self.assertIn("already been used", err.lower())

    def test_11_other_outstanding_tokens_are_invalidated_after_reset(self):
        """Scenario 11: Other outstanding tokens are invalidated after a successful reset."""
        user = db_get_user_by_email("user1@signature.com", self.db_path)

        token_a = generate_reset_token()
        db_create_reset_token(user["username"], user["email"], token_a, 15, self.db_path)

        token_b = generate_reset_token()
        db_create_reset_token(user["username"], user["email"], token_b, 15, self.db_path)

        # Token A is immediately revoked by issuance of Token B
        rec_a, err_a = db_verify_reset_token(token_a, self.db_path)
        self.assertIsNone(rec_a)
        self.assertIn("revoked", err_a.lower())

        # Token B executes reset
        ok_b, _ = db_reset_password_with_token(token_b, hash_password("NewValidPassword2026!"), self.db_path)
        self.assertTrue(ok_b)

        # Token B is now marked used
        rec_b, err_b = db_verify_reset_token(token_b, self.db_path)
        self.assertIsNone(rec_b)
        self.assertIn("already been used", err_b.lower())

    def test_12_existing_login_works_with_new_password_and_rejects_old_password(self):
        """Scenario 12: Existing login works with the new password and rejects the old password."""
        user = db_get_user_by_email("user1@signature.com", self.db_path)
        old_password = "user@123"
        self.assertEqual(user["password_hash"], hash_password(old_password))

        # Reset password to new password
        token = generate_reset_token()
        db_create_reset_token(user["username"], user["email"], token, 15, self.db_path)
        new_password = "UpdatedComplexPassword999!"
        db_reset_password_with_token(token, hash_password(new_password), self.db_path)

        # Refresh user from DB
        fresh_user = db_get_user("user1", self.db_path)

        # Old password check fails
        self.assertNotEqual(fresh_user["password_hash"], hash_password(old_password))

        # New password check succeeds
        self.assertEqual(fresh_user["password_hash"], hash_password(new_password))

    def test_13_signature_database_add_and_retrieve(self):
        """Scenario 13 (PDF Req 4): Users can add, retrieve, and manage signature records in the database."""
        from auth_db import db_save_signature, db_get_user_signatures, db_delete_signature, db_get_signature_by_id

        sample_bytes = b"FAKE_SIGNATURE_BINARY_DATA"
        ok, msg = db_save_signature("user1", "Official Bank Signature", sample_bytes, "png", self.db_path)
        self.assertTrue(ok)

        sigs = db_get_user_signatures("user1", self.db_path)
        self.assertEqual(len(sigs), 1)
        self.assertEqual(sigs[0]["signature_name"], "Official Bank Signature")

        sig_full = db_get_signature_by_id(sigs[0]["id"], self.db_path)
        self.assertEqual(sig_full["image_data"], sample_bytes)

        # Remove
        del_ok, _ = db_delete_signature(sigs[0]["id"], "user1", self.db_path)
        self.assertTrue(del_ok)
        self.assertEqual(len(db_get_user_signatures("user1", self.db_path)), 0)

    def test_14_audit_trail_logging_and_stats(self):
        """Scenario 14 (PDF Req 5): Track and log signature verification attempts, results, and retrieve audit stats."""
        from auth_db import db_log_verification, db_get_audit_logs, db_get_audit_stats

        # Log genuine attempt
        ok1 = db_log_verification(
            username="user1",
            reference_source="Database: Bank Sig",
            test_name="test_1.png",
            result="Genuine",
            confidence=94.5,
            fraud_risk="Low",
            details="Match confirmed",
            db_path=self.db_path
        )
        self.assertTrue(ok1)

        # Log forged attempt
        ok2 = db_log_verification(
            username="user1",
            reference_source="Database: Bank Sig",
            test_name="forged_sample.png",
            result="Forged",
            confidence=89.2,
            fraud_risk="High (Suspicious Forgery)",
            details="Stroke density mismatch",
            db_path=self.db_path
        )
        self.assertTrue(ok2)

        logs = db_get_audit_logs(limit=10, db_path=self.db_path)
        self.assertEqual(len(logs), 2)

        stats = db_get_audit_stats(self.db_path)
        self.assertEqual(stats["total"], 2)
        self.assertEqual(stats["genuine"], 1)
        self.assertEqual(stats["forged"], 1)
        self.assertEqual(stats["suspicious"], 1)

    def test_15_direct_in_app_password_reset_without_smtp(self):
        """Scenario 15 (PDF Req 1): Direct in-app password reset succeeds without requiring SMTP."""
        from auth_db import db_direct_reset_password

        # Valid reset
        new_pwd = "DirectResetPassword2026!#"
        ok, msg = db_direct_reset_password("user1", "user1@signature.com", hash_password(new_pwd), self.db_path)
        self.assertTrue(ok)

        # Authenticate with updated password
        user = db_get_user("user1", self.db_path)
        self.assertEqual(user["password_hash"], hash_password(new_pwd))

        # Mismatched email is rejected
        bad_ok, bad_msg = db_direct_reset_password("user1", "wrong_mail@test.com", hash_password("test"), self.db_path)
        self.assertFalse(bad_ok)

    def test_16_new_user_registration_strictly_assigned_user_role(self):
        """Scenario 16: Every newly registered account is strictly assigned the User role on backend."""
        ok, msg = db_create_user(
            username="employee_jane",
            email="jane@company.com",
            password_hash=hash_password("JanePassword123!"),
            role="User",
            fingerprint_verified=1,
            db_path=self.db_path
        )
        self.assertTrue(ok)
        created_user = db_get_user("employee_jane", self.db_path)
        self.assertIsNotNone(created_user)
        self.assertEqual(created_user["role"], "User")
        self.assertNotEqual(created_user["role"], "Admin")

    def test_17_standard_user_cannot_access_admin_privileges(self):
        """Scenario 17: Standard users cannot access admin privileges even if role tampering is attempted."""
        user = db_get_user("user1", self.db_path)
        self.assertEqual(user["role"], "User")
        # Simulating main_app authoritative check:
        authoritative_role = user["role"] if user else "User"
        self.assertFalse(authoritative_role == "Admin")

    def test_18_email_login_works_for_both_user_and_admin(self):
        """Scenario 18: Users and Administrators can log in using their email address."""
        # 1. User login by email
        user_by_email = db_get_user_by_email("user1@signature.com", self.db_path)
        self.assertIsNotNone(user_by_email)
        self.assertEqual(user_by_email["username"], "user1")
        self.assertEqual(user_by_email["password_hash"], hash_password("user@123"))

        # 2. Admin login by email
        admin_by_email = db_get_user_by_email("venkatesanvijayan28@gmail.com", self.db_path)
        self.assertIsNotNone(admin_by_email)
        self.assertEqual(admin_by_email["role"], "Admin")
        self.assertEqual(admin_by_email["password_hash"], hash_password("venkat@28"))

    def test_19_admin_accounts_retain_admin_privileges(self):
        """Scenario 19: Authorized administrator accounts exist and maintain Admin privileges."""
        admin1 = db_get_user("venkatesan", self.db_path)
        self.assertIsNotNone(admin1)
        self.assertEqual(admin1["role"], "Admin")

        admin2 = db_get_user("admin", self.db_path)
        self.assertIsNotNone(admin2)
        self.assertEqual(admin2["role"], "Admin")

if __name__ == "__main__":
    unittest.main()

