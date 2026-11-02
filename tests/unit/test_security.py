from __future__ import annotations

from datetime import UTC, datetime, timedelta

import jwt
import pytest

from tripmate.errors import UnauthorizedError
from tripmate.security.passwords import hash_password, validate_password_strength, verify_password
from tripmate.security.tokens import decode_token, issue_token

NOW = datetime(2026, 10, 22, 4, 0, tzinfo=UTC)
SECRET = "s" * 40


class TestPasswords:
    def test_roundtrip_and_salting(self):
        a, b = hash_password("correct horse battery"), hash_password("correct horse battery")
        assert (
            a != b
            and verify_password("correct horse battery", a)
            and verify_password("correct horse battery", b)
        )

    def test_wrong_password_rejected(self):
        assert not verify_password("wrong password!!", hash_password("correct horse battery"))

    def test_unknown_user_never_verifies_but_still_does_work(self):
        assert verify_password("anything at all", None) is False

    @pytest.mark.parametrize(
        "stored", ["", "garbage", "scrypt$1$2", "bcrypt$1$2$3$4$5", "scrypt$x$y$z$!!$!!"]
    )
    def test_malformed_hashes_fail_closed(self, stored):
        assert verify_password("whatever12345", stored) is False

    def test_hash_does_not_contain_password(self):
        assert "correct horse" not in hash_password("correct horse battery")

    @pytest.mark.parametrize(
        "pw,ok",
        [
            ("short", False),
            ("x" * 200, False),
            ("aaaaaaaaaaaa", False),
            ("password123", False),
            ("a-good-passphrase-42", True),
        ],
    )
    def test_strength_policy(self, pw, ok):
        assert (validate_password_strength(pw) is None) is ok


class TestTokens:
    def test_roundtrip(self):
        t = issue_token("u1", SECRET, NOW, timedelta(minutes=30))
        assert decode_token(t, SECRET, NOW + timedelta(minutes=5)).user_id == "u1"

    def test_expired(self):
        t = issue_token("u1", SECRET, NOW, timedelta(minutes=30))
        with pytest.raises(UnauthorizedError, match="expired"):
            decode_token(t, SECRET, NOW + timedelta(minutes=31))

    def test_wrong_secret_and_tampering(self):
        t = issue_token("u1", SECRET, NOW, timedelta(minutes=30))
        with pytest.raises(UnauthorizedError):
            decode_token(t, "o" * 40, NOW)
        with pytest.raises(UnauthorizedError):
            decode_token(t[:-3] + "abc", SECRET, NOW)

    def test_alg_none_is_rejected(self):
        forged = jwt.encode(
            {"sub": "admin", "iss": "tripmate", "exp": int((NOW + timedelta(days=1)).timestamp())},
            key=None,
            algorithm="none",
        )
        with pytest.raises(UnauthorizedError):
            decode_token(forged, SECRET, NOW)

    def test_wrong_issuer_and_missing_claims(self):
        bad = jwt.encode(
            {"sub": "u", "iss": "evil", "exp": int((NOW + timedelta(days=1)).timestamp())},
            SECRET,
            algorithm="HS256",
        )
        with pytest.raises(UnauthorizedError):
            decode_token(bad, SECRET, NOW)
        nosub = jwt.encode(
            {"iss": "tripmate", "exp": int((NOW + timedelta(days=1)).timestamp())},
            SECRET,
            algorithm="HS256",
        )
        with pytest.raises(UnauthorizedError):
            decode_token(nosub, SECRET, NOW)

    @pytest.mark.parametrize("junk", ["", "a.b.c", "not-a-token"])
    def test_garbage(self, junk):
        with pytest.raises(UnauthorizedError):
            decode_token(junk, SECRET, NOW)
