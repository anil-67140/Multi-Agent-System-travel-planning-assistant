"""scrypt password hashing (stdlib; memory-hard; no extra dependency)."""

from __future__ import annotations

import base64
import hashlib
import hmac
import secrets

N, R, P = 2**14, 8, 1
KEY_LEN = 32
MIN_LENGTH = 10
MAX_LENGTH = 128  # bounds hashing cost for hostile input
_MAXMEM = 64 * 1024 * 1024


def validate_password_strength(password: str) -> str | None:
    """Return a problem description, or ``None`` if acceptable."""
    if len(password) < MIN_LENGTH:
        return f"Password must be at least {MIN_LENGTH} characters."
    if len(password) > MAX_LENGTH:
        return f"Password must be at most {MAX_LENGTH} characters."
    if password.lower() in {"password123", "1234567890", "qwertyuiop", "letmein1234"}:
        return "That password is too common."
    if len(set(password)) < 4:
        return "Password is too repetitive."
    return None


def _b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(
        password.encode(), salt=salt, n=N, r=R, p=P, dklen=KEY_LEN, maxmem=_MAXMEM
    )
    return f"scrypt${N}${R}${P}${_b64(salt)}${_b64(digest)}"


# Verified against when the user does not exist, so login time does not reveal valid emails.
_DUMMY = hash_password("not-a-real-password")


def verify_password(password: str, stored: str | None) -> bool:
    stored = stored or _DUMMY
    try:
        scheme, n, r, p, salt_b64, digest_b64 = stored.split("$")
        if scheme != "scrypt":
            return False
        salt, expected = base64.b64decode(salt_b64), base64.b64decode(digest_b64)
        actual = hashlib.scrypt(
            password[:MAX_LENGTH].encode(),
            salt=salt,
            n=int(n),
            r=int(r),
            p=int(p),
            dklen=len(expected),
            maxmem=_MAXMEM,
        )
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(actual, expected) and stored is not _DUMMY
