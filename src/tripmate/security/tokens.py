"""Short-lived signed access tokens (HS256 JWT)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import jwt

from tripmate.errors import UnauthorizedError

ALGORITHM = "HS256"
ISSUER = "tripmate"


@dataclass(frozen=True)
class TokenClaims:
    user_id: str
    expires_at: datetime


def issue_token(user_id: str, secret: str, now: datetime, ttl: timedelta) -> str:
    payload = {
        "sub": user_id,
        "iss": ISSUER,
        "iat": int(now.timestamp()),
        "exp": int((now + ttl).timestamp()),
    }
    return jwt.encode(payload, secret, algorithm=ALGORITHM)


def decode_token(token: str, secret: str, now: datetime) -> TokenClaims:
    try:
        # Algorithms are pinned (blocks alg=none / key-confusion); time is injectable for tests.
        data = jwt.decode(
            token,
            secret,
            algorithms=[ALGORITHM],
            issuer=ISSUER,
            options={"require": ["sub", "exp", "iss"], "verify_exp": False},
        )
        exp = datetime.fromtimestamp(int(data["exp"]), tz=now.tzinfo)
        if exp <= now:
            raise UnauthorizedError("Your session has expired. Please sign in again.")
        return TokenClaims(user_id=str(data["sub"]), expires_at=exp)
    except UnauthorizedError:
        raise
    except (jwt.PyJWTError, KeyError, ValueError) as exc:
        raise UnauthorizedError("Invalid authentication token.") from exc
