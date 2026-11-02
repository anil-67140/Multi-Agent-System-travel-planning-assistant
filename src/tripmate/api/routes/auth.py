from __future__ import annotations

from datetime import timedelta

from fastapi import APIRouter, Depends, status

from tripmate.api.deps import LOCKOUT, LOCKOUT_THRESHOLD, ContainerDep, CurrentUser, rate_limit
from tripmate.api.schemas import Credentials, TokenResponse
from tripmate.errors import ConflictError, ForbiddenError, UnauthorizedError, ValidationFailure
from tripmate.security.passwords import hash_password, validate_password_strength, verify_password
from tripmate.security.tokens import issue_token

router = APIRouter(prefix="/v1/auth", tags=["auth"])

_login_limit = rate_limit("login", per=10, window=300)
_register_limit = rate_limit("register", per=5, window=3600)


def _token(c: ContainerDep, user_id: str) -> TokenResponse:
    ttl = timedelta(minutes=c.settings.jwt_ttl_minutes)
    token = issue_token(user_id, c.settings.jwt_secret.get_secret_value(), c.clock.now(), ttl)
    return TokenResponse(access_token=token, expires_in=int(ttl.total_seconds()))


@router.post(
    "/register",
    response_model=TokenResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(_register_limit)],
)
async def register(body: Credentials, c: ContainerDep) -> TokenResponse:
    if not c.settings.allow_registration:
        raise ForbiddenError("Registration is disabled.")
    problem = validate_password_strength(body.password)
    if problem:
        raise ValidationFailure(problem)
    try:
        user = await c.repo.create_user(body.email, hash_password(body.password))
    except ConflictError:
        # Same outward behaviour as a weak password would be a lie; a conflict is honest and
        # registration is rate limited, which bounds email enumeration.
        raise
    return _token(c, user.id)


@router.post("/login", response_model=TokenResponse, dependencies=[Depends(_login_limit)])
async def login(body: Credentials, c: ContainerDep) -> TokenResponse:
    generic = UnauthorizedError("Incorrect email or password.")
    user = await c.repo.get_user_by_email(body.email)
    now = c.clock.now()
    if user and user.locked_until and user.locked_until > now:
        verify_password(body.password, user.password_hash)  # keep timing uniform
        raise generic
    ok = verify_password(body.password, user.password_hash if user else None)
    if user is None or not user.is_active or not ok:
        if user is not None:
            user.failed_logins += 1
            if user.failed_logins >= LOCKOUT_THRESHOLD:
                user.locked_until, user.failed_logins = now + LOCKOUT, 0
            await c.repo.save_user(user)
        raise generic
    if user.failed_logins or user.locked_until:
        user.failed_logins, user.locked_until = 0, None
        await c.repo.save_user(user)
    return _token(c, user.id)


@router.get("/me")
async def me(user: CurrentUser) -> dict[str, str]:
    return {"id": user.id, "email": user.email}
