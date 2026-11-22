"""Authentication (JWT) and workspace authorization.

* Passwords: scrypt with a per-user random salt (stdlib hashlib, no extra dep).
* Tokens: HS256 JWTs. "access" lives ACCESS_TOKEN_MINUTES, "refresh" lives
  REFRESH_TOKEN_DAYS and can only be used at /api/auth/refresh. The token
  `type` claim is checked so a refresh token can't be used as an access token.
* Authorization: every workspace route calls require_role(). Users with no
  access to a private workspace get 404, not 403, so ids can't be probed.
"""
import base64
import hashlib
import hmac
import logging
import os
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional

import jwt
from fastapi import Depends, HTTPException, Request
from sqlalchemy.orm import Session

from .config import settings
from .db import get_db
from .models import Membership, User, Workspace

log = logging.getLogger("nexus.auth")

ROLE_RANK = {"viewer": 1, "editor": 2, "owner": 3}
ALGORITHM = "HS256"

_secret = settings.jwt_secret
if settings.app_env in ("production", "prod"):
    if not _secret or len(_secret) < 32:
        raise RuntimeError(
            "APP_ENV=production requires JWT_SECRET with at least 32 characters "
            "(e.g. `python -c \"import secrets; print(secrets.token_urlsafe(48))\"`)")
if not _secret:
    _secret = secrets.token_urlsafe(48)
    log.warning("JWT_SECRET is not set: using a random per-process secret. "
                "Tokens will be invalid after a restart. Set JWT_SECRET in production.")
elif len(_secret) < 32:
    log.warning("JWT_SECRET is shorter than 32 characters; use a longer one.")


# --- passwords --------------------------------------------------------------

_SCRYPT = dict(n=2 ** 14, r=8, p=1, dklen=32)


def hash_password(password: str) -> str:
    salt = os.urandom(16)
    dk = hashlib.scrypt(password.encode(), salt=salt, **_SCRYPT)
    return "scrypt$" + base64.b64encode(salt).decode() + "$" + base64.b64encode(dk).decode()


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, salt_b64, dk_b64 = stored.split("$")
        salt, expected = base64.b64decode(salt_b64), base64.b64decode(dk_b64)
        if scheme != "scrypt":
            return False
    except ValueError:
        return False
    actual = hashlib.scrypt(password.encode(), salt=salt, **_SCRYPT)
    return hmac.compare_digest(actual, expected)


# --- tokens -----------------------------------------------------------------

def make_token(user_id: int, kind: str) -> str:
    ttl = (timedelta(minutes=settings.access_token_minutes) if kind == "access"
           else timedelta(days=settings.refresh_token_days))
    now = datetime.now(timezone.utc)
    return jwt.encode({"sub": str(user_id), "type": kind, "iat": now,
                       "exp": now + ttl, "jti": secrets.token_hex(8)},
                      _secret, algorithm=ALGORITHM)


def decode_token(token: str, kind: str) -> int:
    try:
        payload = jwt.decode(token, _secret, algorithms=[ALGORITHM])  # alg pinned
    except jwt.ExpiredSignatureError:
        raise HTTPException(401, "Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(401, "Invalid token")
    if payload.get("type") != kind:
        raise HTTPException(401, "Wrong token type")
    return int(payload["sub"])


# --- dependencies -------------------------------------------------------------

def _bearer(request: Request) -> Optional[str]:
    h = request.headers.get("authorization", "")
    if h.lower().startswith("bearer "):
        return h[7:].strip()
    return None


def optional_user(request: Request, db: Session = Depends(get_db)) -> Optional[User]:
    token = _bearer(request)
    if token is None:
        return None
    user = db.query(User).filter_by(id=decode_token(token, "access")).first()
    if user is None:
        raise HTTPException(401, "Unknown user")
    return user


def require_user(user: Optional[User] = Depends(optional_user)) -> User:
    if user is None:
        raise HTTPException(401, "Login required",
                            headers={"WWW-Authenticate": "Bearer"})
    return user


def role_in(db: Session, ws: Workspace, user: Optional[User]) -> Optional[str]:
    if user is not None:
        if ws.owner_id == user.id:
            return "owner"
        m = db.query(Membership).filter_by(workspace_id=ws.id, user_id=user.id).first()
        if m:
            return m.role
    return "viewer" if ws.is_public else None


def require_role(db: Session, workspace_id: int, user: Optional[User],
                 minimum: str = "viewer") -> Workspace:
    ws = db.query(Workspace).filter_by(id=workspace_id).first()
    role = role_in(db, ws, user) if ws else None
    if ws is None or role is None:
        raise HTTPException(404, "Workspace not found")
    if ROLE_RANK[role] < ROLE_RANK[minimum]:
        if user is None:
            raise HTTPException(401, "Login required",
                                headers={"WWW-Authenticate": "Bearer"})
        raise HTTPException(403, f"Requires {minimum} access")
    return ws
