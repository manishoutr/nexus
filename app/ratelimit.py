"""Rate limiting with slowapi. Authenticated calls are keyed by user id
(token subject), anonymous calls by client IP, so one user can't dodge limits
by changing address and one address can't starve other users."""
from fastapi import Request
from slowapi import Limiter
from slowapi.util import get_remote_address

from .config import settings


def key_func(request: Request) -> str:
    h = request.headers.get("authorization", "")
    if h.lower().startswith("bearer "):
        try:
            import jwt

            from .auth import ALGORITHM, _secret
            sub = jwt.decode(h[7:].strip(), _secret, algorithms=[ALGORITHM])["sub"]
            return f"user:{sub}"
        except Exception:  # noqa: BLE001 - bad token: fall back to IP
            pass
    return get_remote_address(request)


limiter = Limiter(key_func=key_func, storage_uri=settings.rate_limit_storage,
                  enabled=settings.rate_limit_enabled)
