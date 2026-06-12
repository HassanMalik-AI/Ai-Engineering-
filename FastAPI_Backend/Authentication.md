"""
Module 06 — Authentication (JWT + OAuth2)
==========================================
Topics:
  - Password hashing with bcrypt
  - JWT token creation & verification
  - OAuth2PasswordBearer flow
  - Protected routes with Depends
  - Refresh tokens concept
  - Role-based access control (RBAC)
"""

from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Config (use .env in production — never hardcode secrets!)
# ---------------------------------------------------------------------------
SECRET_KEY = "CHANGE_THIS_TO_A_REAL_SECRET_IN_PRODUCTION_64_CHARS_MINIMUM"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# ---------------------------------------------------------------------------
# Password hashing
# ---------------------------------------------------------------------------
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


# ---------------------------------------------------------------------------
# Fake user DB (replace with real DB in Module 05 style)
# ---------------------------------------------------------------------------
fake_users_db: dict = {
    "alice": {
        "username": "alice",
        "hashed_password": hash_password("secret123"),
        "role": "admin",
        "disabled": False,
    },
    "bob": {
        "username": "bob",
        "hashed_password": hash_password("bobpass"),
        "role": "user",
        "disabled": False,
    },
}

# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------
class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: Optional[str] = None


class User(BaseModel):
    username: str
    role: str
    disabled: bool


# ---------------------------------------------------------------------------
# JWT utilities
# ---------------------------------------------------------------------------
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    to_encode["exp"] = expire
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


# ---------------------------------------------------------------------------
# OAuth2 scheme + dependency chain
# ---------------------------------------------------------------------------
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")


async def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception

    user_dict = fake_users_db.get(token_data.username)
    if user_dict is None:
        raise credentials_exception
    return User(**user_dict)


async def get_active_user(current_user: User = Depends(get_current_user)) -> User:
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


# Role guard factory
def require_role(role: str):
    async def role_checker(user: User = Depends(get_active_user)) -> User:
        if user.role != role:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Requires role: {role}",
            )
        return user
    return role_checker


# ---------------------------------------------------------------------------
# App & routes
# ---------------------------------------------------------------------------
app = FastAPI(title="Authentication Module")


@app.post("/auth/token", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    Standard OAuth2 password flow.
    Clients send username + password as form data.
    Returns a JWT bearer token.
    """
    user_dict = fake_users_db.get(form_data.username)
    if not user_dict or not verify_password(form_data.password, user_dict["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = create_access_token(data={"sub": form_data.username})
    return Token(access_token=access_token, token_type="bearer")


@app.get("/users/me")
async def read_own_profile(current_user: User = Depends(get_active_user)):
    """Any authenticated user can access this."""
    return current_user


@app.get("/admin/dashboard")
async def admin_dashboard(admin: User = Depends(require_role("admin"))):
    """Only users with role=admin can access this."""
    return {"message": f"Welcome to admin panel, {admin.username}!"}


@app.get("/public")
def public_route():
    """No auth required."""
    return {"message": "Anyone can see this"}


# ---------------------------------------------------------------------------
# Run:  uvicorn main:app --reload
# Flow in /docs:
#   1. POST /auth/token with username=alice password=secret123
#   2. Copy access_token
#   3. Click "Authorize" → paste token
#   4. GET /users/me  ← works
#   5. GET /admin/dashboard ← works for alice (admin), fails for bob (user)
# ---------------------------------------------------------------------------