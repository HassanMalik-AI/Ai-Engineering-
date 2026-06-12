"""
Module 04 — Response Models & Error Handling
=============================================
Topics:
  - response_model= to shape output
  - Hiding sensitive fields (passwords, internal IDs)
  - HTTP status codes
  - HTTPException
  - Custom exception handlers
  - response_model_exclude_unset
"""

from typing import Optional
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, EmailStr

app = FastAPI(title="Response Models & Errors")


# --- Input vs Output Models (never return passwords!) ----------------------
class UserIn(BaseModel):
    username: str
    email: EmailStr
    password: str          # accepted on input


class UserOut(BaseModel):
    id: int
    username: str
    email: EmailStr
    # password intentionally absent → never returned


class UserDB(UserOut):
    hashed_password: str   # internal — also never returned


fake_users: dict[int, UserDB] = {}


@app.post(
    "/users",
    response_model=UserOut,          # ← FastAPI filters the response
    status_code=status.HTTP_201_CREATED,
)
def create_user(user_in: UserIn):
    uid = len(fake_users) + 1
    db_user = UserDB(
        id=uid,
        username=user_in.username,
        email=user_in.email,
        hashed_password=f"bcrypt_{user_in.password}_hashed",  # fake hash
    )
    fake_users[uid] = db_user
    return db_user    # FastAPI strips hashed_password via response_model


@app.get("/users/{user_id}", response_model=UserOut)
def get_user(user_id: int):
    if user_id not in fake_users:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User {user_id} not found",
        )
    return fake_users[user_id]


# --- Status Codes -----------------------------------------------------------
@app.delete("/users/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_user(user_id: int):
    if user_id not in fake_users:
        raise HTTPException(status_code=404, detail="User not found")
    del fake_users[user_id]
    # 204 No Content → don't return a body


# --- response_model_exclude_unset -------------------------------------------
class Item(BaseModel):
    name: str
    description: Optional[str] = None
    price: Optional[float] = None
    tax: Optional[float] = None


@app.get("/items/{item_id}", response_model=Item, response_model_exclude_unset=True)
def get_item(item_id: int):
    """
    response_model_exclude_unset=True:
    Only fields explicitly set are returned.
    Avoids polluting responses with null fields.
    """
    if item_id == 1:
        return Item(name="Laptop", price=999.99)   # description & tax omitted
    raise HTTPException(status_code=404, detail="Item not found")


# --- Custom Exception Class + Handler ---------------------------------------
class AppError(Exception):
    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message


@app.exception_handler(AppError)
async def app_error_handler(request: Request, exc: AppError):
    return JSONResponse(
        status_code=400,
        content={"error_code": exc.code, "message": exc.message},
    )


@app.get("/risky/{value}")
def risky_route(value: int):
    if value < 0:
        raise AppError(code="NEGATIVE_VALUE", message="Value must be non-negative")
    return {"result": value * 2}


# --- Global 404 override (optional) ----------------------------------------
@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    return JSONResponse(
        status_code=404,
        content={"message": "This route does not exist", "path": str(request.url)},
    )


# ---------------------------------------------------------------------------
# Run:  uvicorn main:app --reload
# Try:
#   POST /users → note password is absent in response
#   GET  /users/999 → 404 with clean message
#   GET  /risky/-5  → custom error format
# ---------------------------------------------------------------------------