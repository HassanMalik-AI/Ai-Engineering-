"""
Module 03 — Request Handling & Pydantic
========================================
Topics:
  - Pydantic models as request bodies
  - Field validation (min/max, regex, etc.)
  - Nested models
  - Body + path + query together
  - Form data
  - Request headers & cookies
"""

from typing import Optional
from fastapi import Body, Cookie, FastAPI, Form, Header, HTTPException
from pydantic import BaseModel, EmailStr, Field, field_validator

app = FastAPI(title="Request Handling")


# --- Pydantic Request Body Model -------------------------------------------
class Item(BaseModel):
    name: str = Field(..., min_length=1, max_length=100, description="Item name")
    description: Optional[str] = Field(None, max_length=500)
    price: float = Field(..., gt=0, description="Price must be positive")
    in_stock: bool = True

    # Custom validator
    @field_validator("name")
    @classmethod
    def name_must_not_be_empty(cls, v: str) -> str:
        if v.strip() == "":
            raise ValueError("Name cannot be blank")
        return v.strip().title()


@app.post("/items")
def create_item(item: Item):  # FastAPI reads body → validates → injects
    """
    FastAPI automatically:
    1. Reads the JSON body
    2. Validates it against `Item`
    3. Returns a 422 Unprocessable Entity with details if invalid
    """
    return {"received": item.model_dump(), "computed_tax": item.price * 0.1}


# --- Nested Models ----------------------------------------------------------
class Address(BaseModel):
    street: str
    city: str
    country: str = "Pakistan"


class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    age: int = Field(..., ge=13, le=120)
    address: Address          # nested model — FastAPI handles recursively


@app.post("/users")
def create_user(user: UserCreate):
    return {
        "message": f"User {user.username} created",
        "city": user.address.city,
    }


# --- Body + Path + Query Together ------------------------------------------
@app.put("/items/{item_id}")
def update_item(
    item_id: int,                    # from path
    item: Item,                      # from body
    notify: bool = False,            # from query string
    importance: int = Body(default=1, ge=1, le=5),  # extra body field
):
    return {
        "item_id": item_id,
        "updated": item.model_dump(),
        "importance": importance,
        "notification_sent": notify,
    }


# --- Form Data (e.g. HTML forms, login endpoints) --------------------------
@app.post("/login")
def login(
    username: str = Form(...),
    password: str = Form(...),
):
    """
    Use Form() for application/x-www-form-urlencoded or multipart/form-data.
    Note: you CANNOT mix JSON body and Form in the same endpoint.
    """
    if username == "admin" and password == "secret":
        return {"access": "granted", "user": username}
    raise HTTPException(status_code=401, detail="Invalid credentials")


# --- Headers & Cookies ------------------------------------------------------
@app.get("/profile")
def get_profile(
    user_agent: Optional[str] = Header(None),  # auto-converts User-Agent → user_agent
    session_id: Optional[str] = Cookie(None),
):
    """Headers and cookies are injected the same way as query params."""
    return {
        "user_agent": user_agent,
        "session_id": session_id or "no session cookie found",
    }


# ---------------------------------------------------------------------------
# Run:  uvicorn main:app --reload
# Try in /docs:
#   POST /items  with { "name": "  laptop  ", "price": -1 }  → 422 error
#   POST /items  with { "name": "Laptop", "price": 999.99 }  → success
#   POST /users  with nested address object
# ---------------------------------------------------------------------------