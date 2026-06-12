"""
Module 02 — Routing
===================
Topics:
  - Path parameters
  - Query parameters
  - Optional parameters
  - Enum path params
  - APIRouter (splitting routes into files)
  - Route ordering (specificity matters!)
"""

from enum import Enum
from typing import Optional

from fastapi import APIRouter, FastAPI

app = FastAPI(title="Routing Deep Dive")


# --- Path Parameters -------------------------------------------------------
@app.get("/users/{user_id}")
def get_user(user_id: int):   # FastAPI auto-validates + converts to int
    """
    Path params are part of the URL path.
    FastAPI validates the type automatically — passing 'abc' returns 422.
    """
    return {"user_id": user_id, "name": f"User #{user_id}"}


# --- Route Ordering (IMPORTANT!) -------------------------------------------
# Fixed routes MUST be defined BEFORE dynamic ones with same prefix
@app.get("/users/me")          # ✅ defined first → matched first
def get_current_user():
    return {"user": "current logged-in user"}


# --- Enum Path Parameters --------------------------------------------------
class ModelName(str, Enum):
    alexnet = "alexnet"
    resnet = "resnet"
    lenet = "lenet"


@app.get("/models/{model_name}")
def get_model(model_name: ModelName):
    """
    Enum params appear as a dropdown in Swagger docs.
    Passing an invalid value returns a clear 422 error.
    """
    if model_name == ModelName.alexnet:
        return {"model": model_name, "message": "Deep Learning FTW!"}
    return {"model": model_name, "message": "Another great model"}


# --- Query Parameters -------------------------------------------------------
fake_items = [
    {"name": "Laptop", "category": "electronics"},
    {"name": "Python Book", "category": "books"},
    {"name": "Keyboard", "category": "electronics"},
]


@app.get("/items")
def list_items(
    skip: int = 0,                  # default = 0
    limit: int = 10,                # default = 10
    category: Optional[str] = None  # optional filter
):
    """
    Query params come after '?' in the URL:
    GET /items?skip=0&limit=2&category=electronics
    """
    result = fake_items
    if category:
        result = [i for i in result if i["category"] == category]
    return result[skip: skip + limit]


# --- Multiple Path + Query Params ------------------------------------------
@app.get("/users/{user_id}/posts/{post_id}")
def get_user_post(
    user_id: int,
    post_id: int,
    include_comments: bool = False,  # query param
):
    return {
        "user_id": user_id,
        "post_id": post_id,
        "comments_included": include_comments,
    }


# --- APIRouter (organising routes in separate files) -----------------------
# In real apps you'd put these in separate files like:
#   routers/products.py
#   routers/orders.py

products_router = APIRouter(prefix="/products", tags=["Products"])


@products_router.get("/")
def list_products():
    return [{"id": 1, "name": "Widget"}, {"id": 2, "name": "Gadget"}]


@products_router.get("/{product_id}")
def get_product(product_id: int):
    return {"id": product_id, "name": "Widget"}


app.include_router(products_router)  # register the router


# ---------------------------------------------------------------------------
# Run:  uvicorn main:app --reload
# Try:
#   GET /users/42
#   GET /users/me                          ← note ordering
#   GET /models/alexnet
#   GET /items?category=electronics
#   GET /products/
# ---------------------------------------------------------------------------