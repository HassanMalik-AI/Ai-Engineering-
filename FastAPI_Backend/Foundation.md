"""
Module 01 — Foundations
========================
Topics:
  - Creating a FastAPI app instance
  - Defining path operations (routes)
  - HTTP methods: GET, POST, PUT, DELETE
  - Automatic interactive docs (/docs, /redoc)
  - Async vs sync route handlers
"""

from fastapi import FastAPI

# --- App instance -----------------------------------------------------------
# metadata shown in /docs
app = FastAPI(
    title="FastAPI Foundations",
    description="Your very first FastAPI application",
    version="1.0.0",
)


# --- Basic GET route --------------------------------------------------------
@app.get("/")
def root():
    """The simplest possible route — returns a dict (auto-converted to JSON)."""
    return {"message": "Hello, FastAPI! 🚀"}


# --- Async route ------------------------------------------------------------
@app.get("/async-hello")
async def async_hello():
    """
    Use `async def` when you await I/O (DB queries, HTTP calls).
    Use plain `def` for CPU-bound or simple sync logic.
    FastAPI handles both correctly.
    """
    return {"message": "Hello from an async route!"}


# --- Multiple HTTP methods --------------------------------------------------
items_db = {}   # in-memory store for demo


@app.get("/items")
def list_items():
    """GET /items — return all items."""
    return {"items": list(items_db.values())}


@app.post("/items")
def create_item(name: str, price: float):
    """POST /items — create a new item (simple query-param version)."""
    item_id = len(items_db) + 1
    items_db[item_id] = {"id": item_id, "name": name, "price": price}
    return {"created": items_db[item_id]}


@app.put("/items/{item_id}")
def update_item(item_id: int, name: str):
    """PUT /items/{id} — update item name."""
    if item_id not in items_db:
        return {"error": "Item not found"}
    items_db[item_id]["name"] = name
    return {"updated": items_db[item_id]}


@app.delete("/items/{item_id}")
def delete_item(item_id: int):
    """DELETE /items/{id} — remove an item."""
    if item_id not in items_db:
        return {"error": "Item not found"}
    removed = items_db.pop(item_id)
    return {"deleted": removed}


# ---------------------------------------------------------------------------
# Run:  uvicorn main:app --reload
# Docs: http://127.0.0.1:8000/docs
# ---------------------------------------------------------------------------