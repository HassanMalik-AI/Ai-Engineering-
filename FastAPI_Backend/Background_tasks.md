"""
Module 08 — Background Tasks
=============================
Topics:
  - FastAPI BackgroundTasks (lightweight, in-process)
  - Use cases: sending emails, logging, webhooks
  - Celery + Redis (heavy tasks, retries, scheduling)
  - asyncio.create_task for fire-and-forget
"""

import asyncio
import time
from fastapi import BackgroundTasks, FastAPI
from pydantic import BaseModel

app = FastAPI(title="Background Tasks")

# ---------------------------------------------------------------------------
# 1. FastAPI built-in BackgroundTasks
#    ✅ Good for: send email after registration, log to file, ping webhook
#    ❌ Not for: heavy CPU work, retries, distributed tasks
# ---------------------------------------------------------------------------

def send_welcome_email(email: str, username: str):
    """
    This runs AFTER the response is sent to the client.
    The user doesn't wait for the email to send!
    """
    time.sleep(2)  # simulate email API call
    print(f"📧 Welcome email sent to {email} for user {username}")


def log_audit_event(action: str, user_id: int):
    """Audit logging without blocking the response."""
    print(f"📝 Audit: user={user_id} action={action} at {time.time()}")


class UserRegister(BaseModel):
    username: str
    email: str


@app.post("/register")
def register_user(user: UserRegister, background_tasks: BackgroundTasks):
    """
    Registers user instantly, sends email in the background.
    Client gets a response in milliseconds regardless of email speed.
    """
    # Do the "real" work
    user_id = 42  # pretend we saved to DB

    # Queue background tasks — run after response is returned
    background_tasks.add_task(send_welcome_email, user.email, user.username)
    background_tasks.add_task(log_audit_event, "user_registered", user_id)

    return {
        "message": f"User {user.username} registered successfully!",
        "note": "Welcome email is being sent in the background",
    }


# ---------------------------------------------------------------------------
# 2. Multiple background tasks in one request
# ---------------------------------------------------------------------------
@app.post("/orders/{order_id}/confirm")
async def confirm_order(order_id: int, background_tasks: BackgroundTasks):
    """Trigger multiple side effects without blocking."""
    background_tasks.add_task(send_order_confirmation, order_id)
    background_tasks.add_task(update_inventory, order_id)
    background_tasks.add_task(notify_warehouse, order_id)

    return {"order_id": order_id, "status": "confirmed"}


def send_order_confirmation(order_id: int):
    print(f"📦 Confirmation email for order {order_id}")


def update_inventory(order_id: int):
    print(f"📊 Inventory updated for order {order_id}")


def notify_warehouse(order_id: int):
    print(f"🏭 Warehouse notified for order {order_id}")


# ---------------------------------------------------------------------------
# 3. asyncio.create_task — async fire-and-forget
# ---------------------------------------------------------------------------
async def async_log(message: str):
    await asyncio.sleep(1)  # async I/O (aiohttp call, async DB write, etc.)
    print(f"[async log] {message}")


@app.get("/async-task-demo")
async def async_task_demo():
    """Fire and forget an async task."""
    asyncio.create_task(async_log("route visited"))
    return {"message": "Task fired, response returned immediately"}


# ---------------------------------------------------------------------------
# 4. Celery setup note (requires Redis)
# ---------------------------------------------------------------------------
"""
For production heavy tasks, use Celery:

# tasks.py
from celery import Celery

celery_app = Celery("tasks", broker="redis://localhost:6379/0")

@celery_app.task
def process_large_file(file_path: str):
    # heavy processing...
    pass

# In your FastAPI route:
from tasks import process_large_file

@app.post("/process")
def trigger(file: str):
    process_large_file.delay(file)   # queued in Redis, runs in worker
    return {"status": "queued"}

# Run worker in terminal:
#   celery -A tasks worker --loglevel=info
"""

# ---------------------------------------------------------------------------
# Run:  uvicorn main:app --reload
# Watch the terminal — background tasks print AFTER the response
# ---------------------------------------------------------------------------