"""
Module 07 — Middleware
=======================
Topics:
  - What middleware is and when to use it
  - CORS configuration
  - Request timing / logging middleware
  - GZip compression
  - Trusted hosts
  - Custom middleware class
"""

import time
import uuid
import logging

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Middleware Module")

# ---------------------------------------------------------------------------
# 1. CORS — Cross-Origin Resource Sharing
#    Required when your frontend (e.g. React on :3000) calls this API on :8000
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://myapp.com"],  # list of allowed origins
    allow_credentials=True,
    allow_methods=["*"],   # GET, POST, PUT, DELETE, etc.
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# 2. GZip — compresses responses > 1000 bytes automatically
# ---------------------------------------------------------------------------
app.add_middleware(GZipMiddleware, minimum_size=1000)

# ---------------------------------------------------------------------------
# 3. Trusted Hosts — reject requests with unexpected Host headers
# ---------------------------------------------------------------------------
# app.add_middleware(TrustedHostMiddleware, allowed_hosts=["myapp.com", "*.myapp.com"])
# (commented out so localhost dev works)

# ---------------------------------------------------------------------------
# 4. Custom Middleware — request timing + request ID injection
# ---------------------------------------------------------------------------
class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        # Generate a unique ID for this request (useful for log tracing)
        request_id = str(uuid.uuid4())[:8]
        start_time = time.perf_counter()

        # Attach to request state (accessible in route handlers)
        request.state.request_id = request_id

        logger.info(f"[{request_id}] ▶ {request.method} {request.url.path}")

        response: Response = await call_next(request)

        duration_ms = (time.perf_counter() - start_time) * 1000
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = f"{duration_ms:.2f}ms"

        logger.info(
            f"[{request_id}] ◀ {response.status_code} ({duration_ms:.2f}ms)"
        )
        return response


app.add_middleware(RequestLoggingMiddleware)


# ---------------------------------------------------------------------------
# 5. @app.middleware — simpler decorator style for small middleware
# ---------------------------------------------------------------------------
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    """Adds basic security headers to every response."""
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    return response


# ---------------------------------------------------------------------------
# Sample routes to test middleware effects
# ---------------------------------------------------------------------------
@app.get("/")
def root(request: Request):
    return {
        "message": "Check the response headers in /docs or curl!",
        "request_id": request.state.request_id,
    }


@app.get("/slow")
async def slow_route():
    import asyncio
    await asyncio.sleep(0.5)   # simulate slow DB call
    return {"message": "That took a while..."}


@app.get("/big-response")
def big_response():
    """Returns a large payload — GZip middleware will compress it."""
    return {"data": "x" * 5000}


# ---------------------------------------------------------------------------
# Run:  uvicorn main:app --reload
# Check headers:  curl -I http://127.0.0.1:8000/
# ---------------------------------------------------------------------------