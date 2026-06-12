"""
Module 09 — File Handling
==========================
Topics:
  - Upload single & multiple files
  - Validate file type & size
  - Serve static files
  - Stream file downloads
  - Save to disk / cloud storage pattern
"""

import os
import shutil
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="File Handling")

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif"}
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB


# ---------------------------------------------------------------------------
# 1. Upload a single file
# ---------------------------------------------------------------------------
@app.post("/upload/image")
async def upload_image(file: UploadFile = File(...)):
    """
    UploadFile gives you:
    - file.filename — original name
    - file.content_type — MIME type
    - file.size — byte size (may be None; read to check)
    - await file.read() — full content
    - file.file — underlying SpooledTemporaryFile
    """
    # Validate type
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"File type {file.content_type} not allowed. Use: {ALLOWED_IMAGE_TYPES}",
        )

    # Read content & validate size
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File too large (max 5 MB)")

    # Save to disk
    dest = UPLOAD_DIR / file.filename
    with open(dest, "wb") as f:
        f.write(content)

    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "size_bytes": len(content),
        "saved_to": str(dest),
    }


# ---------------------------------------------------------------------------
# 2. Upload multiple files
# ---------------------------------------------------------------------------
@app.post("/upload/multiple")
async def upload_multiple(files: List[UploadFile] = File(...)):
    results = []
    for file in files:
        content = await file.read()
        dest = UPLOAD_DIR / file.filename
        with open(dest, "wb") as f:
            f.write(content)
        results.append({"filename": file.filename, "size": len(content)})
    return {"uploaded": results}


# ---------------------------------------------------------------------------
# 3. Chunked upload for large files (memory-efficient)
# ---------------------------------------------------------------------------
@app.post("/upload/large")
async def upload_large_file(file: UploadFile = File(...)):
    """
    Stream chunks directly to disk — doesn't load entire file into RAM.
    Essential for large files (videos, datasets, etc.)
    """
    dest = UPLOAD_DIR / file.filename
    with open(dest, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)   # streams chunk by chunk

    size = os.path.getsize(dest)
    return {"filename": file.filename, "size_bytes": size}


# ---------------------------------------------------------------------------
# 4. Download a file
# ---------------------------------------------------------------------------
@app.get("/download/{filename}")
def download_file(filename: str):
    file_path = UPLOAD_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(
        path=file_path,
        filename=filename,                       # sets Content-Disposition header
        media_type="application/octet-stream",
    )


# ---------------------------------------------------------------------------
# 5. Stream a generated file (e.g. CSV export)
# ---------------------------------------------------------------------------
@app.get("/export/csv")
def export_csv():
    """Streams a CSV without writing it to disk first."""
    def generate_rows():
        yield "id,name,email\n"
        for i in range(1, 1001):
            yield f"{i},User {i},user{i}@example.com\n"

    return StreamingResponse(
        generate_rows(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=users.csv"},
    )


# ---------------------------------------------------------------------------
# 6. Serve static files (CSS, JS, images for a frontend)
# ---------------------------------------------------------------------------
# app.mount("/static", StaticFiles(directory="static"), name="static")
# Access via: http://localhost:8000/static/logo.png

# ---------------------------------------------------------------------------
# Run:  uvicorn main:app --reload
# Test upload via /docs → POST /upload/image
# Test download:  curl http://localhost:8000/download/yourfile.jpg --output out.jpg
# Test CSV:       curl http://localhost:8000/export/csv --output users.csv
# ---------------------------------------------------------------------------