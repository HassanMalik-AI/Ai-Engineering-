# 🚀 FastAPI Complete Learning Repository

> A structured, hands-on guide to mastering backend development with FastAPI — from zero to production-ready.

---

## 📚 Curriculum Overview

| # | Module | Topics | Difficulty |
|---|--------|---------|------------|
| 00 | [Setup & Environment](./00_setup/) | Python, venv, dependencies | 🟢 Beginner |
| 01 | [Foundations](./01_foundations/) | First app, Uvicorn, auto-docs | 🟢 Beginner |
| 02 | [Routing](./02_routing/) | Path params, query params, routers | 🟢 Beginner |
| 03 | [Request Handling](./03_request_handling/) | Body, forms, validation, Pydantic | 🟡 Intermediate |
| 04 | [Response Models](./04_response_models/) | Schemas, status codes, errors | 🟡 Intermediate |
| 05 | [Database](./05_database/) | SQLAlchemy, Alembic, async DB | 🟡 Intermediate |
| 06 | [Authentication](./06_authentication/) | JWT, OAuth2, hashing, guards | 🔴 Advanced |
| 07 | [Middleware](./07_middleware/) | CORS, logging, custom middleware | 🟡 Intermediate |
| 08 | [Background Tasks](./08_background_tasks/) | Celery, async tasks, scheduling | 🔴 Advanced |
| 09 | [File Handling](./09_file_handling/) | Upload, download, storage | 🟡 Intermediate |
| 10 | [Testing](./10_testing/) | Pytest, TestClient, mocking | 🔴 Advanced |
| 11 | [Deployment](./11_deployment/) | Docker, CI/CD, env config | 🔴 Advanced |
| 12 | [Advanced Patterns](./12_advanced_patterns/) | WebSockets, DI, caching, events | 🔴 Advanced |

---

## 🗺️ Learning Path

```
START
  │
  ▼
[00] Setup ──► [01] Foundations ──► [02] Routing
                                         │
                                         ▼
                              [03] Request Handling
                                         │
                                         ▼
                              [04] Response Models
                                         │
                              ┌──────────┤
                              ▼          ▼
                         [05] DB    [07] Middleware
                              │          │
                              └────┬─────┘
                                   ▼
                          [06] Authentication
                                   │
                         ┌─────────┼─────────┐
                         ▼         ▼         ▼
                    [08] Tasks [09] Files [10] Tests
                         │
                         ▼
               [11] Deployment ──► [12] Advanced
                                         │
                                         ▼
                                      END (PRO)
```


# Core
fastapi==0.111.0
uvicorn[standard]==0.30.1
python-multipart==0.0.9

# Validation
pydantic[email]==2.7.1
pydantic-settings==2.3.1

# Database
sqlalchemy==2.0.30
alembic==1.13.1
asyncpg==0.29.0          # async postgres driver
aiosqlite==0.20.0        # async sqlite (for dev/testing)

# Auth
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4

# HTTP client (for testing/external requests)
httpx==0.27.0

# Background tasks
celery==5.4.0
redis==5.0.4

# Testing
pytest==8.2.0
pytest-asyncio==0.23.7
pytest-cov==5.0.0

# Dev tools
python-dotenv==1.0.1
---

## ⚡ Quick Start

```bash
# Clone / enter the repo
cd fastapi-learning-repo

# Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install all dependencies
pip install -r requirements.txt

# Run any module's main app
cd 01_foundations
uvicorn main:app --reload
```

Then open: **http://127.0.0.1:8000/docs** (Swagger UI auto-generated!)

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| **FastAPI** | Web framework |
| **Uvicorn** | ASGI server |
| **Pydantic v2** | Data validation |
| **SQLAlchemy 2** | ORM |
| **Alembic** | DB migrations |
| **JWT / OAuth2** | Auth |
| **Pytest** | Testing |
| **Docker** | Containerization |
| **Redis** | Caching / task queue |
| **Celery** | Background tasks |

---

## 📌 Prerequisites

- Python 3.11+
- Basic Python knowledge (functions, classes, decorators)
- Basic understanding of HTTP (GET, POST, status codes)

---

## 💡 How to Use This Repo

Each module has:
- `README.md` — concept explanation + diagrams
- `main.py` — runnable working example
- `exercises/` — problems for you to solve
- `solutions/` — reference solutions

Work through each module sequentially, run the examples, then attempt the exercises before peeking at solutions.

---

## 🤝 Contributing

PRs welcome! See [CONTRIBUTING.md](./CONTRIBUTING.md).

---

*Built for developers who learn by doing.*