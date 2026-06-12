"""
Module 05 — Database with SQLAlchemy 2 + Alembic
==================================================
Topics:
  - Async SQLAlchemy setup
  - Defining models
  - Sessions as dependencies
  - CRUD operations
  - Relationships (one-to-many)
  - Alembic migrations (see alembic/ folder)
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator, List, Optional

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy import ForeignKey, String, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

# ---------------------------------------------------------------------------
# 1. Database engine & session factory
# ---------------------------------------------------------------------------
DATABASE_URL = "sqlite+aiosqlite:///./learning.db"  # swap to postgres in prod

engine = create_async_engine(DATABASE_URL, echo=True)  # echo=True logs SQL
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)


# ---------------------------------------------------------------------------
# 2. ORM Base & Models
# ---------------------------------------------------------------------------
class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str] = mapped_column(String(50), unique=True, index=True)
    email: Mapped[str] = mapped_column(String(200), unique=True)
    posts: Mapped[List["Post"]] = relationship(back_populates="author")


class Post(Base):
    __tablename__ = "posts"

    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column(String(200))
    body: Mapped[Optional[str]] = mapped_column(default=None)
    author_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    author: Mapped["User"] = relationship(back_populates="posts")


# ---------------------------------------------------------------------------
# 3. Pydantic Schemas (separate from ORM models!)
# ---------------------------------------------------------------------------
class UserCreate(BaseModel):
    username: str
    email: str


class UserRead(BaseModel):
    id: int
    username: str
    email: str
    model_config = {"from_attributes": True}  # allows ORM → Pydantic


class PostCreate(BaseModel):
    title: str
    body: Optional[str] = None


class PostRead(BaseModel):
    id: int
    title: str
    body: Optional[str]
    author_id: int
    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# 4. Database dependency + lifespan
# ---------------------------------------------------------------------------
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        yield session


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create tables on startup (use Alembic in production)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    await engine.dispose()


app = FastAPI(title="Database Module", lifespan=lifespan)


# ---------------------------------------------------------------------------
# 5. CRUD Routes
# ---------------------------------------------------------------------------
@app.post("/users", response_model=UserRead, status_code=201)
async def create_user(user_in: UserCreate, db: AsyncSession = Depends(get_db)):
    user = User(username=user_in.username, email=user_in.email)
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return user


@app.get("/users", response_model=List[UserRead])
async def list_users(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User))
    return result.scalars().all()


@app.get("/users/{user_id}", response_model=UserRead)
async def get_user(user_id: int, db: AsyncSession = Depends(get_db)):
    user = await db.get(User, user_id)
    if not user:
        raise HTTPException(404, "User not found")
    return user


@app.post("/users/{user_id}/posts", response_model=PostRead, status_code=201)
async def create_post(
    user_id: int,
    post_in: PostCreate,
    db: AsyncSession = Depends(get_db),
):
    user = await db.get(User, user_id)
    if not user:
        raise HTTPException(404, "User not found")
    post = Post(title=post_in.title, body=post_in.body, author_id=user_id)
    db.add(post)
    await db.commit()
    await db.refresh(post)
    return post


@app.get("/users/{user_id}/posts", response_model=List[PostRead])
async def list_user_posts(user_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Post).where(Post.author_id == user_id))
    return result.scalars().all()


@app.delete("/users/{user_id}", status_code=204)
async def delete_user(user_id: int, db: AsyncSession = Depends(get_db)):
    user = await db.get(User, user_id)
    if not user:
        raise HTTPException(404, "User not found")
    await db.delete(user)
    await db.commit()


# ---------------------------------------------------------------------------
# Run:  uvicorn main:app --reload
# DB is auto-created as learning.db in this directory
# ---------------------------------------------------------------------------