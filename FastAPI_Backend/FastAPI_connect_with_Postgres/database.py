from app.config import *
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.orm import sessionmaker

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import sessionmaker


DATABASE_URL = (
    f"postgresql+asyncpg://{DB_USER}:{DB_PASSWORD}"
    f"@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

from sqlalchemy.orm import declarative_base

Base = declarative_base()

engine = create_async_engine(DATABASE_URL)

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False
)

SessionLocal = sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False
)