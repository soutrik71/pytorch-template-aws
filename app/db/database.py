from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from app.core.config import settings

# Create an async engine
engine = create_async_engine(settings.database_url, echo=True)

# Async session factory
SessionLocal = sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)

Base = declarative_base()


# Dependency for asynchronous database session
async def get_db():
    async with SessionLocal() as session:
        yield session
