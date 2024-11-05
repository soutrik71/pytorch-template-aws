from fastapi import FastAPI
from app.api import chat
from app.db.database import Base, engine
from app.core.config import settings
from aiocache import caches

app = FastAPI()


# Create tables on startup
@app.on_event("startup")
async def startup():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Configure aiocache with in-memory backend
    caches.set_config(
        {
            "default": {
                "cache": settings.cache_backend,
                "ttl": 300,  # Default Time-To-Live for cache entries (in seconds)
            }
        }
    )


# Include the chat route
app.include_router(chat.router)
