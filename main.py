from fastapi import FastAPI
from app.api import chat
from app.db.database import Base, engine  # Import Base and engine for table creation

app = FastAPI()


# Create tables on startup
@app.on_event("startup")
async def startup():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


# Include the chat route
app.include_router(chat.router)
