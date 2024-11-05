from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from app.schemas.chat import ChatMessage, ChatResponse
from app.crud.chat_crud import create_chat_message
from app.tasks.chat_task import process_chat_message
from src.number_manipulation import add_random_number
from app.db.database import get_db
from aiocache import caches

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat(message: ChatMessage, db: AsyncSession = Depends(get_db)):
    cache = caches.get("default")
    cache_key = f"chat:{message.user_input}:{message.content}"

    # Attempt to retrieve the cached response
    cached_response = await cache.get(cache_key)
    if cached_response:
        return ChatResponse(**cached_response)

    # Process user input if not cached
    processed_value = add_random_number(message.user_input)
    message_id = await create_chat_message(
        db=db,
        content=message.content,
        user_input=message.user_input,
        processed_value=processed_value,
    )

    # Trigger background task
    process_chat_message.delay(message.content)

    # Prepare the response data
    response_data = {
        "message_id": message_id,
        "status": "Message received",
        "processed_value": processed_value,
    }

    # Cache the response data
    await cache.set(cache_key, response_data, ttl=300)  # Cache for 5 minutes

    return ChatResponse(**response_data)
