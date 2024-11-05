from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from app.schemas.chat import ChatMessage, ChatResponse
from app.crud.chat_crud import create_chat_message
from app.tasks.chat_task import process_chat_message
from src.number_manipulation import add_random_number
from app.db.database import get_db

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat(message: ChatMessage, db: AsyncSession = Depends(get_db)):
    # Process user input using add_random_number
    processed_value = add_random_number(message.user_input)

    # Save message to database along with the user input and processed value
    message_id = await create_chat_message(
        db=db,
        content=message.content,
        user_input=message.user_input,
        processed_value=processed_value,
    )

    # Trigger async processing of the chat message content in the background
    process_chat_message.delay(message.content)

    return ChatResponse(
        message_id=message_id,
        status="Message received",
        processed_value=processed_value,
    )
