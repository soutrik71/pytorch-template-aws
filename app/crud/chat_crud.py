from sqlalchemy.ext.asyncio import AsyncSession
from app.db.models import ChatMessageModel


async def create_chat_message(
    db: AsyncSession, content: str, user_input: int, processed_value: int
):
    new_message = ChatMessageModel(
        content=content, user_input=user_input, processed_value=processed_value
    )
    db.add(new_message)
    await db.commit()
    await db.refresh(new_message)  # Fetches the latest state after commit
    return new_message.id
