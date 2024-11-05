from sqlalchemy import Column, Integer, String, DateTime, func
from app.db.database import Base


class ChatMessageModel(Base):
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, index=True)
    content = Column(String, nullable=False)
    user_input = Column(Integer, nullable=False)
    processed_value = Column(Integer, nullable=False)
    timestamp = Column(DateTime, server_default=func.now())
