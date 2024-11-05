from pydantic import BaseModel


class ChatMessage(BaseModel):
    content: str
    user_input: int


class ChatResponse(BaseModel):
    message_id: int
    status: str
    processed_value: int
