from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum


class Role(str, Enum):
    user = "user"
    assistant = "assistant"
    system = "system"


class Message(BaseModel):
    role: Role
    content: str


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000, description="User's message")
    conversation_history: Optional[List[Message]] = Field(
        default=[], description="Previous conversation turns for context"
    )
    system_prompt: Optional[str] = Field(
        default=None, description="Optional system prompt override"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "message": "Tell me about your projects",
                "conversation_history": [],
                "system_prompt": None,
            }
        }


class ChatResponse(BaseModel):
    reply: str
    provider: str = Field(description="AI provider that handled the request")
    model: str = Field(description="Model used for the response")
    success: bool = True

    class Config:
        json_schema_extra = {
            "example": {
                "reply": "I have worked on several exciting projects...",
                "provider": "gemini",
                "model": "gemini-1.5-flash",
                "success": True,
            }
        }


class ErrorResponse(BaseModel):
    detail: str
    success: bool = False
