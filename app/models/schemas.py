from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Optional, List, Union
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
    conversation_history: Optional[List[Union[Message, str]]] = Field(
        default=[], description="Previous conversation turns. Each item can be a Message object {role, content} or a plain string (treated as user message)."
    )
    system_prompt: Optional[str] = Field(
        default=None, description="Optional system prompt override"
    )

    @field_validator("conversation_history", mode="before")
    @classmethod
    def normalize_history(cls, v):
        """Accept plain strings, dicts, or Message objects in conversation_history."""
        if not v:
            return []
        normalized = []
        for i, item in enumerate(v):
            if isinstance(item, str):
                # Plain string → treat as user message
                normalized.append({"role": "user", "content": item})
            elif isinstance(item, dict):
                # Dict without role → infer role (even=user, odd=assistant)
                if "role" not in item:
                    item["role"] = "user" if i % 2 == 0 else "assistant"
                if "content" not in item:
                    item["content"] = str(item)
                normalized.append(item)
            else:
                normalized.append(item)
        return normalized

    class Config:
        json_schema_extra = {
            "example": {
                "message": "which is the best project he did?",
                "conversation_history": [
                    {"role": "user", "content": "Tell me about your projects"},
                    {"role": "assistant", "content": "Vamshi has worked on several projects including..."}
                ],
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
                "reply": "Vamshi's best project is Blue Book Services APIs...",
                "provider": "groq",
                "model": "llama-3.3-70b-versatile",
                "success": True,
            }
        }


class ErrorResponse(BaseModel):
    detail: str
    success: bool = False
