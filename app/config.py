from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import Optional


class Settings(BaseSettings):
    # --- Gemini AI ---
    GEMINI_API_KEY: str = "YOUR_GEMINI_API_KEY_HERE"
    GEMINI_MODEL: str = "gemini-1.5-flash"

    # --- OpenAI / ChatGPT ---
    OPENAI_API_KEY: str = "YOUR_OPENAI_API_KEY_HERE"
    OPENAI_MODEL: str = "gpt-3.5-turbo"

    # --- HuggingFace (free fallback) ---
    HUGGINGFACE_API_KEY: Optional[str] = "YOUR_HUGGINGFACE_API_KEY_HERE"
    HUGGINGFACE_MODEL: str = "mistralai/Mixtral-8x7B-Instruct-v0.1"

    # --- General ---
    MAX_TOKENS: int = 512
    TEMPERATURE: float = 0.7
    APP_ENV: str = "production"

    # --- System Prompt ---
    DEFAULT_SYSTEM_PROMPT: str = (
        "You are a helpful AI assistant on Vamshi's portfolio website (vamshi.site). "
        "You represent Vamshi, a skilled software developer. "
        "Answer questions about Vamshi's skills, projects, experience, and expertise in a "
        "friendly, professional, and concise manner. "
        "If asked something unrelated to Vamshi or software development, politely redirect "
        "the conversation back to relevant topics. "
        "Keep responses concise (2-4 sentences) unless more detail is specifically requested."
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    return Settings()
