from pydantic_settings import BaseSettings
from pydantic import field_validator
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

    # --- System Prompt (fallback if scraper fails) ---
    DEFAULT_SYSTEM_PROMPT: str = (
        "You are a helpful AI assistant on Vamshi's portfolio website (vamshi.site). "
        "You represent Vamshidhar Reddy Beecharla, a skilled Python backend / full-stack developer. "
        "Answer questions about Vamshi's skills, projects, and experience in a friendly, "
        "professional, and concise manner."
    )

    @field_validator("TEMPERATURE", mode="before")
    @classmethod
    def fix_temperature(cls, v):
        """Accept both '0.7' and '0,7' (comma as decimal separator)."""
        if isinstance(v, str):
            v = v.replace(",", ".")
        return float(v)

    @field_validator("MAX_TOKENS", mode="before")
    @classmethod
    def fix_max_tokens(cls, v):
        return int(str(v).strip())

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    return Settings()
