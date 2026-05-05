from pydantic_settings import BaseSettings
from pydantic import field_validator
from functools import lru_cache
from typing import Optional


class Settings(BaseSettings):
    # --- Gemini AI ---
    GEMINI_API_KEY: str = "YOUR_GEMINI_API_KEY_HERE"
    GEMINI_MODEL: str = "gemini-1.5-flash-latest"

    # --- OpenAI / ChatGPT ---
    OPENAI_API_KEY: str = "YOUR_OPENAI_API_KEY_HERE"
    OPENAI_MODEL: str = "gpt-4o-mini"

    # --- HuggingFace (free fallback) ---
    HUGGINGFACE_API_KEY: Optional[str] = "YOUR_HUGGINGFACE_API_KEY_HERE"
    HUGGINGFACE_MODEL: str = "mistralai/Mixtral-8x7B-Instruct-v0.1"

    # --- Groq (free, fast fallback) ---
    GROQ_API_KEY: Optional[str] = "YOUR_GROQ_API_KEY_HERE"

    # --- General ---
    MAX_TOKENS: int = 512
    TEMPERATURE: float = 0.7
    APP_ENV: str = "production"

    # --- System Prompt (fallback if scraper fails) ---
    DEFAULT_SYSTEM_PROMPT: str = (
        "You are the AI assistant on Vamshidhar Reddy Beecharla's portfolio website (vamshi.site). "
        "Vamshi is a Senior Python Backend Engineer specializing in FastAPI, Django, microservices, AWS, and Azure. "
        "YOUR ROLE: "
        "1. Answer questions about Vamshi's portfolio — skills, experience, projects, education, contact. "
        "2. Help with general tech questions: Python, FastAPI, Django, Docker, SQL, AWS, Git, etc. "
        "   Write clean working code examples when asked — this showcases Vamshi's technical domain. "
        "3. For completely off-topic questions (movies, politics, cooking, etc.), politely say you focus "
        "   on Vamshi's portfolio and software development topics. "
        "TONE: Friendly, professional, concise. For code — give working examples with brief explanations."
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