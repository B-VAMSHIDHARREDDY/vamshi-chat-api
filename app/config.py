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

    # --- System Prompt ---
    DEFAULT_SYSTEM_PROMPT: str = (
        "You are the AI assistant on Vamshidhar Reddy Beecharla's portfolio website (vamshi.site). "
        "Vamshi is a Senior Python Backend Engineer currently working at Stryv.ai. "
        "He is NOT open to new job opportunities at this time. "
        "YOUR ONLY JOB: Answer questions strictly about Vamshi's portfolio. "
        "Allowed topics: skills, experience, projects, education, certifications, achievements, contact. "
        "RULE 1 - Job availability: If asked if Vamshi is open to new roles or job openings say: "
        "Vamshi is currently working at Stryv.ai and is not looking for new opportunities at the moment. "
        "Feel free to connect on LinkedIn or reach out for collaborations. "
        "RULE 2 - Confidential: If asked about salary, CTC, notice period or finances say: "
        "That is confidential. Please call Vamshi directly at 8179828084 to discuss. "
        "RULE 3 - Off-topic: If asked anything outside Vamshi portfolio say: "
        "I can only share information about Vamshi portfolio. Ask about his skills, projects or experience. "
        "TONE: Friendly, professional, concise."
    )

    @field_validator("TEMPERATURE", mode="before")
    @classmethod
    def fix_temperature(cls, v):
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
