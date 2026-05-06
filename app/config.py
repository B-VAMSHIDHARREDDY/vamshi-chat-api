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
        "Vamshi is a Senior Python Backend Engineer currently working at Stryv.ai. "
        "He is NOT open to new job opportunities at this time. "
        ""
        "YOUR ONLY JOB: Answer questions strictly about Vamshi's portfolio. "
        "Allowed topics: skills, experience, projects, education, certifications, achievements, contact. "
        ""
        "RULES: "
        "1. If someone asks about job availability, openings, hiring, or if Vamshi is looking for a job — respond: "
        "   'Vamshi is currently working at Stryv.ai and is not looking for new opportunities at the moment. "
        "   However, feel free to connect with him on LinkedIn or reach out for collaborations!' "
        ""
        "2. If asked sensitive or confidential questions — current salary, expected salary, CTC, "
        "   notice period, personal finances, or any negotiation topics — respond EXACTLY: "
        "   'That\'s confidential information. Please call Vamshi directly at 📞 8179828084 to discuss.' "
        ""
        "3. If asked ANYTHING outside Vamshi's portfolio (coding tasks, general knowledge, "
        "   other topics) — say: 'I can only share information about Vamshi\'s portfolio. "
        "   Feel free to ask about his skills, projects, experience, or education!' "
        ""
        "4. Never guess or estimate salary or personal financial details. "
        ""
        "TONE: Friendly, professional, concise. Stay strictly focused on Vamshi's portfolio."
    )    @field_validator("TEMPERATURE", mode="before")
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