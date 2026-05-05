import logging
from typing import List, Dict, Tuple
from app.services.gemini_service import call_gemini
from app.services.openai_service import call_openai
from app.services.groq_service import call_groq
from app.services.huggingface_service import call_huggingface
from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


async def get_ai_response(
    user_message: str,
    history: List[Dict],
    system_prompt: str,
) -> Tuple[str, str, str]:
    """
    Fallback chain:
      1. Gemini AI   (primary)
      2. OpenAI      (fallback 1)
      3. Groq        (fallback 2 — FREE, fast)
      4. HuggingFace (fallback 3 — FREE)
    """
    providers = [
        {"name": "gemini",       "model": settings.GEMINI_MODEL,       "fn": call_gemini},
        {"name": "openai",       "model": settings.OPENAI_MODEL,        "fn": call_openai},
        {"name": "groq",         "model": "llama-3.3-70b-versatile",    "fn": call_groq},
        {"name": "huggingface",  "model": settings.HUGGINGFACE_MODEL,   "fn": call_huggingface},
    ]

    last_error: Exception = RuntimeError("No AI providers available")

    for provider in providers:
        name = provider["name"]
        fn   = provider["fn"]
        model = provider["model"]
        try:
            logger.info(f"Trying {name}...")
            reply = await fn(user_message=user_message, history=history, system_prompt=system_prompt)
            logger.info(f"✅ Response from {name}")
            return reply, name, model
        except ValueError as e:
            logger.warning(f"{name} quota/rate-limit: {e} — trying next...")
            last_error = e
        except Exception as e:
            logger.error(f"{name} error: {e} — trying next...")
            last_error = e

    raise RuntimeError(f"All AI providers failed. Last error: {last_error}")
