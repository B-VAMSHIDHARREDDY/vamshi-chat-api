import httpx
import logging
from typing import List, Dict
from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# Groq free tier models (very fast, generous limits)
GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "gemma2-9b-it",
    "mixtral-8x7b-32768",
]


async def call_groq(user_message: str, history: List[Dict], system_prompt: str) -> str:
    if not settings.GROQ_API_KEY or "YOUR_" in settings.GROQ_API_KEY:
        raise RuntimeError("Groq API key not configured")

    headers = {
        "Authorization": f"Bearer {settings.GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    messages = [{"role": "system", "content": system_prompt}]
    for msg in history[-6:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_message})

    for model in GROQ_MODELS:
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": settings.MAX_TOKENS,
            "temperature": settings.TEMPERATURE,
        }
        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                r = await client.post(GROQ_API_URL, json=payload, headers=headers)

            if r.status_code == 200:
                text = r.json()["choices"][0]["message"]["content"].strip()
                if text:
                    logger.info(f"Groq responded via {model}")
                    return text

            elif r.status_code in (429, 402):
                raise ValueError("Groq rate limit exceeded")
            elif r.status_code == 401:
                raise RuntimeError("Groq API key invalid")
            else:
                logger.warning(f"Groq {model} → {r.status_code}, trying next...")
                continue

        except ValueError:
            raise
        except Exception as e:
            logger.warning(f"Groq {model} error: {e}, trying next...")
            continue

    raise RuntimeError("All Groq models failed")
