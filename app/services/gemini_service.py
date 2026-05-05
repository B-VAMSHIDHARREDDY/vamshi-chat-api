import httpx
import logging
from typing import List, Dict, Any
from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Try these models in order until one works
GEMINI_MODELS_FALLBACK = [
    "gemini-1.5-flash-latest",
    "gemini-1.5-flash",
    "gemini-1.5-pro-latest",
    "gemini-pro",
]

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"


def _build_gemini_contents(system_prompt: str, history: List[Dict], user_message: str) -> List[Dict[str, Any]]:
    contents = []
    contents.append({"role": "user", "parts": [{"text": system_prompt}]})
    contents.append({"role": "model", "parts": [{"text": "Understood. I'll follow those instructions."}]})
    for msg in history:
        role = "user" if msg["role"] == "user" else "model"
        contents.append({"role": role, "parts": [{"text": msg["content"]}]})
    contents.append({"role": "user", "parts": [{"text": user_message}]})
    return contents


async def call_gemini(user_message: str, history: List[Dict], system_prompt: str) -> str:
    last_error = None

    for model in GEMINI_MODELS_FALLBACK:
        url = f"{GEMINI_BASE_URL}/{model}:generateContent?key={settings.GEMINI_API_KEY}"
        payload = {
            "contents": _build_gemini_contents(system_prompt, history, user_message),
            "generationConfig": {
                "maxOutputTokens": settings.MAX_TOKENS,
                "temperature": settings.TEMPERATURE,
            },
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(url, json=payload)

            if response.status_code == 404:
                logger.warning(f"Gemini model {model} not found, trying next...")
                last_error = f"Model {model} not found"
                continue

            if response.status_code == 429 or (response.status_code == 200 and "quota" in response.text.lower()):
                raise ValueError("Gemini rate limit / quota exceeded")

            if response.status_code != 200:
                error_detail = response.json().get("error", {}).get("message", response.text)
                raise RuntimeError(f"Gemini API error {response.status_code}: {error_detail}")

            data = response.json()
            candidates = data.get("candidates", [])
            if not candidates:
                raise RuntimeError("Gemini returned no candidates")

            parts = candidates[0].get("content", {}).get("parts", [])
            if not parts:
                raise RuntimeError("Gemini returned empty content parts")

            logger.info(f"Gemini responded using model: {model}")
            return parts[0].get("text", "").strip()

        except ValueError:
            raise  # Re-raise quota errors immediately
        except RuntimeError as e:
            last_error = str(e)
            continue

    raise RuntimeError(f"All Gemini models failed. Last error: {last_error}")
