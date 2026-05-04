import httpx
import logging
from typing import List, Dict, Any
from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"


def _build_gemini_contents(
    system_prompt: str, history: List[Dict], user_message: str
) -> List[Dict[str, Any]]:
    """Convert conversation history to Gemini's `contents` format."""
    contents = []

    # Gemini doesn't have a system role in contents; prepend it as a user/model pair
    contents.append({"role": "user", "parts": [{"text": system_prompt}]})
    contents.append(
        {
            "role": "model",
            "parts": [{"text": "Understood. I'll follow those instructions."}],
        }
    )

    for msg in history:
        role = "user" if msg["role"] == "user" else "model"
        contents.append({"role": role, "parts": [{"text": msg["content"]}]})

    contents.append({"role": "user", "parts": [{"text": user_message}]})
    return contents


async def call_gemini(
    user_message: str,
    history: List[Dict],
    system_prompt: str,
) -> str:
    """
    Call Gemini AI API. Raises ValueError on quota/limit exceeded so the
    orchestrator can trigger the fallback chain.
    """
    url = (
        f"{GEMINI_BASE_URL}/{settings.GEMINI_MODEL}"
        f":generateContent?key={settings.GEMINI_API_KEY}"
    )

    payload = {
        "contents": _build_gemini_contents(system_prompt, history, user_message),
        "generationConfig": {
            "maxOutputTokens": settings.MAX_TOKENS,
            "temperature": settings.TEMPERATURE,
        },
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(url, json=payload)

    if response.status_code == 429 or (
        response.status_code == 200
        and "quota" in response.text.lower()
    ):
        raise ValueError("Gemini rate limit / quota exceeded")

    if response.status_code != 200:
        error_detail = response.json().get("error", {}).get("message", response.text)
        raise RuntimeError(f"Gemini API error {response.status_code}: {error_detail}")

    data = response.json()

    # Handle safety blocks / empty candidates
    candidates = data.get("candidates", [])
    if not candidates:
        raise RuntimeError("Gemini returned no candidates")

    parts = candidates[0].get("content", {}).get("parts", [])
    if not parts:
        raise RuntimeError("Gemini returned empty content parts")

    return parts[0].get("text", "").strip()
