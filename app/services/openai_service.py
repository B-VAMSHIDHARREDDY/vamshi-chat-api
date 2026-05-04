import httpx
import logging
from typing import List, Dict
from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

OPENAI_BASE_URL = "https://api.openai.com/v1/chat/completions"


def _build_openai_messages(
    system_prompt: str, history: List[Dict], user_message: str
) -> List[Dict[str, str]]:
    """Convert conversation to OpenAI messages format."""
    messages = [{"role": "system", "content": system_prompt}]
    for msg in history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_message})
    return messages


async def call_openai(
    user_message: str,
    history: List[Dict],
    system_prompt: str,
) -> str:
    """
    Call OpenAI ChatGPT API. Raises ValueError on quota/billing exceeded.
    """
    headers = {
        "Authorization": f"Bearer {settings.OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": settings.OPENAI_MODEL,
        "messages": _build_openai_messages(system_prompt, history, user_message),
        "max_tokens": settings.MAX_TOKENS,
        "temperature": settings.TEMPERATURE,
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(OPENAI_BASE_URL, json=payload, headers=headers)

    if response.status_code in (429, 402):
        raise ValueError("OpenAI rate limit / quota exceeded")

    if response.status_code != 200:
        error_detail = response.json().get("error", {}).get("message", response.text)
        raise RuntimeError(f"OpenAI API error {response.status_code}: {error_detail}")

    data = response.json()
    choices = data.get("choices", [])
    if not choices:
        raise RuntimeError("OpenAI returned no choices")

    return choices[0]["message"]["content"].strip()
