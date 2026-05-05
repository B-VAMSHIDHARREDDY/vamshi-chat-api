import httpx
import logging
from typing import List, Dict
from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

OPENAI_BASE_URL = "https://api.openai.com/v1/chat/completions"

# Try cheaper/free models first to avoid quota issues
OPENAI_MODELS_FALLBACK = [
    "gpt-4o-mini",       # cheapest, most available
    "gpt-3.5-turbo",     # classic fallback
    "gpt-4o",            # if user has access
]


def _build_openai_messages(system_prompt: str, history: List[Dict], user_message: str) -> List[Dict[str, str]]:
    messages = [{"role": "system", "content": system_prompt}]
    for msg in history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_message})
    return messages


async def call_openai(user_message: str, history: List[Dict], system_prompt: str) -> str:
    headers = {
        "Authorization": f"Bearer {settings.OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    for model in OPENAI_MODELS_FALLBACK:
        payload = {
            "model": model,
            "messages": _build_openai_messages(system_prompt, history, user_message),
            "max_tokens": settings.MAX_TOKENS,
            "temperature": settings.TEMPERATURE,
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(OPENAI_BASE_URL, json=payload, headers=headers)

            if response.status_code in (429, 402):
                error_body = response.json().get("error", {})
                error_code = error_body.get("code", "")
                # If billing issue (not just rate limit), no point trying other models
                if error_code in ("insufficient_quota", "billing_not_active"):
                    raise ValueError(f"OpenAI billing/quota issue: {error_body.get('message', '')}")
                logger.warning(f"OpenAI model {model} rate limited, trying next...")
                continue

            if response.status_code == 404:
                logger.warning(f"OpenAI model {model} not available, trying next...")
                continue

            if response.status_code != 200:
                error_detail = response.json().get("error", {}).get("message", response.text)
                raise RuntimeError(f"OpenAI API error {response.status_code}: {error_detail}")

            data = response.json()
            choices = data.get("choices", [])
            if not choices:
                raise RuntimeError("OpenAI returned no choices")

            logger.info(f"OpenAI responded using model: {model}")
            return choices[0]["message"]["content"].strip()

        except ValueError:
            raise  # Re-raise quota/billing errors
        except RuntimeError as e:
            logger.error(f"OpenAI model {model} error: {e}")
            continue

    raise ValueError("OpenAI quota exceeded or no models available")
