import httpx
import logging
from typing import List, Dict
from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

HF_BASE_URL = "https://api-inference.huggingface.co/models"


def _build_prompt(system_prompt: str, history: List[Dict], user_message: str) -> str:
    """Build an instruction-formatted prompt for HuggingFace Inference API."""
    prompt = f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n"

    for i, msg in enumerate(history):
        if msg["role"] == "user":
            if i == 0:
                prompt += f"{msg['content']} [/INST] "
            else:
                prompt += f"<s>[INST] {msg['content']} [/INST] "
        else:
            prompt += f"{msg['content']} </s>"

    prompt += f"{user_message} [/INST]"
    return prompt


async def call_huggingface(
    user_message: str,
    history: List[Dict],
    system_prompt: str,
) -> str:
    """
    Call HuggingFace Inference API (free tier available).
    Uses Mixtral-8x7B-Instruct by default.
    """
    if not settings.HUGGINGFACE_API_KEY:
        raise RuntimeError("HuggingFace API key not configured")

    url = f"{HF_BASE_URL}/{settings.HUGGINGFACE_MODEL}"
    headers = {
        "Authorization": f"Bearer {settings.HUGGINGFACE_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "inputs": _build_prompt(system_prompt, history, user_message),
        "parameters": {
            "max_new_tokens": settings.MAX_TOKENS,
            "temperature": settings.TEMPERATURE,
            "return_full_text": False,
        },
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(url, json=payload, headers=headers)

    if response.status_code == 503:
        # Model loading — retry once after a short wait
        import asyncio
        await asyncio.sleep(5)
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(url, json=payload, headers=headers)

    if response.status_code in (429, 402):
        raise ValueError("HuggingFace rate limit exceeded")

    if response.status_code != 200:
        raise RuntimeError(f"HuggingFace API error {response.status_code}: {response.text}")

    data = response.json()

    if isinstance(data, list) and data:
        return data[0].get("generated_text", "").strip()

    raise RuntimeError("HuggingFace returned unexpected response format")
