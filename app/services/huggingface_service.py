import httpx
import logging
import asyncio
from typing import List, Dict
from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Models to try in order — all free on HuggingFace Inference API
HF_MODELS_FALLBACK = [
    "microsoft/DialoGPT-large",
    "HuggingFaceH4/zephyr-7b-beta",
    "tiiuae/falcon-7b-instruct",
    "google/flan-t5-large",
]

HF_BASE_URL = "https://api-inference.huggingface.co/models"


def _build_prompt(system_prompt: str, history: List[Dict], user_message: str, model: str) -> str:
    # flan-t5 uses plain text
    if "flan-t5" in model:
        context = f"{system_prompt}\n\nUser: {user_message}\nAssistant:"
        return context

    # Zephyr / Falcon use ChatML or simple format
    if "zephyr" in model:
        prompt = f"<|system|>\n{system_prompt}</s>\n"
        for msg in history[-4:]:  # last 4 messages for context
            role = "user" if msg["role"] == "user" else "assistant"
            prompt += f"<|{role}|>\n{msg['content']}</s>\n"
        prompt += f"<|user|>\n{user_message}</s>\n<|assistant|>\n"
        return prompt

    # Default instruction format
    prompt = f"### System:\n{system_prompt}\n\n"
    for msg in history[-4:]:
        role = "Human" if msg["role"] == "user" else "Assistant"
        prompt += f"### {role}:\n{msg['content']}\n\n"
    prompt += f"### Human:\n{user_message}\n\n### Assistant:\n"
    return prompt


async def call_huggingface(user_message: str, history: List[Dict], system_prompt: str) -> str:
    if not settings.HUGGINGFACE_API_KEY:
        raise RuntimeError("HuggingFace API key not configured")

    headers = {
        "Authorization": f"Bearer {settings.HUGGINGFACE_API_KEY}",
        "Content-Type": "application/json",
    }

    for model in HF_MODELS_FALLBACK:
        url = f"{HF_BASE_URL}/{model}"
        payload = {
            "inputs": _build_prompt(system_prompt, history, user_message, model),
            "parameters": {
                "max_new_tokens": settings.MAX_TOKENS,
                "temperature": settings.TEMPERATURE,
                "return_full_text": False,
                "do_sample": True,
            },
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(url, json=payload, headers=headers)

            if response.status_code == 503:
                # Model loading — wait and retry once
                logger.warning(f"HuggingFace model {model} loading, waiting 8s...")
                await asyncio.sleep(8)
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.post(url, json=payload, headers=headers)

            if response.status_code == 404:
                logger.warning(f"HuggingFace model {model} not found, trying next...")
                continue

            if response.status_code in (429, 402):
                raise ValueError("HuggingFace rate limit exceeded")

            if response.status_code == 401:
                raise RuntimeError("HuggingFace API key is invalid")

            if response.status_code != 200:
                logger.warning(f"HuggingFace model {model} returned {response.status_code}, trying next...")
                continue

            data = response.json()
            if isinstance(data, list) and data:
                text = data[0].get("generated_text", "").strip()
                if text:
                    logger.info(f"HuggingFace responded using model: {model}")
                    return text

            logger.warning(f"HuggingFace model {model} returned empty response, trying next...")
            continue

        except ValueError:
            raise
        except Exception as e:
            logger.warning(f"HuggingFace model {model} error: {e}, trying next...")
            continue

    raise RuntimeError("All HuggingFace models failed or returned empty responses")
