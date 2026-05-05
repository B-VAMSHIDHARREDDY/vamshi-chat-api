import httpx
import logging
from typing import List, Dict
from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Use the newer Inference Providers API endpoint
HF_API_URL = "https://api-inference.huggingface.co/v1/chat/completions"

# Models available on HuggingFace Inference API (free tier)
HF_MODELS = [
    "meta-llama/Llama-3.2-3B-Instruct",
    "microsoft/Phi-3.5-mini-instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
]


async def call_huggingface(user_message: str, history: List[Dict], system_prompt: str) -> str:
    if not settings.HUGGINGFACE_API_KEY or "YOUR_" in settings.HUGGINGFACE_API_KEY:
        raise RuntimeError("HuggingFace API key not configured")

    headers = {
        "Authorization": f"Bearer {settings.HUGGINGFACE_API_KEY}",
        "Content-Type": "application/json",
    }

    messages = [{"role": "system", "content": system_prompt}]
    for msg in history[-6:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_message})

    for model in HF_MODELS:
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": settings.MAX_TOKENS,
            "temperature": settings.TEMPERATURE,
            "stream": False,
        }
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                r = await client.post(HF_API_URL, json=payload, headers=headers)

            if r.status_code == 200:
                data = r.json()
                text = data["choices"][0]["message"]["content"].strip()
                if text:
                    logger.info(f"HuggingFace responded via {model}")
                    return text

            elif r.status_code == 429:
                raise ValueError("HuggingFace rate limit exceeded")
            elif r.status_code == 401:
                raise RuntimeError("HuggingFace token invalid or no Inference API access")
            else:
                logger.warning(f"HuggingFace {model} → {r.status_code}, trying next...")
                continue

        except ValueError:
            raise
        except Exception as e:
            logger.warning(f"HuggingFace {model} error: {e}, trying next...")
            continue

    raise RuntimeError("All HuggingFace models failed")
