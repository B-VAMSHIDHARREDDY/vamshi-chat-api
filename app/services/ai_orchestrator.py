import logging
from typing import List, Dict, Tuple
from app.services.gemini_service import call_gemini
from app.services.openai_service import call_openai
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
    Orchestrates AI calls with a fallback chain:
      1. Gemini AI  (primary)
      2. OpenAI ChatGPT  (fallback on quota/error)
      3. HuggingFace  (free fallback)

    Returns: (reply_text, provider_name, model_name)
    """
    providers = [
        {
            "name": "gemini",
            "model": settings.GEMINI_MODEL,
            "fn": call_gemini,
        },
        {
            "name": "openai",
            "model": settings.OPENAI_MODEL,
            "fn": call_openai,
        },
        {
            "name": "huggingface",
            "model": settings.HUGGINGFACE_MODEL,
            "fn": call_huggingface,
        },
    ]

    last_error: Exception = RuntimeError("No AI providers available")

    for provider in providers:
        name = provider["name"]
        model = provider["model"]
        fn = provider["fn"]

        try:
            logger.info(f"Attempting AI call via {name} ({model})")
            reply = await fn(
                user_message=user_message,
                history=history,
                system_prompt=system_prompt,
            )
            logger.info(f"Successfully got response from {name}")
            return reply, name, model

        except ValueError as e:
            # Quota / rate-limit — try next provider
            logger.warning(f"{name} quota exceeded: {e}. Falling back...")
            last_error = e
            continue

        except Exception as e:
            # Any other error — log and try next provider
            logger.error(f"{name} failed with error: {e}. Falling back...")
            last_error = e
            continue

    # All providers failed
    raise RuntimeError(
        f"All AI providers failed. Last error: {last_error}"
    )
