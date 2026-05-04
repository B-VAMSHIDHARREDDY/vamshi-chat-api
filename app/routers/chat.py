from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from app.models.schemas import ChatRequest, ChatResponse, ErrorResponse
from app.services.ai_orchestrator import get_ai_response
from app.config import get_settings
import logging
import time

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter()


@router.post(
    "/chat",
    response_model=ChatResponse,
    responses={
        200: {"model": ChatResponse},
        422: {"description": "Validation error"},
        503: {"model": ErrorResponse, "description": "All AI providers unavailable"},
    },
    summary="Send a chat message",
    description=(
        "Send a message to the portfolio chat assistant. "
        "Uses Gemini AI with automatic fallback to ChatGPT and HuggingFace."
    ),
)
async def chat(request: ChatRequest):
    start = time.time()

    # Build history list from pydantic models → plain dicts
    history = [
        {"role": msg.role.value, "content": msg.content}
        for msg in (request.conversation_history or [])
    ]

    system_prompt = request.system_prompt or settings.DEFAULT_SYSTEM_PROMPT

    try:
        reply, provider, model = await get_ai_response(
            user_message=request.message,
            history=history,
            system_prompt=system_prompt,
        )
        elapsed = round(time.time() - start, 3)
        logger.info(f"Chat response in {elapsed}s via {provider}/{model}")

        return ChatResponse(
            reply=reply,
            provider=provider,
            model=model,
            success=True,
        )

    except RuntimeError as e:
        logger.error(f"All AI providers failed: {e}")
        raise HTTPException(
            status_code=503,
            detail="All AI providers are currently unavailable. Please try again shortly.",
        )

    except Exception as e:
        logger.error(f"Unexpected error in /chat: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get(
    "/chat/providers",
    summary="List configured AI providers",
    description="Returns the configured AI providers and their status.",
)
async def list_providers():
    return {
        "providers": [
            {
                "name": "gemini",
                "model": settings.GEMINI_MODEL,
                "priority": 1,
                "configured": settings.GEMINI_API_KEY != "YOUR_GEMINI_API_KEY_HERE",
            },
            {
                "name": "openai",
                "model": settings.OPENAI_MODEL,
                "priority": 2,
                "configured": settings.OPENAI_API_KEY != "YOUR_OPENAI_API_KEY_HERE",
            },
            {
                "name": "huggingface",
                "model": settings.HUGGINGFACE_MODEL,
                "priority": 3,
                "configured": bool(settings.HUGGINGFACE_API_KEY)
                and settings.HUGGINGFACE_API_KEY != "YOUR_HUGGINGFACE_API_KEY_HERE",
            },
        ]
    }
