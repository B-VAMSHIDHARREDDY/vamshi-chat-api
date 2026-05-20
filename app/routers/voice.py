from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from app.models.schemas import ChatRequest, ChatResponse
from app.services.ai_orchestrator import get_ai_response
from app.services.scraper_service import fetch_portfolio_prompt
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/voice/chat",
    response_model=ChatResponse,
    summary="Voice assistant chat endpoint",
    description="Optimized endpoint for voice assistant with personalized Vamshi intro",
)
async def voice_chat(request: ChatRequest):
    """
    Voice-optimized chat endpoint with special intro message for wake word activation
    """
    
    # Build history
    history = [
        {"role": msg.role.value, "content": msg.content}
        for msg in (request.conversation_history or [])
    ]
    
    # Get portfolio data
    portfolio_prompt = await fetch_portfolio_prompt()
    
    # Enhanced system prompt for voice assistant
    voice_system_prompt = f"""You are Vamshi's AI Voice Assistant. You represent Vamshidhar Reddy Beecharla.

PERSONALITY:
- Friendly, professional, and conversational
- Keep responses concise for voice (2-3 sentences max)
- Sound natural when spoken aloud
- Use "I" when referring to Vamshi's achievements/skills (you represent him)

INTRO MESSAGE (when user says "Hey Vamshi" or "Hi Vamshi"):
"Hey! I'm Vamshi's AI Voice Assistant. I can tell you about my experience, skills, projects, or how to reach me. What would you like to know?"

KEY INFO:
{portfolio_prompt}

VOICE GUIDELINES:
- Short, clear answers
- Avoid long lists (say "I specialize in Python, FastAPI, and cloud architecture" not listing 20 technologies)
- If asked about projects, briefly mention 1-2 highlights
- For contact: "You can reach me at vamshiyvu@gmail.com or check my LinkedIn in the portfolio"
- Always stay in character as Vamshi's voice
"""
    
    try:
        # Check if this is a wake word activation
        user_msg = request.message.lower().strip()
        is_wake_word = any(wake in user_msg for wake in ["hey vamshi", "hi vamshi", "hello vamshi"])
        
        if is_wake_word and len(history) == 0:
            # First interaction - give intro
            reply = "Hey! I'm Vamshi's AI Voice Assistant. I can tell you about my experience, skills, projects, or how to reach me. What would you like to know?"
            return ChatResponse(
                reply=reply,
                provider="voice_assistant",
                model="intro",
                success=True,
            )
        
        # Regular voice query
        reply, provider, model = await get_ai_response(
            user_message=request.message,
            history=history,
            system_prompt=voice_system_prompt,
        )
        
        logger.info(f"Voice chat response via {provider}/{model}")
        
        return ChatResponse(
            reply=reply,
            provider=provider,
            model=model,
            success=True,
        )
        
    except RuntimeError as e:
        logger.error(f"Voice assistant AI providers failed: {e}")
        raise HTTPException(
            status_code=503,
            detail="Voice assistant temporarily unavailable. Please try again.",
        )
    except Exception as e:
        logger.error(f"Voice assistant error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Voice assistant error")


@router.get(
    "/voice/status",
    summary="Voice assistant status",
    description="Check if voice assistant is ready",
)
async def voice_status():
    return {
        "status": "online",
        "features": ["wake_word_detection", "voice_optimized_responses"],
        "wake_words": ["Hey Vamshi", "Hi Vamshi", "Hello Vamshi"],
    }
