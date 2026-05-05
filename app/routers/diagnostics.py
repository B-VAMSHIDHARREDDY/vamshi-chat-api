import httpx
import logging
from fastapi import APIRouter
from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter()


async def _test_gemini() -> dict:
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-1.5-flash-latest:generateContent?key={settings.GEMINI_API_KEY}"
    )
    payload = {
        "contents": [{"role": "user", "parts": [{"text": "Say OK"}]}],
        "generationConfig": {"maxOutputTokens": 10},
    }
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.post(url, json=payload)
        body = r.json()
        if r.status_code == 200:
            text = body.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
            return {"status": "ok", "http_code": 200, "response": text.strip()}
        else:
            error_msg = body.get("error", {}).get("message", str(body))
            return {"status": "error", "http_code": r.status_code, "error": error_msg}
    except Exception as e:
        return {"status": "exception", "error": str(e)}


async def _test_openai() -> dict:
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {settings.OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": settings.OPENAI_MODEL,
        "messages": [{"role": "user", "content": "Say OK"}],
        "max_tokens": 10,
    }
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.post(url, json=payload, headers=headers)
        body = r.json()
        if r.status_code == 200:
            text = body.get("choices", [{}])[0].get("message", {}).get("content", "")
            return {"status": "ok", "http_code": 200, "response": text.strip()}
        else:
            error_msg = body.get("error", {}).get("message", str(body))
            return {"status": "error", "http_code": r.status_code, "error": error_msg}
    except Exception as e:
        return {"status": "exception", "error": str(e)}


async def _test_huggingface() -> dict:
    url = f"https://api-inference.huggingface.co/models/{settings.HUGGINGFACE_MODEL}"
    headers = {"Authorization": f"Bearer {settings.HUGGINGFACE_API_KEY}"}
    payload = {
        "inputs": "<s>[INST] Say OK [/INST]",
        "parameters": {"max_new_tokens": 10, "return_full_text": False},
    }
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            r = await client.post(url, json=payload, headers=headers)
        if r.status_code == 200:
            data = r.json()
            text = data[0].get("generated_text", "") if isinstance(data, list) else str(data)
            return {"status": "ok", "http_code": 200, "response": text.strip()}
        elif r.status_code == 503:
            return {"status": "loading", "http_code": 503, "error": "Model is loading on HuggingFace, retry in ~20s"}
        else:
            try:
                error_msg = r.json()
            except Exception:
                error_msg = r.text
            return {"status": "error", "http_code": r.status_code, "error": str(error_msg)}
    except Exception as e:
        return {"status": "exception", "error": str(e)}


def _key_hint(key: str) -> str:
    """Return a masked version of an API key for safe display."""
    if not key or "YOUR_" in key or len(key) < 8:
        return "NOT SET (placeholder value)"
    return f"{key[:6]}...{key[-4:]}"


@router.get(
    "/diagnostics/gemini",
    summary="Test Gemini AI connection",
    tags=["Diagnostics"],
)
async def test_gemini():
    result = await _test_gemini()
    return {
        "provider": "Gemini AI",
        "model": settings.GEMINI_MODEL,
        "api_key_hint": _key_hint(settings.GEMINI_API_KEY),
        **result,
        "fix": _gemini_fix(result),
    }


@router.get(
    "/diagnostics/openai",
    summary="Test OpenAI / ChatGPT connection",
    tags=["Diagnostics"],
)
async def test_openai():
    result = await _test_openai()
    return {
        "provider": "OpenAI ChatGPT",
        "model": settings.OPENAI_MODEL,
        "api_key_hint": _key_hint(settings.OPENAI_API_KEY),
        **result,
        "fix": _openai_fix(result),
    }


@router.get(
    "/diagnostics/huggingface",
    summary="Test HuggingFace connection",
    tags=["Diagnostics"],
)
async def test_huggingface():
    result = await _test_huggingface()
    return {
        "provider": "HuggingFace",
        "model": settings.HUGGINGFACE_MODEL,
        "api_key_hint": _key_hint(settings.HUGGINGFACE_API_KEY),
        **result,
        "fix": _hf_fix(result),
    }


@router.get(
    "/diagnostics/all",
    summary="Test ALL AI providers at once",
    tags=["Diagnostics"],
)
async def test_all():
    import asyncio
    gemini, openai, hf = await asyncio.gather(
        _test_gemini(),
        _test_openai(),
        _test_huggingface(),
    )

    providers = [
        {
            "provider": "Gemini AI",
            "model": settings.GEMINI_MODEL,
            "api_key_hint": _key_hint(settings.GEMINI_API_KEY),
            **gemini,
            "fix": _gemini_fix(gemini),
        },
        {
            "provider": "OpenAI ChatGPT",
            "model": settings.OPENAI_MODEL,
            "api_key_hint": _key_hint(settings.OPENAI_API_KEY),
            **openai,
            "fix": _openai_fix(openai),
        },
        {
            "provider": "HuggingFace",
            "model": settings.HUGGINGFACE_MODEL,
            "api_key_hint": _key_hint(settings.HUGGINGFACE_API_KEY),
            **hf,
            "fix": _hf_fix(hf),
        },
    ]

    working = [p for p in providers if p["status"] == "ok"]
    overall = "healthy" if working else "all_providers_failing"

    return {
        "overall_status": overall,
        "working_providers": len(working),
        "total_providers": len(providers),
        "chat_will_work": len(working) > 0,
        "providers": providers,
    }


# ── Human-readable fix hints ──────────────────────────────────────────────────

def _gemini_fix(result: dict) -> str:
    if result["status"] == "ok":
        return "✅ Gemini is working fine."
    code = result.get("http_code")
    err = result.get("error", "")
    if "NOT SET" in str(result.get("api_key_hint", "")):
        return "❌ GEMINI_API_KEY is not set. Get a free key at https://aistudio.google.com/app/apikey and add it to Render → Environment."
    if code == 400:
        return "❌ Bad request — check GEMINI_MODEL value in Render env vars. Default: gemini-1.5-flash"
    if code == 403 or "API_KEY_INVALID" in err:
        return "❌ Invalid API key. Check GEMINI_API_KEY in Render → Environment → make sure no spaces."
    if code == 429 or "quota" in err.lower():
        return "⚠️ Quota exceeded. Free tier limit hit. Wait or upgrade at https://aistudio.google.com"
    return f"❌ Unexpected error (HTTP {code}). Check Render logs for details."


def _openai_fix(result: dict) -> str:
    if result["status"] == "ok":
        return "✅ OpenAI is working fine."
    code = result.get("http_code")
    err = result.get("error", "")
    if "NOT SET" in str(result.get("api_key_hint", "")):
        return "❌ OPENAI_API_KEY is not set. Get a key at https://platform.openai.com/api-keys and add to Render env vars."
    if code == 401 or "invalid_api_key" in err:
        return "❌ Invalid API key. Check OPENAI_API_KEY in Render → Environment."
    if code == 429:
        return "⚠️ Rate limit or billing issue. Check your OpenAI usage at https://platform.openai.com/usage"
    if code == 402:
        return "❌ No credits. Add billing at https://platform.openai.com/account/billing"
    return f"❌ Unexpected error (HTTP {code}). Check error field above."


def _hf_fix(result: dict) -> str:
    if result["status"] == "ok":
        return "✅ HuggingFace is working fine."
    if result["status"] == "loading":
        return "⚠️ Model is cold-starting. Wait ~20 seconds and retry."
    code = result.get("http_code")
    err = str(result.get("error", ""))
    if "NOT SET" in str(result.get("api_key_hint", "")):
        return "❌ HUGGINGFACE_API_KEY not set. Get a free token at https://huggingface.co/settings/tokens"
    if code == 401 or "unauthorized" in err.lower():
        return "❌ Invalid HuggingFace token. Check HUGGINGFACE_API_KEY in Render env vars."
    if code == 429:
        return "⚠️ HuggingFace rate limit hit. Wait and retry."
    return f"❌ Unexpected error (HTTP {code}). Check error field above."
