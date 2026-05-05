import httpx
import logging
import asyncio
from fastapi import APIRouter
from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter()

# Try both v1 and v1beta, and multiple model names
GEMINI_MODELS_TO_TRY = [
    ("v1", "gemini-2.0-flash"),
    ("v1", "gemini-1.5-flash"),
    ("v1", "gemini-1.5-pro"),
    ("v1beta", "gemini-2.0-flash"),
    ("v1beta", "gemini-1.5-flash"),
    ("v1beta", "gemini-1.5-flash-latest"),
    ("v1beta", "gemini-pro"),
]

OPENAI_MODELS_TO_TRY = ["gpt-4o-mini", "gpt-3.5-turbo", "gpt-4o"]

HF_MODELS_TO_TRY = [
    "microsoft/DialoGPT-large",
    "HuggingFaceH4/zephyr-7b-beta",
    "google/flan-t5-large",
    "tiiuae/falcon-7b-instruct",
]


def _key_hint(key: str) -> str:
    if not key or "YOUR_" in key or len(key) < 8:
        return "NOT SET"
    return f"{key[:6]}...{key[-4:]}"


# ── Gemini ────────────────────────────────────────────────────────────────────

async def _test_gemini() -> dict:
    payload = {
        "contents": [{"role": "user", "parts": [{"text": "Say OK"}]}],
        "generationConfig": {"maxOutputTokens": 10},
    }

    tried = []
    async with httpx.AsyncClient(timeout=15.0) as client:
        for api_ver, model in GEMINI_MODELS_TO_TRY:
            url = f"https://generativelanguage.googleapis.com/{api_ver}/models/{model}:generateContent?key={settings.GEMINI_API_KEY}"
            try:
                r = await client.post(url, json=payload)
                tried.append(f"{api_ver}/{model}={r.status_code}")
                if r.status_code == 200:
                    text = (r.json().get("candidates", [{}])[0]
                            .get("content", {}).get("parts", [{}])[0].get("text", ""))
                    return {
                        "status": "ok",
                        "http_code": 200,
                        "working_model": model,
                        "working_api_version": api_ver,
                        "response": text.strip(),
                        "tried": tried,
                    }
                if r.status_code == 429:
                    return {"status": "error", "http_code": 429,
                            "error": "Quota exceeded", "tried": tried}
                if r.status_code == 403:
                    body = r.json()
                    return {"status": "error", "http_code": 403,
                            "error": body.get("error", {}).get("message", "Invalid API key"),
                            "tried": tried}
            except Exception as e:
                tried.append(f"{api_ver}/{model}=exception:{e}")

    return {
        "status": "error",
        "http_code": 404,
        "error": "No working Gemini model found for your API key",
        "tried": tried,
    }


async def _list_gemini_models() -> dict:
    """Call ListModels to see what's actually available for this key."""
    results = {}
    async with httpx.AsyncClient(timeout=15.0) as client:
        for api_ver in ["v1", "v1beta"]:
            url = f"https://generativelanguage.googleapis.com/{api_ver}/models?key={settings.GEMINI_API_KEY}"
            try:
                r = await client.get(url)
                if r.status_code == 200:
                    models = r.json().get("models", [])
                    results[api_ver] = [
                        m["name"] for m in models
                        if "generateContent" in m.get("supportedGenerationMethods", [])
                    ]
                else:
                    results[api_ver] = f"HTTP {r.status_code}: {r.text[:200]}"
            except Exception as e:
                results[api_ver] = f"Exception: {e}"
    return results


# ── OpenAI ────────────────────────────────────────────────────────────────────

async def _test_openai() -> dict:
    headers = {
        "Authorization": f"Bearer {settings.OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    tried = []
    async with httpx.AsyncClient(timeout=15.0) as client:
        for model in OPENAI_MODELS_TO_TRY:
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": "Say OK"}],
                "max_tokens": 10,
            }
            try:
                r = await client.post("https://api.openai.com/v1/chat/completions",
                                      json=payload, headers=headers)
                tried.append(f"{model}={r.status_code}")
                body = r.json()
                if r.status_code == 200:
                    text = body.get("choices", [{}])[0].get("message", {}).get("content", "")
                    return {"status": "ok", "http_code": 200,
                            "working_model": model, "response": text.strip(), "tried": tried}
                if r.status_code in (401, 403):
                    return {"status": "error", "http_code": r.status_code,
                            "error": "Invalid API key", "tried": tried}
                if r.status_code in (429, 402):
                    err = body.get("error", {}).get("message", "")
                    code = body.get("error", {}).get("code", "")
                    if code == "insufficient_quota":
                        return {"status": "error", "http_code": r.status_code,
                                "error": f"No billing credits. Add credits at platform.openai.com/billing. Detail: {err}",
                                "tried": tried}
                    tried.append(f"rate_limit_on_{model}")
                    continue
            except Exception as e:
                tried.append(f"{model}=exception:{e}")

    return {"status": "error", "http_code": 429,
            "error": "All OpenAI models rate-limited or quota exceeded", "tried": tried}


# ── HuggingFace ───────────────────────────────────────────────────────────────

async def _test_huggingface() -> dict:
    headers = {"Authorization": f"Bearer {settings.HUGGINGFACE_API_KEY}"}
    tried = []
    async with httpx.AsyncClient(timeout=20.0) as client:
        for model in HF_MODELS_TO_TRY:
            # Use the correct Inference API URL
            url = f"https://api-inference.huggingface.co/models/{model}"
            payload = {"inputs": "Say OK", "parameters": {"max_new_tokens": 10, "return_full_text": False}}
            try:
                r = await client.post(url, json=payload, headers=headers)
                tried.append(f"{model}={r.status_code}")
                if r.status_code == 200:
                    data = r.json()
                    text = data[0].get("generated_text", "") if isinstance(data, list) else str(data)
                    return {"status": "ok", "http_code": 200,
                            "working_model": model, "response": text.strip()[:100], "tried": tried}
                if r.status_code == 503:
                    # Model loading — count as partial success
                    return {"status": "loading", "http_code": 503,
                            "working_model": model,
                            "error": f"Model {model} is loading (cold start). Retry in 20s.",
                            "tried": tried}
                if r.status_code == 401:
                    return {"status": "error", "http_code": 401,
                            "error": "Invalid HuggingFace token", "tried": tried}
                if r.status_code == 429:
                    tried.append(f"rate_limited_{model}")
                    continue
            except Exception as e:
                tried.append(f"{model}=exception:{e}")

    return {"status": "error", "http_code": 404,
            "error": "All HuggingFace models failed", "tried": tried}


# ── Fix hints ─────────────────────────────────────────────────────────────────

def _gemini_fix(r):
    if r["status"] == "ok": return "✅ Gemini is working."
    if r.get("http_code") == 429: return "⚠️ Quota exceeded. Check https://aistudio.google.com"
    if r.get("http_code") == 403: return "❌ Invalid API key. Check GEMINI_API_KEY in Render env vars."
    if "NOT SET" in _key_hint(settings.GEMINI_API_KEY): return "❌ GEMINI_API_KEY not set in Render env vars."
    return "❌ No working model found. Check /api/v1/diagnostics/gemini-models to see available models."

def _openai_fix(r):
    if r["status"] == "ok": return "✅ OpenAI is working."
    if "billing" in r.get("error","").lower() or "quota" in r.get("error","").lower():
        return "❌ No credits. Add billing at https://platform.openai.com/billing (even $5 works)"
    if r.get("http_code") in (401,403): return "❌ Invalid API key. Check OPENAI_API_KEY in Render env vars."
    return "⚠️ Rate limited. Try again in a minute."

def _hf_fix(r):
    if r["status"] in ("ok","loading"): return "✅ HuggingFace key is valid. Model may need warm-up."
    if r.get("http_code") == 401: return "❌ Invalid token. Check HUGGINGFACE_API_KEY in Render env vars."
    if "NOT SET" in _key_hint(settings.HUGGINGFACE_API_KEY): return "❌ HUGGINGFACE_API_KEY not set."
    return "❌ All models failed. HuggingFace free tier may be overloaded — try again later."


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("/diagnostics/all", summary="Test ALL AI providers at once", tags=["Diagnostics"])
async def test_all():
    gemini, openai, hf = await asyncio.gather(_test_gemini(), _test_openai(), _test_huggingface())
    providers = [
        {"provider": "Gemini AI", "api_key_hint": _key_hint(settings.GEMINI_API_KEY), **gemini, "fix": _gemini_fix(gemini)},
        {"provider": "OpenAI ChatGPT", "api_key_hint": _key_hint(settings.OPENAI_API_KEY), **openai, "fix": _openai_fix(openai)},
        {"provider": "HuggingFace", "api_key_hint": _key_hint(settings.HUGGINGFACE_API_KEY), **hf, "fix": _hf_fix(hf)},
    ]
    working = [p for p in providers if p["status"] in ("ok", "loading")]
    return {
        "overall_status": "healthy" if working else "all_providers_failing",
        "working_providers": len(working),
        "total_providers": 3,
        "chat_will_work": len(working) > 0,
        "providers": providers,
    }


@router.get("/diagnostics/gemini", summary="Test Gemini AI", tags=["Diagnostics"])
async def test_gemini():
    result = await _test_gemini()
    return {"provider": "Gemini AI", "api_key_hint": _key_hint(settings.GEMINI_API_KEY),
            **result, "fix": _gemini_fix(result)}


@router.get("/diagnostics/gemini-models", summary="List available Gemini models for your API key", tags=["Diagnostics"])
async def list_gemini_models():
    models = await _list_gemini_models()
    return {"api_key_hint": _key_hint(settings.GEMINI_API_KEY), "available_models": models,
            "note": "Only models supporting generateContent are shown"}


@router.get("/diagnostics/openai", summary="Test OpenAI ChatGPT", tags=["Diagnostics"])
async def test_openai():
    result = await _test_openai()
    return {"provider": "OpenAI ChatGPT", "api_key_hint": _key_hint(settings.OPENAI_API_KEY),
            **result, "fix": _openai_fix(result)}


@router.get("/diagnostics/huggingface", summary="Test HuggingFace", tags=["Diagnostics"])
async def test_huggingface():
    result = await _test_huggingface()
    return {"provider": "HuggingFace", "api_key_hint": _key_hint(settings.HUGGINGFACE_API_KEY),
            **result, "fix": _hf_fix(result)}
