from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from app.routers import chat
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup — warm portfolio cache (non-fatal if it fails)
    try:
        from app.services.scraper_service import warm_cache
        logger.info("Warming portfolio cache from vamshi.site...")
        await warm_cache()
        logger.info("Portfolio cache ready.")
    except Exception as e:
        logger.warning(f"Portfolio cache warm-up failed (non-fatal): {e}")
    yield
    # Shutdown — nothing to clean up


app = FastAPI(
    title="Vamshi Portfolio Chat API",
    description="AI-powered chat API for vamshi.site — Gemini AI with ChatGPT fallback.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://vamshi.site",
        "https://www.vamshi.site",
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5500",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router, prefix="/api/v1", tags=["Chat"])


@app.get("/", tags=["Health"])
async def root():
    return {
        "status": "online",
        "service": "Vamshi Portfolio Chat API",
        "website": "https://vamshi.site",
        "docs": "/docs",
        "version": "1.0.0",
    }


@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "healthy"}


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal server error occurred. Please try again later."},
    )
