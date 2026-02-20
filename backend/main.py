"""
DeepShield — FastAPI Application Entry Point

Startup sequence:
  1. Initialise SQLite database
  2. Load image model (EfficientNet-B0)
  3. Load audio model (AudioAntiSpoofCNN)
  4. Register routers
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

from config import RATE_LIMIT
from database import init_db
from models.image_model import load_image_model
from models.audio_model import load_audio_model
from routers import image_router, video_router, audio_router, health_router


# ─── Lifespan (startup / shutdown) ───────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    app.state.models_ready = False
    await init_db()
    load_image_model()
    load_audio_model()
    app.state.models_ready = True
    yield
    # Shutdown (cleanup if needed)
    app.state.models_ready = False


# ─── Rate limiter ─────────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address, default_limits=[RATE_LIMIT])


# ─── App factory ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="DeepShield API",
    description=(
        "Real-time deepfake detection for images, videos, and audio. "
        "Built for the DeepShield Chrome Extension."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── Middleware ────────────────────────────────────────────────────────────────
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # Chrome extensions use null origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-RateLimit-Limit", "X-RateLimit-Remaining"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(health_router.router)
app.include_router(image_router.router)
app.include_router(video_router.router)
app.include_router(audio_router.router)


# ── Root redirect → /docs ────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
async def root():
    """Redirect browser visits to the interactive API docs."""
    return RedirectResponse(url="/docs")


# ── Global exception handler ──────────────────────────────────────────────────
@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {type(exc).__name__}: {exc}"},
    )
