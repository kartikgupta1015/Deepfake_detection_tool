"""
DeepShield — POST /analyze-audio
"""
import os
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from slowapi import Limiter
from slowapi.util import get_remote_address

from config import get_risk_level, RATE_LIMIT
from database import log_detection
from models.audio_model import analyze_audio
from utils.download import download_media
from utils.validators import MediaRequest

router  = APIRouter(tags=["Audio Analysis"])
limiter = Limiter(key_func=get_remote_address)


@router.post("/analyze-audio", summary="Detect deepfakes / synthetic voices in audio")
@limiter.limit(RATE_LIMIT)
async def analyze_audio_endpoint(payload: MediaRequest, request: Request) -> dict:
    """
    Download an audio file from the given URL and return a synthetic-voice
    probability along with pitch stability analysis.
    """
    tmp_path: Optional[str] = None
    try:
        # ── Download ─────────────────────────────────────────────────────────
        try:
            tmp_path = await download_media(payload.url, "audio")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Download failed: {exc}")

        # ── Model inference ───────────────────────────────────────────────────
        try:
            result = analyze_audio(tmp_path)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Analysis error: {exc}")

        auth_score = result["authenticity_score"]
        risk_level = get_risk_level(auth_score)

        # ── Persist ───────────────────────────────────────────────────────────
        await log_detection("audio", payload.url, auth_score, risk_level)

        return {
            "type":               "audio",
            "authenticity_score":  auth_score,
            "risk_level":          risk_level,
            "analysis": {
                "synthetic_probability": result["synthetic_probability"],
                "pitch_irregularity":    result["pitch_irregularity"],
            },
        }

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
