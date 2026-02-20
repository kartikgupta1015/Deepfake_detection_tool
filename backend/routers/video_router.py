"""
DeepShield — POST /analyze-video
"""
import asyncio
import os
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from slowapi import Limiter
from slowapi.util import get_remote_address

from config import get_risk_level, RATE_LIMIT
from database import log_detection
from models.video_model import analyze_video
from models.multimodal import multimodal_score
from utils.download import download_media
from utils.validators import MediaRequest

router  = APIRouter(tags=["Video Analysis"])
limiter = Limiter(key_func=get_remote_address)


@router.post("/analyze-video", summary="Detect deepfakes in a video (including audio track)")
@limiter.limit(RATE_LIMIT)
async def analyze_video_endpoint(payload: MediaRequest, request: Request) -> dict:
    """
    Download a video, analyse it frame-by-frame, extract the audio track,
    run both modality models, perform face-voice consistency check, and
    return a combined authenticity score.
    """
    tmp_path: Optional[str] = None
    try:
        # ── Download ──────────────────────────────────────────────────────────
        try:
            tmp_path = await download_media(payload.url, "video")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Download failed: {exc}")

        # ── Video analysis runs in thread pool (CPU-bound) ────────────────────
        try:
            loop   = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, analyze_video, tmp_path)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Video analysis error: {exc}")

        auth_score  = result["authenticity_score"]
        video_score = result["video_score"]
        audio_score = result["audio_score"]
        risk_level  = get_risk_level(auth_score)

        # ── Multimodal consistency ────────────────────────────────────────────
        mm = multimodal_score(video_score, audio_score)

        # ── Persist ───────────────────────────────────────────────────────────
        await log_detection("video", payload.url, auth_score, risk_level)

        return {
            "type":                "video",
            "authenticity_score":   auth_score,
            "risk_level":           risk_level,
            "video_score":          video_score,
            "audio_score":          audio_score,
            "face_voice_match":     mm["face_voice_match"],
            "frame_analysis": {
                "total_frames":      result["frame_analysis"]["total_frames"],
                "suspicious_frames": result["frame_analysis"]["suspicious_frames"],
            },
        }

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
