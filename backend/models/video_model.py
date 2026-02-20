"""
DeepShield — Video Deepfake Detection Model

Pipeline:
  1. Extract frames at 1 fps (max MAX_VIDEO_FRAMES) via OpenCV
  2. Run image_model on each frame → per-frame score
  3. Extract audio track via ffmpeg subprocess
  4. Run audio_model on extracted audio
  5. Combine scores: video_weight * video_avg + audio_weight * audio_score
"""
from __future__ import annotations

import asyncio
import os
import subprocess
import tempfile

import cv2
import numpy as np

from config import MAX_VIDEO_FRAMES, VIDEO_AUDIO_WEIGHT, AUDIO_WEIGHT
from models.image_model import analyze_image
from models.audio_model import analyze_audio


def _extract_audio_from_video(video_path: str) -> str | None:
    """
    Use ffmpeg to extract the audio track from a video file.
    Returns path to a temp WAV file, or None if no audio stream.
    """
    tmp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp_audio.close()
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn",                    # no video
        "-acodec", "pcm_s16le",   # PCM WAV
        "-ar", "16000",           # 16 kHz
        "-ac", "1",               # mono
        tmp_audio.name,
    ]
    result = subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        timeout=60,
    )
    if result.returncode != 0:
        # No audio stream or extraction failed
        os.unlink(tmp_audio.name)
        return None
    # Check file is non-empty
    if os.path.getsize(tmp_audio.name) < 1024:
        os.unlink(tmp_audio.name)
        return None
    return tmp_audio.name


def _extract_frames(video_path: str, fps: int = 1, max_frames: int = MAX_VIDEO_FRAMES) -> list[str]:
    """
    Extract frames from a video at *fps* frames per second.
    Saves each frame as a temp JPEG. Returns list of file paths.
    """
    cap    = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    interval  = max(1, int(round(video_fps / fps)))   # frame interval

    frame_paths: list[str] = []
    frame_idx   = 0
    saved       = 0

    while saved < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % interval == 0:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
            cv2.imwrite(tmp.name, frame)
            tmp.close()
            frame_paths.append(tmp.name)
            saved += 1
        frame_idx += 1

    cap.release()
    return frame_paths


def analyze_video(video_path: str) -> dict:
    """
    Full video deepfake analysis.

    Returns
    -------
    dict
        authenticity_score, video_score, audio_score, face_voice_match,
        frame_analysis (total_frames, suspicious_frames counts).
    """
    # ── 1. Extract frames ────────────────────────────────────────────────────
    frame_paths = _extract_frames(video_path)
    total_frames = len(frame_paths)

    if total_frames == 0:
        raise ValueError("Could not extract any frames from the video.")

    # ── 2. Analyse each frame ────────────────────────────────────────────────
    frame_scores:     list[float] = []
    suspicious_frames = 0
    SUSPICIOUS_THRESHOLD = 50.0

    for path in frame_paths:
        try:
            result     = analyze_image(path)
            score      = result["authenticity_score"]
            frame_scores.append(score)
            if score > SUSPICIOUS_THRESHOLD:
                suspicious_frames += 1
        except Exception:
            pass   # skip corrupted frames
        finally:
            try:
                os.unlink(path)
            except Exception:
                pass

    video_avg = float(np.mean(frame_scores)) if frame_scores else 50.0

    # ── 3. Extract & analyse audio ───────────────────────────────────────────
    audio_path  = _extract_audio_from_video(video_path)
    audio_score = 50.0   # neutral default if no audio
    audio_detail: dict = {}

    if audio_path:
        try:
            audio_result = analyze_audio(audio_path)
            audio_score  = audio_result["authenticity_score"]
            audio_detail = audio_result
        except Exception:
            audio_score = 50.0
        finally:
            try:
                os.unlink(audio_path)
            except Exception:
                pass

    # ── 4. Combined score ────────────────────────────────────────────────────
    combined = VIDEO_AUDIO_WEIGHT * video_avg + AUDIO_WEIGHT * audio_score
    auth_score = round(float(np.clip(combined, 0, 100)), 2)

    return {
        "authenticity_score": auth_score,
        "video_score":   round(video_avg, 2),
        "audio_score":   round(audio_score, 2),
        "frame_analysis": {
            "total_frames":      total_frames,
            "suspicious_frames": suspicious_frames,
        },
        # multimodal decision injected by multimodal.py
        "_audio_detail": audio_detail,
        "_video_avg":    video_avg,
    }
