"""
DeepShield — Multimodal Face-Voice Consistency Checker

Heuristic approach:
  A significant gap between the video (face) score and the audio (voice) score
  suggests that one modality was manipulated while the other was left intact —
  a strong indicator of a deepfake composite.

  |video_score - audio_score| > MISMATCH_THRESHOLD  →  "Mismatch detected"
  otherwise                                          →  "Match"
"""
from __future__ import annotations

MISMATCH_THRESHOLD = 20.0   # score-point gap that signals inconsistency


def check_face_voice_consistency(video_score: float, audio_score: float) -> str:
    """
    Return a human-readable consistency verdict.

    Parameters
    ----------
    video_score : float   (0-100 deepfake score for the video track)
    audio_score : float   (0-100 deepfake score for the audio track)

    Returns
    -------
    str — "Match" | "Mismatch detected"
    """
    gap = abs(video_score - audio_score)
    if gap > MISMATCH_THRESHOLD:
        return "Mismatch detected"
    return "Match"


def multimodal_score(video_score: float, audio_score: float) -> dict:
    """
    Produce the full multimodal analysis block for the /analyze-video response.
    """
    verdict = check_face_voice_consistency(video_score, audio_score)
    return {
        "face_voice_match": verdict,
        "score_delta": round(abs(video_score - audio_score), 2),
    }
