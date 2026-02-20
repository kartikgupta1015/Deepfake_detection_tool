"""
DeepShield — Integration Test Suite
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    """
    Create a test client with models loaded.
    We monkey-patch the model loaders to be no-ops so tests run without
    GPU/large-download overhead — the routers still exercise all logic
    including validators, DB logging, and response schemas.
    """
    import models.image_model as im
    import models.audio_model as am

    # Patch loaders to skip actual weight download in CI
    original_img   = im.load_image_model
    original_audio = am.load_audio_model

    def _noop_image():
        """Stub: create the model with random weights (no pretrained download)."""
        import torch
        from models.image_model import DeepfakeImageDetector
        im._device = torch.device("cpu")
        im._model  = DeepfakeImageDetector()
        im._model.eval()

    def _noop_audio():
        import torch
        from models.audio_model import AudioAntiSpoofCNN
        am._device = torch.device("cpu")
        am._model  = AudioAntiSpoofCNN()
        am._model.eval()

    im.load_image_model   = _noop_image
    am.load_audio_model   = _noop_audio

    from main import app
    c = TestClient(app, raise_server_exceptions=False)

    # Restore
    im.load_image_model   = original_img
    am.load_audio_model   = original_audio

    yield c


# ─────────────────────────────────────────────────────────────────────────────
# GET /health
# ─────────────────────────────────────────────────────────────────────────────

class TestHealth:
    def test_returns_200(self, client):
        r = client.get("/health")
        assert r.status_code == 200

    def test_schema(self, client):
        r = client.get("/health")
        body = r.json()
        assert "status" in body
        assert body["status"] == "ok"
        assert "models_loaded" in body


# ─────────────────────────────────────────────────────────────────────────────
# POST /analyze-image
# ─────────────────────────────────────────────────────────────────────────────

class TestAnalyzeImage:
    def test_invalid_url_rejected(self, client):
        r = client.post("/analyze-image", json={"url": "ftp://bad.url/img.jpg"})
        assert r.status_code == 422

    def test_private_ip_blocked(self, client):
        r = client.post("/analyze-image", json={"url": "http://192.168.1.1/img.jpg"})
        assert r.status_code in (400, 422, 500)

    def test_missing_url_field(self, client):
        r = client.post("/analyze-image", json={})
        assert r.status_code == 422

    def test_response_schema_keys(self, client):
        """
        Use a real public image (Wikipedia Commons CC0).
        Skip gracefully if network unavailable in CI.
        """
        url = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/481px-Cat03.jpg"
        r = client.post("/analyze-image", json={"url": url}, timeout=30)
        if r.status_code == 502:
            pytest.skip("Network unavailable in CI")
        assert r.status_code == 200
        body = r.json()
        assert body["type"] == "image"
        assert 0 <= body["authenticity_score"] <= 100
        assert body["risk_level"] in ("Low", "Medium", "High")
        assert "analysis" in body
        assert "facial_inconsistency" in body["analysis"]
        assert "lighting_mismatch"    in body["analysis"]
        assert "gan_artifacts"        in body["analysis"]


# ─────────────────────────────────────────────────────────────────────────────
# POST /analyze-audio
# ─────────────────────────────────────────────────────────────────────────────

class TestAnalyzeAudio:
    def test_invalid_url_rejected(self, client):
        r = client.post("/analyze-audio", json={"url": "javascript:alert(1)"})
        assert r.status_code == 422

    def test_response_schema_keys(self, client):
        url = "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3"
        r = client.post("/analyze-audio", json={"url": url}, timeout=60)
        if r.status_code in (502, 400):
            pytest.skip("Network unavailable in CI")
        assert r.status_code == 200
        body = r.json()
        assert body["type"] == "audio"
        assert 0 <= body["authenticity_score"] <= 100
        assert body["risk_level"] in ("Low", "Medium", "High")
        assert "synthetic_probability" in body["analysis"]
        assert "pitch_irregularity"    in body["analysis"]


# ─────────────────────────────────────────────────────────────────────────────
# POST /analyze-video
# ─────────────────────────────────────────────────────────────────────────────

class TestAnalyzeVideo:
    def test_invalid_url_rejected(self, client):
        r = client.post("/analyze-video", json={"url": "not-a-url"})
        assert r.status_code == 422

    def test_response_schema_keys(self, client):
        # Small public-domain MP4 from the Internet Archive
        url = "https://archive.org/download/SampleVideo1280x7205mb/SampleVideo_1280x720_5mb.mp4"
        r = client.post("/analyze-video", json={"url": url}, timeout=120)
        if r.status_code in (502, 400):
            pytest.skip("Network unavailable in CI")
        assert r.status_code == 200
        body = r.json()
        assert body["type"] == "video"
        assert 0 <= body["authenticity_score"] <= 100
        assert body["risk_level"] in ("Low", "Medium", "High")
        assert "video_score"  in body
        assert "audio_score"  in body
        assert "face_voice_match" in body
        assert "frame_analysis"   in body
        assert "total_frames"      in body["frame_analysis"]
        assert "suspicious_frames" in body["frame_analysis"]
