"""Integration tests for the POST /news ingest endpoint.

Tests auth (401 / 503 / duplicate) without any ML models or filesystem I/O.
Uses FastAPI's TestClient (sync) so no event loop boilerplate is needed.
"""
import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Minimal env so server.py imports without crashing on missing vars
os.environ.setdefault("INGEST_API_KEY", "test-secret-key-abc123")
os.environ.setdefault("LOG_LEVEL", "WARNING")   # quieten startup logs in tests


@pytest.fixture(scope="module")
def client():
    """Create a TestClient with a patched news_cache.json path."""
    import api.server as srv
    from fastapi.testclient import TestClient

    with patch.object(srv, "CACHE_FILE", Path("/nonexistent/cache.json")):
        with patch.object(srv, "HIST_CSV",   Path("/nonexistent/hist.csv")):
            with patch.object(srv, "historical_news", []):
                with patch.object(srv, "all_news", []):
                    with patch.object(srv, "hot_news", []):
                        # Pin the ingest key to a known test value regardless of .env
                        with patch.object(srv, "_INGEST_API_KEY", "test-secret-key-abc123"):
                            yield TestClient(srv.app)


VALID_ITEM = {
    "title": "SEC approves Bitcoin spot ETF in landmark decision",
    "channel": "cointelegraph",
    "model_score": 0.72,
    "model_score_1h": 0.61,
    "published_ts": 1_700_000_000,
    "type": "BUY",
    "confidence": 78.0,
    "sentiment": "positive",
    "id": "abc123def456",
}


def test_auth_503_when_key_not_configured(client):
    """Endpoint must return 503 when INGEST_API_KEY is unset."""
    import api.server as srv
    original = srv._INGEST_API_KEY
    srv._INGEST_API_KEY = ""
    try:
        resp = client.post("/news", json=VALID_ITEM, headers={"X-API-Key": ""})
        assert resp.status_code == 503
    finally:
        srv._INGEST_API_KEY = original


def test_auth_401_wrong_key(client):
    """Wrong key must return 401, not 200."""
    resp = client.post("/news", json=VALID_ITEM,
                       headers={"X-API-Key": "wrong-key"})
    assert resp.status_code == 401


def test_auth_401_missing_key(client):
    """Missing X-API-Key header must return 401."""
    resp = client.post("/news", json=VALID_ITEM)
    assert resp.status_code == 401


def test_auth_valid_key_accepted(client):
    """Correct key with a valid payload must not return 4xx/5xx."""
    import api.server as srv
    with patch.object(srv, "all_news", []):
        with patch.object(srv, "hot_news", []):
            # Also stub out the WebSocket broadcast and cache write
            with patch.object(srv, "_broadcast", return_value=None):
                with patch("asyncio.create_task"):
                    resp = client.post(
                        "/news", json=VALID_ITEM,
                        headers={"X-API-Key": "test-secret-key-abc123"},
                    )
    assert resp.status_code == 200
    body = resp.json()
    assert body.get("status") in ("ok", "duplicate", "filtered", "blocked")


def test_ingest_rejects_missing_title(client):
    """POST /news without `title` must fail Pydantic validation (422)."""
    bad = {k: v for k, v in VALID_ITEM.items() if k != "title"}
    resp = client.post("/news", json=bad,
                       headers={"X-API-Key": "test-secret-key-abc123"})
    assert resp.status_code == 422


def test_analyze_custom_rejects_short_title(client):
    """POST /analyze/custom with a 1-char title must fail validation (422)."""
    resp = client.post("/analyze/custom", json={"title": "X"},
                       headers={"X-API-Key": "test-secret-key-abc123"})
    assert resp.status_code == 422


def test_analyze_custom_rejects_empty_title(client):
    """POST /analyze/custom with empty title must fail (422)."""
    resp = client.post("/analyze/custom", json={"title": ""},
                       headers={"X-API-Key": "test-secret-key-abc123"})
    assert resp.status_code == 422


def test_news_all_returns_list(client):
    """GET /news/all must return a JSON array."""
    resp = client.get("/news/all")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


def test_news_all_pagination(client):
    """?limit=10&offset=0 must be accepted without errors."""
    resp = client.get("/news/all?limit=10&offset=0")
    assert resp.status_code == 200
    assert "X-Total-Count" in resp.headers


def test_health_endpoint(client):
    """GET /health must return 200."""
    resp = client.get("/health")
    assert resp.status_code == 200
