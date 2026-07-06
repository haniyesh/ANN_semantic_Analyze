"""Cross-file contract tests.

These tests enforce that thresholds, vocabulary, and feature dimensions
stay consistent across config.py, api/server.py, and any module that
scores news. If one of these fails after a change, it means a constant
was updated in one place but not propagated everywhere.

Run with: pytest tests/test_contract.py -v
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ── 1. Config → Server threshold contract ────────────────────────────────────

def test_server_imports_thresholds_from_config():
    """server.py must derive SCORE_HOT/MED/SHOW/CONF_MIN from config, not redefine."""
    import api.server as srv
    from config import (
        SCORE_THRESHOLD_HOT, SCORE_THRESHOLD_MEDIUM,
        SCORE_THRESHOLD_SHOW, CONF_SHOW,
    )
    assert srv.SCORE_HOT  == SCORE_THRESHOLD_HOT,    f"server SCORE_HOT={srv.SCORE_HOT} != config {SCORE_THRESHOLD_HOT}"
    assert srv.SCORE_MED  == SCORE_THRESHOLD_MEDIUM, f"server SCORE_MED={srv.SCORE_MED} != config {SCORE_THRESHOLD_MEDIUM}"
    assert srv.SCORE_SHOW == SCORE_THRESHOLD_SHOW,   f"server SCORE_SHOW={srv.SCORE_SHOW} != config {SCORE_THRESHOLD_SHOW}"
    assert srv.CONF_MIN   == CONF_SHOW * 100,        f"server CONF_MIN={srv.CONF_MIN} != config {CONF_SHOW * 100}"


def test_config_endpoint_values_match_config():
    """/config endpoint must return the values actually in use, not stale copies."""
    from fastapi.testclient import TestClient
    import api.server as srv
    from config import (
        SCORE_THRESHOLD_HOT, SCORE_THRESHOLD_MEDIUM,
        SCORE_THRESHOLD_SHOW, CONF_SHOW,
    )
    client = TestClient(srv.app)
    r = client.get("/config")
    assert r.status_code == 200
    body = r.json()
    assert body["score_hot"]    == SCORE_THRESHOLD_HOT
    assert body["score_medium"] == SCORE_THRESHOLD_MEDIUM
    assert body["score_show"]   == SCORE_THRESHOLD_SHOW
    assert body["conf_min"]     == CONF_SHOW * 100


# ── 2. Impact vocabulary contract ─────────────────────────────────────────────

VALID_IMPACT_LABELS = {"Hot", "Medium", "Show", "Low"}


def test_config_impact_tier_vocabulary():
    """config.impact_tier() must return only valid labels."""
    from config import impact_tier
    for s15, s1h in [(0.0, 0.0), (0.29, 0.29), (0.30, 0.30), (0.55, 0.55), (0.80, 0.80), (1.0, 1.0)]:
        label = impact_tier(s15, s1h)
        assert label in VALID_IMPACT_LABELS, f"impact_tier({s15},{s1h}) returned unexpected '{label}'"


def test_server_recompute_impact_vocabulary():
    """_recompute_impact() must return only the canonical vocabulary."""
    import api.server as srv
    for score in (0.0, 0.25, 0.30, 0.54, 0.55, 0.79, 0.80, 1.0):
        item = {"model_score": score, "model_score_1h": score}
        label = srv._recompute_impact(item)
        assert label in VALID_IMPACT_LABELS, f"_recompute_impact at score={score} returned '{label}'"


def test_impact_tier_boundaries():
    """Boundary values must land in the correct tiers."""
    from config import impact_tier
    assert impact_tier(0.80, 0.00) == "Hot"
    assert impact_tier(0.00, 0.80) == "Hot"
    assert impact_tier(0.55, 0.00) == "Medium"
    assert impact_tier(0.30, 0.00) == "Show"
    assert impact_tier(0.29, 0.29) == "Low"


# ── 3. Feature dimension contract ─────────────────────────────────────────────

def test_build_xgb_features_dimension():
    """build_xgb_features must produce exactly the number of features the model expects."""
    import numpy as np
    try:
        import xgboost as xgb
        import pickle
    except ImportError:
        import pytest; pytest.skip("xgboost not installed")

    clf_path    = ROOT / "xgb_impact_clf_15m_bert.json"
    scaler_path = ROOT / "xgb_feature_scaler_bert.pkl"
    if not clf_path.exists() or not scaler_path.exists():
        import pytest; pytest.skip("model files not present")

    clf = xgb.XGBClassifier()
    clf.load_model(str(clf_path))
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    n_model  = clf.n_features_in_
    n_scaler = scaler.n_features_in_
    assert n_model == n_scaler, (
        f"Model expects {n_model} features but scaler was fit on {n_scaler}. "
        "Re-run training/xgboost_train_bert.py to rebuild both."
    )

    # Simulate build_xgb_features with zero inputs
    import sys; sys.path.insert(0, str(ROOT))
    from main import build_xgb_features
    dummy_emb  = np.zeros(768, dtype=np.float32)
    dummy_sent = {k: 0.0 for k in [
        "cb_prob_pos", "cb_prob_neg", "cb_prob_neu",
        "fb_prob_pos", "fb_prob_neg", "fb_prob_neu",
        "rb_prob_pos", "rb_prob_neg", "rb_prob_neu",
        "net_agreement", "sentiment_score", "weight", "confidence",
    ]}
    dummy_macro = np.zeros(8, dtype=np.float32)
    features = build_xgb_features(dummy_sent, dummy_emb, dummy_emb, dummy_macro)
    assert features.shape[0] == n_model, (
        f"build_xgb_features produced {features.shape[0]} dims, model expects {n_model}. "
        "Check RAG padding in build_xgb_features."
    )
