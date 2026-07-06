"""Train/serve parity guard for the sentiment feature contract.

The XGBoost model consumes a fixed set of per-model ensemble columns
(cb_/fb_/rb_prob_* + net_agreement). Both the offline scorer
(services.sentiment_score) and the live feature builder (main.build_xgb_features)
must agree on exactly those names, or training and inference silently diverge.

These tests are dependency-light: they inspect source text rather than
importing torch/httpx, so they run in CI without the full ML stack. The math
test extracts and execs only the pure-python helper.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# The canonical ensemble sentiment columns the model is trained on.
ENSEMBLE_COLS = [
    "cb_prob_pos", "cb_prob_neg", "cb_prob_neu",
    "fb_prob_pos", "fb_prob_neg", "fb_prob_neu",
    "rb_prob_pos", "rb_prob_neg", "rb_prob_neu",
    "net_agreement",
]


def _extract_function(path: Path, name: str) -> str:
    src = path.read_text()
    m = re.search(rf"(?ms)^def {name}\(.*?(?=^\S|\Z)", src)
    assert m, f"{name} not found in {path.name}"
    return m.group(0)


def test_offline_scorer_emits_ensemble_columns():
    src = (ROOT / "services" / "sentiment_score.py").read_text()
    for col in ENSEMBLE_COLS:
        assert f'"{col}"' in src, f"offline scorer missing {col}"


def test_live_builder_reads_ensemble_columns():
    src = (ROOT / "main.py").read_text()
    for col in ENSEMBLE_COLS:
        assert col in src, f"live build_xgb_features does not reference {col}"


def test_ensemble_math_runs_without_torch():
    """Exec just the pure-python helper and check averaging + agreement logic.
    Function was moved from sentiment_score.py to services/ensemble.py as ensemble_probs."""
    fn_src = _extract_function(ROOT / "services" / "ensemble.py", "ensemble_probs")
    ns = {}
    exec(fn_src, ns)
    _ensemble_columns = ns["ensemble_probs"]

    # All bullish → positive net, full agreement.
    out = _ensemble_columns((0.7, 0.1, 0.2), (0.6, 0.2, 0.2), (0.8, 0.1, 0.1))
    assert out["net_agreement"] > 0
    for col in ENSEMBLE_COLS:
        assert col in out

    # Mixed signs → agreement penalty shrinks the magnitude.
    mixed = _ensemble_columns((0.7, 0.1, 0.2), (0.1, 0.7, 0.2), (0.5, 0.5, 0.0))
    assert abs(mixed["net_agreement"]) < abs(out["net_agreement"])
