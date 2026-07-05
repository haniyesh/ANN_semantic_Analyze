"""
ensemble.py
===========
Canonical 3-model sentiment ensemble — single source of truth.

Import from here in training, live bot, and API. Never copy the logic.
Any change to weights, prompts, or neutral logic here automatically
propagates to all three paths, keeping train/serve parity.
"""

# Prompts MUST match what XGBoost v9 was trained on.
# Changing either string requires retraining from scratch.
FB_PROMPT = "Bitcoin crypto market: {title}"
RB_PROMPT = "BREAKING: {title} #Bitcoin #Crypto"


def ensemble_probs(cb: tuple, fb: tuple, rb: tuple) -> dict:
    """Compute ensemble feature columns from per-model (pos, neg, neu) triples.

    Equal 1/3 weights — matches XGBoost v9 training data in
    services/sentiment_score.py score_batch().

    Returns all 9 per-model probability columns, net_agreement, and
    '_avg' tuple (avg_pos, avg_neg, avg_neu) for downstream label derivation.
    Pop '_avg' before storing in a payload dict.
    """
    cb_pos, cb_neg, cb_neu = cb
    fb_pos, fb_neg, fb_neu = fb
    rb_pos, rb_neg, rb_neu = rb

    avg_pos = (cb_pos + fb_pos + rb_pos) / 3
    avg_neg = (cb_neg + fb_neg + rb_neg) / 3
    avg_neu = (cb_neu + fb_neu + rb_neu) / 3

    nets = [cb_pos - cb_neg, fb_pos - fb_neg, rb_pos - rb_neg]
    mean_net = sum(nets) / 3
    signs = [1 if n > 0 else (-1 if n < 0 else 0) for n in nets]
    agreement = 1.0 if len(set(signs)) == 1 else 0.5

    return {
        "cb_prob_pos": round(cb_pos, 4), "cb_prob_neg": round(cb_neg, 4), "cb_prob_neu": round(cb_neu, 4),
        "fb_prob_pos": round(fb_pos, 4), "fb_prob_neg": round(fb_neg, 4), "fb_prob_neu": round(fb_neu, 4),
        "rb_prob_pos": round(rb_pos, 4), "rb_prob_neg": round(rb_neg, 4), "rb_prob_neu": round(rb_neu, 4),
        "net_agreement": round(mean_net * agreement, 4),
        "_avg": (avg_pos, avg_neg, avg_neu),
    }


def sentiment_from_probs(avg_pos: float, avg_neg: float, avg_neu: float) -> tuple:
    """Derive (label, score_int, confidence) from averaged ensemble probabilities.

    Neutral wins when avg_neu > max(avg_pos, avg_neg) — matches the label
    derivation used to build the XGBoost v9 training targets in score_batch().
    """
    if avg_neu > max(avg_pos, avg_neg):
        return "neutral", 0, avg_neu
    net = avg_pos - avg_neg
    disc = (3 if net > 0.50 else 2 if net > 0.25 else 1 if net > 0.05 else
           -3 if net < -0.50 else -2 if net < -0.25 else -1 if net < -0.05 else 0)
    label = "positive" if disc > 0 else ("negative" if disc < 0 else "neutral")
    conf  = avg_pos if disc > 0 else (avg_neg if disc < 0 else avg_neu)
    return label, disc, conf


def reliability(cb: tuple, fb: tuple, rb: tuple,
                avg_pos: float, avg_neg: float) -> bool:
    """True when no single model diverges >0.3 from the ensemble direction."""
    cb_pos, cb_neg, _ = cb
    fb_pos, fb_neg, _ = fb
    rb_pos, rb_neg, _ = rb
    ens_net = abs(avg_pos - avg_neg)
    spread = max(
        abs(cb_pos - cb_neg) - ens_net,
        abs(fb_pos - fb_neg) - ens_net,
        abs(rb_pos - rb_neg) - ens_net,
    )
    return spread < 0.3
