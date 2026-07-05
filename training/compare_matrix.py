"""
2x2 Comparison Matrix — Model Family x Sentiment Source
=========================================================
Compares on the SAME chronological test set:

                 BERT ensemble sentiment      Groq/Llama-3.3-70B sentiment
  Multi-stream   ann_bert_test_preds.npz   ann_groq_test_preds.npz
  ANN
  XGBoost        xgb_bert_test_preds.npz    xgb_groq_test_preds.npz

For every cell: F1 / Precision / Recall / ROC-AUC with bootstrap 95% CIs.
Paired comparisons (same test rows, paired bootstrap, two-sided p):
  - ANN vs XGBoost (within each sentiment source)
  - BERT vs Groq   (within each model family)

Run AFTER the four training runs:
    python training/xgboost_train_bert.py
    python training/xgboost_train_groq.py
    python training/ann_train.py --sentiment bert
    python training/ann_train_groq.py
    python training/compare_matrix.py

Outputs: comparison_matrix.md (markdown + LaTeX tables) + console report.
"""

import sys, json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent

N_BOOT = 2000
SEED   = 42

CELLS = {
    ("ANN",     "BERT"): ROOT / "ann_bert_test_preds.npz",
    ("ANN",     "Groq"): ROOT / "ann_groq_test_preds.npz",
    ("XGBoost", "BERT"): ROOT / "xgb_bert_test_preds.npz",
    ("XGBoost", "Groq"): ROOT / "xgb_groq_test_preds.npz",
}

HORIZONS = {"15m": ("p15", "y_c15", "thr_15m"), "1h": ("p1h", "y_c1h", "thr_1h")}


def load_cells():
    cells, missing = {}, []
    for key, path in CELLS.items():
        if path.exists():
            cells[key] = np.load(path)
        else:
            missing.append((key, path.name))
    for key, name in missing:
        print(f"  ⚠ missing {name} — cell {key} skipped")
    return cells


def check_alignment(cells):
    """Paired tests require identical test rows. Verify via published timestamps."""
    refs = [(k, v["published"]) for k, v in cells.items() if "published" in v]
    if len(refs) < 2:
        return True
    k0, ts0 = refs[0]
    for k, ts in refs[1:]:
        if len(ts) != len(ts0) or not np.array_equal(ts, ts0):
            print(f"  ⚠ TEST ROWS DIFFER between {k0} and {k} — paired tests are INVALID.")
            print("    Re-run all four scripts on the same frozen dataset.")
            return False
    return True


def point_metrics(p, y, thr):
    pred = (p >= thr).astype(int)
    return {
        "F1":   f1_score(y, pred, zero_division=0),
        "Prec": precision_score(y, pred, zero_division=0),
        "Rec":  recall_score(y, pred, zero_division=0),
        "AUC":  roc_auc_score(y, p) if len(np.unique(y)) > 1 else float("nan"),
    }


def bootstrap_ci(p, y, thr, n_boot=N_BOOT, seed=SEED):
    """Percentile bootstrap 95% CIs for F1 and AUC."""
    rng = np.random.default_rng(seed)
    n = len(y)
    f1s, aucs = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb, pb = y[idx], p[idx]
        if len(np.unique(yb)) < 2:
            continue
        f1s.append(f1_score(yb, (pb >= thr).astype(int), zero_division=0))
        aucs.append(roc_auc_score(yb, pb))
    return (np.percentile(f1s, [2.5, 97.5]), np.percentile(aucs, [2.5, 97.5]))


def paired_bootstrap_test(p_a, p_b, y, thr_a, thr_b, metric="AUC",
                          n_boot=N_BOOT, seed=SEED):
    """
    Two-sided paired bootstrap p-value for metric(A) - metric(B).
    Same resampled rows applied to both models (paired design).
    """
    rng = np.random.default_rng(seed)
    n = len(y)

    def m(p, yy, thr):
        if metric == "AUC":
            return roc_auc_score(yy, p) if len(np.unique(yy)) > 1 else np.nan
        return f1_score(yy, (p >= thr).astype(int), zero_division=0)

    obs = m(p_a, y, thr_a) - m(p_b, y, thr_b)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        if len(np.unique(yb)) < 2:
            continue
        diffs.append(m(p_a[idx], yb, thr_a) - m(p_b[idx], yb, thr_b))
    diffs = np.array(diffs)
    # p-value: how often the sign flips relative to observed difference
    p_val = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    ci = np.percentile(diffs, [2.5, 97.5])
    return obs, ci, min(p_val, 1.0)


def fmt_ci(lo_hi):
    return f"[{lo_hi[0]:.3f}, {lo_hi[1]:.3f}]"


def main():
    print("=" * 70)
    print("  2x2 COMPARISON — Model family x Sentiment source")
    print("=" * 70)

    cells = load_cells()
    if not cells:
        sys.exit("No prediction files found. Run the four training scripts first.")
    aligned = check_alignment(cells)

    lines_md = ["# Model x Sentiment Comparison Matrix\n",
                f"Bootstrap resamples: {N_BOOT}. CIs are percentile 95%.\n"]

    all_results = {}
    for hz, (p_key, y_key, thr_key) in HORIZONS.items():
        print(f"\n── Horizon: {hz} ──")
        lines_md.append(f"\n## Horizon: {hz}\n")
        lines_md.append("| Model | Sentiment | F1 [95% CI] | Precision | Recall | AUC [95% CI] |")
        lines_md.append("|---|---|---|---|---|---|")

        for (model, sent), d in cells.items():
            p, y, thr = d[p_key], d[y_key], float(d[thr_key])
            pm = point_metrics(p, y, thr)
            f1_ci, auc_ci = bootstrap_ci(p, y, thr)
            all_results[(model, sent, hz)] = pm
            row = (f"| {model} | {sent} | {pm['F1']:.3f} {fmt_ci(f1_ci)} "
                   f"| {pm['Prec']:.3f} | {pm['Rec']:.3f} "
                   f"| {pm['AUC']:.3f} {fmt_ci(auc_ci)} |")
            lines_md.append(row)
            print(f"  {model:<8} {sent:<5}  F1={pm['F1']:.3f} {fmt_ci(f1_ci)}  "
                  f"AUC={pm['AUC']:.3f} {fmt_ci(auc_ci)}")

        # Paired significance tests
        if aligned:
            lines_md.append(f"\n### Paired comparisons ({hz})\n")
            lines_md.append("| Comparison | Metric | Δ (A−B) | 95% CI | p-value |")
            lines_md.append("|---|---|---|---|---|")
            pairs = [
                (("ANN", "BERT"), ("XGBoost", "BERT"), "ANN vs XGBoost (BERT)"),
                (("ANN", "Groq"), ("XGBoost", "Groq"), "ANN vs XGBoost (Groq)"),
                (("ANN", "BERT"), ("ANN", "Groq"),     "BERT vs Groq (ANN)"),
                (("XGBoost", "BERT"), ("XGBoost", "Groq"), "BERT vs Groq (XGBoost)"),
            ]
            print(f"\n  Paired tests ({hz}):")
            for a_key, b_key, label in pairs:
                if a_key not in cells or b_key not in cells:
                    continue
                da, db = cells[a_key], cells[b_key]
                y = da[y_key]
                for metric in ("AUC", "F1"):
                    obs, ci, pv = paired_bootstrap_test(
                        da[p_key], db[p_key], y,
                        float(da[thr_key]), float(db[thr_key]), metric=metric)
                    sig = " *" if pv < 0.05 else ""
                    lines_md.append(f"| {label} | {metric} | {obs:+.3f} | {fmt_ci(ci)} | {pv:.3f}{sig} |")
                    print(f"    {label:<26} {metric:<4} Δ={obs:+.3f} {fmt_ci(ci)} p={pv:.3f}{sig}")

    # LaTeX table for the paper
    lines_md.append("\n## LaTeX (main results table)\n")
    lines_md.append("```latex")
    lines_md.append(r"\begin{table}[t]\centering")
    lines_md.append(r"\caption{Impact classification on the chronological test set. "
                    r"95\% bootstrap CIs in brackets.}")
    lines_md.append(r"\begin{tabular}{llcccc}")
    lines_md.append(r"\toprule")
    lines_md.append(r"Model & Sentiment & F1$_{15m}$ & AUC$_{15m}$ & F1$_{1h}$ & AUC$_{1h}$ \\")
    lines_md.append(r"\midrule")
    for model in ("ANN", "XGBoost"):
        for sent in ("BERT", "Groq"):
            r15 = all_results.get((model, sent, "15m"))
            r1h = all_results.get((model, sent, "1h"))
            if r15 and r1h:
                lines_md.append(
                    f"{model} & {sent} & {r15['F1']:.3f} & {r15['AUC']:.3f} "
                    f"& {r1h['F1']:.3f} & {r1h['AUC']:.3f} \\\\")
    lines_md.append(r"\bottomrule")
    lines_md.append(r"\end{tabular}\end{table}")
    lines_md.append("```")

    out = ROOT / "comparison_matrix.md"
    out.write_text("\n".join(lines_md), encoding="utf-8")
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
