# Publication Roadmap — Workshop Paper

Target: NLP/finance workshop (e.g., FinNLP @ ACL/EMNLP/COLING, ECONLP, or an IEEE conference special session). Format: 4–8 pages. Realistic timeline: **6–10 weeks** part-time.

## Where the project already stands

The hard methodological groundwork is done, and it's the part most student projects get wrong:

- Chronological 70/15/15 split with no lookahead
- RAG index built from training rows only (no leakage)
- Four naive baselines reported alongside model metrics
- Unreliable (date-only) timestamps dropped by default
- Honest limitations documented

What's missing is not rigor in the code — it's **reproducible results, a clear scientific claim, and statistical evidence**.

---

## Blocking items (must do, in order)

### B1. Re-measure everything on the leak-free pipeline
The README itself says previously reported metrics are invalid. Right now the paper has **no valid numbers**.

- Run `scripts/build_dataset.py` end-to-end from the raw Kaggle inputs
- Train v9 (XGBoost) and record final test metrics + all four baselines
- Freeze the dataset: record exact row counts, date range, class balance — these go in the paper's Data section
- Commit a dataset manifest (hashes + counts), even if the data itself can't be committed

### B2. Decide the paper's claim
Your citation says "multi-stream neural architecture," but the repo's production model is XGBoost, and the ANN (`v9.py`) isn't in the repo — only its results JSON is referenced. Pick one story:

- **Option A (recommended): comparison study.** "Do transformer embeddings + market context predict short-term BTC news impact? A rigorous comparison of a multi-stream ANN vs. gradient boosting vs. naive baselines." This makes the ANN-vs-XGBoost comparison the contribution, and honest negative/modest results are publishable at workshops.
- **Option B: the ANN is the contribution.** Then the ANN must be in the repo, trained on the same frozen dataset, and it should beat XGBoost — otherwise reviewers will ask why it exists.

Either way: **restore/recreate `v9.py` (the ANN) in the repo** and retrain it on the frozen dataset.

### B3. Address the noisy-label problem head-on
BTC routinely moves >0.3% in 15 min without news, so part of the positive class is noise. Reviewers will attack this first. Two defenses (do both):

- **Threshold sensitivity analysis**: report results at 0.3%, 0.5%, 1.0% (15m) — if the signal is real, precision should rise with the threshold
- **News vs. no-news control**: sample random no-news timestamps, compute the same labels, and show the base rate of "impact" without news. The model's value is the lift over this base rate.

### B4. Statistical significance
Point estimates of F1/AUC are not enough:

- Bootstrap 95% confidence intervals on test F1, precision, AUC (resample test set ~1000×)
- Train with 3–5 random seeds, report mean ± std
- DeLong test (or bootstrap) for ANN vs. XGBoost AUC difference

### B5. Ablation study
The multi-stream design is only a contribution if the streams matter. One table:

| Feature set | F1 | AUC |
|---|---|---|
| Embeddings only (1536d) | | |
| + sentiment ensemble | | |
| + news type | | |
| + macro/price context | | |
| + RAG (full model) | | |

This is cheap to run (same pipeline, masked feature blocks) and is the single most reviewer-pleasing addition.

---

## Strongly recommended (not blocking)

- **R1. Economic significance**: a toy backtest (long/short on predicted impactful+direction, with fees ~0.1%) — even a negative result is informative and honest
- **R2. Related-work section**: position against 3–5 papers on news-based crypto prediction (CryptoBERT paper, FinBERT applications, news-impact studies). I can help compile this.
- **R3. Error analysis**: which news types does the model get right/wrong (you already have the 11-type classifier)
- **R4. Unit tests for the leakage guards** (split assertion, RAG train-only) — reviewers increasingly check repos

## Skip for the paper

- Dashboard/live pipeline polish (mention in one paragraph as "deployment"; it's a nice differentiator but not the contribution)
- ETH modeling
- More model versions — freeze at v9/v10, no new variants

---

## Paper skeleton (workshop, ~6 pages)

1. **Intro** — question: does news content add predictive signal for short-horizon BTC moves beyond volatility itself?
2. **Related work** (R2)
3. **Data** — Telegram + Kaggle sources, labeling, timestamp handling, frozen stats (B1)
4. **Method** — feature streams, ANN + XGBoost, leakage controls
5. **Experiments** — main results vs. baselines (B1), significance (B4), ablations (B5), threshold sensitivity + no-news control (B3), backtest (R1)
6. **Limitations** — you've already written this section in the README; reviewers reward this honesty
7. **Conclusion**

## Suggested order of work

Week 1–2: B1 (rebuild + re-measure) → Week 2–3: B2 (restore ANN, retrain) → Week 3–4: B3 + B4 → Week 4–5: B5 + R1 → Week 5+: writing, R2, professor review loop.

**First concrete step:** run `python scripts/build_dataset.py --check`, fix whatever raw inputs are missing, and get one clean end-to-end training run. Everything else depends on that.
