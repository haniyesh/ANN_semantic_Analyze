# Model x Sentiment Comparison Matrix

Bootstrap resamples: 2000. CIs are percentile 95%.


## Horizon: 15m

| Model | Sentiment | F1 [95% CI] | Precision | Recall | AUC [95% CI] |
|---|---|---|---|---|---|
| ANN | BERT | 0.441 [0.425, 0.457] | 0.337 | 0.641 | 0.739 [0.727, 0.751] |
| ANN | Groq | 0.428 [0.411, 0.445] | 0.358 | 0.533 | 0.734 [0.722, 0.747] |
| XGBoost | BERT | 0.507 [0.490, 0.522] | 0.406 | 0.676 | 0.807 [0.796, 0.816] |
| XGBoost | Groq | 0.496 [0.480, 0.512] | 0.382 | 0.707 | 0.810 [0.800, 0.820] |

### Paired comparisons (15m)

| Comparison | Metric | Δ (A−B) | 95% CI | p-value |
|---|---|---|---|---|
| ANN vs XGBoost (BERT) | AUC | -0.068 | [-0.078, -0.058] | 0.000 * |
| ANN vs XGBoost (BERT) | F1 | -0.066 | [-0.079, -0.053] | 0.000 * |
| ANN vs XGBoost (Groq) | AUC | -0.076 | [-0.086, -0.065] | 0.000 * |
| ANN vs XGBoost (Groq) | F1 | -0.068 | [-0.082, -0.054] | 0.000 * |
| BERT vs Groq (ANN) | AUC | +0.004 | [0.001, 0.008] | 0.008 * |
| BERT vs Groq (ANN) | F1 | +0.013 | [0.004, 0.023] | 0.007 * |
| BERT vs Groq (XGBoost) | AUC | -0.003 | [-0.007, 0.001] | 0.089 |
| BERT vs Groq (XGBoost) | F1 | +0.011 | [0.002, 0.020] | 0.028 * |

## Horizon: 1h

| Model | Sentiment | F1 [95% CI] | Precision | Recall | AUC [95% CI] |
|---|---|---|---|---|---|
| ANN | BERT | 0.443 [0.428, 0.456] | 0.336 | 0.648 | 0.690 [0.679, 0.702] |
| ANN | Groq | 0.443 [0.428, 0.456] | 0.338 | 0.640 | 0.691 [0.680, 0.703] |
| XGBoost | BERT | 0.454 [0.441, 0.467] | 0.330 | 0.728 | 0.712 [0.701, 0.723] |
| XGBoost | Groq | 0.448 [0.434, 0.462] | 0.338 | 0.665 | 0.709 [0.697, 0.720] |

### Paired comparisons (1h)

| Comparison | Metric | Δ (A−B) | 95% CI | p-value |
|---|---|---|---|---|
| ANN vs XGBoost (BERT) | AUC | -0.022 | [-0.028, -0.015] | 0.000 * |
| ANN vs XGBoost (BERT) | F1 | -0.012 | [-0.020, -0.003] | 0.009 * |
| ANN vs XGBoost (Groq) | AUC | -0.018 | [-0.024, -0.011] | 0.000 * |
| ANN vs XGBoost (Groq) | F1 | -0.005 | [-0.014, 0.004] | 0.230 |
| BERT vs Groq (ANN) | AUC | -0.001 | [-0.004, 0.002] | 0.555 |
| BERT vs Groq (ANN) | F1 | -0.000 | [-0.005, 0.005] | 0.960 |
| BERT vs Groq (XGBoost) | AUC | +0.003 | [-0.001, 0.008] | 0.171 |
| BERT vs Groq (XGBoost) | F1 | +0.006 | [-0.001, 0.014] | 0.113 |

## LaTeX (main results table)

```latex
\begin{table}[t]\centering
\caption{Impact classification on the chronological test set. 95\% bootstrap CIs in brackets.}
\begin{tabular}{llcccc}
\toprule
Model & Sentiment & F1$_{15m}$ & AUC$_{15m}$ & F1$_{1h}$ & AUC$_{1h}$ \\
\midrule
ANN & BERT & 0.441 & 0.739 & 0.443 & 0.690 \\
ANN & Groq & 0.428 & 0.734 & 0.443 & 0.691 \\
XGBoost & BERT & 0.507 & 0.807 & 0.454 & 0.712 \\
XGBoost & Groq & 0.496 & 0.810 & 0.448 & 0.709 \\
\bottomrule
\end{tabular}\end{table}
```