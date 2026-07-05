"""
Multi-Stream ANN Trainer
=========================
The multi-stream (multi-tower) neural architecture, ported onto the EXACT
same feature pipeline as xgboost_train_bert.py so the ANN-vs-XGBoost comparison is
fair (identical data, identical chronological split, identical features):

  - Dual CryptoBERT+FinBERT embeddings (1536 dims, projected 768->16 each)
  - Sentiment: --sentiment bert  -> 3-BERT ensemble (same as xgboost_train_bert)
               --sentiment groq  -> Groq/Llama-3.3-70B one-hot (same as xgboost_train_groq)
  - News-type probs (11), macro timing (5) + price context (3), RAG (train-only index)
  - Chronological 70/15/15 split with temporal-overlap assertions
  - Same threshold search (MIN_PRECISION=0.20) and same baselines

Architecture (adapted from archive/production_system_v8.py CryptoImpactNetV5):
  sem_tower (proj emb + sentiment + type) -> NewsTypeGating -> cat(mac_tower)
  -> bidirectional cross-attention with rag_tower -> fusion -> heads
  Heads: cls_15m, cls_1h, reg_15m, direction

Outputs (used by training/compare_matrix.py):
  ann_{bert|groq}_results.json       — metrics
  ann_{bert|groq}_test_preds.npz     — per-row test predictions

Usage (run from project root):
    python training/ann_train.py --sentiment bert
    python training/ann_train.py --sentiment groq
    python training/ann_train.py --sentiment bert --skip-rag   # debug
    python training/ann_train.py --sentiment bert --seed 44    # multi-seed runs
"""

import os, sys, json, warnings, argparse
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))                # pipeline.*
sys.path.insert(0, str(ROOT / "training"))   # sibling training scripts

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from dotenv import load_dotenv

load_dotenv()

# Reuse the EXACT feature builders from the XGBoost scripts — this is what
# guarantees the comparison is apples-to-apples.
from xgboost_train_bert import (
    load_data, compute_cryptobert_embeddings, compute_finbert_embeddings,
    crypto_news_type_classify, build_macro_features, compute_price_context,
    find_threshold, eval_horizon, evaluate_baselines,
    NEWS_TYPE_LABELS, MIN_PRECISION, THRESHOLD_15M, THRESHOLD_1H,
)
from xgboost_train_groq import build_groq_features

# ══════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════
HERE          = ROOT
EMB_PROJ_DIMS = 16
LEARNING_RATE = 3e-4
WEIGHT_DECAY  = 1e-3
EPOCHS        = 200
PATIENCE      = 20
BATCH_SIZE    = 64
GRAD_CLIP     = 1.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ══════════════════════════════════════════════════════════════════
# FEATURE BLOCKS  (kept separate per stream — unlike XGBoost's flat X)
# ══════════════════════════════════════════════════════════════════
def build_feature_blocks(df: pd.DataFrame, train_idx: np.ndarray,
                         sentiment: str, skip_rag: bool):
    print("[2/7] FEATURE ENGINEERING (multi-stream)")
    cb_emb = compute_cryptobert_embeddings(df)   # (N, 768)
    fb_emb = compute_finbert_embeddings(df)      # (N, 768)

    # News-type probabilities (same as xgboost_train_bert)
    if "news_type" in df.columns and df["news_type"].notna().mean() > 0.5:
        type_onehot = pd.get_dummies(df["news_type"].fillna("market_analysis"))
        for label in NEWS_TYPE_LABELS:
            if label not in type_onehot.columns:
                type_onehot[label] = 0
        type_probs = type_onehot[NEWS_TYPE_LABELS].values.astype(np.float32)
    else:
        type_probs = crypto_news_type_classify(cb_emb)

    # Sentiment block — the experimental variable
    if sentiment == "groq":
        print("  Sentiment source: Groq/Llama-3.3-70B (same as xgboost_train_groq)")
        groq_feats  = build_groq_features(df)                       # (N, 3)
        scalar_cols = ["sentiment_score", "weight", "confidence"]
        scalar_df   = df[scalar_cols].fillna(0).values.astype(np.float32)
        sent        = np.hstack([groq_feats, scalar_df]).astype(np.float32)  # (N, 6)
    else:
        ensemble_cols = [
            "cb_prob_pos", "cb_prob_neg", "cb_prob_neu",
            "fb_prob_pos", "fb_prob_neg", "fb_prob_neu",
            "rb_prob_pos", "rb_prob_neg", "rb_prob_neu",
            "net_agreement",
        ]
        if all(c in df.columns for c in ensemble_cols):
            sent_cols = ensemble_cols + ["sentiment_score", "weight", "confidence"]
            print(f"  Sentiment source: 3-BERT ensemble ({len(sent_cols)} dims, same as xgboost_train_bert)")
        else:
            sent_cols = ["sentiment_score", "weight", "confidence",
                         "prob_positive", "prob_negative", "prob_neutral"]
            print(f"  Sentiment source: legacy ({len(sent_cols)} dims)")
        sent = df[sent_cols].fillna(0).values.astype(np.float32)

    # Macro stream: 5 timing + 3 price context (same as xgboost_train_bert)
    macro = np.hstack([build_macro_features(df),
                       compute_price_context(df)]).astype(np.float32)

    # RAG stream — train-only index, no leakage (same as xgboost_train_bert)
    if skip_rag:
        print("  RAG      : SKIPPED (--skip-rag flag)")
        rag = np.zeros((len(df), 1), dtype=np.float32)
    else:
        from pipeline.rag_news import build_rag_features_qdrant
        ch_rates = df.iloc[train_idx].groupby("channel")["is_impactful_15m"].mean().to_dict()
        rag, _ = build_rag_features_qdrant(
            df, channel_impact_rates=ch_rates,
            train_idx=train_idx, rebuild=True,
        )
        rag = rag.astype(np.float32)
        print(f"  RAG      : {rag.shape[1]} dims (train-only index)")

    print(f"  Streams  : cb {cb_emb.shape[1]} | fb {fb_emb.shape[1]} | "
          f"sent {sent.shape[1]} | type {type_probs.shape[1]} | "
          f"macro {macro.shape[1]} | rag {rag.shape[1]}")
    return cb_emb, fb_emb, sent, type_probs, macro, rag


# ══════════════════════════════════════════════════════════════════
# MODEL  (multi-stream, adapted from v8 CryptoImpactNetV5)
# ══════════════════════════════════════════════════════════════════
class EmbProjection(nn.Module):
    def __init__(self, in_dim=768, out_dim=EMB_PROJ_DIMS):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, 32), nn.LayerNorm(32), nn.GELU(),
            nn.Dropout(0.5), nn.Linear(32, out_dim),
        )

    def forward(self, x):
        return self.proj(x)


class SmallTower(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim), nn.BatchNorm1d(out_dim),
            nn.ReLU(), nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class CrossAttentionFusion(nn.Module):
    """Bidirectional cross-attention between two feature towers."""

    def __init__(self, dim_a, dim_b, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.proj_a   = nn.Linear(dim_a, d_model)
        self.proj_b   = nn.Linear(dim_b, d_model)
        self.attn_a2b = nn.MultiheadAttention(d_model, num_heads, batch_first=True, dropout=0.1)
        self.attn_b2a = nn.MultiheadAttention(d_model, num_heads, batch_first=True, dropout=0.1)
        self.norm_a   = nn.LayerNorm(d_model)
        self.norm_b   = nn.LayerNorm(d_model)
        self.out_dim  = d_model * 2

    def forward(self, a, b):
        a_p = self.proj_a(a).unsqueeze(1)
        b_p = self.proj_b(b).unsqueeze(1)
        a_e, _ = self.attn_a2b(query=a_p, key=b_p, value=b_p)
        b_e, _ = self.attn_b2a(query=b_p, key=a_p, value=a_p)
        a_out = self.norm_a((a_p + a_e).squeeze(1))
        b_out = self.norm_b((b_p + b_e).squeeze(1))
        return torch.cat([a_out, b_out], dim=1)


class NewsTypeGating(nn.Module):
    """Soft-gates the semantic tower output conditioned on news-type probs."""

    def __init__(self, type_dim, sem_dim):
        super().__init__()
        self.gate_net = nn.Sequential(nn.Linear(type_dim, sem_dim), nn.Sigmoid())

    def forward(self, sem_out, type_probs):
        return sem_out * self.gate_net(type_probs)


class MultiStreamNet(nn.Module):
    """
    Streams:
      sem  = [cb_proj(16), fb_proj(16), sentiment, type(11)]  -> sem_tower -> 24
      mac  = macro timing + price context (8)                 -> mac_tower -> 6
      rag  = RAG retrieval features                           -> rag_tower -> 6
    Fusion:
      NewsTypeGating(sem_out) -> cat(mac_out) -> cross-attn with rag_out
      -> fusion MLP -> heads (cls_15m, cls_1h, reg_15m, direction)
    """

    def __init__(self, sent_dim, type_dim, macro_dim, rag_dim):
        super().__init__()
        self.cb_proj = EmbProjection()
        self.fb_proj = EmbProjection()
        sem_in = EMB_PROJ_DIMS * 2 + sent_dim + type_dim

        self.sem_tower = SmallTower(sem_in,    24, dropout=0.3)
        self.mac_tower = SmallTower(macro_dim,  6, dropout=0.4)
        self.rag_tower = SmallTower(rag_dim,    6, dropout=0.4)

        self.type_gate = NewsTypeGating(type_dim, 24)
        self.cross     = CrossAttentionFusion(dim_a=30, dim_b=6, d_model=12, num_heads=2)

        self.fusion = nn.Sequential(
            nn.Linear(24, 16), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(16, 12), nn.ReLU(),
        )
        self.head_cls_15m = nn.Linear(12, 1)
        self.head_cls_1h  = nn.Linear(12, 1)
        self.head_reg_15m = nn.Linear(12, 1)
        self.head_dir     = nn.Linear(12, 2)

    def forward(self, cb, fb, sent, type_probs, macro, rag):
        sem = torch.cat([self.cb_proj(cb), self.fb_proj(fb), sent, type_probs], dim=1)
        s = self.sem_tower(sem)
        s = self.type_gate(s, type_probs)
        m = self.mac_tower(macro)
        r = self.rag_tower(rag)
        fused = self.fusion(self.cross(torch.cat([s, m], dim=1), r))
        return {
            "cls_15m":   self.head_cls_15m(fused).squeeze(1),
            "cls_1h":    self.head_cls_1h(fused).squeeze(1),
            "reg_15m":   self.head_reg_15m(fused).squeeze(1),
            "direction": self.head_dir(fused),
        }


# ══════════════════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════════════════
def make_loader(tensors, shuffle, drop_last=False):
    return DataLoader(TensorDataset(*tensors), batch_size=BATCH_SIZE,
                      shuffle=shuffle, drop_last=drop_last)


def multitask_loss(out, y_c15, y_c1h, y_r15, y_dir, pw15, pw1h):
    dir_mask = (y_r15.abs() > THRESHOLD_15M).float()
    loss_c15 = nn.BCEWithLogitsLoss(pos_weight=pw15)(out["cls_15m"], y_c15)
    loss_c1h = nn.BCEWithLogitsLoss(pos_weight=pw1h)(out["cls_1h"],  y_c1h)
    ce       = nn.CrossEntropyLoss(reduction="none")
    loss_dir = (ce(out["direction"], y_dir) * dir_mask).sum() / (dir_mask.sum() + 1e-8)
    loss_reg = nn.MSELoss()(out["reg_15m"], y_r15)
    return 2.0 * loss_c15 + 2.0 * loss_c1h + 1.0 * loss_dir + 0.3 * loss_reg


@torch.no_grad()
def predict(model, loader):
    model.eval()
    p15, p1h, r15 = [], [], []
    for batch in loader:
        cb, fb, sent, tp, mac, rag = [t.to(device) for t in batch[:6]]
        out = model(cb, fb, sent, tp, mac, rag)
        p15.append(torch.sigmoid(out["cls_15m"]).cpu().numpy())
        p1h.append(torch.sigmoid(out["cls_1h"]).cpu().numpy())
        r15.append(out["reg_15m"].cpu().numpy())
    return np.concatenate(p15), np.concatenate(p1h), np.concatenate(r15)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sentiment", choices=["bert", "groq"], default="bert")
    parser.add_argument("--skip-rag", action="store_true")
    parser.add_argument("--seed", type=int, default=43)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    results_path = HERE / f"ann_{args.sentiment}_results.json"
    preds_path   = HERE / f"ann_{args.sentiment}_test_preds.npz"
    model_path   = HERE / f"ann_{args.sentiment}_weights.pt"

    print("=" * 65)
    print(f"  MULTI-STREAM ANN v9 — sentiment={args.sentiment}  seed={args.seed}")
    print(f"  device={device}  threshold_15m={THRESHOLD_15M}  min_precision={MIN_PRECISION}")
    print("=" * 65)

    df = load_data()

    print("\n[3/7] CHRONOLOGICAL SPLIT (70/15/15 by time)")
    n     = len(df)
    n_tr  = int(n * 0.70)
    n_val = int(n * 0.15)
    tri   = np.arange(0, n_tr)
    vi    = np.arange(n_tr, n_tr + n_val)
    tei   = np.arange(n_tr + n_val, n)

    # Hard guarantee: no temporal overlap across splits (same as xgboost_train_bert).
    assert df["published"].iloc[tri].max() <= df["published"].iloc[vi].min()
    assert df["published"].iloc[vi].max()  <= df["published"].iloc[tei].min()
    print(f"  Train: {len(tri):,} | Val: {len(vi):,} | Test: {len(tei):,}")

    cb, fb, sent, tp, macro, rag = build_feature_blocks(
        df, tri, sentiment=args.sentiment, skip_rag=args.skip_rag)

    y_c15 = df["is_impactful_15m"].values.astype(np.float32)
    y_c1h = df["is_impactful_1h"].values.astype(np.float32)
    y_r15 = df["btc_change_15m"].values.astype(np.float32)
    y_r1h = df["btc_change_1h"].values.astype(np.float32)
    y_dir = df["direction_15m"].values.astype(np.int64)

    # Scale each stream on TRAIN only
    print("\n[4/7] TRAINING")
    blocks, scaled = [cb, fb, sent, macro, rag], {}
    names          = ["cb", "fb", "sent", "macro", "rag"]
    for name, x in zip(names, blocks):
        sc = StandardScaler().fit(x[tri])
        scaled[name] = sc.transform(x).astype(np.float32)
    scaled["tp"] = tp  # probabilities — no scaling

    def split_tensors(idx, with_labels=True):
        t = [torch.from_numpy(scaled[k][idx]) for k in ["cb", "fb", "sent", "tp", "macro", "rag"]]
        if with_labels:
            t += [torch.from_numpy(y_c15[idx]), torch.from_numpy(y_c1h[idx]),
                  torch.from_numpy(y_r15[idx]), torch.from_numpy(y_dir[idx])]
        return t

    tr_loader = make_loader(split_tensors(tri), shuffle=True, drop_last=True)
    vl_loader = make_loader(split_tensors(vi),  shuffle=False)

    pos15 = y_c15[tri].mean()
    pos1h = y_c1h[tri].mean()
    pw15  = torch.tensor([(1 - pos15) / max(pos15, 1e-6)], device=device)
    pw1h  = torch.tensor([(1 - pos1h) / max(pos1h, 1e-6)], device=device)
    print(f"  pos_weight 15m={pw15.item():.2f}  1h={pw1h.item():.2f}")

    model = MultiStreamNet(sent_dim=sent.shape[1], type_dim=tp.shape[1],
                           macro_dim=macro.shape[1], rag_dim=rag.shape[1]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    opt   = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    best_val, patience, best_state = float("inf"), 0, None
    for epoch in range(EPOCHS):
        model.train()
        tr_loss, nb = 0.0, 0
        for batch in tr_loader:
            cb_b, fb_b, s_b, tp_b, m_b, rg_b, c15, c1h, r15, d = [t.to(device) for t in batch]
            opt.zero_grad()
            out  = model(cb_b, fb_b, s_b, tp_b, m_b, rg_b)
            loss = multitask_loss(out, c15, c1h, r15, d, pw15, pw1h)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            tr_loss += loss.item(); nb += 1
        sched.step()

        model.eval()
        vl_loss, nvb = 0.0, 0
        with torch.no_grad():
            for batch in vl_loader:
                cb_b, fb_b, s_b, tp_b, m_b, rg_b, c15, c1h, r15, d = [t.to(device) for t in batch]
                out = model(cb_b, fb_b, s_b, tp_b, m_b, rg_b)
                vl_loss += multitask_loss(out, c15, c1h, r15, d, pw15, pw1h).item(); nvb += 1
        vl_loss /= nvb

        if vl_loss < best_val - 1e-5:
            best_val, patience = vl_loss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
        if epoch % 10 == 0 or patience == 0:
            print(f"  epoch {epoch:>3}  train {tr_loss/nb:.4f}  val {vl_loss:.4f}"
                  f"{'  *' if patience == 0 else ''}")
        if patience >= PATIENCE:
            print(f"  Early stop at epoch {epoch} (best val {best_val:.4f})")
            break

    model.load_state_dict(best_state)
    torch.save({"state_dict": best_state,
                "dims": {"sent": sent.shape[1], "type": tp.shape[1],
                         "macro": macro.shape[1], "rag": rag.shape[1]},
                "sentiment": args.sentiment, "seed": args.seed}, model_path)

    print(f"\n[5/7] THRESHOLD SEARCH (min_precision={MIN_PRECISION})")
    p15_vl, p1h_vl, _ = predict(model, vl_loader)
    thr_15m = find_threshold(p15_vl, y_c15[vi], "15m")
    thr_1h  = find_threshold(p1h_vl, y_c1h[vi], "1h")

    print(f"\n[6/7] TEST SET EVALUATION")
    te_loader = make_loader(split_tensors(tei), shuffle=False)
    p15_te, p1h_te, r15_te = predict(model, te_loader)
    dir_pred = (p15_te >= 0.5).astype(int)

    np.savez(
        preds_path,
        p15=p15_te, p1h=p1h_te, r15=r15_te,
        y_c15=y_c15[tei], y_c1h=y_c1h[tei],
        y_r15=y_r15[tei], y_r1h=y_r1h[tei],
        thr_15m=thr_15m, thr_1h=thr_1h,
        published=df["published"].iloc[tei].astype("int64").values,
    )
    print(f"  Test predictions saved → {preds_path}")

    r15 = eval_horizon("15-minute", p15_te, thr_15m, y_c15[tei], r15_te, y_r15[tei])
    r1h = eval_horizon("1-hour",    p1h_te, thr_1h,  y_c1h[tei], r15_te, y_r1h[tei])

    base_15 = evaluate_baselines(y_c15[tei], y_r15[tei], "15m")
    base_1h = evaluate_baselines(y_c1h[tei], y_r1h[tei], "1h")
    if r15["F1"] <= max(b["F1"] for b in base_15.values()):
        print("  ⚠️  15m model does NOT beat naive baselines — result is not meaningful.")
    if r1h["F1"] <= max(b["F1"] for b in base_1h.values()):
        print("  ⚠️  1h model does NOT beat naive baselines — result is not meaningful.")

    dir_acc = accuracy_score(y_dir[tei], dir_pred)
    dir_f1  = f1_score(y_dir[tei], dir_pred, zero_division=0)

    results = {
        "model": "multi_stream_ann",
        "sentiment": args.sentiment,
        "seed": args.seed,
        "n_params": int(n_params),
        "15_minute": r15, "1_hour": r1h,
        "direction": {"Acc": float(dir_acc), "F1": float(dir_f1)},
        "threshold_15m": float(thr_15m), "threshold_1h": float(thr_1h),
        "baselines_15m": base_15, "baselines_1h": base_1h,
        "split": "chronological_70_15_15",
    }
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[7/7] Saved → {results_path}")

    # ann_bert_results.json is already written above via results_path

    print(f"\n{'='*65}\n  SUMMARY — Multi-Stream ANN v9 ({args.sentiment})\n{'='*65}")
    print(f"  15m F1: {r15['F1']:.3f}  AUC: {r15['ROC_AUC']:.3f}")
    print(f"  1h  F1: {r1h['F1']:.3f}  AUC: {r1h['ROC_AUC']:.3f}")


if __name__ == "__main__":
    main()
