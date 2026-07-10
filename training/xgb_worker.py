"""
Standalone XGBoost training + prediction worker.

Run as a subprocess from xgboost_train_bert.py to avoid the glibc tcache
double-free bug that XGBoost triggers on WSL2.

After training (or loading in --predict-only mode) the worker computes
predict_proba on val and test sets and saves them alongside feature importances
in --preds-out so the main process never has to import xgboost.
"""

import sys
import json
import argparse
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features",      required=True, help="Path to .npz with arrays")
    parser.add_argument("--params",        required=False, default=None,
                        help="Path to JSON file with base_params (not needed for --predict-only)")
    parser.add_argument("--clf15-out",     required=True)
    parser.add_argument("--clf1h-out",     required=True)
    parser.add_argument("--reg15-out",     required=True)
    parser.add_argument("--preds-out",     required=True, help="Path to .npz for predictions output")
    parser.add_argument("--predict-only",  action="store_true",
                        help="Skip training; load saved models and predict only")
    args = parser.parse_args()

    data = np.load(args.features)
    X_vl = data["X_vl"]
    X_te = data["X_te"]

    import xgboost as xgb

    if args.predict_only:
        print("  [predict-only] Loading saved models...", flush=True)
        clf_15m = xgb.XGBClassifier(); clf_15m.load_model(args.clf15_out)
        clf_1h  = xgb.XGBClassifier(); clf_1h.load_model(args.clf1h_out)
        reg_15m = xgb.XGBRegressor();  reg_15m.load_model(args.reg15_out)
        print("  Models loaded.", flush=True)
    else:
        X_tr     = data["X_tr"]
        y_c15_tr = data["y_c15_tr"]; y_c15_vl = data["y_c15_vl"]
        y_c1h_tr = data["y_c1h_tr"]; y_c1h_vl = data["y_c1h_vl"]
        y_r15_tr = data["y_r15_tr"]; y_r15_vl = data["y_r15_vl"]

        with open(args.params) as f:
            base_params = json.load(f)

        neg_15 = int((y_c15_tr == 0).sum())
        pos_15 = int((y_c15_tr == 1).sum())
        neg_1h = int((y_c1h_tr == 0).sum())
        pos_1h = int((y_c1h_tr == 1).sum())
        print(f"  15m — neg:{neg_15}  pos:{pos_15}  scale:{neg_15/pos_15:.2f}", flush=True)
        print(f"  1h  — neg:{neg_1h}  pos:{pos_1h}  scale:{neg_1h/pos_1h:.2f}", flush=True)

        print("  Training clf_15m...", flush=True)
        clf_15m = xgb.XGBClassifier(
            **base_params,
            objective="binary:logistic",
            scale_pos_weight=neg_15 / pos_15,
        )
        clf_15m.fit(X_tr, y_c15_tr, eval_set=[(X_vl, y_c15_vl)], verbose=False)
        print(f"    Best iteration: {clf_15m.best_iteration}", flush=True)
        clf_15m.get_booster().save_model(args.clf15_out)

        print("  Training clf_1h...", flush=True)
        clf_1h = xgb.XGBClassifier(
            **base_params,
            objective="binary:logistic",
            scale_pos_weight=neg_1h / pos_1h,
        )
        clf_1h.fit(X_tr, y_c1h_tr, eval_set=[(X_vl, y_c1h_vl)], verbose=False)
        print(f"    Best iteration: {clf_1h.best_iteration}", flush=True)
        clf_1h.get_booster().save_model(args.clf1h_out)

        reg_params = {k: v for k, v in base_params.items()
                      if k not in ["scale_pos_weight", "eval_metric"]}
        reg_params["eval_metric"] = "rmse"
        print("  Training reg_15m...", flush=True)
        reg_15m = xgb.XGBRegressor(**reg_params, objective="reg:squarederror")
        reg_15m.fit(X_tr, y_r15_tr, eval_set=[(X_vl, y_r15_vl)], verbose=False)
        reg_15m.get_booster().save_model(args.reg15_out)
        print("  XGBoost training complete.", flush=True)

    # Compute predictions in the subprocess (avoids xgboost import in main process)
    print("  Computing predictions...", flush=True)
    p15_vl = clf_15m.predict_proba(X_vl)[:, 1]
    p1h_vl = clf_1h.predict_proba(X_vl)[:, 1]
    p15_te = clf_15m.predict_proba(X_te)[:, 1]
    p1h_te = clf_1h.predict_proba(X_te)[:, 1]
    r15_te = reg_15m.predict(X_te)
    feat_imp_15m = clf_15m.feature_importances_
    print(f"  Predictions done: val={len(p15_vl)} test={len(p15_te)}", flush=True)

    np.savez(args.preds_out,
             p15_vl=p15_vl, p1h_vl=p1h_vl,
             p15_te=p15_te, p1h_te=p1h_te,
             r15_te=r15_te, feat_imp_15m=feat_imp_15m)
    print(f"  Predictions saved → {args.preds_out}", flush=True)

    import os as _os
    _os._exit(0)


if __name__ == "__main__":
    main()
