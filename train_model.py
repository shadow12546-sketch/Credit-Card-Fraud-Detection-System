"""Standalone fraud-model training (XGBoost) - one file, no project structure needed.

Put this file next to your CSV and run:
    python train_model.py --data advanced_credit_card_transactions.csv

What it does
  * time-based split (oldest 70% train / next 15% validation / newest 15% test)
  * card-history features computed from PAST transactions only (no leakage)
  * leaky / low-value columns are not used (see NUMERIC / CATEGORICAL below)
  * XGBoost with class weighting + early stopping on PR-AUC
  * decision threshold tuned on validation data for recall (F2)
  * saves model + metrics + SHAP plot to ./model_output
"""
import argparse
import json
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             fbeta_score, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score)
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBClassifier

TARGET = "is_fraud"
TIME_COL = "unix_time"
CARD_COL = "cc_num"

# Columns used by the model. Leaky or useless columns from your file are NOT here:
# unix_time (raw), gender, tx_count_card, avg_amt_card, amt_ratio, day, month, weekend,
# merch_zipcode, city, zip, job, lat, long, merch_lat, merch_long.
NUMERIC = ["amt", "age", "hour", "night_tx", "weekday", "distance_km", "city_pop"]
CATEGORICAL = ["category", "merchant", "state"]
HISTORY = ["card_avg_amt_before", "amt_ratio_before", "secs_since_last_tx"]  # built below
SEED = 42


def add_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per-card behaviour using only earlier transactions (df must be sorted by time)."""
    g = df.groupby(CARD_COL, sort=False)
    n_before = g.cumcount()
    amt_before = g["amt"].cumsum() - df["amt"]
    df["card_avg_amt_before"] = amt_before / n_before.replace(0, np.nan)
    df["amt_ratio_before"] = df["amt"] / df["card_avg_amt_before"].replace(0, np.nan)
    df["secs_since_last_tx"] = g[TIME_COL].diff()
    return df


def evaluate(y, proba, thr):
    pred = (proba >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "roc_auc": float(roc_auc_score(y, proba)),
        "pr_auc": float(average_precision_score(y, proba)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "f2": float(fbeta_score(y, pred, beta=2, zero_division=0)),
        "threshold": float(thr),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def best_threshold(y, proba, beta=2.0):
    p, r, t = precision_recall_curve(y, proba)
    p, r = p[:-1], r[:-1]
    f = (1 + beta**2) * p * r / np.maximum(beta**2 * p + r, 1e-12)
    return float(t[int(np.nanargmax(f))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="advanced_credit_card_transactions.csv")
    ap.add_argument("--out-dir", default="model_output")
    ap.add_argument("--max-rows", type=int, default=None, help="random subset for a quick test run")
    args = ap.parse_args()
    t0 = time.time()

    df = pd.read_csv(args.data, low_memory=False)
    for col in (TARGET, TIME_COL, CARD_COL, "amt"):
        if col not in df.columns:
            raise SystemExit(f"Required column '{col}' not found. Columns: {list(df.columns)}")
    if args.max_rows and len(df) > args.max_rows:
        df = df.sample(args.max_rows, random_state=SEED)
    df = df.sort_values(TIME_COL, kind="stable").reset_index(drop=True)
    df = add_history_features(df)

    numeric = [c for c in NUMERIC + HISTORY if c in df.columns]
    categorical = [c for c in CATEGORICAL if c in df.columns]
    missing = [c for c in NUMERIC + CATEGORICAL if c not in df.columns]
    if missing:
        print(f"[warn] columns not in your file, skipped: {missing}")
    print(f"Rows: {len(df):,} | fraud rate: {df[TARGET].mean():.4%}")
    print(f"Numeric features:     {numeric}")
    print(f"Categorical features: {categorical}")

    # ---- time-based split
    n = len(df)
    i1, i2 = int(n * 0.70), int(n * 0.85)
    train, val, test = df.iloc[:i1], df.iloc[i1:i2], df.iloc[i2:]
    for name, part in (("train", train), ("validation", val), ("test", test)):
        if part[TARGET].nunique() < 2:
            raise SystemExit(f"The {name} split has only one class - use more rows.")
        print(f"{name:<10} {len(part):>9,} rows | fraud {part[TARGET].mean():.3%}")

    # ---- encode (fit on train only)
    enc = None
    if categorical:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        enc.fit(train[categorical].astype(str))

    def make_X(part):
        X = part[numeric].astype("float64").reset_index(drop=True)
        if enc is not None:
            codes = enc.transform(part[categorical].astype(str))
            X = pd.concat([X, pd.DataFrame(codes, columns=categorical)], axis=1)
        return X

    Xtr, Xva, Xte = make_X(train), make_X(val), make_X(test)
    ytr, yva, yte = train[TARGET].values, val[TARGET].values, test[TARGET].values

    # ---- XGBoost
    pos = int(ytr.sum())
    spw = (len(ytr) - pos) / max(pos, 1)
    model = XGBClassifier(
        n_estimators=1000, learning_rate=0.05, max_depth=6, subsample=0.8,
        colsample_bytree=0.8, scale_pos_weight=spw, eval_metric="aucpr",
        early_stopping_rounds=50, tree_method="hist", n_jobs=-1, random_state=SEED)
    model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)

    thr = best_threshold(yva, model.predict_proba(Xva)[:, 1])
    metrics = evaluate(yte, model.predict_proba(Xte)[:, 1], thr)

    print("\n=== TEST RESULTS (newest 15% of transactions) ===")
    for k in ("roc_auc", "pr_auc", "precision", "recall", "f1", "f2", "threshold"):
        print(f"{k:<10} {metrics[k]:.4f}")
    print(f"confusion  TN={metrics['tn']:,} FP={metrics['fp']:,} FN={metrics['fn']:,} TP={metrics['tp']:,}")

    gain = pd.Series(model.get_booster().get_score(importance_type="gain")).sort_values(ascending=False)
    print("\nTop features by gain:")
    print(gain.head(10).round(1).to_string())

    # ---- save
    out = Path(args.out_dir)
    out.mkdir(exist_ok=True)
    joblib.dump({"model": model, "encoder": enc, "numeric": numeric, "categorical": categorical,
                 "features": list(Xtr.columns), "threshold": thr}, out / "fraud_model.joblib", compress=3)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))

    try:
        import shap
        sample = Xte.sample(min(2000, len(Xte)), random_state=SEED)
        sv = shap.TreeExplainer(model).shap_values(sample)
        plt.figure()
        shap.summary_plot(sv, sample, show=False, max_display=15)
        plt.savefig(out / "shap_summary.png", dpi=150, bbox_inches="tight")
        plt.close()
    except Exception as exc:
        print(f"[warn] SHAP plot skipped: {exc}")

    print(f"\nSaved to {out.resolve()} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

