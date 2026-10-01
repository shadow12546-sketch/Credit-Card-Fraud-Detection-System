"""Error analysis, rule baseline and threshold table for the trained fraud model.

Put next to train_model.py and run:
    python error_analysis.py --data advanced_credit_card_transactions.csv

Reuses add_history_features from train_model.py and the same 70/15/15 time split,
so the test set is exactly the one used in training.
"""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import fbeta_score, precision_score, recall_score

from train_model import TARGET, TIME_COL, add_history_features

NIGHT_HOURS = [22, 23, 0, 1, 2, 3]


def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="advanced_credit_card_transactions.csv")
    ap.add_argument("--model", default="model_output/fraud_model.joblib")
    ap.add_argument("--out-dir", default="model_output")
    args = ap.parse_args()
    out = Path(args.out_dir)

    b = joblib.load(args.model)
    df = pd.read_csv(args.data, low_memory=False)
    df = df.sort_values(TIME_COL, kind="stable").reset_index(drop=True)
    df = add_history_features(df)
    n = len(df)
    i1, i2 = int(n * 0.70), int(n * 0.85)
    val, test = df.iloc[i1:i2].copy(), df.iloc[i2:].copy()

    # ---- 1. how were hour / weekday / night_tx / distance_km derived? (the API must match)
    ts = pd.to_datetime(df[TIME_COL], unit="s")
    print("=== FEATURE DEFINITION CHECKS (the API relies on these) ===")
    print(f"hour == UTC hour of unix_time:      {(ts.dt.hour == df['hour']).mean():.4f}")
    print(f"weekday == weekday of (date + 7 years): "
          f"{((ts + pd.DateOffset(years=7)).dt.weekday == df['weekday']).mean():.4f}")
    print(f"night_tx == hour in {NIGHT_HOURS}: "
          f"{(df['night_tx'] == df['hour'].isin(NIGHT_HOURS).astype(int)).mean():.4f}")
    hv = haversine(df["lat"], df["long"], df["merch_lat"], df["merch_long"])
    print(f"distance_km vs haversine(lat,long,merch_lat,merch_long), mean abs diff: "
          f"{(hv - df['distance_km']).abs().mean():.4f} km")
    print("Anything well below 1.0 (or a large distance diff) means the API must be changed to match.\n")

    # ---- 2. score the test split
    def score(part):
        X = part[b["numeric"]].astype("float64").reset_index(drop=True)
        codes = b["encoder"].transform(part[b["categorical"]].astype(str))
        X = pd.concat([X, pd.DataFrame(codes, columns=b["categorical"])], axis=1)
        return b["model"].predict_proba(X[b["features"]])[:, 1]

    thr = b["threshold"]
    test["proba"] = score(test)
    test["pred"] = (test["proba"] >= thr).astype(int)
    fr = test[test[TARGET] == 1]
    missed, caught = fr[fr["pred"] == 0], fr[fr["pred"] == 1]
    print(f"=== TEST: {len(fr)} frauds | caught {len(caught)} | missed {len(missed)} (threshold {thr:.4f}) ===")

    # ---- 3. what do the misses look like?
    def profile(g):
        return pd.Series({
            "n": len(g),
            "night_share": g["night_tx"].mean(),
            "median_amt": g["amt"].median(),
            "median_amt_ratio": g["amt_ratio_before"].median(),
            "no_history_share": g["card_avg_amt_before"].isna().mean(),
            "median_score": g["proba"].median(),
        })

    print(pd.DataFrame({"missed": profile(missed), "caught": profile(caught)}).round(3).to_string())
    print("\nMissed frauds by category:")
    print(missed["category"].value_counts().head(8).to_string())
    print("\nMissed frauds: night_tx (rows) x amount bucket (columns):")
    print(pd.crosstab(missed["night_tx"], pd.cut(missed["amt"], [0, 50, 100, 250, 500, 1e9])).to_string())
    out.mkdir(exist_ok=True)
    missed.drop(columns=["proba"]).assign(score=missed["proba"]).to_csv(out / "missed_frauds.csv", index=False)
    print(f"\nSaved {out / 'missed_frauds.csv'}")

    # ---- 4. baseline: night AND amount >= T, T tuned on validation for F2
    def rule(part, t):
        return ((part["night_tx"] == 1) & (part["amt"] >= t)).astype(int)

    grid = np.arange(50, 1000, 10)
    best_t = max(grid, key=lambda t: fbeta_score(val[TARGET], rule(val, t), beta=2, zero_division=0))
    rp = rule(test, best_t)
    y = test[TARGET]
    print(f"\n=== BASELINE RULE: night_tx == 1 AND amt >= {best_t} (tuned on validation) vs MODEL ===")
    comp = pd.DataFrame({
        "precision": [precision_score(y, rp, zero_division=0), precision_score(y, test["pred"])],
        "recall": [recall_score(y, rp), recall_score(y, test["pred"])],
        "F2": [fbeta_score(y, rp, beta=2), fbeta_score(y, test["pred"], beta=2)],
        "alerts": [int(rp.sum()), int(test["pred"].sum())],
    }, index=["rule", "model"])
    print(comp.round(4).to_string())

    # ---- 5. threshold table
    rows = []
    for t in sorted({0.3, 0.4, 0.5, 0.6, 0.7, 0.8, round(thr, 4), 0.9, 0.95, 0.98}):
        flag = test["proba"] >= t
        tp = int((flag & (y == 1)).sum())
        rows.append({
            "threshold": t, "alerts": int(flag.sum()), "alert_rate_pct": round(100 * flag.mean(), 3),
            "frauds_caught": tp, "false_alarms": int(flag.sum()) - tp,
            "precision": round(tp / max(flag.sum(), 1), 4), "recall": round(tp / max(y.sum(), 1), 4),
        })
    table = pd.DataFrame(rows)
    print("\n=== THRESHOLD TABLE (test split) ===")
    print(table.to_string(index=False))
    table.to_csv(out / "threshold_table.csv", index=False)
    print(f"\nSaved {out / 'threshold_table.csv'}")


if __name__ == "__main__":
    main()
