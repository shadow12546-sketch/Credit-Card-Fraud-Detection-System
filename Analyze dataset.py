#!/usr/bin/env python
"""Data-driven analysis of the fraud dataset -> reports/dataset_analysis.md + assets/eda/*.png

    python scripts/analyze_dataset.py --data data/raw/advanced_credit_card_transactions.csv

Every number in the report is computed from your file. Sections that need human judgement
(3, 9, 10, 11) get the measured evidence; paste the report back for the written assessment.
"""
from __future__ import annotations

import argparse
import re
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
EDA = ROOT / "assets" / "eda"

TARGETS = ["is_fraud", "isfraud", "fraud", "fraud_flag", "is_fraudulent", "fraudulent", "class", "label", "target"]
ID_RE = re.compile(r"(^|_)(id|uuid|index)$|^unnamed|card_?num|cc_?num|trans_?num", re.I)
MEANING = {
    "amt": "Transaction amount", "amount": "Transaction amount",
    "cc_num": "Card identifier", "card_number": "Card identifier", "merchant": "Merchant name",
    "category": "Merchant / purchase category", "gender": "Cardholder gender", "state": "Cardholder state",
    "city": "Cardholder city", "zip": "Cardholder ZIP code", "lat": "Cardholder latitude",
    "long": "Cardholder longitude", "city_pop": "Population of cardholder's city", "job": "Cardholder occupation",
    "dob": "Cardholder date of birth", "trans_num": "Unique transaction id", "unix_time": "Transaction unix time",
    "merch_lat": "Merchant latitude", "merch_long": "Merchant longitude", "first": "First name",
    "last": "Last name", "street": "Street address", "trans_date_trans_time": "Transaction timestamp",
}


# ------------------------------------------------------------------ helpers
def fmt(v) -> str:
    if isinstance(v, (float, np.floating)):
        return "" if np.isnan(v) else f"{v:,.4g}"
    if isinstance(v, (int, np.integer)):
        return f"{v:,}"
    return str(v).replace("|", "/")


def table(df: pd.DataFrame, index: bool = True) -> str:
    d = df.reset_index() if index else df
    cols = [str(c) for c in d.columns]
    rows = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    rows += ["| " + " | ".join(fmt(v) for v in r) + " |" for r in d.itertuples(index=False)]
    return "\n".join(rows)


def rate_table(s: pd.Series, y: pd.Series, min_n: int = 30) -> pd.DataFrame:
    g = pd.DataFrame({"k": s, "y": y}).dropna().groupby("k")["y"].agg(["size", "sum", "mean"])
    g.columns = ["transactions", "frauds", "fraud_rate"]
    g["lift_vs_avg"] = g["fraud_rate"] / y.mean()
    return g[g["transactions"] >= min_n].sort_values("fraud_rate", ascending=False)


def auc(x: pd.Series, y: pd.Series) -> float:
    m = x.notna() & np.isfinite(pd.to_numeric(x, errors="coerce"))
    if m.sum() < 50 or y[m].nunique() < 2 or x[m].nunique() < 2:
        return np.nan
    return float(roc_auc_score(y[m], x[m]))


def bar(rt: pd.DataFrame, title: str, fname: str, top: int | None = None, sort_index: bool = False):
    d = rt.sort_index() if sort_index else rt.head(top) if top else rt
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(d.index.astype(str), d["fraud_rate"] * 100, color="#d62728")
    ax.set_ylabel("fraud rate (%)")
    ax.set_title(title)
    plt.xticks(rotation=60, ha="right")
    fig.savefig(EDA / fname, dpi=130, bbox_inches="tight")
    plt.close(fig)


def detect_target(df: pd.DataFrame, given: str | None) -> str:
    if given:
        return given
    low = {c.lower().strip().replace(" ", "_"): c for c in df.columns}
    for t in TARGETS:
        if t in low and df[low[t]].nunique() == 2:
            return low[t]
    for c in df.columns:
        if "fraud" in c.lower() and df[c].nunique() == 2:
            return c
    raise SystemExit(f"Cannot detect target. Columns: {list(df.columns)}. Use --target")


def to_binary(y: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(y) or y.dtype == bool:
        return y.astype(int)
    return y.astype(str).str.strip().str.lower().isin({"1", "yes", "y", "true", "fraud", "fraudulent"}).astype(int)


def is_text(s: pd.Series) -> bool:
    return not (pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s))


def find(df, pattern, exclude=None):
    for c in df.columns:
        if re.search(pattern, c, re.I) and not (exclude and re.search(exclude, c, re.I)):
            return c
    return None


def haversine(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--target")
    ap.add_argument("--out", default=str(ROOT / "reports" / "dataset_analysis.md"))
    args = ap.parse_args()

    EDA.mkdir(parents=True, exist_ok=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.data, low_memory=False)
    tgt = detect_target(df, args.target)
    y = to_binary(df[tgt])
    raw = df.drop(columns=[tgt])
    out: list[str] = []
    W = out.append

    # ---------- derive time / person / geo / card features
    derived = pd.DataFrame(index=df.index)
    ts_col = None
    for c in raw.columns:
        if is_text(raw[c]) and re.search(r"date|time|ts|timestamp", c, re.I):
            t = pd.to_datetime(raw[c], errors="coerce", utc=True)
            if t.notna().mean() > 0.9:
                ts_col, ts = c, t
                break
    if ts_col is None and (u := find(raw, r"unix")) is not None:
        ts_col, ts = u, pd.to_datetime(raw[u], unit="s", errors="coerce", utc=True)
    if ts_col is not None:
        derived["hour"] = ts.dt.hour
        derived["weekday"] = ts.dt.dayofweek
        derived["month"] = ts.dt.month
        derived["is_weekend"] = (ts.dt.dayofweek >= 5).astype(float).where(ts.notna())
        derived["night_tx"] = ((ts.dt.hour >= 22) | (ts.dt.hour < 6)).astype(float).where(ts.notna())
    dob_col = find(raw, r"dob|birth")
    if dob_col is not None and ts_col is not None:
        dob = pd.to_datetime(raw[dob_col], errors="coerce", utc=True)
        derived["age"] = (ts - dob).dt.days / 365.25
    lat, lon = find(raw, r"^lat", r"merch"), find(raw, r"^(long|lon)", r"merch")
    mlat, mlon = find(raw, r"merch.*lat"), find(raw, r"merch.*(long|lon)")
    if all([lat, lon, mlat, mlon]):
        derived["distance_km"] = haversine(raw[lat], raw[lon], raw[mlat], raw[mlon])
    amt_col = find(raw, r"^(amt|amount)", r"ratio|avg")
    card_col = find(raw, r"cc_?num|card")
    if amt_col and card_col and ts_col is not None:
        order = ts.sort_values().index
        a = pd.to_numeric(raw.loc[order, amt_col], errors="coerce")
        g = a.groupby(raw.loc[order, card_col])
        derived["tx_count_card"] = g.cumcount()
        prior_mean = g.transform(lambda s: s.expanding().mean().shift(1))
        derived["avg_amt_card"] = prior_mean
        derived["amt_ratio"] = a / prior_mean.replace(0, np.nan)
        derived = derived.reindex(df.index)
    age_col = "age" if "age" in derived else None

    # ============ 1. OVERVIEW
    fr = y.mean()
    W("# Dataset analysis\n")
    W("## 1. Dataset overview\n")
    W(f"- Rows: **{len(df):,}**\n- Columns: **{df.shape[1]}** (including the target)\n- Target column: **`{tgt}`**")
    W(f"- Fraud: **{int(y.sum()):,}** ({fr:.4%}) | Non-fraud: **{int((1 - y).sum()):,}** ({1 - fr:.4%})")
    W(f"- Imbalance ratio: **1 : {(1 - fr) / max(fr, 1e-12):,.1f}**")
    W("- Time span: " + (f"{ts.min():%Y-%m-%d} to {ts.max():%Y-%m-%d}" if ts_col is not None else "no timestamp column detected"))
    W("- Purpose / business problem: *(written assessment - see note at the end)*\n")

    # ============ 2. FEATURES
    W("## 2. Feature description\n")
    rows = []
    for c in raw.columns:
        s = raw[c]
        num = pd.api.types.is_numeric_dtype(s) and s.dtype != bool
        rows.append({
            "feature": c, "dtype": str(s.dtype), "unique": int(s.nunique()),
            "missing_%": round(s.isna().mean() * 100, 3),
            "example": str(s.dropna().iloc[0])[:30] if s.notna().any() else "",
            "min": s.min() if num else "", "median": s.median() if num else "", "max": s.max() if num else "",
            "likely_meaning": MEANING.get(c.lower(), "(describe manually)"),
            "single_feature_AUC": auc(s, y) if num else np.nan,
        })
    W(table(pd.DataFrame(rows), index=False) + "\n")

    # ============ 3. ENGINEERED FEATURES
    W("## 3. Feature engineering analysis\n")
    pre = [c for c in raw.columns if re.search(r"distance|night|weekend|avg_amt|amt_ratio|tx_count|^hour|^age$|velocity", c, re.I)]
    W(f"Columns already engineered in the file: {pre if pre else 'none detected'}\n")
    W("Features derived here from raw columns, with measured signal "
      "(AUC 0.5 = no signal; card-history features use only *earlier* rows per card):\n")
    ev = []
    for c in derived.columns:
        s = derived[c]
        a = auc(s, y)
        row = {"feature": c, "AUC": a, "signal_strength": abs(a - 0.5) * 2 if pd.notna(a) else np.nan,
               "mean_if_fraud": s[y == 1].mean(), "mean_if_legit": s[y == 0].mean()}
        ev.append(row)
    if ev:
        W(table(pd.DataFrame(ev), index=False) + "\n")
    else:
        W("No derivable features detected (no timestamp / geo / card columns found).\n")

    # ============ 4. DATA QUALITY
    W("## 4. Data quality report\n")
    miss = df.isna().sum()
    miss = miss[miss > 0]
    W("**Missing values:** " + ("none" if miss.empty else "\n\n" + table(pd.DataFrame({"missing": miss, "%": miss / len(df) * 100}))) + "\n")
    idc = find(raw, r"trans_?num|transaction_?id")
    W(f"**Duplicate rows (all columns):** {int(df.duplicated().sum()):,}")
    if idc:
        W(f"**Duplicate `{idc}` values:** {int(raw[idc].duplicated().sum()):,}")
    W("\n**Outliers (IQR rule, numeric columns):**\n")
    orows = []
    for c in raw.select_dtypes("number").columns:
        if ID_RE.search(c) or raw[c].nunique() < 10:
            continue
        q1, q3 = raw[c].quantile([.25, .75])
        i = q3 - q1
        n = int(((raw[c] < q1 - 1.5 * i) | (raw[c] > q3 + 1.5 * i)).sum())
        orows.append({"feature": c, "outliers": n, "outlier_%": n / len(df) * 100, "skew": raw[c].skew()})
    W(table(pd.DataFrame(orows), index=False) + "\n" if orows else "n/a\n")
    W("**Invalid-value checks:**\n")
    inv = []
    if amt_col:
        a = pd.to_numeric(raw[amt_col], errors="coerce")
        inv += [("amount <= 0", int((a <= 0).sum())), ("amount not numeric", int(a.isna().sum() - raw[amt_col].isna().sum()))]
    for c, lo, hi in ((lat, -90, 90), (lon, -180, 180), (mlat, -90, 90), (mlon, -180, 180)):
        if c:
            inv.append((f"{c} outside [{lo},{hi}]", int(((raw[c] < lo) | (raw[c] > hi)).sum())))
    if age_col:
        inv.append(("age < 0 or > 110", int(((derived.age < 0) | (derived.age > 110)).sum())))
    if ts_col is not None:
        inv.append(("unparseable timestamps", int(ts.isna().sum())))
    W(table(pd.DataFrame(inv, columns=["check", "violations"]), index=False) + "\n" if inv else "no checks applicable\n")
    W(f"**Class imbalance:** fraud rate {fr:.4%}\n")

    # ============ 5. EDA
    W("## 5. Exploratory data analysis\n")
    if amt_col:
        a = pd.to_numeric(raw[amt_col], errors="coerce")
        W(f"**`{amt_col}` by class:**\n")
        W(table(a.groupby(y).describe(percentiles=[.25, .5, .75, .95, .99]).rename(index={0: "legit", 1: "fraud"})) + "\n")
        fig, ax = plt.subplots(figsize=(8, 4))
        for k, col in ((0, "#1f77b4"), (1, "#d62728")):
            ax.hist(np.log1p(a[y == k].clip(lower=0)), bins=60, alpha=.55, density=True, color=col, label=["legit", "fraud"][k])
        ax.set_title("log(1+amount) by class")
        ax.legend()
        fig.savefig(EDA / "amount_dist.png", dpi=130, bbox_inches="tight")
        plt.close(fig)
    if age_col:
        W("**Age by class:**\n")
        W(table(derived.age.groupby(y).describe().rename(index={0: "legit", 1: "fraud"})) + "\n")
    gen = find(raw, r"gender|sex")
    if gen:
        W("**Fraud by gender:**\n\n" + table(rate_table(raw[gen], y, 1)) + "\n")
    st = find(raw, r"^state")
    if st:
        rt = rate_table(raw[st], y)
        W("**Fraud by state (top 10 / bottom 5, min 30 txns):**\n\n" + table(pd.concat([rt.head(10), rt.tail(5)])) + "\n")
        bar(rt, "Fraud rate by state (highest)", "fraud_by_state.png", top=15)
    cat = find(raw, r"categor")
    if cat:
        rt = rate_table(raw[cat], y)
        W("**Fraud by category:**\n\n" + table(rt) + "\n")
        bar(rt, "Fraud rate by category", "fraud_by_category.png")
    for name, col in (("hour", "hour"), ("day of week (0=Mon)", "weekday"), ("month", "month")):
        if col in derived:
            rt = rate_table(derived[col], y, 1)
            W(f"**Fraud by {name}:**\n\n" + table(rt.sort_index()) + "\n")
            bar(rt, f"Fraud rate by {name}", f"fraud_by_{col}.png", sort_index=True)

    # ============ 6. PATTERNS
    W("## 6. Fraud pattern discovery\n")
    W("Buckets with at least 30 transactions, ranked by fraud rate (lift = rate / overall rate):\n")
    if cat:
        W("**Highest-risk categories:**\n\n" + table(rate_table(raw[cat], y).head(5)) + "\n")
    if "hour" in derived:
        W("**Highest-risk hours:**\n\n" + table(rate_table(derived.hour, y).head(5)) + "\n")
    if st:
        W("**Highest-risk states:**\n\n" + table(rate_table(raw[st], y).head(5)) + "\n")
    if age_col:
        bins = pd.cut(derived.age, [0, 25, 35, 45, 55, 65, 75, 120])
        W("**Age groups:**\n\n" + table(rate_table(bins.astype(str), y, 1)) + "\n")
    if amt_col:
        a = pd.to_numeric(raw[amt_col], errors="coerce")
        dec = pd.qcut(a, 10, duplicates="drop").astype(str)
        W("**Amount deciles:**\n\n" + table(rate_table(dec, y, 1)) + "\n")

    # ============ 7. IMPORTANCE ESTIMATE
    W("## 7. Feature importance estimate (no model trained)\n")
    W("Mutual information with the label on a sample, plus single-feature AUC. Higher = more predictive signal on its own; "
      "interactions are not captured.\n")
    # derived columns whose names already exist in the file are skipped (avoids duplicate column names)
    extra = derived.loc[:, [c for c in derived.columns if c not in raw.columns]]
    use = pd.concat([raw, extra], axis=1)
    use = use[[c for c in use.columns if not ID_RE.search(c) and c not in (ts_col, dob_col)]]
    use = use.loc[:, use.nunique() > 1]
    use = use.loc[:, [c for c in use.columns if not (is_text(use[c]) and use[c].nunique() > 0.5 * len(use))]]
    samp = use.sample(min(200_000, len(use)), random_state=0)
    ys = y.loc[samp.index]
    X, disc = {}, []
    for c in samp.columns:
        if is_text(samp[c]) or pd.api.types.is_bool_dtype(samp[c]):
            X[c] = pd.factorize(samp[c].astype(str))[0]
            disc.append(True)
        else:
            X[c] = pd.to_numeric(samp[c], errors="coerce").fillna(samp[c].median())
            disc.append(False)
    mi = pd.Series(mutual_info_classif(pd.DataFrame(X), ys, discrete_features=np.array(disc), random_state=0), index=samp.columns)
    mi = mi.sort_values(ascending=False)
    hi, lo = mi.quantile(.67), mi.quantile(.33)
    imp = pd.DataFrame({"mutual_info": mi, "tier": np.where(mi >= hi, "HIGH", np.where(mi >= lo, "MEDIUM", "LOW"))})
    imp["single_AUC"] = [np.nan if (is_text(use[c]) or pd.api.types.is_bool_dtype(use[c])) else auc(use[c], y) for c in imp.index]
    W(table(imp) + "\n")

    # ============ 8. ML READINESS (facts)
    W("## 8. Machine-learning readiness (measured facts)\n")
    cats = {c: int(raw[c].nunique()) for c in raw.columns if is_text(raw[c])}
    W(f"- Positive samples: {int(y.sum()):,} (rule of thumb: >1,000 positives is comfortable for gradient boosting)")
    W(f"- Categorical cardinalities: {cats}")
    W(f"- Highly skewed numeric columns (|skew|>2): {[r['feature'] for r in orows if abs(r['skew']) > 2]}")
    W("- Tree models need no scaling; linear/NN baselines do. SMOTE is usually unnecessary when positives are in the "
      "thousands - try `scale_pos_weight` and threshold tuning first.\n")

    # ============ 9-11
    W("## 9-11. Deployment readiness, improvements, final assessment\n")
    W("These need interpretation of the tables above. Paste this report (or upload it) and I will write them with your real numbers.\n")

    Path(args.out).write_text("\n".join(out), encoding="utf-8")
    print(f"Report written to {args.out}\nCharts in {EDA}")


if __name__ == "__main__":
    main()
