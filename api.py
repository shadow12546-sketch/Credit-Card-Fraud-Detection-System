"""Fraud scoring API.

    pip install fastapi uvicorn joblib xgboost scikit-learn pandas
    set HISTORY_CSV=advanced_credit_card_transactions.csv     (Windows; optional but recommended)
    uvicorn api:app --port 8000

The model needs per-card history (average amount so far, time since last transaction).
The API keeps that in memory. HISTORY_CSV seeds it at startup; without it every card
is "unseen" and the three history features are missing. History is lost on restart.

Feature definitions mirror train_model.py. hour/weekday are taken in UTC and
night_tx = hour in 22..3. Run error_analysis.py first: it checks these against your CSV.
"""
import os
import threading
from datetime import datetime, timezone
from typing import Optional, Union

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

BUNDLE = joblib.load(os.getenv("MODEL_PATH", "model_output/fraud_model.joblib"))
MODEL, ENC = BUNDLE["model"], BUNDLE["encoder"]
NUMERIC, CATEGORICAL, FEATURES, THR = (BUNDLE[k] for k in ("numeric", "categorical", "features", "threshold"))
KNOWN = {c: set(map(str, cats)) for c, cats in zip(CATEGORICAL, ENC.categories_)}
NIGHT_HOURS = {22, 23, 0, 1, 2, 3}
PROFILE_KEYS = ("age", "city_pop", "lat", "long")

CARDS: dict = {}  # token -> {n, sum_amt, last_ts, age, city_pop, lat, long}
LOCK = threading.Lock()


def seed_history(path: str):
    cols = ["cc_num", "amt", "unix_time", "age", "city_pop", "lat", "long"]
    d = pd.read_csv(path, usecols=cols).sort_values("unix_time", kind="stable")
    g = d.groupby("cc_num")
    agg = g.agg(n=("amt", "size"), sum_amt=("amt", "sum"), last_ts=("unix_time", "max"))
    last = g[list(PROFILE_KEYS)].last()
    for card, r in agg.join(last).iterrows():
        CARDS[str(card)] = {"n": int(r["n"]), "sum_amt": float(r["sum_amt"]), "last_ts": float(r["last_ts"]),
                            **{k: float(r[k]) for k in PROFILE_KEYS}}


def seed_summary(path: str):
    """Load the small per-card summary written by make_seed.py (same numbers seed_history computes)."""
    d = pd.read_csv(path, dtype={"cc_num": str})
    for r in d.itertuples(index=False):
        CARDS[r.cc_num] = {"n": int(r.n), "sum_amt": float(r.sum_amt), "last_ts": float(r.last_ts),
                           **{k: float(getattr(r, k)) for k in PROFILE_KEYS}}


SUMMARY = os.getenv("HISTORY_SUMMARY", "model_output/cards_seed.csv")
if os.getenv("HISTORY_CSV"):
    seed_history(os.environ["HISTORY_CSV"])
elif os.path.exists(SUMMARY):
    seed_summary(SUMMARY)

app = FastAPI(title="Fraud scoring API")



class Tx(BaseModel):
    card_token: str
    amount: float = Field(gt=0)
    category: str
    merchant: str
    state: str
    timestamp: Union[int, float, str]  # unix seconds, or ISO-8601 (no timezone = UTC)
    # Optional. Cardholder fields default to what is stored for the card.
    age: Optional[float] = None
    city_pop: Optional[float] = None
    lat: Optional[float] = None
    long: Optional[float] = None
    merch_lat: Optional[float] = None
    merch_long: Optional[float] = None
    commit: bool = True  # False = score only, do not add this transaction to the card's history


def parse_ts(v) -> float:
    if isinstance(v, (int, float)):
        return float(v)
    try:
        return float(v)
    except ValueError:
        pass
    try:
        dt = datetime.fromisoformat(v.replace("Z", "+00:00"))
    except ValueError:
        raise HTTPException(422, "timestamp must be unix seconds or ISO-8601")
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def dataset_weekday(dt: datetime) -> int:
    """The CSV's `weekday` is NOT the weekday of unix_time. It equals the weekday of the
    same calendar date 7 years later (unix_time is the original date shifted back 7 years).
    Verified against the CSV with error_analysis.py; this reproduces the training feature."""
    try:
        shifted = dt.replace(year=dt.year + 7)
    except ValueError:  # Feb 29
        shifted = dt.replace(year=dt.year + 7, day=28)
    return shifted.weekday()


def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return float(6371 * 2 * np.arcsin(np.sqrt(a)))


def clean(v):
    if isinstance(v, str):
        return v
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else float(v)


@app.get("/health")
def health():
    return {"status": "ok", "cards_in_memory": len(CARDS), "threshold": THR}


@app.get("/meta")
def meta():
    return {"categories": {c: sorted(map(str, cats)) for c, cats in zip(CATEGORICAL, ENC.categories_)},
            "threshold": THR, "sample_cards": list(CARDS)[:5]}


@app.get("/card/{token}")
def card(token: str):
    if token not in CARDS:
        raise HTTPException(404, "unknown card")
    return CARDS[token]


@app.post("/score")
def score(tx: Tx):
    ts = parse_ts(tx.timestamp)
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    warnings = []
    for c, val in (("category", tx.category), ("merchant", tx.merchant), ("state", tx.state)):
        if c in KNOWN and val not in KNOWN[c]:
            warnings.append(f"{c} '{val}' was not seen in training")

    with LOCK:
        st = CARDS.get(tx.card_token)
        if st is None:
            st = {"n": 0, "sum_amt": 0.0, "last_ts": None}
            warnings.append("unseen card: history features are missing")
        n, last = st["n"], st["last_ts"]
        avg = st["sum_amt"] / n if n > 0 else np.nan
        ratio = tx.amount / avg if n > 0 and avg > 0 else np.nan
        since = ts - last if last is not None else np.nan
        if last is not None and ts < last:
            warnings.append("timestamp is earlier than the card's last transaction; seconds-since-last is negative")

        prof = {k: (getattr(tx, k) if getattr(tx, k) is not None else st.get(k)) for k in PROFILE_KEYS}
        coords = [prof["lat"], prof["long"], tx.merch_lat, tx.merch_long]
        dist = haversine(*coords) if all(v is not None for v in coords) else np.nan
        if np.isnan(dist):
            warnings.append("distance_km missing (need merch_lat and merch_long)")
        for k in ("age", "city_pop"):
            if prof[k] is None:
                warnings.append(f"{k} missing")

        num = {"amt": tx.amount, "age": prof["age"], "hour": dt.hour, "night_tx": int(dt.hour in NIGHT_HOURS),
               "weekday": dataset_weekday(dt), "distance_km": dist, "city_pop": prof["city_pop"],
               "card_avg_amt_before": avg, "amt_ratio_before": ratio, "secs_since_last_tx": since}
        cats = {"category": tx.category, "merchant": tx.merchant, "state": tx.state}

        X = pd.DataFrame([[num.get(c) for c in NUMERIC]], columns=NUMERIC).astype("float64")
        codes = ENC.transform(pd.DataFrame([[cats[c] for c in CATEGORICAL]], columns=CATEGORICAL).astype(str))
        X = pd.concat([X, pd.DataFrame(codes, columns=CATEGORICAL)], axis=1)[FEATURES]

        proba = float(MODEL.predict_proba(X)[0, 1])
        contrib = MODEL.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)[0][:-1]  # last = bias

        if tx.commit:
            st["n"] += 1
            st["sum_amt"] += tx.amount
            st["last_ts"] = ts if last is None else max(last, ts)
            for k, v in prof.items():
                if v is not None:
                    st[k] = v
            CARDS[tx.card_token] = st

    shown = {**{c: X.iloc[0][c] for c in NUMERIC}, **cats}
    order = np.argsort(-np.abs(contrib))[:6]
    reasons = [{"feature": FEATURES[i], "value": clean(shown[FEATURES[i]]),
                "contribution": round(float(contrib[i]), 4),
                "effect": "raises risk" if contrib[i] > 0 else "lowers risk"} for i in order]
    return {
        "risk_score": round(proba, 4), "threshold": round(THR, 4),
        "decision": "ALERT" if proba >= THR else "OK",
        "reasons": reasons,  # contributions are in log-odds (XGBoost SHAP values)
        "history": {"prior_transactions": n, "card_avg_amt_before": clean(avg),
                    "amt_ratio_before": clean(ratio), "secs_since_last_tx": clean(since)},
        "warnings": warnings, "committed": tx.commit,
    }
