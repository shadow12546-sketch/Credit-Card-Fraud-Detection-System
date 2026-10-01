"""Build a small per-card history file so the API can run without the 316 MB CSV.

    python make_seed.py                                   (reads advanced_credit_card_transactions.csv)
    python make_seed.py other_file.csv

Writes model_output/cards_seed.csv with the same per-card numbers api.py's seed_history computes:
transaction count, total amount, last timestamp and the cardholder profile.
"""
import sys
from pathlib import Path

import pandas as pd

SRC = sys.argv[1] if len(sys.argv) > 1 else "advanced_credit_card_transactions.csv"
OUT = Path("model_output/cards_seed.csv")
PROFILE_KEYS = ["age", "city_pop", "lat", "long"]

d = pd.read_csv(SRC, usecols=["cc_num", "amt", "unix_time", *PROFILE_KEYS]).sort_values("unix_time", kind="stable")
g = d.groupby("cc_num")
agg = g.agg(n=("amt", "size"), sum_amt=("amt", "sum"), last_ts=("unix_time", "max"))
seed = agg.join(g[PROFILE_KEYS].last())

OUT.parent.mkdir(exist_ok=True)
seed.to_csv(OUT)  # cc_num is written as the first column
print(f"{len(seed):,} cards written to {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")
