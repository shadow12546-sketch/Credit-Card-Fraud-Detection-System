"""Streamlit dashboard for the fraud API.

    pip install streamlit requests pandas altair
    streamlit run Dashboard.py        (with the API already running on port 8000)

Needs Streamlit 1.26 or newer (for the compact layout toggle).
"""
import html
import json
import math
import os
import random
from datetime import datetime, time
from pathlib import Path

import inspect

import altair as alt
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Fraud scorer", page_icon="🛡️", layout="wide")

# Newer Streamlit replaced use_container_width with width="stretch"; support both.
STRETCH = {"width": "stretch"} if "width" in inspect.signature(st.button).parameters else {"use_container_width": True}

# ================================================================ theme
T = dict(
    bg="#0F141B", panel="#171E28", panel2="#1F2835", border="#2A3441", sidebar="#121821",
    text="#E6EAF0", muted="#93A0B2", safe="#3CCBA5", alert="#FF6B66", warn="#F2B84B",
    track="#2A3441", safe_bg="#12261F", alert_bg="#2A1819", warn_bg="#2A2210",
    btn="#5EC8B0", btn_text="#0D1117",
)
# The toggle value from the previous run is already in session_state when the script reruns.
compact = bool(st.session_state.get("compact", False))

FONT_IMPORT = "@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');"

BASE_CSS = """
.stApp, .stApp p, .stApp label, .stApp input, .stApp button, .stApp textarea, .stApp li,
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp td, .stApp th { font-family: 'Manrope', sans-serif !important; }
.block-container { padding-top: 2rem; max-width: 1280px; }
#MainMenu, footer { visibility: hidden; }
.stApp { background: var(--bg); color: var(--text); }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] { background: var(--sidebar); border-right: 1px solid var(--border); }

.stApp [data-testid="stMarkdownContainer"] p,
.stApp [data-testid="stMarkdownContainer"] h3,
.stApp [data-testid="stWidgetLabel"] p,
.stApp label { color: var(--text) !important; }
.stApp [data-testid="stCaptionContainer"],
.stApp [data-testid="stCaptionContainer"] p { color: var(--muted) !important; }

.app-title { font-size: 1.9rem; font-weight: 800; color: var(--text); letter-spacing: -0.02em; line-height: 1.2; }
.app-sub { color: var(--muted); margin: 0.2rem 0 0 0; }
.section { font-weight: 700; color: var(--text); font-size: 1.05rem; margin: 0 0 0.5rem 0; }

[data-testid="stForm"] { border: 1px solid var(--border); border-radius: 14px; padding: 1.2rem 1.3rem; background: var(--panel); }
[data-testid="stFormSubmitButton"], [data-testid="stFormSubmitButton"] > div { width: 100%; }
[data-testid="stFormSubmitButton"] button { width: 100%; background: var(--btn); border: 0; border-radius: 10px; padding: 0.65rem 0; }
[data-testid="stFormSubmitButton"] button p { color: var(--btn_text) !important; font-weight: 700; }
[data-testid="stFormSubmitButton"] button:hover { filter: brightness(1.12); }
.stButton button, [data-testid="stDownloadButton"] button {
    background: var(--panel); border: 1px solid var(--border); border-radius: 10px; }
.stButton button p, [data-testid="stDownloadButton"] button p { color: var(--text) !important; font-weight: 600; }
.stButton button:hover, [data-testid="stDownloadButton"] button:hover { border-color: var(--btn); }
.stApp button:focus-visible { outline: 3px solid var(--btn) !important; outline-offset: 2px; }

[data-testid="stExpander"] details { background: var(--panel); border: 1px solid var(--border); border-radius: 12px; }
[data-testid="stExpander"] summary p { color: var(--text) !important; font-weight: 600; }

button[data-baseweb="tab"] p { color: var(--muted) !important; font-weight: 600; }
button[data-baseweb="tab"][aria-selected="true"] p { color: var(--text) !important; }
[data-baseweb="tab-highlight"] { background: var(--btn) !important; }
[data-baseweb="tab-border"] { background: var(--border) !important; }

.tile { background: var(--panel); border: 1px solid var(--border); border-radius: 12px; padding: 0.75rem 1rem; }
.tile .k { color: var(--muted); font-size: 0.8rem; }
.tile .v { color: var(--text); font-size: 1.35rem; font-weight: 700; line-height: 1.3; }
.grid4 { display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.8rem; margin: 1.1rem 0 1.4rem 0; }
.grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 0.6rem; }
@media (max-width: 800px) { .grid4 { grid-template-columns: 1fr 1fr; } }

.panel { background: var(--panel); border: 1px solid var(--border); border-radius: 14px; padding: 1rem 1.2rem; margin-bottom: 1rem; }
.gauge { text-align: center; }
.gauge svg { display: block; margin: 0 auto; }
.gauge .cap { color: var(--muted); font-size: 0.85rem; margin-top: 0.2rem; }

.verdict { border-radius: 14px; padding: 1rem 1.2rem; border: 1px solid; display: flex; align-items: center;
    justify-content: space-between; gap: 1rem; flex-wrap: wrap; margin-bottom: 1rem; }
.verdict.alert { background: var(--alert_bg); border-color: color-mix(in srgb, var(--alert) 40%, transparent); }
.verdict.warn { background: var(--warn_bg); border-color: color-mix(in srgb, var(--warn) 40%, transparent); }
.verdict.safe { background: var(--safe_bg); border-color: color-mix(in srgb, var(--safe) 40%, transparent); }
.verdict .head { font-size: 1.2rem; font-weight: 800; }
.verdict.alert .head { color: var(--alert); }
.verdict.warn .head { color: var(--warn); }
.verdict.safe .head { color: var(--safe); }
.verdict .sub { color: var(--text); margin-top: 0.1rem; }
.pill { display: inline-block; padding: 0.25rem 0.8rem; border-radius: 999px; font-weight: 700; font-size: 0.85rem; color: var(--btn_text); }

.empty { border: 1.5px dashed var(--border); border-radius: 14px; padding: 3rem 1.5rem; text-align: center; color: var(--muted); }
.empty b { color: var(--text); font-size: 1.05rem; }
.note { background: var(--warn_bg); border: 1px solid color-mix(in srgb, var(--warn) 35%, transparent);
    color: var(--text); border-radius: 10px; padding: 0.55rem 0.8rem; margin-bottom: 0.5rem; font-size: 0.92rem; }

.dwrap { overflow-x: auto; border: 1px solid var(--border); border-radius: 12px; background: var(--panel); }
.dtable { width: 100%; border-collapse: collapse; font-size: 0.9rem; }
.dtable th { text-align: left; color: var(--muted); font-weight: 600; padding: 0.55rem 0.8rem; background: var(--panel2); border-bottom: 1px solid var(--border); }
.dtable td { padding: 0.5rem 0.8rem; color: var(--text); border-bottom: 1px solid var(--border); }
.dtable tr:last-child td { border-bottom: 0; }
.tag { font-weight: 700; }
.tag.ALERT { color: var(--alert); }
.tag.ok { color: var(--safe); }

.status { display: inline-flex; align-items: center; gap: 0.5rem; font-weight: 600; color: var(--text); margin-bottom: 0.8rem; }
.dot { width: 0.65rem; height: 0.65rem; border-radius: 50%; display: inline-block; }
.legend { color: var(--muted); font-size: 0.85rem; margin: 0.4rem 0; }
"""

DARK_CSS = """
div[data-baseweb="input"], div[data-baseweb="base-input"], div[data-baseweb="select"] > div, div[data-baseweb="textarea"] {
    background: var(--bg) !important; border-color: var(--border) !important; }
.stApp input, .stApp textarea { color: var(--text) !important; -webkit-text-fill-color: var(--text) !important; background: transparent !important; }
div[data-baseweb="select"] div, div[data-baseweb="select"] span { color: var(--text) !important; }
div[data-baseweb="select"] svg, [data-testid="stNumberInput"] svg { fill: var(--muted) !important; }
[data-testid="stNumberInput"] button { background: var(--panel2) !important; color: var(--text) !important; border-color: var(--border) !important; }
div[data-baseweb="popover"], div[data-baseweb="popover"] > div, ul[role="listbox"], div[data-baseweb="menu"] {
    background: var(--panel) !important; color: var(--text) !important; border-color: var(--border) !important; }
li[role="option"] { background: var(--panel) !important; color: var(--text) !important; }
li[role="option"]:hover, li[aria-selected="true"] { background: var(--panel2) !important; }
div[data-baseweb="calendar"], div[data-baseweb="calendar"] div, div[data-baseweb="calendar"] button {
    background: var(--panel) !important; color: var(--text) !important; }
[data-testid="stExpander"] summary svg { color: var(--text); fill: var(--text); }
[data-testid="stAlert"] { background: var(--panel2); }
"""

COMPACT_CSS = """
.block-container { padding-top: 0.8rem !important; }
.stApp [data-testid="stVerticalBlock"] { gap: 0.55rem; }
[data-testid="stForm"] { padding: 0.8rem 1rem; }
.grid4 { margin: 0.6rem 0 0.8rem 0; }
.tile { padding: 0.5rem 0.8rem; }
.tile .v { font-size: 1.1rem; }
.gauge svg { max-width: 190px !important; }
.panel { padding: 0.6rem 1rem; margin-bottom: 0.6rem; }
.app-title { font-size: 1.5rem; }
.verdict { padding: 0.7rem 1rem; margin-bottom: 0.6rem; }
"""

css_vars = ":root{" + ";".join(f"--{k}:{v}" for k, v in T.items()) + "}"
st.markdown(
    f"<style>{FONT_IMPORT}{css_vars}{BASE_CSS}{DARK_CSS}{COMPACT_CSS if compact else ''}</style>",
    unsafe_allow_html=True,
)


# ================================================================ helpers
def flat(s: str) -> str:
    """Collapse multi-line HTML so Markdown never treats indented lines as code."""
    return " ".join(line.strip() for line in s.splitlines() if line.strip())


def tile(label, value) -> str:
    return f'<div class="tile"><div class="k">{html.escape(str(label))}</div><div class="v">{html.escape(str(value))}</div></div>'


def fmt(v, suffix=""):
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:,.2f}{suffix}"
    return f"{v}{suffix}"


def html_table(df: pd.DataFrame) -> str:
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns)
    rows = []
    for _, r in df.iterrows():
        cells = []
        for c in df.columns:
            v = r[c]
            if c == "Decision":
                cells.append(f'<td><span class="tag {"ALERT" if v == "ALERT" else "ok"}">{html.escape(str(v))}</span></td>')
                continue
            if v is None or (isinstance(v, float) and math.isnan(v)):
                txt = "n/a"
            elif isinstance(v, float):
                txt = f"{v:,.4f}"
            else:
                txt = str(v)
            cells.append(f"<td>{html.escape(txt)}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return f'<div class="dwrap"><table class="dtable"><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'


def level_of(score, thr, decision):
    """Low / Medium / High. High follows the API's ALERT decision; Medium is 40%+ of the threshold."""
    if str(decision).upper() == "ALERT":
        return "High", "alert", T["alert"]
    if score >= 0.4 * thr:
        return "Medium", "warn", T["warn"]
    return "Low", "safe", T["safe"]


def gauge_svg(score, thr, color):
    length = math.pi * 80
    fill = max(0.0, min(score, 1.0)) * length
    ang = math.pi * (1 - max(0.0, min(thr, 1.0)))
    x1, y1 = 100 + 66 * math.cos(ang), 100 - 66 * math.sin(ang)
    x2, y2 = 100 + 94 * math.cos(ang), 100 - 94 * math.sin(ang)
    arc = "M20 100 A80 80 0 0 1 180 100"
    return flat(f"""
    <svg viewBox="0 0 200 120" role="img" aria-label="Risk score {score:.3f}, alert threshold {thr:.3f}" style="width:100%;max-width:260px">
      <path d="{arc}" fill="none" stroke-width="14" stroke-linecap="round" style="stroke:var(--track)"/>
      <path d="{arc}" fill="none" stroke-width="14" stroke-linecap="round" stroke-dasharray="{fill:.1f} {length + 5:.1f}" style="stroke:{color}"/>
      <line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke-width="3" stroke-linecap="round" style="stroke:var(--text)"/>
      <text x="100" y="88" text-anchor="middle" font-size="30" font-weight="800" style="fill:var(--text)">{score:.3f}</text>
      <text x="100" y="108" text-anchor="middle" font-size="11" style="fill:var(--muted)">risk score</text>
    </svg>""")


def style_chart(chart):
    return (
        chart.configure(background="transparent")
        .configure_view(stroke=None)
        .configure_axis(gridColor=T["border"], domainColor=T["border"], tickColor=T["border"],
                        labelColor=T["muted"], titleColor=T["muted"], labelFont="Manrope", titleFont="Manrope")
        .configure_legend(labelColor=T["text"], labelFont="Manrope", title=None)
    )


def shap_chart(reasons):
    df = pd.DataFrame(reasons)
    df["abs"] = df["contribution"].abs()
    df = df.sort_values("abs", ascending=False).head(8)
    df["direction"] = df["contribution"].apply(lambda v: "Toward fraud" if v > 0 else "Away from fraud")
    chart = (
        alt.Chart(df)
        .mark_bar(cornerRadiusEnd=4, size=18)
        .encode(
            x=alt.X("contribution:Q", title="Contribution (log-odds)"),
            y=alt.Y("feature:N", sort=list(df["feature"]), title=None),
            color=alt.Color("direction:N",
                            scale=alt.Scale(domain=["Toward fraud", "Away from fraud"], range=[T["alert"], T["safe"]]),
                            legend=alt.Legend(orient="bottom")),
            tooltip=["feature", alt.Tooltip("contribution:Q", format="+.3f")],
        )
        .properties(height=max(150, 34 * len(df)))
    )
    return style_chart(chart)


def trend_chart(df):
    thr = float(df["threshold"].iloc[-1])
    line = alt.Chart(df).mark_line(color=T["muted"]).encode(
        x=alt.X("n:O", title="Transaction number"),
        y=alt.Y("score:Q", title="Risk score", scale=alt.Scale(domain=[0, 1])),
    )
    pts = alt.Chart(df).mark_point(filled=True, size=80, opacity=1).encode(
        x="n:O", y="score:Q",
        color=alt.Color("level:N",
                        scale=alt.Scale(domain=["Low", "Medium", "High"], range=[T["safe"], T["warn"], T["alert"]]),
                        legend=alt.Legend(orient="bottom")),
        tooltip=["n", "time", "category", alt.Tooltip("score:Q", format=".3f"), "level"],
    )
    rule = alt.Chart(pd.DataFrame({"thr": [thr]})).mark_rule(strokeDash=[5, 4], color=T["text"]).encode(y="thr:Q")
    return style_chart((line + pts + rule).properties(height=220))


def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 6371 * 2 * math.asin(math.sqrt(a))


# ================================================================ sidebar
st.sidebar.markdown("### Settings")
def default_api_url():
    """Deployed apps set API_URL in Streamlit secrets (or an env var); locally it falls back to localhost."""
    try:
        if "API_URL" in st.secrets:
            return str(st.secrets["API_URL"]).rstrip("/")
    except Exception:  # no secrets file locally
        pass
    return os.environ.get("API_URL", "http://localhost:8000").rstrip("/")


api = st.sidebar.text_input("API URL", default_api_url()).rstrip("/")

# ================================================================ header
hl, hr = st.columns([3, 1])
with hl:
    st.markdown(
        flat("""<div class="app-title">Card fraud scorer</div>
        <div class="app-sub">Score a transaction and see which factors drove the result.</div>"""),
        unsafe_allow_html=True,
    )
with hr:
    st.toggle("Compact layout", key="compact")

summary_slot = st.empty()  # filled in after scoring so the tiles are never one step behind


@st.cache_data(ttl=60)
def get_meta(url):
    r = requests.get(f"{url}/meta", timeout=10)
    r.raise_for_status()
    return r.json()


try:
    meta = get_meta(api)
    st.sidebar.markdown(
        '<div class="status"><span class="dot" style="background:var(--safe)"></span>API connected</div>',
        unsafe_allow_html=True,
    )
except Exception as exc:
    st.sidebar.markdown(
        '<div class="status"><span class="dot" style="background:var(--alert)"></span>API offline</div>',
        unsafe_allow_html=True,
    )
    st.error(f"Can't reach the API at {api}. Start it with `uvicorn api:app --port 8000`, then reload.\n\n{exc}")
    st.stop()

metrics_file = Path("model_output/metrics.json")
if metrics_file.exists():
    st.sidebar.markdown("### Model performance")
    st.sidebar.caption("Test set, time-based holdout")
    try:
        metrics = json.loads(metrics_file.read_text())
    except Exception as exc:
        metrics = None
        st.sidebar.warning(f"Could not read metrics.json: {exc}")
    if isinstance(metrics, dict):
        flat_m = {k: v for k, v in metrics.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
        if flat_m:
            tiles_html = "".join(
                tile(k.replace("_", " "), f"{v:.3f}" if isinstance(v, float) else v) for k, v in flat_m.items()
            )
            st.sidebar.markdown(f'<div class="grid2">{tiles_html}</div>', unsafe_allow_html=True)
        rest = {k: v for k, v in metrics.items() if k not in flat_m}
        if rest:
            with st.sidebar.expander("More metrics"):
                st.json(rest)

cats = meta["categories"]
sample_cards = meta.get("sample_cards") or [""]

# ================================================================ state
defaults = {
    "f_card": sample_cards[0],
    "f_amount": 120.0,
    "f_category": cats["category"][0],
    "f_merchant": cats["merchant"][0],
    "f_state": cats["state"][0],
    "f_date": datetime(2013, 6, 22).date(),
    "f_time": time(23, 30),
    "f_loc": False,
    "f_lat": 36.0,
    "f_long": -82.0,
    "f_commit": False,
    "last": None,
    "last_body": None,
    "log": [],
    "found": None,
    "flt_level": [], "flt_decision": [], "flt_cat": [],
    "flt_score": (0.0, 1.0), "flt_amin": 0.0, "flt_amax": 0.0,
}
for k, v in defaults.items():
    st.session_state.setdefault(k, v)


def preset(amount, t, hint=None):
    st.session_state.f_amount = amount
    st.session_state.f_time = t
    if hint:  # pick the first category whose name contains the hint, if there is one
        for c in cats["category"]:
            if hint in c:
                st.session_state.f_category = c
                break


def load_case(case):
    ss = st.session_state
    ss.f_amount, ss.f_category, ss.f_merchant, ss.f_state = case["amount"], case["category"], case["merchant"], case["state"]
    ss.f_time = time(case["hour"], 30)


def search_examples(api_url, card, day, n=40):
    """Score random risky-looking combinations (never saved to history) and return the top 5."""
    rng = random.Random(7)
    risky = [c for c in cats["category"] if "net" in c or "pos" in c] or cats["category"]
    rows, thr = [], None
    bar = st.progress(0.0)
    for i in range(n):
        case = {
            "amount": rng.choice([250.0, 600.0, 900.0, 1400.0]),
            "hour": rng.choice([1, 2, 3, 22, 23]),
            "category": rng.choice(risky),
            "merchant": rng.choice(cats["merchant"]),
            "state": rng.choice(cats["state"]),
        }
        body = {"card_token": card, "amount": case["amount"], "category": case["category"],
                "merchant": case["merchant"], "state": case["state"],
                "timestamp": datetime.combine(day, time(case["hour"], 30)).isoformat(), "commit": False}
        r = requests.post(f"{api_url}/score", json=body, timeout=10)
        r.raise_for_status()
        res = r.json()
        thr = float(res["threshold"])
        rows.append({**case, "score": float(res["risk_score"]), "decision": res["decision"]})
        bar.progress((i + 1) / n)
    bar.empty()
    rows.sort(key=lambda x: -x["score"])
    return {"rows": rows[:5], "thr": thr, "tried": n}


def reset_filters():
    st.session_state.update(flt_level=[], flt_decision=[], flt_cat=[], flt_score=(0.0, 1.0), flt_amin=0.0, flt_amax=0.0)


def clear_log():
    st.session_state.log = []


# ================================================================ layout: input
left, right = st.columns([5, 6], gap="large")

with left:
    st.markdown('<div class="section">Quick scenarios</div>', unsafe_allow_html=True)
    b1, b2, b3 = st.columns(3)
    b1.button("Everyday", on_click=preset, args=(42.0, time(14, 10), "gas"), **STRETCH)
    b2.button("Late-night", on_click=preset, args=(480.0, time(2, 15), "misc_net"), **STRETCH)
    b3.button("Large amount", on_click=preset, args=(1450.0, time(23, 45), "shopping_net"), **STRETCH)

    with st.form("tx"):
        st.markdown('<div class="section">Transaction</div>', unsafe_allow_html=True)
        c1, c2 = st.columns([3, 2])
        c1.text_input("Card token", key="f_card")
        c2.number_input("Amount", min_value=0.01, step=10.0, key="f_amount")

        c3, c4 = st.columns(2)
        c3.selectbox("Category", cats["category"], key="f_category")
        c4.selectbox("State", cats["state"], key="f_state")
        st.selectbox("Merchant", cats["merchant"], key="f_merchant")

        c5, c6 = st.columns(2)
        c5.date_input("Date (UTC)", key="f_date")
        c6.time_input("Time (UTC)", key="f_time")

        with st.expander("Merchant location (optional)"):
            st.checkbox("Include location to calculate distance", key="f_loc")
            l1, l2 = st.columns(2)
            l1.number_input("Latitude", format="%.4f", key="f_lat")
            l2.number_input("Longitude", format="%.4f", key="f_long")

        st.checkbox("Save to this card's history after scoring", key="f_commit")
        go = st.form_submit_button("Score transaction")

    with st.expander("Find high-risk examples"):
        st.caption(
            "Tries 40 random late-night, higher-amount combinations against the API and lists the highest scores. "
            "Nothing is saved to card history. Use a known card token for the most realistic results."
        )
        if st.button("Search for examples", **STRETCH):
            try:
                with st.spinner("Scoring combinations..."):
                    st.session_state.found = search_examples(api, st.session_state.f_card, st.session_state.f_date)
            except Exception as exc:
                st.session_state.found = None
                st.error(f"Search failed: {exc}")
        fd = st.session_state.found
        if fd:
            hits = sum(1 for r in fd["rows"] if r["decision"] == "ALERT")
            st.caption(f"Top 5 of {fd['tried']} tried. {hits} reached the alert threshold ({fd['thr']:.3f}).")
            for i, r in enumerate(fd["rows"]):
                c1, c2 = st.columns([4, 1])
                c1.markdown(
                    f"**{r['score']:.3f}** &nbsp; amount {r['amount']:,.0f}, {r['category']}, "
                    f"{r['hour']:02d}:30, {r['state']}"
                )
                c2.button("Load", key=f"load_{i}", on_click=load_case, args=(r,), **STRETCH)

# ================================================================ scoring
score_error = None
if go:
    ss = st.session_state
    body = {
        "card_token": ss.f_card, "amount": ss.f_amount, "category": ss.f_category,
        "merchant": ss.f_merchant, "state": ss.f_state,
        "timestamp": datetime.combine(ss.f_date, ss.f_time).isoformat(), "commit": ss.f_commit,
    }
    if ss.f_loc:
        body.update(merch_lat=ss.f_lat, merch_long=ss.f_long)
    try:
        r = requests.post(f"{api}/score", json=body, timeout=15)
        r.raise_for_status()
        res = r.json()
        ss.last, ss.last_body = res, body
        level = level_of(float(res["risk_score"]), float(res["threshold"]), res["decision"])[0]
        ss.log.insert(0, {
            "n": len(ss.log) + 1,
            "time": datetime.now().strftime("%H:%M:%S"),
            "card": str(body["card_token"])[:12],
            "amount": body["amount"],
            "category": body["category"],
            "score": round(float(res["risk_score"]), 4),
            "threshold": float(res["threshold"]),
            "level": level,
            "decision": res["decision"],
        })
    except Exception as exc:
        ss.last, ss.last_body = None, None
        score_error = f"The scoring request failed: {exc}"

# ================================================================ summary tiles
log = st.session_state.log
n_scored = len(log)
n_alerts = sum(1 for e in log if e["decision"] == "ALERT")
avg_score = f"{sum(e['score'] for e in log) / n_scored:.3f}" if n_scored else "n/a"
max_score = f"{max(e['score'] for e in log):.3f}" if n_scored else "n/a"
summary_slot.markdown(
    f'<div class="grid4">{tile("Transactions scored", n_scored)}{tile("Alerts raised", n_alerts)}'
    f'{tile("Average risk score", avg_score)}{tile("Highest risk score", max_score)}</div>',
    unsafe_allow_html=True,
)

# ================================================================ results
with right:
    if score_error:
        st.error(score_error)

    res = st.session_state.last
    if res is None:
        st.markdown(
            '<div class="empty"><b>No result yet</b><br>'
            "Fill in the transaction and press Score, or start from a quick scenario.</div>",
            unsafe_allow_html=True,
        )
    else:
        score, thr = float(res["risk_score"]), float(res["threshold"])
        level, tone, color = level_of(score, thr, res["decision"])
        headline = {
            "High": "Alert: review this transaction",
            "Medium": "Below the alert threshold, but elevated",
            "Low": "Below the alert threshold",
        }[level]

        st.markdown(
            flat(f"""<div class="verdict {tone}">
              <div><div class="head">{headline}</div>
              <div class="sub">Risk score {score:.3f} against a threshold of {thr:.3f}.</div></div>
              <span class="pill" style="background:{color}">{level} risk</span></div>"""),
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="panel gauge">{gauge_svg(score, thr, color)}'
            f'<div class="cap">The dark line marks the alert threshold ({thr:.3f}).</div></div>',
            unsafe_allow_html=True,
        )

        tab_why, tab_hist, tab_map, tab_log = st.tabs(["Why this score", "Card history", "Map", "Session"])

        with tab_why:
            st.caption("Red bars push the score toward fraud. Green bars push it away.")
            st.altair_chart(shap_chart(res["reasons"]), **STRETCH, theme=None)
            with st.expander("Show values"):
                st.markdown(html_table(pd.DataFrame(res["reasons"])), unsafe_allow_html=True)

        with tab_hist:
            h = res["history"]
            tiles_html = "".join([
                tile("Prior transactions", fmt(h.get("prior_transactions"))),
                tile("Average amount before", fmt(h.get("card_avg_amt_before"))),
                tile("Amount vs average", fmt(h.get("amt_ratio_before"), "x")),
                tile("Seconds since last", fmt(h.get("secs_since_last_tx"))),
            ])
            st.markdown(f'<div class="grid2">{tiles_html}</div>', unsafe_allow_html=True)

        with tab_map:
            body = st.session_state.last_body or {}
            points = []
            if body.get("merch_lat") is not None and body.get("merch_long") is not None:
                points.append({"label": "Merchant", "lat": body["merch_lat"], "lon": body["merch_long"], "color": T["alert"]})
            # The home location is optional: it only shows if the API returns it.
            home = res.get("home") if isinstance(res.get("home"), dict) else {}
            hlat = res.get("home_lat", home.get("lat"))
            hlon = res.get("home_long", home.get("long", home.get("lon")))
            if hlat is not None and hlon is not None:
                points.append({"label": "Cardholder home", "lat": hlat, "lon": hlon, "color": "#3B82F6"})

            if points:
                pdf = pd.DataFrame(points)
                legend = " ".join(
                    f'<span class="dot" style="background:{p["color"]}"></span> {html.escape(p["label"])}&nbsp;&nbsp;'
                    for p in points
                )
                st.markdown(f'<div class="legend">{legend}</div>', unsafe_allow_html=True)
                try:
                    st.map(pdf, latitude="lat", longitude="lon", color="color", size=3000)
                except TypeError:  # older Streamlit without color/size
                    st.map(pdf, latitude="lat", longitude="lon")
                if len(points) == 2:
                    dist = haversine_km(points[0]["lat"], points[0]["lon"], points[1]["lat"], points[1]["lon"])
                    st.markdown(f'<div class="grid2">{tile("Distance, home to merchant", f"{dist:,.1f} km")}</div>',
                                unsafe_allow_html=True)
            else:
                st.caption(
                    "No location to show. Turn on merchant location in the form to place the merchant on the map. "
                    "The cardholder's home appears too if the API returns it as home_lat and home_long."
                )

        with tab_log:
            if log:
                all_df = pd.DataFrame(log)
                with st.expander("Filter the log"):
                    st.caption("Leave a list empty to include everything. The chart, table and CSV all use these filters.")
                    f1, f2, f3 = st.columns(3)
                    f1.multiselect("Level", ["Low", "Medium", "High"], key="flt_level")
                    f2.multiselect("Decision", sorted(all_df["decision"].unique()), key="flt_decision")
                    f3.multiselect("Category", sorted(all_df["category"].unique()), key="flt_cat")
                    st.slider("Risk score range", 0.0, 1.0, step=0.01, key="flt_score")
                    a1, a2 = st.columns(2)
                    a1.number_input("Min amount (0 = no minimum)", min_value=0.0, step=50.0, key="flt_amin")
                    a2.number_input("Max amount (0 = no maximum)", min_value=0.0, step=50.0, key="flt_amax")
                    st.button("Reset filters", on_click=reset_filters, **STRETCH)

                ss = st.session_state
                mask = all_df["score"].between(ss.flt_score[0], ss.flt_score[1])
                if ss.flt_level:
                    mask &= all_df["level"].isin(ss.flt_level)
                if ss.flt_decision:
                    mask &= all_df["decision"].isin(ss.flt_decision)
                if ss.flt_cat:
                    mask &= all_df["category"].isin(ss.flt_cat)
                if ss.flt_amin > 0:
                    mask &= all_df["amount"] >= ss.flt_amin
                if ss.flt_amax > 0:
                    mask &= all_df["amount"] <= ss.flt_amax
                ldf = all_df[mask].sort_values("n")

                st.caption(f"Showing {len(ldf)} of {len(all_df)} transactions.")
                if ldf.empty:
                    st.markdown('<div class="empty">No transactions match these filters.</div>', unsafe_allow_html=True)
                else:
                    st.altair_chart(trend_chart(ldf), **STRETCH, theme=None)
                    shown = ldf.sort_values("n", ascending=False).rename(columns={
                        "n": "#", "time": "Time", "card": "Card", "amount": "Amount", "category": "Category",
                        "score": "Risk score", "level": "Level", "decision": "Decision",
                    }).drop(columns=["threshold"])
                    st.markdown(html_table(shown), unsafe_allow_html=True)

                d1, d2 = st.columns(2)
                d1.download_button(
                    f"Download filtered CSV ({len(ldf)} rows)",
                    shown.to_csv(index=False).encode("utf-8") if not ldf.empty else b"",
                    file_name="scored_transactions_filtered.csv", mime="text/csv", disabled=ldf.empty, **STRETCH,
                )
                d2.button("Clear session", on_click=clear_log, **STRETCH)
            else:
                st.caption("Scored transactions will appear here.")

        warnings = res.get("warnings") or []
        if warnings:
            with st.expander(f"Data notes ({len(warnings)})"):
                st.markdown("".join(f'<div class="note">{html.escape(str(w))}</div>' for w in warnings),
                            unsafe_allow_html=True)

