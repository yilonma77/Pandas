"""
Smart Portfolio Manager  ·  Streamlit App
==========================================
End-to-end systematic portfolio construction, backtesting and live rebalancing.
Powered by sprv2.py
"""

import io
import os
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.show = lambda *a, **kw: None   # suppress blocking plt.show from sprv2

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

warnings.filterwarnings("ignore")

# ── Path resolution ───────────────────────────────────────────────────────────
# app.py is at  Pandas/streamlit/portfolio_manager/app.py
# sprv2.py  is at  Pandas/sprv2.py
ROOT   = Path(__file__).resolve().parents[2]   # → Pandas/
SP_CSV = str(ROOT / "csv" / "SPfull.csv")
sys.path.insert(0, str(ROOT))
import sprv2 as sp

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Portfolio Manager",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Background */
.stApp { background-color: #0d1117; }

/* Metric cards */
[data-testid="metric-container"] {
    background: linear-gradient(135deg, #161d2e 0%, #1a2540 100%);
    border: 1px solid #2a3a5c;
    border-radius: 14px;
    padding: 18px 22px !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.4);
}
[data-testid="metric-container"] label {
    color: #7d92b8 !important;
    font-size: 12px !important;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
[data-testid="stMetricValue"] { color: #e2eaff !important; font-size: 24px !important; font-weight: 700 !important; }
[data-testid="stMetricDelta"] { font-size: 13px !important; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1117 0%, #111827 100%);
    border-right: 1px solid #1e2d4a;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #161d2e;
    border-radius: 12px;
    padding: 6px 8px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    padding: 8px 16px;
    font-weight: 600;
    font-size: 13px;
    color: #7d92b8;
    background: transparent;
    border: none;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
    color: #fff !important;
    box-shadow: 0 3px 12px rgba(37,99,235,0.5);
}

/* Run button */
.stButton > button {
    background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 14px 0;
    font-weight: 700;
    font-size: 15px;
    width: 100%;
    box-shadow: 0 4px 20px rgba(37,99,235,0.45);
    transition: all 0.2s ease;
    letter-spacing: 0.03em;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 28px rgba(37,99,235,0.6);
}

/* Section headings */
h1 { color: #e2eaff !important; font-weight: 800 !important; }
h2 { color: #a8bcdb !important; font-weight: 700 !important; }
h3 { color: #7d92b8 !important; }

/* Card/panel look for markdown blocks */
.card {
    background: linear-gradient(135deg, #161d2e 0%, #1a2540 100%);
    border: 1px solid #2a3a5c;
    border-radius: 14px;
    padding: 20px 24px;
    margin-bottom: 16px;
}

/* DataFrames */
[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }

/* Toast / success colours */
div[data-testid="stToast"] { background: #0f2027 !important; border-left: 4px solid #22c55e; }
</style>
""", unsafe_allow_html=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HELPERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DARK_BG    = "#161d2e"
DARK_PAPER = "#0d1117"
COLORS     = px.colors.qualitative.Plotly

BENCHMARK_OPTIONS = {
    "S&P 500  (^GSPC)":        "^GSPC",
    "NASDAQ 100  (^NDX)":      "^NDX",
    "NASDAQ Composite  (^IXIC)": "^IXIC",
    "Dow Jones  (^DJI)":       "^DJI",
    "FTSE 100  (^FTSE)":       "^FTSE",
    "Euro Stoxx 50  (^STOXX50E)": "^STOXX50E",
    "Nikkei 225  (^N225)":     "^N225",
}

def _plotly_base(height=420):
    return dict(
        template="plotly_dark",
        paper_bgcolor=DARK_PAPER,
        plot_bgcolor=DARK_BG,
        height=height,
        margin=dict(l=20, r=20, t=55, b=20),
        font=dict(family="Inter, sans-serif"),
    )

def _df_fmt(df, fn):
    """Apply a format function to every cell of a DataFrame (pandas-version-safe)."""
    return df.apply(lambda col: col.map(fn))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CACHED COMPUTATION FUNCTIONS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@st.cache_data(show_spinner=False)
def _load_sp500(sp_csv, interval, period, sample_per_industry,
                exclude_industries, only_industries, elements_to_remove):
    return sp.load_universe(
        sp_csv=sp_csv, interval=interval, period=period,
        sample_per_industry=sample_per_industry,
        exclude_industries=list(exclude_industries),
        only_industries=list(only_industries),
        elements_to_remove=list(elements_to_remove),
    )


@st.cache_data(show_spinner=False)
def _load_custom(tickers_tuple, period, interval):
    import yfinance as yf
    raw = yf.download(list(tickers_tuple), period=period, interval=interval)
    raw.index = pd.to_datetime(raw.index)
    # flatten MultiIndex if single ticker was downloaded
    if isinstance(raw.columns, pd.MultiIndex):
        pass  # keep as is — clean_df expects MultiIndex
    raw = sp.clean_df(raw)
    raw = sp.trim_incomplete_period(raw, interval)

    mkt = yf.download("^GSPC", interval=interval).dropna()
    if isinstance(mkt.columns, pd.MultiIndex):
        mkt.columns = mkt.columns.droplevel(1)
    mkt["Chg"] = mkt["Close"].pct_change()

    tnx = yf.download("^TNX").dropna()
    if isinstance(tnx.columns, pd.MultiIndex):
        tnx.columns = tnx.columns.droplevel(1)
    ten_year = tnx.reindex(mkt.index)["Close"].div(100).ffill()

    mkt      = mkt.reindex(raw.index)
    df_return = raw["Close"].pct_change()
    actual_tickers = raw["Close"].columns.tolist()
    industries = pd.DataFrame({
        "Ticker": actual_tickers,
        "Industry": ["Custom Portfolio"] * len(actual_tickers),
    })
    return raw, mkt, ten_year, df_return, actual_tickers, industries


@st.cache_data(show_spinner=False)
def _compute_signals(_df, _market_df, tickers_tuple, nb_rolling):
    plt.close("all")
    out = sp.compute_signals(_df, _market_df, list(tickers_tuple), nb_rolling)
    plt.close("all")
    return out


@st.cache_data(show_spinner=False)
def _find_best_signal(_df_signal, tickers_tuple, period_to_start,
                      nombre_tickers, nb_rolling, best, criteria, test_size):
    plt.close("all")
    out = sp.find_best_signal(
        _df_signal, list(tickers_tuple), period_to_start,
        nombre_tickers, nb_rolling, best, criteria, test_size, trace=False,
    )
    plt.close("all")
    return out


@st.cache_data(show_spinner=False)
def _rebalancing_table(_df_signal, tickers_tuple, best_signal, period_to_start):
    return sp.create_rebalancing_table(
        _df_signal, list(tickers_tuple), best_signal, period_to_start
    )


@st.cache_data(show_spinner=False)
def _load_benchmark(ticker: str, interval: str, period: str) -> pd.Series:
    """Download a benchmark and return its periodic return Series."""
    import yfinance as yf
    raw = yf.download(ticker, interval=interval, period=period).dropna()
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.droplevel(1)
    return raw["Close"].pct_change()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SIDEBAR — CONFIGURATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

with st.sidebar:
    st.markdown("## 📈 Smart Portfolio")

    # ── Run button at the very top ────────────────────────────────────────────
    run_btn = st.button("🚀  Run Analysis", use_container_width=True, type="primary")

    # ── Show selected tickers after a successful run ──────────────────────────
    _tts_val = st.session_state.get("tickers_to_see")
    if st.session_state.get("ready") and _tts_val is not None and len(_tts_val) > 0:
        _tts_idx  = st.session_state.get("tts", 0)
        _selected = list(_tts_val)
        st.markdown(
            f'<div style="background:#0d2b1e;border:1px solid #10b981;border-radius:6px;'
            f'padding:7px 11px;margin:8px 0 4px 0">'
            f'<div style="color:#6ee7b7;font-size:0.78em;font-weight:700;margin-bottom:4px">'
            f'✅ {len(_selected)} TICKERS SELECTED — [{_tts_idx}] {sp.TTS_MAPPING[_tts_idx]}</div>'
            f'<div style="color:#a7f3d0;font-size:0.75em;line-height:1.6">'
            + "  ".join(f'<code style="background:#134e2a;border-radius:3px;padding:1px 4px">{t}</code>'
                        for t in sorted(_selected))
            + '</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── Data ──────────────────────────────────────────────────────────────────
    with st.expander("📅  Data", expanded=True):
        interval = st.selectbox(
            "Frequency", ["1mo", "1wk"], index=0,
            help="Monthly: faster & less noise.  Weekly: finer signal resolution."
        )
        period = st.selectbox("History", ["2Y", "3Y", "4Y", "5Y", "8Y", "10Y"], index=2)
        benchmark_opt = st.selectbox(
            "Benchmark", list(BENCHMARK_OPTIONS.keys()), index=0,
            help="Index used for performance comparison in all charts."
        )
    benchmark_ticker = BENCHMARK_OPTIONS[benchmark_opt]
    benchmark_label  = benchmark_opt.split("  ")[0]   # short name e.g. 'S&P 500'

    # ── Universe ──────────────────────────────────────────────────────────────
    with st.expander("🌍  Universe", expanded=True):
        mode = st.radio("Source", ["S&P 500 Sample", "Custom Tickers"], horizontal=True)

        try:
            sp_df          = pd.read_csv(SP_CSV, sep=";")
            all_industries = sorted(sp_df["Industry"].dropna().unique().tolist())
        except Exception:
            sp_df, all_industries = pd.DataFrame(), []

        custom_tickers     = []
        exclude_industries = []
        only_industries    = []
        elements_to_remove = []
        sample_per_industry = 0

        if mode == "S&P 500 Sample":
            sample_per_industry = st.slider("Tickers per industry", 1, 10, 3)
            only_industries     = st.multiselect("Only these industries (blank = all)", all_industries)
            exclude_industries  = st.multiselect(
                "Exclude industries", all_industries,
                default=[i for i in ["Energy", "Real Estate"] if i in all_industries]
            )
            remove_txt         = st.text_input("Remove specific tickers (comma-sep.)", "TSLA")
            elements_to_remove = [t.strip().upper() for t in remove_txt.split(",") if t.strip()]
        else:
            ticker_txt = st.text_area(
                "Tickers (one per line or comma-separated)",
                "AAPL\nMSFT\nNVDA\nGOOGL\nAMZN\nMETA\nTSM\nJPM\nV\nUNH",
                height=160,
            )
            custom_tickers = [t.strip().upper() for t in ticker_txt.replace(",", "\n").split("\n") if t.strip()]

    # ── Backtest window ───────────────────────────────────────────────────────
    with st.expander("🗓️  Backtest Window", expanded=False):
        period_to_start = st.text_input("Start date", "2022-01-01",
                                         help="Must be within the downloaded history. "
                                              "For 4Y of data this should be ≥ 2022.")
        nombre_tickers  = st.slider("Tickers per strategy", 5, 30, 10)
        default_roll    = 52 if interval == "1wk" else 12
        nb_rolling      = st.number_input("Rolling window (periods)", 4, 104, default_roll)
        default_cov     = 32 if interval == "1mo" else 8
        cov_ma_period   = st.number_input("Cov. window (periods)", 4, 200, default_cov)

    # ── Signal ────────────────────────────────────────────────────────────────
    with st.expander("📡  Signal", expanded=False):
        criteria         = st.selectbox("Selection criteria", ["sharpe", "return", "sd"])
        test_size        = st.slider("In-sample fraction", 0.3, 1.0, 0.55, 0.05)
        best_signal_opt  = st.selectbox(
            "Force strategy (optional)",
            ["default"] + list(range(17)), index=0,
            format_func=lambda x: x if x == "default" else f"[{x}] {sp.STRATEGIES[x]}"
        )

    # ── Optimisation ──────────────────────────────────────────────────────────
    with st.expander("⚖️  Optimisation", expanded=False):
        objective    = st.selectbox("Objective", ["sortino", "sharpe", "variance", "risk_parity", "treynor"])
        min_pos      = st.number_input("Min positions", 2, 15, 3)
        min_w        = st.number_input("Min weight", 0.01, 0.30, 0.047, 0.005, format="%.3f")
        max_w_ratio  = st.number_input("Max weight ratio", 1.0, 10.0, 2.86, 0.1, format="%.2f")
        corr_constr  = st.checkbox("Correlation constraint", True)
        max_corr     = st.slider("Max pair correlation", 0.0, 1.0, 0.40, 0.05) if corr_constr else 1.0

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN HEADER + TABS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

st.markdown("# 📈 Smart Portfolio Manager")
st.markdown("*Systematic portfolio construction — backtesting to live rebalancing*")
st.markdown("---")

tabs = st.tabs([
    "🏠  Dashboard",
    "🌍  Universe",
    "🔍  Ticker Selection",
    "📡  Signals",
    "⚖️  Optimisation",
    "📊  Performance",
    "🎯  Next Rebalancing",
])

# ── Session state defaults ────────────────────────────────────────────────────
_DEFAULTS = {
    "ready": False,
    "df": None, "market_df": None, "ten_year": None, "df_return": None,
    "tickers": None, "industries": None, "tickers_to_see": None,
    "tts": None, "list_tts": None, "df_signal": None,
    "best_signal_set": None, "rebalancing_table": None,
    "weights_df": None, "equal_w_df": None, "returns": None,
    "returns_benchmark": None, "max_w_val": None,
    "cfg_nb_rolling": None, "cfg_period_to_start": None,
    "benchmark_label": "S&P 500", "benchmark_ticker": "^GSPC",
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# RUN PIPELINE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if run_btn:
    st.session_state["ready"] = False

    # ── Step 1: Download data ─────────────────────────────────────────────────
    with st.spinner("📥  Downloading market data…"):
        try:
            if mode == "Custom Tickers":
                df, market_df, ten_year, df_return, tickers, industries = _load_custom(
                    tuple(custom_tickers), period, interval
                )
            else:
                df, market_df, ten_year, df_return, tickers, industries = _load_sp500(
                    SP_CSV, interval, period, sample_per_industry,
                    tuple(exclude_industries), tuple(only_industries), tuple(elements_to_remove),
                )
            st.session_state.update({
                "df": df, "market_df": market_df, "ten_year": ten_year,
                "df_return": df_return, "tickers": tickers, "industries": industries,
            })
            st.toast(f"✅  {df['Close'].shape[1]} tickers downloaded", icon="📥")
        except Exception as e:
            st.error(f"**Data download failed:** {e}")
            st.stop()

    df         = st.session_state["df"]
    market_df  = st.session_state["market_df"]
    ten_year   = st.session_state["ten_year"]
    df_return  = st.session_state["df_return"]
    industries = st.session_state["industries"]

    # ── Step 2: Ticker selection ──────────────────────────────────────────────
    with st.spinner("🔍  Selecting tickers…"):
        try:
            plt.close("all")
            tickers_to_see, tts, list_tts = sp.plot_ticker_configs(
                df=df, df_return=df_return, market_df=market_df,
                tickers_to_see=df["Close"].columns,
                nombre_tickers=nombre_tickers,
                period_to_start=period_to_start,
                nb_rolling=nb_rolling,
                list_tts="default",
            )
            plt.close("all")
            st.session_state.update({
                "tickers_to_see": tickers_to_see,
                "tts": tts,
                "list_tts": list_tts,
            })
            st.toast(f"✅  {len(tickers_to_see)} tickers selected — strategy: {sp.TTS_MAPPING[tts]}", icon="🔍")
        except Exception as e:
            st.error(f"**Ticker selection failed:** {e}")
            st.stop()

    tickers_to_see = st.session_state["tickers_to_see"]
    tts            = st.session_state["tts"]
    list_tts       = st.session_state["list_tts"]

    # ── Step 3: Signals ───────────────────────────────────────────────────────
    with st.spinner("📡  Computing technical indicators…"):
        try:
            df_signal = _compute_signals(df, market_df, tuple(tickers_to_see), nb_rolling)
            st.session_state["df_signal"] = df_signal
            st.toast("✅  Signals computed", icon="📡")
        except Exception as e:
            st.error(f"**Signal computation failed:** {e}")
            st.stop()

    df_signal = st.session_state["df_signal"]

    # ── Step 4: Best strategy ─────────────────────────────────────────────────
    with st.spinner("🏆  Evaluating 17 strategies…"):
        try:
            best_signal_set = _find_best_signal(
                df_signal, tuple(tickers_to_see), period_to_start,
                nombre_tickers, nb_rolling, best_signal_opt, criteria, test_size,
            )
            st.session_state["best_signal_set"] = best_signal_set
            st.toast(f"✅  Strategy [{best_signal_set[0]}] selected", icon="🏆")
        except Exception as e:
            st.error(f"**Strategy selection failed:** {e}")
            st.stop()

    best_signal_set = st.session_state["best_signal_set"]

    # Ensure buy-signal columns are always present on the session-state df_signal.
    # _find_best_signal may return a cached result without running the function body,
    # so assign_signals may never execute its in-place writes on the current object.
    df_signal = sp.assign_signals(df_signal, list(tickers_to_see), nombre_tickers)
    st.session_state["df_signal"] = df_signal

    # ── Step 5: Rebalancing table ─────────────────────────────────────────────
    with st.spinner("📋  Building rebalancing table…"):
        try:
            rebalancing_table = _rebalancing_table(
                df_signal, tuple(tickers_to_see), best_signal_set[0], period_to_start
            )
            st.session_state["rebalancing_table"] = rebalancing_table
        except Exception as e:
            st.error(f"**Rebalancing table failed:** {e}")
            st.stop()

    rebalancing_table = st.session_state["rebalancing_table"]

    # ── Step 6: Optimisation ──────────────────────────────────────────────────
    with st.spinner("⚖️  Optimising walk-forward weights — this may take 1–2 min…"):
        try:
            weights_df, equal_w_df = sp.run_optimization(
                df=df, df_return=df_return, market_df=market_df,
                rebalancing_table=rebalancing_table,
                tickers_to_see=tickers_to_see,
                period_to_start=period_to_start,
                min_pos=min_pos, min_w=min_w, max_w_ratio=max_w_ratio,
                cov_ma_period=cov_ma_period, nb_rolling=nb_rolling,
                objective=objective, corr_constraint=corr_constr, max_corr=max_corr,
            )
            returns = (df_return.loc[period_to_start:].shift(-1) * weights_df).sum(axis=1)
            # Download benchmark fresh and align to rebalancing dates
            # (avoids NaN from the reindexed market_df)
            bm_raw  = _load_benchmark(benchmark_ticker, interval, period)
            returns_benchmark = (
                bm_raw
                .reindex(returns.index, method="ffill")
                .fillna(0)
            )
            n_pos     = rebalancing_table[period_to_start:].count(axis=1).replace(0, np.nan).mean()
            max_w_val = sp._max_w_bound(int(n_pos), max_w_ratio)
            st.session_state.update({
                "weights_df": weights_df, "equal_w_df": equal_w_df,
                "returns": returns, "returns_benchmark": returns_benchmark,
                "max_w_val": max_w_val,
                "cfg_nb_rolling": nb_rolling,
                "cfg_period_to_start": period_to_start,
                "benchmark_label": benchmark_label,
                "benchmark_ticker": benchmark_ticker,
            })
            st.toast("✅  Optimisation complete", icon="⚖️")
        except Exception as e:
            st.error(f"**Optimisation failed:** {e}")
            st.stop()

    st.session_state["ready"] = True
    st.rerun()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# WELCOME SCREEN (no results yet)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if not st.session_state["ready"]:
    with tabs[0]:
        st.markdown("""
        <div class="card">
        <h3>👋 Welcome — configure your portfolio in the sidebar then click <em>🚀 Run Analysis</em></h3>
        </div>
        """, unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Strategies evaluated", "17", "technical signals")
        c2.metric("Optimisation objectives", "5", "Sortino · Sharpe · Variance…")
        c3.metric("Walk-forward periods", "—", "every rebalancing date")
        c4.metric("Live instructions", "✓", "BUY / SELL / HOLD / ADJUST")

        st.markdown("---")
        st.markdown("""
        #### What this tool does, step by step
        | Step | Description |
        |------|-------------|
        | 1 | Downloads OHLCV for your universe + S&P 500 benchmark + 10y yield |
        | 2 | Compares **10 ticker-selection strategies** (correlation, vol, volume, price, beta) |
        | 3 | Computes **17 technical signal strategies** (BB, RSI, MA, MACD, ZL-MACD, beta…) |
        | 4 | Selects the best strategy by in-sample Sharpe / return / volatility |
        | 5 | Runs **walk-forward portfolio optimisation** (Sortino / Sharpe / Variance / Risk Parity) |
        | 6 | Shows **precise rebalancing instructions** for the next period |
        """)
    for t in tabs[1:]:
        with t:
            st.info("Run the analysis first (sidebar → 🚀 Run Analysis).")
    st.stop()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PULL FROM SESSION STATE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

df                = st.session_state["df"]
market_df         = st.session_state["market_df"]
ten_year          = st.session_state["ten_year"]
df_return         = st.session_state["df_return"]
industries        = st.session_state["industries"]
tickers_to_see    = st.session_state["tickers_to_see"]
tts               = st.session_state["tts"]
list_tts          = st.session_state["list_tts"]
df_signal         = st.session_state["df_signal"]
best_signal_set   = st.session_state["best_signal_set"]
rebalancing_table = st.session_state["rebalancing_table"]
weights_df        = st.session_state["weights_df"]
equal_w_df        = st.session_state["equal_w_df"]
returns           = st.session_state["returns"]
returns_benchmark = st.session_state["returns_benchmark"]
max_w_val         = st.session_state["max_w_val"]
nb_rolling        = st.session_state.get("cfg_nb_rolling") or nb_rolling
period_to_start   = st.session_state.get("cfg_period_to_start") or period_to_start
benchmark_label   = st.session_state.get("benchmark_label") or "S&P 500"
benchmark_ticker  = st.session_state.get("benchmark_ticker") or "^GSPC"

# ── Pre-compute summary stats ─────────────────────────────────────────────────
cum_reb  = (1 + returns).prod()
cum_mkt  = (1 + returns_benchmark).prod()
cum_eqw  = (1 + (equal_w_df * df_return[period_to_start:].shift(-1)).sum(axis=1)).prod()
sharpe   = (returns.mean() / returns.std() * np.sqrt(nb_rolling)) if returns.std() > 0 else 0
cum_arr  = np.cumprod(1 + returns.values)
peaks    = np.maximum.accumulate(cum_arr)
max_dd   = ((cum_arr - peaks) / peaks).min()
avg_pos  = rebalancing_table[period_to_start:].count(axis=1).mean()

# Safe anchor point: first row of data at or just before period_to_start;
# fall back to the first available row when  period_to_start pre-dates the data.
_before  = df_return.loc[:period_to_start]
prev_idx = _before.iloc[-1].name if len(_before) > 0 else df_return.index[0]
reb_c = pd.concat([pd.Series({prev_idx: 1}),
                   (1 + (df_return.loc[period_to_start:].shift(-1) * weights_df).sum(axis=1)).cumprod()])
mkt_c = pd.concat([pd.Series({prev_idx: 1}), (returns_benchmark + 1).cumprod()])
eqw_c = pd.concat([pd.Series({prev_idx: 1}),
                   (1 + (equal_w_df * df_return[period_to_start:].shift(-1)).sum(axis=1)).cumprod()])
ini_c = pd.concat([pd.Series({prev_idx: 1}),
                   (1 + (1 / df_return.shape[1]) * df_return.loc[period_to_start:].shift(-1).sum(axis=1)).cumprod()])

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 0 — DASHBOARD
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

with tabs[0]:
    st.markdown("## Summary Dashboard")

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Portfolio Return",   f"{cum_reb:.2f}×", f"{cum_reb - cum_mkt:+.2f}× vs mkt")
    c2.metric("Benchmark Return",   f"{cum_mkt:.2f}×")
    c3.metric("Equal-Weight Return",f"{cum_eqw:.2f}×")
    c4.metric("Annualised Sharpe",  f"{sharpe:.2f}")
    c5.metric("Max Drawdown",       f"{max_dd:.1%}")
    c6.metric("Avg Positions",      f"{avg_pos:.1f}",   f"strategy [{best_signal_set[0]}]")

    st.markdown("---")

    col_chart, col_alloc = st.columns([3, 1])

    with col_chart:
        fig = go.Figure()
        for name, series, color, dash, width in [
            ("Rebalanced",       reb_c, "#3b82f6", "solid",  3),
            ("EW + Signal",      eqw_c, "#f59e0b", "dot",    1.8),
            ("Equal Weight",     ini_c, "#f97316", "dot",    1.5),
            (benchmark_label,    mkt_c, "#6b7280", "dash",   1.8),
        ]:
            fig.add_trace(go.Scatter(
                x=series.index, y=series.values, name=name,
                line=dict(color=color, width=width, dash=dash),
                hovertemplate="%{x|%Y-%m-%d}<br>%{y:.3f}×<extra>" + name + "</extra>",
            ))
        fig.add_hline(y=1, line_dash="dot", line_color="rgba(255,255,255,0.25)", line_width=1)
        fig.update_layout(
            **_plotly_base(380),
            title="Cumulative Return",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_alloc:
        st.markdown("### Current Allocation")
        curr_w = weights_df.iloc[-1]
        curr_w = curr_w[curr_w > 0.001].sort_values(ascending=False)
        fig_pie = go.Figure(go.Pie(
            labels=curr_w.index.tolist(),
            values=curr_w.values,
            hole=0.48,
            textinfo="label+percent",
            textfont=dict(size=12),
            marker=dict(colors=COLORS, line=dict(color=DARK_PAPER, width=2)),
        ))
        fig_pie.update_layout(
            **_plotly_base(300),
            title=f"As of {weights_df.index[-1].date()}",
            showlegend=False,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

        for t, w in curr_w.items():
            avg = weights_df[t].mean()
            color = "#22c55e" if w >= avg else "#f59e0b"
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;padding:5px 0;"
                f"border-bottom:1px solid #1e2d4a'>"
                f"<span style='color:#c9d8f0;font-size:14px;font-weight:600'>{t}</span>"
                f"<span style='color:{color};font-weight:700;font-size:14px'>{w:.1%}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 1 — UNIVERSE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

with tabs[1]:
    st.markdown("## Universe & Data")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Tickers",  df["Close"].shape[1])
    c2.metric("Industries",     industries["Industry"].nunique())
    c3.metric("Start",          str(df.index[0].date()))
    c4.metric("End",            str(df.index[-1].date()))

    col_l, col_r = st.columns(2)

    with col_l:
        univ_tickers = df["Close"].columns.intersection(industries["Ticker"])
        ind_counts = (
            industries[industries["Ticker"].isin(univ_tickers)]["Industry"]
            .value_counts().reset_index()
        )
        ind_counts.columns = ["Industry", "Count"]
        fig = px.bar(
            ind_counts.sort_values("Count"), x="Count", y="Industry",
            orientation="h", color="Count", color_continuous_scale="Blues",
            title="Tickers per Industry",
        )
        fig.update_layout(**_plotly_base(420), showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        fig2 = px.pie(
            ind_counts, values="Count", names="Industry", hole=0.38,
            title="Industry Distribution",
            color_discrete_sequence=px.colors.qualitative.Dark24,
        )
        fig2.update_layout(**_plotly_base(420))
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("#### Correlation Heatmap — Selected Universe")
    corr = df["Close"][list(tickers_to_see)].pct_change().corr()
    fig3 = px.imshow(
        corr, text_auto=".2f", color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1, title="Return Correlation Matrix",
    )
    fig3.update_layout(**_plotly_base(460))
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("#### Cumulative Return per Industry")
    ind_map  = dict(zip(industries["Ticker"], industries["Industry"]))
    sel_cols = [t for t in tickers_to_see if t in df_return.columns]
    cumrets  = (1 + df_return[sel_cols].loc[period_to_start:]).cumprod()
    cumrets.columns = [f"{c} ({ind_map.get(c,'?')})" for c in cumrets.columns]
    fig4 = go.Figure()
    for col in cumrets.columns:
        fig4.add_trace(go.Scatter(x=cumrets.index, y=cumrets[col], name=col,
                                  mode="lines", line=dict(width=1.5)))
    fig4.add_hline(y=1, line_dash="dot", line_color="rgba(255,255,255,0.3)")
    fig4.update_layout(**_plotly_base(360), title="Cumulative return per ticker",
                       legend=dict(orientation="h", y=-0.25, font=dict(size=10)))
    st.plotly_chart(fig4, use_container_width=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 2 — TICKER SELECTION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

with tabs[2]:
    st.markdown("## Ticker Selection Strategies")
    st.success(
        f"**Selected:** `{sp.TTS_MAPPING[tts]}`  —  {len(tickers_to_see)} tickers: "
        f"{', '.join(tickers_to_see)}"
    )

    fig = make_subplots(
        rows=5, cols=2,
        shared_xaxes=False,
        subplot_titles=[f"[{i}] {sp.TTS_MAPPING[i]}" for i in range(10)],
        vertical_spacing=0.055,
        horizontal_spacing=0.08,
    )
    for num, tlist in enumerate(list_tts):
        row, col = divmod(num, 2)
        ret      = df_return.loc[:, tlist]
        cum      = ((1 / ret.shape[1]) * ret.loc[period_to_start:]).sum(axis=1).add(1).cumprod()
        cum_full = pd.concat([pd.Series({prev_idx: 1}), cum])
        selected = num == tts
        clr      = "#3b82f6" if selected else "#475569"
        lw       = 2.5 if selected else 1.2

        fig.add_trace(go.Scatter(
            x=cum_full.index, y=cum_full.values,
            name=sp.TTS_MAPPING[num],
            line=dict(color=clr, width=lw),
            showlegend=False,
            hovertemplate="%{x|%Y-%m-%d}  %{y:.3f}×<extra>" + sp.TTS_MAPPING[num] + "</extra>",
        ), row=row + 1, col=col + 1)

        fig.add_trace(go.Scatter(
            x=mkt_c.index, y=mkt_c.values,
            line=dict(color="#ef4444", width=1, dash="dash"),
            showlegend=False, hoverinfo="skip",
        ), row=row + 1, col=col + 1)

        fig.add_annotation(
            x=cum_full.index[-1], y=cum_full.iloc[-1],
            text=f"<b>{cum_full.iloc[-1]:.2f}×</b>",
            showarrow=False, row=row + 1, col=col + 1,
            font=dict(color=clr, size=11),
            xanchor="right",
        )

    fig.update_layout(
        **_plotly_base(950),
        title=f"10 Ticker-Selection Strategies  (red dashed = {benchmark_label})",
    )
    st.plotly_chart(fig, use_container_width=True)

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("#### Selected Tickers")
        sel_detail = industries[industries["Ticker"].isin(list(tickers_to_see))][["Ticker","Industry"]]
        st.dataframe(sel_detail.set_index("Ticker"), use_container_width=True)
    with col_r:
        st.markdown("#### Rolling Volatility Ranking")
        vol_rank = (
            df["Close"][list(tickers_to_see)].pct_change().fillna(0)
            .rolling(nb_rolling).var()
            .resample("YE").last()
            .dropna(axis=0)
            .rank(axis=1, ascending=False)
            .T.sort_index()
        )
        if not vol_rank.empty:
            fig_vr = px.imshow(
                vol_rank, text_auto=True, color_continuous_scale="plasma",
                title="Annual Rolling Volatility Rank (1 = highest vol)",
            )
            fig_vr.update_layout(**_plotly_base(300), coloraxis_showscale=False)
            fig_vr.update_xaxes(tickvals=vol_rank.columns, ticktext=vol_rank.columns.year)
            st.plotly_chart(fig_vr, use_container_width=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 3 — SIGNALS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

with tabs[3]:
    st.markdown("## Signal & Strategy Evaluation")
    st.success(f"**Best Strategy:** [{best_signal_set[0]}] — {sp.STRATEGIES[best_signal_set[0]]}")

    # Build recap for all strategies
    recap_rows = []
    for strat in sp.STRATEGIES:
        try:
            tx = sp._build_tx_df(df_signal, strat, period_to_start)
            r  = tx.loc[period_to_start:, "Cumprod"][list(tickers_to_see)].iloc[-2].mean()
            v  = tx.loc[period_to_start:, "Return"][list(tickers_to_see)].std().mean()
        except Exception:
            r, v = 1.0, 0.01
        recap_rows.append({"id": strat, "Strategy": sp.STRATEGIES[strat], "Return": r, "Vol": v})
    recap_df = pd.DataFrame(recap_rows)
    recap_df["Sharpe"] = (recap_df["Return"] - 1) / recap_df["Vol"].replace(0, np.nan)

    col_l, col_r = st.columns([3, 2])
    with col_l:
        best_row = recap_df[recap_df["id"] == best_signal_set[0]]
        fig = px.scatter(
            recap_df, x="Vol", y="Return",
            color="Sharpe", size="Return",
            color_continuous_scale="RdYlGn",
            hover_data=["Strategy"],
            labels={"Vol": "Avg Volatility", "Return": "Avg Cumulative Return"},
            title="Strategy Risk-Return Map",
        )
        fig.add_trace(go.Scatter(
            x=best_row["Vol"], y=best_row["Return"], mode="markers+text",
            marker=dict(color="#3b82f6", size=18, symbol="star",
                        line=dict(color="white", width=1.5)),
            text=[f"[{best_signal_set[0]}]"], textposition="top center",
            name="Selected", showlegend=True,
        ))
        for _, row in recap_df.iterrows():
            fig.add_annotation(
                x=row["Vol"], y=row["Return"] * 1.012,
                text=str(int(row["id"])), showarrow=False,
                font=dict(size=10, color="#94a3b8"),
            )
        fig.update_layout(**_plotly_base(440))
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown("#### Strategy Ranking")
        disp = recap_df[["id","Strategy","Return","Vol","Sharpe"]].sort_values("Sharpe", ascending=False).copy()
        disp["Return"] = disp["Return"].map("{:.2f}×".format)
        disp["Vol"]    = disp["Vol"].map("{:.2%}".format)
        disp["Sharpe"] = disp["Sharpe"].map("{:.2f}".format)
        disp["id"]     = disp["id"].map(lambda x: f"[{x}]")
        st.dataframe(disp.set_index("id"), use_container_width=True, height=440)

    st.markdown("#### Rebalancing Table — last 10 periods")
    rt_disp = rebalancing_table.tail(10).fillna("—")
    st.dataframe(rt_disp, use_container_width=True)

    buf_rt = io.BytesIO()
    rebalancing_table.fillna("").to_csv(buf_rt, index=True)
    st.download_button("⬇️  Download full rebalancing table (CSV)",
                       buf_rt.getvalue(), "rebalancing_table.csv", "text/csv")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 4 — OPTIMISATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

with tabs[4]:
    st.markdown("## Portfolio Optimisation")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Optimisation Objective", objective.capitalize())
    c2.metric("Periods Optimised", len(weights_df))
    c3.metric("Avg Positions",     f"{avg_pos:.1f}")
    c4.metric("Max Weight Cap",    f"{max_w_val:.1%}")

    # Stacked area — weight allocation over time
    active_w = weights_df[weights_df.sum(axis=1) > 0].copy()
    fig = go.Figure()
    for i, ticker in enumerate(active_w.columns):
        fig.add_trace(go.Scatter(
            x=active_w.index, y=active_w[ticker],
            name=ticker, stackgroup="one", mode="lines",
            line=dict(width=0),
            fillcolor=COLORS[i % len(COLORS)],
            hovertemplate=ticker + "  %{y:.1%}<extra></extra>",
        ))
    fig.update_layout(
        **_plotly_base(400),
        title="Weight Allocation Over Time (stacked)",
        yaxis_tickformat=".0%",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(size=11)),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Line chart: individual weight trajectories
    col_l, col_r = st.columns([2, 1])
    with col_l:
        fig2 = go.Figure()
        for i, ticker in enumerate(active_w.columns):
            fig2.add_trace(go.Scatter(
                x=active_w.index, y=active_w[ticker],
                name=ticker, mode="lines",
                line=dict(color=COLORS[i % len(COLORS)], width=2),
                hovertemplate=ticker + "  %{y:.1%}<extra></extra>",
            ))
        fig2.update_layout(
            **_plotly_base(360),
            title="Individual Weight Trajectories",
            yaxis_tickformat=".1%",
            legend=dict(orientation="h", y=-0.25, font=dict(size=11)),
        )
        st.plotly_chart(fig2, use_container_width=True)

    with col_r:
        st.markdown("#### Latest 3 Periods")
        for i in range(-3, 0):
            w = weights_df.iloc[i]
            w = w[w > 0.001].sort_values(ascending=False)
            date_lbl = weights_df.index[i].strftime("%Y-%m-%d")
            fp = go.Figure(go.Pie(
                labels=w.index.tolist(), values=w.values, hole=0.42,
                textinfo="label+percent", textfont=dict(size=10),
                marker=dict(colors=COLORS, line=dict(color=DARK_PAPER, width=1.5)),
            ))
            fp.update_layout(**_plotly_base(220),
                             title=date_lbl, showlegend=False)
            st.plotly_chart(fp, use_container_width=True)

    st.markdown("#### Weight History — last 8 periods")
    wt_disp = weights_df.iloc[-8:].copy()
    wt_fmt = wt_disp.apply(lambda col: col.map(lambda x: f"{x:.1%}" if x > 0.001 else "—"))
    st.dataframe(wt_fmt, use_container_width=True)

    buf_w = io.BytesIO()
    weights_df.to_csv(buf_w, index=True)
    st.download_button("⬇️  Download full weight history (CSV)",
                       buf_w.getvalue(), "weights_history.csv", "text/csv")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 5 — PERFORMANCE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

with tabs[5]:
    st.markdown("## Performance Analysis")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Portfolio Return",    f"{cum_reb:.2f}×")
    c2.metric("vs Benchmark",        f"{(cum_reb / cum_mkt - 1):+.1%}")
    c3.metric("Ann. Sharpe",         f"{sharpe:.2f}")
    c4.metric("Max Drawdown",        f"{max_dd:.1%}")
    c5.metric(f"Periods beating {benchmark_label}",  f"{(returns > returns_benchmark).mean():.0%}")

    # ── Cumulative return + drawdown chart ────────────────────────────────────
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.65, 0.35],
        subplot_titles=["Cumulative Return", "Portfolio Drawdown"],
        vertical_spacing=0.06,
    )
    for name, series, color, dash, lw in [
        ("Rebalanced",      reb_c, "#3b82f6", "solid", 3),
        ("EW + Signal",     eqw_c, "#f59e0b", "dot",   1.8),
        ("Equal Weight",    ini_c, "#f97316", "dot",   1.5),
        (benchmark_label,   mkt_c, "#6b7280", "dash",  1.8),
    ]:
        fig.add_trace(go.Scatter(
            x=series.index, y=series.values, name=name,
            line=dict(color=color, width=lw, dash=dash),
            hovertemplate="%{x|%Y-%m-%d}  %{y:.3f}×<extra>" + name + "</extra>",
        ), row=1, col=1)

    dd_series = (cum_arr - peaks) / peaks
    fig.add_trace(go.Scatter(
        x=returns.index, y=dd_series,
        fill="tozeroy", fillcolor="rgba(239,68,68,0.18)",
        line=dict(color="#ef4444", width=1.2),
        name="Drawdown",
    ), row=2, col=1)

    fig.add_hline(y=1, line_dash="dot",
                  line_color="rgba(255,255,255,0.25)", row=1, col=1)
    fig.update_yaxes(tickformat=".0%", row=2, col=1)
    fig.update_layout(
        **_plotly_base(560),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Per-share scatter ─────────────────────────────────────────────────────
    st.markdown("#### Per-Share Return vs Volatility")
    ret_ps  = (weights_df.fillna(0) * df_return.loc[:, list(tickers_to_see)].shift(-1)).dropna()
    shares  = pd.DataFrame({
        "Cumulative Return": (ret_ps + 1).prod(),
        "Annualised Vol":    ret_ps.std() * np.sqrt(nb_rolling),
    })
    shares  = shares[shares["Annualised Vol"] > 0].copy()
    shares["Sharpe"]   = (shares["Cumulative Return"] - 1) / shares["Annualised Vol"]
    shares["Industry"] = shares.index.map(lambda t: ind_map.get(t, "Custom"))  # type: ignore[name-defined]

    fig2 = px.scatter(
        shares.reset_index().rename(columns={"index": "Ticker"}),
        x="Annualised Vol", y="Cumulative Return", text="Ticker",
        color="Industry", size=shares["Sharpe"].clip(lower=0.01).values.tolist(),
        color_discrete_sequence=px.colors.qualitative.Dark24,
        title="Per-Share Contribution (bubble size = Sharpe)",
    )
    fig2.update_traces(textposition="top center", textfont=dict(size=11))
    fig2.add_hline(y=1, line_dash="dash",
                   line_color="rgba(255,255,255,0.25)", annotation_text="breakeven")
    fig2.update_layout(**_plotly_base(440))
    st.plotly_chart(fig2, use_container_width=True)

    # ── CAPM ─────────────────────────────────────────────────────────────────
    try:
        import statsmodels.api as sm  # local import to avoid error if not installed
        rf    = ten_year.reindex(returns.index).ffill().fillna(0)
        rf_b  = ten_year.reindex(returns_benchmark.index).ffill().fillna(0)
        exc   = (returns - rf).dropna()
        mkt_p = (returns_benchmark - rf_b).dropna()
        common = exc.index.intersection(mkt_p.index)
        capm   = sm.OLS(exc.loc[common], sm.add_constant(mkt_p.loc[common])).fit()
        alpha_v, beta_v = capm.params.values
        pval   = capm.pvalues.iloc[0]
        sig    = " ⭐ significant" if pval < 0.05 else ""

        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.markdown("#### CAPM Attribution")
            st.markdown(f"""
| Metric | Value |
|--------|-------|
| Jensen's α | **{alpha_v:.4f}**{sig} (p = {pval:.3f}) |
| Market β | **{beta_v:.3f}** |
| R² | **{capm.rsquared:.2%}** |
| Observations | **{len(common)}** |
""")
        with col_c2:
            fig_capm = px.scatter(
                x=mkt_p.loc[common].values, y=exc.loc[common].values,
                labels={"x": "Market excess return", "y": "Portfolio excess return"},
                title="CAPM Scatter",
                trendline="ols",
                trendline_color_override="#3b82f6",
            )
            fig_capm.update_layout(**_plotly_base(300))
            st.plotly_chart(fig_capm, use_container_width=True)
    except Exception:
        pass

    # ── Rolling Sharpe ────────────────────────────────────────────────────────
    st.markdown("#### Rolling Sharpe Ratio")
    roll_win = max(nb_rolling, 8)
    rs = (returns.rolling(roll_win).mean() / returns.rolling(roll_win).std()) * np.sqrt(nb_rolling)
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=rs.index, y=rs.values, name="Rolling Sharpe",
        fill="tozeroy",
        fillcolor="rgba(59,130,246,0.15)",
        line=dict(color="#3b82f6", width=2),
    ))
    fig3.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.3)")
    fig3.update_layout(**_plotly_base(280), title=f"Rolling Sharpe ({roll_win}-period window)")
    st.plotly_chart(fig3, use_container_width=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 6 — NEXT REBALANCING  (THE ACTIONABLE PAGE)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

with tabs[6]:
    st.markdown("## 🎯 Next Rebalancing Instructions")

    next_date = weights_df.index[-1]
    curr_w    = weights_df.iloc[-1]
    curr_w    = curr_w[curr_w > 0.001].sort_values(ascending=False)
    prev_w    = weights_df.iloc[-2] if len(weights_df) > 1 else pd.Series(dtype=float)
    prev_w    = prev_w[prev_w > 0.001]

    st.markdown(
        f"<div class='card'>"
        f"<b>Signal period ending:</b>  <code>{next_date.date()}</code>  ·  "
        f"<b>Strategy:</b>  [{best_signal_set[0]}] {sp.STRATEGIES[best_signal_set[0]]}  ·  "
        f"<b>Objective:</b>  {objective.capitalize()}"
        f"</div>",
        unsafe_allow_html=True,
    )

    # Classify changes
    ACTION_STYLE = {
        "BUY":      ("🟢", "#065f46", "#34d399"),
        "SELL":     ("🔴", "#7f1d1d", "#f87171"),
        "INCREASE": ("🔵", "#1e3a5f", "#93c5fd"),
        "REDUCE":   ("🟡", "#78350f", "#fcd34d"),
        "HOLD":     ("⚪", "#1e2d3a", "#94a3b8"),
    }
    changes = []
    for t in sorted(set(curr_w.index) | set(prev_w.index)):
        cw = curr_w.get(t, 0.0)
        pw = prev_w.get(t, 0.0)
        if pw == 0 and cw > 0:
            action = "BUY"
        elif cw == 0 and pw > 0:
            action = "SELL"
        elif abs(cw - pw) < 0.005:
            action = "HOLD"
        elif cw > pw:
            action = "INCREASE"
        else:
            action = "REDUCE"
        changes.append({
            "Ticker": t, "Action": action,
            "Previous Weight": pw, "Target Weight": cw,
            "Δ Weight": cw - pw,
        })
    changes_df = pd.DataFrame(changes)

    # ── KPI row ───────────────────────────────────────────────────────────────
    action_counts = changes_df["Action"].value_counts()
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Positions",   len(curr_w))
    k2.metric("New BUYs",    action_counts.get("BUY", 0))
    k3.metric("SELLs",       action_counts.get("SELL", 0))
    k4.metric("Adjustments", action_counts.get("INCREASE", 0) + action_counts.get("REDUCE", 0))
    k5.metric("HOLDs",       action_counts.get("HOLD", 0))

    # ── Action badges + Pie ───────────────────────────────────────────────────
    col_a, col_b, col_c = st.columns([1, 1, 2])

    with col_a:
        st.markdown("#### Action Summary")
        for action, (icon, bg, fg) in ACTION_STYLE.items():
            cnt = action_counts.get(action, 0)
            if cnt:
                st.markdown(
                    f"<div style='background:{bg};color:{fg};border-radius:10px;padding:12px 18px;"
                    f"margin:6px 0;font-weight:700;font-size:15px;letter-spacing:0.02em'>"
                    f"{icon}  {action}  ·  {cnt} position{'s' if cnt > 1 else ''}</div>",
                    unsafe_allow_html=True,
                )

    with col_b:
        st.markdown("#### Portfolio Metrics")
        sel_cov = df_return.loc[:, list(curr_w.index)].dropna()
        exp_ret = float((curr_w * sel_cov.mean()).sum())
        cov_m   = sel_cov.cov().values
        exp_vol = float(np.sqrt(curr_w.values @ cov_m @ curr_w.values) * np.sqrt(nb_rolling))
        exp_sr  = exp_ret / exp_vol if exp_vol > 0 else 0

        st.metric("Active Positions",       len(curr_w))
        st.metric("Expected Period Return", f"{exp_ret:.2%}")
        st.metric("Expected Ann. Vol",      f"{exp_vol:.2%}")
        st.metric("Expected Sharpe",        f"{exp_sr:.2f}")

    with col_c:
        fig_pie = go.Figure(go.Pie(
            labels=curr_w.index.tolist(),
            values=curr_w.values,
            hole=0.50,
            textinfo="label+percent",
            textfont=dict(size=13),
            marker=dict(colors=COLORS, line=dict(color=DARK_PAPER, width=2)),
            pull=[0.06 if a in ("BUY", "INCREASE") else 0
                  for t, a in zip(curr_w.index,
                                  changes_df.set_index("Ticker").reindex(curr_w.index)["Action"].fillna("HOLD"))],
        ))
        fig_pie.update_layout(
            **_plotly_base(360),
            title=f"Target Allocation  —  {next_date.date()}",
            showlegend=False,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    # ── Order sheet table ─────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 📋 Order Sheet")

    display_df = changes_df.copy()
    display_df["Previous Weight"] = display_df["Previous Weight"].map(
        lambda x: f"{x:.1%}" if x > 0 else "—"
    )
    display_df["Target Weight"]   = display_df["Target Weight"].map(
        lambda x: f"{x:.1%}" if x > 0 else "EXIT"
    )
    display_df["Δ Weight"]        = display_df["Δ Weight"].map(
        lambda x: f"{x:+.1%}" if abs(x) > 0.001 else " ≈ "
    )

    def _style_action(v):
        m = {"BUY": "065f46", "SELL": "7f1d1d", "INCREASE": "1e3a5f",
             "REDUCE": "78350f", "HOLD": "1e2d3a"}
        fg = {"BUY": "34d399", "SELL": "f87171", "INCREASE": "93c5fd",
              "REDUCE": "fcd34d", "HOLD": "94a3b8"}
        bg = m.get(v, "")
        f  = fg.get(v, "fff")
        if bg:
            return f"background-color:#{bg};color:#{f};font-weight:700"
        return ""

    styled = (
        display_df.set_index("Ticker")
        .style
        .map(_style_action, subset=["Action"])
    )
    st.dataframe(styled, use_container_width=True, height=min(40 * len(changes_df) + 45, 500))

    col_d1, col_d2 = st.columns(2)
    buf_ord = io.BytesIO()
    changes_df.to_csv(buf_ord, index=False)
    col_d1.download_button(
        "⬇️  Download Order Sheet (CSV)",
        buf_ord.getvalue(),
        f"order_sheet_{next_date.date()}.csv",
        "text/csv",
        use_container_width=True,
    )
    buf_cw = io.BytesIO()
    curr_w.to_frame("Target Weight").to_csv(buf_cw)
    col_d2.download_button(
        "⬇️  Download Target Weights (CSV)",
        buf_cw.getvalue(),
        f"target_weights_{next_date.date()}.csv",
        "text/csv",
        use_container_width=True,
    )

    # ── Weight evolution (last 8 periods) ────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 📈 Weight Evolution — Last 8 Periods")
    last8 = weights_df.iloc[-8:][list(curr_w.index)].copy()
    if not last8.empty:
        fig_bar = go.Figure()
        for i, t in enumerate(last8.columns):
            fig_bar.add_trace(go.Bar(
                x=last8.index.strftime("%Y-%m-%d"), y=last8[t],
                name=t, marker_color=COLORS[i % len(COLORS)],
                hovertemplate=t + "  %{y:.1%}<extra></extra>",
            ))
        fig_bar.update_layout(
            **_plotly_base(340),
            barmode="stack", title="Stacked Weight History",
            yaxis_tickformat=".0%",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(size=11)),
        )
        st.plotly_chart(fig_bar, use_container_width=True)
