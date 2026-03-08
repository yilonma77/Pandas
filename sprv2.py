"""
Smart Portfolio Rebalancing v2  —  sprv2.py
============================================
Modular functions library for systematic portfolio construction.

Sections
--------
1. Data Utilities        — ticker sampling, OHLCV loading, index trimming
2. Technical Signals     — Bollinger Bands, RSI, MA, MACD, Zero-Lag MACD,
                           Rolling Beta / Vol / Correlation
3. Signal Strategies     — buy-signal generation, strategy backtesting,
                           rebalancing table
4. Portfolio Optimization — weight constraints, objective functions,
                            optimisation loop
5. ML Alpha              — feature engineering, position sizing, PnL analytics
6. Visualisation         — performance charts, 3-D plots, heatmaps
"""

# ── Imports ──────────────────────────────────────────────────────────────────
import time
import warnings

import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sb
import statsmodels.api as sm
import yfinance as yf
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf
from sklearn.preprocessing import MinMaxScaler
from tqdm.notebook import tqdm

warnings.filterwarnings("ignore", category=FutureWarning, module="yfinance")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. DATA UTILITIES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def get_sub(x: str) -> str:
    """Convert ASCII characters to their subscript Unicode equivalents."""
    normal = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-=()"
    sub_s  = "ₐ₈CDₑբGₕᵢⱼₖₗₘₙₒₚQᵣₛₜᵤᵥwₓᵧZₐ♭꜀ᑯₑբ₉ₕᵢⱼₖₗₘₙₒₚ૧ᵣₛₜᵤᵥwₓᵧ₂₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎"
    return x.translate(x.maketrans("".join(normal), "".join(sub_s)))


def equal_industry(
    industries_df: pd.DataFrame,
    sample: int = 1,
    industry_col: str = "Industry",
    ticker_col: str = "Ticker",
    exclude: list = [],
    only: list = [],
) -> tuple:
    """
    Sample `sample` tickers per industry.

    Returns
    -------
    tickers : list[str]   — sampled ticker symbols
    ind_df  : DataFrame   — tidy [Ticker, Industry] mapping
    """
    industries_to_see = (
        only if only
        else np.setdiff1d(industries_df[industry_col].unique(), exclude)
    )
    rows = []
    for industry in industries_to_see:
        subset = industries_df[industries_df[industry_col] == industry]
        try:
            rows.append(subset.sample(n=sample))
        except ValueError:          # fewer tickers than requested
            rows.append(subset)

    ind_df = industries_df[[ticker_col, industry_col]].rename(
        columns={ticker_col: "Ticker", industry_col: "Industry"}
    )
    return pd.concat(rows)[ticker_col].tolist(), ind_df


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """Drop tickers whose Close has > 50 % missing values."""
    na_tickers = (df["Close"].isna().sum() > df["Close"].shape[0] / 2)
    na_tickers = na_tickers[na_tickers].index
    valid = pd.Index(set(df["Close"].columns) - set(na_tickers))
    return df.loc[:, pd.IndexSlice[:, valid]].sort_index(axis=1)


def trim_incomplete_period(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    """
    Remove the last row when the most-recent bar is still open
    (common with weekly / monthly yfinance downloads mid-period).
    """
    if interval == "1wk":
        if not (df.index[-1] + pd.offsets.Week(-1)) == df.index[-2]:
            df = df[:-1]
    elif interval == "1mo":
        if not (df.index[-1] + pd.DateOffset(months=-1)) == df.index[-2]:
            df = df[:-1]
    return df


def load_universe(
    sp_csv: str,
    interval: str = "1mo",
    period: str = "16Y",
    sample_per_industry: int = 4,
    exclude_industries: list = ["Energy", "Real Estate"],
    only_industries: list = [],
    elements_to_remove: list = [],
) -> tuple:
    """
    End-to-end data pipeline:
      1. Read S&P CSV, sample tickers per industry
      2. Download OHLCV + ^GSPC benchmark + ^TNX 10-year rate
      3. Clean and trim DataFrames

    Returns
    -------
    df, market_df, ten_year, df_return, tickers, industries
    """
    sp_full = pd.read_csv(sp_csv, sep=";")
    tickers, industries = equal_industry(
        sp_full, sample_per_industry,
        industry_col="Industry",
        exclude=exclude_industries,
        only=only_industries,
    )
    for t in elements_to_remove:
        try:
            tickers.remove(t)
        except ValueError:
            pass

    market_df = yf.download("^GSPC", interval=interval).dropna()
    # yfinance ≥ 0.2.x returns a MultiIndex even for a single ticker — flatten it
    if isinstance(market_df.columns, pd.MultiIndex):
        market_df.columns = market_df.columns.droplevel(1)
    market_df["Chg"] = market_df["Close"].pct_change()

    tnx_raw = yf.download("^TNX").dropna()
    if isinstance(tnx_raw.columns, pd.MultiIndex):
        tnx_raw.columns = tnx_raw.columns.droplevel(1)
    ten_year = tnx_raw.reindex(market_df.index)["Close"].div(100).ffill()

    df = yf.download(tickers, period=period, interval=interval)
    df.index = pd.to_datetime(df.index)
    df = clean_df(df)
    df = trim_incomplete_period(df, interval)
    market_df = market_df.reindex(df.index)
    df_return = df["Close"].pct_change()

    return df, market_df, ten_year, df_return, tickers, industries


def _norm_mkt(mdf: pd.DataFrame) -> pd.DataFrame:
    """Normalise a yfinance market DataFrame regardless of API version.

    yfinance ≥ 0.2.x returns a MultiIndex even for a single ticker.
    This helper flattens the columns and ensures the 'Chg' column exists
    so every downstream function receives a plain flat DataFrame.
    """
    if isinstance(mdf.columns, pd.MultiIndex):
        mdf = mdf.copy()
        mdf.columns = mdf.columns.droplevel(1)
    if "Chg" not in mdf.columns:
        mdf = mdf.copy()
        mdf["Chg"] = mdf["Close"].pct_change()
    return mdf


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. TECHNICAL SIGNALS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def compute_signals(
    df: pd.DataFrame,
    market_df: pd.DataFrame,
    tickers_to_see,
    nb_rolling: int = 12,
) -> pd.DataFrame:
    """
    Compute all technical indicators for every ticker in df['Close']:
      - Bollinger Bands (±2σ and ±3σ)
      - RSI (14)
      - Moving Averages (MA13, MA20)
      - MACD & Signal line
      - Zero-Lag MACD & Signal line
      - Rolling Beta (vs ^GSPC)
      - Rolling Volatility
      - Rolling Intra-universe Correlation (tickers_to_see only)

    Returns a wide MultiIndex DataFrame (indicator, ticker).
    """
    market_df = _norm_mkt(market_df)
    df_signal = pd.DataFrame(df[["Close", "Volume", "Open", "High", "Low"]])
    periodma, periodma2 = 13, 20

    for ticker in df_signal["Close"].columns:
        print(ticker, end="\r")
        close = df_signal["Close", ticker]

        # ── Bollinger Bands ───────────────────────────────────────────────
        std_mult, length = 2, 20
        mid      = close.rolling(length).mean()
        roll_std = close.rolling(length).std()
        df_signal = pd.concat([
            df_signal,
            (mid + std_mult       * roll_std).rename(("upperband",  ticker)),
            (mid - std_mult       * roll_std).rename(("lowerband",  ticker)),
            (mid + (std_mult + 1) * roll_std).rename(("upperband2", ticker)),
            (mid - (std_mult + 1) * roll_std).rename(("lowerband2", ticker)),
        ], axis=1)

        # ── RSI (14) ──────────────────────────────────────────────────────
        delta  = close.diff()
        gains  = delta.where(delta > 0, 0).rolling(14).mean()
        losses = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rsi    = 100 - (100 / (1 + gains / losses))
        df_signal = pd.concat([df_signal, rsi.rename(("rsi", ticker))], axis=1)

        # ── Moving Averages ───────────────────────────────────────────────
        df_signal = pd.concat([
            df_signal,
            close.rolling(periodma,  min_periods=periodma).mean().rename(("MA",  ticker)),
            close.rolling(periodma2, min_periods=periodma).mean().rename(("MA2", ticker)),
        ], axis=1)

        # ── MACD & Signal ─────────────────────────────────────────────────
        ema12  = close.ewm(span=12, adjust=False, min_periods=12).mean()
        ema26  = close.ewm(span=26, adjust=False, min_periods=26).mean()
        macd   = ema12 - ema26
        macd_s = macd.ewm(span=9, adjust=False, min_periods=9).mean()
        df_signal = pd.concat([
            df_signal,
            macd.rename(("macd",        ticker)),
            macd_s.rename(("macd_signal", ticker)),
        ], axis=1)

        # ── Zero-Lag MACD & Signal ────────────────────────────────────────
        zl_fast = 2 * ema12 - ema12.ewm(span=12, adjust=False, min_periods=12).mean()
        zl_slow = 2 * ema26 - ema26.ewm(span=26, adjust=False, min_periods=26).mean()
        zl_macd = zl_fast - zl_slow
        zl_sig  = zl_macd.ewm(span=13, min_periods=12).mean()
        zl_sig  = 2 * zl_sig - zl_sig.ewm(span=13, min_periods=12).mean()
        df_signal = pd.concat([
            df_signal,
            zl_macd.rename(("zl_macd",   ticker)),
            zl_sig.rename(("zl_macd_s", ticker)),
        ], axis=1)

        # ── Rolling Beta ──────────────────────────────────────────────────
        pct  = close.pct_change()
        mkt  = market_df["Close"].pct_change()
        beta = (
            mkt.rolling(int(periodma2 / 2)).cov(pct)
            / mkt.rolling(int(periodma2 / 2)).var()
        )
        df_signal = pd.concat([df_signal, beta.rename(("beta", ticker))], axis=1)

        # ── Rolling Volatility ────────────────────────────────────────────
        df_signal = pd.concat([
            df_signal, pct.rolling(14).std().rename(("roll_vol", ticker))
        ], axis=1)

        # ── Rolling Intra-universe Correlation ────────────────────────────
        if ticker in tickers_to_see:
            roll_corr = (
                df_signal["Close"][tickers_to_see]
                .pct_change()
                .rolling(int(periodma2 / 2))
                .corr()
                .unstack(level=1)[ticker]
                .drop(ticker, axis=1)
                .mean(axis=1)
            )
            df_signal = pd.concat([
                df_signal, roll_corr.rename(("roll_corr", ticker))
            ], axis=1)

    print(" " * 30)
    return df_signal


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. SIGNAL STRATEGIES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

STRATEGIES = {
    0:  "Price > MA",
    1:  "Price > MA & MA1 > MA2",
    2:  "Price > MA & MACD",
    3:  "Price > MA & MACD & Close < BB 3σ",
    4:  "Price > MA & ZL-MACD & Close < BB 3σ & RSI > 10",
    5:  "ZL-MACD & RSI < 85",
    6:  "ZL-MACD & RSI > 25",
    7:  "RSI (32–72)",
    8:  "RSI (32–72) & Close < BB 3σ",
    9:  "Price > MA & MACD & Close < BB 3σ & MA1 > MA2",
    10: "Rolling β > Avg β",
    11: "Rolling β > Avg β & ZL-MACD",
    12: "Lowest ½ rolling correlation",
    13: "Highest ½ rolling correlation",
    14: "Lowest ½ rolling volatility",
    15: "Highest ½ rolling volatility",
    16: "Always invested (ø)",
}

TTS_MAPPING = {
    0: "Correlation: Ascending",  1: "Correlation: Descending",
    2: "Volatility: Ascending",   3: "Volatility: Descending",
    4: "Volume: Ascending",       5: "Volume: Descending",
    6: "Price: Ascending",        7: "Price: Descending",
    8: "Beta: Ascending",         9: "Beta: Descending",
}


def assign_signals(
    df_signal: pd.DataFrame,
    tickers_to_see,
    nombre_tickers: int,
) -> pd.DataFrame:
    """Append binary buy{0..16} columns to df_signal for every ticker."""
    half = int(nombre_tickers / 2)
    for ticker in df_signal["Close"].columns:
        c    = df_signal["Close",       ticker]
        ma   = df_signal["MA",          ticker]
        ma2  = df_signal["MA2",         ticker]
        mac  = df_signal["macd",        ticker]
        macs = df_signal["macd_signal", ticker]
        zlm  = df_signal["zl_macd",     ticker]
        zls  = df_signal["zl_macd_s",   ticker]
        rsi  = df_signal["rsi",         ticker]
        bb2  = df_signal["upperband2",  ticker]
        beta = df_signal["beta",        ticker]

        df_signal.loc[:, ("buy0",  ticker)] = np.where(c > ma, 1, 0)
        df_signal.loc[:, ("buy1",  ticker)] = np.where((c > ma) & (ma > ma2), 1, 0)
        df_signal.loc[:, ("buy2",  ticker)] = np.where((c > ma) & (mac > macs), 1, 0)
        df_signal.loc[:, ("buy3",  ticker)] = np.where((c > ma) & (mac > macs) & (c < bb2), 1, 0)
        df_signal.loc[:, ("buy4",  ticker)] = np.where((c > ma) & (zlm > zls) & (c < bb2) & (rsi > 10), 1, 0)
        df_signal.loc[:, ("buy5",  ticker)] = np.where((zlm > zls) & (rsi < 85), 1, 0)
        df_signal.loc[:, ("buy6",  ticker)] = np.where((zlm > zls) & (rsi > 25), 1, 0)
        df_signal.loc[:, ("buy7",  ticker)] = np.where((rsi > 32) & (rsi < 72), 1, 0)
        df_signal.loc[:, ("buy8",  ticker)] = np.where((rsi > 32) & (rsi < 72) & (c < bb2), 1, 0)
        df_signal.loc[:, ("buy9",  ticker)] = np.where((c > ma) & (mac > macs) & (c < bb2) & (ma > ma2), 1, 0)
        df_signal.loc[:, ("buy10", ticker)] = np.where(beta > beta.mean(), 1, 0)
        df_signal.loc[:, ("buy11", ticker)] = np.where((beta > beta.mean()) & (zlm > zls), 1, 0)
        df_signal.loc[:, ("buy16", ticker)] = 1

        if ticker in tickers_to_see:
            rv = df_signal["roll_vol"][list(tickers_to_see)]
            rc = df_signal["roll_corr"][list(tickers_to_see)]
            df_signal.loc[:, ("buy12", ticker)] = np.where(rc.rank(axis=1)[ticker] < half, 1, 0)
            df_signal.loc[:, ("buy13", ticker)] = np.where(rc.rank(axis=1)[ticker] > half, 1, 0)
            df_signal.loc[:, ("buy14", ticker)] = np.where(rv.rank(axis=1)[ticker] < half, 1, 0)
            df_signal.loc[:, ("buy15", ticker)] = np.where(rv.rank(axis=1)[ticker] > half, 1, 0)
        else:
            for s in [12, 13, 14, 15]:
                df_signal.loc[:, (f"buy{s}", ticker)] = 0

    return df_signal


def _build_tx_df(df_signal, strategy_id, period_to_start):
    """Internal: compute per-ticker Return / Cumprod DataFrame for one strategy."""
    df_tx = pd.DataFrame(index=df_signal.index)
    for ticker in df_signal["Close"].columns:
        chg = df_signal["Close", ticker].pct_change().shift(-1)
        ret = chg * df_signal[f"buy{strategy_id}", ticker]
        cum = (1 + ret[period_to_start:]).cumprod()
        df_tx = pd.concat([
            df_tx,
            ret.rename(f"Return_{ticker}"),
            cum.rename(f"Cumprod_{ticker}"),
        ], axis=1)
    df_tx.columns = df_tx.columns.str.split("_", expand=True)
    return df_tx


def find_best_signal(
    df_signal: pd.DataFrame,
    tickers_to_see,
    period_to_start: str,
    nombre_tickers: int,
    nb_rolling: int,
    best: int | str = "default",
    criteria: str = "sharpe",
    test_size: float = 0.8,
    trace: bool = True,
) -> tuple:
    """
    Evaluate all 17 strategies and return the best one.

    Parameters
    ----------
    criteria : 'sharpe' | 'return' | 'sd'
    test_size: fraction of period_to_start→end used for in-sample optimisation

    Returns
    -------
    (best_strategy_id, end_of_test_date)
    """
    df_signal   = assign_signals(df_signal, tickers_to_see, nombre_tickers)
    recap_df    = pd.DataFrame(columns=["Return", "Volatility"])
    n_test      = int(test_size * len(df_signal.loc[period_to_start:]))
    test_period = (
        df_signal.loc[period_to_start:].iloc[:n_test].index
        if test_size != 1
        else df_signal.loc[period_to_start:].index
    )

    score_map = {}
    for strat in STRATEGIES:
        df_tx = _build_tx_df(df_signal, strat, period_to_start)

        r_test = df_tx.loc[test_period, "Cumprod"][list(tickers_to_see)].iloc[-2].mean()
        v_test = df_tx.loc[test_period, "Return"][list(tickers_to_see)].std().mean()
        score  = {
            "sharpe": (r_test - 1) / v_test,
            "return": r_test,
            "sd":     1 / v_test,
        }.get(criteria, (r_test - 1) / v_test)
        score_map[strat] = score

        r_full = df_tx.loc[period_to_start:, "Cumprod"][list(tickers_to_see)].iloc[-2].mean()
        v_full = df_tx.loc[period_to_start:, "Return"][list(tickers_to_see)].std().mean()
        sr_full= (
            df_tx.loc[period_to_start:, "Return"][list(tickers_to_see)].sum(axis=1).mean()
            / df_tx.loc[period_to_start:, "Return"][list(tickers_to_see)].sum(axis=1).std()
            * np.sqrt(nb_rolling)
        )
        recap_df.loc[len(recap_df)] = [r_full, v_full]
        if trace:
            print(f"[{strat:>2}] {STRATEGIES[strat]:<55} "
                  f"Ret: {r_full:.2f} | σ: {v_full:.2%} | Sharpe: {sr_full:.2f}")

    # Risk-return scatter
    plt.figure(figsize=(11, 3.5))
    plt.scatter(recap_df["Volatility"], recap_df["Return"], zorder=3)
    best_idx = (recap_df["Return"] / recap_df["Volatility"]).idxmax()
    plt.scatter(recap_df.loc[best_idx, "Volatility"], recap_df.loc[best_idx, "Return"],
                color="red", zorder=4, label="Best Sharpe")
    for i in recap_df.index:
        plt.annotate(i, (recap_df.loc[i, "Volatility"], recap_df.loc[i, "Return"] * 1.015))
    plt.xlabel("Volatility")
    plt.ylabel("Return")
    plt.title("Strategy Risk-Return Map")
    plt.legend()
    plt.tight_layout()
    plt.show()

    chosen = max(score_map, key=score_map.get) if best == "default" else best
    return chosen, test_period[-1]


def create_rebalancing_table(
    df_signal: pd.DataFrame,
    tickers_to_see,
    best_signal: int,
    period_to_start: str,
) -> pd.DataFrame:
    """Build a date × position rebalancing table from the chosen strategy."""
    sig_df = df_signal[f"buy{best_signal}"][list(tickers_to_see)]
    rows = [
        pd.Series(sig_df.loc[d][sig_df.loc[d] == True].index, name=d)
        for d in sig_df.index
    ]
    return pd.DataFrame(rows)[period_to_start:]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. PORTFOLIO OPTIMISATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _cov_matrix(date, df, rebalancing_table, cov_ma_period, nb_rolling):
    start   = str(pd.to_datetime(date) - pd.DateOffset(weeks=int(cov_ma_period)))
    tickers = rebalancing_table.loc[date].dropna()
    ret = (df["Close"][start:date][tickers].pct_change() * nb_rolling).dropna()
    return ret.cov(ddof=1)


def _past_return(wght, date, df, rebalancing_table):
    tickers = rebalancing_table.loc[date].dropna()
    return df["Close"][:date][tickers].pct_change().iloc[-1] @ wght


def _portfolio_beta(wght, date, df, market_df, rebalancing_table, cov_ma_period):
    market_df = _norm_mkt(market_df)
    start = str(pd.to_datetime(date) - pd.DateOffset(weeks=int(2 * cov_ma_period)))
    mkt   = market_df["Close"].pct_change()[start:date].dropna()
    pf    = (
        df["Close"].pct_change()[start:date][rebalancing_table.loc[date].dropna()]
        .dropna() * wght
    ).sum(axis=1)
    return mkt.cov(pf) / mkt.var()


# ── Objectives ───────────────────────────────────────────────────────────────

def obj_variance(wght, date, df, rebalancing_table, cov_ma_period, nb_rolling):
    cov = _cov_matrix(date, df, rebalancing_table, cov_ma_period, nb_rolling)
    return float(wght @ cov @ wght)


def obj_sharpe(wght, date, df, rebalancing_table, cov_ma_period, nb_rolling):
    ret = _past_return(wght, date, df, rebalancing_table)
    var = obj_variance(wght, date, df, rebalancing_table, cov_ma_period, nb_rolling)
    return -ret / var


def obj_sortino(wght, date, df, rebalancing_table, cov_ma_period, nb_rolling):
    start       = str(pd.to_datetime(date) - pd.DateOffset(weeks=int(cov_ma_period)))
    ticker_list = list(rebalancing_table.loc[date].dropna().values)
    avail       = [t for t in ticker_list if t in df["Close"].columns]
    if not avail or len(avail) < 2:
        return -_past_return(wght, date, df, rebalancing_table) / 1e-8
    raw = df["Close"].reindex(columns=avail)[start:date].pct_change().fillna(0) * nb_rolling
    neg_ret = raw.clip(upper=0)
    X = np.nan_to_num(neg_ret.to_numpy(), nan=0.0)
    if X.shape[0] < 2:
        return -_past_return(wght, date, df, rebalancing_table) / 1e-8
    downside = wght @ LedoitWolf().fit(X).covariance_ @ wght
    return -_past_return(wght, date, df, rebalancing_table) / max(downside, 1e-10)


def obj_treynor(wght, date, df, market_df, rebalancing_table, cov_ma_period):
    ret  = _past_return(wght, date, df, rebalancing_table)
    beta = _portfolio_beta(wght, date, df, market_df, rebalancing_table, cov_ma_period)
    return -ret / beta


def obj_risk_parity(wght, date, df, rebalancing_table, cov_ma_period, nb_rolling):
    cov = _cov_matrix(date, df, rebalancing_table, cov_ma_period, nb_rolling)
    cx  = cov @ wght
    tot = wght @ cov @ wght
    n   = cov.shape[1]
    return float(np.sum(np.abs(cx * wght / tot - 1 / n)))


# ── Constraints ──────────────────────────────────────────────────────────────

def constr_weights_sum(wght):
    return abs(np.sum(wght)) - 1


def constr_max_corr(wght, date, df, rebalancing_table, cov_ma_period, min_weight, max_corr=1.0):
    """Equalise weights of the most-correlated pair (empirical hedging)."""
    try:
        start   = str(pd.to_datetime(date) - pd.DateOffset(weeks=int(cov_ma_period / 1.25)))
        tickers = rebalancing_table.loc[date].dropna()
        corr    = df["Close"][start:date][tickers].pct_change().dropna().corr()
        if corr.min().sort_values().iloc[0] < max_corr:
            t1 = corr.min().sort_values().index[0]
            t2 = corr[t1].idxmin()
            idx = rebalancing_table.loc[date].dropna()
            w1  = idx[idx == t1].index.values[0]
            w2  = idx[idx == t2].index.values[0]
            return abs(wght[w1] - min_weight) + abs(wght[w2] - min_weight)
    except Exception:
        pass
    return 0


# ── Weight helpers ────────────────────────────────────────────────────────────

def cap_weight(weights: np.ndarray, cap: float, method: str = "proportional") -> np.ndarray:
    """Apply a weight cap and redistribute excess proportionally / equally / flipped."""
    weights = np.array(weights, dtype=float)
    excess  = (weights[weights > cap] - cap).sum()
    if excess > 0:
        weights[weights > cap] = cap
        below      = weights < cap
        below_sum  = weights[below].sum()
        if method == "proportional" and below_sum > 0:
            weights[below] += weights[below] / below_sum * excess
        elif method == "equal" and below.sum() > 0:
            weights[below] += excess / below.sum()
        elif method == "flip" and below_sum > 0:
            weights[below] += np.flip(weights[below] / below_sum) * excess
    return weights


def volume_weights(date, df, rebalancing_table, cap: float, method: str = "proportional") -> np.ndarray:
    """Volume-proportional weights with capping."""
    tickers = rebalancing_table.loc[date].dropna()
    vol_sum = df["Volume"][tickers].sum(axis=1)
    raw     = df["Volume"][tickers].div(vol_sum, axis=0).loc[date].values
    return cap_weight(raw, cap, method)


def _min_w_bound(min_w: float, n: int) -> float:
    return min_w if n * min_w <= 1 else 0.35 / n


def _max_w_bound(n: int, threshold: float) -> float:
    return threshold / n


# ── Main optimisation loop ────────────────────────────────────────────────────

def run_optimization(
    df,
    df_return,
    market_df,
    rebalancing_table,
    tickers_to_see,
    period_to_start: str,
    min_pos: int = 3,
    min_w: float = 0.04725,
    max_w_ratio: float = 2.86,
    cov_ma_period: int = 32,
    nb_rolling: int = 12,
    objective: str = "sortino",
    corr_constraint: bool = True,
    max_corr: float = 0.4,
) -> tuple:
    """
    Walk-forward portfolio optimisation.

    Parameters
    ----------
    objective : 'sortino' | 'sharpe' | 'variance' | 'treynor' | 'risk_parity'

    Returns
    -------
    weights_df, equal_w_df
    """
    _OBJ = {
        "variance":   lambda w, d: obj_variance(w, d, df, rebalancing_table, cov_ma_period, nb_rolling),
        "sharpe":     lambda w, d: obj_sharpe(w, d, df, rebalancing_table, cov_ma_period, nb_rolling),
        "sortino":    lambda w, d: obj_sortino(w, d, df, rebalancing_table, cov_ma_period, nb_rolling),
        "treynor":    lambda w, d: obj_treynor(w, d, df, market_df, rebalancing_table, cov_ma_period),
        "risk_parity":lambda w, d: obj_risk_parity(w, d, df, rebalancing_table, cov_ma_period, nb_rolling),
    }
    obj_fn = _OBJ.get(objective, _OBJ["sortino"])

    start_t   = time.time()
    total_days= (pd.to_datetime(df.iloc[-1].name) - pd.to_datetime(period_to_start)).days
    history   = []

    for date in tqdm(df.loc[period_to_start:].index):
        signal = rebalancing_table.loc[date].dropna()
        n      = signal.count()
        pct    = 1 - (pd.to_datetime(df.iloc[-1].name) - pd.to_datetime(date)).days / total_days
        print(f"{date.date()} / {df.iloc[-1].name.date()}  \033[1m({pct:.2%})\033[0m  {n} positions   ", end="\r")

        if n > min_pos:
            bounds = [(_min_w_bound(min_w, n), _max_w_bound(n, max_w_ratio))] * n
            w0     = np.ones(n) / n
            constraints = [{"type": "eq", "fun": constr_weights_sum}]
            if corr_constraint:
                constraints.append({
                    "type": "eq",
                    "fun":  constr_max_corr,
                    "args": [date, df, rebalancing_table, cov_ma_period, min_w, max_corr],
                })
            result = minimize(obj_fn, x0=w0, args=(date,), method="SLSQP",
                              bounds=bounds, constraints=constraints,
                              options={"disp": False})
            history.append(dict(zip(signal.values, result.x)))
        else:
            history.append(dict(zip(list(tickers_to_see), np.zeros(len(tickers_to_see)))))

    weights_df = (
        pd.DataFrame(history, columns=list(tickers_to_see), index=df.loc[period_to_start:].index)
        .fillna(0)
        .reindex(sorted(list(tickers_to_see)), axis=1)
    )
    equal_w_df = weights_df.replace(0, np.nan).abs()
    equal_w_df[equal_w_df > 0] = 1
    equal_w_df = equal_w_df.div(equal_w_df.count(axis=1), axis=0)

    elapsed = time.time() - start_t
    ret = (1 + (df_return.loc[period_to_start:].shift(-1) * weights_df).sum(axis=1)).prod()
    vol = (1 + (df_return.loc[period_to_start:].shift(-1) * weights_df).sum(axis=1)).std()
    print(f"\n\033[1mOptimisation complete\033[0m in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"Cumulative return: {ret:.2f}  |  σ per period: {vol:.2%}")

    return weights_df, equal_w_df


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. ML ALPHA
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def alpha_to_positions(alpha: pd.DataFrame, short: bool = False) -> pd.DataFrame:
    """Rank-based position sizing; dollar-neutral when short=True."""
    pos = alpha.rank(axis=1)
    pos = pos.div(pos.sum(axis=1), axis=0)
    if short:
        pos = pos.sub(pos.mean(axis=1), axis=0)
    return pos


def sharpe_ratio(returns: pd.Series, nb_rolling: int = 12) -> float:
    """Annualised Sharpe ratio from a periodic return series."""
    r = returns.replace(0, np.nan).dropna()
    years   = r.index[-1].year - r.index[0].year
    cum     = (1 + r).cumprod()
    ann_ret = (cum.iloc[-1] / cum.iloc[0]) ** (1 / years) - 1
    ann_vol = r.std() * np.sqrt(nb_rolling)
    return ann_ret / ann_vol


def pnl_analytics(positions: pd.DataFrame, df_return, lag: int, nb_rolling: int) -> tuple:
    """
    Compute P&L from shifted positions.

    Returns
    -------
    pnl (Series), sharpe (float)
    """
    tickers        = positions.columns
    shifted_rets   = df_return[tickers].shift(-1)[positions.index[0]:positions.index[-1]]
    pnl            = shifted_rets.mul(positions.shift(lag)).sum(axis=1)
    clean          = pnl.replace(0, np.nan).dropna()
    sharpe         = clean.mean() / clean.std() * np.sqrt(nb_rolling)
    return pnl, sharpe


def rolling_autocorr(series: pd.Series, window: int) -> pd.Series:
    """Rolling autocorrelation of a return series (used as an alpha feature)."""
    return series.rolling(window, int(window / 2)).corr(series.shift(1))


def macd_distance(series: pd.Series) -> pd.Series:
    """
    Normalised MACD–Signal distance as a cross-sectional alpha signal.
    Higher value → stronger sell pressure (mean-reversion flavour).
    """
    mn, mx = series.min(), series.max()
    s    = (series - mn) / (mx - mn) if mx > mn else series
    fast = s.ewm(span=12, adjust=False, min_periods=12).mean()
    slow = s.ewm(span=26, adjust=False, min_periods=26).mean()
    macd = fast - slow
    sig  = macd.ewm(span=9, adjust=False, min_periods=9).mean()
    return sig - macd


def build_ml_features(df_signal: pd.DataFrame) -> dict:
    """Return a dict of raw alpha DataFrames (one per feature name)."""
    return {
        "pct_change":    df_signal["Close"].pct_change(),
        "volume":       -(df_signal["Volume"] * df_signal["Close"].pct_change())
                         / df_signal["Volume"].rolling(2).mean(),
        "autocorr":      df_signal["Close"].pct_change().apply(rolling_autocorr, args=([6])),
        "macd_distance": df_signal["Close"].pct_change().apply(macd_distance),
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6. VISUALISATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

BBOX = dict(boxstyle="square", pad=0.3, fc="white", ec="black", lw=1.65)


def industry_color(industries: pd.DataFrame, tickers_to_see) -> pd.DataFrame:
    """Assign a distinct Plotly colour to each industry in the universe."""
    df = industries.copy()
    for t in tickers_to_see:
        if not df[df["Ticker"] == t].any().any():
            df.loc[len(df)] = [t, "#"]
    df["Colorid"] = pd.factorize(df["Industry"])[0]
    df["Color"]   = [px.colors.qualitative.Dark24[int(c)] for c in df["Colorid"]]
    return df


def plot_industry_overview(df, market_df, industries, period: str = "2020"):
    """Industry correlation heatmap + cumulative return per industry."""
    market_df = _norm_mkt(market_df)
    # Determine actual index name (yfinance version may differ: "Date" / "Datetime" / "Price")
    _idx = df.index.name or "Date"
    basis = (
        df["Close"].loc[period:]
        .pct_change()
        .rename_axis("Date")
        .reset_index().iloc[1:]
        .melt(id_vars="Date", var_name="Ticker", value_name="Price")
        .merge(industries, on="Ticker")
        .groupby(["Industry", "Date"]).sum(numeric_only=True)
        .div(df["Close"].shape[1])
        .unstack(level=0)
    )
    basis.columns = basis.columns.droplevel(0)
    corr   = basis.corr()
    cumret = (1 + basis).cumprod()
    n_tick = (
        df["Close"].pct_change()
        .rename_axis("Date")
        .reset_index().iloc[1:]
        .melt(id_vars="Date", var_name="Ticker", value_name="Price")
        .merge(industries, on="Ticker")
        .drop_duplicates("Ticker")["Industry"]
        .value_counts()
    )

    fig, axes = plt.subplots(1, 2, figsize=(19, 8.5))
    fig.suptitle("Securities Snapshot: Correlation & Cumulative Return", fontsize=15)

    mask = np.triu(np.ones_like(corr, dtype=bool), 1)
    sb.heatmap(corr, mask=mask, annot=True, cmap="coolwarm",
               ax=axes[0], fmt=".2g")
    axes[0].set_yticklabels(corr.columns, rotation=0, fontsize=7.5)
    axes[0].set_xticks([])
    axes[0].set_title("Industry Correlation Heatmap", fontsize=13)
    top20 = "\n".join(
        f"{t}: {int(r)}"
        for t, r in df["Close"].pct_change().std().rank(ascending=False).sort_values()[:20].items()
    )
    axes[0].text(0.75, 1, "Top 20 σ ranks:\n" + top20, transform=axes[0].transAxes,
                 fontsize=9, va="top", bbox=dict(boxstyle="round", fc="wheat", alpha=0.5))

    cumret.plot(ax=axes[1])
    legend = [f"{c}: {n_tick.get(c, 0)} tickers" for c in cumret] + ["Benchmark"]
    mkt_raw = market_df["Close"].pct_change()
    if isinstance(mkt_raw, pd.DataFrame):
        mkt_raw = mkt_raw.iloc[:, 0]      # single-ticker yfinance returns a DataFrame
    mkt = (1 + mkt_raw.div(df["Close"].shape[1])).reindex(cumret.index).cumprod()
    mkt.plot(ax=axes[1], c="black", ls="--", label="Benchmark")
    axes[1].fill_between(mkt.index, mkt.values, axes[1].get_ylim()[0], color="red", alpha=0.04)
    axes[1].axhline(y=1, lw=3, ls="--", c="royalblue")
    axes[1].set_ylabel("Cumulative return", fontsize=9)
    axes[1].set_title("Cumulative Return per Industry", fontsize=13)
    axes[1].legend(legend, fontsize=8)
    plt.tight_layout()
    plt.show()


def plot_ticker_configs(
    df, df_return, market_df, tickers_to_see,
    nombre_tickers, period_to_start, nb_rolling,
    list_tts: int | str = "default",
):
    """
    10-panel figure comparing ticker selection strategies
    (correlation, volatility, volume, price change, beta — ascending/descending).

    Returns
    -------
    chosen_tickers, chosen_idx, all_lists
    """
    market_df = _norm_mkt(market_df)
    corr_p = df["Close"].pct_change().corr()
    ttsac  = corr_p.min().sort_values(ascending=True)[:nombre_tickers].index
    ttsdc  = corr_p.min().sort_values(ascending=False)[:nombre_tickers].index
    ttsas  = df["Close"].pct_change().std().sort_values()[:nombre_tickers].index
    ttsds  = df["Close"].pct_change().std().sort_values(ascending=False)[:nombre_tickers].index
    ttsav  = df["Volume"].iloc[-nb_rolling:].mean().sort_values()[:nombre_tickers].index
    ttsdv  = df["Volume"].iloc[-nb_rolling:].mean().sort_values(ascending=False)[:nombre_tickers].index
    pdiff  = pd.Series(
        [df["Close"][t].dropna().iat[-1] - df["Close"][t].dropna().iat[0] for t in df["Close"].columns],
        index=df["Close"].columns,
    )
    ttsap  = pdiff.sort_values(ascending=False)[:nombre_tickers].index
    ttsdp  = pdiff.sort_values()[:nombre_tickers].index
    # Use a guaranteed unique name for the market column so it is never
    # accidentally dropped or confused with a ticker.  dropna(subset=) only
    # removes rows where the market has no value, keeping all stock columns.
    _MKT = "__MKT__"
    mkt_s    = market_df["Close"].squeeze().rename(_MKT).pct_change().iloc[1:]
    beta_cov = pd.concat([df["Close"].pct_change().iloc[1:], mkt_s], axis=1) \
                 .dropna(subset=[_MKT]).cov()
    betas    = [(s, beta_cov.at[s, _MKT] / beta_cov.at[_MKT, _MKT])
                for s in beta_cov.columns.difference([_MKT])
                if s in beta_cov.index]
    ttsab  = pd.DataFrame(betas).set_index(0).sort_values(1)[:nombre_tickers].index
    ttsdb  = pd.DataFrame(betas).set_index(0).sort_values(1, ascending=False)[:nombre_tickers].index

    # When the universe is too small, cap nombre_tickers so each strategy
    # still picks a meaningful (non-identical) subset of tickers.
    if len(df["Close"].columns) <= nombre_tickers:
        nombre_tickers = max(2, len(df["Close"].columns) // 2)
        ttsac  = corr_p.min().sort_values(ascending=True)[:nombre_tickers].index
        ttsdc  = corr_p.min().sort_values(ascending=False)[:nombre_tickers].index
        ttsas  = df["Close"].pct_change().std().sort_values()[:nombre_tickers].index
        ttsds  = df["Close"].pct_change().std().sort_values(ascending=False)[:nombre_tickers].index
        ttsav  = df["Volume"].iloc[-nb_rolling:].mean().sort_values()[:nombre_tickers].index
        ttsdv  = df["Volume"].iloc[-nb_rolling:].mean().sort_values(ascending=False)[:nombre_tickers].index
        ttsap  = pdiff.sort_values(ascending=False)[:nombre_tickers].index
        ttsdp  = pdiff.sort_values()[:nombre_tickers].index
        ttsab  = pd.DataFrame(betas).set_index(0).sort_values(1)[:nombre_tickers].index
        ttsdb  = pd.DataFrame(betas).set_index(0).sort_values(1, ascending=False)[:nombre_tickers].index
    all_lists = [ttsac, ttsdc, ttsas, ttsds, ttsav, ttsdv, ttsap, ttsdp, ttsab, ttsdb]

    mkt_ret  = (market_df[period_to_start:]["Chg"] + 1).cumprod()
    fig, axes = plt.subplots(5, 2, figsize=(16, 10.5))
    fig.suptitle(f"Ticker Selection Strategies  —  {nombre_tickers} tickers", fontsize=20)

    best_val, best_idx = 0, 0
    for ax, tlist, num in zip(axes.ravel(), all_lists, range(10)):
        ret = df_return.loc[:, tlist]
        cum = ((1 / ret.shape[1]) * ret.loc[period_to_start:]).sum(axis=1).add(1).cumprod()
        cum.plot(ax=ax)
        if cum.iloc[-1] > best_val:
            best_val, best_idx = cum.iloc[-1], num
        maxv = max(cum.max(), mkt_ret.max()) + 0.1
        minv = min(cum.min(), mkt_ret.min()) - 0.05
        ax.fill_between(mkt_ret.index, mkt_ret, maxv, color="limegreen", alpha=0.08)
        ax.fill_between(mkt_ret.index, mkt_ret, -3, color="red", alpha=0.08)
        ax.axhline(y=1, color="black", ls="--", lw=2)
        ax.set_ylim(minv, maxv)
        ax.set_xlim(mkt_ret.index[0], mkt_ret.index[-1])
        ax.set_title(f"{TTS_MAPPING[num]}  ({ret.shape[0]} periods)")
        ax.annotate(f"{cum.iloc[-1]:.2f}", xy=(cum.index[-1], cum.iloc[-1]),
                    xytext=(5, 0), textcoords="offset points", bbox=BBOX)
    plt.tight_layout()
    plt.show()

    chosen     = all_lists[list_tts] if list_tts != "default" else all_lists[best_idx]
    chosen_idx = list_tts if list_tts != "default" else best_idx
    return chosen, chosen_idx, all_lists


def plot_ticker_detail(df, industries, tickers_to_see, tts_text, nb_rolling):
    """Correlation heatmap + annual rolling volatility ranking (side-by-side)."""
    fig, axes = plt.subplots(1, 2, figsize=(30, 9.5))
    fig.suptitle(f"Ticker Detail  —  {tts_text}", fontsize=30)

    corr = df["Close"][list(tickers_to_see)].pct_change().corr().sort_index().sort_index(axis=1)
    mask = np.triu(np.ones_like(corr, dtype=bool), 1)
    sb.heatmap(corr, mask=mask, annot=True, ax=axes[0], cmap="Spectral", fmt=".2f")
    axes[0].set_title("Correlation Heatmap", size=23)
    txt = "\n".join(
        f"{ind}: {cnt} tickers"
        for ind, cnt in industries[industries["Ticker"].isin(list(tickers_to_see))]["Industry"]
        .value_counts().items()
    )
    axes[0].text(0.6, 0.75, txt, bbox=BBOX, fontsize=15, ha="left", va="center",
                 transform=axes[0].transAxes)

    vol_rank = (
        df["Close"][list(tickers_to_see)].pct_change().fillna(0)
        .rolling(nb_rolling).var()
        .resample("YE").last()
        .dropna(axis=0)
        .rank(axis=1, ascending=False)
        .T.sort_index()
    )
    ax2 = sb.heatmap(vol_rank, linewidth=2.5, annot=True, ax=axes[1], cmap="rocket")
    ax2.set(xlabel="Year", ylabel="Ticker", xticklabels=vol_rank.columns.year)
    ax2.set_yticklabels(vol_rank.index, rotation=0)
    axes[1].set_title("Annual Rolling Volatility Ranking", size=23)
    plt.tight_layout()
    plt.show()


def plot_signal_returns(
    df_signal, df, df_return, industries,
    tickers_to_see, list_tickers_to_see,
    best_signal_set, period_to_start,
):
    """Interactive Plotly chart: per-ticker cumulative return under the chosen strategy."""
    df_tx = _build_tx_df(df_signal, best_signal_set[0], period_to_start)
    inds  = industry_color(industries, list(tickers_to_see))

    fig = go.Figure()
    for t in df_tx["Cumprod"][list(tickers_to_see)]:
        ret = pd.concat((
            pd.Series({df[:period_to_start].index[-2]: 1}),
            df_tx["Cumprod", t].dropna(),
        ))
        row = inds[inds["Ticker"] == t]
        fig.add_trace(go.Scatter(
            x=ret.index, y=ret,
            mode="lines+markers", name=t,
            legendrank=row["Colorid"].values[0],
            legendgroup=row["Industry"].values[0],
            legendgrouptitle_text=row["Industry"].values[0],
            line_color=row["Color"].values[0],
        ))

    fig.add_hrect(
        y0=1, y1=df_tx["Cumprod"][list(tickers_to_see)].min().min(),
        line_width=0, fillcolor="red", opacity=0.05,
    )
    fig.add_vline(x=best_signal_set[1], line_width=5, line_dash="dot", line_color="black")
    fig.add_shape(
        type="line",
        x0=df_tx.index[0], x1=df_tx.index[-1], y0=1, y1=1,
        line=dict(color="black", width=4, dash="dash"),
    )
    txt_rets = "<br>".join(
        f"{TTS_MAPPING[n]}: {df_tx['Cumprod'][list(list_tickers_to_see[n])].iloc[-2].mean():.2f}  |  "
        f"{TTS_MAPPING[n+1]}: {df_tx['Cumprod'][list(list_tickers_to_see[n+1])].iloc[-2].mean():.2f}"
        for n in range(0, len(list_tickers_to_see), 2)
    )
    fig.add_annotation(
        text=(f"Avg return ({df['Close'].shape[1]} tickers): "
              f"{df_tx['Cumprod'].iloc[-2].mean():.2f}<br>{txt_rets}"),
        xref="paper", yref="paper", x=0, y=1, showarrow=False,
        font=dict(size=10), bgcolor="lightgrey", opacity=0.7,
    )
    fig.update_layout(
        title=(f"Signal Returns — Strategy {best_signal_set[0]}: "
               f"{STRATEGIES[best_signal_set[0]]}"),
        xaxis_title="Date", yaxis_title="Cumulative Return (log)",
        template="plotly_white", yaxis_type="log",
        legend=dict(font=dict(size=9.5)), width=950, height=750,
    )
    fig.show()
    return df_tx


def plot_portfolio_performance(
    returns, returns_benchmark, ten_year,
    weights_df, equal_w_df, df_return, rebalancing_table,
    tickers_to_see, period_to_start, nb_rolling, min_pos,
    min_w, max_w_val, cov_ma_period, best_signal_id,
):
    """
    Main performance chart displaying:
      - Rebalanced portfolio vs equal-weight (signal) vs equal-weight (all) vs benchmark
      - Fill between lines, drawdown annotation, position count labels, CAPM α/β
    """
    fig, ax = plt.subplots(figsize=(26.5, 9.5))

    # Align risk-free rate to the rebalancing frequency of returns
    rf          = ten_year.reindex(returns.index).ffill().fillna(0)
    excess_rets = (returns - rf).dropna()
    rf_b        = ten_year.reindex(returns_benchmark.index).ffill().fillna(0)
    mkt_premium = (returns_benchmark - rf_b).dropna()
    # Align the two series to a common index before OLS
    common      = excess_rets.index.intersection(mkt_premium.index)
    capm        = sm.OLS(excess_rets.loc[common],
                         sm.add_constant(mkt_premium.loc[common])).fit()
    alpha_beta  = capm.params.values

    prev_idx = df_return.loc[:period_to_start].iloc[-1].name
    reb  = pd.concat([pd.Series({prev_idx: 1}),
                      (1 + (df_return.loc[period_to_start:].shift(-1) * weights_df).sum(axis=1)).cumprod()])
    eqw  = pd.concat([pd.Series({prev_idx: 1}),
                      (1 + (equal_w_df * df_return[period_to_start:].shift(-1)).sum(axis=1)).cumprod()])
    ini  = pd.concat([pd.Series({prev_idx: 1}),
                      (1 + ((1 / df_return.shape[1]) * df_return.loc[period_to_start:].shift(-1)).sum(axis=1)).cumprod()])
    mkt  = pd.concat([pd.Series({prev_idx: 1}), (returns_benchmark + 1).cumprod()])

    series = [reb, eqw, ini, mkt]
    labels = [
        f"Rebalanced       σ={reb.pct_change().std():.2%}",
        f"Equal-w + signal σ={eqw.pct_change().std():.2%}",
        f"Equal-w          σ={ini.pct_change().std():.2%}",
        f"Benchmark        σ={mkt.pct_change().std():.2%}",
    ]
    colors = ["orangered", "navy", "orange", "cornflowerblue"]
    styles = ["-", "-", "-", "-."]

    for s, lab, c, ls in zip(series, labels, colors, styles):
        s.plot(ax=ax, label=lab, color=c, linestyle=ls)

    maxv = max(s.max() for s in series) + 0.1
    minv = min(s.min() for s in series) - 0.05
    ax.fill_between(mkt.index, mkt, maxv, color="limegreen", alpha=0.08)
    ax.fill_between(mkt.index, mkt, minv, color="red",       alpha=0.08)
    ax.fill_between(reb.index, reb, eqw, interpolate=True,
                    where=reb > eqw, color="chartreuse", alpha=0.175)
    ax.fill_between(reb.index, reb, eqw, interpolate=True,
                    where=reb < eqw, color="red",         alpha=0.155)
    ax.set_ylim(minv, maxv)
    ax.set_xlim(mkt.index[0], mkt.index[-1])
    ax.axhline(y=1, color="black", ls="--", lw=2)

    # Max drawdown
    cum   = np.cumprod(1 + returns)
    peaks = np.maximum.accumulate(cum)
    dd    = (cum - peaks) / peaks
    max_dd, max_dd_date = dd.min(), pd.to_datetime(dd.idxmin())

    # Annotations
    for s, c in zip(series, colors):
        ax.annotate(f"{s.iloc[-1]:.2f}", xy=(s.index[-2], s.iloc[-1]),
                    xytext=(20, 5), textcoords="offset points", color=c, bbox=BBOX)

    for idx, val in rebalancing_table[str(period_to_start):df_return.shift(-1).index.max()].count(axis=1).items():
        ax.annotate(str(val), xy=(idx, minv), xytext=(-2, 5),
                    textcoords="offset points",
                    color="crimson" if val <= min_pos else "darkgreen")

    ax.annotate(f"{reb.min():.2f}", xy=(reb.idxmin(), reb.min()),
                xytext=(reb.idxmin(), reb.min() + 0.2),
                arrowprops=dict(facecolor="red",   arrowstyle="->"), fontsize=12, color="red")
    ax.scatter(reb.idxmin(), reb.min(), color="red",   marker="v")
    ax.annotate(f"{reb.max():.3f}", xy=(reb.idxmax(), reb.max()),
                xytext=(reb.idxmax(), reb.max() * 0.935),
                arrowprops=dict(facecolor="green", arrowstyle="->"), fontsize=12, color="green")
    ax.scatter(reb.idxmax(), reb.max(), color="green", marker="^")
    ax.annotate(f"{max_dd:.2%}", xy=(max_dd_date, reb[max_dd_date]),
                xytext=(max_dd_date, reb[max_dd_date] - 0.15),
                arrowprops=dict(facecolor="blue",  arrowstyle="->"), fontsize=10, color="blue")

    box_y = dict(boxstyle="round,pad=0.2", facecolor="yellow", alpha=0.5)
    ax.text(0.11, 0.75,
            f"Avg return/period: {returns.mean():.2%}\nMax drawdown: {max_dd:.0%}",
            transform=ax.transAxes, fontsize=10.5, bbox=box_y, ha="center", va="center")
    box_o = dict(boxstyle="round,pad=0.2", facecolor="orange", alpha=0.5)
    sig   = "$^{**}$" if capm.pvalues.iloc[0] < 0.03 else ""
    ax.text(0.5, 0.95,
            f"β: {alpha_beta[1]:.2f}   α{sig}: {alpha_beta[0]:.2f}",
            transform=ax.transAxes, fontsize=10.5, bbox=box_o, ha="center", va="center")

    avg_pos = rebalancing_table[str(period_to_start):].count(axis=1).mean()
    ax.set_ylabel("Cumulative return")
    ax.set_title(
        f"Portfolio performance  —  {len(weights_df)} periods  —  {len(tickers_to_see)} tickers  "
        f"—  avg {avg_pos:.1f} positions\n"
        f"W{get_sub('mn')}={min_w:.2%}  W{get_sub('max')}={max_w_val:.2%}  "
        f"cov window={cov_ma_period}  |  Strategy: {STRATEGIES[best_signal_id]}",
        fontsize=14,
    )
    fig.set_facecolor("snow")
    ax.legend(shadow=True, ncols=2, fancybox=True, title="Legend")
    ax.grid(True, alpha=0.55, ls=":")
    plt.tight_layout()
    plt.show()


def plot_returns_per_share(weights_df, df_return, tickers_to_see, nb_rolling):
    """Scatter plot: per-share cumulative return vs annualised volatility."""
    ret_ps  = (weights_df.fillna(0) * df_return.loc[:, list(tickers_to_see)].shift(-1)).dropna()
    shares  = pd.DataFrame({
        "Vol": ret_ps.std() * np.sqrt(nb_rolling),
        "Ret": (ret_ps + 1).prod(),
    }).pipe(lambda d: d[d["Vol"] != 0])
    shares["Sharpe"] = shares["Ret"] / shares["Vol"]

    plt.figure(figsize=(18.5, 4.5))
    x, y   = shares["Vol"], shares["Ret"]
    scaler = MinMaxScaler(feature_range=(10, 16))
    sizes  = scaler.fit_transform(y.values.reshape(-1, 1)).flatten()
    plt.scatter(x, y, marker="o", color="steelblue")
    top = shares[shares["Ret"] > 1]["Sharpe"].idxmax()
    plt.scatter(shares.loc[top, "Vol"], shares.loc[top, "Ret"], color="red",
                zorder=5, label="Best Sharpe")
    y_lim = 1 + (plt.gca().get_ylim()[1] - 1) / 50
    for i, txt in enumerate(shares.index):
        plt.annotate(txt, (x.iloc[i], y_lim * y.iloc[i]), size=sizes[i])
    plt.axhline(y=1, ls="--", color="black")
    plt.xlabel("Annualised σ")
    plt.ylabel("Cumulative Return")
    plt.title("Return vs Volatility per Share")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_returns_share_industry(weights_df, df_return, tickers_to_see, industries):
    """Two panels: per-share cumulative return + per-industry cumulative return."""
    fig, axes = plt.subplots(1, 2, figsize=(18.5, 5.5))
    tickers = list(tickers_to_see)

    # Panel 1 — per share
    ((weights_df.fillna(0) * df_return.loc[:, tickers].shift(-1)).dropna().add(1)
     .cumprod()).plot(ax=axes[0])
    axes[0].axhline(y=1, color="black", ls="--")
    count_pos = weights_df[weights_df > 0.01].count()
    axes[0].legend([f"{c}: {count_pos[c]}" for c in count_pos.index],
                   loc="upper left", fontsize="x-small")
    axes[0].set_title("Return per Share (# buy signals)")
    axes[0].set_yscale("linear")

    # Panel 2 — per industry
    ind_gains = (
        (weights_df.fillna(0) * df_return.loc[:, tickers].shift(-1)).dropna()
        .reset_index().melt(id_vars="Date", var_name="Ticker", value_name="Return")
        .merge(industries, on="Ticker")
        .groupby(["Industry", "Date"]).sum()["Return"]
        .unstack(0)
        .pipe(lambda d: (1 + d).cumprod())
    )
    ind_gains.plot(ax=axes[1])
    ind_count = industries[industries["Ticker"].isin(tickers)]["Industry"].value_counts()
    axes[1].legend(
        [f"{ind}: {ind_count.get(ind, 0)} stocks" for ind in ind_gains.columns],
        fontsize="x-small",
    )
    axes[1].axhline(y=1, color="black", ls="--")
    axes[1].set_title("Return per Industry")
    plt.tight_layout()
    plt.show()


def market_plot_3d(returns, returns_benchmark):
    """3-D surface: cumulative return as a function of allocation weight × time."""
    dates  = np.arange(len(returns))
    w      = np.arange(0, 1.01, 0.2)
    d_msh, w_msh = np.meshgrid(dates, w)

    def _ret(wt, dt):
        return wt * returns.iloc[dt] + (1 - wt) * returns_benchmark.iloc[dt]

    rets = np.array([_ret(wi, di) for wi, di in zip(w_msh.ravel(), d_msh.ravel())]).reshape(d_msh.shape)

    fig = plt.figure(figsize=(23, 11.5))
    ax  = fig.add_subplot(111, projection="3d")
    ax.plot_surface(w_msh, d_msh, (1 + rets).cumprod(axis=1),
                    cmap="turbo", edgecolor="black", alpha=0.45)
    ax.set_xlabel("Portfolio Allocation")
    ax.set_ylabel("Period")
    ax.set_zlabel("Cumulative Return")
    ax.set_title("3D: Allocation × Time → Cumulative Return")
    ax.xaxis.set_major_formatter("{x:.0%}")
    ax.view_init(elev=20, azim=-40)
    plt.tight_layout()
    plt.show()


def analyze_alpha_signal(positions, df_return, title, nb_rolling, lags=(0, 1, 2), short=False):
    """Plot cumulative PnL curves for a given alpha across multiple lags."""
    pos = alpha_to_positions(positions, short)
    fig = go.Figure()
    for lag in lags:
        pnl, sharpe = pnl_analytics(pos, df_return, lag, nb_rolling)
        fig.add_trace(go.Scatter(
            x=pnl.index, y=(1 + pnl).cumprod(),
            name=f"Lag {lag}  |  Sharpe {sharpe:.2f}",
        ))
    fig.update_layout(title=title, template="plotly_white",
                      xaxis_title="Date", yaxis_title="Cumulative PnL")
    fig.show()
