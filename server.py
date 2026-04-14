#!/usr/bin/env python3
"""
Smart Portfolio Manager - FastAPI Backend
==========================================
Place this file in the SAME directory as sprv2.py (your Pandas/ folder).

Install deps (once):
    pip install fastapi "uvicorn[standard]" yfinance statsmodels

Run:
    uvicorn server:app --host 0.0.0.0 --port 8000 --reload

On a real iPhone (not Simulator), replace 127.0.0.1 in APIService.swift
with your Mac's local IP  e.g.  http://192.168.1.42:8000
"""

import sys
import uuid
import threading
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.show = lambda *a, **kw: None

warnings.filterwarnings("ignore")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Path setup ─────────────────────────────────────────────────────────────────
ROOT   = Path(__file__).resolve().parent       # same dir as sprv2.py
SP_CSV = str(ROOT / "csv" / "SPfull.csv")
sys.path.insert(0, str(ROOT))
import sprv2 as sp                             # noqa: E402  (path must be set first)

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(title="Smart Portfolio Manager API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory job store ────────────────────────────────────────────────────────
jobs: dict = {}


# ── Request model  (CodingKeys must match Swift AnalysisConfig exactly) ────────
class AnalysisConfig(BaseModel):
    mode:                str       = "sp500"
    interval:            str       = "1mo"
    period:              str       = "4Y"
    benchmark_ticker:    str       = "^GSPC"
    benchmark_label:     str       = "S&P 500"
    sample_per_industry: int       = 3
    only_industries:     list      = []
    exclude_industries:  list      = ["Energy", "Real Estate"]
    elements_to_remove:  list      = ["TSLA"]
    custom_tickers:      list      = []
    period_to_start:     str       = "2022-01-01"
    nombre_tickers:      int       = 10
    nb_rolling:          int       = 12
    cov_ma_period:       int       = 8
    criteria:            str       = "sharpe"
    test_size:           float     = 0.55
    best_signal_opt:     str       = "default"
    objective:           str       = "sortino"
    min_pos:             int       = 3
    min_w:               float     = 0.047
    max_w_ratio:         float     = 2.86
    corr_constraint:     bool      = True
    max_corr:            float     = 0.40


# ──────────────────────────────────────────────────────────────────────────────
# ENDPOINTS
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/industries")
def industries():
    try:
        df = pd.read_csv(SP_CSV, sep=";")
        return sorted(df["Industry"].dropna().unique().tolist())
    except Exception:
        return []


@app.post("/analyze/start")
def start_analysis(config: AnalysisConfig):
    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "status": "queued",
        "step": "Queued...",
        "progress": 0.0,
        "result": None,
        "error": None,
    }
    t = threading.Thread(target=_run_analysis, args=(job_id, config), daemon=True)
    t.start()
    return {"job_id": job_id}


@app.get("/analyze/status/{job_id}")
def job_status(job_id: str):
    return jobs.get(job_id, {
        "status": "error",
        "step": "Job not found",
        "progress": 0,
        "result": None,
        "error": "Job not found",
    })


# ──────────────────────────────────────────────────────────────────────────────
# BACKGROUND WORKER
# ──────────────────────────────────────────────────────────────────────────────

def _upd(jid, step, prog):
    jobs[jid].update(status="running", step=step, progress=prog)


def _run_analysis(job_id: str, cfg: AnalysisConfig):
    try:
        import yfinance as yf

        # Step 1 - Load universe
        if cfg.mode == "custom":
            ticker_preview = ", ".join(cfg.custom_tickers[:5])
            if len(cfg.custom_tickers) > 5:
                ticker_preview += f" +{len(cfg.custom_tickers) - 5} more"
            _upd(job_id, f"Downloading {ticker_preview}…", 0.05)
            raw = yf.download(cfg.custom_tickers, period=cfg.period, interval=cfg.interval, progress=False)
            raw.index = pd.to_datetime(raw.index)
            raw = sp.clean_df(raw)
            raw = sp.trim_incomplete_period(raw, cfg.interval)

            _upd(job_id, "Downloading index & risk-free rate…", 0.12)
            mkt = yf.download("^GSPC", interval=cfg.interval, progress=False).dropna()
            if isinstance(mkt.columns, pd.MultiIndex):
                mkt.columns = mkt.columns.droplevel(1)
            mkt["Chg"] = mkt["Close"].pct_change()

            tnx = yf.download("^TNX", progress=False).dropna()
            if isinstance(tnx.columns, pd.MultiIndex):
                tnx.columns = tnx.columns.droplevel(1)
            ten_year = tnx.reindex(mkt.index)["Close"].div(100).ffill()

            mkt       = mkt.reindex(raw.index)
            df_return = raw["Close"].pct_change()
            tickers   = raw["Close"].columns.tolist()
            industries_df = pd.DataFrame({
                "Ticker":   tickers,
                "Industry": ["Custom Portfolio"] * len(tickers),
            })
        else:
            _upd(job_id, f"Downloading S&P 500 universe ({cfg.period})…", 0.05)
            raw, mkt, ten_year, df_return, tickers, industries_df = sp.load_universe(
                sp_csv=SP_CSV,
                interval=cfg.interval,
                period=cfg.period,
                sample_per_industry=cfg.sample_per_industry,
                exclude_industries=cfg.exclude_industries,
                only_industries=cfg.only_industries,
                elements_to_remove=cfg.elements_to_remove,
            )

        # Step 2 - Ticker selection
        n_universe = raw["Close"].shape[1]
        _upd(job_id, f"Selecting top {cfg.nombre_tickers} from {n_universe} tickers…", 0.20)
        plt.close("all")
        tickers_to_see, tts, list_tts = sp.plot_ticker_configs(
            df=raw,
            df_return=df_return,
            market_df=mkt,
            tickers_to_see=raw["Close"].columns,
            nombre_tickers=cfg.nombre_tickers,
            period_to_start=cfg.period_to_start,
            nb_rolling=cfg.nb_rolling,
            list_tts="default",
        )
        plt.close("all")

        # Step 3 - Signals
        _upd(job_id, f"Computing signals for {len(list(tickers_to_see))} tickers…", 0.38)
        plt.close("all")
        df_signal = sp.compute_signals(raw, mkt, list(tickers_to_see), cfg.nb_rolling)
        plt.close("all")

        # Step 4 - Best strategy
        _upd(job_id, "Evaluating 17 strategies...", 0.54)
        plt.close("all")
        best_signal_set = sp.find_best_signal(
            df_signal, list(tickers_to_see),
            cfg.period_to_start, cfg.nombre_tickers, cfg.nb_rolling,
            cfg.best_signal_opt, cfg.criteria, cfg.test_size, trace=False,
        )
        plt.close("all")
        df_signal = sp.assign_signals(df_signal, list(tickers_to_see), cfg.nombre_tickers)

        # Step 5 - Rebalancing table
        _upd(job_id, "Building rebalancing table...", 0.65)
        rebalancing_table = sp.create_rebalancing_table(
            df_signal, list(tickers_to_see), best_signal_set[0], cfg.period_to_start
        )

        # Step 6 - Optimisation
        _upd(job_id, "Optimising walk-forward weights...", 0.75)
        weights_df, equal_w_df = sp.run_optimization(
            df=raw, df_return=df_return, market_df=mkt,
            rebalancing_table=rebalancing_table,
            tickers_to_see=tickers_to_see,
            period_to_start=cfg.period_to_start,
            min_pos=cfg.min_pos, min_w=cfg.min_w, max_w_ratio=cfg.max_w_ratio,
            cov_ma_period=cfg.cov_ma_period, nb_rolling=cfg.nb_rolling,
            objective=cfg.objective,
            corr_constraint=cfg.corr_constraint, max_corr=cfg.max_corr,
        )

        # Compute returns & benchmark
        _upd(job_id, "Computing performance metrics...", 0.90)
        returns = (df_return.loc[cfg.period_to_start:].shift(-1) * weights_df).sum(axis=1)

        bm_raw = yf.download(cfg.benchmark_ticker, interval=cfg.interval, period=cfg.period, progress=False).dropna()
        if isinstance(bm_raw.columns, pd.MultiIndex):
            bm_raw.columns = bm_raw.columns.droplevel(1)
        returns_benchmark = (
            bm_raw["Close"].pct_change()
            .reindex(returns.index, method="ffill")
            .fillna(0)
        )

        n_pos     = rebalancing_table[cfg.period_to_start:].count(axis=1).replace(0, np.nan).mean()
        max_w_val = sp._max_w_bound(int(n_pos), cfg.max_w_ratio)

        result = _build_result(
            cfg=cfg,
            raw=raw, df_return=df_return, ten_year=ten_year,
            tickers_to_see=tickers_to_see, tts=tts, list_tts=list_tts,
            industries_df=industries_df,
            df_signal=df_signal, best_signal_set=best_signal_set,
            rebalancing_table=rebalancing_table,
            weights_df=weights_df, equal_w_df=equal_w_df,
            returns=returns, returns_benchmark=returns_benchmark,
            max_w_val=max_w_val,
        )

        jobs[job_id] = {
            "status":   "done",
            "step":     "Analysis complete",
            "progress": 1.0,
            "result":   result,
            "error":    None,
        }

    except Exception as exc:
        jobs[job_id] = {
            "status":   "error",
            "step":     "Error",
            "progress": 0.0,
            "result":   None,
            "error":    f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
        }


# ──────────────────────────────────────────────────────────────────────────────
# RESULT BUILDER  -  must match Swift AnalysisResult / CodingKeys exactly
# ──────────────────────────────────────────────────────────────────────────────

def _build_result(cfg, raw, df_return, ten_year,
                  tickers_to_see, tts, list_tts, industries_df,
                  df_signal, best_signal_set, rebalancing_table,
                  weights_df, equal_w_df, returns, returns_benchmark, max_w_val):

    pts = cfg.period_to_start
    nb  = cfg.nb_rolling

    # Summary stats
    cum_reb = float((1 + returns).prod())
    cum_mkt = float((1 + returns_benchmark).prod())
    cum_eqw = float((1 + (equal_w_df * df_return[pts:].shift(-1)).sum(axis=1)).prod())
    sharpe  = float(returns.mean() / returns.std() * np.sqrt(nb)) if returns.std() > 0 else 0.0
    cum_arr = np.cumprod(1 + returns.values)
    peaks   = np.maximum.accumulate(cum_arr)
    max_dd  = float(((cum_arr - peaks) / peaks).min())
    avg_pos = float(rebalancing_table[pts:].count(axis=1).mean())

    # Anchor point for cumulative curves
    _before  = df_return.loc[:pts]
    prev_idx = _before.iloc[-1].name if len(_before) > 0 else df_return.index[0]

    def _curve(series):
        c = pd.concat([pd.Series({prev_idx: 1.0}), (1 + series).cumprod()])
        return [{"date": str(d.date()), "value": float(v)} for d, v in c.items()]

    reb_ser = (df_return.loc[pts:].shift(-1) * weights_df).sum(axis=1)
    eqw_ser = (equal_w_df * df_return[pts:].shift(-1)).sum(axis=1)
    ini_ser = (1 / df_return.shape[1]) * df_return.loc[pts:].shift(-1).sum(axis=1)

    # Current allocation
    curr_w       = weights_df.iloc[-1]
    curr_w       = curr_w[curr_w > 0.001].sort_values(ascending=False)
    current_alloc = [{"ticker": t, "weight": float(w)} for t, w in curr_w.items()]

    # Industry counts
    ic = industries_df["Industry"].value_counts().reset_index()
    ic.columns = ["industry", "count"]

    # Ticker selection strategies (10 strategies)
    ticker_strategies = []
    for num, tlist in enumerate(list_tts):
        ret      = df_return.loc[:, tlist]
        cum      = ((1 / ret.shape[1]) * ret.loc[pts:]).sum(axis=1).add(1).cumprod()
        cum_full = pd.concat([pd.Series({prev_idx: 1.0}), cum])
        ticker_strategies.append({
            "id":           num,
            "name":         sp.TTS_MAPPING[num],
            "final_return": float(cum_full.iloc[-1]),
            "curve":        [{"date": str(d.date()), "value": float(v)}
                             for d, v in cum_full.items()],
        })

    # Signal strategy ranking (17 strategies)
    signal_strategies = []
    for strat in sp.STRATEGIES:
        try:
            tx  = sp._build_tx_df(df_signal, strat, pts)
            r   = tx.loc[pts:, "Cumprod"][list(tickers_to_see)].iloc[-2].mean()
            vol = tx.loc[pts:, "Return"][list(tickers_to_see)].std().mean()
        except Exception:
            r, vol = 1.0, 0.01
        sh = float((r - 1) / vol) if vol > 0 else 0.0
        signal_strategies.append({
            "id":     strat,
            "name":   sp.STRATEGIES[strat],
            "return": float(r),
            "vol":    float(vol),
            "sharpe": sh,
        })

    # Recent rebalancing periods (last 10)
    reb_rows = []
    for date, row in rebalancing_table.tail(10).iterrows():
        tks = [t for t in row if pd.notna(t)]
        reb_rows.append({"date": str(date.date()), "tickers": tks})

    # Drawdown & rolling Sharpe
    dd_series = (cum_arr - peaks) / peaks
    roll_win  = max(nb, 8)
    rs = (returns.rolling(roll_win).mean() / returns.rolling(roll_win).std()) * np.sqrt(nb)

    # Per-share contribution
    ind_map = dict(zip(industries_df["Ticker"], industries_df["Industry"]))
    ret_ps  = (weights_df.fillna(0) * df_return.loc[:, list(tickers_to_see)].shift(-1)).dropna()
    per_share = []
    for t in ret_ps.columns:
        s_ret = ret_ps[t]
        cum_r = float((1 + s_ret).prod())
        ann_v = float(s_ret.std() * np.sqrt(nb))
        sh    = float((cum_r - 1) / ann_v) if ann_v > 0 else 0.0
        per_share.append({
            "ticker":     t,
            "industry":   ind_map.get(t, "Custom"),
            "cum_return": cum_r,
            "ann_vol":    ann_v,
            "sharpe":     sh,
        })

    # CAPM (optional — silently skipped if statsmodels not installed)
    capm_result = None
    try:
        import statsmodels.api as sm
        rf     = ten_year.reindex(returns.index).ffill().fillna(0)
        rf_b   = ten_year.reindex(returns_benchmark.index).ffill().fillna(0)
        exc    = (returns - rf).dropna()
        mktp   = (returns_benchmark - rf_b).dropna()
        common = exc.index.intersection(mktp.index)
        fit    = sm.OLS(exc.loc[common], sm.add_constant(mktp.loc[common])).fit()
        alpha_v, beta_v = fit.params.values
        pval = float(fit.pvalues.iloc[0])
        capm_result = {
            "alpha":        float(alpha_v),
            "beta":         float(beta_v),
            "r_squared":    float(fit.rsquared),
            "p_value":      pval,
            "significant":  bool(pval < 0.05),
            "observations": int(len(common)),
            "scatter":      [{"x": float(x), "y": float(y)}
                             for x, y in zip(mktp.loc[common].values[:200],
                                             exc.loc[common].values[:200])],
        }
    except Exception:
        pass

    # Returns series (date, portfolio, benchmark)
    returns_series = [
        {"date": str(d.date()), "portfolio": float(p), "benchmark": float(b)}
        for d, p, b in zip(returns.index, returns.values, returns_benchmark.values)
    ]

    # Orders
    prev_w_raw   = weights_df.iloc[-2] if len(weights_df) > 1 else pd.Series(dtype=float)
    prev_w_clean = prev_w_raw[prev_w_raw > 0.001] if len(prev_w_raw) > 0 else pd.Series(dtype=float)

    orders = []
    for t in sorted(set(curr_w.index) | set(prev_w_clean.index)):
        cw = float(curr_w.get(t, 0.0))
        pw = float(prev_w_clean.get(t, 0.0))
        if   pw == 0 and cw > 0:    action = "BUY"
        elif cw == 0 and pw > 0:    action = "SELL"
        elif abs(cw - pw) < 0.005:  action = "HOLD"
        elif cw > pw:               action = "INCREASE"
        else:                       action = "REDUCE"
        orders.append({
            "ticker":        t,
            "action":        action,
            "prev_weight":   pw,
            "target_weight": cw,
            "delta":         cw - pw,
        })

    ac = {a: sum(1 for o in orders if o["action"] == a)
          for a in ("BUY", "SELL", "INCREASE", "REDUCE", "HOLD")}

    # Expected portfolio metrics
    sel_cov = df_return.loc[:, list(curr_w.index)].dropna()
    exp_ret = float((curr_w * sel_cov.mean()).sum())
    cov_m   = sel_cov.cov().values
    exp_vol = float(np.sqrt(curr_w.values @ cov_m @ curr_w.values) * np.sqrt(nb))
    exp_sr  = float(exp_ret / exp_vol) if exp_vol > 0 else 0.0

    # Weight history (last 8 periods)
    weight_history = []
    for idx, row in weights_df.tail(8).iterrows():
        w_dict = {t: float(w) for t, w in row.items() if w > 0.001}
        weight_history.append({"date": str(idx.date()), "weights": w_dict})

    # Final assembled response
    return {
        "meta": {
            "period_to_start":  pts,
            "nb_rolling":       nb,
            "benchmark_label":  cfg.benchmark_label,
            "objective":        cfg.objective,
            "best_signal_id":   int(best_signal_set[0]),
            "best_signal_name": sp.STRATEGIES[best_signal_set[0]],
            "tts_id":           int(tts),
            "tts_name":         sp.TTS_MAPPING[tts],
            "max_w_val":        float(max_w_val),
        },
        "dashboard": {
            "cum_reb":            cum_reb,
            "cum_mkt":            cum_mkt,
            "cum_eqw":            cum_eqw,
            "sharpe":             sharpe,
            "max_dd":             max_dd,
            "avg_pos":            avg_pos,
            "reb_curve":          _curve(reb_ser),
            "mkt_curve":          _curve(returns_benchmark),
            "eqw_curve":          _curve(eqw_ser),
            "ini_curve":          _curve(ini_ser),
            "current_allocation": current_alloc,
        },
        "universe": {
            "total_tickers":    int(raw["Close"].shape[1]),
            "total_industries": int(industries_df["Industry"].nunique()),
            "start_date":       str(raw.index[0].date()),
            "end_date":         str(raw.index[-1].date()),
            "industry_counts":  ic.to_dict(orient="records"),
            "selected_tickers": list(tickers_to_see),
        },
        "signals": {
            "ticker_strategies":  ticker_strategies,
            "signal_strategies":  signal_strategies,
            "best_signal_id":     int(best_signal_set[0]),
            "best_signal_name":   sp.STRATEGIES[best_signal_set[0]],
            "tts_id":             int(tts),
            "rebalancing_table":  reb_rows,
        },
        "performance": {
            "cum_return":      float((1 + returns).prod()),
            "cum_mkt":         cum_mkt,
            "vs_benchmark":    float((1 + returns).prod() / cum_mkt - 1),
            "sharpe":          sharpe,
            "max_dd":          max_dd,
            "pct_beating":     float((returns > returns_benchmark).mean()),
            "returns_series":  returns_series,
            "drawdown_series": [{"date": str(d.date()), "value": float(v)}
                                for d, v in zip(returns.index, dd_series)],
            "rolling_sharpe":  [{"date": str(d.date()), "value": float(v)}
                                for d, v in rs.dropna().items()],
            "capm":            capm_result,
            "per_share":       per_share,
        },
        "rebalancing": {
            "next_date":       str(weights_df.index[-1].date()),
            "strategy_id":     int(best_signal_set[0]),
            "strategy_name":   sp.STRATEGIES[best_signal_set[0]],
            "objective":       cfg.objective,
            "positions":       int(len(curr_w)),
            "new_buys":        int(ac["BUY"]),
            "sells":           int(ac["SELL"]),
            "adjustments":     int(ac["INCREASE"] + ac["REDUCE"]),
            "holds":           int(ac["HOLD"]),
            "expected_return": exp_ret,
            "expected_vol":    exp_vol,
            "expected_sharpe": exp_sr,
            "orders":          orders,
            "weight_history":  weight_history,
        },
    }


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
