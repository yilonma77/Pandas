import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
TICKERS  = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA"]
CLIENTS  = ["Client A", "Client B", "Client C", "Client D"]
N_QUOTES = 1000   # simulated quotes per (client, ticker)
SEED     = 42

st.set_page_config(
    page_title="Allocation Optimizer – Market Maker",
    layout="wide",
    page_icon="📊",
)

st.title("📊 Market Maker — Allocation Optimizer")
st.caption("Simulez l'impact de vos allocations par client/ticker sur les hit ratios.")

# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────
def simulate_hits(
    alloc_df: pd.DataFrame,
    buy_disp: dict,
    sell_disp: dict,
    n: int = N_QUOTES,
    seed: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Simulate hit events.

    For each (client, ticker):
        alloc  = allocation score [0–100] from market maker
        p_buy  = buy_disp[client]  * (alloc / 100)
        p_sell = sell_disp[client] * (alloc / 100)
        Each simulated quote independently triggers a buy-hit, sell-hit, or nothing.

    Returns:
        hit_ratio_df   – total hit ratio  (hits / n_quotes)
        buy_hit_df     – buy  hit ratio
        sell_hit_df    – sell hit ratio
    """
    rng = np.random.default_rng(seed)

    hit_ratio_df  = pd.DataFrame(index=CLIENTS, columns=TICKERS, dtype=float)
    buy_hit_df    = pd.DataFrame(index=CLIENTS, columns=TICKERS, dtype=float)
    sell_hit_df   = pd.DataFrame(index=CLIENTS, columns=TICKERS, dtype=float)

    for client in CLIENTS:
        for ticker in TICKERS:
            alloc  = alloc_df.loc[client, ticker] / 100.0
            p_buy  = min(buy_disp[client]  / 100.0 * alloc, 1.0)
            p_sell = min(sell_disp[client] / 100.0 * alloc, 1.0)

            u = rng.uniform(0, 1, n)
            buy_hits  = (u < p_buy).sum()
            sell_hits = ((u >= p_buy) & (u < p_buy + p_sell)).sum()
            total     = buy_hits + sell_hits

            hit_ratio_df.loc[client, ticker]  = round(total   / n * 100, 2)
            buy_hit_df.loc[client, ticker]    = round(buy_hits / n * 100, 2)
            sell_hit_df.loc[client, ticker]   = round(sell_hits / n * 100, 2)

    return hit_ratio_df.astype(float), buy_hit_df.astype(float), sell_hit_df.astype(float)


def heatmap_fig(df: pd.DataFrame, title: str, colorscale: str = "Blues") -> go.Figure:
    fig = go.Figure(go.Heatmap(
        z=df.values,
        x=df.columns.tolist(),
        y=df.index.tolist(),
        colorscale=colorscale,
        text=df.values.round(1),
        texttemplate="%{text}%",
        textfont={"size": 13},
        hoverongaps=False,
        colorbar=dict(title="Hit %"),
        zmin=0,
        zmax=df.values.max() + 5,
    ))
    fig.update_layout(
        title=title,
        height=300,
        margin=dict(t=40, b=10, l=10, r=10),
        xaxis_title="Ticker",
        yaxis_title="Client",
    )
    return fig


# ──────────────────────────────────────────────
# SIDEBAR — CLIENT DISPOSITIONS
# ──────────────────────────────────────────────
st.sidebar.header("📋 Dispositions clients")
st.sidebar.caption("Tendance naturelle du client à acheter / vendre (ne doit pas sommer à 100 %).")

buy_disp  = {}
sell_disp = {}

DEFAULTS = {
    "Client A": (30, 40),
    "Client B": (50, 20),
    "Client C": (15, 55),
    "Client D": (35, 35),
}

for client in CLIENTS:
    b_def, s_def = DEFAULTS[client]
    st.sidebar.markdown(f"**{client}**")
    c1, c2 = st.sidebar.columns(2)
    buy_disp[client]  = c1.number_input(
        "Long %", min_value=0, max_value=100, value=b_def,
        key=f"buy_{client}", step=5,
    )
    sell_disp[client] = c2.number_input(
        "Short %", min_value=0, max_value=100, value=s_def,
        key=f"sell_{client}", step=5,
    )
    st.sidebar.divider()

# ──────────────────────────────────────────────
# MAIN — ALLOCATION MATRIX
# ──────────────────────────────────────────────
st.subheader("🎛️ Matrice d'allocation (score 0–100)")
st.caption(
    "L'allocation représente l'agressivité de votre quote pour chaque couple client/ticker. "
    "100 = quote au max de compétitivité → probabilité de hit maximale."
)

DEFAULT_ALLOC = pd.DataFrame(
    {
        "AAPL":  [80, 50, 30, 70],
        "MSFT":  [60, 80, 40, 50],
        "GOOGL": [40, 60, 80, 30],
        "AMZN":  [70, 40, 60, 90],
        "META":  [50, 70, 50, 40],
        "TSLA":  [90, 30, 70, 60],
    },
    index=CLIENTS,
)

# Editable allocation table
if "alloc_df" not in st.session_state:
    st.session_state["alloc_df"] = DEFAULT_ALLOC.copy()

edited = st.data_editor(
    st.session_state["alloc_df"],
    width="stretch",
    num_rows="fixed",
    column_config={
        t: st.column_config.NumberColumn(t, min_value=0, max_value=100, step=5)
        for t in TICKERS
    },
    key="alloc_editor",
)
st.session_state["alloc_df"] = edited

col_reset, col_run = st.columns([1, 5])
with col_reset:
    if st.button("↩️ Réinitialiser", width="stretch"):
        st.session_state["alloc_df"] = DEFAULT_ALLOC.copy()
        st.rerun()

# ──────────────────────────────────────────────
# SIMULATION
# ──────────────────────────────────────────────
hit_df, buy_df, sell_df = simulate_hits(
    edited, buy_disp, sell_disp, n=N_QUOTES, seed=SEED
)

# ──────────────────────────────────────────────
# HEATMAPS
# ──────────────────────────────────────────────
st.divider()
st.subheader("📈 Hit Ratios simulés")

t1, t2, t3 = st.tabs(["Total", "🟢 Buy hits", "🔴 Sell hits"])
with t1:
    st.plotly_chart(heatmap_fig(hit_df, "Hit ratio total (%)", "Teal"), width="stretch")
with t2:
    st.plotly_chart(heatmap_fig(buy_df, "Buy hits (%)", "Blues"), width="stretch")
with t3:
    st.plotly_chart(heatmap_fig(sell_df, "Sell hits (%)", "Reds"), width="stretch")

# ──────────────────────────────────────────────
# RÉSUMÉ PAR CLIENT ET PAR TICKER
# ──────────────────────────────────────────────
st.divider()
st.subheader("📊 Analyse agrégée")

col_a, col_b = st.columns(2)

with col_a:
    st.markdown("**Hit ratio moyen par client**")
    client_summary = pd.DataFrame({
        "Total %":     hit_df.mean(axis=1).round(2),
        "Buy %":       buy_df.mean(axis=1).round(2),
        "Sell %":      sell_df.mean(axis=1).round(2),
        "Long disp.":  [buy_disp[c]  for c in CLIENTS],
        "Short disp.": [sell_disp[c] for c in CLIENTS],
    })
    st.dataframe(
        client_summary.style.format("{:.2f}", subset=["Total %", "Buy %", "Sell %"]),
        width="stretch",
    )

    fig_client = px.bar(
        client_summary.reset_index().rename(columns={"index": "Client"}),
        x="Client", y=["Buy %", "Sell %"],
        barmode="stack", color_discrete_map={"Buy %": "#2196F3", "Sell %": "#F44336"},
        title="Stack Buy / Sell hits par client",
    )
    fig_client.update_layout(height=280, margin=dict(t=40, b=10))
    st.plotly_chart(fig_client, width="stretch")

with col_b:
    st.markdown("**Hit ratio moyen par ticker**")
    ticker_summary = pd.DataFrame({
        "Total %": hit_df.mean(axis=0).round(2),
        "Buy %":   buy_df.mean(axis=0).round(2),
        "Sell %":  sell_df.mean(axis=0).round(2),
    })
    st.dataframe(
        ticker_summary.style.format("{:.2f}", subset=["Total %", "Buy %", "Sell %"]),
        width="stretch",
    )

    fig_ticker = px.bar(
        ticker_summary.reset_index().rename(columns={"index": "Ticker"}),
        x="Ticker", y=["Buy %", "Sell %"],
        barmode="stack", color_discrete_map={"Buy %": "#2196F3", "Sell %": "#F44336"},
        title="Stack Buy / Sell hits par ticker",
    )
    fig_ticker.update_layout(height=280, margin=dict(t=40, b=10))
    st.plotly_chart(fig_ticker, width="stretch")

# ──────────────────────────────────────────────
# SENSIBILITÉ : IMPACT D'UNE VARIATION D'ALLOCATION
# ──────────────────────────────────────────────
st.divider()
st.subheader("🔬 Analyse de sensibilité")
st.caption("Observez l'impact d'un changement d'allocation sur le hit ratio total d'un couple client/ticker.")

col_s1, col_s2 = st.columns(2)
with col_s1:
    sens_client = st.selectbox("Client", CLIENTS, key="sens_client")
with col_s2:
    sens_ticker = st.selectbox("Ticker", TICKERS, key="sens_ticker")

alloc_range = np.arange(0, 105, 5)
total_hits, buy_hits_arr, sell_hits_arr = [], [], []

for alloc_val in alloc_range:
    tmp = edited.copy()
    tmp.loc[sens_client, sens_ticker] = alloc_val
    h, b, s = simulate_hits(tmp, buy_disp, sell_disp, n=N_QUOTES, seed=SEED)
    total_hits.append(h.loc[sens_client, sens_ticker])
    buy_hits_arr.append(b.loc[sens_client, sens_ticker])
    sell_hits_arr.append(s.loc[sens_client, sens_ticker])

fig_sens = go.Figure()
fig_sens.add_trace(go.Scatter(
    x=alloc_range, y=total_hits, mode="lines+markers",
    name="Total hit %", line=dict(color="#00BCD4", width=2)
))
fig_sens.add_trace(go.Scatter(
    x=alloc_range, y=buy_hits_arr, mode="lines",
    name="Buy hit %", line=dict(color="#2196F3", dash="dash")
))
fig_sens.add_trace(go.Scatter(
    x=alloc_range, y=sell_hits_arr, mode="lines",
    name="Sell hit %", line=dict(color="#F44336", dash="dash")
))
# Current allocation marker
curr_alloc = edited.loc[sens_client, sens_ticker]
curr_total  = hit_df.loc[sens_client, sens_ticker]
fig_sens.add_vline(
    x=curr_alloc, line_dash="dot", line_color="orange",
    annotation_text=f"Actuel: {int(curr_alloc)}", annotation_position="top right"
)
fig_sens.update_layout(
    title=f"Sensibilité hit ratio — {sens_client} / {sens_ticker}",
    xaxis_title="Allocation (score)",
    yaxis_title="Hit ratio (%)",
    height=350,
    legend=dict(orientation="h", y=1.1),
    margin=dict(t=60, b=30),
)
st.plotly_chart(fig_sens, width="stretch")

# ──────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────
st.divider()
st.caption(
    f"Simulation basée sur {N_QUOTES:,} quotes par couple (client, ticker). "
    "Les hit ratios sont stochastiques : le seed est fixé à "
    f"{SEED} pour la reproductibilité."
)
