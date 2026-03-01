# app.py — P2-ETF-DQN-ENGINE  Streamlit UI

import json
import os
import shutil
from datetime import datetime, date, timedelta, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import config

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="P2 ETF DQN Engine",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.main { background-color: #ffffff; color: #1a1a1a; }
div[data-testid="stMetric"] {
    background: #f8f9fa; border: 1px solid #e9ecef;
    border-radius: 10px; padding: 15px;
}
[data-testid="stMetricValue"] { color: #0066cc !important; font-size: 26px !important; font-weight: 700 !important; }
[data-testid="stMetricLabel"] { color: #6c757d !important; font-size: 11px !important; text-transform: uppercase; }
.hero-card {
    background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
    border: 2px solid #0066cc; border-radius: 16px;
    padding: 32px; text-align: center; margin-bottom: 24px;
}
.hero-label { color: #6c757d; font-size: 13px; text-transform: uppercase; letter-spacing: 2px; }
.hero-value { color: #0066cc; font-size: 72px; font-weight: 900; margin: 8px 0; line-height: 1; }
.hero-sub   { color: #495057; font-size: 14px; margin-top: 8px; }
.cash-card  {
    background: linear-gradient(135deg, #fff8f0 0%, #ffe4cc 100%);
    border: 2px solid #cc6600; border-radius: 16px;
    padding: 32px; text-align: center; margin-bottom: 24px;
}
.provenance { background: #f8f9fa; border-left: 4px solid #0066cc; padding: 10px 16px;
              border-radius: 4px; font-size: 13px; color: #6c757d; margin-top: 8px; }
.method-box { background: #ffffff; border: 1px solid #dee2e6; border-radius: 12px; padding: 20px; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_json(path: str) -> dict:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def _next_trading_day() -> date:
    """Uses NYSE calendar via pandas_market_calendars — no hardcoded holidays."""
    try:
        import pandas_market_calendars as mcal
        nyse    = mcal.get_calendar("NYSE")
        now_est = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=5)
        today   = now_est.date()
        sched   = nyse.schedule(
            start_date=today.strftime("%Y-%m-%d"),
            end_date=(today + timedelta(days=10)).strftime("%Y-%m-%d"),
        )
        trading_dates = [d.date() for d in mcal.date_range(sched, frequency="1D")]
        if trading_dates and trading_dates[0] == today and now_est.hour < 16:
            return today
        for d in trading_dates:
            if d > today:
                return d
    except Exception:
        pass
    now_est = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=5)
    d = now_est.date()
    if now_est.hour >= 16 or d.weekday() >= 5:
        d += timedelta(days=1)
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d


def _trigger_github(start_year: int, fee_bps: int,
                    tsl_pct: float, z_reentry: float) -> bool:
    try:
        import requests
        token = config.GITHUB_TOKEN if hasattr(config, "GITHUB_TOKEN") else os.getenv("GITHUB_TOKEN", "")
        if not token:
            st.error("❌ Debug: GITHUB_TOKEN is empty — check HF Space secrets.")
            return False
        url  = f"https://api.github.com/repos/{config.GITHUB_REPO}/actions/workflows/train_models.yml/dispatches"
        resp = requests.post(url,
            headers={"Authorization": f"token {token}",
                     "Accept": "application/vnd.github+json"},
            json={"ref": "main",
                  "inputs": {
                      "start_year": str(start_year),
                      "fee_bps":    str(fee_bps),
                      "tsl_pct":    str(tsl_pct),
                      "z_reentry":  str(z_reentry),
                  }},
            timeout=10,
        )
        if resp.status_code != 204:
            st.error(f"❌ Debug: GitHub API returned HTTP {resp.status_code} — {resp.text[:300]}")
        return resp.status_code == 204
    except Exception as e:
        st.error(f"❌ Debug: Exception — {str(e)}")
        return False
        url  = f"https://api.github.com/repos/{config.GITHUB_REPO}/actions/workflows/train_models.yml/dispatches"
        resp = requests.post(url,
            headers={"Authorization": f"token {token}",
                     "Accept": "application/vnd.github+json"},
            json={"ref": "main",
                  "inputs": {
                      "start_year": str(start_year),
                      "episodes":   str(episodes),
                      "fee_bps":    str(fee_bps),
                      "tsl_pct":    str(tsl_pct),
                      "z_reentry":  str(z_reentry),
                  }},
            timeout=10,
        )
        return resp.status_code == 204
    except Exception:
        return False


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    if st.button("🔄 Refresh Data & Clear Cache"):
        st.cache_data.clear()
        st.toast("Cache cleared — reloading...")
        st.rerun()

    st.divider()
    st.markdown("### 📅 Training Parameters")
    st.caption("Changes here require retraining via GitHub Actions.")

    start_year = st.slider("Start Year", config.START_YEAR_MIN if hasattr(config, "START_YEAR_MIN") else 2008,
                           2024, 2015)
    fee_bps    = st.number_input("T-Costs (bps)", 0, 50, 10)

    st.divider()
    st.markdown("### 🛡️ Risk Controls")
    st.caption("Instant — no retraining needed.")

    tsl_pct   = st.slider("Trailing Stop Loss (%)", 5.0, 25.0, 10.0, 0.5)
    z_reentry = st.slider("Re-entry Z-Score Threshold", 0.5, 3.0, 1.1, 0.1)

    st.divider()
    run_btn = st.button("🚀 Retrain DQN Agent",
                        help="Triggers GitHub Actions training job",
                        use_container_width=True)

    if run_btn:
        triggered = _trigger_github(start_year, fee_bps, tsl_pct, z_reentry)
        if triggered:
            st.success(
                f"✅ Training triggered!\n\n"
                f"Training from **{start_year}** · 250 episodes · **{fee_bps}bps** fees"
                f"**{fee_bps}bps** fees\n\n"
                f"Results update here in ~50–65 min."
            )
        else:
            st.warning(
                "⚠️ Could not trigger GitHub Actions automatically.\n\n"
                f"**Manual steps:**\n"
                f"- Go to GitHub → Actions → Train DQN Agent\n"
                f"- Set `start_year = {start_year}`\n"
                f"- Or add `GITHUB_TOKEN` to HF Space secrets."
            )

    st.caption(f"↑ Trains from {start_year} onwards · 250 episodes (hardcoded in train_models.yml)")


# ── Load outputs ──────────────────────────────────────────────────────────────
pred  = _load_json("latest_prediction.json")
evalu = _load_json("evaluation_results.json")

next_td          = _next_trading_day()
final_signal     = pred.get("final_signal", "—")
z_score          = pred.get("z_score", 0.0)
confidence       = pred.get("confidence", pred.get("final_confidence", 0.0))
tsl_stat         = pred.get("tsl_status", {})
tbill_rt         = pred.get("tbill_rate", 3.6)
probs            = pred.get("probabilities", {})
q_vals           = pred.get("q_values", {})
trained_from_year= pred.get("trained_from_year")
trained_at       = pred.get("trained_at")
in_cash          = tsl_stat.get("in_cash", False)
tsl_triggered    = tsl_stat.get("tsl_triggered", False)
two_day_ret      = tsl_stat.get("two_day_cumul_pct", 0.0)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🤖 P2 ETF DQN Engine")
st.caption("Dueling Deep Q-Network · Multi-Asset ETF Selection · arXiv:2411.07585")

# ── Provenance banner ─────────────────────────────────────────────────────────
if trained_from_year and trained_at:
    trained_date = trained_at[:10]
    st.markdown(
        f'<div class="provenance">📋 Active model trained from '
        f'<b>{trained_from_year}</b> · Generated <b>{trained_date}</b> · '
        f'Val Sharpe <b>{evalu.get("sharpe", "—")}</b></div>',
        unsafe_allow_html=True
    )
else:
    st.info("⚠️ No trained model found. Click **🚀 Retrain DQN Agent** to train.")

st.markdown("---")

# ── TSL override banner ───────────────────────────────────────────────────────
if tsl_triggered:
    st.markdown(f"""
    <div style="background:#fff8f0;border:2px solid #cc6600;border-radius:10px;
                padding:16px;margin-bottom:16px;">
      🔴 <b>TRAILING STOP LOSS TRIGGERED</b> — 2-day return
      ({float(two_day_ret):+.1f}%) breached −{tsl_pct:.0f}% threshold.
      Holding CASH @ {tbill_rt:.2f}% T-bill until Z ≥ {z_reentry:.1f}σ.
    </div>""", unsafe_allow_html=True)

# ── Signal Hero Card ──────────────────────────────────────────────────────────
now_est  = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=5)
is_today = (next_td == now_est.date())
td_label = "TODAY'S SIGNAL" if is_today else "NEXT TRADING DAY"

if in_cash or not pred:
    st.markdown(f"""
    <div class="cash-card">
      <div class="hero-label">⚠️ Risk Override Active · {td_label}</div>
      <div class="hero-value" style="color:#cc6600;">💵 CASH</div>
      <div class="hero-sub">
        Earning 3m T-bill: <b>{tbill_rt:.2f}% p.a.</b> &nbsp;|&nbsp;
        Re-entry when Z ≥ {z_reentry:.1f}σ
      </div>
    </div>""", unsafe_allow_html=True)
else:
    prov_str = ""
    if trained_from_year and trained_at:
        prov_str = (f"📋 Trained from {trained_from_year} · "
                    f"Generated {trained_at[:10]} · Z-Score {z_score:.2f}σ")
    st.markdown(f"""
    <div class="hero-card">
      <div class="hero-label">Dueling DQN · {td_label}</div>
      <div class="hero-value">{final_signal}</div>
      <div class="hero-sub">
        🎯 {next_td} &nbsp;|&nbsp; Confidence {float(confidence):.1%}
        &nbsp;|&nbsp; Z-Score {float(z_score):.2f}σ
      </div>
      {"<div class='hero-sub' style='margin-top:6px;font-size:12px;opacity:0.7;'>" + prov_str + "</div>" if prov_str else ""}
    </div>""", unsafe_allow_html=True)

# ── Key Metrics ───────────────────────────────────────────────────────────────
if evalu:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Ann. Return",   f"{evalu.get('ann_return', 0):.1%}")
    c2.metric("Sharpe Ratio",  f"{evalu.get('sharpe', 0):.2f}")
    c3.metric("Max Drawdown",  f"{evalu.get('max_drawdown', 0):.1%}")
    c4.metric("Calmar Ratio",  f"{evalu.get('calmar', 0):.2f}")
    c5.metric("Hit Ratio",     f"{evalu.get('hit_ratio', 0):.1%}")

    # Benchmark comparison — show annualised return
    st.markdown("**Benchmark Comparison (Test Period)**")
    try:
        from data_download import load_local
        bdata = load_local()
        if bdata and "etf_prices" in bdata:
            prices   = bdata["etf_prices"]
            n_test   = evalu.get("n_test_days", 252)
            bc1, bc2 = st.columns(2)
            for col, b in zip([bc1, bc2], config.BENCHMARKS):
                if b in prices.columns:
                    bp      = prices[b].dropna()
                    bp_test = bp.iloc[-n_test:]
                    b_ret   = float((bp_test.iloc[-1] / bp_test.iloc[0]) - 1)
                    b_ann   = float((1 + b_ret) ** (252 / max(len(bp_test), 1)) - 1)
                    strat_ann = evalu.get("ann_return", 0)
                    delta   = strat_ann - b_ann
                    col.metric(f"{b} Ann. Return", f"{b_ann:.1%}",
                               delta=f"{delta:+.1%} vs strategy")
    except Exception:
        bench = evalu.get("benchmark_sharpe", {})
        if bench:
            bc1, bc2 = st.columns(2)
            for col, (k, v) in zip([bc1, bc2], bench.items()):
                col.metric(f"{k} Sharpe", f"{v:.2f}")

st.markdown("---")

# ── Q-Value / Probability Bar Chart ──────────────────────────────────────────
if probs:
    st.subheader("📊 Agent Conviction — Action Probabilities")
    actions = list(probs.keys())
    values  = [probs[a] for a in actions]
    colours = ["#fd7e14" if a == "CASH" else
               "#0066cc" if a == final_signal else "#adb5bd"
               for a in actions]
    text_colours = ["#ffffff" for _ in actions]

    fig = go.Figure(go.Bar(
        x=actions, y=values,
        marker=dict(
            color=colours,
            line=dict(color=["#cc5500" if a == "CASH" else
                             "#004499" if a == final_signal else "#6c757d"
                             for a in actions], width=2)
        ),
        text=[f"{v:.1%}" for v in values],
        textposition="outside",
        textfont=dict(size=12, color="#1a1a1a"),
        cliponaxis=False,
    ))
    fig.update_layout(
        paper_bgcolor="#f8f9fa", plot_bgcolor="#ffffff",
        font_color="#1a1a1a",
        yaxis_title="Probability",
        xaxis_title="",
        height=340,
        margin=dict(t=10, b=10, l=50, r=20),
        yaxis=dict(gridcolor="#dee2e6", tickformat=".0%",
                   range=[0, max(values) * 1.3]),
        xaxis=dict(tickfont=dict(size=13, color="#1a1a1a")),
        bargap=0.35,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "**How to read this chart:** Each bar shows the agent's probability of choosing "
        "that action today, derived from softmax of the DQN's Q-values. "
        "🔵 Blue = today's chosen action. 🟠 Orange = CASH. Grey = rejected actions. "
        "A dominant single bar = high conviction. "
        "Bars of similar height = agent is uncertain (low Z-score) — treat signal with caution."
    )

# ── Equity Curve ──────────────────────────────────────────────────────────────
if evalu and "equity_curve" in evalu:
    st.subheader("📈 Test-Set Equity Curve vs Benchmarks")
    st.caption("Normalised to 1.0 at start of test period (final 10% of data). "
               "SPY and AGG shown for comparison.")
    equity  = evalu["equity_curve"]
    n       = len(equity)

    # Build date index for X axis from price data
    x_dates = None
    try:
        from data_download import load_local
        d = load_local()
        if d and "etf_prices" in d:
            all_dates = d["etf_prices"].dropna(how="all").index
            if len(all_dates) >= n:
                x_dates = [str(dt.date()) for dt in all_dates[-n:]]
    except Exception:
        pass
    x_axis = x_dates if x_dates else list(range(n))

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=x_axis, y=equity, mode="lines", name="DQN Strategy",
        line=dict(color="#0066cc", width=2.5),
    ))

    # SPY and AGG lines
    bench_colours = {"SPY": "#e63946", "AGG": "#2a9d8f"}
    try:
        from data_download import load_local
        d = load_local()
        if d and "etf_prices" in d:
            prices = d["etf_prices"]
            for b in config.BENCHMARKS:
                if b in prices.columns:
                    bp    = prices[b].dropna()
                    bp_n  = bp.iloc[-n:]
                    brets = bp_n.pct_change().fillna(0)
                    beq   = (1 + brets).cumprod().values
                    beq   = beq / beq[0]
                    bx    = [str(dt.date()) for dt in bp_n.index] if x_dates else list(range(len(beq)))
                    fig2.add_trace(go.Scatter(
                        x=bx, y=beq, mode="lines", name=b,
                        line=dict(width=1.5, dash="dot",
                                  color=bench_colours.get(b, "#888888")),
                    ))
    except Exception:
        pass

    fig2.update_layout(
        paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
        font_color="#1a1a1a", height=420,
        yaxis_title="Normalised Equity (start = 1.0)",
        xaxis_title="Date",
        legend=dict(bgcolor="#f8f9fa", orientation="h",
                    yanchor="bottom", y=1.02, xanchor="left", x=0),
        yaxis=dict(gridcolor="#e9ecef", tickformat=".2f"),
        xaxis=dict(type="category", tickangle=-45,
                   nticks=12, gridcolor="#e9ecef"),
        margin=dict(t=40, b=60),
    )
    st.plotly_chart(fig2, use_container_width=True)

# ── Allocation Breakdown ──────────────────────────────────────────────────────
if evalu and "allocation_pct" in evalu:
    st.subheader("📊 Allocation Breakdown — Out-of-Sample Test Period")
    alloc     = evalu["allocation_pct"]
    n_test    = evalu.get("n_test_days", "?")
    start_yr  = evalu.get("start_year", "?")
    fig3  = go.Figure(go.Pie(
        labels=list(alloc.keys()),
        values=list(alloc.values()),
        hole=0.45,
        marker_colors=["#0066cc","#28a745","#ffc107","#fd7e14",
                       "#6f42c1","#e83e8c","#17a2b8","#adb5bd"],
        textinfo="label+percent",
        textfont=dict(size=13),
        hovertemplate="<b>%{label}</b><br>%{percent} of test days<extra></extra>",
    ))
    fig3.update_layout(
        paper_bgcolor="#ffffff", font_color="#1a1a1a",
        height=360, margin=dict(t=20, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=-0.15),
    )
    st.plotly_chart(fig3, use_container_width=True)
    st.caption(
        f"**What this shows:** Percentage of the {n_test} out-of-sample (OOS) test days "
        f"the agent held each position. This is the final 10% of data from {start_yr} onwards — "
        f"data the model never saw during training. "
        "A well-trained agent rotates across multiple ETFs. "
        "Heavy concentration in one slice (80%+) indicates the agent defaulted to a "
        "single-asset strategy — a sign of poor learning or regime lock-in."
    )

# ── 15-Day Audit Trail ───────────────────────────────────────────────────────
if evalu and "allocations" in evalu and len(evalu["allocations"]) > 0:
    st.subheader("🗓️ 15-Day Audit Trail — Most Recent OOS Days")
    st.caption("Last 15 trading days from the out-of-sample test period. "
               "Shows daily allocation decision, return, and running equity.")

    allocs     = evalu["allocations"]
    eq_curve   = evalu.get("equity_curve", [])

    # Reconstruct daily returns from equity curve
    daily_rets = []
    if len(eq_curve) > 1:
        for i in range(1, len(eq_curve)):
            daily_rets.append(eq_curve[i] / eq_curve[i-1] - 1)

    n_show      = min(15, len(allocs))
    last_allocs = allocs[-n_show:]
    last_rets   = daily_rets[-n_show:] if daily_rets else [0.0] * n_show
    last_equity = eq_curve[-n_show:]   if eq_curve   else [1.0] * n_show

    # Try to get actual dates
    date_labels = [f"Day {i+1}" for i in range(n_show)]
    try:
        from data_download import load_local
        d = load_local()
        if d and "etf_prices" in d:
            all_dates = d["etf_prices"].dropna(how="all").index
            n_test    = evalu.get("n_test_days", len(allocs))
            test_dates = all_dates[-n_test:]
            if len(test_dates) >= n_show:
                date_labels = [str(dt.date()) for dt in test_dates[-n_show:]]
    except Exception:
        pass

    audit_rows = []
    for i in range(n_show):
        ret = last_rets[i] if i < len(last_rets) else 0.0
        eq  = last_equity[i] if i < len(last_equity) else 1.0
        audit_rows.append({
            "Date"      : date_labels[i],
            "Allocation": last_allocs[i],
            "Daily Ret" : ret,
            "Equity"    : eq,
        })

    audit_df = pd.DataFrame(audit_rows)

    def _colour_ret(val):
        color = "#d93025" if val < 0 else "#188038"
        return f"color: {color}; font-weight: bold"

    styled = (
        audit_df.style
        .format({"Daily Ret": "{:+.2%}", "Equity": "{:.4f}"})
        .applymap(_colour_ret, subset=["Daily Ret"])
        .set_properties(**{"text-align": "center"})
        .hide(axis="index")
    )
    st.dataframe(styled, use_container_width=True)

st.markdown("---")

# ── Methodology Section ───────────────────────────────────────────────────────
st.subheader("🧠 Methodology")
st.markdown("""
<div class="method-box">

<h4 style="color:#0066cc;">Reinforcement Learning Framework — Dueling DQN</h4>

<p>This engine implements a <b>Dueling Deep Q-Network (Dueling DQN)</b> for daily ETF
selection, directly extending the RL framework proposed by
<b>Yasin & Gill (2024)</b> — <i>"Reinforcement Learning Framework for Quantitative Trading"</i>,
presented at the <b>ICAIF 2024 FM4TS Workshop</b>
(<a href="https://arxiv.org/abs/2411.07585" style="color:#0066cc;">arXiv:2411.07585</a>).</p>

<h5 style="color:#0066cc;">From the Paper → Our Implementation</h5>

<p>The paper benchmarks DQN, PPO, and A2C agents on single-stock buy/sell decisions using
20 technical indicators, finding that <b>DQN with MLP policy significantly outperforms
policy-gradient methods</b> (PPO, A2C) on daily financial time-series, and that
<b>higher learning rates</b> (lr = 0.001) produce the most profitable signals.</p>

<p>We extend this methodology in three key ways:</p>

<ol>
<li><b>Multi-Asset Action Space:</b> Rather than binary buy/sell on a single asset,
the agent selects from 8 discrete actions — CASH or one of 7 ETFs
(TLT, VCIT, LQD, HYG, VNQ, GLD, SLV). This is fundamentally a harder problem than
the paper's setup, requiring the agent to learn relative value across assets.</li>

<li><b>Dueling Architecture</b> (Wang et al., 2016): We replace the paper's standard DQN
with a <b>Dueling DQN</b>, which separates the Q-function into a state-value stream V(s)
and an advantage stream A(s,a):
<br><code>Q(s,a) = V(s) + A(s,a) − mean_a(A(s,a))</code><br>
This is specifically more effective for multi-action spaces because it explicitly learns
which state is valuable independent of which action to take — critical when TLT and VCIT
have similar Q-values in a rate-falling regime.</li>

<li><b>Macro State Augmentation:</b> The paper's state space uses only price-derived
technical indicators. We add six FRED macro signals to the state:
VIX, T10Y2Y (yield curve slope), TBILL_3M, DXY, Corp Spread, and HY Spread.
These directly encode the macro regime that drives fixed-income and credit ETF selection.</li>
</ol>

<h5 style="color:#0066cc;">State Space (per trading day)</h5>
<p>20 technical indicators per ETF × 7 ETFs + 6 macro signals (+ z-scored variants),
all computed over a rolling <b>20-day lookback window</b>. The flattened window is fed
to the DQN as a single state vector. Indicators follow the paper exactly:
RSI(14), MACD(12/26/9), Stochastic(14), CCI(20), ROC(10), CMO(14), Williams%R,
ATR, Bollinger %B + Width, StochRSI, Ultimate Oscillator, Momentum(10),
rolling returns at 1/5/10/21d, and 21d realised volatility.</p>

<h5 style="color:#0066cc;">Reward Function</h5>
<p>Reward = excess daily return over 3m T-bill, minus transaction cost on switches,
scaled by inverse 21d realised volatility to penalise drawdown-prone positions.
This replaces the paper's raw P&L reward with a risk-adjusted signal aligned with
Sharpe Ratio maximisation.</p>

<h5 style="color:#0066cc;">Training</h5>
<p>Data split is 80/10/10 (train/val/test) from the user-selected start year to present.
Best weights are saved by <b>validation-set Sharpe Ratio</b>. The agent uses
<b>Double DQN</b> (online network selects action, frozen target network evaluates)
to reduce Q-value overestimation — a known instability in financial RL applications.
Experience replay buffer of 100k transitions; hard target network update every 500 steps;
ε-greedy exploration decaying from 1.0 → 0.05 over the first 50% of training.
Training runs for <b>250 episodes</b> (one episode = one full pass over the training set).
<br><small style="color:#6c757d;">
⚙️ To change episode count: edit <code>train_models.yml</code> in GitHub →
find <code>--episodes "250"</code> in the <i>Train Dueling DQN Agent</i> step → update the number.
</small></p>

<h5 style="color:#0066cc;">Risk Controls</h5>
<p>A post-signal <b>Trailing Stop Loss</b> overrides the DQN signal to CASH if the
2-day cumulative return of the held ETF breaches the configured threshold.
Re-entry from CASH requires the DQN's best-action Z-score to clear the re-entry
threshold, ensuring the model has recovered conviction before re-entering risk.</p>

</div>
""", unsafe_allow_html=True)

# ── Reference ─────────────────────────────────────────────────────────────────
st.markdown("""
<div style="background:#f8f9fa;border:1px solid #dee2e6;border-radius:8px;
            padding:14px;font-size:12px;color:#6c757d;margin-top:8px;">
<b>Reference:</b> Yasin, A.S. & Gill, P.S. (2024).
<i>Reinforcement Learning Framework for Quantitative Trading.</i>
arXiv:2411.07585 [q-fin.TR]. Accepted at ICAIF 2024 FM4TS Workshop.
&nbsp;·&nbsp;
<b>Dueling DQN:</b> Wang, Z. et al. (2016).
<i>Dueling Network Architectures for Deep Reinforcement Learning.</i>
ICML 2016.
</div>
""", unsafe_allow_html=True)
