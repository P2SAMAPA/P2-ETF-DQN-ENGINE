# app.py — P2-ETF-DQN-ENGINE  Streamlit UI
# Supports both Option A (FI/Commodities) and Option B (Equity Sectors).

import json
import os
import shutil
from datetime import datetime, date, timedelta, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests as req
import streamlit as st

import config

WORKFLOW_FILE  = "train_models.yml"

# Colour maps for both universes
FI_COLOURS = {
    "TLT": "#4e79a7", "VCIT": "#f28e2b", "LQD": "#59a14f",
    "HYG": "#e15759", "VNQ": "#76b7b2", "SLV": "#edc948",
    "GLD": "#b07aa1", "CASH": "#aaaaaa",
}
EQ_COLOURS = {
    "SPY": "#4e79a7", "QQQ": "#f28e2b", "XLK": "#59a14f",
    "XLF": "#e15759", "XLE": "#76b7b2", "XLV": "#edc948",
    "XLI": "#b07aa1", "XLY": "#ff9da7", "XLP": "#9c755f",
    "XLU": "#86b875", "GDX": "#bab0ac", "XME": "#f1ce63",
    "CASH": "#aaaaaa",
}

def get_colour_map(option: str) -> dict:
    return FI_COLOURS if option == 'a' else EQ_COLOURS


# ── Session state for option selection ──────────────────────────────────────
if "option" not in st.session_state:
    st.session_state.option = "Option A (FI/Commodities)"
    st.session_state.opt_code = 'a'
option = st.session_state.option
opt_code = st.session_state.opt_code


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_json(path: str) -> dict:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def _next_trading_day() -> date:
    US_HOLIDAYS = {
        date(2025,1,1), date(2025,1,20), date(2025,2,17), date(2025,4,18),
        date(2025,5,26), date(2025,6,19), date(2025,7,4), date(2025,9,1),
        date(2025,11,27), date(2025,12,25),
        date(2026,1,1), date(2026,1,19), date(2026,2,16), date(2026,4,3),
        date(2026,5,25), date(2026,6,19), date(2026,7,3), date(2026,9,7),
        date(2026,11,26), date(2026,12,25),
    }
    now_est = datetime.utcnow() - timedelta(hours=5)
    today   = now_est.date()
    if today.weekday() < 5 and today not in US_HOLIDAYS and now_est.hour < 16:
        return today
    d = today + timedelta(days=1)
    while d.weekday() >= 5 or d in US_HOLIDAYS:
        d += timedelta(days=1)
    return d


def _trigger_github(start_year: int, fee_bps: int,
                    tsl_pct: float, z_reentry: float,
                    sweep_mode: str = "") -> bool:
    try:
        token = os.getenv("GITHUB_TOKEN", "")
        if not token:
            st.error("❌ GITHUB_TOKEN not found in Space secrets.")
            return False
        url  = f"https://api.github.com/repos/{config.GITHUB_REPO}/actions/workflows/{WORKFLOW_FILE}/dispatches"
        resp = req.post(url,
            headers={"Authorization": f"token {token}",
                     "Accept": "application/vnd.github+json"},
            json={"ref": "main",
                  "inputs": {
                      "start_year": str(start_year),
                      "fee_bps":    str(fee_bps),
                      "tsl_pct":    str(tsl_pct),
                      "z_reentry":  str(z_reentry),
                      "sweep_mode": sweep_mode,
                  }},
            timeout=10,
        )
        if resp.status_code != 204:
            st.error(f"❌ GitHub API returned HTTP {resp.status_code} — {resp.text[:300]}")
        return resp.status_code == 204
    except Exception as e:
        st.error(f"❌ Exception: {str(e)}")
        return False


def _get_latest_workflow_run() -> dict:
    try:
        token = os.getenv("GITHUB_TOKEN", "")
        if not token:
            return {}
        url = f"https://api.github.com/repos/{config.GITHUB_REPO}/actions/workflows/{WORKFLOW_FILE}/runs?per_page=1"
        r = req.get(url, headers={"Authorization": f"token {token}",
                                   "Accept": "application/vnd.github+json"}, timeout=10)
        if r.status_code == 200:
            runs = r.json().get("workflow_runs", [])
            return runs[0] if runs else {}
    except Exception:
        pass
    return {}


def _today_est() -> date:
    return (datetime.now(timezone.utc) - timedelta(hours=5)).date()


def _load_sweep_cache(for_date: date, option: str) -> dict:
    """Load date-stamped sweep files from HF Dataset (subdirectory)."""
    cache = {}
    try:
        from huggingface_hub import hf_hub_download, HfApi
        token   = os.getenv("HF_TOKEN")
        repo_id = os.getenv("HF_DATASET_REPO", "P2SAMAPA/P2-ETF-DQN-ENGINE-DATASET")
        date_tag = for_date.strftime("%Y%m%d")
        # List files in the subdirectory for this option
        api = HfApi()
        prefix = f"sweep/option_{option}/sweep_"
        files = list(api.list_repo_files(repo_id=repo_id, repo_type="dataset", token=token))
        for f in files:
            if f.startswith(prefix) and f.endswith(".json"):
                # Extract year from filename e.g. sweep_2008_20260328.json
                parts = os.path.basename(f).replace(".json","").split("_")
                if len(parts) == 3:
                    try:
                        yr = int(parts[1])
                        if parts[2] == date_tag:
                            path = hf_hub_download(repo_id=repo_id, filename=f,
                                                   repo_type="dataset", token=token, force_download=True)
                            with open(path) as fh:
                                cache[yr] = json.load(fh)
                    except Exception:
                        pass
    except Exception as e:
        st.warning(f"Could not load sweep results: {e}")
    return cache


def _load_sweep_cache_any(option: str) -> tuple:
    """Load most recent sweep files from HF Dataset regardless of date. Returns (cache, date)."""
    found, best_date = {}, None
    try:
        from huggingface_hub import HfApi, hf_hub_download
        token   = os.getenv("HF_TOKEN")
        repo_id = os.getenv("HF_DATASET_REPO", "P2SAMAPA/P2-ETF-DQN-ENGINE-DATASET")
        api     = HfApi()
        prefix = f"sweep/option_{option}/sweep_"
        files   = list(api.list_repo_files(repo_id=repo_id, repo_type="dataset", token=token))
        # Find most recent date across all sweep files for this option
        for fname in files:
            if not fname.startswith(prefix):
                continue
            bn = os.path.basename(fname)
            parts = bn.replace(".json","").split("_")
            if len(parts) == 3:
                try:
                    dt = datetime.strptime(parts[2], "%Y%m%d").date()
                    if best_date is None or dt > best_date:
                        best_date = dt
                except Exception:
                    pass
        if best_date:
            found = _load_sweep_cache(best_date, option)
    except Exception:
        pass
    return found, best_date


def _compute_consensus(sweep_data: dict) -> dict:
    """40% Return · 20% Z · 20% Sharpe · 20% (-MaxDD), min-max normalised."""
    rows = []
    for yr, sig in sweep_data.items():
        rows.append({
            "year":       yr,
            "signal":     sig.get("signal", "?"),
            "ann_return": sig.get("ann_return", 0.0),
            "z_score":    sig.get("z_score", 0.0),
            "sharpe":     sig.get("sharpe", 0.0),
            "max_dd":     sig.get("max_dd", 0.0),
            "conviction": sig.get("conviction", "?"),
            "lookback":   sig.get("lookback", "?"),
        })
    if not rows:
        return {}
    df = pd.DataFrame(rows)

    def _mm(s):
        mn, mx = s.min(), s.max()
        return (s - mn) / (mx - mn + 1e-9)

    df["n_ret"]    = _mm(df["ann_return"])
    df["n_z"]      = _mm(df["z_score"])
    df["n_sharpe"] = _mm(df["sharpe"])
    df["n_negdd"]  = _mm(-df["max_dd"])
    df["wtd"]      = 0.40*df["n_ret"] + 0.20*df["n_z"] + 0.20*df["n_sharpe"] + 0.20*df["n_negdd"]

    etf_agg = {}
    for _, row in df.iterrows():
        e = row["signal"]
        etf_agg.setdefault(e, {"years": [], "scores": [], "returns": [],
                                "zs": [], "sharpes": [], "dds": []})
        etf_agg[e]["years"].append(row["year"])
        etf_agg[e]["scores"].append(row["wtd"])
        etf_agg[e]["returns"].append(row["ann_return"])
        etf_agg[e]["zs"].append(row["z_score"])
        etf_agg[e]["sharpes"].append(row["sharpe"])
        etf_agg[e]["dds"].append(row["max_dd"])

    total = sum(sum(v["scores"]) for v in etf_agg.values()) + 1e-9
    summary = {}
    for e, v in etf_agg.items():
        cs = sum(v["scores"])
        summary[e] = {
            "cum_score":   round(cs, 4),
            "score_share": round(cs / total, 3),
            "n_years":     len(v["years"]),
            "years":       v["years"],
            "avg_return":  round(float(np.mean(v["returns"])), 4),
            "avg_z":       round(float(np.mean(v["zs"])), 3),
            "avg_sharpe":  round(float(np.mean(v["sharpes"])), 3),
            "avg_max_dd":  round(float(np.mean(v["dds"])), 4),
        }
    winner = max(summary, key=lambda e: summary[e]["cum_score"])
    return {"winner": winner, "etf_summary": summary,
            "per_year": df.to_dict("records"), "n_years": len(rows)}


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    # Update session state with user selection
    selected = st.radio("Select Option", ["Option A (FI/Commodities)", "Option B (Equity Sectors)"],
                         index=0, horizontal=True)
    st.session_state.option = selected
    st.session_state.opt_code = 'a' if "Option A" in selected else 'b'
    # Update local variables for this run
    option = st.session_state.option
    opt_code = st.session_state.opt_code

    if st.button("🔄 Refresh Data & Clear Cache"):
        st.cache_data.clear()
        st.toast("Cache cleared — reloading...")
        st.rerun()

    st.divider()
    st.markdown("### 📅 Training Parameters")
    st.caption("Changes here require retraining via GitHub Actions.")

    start_year = st.slider("Start Year", config.DEFAULT_START_YEAR,
                           2024, 2015, key=f"start_{opt_code}")
    fee_bps    = st.number_input("T-Costs (bps)", 0, 50, 10, key=f"fee_{opt_code}")

    st.divider()
    st.markdown("### 🛡️ Risk Controls")
    st.caption("Instant — no retraining needed.")

    tsl_pct   = st.slider("Trailing Stop Loss (%)", 5.0, 25.0, 10.0, 0.5, key=f"tsl_{opt_code}")
    z_reentry = st.slider("Re-entry Z-Score Threshold", 0.5, 3.0, 1.1, 0.1, key=f"z_{opt_code}")

    st.divider()
    run_btn = st.button("🚀 Retrain DQN Agent",
                        help="Triggers GitHub Actions training job",
                        use_container_width=True)

    if run_btn:
        triggered = _trigger_github(start_year, fee_bps, tsl_pct, z_reentry, sweep_mode="")
        if triggered:
            st.success(
                f"✅ Training triggered!\n\n"
                f"Training from **{start_year}** · 200 episodes · "
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

    st.caption(f"↑ Trains from {start_year} onwards · 200 episodes (hardcoded in train_models.yml)")


# ── Load outputs for selected option ──────────────────────────────────────────
pred_file = f"latest_prediction_{opt_code}.json"
eval_file = f"evaluation_results_{opt_code}.json"
pred  = _load_json(pred_file)
evalu = _load_json(eval_file)

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

# ── Check latest workflow run status ─────────────────────────────────────────
latest_run  = _get_latest_workflow_run()
is_training = latest_run.get("status") in ("queued", "in_progress")
run_started = latest_run.get("created_at", "")[:16].replace("T", " ") if latest_run else ""

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["📊 Single-Year Results", "🔄 Multi-Year Consensus Sweep"])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Single-Year Results
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
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
    now_est  = datetime.utcnow() - timedelta(hours=5)
    is_today = (next_td == now_est.date())
    td_label = "TODAY'S SIGNAL" if is_today else "NEXT TRADING DAY"

    if in_cash or not pred:
        st.markdown(f"""
        <div class="cash-card" style="background: linear-gradient(135deg, #fff8f0 0%, #ffe4cc 100%); border: 2px solid #cc6600; border-radius: 16px; padding: 32px; text-align: center; margin-bottom: 24px;">
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
        <div class="hero-card" style="background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); border: 2px solid #0066cc; border-radius: 16px; padding: 32px; text-align: center; margin-bottom: 24px;">
          <div class="hero-label">Dueling DQN · {td_label}</div>
          <div class="hero-value" style="color:#0066cc;">{final_signal}</div>
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

        # Benchmark comparison — ann return
        bench_ann = evalu.get("benchmark_ann", {})
        if bench_ann:
            bc1, bc2  = st.columns(2)
            strat_ann = evalu.get("ann_return", 0)
            for col, (k, v) in zip([bc1, bc2], bench_ann.items()):
                delta = strat_ann - v
                col.metric(f"{k} Ann. Return", f"{v:.1%}",
                           delta=f"{delta:+.1%} vs strategy")

    st.markdown("---")

    # ── Q-Value / Probability Bar Chart ──────────────────────────────────────────
    if probs:
        st.subheader("📊 Action Probabilities (Softmax Q-Values)")
        actions = list(probs.keys())
        values  = [probs[a] for a in actions]
        colours = ["#cc6600" if a == "CASH" else
                   "#0066cc" if a == final_signal else "#6c757d"
                   for a in actions]

        fig = go.Figure(go.Bar(
            x=actions, y=values,
            marker_color=colours,
            text=[f"{v:.1%}" for v in values],
            textposition="outside",
        ))
        fig.update_layout(
            paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
            font_color="#1a1a1a",
            yaxis_title="Probability", xaxis_title="Action",
            height=300, margin=dict(t=20, b=20),
            yaxis=dict(gridcolor="#e9ecef"),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "**How to read:** Each bar is the agent's probability of choosing that action today, "
            "derived from softmax of the DQN Q-values. 🔵 Blue = chosen action. 🟠 Orange = CASH. "
            "Grey = rejected. A dominant bar = high conviction. Similar-height bars = low conviction / uncertain signal."
        )

    # ── Equity Curve ──────────────────────────────────────────────────────────────
    if evalu and "equity_curve" in evalu:
        st.subheader("📈 Test-Set Equity Curve vs Benchmarks")
        st.caption("Normalised to 1.0 at start of test period. SPY and AGG shown for comparison.")
        equity     = evalu["equity_curve"]
        test_dates = evalu.get("test_dates", [])
        x_axis     = test_dates if len(test_dates) == len(equity) else list(range(len(equity)))

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=x_axis, y=equity, mode="lines", name="DQN Strategy",
            line=dict(color="#0066cc", width=2.5),
        ))

        # SPY and AGG from json — no load_local needed
        bench_equity = evalu.get("benchmark_equity", {})
        bench_colours = {"SPY": "#e63946", "AGG": "#2a9d8f"}
        for b, beq in bench_equity.items():
            bx = test_dates if len(test_dates) == len(beq) else list(range(len(beq)))
            fig2.add_trace(go.Scatter(
                x=bx, y=beq, mode="lines", name=b,
                line=dict(width=1.5, dash="dot", color=bench_colours.get(b, "#888888")),
            ))

        fig2.update_layout(
            paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
            font_color="#1a1a1a", height=420,
            yaxis_title="Normalised Equity (start = 1.0)",
            xaxis_title="Date",
            legend=dict(bgcolor="#f8f9fa", orientation="h",
                        yanchor="bottom", y=1.02, xanchor="left", x=0),
            yaxis=dict(gridcolor="#e9ecef", tickformat=".2f"),
            xaxis=dict(tickangle=-45, nticks=12, gridcolor="#e9ecef"),
            margin=dict(t=40, b=60),
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── Allocation Breakdown ──────────────────────────────────────────────────────
    if evalu and "allocation_pct" in evalu:
        st.subheader("📊 Allocation Breakdown (Test Set)")
        alloc = evalu["allocation_pct"]
        colour_map = get_colour_map(opt_code)
        fig3  = go.Figure(go.Pie(
            labels=list(alloc.keys()),
            values=list(alloc.values()),
            hole=0.45,
            marker_colors=[colour_map.get(e, "#888") for e in alloc.keys()],
        ))
        fig3.update_layout(
            paper_bgcolor="#ffffff", font_color="#1a1a1a",
            height=320, margin=dict(t=20),
        )
        st.plotly_chart(fig3, use_container_width=True)
        st.caption(
            "**How to read:** Percentage of out-of-sample (OOS) test days the agent held each position. "
            "A well-trained agent rotates across ETFs. Heavy concentration in one slice (80%+) = "
            "agent defaulted to a single-asset strategy — sign of poor learning."
        )

    st.markdown("---")

    # ── 15-Day Audit Trail ────────────────────────────────────────────────────────
    if evalu and "allocations" in evalu and len(evalu["allocations"]) > 0:
        st.subheader("🗓️ 15-Day Audit Trail — Most Recent OOS Days")
        st.caption("Last 15 trading days from the out-of-sample test period.")

        allocs     = evalu["allocations"]
        eq_curve   = evalu.get("equity_curve", [])
        test_dates = evalu.get("test_dates", [])

        daily_rets = []
        if len(eq_curve) > 1:
            for i in range(1, len(eq_curve)):
                daily_rets.append(eq_curve[i] / eq_curve[i-1] - 1)

        n_show      = min(15, len(allocs))
        last_allocs = allocs[-n_show:]
        last_rets   = daily_rets[-n_show:]   if daily_rets   else [0.0] * n_show
        last_equity = eq_curve[-n_show:]     if eq_curve     else [1.0] * n_show
        last_dates  = test_dates[-n_show:]   if len(test_dates) >= n_show else [f"Day {i+1}" for i in range(n_show)]

        audit_df = pd.DataFrame({
            "Date"      : last_dates,
            "Allocation": last_allocs,
            "Daily Ret" : last_rets,
            "Equity"    : last_equity,
        })

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

    # ── Methodology Section (shortened) ───────────────────────────────────────────
    st.subheader("🧠 Methodology")
    st.markdown("""
    **Dueling DQN** for ETF rotation, extending **Yasin & Gill (2024)**.

    - **Multi-asset action space** – CASH + 7 ETFs (Option A) or 12 ETFs (Option B)  
    - **Macro state augmentation** – VIX, yield curve slope, T-bill rate, USD index, credit spreads  
    - **Reward** = excess daily return over T-bill, minus transaction cost, scaled by inverse realised volatility  
    - **Training** – 80/10/10 train/val/test split; best model selected by validation Sharpe; Double DQN; experience replay; ε‑greedy decay  
    - **Risk controls** – trailing stop loss (overrides to CASH) + Z‑score re‑entry threshold  

    📄 **References**  
    - Yasin, A.S. & Gill, P.S. (2024). *Reinforcement Learning Framework for Quantitative Trading*. [arXiv:2411.07585](https://arxiv.org/abs/2411.07585)  
    - Wang, Z. et al. (2016). *Dueling Network Architectures for Deep Reinforcement Learning*. ICML 2016.
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


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Multi-Year Consensus Sweep
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader(f"🔄 Multi-Year Consensus Sweep — {option}")
    st.markdown(
        f"**All start years 2008–2025** are precomputed daily (after market close). "
        f"The table below shows the performance of each start year’s model on its out‑of‑sample test period. "
        f"**Weighted score** = 40% Ann. Return + 20% Z-Score + 20% Sharpe + 20% (–MaxDD), min‑max normalised."
    )

    # Load the most recent sweep results from HF (subdirectory)
    sweep_data, sweep_date = _load_sweep_cache_any(opt_code)
    if not sweep_data:
        st.info("No sweep results found. They will be generated after the next daily run.")
        st.stop()

    st.caption(f"Results date: {sweep_date}")

    # Compute consensus
    consensus = _compute_consensus(sweep_data)
    if not consensus:
        st.warning("Could not compute consensus from sweep data.")
        st.stop()

    winner    = consensus["winner"]
    w_info    = consensus["etf_summary"][winner]
    colour_map = get_colour_map(opt_code)
    win_color = colour_map.get(winner, "#0066cc")
    score_pct = w_info["score_share"] * 100
    split_sig = w_info["score_share"] < 0.40
    sig_label = "⚠️ Split Signal" if split_sig else "✅ Clear Signal"
    note      = f"Score share {score_pct:.0f}% · {w_info['n_years']} years · avg score {w_info['cum_score']:.4f}"

    # Winner banner
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#1a1a2e,#16213e);
                border:2px solid {win_color};border-radius:16px;
                padding:32px;text-align:center;margin:16px 0;
                box-shadow:0 8px 24px rgba(0,0,0,0.4);">
      <div style="font-size:11px;letter-spacing:3px;color:#aaa;margin-bottom:8px;">
        WEIGHTED CONSENSUS · DQN · {len(sweep_data)} START YEARS · Results {sweep_date}
      </div>
      <div style="font-size:72px;font-weight:900;color:{win_color};
                  text-shadow:0 0 30px {win_color}88;letter-spacing:2px;">
        {winner}
      </div>
      <div style="font-size:14px;color:#ccc;margin-top:8px;">{sig_label} · {note}</div>
      <div style="display:flex;justify-content:center;gap:32px;margin-top:20px;flex-wrap:wrap;">
        <div style="text-align:center;">
          <div style="font-size:11px;color:#aaa;">Avg Return</div>
          <div style="font-size:22px;font-weight:700;color:{'#00b894' if w_info['avg_return']>0 else '#d63031'};">
            {w_info['avg_return']*100:.1f}%</div>
        </div>
        <div style="text-align:center;">
          <div style="font-size:11px;color:#aaa;">Avg Z</div>
          <div style="font-size:22px;font-weight:700;color:#74b9ff;">{w_info['avg_z']:.2f}σ</div>
        </div>
        <div style="text-align:center;">
          <div style="font-size:11px;color:#aaa;">Avg Sharpe</div>
          <div style="font-size:22px;font-weight:700;color:#a29bfe;">{w_info['avg_sharpe']:.2f}</div>
        </div>
        <div style="text-align:center;">
          <div style="font-size:11px;color:#aaa;">Avg MaxDD</div>
          <div style="font-size:22px;font-weight:700;color:#fd79a8;">{w_info['avg_max_dd']*100:.1f}%</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Also-ranked
    others = sorted([(e, v) for e, v in consensus["etf_summary"].items() if e != winner],
                    key=lambda x: -x[1]["cum_score"])
    parts = []
    for etf, v in others:
        c = colour_map.get(etf, "#888")
        parts.append(f'<span style="color:{c};font-weight:600;">{etf}</span> '
                     f'<span style="color:#aaa;">({v["cum_score"]:.2f} · {v["n_years"]}yr)</span>')
    st.markdown(
        '<div style="text-align:center;margin-bottom:12px;font-size:13px;">'
        'Also ranked: ' + " &nbsp;|&nbsp; ".join(parts) + '</div>',
        unsafe_allow_html=True
    )
    st.divider()

    # ── Charts ────────────────────────────────────────────────────────────────
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Weighted Score per ETF**")
        es = consensus["etf_summary"]
        sorted_etfs = sorted(es.keys(), key=lambda e: -es[e]["cum_score"])
        fig_bar = go.Figure(go.Bar(
            x=sorted_etfs,
            y=[es[e]["cum_score"] for e in sorted_etfs],
            marker_color=[colour_map.get(e, "#888") for e in sorted_etfs],
            text=[f"{es[e]['n_years']}yr · {es[e]['score_share']*100:.0f}%<br>{es[e]['cum_score']:.2f}"
                  for e in sorted_etfs],
            textposition="outside",
        ))
        fig_bar.update_layout(
            template="plotly_dark", height=360,
            yaxis_title="Cumulative Weighted Score",
            showlegend=False, margin=dict(t=20, b=20)
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with c2:
        st.markdown("**Conviction Z-Score by Start Year**")
        per_year = consensus["per_year"]
        fig_sc = go.Figure()
        for row in per_year:
            etf = row["signal"]
            col = colour_map.get(etf, "#888")
            fig_sc.add_trace(go.Scatter(
                x=[row["year"]], y=[row["z_score"]],
                mode="markers+text",
                marker=dict(size=18, color=col, line=dict(color="white", width=1)),
                text=[etf], textposition="top center",
                name=etf, showlegend=False,
                hovertemplate=f"<b>{etf}</b><br>Year: {row['year']}<br>"
                              f"Z: {row['z_score']:.2f}σ<br>"
                              f"Return: {row['ann_return']*100:.1f}%<extra></extra>"
            ))
        fig_sc.add_hline(y=0, line_dash="dot",
                         line_color="rgba(255,255,255,0.3)",
                         annotation_text="Neutral")
        fig_sc.update_layout(
            template="plotly_dark", height=360,
            xaxis_title="Start Year", yaxis_title="Z-Score (σ)",
            margin=dict(t=20, b=20)
        )
        st.plotly_chart(fig_sc, use_container_width=True)

    # ── Per-year breakdown table ──────────────────────────────────────────────
    st.subheader("📋 Full Per-Year Breakdown")
    st.caption(
        "**Wtd Score** = 40% Ann. Return + 20% Z-Score + 20% Sharpe + 20% (–Max DD), "
        "min-max normalised across years. "
        f"Results dated: **{sweep_date}**"
    )

    tbl_rows = []
    for row in sorted(per_year, key=lambda r: r["year"]):
        tbl_rows.append({
            "Start Year":   row["year"],
            "Signal":       row["signal"],
            "Wtd Score":    round(row["wtd"], 3),
            "Conviction":   row["conviction"],
            "Z-Score":      f"{row['z_score']:.2f}σ",
            "Ann. Return":  f"{row['ann_return']*100:.2f}%",
            "Sharpe":       f"{row['sharpe']:.2f}",
            "Max Drawdown": f"{row['max_dd']*100:.2f}%",
            "Lookback":     f"{row['lookback']}d",
        })

    tbl_df = pd.DataFrame(tbl_rows)

    def _style_sig(val):
        c = colour_map.get(val, "#888")
        return f"background-color:{c}22;color:{c};font-weight:700;"

    def _style_ret(val):
        try:
            v = float(val.replace("%", ""))
            return "color:#00b894;font-weight:600" if v > 0 else "color:#d63031;font-weight:600"
        except Exception:
            return ""

    styled_tbl = (tbl_df.style
                  .applymap(_style_sig, subset=["Signal"])
                  .applymap(_style_ret, subset=["Ann. Return"])
                  .set_properties(**{"text-align": "center", "font-size": "14px"})
                  .set_table_styles([
                      {"selector": "th", "props": [("font-size", "14px"),
                                                   ("font-weight", "bold"),
                                                   ("text-align", "center"),
                                                   ("background-color", "#1a1a2e"),
                                                   ("color", "#0066cc")]},
                      {"selector": "td", "props": [("padding", "10px")]}
                  ]))
    st.dataframe(styled_tbl, use_container_width=True, height=280)

    # ── How to read ───────────────────────────────────────────────────────────
    st.divider()
    st.subheader("📖 How to Read These Results")
    st.markdown(f"""
**Why does the signal change by start year?**
Each start year defines the market regime the DQN agent trains on.
A model trained from 2008 has seen the GFC, 2013 taper tantrum, COVID, and rate hike cycles.
A model from 2019 focuses on post-COVID dynamics. The consensus aggregates all regime views.

**How is the winner chosen?**
Each year's signal is scored: 40% Ann. Return + 20% Z-Score + 20% Sharpe + 20% (–MaxDD),
all min-max normalised so no metric dominates. The ETF with the highest total weighted score wins.

**Why is the button disabled when today's results exist?**
To protect your GitHub Actions minutes (~{len(sweep_data) * 90} mins per full sweep).
The sweep auto-runs at 8pm EST daily so you rarely need to trigger it manually.

**Split Signal warning (score share < 40%)**
Signals are fragmented — no single ETF dominates across regimes. Treat with caution.

**Date stamp**
Results filename includes the run date (e.g. `sweep_20260304.json`).
The app only shows today's results if available, otherwise yesterday's with a warning banner.
Previous day's files are automatically deleted at 8pm EST before the new sweep starts.
""")
