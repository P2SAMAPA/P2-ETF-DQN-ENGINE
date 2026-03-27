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

SWEEP_YEARS    = [2008, 2013, 2015, 2017, 2019, 2021]   # not used in new sweep, kept for reference
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


# ── Helpers (unchanged from previous version) ─────────────────────────────────
# ... (keep all the helper functions: _load_json, _next_trading_day, _trigger_github,
# _get_latest_workflow_run, _today_est, _load_sweep_cache, _load_sweep_cache_any,
# _compute_consensus, etc.) 
# To save space, I'll assume they are identical to the last version we had.
# In the final answer, I will include the full app.py with these helpers unchanged.

# For brevity, I'll present only the changed part. But in the final answer, I'll give the full file.

# ...

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["📊 Single-Year Results", "🔄 Multi-Year Consensus Sweep"])

# ... (tab1 unchanged) ...

with tab2:
    st.subheader(f"🔄 Multi-Year Consensus Sweep — {option}")
    st.markdown(
        f"**All start years 2008–2025** are precomputed daily (after market close). "
        f"The table below shows the performance of each start year’s model on its out‑of‑sample test period. "
        f"**Weighted score** = 40% Ann. Return + 20% Z-Score + 20% Sharpe + 20% (–MaxDD), min‑max normalised."
    )

    # Load the most recent walkforward results from HF
    try:
        from huggingface_hub import hf_hub_download, HfApi
        token = os.getenv("HF_TOKEN")
        repo_id = os.getenv("HF_DATASET_REPO", "P2SAMAPA/P2-ETF-DQN-ENGINE-DATASET")
        api = HfApi()
        # List files in the walkforward directory for this option
        prefix = f"results/option_{opt_code}/walkforward/"
        files = list(api.list_repo_files(repo_id=repo_id, repo_type="dataset", token=token))
        sweep_files = [f for f in files if f.startswith(prefix) and f.endswith(".json")]
        if not sweep_files:
            st.info("No walkforward results found. They will be generated after the next daily run.")
            st.stop()
        # Sort by date (filename format sweep_YYYYMMDD.json)
        sweep_files.sort(reverse=True)
        latest_file = sweep_files[0]
        # Download the file
        path = hf_hub_download(repo_id=repo_id, filename=latest_file, repo_type="dataset", token=token)
        with open(path) as f:
            results_by_year = json.load(f)
        results_date = latest_file.split('_')[-1].split('.')[0]
        st.caption(f"Results date: {results_date}")
    except Exception as e:
        st.warning(f"Could not load walkforward results: {e}")
        st.stop()

    # Build a DataFrame from results_by_year (which is dict year -> metrics)
    rows = []
    for year, m in results_by_year.items():
        rows.append({
            "Start Year":   year,
            "Signal":       m.get("signal", "?"),
            "Conviction":   m.get("conviction", "?"),
            "Z-Score":      f"{m.get('z_score', 0):.2f}σ",
            "Ann. Return":  f"{m.get('ann_return', 0)*100:.2f}%",
            "Sharpe":       f"{m.get('sharpe', 0):.2f}",
            "Max Drawdown": f"{m.get('max_dd', 0)*100:.2f}%",
            "Lookback":     f"{m.get('lookback', config.LOOKBACK_WINDOW)}d",
            "_ann_return_raw": m.get('ann_return', 0),
            "_z_score_raw":    m.get('z_score', 0),
            "_sharpe_raw":     m.get('sharpe', 0),
            "_max_dd_raw":     m.get('max_dd', 0),
        })
    df = pd.DataFrame(rows).sort_values("Start Year")

    # Compute min-max normalised scores across all years
    def mm(s):
        mn, mx = s.min(), s.max()
        return (s - mn) / (mx - mn + 1e-9)
    df["n_ret"]    = mm(df["_ann_return_raw"])
    df["n_z"]      = mm(df["_z_score_raw"])
    df["n_sharpe"] = mm(df["_sharpe_raw"])
    df["n_negdd"]  = mm(-df["_max_dd_raw"])
    df["Wtd Score"] = 0.40 * df["n_ret"] + 0.20 * df["n_z"] + 0.20 * df["n_sharpe"] + 0.20 * df["n_negdd"]
    df["Wtd Score"] = df["Wtd Score"].round(3)

    # Aggregate by ETF to find winner
    etf_agg = {}
    for _, row in df.iterrows():
        e = row["Signal"]
        etf_agg.setdefault(e, {"scores": [], "n_years": 0})
        etf_agg[e]["scores"].append(row["Wtd Score"])
        etf_agg[e]["n_years"] += 1
    total_score = sum(sum(v["scores"]) for v in etf_agg.values()) + 1e-9
    etf_summary = {}
    for e, v in etf_agg.items():
        cs = sum(v["scores"])
        etf_summary[e] = {
            "cum_score": round(cs, 4),
            "score_share": round(cs / total_score, 3),
            "n_years": v["n_years"],
            "avg_return": df[df["Signal"] == e]["_ann_return_raw"].mean(),
            "avg_z": df[df["Signal"] == e]["_z_score_raw"].mean(),
            "avg_sharpe": df[df["Signal"] == e]["_sharpe_raw"].mean(),
            "avg_max_dd": df[df["Signal"] == e]["_max_dd_raw"].mean(),
        }

    winner = max(etf_summary, key=lambda e: etf_summary[e]["cum_score"])
    w_info = etf_summary[winner]
    colour_map = get_colour_map(opt_code)
    win_color = colour_map.get(winner, "#0066cc")
    score_pct = w_info["score_share"] * 100
    split_sig = w_info["score_share"] < 0.40
    sig_label = "⚠️ Split Signal" if split_sig else "✅ Clear Signal"
    note = f"Score share {score_pct:.0f}% · {w_info['n_years']} years · avg score {w_info['cum_score']:.4f}"

    # Winner banner
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#1a1a2e,#16213e);
                border:2px solid {win_color};border-radius:16px;
                padding:32px;text-align:center;margin:16px 0;
                box-shadow:0 8px 24px rgba(0,0,0,0.4);">
      <div style="font-size:11px;letter-spacing:3px;color:#aaa;margin-bottom:8px;">
        WEIGHTED CONSENSUS · DQN · {len(df)} START YEARS · Results {results_date}
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
    others = sorted([(e, v) for e, v in etf_summary.items() if e != winner],
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
        sorted_etfs = sorted(etf_summary.keys(), key=lambda e: -etf_summary[e]["cum_score"])
        fig_bar = go.Figure(go.Bar(
            x=sorted_etfs,
            y=[etf_summary[e]["cum_score"] for e in sorted_etfs],
            marker_color=[colour_map.get(e, "#888") for e in sorted_etfs],
            text=[f"{etf_summary[e]['n_years']}yr · {etf_summary[e]['score_share']*100:.0f}%<br>{etf_summary[e]['cum_score']:.2f}"
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
        fig_sc = go.Figure()
        for _, row in df.iterrows():
            etf = row["Signal"]
            col = colour_map.get(etf, "#888")
            fig_sc.add_trace(go.Scatter(
                x=[row["Start Year"]], y=[row["_z_score_raw"]],
                mode="markers+text",
                marker=dict(size=18, color=col, line=dict(color="white", width=1)),
                text=[etf], textposition="top center",
                name=etf, showlegend=False,
                hovertemplate=f"<b>{etf}</b><br>Year: {row['Start Year']}<br>"
                              f"Z: {row['_z_score_raw']:.2f}σ<br>"
                              f"Return: {row['_ann_return_raw']*100:.1f}%<extra></extra>"
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
        f"Results dated: **{results_date}**"
    )

    tbl_df = df[["Start Year", "Signal", "Wtd Score", "Conviction", "Z-Score",
                 "Ann. Return", "Sharpe", "Max Drawdown", "Lookback"]].copy()

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
To protect your GitHub Actions minutes (~{len(df) * 90} mins per full sweep).
The sweep auto-runs at 8pm EST daily so you rarely need to trigger it manually.

**Split Signal warning (score share < 40%)**
Signals are fragmented — no single ETF dominates across regimes. Treat with caution.

**Date stamp**
Results filename includes the run date (e.g. `sweep_20260304.json`).
The app only shows today's results if available, otherwise yesterday's with a warning banner.
Previous day's files are automatically deleted at 8pm EST before the new sweep starts.
""")
