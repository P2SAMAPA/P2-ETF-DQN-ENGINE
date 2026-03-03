# predict.py
# Generates next-trading-day ETF signal from saved DQN weights.
# Usage:
#   python predict.py --tsl 10 --z 1.1

import argparse
import json
import os
import shutil
from datetime import datetime, date, timedelta, timezone

import numpy as np
import pandas as pd

import config
from data_download import load_local
from features import build_features
from agent import DQNAgent

WEIGHTS_PATH = os.path.join(config.MODELS_DIR, "dqn_best.pt")
SUMMARY_PATH = os.path.join(config.MODELS_DIR, "training_summary.json")
PRED_PATH    = "latest_prediction.json"

def next_trading_day(from_date=None) -> date:
    """Returns next NYSE trading day using pandas_market_calendars — no hardcoded holidays."""
    try:
        import pandas_market_calendars as mcal
        nyse  = mcal.get_calendar("NYSE")
        start = from_date or date.today()
        sched = nyse.schedule(
            start_date=start.strftime("%Y-%m-%d"),
            end_date=(start + timedelta(days=14)).strftime("%Y-%m-%d"),
        )
        trading_dates = [d.date() for d in mcal.date_range(sched, frequency="1D")]
        for d in trading_dates:
            if d > start:
                return d
    except Exception:
        pass
    # Fallback: weekend skip only
    d = (from_date or date.today()) + timedelta(days=1)
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d


def _q_zscore(q_vals: np.ndarray) -> np.ndarray:
    mu  = q_vals.mean()
    std = q_vals.std() + 1e-9
    return (q_vals - mu) / std


def download_from_hf():
    """Pull weights + data from HF Dataset if not present locally."""
    try:
        from huggingface_hub import hf_hub_download
        token = config.HF_TOKEN or None
        os.makedirs(config.DATA_DIR,   exist_ok=True)
        os.makedirs(config.MODELS_DIR, exist_ok=True)

        for f in ["etf_prices", "macro"]:
            try:
                dl = hf_hub_download(repo_id=config.HF_DATASET_REPO,
                                     filename=f"data/{f}.parquet",
                                     repo_type="dataset", token=token)
                shutil.copy(dl, os.path.join(config.DATA_DIR, f"{f}.parquet"))
            except Exception as e:
                print(f"  data/{f}: {e}")

        for f in ["dqn_best.pt", "training_summary.json"]:
            try:
                dl = hf_hub_download(repo_id=config.HF_DATASET_REPO,
                                     filename=f"models/{f}",
                                     repo_type="dataset", token=token)
                shutil.copy(dl, os.path.join(config.MODELS_DIR, f))
                print(f"  ✓ models/{f}")
            except Exception as e:
                print(f"  models/{f}: {e}")
    except Exception as e:
        print(f"  HF download failed: {e}")


def run_predict(tsl_pct:   float = config.DEFAULT_TSL_PCT,
                z_reentry: float = config.DEFAULT_Z_REENTRY) -> dict:

    print(f"\n{'='*60}")
    print(f"  Predict — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    # ── Ensure data + weights ─────────────────────────────────────────────────
    data = load_local()
    if not data:
        print("  No local data — downloading from HF...")
        download_from_hf()
        data = load_local()
    if not data:
        print("  ERROR: No data available.")
        return {}

    if not os.path.exists(WEIGHTS_PATH):
        print("  No local weights — downloading from HF...")
        download_from_hf()
    if not os.path.exists(WEIGHTS_PATH):
        print("  ERROR: No weights available.")
        return {}

    # ── Load training metadata ────────────────────────────────────────────────
    trained_from_year = None
    trained_at        = None
    lookback          = config.LOOKBACK_WINDOW
    if os.path.exists(SUMMARY_PATH):
        with open(SUMMARY_PATH) as f:
            summary = json.load(f)
        trained_from_year = summary.get("start_year")
        trained_at        = summary.get("trained_at")
        lookback          = summary.get("lookback", config.LOOKBACK_WINDOW)

    # ── Build features ────────────────────────────────────────────────────────
    etf_prices = data["etf_prices"]
    macro      = data["macro"]
    feat_df    = build_features(etf_prices, macro)

    # ── Load agent ────────────────────────────────────────────────────────────
    # FIX: state_size must match env.py — flattened window + one-hot position (n_actions)
    state_size = feat_df.shape[1] * lookback + config.N_ACTIONS
    agent      = DQNAgent(state_size=state_size)
    agent.load(WEIGHTS_PATH)

    # ── Build current state (last lookback rows) ──────────────────────────────
    window = feat_df.iloc[-lookback:].values.astype(np.float32)
    if len(window) < lookback:
        pad    = np.zeros((lookback - len(window), feat_df.shape[1]), dtype=np.float32)
        window = np.vstack([pad, window])
    # FIX: append one-hot position — assume CASH at inference start (index 0)
    position = np.zeros(config.N_ACTIONS, dtype=np.float32)
    position[0] = 1.0   # CASH
    state = np.concatenate([window.flatten(), position])

    # ── Inference ─────────────────────────────────────────────────────────────
    q_values = agent.q_values(state)
    z_scores = _q_zscore(q_values)
    best_idx = int(q_values.argmax())
    best_z   = float(z_scores[best_idx])

    # T-bill rate
    tbill_rate = 3.6
    if "macro_TBILL_3M" in feat_df.columns:
        val = feat_df["macro_TBILL_3M"].iloc[-1]
        if not np.isnan(val):
            tbill_rate = float(val)

    # TSL / re-entry check using last 2 days
    tsl_triggered = False
    in_cash       = False
    two_day_ret   = 0.0
    if best_idx != 0:
        etf = config.ACTIONS[best_idx]
        if etf in etf_prices.columns:
            last2       = etf_prices[etf].iloc[-3:]
            two_day_ret = float((last2.iloc[-1] / last2.iloc[0]) - 1) * 100
            if two_day_ret <= -tsl_pct:
                tsl_triggered = True
                if best_z < z_reentry:
                    in_cash = True

    final_signal = "CASH" if in_cash else config.ACTIONS[best_idx]

    # Signal date — use NYSE calendar
    now_est  = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=5)
    today    = now_est.date()
    try:
        import pandas_market_calendars as mcal
        nyse  = mcal.get_calendar("NYSE")
        sched = nyse.schedule(
            start_date=today.strftime("%Y-%m-%d"),
            end_date=today.strftime("%Y-%m-%d"),
        )
        is_trading_day = not sched.empty
    except Exception:
        is_trading_day = today.weekday() < 5
    if is_trading_day and now_est.hour < 16:
        signal_date = today
    else:
        signal_date = next_trading_day(today)

    # Per-action probabilities (softmax of Q-values for display)
    q_shifted = q_values - q_values.max()
    exp_q     = np.exp(q_shifted / 0.1)
    probs     = exp_q / exp_q.sum()
    prob_dict = {config.ACTIONS[i]: round(float(probs[i]), 4)
                 for i in range(config.N_ACTIONS)}

    output = dict(
        as_of_date        = str(signal_date),
        final_signal      = final_signal,
        final_confidence  = round(float(probs[best_idx]), 4),
        z_score           = round(best_z, 3),
        q_values          = {config.ACTIONS[i]: round(float(q_values[i]), 4)
                             for i in range(config.N_ACTIONS)},
        probabilities     = prob_dict,
        tbill_rate        = round(tbill_rate, 3),
        tsl_status        = dict(
            two_day_cumul_pct = round(two_day_ret, 2),
            tsl_triggered     = tsl_triggered,
            in_cash           = in_cash,
            z_reentry         = z_reentry,
            tsl_pct           = tsl_pct,
        ),
        trained_from_year = trained_from_year,
        trained_at        = trained_at,
    )

    with open(PRED_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n  Signal date  : {signal_date}")
    print(f"  Final signal : {final_signal}")
    print(f"  Z-score      : {best_z:.2f}σ")
    print(f"  Confidence   : {float(probs[best_idx]):.1%}")
    for act, p in prob_dict.items():
        print(f"    {act:<8} Q={q_values[config.ACTIONS.index(act)]:.3f}  p={p:.3f}")
    print(f"\n  Saved → {PRED_PATH}")

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tsl", type=float, default=config.DEFAULT_TSL_PCT)
    parser.add_argument("--z",   type=float, default=config.DEFAULT_Z_REENTRY)
    args = parser.parse_args()
    run_predict(tsl_pct=args.tsl, z_reentry=args.z)
