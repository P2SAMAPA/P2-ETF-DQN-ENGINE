# predict.py
# Generates next-trading-day ETF signal from saved DQN weights.
# Supports Option A and Option B via --option.
# Usage:
#   python predict.py --option a --tsl 10 --z 1.1
#   python predict.py --option b --tsl 10 --z 1.1

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


def get_action_names(option: str) -> list:
    if option == 'a':
        etfs = config.ETFS
    elif option == 'b':
        etfs = config.OPTION_B_ETFS
    else:
        raise ValueError(f"Unknown option: {option}")
    return ["CASH"] + etfs


def get_model_dir(option: str) -> str:
    base = config.MODELS_DIR
    if option == 'a':
        return base
    else:
        return os.path.join(base, f"option_{option}")


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
    d = (from_date or date.today()) + timedelta(days=1)
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d


def _q_zscore(q_vals: np.ndarray) -> np.ndarray:
    mu  = q_vals.mean()
    std = q_vals.std() + 1e-9
    return (q_vals - mu) / std


def download_from_hf(option: str):
    """Pull weights + data from HF Dataset if not present locally."""
    try:
        from huggingface_hub import hf_hub_download
        token = config.HF_TOKEN or None
        os.makedirs(config.DATA_DIR,   exist_ok=True)
        model_dir = get_model_dir(option)
        os.makedirs(model_dir, exist_ok=True)

        for f in ["etf_prices", "macro"]:
            try:
                dl = hf_hub_download(repo_id=config.HF_DATASET_REPO,
                                     filename=f"data/{f}.parquet",
                                     repo_type="dataset", token=token)
                shutil.copy(dl, os.path.join(config.DATA_DIR, f"{f}.parquet"))
                print(f"  ✓ data/{f}.parquet")
            except Exception as e:
                print(f"  data/{f}: {e}")

        for f in ["dqn_best.pt", "training_summary.json"]:
            try:
                remote_path = f"models/{f}" if option == 'a' else f"models/option_{option}/{f}"
                dl = hf_hub_download(repo_id=config.HF_DATASET_REPO,
                                     filename=remote_path,
                                     repo_type="dataset", token=token)
                shutil.copy(dl, os.path.join(model_dir, f))
                print(f"  ✓ {remote_path}")
            except Exception as e:
                print(f"  {remote_path}: {e}")
    except Exception as e:
        print(f"  HF download failed: {e}")


def run_predict(option: str,
                tsl_pct:   float = config.DEFAULT_TSL_PCT,
                z_reentry: float = config.DEFAULT_Z_REENTRY) -> dict:

    print(f"\n{'='*60}")
    print(f"  Predict — Option {option.upper()} · {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    # ── Ensure data + weights ─────────────────────────────────────────────────
    data = load_local()
    if not data:
        print("  No local data — downloading from HF...")
        download_from_hf(option)
        data = load_local()
    if not data:
        print("  ERROR: No data available.")
        return {}

    weights_path = os.path.join(get_model_dir(option), "dqn_best.pt")
    summary_path = os.path.join(get_model_dir(option), "training_summary.json")
    if not os.path.exists(weights_path):
        print("  No local weights — downloading from HF...")
        download_from_hf(option)
    if not os.path.exists(weights_path):
        print("  ERROR: No weights available.")
        return {}

    # ── Load training metadata ────────────────────────────────────────────────
    trained_from_year = None
    trained_at        = None
    lookback          = config.LOOKBACK_WINDOW
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            summary = json.load(f)
        trained_from_year = summary.get("start_year")
        trained_at        = summary.get("trained_at")
        lookback          = summary.get("lookback", config.LOOKBACK_WINDOW)
        print(f"  Loaded training summary: start_year={trained_from_year}, lookback={lookback}")

    action_names = get_action_names(option)
    n_actions = len(action_names)
    etf_list = action_names[1:]

    # ── Build features ────────────────────────────────────────────────────────
    etf_prices = data["etf_prices"]
    macro      = data["macro"]

    # Check macro data
    if macro is None or macro.empty:
        print("  WARNING: Macro data is empty. Will attempt to re-download.")
        download_from_hf(option)
        data = load_local()
        macro = data["macro"] if data else None

    # Check that all expected ETFs are present
    missing_etfs = [t for t in etf_list if t not in etf_prices.columns]
    if missing_etfs:
        print(f"  WARNING: Missing ETFs in price data: {missing_etfs}")
        print("  This will reduce the feature count and cause a state size mismatch.")
        print("  Run a full reseed to restore complete data.")

    feat_df = build_features(etf_prices, macro, etf_list=etf_list)
    n_features = feat_df.shape[1]
    print(f"  Feature matrix: {len(feat_df)} days × {n_features} features")
    print(f"  Macro data shape: {macro.shape if macro is not None else 'None'}")
    print(f"  Expected features for {len(etf_list)} ETFs: {len(etf_list)*20 + 12}")

    # ── Load agent ────────────────────────────────────────────────────────────
    state_size = n_features * lookback + n_actions
    print(f"  Computed state size: {state_size}")

    agent = DQNAgent(state_size=state_size, n_actions=n_actions)
    try:
        agent.load(weights_path)
    except RuntimeError as e:
        print(f"  ERROR loading model: {e}")
        print("  This is likely because the feature count has changed since training.")
        print("  Please ensure the dataset contains all required data and reseed if necessary.")
        return {}

    # ── Build current state (last lookback rows) ──────────────────────────────
    window = feat_df.iloc[-lookback:].values.astype(np.float32)
    if len(window) < lookback:
        pad    = np.zeros((lookback - len(window), n_features), dtype=np.float32)
        window = np.vstack([pad, window])
    # append one-hot position — assume CASH at inference start (index 0)
    position = np.zeros(n_actions, dtype=np.float32)
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
        etf = action_names[best_idx]
        if etf in etf_prices.columns:
            last2       = etf_prices[etf].iloc[-3:]
            two_day_ret = float((last2.iloc[-1] / last2.iloc[0]) - 1) * 100
            if two_day_ret <= -tsl_pct:
                tsl_triggered = True
                if best_z < z_reentry:
                    in_cash = True

    final_signal = "CASH" if in_cash else action_names[best_idx]

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
    prob_dict = {action_names[i]: round(float(probs[i]), 4)
                 for i in range(n_actions)}

    output = dict(
        option            = option,
        as_of_date        = str(signal_date),
        final_signal      = final_signal,
        final_confidence  = round(float(probs[best_idx]), 4),
        z_score           = round(best_z, 3),
        q_values          = {action_names[i]: round(float(q_values[i]), 4)
                             for i in range(n_actions)},
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

    pred_path = f"latest_prediction_{option}.json"
    with open(pred_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n  Signal date  : {signal_date}")
    print(f"  Final signal : {final_signal}")
    print(f"  Z-score      : {best_z:.2f}σ")
    print(f"  Confidence   : {float(probs[best_idx]):.1%}")
    for act, p in prob_dict.items():
        print(f"    {act:<8} Q={q_values[action_names.index(act)]:.3f}  p={p:.3f}")
    print(f"\n  Saved → {pred_path}")

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--option", choices=["a", "b"], default="a",
                        help="Option to predict: a (FI/Commodities) or b (Equity)")
    parser.add_argument("--tsl", type=float, default=config.DEFAULT_TSL_PCT)
    parser.add_argument("--z",   type=float, default=config.DEFAULT_Z_REENTRY)
    args = parser.parse_args()
    run_predict(option=args.option, tsl_pct=args.tsl, z_reentry=args.z)
