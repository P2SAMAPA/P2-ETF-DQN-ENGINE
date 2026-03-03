# evaluate.py
# Runs backtest on test set, computes performance vs SPY/AGG benchmarks.
# Usage:
#   python evaluate.py --start_year 2015 --fee_bps 10 --tsl 10 --z 1.1

import argparse
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd

import config
from data_download import load_local
from features import build_features
from env import make_splits
from agent import DQNAgent

WEIGHTS_PATH = os.path.join(config.MODELS_DIR, "dqn_best.pt")
SUMMARY_PATH = os.path.join(config.MODELS_DIR, "training_summary.json")
EVAL_PATH    = "evaluation_results.json"


def _sharpe(rets: np.ndarray, tbill: float = 0.036) -> float:
    excess = rets - tbill / 252
    return float(excess.mean() / (excess.std() + 1e-9) * np.sqrt(252))


def _max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    dd   = (equity - peak) / (peak + 1e-9)
    return float(dd.min())


def _calmar(ann_ret: float, max_dd: float) -> float:
    return ann_ret / (abs(max_dd) + 1e-9)


def run_backtest(start_year: int,
                 fee_bps:    int   = config.DEFAULT_FEE_BPS,
                 tsl_pct:    float = config.DEFAULT_TSL_PCT,
                 z_reentry:  float = config.DEFAULT_Z_REENTRY) -> dict:

    print(f"\n{'='*60}")
    print(f"  P2-ETF-DQN-ENGINE — Evaluation")
    print(f"  Start year : {start_year}")
    print(f"{'='*60}")

    # ── Load ──────────────────────────────────────────────────────────────────
    data = load_local()
    if not data:
        raise RuntimeError("No local data. Run data_download.py first.")

    etf_prices = data["etf_prices"]
    macro      = data["macro"]

    feat_df = build_features(etf_prices, macro, start_year=start_year)

    fee_pct = fee_bps / 10_000
    _, _, test_env = make_splits(feat_df, etf_prices, macro, start_year,
                                 fee_pct=fee_pct)

    # ── Load agent ────────────────────────────────────────────────────────────
    agent = DQNAgent(state_size=test_env.observation_size)
    agent.load(WEIGHTS_PATH)

    # ── Backtest with TSL ─────────────────────────────────────────────────────
    # FIX: force reset to start_idx (not random) for deterministic evaluation
    test_env.current_idx    = test_env.start_idx
    test_env.held_action    = 0
    test_env.peak_equity    = 1.0
    test_env.equity         = 1.0
    test_env.is_stopped_out = False
    state        = test_env._get_state()
    rets         = []
    allocations  = []
    q_vals_log   = []
    equity_curve = [1.0]
    peak_equity  = 1.0
    is_stopped   = False
    done         = False

    while not done:
        q_values = agent.q_values(state)
        z_scores = _q_zscore(q_values)

        if is_stopped:
            # Re-enter when best z-score clears threshold
            if z_scores.max() >= z_reentry:
                is_stopped = False
                action     = int(q_values.argmax())
            else:
                action = 0  # stay in CASH
        else:
            action = int(q_values.argmax())

        next_state, reward, done, info = test_env.step(action)

        # TSL check
        eq = test_env.equity
        if action != 0:
            if eq > peak_equity:
                peak_equity = eq
            if eq < peak_equity * (1 - tsl_pct / 100):
                is_stopped  = True

        rets.append(info["day_ret"])
        allocations.append(config.ACTIONS[action])
        q_vals_log.append(q_values.tolist())
        equity_curve.append(eq)
        state = next_state

    rets   = np.array(rets)
    equity = np.array(equity_curve[1:])

    # ── Benchmark returns (over same test period) ─────────────────────────────
    test_dates = feat_df.index[
        int(len(feat_df) * (config.TRAIN_SPLIT + config.VAL_SPLIT)):
    ]
    bench        = {}
    bench_ann    = {}
    bench_equity = {}
    for b in config.BENCHMARKS:
        if b in etf_prices.columns:
            bp = etf_prices[b].reindex(test_dates).ffill().pct_change().dropna()
            bench[b]        = _sharpe(bp.values)
            cum             = float((1 + bp).prod())
            bench_ann[b]    = round(float(cum ** (252 / max(len(bp), 1)) - 1), 4)
            beq             = (1 + bp).cumprod().values
            bench_equity[b] = [round(float(v / beq[0]), 4) for v in beq]

    # ── Metrics ───────────────────────────────────────────────────────────────
    n_days  = len(rets)
    ann_ret = float((equity[-1]) ** (252 / n_days) - 1) if n_days > 0 else 0.0
    sharpe  = _sharpe(rets)
    max_dd  = _max_drawdown(equity)
    calmar  = _calmar(ann_ret, max_dd)
    hit     = float((rets > 0).mean())

    # Allocation breakdown
    alloc_counts = pd.Series(allocations).value_counts(normalize=True).to_dict()

    results = dict(
        start_year       = start_year,
        evaluated_at     = datetime.now().isoformat(),
        n_test_days      = n_days,
        ann_return       = round(ann_ret, 4),
        sharpe           = round(sharpe,  4),
        max_drawdown     = round(max_dd,  4),
        calmar           = round(calmar,  4),
        hit_ratio        = round(hit,     4),
        final_equity     = round(float(equity[-1]), 4),
        benchmark_sharpe = {k: round(v, 4) for k, v in bench.items()},
        benchmark_ann    = bench_ann,
        benchmark_equity = bench_equity,
        test_dates       = [str(d.date()) for d in test_dates],
        allocation_pct   = {k: round(v, 4) for k, v in alloc_counts.items()},
        equity_curve     = [round(float(e), 4) for e in equity],
        allocations      = allocations,
        fee_bps          = fee_bps,
        tsl_pct          = tsl_pct,
        z_reentry        = z_reentry,
    )

    with open(EVAL_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  Ann. Return  : {ann_ret:.2%}")
    print(f"  Sharpe Ratio : {sharpe:.3f}")
    print(f"  Max Drawdown : {max_dd:.2%}")
    print(f"  Calmar Ratio : {calmar:.3f}")
    print(f"  Hit Ratio    : {hit:.1%}")
    print(f"  Benchmarks   : {bench}")
    print(f"\n  Results saved → {EVAL_PATH}")

    return results


def _q_zscore(q_vals: np.ndarray) -> np.ndarray:
    mu  = q_vals.mean()
    std = q_vals.std() + 1e-9
    return (q_vals - mu) / std


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_year", type=int, default=config.DEFAULT_START_YEAR)
    parser.add_argument("--fee_bps",    type=int, default=config.DEFAULT_FEE_BPS)
    parser.add_argument("--tsl",        type=float, default=config.DEFAULT_TSL_PCT)
    parser.add_argument("--z",          type=float, default=config.DEFAULT_Z_REENTRY)
    args = parser.parse_args()

    run_backtest(args.start_year, args.fee_bps, args.tsl, args.z)
