# evaluate.py
# Runs backtest on test set, computes performance vs SPY/AGG benchmarks.
# Supports Option A and Option B via --option.
# Usage:
#   python evaluate.py --option a --start_year 2015 --fee_bps 10 --tsl 10 --z 1.1
#   python evaluate.py --option b --start_year 2015 --fee_bps 10 --tsl 10 --z 1.1
#   python evaluate.py --walkforward-all --option a    # compute all years 2008-2025

import argparse
import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import config
from data_download import load_local
from features import build_features
from env import make_splits
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


def _sharpe(rets: np.ndarray, tbill: float = 0.036) -> float:
    excess = rets - tbill / 252
    return float(excess.mean() / (excess.std() + 1e-9) * np.sqrt(252))


def _max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    dd   = (equity - peak) / (peak + 1e-9)
    return float(dd.min())


def _calmar(ann_ret: float, max_dd: float) -> float:
    return ann_ret / (abs(max_dd) + 1e-9)


def _q_zscore(q_vals: np.ndarray) -> np.ndarray:
    mu  = q_vals.mean()
    std = q_vals.std() + 1e-9
    return (q_vals - mu) / std


def run_backtest(option: str,
                 start_year: int,
                 fee_bps:    int   = config.DEFAULT_FEE_BPS,
                 tsl_pct:    float = config.DEFAULT_TSL_PCT,
                 z_reentry:  float = config.DEFAULT_Z_REENTRY) -> dict:

    # Paths
    model_dir = get_model_dir(option)
    weights_path = os.path.join(model_dir, "dqn_best.pt")
    summary_path = os.path.join(model_dir, "training_summary.json")

    action_names = get_action_names(option)
    etf_list = action_names[1:]

    print(f"\n{'='*60}")
    print(f"  P2-ETF-DQN-ENGINE — Evaluation (Option {option.upper()})")
    print(f"  Start year : {start_year}")
    print(f"{'='*60}")

    # ── Load ──────────────────────────────────────────────────────────────────
    data = load_local()
    if not data:
        raise RuntimeError("No local data. Run data_download.py first.")

    etf_prices = data["etf_prices"]
    macro      = data["macro"]

    feat_df = build_features(etf_prices, macro, start_year=start_year, etf_list=etf_list)

    fee_pct = fee_bps / 10_000
    _, _, test_env = make_splits(feat_df, etf_prices, macro, start_year,
                                 fee_pct=fee_pct, action_names=action_names)

    # ── Load agent ────────────────────────────────────────────────────────────
    agent = DQNAgent(state_size=test_env.observation_size, n_actions=len(action_names))
    agent.load(weights_path)

    # ── Backtest with TSL ─────────────────────────────────────────────────────
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
            if z_scores.max() >= z_reentry:
                is_stopped = False
                action     = int(q_values.argmax())
            else:
                action = 0
        else:
            action = int(q_values.argmax())

        next_state, reward, done, info = test_env.step(action)

        eq = test_env.equity
        if action != 0:
            if eq > peak_equity:
                peak_equity = eq
            if eq < peak_equity * (1 - tsl_pct / 100):
                is_stopped  = True

        rets.append(info["day_ret"])
        allocations.append(action_names[action])
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

    # Final signal and conviction
    last_q = np.array(q_vals_log[-1]) if q_vals_log else np.zeros(len(action_names))
    last_z_arr = _q_zscore(last_q)
    last_action_idx = int(last_q.argmax())
    best_z = float(last_z_arr[last_action_idx])
    best_z = best_z if np.isfinite(best_z) else 0.0
    final_signal = action_names[last_action_idx] if allocations else "CASH"
    conviction = ("Very High" if best_z >= 2.0 else
                  "High"      if best_z >= 1.5 else
                  "Moderate"  if best_z >= 1.0 else "Low")

    results = dict(
        option           = option,
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
        final_signal     = final_signal,
        conviction       = conviction,
        z_score          = round(best_z, 4),
    )

    eval_path = f"evaluation_results_{option}.json"
    with open(eval_path, "w") as f:
        json.dump(results, f, indent=2)

    # ── Write date-stamped sweep cache if this is a sweep year ────────────────
    sweep_years = [2008, 2013, 2015, 2017, 2019, 2021]
    if start_year in sweep_years:
        today_str = datetime.now(timezone.utc).strftime("%Y%m%d")
        sweep_payload = {
            "signal":     final_signal,
            "top_held":   final_signal,
            "ann_return": round(ann_ret, 6),
            "z_score":    round(best_z,  4),
            "sharpe":     round(sharpe,  4),
            "max_dd":     round(max_dd,  6),
            "conviction": conviction,
            "lookback":   results.get("lookback", config.LOOKBACK_WINDOW),
            "start_year": start_year,
            "sweep_date": today_str,
        }
        sweep_dir = os.path.join("results", f"option_{option}")
        os.makedirs(sweep_dir, exist_ok=True)
        sweep_fname = os.path.join(sweep_dir, f"sweep_{start_year}_{today_str}.json")
        with open(sweep_fname, "w") as sf:
            json.dump(sweep_payload, sf, indent=2)
        print(f"  Sweep cache saved → {sweep_fname}")
        print(f"  Sweep signal: {final_signal}  z={best_z:.3f}  conviction={conviction}")

    print(f"\n  Ann. Return  : {ann_ret:.2%}")
    print(f"  Sharpe Ratio : {sharpe:.3f}")
    print(f"  Max Drawdown : {max_dd:.2%}")
    print(f"  Calmar Ratio : {calmar:.3f}")
    print(f"  Hit Ratio    : {hit:.1%}")
    print(f"  Benchmarks   : {bench}")
    print(f"\n  Results saved → {eval_path}")

    return results


def run_walkforward_all(option: str, start_years: list = None) -> dict:
    """
    For each start_year in start_years (default 2008-2025), run backtest
    and save per‑year results to a JSON file stamped with today's date.
    """
    if start_years is None:
        start_years = list(range(2008, 2026))  # 2008..2025 inclusive

    today_str = datetime.now(timezone.utc).strftime("%Y%m%d")
    out_dir = f"results/option_{option}/walkforward"
    os.makedirs(out_dir, exist_ok=True)

    results_by_year = {}
    for year in start_years:
        print(f"\n{'='*60}")
        print(f"  Walk-forward for start year {year} (Option {option})")
        print(f"{'='*60}")

        # Reuse the existing backtest function (which already does train+eval)
        res = run_backtest(option, year, fee_bps=10, tsl_pct=10, z_reentry=1.1)

        # Extract the fields we need for consensus
        metrics = {
            "year":         year,
            "signal":       res.get("final_signal", "?"),
            "ann_return":   res.get("ann_return", 0.0),
            "z_score":      res.get("z_score", 0.0),
            "sharpe":       res.get("sharpe", 0.0),
            "max_dd":       res.get("max_drawdown", 0.0),
            "conviction":   res.get("conviction", "?"),
            "lookback":     res.get("lookback", config.LOOKBACK_WINDOW),
            "option":       option,
            "sweep_date":   today_str,
        }
        results_by_year[year] = metrics

        # Save after each year in case of interruption
        out_file = os.path.join(out_dir, f"sweep_{today_str}.json")
        with open(out_file, "w") as f:
            json.dump(results_by_year, f, indent=2, default=str)

    print(f"\n✅ Walk-forward complete for Option {option}. Results saved to {out_file}")
    return results_by_year


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--option", choices=["a", "b"], default="a",
                        help="Option to evaluate: a (FI/Commodities) or b (Equity)")
    parser.add_argument("--start_year", type=int, default=config.DEFAULT_START_YEAR)
    parser.add_argument("--fee_bps",    type=int, default=config.DEFAULT_FEE_BPS)
    parser.add_argument("--tsl",        type=float, default=config.DEFAULT_TSL_PCT)
    parser.add_argument("--z",          type=float, default=config.DEFAULT_Z_REENTRY)
    parser.add_argument("--walkforward-all", action="store_true",
                        help="Run walk-forward for all start years (2008-2025)")
    args = parser.parse_args()

    if args.walkforward_all:
        run_walkforward_all(args.option)
    else:
        run_backtest(args.option, args.start_year, args.fee_bps, args.tsl, args.z)
