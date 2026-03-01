# train.py
# Trains the Dueling DQN agent across train/val sets.
# Usage:
#   python train.py --start_year 2015 --episodes 300
#   python train.py --start_year 2008 --episodes 500 --fee_bps 10

import argparse
import json
import os
from datetime import datetime

import numpy as np

import config
from data_download import load_local, seed, incremental_update
from features import build_features
from env import make_splits
from agent import DQNAgent

os.makedirs(config.MODELS_DIR, exist_ok=True)
WEIGHTS_PATH = os.path.join(config.MODELS_DIR, "dqn_best.pt")
SUMMARY_PATH = os.path.join(config.MODELS_DIR, "training_summary.json")


# ── Sharpe helper ─────────────────────────────────────────────────────────────

def _episode_sharpe(rewards: list) -> float:
    if len(rewards) < 2:
        return 0.0
    r = np.array(rewards)
    return float(r.mean() / (r.std() + 1e-9) * np.sqrt(252))


# ── Single episode runner ─────────────────────────────────────────────────────

def run_episode(env, agent: DQNAgent, train: bool = True) -> dict:
    state    = env.reset()
    rewards  = []
    actions  = []
    losses   = []
    done     = False

    while not done:
        action              = agent.select_action(state, greedy=not train)
        next_state, reward, done, info = env.step(action)

        if train:
            agent.push(state, action, reward, next_state, done)
            loss = agent.learn()
            if loss > 0:
                losses.append(loss)

        rewards.append(reward)
        actions.append(action)
        state = next_state

    return dict(
        total_reward = sum(rewards),
        sharpe       = _episode_sharpe(rewards),
        equity       = env.equity,
        n_steps      = len(rewards),
        avg_loss     = np.mean(losses) if losses else 0.0,
        epsilon      = agent.epsilon,
        actions      = actions,
    )


# ── Main training loop ────────────────────────────────────────────────────────

def run_training(start_year: int,
                 n_episodes: int,
                 fee_bps:    int,
                 lookback:   int = config.LOOKBACK_WINDOW) -> dict:

    print(f"\n{'='*60}")
    print(f"  P2-ETF-DQN-ENGINE — Training")
    print(f"  Start year : {start_year}")
    print(f"  Episodes   : {n_episodes}")
    print(f"  Fee        : {fee_bps} bps")
    print(f"  Started    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    # ── Load data ─────────────────────────────────────────────────────────────
    data = load_local()
    if not data:
        print("No local data — downloading...")
        data = seed()

    etf_prices = data["etf_prices"]
    macro      = data["macro"]

    # ── Build features ────────────────────────────────────────────────────────
    print("\nBuilding features...")
    feat_df = build_features(etf_prices, macro, start_year=start_year)
    print(f"  Feature matrix: {feat_df.shape[0]} days × {feat_df.shape[1]} features")

    # ── Make environments ─────────────────────────────────────────────────────
    fee_pct = fee_bps / 10_000
    train_env, val_env, test_env = make_splits(
        feat_df, etf_prices, macro, start_year, fee_pct=fee_pct, lookback=lookback
    )
    print(f"  Train: {train_env.end_idx - train_env.start_idx} steps | "
          f"Val: {val_env.end_idx - val_env.start_idx} steps | "
          f"Test: {test_env.end_idx - test_env.start_idx} steps")

    # ── Initialise agent ──────────────────────────────────────────────────────
    state_size  = train_env.observation_size
    total_steps = (train_env.end_idx - train_env.start_idx) * n_episodes
    agent       = DQNAgent(state_size=state_size, total_steps=total_steps)
    print(f"  State size : {state_size}")
    print(f"  Actions    : {config.ACTIONS}")

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_sharpe = -np.inf
    history = []

    for ep in range(1, n_episodes + 1):
        train_stats = run_episode(train_env, agent, train=True)
        val_stats   = run_episode(val_env,   agent, train=False)

        row = dict(
            episode        = ep,
            train_sharpe   = round(train_stats["sharpe"], 3),
            train_equity   = round(train_stats["equity"], 4),
            val_sharpe     = round(val_stats["sharpe"], 3),
            val_equity     = round(val_stats["equity"], 4),
            avg_loss       = round(train_stats["avg_loss"], 6),
            epsilon        = round(train_stats["epsilon"], 4),
        )
        history.append(row)

        # ── Save best model (by val Sharpe) ───────────────────────────────────
        if val_stats["sharpe"] > best_val_sharpe:
            best_val_sharpe = val_stats["sharpe"]
            agent.save(WEIGHTS_PATH)
            improved = " ✓ saved"
        else:
            improved = ""

        if ep % 10 == 0 or ep == 1:
            print(f"  Ep {ep:4d}/{n_episodes} | "
                  f"train_S={train_stats['sharpe']:+.2f} "
                  f"eq={train_stats['equity']:.3f} | "
                  f"val_S={val_stats['sharpe']:+.2f} "
                  f"eq={val_stats['equity']:.3f} | "
                  f"ε={train_stats['epsilon']:.3f} "
                  f"loss={train_stats['avg_loss']:.5f}"
                  f"{improved}")

    # ── Final test evaluation ─────────────────────────────────────────────────
    print("\nRunning test evaluation with best weights...")
    agent.load(WEIGHTS_PATH)
    test_stats = run_episode(test_env, agent, train=False)
    print(f"  Test Sharpe : {test_stats['sharpe']:+.3f}")
    print(f"  Test Equity : {test_stats['equity']:.4f}")

    # ── Save training summary ─────────────────────────────────────────────────
    summary = dict(
        start_year       = start_year,
        n_episodes       = n_episodes,
        fee_bps          = fee_bps,
        lookback         = lookback,
        state_size       = state_size,
        n_features       = feat_df.shape[1],
        trained_at       = datetime.now().isoformat(),
        best_val_sharpe  = round(best_val_sharpe, 4),
        test_sharpe      = round(test_stats["sharpe"], 4),
        test_equity      = round(test_stats["equity"], 4),
        train_days       = train_env.end_idx - train_env.start_idx,
        val_days         = val_env.end_idx   - val_env.start_idx,
        test_days        = test_env.end_idx  - test_env.start_idx,
        history          = history[-20:],    # last 20 episodes for UI display
    )

    with open(SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Training summary saved → {SUMMARY_PATH}")
    print(f"  Best weights saved    → {WEIGHTS_PATH}")
    print(f"\n{'='*60}")
    print("  Training complete.")
    print(f"{'='*60}")

    return summary


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_year", type=int, default=config.DEFAULT_START_YEAR,
                        help="Training data start year (e.g. 2015)")
    parser.add_argument("--episodes",   type=int, default=config.DEFAULT_EPISODES,
                        help="Number of training episodes")
    parser.add_argument("--fee_bps",    type=int, default=config.DEFAULT_FEE_BPS,
                        help="Transaction fee in basis points")
    parser.add_argument("--lookback",   type=int, default=config.LOOKBACK_WINDOW,
                        help="Lookback window size for state")
    args = parser.parse_args()

    run_training(
        start_year = args.start_year,
        n_episodes = args.episodes,
        fee_bps    = args.fee_bps,
        lookback   = args.lookback,
    )
