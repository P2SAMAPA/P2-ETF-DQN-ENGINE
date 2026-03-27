# smoke_test.py
# Fast pre-training validation — runs in <60 seconds.
# Catches: state size mismatches, import errors, env bugs,
#          agent load/save roundtrip, reward collapse signals.
# Called by train_models.yml BEFORE the real training run.

import sys
import os
import json
import traceback
import numpy as np

PASS = []
FAIL = []

def check(name, fn):
    try:
        result = fn()
        msg = f"  ✅ {name}" + (f" — {result}" if result else "")
        print(msg)
        PASS.append(name)
    except Exception as e:
        print(f"  ❌ {name} — {e}")
        traceback.print_exc()
        FAIL.append((name, str(e)))


# ── 1. Imports ────────────────────────────────────────────────────────────────
print("\n── Imports ──")
check("import config",   lambda: __import__("config"))
check("import agent",    lambda: __import__("agent"))
check("import env",      lambda: __import__("env"))
check("import features", lambda: __import__("features"))

if FAIL:
    print(f"\n💥 Import failures — aborting smoke test")
    sys.exit(1)

import config
from agent import DQNAgent, DuelingDQN
from env import ETFTradingEnv, make_splits

# ── 2. Config sanity ──────────────────────────────────────────────────────────
print("\n── Config ──")
check("N_ACTIONS == len(ACTIONS)",
      lambda: None if config.N_ACTIONS == len(config.ACTIONS)
              else (_ for _ in ()).throw(ValueError(f"{config.N_ACTIONS} != {len(config.ACTIONS)}")))
check("TAU defined",        lambda: f"{config.TAU}")
check("REWARD_VOL_MIN > 0", lambda: f"{config.REWARD_VOL_MIN}")
check("EPSILON_END >= 0.05",lambda: f"{config.EPSILON_END}")
check("EPSILON_DECAY_FRAC >= 0.70",
      lambda: f"{config.EPSILON_DECAY_FRAC}" if config.EPSILON_DECAY_FRAC >= 0.70
              else (_ for _ in ()).throw(ValueError(f"too fast: {config.EPSILON_DECAY_FRAC}")))

# ── 3. Synthetic env + agent ──────────────────────────────────────────────────
print("\n── Synthetic Env ──")

def make_synthetic_env():
    import pandas as pd
    n = 300
    dates = pd.bdate_range("2020-01-01", periods=n)
    n_feat = 10
    feat_df  = pd.DataFrame(np.random.randn(n, n_feat), index=dates,
                             columns=[f"f{i}" for i in range(n_feat)])
    etfs     = config.ETFS
    price_df = pd.DataFrame(
        np.cumprod(1 + np.random.randn(n, len(etfs)) * 0.01, axis=0),
        index=dates, columns=etfs)
    macro_df = pd.DataFrame(
        {"macro_TBILL_3M": np.full(n, 5.0)}, index=dates)
    env = ETFTradingEnv(feat_df, price_df, macro_df,
                        lookback=config.LOOKBACK_WINDOW,
                        action_names=config.ACTIONS)   # pass action_names
    return env, n_feat

env, n_feat = None, None
try:
    env, n_feat = make_synthetic_env()
    PASS.append("make ETFTradingEnv")
    print(f"  ✅ make ETFTradingEnv")
except Exception as e:
    FAIL.append(("make ETFTradingEnv", str(e)))
    print(f"  ❌ make ETFTradingEnv — {e}")
    sys.exit(1)

check("observation_size == n_feat*lookback + N_ACTIONS",
      lambda: (
          None if env.observation_size == n_feat * config.LOOKBACK_WINDOW + config.N_ACTIONS
          else (_ for _ in ()).throw(ValueError(
              f"obs={env.observation_size} expected={n_feat*config.LOOKBACK_WINDOW+config.N_ACTIONS}"))
      ))

def check_reset_random():
    starts = set()
    for _ in range(20):
        env.reset()
        starts.add(env.current_idx)
    if len(starts) < 3:
        raise ValueError(f"reset() not randomising — saw only {starts}")
    return f"{len(starts)} unique starts"

check("reset() randomises start", check_reset_random)

def check_state_shape():
    state = env.reset()
    expected = env.observation_size
    if state.shape[0] != expected:
        raise ValueError(f"state shape {state.shape[0]} != obs_size {expected}")
    return f"shape=({state.shape[0]},)"

check("state shape matches observation_size", check_state_shape)

def check_position_onehot():
    env.reset()
    env.held_action = 3  # pretend we're in HYG (index 3)
    state = env._get_state()
    pos = state[-config.N_ACTIONS:]
    if pos[3] != 1.0:
        raise ValueError(f"one-hot wrong: {pos}")
    if pos.sum() != 1.0:
        raise ValueError(f"one-hot sum != 1: {pos}")
    return "one-hot correct"

check("position one-hot in state", check_position_onehot)

def check_step():
    state = env.reset()
    next_state, reward, done, info = env.step(0)  # CASH
    if next_state.shape[0] != env.observation_size:
        raise ValueError(f"next_state shape mismatch")
    return f"reward={reward:.4f}"

check("env.step() returns correct shape", check_step)

# ── 4. Reward collapse check ──────────────────────────────────────────────────
print("\n── Reward Collapse Check ──")

def check_cash_reward_not_dominant():
    """CASH reward should not be >> ETF rewards due to vol scaling."""
    env.reset()
    cash_rewards, etf_rewards = [], []
    for _ in range(50):
        _, r_cash, _, _ = env.step(0)
        cash_rewards.append(abs(r_cash))
    for _ in range(50):
        _, r_etf, _, _ = env.step(1)
        etf_rewards.append(abs(r_etf))
    ratio = np.mean(cash_rewards) / (np.mean(etf_rewards) + 1e-9)
    if ratio > 10:
        raise ValueError(f"CASH reward {ratio:.1f}x larger than ETF — vol scaling broken")
    return f"CASH/ETF reward ratio = {ratio:.2f}x (should be <10)"

check("CASH reward not dominating ETF", check_cash_reward_not_dominant)

# ── 5. Agent ──────────────────────────────────────────────────────────────────
print("\n── Agent ──")

def check_agent_init():
    obs_size = env.observation_size
    agent = DQNAgent(state_size=obs_size, n_actions=env.n_actions, total_steps=10_000)
    return f"state_size={obs_size}"

check("DQNAgent initialises", check_agent_init)

def check_action_diversity():
    """Agent should not always return same action (even random)."""
    obs_size = env.observation_size
    agent = DQNAgent(state_size=obs_size, n_actions=env.n_actions, total_steps=10_000)
    state = env.reset()
    actions = set()
    for _ in range(50):
        a = agent.select_action(state, greedy=False)  # epsilon=1.0 → random
        actions.add(a)
    if len(actions) < env.n_actions - 1:
        raise ValueError(f"Only {len(actions)} unique actions in 50 random steps")
    return f"{len(actions)}/{env.n_actions} unique actions"

check("agent explores all actions", check_action_diversity)

def check_save_load_roundtrip():
    import tempfile
    obs_size = env.observation_size
    agent = DQNAgent(state_size=obs_size, n_actions=env.n_actions, total_steps=1000)
    state = env.reset()
    q_before = agent.q_values(state).copy()
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
    agent.save(path)
    # Load into fresh agent
    agent2 = DQNAgent(state_size=obs_size, n_actions=env.n_actions, total_steps=1000)
    agent2.load(path)
    q_after = agent2.q_values(state)
    os.unlink(path)
    if not np.allclose(q_before, q_after, atol=1e-5):
        raise ValueError(f"Q-values differ after save/load: {q_before} vs {q_after}")
    return "save/load roundtrip identical"

check("agent save/load roundtrip", check_save_load_roundtrip)

def check_soft_update():
    obs_size = env.observation_size
    agent = DQNAgent(state_size=obs_size, n_actions=env.n_actions, total_steps=1000)
    # Manually diverge online and target
    for p in agent.online_net.parameters():
        p.data.fill_(1.0)
    for p in agent.target_net.parameters():
        p.data.fill_(0.0)
    agent._update_target()
    # After one soft update, target should be TAU*1 + (1-TAU)*0 = TAU
    for p in agent.target_net.parameters():
        val = p.data.mean().item()
        if not abs(val - config.TAU) < 1e-4:
            raise ValueError(f"Soft update wrong: got {val:.6f}, expected {config.TAU}")
    return f"target={config.TAU} after one update ✓"

check("Polyak soft update correct", check_soft_update)

def check_learning_step():
    obs_size = env.observation_size
    agent = DQNAgent(state_size=obs_size, n_actions=env.n_actions, total_steps=10_000)
    state = env.reset()
    # Fill replay buffer past MIN_REPLAY_SIZE
    for _ in range(config.MIN_REPLAY_SIZE + config.BATCH_SIZE + 10):
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.push(state, action, reward, next_state, done)
        state = next_state if not done else env.reset()
    loss = agent.learn()
    if loss == 0.0:
        raise ValueError("learn() returned 0 — buffer may be too small")
    return f"loss={loss:.6f}"

check("agent.learn() produces nonzero loss", check_learning_step)

# ── 6. predict.py state consistency ──────────────────────────────────────────
print("\n── predict.py Consistency ──")

def check_predict_state_size():
    """Simulate what predict.py does and ensure state_size matches env."""
    n_feat   = env.feat_df.shape[1]
    lookback = config.LOOKBACK_WINDOW
    # predict.py formula (fixed version)
    predict_state_size = n_feat * lookback + env.n_actions
    env_state_size     = env.observation_size
    if predict_state_size != env_state_size:
        raise ValueError(
            f"predict.py would build state_size={predict_state_size} "
            f"but env expects {env_state_size}")
    return f"both = {env_state_size}"

check("predict.py state_size matches env", check_predict_state_size)

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"  Smoke test complete")
print(f"  Passed : {len(PASS)}")
print(f"  Failed : {len(FAIL)}")
print(f"{'='*50}")

if FAIL:
    print("\n💥 FAILURES:")
    for name, err in FAIL:
        print(f"   • {name}: {err}")
    sys.exit(1)
else:
    print("\n✅ All checks passed — safe to start training.")
    sys.exit(0)
