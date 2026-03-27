# env.py
# Custom Gym-style trading environment for the Dueling DQN agent.
# State  : flattened lookback window of technical indicators + macro
# Actions: 0=CASH, 1..N = ETF indices (maps to provided action list)
# Reward : excess return over T-bill - transaction cost on switches,
#          scaled by inverse realised volatility to penalise drawdowns

import numpy as np
import pandas as pd

import config


class ETFTradingEnv:
    """
    Single-asset-at-a-time ETF selection environment.

    Parameters
    ----------
    feat_df      : pd.DataFrame — full feature matrix (one row per trading day)
    price_df     : pd.DataFrame — ETF close prices aligned to feat_df index
    macro_df     : pd.DataFrame — macro data (for TBILL_3M fallback)
    start_idx    : int — first index to start an episode from
    end_idx      : int — last index (exclusive)
    fee_pct      : float — one-way transaction cost as fraction (e.g. 0.001 = 10bps)
    lookback     : int — window size fed as state
    tsl_pct      : float — trailing stop loss % (applied post-signal in backtest)
    action_names : list — list of ETF names (including CASH as first element)
    """

    def __init__(self,
                 feat_df:   pd.DataFrame,
                 price_df:  pd.DataFrame,
                 macro_df:  pd.DataFrame,
                 start_idx: int   = 0,
                 end_idx:   int   = None,
                 fee_pct:   float = config.DEFAULT_FEE_BPS / 10_000,
                 lookback:  int   = config.LOOKBACK_WINDOW,
                 tsl_pct:   float = config.DEFAULT_TSL_PCT / 100,
                 action_names: list = None):

        self.feat_df   = feat_df.reset_index(drop=True)
        self.price_df  = price_df.reindex(feat_df.index).reset_index(drop=True)
        self.macro_df  = macro_df.reindex(feat_df.index).ffill().reset_index(drop=True)
        self.fee_pct   = fee_pct
        self.lookback  = lookback
        self.tsl_pct   = tsl_pct

        # Action space
        if action_names is None:
            # Default: Option A (existing)
            self.actions = config.ACTIONS
        else:
            self.actions = action_names
        self.n_actions = len(self.actions)

        self.start_idx = max(start_idx, lookback - 1)
        self.end_idx   = end_idx if end_idx is not None else len(self.feat_df) - 1

        self.n_features = feat_df.shape[1] * lookback   # flattened state size

        self.reset()

    # ── Gym-style interface ───────────────────────────────────────────────────

    def reset(self) -> np.ndarray:
        # FIX: randomise start within first 50% of window so agent sees diverse sequences
        max_rand = self.start_idx + (self.end_idx - self.start_idx) // 2
        self.current_idx    = int(np.random.randint(self.start_idx, max(self.start_idx + 1, max_rand)))
        self.held_action    = 0          # start in CASH
        self.peak_equity    = 1.0
        self.equity         = 1.0
        self.is_stopped_out = False
        return self._get_state()

    def step(self, action: int):
        """
        Execute one step.
        Returns (next_state, reward, done, info)
        """
        assert 0 <= action < self.n_actions

        prev_idx = self.current_idx
        self.current_idx += 1
        done = (self.current_idx >= self.end_idx)

        # ── Transaction cost on switch ────────────────────────────────────────
        switched   = (action != self.held_action)
        t_cost     = self.fee_pct if switched else 0.0
        if switched:
            self.held_action = action
            self.peak_equity = self.equity   # reset peak on new position

        # ── Daily return of chosen action ─────────────────────────────────────
        if action == 0:  # CASH — earn T-bill
            tbill_rate = self._get_tbill(prev_idx)
            day_ret    = tbill_rate / 252
        else:
            etf = self.actions[action]
            if etf in self.price_df.columns:
                p0 = self.price_df[etf].iloc[prev_idx]
                p1 = self.price_df[etf].iloc[self.current_idx]
                day_ret = (p1 / (p0 + 1e-9)) - 1.0
            else:
                day_ret = 0.0

        day_ret -= t_cost
        self.equity *= (1.0 + day_ret)

        # ── Trailing stop-loss check ──────────────────────────────────────────
        if action != 0:
            if self.equity > self.peak_equity:
                self.peak_equity = self.equity
            if self.equity < self.peak_equity * (1 - self.tsl_pct):
                self.is_stopped_out = True

        # ── Risk-adjusted reward ──────────────────────────────────────────────
        # FIX: clamp vol so CASH (vol~0.005) doesn't get 30x reward amplification
        tbill_daily = self._get_tbill(prev_idx) / 252
        excess_ret  = day_ret - tbill_daily
        vol_21d     = self._get_vol(action, prev_idx)
        vol_scale   = float(np.clip(vol_21d, config.REWARD_VOL_MIN, config.REWARD_VOL_MAX))
        reward      = excess_ret / vol_scale
        # FIX: small bonus when ETF (not CASH) beats T-bill — discourage CASH collapse
        if action != 0 and excess_ret > 0:
            reward *= config.REWARD_ETF_BONUS

        next_state  = self._get_state()
        info        = dict(day_ret=day_ret, equity=self.equity,
                           action_name=self.actions[action],
                           tsl_triggered=self.is_stopped_out)

        return next_state, reward, done, info

    @property
    def observation_size(self) -> int:
        # FIX: +n_actions for one-hot position encoding appended to state
        return self.n_features + self.n_actions

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_state(self) -> np.ndarray:
        """Return flattened lookback window + one-hot current position."""
        start = self.current_idx - self.lookback + 1
        end   = self.current_idx + 1
        window = self.feat_df.iloc[start:end].values.astype(np.float32)
        if len(window) < self.lookback:
            pad    = np.zeros((self.lookback - len(window), self.feat_df.shape[1]),
                              dtype=np.float32)
            window = np.vstack([pad, window])
        # FIX: append one-hot current position so agent knows what it is holding
        position = np.zeros(self.n_actions, dtype=np.float32)
        position[self.held_action] = 1.0
        return np.concatenate([window.flatten(), position])

    def _get_tbill(self, idx: int) -> float:
        """Annual T-bill rate at given index (fraction)."""
        if "macro_TBILL_3M" in self.macro_df.columns:
            val = self.macro_df["macro_TBILL_3M"].iloc[idx]
            if not np.isnan(val):
                return float(val) / 100.0
        return 0.036   # fallback 3.6%

    def _get_vol(self, action: int, idx: int) -> float:
        """21d annualised vol for scaling reward."""
        if action == 0:
            return 0.005   # CASH ~ zero vol
        etf     = self.actions[action]
        vol_col = f"{etf}_Vol21d"
        if vol_col in self.feat_df.columns:
            val = self.feat_df[vol_col].iloc[idx]
            if not np.isnan(val) and val > 0:
                return float(val)
        return 0.15   # fallback 15% vol


# ── Train / Val / Test splitter ───────────────────────────────────────────────

def make_splits(feat_df: pd.DataFrame,
                price_df: pd.DataFrame,
                macro_df: pd.DataFrame,
                start_year: int,
                fee_pct: float = config.DEFAULT_FEE_BPS / 10_000,
                lookback:  int = config.LOOKBACK_WINDOW,
                action_names: list = None):
    """
    Returns three ETFTradingEnv instances: train, val, test.
    Split is 80/10/10 of the date range from start_year onwards.
    If action_names is provided, use that list; otherwise default to config.ACTIONS.
    """
    # Filter by start year
    mask     = feat_df.index.year >= start_year
    feat_sub = feat_df[mask].copy()
    n        = len(feat_sub)

    n_train = int(n * config.TRAIN_SPLIT)
    n_val   = int(n * config.VAL_SPLIT)

    train_env = ETFTradingEnv(feat_sub.iloc[:n_train],
                              price_df, macro_df,
                              fee_pct=fee_pct, lookback=lookback,
                              action_names=action_names)

    val_env   = ETFTradingEnv(feat_sub.iloc[n_train : n_train + n_val],
                              price_df, macro_df,
                              fee_pct=fee_pct, lookback=lookback,
                              action_names=action_names)

    test_env  = ETFTradingEnv(feat_sub.iloc[n_train + n_val:],
                              price_df, macro_df,
                              fee_pct=fee_pct, lookback=lookback,
                              action_names=action_names)

    return train_env, val_env, test_env
