# features.py
# Builds the DQN state matrix from raw price and macro data.
# Implements the 20 technical indicators from:
#   Yasin & Gill (2024) "Reinforcement Learning Framework for Quantitative Trading"
#   arXiv:2411.07585  — ICAIF 2024 FM4TS Workshop

import numpy as np
import pandas as pd

import config


# ── Individual indicator functions ───────────────────────────────────────────

def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))


def _macd(series: pd.Series, fast=12, slow=26, signal=9):
    ema_fast   = series.ewm(span=fast, adjust=False).mean()
    ema_slow   = series.ewm(span=slow, adjust=False).mean()
    macd_line  = ema_fast - ema_slow
    signal_line= macd_line.ewm(span=signal, adjust=False).mean()
    histogram  = macd_line - signal_line
    return macd_line, signal_line, histogram


def _stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
                period: int = 14) -> pd.Series:
    lowest  = low.rolling(period).min()
    highest = high.rolling(period).max()
    return 100 * (close - lowest) / (highest - lowest + 1e-9)


def _cci(high: pd.Series, low: pd.Series, close: pd.Series,
         period: int = 20) -> pd.Series:
    tp      = (high + low + close) / 3
    sma_tp  = tp.rolling(period).mean()
    mad     = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    return (tp - sma_tp) / (0.015 * mad + 1e-9)


def _roc(series: pd.Series, period: int = 10) -> pd.Series:
    return ((series - series.shift(period)) / (series.shift(period) + 1e-9)) * 100


def _cmo(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up    = delta.clip(lower=0).rolling(period).sum()
    down  = (-delta.clip(upper=0)).rolling(period).sum()
    return 100 * (up - down) / (up + down + 1e-9)


def _williams_r(high: pd.Series, low: pd.Series, close: pd.Series,
                period: int = 14) -> pd.Series:
    highest = high.rolling(period).max()
    lowest  = low.rolling(period).min()
    return -100 * (highest - close) / (highest - lowest + 1e-9)


def _atr(high: pd.Series, low: pd.Series, close: pd.Series,
         period: int = 14) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def _bbands(series: pd.Series, period: int = 20):
    sma   = series.rolling(period).mean()
    std   = series.rolling(period).std()
    upper = sma + 2 * std
    lower = sma - 2 * std
    pct_b = (series - lower) / (upper - lower + 1e-9)  # %B oscillator
    width = (upper - lower) / (sma + 1e-9)
    return pct_b, width


def _stoch_rsi(series: pd.Series, rsi_period=14, stoch_period=14) -> pd.Series:
    rsi     = _rsi(series, rsi_period)
    min_rsi = rsi.rolling(stoch_period).min()
    max_rsi = rsi.rolling(stoch_period).max()
    return (rsi - min_rsi) / (max_rsi - min_rsi + 1e-9)


def _ultimate_oscillator(high, low, close, s=7, m=14, l=28) -> pd.Series:
    prev_close = close.shift(1)
    bp  = close - pd.concat([low, prev_close], axis=1).min(axis=1)
    tr  = pd.concat([high, prev_close], axis=1).max(axis=1) - \
          pd.concat([low,  prev_close], axis=1).min(axis=1)
    avg_s = bp.rolling(s).sum() / (tr.rolling(s).sum() + 1e-9)
    avg_m = bp.rolling(m).sum() / (tr.rolling(m).sum() + 1e-9)
    avg_l = bp.rolling(l).sum() / (tr.rolling(l).sum() + 1e-9)
    return 100 * (4 * avg_s + 2 * avg_m + avg_l) / 7


def _momentum(series: pd.Series, period: int = 10) -> pd.Series:
    return series - series.shift(period)


def _rolling_zscore(series: pd.Series, window: int = 60) -> pd.Series:
    mu  = series.rolling(window, min_periods=10).mean()
    std = series.rolling(window, min_periods=10).std()
    return (series - mu) / (std + 1e-9)


# ── Per-ETF feature builder ───────────────────────────────────────────────────

def _etf_features(prices: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Builds the 20 technical indicators for one ETF.
    Requires OHLCV — if only Close is available, High=Low=Close (safe fallback).
    """
    close = prices[ticker]
    # yfinance returns OHLCV; if we only have close use it for H/L too
    high  = prices.get(f"{ticker}_High",  close)
    low   = prices.get(f"{ticker}_Low",   close)

    ret   = close.pct_change()
    feat  = pd.DataFrame(index=prices.index)

    # 1. RSI
    feat[f"{ticker}_RSI"]       = _rsi(close, config.RSI_PERIOD)
    # 2-4. MACD line, signal, histogram
    ml, sl, hist                = _macd(close, config.MACD_FAST, config.MACD_SLOW, config.MACD_SIGNAL)
    feat[f"{ticker}_MACD"]      = ml
    feat[f"{ticker}_MACD_sig"]  = sl
    feat[f"{ticker}_MACD_hist"] = hist
    # 5. Stochastic %K
    feat[f"{ticker}_Stoch"]     = _stochastic(high, low, close, config.STOCH_PERIOD)
    # 6. CCI
    feat[f"{ticker}_CCI"]       = _cci(high, low, close, config.CCI_PERIOD)
    # 7. ROC
    feat[f"{ticker}_ROC"]       = _roc(close, config.ROC_PERIOD)
    # 8. CMO
    feat[f"{ticker}_CMO"]       = _cmo(close, config.CMO_PERIOD)
    # 9. Williams %R
    feat[f"{ticker}_WillR"]     = _williams_r(high, low, close)
    # 10. ATR (normalised by close)
    feat[f"{ticker}_ATR"]       = _atr(high, low, close, config.ATR_PERIOD) / (close + 1e-9)
    # 11-12. Bollinger %B and width
    pctb, bw                    = _bbands(close, config.BBANDS_PERIOD)
    feat[f"{ticker}_BB_pctB"]   = pctb
    feat[f"{ticker}_BB_width"]  = bw
    # 13. StochRSI
    feat[f"{ticker}_StochRSI"]  = _stoch_rsi(close, config.RSI_PERIOD, config.STOCH_PERIOD)
    # 14. Ultimate Oscillator
    feat[f"{ticker}_UO"]        = _ultimate_oscillator(high, low, close)
    # 15. Momentum
    feat[f"{ticker}_Mom"]       = _momentum(close, config.ROC_PERIOD)
    # 16-19. Rolling returns: 1d, 5d, 10d, 21d
    for lag in [1, 5, 10, 21]:
        feat[f"{ticker}_Ret{lag}d"] = close.pct_change(lag)
    # 20. Realised vol (21d annualised)
    feat[f"{ticker}_Vol21d"]    = ret.rolling(21).std() * np.sqrt(252)

    return feat


# ── Macro feature builder ─────────────────────────────────────────────────────

def _macro_features(macro: pd.DataFrame) -> pd.DataFrame:
    feat = pd.DataFrame(index=macro.index)
    for col in macro.columns:
        feat[f"macro_{col}"] = macro[col]
        # z-score each macro series with 60d rolling window
        feat[f"macro_{col}_Z"] = _rolling_zscore(macro[col], 60)
    return feat


# ── Master feature matrix ─────────────────────────────────────────────────────

def build_features(etf_prices: pd.DataFrame,
                   macro: pd.DataFrame,
                   start_year: int = None) -> pd.DataFrame:
    """
    Builds the full feature matrix aligned on trading days.
    Returns a DataFrame where each row is one day's state vector.
    """
    frames = []

    # Technical indicators for each target ETF
    for ticker in config.ETFS:
        if ticker in etf_prices.columns:
            frames.append(_etf_features(etf_prices, ticker))

    # Macro features
    if not macro.empty:
        frames.append(_macro_features(macro))

    feat = pd.concat(frames, axis=1, join="outer")

    # Forward-fill macro (released less frequently than daily)
    feat = feat.ffill(limit=5)

    # Align to ETF trading days only (drop weekends / holidays)
    etf_idx = etf_prices[config.ETFS[0]].dropna().index
    feat    = feat.reindex(etf_idx).ffill(limit=5)

    # Filter by start year if provided
    if start_year:
        feat = feat[feat.index.year >= start_year]

    # Drop rows where >50% of features are NaN (warm-up rows)
    thresh = int(len(feat.columns) * 0.5)
    feat   = feat.dropna(thresh=thresh)

    # Fill remaining NaNs with 0 (edge-of-history cases)
    feat   = feat.fillna(0.0)

    return feat


def get_feature_names(etf_prices: pd.DataFrame, macro: pd.DataFrame) -> list:
    """Returns the list of feature column names (for debugging / UI display)."""
    feat = build_features(etf_prices, macro)
    return list(feat.columns)


def make_state_windows(feat: pd.DataFrame,
                       lookback: int = None) -> np.ndarray:
    """
    Converts feature DataFrame into 3D array of shape
    (n_steps, lookback, n_features) for DQN consumption.
    Windows are stride-1 over the date axis.
    """
    lb     = lookback or config.LOOKBACK_WINDOW
    values = feat.values.astype(np.float32)
    n      = len(values)
    if n < lb:
        raise ValueError(f"Not enough data: {n} rows < lookback {lb}")
    windows = np.stack([values[i : i + lb] for i in range(n - lb + 1)])
    return windows   # (n - lb + 1, lb, n_features)
