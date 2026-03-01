# data_download.py
# Downloads ETF prices + FRED macro signals.
# Usage:
#   python data_download.py --mode seed          # 2008-01-01 → today
#   python data_download.py --mode incremental   # last stored date → today

import argparse
import os
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
from fredapi import Fred
from tqdm import tqdm

import config

warnings.filterwarnings("ignore")
os.makedirs(config.DATA_DIR, exist_ok=True)


# ── Price fetcher ─────────────────────────────────────────────────────────────

def fetch_prices(tickers: list, start: str, end: str) -> pd.DataFrame:
    """Fetch daily adjusted close prices via yfinance."""
    print(f"  Fetching prices: {tickers}")
    frames = []
    for ticker in tqdm(tickers, desc="Prices"):
        try:
            df = yf.download(ticker, start=start, end=end,
                             auto_adjust=True, progress=False)
            if df.empty:
                print(f"  WARNING: No data for {ticker}")
                continue
            close = df[["Close"]].rename(columns={"Close": ticker})
            frames.append(close)
        except Exception as e:
            print(f"  WARNING: Failed {ticker}: {e}")

    if not frames:
        raise RuntimeError("No price data fetched.")

    prices = pd.concat(frames, axis=1)
    if isinstance(prices.columns, pd.MultiIndex):
        prices.columns = [col[0] if col[1] == col[0] else col[0]
                          for col in prices.columns]
    prices.columns = [str(c).strip() for c in prices.columns]
    prices.index   = pd.to_datetime(prices.index)
    prices.index.name = "Date"
    if prices.index.tzinfo:
        prices.index = prices.index.tz_localize(None)
    return prices.sort_index()


# ── FRED macro fetcher ────────────────────────────────────────────────────────

def fetch_macro(start: str, end: str) -> pd.DataFrame:
    """Fetch all macro signals from FRED."""
    fred   = Fred(api_key=config.FRED_API_KEY)
    frames = {}
    for name, series_id in tqdm(config.MACRO_SERIES.items(), desc="Macro"):
        try:
            s = fred.get_series(series_id,
                                observation_start=start,
                                observation_end=end)
            s       = pd.Series(s, name=name)
            s.index = pd.to_datetime(s.index)
            if s.index.tzinfo:
                s.index = s.index.tz_localize(None)
            frames[name] = s
        except Exception as e:
            print(f"  WARNING: FRED failed for {name} ({series_id}): {e}")

    macro = pd.DataFrame(frames)
    macro.index.name = "Date"
    macro = macro.ffill().dropna(how="all")
    return macro.sort_index()


# ── Dataset builder ───────────────────────────────────────────────────────────

def build_dataset(start: str, end: str) -> dict:
    print(f"\n{'='*60}")
    print(f"  Downloading: {start} → {end}")
    print(f"{'='*60}")

    etf_prices  = fetch_prices(config.ALL_TICKERS, start, end)
    macro       = fetch_macro(start, end)

    return dict(etf_prices=etf_prices, macro=macro)


def save_local(data: dict):
    os.makedirs(config.DATA_DIR, exist_ok=True)
    for name, df in data.items():
        path = os.path.join(config.DATA_DIR, f"{name}.parquet")
        df.to_parquet(path)
        print(f"  Saved {path}  ({len(df)} rows × {len(df.columns)} cols)")


def load_local() -> dict:
    data = {}
    for name in ["etf_prices", "macro"]:
        path = os.path.join(config.DATA_DIR, f"{name}.parquet")
        if os.path.exists(path):
            df = pd.read_parquet(path)
            df.index = pd.to_datetime(df.index)
            data[name] = df
    return data


def incremental_update() -> dict:
    existing = load_local()
    if not existing or "etf_prices" not in existing:
        print("No existing data — running full seed.")
        return seed()

    last_date = existing["etf_prices"].index.max()
    start     = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
    end       = datetime.today().strftime("%Y-%m-%d")

    if start >= end:
        print(f"Already up to date through {last_date.date()}.")
        return existing

    print(f"Incremental update: {start} → {end}")
    new_data = build_dataset(start, end)

    merged = {}
    for key in ["etf_prices", "macro"]:
        if key in existing and key in new_data and not new_data[key].empty:
            combined = pd.concat([existing[key], new_data[key]])
            combined = combined[~combined.index.duplicated(keep="last")]
            combined.sort_index(inplace=True)
            merged[key] = combined
        else:
            merged[key] = existing.get(key, new_data.get(key, pd.DataFrame()))

    save_local(merged)
    return merged


def seed() -> dict:
    end  = datetime.today().strftime("%Y-%m-%d")
    data = build_dataset(config.SEED_START, end)
    save_local(data)
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["seed", "incremental"],
                        default="incremental")
    args = parser.parse_args()

    if args.mode == "seed":
        seed()
    else:
        incremental_update()

    print("\nDone.")
