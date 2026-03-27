---
title: P2 ETF DQN Engine
emoji: 📈
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.28.0
app_file: app.py
pinned: false
---
# P2-ETF-DQN-ENGINE

> **Dueling Deep Q-Network for multi-asset ETF selection**  
> Two independent modules: Option A (FI/Commodities) and Option B (Equity Sectors)  
> Next-trading-day signal across TLT · VCIT · LQD · HYG · VNQ · GLD · SLV · *or* SPY · QQQ · XLK · XLF · XLE · XLV · XLI · XLY · XLP · XLU · GDX · XME · or CASH

[![Hugging Face Space](https://img.shields.io/badge/🤗%20Space-P2--ETF--DQN--ENGINE-blue)](https://huggingface.co/spaces/P2SAMAPA/P2-ETF-DQN-ENGINE)
[![Dataset](https://img.shields.io/badge/🤗%20Dataset-P2--ETF--DQN--ENGINE--DATASET-green)](https://huggingface.co/datasets/P2SAMAPA/P2-ETF-DQN-ENGINE-DATASET)
[![arXiv](https://img.shields.io/badge/arXiv-2411.07585-red)](https://arxiv.org/abs/2411.07585)

---

## Overview

This project implements a **Dueling Deep Q-Network (Dueling DQN)** reinforcement learning agent that selects the highest-return ETF to hold each day, or moves to CASH protected by a trailing stop loss. The methodology directly extends the RL framework proposed by **Yasin & Gill (2024)** at ICAIF 2024, adapted from single-stock buy/sell to multi-asset ETF selection.

The engine now supports **two independent options**:

- **Option A – Fixed Income / Commodities** (7 ETFs: TLT, VCIT, LQD, HYG, VNQ, GLD, SLV)
- **Option B – Equity Sectors** (12 ETFs: SPY, QQQ, XLK, XLF, XLE, XLV, XLI, XLY, XLP, XLU, GDX, XME)

The agent can be trained separately for each option using the `--option` flag. The Streamlit app shows both options in separate tabs, each with its own signal, backtest results, and consensus sweep.

The agent is trained entirely on historical data from a user-selected start year, with an 80/10/10 train/validation/test split. Training runs automatically via GitHub Actions (≈20–30 min on CPU) and results are deployed live to a Hugging Face Space.

---

## Methodology

### Foundation: Yasin & Gill (2024)

This engine is built on the framework introduced in:

> **Yasin, A.S. & Gill, P.S. (2024)**  
> *Reinforcement Learning Framework for Quantitative Trading*  
> arXiv:2411.07585 [q-fin.TR] · ICAIF 2024 FM4TS Workshop  
> https://arxiv.org/abs/2411.07585

The paper benchmarks three RL algorithms — DQN, PPO, and A2C — on single-stock trading using 20 technical indicators as the state space. Its key findings, which directly inform the design choices here:

- **DQN with MLP policy significantly outperforms PPO and A2C** on daily financial time-series. PPO struggles with action timing; A2C converges too slowly on short sequences.
- **Higher learning rates (lr = 0.001)** produce more profitable signals than conservative rates.
- **20 technical indicators** (RSI, MACD, Stochastic, CCI, ROC, CMO, Williams %R, UO, StochRSI, ATR, Bollinger Bands, Momentum, rolling returns, realised vol) constitute a sufficient and non-redundant state representation.
- Normalisation choice has minor impact; the indicator selection is the dominant factor.

### Extensions Beyond the Paper

We extend the paper's methodology in three important ways:

**1. Multi-Asset Action Space**  
Rather than binary buy/sell on a single stock, the agent chooses from discrete actions: CASH, or one of the ETFs in the selected universe (7 for Option A, 12 for Option B). This is a fundamentally harder problem — the agent must learn relative value across assets, not just directionality of one.

**2. Dueling DQN Architecture** (Wang et al., 2016)  
We replace the paper's standard DQN with a Dueling DQN, which decomposes the Q-function into separate Value and Advantage streams:
Q(s, a) = V(s) + A(s, a) − mean_a(A(s, a))

text

This is specifically more effective for multi-action spaces because it separates *"how good is this state overall"* (V) from *"which action is incrementally better"* (A). In practice, when two ETFs have similar Q‑values in a given regime, the Dueling architecture resolves the tie more stably than a standard DQN.

**3. Macro State Augmentation**  
The paper's state uses only price-derived indicators. We add six FRED macro signals to the state vector — VIX, T10Y2Y (yield curve slope), TBILL_3M, DXY (dollar index), Corporate Spread, and HY Spread — all z-scored with a 60-day rolling window. These directly encode the macro regime that drives fixed-income and credit ETF rotation.

### Architecture
State (flattened 20-day window):
├── 20 technical indicators × N_ETFs = N_ETFs × 20 features
├── 6 macro signals + 6 z-scored = 12 features
└── Total per day ~ (N_ETFs × 20 + 12) features
× 20-day lookback window ~ state size

Dueling DQN:
Input (state_size)
→ Dense(256) + LayerNorm + ReLU
→ Dense(256) + LayerNorm + ReLU
↙ ↘
Value stream Advantage stream
Dense(128) → ReLU Dense(128) → ReLU
Dense(1) = V(s) Dense(N_ACTIONS) = A(s,a)
↘ ↙
Q(s,a) = V(s) + A(s,a) − mean(A)

Output: N_ACTIONS Q-values → argmax = selected action

text

### Reward Function
reward = (daily_return − tbill_daily) / realised_vol_21d − transaction_cost

text

- **Excess return over T-bill**: aligns with Sharpe Ratio maximisation
- **Volatility scaling**: penalises drawdown-prone positions, not just negative returns
- **Transaction cost penalty**: discourages excessive switching (configurable, default 10bps)

### Training

| Parameter | Value |
|---|---|
| Data split | 80 / 10 / 10 (train / val / test) |
| Start year | User-selected (2008–2024) |
| Replay buffer | 100,000 transitions |
| Batch size | 64 |
| ε-greedy decay | 1.0 → 0.05 over first 50% of steps |
| Target network update | Every 500 steps (hard copy) |
| Optimiser | Adam, lr = 0.001 |
| Loss | Huber (Smooth L1) |
| Best model selection | Highest validation-set Sharpe Ratio |
| Training time (CPU) | ~20–30 min on GitHub Actions ubuntu-latest |

### Risk Controls (post-signal)

- **Trailing Stop Loss**: if the held ETF's 2-day cumulative return breaches the configured threshold (default −10%), the signal is overridden to CASH
- **Z-Score Re-entry**: re-entry from CASH requires the DQN's best-action Q-value Z-score to clear a configurable threshold (default 1.1σ), ensuring the model has recovered conviction before re-entering risk
- **CASH return**: earns the 3m T-bill rate while in CASH (sourced from FRED DTB3)

---

## ETF Universe

The engine supports two independent universes. Each universe is trained separately and appears as a separate tab in the Streamlit app.

### Option A — Fixed Income / Commodities

| ETF | Category | Role |
|---|---|---|
| TLT | Long-term US Treasuries | Duration / flight-to-quality |
| VCIT | Investment Grade Corp (Vanguard) | Credit / income |
| LQD | Investment Grade Corp (iShares) | Credit / income |
| HYG | High Yield Corp | Risk-on credit |
| VNQ | US Real Estate (REITs) | Inflation hedge / yield |
| GLD | Gold | Safe haven / macro hedge |
| SLV | Silver | Commodity / inflation |

### Option B — Equity Sectors

| ETF | Category | Role |
|---|---|---|
| SPY | S&P 500 | Core equity benchmark |
| QQQ | NASDAQ 100 | Technology‑heavy growth |
| XLK | Technology | Sector exposure |
| XLF | Financials | Financial sector |
| XLE | Energy | Energy sector |
| XLV | Health Care | Health care sector |
| XLI | Industrials | Industrials sector |
| XLY | Consumer Discretionary | Consumer spending |
| XLP | Consumer Staples | Defensive consumer |
| XLU | Utilities | Defensive utilities |
| GDX | Gold Miners | Gold mining equities |
| XME | Metals & Mining | Commodity producers |

### Benchmarks (both options)

| Ticker | Description |
|---|---|
| SPY | S&P 500 (used for Option A backtest comparison) |
| AGG | US Aggregate Bond (used for both options) |
| CASH | 3m T-bill rate (risk‑free asset) |

---

## Project Structure
P2-ETF-DQN-ENGINE/
├── .github/
│ └── workflows/
│ ├── daily_data_update.yml # 2am UTC weekdays: data sync + prediction
│ ├── train_models.yml # On-demand: full DQN training pipeline
│ └── seed.yml # One‑time dataset seed (manual)
├── config.py # Single source of truth — all constants, ETF lists
├── data_download.py # yfinance + FRED data pipeline
├── features.py # 20 technical indicators + macro feature builder
├── env.py # Gym-style trading environment with TSL
├── agent.py # Dueling DQN + replay buffer + target network
├── train.py # Training loop — saves best weights by val Sharpe
├── evaluate.py # Backtest on test set vs SPY/AGG benchmarks
├── predict.py # Next-day signal from saved weights
├── app.py # Streamlit UI (two tabs: Option A & B)
├── smoke_test.py # Fast pre‑training validation
├── requirements.txt
├── data/ # Local parquet cache (gitignored)
└── models/ # Weights + training_summary.json (gitignored)

text

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/P2SAMAPA/P2-ETF-DQN-ENGINE
cd P2-ETF-DQN-ENGINE
pip install -r requirements.txt
2. Environment variables
Create a .env file (never commit this):

text
FRED_API_KEY=your_fred_key
HF_TOKEN=your_hf_token
HF_DATASET_REPO=P2SAMAPA/P2-ETF-DQN-ENGINE-DATASET
GITHUB_TOKEN=your_github_token   # optional, enables one-click retrain from UI
Add the same keys as GitHub repository secrets and Hugging Face Space secrets for the automated pipelines.

3. Seed the dataset
bash
python data_download.py --mode seed
This will download all ETFs (both Option A and Option B) from 2008 and store them locally.

4. Train
Train for Option A:

bash
python train.py --option a --start_year 2015 --episodes 300 --fee_bps 10
Train for Option B:

bash
python train.py --option b --start_year 2015 --episodes 300 --fee_bps 10
5. Evaluate and predict
bash
python evaluate.py --option a --start_year 2015
python predict.py --option a
6. Run the UI locally
bash
streamlit run app.py
Automated Pipelines
Daily Data Update (daily_data_update.yml)
Runs at 2:00 AM UTC Monday–Friday (after US market close):

Incremental price + macro data download

Push updated parquets to HF Dataset

Run predict.py for both Option A and Option B with latest weights

Push latest_prediction_a.json and latest_prediction_b.json to GitHub

Train DQN Agent (train_models.yml)
Triggered manually via GitHub Actions → workflow_dispatch with inputs:

Input	Default	Description
start_year	2015	Training data start year
episodes	300	Number of training episodes
fee_bps	10	Transaction cost in basis points
tsl_pct	10	Trailing stop loss %
z_reentry	1.1	Z-score re-entry threshold
sweep_mode	(blank)	Comma‑separated years for consensus sweep
Pipeline steps: download data → incremental update → train → evaluate → predict → push weights to HF Dataset → push code + results to HF Space → commit json to GitHub.

The --option flag is used to run separate jobs for Option A and Option B (parallel).

Seed Dataset (seed.yml)
Manual workflow to run the full seed once.

UI Features
Two tabs – Option A (FI/Commodities) and Option B (Equity Sectors)

Hero card: next-trading-day signal with training provenance stamp (trained from year, generated date, validation Sharpe)

Trailing stop loss override: orange banner when TSL is active

Q-value probability chart: softmax of Q-values across all actions

Equity curve: test-set performance vs SPY and AGG benchmarks

Allocation breakdown: pie chart of test-set time allocation per ETF

Key metrics: annualised return, Sharpe, max drawdown, Calmar ratio, hit ratio

Methodology section: full explanation of Dueling DQN, state space, reward function, with direct reference to arXiv:2411.07585

One-click retrain: sidebar button triggers GitHub Actions training job with chosen parameters

Consensus Sweep: multi-year backtest aggregation for all start years

References
Yasin, A.S. & Gill, P.S. (2024). Reinforcement Learning Framework for Quantitative Trading. arXiv:2411.07585. Accepted at ICAIF 2024 FM4TS Workshop. https://arxiv.org/abs/2411.07585

Wang, Z., Schaul, T., Hessel, M., van Hasselt, H., Lanctot, M., & de Freitas, N. (2016). Dueling Network Architectures for Deep Reinforcement Learning. ICML 2016.

Mnih, V. et al. (2015). Human-level control through deep reinforcement learning. Nature 518, 529–533.

van Hasselt, H., Guez, A., & Silver, D. (2016). Deep Reinforcement Learning with Double Q-learning. AAAI 2016.

Disclaimer
This project is for research and educational purposes only. It is not financial advice. Past performance of backtested strategies does not guarantee future results. All investment decisions carry risk.
