# config.py — single source of truth for P2-ETF-DQN-ENGINE
import os
from dotenv import load_dotenv
load_dotenv()

# ── Repos ─────────────────────────────────────────────────────────────────────
HF_DATASET_REPO = os.getenv("HF_DATASET_REPO", "P2SAMAPA/P2-ETF-DQN-ENGINE-DATASET")
HF_SPACE_REPO   = "P2SAMAPA/P2-ETF-DQN-ENGINE"
GITHUB_REPO     = "P2SAMAPA/P2-ETF-DQN-ENGINE"

# ── API Keys ──────────────────────────────────────────────────────────────────
HF_TOKEN        = os.getenv("HF_TOKEN", "")
FRED_API_KEY    = os.getenv("FRED_API_KEY", "")
GITHUB_TOKEN    = os.getenv("GITHUB_TOKEN", "")

# ── ETF Universe ──────────────────────────────────────────────────────────────
ETFS            = ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV"]
BENCHMARKS      = ["SPY", "AGG"]
ALL_TICKERS     = ETFS + BENCHMARKS

# ── Action Space: 0=CASH, 1..7 = ETFs ────────────────────────────────────────
ACTIONS         = ["CASH"] + ETFS          # len = 8
N_ACTIONS       = len(ACTIONS)             # 8

# ── FRED Macro Series ─────────────────────────────────────────────────────────
MACRO_SERIES = {
    "VIX"        : "VIXCLS",
    "T10Y2Y"     : "T10Y2Y",
    "TBILL_3M"   : "DTB3",
    "DXY"        : "DTWEXBGS",
    "CORP_SPREAD": "BAMLC0A0CM",
    "HY_SPREAD"  : "BAMLH0A0HYM2",
}

# ── Data ──────────────────────────────────────────────────────────────────────
SEED_START          = "2008-01-01"
DEFAULT_START_YEAR  = 2008

# ── Feature Engineering ───────────────────────────────────────────────────────
LOOKBACK_WINDOW = 20          # days of history fed into DQN state
RSI_PERIOD      = 14
MACD_FAST       = 12
MACD_SLOW       = 26
MACD_SIGNAL     = 9
STOCH_PERIOD    = 14
CCI_PERIOD      = 20
ROC_PERIOD      = 10
CMO_PERIOD      = 14
ATR_PERIOD      = 14
BBANDS_PERIOD   = 20

# ── Dueling DQN Hyperparameters ───────────────────────────────────────────────
HIDDEN_UNITS        = 256
LEARNING_RATE       = 0.0005        # FIX: reduced from 0.001
GAMMA               = 0.99          # discount factor
EPSILON_START       = 1.0
EPSILON_END         = 0.10          # FIX: was 0.05
EPSILON_DECAY_FRAC  = 0.80          # FIX: was 0.50 decay over 80% not 50%
REPLAY_BUFFER_SIZE  = 100_000
BATCH_SIZE          = 64
TARGET_UPDATE_FREQ  = 1             # FIX: soft Polyak update every step
TAU                 = 0.005         # FIX: Polyak factor
MIN_REPLAY_SIZE     = 1_000         # steps before training starts
DEFAULT_EPISODES    = 500           # FIX: was 300

# Reward shaping
REWARD_VOL_MIN      = 0.05          # FIX: min vol floor prevents CASH 30x amplification
REWARD_VOL_MAX      = 0.40          # FIX: max vol ceiling
REWARD_ETF_BONUS    = 1.10          # FIX: bonus when ETF beats T-bill

# ── Risk Controls (defaults — overridden by UI sliders) ───────────────────────
DEFAULT_TSL_PCT     = 10.0          # trailing stop loss %
DEFAULT_Z_REENTRY   = 1.1           # z-score to re-enter from CASH
DEFAULT_FEE_BPS     = 10            # transaction cost in basis points

# ── Train/Val/Test Split ──────────────────────────────────────────────────────
TRAIN_SPLIT     = 0.80
VAL_SPLIT       = 0.10
# TEST = remaining 0.10

# ── Paths ─────────────────────────────────────────────────────────────────────
MODELS_DIR      = "models"
DATA_DIR        = "data"
