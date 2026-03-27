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

# ── ETF Universes ──────────────────────────────────────────────────────────────
# Option A: Fixed Income / Commodities (existing)
OPTION_A_ETFS = ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV"]

# Option B: Equity Sectors (new)
OPTION_B_ETFS = [
    "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV",
    "XLI", "XLY", "XLP", "XLU", "GDX", "XME"
]

# Combined list for data fetching (all tickers)
ALL_TICKERS = OPTION_A_ETFS + OPTION_B_ETFS + ["SPY", "AGG"]  # benchmarks already in ALL_TICKERS? SPY is also in Option B, but that's fine.

# Action spaces:
# Option A: 0=CASH, 1..7 = OPTION_A_ETFS
ACTIONS_A = ["CASH"] + OPTION_A_ETFS          # len = 8
N_ACTIONS_A = len(ACTIONS_A)

# Option B: 0=CASH, 1..12 = OPTION_B_ETFS
ACTIONS_B = ["CASH"] + OPTION_B_ETFS          # len = 13
N_ACTIONS_B = len(ACTIONS_B)

# Keep backward compatibility (for existing code that expects ETFS, ACTIONS, N_ACTIONS)
ETFS            = OPTION_A_ETFS   # for backward compatibility
ACTIONS         = ACTIONS_A       # for backward compatibility
N_ACTIONS       = N_ACTIONS_A     # for backward compatibility
BENCHMARKS      = ["SPY", "AGG"]

# ── FRED Macro Series (unchanged) ─────────────────────────────────────────────
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

# ── Feature Engineering (unchanged) ───────────────────────────────────────────
LOOKBACK_WINDOW = 20
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

# ── Dueling DQN Hyperparameters (unchanged) ───────────────────────────────────
HIDDEN_UNITS        = 256
LEARNING_RATE       = 0.0005
GAMMA               = 0.99
EPSILON_START       = 1.0
EPSILON_END         = 0.10
EPSILON_DECAY_FRAC  = 0.80
REPLAY_BUFFER_SIZE  = 100_000
BATCH_SIZE          = 64
TARGET_UPDATE_FREQ  = 1
TAU                 = 0.005
MIN_REPLAY_SIZE     = 1_000
DEFAULT_EPISODES    = 500

# Reward shaping (unchanged)
REWARD_VOL_MIN      = 0.05
REWARD_VOL_MAX      = 0.40
REWARD_ETF_BONUS    = 1.10

# ── Risk Controls (defaults) ──────────────────────────────────────────────────
DEFAULT_TSL_PCT     = 10.0
DEFAULT_Z_REENTRY   = 1.1
DEFAULT_FEE_BPS     = 10

# ── Train/Val/Test Split (unchanged) ──────────────────────────────────────────
TRAIN_SPLIT     = 0.80
VAL_SPLIT       = 0.10

# ── Paths (unchanged) ─────────────────────────────────────────────────────────
MODELS_DIR      = "models"
DATA_DIR        = "data"
