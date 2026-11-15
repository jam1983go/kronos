"""Frozen settings for the three-route test. See PROTOCOL.md."""

from __future__ import annotations

SEED = 20261115
AS_OF = "2026-11-15"
HISTORY_BEG = "20230101"
HISTORY_END = "20261114"

POOL = (
    {"code": "600036", "market": "SH"},
    {"code": "601899", "market": "SH"},
    {"code": "000333", "market": "SZ"},
    {"code": "002415", "market": "SZ"},
    {"code": "300750", "market": "SZ"},
    {"code": "600900", "market": "SH"},
    {"code": "600435", "market": "SH"},
    {"code": "600967", "market": "SH"},
    {"code": "600863", "market": "SH"},
    {"code": "300059", "market": "SZ"},
    {"code": "600519", "market": "SH"},
    {"code": "000858", "market": "SZ"},
    {"code": "000651", "market": "SZ"},
    {"code": "601318", "market": "SH"},
    {"code": "601888", "market": "SH"},
    {"code": "600887", "market": "SH"},
    {"code": "002594", "market": "SZ"},
    {"code": "601166", "market": "SH"},
    {"code": "600030", "market": "SH"},
    {"code": "000001", "market": "SZ"},
)

FIT = ("2024-07-01", "2024-10-31")
CHECK = ("2024-11-15", "2024-12-20")
REVAL = ("2025-01-15", "2026-10-24")
DATA_LAST = "2026-11-13"

ACCOUNT = 300_000.0
MAX_POSITIONS = 3
RISK_UP = 10_000.0
RISK_MIXED = 7_000.0
GAP_SKIP = 0.015
STOP_MIN = 0.005
STOP_MAX = 0.12
HOLD_HALF = 5
HOLD_FINAL = 10
DEFER_LIMIT = 3
COMMISSION = 0.00025
MIN_COMMISSION = 5.0
STAMP = 0.0005
SLIPPAGE = 0.0005
COST_HURDLE = 0.002
FLAT_EPS = 0.005

LOOKBACK = 120
PRED_LEN = 3
SAMPLES = 5
FT_STEPS = 20
FT_BATCH = 4
FT_SEQ = 64
FT_LR = 1e-4
BOOT_REPS = 1000
BOOT_BLOCK = 10

CANDIDATES = (
    "hit_raw",
    "hit_kronos_pos",
    "hit_kronos_cost",
    "logit_c1",
    "logit_c1_kronos",
    "logit_c01_kronos",
    "tree_d2_kronos",
    "hit_finetune_pos",
)

assert len(CANDIDATES) <= 12
assert len(POOL) == 20
