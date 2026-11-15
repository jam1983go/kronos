"""Frozen win-rate experiment settings. See PROTOCOL.md. Do not retune after scores."""

from __future__ import annotations

MODEL_ID = "NeoQuasar/Kronos-small"
MODEL_REVISION = "901c26c1332695a2a8f243eb2f37243a37bea320"
TOKENIZER_ID = "NeoQuasar/Kronos-Tokenizer-base"
TOKENIZER_REVISION = "0e0117387f39004a9016484a186a908917e22426"
MAX_CONTEXT = 512
SEED = 20261115
TEMPERATURE = 1.0
TOP_K = 0
TOP_P = 0.9
CLIP = 5.0
PRED_LEN = 3
HOLD_DAYS = 3
GRID_STRIDE = 4
BOOTSTRAP_BLOCK = 5
BOOTSTRAP_REPS = 2000
FLAT_PRICE_EPS = 0.005
LIMIT = 0.10
LIMIT_BUFFER = 0.001
MAX_LIMIT_DOWN_DEFER = 3
MAX_HOLD_CALENDAR_DAYS = 12

COMMISSION = 0.00025
SLIPPAGE = 0.0005
STAMP_ETF = 0.0
COST_HURDLE = 2 * COMMISSION + 2 * SLIPPAGE  # 0.0015, fee schedule, not fitted

LIQUIDITY_START = "2023-01-03"
LIQUIDITY_END = "2024-06-30"
MIN_MEDIAN_AMOUNT = 100_000_000.0

POOL = (
    {"code": "510300", "market": "SH"},
    {"code": "159915", "market": "SZ"},
    {"code": "510500", "market": "SH"},
    {"code": "510050", "market": "SH"},
)

# signal_start, signal_end, exit_before
WINDOWS = {
    "selection": ("2024-07-01", "2024-10-31", "2024-11-15"),
    "confirmation": ("2024-11-15", "2024-12-20", "2025-01-15"),
    "revalidation": ("2025-01-15", "2026-11-06", "2026-11-14"),
}

AS_OF_DATE = "2026-11-15"
HISTORY_START = "20230101"
HISTORY_END = "20261114"

MIN_TRADES = 30
MIN_LOSSES = 5
MIN_COVERAGE = 0.15
MIN_PROFIT_FACTOR = 1.0
MIN_PAYOFF = 0.75
MATERIAL_WIN_GAIN = 0.05
REVALIDATION_MIN_TRADES_PER_ETF = 20

CONFIGS = (
    {"id": "no_trade", "kind": "baseline", "lookback": None, "samples": None, "rule": "none"},
    {"id": "momentum", "kind": "baseline", "lookback": None, "samples": None, "rule": "mom>0"},
    {"id": "momentum_gt_cost", "kind": "baseline", "lookback": None, "samples": None, "rule": "mom>cost"},
    {"id": "majority_120", "kind": "baseline", "lookback": 120, "samples": None, "rule": "majority"},
    {"id": "kronos_lb120_s5_mean_gt0", "kind": "original", "lookback": 120, "samples": 5, "rule": "mean>0"},
    {"id": "kronos_lb120_s5_mean_gt_cost", "kind": "modified", "lookback": 120, "samples": 5, "rule": "mean>cost"},
    {"id": "kronos_lb120_s5_median_gt0", "kind": "modified", "lookback": 120, "samples": 5, "rule": "median>0"},
    {"id": "kronos_lb120_s5_median_gt_cost", "kind": "modified", "lookback": 120, "samples": 5, "rule": "median>cost"},
    {"id": "kronos_lb120_s5_agree4_mean_gt_cost", "kind": "modified", "lookback": 120, "samples": 5, "rule": "agree4_mean>cost"},
    {"id": "kronos_lb120_s5_mean_median_agree_gt_cost", "kind": "modified", "lookback": 120, "samples": 5, "rule": "mean>cost_median>0"},
    {"id": "kronos_lb120_s20_agree15_median_gt_cost", "kind": "modified", "lookback": 120, "samples": 20, "rule": "agree15_median>cost"},
    {"id": "kronos_lb60_s5_mean_gt_cost", "kind": "modified", "lookback": 60, "samples": 5, "rule": "mean>cost"},
    {"id": "kronos_lb60_s5_agree4_mean_gt_cost", "kind": "modified", "lookback": 60, "samples": 5, "rule": "agree4_mean>cost"},
    {"id": "kronos_lb240_s5_mean_gt_cost", "kind": "modified", "lookback": 240, "samples": 5, "rule": "mean>cost"},
    {"id": "kronos_lb240_s5_agree4_mean_gt_cost", "kind": "modified", "lookback": 240, "samples": 5, "rule": "agree4_mean>cost"},
)

INFERENCE_SPECS = (
    {"lookback": 60, "samples": 5},
    {"lookback": 120, "samples": 5},
    {"lookback": 120, "samples": 20},
    {"lookback": 240, "samples": 5},
)

assert len(CONFIGS) <= 18
assert abs(COST_HURDLE - 0.0015) < 1e-12
assert CONFIGS[4]["kind"] == "original"
