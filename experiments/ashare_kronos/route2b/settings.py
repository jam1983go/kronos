"""Frozen pool, folds and model list. See PROTOCOL.md."""

from __future__ import annotations

SEED = 20261115
AS_OF = "2026-11-15"
FETCH_BEG = "20160101"
FETCH_END = "20261114"
LIQ_START = "2019-01-01"
LIQ_END = "2021-12-31"
MIN_MEDIAN_AMOUNT = 50_000_000.0
FIRST_SIGNAL = "2018-01-01"
EXTENDED_SIGNAL = "2016-04-01"
MIN_TRAIN = 80
MIN_CLASS = 25
MIN_FOLD_TRAIN = 40
MIN_FOLD_CLASS = 10
REVAL_TRAIN_BEFORE = "2025-01-15"
BOOT_REPS = 1000
BOOT_BLOCK = 10
LOOKBACK = 120
PRED_LEN = 3
SAMPLES = 5
COST_HURDLE = 0.0  # kronos_pos uses a sign, not a fitted hurdle

FEATURES = ["dif", "hist", "vol_ratio", "atr_pct", "ret_5", "ret_20", "regime_ord", "stop_dist"]
CANDIDATES = (
    "hit_raw",
    "kronos_pos",
    "logit_c1",
    "logit_c1_kronos",
    "logit_c01",
    "logit_c01_kronos",
    "tree_d2",
    "tree_d2_kronos",
)

# signal_start, signal_end, exit_before, train_exit_before, exploratory
FOLDS = (
    ("wf_2022", "2022-01-04", "2022-12-30", "2023-01-03", "2022-01-04", False),
    ("wf_2023", "2023-01-03", "2023-12-29", "2024-01-02", "2023-01-03", False),
    ("wf_2024", "2024-01-02", "2024-12-31", "2025-01-15", "2024-01-02", False),
    ("reval_2025_2026", "2025-01-15", "2026-10-24", "2026-11-14", "2025-01-15", True),
)

def _pairs(codes: str, market: str) -> tuple[dict, ...]:
    return tuple({"code": code, "market": market} for code in codes.split())


POOL_A = (
    _pairs("600036 601166 600000 600016 601328 601288 601398 601939 601988 601818 600015 601169 601009", "SH")
    + _pairs("000001 002142", "SZ")
    + _pairs("601318 601601 601628 601336 600030 600837 601211 601688 600999", "SH")
    + _pairs("000776", "SZ")
    + _pairs("600519 600809 603288 600887", "SH")
    + _pairs("000858 000568 002304 000596", "SZ")
    + _pairs("600690", "SH")
    + _pairs("000333 000651 002594 000625", "SZ")
    + _pairs("601633 600104 601857 600028 601088 601225 600188 601898 601899 600111 601600 600362 603993 600585 600019", "SH")
    + _pairs("600900 600886 600795 601985 601006 601111 600029 601919 601872 601888 600009", "SH")
    + _pairs("002415 000725 002475 000063 002241 000977 002230", "SZ")
    + _pairs("600183", "SH")
    + _pairs("300059 300750 300014 300124 300274", "SZ")
    + _pairs("601012 600438 600435 600967 600863 600893 601668 601390 601186 601800 600048 600406 600031 601766", "SH")
    + _pairs("000768 000002 001979 000338", "SZ")
)

POOL_B = (
    _pairs("600050 600741 600660 600346 601233 600309 600547 601168 600176 601636 600011 600642 601991 600004 601021 600115 601866 600018 601018 600498 600570 600588 600150 601989 600482", "SH")
    + _pairs("002027 002714 000895 002352 002493 000630 000066 002236 002410 000539", "SZ")
)

assert len(CANDIDATES) <= 8
assert len(POOL_A) >= 80
