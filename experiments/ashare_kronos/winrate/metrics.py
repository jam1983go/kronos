"""Trade statistics. Thresholds live in settings.py and are not fit here."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from experiments.ashare_kronos.winrate.settings import (
    BOOTSTRAP_BLOCK,
    BOOTSTRAP_REPS,
    COMMISSION,
    COST_HURDLE,
    FLAT_PRICE_EPS,
    MATERIAL_WIN_GAIN,
    MIN_COVERAGE,
    MIN_LOSSES,
    MIN_PAYOFF,
    MIN_PROFIT_FACTOR,
    MIN_TRADES,
    SEED,
    SLIPPAGE,
    STAMP_ETF,
)


def net_return(entry: float, exit_px: float, commission: float = COMMISSION, slippage: float = SLIPPAGE, stamp: float = STAMP_ETF) -> float:
    entry_fill = entry * (1 + slippage)
    exit_fill = exit_px * (1 - slippage)
    return exit_fill * (1 - commission - stamp) / (entry_fill * (1 + commission)) - 1


def wilson_interval(wins: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    if n <= 0:
        return (float("nan"), float("nan"), float("nan"))
    p = wins / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, max(0.0, center - half), min(1.0, center + half)


def _block_ids(dates: pd.Series, block: int) -> pd.Series:
    ordered = sorted(pd.to_datetime(dates).unique())
    mapping = {date: index // block for index, date in enumerate(ordered)}
    return pd.to_datetime(dates).map(mapping)


def block_bootstrap_rate(flags: np.ndarray, block_ids: np.ndarray, reps: int, seed: int) -> tuple[float, float]:
    if len(flags) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    blocks = sorted(set(block_ids.tolist()))
    grouped = {block: flags[block_ids == block] for block in blocks}
    rates = []
    for _ in range(reps):
        draw = rng.choice(blocks, size=len(blocks), replace=True)
        sample = np.concatenate([grouped[block] for block in draw])
        if len(sample) == 0:
            continue
        rates.append(sample.mean())
    if not rates:
        return (float("nan"), float("nan"))
    return (float(np.quantile(rates, 0.025)), float(np.quantile(rates, 0.975)))


def block_bootstrap_diff(flags_a: np.ndarray, blocks_a: np.ndarray, flags_b: np.ndarray, blocks_b: np.ndarray, reps: int, seed: int) -> tuple[float, float]:
    """Paired by resampling the union of signal-date blocks."""
    if len(flags_a) == 0 or len(flags_b) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    blocks = sorted(set(blocks_a.tolist()) | set(blocks_b.tolist()))
    group_a = {block: flags_a[blocks_a == block] for block in blocks}
    group_b = {block: flags_b[blocks_b == block] for block in blocks}
    diffs = []
    for _ in range(reps):
        draw = rng.choice(blocks, size=len(blocks), replace=True)
        sample_a = [group_a[block] for block in draw if len(group_a[block])]
        sample_b = [group_b[block] for block in draw if len(group_b[block])]
        if not sample_a or not sample_b:
            continue
        diffs.append(float(np.concatenate(sample_a).mean() - np.concatenate(sample_b).mean()))
    if not diffs:
        return (float("nan"), float("nan"))
    return (float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975)))


def summarize_trades(trades: pd.DataFrame, n_slots: int, n_etfs: int, span_days: float) -> dict:
    if trades.empty:
        return {
            "n_trades": 0,
            "n_wins": 0,
            "n_losses": 0,
            "n_flat_gross": 0,
            "win_rate": float("nan"),
            "wilson_low": float("nan"),
            "wilson_high": float("nan"),
            "block_low": float("nan"),
            "block_high": float("nan"),
            "coverage": 0.0,
            "trades_per_etf_year": 0.0,
            "mean_net": float("nan"),
            "payoff": float("nan"),
            "profit_factor": float("nan"),
            "direction_accuracy": float("nan"),
            "n_direction": 0,
            "n_deferred": 0,
        }
    nets = trades["net"].to_numpy(dtype=float)
    wins = nets > 0
    losses = nets <= 0
    n = int(len(nets))
    n_wins = int(wins.sum())
    n_losses = int(losses.sum())
    win_rate, wilson_low, wilson_high = wilson_interval(n_wins, n)
    blocks = _block_ids(trades["signal_date"], BOOTSTRAP_BLOCK).to_numpy()
    block_low, block_high = block_bootstrap_rate(wins.astype(float), blocks, BOOTSTRAP_REPS, SEED)
    gross = trades["gross"].to_numpy(dtype=float)
    nonflat = np.abs(trades["exit_open"].to_numpy(dtype=float) - trades["entry_open"].to_numpy(dtype=float)) >= FLAT_PRICE_EPS
    direction_hits = (gross > 0) & nonflat
    gain = nets[wins]
    loss = nets[losses]
    payoff = float(gain.mean() / abs(loss.mean())) if len(gain) and len(loss) and loss.mean() != 0 else float("nan")
    profit_factor = float(gain.sum() / abs(loss.sum())) if len(gain) and len(loss) and loss.sum() != 0 else float("nan")
    years = max(span_days, 1) / 365.25
    return {
        "n_trades": n,
        "n_wins": n_wins,
        "n_losses": n_losses,
        "n_flat_gross": int((~nonflat).sum()),
        "win_rate": win_rate,
        "wilson_low": wilson_low,
        "wilson_high": wilson_high,
        "block_low": block_low,
        "block_high": block_high,
        "coverage": float(n / n_slots) if n_slots else float("nan"),
        "trades_per_etf_year": float(n / max(n_etfs, 1) / years),
        "mean_net": float(nets.mean()),
        "payoff": payoff,
        "profit_factor": profit_factor,
        "direction_accuracy": float(direction_hits.sum() / nonflat.sum()) if nonflat.any() else float("nan"),
        "n_direction": int(nonflat.sum()),
        "n_deferred": int(trades["deferred"].sum()) if "deferred" in trades else 0,
    }


def eligible(summary: dict) -> bool:
    return (
        summary["n_trades"] >= MIN_TRADES
        and summary["n_losses"] >= MIN_LOSSES
        and summary["coverage"] >= MIN_COVERAGE
        and math.isfinite(summary["profit_factor"])
        and summary["profit_factor"] >= MIN_PROFIT_FACTOR
        and math.isfinite(summary["payoff"])
        and summary["payoff"] >= MIN_PAYOFF
    )


def choose_modification(selection_scores: list[dict]) -> dict:
    raw = next(row for row in selection_scores if row["id"] == "kronos_lb120_s5_mean_gt0")
    qualified = [row for row in selection_scores if row["kind"] == "modified" and row["eligible"]]
    qualified.sort(key=lambda row: (-row["win_rate"], -row["profit_factor"], -row["coverage"], row["id"]))
    if not qualified:
        return {
            "selected_id": None,
            "selection_reason": "none_eligible",
            "material_gain": False,
            "selection_gain": float("nan"),
            "raw_win_rate": raw["win_rate"],
        }
    best = qualified[0]
    gain = best["win_rate"] - raw["win_rate"]
    return {
        "selected_id": best["id"],
        "selection_reason": "highest_eligible_win_rate",
        "material_gain": bool(gain >= MATERIAL_WIN_GAIN),
        "selection_gain": float(gain),
        "raw_win_rate": raw["win_rate"],
    }


def decide_signal(rule: str, mean_ret: float | None, median_ret: float | None, positive_paths: int | None, sample_count: int | None, mom_ret: float | None, majority: int | None) -> bool:
    if rule == "none":
        return False
    if rule == "mom>0":
        return mom_ret is not None and mom_ret > 0
    if rule == "mom>cost":
        return mom_ret is not None and mom_ret > COST_HURDLE
    if rule == "majority":
        return majority == 1
    if rule == "mean>0":
        return mean_ret is not None and mean_ret > 0
    if rule == "mean>cost":
        return mean_ret is not None and mean_ret > COST_HURDLE
    if rule == "median>0":
        return median_ret is not None and median_ret > 0
    if rule == "median>cost":
        return median_ret is not None and median_ret > COST_HURDLE
    if rule == "agree4_mean>cost":
        return positive_paths is not None and sample_count == 5 and positive_paths >= 4 and mean_ret is not None and mean_ret > COST_HURDLE
    if rule == "mean>cost_median>0":
        return mean_ret is not None and median_ret is not None and mean_ret > COST_HURDLE and median_ret > 0
    if rule == "agree15_median>cost":
        return positive_paths is not None and sample_count == 20 and positive_paths >= 15 and median_ret is not None and median_ret > COST_HURDLE
    raise ValueError(rule)
