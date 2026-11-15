"""Fixed non-overlapping execution slots. No model scores."""

from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.ashare_kronos.winrate.settings import (
    FLAT_PRICE_EPS,
    GRID_STRIDE,
    HOLD_DAYS,
    LIMIT,
    LIMIT_BUFFER,
    MAX_HOLD_CALENDAR_DAYS,
    MAX_LIMIT_DOWN_DEFER,
    WINDOWS,
)


def _limit_up(open_px: float, prev_close: float) -> bool:
    return prev_close > 0 and open_px / prev_close - 1 >= LIMIT - LIMIT_BUFFER


def _limit_down(open_px: float, prev_close: float) -> bool:
    return prev_close > 0 and open_px / prev_close - 1 <= -(LIMIT - LIMIT_BUFFER)


def build_slots(panels: dict[str, dict[str, pd.DataFrame]]) -> pd.DataFrame:
    calendars = {code: set(pack["qfq"]["date"]) for code, pack in panels.items()}
    common = sorted(set.intersection(*calendars.values()))
    common_index = {date: i for i, date in enumerate(common)}
    rows = []
    for code, pack in panels.items():
        qfq = pack["qfq"].reset_index(drop=True)
        raw = pack["raw"].reset_index(drop=True)
        if not qfq["date"].equals(raw["date"]):
            raise ValueError(f"{code} qfq/raw calendar mismatch")
        date_to_i = {date: i for i, date in enumerate(qfq["date"])}
        for date in common:
            if common_index[date] % GRID_STRIDE != 0:
                continue
            signal_i = date_to_i[date]
            if signal_i < 240 or signal_i + 1 + HOLD_DAYS >= len(qfq):
                continue
            entry_i = signal_i + 1
            exit_i = entry_i + HOLD_DAYS
            deferred = 0
            blocked = False
            while deferred <= MAX_LIMIT_DOWN_DEFER:
                if exit_i + deferred >= len(raw):
                    blocked = True
                    break
                bar = raw.iloc[exit_i + deferred]
                prev = raw.iloc[exit_i + deferred - 1]
                if bar["volume"] <= 0 or _limit_down(float(bar["open"]), float(prev["close"])):
                    deferred += 1
                    continue
                break
            else:
                blocked = True
            entry = raw.iloc[entry_i]
            prev_close = float(raw.iloc[signal_i]["close"])
            entry_open = float(entry["open"])
            limit_up = _limit_up(entry_open, prev_close) or entry["volume"] <= 0 or entry_open <= 0
            if blocked:
                exit_open = np.nan
                exit_date = pd.NaT
                executable = False
                deferred_out = 0
            else:
                exit_open = float(raw.iloc[exit_i + deferred]["open"])
                exit_date = raw.iloc[exit_i + deferred]["date"]
                deferred_out = deferred
                executable = not limit_up
            if executable and (pd.Timestamp(exit_date) - pd.Timestamp(date)).days > MAX_HOLD_CALENDAR_DAYS:
                executable = False
            target_i = signal_i + HOLD_DAYS
            rows.append(
                {
                    "code": code,
                    "signal_date": pd.Timestamp(date),
                    "entry_date": qfq.iloc[entry_i]["date"],
                    "exit_date": pd.Timestamp(exit_date) if executable else pd.NaT,
                    "signal_close_qfq": float(qfq.iloc[signal_i]["close"]),
                    "signal_close_raw": float(raw.iloc[signal_i]["close"]),
                    "target_close_raw": float(raw.iloc[target_i]["close"]),
                    "entry_open": entry_open,
                    "exit_open": exit_open if executable else np.nan,
                    "mom_ret": float(qfq.iloc[signal_i]["close"] / qfq.iloc[signal_i - HOLD_DAYS]["close"] - 1),
                    "majority": _majority(qfq.iloc[signal_i - 119 : signal_i + 1]["close"].to_numpy()),
                    "executable": bool(executable),
                    "limit_up": bool(limit_up),
                    "deferred": int(deferred_out),
                    "flat_target": abs(float(raw.iloc[target_i]["close"]) - float(raw.iloc[signal_i]["close"])) < FLAT_PRICE_EPS,
                }
            )
    slots = pd.DataFrame(rows)
    slots["window"] = slots.apply(_window_of, axis=1)
    return slots


def _majority(closes: np.ndarray) -> int:
    diffs = np.diff(closes)
    ups = int((diffs > FLAT_PRICE_EPS).sum())
    downs = int((diffs < -FLAT_PRICE_EPS).sum())
    if ups > downs:
        return 1
    if downs > ups:
        return -1
    return 0


def _window_of(row: pd.Series) -> str:
    signal = pd.Timestamp(row["signal_date"])
    exit_date = pd.Timestamp(row["exit_date"]) if pd.notna(row["exit_date"]) else pd.NaT
    for name, (start, end, exit_before) in WINDOWS.items():
        if pd.Timestamp(start) <= signal <= pd.Timestamp(end):
            if pd.isna(exit_date) or exit_date >= pd.Timestamp(exit_before):
                return "purged"
            return name
    return "outside"
