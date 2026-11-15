"""HIT research-approximation signals with a date-aware price limit."""

from __future__ import annotations

import json
import math

import pandas as pd

from experiments.ashare_kronos.routes3.engine import (
    add_indicators,
    run_exit,
    simulate_trade,
    stop_distance,
)
from experiments.ashare_kronos.routes3.settings import FLAT_EPS, RISK_MIXED, RISK_UP


def limit_for(code: str, name: str, when: pd.Timestamp) -> float:
    upper = name.upper()
    if "ST" in upper:
        return 0.05
    if code.startswith("300") and pd.Timestamp(when) >= pd.Timestamp("2020-08-24"):
        return 0.20
    if code.startswith("300"):
        return 0.10
    return 0.10


def build_signals(code: str, name: str, qfq: pd.DataFrame, raw: pd.DataFrame, regimes: pd.DataFrame) -> pd.DataFrame:
    ind = add_indicators(qfq)
    raw = raw.sort_values("date").reset_index(drop=True)
    if not ind["date"].equals(raw["date"]):
        raise ValueError(f"{code} calendars differ")
    merged = ind.merge(regimes, on="date", how="left")
    usable = merged.index[
        (merged["cross"] == True)  # noqa: E712
        & (merged["vol_ok"] == True)  # noqa: E712
        & merged["regime"].isin(["ONLY_UP", "MIXED"])
    ]
    rows = []
    for i in usable:
        if int(i) < 60:
            continue
        row = merged.loc[i]
        limit = limit_for(code, name, row["date"])
        dist = stop_distance(float(row["close"]), float(row["low_10"]), float(row["atr"]))
        plan = simulate_trade(raw, int(i), dist, limit)
        if plan is None or plan["status"] != "sized":
            continue
        risk_budget = RISK_UP if row["regime"] == "ONLY_UP" else RISK_MIXED
        shares = int(math.floor(risk_budget / plan["risk_per_share"] / 100.0) * 100)
        if shares < 100:
            continue
        trade = run_exit(raw, plan["entry_i"], shares, plan["stop_px"], plan["target_px"], limit, 1.0)
        trade2 = run_exit(raw, plan["entry_i"], shares, plan["stop_px"], plan["target_px"], limit, 2.0)
        if trade is None or trade2 is None:
            continue
        target_i = int(i) + 5
        direction = float("nan")
        if target_i < len(raw):
            move = float(raw.iloc[target_i]["close"]) - float(raw.iloc[int(i)]["close"])
            direction = 0 if abs(move) < FLAT_EPS else (1 if move > 0 else -1)
        rows.append(
            {
                "code": code,
                "name": name,
                "status": "candidate",
                "signal_date": pd.Timestamp(row["date"]),
                "entry_date": raw.iloc[plan["entry_i"]]["date"],
                "exit_date": pd.Timestamp(trade["exit_date"]),
                "exit_session": trade["exit_session"],
                "regime": row["regime"],
                "gap": plan["gap"],
                "shares": shares,
                "entry": plan["entry"],
                "stop_dist": dist,
                "pnl": trade["pnl"],
                "ret": trade["ret"],
                "pnl_stress": trade2["pnl"],
                "ret_stress": trade2["ret"],
                "win": bool(trade["win"]),
                "win_stress": bool(trade2["win"]),
                "buy": trade["buy"],
                "amount": float(row["amount"]),
                "dif": float(row["dif"]),
                "hist": float(row["hist"]),
                "vol_ratio": float(row["volume"] / row["vol_ma5_yday"]),
                "atr_pct": float(row["atr"] / row["close"]),
                "ret_5": float(row["ret_5"]),
                "ret_20": float(row["ret_20"]),
                "regime_ord": 1 if row["regime"] == "ONLY_UP" else 0,
                "direction_5": direction,
                "fills": json.dumps(trade["fills"], ensure_ascii=False),
                "fills_stress": json.dumps(trade2["fills"], ensure_ascii=False),
            }
        )
    return pd.DataFrame(rows)
