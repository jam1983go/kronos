"""Research-approximation HIT signals and fills. Indicators use adjusted prices."""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

from experiments.ashare_kronos.routes3.settings import (
    ACCOUNT,
    COMMISSION,
    DEFER_LIMIT,
    FLAT_EPS,
    GAP_SKIP,
    HOLD_FINAL,
    HOLD_HALF,
    MAX_POSITIONS,
    MIN_COMMISSION,
    RISK_MIXED,
    RISK_UP,
    SLIPPAGE,
    STAMP,
    STOP_MAX,
    STOP_MIN,
)


def limit_pct(code: str, name: str) -> float:
    upper = name.upper()
    if "ST" in upper:
        return 0.05
    if code.startswith(("300", "688")):
        return 0.20
    return 0.10


def excluded_name(name: str, code: str) -> str:
    if code.startswith(("688", "689")):
        return "star_board"
    text = name.upper()
    for token in ("ST", "退", "医药", "制药", "生物", "药业", "津投"):
        if token.upper() in text or token in name:
            return token
    return ""


def add_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.sort_values("date").reset_index(drop=True).copy()
    close = out["close"]
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    dif = ema12 - ema26
    dea = dif.ewm(span=9, adjust=False).mean()
    hist = 2 * (dif - dea)
    prev_close = close.shift(1)
    tr = pd.concat([(out["high"] - out["low"]), (out["high"] - prev_close).abs(), (out["low"] - prev_close).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / 14, adjust=False).mean()
    out["dif"] = dif
    out["hist"] = hist
    out["atr"] = atr
    out["vol_ma5_yday"] = out["volume"].shift(1).rolling(5).mean()
    out["low_10"] = out["low"].rolling(10).min()
    out["ret_5"] = close / close.shift(5) - 1
    out["ret_20"] = close / close.shift(20) - 1
    out["cross"] = (hist.shift(1) <= 0) & (hist > 0) & (dif > 0)
    out["vol_ok"] = out["volume"] > 1.5 * out["vol_ma5_yday"]
    return out


def regime_of(index_frame: pd.DataFrame) -> pd.DataFrame:
    out = index_frame.sort_values("date").reset_index(drop=True).copy()
    close = out["close"]
    ma20 = close.rolling(20).mean()
    ma60 = close.rolling(60).mean()
    regime = np.full(len(out), "MIXED", dtype=object)
    up = (close > ma20) & (ma20 > ma60)
    down = (close < ma20) & (ma20 < ma60)
    regime[up.fillna(False).to_numpy()] = "ONLY_UP"
    regime[down.fillna(False).to_numpy()] = "DOWN"
    out["regime"] = regime
    return out[["date", "regime"]]


def _buy_cost(shares: int, px: float, scale: float) -> float:
    fill = px * (1 + SLIPPAGE * scale)
    return shares * fill + max(shares * fill * COMMISSION * scale, MIN_COMMISSION * scale)


def _sell_proceeds(shares: int, px: float, scale: float) -> float:
    fill = px * (1 - SLIPPAGE * scale)
    notional = shares * fill
    comm = max(notional * COMMISSION * scale, MIN_COMMISSION * scale)
    return notional - comm - notional * STAMP


def _limit_down(px: float, prev_close: float, limit: float) -> bool:
    return prev_close > 0 and px / prev_close - 1 <= -(limit - 0.001)


def _limit_up(px: float, prev_close: float, limit: float) -> bool:
    return prev_close > 0 and px / prev_close - 1 >= limit - 0.001


def simulate_trade(raw: pd.DataFrame, signal_i: int, stop_dist: float, limit: float, scale: float = 1.0) -> dict | None:
    entry_i = signal_i + 1
    final_i = entry_i + HOLD_FINAL
    if final_i >= len(raw) or not math.isfinite(stop_dist):
        return None
    entry = float(raw.iloc[entry_i]["open"])
    signal_close = float(raw.iloc[signal_i]["close"])
    if entry <= 0 or signal_close <= 0:
        return None
    gap = entry / signal_close - 1
    if gap > GAP_SKIP:
        return {"status": "gap_skip", "gap": gap}
    prev_for_entry = signal_close
    if _limit_up(entry, prev_for_entry, limit) or float(raw.iloc[entry_i]["volume"]) <= 0:
        return {"status": "limit_or_halt", "gap": gap}
    stop_px = entry * (1 - stop_dist)
    target_px = entry + 2 * (entry - stop_px)
    risk_per_share = entry - stop_px
    if risk_per_share <= 0:
        return None
    shares = int(math.floor(1 / risk_per_share / 100) * 100)  # placeholder, resized by caller
    # caller passes sized shares via stop only; sizing is outside. This function uses shares set later.
    return {
        "status": "sized",
        "entry_i": entry_i,
        "final_i": final_i,
        "entry": entry,
        "gap": gap,
        "stop_px": stop_px,
        "target_px": target_px,
        "risk_per_share": risk_per_share,
        "signal_close": signal_close,
    }


def run_exit(raw: pd.DataFrame, entry_i: int, shares: int, stop_px: float, target_px: float, limit: float, scale: float) -> dict | None:
    final_i = entry_i + HOLD_FINAL
    half_i = entry_i + HOLD_HALF
    if shares < 100 or final_i >= len(raw):
        return None
    remaining = shares
    cash_in = 0.0
    defer = 0
    fills = []
    exit_session = "open"
    exit_i = final_i
    half_done = False
    for d in range(entry_i + 1, final_i + 1 + DEFER_LIMIT):
        if d >= len(raw) or remaining <= 0:
            break
        row = raw.iloc[d]
        prev = float(raw.iloc[d - 1]["close"])
        open_px = float(row["open"])
        high = float(row["high"])
        low = float(row["low"])
        locked = _limit_down(float(row["high"]), prev, limit) and _limit_down(low, prev, limit)
        sold_today = False
        if locked:
            defer += 1
            if defer > DEFER_LIMIT:
                cash_in += _sell_proceeds(remaining, open_px, scale)
                fills.append((str(row["date"].date()), remaining, open_px, "forced_limit"))
                remaining = 0
                exit_session = "open"
                exit_i = d
            continue
        if open_px <= stop_px:
            cash_in += _sell_proceeds(remaining, open_px, scale)
            fills.append((str(row["date"].date()), remaining, open_px, "gap_stop"))
            remaining = 0
            exit_session = "open"
            exit_i = d
            break
        if d == half_i and not half_done and remaining >= 200:
            half_shares = (shares // 200) * 100
            half_shares = min(half_shares, remaining - 100)
            if half_shares >= 100:
                cash_in += _sell_proceeds(half_shares, open_px, scale)
                fills.append((str(row["date"].date()), half_shares, open_px, "half"))
                remaining -= half_shares
                half_done = True
                sold_today = True
        if d >= final_i and remaining > 0:
            cash_in += _sell_proceeds(remaining, open_px, scale)
            fills.append((str(row["date"].date()), remaining, open_px, "time"))
            remaining = 0
            exit_session = "open"
            exit_i = d
            break
        if remaining > 0 and low <= stop_px:
            cash_in += _sell_proceeds(remaining, stop_px, scale)
            fills.append((str(row["date"].date()), remaining, stop_px, "stop"))
            remaining = 0
            exit_session = "intraday"
            exit_i = d
            break
        if remaining > 0 and high >= target_px and not sold_today:
            cash_in += _sell_proceeds(remaining, target_px, scale)
            fills.append((str(row["date"].date()), remaining, target_px, "target"))
            remaining = 0
            exit_session = "intraday"
            exit_i = d
            break
        if remaining > 0 and high >= target_px and sold_today:
            cash_in += _sell_proceeds(remaining, target_px, scale)
            fills.append((str(row["date"].date()), remaining, target_px, "target"))
            remaining = 0
            exit_session = "intraday"
            exit_i = d
            break
    if remaining > 0:
        return None
    buy = _buy_cost(shares, float(raw.iloc[entry_i]["open"]), scale)
    pnl = cash_in - buy
    return {
        "pnl": pnl,
        "ret": pnl / buy,
        "buy": buy,
        "cash_in": cash_in,
        "fills": fills,
        "exit_i": exit_i,
        "exit_session": exit_session,
        "exit_date": raw.iloc[exit_i]["date"],
        "win": pnl > 0,
    }


def stop_distance(qfq_row_close: float, low_10: float, atr: float) -> float:
    if not (qfq_row_close > 0 and np.isfinite(low_10) and np.isfinite(atr)):
        return float("nan")
    dist = (qfq_row_close - (low_10 - 0.5 * atr)) / qfq_row_close
    return float(min(max(dist, STOP_MIN), STOP_MAX))


def build_signals(code: str, name: str, qfq: pd.DataFrame, raw: pd.DataFrame, regimes: pd.DataFrame) -> pd.DataFrame:
    ind = add_indicators(qfq)
    raw = raw.sort_values("date").reset_index(drop=True)
    if not ind["date"].equals(raw["date"]):
        raise ValueError(f"{code} adjusted and raw calendars differ")
    merged = ind.merge(regimes, on="date", how="left")
    limit = limit_pct(code, name)
    rows = []
    for i, row in merged.iterrows():
        if i < 60 or row["cross"] != True or row["vol_ok"] != True:
            continue
        if row["regime"] == "DOWN" or not isinstance(row["regime"], str):
            continue
        dist = stop_distance(float(row["close"]), float(row["low_10"]), float(row["atr"]))
        plan = simulate_trade(raw, int(i), dist, limit)
        if plan is None:
            continue
        if plan["status"] != "sized":
            rows.append(
                {
                    "code": code,
                    "name": name,
                    "signal_date": row["date"],
                    "status": plan["status"],
                    "gap": plan.get("gap", np.nan),
                    "regime": row["regime"],
                }
            )
            continue
        risk_budget = RISK_UP if row["regime"] == "ONLY_UP" else RISK_MIXED
        shares = int(math.floor(risk_budget / plan["risk_per_share"] / 100.0) * 100)
        if shares < 100:
            rows.append({"code": code, "name": name, "signal_date": row["date"], "status": "too_small", "regime": row["regime"]})
            continue
        trade = run_exit(raw, plan["entry_i"], shares, plan["stop_px"], plan["target_px"], limit, 1.0)
        trade2 = run_exit(raw, plan["entry_i"], shares, plan["stop_px"], plan["target_px"], limit, 2.0)
        fills_json = json.dumps(trade["fills"], ensure_ascii=False) if trade else "[]"
        if trade is None or trade2 is None:
            continue
        target_i = int(i) + 5
        direction = np.nan
        if target_i < len(raw):
            move = float(raw.iloc[target_i]["close"]) - float(raw.iloc[i]["close"])
            direction = 0 if abs(move) < FLAT_EPS else (1 if move > 0 else -1)
        rows.append(
            {
                "code": code,
                "name": name,
                "signal_date": pd.Timestamp(row["date"]),
                "entry_date": raw.iloc[plan["entry_i"]]["date"],
                "exit_date": trade["exit_date"],
                "exit_session": trade["exit_session"],
                "status": "candidate",
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
                "fills": fills_json,
                "fills_stress": json.dumps(trade2["fills"], ensure_ascii=False),
                "signal_i": int(i),
                "entry_i": plan["entry_i"],
                "exit_i": trade["exit_i"],
            }
        )
    return pd.DataFrame(rows)


def apply_portfolio(candidates: pd.DataFrame) -> pd.DataFrame:
    pool = candidates[candidates["status"] == "candidate"].sort_values(["entry_date", "amount"], ascending=[True, False])
    cash = ACCOUNT
    open_slots: list[dict] = []
    accepted = []
    for row in pool.itertuples(index=False):
        entry_date = pd.Timestamp(row.entry_date)
        still = []
        for slot in open_slots:
            exit_date = pd.Timestamp(slot["exit_date"])
            freed = exit_date < entry_date or (exit_date == entry_date and slot["exit_session"] == "open")
            if freed:
                cash += slot["proceeds"]
            else:
                still.append(slot)
        open_slots = still
        if any(slot["code"] == row.code for slot in open_slots) or len(open_slots) >= MAX_POSITIONS:
            continue
        if row.buy > cash:
            continue
        cash -= row.buy
        open_slots.append(
            {
                "code": row.code,
                "exit_date": row.exit_date,
                "exit_session": row.exit_session,
                "proceeds": row.buy + row.pnl,
            }
        )
        accepted.append(row._asdict())
    return pd.DataFrame(accepted)


def equity_curve(trades: pd.DataFrame, raws: dict[str, pd.DataFrame], stress: bool = False) -> tuple[float, pd.DataFrame]:
    """Cash plus remaining shares marked at the raw close. Sells are booked at their fill prices."""
    if trades.empty:
        return 0.0, pd.DataFrame(columns=["date", "equity"])
    dates = sorted(set.union(*[set(pd.to_datetime(raw["date"])) for raw in raws.values()]))
    start = pd.Timestamp(trades["entry_date"].min())
    end = pd.Timestamp(trades["exit_date"].max())
    dates = [date for date in dates if start <= date <= end]
    close_map = {code: pd.Series(raw["close"].to_numpy(), index=pd.to_datetime(raw["date"])) for code, raw in raws.items()}
    events: dict[pd.Timestamp, list] = {}
    for trade in trades.itertuples(index=False):
        scale = 2.0 if stress else 1.0
        buy = _buy_cost(int(trade.shares), float(trade.entry), scale)
        fill_key = "fills_stress" if stress else "fills"
        events.setdefault(pd.Timestamp(trade.entry_date), []).append(("buy", trade.code, int(trade.shares), buy))
        for date_text, shares, price, _reason in json.loads(getattr(trade, fill_key)):
            events.setdefault(pd.Timestamp(date_text), []).append(("sell", trade.code, int(shares), float(price), scale))
    cash = ACCOUNT
    held: dict[str, int] = {}
    peak = ACCOUNT
    worst = 0.0
    rows = []
    for date in dates:
        for kind, code, shares, value, *extra in events.get(date, []):
            if kind == "sell":
                cash += _sell_proceeds(shares, value, extra[0])
                held[code] = held.get(code, 0) - shares
        for kind, code, shares, value, *extra in events.get(date, []):
            if kind == "buy":
                cash -= value
                held[code] = held.get(code, 0) + shares
        equity = cash
        for code, shares in held.items():
            if shares > 0 and date in close_map[code].index:
                equity += shares * float(close_map[code].loc[date])
        peak = max(peak, equity)
        worst = min(worst, equity / peak - 1 if peak else 0.0)
        rows.append({"date": date, "equity": equity})
    return float(worst), pd.DataFrame(rows)
