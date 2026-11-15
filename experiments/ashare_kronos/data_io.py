"""Fetch and check real A-share daily OHLCV. No synthetic bars."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from experiments.ashare_kronos.config import (
    AS_OF_DATE,
    HISTORY_START,
    UNIVERSE,
)

EM_URL = "http://push2his.eastmoney.com/api/qt/stock/kline/get"
TX_URL = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
SINA_URL = "https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData"
HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Referer": "https://quote.eastmoney.com/",
}


def market_prefix(market: str) -> tuple[str, str]:
    if market == "SH":
        return "1", "sh"
    if market == "SZ":
        return "0", "sz"
    raise ValueError(market)


def fetch_eastmoney(code: str, market: str, fqt: int, beg: str, end: str) -> tuple[str, pd.DataFrame]:
    secid_mkt, _ = market_prefix(market)
    params = {
        "secid": f"{secid_mkt}.{code}",
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
        "klt": "101",
        "fqt": str(fqt),
        "beg": beg,
        "end": end,
        "lmt": "10000",
    }
    last_err: Exception | None = None
    for attempt in range(4):
        try:
            response = requests.get(EM_URL, params=params, headers=HEADERS, timeout=30)
            response.raise_for_status()
            payload = response.json()
            data = payload.get("data") or {}
            klines = data.get("klines") or []
            if not klines:
                raise RuntimeError(f"empty klines rc={payload.get('rc')} code={code} fqt={fqt}")
            name = str(data.get("name") or "")
            rows = []
            for line in klines:
                parts = line.split(",")
                if len(parts) < 7:
                    raise RuntimeError(f"short kline row {line}")
                rows.append(
                    {
                        "date": parts[0],
                        "open": float(parts[1]),
                        "close": float(parts[2]),
                        "high": float(parts[3]),
                        "low": float(parts[4]),
                        "volume": float(parts[5]),
                        "amount": float(parts[6]),
                        "amplitude": float(parts[7]) if len(parts) > 7 and parts[7] else np.nan,
                        "pct_chg": float(parts[8]) if len(parts) > 8 and parts[8] else np.nan,
                        "change": float(parts[9]) if len(parts) > 9 and parts[9] else np.nan,
                        "turnover": float(parts[10]) if len(parts) > 10 and parts[10] else np.nan,
                    }
                )
            frame = pd.DataFrame(rows)
            frame["date"] = pd.to_datetime(frame["date"])
            frame = frame.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
            return name, frame
        except Exception as exc:  # noqa: BLE001 - retry network/parse failures
            last_err = exc
            time.sleep(1.2 * (attempt + 1))
    raise RuntimeError(f"eastmoney failed for {code} fqt={fqt}: {last_err}")


def fetch_tencent_qfq(code: str, market: str, beg: str, end: str) -> pd.DataFrame:
    _, prefix = market_prefix(market)
    param = f"{prefix}{code},day,{beg},{end},2000,qfq"
    response = requests.get(TX_URL, params={"param": param}, headers=HEADERS, timeout=30)
    response.raise_for_status()
    block = response.json()["data"][f"{prefix}{code}"]
    rows = block.get("qfqday") or block.get("day") or []
    if not rows:
        return pd.DataFrame(columns=["date", "open", "close", "high", "low", "volume"])
    # Tencent order is date, open, close, high, low, volume.
    frame = pd.DataFrame(rows).iloc[:, :6]
    frame.columns = ["date", "open", "close", "high", "low", "volume"]
    for col in ("open", "close", "high", "low", "volume"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame["date"] = pd.to_datetime(frame["date"])
    return frame.sort_values("date").drop_duplicates("date").reset_index(drop=True)


def fetch_sina_raw(code: str, market: str, datalen: int = 80) -> pd.DataFrame:
    _, prefix = market_prefix(market)
    response = requests.get(
        SINA_URL,
        params={"symbol": f"{prefix}{code}", "scale": 240, "ma": "no", "datalen": datalen},
        headers=HEADERS,
        timeout=30,
    )
    response.raise_for_status()
    frame = pd.DataFrame(response.json())
    if frame.empty:
        return frame
    frame["date"] = pd.to_datetime(frame["day"])
    for col in ("open", "high", "low", "close", "volume"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    # Sina volume is shares; Eastmoney/Tencent daily volume is lots (手).
    frame["volume_lot"] = frame["volume"] / 100.0
    return frame.sort_values("date").reset_index(drop=True)


def _ohlc_bad(frame: pd.DataFrame) -> pd.Series:
    return (
        (frame[["open", "high", "low", "close"]] <= 0).any(axis=1)
        | frame[["open", "high", "low", "close", "volume", "amount"]].isna().any(axis=1)
        | (frame["high"] + 1e-6 < frame["low"])
        | (frame["high"] + 1e-4 < frame[["open", "close"]].max(axis=1))
        | (frame["low"] - 1e-4 > frame[["open", "close"]].min(axis=1))
    )


def quality_report(code: str, meta: dict, qfq: pd.DataFrame, raw: pd.DataFrame) -> dict:
    qfq = qfq.copy()
    raw = raw.copy()
    merged = qfq.merge(raw, on="date", suffixes=("_qfq", "_raw"), how="outer", indicator=True)
    both = merged[merged["_merge"] == "both"].copy()
    q_ret = both["close_qfq"].pct_change()
    r_ret = both["close_raw"].pct_change()
    ratio = both["close_qfq"] / both["close_raw"]
    ratio_jump = ratio.pct_change().abs()
    calendar_gap = both["date"].diff().dt.days
    limit = 0.05 if "ST" in str(meta.get("vendor_name", "")).upper() else float(meta["limit"])
    unexplained = (q_ret.abs() > max(0.25, limit * 1.4)) & (r_ret.abs() > max(0.25, limit * 1.4))
    multiplier = (r_ret.abs() > 4.0) | (q_ret.abs() > 4.0)
    adjustment = (r_ret.abs() > 0.12) & (q_ret.abs() < 0.08)
    zero_vol = both["volume_qfq"] <= 0
    long_gap = calendar_gap > 15
    bad = _ohlc_bad(
        pd.DataFrame(
            {
                "open": both["open_qfq"],
                "high": both["high_qfq"],
                "low": both["low_qfq"],
                "close": both["close_qfq"],
                "volume": both["volume_qfq"],
                "amount": both["amount_qfq"],
            }
        )
    )
    both = both.assign(
        bad_ohlc=bad.to_numpy(),
        unexplained_jump=unexplained.fillna(False).to_numpy(),
        multiplier_glitch=multiplier.fillna(False).to_numpy(),
        adjustment_event=adjustment.fillna(False).to_numpy(),
        zero_volume=zero_vol.to_numpy(),
        long_gap=long_gap.fillna(False).to_numpy(),
        qfq_raw_ratio=ratio.to_numpy(),
    )
    return {
        "summary": {
            "code": code,
            "vendor_name": meta.get("vendor_name"),
            "industry": meta["industry"],
            "board": meta["board"],
            "limit_used": limit,
            "n_qfq": int(len(qfq)),
            "n_raw": int(len(raw)),
            "n_date_mismatch": int((merged["_merge"] != "both").sum()),
            "first_date": str(both["date"].min().date()) if len(both) else None,
            "last_date": str(both["date"].max().date()) if len(both) else None,
            "n_bad_ohlc": int(both["bad_ohlc"].sum()),
            "n_zero_volume": int(both["zero_volume"].sum()),
            "n_long_gap": int(both["long_gap"].sum()),
            "n_adjustment_like": int(both["adjustment_event"].sum()),
            "n_unexplained_jump": int(both["unexplained_jump"].sum()),
            "n_multiplier_glitch": int(both["multiplier_glitch"].sum()),
            "max_abs_qfq_ret": float(q_ret.abs().max()) if q_ret.notna().any() else None,
            "max_abs_raw_ret": float(r_ret.abs().max()) if r_ret.notna().any() else None,
            "ratio_end": float(ratio.iloc[-1]) if len(ratio) else None,
            "ratio_min": float(ratio.min()) if len(ratio) else None,
            "ratio_max": float(ratio.max()) if len(ratio) else None,
            "st_in_name": "ST" in str(meta.get("vendor_name", "")).upper(),
        },
        "bars": both,
    }


def cross_check(code: str, market: str, qfq: pd.DataFrame, raw: pd.DataFrame) -> dict:
    beg = HISTORY_START
    end = AS_OF_DATE.replace("-", "")
    tencent = fetch_tencent_qfq(code, market, beg, end[:4] + "-" + end[4:6] + "-" + end[6:])
    sina = fetch_sina_raw(code, market)
    out = {"code": code, "tencent_rows": int(len(tencent)), "sina_rows": int(len(sina))}
    if not tencent.empty:
        joined = qfq.merge(tencent, on="date", suffixes=("_em", "_tx"))
        ret_em = joined["close_em"].pct_change()
        ret_tx = joined["close_tx"].pct_change()
        out.update(
            {
                "tencent_overlap": int(len(joined)),
                "tencent_max_abs_close_rel": float(((joined["close_em"] - joined["close_tx"]).abs() / joined["close_tx"]).max()),
                "tencent_median_abs_close_rel": float(((joined["close_em"] - joined["close_tx"]).abs() / joined["close_tx"]).median()),
                "tencent_return_corr": float(ret_em.corr(ret_tx)) if ret_em.notna().sum() > 5 else None,
                "tencent_max_abs_return_diff": float((ret_em - ret_tx).abs().max()) if ret_em.notna().any() else None,
                "tencent_last_date": str(tencent["date"].max().date()),
                "tencent_last_close": float(tencent["close"].iloc[-1]),
                "eastmoney_last_close_qfq": float(qfq["close"].iloc[-1]),
            }
        )
    if not sina.empty:
        joined = raw.merge(sina, on="date", suffixes=("_em", "_sina"))
        if len(joined) >= 5:
            vol_rel = (joined["volume_em"] - joined["volume_lot"]).abs() / joined["volume_em"].replace(0, np.nan)
            out.update(
                {
                    "sina_overlap": int(len(joined)),
                    "sina_close_max_abs_rel": float(((joined["close_em"] - joined["close_sina"]).abs() / joined["close_sina"]).max()),
                    "sina_volume_median_abs_rel": float(vol_rel.median()),
                    "sina_last_date": str(sina["date"].max().date()),
                }
            )
    return out


def load_universe(data_dir: Path) -> dict[str, dict]:
    loaded = {}
    for spec in UNIVERSE:
        code = spec["code"]
        qfq = pd.read_csv(data_dir / f"{code}_qfq.csv", parse_dates=["date"])
        raw = pd.read_csv(data_dir / f"{code}_raw.csv", parse_dates=["date"])
        meta = dict(spec)
        name_path = data_dir / f"{code}_name.txt"
        meta["vendor_name"] = name_path.read_text(encoding="utf-8").strip() if name_path.exists() else ""
        meta["st_in_name"] = "ST" in meta["vendor_name"].upper()
        loaded[code] = {"meta": meta, "qfq": qfq, "raw": raw}
    return loaded


def fetch_all(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    checks = []
    beg = HISTORY_START.replace("-", "")
    end = "20261114"  # last calendar day before AS_OF_DATE
    for spec in UNIVERSE:
        code = spec["code"]
        name, qfq = fetch_eastmoney(code, spec["market"], fqt=1, beg=beg, end=end)
        _, raw = fetch_eastmoney(code, spec["market"], fqt=0, beg=beg, end=end)
        qfq = qfq[qfq["date"] < pd.Timestamp(AS_OF_DATE)].reset_index(drop=True)
        raw = raw[raw["date"] < pd.Timestamp(AS_OF_DATE)].reset_index(drop=True)
        if qfq.empty or raw.empty:
            raise RuntimeError(f"{code} has no bars before {AS_OF_DATE}")
        if qfq["date"].max() != raw["date"].max():
            raise RuntimeError(f"{code} qfq/raw last date mismatch")
        meta = dict(spec)
        meta["vendor_name"] = name
        (data_dir / f"{code}_name.txt").write_text(name + "\n", encoding="utf-8")
        qfq.to_csv(data_dir / f"{code}_qfq.csv", index=False)
        raw.to_csv(data_dir / f"{code}_raw.csv", index=False)
        report = quality_report(code, meta, qfq, raw)
        summaries.append(report["summary"])
        report["bars"].to_csv(data_dir / f"{code}_bar_flags.csv", index=False)
        checks.append(cross_check(code, spec["market"], qfq, raw))
        time.sleep(0.4)
    summary = pd.DataFrame(summaries)
    check = pd.DataFrame(checks)
    summary.to_csv(data_dir / "quality_summary.csv", index=False)
    check.to_csv(data_dir / "crosscheck.csv", index=False)
    return summary, check
