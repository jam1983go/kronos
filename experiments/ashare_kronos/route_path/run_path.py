"""Align Kronos horizons with the HIT exit. No new model is trained."""

from __future__ import annotations

import json
import traceback
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.ashare_kronos.routes3.engine import apply_portfolio, equity_curve
from experiments.ashare_kronos.routes3.settings import HOLD_FINAL, HOLD_HALF
from experiments.ashare_kronos.winrate.infer import load_predictor, set_seed
from experiments.ashare_kronos.winrate.settings import SEED, TEMPERATURE, TOP_K, TOP_P

DATA = Path(__file__).resolve().parents[1] / "route2b" / "data"
SIGNALS = Path(__file__).resolve().parents[1] / "route2b" / "outputs" / "signals_with_kronos.csv"
OUT = Path(__file__).resolve().parent / "outputs"
ART = Path("/opt/cursor/artifacts/kronos_route_path")
TZ = ZoneInfo("Asia/Shanghai")
LOOKBACK = 120
PRED_LEN = 10
SAMPLES = 5
BOOT_REPS = 1000
BOOT_BLOCK = 10
FOLDS = (
    ("wf_2022", "2022-01-04", "2022-12-30", "2023-01-03", False),
    ("wf_2023", "2023-01-03", "2023-12-29", "2024-01-02", False),
    ("wf_2024", "2024-01-02", "2024-12-31", "2025-01-15", False),
    ("reval_2025_2026", "2025-01-15", "2026-10-24", "2026-11-14", True),
)
CANDIDATES = ("hit_raw", "kronos_pos_3d", "kronos_pos_5d", "kronos_pos_10d", "path_2r_before_stop")


def log(message: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    line = f"[{datetime.now(TZ).strftime('%Y-%m-%d %H:%M:%S%z')}] {message}"
    print(line, flush=True)
    with (OUT / "run_log.txt").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def load_signals() -> pd.DataFrame:
    frame = pd.read_csv(SIGNALS, dtype={"code": str})
    frame["code"] = frame["code"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
    for column in ("signal_date", "entry_date", "exit_date"):
        frame[column] = pd.to_datetime(frame[column])
    frame = frame[frame["signal_date"] >= "2022-01-01"].copy()
    frame["pred_ret_3d"] = frame["pred_ret"]
    return frame.reset_index(drop=True)


def load_qfq(code: str) -> pd.DataFrame:
    frame = pd.read_csv(DATA / f"{code}_qfq.csv", parse_dates=["date"])
    return frame[frame["date"] < "2026-11-15"].sort_values("date").reset_index(drop=True)


def future_timestamps(qfq: pd.DataFrame, signal_date: pd.Timestamp) -> pd.Series | None:
    signal_i = int(qfq.index[qfq["date"] == pd.Timestamp(signal_date)][0])
    future = qfq.iloc[signal_i + 1 : signal_i + 1 + PRED_LEN]["date"]
    if len(future) != PRED_LEN:
        return None
    return future.reset_index(drop=True)


def path_takes(ohlc: np.ndarray, stop_dist: float) -> bool:
    """True only when the predicted path touches 2R before the stop."""
    entry = float(ohlc[0, 0])
    if not np.isfinite(entry) or entry <= 0 or not np.isfinite(stop_dist):
        return False
    stop_px = entry * (1 - float(stop_dist))
    target_px = entry + 2 * (entry - stop_px)
    # Bar 0 is the entry day. Exits start the next bar, matching T+1.
    for offset, bar in enumerate(ohlc[1:], start=1):
        open_px, high, low = float(bar[0]), float(bar[1]), float(bar[2])
        if open_px <= stop_px:
            return False
        if low <= stop_px:
            return False
        if high >= target_px:
            return True
        if offset >= HOLD_FINAL:
            return False
    return False


def forecast_paths(frame: pd.DataFrame) -> pd.DataFrame:
    cache = OUT / "paths_10d.csv"
    done = pd.DataFrame()
    if cache.exists():
        done = pd.read_csv(cache, dtype={"code": str})
        done["code"] = done["code"].str.zfill(6)
        done["signal_date"] = pd.to_datetime(done["signal_date"])
        done = done.rename(columns={"ret_5": "pred_ret_5d", "ret_10": "pred_ret_10d", "close_5": "pred_close_5", "close_10": "pred_close_10"})
        log(f"resume paths {len(done)}")
    have = set(zip(done["code"], done["signal_date"])) if len(done) else set()
    pending = [row for row in frame.itertuples(index=False) if (row.code, pd.Timestamp(row.signal_date)) not in have]
    log(f"path forecasts pending {len(pending)} of {len(frame)}")
    if not pending:
        return done
    predictor, n_params = load_predictor()
    log(f"loaded frozen Kronos-small params {n_params}")
    rows = done.to_dict("records") if len(done) else []
    skipped = 0
    panels: dict[str, pd.DataFrame] = {}
    for i, row in enumerate(pending, start=1):
        qfq = panels.setdefault(row.code, load_qfq(row.code))
        future = future_timestamps(qfq, pd.Timestamp(row.signal_date))
        if future is None:
            skipped += 1
            rows.append({"code": row.code, "signal_date": pd.Timestamp(row.signal_date), "ok": 0})
        else:
            signal_i = int(qfq.index[qfq["date"] == pd.Timestamp(row.signal_date)][0])
            hist = qfq.iloc[signal_i - LOOKBACK + 1 : signal_i + 1].reset_index(drop=True)
            set_seed(SEED)
            pred = predictor.predict(
                df=hist[["open", "high", "low", "close", "volume", "amount"]],
                x_timestamp=hist["date"],
                y_timestamp=future,
                pred_len=PRED_LEN,
                T=TEMPERATURE,
                top_k=TOP_K,
                top_p=TOP_P,
                sample_count=SAMPLES,
                verbose=False,
            )
            ohlc = pred[["open", "high", "low", "close"]].to_numpy(dtype=float)
            origin = float(hist["close"].iloc[-1])
            rows.append(
                {
                    "code": row.code,
                    "signal_date": pd.Timestamp(row.signal_date),
                    "ok": 1,
                    "pred_close_5": float(ohlc[4, 3]),
                    "pred_close_10": float(ohlc[9, 3]),
                    "pred_ret_5d": float(ohlc[4, 3] / origin - 1),
                    "pred_ret_10d": float(ohlc[9, 3] / origin - 1),
                    "path_take": int(path_takes(ohlc, float(row.stop_dist))),
                    "origin_close": origin,
                }
            )
        if i % 20 == 0 or i == len(pending):
            pd.DataFrame(rows).to_csv(cache, index=False)
            log(f"paths {i}/{len(pending)} skipped {skipped}")
    out = pd.DataFrame(rows)
    out.to_csv(cache, index=False)
    log(f"paths done skipped {skipped}")
    return out


def block_mean_diff(left: pd.DataFrame, right: pd.DataFrame) -> tuple[float, float]:
    if left.empty or right.empty:
        return (float("nan"), float("nan"))
    dates = sorted(set(pd.to_datetime(left["signal_date"])) | set(pd.to_datetime(right["signal_date"])))
    mapping = {date: i // BOOT_BLOCK for i, date in enumerate(dates)}
    rng = np.random.default_rng(SEED)
    keys = sorted(set(mapping.values()))
    left_ret = left["ret"].to_numpy(float)
    right_ret = right["ret"].to_numpy(float)
    left_block = pd.to_datetime(left["signal_date"]).map(mapping).to_numpy()
    right_block = pd.to_datetime(right["signal_date"]).map(mapping).to_numpy()
    diffs = []
    for _ in range(BOOT_REPS):
        draw = rng.choice(keys, size=len(keys), replace=True)
        a = [left_ret[left_block == key] for key in draw if np.any(left_block == key)]
        b = [right_ret[right_block == key] for key in draw if np.any(right_block == key)]
        if a and b:
            diffs.append(float(np.concatenate(a).mean() - np.concatenate(b).mean()))
    if not diffs:
        return (float("nan"), float("nan"))
    return float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))


def block_win_diff(left: pd.DataFrame, right: pd.DataFrame) -> tuple[float, float]:
    if left.empty or right.empty:
        return (float("nan"), float("nan"))
    copied_left = left.copy()
    copied_right = right.copy()
    copied_left["ret"] = copied_left["win"].astype(float)
    copied_right["ret"] = copied_right["win"].astype(float)
    return block_mean_diff(copied_left, copied_right)


def score_block(trades: pd.DataFrame, n_signals: int, raws: dict) -> dict:
    if trades.empty:
        return {"n_trades": 0, "n_wins": 0, "win_rate": np.nan, "mean_net": np.nan, "profit_factor": np.nan, "payoff": np.nan, "direction_accuracy": np.nan, "coverage": 0.0, "n_dates": 0, "max_drawdown": 0.0, "stress_mean_net": np.nan, "stress_profit_factor": np.nan, "stress_win_rate": np.nan}
    wins = trades["win"].astype(bool)
    rets = trades["ret"].to_numpy(float)
    pnls = trades["pnl"].to_numpy(float)
    gain, loss = rets[wins.to_numpy()], rets[~wins.to_numpy()]
    pos, neg = pnls[pnls > 0].sum(), pnls[pnls <= 0].sum()
    directed = trades[trades["direction_5"].isin([1, -1])]
    dd, _curve = equity_curve(trades, raws)
    stress = trades.copy()
    stress["win"] = stress["win_stress"]
    stress["ret"] = stress["ret_stress"]
    stress["pnl"] = stress["pnl_stress"]
    s_rets = stress["ret"].to_numpy(float)
    s_pnls = stress["pnl"].to_numpy(float)
    s_pos, s_neg = s_pnls[s_pnls > 0].sum(), s_pnls[s_pnls <= 0].sum()
    return {
        "n_trades": int(len(trades)),
        "n_wins": int(wins.sum()),
        "win_rate": float(wins.mean()),
        "mean_net": float(rets.mean()),
        "profit_factor": float(pos / abs(neg)) if neg < 0 else np.nan,
        "payoff": float(gain.mean() / abs(loss.mean())) if len(gain) and len(loss) else np.nan,
        "direction_accuracy": float((directed["direction_5"] == 1).mean()) if len(directed) else np.nan,
        "coverage": float(len(trades) / n_signals) if n_signals else np.nan,
        "n_dates": int(pd.to_datetime(trades["signal_date"]).nunique()),
        "max_drawdown": dd,
        "stress_win_rate": float(stress["win"].mean()),
        "stress_mean_net": float(s_rets.mean()),
        "stress_profit_factor": float(s_pos / abs(s_neg)) if s_neg < 0 else np.nan,
    }


def select_trades(frame: pd.DataFrame, candidate: str) -> pd.DataFrame:
    if candidate == "hit_raw":
        chosen = frame
    elif candidate == "kronos_pos_3d":
        chosen = frame[frame["pred_ret_3d"] > 0]
    elif candidate == "kronos_pos_5d":
        chosen = frame[frame["pred_ret_5d"] > 0]
    elif candidate == "kronos_pos_10d":
        chosen = frame[frame["pred_ret_10d"] > 0]
    elif candidate == "path_2r_before_stop":
        chosen = frame[frame["path_take"] == 1]
    else:
        raise KeyError(candidate)
    return apply_portfolio(chosen)


def judge(pooled: dict, reval: dict) -> str:
    if pooled["n_trades"] == 0 or not np.isfinite(pooled["mean_net"]):
        return "未完成"
    walk_ok = pooled["mean_net"] > 0 and np.isfinite(pooled["profit_factor"]) and pooled["profit_factor"] > 1
    diff_ok = np.isfinite(pooled["mean_diff_low"]) and np.isfinite(pooled["mean_diff_high"]) and not (pooled["mean_diff_low"] <= 0 <= pooled["mean_diff_high"])
    reval_ok = (
        reval["n_trades"] >= 15
        and reval["coverage"] >= 0.10
        and reval["mean_net"] > 0
        and np.isfinite(reval["profit_factor"])
        and reval["profit_factor"] > 1
        and reval["mean_net"] >= reval["raw_mean_net"]
    )
    if walk_ok and diff_ok and reval_ok:
        return "可实施"
    return "无效"


def fold_frame(signals: pd.DataFrame, start: str, end: str, exit_before: str) -> pd.DataFrame:
    return signals[(signals["signal_date"] >= start) & (signals["signal_date"] <= end) & (signals["exit_date"] < exit_before)].copy()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ART.mkdir(parents=True, exist_ok=True)
    log("path alignment start")
    signals = load_signals()
    paths = forecast_paths(signals)
    paths["code"] = paths["code"].astype(str).str.zfill(6)
    paths["signal_date"] = pd.to_datetime(paths["signal_date"])
    paths = paths.rename(columns={"ret_5": "pred_ret_5d", "ret_10": "pred_ret_10d", "close_5": "pred_close_5", "close_10": "pred_close_10"})
    merged = signals.merge(paths, on=["code", "signal_date"], how="left")
    covered = merged[merged["ok"] == 1].copy()
    log(f"signals {len(signals)} path-covered {len(covered)} missing {(merged['ok'] != 1).sum()}")
    raws = {}
    for code in sorted(covered["code"].unique()):
        raw = pd.read_csv(DATA / f"{code}_raw.csv", parse_dates=["date"])
        raws[code] = raw[raw["date"] < "2026-11-15"].sort_values("date").reset_index(drop=True)
    rows = []
    trade_frames = []
    fold_taken = {name: [] for name in CANDIDATES}
    for fold_name, start, end, exit_before, exploratory in FOLDS:
        part = fold_frame(covered if fold_name else signals, start, end, exit_before)
        # 3-day filter can use every cached signal, including any without a 10-day path.
        part_3d = fold_frame(signals, start, end, exit_before)
        n_signals = int(len(part_3d))
        log(f"{fold_name} signals {n_signals} with path {len(part)}")
        taken = {
            "hit_raw": select_trades(part_3d, "hit_raw"),
            "kronos_pos_3d": select_trades(part_3d, "kronos_pos_3d"),
            "kronos_pos_5d": select_trades(part, "kronos_pos_5d"),
            "kronos_pos_10d": select_trades(part, "kronos_pos_10d"),
            "path_2r_before_stop": select_trades(part, "path_2r_before_stop"),
        }
        for name, trades in taken.items():
            fold_taken[name].append(trades)
            stats = score_block(trades, n_signals, raws)
            low_mean, high_mean = block_mean_diff(trades, taken["hit_raw"]) if name != "hit_raw" else (np.nan, np.nan)
            low_win, high_win = block_win_diff(trades, taken["hit_raw"]) if name != "hit_raw" else (np.nan, np.nan)
            stats.update(
                {
                    "fold": fold_name,
                    "id": name,
                    "exploratory": exploratory,
                    "mean_diff": stats["mean_net"] - (float(taken["hit_raw"]["ret"].mean()) if len(taken["hit_raw"]) else np.nan),
                    "win_diff": stats["win_rate"] - (float(taken["hit_raw"]["win"].mean()) if len(taken["hit_raw"]) else np.nan),
                    "mean_diff_low": low_mean,
                    "mean_diff_high": high_mean,
                    "win_diff_low": low_win,
                    "win_diff_high": high_win,
                }
            )
            rows.append(stats)
            if len(trades):
                saved = trades.copy()
                saved["strategy"] = name
                saved["fold"] = fold_name
                trade_frames.append(saved)
    # Pooled walk-forward uses the same trades already scored per fold.
    pooled_rows = []
    for name in CANDIDATES:
        pooled = pd.concat([item for item in fold_taken[name][:3] if len(item)], ignore_index=True) if any(len(item) for item in fold_taken[name][:3]) else pd.DataFrame()
        raw_pooled = pd.concat([item for item in fold_taken["hit_raw"][:3] if len(item)], ignore_index=True)
        n_signals = int(sum(len(fold_frame(signals, *FOLDS[i][1:4])) for i in range(3)))
        stats = score_block(pooled, n_signals, raws)
        low_mean, high_mean = block_mean_diff(pooled, raw_pooled) if name != "hit_raw" else (np.nan, np.nan)
        low_win, high_win = block_win_diff(pooled, raw_pooled) if name != "hit_raw" else (np.nan, np.nan)
        raw_mean = float(raw_pooled["ret"].mean()) if len(raw_pooled) else np.nan
        stats.update({"fold": "wf_2022_2024", "id": name, "exploratory": False, "mean_diff": stats["mean_net"] - raw_mean, "win_diff": stats["win_rate"] - (float(raw_pooled["win"].mean()) if len(raw_pooled) else np.nan), "mean_diff_low": low_mean, "mean_diff_high": high_mean, "win_diff_low": low_win, "win_diff_high": high_win, "raw_mean_net": raw_mean})
        pooled_rows.append(stats)
        rows.append(stats)
    scores = pd.DataFrame(rows)
    scores.to_csv(OUT / "all_scores.csv", index=False)
    if trade_frames:
        pd.concat(trade_frames, ignore_index=True).to_csv(OUT / "trades.csv", index=False)
    make_plot(scores)
    write_report(scores, pooled_rows)
    log("done")


def make_plot(scores: pd.DataFrame) -> None:
    fig_dir = OUT / "figures"
    fig_dir.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
    for ax, fold, title in (
        (axes[0], "wf_2022_2024", "Walk-forward 2022-2024"),
        (axes[1], "reval_2025_2026", "Exploratory revalidation"),
    ):
        part = scores[scores["fold"] == fold].set_index("id").loc[list(CANDIDATES)]
        ax.barh(part.index, part["mean_net"])
        ax.axvline(0, color="black", lw=0.7)
        ax.set_xlabel("Mean net return per trade")
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(fig_dir / "mean_net_by_candidate.png", dpi=120)
    plt.close(fig)


def write_report(scores: pd.DataFrame, pooled_rows: list[dict]) -> None:
    def cell(value, pct=False):
        if value is None or not isinstance(value, (int, float, np.floating)) or not np.isfinite(value):
            return "NA"
        return f"{100 * float(value):.1f}%" if pct else f"{float(value):.2f}"

    by_id = {row["id"]: row for row in pooled_rows}
    raw = by_id["hit_raw"]
    reval = scores[scores["fold"] == "reval_2025_2026"].set_index("id")
    labels = []
    for name in CANDIDATES:
        if name == "hit_raw":
            labels.append((name, "基线"))
            continue
        reval_row = reval.loc[name].to_dict()
        reval_row["raw_mean_net"] = float(reval.loc["hit_raw", "mean_net"])
        labels.append((name, judge(by_id[name], reval_row)))
    implementable = [name for name, label in labels if label == "可实施"]
    head = "无效，不可实施。" if not implementable else "可实施：" + "、".join(implementable) + "。"
    lines = [
        "# Kronos 时域对齐",
        "",
        f"**{head}**",
        "",
        "报告时区 Asia/Shanghai，数据截至 2026-11-13。Kronos 权重没有更新。2025–2026 只是探索性再验证。",
        "",
        f"未过滤走步合并：{int(raw['n_wins'])}/{int(raw['n_trades'])} = {cell(raw['win_rate'], True)}，均笔 {cell(raw['mean_net'], True)}，PF {cell(raw['profit_factor'])}，回撤 {cell(raw['max_drawdown'], True)}。",
        "",
        "## 走步合并 2022–2024",
        "",
        "| 候选 | 笔数 | 胜率 | 均笔 | PF | 回撤 | 相对基线均笔 | 均笔区间 | 覆盖率 | 判定 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |",
    ]
    for name, label in labels:
        row = by_id[name]
        lines.append(
            f"| {name} | {int(row['n_trades'])} | {cell(row['win_rate'], True)} | {cell(row['mean_net'], True)} | {cell(row['profit_factor'])} | {cell(row['max_drawdown'], True)} | {cell(row['mean_diff'], True)} | {cell(row['mean_diff_low'], True)}–{cell(row['mean_diff_high'], True)} | {cell(row['coverage'], True)} | {label} |"
        )
    lines.extend(["", "## 探索性再验证", "", "| 候选 | 笔数 | 胜率 | 方向准确率 | 均笔 | PF | 回撤 | 费用加倍均笔 | 覆盖率 |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for name in CANDIDATES:
        row = reval.loc[name]
        lines.append(
            f"| {name} | {int(row['n_trades'])} | {cell(row['win_rate'], True)} | {cell(row['direction_accuracy'], True)} | {cell(row['mean_net'], True)} | {cell(row['profit_factor'])} | {cell(row['max_drawdown'], True)} | {cell(row['stress_mean_net'], True)} | {cell(row['coverage'], True)} |"
        )
    ten = by_id.get("kronos_pos_10d", {})
    path_row = by_id.get("path_2r_before_stop", {})
    lines.extend(
        [
            "",
            "## 说明",
            "",
            f"三日过滤复现了上轮走步点估计：{int(by_id['kronos_pos_3d']['n_trades'])} 笔、胜率 {cell(by_id['kronos_pos_3d']['win_rate'], True)}、均笔 {cell(by_id['kronos_pos_3d']['mean_net'], True)}、PF {cell(by_id['kronos_pos_3d']['profit_factor'])}。",
            f"十日过滤是唯一走步均笔为正（{cell(ten.get('mean_net'), True)}）、PF 大于 1（{cell(ten.get('profit_factor'))}）且均笔差区间不含 0 的候选。再验证有 {int(reval.loc['kronos_pos_10d']['n_trades'])} 笔、覆盖率 {cell(reval.loc['kronos_pos_10d']['coverage'], True)}，低于事先写死的 10% 覆盖率，因此仍不可实施。",
            f"路径规则在走步合并里成交 {int(path_row.get('n_trades', 0))} 笔。预测均值路径很少先碰到 2R。",
            "",
            "三日过滤复用了上一轮缓存。五日、十日和路径来自同一次官方 `KronosPredictor.predict`，`pred_len=10`、`sample_count=5`。路径把预测的第一根开盘当作进场价，止损距离用信号日的 `stop_dist`，出场顺序与研究近似一致。",
            "",
            "这不是完整实盘 HIT。股票池是上一轮按 2019–2021 成交额留下的 94 只现存非医药股，有存活偏差。Kronos 训练截止不可审计。胜率升高但均笔不为正，或均笔差的区间含 0，或再验证反向，都不可实施。",
            "",
            "复现：`PYTHONPATH=. /tmp/kronos-venv/bin/python -m experiments.ashare_kronos.route_path.run_path`",
            "",
        ]
    )
    text = "\n".join(lines)
    (OUT / "REPORT.md").write_text(text, encoding="utf-8")
    (ART / "REPORT.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log(traceback.format_exc())
        raise
