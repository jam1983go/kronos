"""Score frozen configs. Selection uses only the selection window."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.ashare_kronos.winrate.metrics import (
    block_bootstrap_diff,
    choose_modification,
    decide_signal,
    eligible,
    net_return,
    summarize_trades,
)
from experiments.ashare_kronos.winrate.metrics import _block_ids
from experiments.ashare_kronos.winrate.settings import (
    BOOTSTRAP_REPS,
    COMMISSION,
    CONFIGS,
    COST_HURDLE,
    MATERIAL_WIN_GAIN,
    MIN_COVERAGE,
    MIN_PAYOFF,
    MIN_PROFIT_FACTOR,
    REVALIDATION_MIN_TRADES_PER_ETF,
    SEED,
    SLIPPAGE,
    WINDOWS,
)


def _sample_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column.startswith("pred_close_")]


def load_predictions(out_dir: Path) -> dict[tuple[int, int], pd.DataFrame]:
    loaded = {}
    for lookback, samples in ((60, 5), (120, 5), (120, 20), (240, 5)):
        frame = pd.read_csv(out_dir / f"preds_lb{lookback}_s{samples}.csv", dtype={"code": str})
        frame["code"] = frame["code"].str.zfill(6)
        frame["signal_date"] = pd.to_datetime(frame["signal_date"])
        loaded[(lookback, samples)] = frame
    return loaded


def _prediction_features(row: pd.Series) -> tuple[float, float, int, int]:
    values = np.array([row[column] for column in _sample_columns(row.to_frame().T) if pd.notna(row[column])], dtype=float)
    # row is a Series; columns that exist:
    values = []
    for column, value in row.items():
        if column.startswith("pred_close_") and pd.notna(value):
            values.append(float(value))
    closes = np.array(values, dtype=float)
    rets = closes / float(row["origin_close"]) - 1
    return float(rets.mean()), float(np.median(rets)), int((rets > 0).sum()), int(len(rets))


def build_feature_table(slots: pd.DataFrame, predictions: dict[tuple[int, int], pd.DataFrame]) -> pd.DataFrame:
    base = slots[slots["window"].isin(WINDOWS) & slots["executable"]].copy()
    pieces = [base]
    for (lookback, samples), frame in predictions.items():
        slim = frame.copy()
        features = []
        for row in slim.itertuples(index=False):
            closes = [getattr(row, f"pred_close_{i}") for i in range(samples)]
            closes = np.array(closes, dtype=float)
            rets = closes / float(row.origin_close) - 1
            features.append(
                {
                    "code": str(row.code).zfill(6),
                    "signal_date": pd.Timestamp(row.signal_date),
                    f"mean_lb{lookback}_s{samples}": float(rets.mean()),
                    f"median_lb{lookback}_s{samples}": float(np.median(rets)),
                    f"pos_lb{lookback}_s{samples}": int((rets > 0).sum()),
                }
            )
        feat = pd.DataFrame(features)
        base = base.merge(feat, on=["code", "signal_date"], how="left")
    if base.empty:
        raise RuntimeError("no executable slots")
    missing = [column for column in base.columns if column.startswith("mean_") and base[column].isna().any()]
    if missing:
        raise RuntimeError(f"predictions missing for {missing}")
    return base


def _want(config: dict, row: pd.Series) -> bool:
    lookback = config["lookback"]
    samples = config["samples"]
    mean_ret = median_ret = positive = None
    if lookback and samples:
        mean_ret = float(row[f"mean_lb{lookback}_s{samples}"])
        median_ret = float(row[f"median_lb{lookback}_s{samples}"])
        positive = int(row[f"pos_lb{lookback}_s{samples}"])
    return decide_signal(config["rule"], mean_ret, median_ret, positive, samples, float(row["mom_ret"]), int(row["majority"]))


def trades_for(config: dict, slots: pd.DataFrame, cost_scale: float = 1.0) -> pd.DataFrame:
    rows = []
    for row in slots.itertuples(index=False):
        series = pd.Series(row._asdict())
        if not _want(config, series):
            continue
        gross = float(row.exit_open) / float(row.entry_open) - 1
        net = net_return(float(row.entry_open), float(row.exit_open), COMMISSION * cost_scale, SLIPPAGE * cost_scale)
        target_ret = float(row.target_close_raw) / float(row.signal_close_raw) - 1
        rows.append(
            {
                "config": config["id"],
                "code": row.code,
                "signal_date": pd.Timestamp(row.signal_date),
                "entry_date": pd.Timestamp(row.entry_date),
                "exit_date": pd.Timestamp(row.exit_date),
                "entry_open": float(row.entry_open),
                "exit_open": float(row.exit_open),
                "gross": gross,
                "net": net,
                "target_ret": target_ret,
                "deferred": int(row.deferred),
                "window": row.window,
                "cost_scale": cost_scale,
            }
        )
    return pd.DataFrame(rows)


def score_config(config: dict, window_slots: pd.DataFrame, panels: dict, n_etfs: int, span_days: float, cost_scale: float = 1.0) -> tuple[dict, pd.DataFrame]:
    trades = trades_for(config, window_slots, cost_scale=cost_scale)
    summary = summarize_trades(trades, n_slots=len(window_slots), n_etfs=n_etfs, span_days=span_days)
    summary.update({"id": config["id"], "kind": config["kind"], "rule": config["rule"], "eligible": eligible(summary) if config["kind"] == "modified" else False})
    summary["max_drawdown"] = portfolio_drawdown(trades, panels, n_etfs, cost_scale)[0]
    return summary, trades


def portfolio_drawdown(trades: pd.DataFrame, panels: dict, n_etfs: int, cost_scale: float) -> tuple[float, pd.DataFrame]:
    if trades.empty:
        return 0.0, pd.DataFrame(columns=["date", "equity", "drawdown"])
    raws = {code: pack["raw"].set_index("date") for code, pack in panels.items()}
    dates = sorted(set.union(*[set(raw.index) for raw in raws.values()]))
    start = pd.Timestamp(trades["entry_date"].min())
    end = pd.Timestamp(trades["exit_date"].max())
    dates = [date for date in dates if start <= date <= end]
    entries = {pd.Timestamp(date): group for date, group in trades.groupby("entry_date")}
    exits = {pd.Timestamp(date): group for date, group in trades.groupby("exit_date")}
    cash = 1.0
    positions: dict[str, dict] = {}
    peak = 1.0
    worst = 0.0
    curve = []
    commission = COMMISSION * cost_scale
    slippage = SLIPPAGE * cost_scale
    for date in dates:
        if date in exits:
            for trade in exits[date].itertuples(index=False):
                pos = positions.pop(trade.code)
                cash += pos["shares"] * float(trade.exit_open) * (1 - slippage) * (1 - commission)
        equity_open = cash
        for code, pos in positions.items():
            raw = raws[code]
            px = float(raw.loc[date, "open"]) if date in raw.index else pos["last"]
            equity_open += pos["shares"] * px
        if date in entries:
            for trade in entries[date].itertuples(index=False):
                invest = min(equity_open / n_etfs, cash)
                shares = invest / (float(trade.entry_open) * (1 + slippage) * (1 + commission))
                cash -= invest
                positions[trade.code] = {"shares": shares, "last": float(trade.entry_open)}
        equity = cash
        for code, pos in list(positions.items()):
            raw = raws[code]
            if date in raw.index:
                px = float(raw.loc[date, "close"])
                pos["last"] = px
            else:
                px = pos["last"]
            equity += pos["shares"] * px
        peak = max(peak, equity)
        drawdown = equity / peak - 1 if peak else 0.0
        worst = min(worst, drawdown)
        curve.append({"date": date, "equity": equity, "drawdown": drawdown})
    return float(worst), pd.DataFrame(curve)


def _span_days(window: str) -> float:
    start, end, _ = WINDOWS[window]
    return float((pd.Timestamp(end) - pd.Timestamp(start)).days)


def score_window(window: str, featured: pd.DataFrame, panels: dict, n_etfs: int, cost_scale: float = 1.0) -> tuple[list[dict], list[pd.DataFrame]]:
    part = featured[featured["window"] == window].copy()
    scores = []
    trades = []
    for config in CONFIGS:
        summary, trade_frame = score_config(config, part, panels, n_etfs, _span_days(window), cost_scale)
        summary["window"] = window
        summary["cost_scale"] = cost_scale
        scores.append(summary)
        if not trade_frame.empty:
            trades.append(trade_frame)
    return scores, trades


def diff_against_raw(scores: list[dict], trades: list[pd.DataFrame], raw_id: str = "kronos_lb120_s5_mean_gt0") -> list[dict]:
    by_id = {frame["config"].iloc[0]: frame for frame in trades if len(frame)}
    raw = by_id.get(raw_id)
    for summary in scores:
        summary["win_gain_vs_raw"] = summary["win_rate"] - next(row["win_rate"] for row in scores if row["id"] == raw_id)
        other = by_id.get(summary["id"])
        if raw is None or other is None or summary["id"] == raw_id:
            summary["diff_block_low"] = float("nan")
            summary["diff_block_high"] = float("nan")
            continue
        flags_a = (other["net"] > 0).to_numpy(dtype=float)
        flags_b = (raw["net"] > 0).to_numpy(dtype=float)
        low, high = block_bootstrap_diff(
            flags_a,
            _block_ids(other["signal_date"], 5).to_numpy(),
            flags_b,
            _block_ids(raw["signal_date"], 5).to_numpy(),
            BOOTSTRAP_REPS,
            SEED,
        )
        summary["diff_block_low"] = low
        summary["diff_block_high"] = high
    return scores


def slice_stability(trades: pd.DataFrame, slots: pd.DataFrame, config_id: str) -> list[dict]:
    rows = []
    if trades.empty:
        return rows
    part = trades[trades["config"] == config_id]
    for code, group in part.groupby("code"):
        slot_n = int(((slots["code"] == code) & slots["executable"] & slots["window"].isin(part["window"].unique())).sum())
        summary = summarize_trades(group, slot_n, 1, _span_days("revalidation"))
        summary["slice"] = code
        rows.append(summary)
    for year in (2025, 2026):
        group = part[pd.to_datetime(part["signal_date"]).dt.year == year]
        slot_n = int(((slots["window"] == "revalidation") & slots["executable"] & (pd.to_datetime(slots["signal_date"]).dt.year == year)).sum())
        summary = summarize_trades(group, slot_n, part["code"].nunique(), 365.25 if year == 2025 else 320)
        summary["slice"] = str(year)
        rows.append(summary)
    return rows


def finetune_gate(revalidation_row: dict | None, per_etf: list[dict]) -> bool:
    if revalidation_row is None:
        return False
    if not (
        math.isfinite(revalidation_row.get("diff_block_low", float("nan")))
        and revalidation_row["diff_block_low"] > 0
        and revalidation_row["profit_factor"] >= MIN_PROFIT_FACTOR
        and revalidation_row["payoff"] >= MIN_PAYOFF
        and revalidation_row["coverage"] >= MIN_COVERAGE
    ):
        return False
    etf_rows = [row for row in per_etf if row["slice"] not in {"2025", "2026"}]
    return bool(etf_rows) and all(row["n_trades"] >= REVALIDATION_MIN_TRADES_PER_ETF for row in etf_rows)


def make_plots(selection: pd.DataFrame, revalidation: pd.DataFrame, curves: dict[str, pd.DataFrame], fig_dir: Path) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ax.scatter(selection["coverage"], selection["win_rate"], s=36)
    for row in selection.itertuples(index=False):
        if row.kind == "baseline" and row.id == "no_trade":
            continue
        ax.annotate(row.id.replace("kronos_", ""), (row.coverage, row.win_rate), fontsize=6, alpha=0.8)
    ax.axhline(0.5, color="black", lw=0.6)
    ax.set_xlabel("Coverage of executable slots")
    ax.set_ylabel("Post-cost win rate")
    ax.set_title("Selection window only: win rate versus coverage")
    fig.tight_layout()
    fig.savefig(fig_dir / "selection_winrate_vs_coverage.png", dpi=120)
    plt.close(fig)

    compare = revalidation[(revalidation["id"] != "no_trade") & (revalidation["n_trades"] > 0)].copy()
    compare["block_low"] = compare["block_low"].fillna(compare["win_rate"])
    compare["block_high"] = compare["block_high"].fillna(compare["win_rate"])
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    y = np.arange(len(compare))
    ax.barh(
        y,
        compare["win_rate"],
        xerr=[(compare["win_rate"] - compare["block_low"]).clip(lower=0), (compare["block_high"] - compare["win_rate"]).clip(lower=0)],
        capsize=2,
    )
    ax.set_yticks(y, compare["id"], fontsize=7)
    ax.axvline(0.5, color="black", lw=0.6)
    ax.set_xlabel("Post-cost win rate, block-bootstrap 95% interval")
    ax.set_title("Exploratory 2025-2026 revalidation, not a fresh holdout")
    fig.tight_layout()
    fig.savefig(fig_dir / "revalidation_winrate.png", dpi=120)
    plt.close(fig)

    if curves:
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        for name, curve in curves.items():
            if curve.empty:
                continue
            ax.plot(curve["date"], curve["equity"], label=name, lw=1.2)
        ax.legend(fontsize=8)
        ax.set_title("Exploratory revalidation equity, same 1/N sleeves")
        ax.set_ylabel("Equity")
        fig.tight_layout()
        fig.savefig(fig_dir / "revalidation_equity.png", dpi=120)
        plt.close(fig)


def write_report(path: Path, payload: dict) -> None:
    lock = payload["lock"]
    selected = lock["selected_id"]
    gain_pp = lock["selection_gain"] * 100 if math.isfinite(lock["selection_gain"]) else float("nan")
    lines = [
        "# 宽基 ETF 三日决策层改造",
        "",
        f"报告时区 Asia/Shanghai。数据截至 2026-11-13，方案冻结于 2026-11-15。",
        "",
        f"选择结果：`{selected}`。选择窗胜率差 {gain_pp:.2f} 个百分点。" if selected else "选择结果：没有改造候选通过资格线。",
        "",
        "预训练权重没有更新。这次只改变是否根据既有采样做多。不能称为重训成功。",
        "",
        "2025–2026 是探索性历史再验证。上一轮已经看过包含 510300 和 159915 的同期成绩，这里不是完全未见的独立样本外。",
        "",
        "## 选择窗（2024-07-01 至 2024-10-31）",
        "",
        _table(payload["selection"]),
        "",
        f"资格线：成交不少于 30 且亏损不少于 5，覆盖率不少于 15%，profit factor 不少于 1，盈亏比不少于 0.75。实质提升门槛是比原模型无过滤高 {MATERIAL_WIN_GAIN * 100:.0f} 个百分点。",
        "",
        "## 确认窗",
        "",
        "确认窗只核对，不改已经锁定的选择。这个窗口事先只有 28 个可成交信号格，大约五周，年化频率只是按日历外推。区块区间收成一个点时，表示块数太少，不能据此改选择。",
        "",
        _table(payload["confirmation"]),
        "",
        "## 探索性再验证（2025-01-15 至 2026-11-06）",
        "",
        _table(payload["revalidation"]),
        "",
        "## 费用加倍",
        "",
        _table(payload["stress"]),
        "",
        "## 分 ETF 与分年",
        "",
        _slice_table(payload["stability"]),
        "",
        "## 组合回撤",
        "",
        "回撤来自同一资金规则：初始权益 1，每只 ETF 开仓使用当时权益的 1/N，同时最多持有 N 笔。下面的最大回撤是这条权益曲线的数字。",
        "",
        _dd_table(payload["revalidation"]),
        "",
        "## 结论",
        "",
        payload["verdict"],
        "",
        f"下一步微调：{'证据达到事先门槛，可以另开一次小规模微调实验；本次没有做。' if payload['finetune'] else '证据不足，不做权重微调。'}",
        "",
        "## 运行",
        "",
        f"- 权重 `{payload['model']}` `{payload['revision']}`，参数量 {payload['n_params']}，设备 CPU。",
        f"- 通过流动性的证券：{', '.join(payload['pool'])}。",
        f"- 预测墙钟 {payload['forecast_seconds']} 秒。",
        "- 往返成本门槛 0.15%，ETF 无印花税。买入用下一开盘，卖出用三日后开盘。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _fmt(value, digits=3) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "NA"
    return f"{value:.{digits}f}"


def _pct(value) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "NA"
    return f"{100 * value:.1f}%"


def _table(rows: list[dict]) -> str:
    header = "| 方案 | 笔数 | 胜率 | 区块区间 | 覆盖率 | 年化每ETF | 平均净收益 | 盈亏比 | PF | 方向准确率 | 回撤 |"
    sep = "| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    body = []
    for row in rows:
        body.append(
            "| {id} | {n} | {win} | {lo}–{hi} | {cov} | {freq} | {net} | {pay} | {pf} | {dire} | {dd} |".format(
                id=row["id"],
                n=row["n_trades"],
                win=_pct(row["win_rate"]),
                lo=_pct(row["block_low"]),
                hi=_pct(row["block_high"]),
                cov=_pct(row["coverage"]),
                freq=_fmt(row["trades_per_etf_year"], 1),
                net=_pct(row["mean_net"]),
                pay=_fmt(row["payoff"], 2),
                pf=_fmt(row["profit_factor"], 2),
                dire=_pct(row["direction_accuracy"]),
                dd=_pct(row["max_drawdown"]),
            )
        )
    return "\n".join([header, sep, *body])


def _slice_table(rows: list[dict]) -> str:
    if not rows:
        return "无。"
    lines = ["| 切片 | 笔数 | 胜率 | 覆盖率 | PF | 平均净收益 |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in rows:
        lines.append(
            f"| {row['slice']} | {row['n_trades']} | {_pct(row['win_rate'])} | {_pct(row['coverage'])} | {_fmt(row['profit_factor'], 2)} | {_pct(row['mean_net'])} |"
        )
    return "\n".join(lines)


def _dd_table(rows: list[dict]) -> str:
    lines = ["| 方案 | 最大回撤 |", "| --- | ---: |"]
    for row in rows:
        lines.append(f"| {row['id']} | {_pct(row['max_drawdown'])} |")
    return "\n".join(lines)


def verdict_text(lock: dict, revalidation: list[dict], stress: list[dict]) -> str:
    selected = lock["selected_id"]
    if selected is None:
        return "改造候选没有同时满足笔数、覆盖率、profit factor 和盈亏比。不能把高胜率、低覆盖或亏损结构当成有效改造。原模型无过滤、动量、历史多数和不交易的对照仍列在表里。"
    gain_pp = lock["selection_gain"] * 100
    rev = next(row for row in revalidation if row["id"] == selected)
    raw = next(row for row in revalidation if row["id"] == "kronos_lb120_s5_mean_gt0")
    rev_pp = (rev["win_rate"] - raw["win_rate"]) * 100 if math.isfinite(rev["win_rate"]) and math.isfinite(raw["win_rate"]) else float("nan")
    stress_row = next(row for row in stress if row["id"] == selected)
    parts = [
        f"锁定的改造是 `{selected}`。选择窗扣费后胜率比原模型无过滤高 {gain_pp:.1f} 个百分点。"
        if lock["material_gain"]
        else f"锁定观察的最高资格候选是 `{selected}`，选择窗只高 {gain_pp:.1f} 个百分点，没有达到 5 个百分点的实质提升。"
    ]
    parts.append(
        f"探索性再验证里，该方案胜率 {_pct(rev['win_rate'])}，原模型 {_pct(raw['win_rate'])}，差 {rev_pp:.1f} 个百分点；区块区间 {_pct(rev['diff_block_low'])} 至 {_pct(rev['diff_block_high'])}。"
    )
    if not (math.isfinite(rev.get("diff_block_low", float("nan"))) and rev["diff_block_low"] > 0):
        parts.append("再验证的胜率差区间包含 0 或没有足够成交，不能把选择窗的最高分写成可靠优势。")
    if not (math.isfinite(rev["profit_factor"]) and rev["profit_factor"] >= 1 and math.isfinite(rev["payoff"]) and rev["payoff"] >= 0.75):
        parts.append("再验证的盈亏比或 profit factor 没有守住资格线，胜率不能单独算作改进。")
    parts.append(
        f"费用和滑点加倍后，该方案胜率 {_pct(stress_row['win_rate'])}，平均净收益 {_pct(stress_row['mean_net'])}，profit factor {_fmt(stress_row['profit_factor'], 2)}。"
    )
    return "".join(parts)
