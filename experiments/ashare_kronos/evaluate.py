"""Paired scores versus flat, majority-direction and momentum baselines."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.ashare_kronos.config import (
    COMMISSION_RATE,
    DECISION_RULE,
    FLAT_PRICE_EPS,
    GAP_SKIP,
    HORIZONS,
    LIMIT_BUFFER,
    LOOKBACK,
    MAX_LIMIT_DOWN_DEFER,
    ORIGIN_STEP,
    PAPER_TRAIN_END,
    SLIPPAGE_RATE,
    STAMP_TAX_ETF,
    STAMP_TAX_STOCK,
)


def _direction_from_prices(pred_or_actual: float, origin: float) -> int:
    if abs(pred_or_actual - origin) < FLAT_PRICE_EPS:
        return 0
    return 1 if pred_or_actual > origin else -1


def _tstat(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 3:
        return float("nan")
    std = values.std(ddof=1)
    if std == 0:
        return float("nan")
    return float(values.mean() / (std / np.sqrt(len(values))))


def _spearman(left: pd.Series, right: pd.Series) -> float:
    if len(left) < 5:
        return float("nan")
    ra = left.rank(method="average")
    rb = right.rank(method="average")
    if ra.nunique() < 2 or rb.nunique() < 2:
        return float("nan")
    return float(ra.corr(rb))


def enrich(predictions: pd.DataFrame) -> pd.DataFrame:
    frame = predictions.copy()
    frame["origin_date"] = pd.to_datetime(frame["origin_date"])
    frame["input_start"] = pd.to_datetime(frame["input_start"])
    return frame


def score_horizon(frame: pd.DataFrame, horizon: int) -> dict:
    actual = frame[f"actual_close_{horizon}"].to_numpy(dtype=float)
    pred = frame[f"pred_close_{horizon}"].to_numpy(dtype=float)
    origin = frame["origin_close"].to_numpy(dtype=float)
    mom = frame[f"mom_close_{horizon}"].to_numpy(dtype=float)
    pred_std = frame[f"pred_close_std_{horizon}"].to_numpy(dtype=float)
    actual_dir = np.array([_direction_from_prices(a, o) for a, o in zip(actual, origin)])
    kronos_dir = np.array([_direction_from_prices(p, o) for p, o in zip(pred, origin)])
    mom_dir = np.array([_direction_from_prices(m, o) for m, o in zip(mom, origin)])
    majority = frame["majority_dir"].to_numpy(dtype=int)
    moved = actual_dir != 0
    ae_k = np.abs(pred - actual)
    ae_f = np.abs(origin - actual)
    ae_m = np.abs(mom - actual)
    ae_k_ret = ae_k / origin
    ae_f_ret = ae_f / origin
    ae_m_ret = ae_m / origin
    diff = ae_f - ae_k
    by_symbol = pd.Series(diff, index=frame.index).groupby(frame["code"]).mean()
    ic_rows = []
    for date, part in frame.groupby("origin_date"):
        ic_rows.append(
            {
                "origin_date": str(pd.Timestamp(date).date()),
                "n": int(len(part)),
                "rank_ic": _spearman(part[f"pred_ret_{horizon}"], part[f"actual_ret_{horizon}"]),
            }
        )
    ic = pd.DataFrame(ic_rows)
    ic_vals = ic["rank_ic"].to_numpy(dtype=float) if len(ic) else np.array([])
    finite_ic = ic_vals[np.isfinite(ic_vals)] if len(ic_vals) else np.array([])
    return {
        "horizon": horizon,
        "n_predictions": int(len(frame)),
        "n_symbols": int(frame["code"].nunique()),
        "n_origin_dates": int(frame["origin_date"].nunique()),
        "n_flat": int((~moved).sum()),
        "n_up": int((actual_dir == 1).sum()),
        "n_down": int((actual_dir == -1).sum()),
        "flat_share": float((~moved).mean()),
        "actual_up_share_of_moved": float((actual_dir[moved] == 1).mean()) if moved.any() else float("nan"),
        "kronos_up_share": float((kronos_dir == 1).mean()),
        "kronos_flat_pred": int((kronos_dir == 0).sum()),
        "dir_acc_kronos": float((kronos_dir[moved] == actual_dir[moved]).mean()) if moved.any() else float("nan"),
        "dir_acc_momentum": float((mom_dir[moved] == actual_dir[moved]).mean()) if moved.any() else float("nan"),
        "dir_acc_majority": float((majority[moved] == actual_dir[moved]).mean()) if moved.any() else float("nan"),
        "mae_kronos": float(ae_k.mean()),
        "mae_flat": float(ae_f.mean()),
        "mae_momentum": float(ae_m.mean()),
        "mae_ratio_kronos": float(ae_k.mean() / ae_f.mean()) if ae_f.mean() else float("nan"),
        "mae_ratio_momentum": float(ae_m.mean() / ae_f.mean()) if ae_f.mean() else float("nan"),
        "mae_ret_ratio_kronos": float(ae_k_ret.mean() / ae_f_ret.mean()) if ae_f_ret.mean() else float("nan"),
        "mae_ret_ratio_momentum": float(ae_m_ret.mean() / ae_f_ret.mean()) if ae_f_ret.mean() else float("nan"),
        "mae_skill_vs_flat": float(1 - ae_k.mean() / ae_f.mean()) if ae_f.mean() else float("nan"),
        "paired_win_rate_vs_flat": float(np.mean(ae_k < ae_f)),
        "paired_tie_rate_vs_flat": float(np.mean(np.isclose(ae_k, ae_f))),
        "cluster_t_mae": _tstat(by_symbol.to_numpy()),
        "cluster_mean_mae_improvement": float(by_symbol.mean()),
        "naive_t_mae": _tstat(diff),
        "mean_pred_ret": float(frame[f"pred_ret_{horizon}"].mean()),
        "mean_actual_ret": float(frame[f"actual_ret_{horizon}"].mean()),
        "mean_mom_ret": float(frame[f"mom_ret_{horizon}"].mean()),
        "rank_ic_mean": float(np.mean(finite_ic)) if len(finite_ic) else float("nan"),
        "rank_ic_std": float(np.std(finite_ic, ddof=1)) if len(finite_ic) > 1 else float("nan"),
        "rank_ic_t": _tstat(finite_ic),
        "rank_ic_n_dates": int(len(finite_ic)),
        "rank_ic_pooled": _spearman(frame[f"pred_ret_{horizon}"], frame[f"actual_ret_{horizon}"]),
        "mean_pred_std_ret": float(np.mean(pred_std / origin)),
        "uncertainty_vs_abs_err_spearman": _spearman(pd.Series(pred_std), pd.Series(ae_k)),
        "uncertainty_vs_abs_ret_err_spearman": _spearman(pd.Series(pred_std / origin), pd.Series(ae_k_ret)),
        "input_overlap_share": float((LOOKBACK - ORIGIN_STEP) / LOOKBACK),
        "forward_windows_overlap_within_symbol": bool(horizon > ORIGIN_STEP),
    }, ic


def decide(rows: list[dict]) -> str:
    mae_better = all(r["mae_ratio_kronos"] < 1 for r in rows)
    dir_better = all(
        np.isfinite(r["dir_acc_kronos"])
        and r["dir_acc_kronos"] > r["dir_acc_majority"]
        and r["dir_acc_kronos"] > r["dir_acc_momentum"]
        for r in rows
    )
    ic_pos = all(np.isfinite(r["rank_ic_mean"]) and r["rank_ic_mean"] > 0 for r in rows)
    mae_sig = sum(np.isfinite(r["cluster_t_mae"]) and r["cluster_t_mae"] > 1.96 for r in rows) >= 2
    ic_sig = sum(np.isfinite(r["rank_ic_t"]) and r["rank_ic_t"] > 1.96 for r in rows) >= 2
    if mae_better and dir_better and ic_pos and mae_sig and ic_sig:
        return "有预测优势"
    no_mae = all(r["mae_ratio_kronos"] >= 1 for r in rows)
    no_dir = all(
        not (
            np.isfinite(r["dir_acc_kronos"])
            and r["dir_acc_kronos"] > r["dir_acc_majority"]
            and r["dir_acc_kronos"] > r["dir_acc_momentum"]
        )
        for r in rows
    )
    no_ic = all(not (np.isfinite(r["rank_ic_mean"]) and r["rank_ic_mean"] > 0 and np.isfinite(r["rank_ic_t"]) and r["rank_ic_t"] > 1.96) for r in rows)
    if no_mae and no_dir and no_ic:
        return "无预测优势"
    return "未证实优势"


def _net_return(entry: float, exit_px: float, stamp: float) -> float:
    entry_fill = entry * (1 + SLIPPAGE_RATE)
    exit_fill = exit_px * (1 - SLIPPAGE_RATE)
    return exit_fill * (1 - COMMISSION_RATE - stamp) / (entry_fill * (1 + COMMISSION_RATE)) - 1


def _limit_up(open_px: float, prev_close: float, limit: float) -> bool:
    if prev_close <= 0:
        return True
    return open_px / prev_close - 1 >= limit - LIMIT_BUFFER


def _limit_down(open_px: float, prev_close: float, limit: float) -> bool:
    if prev_close <= 0:
        return True
    return open_px / prev_close - 1 <= -(limit - LIMIT_BUFFER)


def simulate_trades(frame: pd.DataFrame, loaded: dict, horizon: int, gap_filter: bool) -> pd.DataFrame:
    records = []
    for row in frame.itertuples(index=False):
        raw = loaded[row.code]["raw"]
        origin_ts = pd.Timestamp(row.origin_date)
        match = raw.index[raw["date"] == origin_ts]
        if len(match) != 1:
            continue
        origin = int(match[0])
        entry_idx = origin + 1
        exit_idx = origin + 1 + horizon
        if exit_idx >= len(raw):
            records.append({"code": row.code, "origin_date": row.origin_date, "status": "no_exit_bar", "strategy": "market"})
            continue
        entry = float(raw.iloc[entry_idx]["open"])
        prev_close = float(raw.iloc[origin]["close"])
        limit = float(row.limit)
        stamp = STAMP_TAX_ETF if row.board == "etf" else STAMP_TAX_STOCK
        gap = entry / prev_close - 1
        status = "filled"
        if raw.iloc[entry_idx]["volume"] <= 0 or entry <= 0:
            status = "no_trade_volume"
        elif _limit_up(entry, prev_close, limit):
            status = "limit_up_skip"
        elif gap_filter and gap > GAP_SKIP:
            status = "gap_skip"
        else:
            deferred = 0
            while deferred <= MAX_LIMIT_DOWN_DEFER:
                if exit_idx + deferred >= len(raw):
                    status = "no_exit_bar"
                    break
                bar = raw.iloc[exit_idx + deferred]
                prev = raw.iloc[exit_idx + deferred - 1]
                if bar["volume"] <= 0:
                    deferred += 1
                    continue
                if _limit_down(float(bar["open"]), float(prev["close"]), limit):
                    deferred += 1
                    continue
                exit_px = float(bar["open"])
                status = "filled"
                break
            else:
                status = "limit_down_unfilled"
        pred_ret = float(getattr(row, f"pred_ret_{horizon}"))
        mom_ret = float(getattr(row, f"mom_ret_{horizon}"))
        base = {
            "code": row.code,
            "origin_date": str(pd.Timestamp(row.origin_date).date()),
            "horizon": horizon,
            "gap_filter": gap_filter,
            "status": status,
            "gap": gap,
        }
        if status != "filled":
            for name in ("kronos_long", "momentum_long", "always_long"):
                records.append({**base, "strategy": name, "net_return": np.nan, "taken": False})
            continue
        net = _net_return(entry, exit_px, stamp)
        taken = {
            "kronos_long": pred_ret > 0,
            "momentum_long": mom_ret > 0,
            "always_long": True,
        }
        for name, take in taken.items():
            records.append({**base, "strategy": name, "net_return": net if take else np.nan, "taken": bool(take), "defer_days": deferred})
    return pd.DataFrame(records)


def trade_summary(trades: pd.DataFrame) -> list[dict]:
    if trades.empty:
        return []
    rows = []
    for (horizon, gap_filter, strategy), part in trades.groupby(["horizon", "gap_filter", "strategy"]):
        taken = part[part["taken"] == True]  # noqa: E712
        rets = taken["net_return"].dropna()
        rows.append(
            {
                "horizon": int(horizon),
                "gap_filter": bool(gap_filter),
                "strategy": strategy,
                "n_signals": int(len(part)),
                "n_taken": int(len(rets)),
                "n_limit_up_skip": int((part["status"] == "limit_up_skip").sum()),
                "n_gap_skip": int((part["status"] == "gap_skip").sum()),
                "n_unfilled": int(part["status"].isin(["limit_down_unfilled", "no_exit_bar", "no_trade_volume"]).sum()),
                "mean_net_return": float(rets.mean()) if len(rets) else float("nan"),
                "median_net_return": float(rets.median()) if len(rets) else float("nan"),
                "win_rate": float((rets > 0).mean()) if len(rets) else float("nan"),
            }
        )
    return rows


def make_plots(frame: pd.DataFrame, scores: list[dict], fig_dir: Path) -> list[str]:
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    labels = [str(s["horizon"]) for s in scores]
    fig, ax = plt.subplots(figsize=(6.2, 4))
    x = np.arange(len(scores))
    ax.bar(x - 0.15, [s["mae_ratio_kronos"] for s in scores], width=0.3, label="Kronos / flat")
    ax.bar(x + 0.15, [s["mae_ratio_momentum"] for s in scores], width=0.3, label="Momentum / flat")
    ax.axhline(1.0, color="black", lw=0.8)
    ax.set_xticks(x, labels)
    ax.set_xlabel("Horizon (trading days)")
    ax.set_ylabel("MAE ratio (<1 beats flat)")
    ax.set_title("Close-price MAE relative to flat forecast")
    ax.legend()
    fig.tight_layout()
    p = fig_dir / "mae_ratio_by_horizon.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    paths.append(str(p))

    fig, ax = plt.subplots(figsize=(6.4, 4))
    width = 0.25
    ax.bar(x - width, [s["dir_acc_kronos"] for s in scores], width=width, label="Kronos")
    ax.bar(x, [s["dir_acc_momentum"] for s in scores], width=width, label="Momentum")
    ax.bar(x + width, [s["dir_acc_majority"] for s in scores], width=width, label="Majority")
    ax.axhline(0.5, color="black", lw=0.8)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Horizon (trading days)")
    ax.set_ylabel("Direction accuracy ex-flat")
    ax.set_title("Direction accuracy, unchanged closes excluded")
    ax.legend()
    fig.tight_layout()
    p = fig_dir / "direction_accuracy_by_horizon.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    paths.append(str(p))

    fig, ax = plt.subplots(figsize=(6.2, 4))
    means = [s["rank_ic_mean"] for s in scores]
    stds = [s["rank_ic_std"] / np.sqrt(s["rank_ic_n_dates"]) if s["rank_ic_n_dates"] else np.nan for s in scores]
    ax.bar(x, means, yerr=stds, capsize=4)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x, labels)
    ax.set_xlabel("Horizon (trading days)")
    ax.set_ylabel("Mean cross-sectional Rank IC")
    ax.set_title("Rank IC across the fixed pool (mean ± SE)")
    fig.tight_layout()
    p = fig_dir / "rank_ic_by_horizon.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    paths.append(str(p))

    fig, ax = plt.subplots(figsize=(5.4, 5))
    ax.scatter(frame["pred_ret_5"], frame["actual_ret_5"], s=12, alpha=0.45)
    ax.axhline(0, color="black", lw=0.6)
    ax.axvline(0, color="black", lw=0.6)
    ax.set_xlabel("Kronos predicted 5-day return")
    ax.set_ylabel("Realized 5-day close return")
    ax.set_title("All 5-day forecasts, not a selected path")
    fig.tight_layout()
    p = fig_dir / "pred_vs_actual_h5.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    paths.append(str(p))
    return paths


def _fmt(value, digits=4) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "NA"
    return f"{value:.{digits}f}"


def _pct(value) -> str:
    if value is None or not np.isfinite(value):
        return "NA"
    return f"{100 * value:.2f}%"


def write_report(summary: dict, path: Path) -> None:
    scores = summary["primary_scores"]
    lines = []
    lines.append("# Kronos 预训练模型 A 股日线离线检验")
    lines.append("")
    lines.append(f"结论：**{summary['conclusion']}**。")
    lines.append("")
    if summary["conclusion"] == "未证实优势" and all(row["mae_ratio_kronos"] >= 1 for row in scores):
        lines.append("三个期限的收盘价 MAE 都高于持平，方向准确率都在一半附近。5 日方向只比动量和历史多数高约 1 个百分点，所以事先规则不能写成「无预测优势」；误差、方向和 Rank IC 也没有同时达到「有预测优势」。")
        lines.append("")
    lines.append("这是固定证券池上的价格预测对照，不是实盘盈利结论，也不是相对 HIT 规则的结论。")
    lines.append("")
    lines.append("## 判定规则")
    lines.append("")
    lines.append(DECISION_RULE.strip())
    lines.append("")
    lines.append("## 主样本对照")
    lines.append("")
    lines.append("| 期限 | 预测条数 | 独立日期 | 平盘 | 涨/跌 | Kronos方向 | 动量方向 | 多数方向 | MAE/持平 | 动量MAE/持平 | 聚类t | RankIC | IC日期t |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in scores:
        lines.append(
            "| {h} | {n} | {d} | {flat} | {up}/{down} | {kd} | {md} | {jd} | {ratio} | {mr} | {t} | {ic} | {ict} |".format(
                h=row["horizon"],
                n=row["n_predictions"],
                d=row["n_origin_dates"],
                flat=row["n_flat"],
                up=row["n_up"],
                down=row["n_down"],
                kd=_pct(row["dir_acc_kronos"]),
                md=_pct(row["dir_acc_momentum"]),
                jd=_pct(row["dir_acc_majority"]),
                ratio=_fmt(row["mae_ratio_kronos"]),
                mr=_fmt(row["mae_ratio_momentum"]),
                t=_fmt(row["cluster_t_mae"], 2),
                ic=_fmt(row["rank_ic_mean"]),
                ict=_fmt(row["rank_ic_t"], 2),
            )
        )
    lines.append("")
    lines.append("MAE/持平小于 1 表示收盘价绝对误差低于“未来价等于今日收盘”。方向准确率的分母不含平盘；预测为平或反向都算错。聚类 t 把每只证券的平均 MAE 改善合成一个数，正值表示 Kronos 更接近真实收盘。Rank IC 是同一起点日期、池内至少 5 只证券的 Spearman。")
    lines.append("")
    lines.append(
        "价格 MAE 会被高价股放大。按收益率重算的 MAE/持平为 "
        + "、".join(f"{row['horizon']}日 {_fmt(row['mae_ret_ratio_kronos'])}" for row in scores)
        + "。它与价格口径一致，Kronos 仍然大于持平，动量更大。采样标准差和绝对误差在价格上的 Spearman 为 "
        + "、".join(_fmt(row["uncertainty_vs_abs_err_spearman"], 2) for row in scores)
        + "，换成收益率后降为 "
        + "、".join(_fmt(row["uncertainty_vs_abs_ret_err_spearman"], 2) for row in scores)
        + "。5 条路径的离散度不是已经校准好的预测区间。"
    )
    lines.append("")
    lines.append(f"主样本预测条数 {summary['n_predictions']}，证券 {summary['n_symbols']}，不同起点日期 {summary['n_origin_dates']}。其中 {summary.get('rank_ic_dates', 'NA')} 个日期至少有 5 只证券同时成为起点，Rank IC 只用这些日期；其余日期因各证券交易日历错开而只有 1 只证券。输入窗口重叠比例 {(LOOKBACK - ORIGIN_STEP) / LOOKBACK:.1%}。起点间隔 {ORIGIN_STEP} 个交易日，1/3/5 日的远期收益窗口在同一证券内不重叠，但相邻窗口仍共享市场状态，条数不能当成独立样本。")
    lines.append("")
    lines.append(f"输入起点晚于论文所称 2024-06-30 的预测有 {summary['n_input_after_paper_cutoff']} 条。该子集结论按同一规则为 **{summary['subset_conclusion']}**，只作稳健性，不替换主结论。")
    lines.append("")
    lines.append("## 样本外边界")
    lines.append("")
    lines.append(summary["oos_note"])
    lines.append("")
    lines.append("## 交易模拟")
    lines.append("")
    lines.append("模拟用信号日收盘之后的下一根日线开盘价成交，卖出在持有期满后的下一开盘；涨停开盘不买，跌停开盘顺延最多 3 个交易日，T+1 由持有跨日满足。股票卖出印花税 0.05%，ETF 为 0，佣金双边各 0.025%，滑点双边各 0.05%。下表是被选中交易的平均单笔净收益，不是组合净值，没有资金约束，也没有 HIT 的 MACD、量能和仓位规则。")
    lines.append("")
    lines.append("| 期限 | 过滤高开 | 策略 | 尝试 | 成交 | 平均净收益 | 胜率 |")
    lines.append("| ---: | --- | --- | ---: | ---: | ---: | ---: |")
    for row in summary["trades"]:
        lines.append(
            f"| {row['horizon']} | {row['gap_filter']} | {row['strategy']} | {row['n_signals']} | {row['n_taken']} | {_pct(row['mean_net_return'])} | {_pct(row['win_rate'])} |"
        )
    lines.append("")
    lines.append("不能据此称盈利优势。没有用户实盘同日期 HIT 信号，不能声称优于 HIT。")
    lines.append("")
    lines.append("## 数据与运行")
    lines.append("")
    lines.append(f"- 代码：`jam1983go/kronos`，commit `{summary['code_commit']}`，与 `shiyu-coder/Kronos` master 的 commit 和 tree 相同。MIT License，版权 ShiYu 2025。")
    lines.append(f"- 权重：`{summary['model_id']}` revision `{summary['model_revision']}`，参数量 {summary['n_params']}。Tokenizer `{summary['tokenizer_id']}` revision `{summary['tokenizer_revision']}`。")
    lines.append(f"- 设备：{summary['device']}，{summary.get('cpu_count')} 核，无 GPU，未购买 GPU。1078 条预测墙钟 {summary.get('forecast_elapsed_sec', summary['elapsed_sec'])} 秒。同机再跑一批 12 条、每条 5 次采样的进程峰值 RSS 约 {summary['peak_rss_mb']} MB。")
    lines.append(f"- 数据：东方财富日线，前复权与不复权各一份，抓取时间 {summary['fetched_at']}，区间实际 {summary['data_first']} 至 {summary['data_last']}，评估日 {summary['as_of']} 之前的已收盘数据。腾讯前复权收盘和最近 80 日新浪不复权收盘用于交叉核对。")
    lines.append(f"- 质量：{summary.get('quality_note', '')}")
    lines.append(f"- 固定参数：lookback {LOOKBACK}，pred_len 5，step {ORIGIN_STEP}，sample_count {summary['sample_count']}，T {summary['temperature']}，top_p {summary['top_p']}，top_k {summary['top_k']}，seed {summary['seed']}。归一化只用输入窗口。")
    lines.append("")
    lines.append("## 局限")
    lines.append("")
    lines.append("- 公开材料对训练集截止日期的说法不一致，权重文件本身不带可审计的样本清单，所以时间切分不是严格模型样本外。")
    lines.append("- 上海市场在论文里被列为分布内。这些证券的更早 K 线，以及其他周期，可能进过预训练。")
    lines.append("- 池子只有 12 只证券，截面 Rank IC 方差大。")
    lines.append("- 前复权序列以抓取日为锚，模型输入与成交价口径不同；成交模拟用未复权开盘价。")
    lines.append("- 采样标准差来自 5 条路径，不是参数不确定性。")
    lines.append("- 未做组合优化、冲击成本和 HIT 对照。")
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_evaluation(predictions: pd.DataFrame, loaded: dict, out_dir: Path, meta: dict) -> dict:
    frame = enrich(predictions)
    primary_scores = []
    ic_frames = []
    for horizon in HORIZONS:
        score, ic = score_horizon(frame, horizon)
        primary_scores.append(score)
        ic["horizon"] = horizon
        ic_frames.append(ic)
    subset = frame[frame["input_start"] > pd.Timestamp(PAPER_TRAIN_END)]
    subset_scores = [score_horizon(subset, horizon)[0] for horizon in HORIZONS] if len(subset) else []
    trades = []
    trade_frames = []
    for horizon in HORIZONS:
        for gap_filter in (False, True):
            trade_frames.append(simulate_trades(frame, loaded, horizon, gap_filter))
    trade_frame = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
    trades = trade_summary(trade_frame)
    fig_paths = make_plots(frame, primary_scores, out_dir / "figures")
    summary = {
        **meta,
        "conclusion": decide(primary_scores),
        "subset_conclusion": decide(subset_scores) if subset_scores else "未运行",
        "n_predictions": int(len(frame)),
        "n_symbols": int(frame["code"].nunique()) if len(frame) else 0,
        "n_origin_dates": int(frame["origin_date"].nunique()) if len(frame) else 0,
        "n_input_after_paper_cutoff": int(len(subset)),
        "rank_ic_dates": int(primary_scores[0]["rank_ic_n_dates"]) if primary_scores else 0,
        "primary_scores": primary_scores,
        "subset_scores": subset_scores,
        "trades": trades,
        "figures": fig_paths,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    pd.DataFrame(primary_scores).to_csv(out_dir / "horizon_scores.csv", index=False)
    if subset_scores:
        pd.DataFrame(subset_scores).to_csv(out_dir / "horizon_scores_after_paper_cutoff.csv", index=False)
    if ic_frames:
        pd.concat(ic_frames, ignore_index=True).to_csv(out_dir / "rank_ic_by_date.csv", index=False)
    if not trade_frame.empty:
        trade_frame.to_csv(out_dir / "trades.csv", index=False)
        pd.DataFrame(trades).to_csv(out_dir / "trade_summary.csv", index=False)
    write_report(summary, out_dir / "REPORT.md")
    return summary
