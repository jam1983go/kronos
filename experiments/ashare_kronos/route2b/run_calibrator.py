"""Walk-forward calibration test. Label counts may expand the sample; scores may not."""

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
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from experiments.ashare_kronos.data_io import fetch_eastmoney
from experiments.ashare_kronos.routes3.engine import apply_portfolio, equity_curve, excluded_name, regime_of
from experiments.ashare_kronos.route2b.settings import (
    AS_OF,
    BOOT_BLOCK,
    BOOT_REPS,
    CANDIDATES,
    EXTENDED_SIGNAL,
    FEATURES,
    FETCH_BEG,
    FETCH_END,
    FIRST_SIGNAL,
    FOLDS,
    LIQ_END,
    LIQ_START,
    LOOKBACK,
    MIN_CLASS,
    MIN_FOLD_CLASS,
    MIN_FOLD_TRAIN,
    MIN_MEDIAN_AMOUNT,
    MIN_TRAIN,
    POOL_A,
    POOL_B,
    REVAL_TRAIN_BEFORE,
    SAMPLES,
    SEED,
)
from experiments.ashare_kronos.route2b.signals import build_signals
from experiments.ashare_kronos.winrate.infer import load_predictor, predict_slot

DATA = Path(__file__).resolve().parent / "data"
OUT = Path(__file__).resolve().parent / "outputs"
TZ = ZoneInfo("Asia/Shanghai")


def log(message: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    line = f"[{datetime.now(TZ).strftime('%Y-%m-%d %H:%M:%S%z')}] {message}"
    print(line, flush=True)
    with (OUT / "run_log.txt").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def load_pair(spec: dict) -> tuple[str, pd.DataFrame, pd.DataFrame]:
    code, market = spec["code"], spec["market"]
    paths = {kind: DATA / f"{code}_{kind}.csv" for kind in ("qfq", "raw")}
    name_path = DATA / f"{code}_name.txt"
    if not paths["qfq"].exists() or not paths["raw"].exists():
        name, qfq = fetch_eastmoney(code, market, 1, FETCH_BEG, FETCH_END)
        _, raw = fetch_eastmoney(code, market, 0, FETCH_BEG, FETCH_END)
        qfq.to_csv(paths["qfq"], index=False)
        raw.to_csv(paths["raw"], index=False)
        name_path.write_text(name + "\n", encoding="utf-8")
    name = name_path.read_text(encoding="utf-8").strip() if name_path.exists() else code
    qfq = pd.read_csv(paths["qfq"], parse_dates=["date"])
    raw = pd.read_csv(paths["raw"], parse_dates=["date"])
    qfq = qfq[qfq["date"] < AS_OF].sort_values("date").reset_index(drop=True)
    raw = raw[raw["date"] < AS_OF].sort_values("date").reset_index(drop=True)
    return name, qfq, raw


def load_index() -> pd.DataFrame:
    path = DATA / "index_sh_qfq.csv"
    if not path.exists():
        name, frame = fetch_eastmoney("000001", "SH", 1, FETCH_BEG, FETCH_END)
        frame.to_csv(path, index=False)
        (DATA / "index_sh_name.txt").write_text(name + "\n", encoding="utf-8")
    frame = pd.read_csv(path, parse_dates=["date"])
    return frame[frame["date"] < AS_OF].sort_values("date").reset_index(drop=True)


def liquid_codes(pool: tuple[dict, ...]) -> list[dict]:
    kept = []
    rows = []
    for spec in pool:
        name, qfq, _raw = load_pair(spec)
        reason = excluded_name(name, spec["code"])
        hist = qfq[(qfq["date"] >= LIQ_START) & (qfq["date"] <= LIQ_END)]
        median_amount = float(hist["amount"].median()) if len(hist) else 0.0
        passed = reason == "" and median_amount >= MIN_MEDIAN_AMOUNT and len(hist) >= 200
        rows.append({"code": spec["code"], "name": name, "median_amount": median_amount, "n": int(len(hist)), "pass": passed, "reason": reason})
        if passed:
            kept.append({**spec, "name": name})
        log(f"liq {spec['code']} {name} {median_amount:.0f} pass {passed} {reason}")
    pd.DataFrame(rows).to_csv(OUT / "liquidity.csv", index=False)
    return kept


def build_all(pool: list[dict], regimes: pd.DataFrame) -> pd.DataFrame:
    frames = []
    panels = {}
    raws = {}
    for spec in pool:
        name, qfq, raw = load_pair(spec)
        panels[spec["code"]] = qfq
        raws[spec["code"]] = raw
        frame = build_signals(spec["code"], name, qfq, raw, regimes)
        log(f"signals {spec['code']} {name} {len(frame)}")
        if len(frame):
            frames.append(frame)
    signals = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return signals, panels, raws


def class_count(frame: pd.DataFrame) -> dict:
    if frame.empty:
        return {"n": 0, "wins": 0, "losses": 0}
    wins = int(frame["win"].sum())
    return {"n": int(len(frame)), "wins": wins, "losses": int(len(frame) - wins)}


def enough(count: dict, minimum: int, each: int) -> bool:
    return count["n"] >= minimum and count["wins"] >= each and count["losses"] >= each


def choose_sample(signals, pool_a, panels, raws, regimes):
    """Expand only by the frozen rule. Returns signals, pool, panels, raws, decision."""
    start = FIRST_SIGNAL
    pool = pool_a
    decision = {"start": start, "batch": "A", "passed": False}
    for step in ("A-2018", "A-2016", "B-2016"):
        if step == "A-2016":
            start = EXTENDED_SIGNAL
        if step == "B-2016":
            extra = liquid_codes(POOL_B)
            pool = pool_a + extra
            signals, panels, raws = build_all(pool, regimes)
            start = EXTENDED_SIGNAL
        count = class_count(signals[(signals["signal_date"] >= start) & (signals["exit_date"] < REVAL_TRAIN_BEFORE)])
        decision = {"step": step, "start": start, "batch": "B" if step.startswith("B") else "A", "train_before_reval": count, "n_names": len(pool), "passed": False}
        log(f"label count {decision}")
        if enough(count, MIN_TRAIN, MIN_CLASS):
            decision["passed"] = True
            return signals, pool, panels, raws, decision
    decision["passed"] = False
    return signals, pool, panels, raws, decision


def attach_predictions(signals: pd.DataFrame, panels: dict, predictor) -> pd.DataFrame:
    out = signals.copy()
    values = []
    for i, row in enumerate(out.itertuples(index=False), start=1):
        qfq = panels[row.code]
        if int((qfq["date"] <= pd.Timestamp(row.signal_date)).sum()) < LOOKBACK:
            values.append(np.nan)
            continue
        samples = predict_slot(predictor, qfq, pd.Timestamp(row.signal_date), LOOKBACK, SAMPLES)
        origin = float(qfq.loc[qfq["date"] == pd.Timestamp(row.signal_date), "close"].iloc[0])
        values.append(float(np.mean(samples) / origin - 1))
        if i % 20 == 0:
            log(f"kronos {i}/{len(out)}")
    out["pred_ret"] = values
    return out


def fit_predict(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, pd.Series]:
    y = train["win"].astype(int).to_numpy()
    mu = train[FEATURES].mean()
    sigma = train[FEATURES].std().replace(0, 1.0)
    kmu = float(train["pred_ret"].mean())
    ksig = float(train["pred_ret"].std())
    if not np.isfinite(ksig) or ksig == 0:
        ksig = 1.0
    def pack(frame: pd.DataFrame, with_kronos: bool) -> pd.DataFrame:
        base = (frame[FEATURES] - mu) / sigma
        if with_kronos:
            base = base.copy()
            base["pred_ret"] = (frame["pred_ret"] - kmu) / ksig
        return base
    models = {
        "logit_c1": (LogisticRegression(C=1.0, max_iter=500), False),
        "logit_c1_kronos": (LogisticRegression(C=1.0, max_iter=500), True),
        "logit_c01": (LogisticRegression(C=0.1, max_iter=500), False),
        "logit_c01_kronos": (LogisticRegression(C=0.1, max_iter=500), True),
        "tree_d2": (DecisionTreeClassifier(max_depth=2, min_samples_leaf=20, random_state=SEED), False),
        "tree_d2_kronos": (DecisionTreeClassifier(max_depth=2, min_samples_leaf=20, random_state=SEED), True),
    }
    out = {}
    for name, (model, with_kronos) in models.items():
        model.fit(pack(train, with_kronos), y)
        proba = model.predict_proba(pack(test, with_kronos))
        # class column order follows model.classes_
        pos = list(model.classes_).index(1) if 1 in list(model.classes_) else None
        out[name] = pd.Series(proba[:, pos] if pos is not None else np.zeros(len(test)), index=test.index)
        if name == "logit_c1_kronos" and hasattr(model, "coef_"):
            out["_coef"] = dict(zip(list(pack(train, True).columns), [float(v) for v in model.coef_.ravel()]))
    return out


def block_diff(left: pd.DataFrame, right: pd.DataFrame) -> tuple[float, float]:
    if left.empty or right.empty:
        return (float("nan"), float("nan"))
    dates = sorted(set(pd.to_datetime(left["signal_date"])) | set(pd.to_datetime(right["signal_date"])))
    mapping = {date: i // BOOT_BLOCK for i, date in enumerate(dates)}
    rng = np.random.default_rng(SEED)
    keys = sorted(set(mapping.values()))
    left_flag = left["win"].astype(float).to_numpy()
    right_flag = right["win"].astype(float).to_numpy()
    left_block = pd.to_datetime(left["signal_date"]).map(mapping).to_numpy()
    right_block = pd.to_datetime(right["signal_date"]).map(mapping).to_numpy()
    diffs = []
    for _ in range(BOOT_REPS):
        draw = rng.choice(keys, size=len(keys), replace=True)
        a = [left_flag[left_block == key] for key in draw if np.any(left_block == key)]
        b = [right_flag[right_block == key] for key in draw if np.any(right_block == key)]
        if a and b:
            diffs.append(float(np.concatenate(a).mean() - np.concatenate(b).mean()))
    if not diffs:
        return (float("nan"), float("nan"))
    return float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))


def metrics(trades: pd.DataFrame, n_signals: int, years: float) -> dict:
    if trades.empty:
        return {"n_trades": 0, "n_wins": 0, "win_rate": np.nan, "coverage": 0.0, "n_dates": 0, "per_year": 0.0, "mean_net": np.nan, "payoff": np.nan, "profit_factor": np.nan, "direction_accuracy": np.nan, "max_drawdown": 0.0}
    wins = trades["win"].astype(bool)
    rets = trades["ret"].to_numpy(float)
    pnls = trades["pnl"].to_numpy(float)
    gain, loss = rets[wins.to_numpy()], rets[~wins.to_numpy()]
    pos, neg = pnls[pnls > 0].sum(), pnls[pnls <= 0].sum()
    directed = trades[trades["direction_5"].isin([1, -1])]
    _dd, _curve = equity_curve(trades, metrics.raws)
    return {
        "n_trades": int(len(trades)),
        "n_wins": int(wins.sum()),
        "win_rate": float(wins.mean()),
        "coverage": float(len(trades) / n_signals) if n_signals else np.nan,
        "n_dates": int(pd.to_datetime(trades["signal_date"]).nunique()),
        "per_year": float(len(trades) / years) if years else np.nan,
        "mean_net": float(rets.mean()),
        "payoff": float(gain.mean() / abs(loss.mean())) if len(gain) and len(loss) else np.nan,
        "profit_factor": float(pos / abs(neg)) if neg < 0 else np.nan,
        "direction_accuracy": float((directed["direction_5"] == 1).mean()) if len(directed) else np.nan,
        "max_drawdown": _dd,
    }


def stress_metrics(trades: pd.DataFrame, n_signals: int) -> dict:
    if trades.empty:
        return {"stress_win_rate": np.nan, "stress_mean_net": np.nan, "stress_profit_factor": np.nan, "stress_drawdown": 0.0}
    copied = trades.copy()
    copied["win"] = copied["win_stress"]
    copied["ret"] = copied["ret_stress"]
    copied["pnl"] = copied["pnl_stress"]
    base = metrics(copied, n_signals, 1.0)
    dd, _curve = equity_curve(trades, metrics.raws, stress=True)
    return {"stress_win_rate": base["win_rate"], "stress_mean_net": base["mean_net"], "stress_profit_factor": base["profit_factor"], "stress_drawdown": dd}


def judge(row: dict) -> str:
    gain = row.get("win_gain", np.nan)
    low = row.get("diff_low", np.nan)
    positive = np.isfinite(row.get("profit_factor", np.nan)) and row["profit_factor"] >= 1 and np.isfinite(row.get("payoff", np.nan)) and row["payoff"] >= 0.75 and row.get("mean_net", -1) > 0
    if not np.isfinite(gain):
        return "未完成"
    if gain > 0 and not positive:
        return "无效"
    passed = gain > 0 and positive and row["n_trades"] >= 30 and row["coverage"] >= 0.20 and np.isfinite(low) and low > 0
    if passed:
        return "有效"
    if gain > 0:
        return "不确定"
    return "无效"


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    log("calibrator start")
    regimes = regime_of(load_index())
    pool_a = liquid_codes(POOL_A)
    signals, panels, raws = build_all(pool_a, regimes)
    signals, pool, panels, raws, decision = choose_sample(signals, pool_a, panels, raws, regimes)
    decision["passed"] = bool(decision.get("passed"))
    (OUT / "label_counts.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
    if len(signals):
        signals.to_csv(OUT / "signals_standalone.csv", index=False)
    if not decision["passed"]:
        log("training labels below the frozen minimum; models not fit")
        write_stop_report(decision)
        return
    # Rebuild panels for the final pool if the last step already did so.
    if not panels:
        signals, panels, raws = build_all(pool, regimes)
    usable = signals[signals["exit_date"] < "2026-11-14"].copy()
    predictor, n_params = load_predictor()
    log(f"zero-shot params {n_params} signals {len(usable)}")
    usable = attach_predictions(usable, panels, predictor)
    usable = usable[usable["pred_ret"].notna()].copy()
    usable.to_csv(OUT / "signals_with_kronos.csv", index=False)
    metrics.raws = raws
    score_rows = []
    trade_frames = []
    coefficients = {}
    for fold in FOLDS:
        name, start, end, exit_before, train_before, exploratory = fold
        train = usable[(usable["signal_date"] >= decision["start"]) & (usable["exit_date"] < train_before)].dropna(subset=FEATURES + ["pred_ret"])
        test = usable[(usable["signal_date"] >= start) & (usable["signal_date"] <= end) & (usable["exit_date"] < exit_before)].dropna(subset=FEATURES + ["pred_ret"])
        count = class_count(train)
        log(f"fold {name} train {count} test {len(test)} exploratory {exploratory}")
        if not enough(count, MIN_FOLD_TRAIN, MIN_FOLD_CLASS) or test.empty or len(np.unique(train["win"].astype(int))) < 2:
            score_rows.append({"fold": name, "id": "ALL", "note": "insufficient_train", **count, "n_test_signals": int(len(test))})
            continue
        fitted = fit_predict(train, test)
        coefficients[name] = fitted.get("_coef", {})
        years = max((pd.Timestamp(end) - pd.Timestamp(start)).days / 365.25, 0.2)
        n_signals = int(len(test))
        raw_taken = apply_portfolio(test)
        taken = {"hit_raw": raw_taken, "kronos_pos": apply_portfolio(test[test["pred_ret"] > 0])}
        for model_name in ("logit_c1", "logit_c1_kronos", "logit_c01", "logit_c01_kronos", "tree_d2", "tree_d2_kronos"):
            mask = fitted[model_name] > 0.5
            taken[model_name] = apply_portfolio(test.loc[mask])
        for candidate in CANDIDATES:
            chosen = taken[candidate]
            row = metrics(chosen, n_signals, years)
            row.update(stress_metrics(chosen, n_signals))
            row.update({"fold": name, "id": candidate, "exploratory": exploratory, "train_n": count["n"], "train_wins": count["wins"], "train_losses": count["losses"]})
            if candidate != "hit_raw":
                low, high = block_diff(chosen, raw_taken)
                row["diff_low_vs_raw"] = low
                row["diff_high_vs_raw"] = high
                row["win_gain_vs_raw"] = row["win_rate"] - (float(raw_taken["win"].mean()) if len(raw_taken) else np.nan)
            if candidate == "logit_c1_kronos":
                base = taken["logit_c1"]
                low, high = block_diff(chosen, base)
                row["diff_low"] = low
                row["diff_high"] = high
                row["win_gain"] = row["win_rate"] - (float(base["win"].mean()) if len(base) else np.nan)
                removed = opportunity(raw_taken, chosen)
                row["removed_winners"] = removed["removed_winners"]
                row["removed_winner_pnl"] = removed["removed_winner_pnl"]
                row["removed_loser_pnl"] = removed["removed_loser_pnl"]
            score_rows.append(row)
            if len(chosen):
                saved = chosen.copy()
                saved["strategy"] = candidate
                saved["fold"] = name
                trade_frames.append(saved)
    scores = pd.DataFrame(score_rows)
    scores.to_csv(OUT / "all_scores.csv", index=False)
    if trade_frames:
        pd.concat(trade_frames, ignore_index=True).to_csv(OUT / "trades.csv", index=False)
    (OUT / "logit_coefficients.json").write_text(json.dumps(coefficients, ensure_ascii=False, indent=2), encoding="utf-8")
    make_plot(scores)
    write_report(scores, decision, coefficients)
    log("done")


def opportunity(raw_taken: pd.DataFrame, kept: pd.DataFrame) -> dict:
    if raw_taken.empty:
        return {"removed_winners": 0, "removed_winner_pnl": 0.0, "removed_loser_pnl": 0.0}
    keys = set(zip(kept["code"], pd.to_datetime(kept["signal_date"]))) if len(kept) else set()
    removed = raw_taken.loc[[key not in keys for key in zip(raw_taken["code"], pd.to_datetime(raw_taken["signal_date"]))]]
    winners = removed[removed["pnl"] > 0]
    losers = removed[removed["pnl"] <= 0]
    return {
        "removed_winners": int(len(winners)),
        "removed_winner_pnl": float(winners["pnl"].sum()) if len(winners) else 0.0,
        "removed_loser_pnl": float(losers["pnl"].sum()) if len(losers) else 0.0,
    }


def make_plot(scores: pd.DataFrame) -> None:
    fig_dir = OUT / "figures"
    fig_dir.mkdir(exist_ok=True)
    part = scores[scores["fold"].isin(["wf_2022", "wf_2023", "wf_2024"]) & scores["id"].isin(CANDIDATES)]
    if part.empty or "win_rate" not in part:
        return
    grouped = part.groupby("id")["win_rate"].mean()
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.barh(grouped.index, grouped.values)
    ax.set_xlabel("Mean walk-forward post-cost win rate, 2022-2024")
    ax.set_title("Not the 2025-2026 exploratory window")
    fig.tight_layout()
    fig.savefig(fig_dir / "walkforward_winrate.png", dpi=120)
    plt.close(fig)


def write_stop_report(decision: dict) -> None:
    count = decision.get("train_before_reval", {})
    text = "\n".join([
        "# 轻量判断模型",
        "",
        "报告时区 Asia/Shanghai。判断模型没有训练。",
        "",
        f"扩样结束后，退出日早于 2025-01-15 的 HIT 单笔仍是 {count}。事先要求至少 {MIN_TRAIN} 笔且两类都不少于 {MIN_CLASS} 笔。",
        "没有放宽 DOWN 门禁，也没有改用 2025–2026 凑样本。",
        "",
    ])
    (OUT / "REPORT.md").write_text(text, encoding="utf-8")


def write_report(scores: pd.DataFrame, decision: dict, coefficients: dict) -> None:
    def cell(value, pct=False):
        if value is None or not isinstance(value, (int, float, np.floating)) or not np.isfinite(value):
            return "NA"
        return f"{100 * value:.1f}%" if pct else f"{value:.2f}"

    lines = [
        "# 轻量判断模型回测",
        "",
        "报告时区 Asia/Shanghai，数据截至 2026-11-13。Kronos 权重没有更新。2025–2026 是探索性再验证，不是独立最终样本外。",
        "",
        f"训练样本决定：{json.dumps(decision.get('train_before_reval'), ensure_ascii=False)}，起点 {decision.get('start')}，名单批次 {decision.get('batch')}，股票 {decision.get('n_names')} 只。",
        "",
        "## 走步 2022–2024",
        "",
        "| 候选 | 折 | 笔数 | 胜率 | 相对无Kronos或未过滤 | 区块下界 | PF | 平均净收益 | 回撤 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    walk = scores[scores["fold"].isin(["wf_2022", "wf_2023", "wf_2024"])]
    for row in walk.itertuples(index=False):
        if row.id == "ALL":
            lines.append(f"| 样本不足 | {row.fold} | {getattr(row, 'n', 0)} | NA | NA | NA | NA | NA | NA |")
            continue
        gain = getattr(row, "win_gain", np.nan) if row.id == "logit_c1_kronos" else getattr(row, "win_gain_vs_raw", np.nan)
        low = getattr(row, "diff_low", np.nan) if row.id == "logit_c1_kronos" else getattr(row, "diff_low_vs_raw", np.nan)
        lines.append(f"| {row.id} | {row.fold} | {int(row.n_trades)} | {cell(row.win_rate, True)} | {cell(gain, True)} | {cell(low, True)} | {cell(row.profit_factor)} | {cell(row.mean_net, True)} | {cell(row.max_drawdown, True)} |")
    primary = walk[walk["id"] == "logit_c1_kronos"]
    label = "未完成"
    if len(primary) and primary["n_trades"].sum() > 0:
        # Pooled judgment uses the reval row separately; walk-forward label is computed on the concatenated trades file if present.
        label = "见逐折，不把单折最高分当成结论"
    reval = scores[scores["fold"] == "reval_2025_2026"]
    lines.extend(["", "## 探索性再验证 2025–2026", "", "| 候选 | 笔数 | 胜率 | 方向准确率 | 平均净收益 | PF | 盈亏比 | 回撤 | 费用加倍胜率 |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for row in reval.itertuples(index=False):
        if row.id == "ALL":
            continue
        lines.append(f"| {row.id} | {int(row.n_trades)} | {cell(row.win_rate, True)} | {cell(row.direction_accuracy, True)} | {cell(row.mean_net, True)} | {cell(row.profit_factor)} | {cell(row.payoff)} | {cell(row.max_drawdown, True)} | {cell(getattr(row, 'stress_win_rate', np.nan), True)} |")
    focus = reval[reval["id"] == "logit_c1_kronos"]
    base = reval[reval["id"] == "logit_c1"]
    conclusion = "判断模型已训练。" if decision.get("passed") else "判断模型没有训练。"
    if len(focus) and len(base) and np.isfinite(focus.iloc[0].get("win_gain", np.nan)):
        row = focus.iloc[0]
        conclusion += f" 探索性再验证里，含 Kronos 的 C=1 logistic 胜率比不含 Kronos 的同一模型 {cell(row['win_gain'], True)}，区块下界 {cell(row['diff_low'], True)}。事先标准判定为{judge(row)}。"
    else:
        conclusion += " 主比较没有同时得到含 Kronos 和不含 Kronos 的再验证成交。"
    lines.extend([
        "",
        "## 结论",
        "",
        conclusion,
        "",
        "走步各折不能挑出最高分再替换模型。Kronos 预训练截止仍不可审计。池子有存活偏差。下一步不做全量微调。",
        "",
        f"主比较系数（各折含 Kronos 的 C=1）已写入 logit_coefficients.json。最近一折系数：{json.dumps(list(coefficients.values())[-1] if coefficients else {}, ensure_ascii=False)}",
        "",
    ])
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log(traceback.format_exc())
        raise
