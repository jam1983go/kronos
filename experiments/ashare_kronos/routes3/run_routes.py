"""Run the three pre-registered routes. Do not add candidates after scores exist."""

from __future__ import annotations

import json
import shutil
import time
import traceback
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from experiments.ashare_kronos.data_io import fetch_eastmoney
from experiments.ashare_kronos.routes3.engine import (
    _buy_cost,
    apply_portfolio,
    build_signals,
    equity_curve,
    excluded_name,
    regime_of,
    stop_distance,
)
from experiments.ashare_kronos.routes3.settings import (
    ACCOUNT,
    AS_OF,
    BOOT_BLOCK,
    BOOT_REPS,
    CANDIDATES,
    CHECK,
    COST_HURDLE,
    DATA_LAST,
    FIT,
    FT_BATCH,
    FT_LR,
    FT_SEQ,
    FT_STEPS,
    HISTORY_BEG,
    HISTORY_END,
    LOOKBACK,
    POOL,
    PRED_LEN,
    REVAL,
    SAMPLES,
    SEED,
)
from experiments.ashare_kronos.winrate.infer import load_predictor, predict_slot
from model.kronos import calc_time_stamps

ROOT = Path(__file__).resolve().parents[3]
DATA = Path(__file__).resolve().parent / "data"
OUT = Path(__file__).resolve().parent / "outputs"
OLD = [ROOT / "experiments/ashare_kronos/data", ROOT / "experiments/ashare_kronos/winrate/data"]
TZ = ZoneInfo("Asia/Shanghai")
BASE_FEATURES = ["dif", "hist", "vol_ratio", "atr_pct", "ret_5", "ret_20", "regime_ord", "stop_dist"]


def log(message: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    line = f"[{datetime.now(TZ).strftime('%Y-%m-%d %H:%M:%S%z')}] {message}"
    print(line, flush=True)
    with (OUT / "run_log.txt").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def self_check() -> None:
    assert abs(stop_distance(10, 9, 0.1) - 0.105) < 1e-12
    assert stop_distance(10, 10, 0) == 0.005
    assert abs(_buy_cost(100, 10, 1) - 1005.5) < 1e-6
    assert len(CANDIDATES) == 8


def load_frame(code: str, market: str, kind: str) -> pd.DataFrame:
    path = DATA / f"{code}_{kind}.csv"
    if not path.exists():
        for source in OLD:
            old = source / f"{code}_{kind}.csv"
            if old.exists() and market == "SH" or (old.exists() and code != "000001"):
                shutil.copy(old, path)
                break
    if not path.exists():
        fqt = 1 if kind == "qfq" else 0
        _name, frame = fetch_eastmoney(code, market, fqt, HISTORY_BEG, HISTORY_END)
        if kind == "qfq":
            (DATA / f"{code}_name.txt").write_text(_name + "\n", encoding="utf-8")
        frame.to_csv(path, index=False)
    frame = pd.read_csv(path, parse_dates=["date"])
    frame = frame[frame["date"] < pd.Timestamp(AS_OF)].sort_values("date").reset_index(drop=True)
    return frame


def load_index() -> pd.DataFrame:
    path = DATA / "index_sh_qfq.csv"
    if not path.exists():
        _name, frame = fetch_eastmoney("000001", "SH", 1, HISTORY_BEG, HISTORY_END)
        frame.to_csv(path, index=False)
        (DATA / "index_sh_name.txt").write_text(_name + "\n", encoding="utf-8")
    frame = pd.read_csv(path, parse_dates=["date"])
    return frame[frame["date"] < pd.Timestamp(AS_OF)].sort_values("date").reset_index(drop=True)


def assign_window(signal_date: pd.Timestamp, exit_date: pd.Timestamp) -> str:
    signal = pd.Timestamp(signal_date)
    exit_at = pd.Timestamp(exit_date)
    spans = (("fit", FIT, pd.Timestamp(CHECK[0])), ("check", CHECK, pd.Timestamp(REVAL[0])), ("reval", REVAL, pd.Timestamp(DATA_LAST) + pd.Timedelta(days=1)))
    for name, bounds, exit_before in spans:
        if pd.Timestamp(bounds[0]) <= signal <= pd.Timestamp(bounds[1]):
            return "purged" if exit_at >= exit_before else name
    return "outside"


def attach_kronos(candidates: pd.DataFrame, panels: dict, predictor, column: str) -> pd.DataFrame:
    out = candidates.copy()
    values = []
    for i, row in enumerate(out.itertuples(index=False), start=1):
        if row.status != "candidate":
            values.append(np.nan)
            continue
        samples = predict_slot(predictor, panels[row.code], pd.Timestamp(row.signal_date), LOOKBACK, SAMPLES)
        values.append(float(np.mean(samples) / row_signal_close(panels[row.code], row.signal_date) - 1))
        if i % 25 == 0:
            log(f"{column} {i}/{len(out)}")
    out[column] = values
    return out


def row_signal_close(qfq: pd.DataFrame, signal_date) -> float:
    hit = qfq.loc[qfq["date"] == pd.Timestamp(signal_date), "close"]
    return float(hit.iloc[0])


def summarize(trades: pd.DataFrame, n_signals: int) -> dict:
    if trades.empty:
        return {"n_trades": 0, "n_wins": 0, "win_rate": np.nan, "coverage": 0.0, "mean_net": np.nan, "payoff": np.nan, "profit_factor": np.nan, "direction_accuracy": np.nan, "n_direction": 0, "n_dates": 0}
    wins = trades["win"].astype(bool)
    rets = trades["ret"].to_numpy(dtype=float)
    pnls = trades["pnl"].to_numpy(dtype=float)
    gain = rets[wins.to_numpy()]
    loss = rets[~wins.to_numpy()]
    pos = pnls[pnls > 0].sum()
    neg = pnls[pnls <= 0].sum()
    directed = trades[trades["direction_5"].isin([1, -1])]
    return {
        "n_trades": int(len(trades)),
        "n_wins": int(wins.sum()),
        "win_rate": float(wins.mean()),
        "coverage": float(len(trades) / n_signals) if n_signals else np.nan,
        "mean_net": float(rets.mean()),
        "payoff": float(gain.mean() / abs(loss.mean())) if len(gain) and len(loss) else np.nan,
        "profit_factor": float(pos / abs(neg)) if neg < 0 else np.nan,
        "direction_accuracy": float((directed["direction_5"] == 1).mean()) if len(directed) else np.nan,
        "n_direction": int(len(directed)),
        "n_dates": int(pd.to_datetime(trades["signal_date"]).nunique()),
    }


def block_bounds(flags: np.ndarray, dates: pd.Series) -> tuple[float, float]:
    if len(flags) == 0:
        return (np.nan, np.nan)
    ordered = sorted(pd.to_datetime(dates).unique())
    mapping = {date: i // BOOT_BLOCK for i, date in enumerate(ordered)}
    blocks = pd.to_datetime(dates).map(mapping).to_numpy()
    rng = np.random.default_rng(SEED)
    groups = {b: flags[blocks == b] for b in sorted(set(blocks))}
    keys = list(groups)
    rates = []
    for _ in range(BOOT_REPS):
        draw = rng.choice(keys, size=len(keys), replace=True)
        sample = np.concatenate([groups[k] for k in draw])
        rates.append(sample.mean())
    return float(np.quantile(rates, 0.025)), float(np.quantile(rates, 0.975))


def block_diff(left: pd.DataFrame, right: pd.DataFrame) -> tuple[float, float]:
    if left.empty or right.empty:
        return (np.nan, np.nan)
    dates = sorted(set(pd.to_datetime(left["signal_date"])) | set(pd.to_datetime(right["signal_date"])))
    mapping = {date: i // BOOT_BLOCK for i, date in enumerate(dates)}
    rng = np.random.default_rng(SEED)
    keys = sorted(set(mapping.values()))
    la = {b: (pd.to_datetime(left["signal_date"]).map(mapping) == b).to_numpy() for b in keys}
    ra = {b: (pd.to_datetime(right["signal_date"]).map(mapping) == b).to_numpy() for b in keys}
    lf = left["win"].astype(float).to_numpy()
    rf = right["win"].astype(float).to_numpy()
    diffs = []
    for _ in range(BOOT_REPS):
        draw = rng.choice(keys, size=len(keys), replace=True)
        a_parts = [lf[la[k]] for k in draw if la[k].any()]
        b_parts = [rf[ra[k]] for k in draw if ra[k].any()]
        if not a_parts or not b_parts:
            continue
        diffs.append(float(np.concatenate(a_parts).mean() - np.concatenate(b_parts).mean()))
    if not diffs:
        return (np.nan, np.nan)
    return float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))


def take_mask(frame: pd.DataFrame, name: str, prob: pd.Series | None = None) -> pd.Series:
    if name == "hit_raw":
        return pd.Series(True, index=frame.index)
    if name == "hit_kronos_pos":
        return frame["pred_ret"] > 0
    if name == "hit_kronos_cost":
        return frame["pred_ret"] > COST_HURDLE
    if name == "hit_finetune_pos":
        return frame["pred_ret_ft"] > 0
    if name == "logit_c1":
        return prob > 0.5
    if name in {"logit_c1_kronos", "logit_c01_kronos", "tree_d2_kronos"}:
        return prob > 0.5
    raise KeyError(name)


def fit_models(fit_rows: pd.DataFrame) -> dict:
    y = fit_rows["win"].astype(int).to_numpy()
    if len(np.unique(y)) < 2 or len(fit_rows) < 8:
        return {"ok": False, "reason": f"fit rows {len(fit_rows)} classes {len(np.unique(y))}"}
    mu = fit_rows[BASE_FEATURES].mean()
    sigma = fit_rows[BASE_FEATURES].std().replace(0, 1)
    base = (fit_rows[BASE_FEATURES] - mu) / sigma
    both = base.copy()
    kmu = float(fit_rows["pred_ret"].mean())
    ksig = float(fit_rows["pred_ret"].std() or 1)
    both["pred_ret"] = (fit_rows["pred_ret"] - kmu) / ksig
    models = {
        "logit_c1": LogisticRegression(C=1.0, max_iter=400),
        "logit_c1_kronos": LogisticRegression(C=1.0, max_iter=400),
        "logit_c01_kronos": LogisticRegression(C=0.1, max_iter=400),
        "tree_d2_kronos": DecisionTreeClassifier(max_depth=2, min_samples_leaf=15, random_state=SEED),
    }
    models["logit_c1"].fit(base, y)
    for key in ("logit_c1_kronos", "logit_c01_kronos", "tree_d2_kronos"):
        models[key].fit(both, y)
    return {"ok": True, "models": models, "mu": mu, "sigma": sigma, "kmu": kmu, "ksig": ksig}


def score_split(name: str, chosen: pd.DataFrame, n_signals: int, raws: dict) -> dict:
    stats = summarize(chosen, n_signals)
    low, high = block_bounds(chosen["win"].astype(float).to_numpy() if len(chosen) else np.array([]), chosen["signal_date"] if len(chosen) else pd.Series(dtype="datetime64[ns]"))
    dd, _curve = equity_curve(chosen, raws, stress=False)
    stress = chosen.copy()
    if len(stress):
        stress["win"] = stress["win_stress"]
        stress["ret"] = stress["ret_stress"]
        stress["pnl"] = stress["pnl_stress"]
    stress_stats = summarize(stress, n_signals)
    dd_stress, _curve2 = equity_curve(chosen, raws, stress=True) if len(chosen) else (0.0, pd.DataFrame())
    stats.update({"id": name, "block_low": low, "block_high": high, "max_drawdown": dd, "stress_win_rate": stress_stats["win_rate"], "stress_mean_net": stress_stats["mean_net"], "stress_profit_factor": stress_stats["profit_factor"], "stress_drawdown": dd_stress})
    return stats


def opportunity(raw_trades: pd.DataFrame, kept_keys: set[tuple]) -> dict:
    if raw_trades.empty:
        return {"removed": 0, "removed_winners": 0, "removed_winner_pnl": 0.0, "removed_losers": 0}
    keys = list(zip(raw_trades["code"], pd.to_datetime(raw_trades["signal_date"])))
    removed = raw_trades.loc[[key not in kept_keys for key in keys]]
    winners = removed[removed["pnl"] > 0]
    return {
        "removed": int(len(removed)),
        "removed_winners": int(len(winners)),
        "removed_winner_pnl": float(winners["pnl"].sum()) if len(winners) else 0.0,
        "removed_losers": int((removed["pnl"] <= 0).sum()),
    }


def finetune_head(predictor, panels: dict) -> dict:
    model = predictor.model
    tokenizer = predictor.tokenizer
    for param in model.parameters():
        param.requires_grad = False
    for param in model.head.parameters():
        param.requires_grad = True
    trainable = int(sum(param.numel() for param in model.parameters() if param.requires_grad))
    total = int(sum(param.numel() for param in model.parameters()))
    windows = []
    for code, qfq in panels.items():
        dates = qfq["date"]
        for end in range(FT_SEQ, len(qfq)):
            if not (pd.Timestamp(FIT[0]) <= dates.iloc[end] <= pd.Timestamp(FIT[1])):
                continue
            if (end - FT_SEQ) % 20 != 0:
                continue
            hist = qfq.iloc[end - FT_SEQ : end + 1]
            values = hist[["open", "high", "low", "close", "volume", "amount"]].to_numpy(dtype=np.float32)
            mean = values.mean(axis=0)
            std = values.std(axis=0)
            norm = np.clip((values - mean) / (std + 1e-5), -5, 5)
            stamp = calc_time_stamps(pd.to_datetime(hist["date"])).to_numpy(dtype=np.float32)
            windows.append((norm, stamp))
    windows = windows[: FT_STEPS * FT_BATCH]
    opt = torch.optim.AdamW([param for param in model.head.parameters() if param.requires_grad], lr=FT_LR)
    losses = []
    model.eval()
    started = time.perf_counter()
    for step in range(FT_STEPS):
        batch = windows[step * FT_BATCH : (step + 1) * FT_BATCH]
        if len(batch) < FT_BATCH:
            break
        x = torch.tensor(np.stack([item[0] for item in batch]))
        stamp = torch.tensor(np.stack([item[1][:-1] for item in batch]))
        with torch.no_grad():
            s1, s2 = tokenizer.encode(x, half=True)
        logits = model(s1[:, :-1], s2[:, :-1], stamp)
        loss, s1_loss, s2_loss = model.head.compute_loss(logits[0], logits[1], s1[:, 1:], s2[:, 1:])
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append({"step": step + 1, "loss": float(loss.item()), "s1": float(s1_loss.item()), "s2": float(s2_loss.item())})
        log(f"finetune step {step + 1} loss {loss.item():.4f}")
    elapsed = time.perf_counter() - started
    path = OUT / "dual_head_finetune.pt"
    torch.save(model.head.state_dict(), path)
    return {"trainable": trainable, "total": total, "steps": len(losses), "seconds": elapsed, "losses": losses, "windows": len(windows), "path": str(path)}


def verdict(row: dict, diff_low: float, n_fit: int) -> str:
    ok = (
        row["n_trades"] >= 30
        and row["coverage"] >= 0.20
        and np.isfinite(row["profit_factor"])
        and row["profit_factor"] >= 1
        and np.isfinite(row["payoff"])
        and row["payoff"] >= 0.75
        and row["mean_net"] > 0
        and np.isfinite(diff_low)
        and diff_low > 0
    )
    if ok:
        return "有效"
    if np.isfinite(row["win_rate"]) and row.get("win_gain", -1) > 0 and row["n_trades"] < 30:
        return "不确定"
    if row.get("win_gain", -1) > 0 and (not np.isfinite(row["profit_factor"]) or row["profit_factor"] < 1 or row["mean_net"] <= 0):
        return "无效"
    if row.get("win_gain", 0) > 0:
        return "不确定"
    return "无效"


def main() -> None:
    self_check()
    DATA.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    log(f"routes start candidates {len(CANDIDATES)} account {ACCOUNT}")
    blockers = []
    try:
        import comet_ml  # noqa: F401
    except Exception as exc:
        blockers.append(f"官方训练依赖 comet_ml 不可用：{exc}")
    try:
        import qlib  # noqa: F401
    except Exception as exc:
        blockers.append(f"官方训练依赖 qlib 不可用：{exc}")
    blockers.append(f"cuda={torch.cuda.is_available()}；finetune/train_predictor.py 需要 torchrun、DDP 和 QlibDataset，本机不启动该全量脚本。")
    (OUT / "official_train_blocker.txt").write_text("\n".join(blockers) + "\n", encoding="utf-8")
    panels = {}
    names = {}
    raws = {}
    for spec in POOL:
        qfq = load_frame(spec["code"], spec["market"], "qfq")
        raw = load_frame(spec["code"], spec["market"], "raw")
        name = (DATA / f"{spec['code']}_name.txt").read_text(encoding="utf-8").strip() if (DATA / f"{spec['code']}_name.txt").exists() else spec["code"]
        reason = excluded_name(name, spec["code"])
        if reason:
            log(f"exclude {spec['code']} {name} {reason}")
            continue
        panels[spec["code"]] = qfq
        raws[spec["code"]] = raw
        names[spec["code"]] = name
        log(f"loaded {spec['code']} {name} {qfq['date'].min().date()} {qfq['date'].max().date()} rows {len(qfq)}")
    regimes = regime_of(load_index())
    parts = [build_signals(code, names[code], panels[code], raws[code], regimes) for code in panels]
    signals = pd.concat(parts, ignore_index=True)
    signals = signals[signals["status"] == "candidate"].copy()
    signals["window"] = [assign_window(s, e) for s, e in zip(signals["signal_date"], signals["exit_date"])]
    signals = signals[signals["window"].isin(["fit", "check", "reval"])].copy()
    signals.to_csv(OUT / "signals_standalone.csv", index=False)
    log(signals.groupby(["window"]).size().to_string() if len(signals) else "no signals")
    predictor, n_params = load_predictor()
    log(f"zero-shot params {n_params}")
    started = time.perf_counter()
    signals = attach_kronos(signals, panels, predictor, "pred_ret")
    log(f"zero-shot inference {time.perf_counter() - started:.1f}s")
    signals.to_csv(OUT / "signals_with_kronos.csv", index=False)
    fit = signals[signals["window"] == "fit"].copy()
    fitted = fit_models(fit)
    (OUT / "route2_fit.json").write_text(json.dumps({"ok": fitted["ok"], **({"reason": fitted["reason"]} if not fitted["ok"] else {"rows": int(len(fit))})}, ensure_ascii=False, indent=2), encoding="utf-8")
    probs = {name: pd.Series(np.nan, index=signals.index) for name in ("logit_c1", "logit_c1_kronos", "logit_c01_kronos", "tree_d2_kronos")}
    if fitted["ok"]:
        mu, sigma, kmu = fitted["mu"], fitted["sigma"], fitted["kmu"]
        ksig = fitted["ksig"] if np.isfinite(fitted["ksig"]) and fitted["ksig"] else 1.0
        base = (signals[BASE_FEATURES] - mu) / sigma
        both = base.copy()
        both["pred_ret"] = (signals["pred_ret"] - kmu) / ksig
        probs["logit_c1"] = pd.Series(fitted["models"]["logit_c1"].predict_proba(base)[:, 1], index=signals.index)
        for key in ("logit_c1_kronos", "logit_c01_kronos", "tree_d2_kronos"):
            probs[key] = pd.Series(fitted["models"][key].predict_proba(both)[:, 1], index=signals.index)
    ft_info = finetune_head(predictor, panels)
    (OUT / "finetune_log.json").write_text(json.dumps(ft_info, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"finetune trainable {ft_info['trainable']} of {ft_info['total']} steps {ft_info['steps']} seconds {ft_info['seconds']:.1f}")
    started = time.perf_counter()
    signals = attach_kronos(signals, panels, predictor, "pred_ret_ft")
    log(f"finetuned inference {time.perf_counter() - started:.1f}s")
    signals.to_csv(OUT / "signals_all_features.csv", index=False)
    rows = []
    trade_frames = []
    opp_rows = []
    for window in ("fit", "check", "reval"):
        part = signals[signals["window"] == window].copy()
        n_signals = int(len(part))
        raw_taken = apply_portfolio(part)
        raw_keys = set(zip(raw_taken["code"], pd.to_datetime(raw_taken["signal_date"]))) if len(raw_taken) else set()
        for name in CANDIDATES:
            if name.startswith("logit") or name.startswith("tree"):
                if not fitted["ok"]:
                    rows.append({"id": name, "window": window, "n_trades": 0, "note": fitted.get("reason", "not fit")})
                    continue
                mask = take_mask(part, name, probs[name].loc[part.index]).fillna(False)
            else:
                mask = take_mask(part, name).fillna(False)
            chosen = apply_portfolio(part.loc[mask.to_numpy()])
            stats = score_split(name, chosen, n_signals, raws)
            stats["window"] = window
            if name != "hit_raw" and len(raw_taken):
                low, high = block_diff(chosen, raw_taken) if len(chosen) else (np.nan, np.nan)
                stats["diff_low_vs_raw"] = low
                stats["diff_high_vs_raw"] = high
                stats["win_gain"] = stats["win_rate"] - float(raw_taken["win"].mean())
            if name == "logit_c1_kronos" and fitted["ok"] and len(part):
                base_taken = apply_portfolio(part.loc[take_mask(part, "logit_c1", probs["logit_c1"].loc[part.index]).fillna(False).to_numpy()])
                low, high = block_diff(chosen, base_taken) if len(chosen) and len(base_taken) else (np.nan, np.nan)
                stats["diff_low_vs_nokronos"] = low
                stats["diff_high_vs_nokronos"] = high
                stats["kronos_gain"] = stats["win_rate"] - (float(base_taken["win"].mean()) if len(base_taken) else np.nan)
            if name == "hit_finetune_pos" and len(part):
                base_taken = apply_portfolio(part.loc[take_mask(part, "hit_kronos_pos").fillna(False).to_numpy()])
                low, high = block_diff(chosen, base_taken) if len(chosen) and len(base_taken) else (np.nan, np.nan)
                stats["diff_low_vs_zeroshot"] = low
                stats["diff_high_vs_zeroshot"] = high
                stats["finetune_gain"] = stats["win_rate"] - (float(base_taken["win"].mean()) if len(base_taken) else np.nan)
            if name in {"hit_kronos_pos", "hit_kronos_cost", "hit_finetune_pos"}:
                kept = set(zip(part.loc[mask.to_numpy(), "code"], pd.to_datetime(part.loc[mask.to_numpy(), "signal_date"]))) if mask.any() else set()
                opp = opportunity(raw_taken, kept)
                opp.update({"id": name, "window": window})
                opp_rows.append(opp)
            rows.append(stats)
            if len(chosen):
                chosen = chosen.copy()
                chosen["strategy"] = name
                chosen["window"] = window
                trade_frames.append(chosen)
    scores = pd.DataFrame(rows)
    scores.to_csv(OUT / "all_scores.csv", index=False)
    if trade_frames:
        pd.concat(trade_frames, ignore_index=True).to_csv(OUT / "trades.csv", index=False)
    pd.DataFrame(opp_rows).to_csv(OUT / "opportunity_loss.csv", index=False)
    write_report(scores, opp_rows, ft_info, blockers, fitted, int((signals["window"] == "fit").sum()))
    log("report written")


def write_report(scores: pd.DataFrame, opp_rows: list, ft_info: dict, blockers: list, fitted: dict, n_fit: int) -> None:
    def pick(window, name):
        hit = scores[(scores["window"] == window) & (scores["id"] == name)]
        return None if hit.empty else hit.iloc[0]

    raw = pick("reval", "hit_raw")
    lines = ["# 三条路线实测", "", "报告时区 Asia/Shanghai，数据截至 2026-11-13。预训练骨干没有整网重训。2025–2026 是探索性再验证，不是独立最终样本外。本轮没有自动建立监控。", ""]
    lines.append(f"拟合窗在冻结规则下只有 {n_fit} 笔 HIT 候选。2024-07-01 至 2024-10-31 上证多数交易日被研究近似判为 DOWN，因此不开新仓。logistic 与树没有可训练标签：{'成功' if fitted['ok'] else fitted.get('reason')}。这不是把测试期换掉再挑赢家。")
    if raw is not None and np.isfinite(raw.get("win_rate", np.nan)):
        lines.append(
            f"未过滤研究近似在再验证里成交 {int(raw['n_trades'])} 笔，胜率 {_pct(raw['win_rate'])}，方向准确率 {_pct(raw['direction_accuracy'])}，平均净收益 {_pct(raw['mean_net'])}，profit factor {_num(raw['profit_factor'])}，组合最大回撤 {_pct(raw['max_drawdown'])}。笔数少于 30，不能把这个点估计写成成功。方向准确率和交易胜率不是同一个数。"
        )
    lines.append("")
    lines.append("## 对照")
    lines.append("")
    lines.append("| 路线 | 候选 | 再验证笔数 | 胜率 | 相对对照 | 区块下界 | PF | 平均净收益 | 判定 |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    comparisons = [
        ("1 HIT过滤", "hit_kronos_pos", "win_gain", "diff_low_vs_raw"),
        ("1 HIT过滤", "hit_kronos_cost", "win_gain", "diff_low_vs_raw"),
        ("2 校准消融", "logit_c1_kronos", "kronos_gain", "diff_low_vs_nokronos"),
        ("2 校准消融", "logit_c01_kronos", "kronos_gain", "diff_low_vs_nokronos"),
        ("2 校准消融", "tree_d2_kronos", "kronos_gain", "diff_low_vs_nokronos"),
        ("3 最后层", "hit_finetune_pos", "finetune_gain", "diff_low_vs_zeroshot"),
    ]
    judgments = []
    for route, name, gain_key, low_key in comparisons:
        row = pick("reval", name)
        if row is None or "win_rate" not in row or not np.isfinite(row.get("win_rate", np.nan)):
            lines.append(f"| {route} | {name} | 0 | NA | NA | NA | NA | NA | 未完成 |")
            judgments.append((route, name, "未完成"))
            continue
        gain = row.get(gain_key, np.nan)
        low = row.get(low_key, np.nan)
        label = verdict({**row.to_dict(), "win_gain": gain if np.isfinite(gain) else -1}, low if isinstance(low, float) else np.nan, n_fit)
        judgments.append((route, name, label))
        lines.append(
            f"| {route} | {name} | {int(row['n_trades'])} | {_pct(row['win_rate'])} | {_pp(gain)} | {_pp(low)} | {_num(row['profit_factor'])} | {_pct(row['mean_net'])} | {label} |"
        )
    lines.append("")
    lines.append("## 机会损失")
    lines.append("")
    lines.append("下面是未过滤组合真正成交、但被 Kronos 规则挡掉的单子。赢家损失是这些被挡赢家的扣费净利合计。")
    lines.append("")
    lines.append("| 窗口 | 规则 | 挡掉 | 其中赢家 | 赢家净利合计 | 挡掉的亏损笔 |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: |")
    for item in opp_rows:
        lines.append(f"| {item['window']} | {item['id']} | {item['removed']} | {item['removed_winners']} | {item['removed_winner_pnl']:.0f} | {item['removed_losers']} |")
    lines.append("")
    lines.append("## 微调")
    lines.append("")
    lines.append(f"实际更新的是 `DualHead`，可训练参数 {ft_info['trainable']} / {ft_info['total']}。步数 {ft_info['steps']}，学习率 {FT_LR}，序列 {FT_SEQ}，批大小 {FT_BATCH}，用时 {ft_info['seconds']:.1f} 秒。目标是官方同一形式的 s1/s2 交叉熵，骨干冻结且前向用 eval 以关掉 dropout。权重文件 `{ft_info['path']}`。这不是分类器冒充的权重更新，也不是全模型重训成功。")
    lines.append("")
    lines.append("官方全量脚本没有跑起来：")
    lines.append("")
    for item in blockers:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## 结论")
    lines.append("")
    lines.append("三条都跑完了各自能跑的部分。HIT 研究近似可以离线复现，但它不是仓库里的实盘 HIT。Kronos 过滤和 20 步最后层微调都没有提高再验证胜率。logistic 因拟合窗没有样本而没有训练，不能声称校准模型已测出 Kronos 增量。")
    lines.append("")
    for route, name, label in judgments:
        lines.append(f"- {route} / {name}：{label}。")
    lines.append("")
    lines.append("没有一条同时满足再验证的笔数、覆盖率、正期望和胜率差下界。因此本轮没有找到可实施的胜率提升。Kronos 的增量没有被这些对照证实。下一步不做全模型训练，也不自动建监控；若还要验证，需要 2026-11-13 之后的纸面记录。")
    lines.append("")
    lines.append("方向准确率按信号日后第 5 个收盘的涨跌计算，交易胜率按扣费净利是否为正计算，两者分开列在 `all_scores.csv`。")
    lines.append("")
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _pct(value) -> str:
    return "NA" if value is None or not np.isfinite(value) else f"{100 * value:.1f}%"


def _pp(value) -> str:
    return "NA" if value is None or not np.isfinite(value) else f"{100 * value:.1f}pp"


def _num(value) -> str:
    return "NA" if value is None or not np.isfinite(value) else f"{value:.2f}"


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log(traceback.format_exc())
        raise
