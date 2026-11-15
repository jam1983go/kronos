"""Run the frozen ETF win-rate experiment.

From the repository root:

    PYTHONPATH=. /tmp/kronos-venv/bin/python -m experiments.ashare_kronos.winrate.run --stage all
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
import traceback
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from experiments.ashare_kronos.data_io import cross_check, fetch_eastmoney
from experiments.ashare_kronos.winrate.evaluate_winrate import (
    build_feature_table,
    diff_against_raw,
    finetune_gate,
    load_predictions,
    make_plots,
    portfolio_drawdown,
    score_window,
    slice_stability,
    trades_for,
    verdict_text,
    write_report,
)
from experiments.ashare_kronos.winrate.infer import load_predictor, official_mean_close, predict_slot, run_inference
from experiments.ashare_kronos.winrate.metrics import choose_modification
from experiments.ashare_kronos.winrate.settings import (
    AS_OF_DATE,
    CONFIGS,
    HISTORY_END,
    HISTORY_START,
    LIQUIDITY_END,
    LIQUIDITY_START,
    MIN_MEDIAN_AMOUNT,
    MODEL_ID,
    MODEL_REVISION,
    POOL,
    SEED,
)
from experiments.ashare_kronos.winrate.slots import build_slots

ROOT = Path(__file__).resolve().parents[3]
OLD_DATA = ROOT / "experiments" / "ashare_kronos" / "data"
DATA = Path(__file__).resolve().parent / "data"
OUT = Path(__file__).resolve().parent / "outputs"
TZ = ZoneInfo("Asia/Shanghai")


def log(message: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S%z")
    line = f"[{stamp}] {message}"
    print(line, flush=True)
    with (OUT / "run_log.txt").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def _load_pair(code: str) -> dict[str, pd.DataFrame]:
    qfq = pd.read_csv(DATA / f"{code}_qfq.csv", parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    raw = pd.read_csv(DATA / f"{code}_raw.csv", parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    qfq = qfq[qfq["date"] < pd.Timestamp(AS_OF_DATE)].reset_index(drop=True)
    raw = raw[raw["date"] < pd.Timestamp(AS_OF_DATE)].reset_index(drop=True)
    return {"qfq": qfq, "raw": raw}


def stage_fetch() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    for spec in POOL:
        code = spec["code"]
        for kind in ("qfq", "raw"):
            dest = DATA / f"{code}_{kind}.csv"
            old = OLD_DATA / f"{code}_{kind}.csv"
            if dest.exists():
                continue
            if old.exists():
                shutil.copy(old, dest)
                log(f"copied {dest.name} from the previous snapshot")
        if not (DATA / f"{code}_qfq.csv").exists():
            name, qfq = fetch_eastmoney(code, spec["market"], 1, HISTORY_START, HISTORY_END)
            _, raw = fetch_eastmoney(code, spec["market"], 0, HISTORY_START, HISTORY_END)
            qfq.to_csv(DATA / f"{code}_qfq.csv", index=False)
            raw.to_csv(DATA / f"{code}_raw.csv", index=False)
            (DATA / f"{code}_name.txt").write_text(name + "\n", encoding="utf-8")
            log(f"fetched {code} {name} {qfq['date'].min().date()} {qfq['date'].max().date()}")
            check = cross_check(code, spec["market"], qfq, raw)
            log(f"crosscheck {code} {check}")
    rows = []
    for spec in POOL:
        pack = _load_pair(spec["code"])
        hist = pack["qfq"][(pack["qfq"]["date"] >= LIQUIDITY_START) & (pack["qfq"]["date"] <= LIQUIDITY_END)]
        median_amount = float(hist["amount"].median())
        rows.append(
            {
                "code": spec["code"],
                "median_amount": median_amount,
                "pass": median_amount >= MIN_MEDIAN_AMOUNT,
                "n": int(len(hist)),
                "last_date": str(pack["qfq"]["date"].max().date()),
            }
        )
    screen = pd.DataFrame(rows)
    screen.to_csv(OUT / "liquidity.csv", index=False)
    log("liquidity\n" + screen.to_string(index=False))
    if screen["last_date"].max() < "2025-01-01":
        raise RuntimeError("vendor history ends before the revalidation window")


def load_panels() -> dict:
    screen = pd.read_csv(OUT / "liquidity.csv", dtype={"code": str})
    screen["code"] = screen["code"].str.zfill(6)
    passed = screen[screen["pass"]]["code"].tolist()
    if not passed:
        raise RuntimeError("no ETF passed the pre-registered liquidity screen")
    return {code: _load_pair(code) for code in passed}


def stage_smoke(predictor, panels, slots: pd.DataFrame) -> None:
    sample = slots[(slots["window"] == "selection") & slots["executable"]].iloc[0]
    qfq = panels[sample.code]["qfq"]
    samples = predict_slot(predictor, qfq, pd.Timestamp(sample.signal_date), 120, 5)
    official = official_mean_close(predictor, qfq, pd.Timestamp(sample.signal_date), 120, 5)
    diff = abs(float(samples.mean()) - official)
    log(f"smoke {sample.code} {pd.Timestamp(sample.signal_date).date()} mean {samples.mean():.6f} official {official:.6f} diff {diff:.8f}")
    if diff > 1e-3:
        raise RuntimeError(f"sample mean diverges from official predict by {diff}")


def stage_select(featured: pd.DataFrame, panels: dict) -> dict:
    if (OUT / "locked_choice.json").exists():
        lock = json.loads((OUT / "locked_choice.json").read_text(encoding="utf-8"))
        log(f"selection already locked: {lock['selected_id']}")
        return lock
    scores, trade_frames = score_window("selection", featured, panels, n_etfs=len(panels))
    scores = diff_against_raw(scores, trade_frames)
    pd.DataFrame(scores).to_csv(OUT / "selection_scores.csv", index=False)
    choice = choose_modification(scores)
    choice["locked_at"] = datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S%z")
    choice["config_ids"] = [config["id"] for config in CONFIGS]
    choice["seed"] = SEED
    (OUT / "locked_choice.json").write_text(json.dumps(choice, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"LOCKED {json.dumps(choice, ensure_ascii=False)}")
    return choice


def stage_report(featured: pd.DataFrame, panels: dict, n_params: int, forecast_seconds: float) -> None:
    lock = json.loads((OUT / "locked_choice.json").read_text(encoding="utf-8"))
    n_etfs = len(panels)
    all_scores = []
    all_trades = []
    curves = {}
    for window in ("selection", "confirmation", "revalidation"):
        scores, trade_frames = score_window(window, featured, panels, n_etfs)
        scores = diff_against_raw(scores, trade_frames)
        all_scores.extend(scores)
        all_trades.extend(trade_frames)
        if window == "revalidation":
            by_id = {frame["config"].iloc[0]: frame for frame in trade_frames if len(frame)}
            for config_id in ("kronos_lb120_s5_mean_gt0", "momentum", lock.get("selected_id")):
                if config_id and config_id in by_id:
                    _, curve = portfolio_drawdown(by_id[config_id], panels, n_etfs, 1.0)
                    curves[config_id] = curve
                    curve.to_csv(OUT / f"equity_{config_id}.csv", index=False)
    stress_scores, stress_trades = score_window("revalidation", featured, panels, n_etfs, cost_scale=2.0)
    stress_scores = diff_against_raw(stress_scores, stress_trades)
    for row in stress_scores:
        row["window"] = "revalidation_double_cost"
    all_scores.extend(stress_scores)
    pd.DataFrame(all_scores).to_csv(OUT / "all_scores.csv", index=False)
    if all_trades:
        pd.concat(all_trades, ignore_index=True).to_csv(OUT / "trades.csv", index=False)
    if stress_trades:
        pd.concat(stress_trades, ignore_index=True).to_csv(OUT / "trades_double_cost.csv", index=False)
    selected = lock.get("selected_id")
    revalidation_slots = featured[featured["window"] == "revalidation"]
    stability = []
    if selected and all_trades:
        stability = slice_stability(pd.concat(all_trades, ignore_index=True), revalidation_slots, selected)
        pd.DataFrame(stability).to_csv(OUT / "stability.csv", index=False)
    by_window = {}
    for window in ("selection", "confirmation", "revalidation"):
        by_window[window] = [row for row in all_scores if row["window"] == window]
    rev_row = next((row for row in by_window["revalidation"] if row["id"] == selected), None) if selected else None
    payload = {
        "lock": lock,
        "selection": by_window["selection"],
        "confirmation": by_window["confirmation"],
        "revalidation": by_window["revalidation"],
        "stress": stress_scores,
        "stability": stability,
        "finetune": finetune_gate(rev_row, stability),
        "verdict": verdict_text(lock, by_window["revalidation"], stress_scores),
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "n_params": n_params,
        "pool": list(panels),
        "forecast_seconds": round(forecast_seconds, 1),
    }
    (OUT / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    make_plots(pd.DataFrame(by_window["selection"]), pd.DataFrame(by_window["revalidation"]), curves, OUT / "figures")
    write_report(OUT / "REPORT.md", payload)
    log(payload["verdict"])
    log(f"finetune_gate {payload['finetune']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["fetch", "smoke", "infer", "select", "report", "all"], default="all")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    log(f"stage {args.stage} configs {len(CONFIGS)} weights_not_updated")
    started = time.perf_counter()
    try:
        if args.stage in {"fetch", "all"} or not (OUT / "liquidity.csv").exists():
            stage_fetch()
        panels = load_panels()
        slots_path = OUT / "slots.csv"
        if args.stage in {"fetch", "smoke", "infer", "all"} or not slots_path.exists():
            slots = build_slots(panels)
            slots.to_csv(slots_path, index=False)
            log(slots.groupby(["window", "executable"]).size().to_string())
        else:
            slots = pd.read_csv(slots_path, dtype={"code": str}, parse_dates=["signal_date", "entry_date", "exit_date"])
            slots["code"] = slots["code"].str.zfill(6)
        predictor = None
        n_params = 0
        if args.stage in {"smoke", "infer", "all"}:
            predictor, n_params = load_predictor()
            log(f"loaded frozen weights params {n_params}")
            (OUT / "model_manifest.json").write_text(
                json.dumps({"model": MODEL_ID, "revision": MODEL_REVISION, "n_params": n_params, "weights_updated": False}, indent=2),
                encoding="utf-8",
            )
        if args.stage in {"smoke", "all"}:
            stage_smoke(predictor, panels, slots)
        infer_started = time.perf_counter()
        if args.stage in {"infer", "all"}:
            run_inference(predictor, panels, slots, OUT, log)
        forecast_seconds = time.perf_counter() - infer_started if args.stage in {"infer", "all"} else _logged_forecast_seconds()
        if args.stage in {"select", "report", "all"}:
            featured = build_feature_table(slots, load_predictions(OUT))
            stage_select(featured, panels)
        if args.stage in {"report", "all"}:
            if "featured" not in locals():
                featured = build_feature_table(slots, load_predictions(OUT))
            stage_report(featured, panels, n_params or json.loads((OUT / "model_manifest.json").read_text())["n_params"], forecast_seconds)
        log(f"done elapsed {time.perf_counter() - started:.1f}s")
    except Exception:
        log(traceback.format_exc())
        raise


def _logged_forecast_seconds() -> float:
    path = OUT / "run_log.txt"
    if not path.exists():
        return float("nan")
    head = path.read_text(encoding="utf-8").split("stage select")[0]
    matches = []
    for line in head.splitlines():
        if "done elapsed" in line:
            matches.append(float(line.split("done elapsed", 1)[1].split("s", 1)[0].strip()))
    return matches[-1] if matches else float("nan")


if __name__ == "__main__":
    main()
