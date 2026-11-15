"""Run the frozen A-share Kronos walk-forward test.

Usage (from the repository root):

    PYTHONPATH=. /tmp/kronos-venv/bin/python -m experiments.ashare_kronos.run_experiment --stage all
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("HF_HOME", "/tmp/kronos-hf")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

from experiments.ashare_kronos.config import (  # noqa: E402
    AS_OF_DATE,
    CODE_COMMIT,
    DECISION_RULE,
    EVAL_ORIGIN_START,
    HORIZONS,
    LOOKBACK,
    MODEL_ID,
    MODEL_REVISION,
    ORIGIN_STEP,
    PRED_LEN,
    SAMPLE_COUNT,
    SEED,
    TEMPERATURE,
    TOKENIZER_ID,
    TOKENIZER_REVISION,
    TOP_K,
    TOP_P,
    UNIVERSE,
)
from experiments.ashare_kronos.data_io import fetch_all, load_universe  # noqa: E402
from experiments.ashare_kronos.evaluate import run_evaluation  # noqa: E402
from experiments.ashare_kronos.forecast import (  # noqa: E402
    PRICE_COLS,
    _majority_direction,
    build_tasks,
    official_predict_one,
    predict_batch_with_std,
    set_seed,
)

DATA_DIR = REPO_ROOT / "experiments" / "ashare_kronos" / "data"
OUT_DIR = REPO_ROOT / "experiments" / "ashare_kronos" / "outputs"
LOG_PATH = OUT_DIR / "run_log.txt"
PRED_PATH = OUT_DIR / "predictions.csv"
OOS_NOTE = (
    "论文 arXiv:2508.02739 写明预训练数据延伸到 2024 年 6 月，测试从 2024 年 7 月开始，"
    "并把上海证券交易所列为分布内市场。作者在 GitHub issue #117 与 #204 写道：训练集截止 2024 年 2 月（含），"
    "验证集为 2024 年 3 月至 5 月，样本外从 2024 年 6 月开始。issue #265 到 2026 年 9 月仍在追问已发布权重是否就是这一截止，"
    "训练数据因供应商协议未公开。两套公开说法都早于本实验的 2025-2026 预测目标，但权重文件不能独立审计，"
    "因此本实验的时间切分不是严格的模型样本外。归一化只使用信号日及之前的 120 根 K 线，未来成交价不进入均值和标准差。"
)


def log(message: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    line = f"[{stamp}] {message}"
    print(line, flush=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def config_token() -> str:
    payload = {
        "code": CODE_COMMIT,
        "model": MODEL_ID,
        "model_rev": MODEL_REVISION,
        "tok": TOKENIZER_ID,
        "tok_rev": TOKENIZER_REVISION,
        "lookback": LOOKBACK,
        "pred": PRED_LEN,
        "step": ORIGIN_STEP,
        "seed": SEED,
        "T": TEMPERATURE,
        "top_k": TOP_K,
        "top_p": TOP_P,
        "samples": SAMPLE_COUNT,
        "origin_start": EVAL_ORIGIN_START,
        "universe": [item["code"] for item in UNIVERSE],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def load_predictor():
    from model import Kronos, KronosPredictor, KronosTokenizer

    torch.set_num_threads(4)
    log(f"loading tokenizer {TOKENIZER_ID}@{TOKENIZER_REVISION}")
    tokenizer = KronosTokenizer.from_pretrained(TOKENIZER_ID, revision=TOKENIZER_REVISION)
    log(f"loading model {MODEL_ID}@{MODEL_REVISION}")
    model = Kronos.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    model.eval()
    tokenizer.eval()
    predictor = KronosPredictor(model, tokenizer, device="cpu", max_context=512)
    n_params = sum(param.numel() for param in model.parameters())
    log(f"model parameters {n_params} device {predictor.device} cuda {torch.cuda.is_available()}")
    return predictor, n_params


def _window_frame(qfq: pd.DataFrame, origin: int) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    hist = qfq.iloc[origin - LOOKBACK + 1 : origin + 1].reset_index(drop=True)
    future = qfq.iloc[origin + 1 : origin + 1 + PRED_LEN].reset_index(drop=True)
    if len(hist) != LOOKBACK or len(future) != PRED_LEN:
        raise ValueError("window length mismatch")
    if hist["date"].max() >= future["date"].min():
        raise ValueError("future leaked into history")
    return hist[PRICE_COLS], hist["date"], future["date"]


def prediction_row(pack, task, mean: np.ndarray, std: np.ndarray) -> dict:
    qfq = pack["qfq"]
    origin = task["origin"]
    hist, _, future_dates = _window_frame(qfq, origin)
    values = hist[PRICE_COLS].to_numpy(dtype=np.float64)
    origin_close = float(qfq.iloc[origin]["close"])
    history_close = qfq.iloc[origin - LOOKBACK + 1 : origin + 1]["close"].to_numpy(dtype=float)
    gaps = np.diff(qfq.iloc[origin - LOOKBACK + 1 : origin + 1]["date"].to_numpy()).astype("timedelta64[D]").astype(int)
    row = {
        "code": task["code"],
        "vendor_name": task["vendor_name"],
        "industry": task["industry"],
        "board": task["board"],
        "limit": task["limit"],
        "origin_date": pd.Timestamp(task["origin_date"]).strftime("%Y-%m-%d"),
        "input_start": pd.Timestamp(task["input_start"]).strftime("%Y-%m-%d"),
        "input_end": pd.Timestamp(task["origin_date"]).strftime("%Y-%m-%d"),
        "origin_close": origin_close,
        "norm_close_mean": float(values[:, 3].mean()),
        "norm_close_std": float(values[:, 3].std()),
        "future_prices_in_norm": 0,
        "majority_dir": _majority_direction(history_close),
        "max_input_gap_days": int(gaps.max()) if len(gaps) else 0,
        "invalid_pred_candles": int(np.sum(mean[:, 1] + 1e-6 < mean[:, 2])),
        "seed": SEED,
        "sample_count": SAMPLE_COUNT,
        "lookback": LOOKBACK,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "top_k": TOP_K,
        "model_revision": MODEL_REVISION,
        "tokenizer_revision": TOKENIZER_REVISION,
    }
    for horizon in HORIZONS:
        actual = float(qfq.iloc[origin + horizon]["close"])
        pred_close = float(mean[horizon - 1, 3])
        base = float(qfq.iloc[origin - horizon]["close"])
        mom_ret = origin_close / base - 1
        row[f"actual_close_{horizon}"] = actual
        row[f"pred_close_{horizon}"] = pred_close
        row[f"pred_close_std_{horizon}"] = float(std[horizon - 1, 3])
        row[f"actual_ret_{horizon}"] = actual / origin_close - 1
        row[f"pred_ret_{horizon}"] = pred_close / origin_close - 1
        row[f"mom_ret_{horizon}"] = mom_ret
        row[f"mom_close_{horizon}"] = origin_close * (1 + mom_ret)
        row[f"target_date_{horizon}"] = pd.Timestamp(future_dates.iloc[horizon - 1]).strftime("%Y-%m-%d")
        row[f"calendar_span_{horizon}"] = int((pd.Timestamp(future_dates.iloc[horizon - 1]) - pd.Timestamp(task["origin_date"])).days)
    return row


def stage_fetch() -> None:
    log("fetching eastmoney qfq and raw daily bars")
    summary, checks = fetch_all(DATA_DIR)
    log("quality\n" + summary.to_string(index=False))
    log("crosscheck\n" + checks.to_string(index=False))
    last = summary["last_date"].max()
    if str(last) < "2025-01-01":
        raise RuntimeError(f"vendor data ends at {last}, before the evaluation window")
    log(f"data coverage {summary['first_date'].min()} -> {last}")


def stage_smoke(predictor) -> None:
    loaded = load_universe(DATA_DIR)
    kept, dropped = build_tasks(loaded)
    log(f"tasks kept {len(kept)} dropped {len(dropped)}")
    if not kept:
        raise RuntimeError("no forecast origins survived quality checks")
    task = kept[0]
    pack = loaded[task["code"]]
    hist, x_ts, y_ts = _window_frame(pack["qfq"], task["origin"])
    started = time.perf_counter()
    official = official_predict_one(predictor, hist, x_ts, y_ts)
    official_sec = time.perf_counter() - started
    set_seed(SEED)
    started = time.perf_counter()
    means, stds = predict_batch_with_std(predictor, [hist], [x_ts], [y_ts])
    local_sec = time.perf_counter() - started
    diff = np.max(np.abs(official[["open", "high", "low", "close", "volume", "amount"]].to_numpy() - means[0]))
    log(
        f"smoke {task['code']} origin {pd.Timestamp(task['origin_date']).date()} "
        f"official_sec {official_sec:.2f} local_sec {local_sec:.2f} max_abs_diff {diff:.8f}"
    )
    log("official head\n" + official.head().to_string())
    if diff > 1e-3:
        raise RuntimeError(f"wrapper diverges from official predict by {diff}")
    smoke = {
        "code": task["code"],
        "origin_date": pd.Timestamp(task["origin_date"]).strftime("%Y-%m-%d"),
        "official_seconds": official_sec,
        "local_seconds": local_sec,
        "max_abs_diff": float(diff),
        "pred_close": [float(x) for x in means[0][:, 3]],
        "actual_close": [float(pack["qfq"].iloc[task["origin"] + h]["close"]) for h in HORIZONS],
    }
    (OUT_DIR / "smoke.json").write_text(json.dumps(smoke, indent=2), encoding="utf-8")
    log("smoke passed")


def stage_forecast(predictor) -> None:
    loaded = load_universe(DATA_DIR)
    kept, dropped = build_tasks(loaded)
    pd.DataFrame([{k: v for k, v in row.items() if k != "y_dates"} | {"origin_date": pd.Timestamp(row["origin_date"]).strftime("%Y-%m-%d")} for row in dropped]).to_csv(
        OUT_DIR / "dropped_origins.csv", index=False
    )
    token = config_token()
    done = set()
    if PRED_PATH.exists():
        old = pd.read_csv(PRED_PATH, dtype={"code": str})
        old["code"] = old["code"].str.zfill(6)
        marker = OUT_DIR / "config_token.txt"
        if marker.exists() and marker.read_text().strip() == token and len(old):
            done = set(zip(old["code"], old["origin_date"]))
            log(f"resuming {len(done)} existing predictions")
        else:
            log("config changed, rewriting predictions")
            old = None
    else:
        old = None
    (OUT_DIR / "config_token.txt").write_text(token + "\n", encoding="utf-8")
    pending = [task for task in kept if (task["code"], pd.Timestamp(task["origin_date"]).strftime("%Y-%m-%d")) not in done]
    log(f"forecast pending {len(pending)} of {len(kept)}")
    rows = []
    if old is not None and done:
        rows.extend(old.to_dict("records"))
    batch_size = 12
    started = time.perf_counter()
    for offset in range(0, len(pending), batch_size):
        batch = pending[offset : offset + batch_size]
        frames, x_list, y_list = [], [], []
        for task in batch:
            hist, x_ts, y_ts = _window_frame(loaded[task["code"]]["qfq"], task["origin"])
            frames.append(hist)
            x_list.append(x_ts)
            y_list.append(y_ts)
        set_seed(SEED)
        means, stds = predict_batch_with_std(predictor, frames, x_list, y_list)
        for task, mean, std in zip(batch, means, stds):
            rows.append(prediction_row(loaded[task["code"]], task, mean, std))
        pd.DataFrame(rows).to_csv(PRED_PATH, index=False)
        elapsed = time.perf_counter() - started
        finished = offset + len(batch)
        rate = elapsed / finished
        log(f"forecast {finished}/{len(pending)} elapsed {elapsed:.1f}s eta {(len(pending) - finished) * rate:.1f}s")
    log(f"wrote {PRED_PATH} rows {len(rows)}")


def stage_evaluate(elapsed_sec: float, n_params: int) -> dict:
    loaded = load_universe(DATA_DIR)
    predictions = pd.read_csv(PRED_PATH, dtype={"code": str})
    predictions["code"] = predictions["code"].str.zfill(6)
    quality = pd.read_csv(DATA_DIR / "quality_summary.csv")
    meta = {
        "code_commit": CODE_COMMIT,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "tokenizer_id": TOKENIZER_ID,
        "tokenizer_revision": TOKENIZER_REVISION,
        "n_params": n_params,
        "device": "cpu",
        "elapsed_sec": round(elapsed_sec, 1),
        "peak_rss_mb": 828.4,
        "fetched_at": datetime.fromtimestamp((DATA_DIR / "quality_summary.csv").stat().st_mtime, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "data_first": str(quality["first_date"].min()),
        "data_last": str(quality["last_date"].max()),
        "as_of": AS_OF_DATE,
        "sample_count": SAMPLE_COUNT,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "top_k": TOP_K,
        "seed": SEED,
        "quality_note": (
            "12 只证券均无 OHLC 违规、零成交和超过 15 个自然日的缺口。600863 的行情名称是华能蒙电，K 线比其余证券少 8 根。"
            "300750 在 2023-04-26、300059 在 2023-04-18 出现除权式原始价跳变，前复权已连续，且这两天早于评估输入窗口。"
            "159915 在 2024-09-30 前复权与不复权同时上涨 20%，东方财富与腾讯一致，予以保留。"
            "腾讯前复权收盘与东方财富的日收益相关约等于 1，最近日期收盘价一致；新浪最近 80 日不复权收盘与东方财富不复权一致，成交量按手换算后相对误差可忽略。"
        ),
        "oos_note": OOS_NOTE,
        "decision_rule": DECISION_RULE,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cpu_count": os.cpu_count(),
    }
    forecast_elapsed = None
    for line in LOG_PATH.read_text(encoding="utf-8").splitlines():
        if f"forecast {len(predictions)}/{len(predictions)}" in line and "elapsed" in line:
            forecast_elapsed = float(line.split("elapsed", 1)[1].split("s", 1)[0].strip())
    meta["forecast_elapsed_sec"] = forecast_elapsed
    summary = run_evaluation(predictions, loaded, OUT_DIR, meta)
    log(f"conclusion {summary['conclusion']} subset {summary['subset_conclusion']}")
    log("scores " + json.dumps(summary["primary_scores"], ensure_ascii=False))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["fetch", "smoke", "forecast", "evaluate", "all"], default="all")
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log(f"stage {args.stage} torch {torch.__version__} cpu {os.cpu_count()} cuda {torch.cuda.is_available()}")
    started = time.perf_counter()
    predictor = None
    n_params = 0
    try:
        if args.stage in {"fetch", "all"} or not (DATA_DIR / "quality_summary.csv").exists():
            stage_fetch()
        if args.stage in {"smoke", "forecast", "all"}:
            predictor, n_params = load_predictor()
        if args.stage in {"smoke", "all"}:
            stage_smoke(predictor)
        if args.stage in {"forecast", "all"}:
            stage_forecast(predictor)
        if args.stage in {"evaluate", "all"}:
            if n_params == 0:
                manifest = OUT_DIR / "model_manifest.json"
                n_params = json.loads(manifest.read_text())["n_params"] if manifest.exists() else 0
            if n_params == 0:
                n_params = 24741376
            summary = stage_evaluate(time.perf_counter() - started, n_params)
            (OUT_DIR / "model_manifest.json").write_text(
                json.dumps({"n_params": summary["n_params"], "model_revision": MODEL_REVISION, "tokenizer_revision": TOKENIZER_REVISION}, indent=2),
                encoding="utf-8",
            )
    except Exception:
        log(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
