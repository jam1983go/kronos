"""Kronos forecasts with history-only normalization and sample dispersion.

`auto_regressive_inference_with_std` matches model/kronos.py at commit
67b630e except it keeps the cross-sample standard deviation. Smoke tests
compare its mean with the official KronosPredictor.predict mean.
"""

from __future__ import annotations

import random
from typing import Iterable

import numpy as np
import pandas as pd
import torch

from model.kronos import calc_time_stamps, sample_from_logits

from experiments.ashare_kronos.config import (
    CLIP,
    EVAL_ORIGIN_START,
    FLAT_PRICE_EPS,
    LOOKBACK,
    ORIGIN_STEP,
    PRED_LEN,
    SAMPLE_COUNT,
    SEED,
    TEMPERATURE,
    TOP_K,
    TOP_P,
)

PRICE_COLS = ["open", "high", "low", "close", "volume", "amount"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def auto_regressive_inference_with_std(
    tokenizer,
    model,
    x,
    x_stamp,
    y_stamp,
    max_context,
    pred_len,
    clip=5,
    T=1.0,
    top_k=0,
    top_p=0.99,
    sample_count=5,
    verbose=False,
):
    with torch.inference_mode():
        x = torch.clip(x, -clip, clip)
        device = x.device
        x = x.unsqueeze(1).repeat(1, sample_count, 1, 1).reshape(-1, x.size(1), x.size(2)).to(device)
        x_stamp = x_stamp.unsqueeze(1).repeat(1, sample_count, 1, 1).reshape(-1, x_stamp.size(1), x_stamp.size(2)).to(device)
        y_stamp = y_stamp.unsqueeze(1).repeat(1, sample_count, 1, 1).reshape(-1, y_stamp.size(1), y_stamp.size(2)).to(device)

        x_token = tokenizer.encode(x, half=True)
        initial_seq_len = x.size(1)
        batch_size = x_token[0].size(0)
        total_seq_len = initial_seq_len + pred_len
        full_stamp = torch.cat([x_stamp, y_stamp], dim=1)

        generated_pre = x_token[0].new_empty(batch_size, pred_len)
        generated_post = x_token[1].new_empty(batch_size, pred_len)
        pre_buffer = x_token[0].new_zeros(batch_size, max_context)
        post_buffer = x_token[1].new_zeros(batch_size, max_context)
        buffer_len = min(initial_seq_len, max_context)
        if buffer_len > 0:
            start_idx = max(0, initial_seq_len - max_context)
            pre_buffer[:, :buffer_len] = x_token[0][:, start_idx : start_idx + buffer_len]
            post_buffer[:, :buffer_len] = x_token[1][:, start_idx : start_idx + buffer_len]

        for i in range(pred_len):
            current_seq_len = initial_seq_len + i
            window_len = min(current_seq_len, max_context)
            if current_seq_len <= max_context:
                input_tokens = [pre_buffer[:, :window_len], post_buffer[:, :window_len]]
            else:
                input_tokens = [pre_buffer, post_buffer]
            context_end = current_seq_len
            context_start = max(0, context_end - max_context)
            current_stamp = full_stamp[:, context_start:context_end, :].contiguous()
            s1_logits, context = model.decode_s1(input_tokens[0], input_tokens[1], current_stamp)
            sample_pre = sample_from_logits(
                s1_logits[:, -1, :], temperature=T, top_k=top_k, top_p=top_p, sample_logits=True
            )
            s2_logits = model.decode_s2(context, sample_pre)
            sample_post = sample_from_logits(
                s2_logits[:, -1, :], temperature=T, top_k=top_k, top_p=top_p, sample_logits=True
            )
            generated_pre[:, i] = sample_pre.squeeze(-1)
            generated_post[:, i] = sample_post.squeeze(-1)
            if current_seq_len < max_context:
                pre_buffer[:, current_seq_len] = sample_pre.squeeze(-1)
                post_buffer[:, current_seq_len] = sample_post.squeeze(-1)
            else:
                pre_buffer.copy_(torch.roll(pre_buffer, shifts=-1, dims=1))
                post_buffer.copy_(torch.roll(post_buffer, shifts=-1, dims=1))
                pre_buffer[:, -1] = sample_pre.squeeze(-1)
                post_buffer[:, -1] = sample_post.squeeze(-1)

        full_pre = torch.cat([x_token[0], generated_pre], dim=1)
        full_post = torch.cat([x_token[1], generated_post], dim=1)
        context_start = max(0, total_seq_len - max_context)
        decoded = tokenizer.decode(
            [full_pre[:, context_start:total_seq_len].contiguous(), full_post[:, context_start:total_seq_len].contiguous()],
            half=True,
        )
        decoded = decoded.reshape(-1, sample_count, decoded.size(1), decoded.size(2))
        array = decoded.cpu().numpy()
        return array.mean(axis=1), array.std(axis=1)


def _normalize_history(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = df[PRICE_COLS].to_numpy(dtype=np.float32)
    if np.isnan(values).any():
        raise ValueError("NaN in history window")
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    norm = np.clip((values - mean) / (std + 1e-5), -CLIP, CLIP)
    return norm, mean, std


def predict_batch_with_std(predictor, frames: list[pd.DataFrame], x_timestamps: list[pd.Series], y_timestamps: list[pd.Series]):
    norms, means, stds, x_stamps, y_stamps = [], [], [], [], []
    for df, x_ts, y_ts in zip(frames, x_timestamps, y_timestamps):
        if len(df) != LOOKBACK:
            raise ValueError(f"history length {len(df)} != {LOOKBACK}")
        if pd.Timestamp(x_ts.max()) >= pd.Timestamp(y_ts.min()):
            raise ValueError("future bar entered the history window")
        if len(y_ts) != PRED_LEN:
            raise ValueError("prediction calendar length mismatch")
        norm, mean, std = _normalize_history(df)
        norms.append(norm)
        means.append(mean)
        stds.append(std)
        x_stamps.append(calc_time_stamps(pd.to_datetime(x_ts)).to_numpy(dtype=np.float32))
        y_stamps.append(calc_time_stamps(pd.to_datetime(y_ts)).to_numpy(dtype=np.float32))
    x = torch.from_numpy(np.stack(norms)).to(predictor.device)
    x_stamp = torch.from_numpy(np.stack(x_stamps)).to(predictor.device)
    y_stamp = torch.from_numpy(np.stack(y_stamps)).to(predictor.device)
    mean_norm, std_norm = auto_regressive_inference_with_std(
        predictor.tokenizer,
        predictor.model,
        x,
        x_stamp,
        y_stamp,
        predictor.max_context,
        PRED_LEN,
        CLIP,
        TEMPERATURE,
        TOP_K,
        TOP_P,
        SAMPLE_COUNT,
        False,
    )
    mean_norm = mean_norm[:, -PRED_LEN:, :]
    std_norm = std_norm[:, -PRED_LEN:, :]
    out_mean, out_std = [], []
    for i, (mean, std) in enumerate(zip(means, stds)):
        scale = std + 1e-5
        out_mean.append(mean_norm[i] * scale + mean)
        out_std.append(std_norm[i] * scale)
    return out_mean, out_std


def official_predict_one(predictor, df, x_ts, y_ts):
    set_seed(SEED)
    return predictor.predict(
        df=df[PRICE_COLS].reset_index(drop=True),
        x_timestamp=pd.to_datetime(x_ts).reset_index(drop=True),
        y_timestamp=pd.to_datetime(y_ts).reset_index(drop=True),
        pred_len=PRED_LEN,
        T=TEMPERATURE,
        top_k=TOP_K,
        top_p=TOP_P,
        sample_count=SAMPLE_COUNT,
        verbose=False,
    )


def _majority_direction(history_close: np.ndarray) -> int:
    rets = np.diff(history_close)
    ups = int((rets > FLAT_PRICE_EPS).sum())
    downs = int((rets < -FLAT_PRICE_EPS).sum())
    if ups > downs:
        return 1
    if downs > ups:
        return -1
    return 0


def iter_origins(qfq: pd.DataFrame, raw: pd.DataFrame) -> Iterable[dict]:
    if not qfq["date"].reset_index(drop=True).equals(raw["date"].reset_index(drop=True)):
        raise ValueError("qfq/raw calendars differ")
    bad = (
        (qfq[["open", "high", "low", "close"]] <= 0).any(axis=1)
        | qfq[PRICE_COLS].isna().any(axis=1)
        | (qfq["high"] + 1e-6 < qfq["low"])
        | (qfq["high"] + 1e-4 < qfq[["open", "close"]].max(axis=1))
        | (qfq["low"] - 1e-4 > qfq[["open", "close"]].min(axis=1))
    ).to_numpy()
    zero_vol = (qfq["volume"] <= 0).to_numpy()
    q_ret = qfq["close"].pct_change().abs().fillna(0).to_numpy()
    r_ret = raw["close"].pct_change().abs().fillna(0).to_numpy()
    glitch = (q_ret > 4) | (r_ret > 4) | ((q_ret > 0.25) & (r_ret > 0.25))
    dates = qfq["date"]
    start_cut = pd.Timestamp(EVAL_ORIGIN_START)
    last_origin = len(qfq) - 1 - PRED_LEN
    first = LOOKBACK - 1
    for origin in range(first, last_origin + 1, ORIGIN_STEP):
        if dates.iloc[origin] < start_cut:
            continue
        window = slice(origin - LOOKBACK + 1, origin + 1)
        horizon = slice(origin + 1, origin + 1 + PRED_LEN)
        reason = ""
        if bad[window].any() or bad[horizon].any() or glitch[window].any() or glitch[horizon].any():
            reason = "bad_price"
        elif zero_vol[origin] or zero_vol[horizon].any():
            reason = "zero_volume"
        yield {
            "origin": origin,
            "reason": reason,
            "input_start": dates.iloc[origin - LOOKBACK + 1],
            "origin_date": dates.iloc[origin],
            "y_dates": dates.iloc[horizon],
        }


def build_tasks(loaded: dict) -> tuple[list[dict], list[dict]]:
    kept, dropped = [], []
    for code, pack in loaded.items():
        qfq, raw, meta = pack["qfq"], pack["raw"], pack["meta"]
        for item in iter_origins(qfq, raw):
            row = {
                "code": code,
                "vendor_name": meta.get("vendor_name", ""),
                "industry": meta["industry"],
                "board": meta["board"],
                "limit": 0.05 if meta.get("st_in_name") else meta["limit"],
                **item,
            }
            if item["reason"]:
                dropped.append(row)
            else:
                kept.append(row)
    return kept, dropped
