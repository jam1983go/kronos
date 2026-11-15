"""Inference-only forecasts. Pretrained weights are loaded and not updated."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from model.kronos import calc_time_stamps, sample_from_logits

from experiments.ashare_kronos.winrate.settings import (
    CLIP,
    MAX_CONTEXT,
    MODEL_ID,
    MODEL_REVISION,
    PRED_LEN,
    SEED,
    TEMPERATURE,
    TOKENIZER_ID,
    TOKENIZER_REVISION,
    TOP_K,
    TOP_P,
)

PRICE_COLS = ["open", "high", "low", "close", "volume", "amount"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_predictor():
    from model import Kronos, KronosPredictor, KronosTokenizer

    torch.set_num_threads(4)
    tokenizer = KronosTokenizer.from_pretrained(TOKENIZER_ID, revision=TOKENIZER_REVISION)
    model = Kronos.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    model.eval()
    tokenizer.eval()
    predictor = KronosPredictor(model, tokenizer, device="cpu", max_context=MAX_CONTEXT)
    n_params = sum(param.numel() for param in model.parameters())
    return predictor, n_params


def _forecast_samples(predictor, history: pd.DataFrame, x_ts: pd.Series, y_ts: pd.Series, sample_count: int) -> np.ndarray:
    """Return denormalized close samples for the last predicted bar. History only is normalized."""
    if len(history) != len(x_ts) or pd.Timestamp(x_ts.max()) >= pd.Timestamp(y_ts.min()):
        raise ValueError("future prices entered the normalization window")
    values = history[PRICE_COLS].to_numpy(dtype=np.float32)
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    norm = np.clip((values - mean) / (std + 1e-5), -CLIP, CLIP)
    x = torch.from_numpy(norm[np.newaxis]).to(predictor.device)
    x_stamp = torch.from_numpy(calc_time_stamps(pd.to_datetime(x_ts)).to_numpy(dtype=np.float32)[np.newaxis]).to(predictor.device)
    y_stamp = torch.from_numpy(calc_time_stamps(pd.to_datetime(y_ts)).to_numpy(dtype=np.float32)[np.newaxis]).to(predictor.device)
    with torch.inference_mode():
        x_clip = torch.clip(x, -CLIP, CLIP)
        x_rep = x_clip.unsqueeze(1).repeat(1, sample_count, 1, 1).reshape(-1, x_clip.size(1), x_clip.size(2))
        x_stamp_rep = x_stamp.unsqueeze(1).repeat(1, sample_count, 1, 1).reshape(-1, x_stamp.size(1), x_stamp.size(2))
        y_stamp_rep = y_stamp.unsqueeze(1).repeat(1, sample_count, 1, 1).reshape(-1, y_stamp.size(1), y_stamp.size(2))
        tokens = predictor.tokenizer.encode(x_rep, half=True)
        initial = x_rep.size(1)
        batch = tokens[0].size(0)
        total = initial + PRED_LEN
        full_stamp = torch.cat([x_stamp_rep, y_stamp_rep], dim=1)
        generated_pre = tokens[0].new_empty(batch, PRED_LEN)
        generated_post = tokens[1].new_empty(batch, PRED_LEN)
        pre_buffer = tokens[0].new_zeros(batch, MAX_CONTEXT)
        post_buffer = tokens[1].new_zeros(batch, MAX_CONTEXT)
        buffer_len = min(initial, MAX_CONTEXT)
        start_idx = max(0, initial - MAX_CONTEXT)
        pre_buffer[:, :buffer_len] = tokens[0][:, start_idx : start_idx + buffer_len]
        post_buffer[:, :buffer_len] = tokens[1][:, start_idx : start_idx + buffer_len]
        for step in range(PRED_LEN):
            current = initial + step
            window_len = min(current, MAX_CONTEXT)
            if current <= MAX_CONTEXT:
                input_tokens = [pre_buffer[:, :window_len], post_buffer[:, :window_len]]
            else:
                input_tokens = [pre_buffer, post_buffer]
            context_start = max(0, current - MAX_CONTEXT)
            stamp = full_stamp[:, context_start:current, :].contiguous()
            s1_logits, context = predictor.model.decode_s1(input_tokens[0], input_tokens[1], stamp)
            sample_pre = sample_from_logits(s1_logits[:, -1, :], temperature=TEMPERATURE, top_k=TOP_K, top_p=TOP_P, sample_logits=True)
            s2_logits = predictor.model.decode_s2(context, sample_pre)
            sample_post = sample_from_logits(s2_logits[:, -1, :], temperature=TEMPERATURE, top_k=TOP_K, top_p=TOP_P, sample_logits=True)
            generated_pre[:, step] = sample_pre.squeeze(-1)
            generated_post[:, step] = sample_post.squeeze(-1)
            if current < MAX_CONTEXT:
                pre_buffer[:, current] = sample_pre.squeeze(-1)
                post_buffer[:, current] = sample_post.squeeze(-1)
            else:
                pre_buffer.copy_(torch.roll(pre_buffer, shifts=-1, dims=1))
                post_buffer.copy_(torch.roll(post_buffer, shifts=-1, dims=1))
                pre_buffer[:, -1] = sample_pre.squeeze(-1)
                post_buffer[:, -1] = sample_post.squeeze(-1)
        full_pre = torch.cat([tokens[0], generated_pre], dim=1)
        full_post = torch.cat([tokens[1], generated_post], dim=1)
        context_start = max(0, total - MAX_CONTEXT)
        decoded = predictor.tokenizer.decode(
            [full_pre[:, context_start:total].contiguous(), full_post[:, context_start:total].contiguous()],
            half=True,
        )
        decoded = decoded.reshape(sample_count, decoded.size(1), decoded.size(2))
        future_close = decoded[:, -1, 3].detach().cpu().numpy()
    return future_close * (std[3] + 1e-5) + mean[3]


def predict_slot(predictor, qfq: pd.DataFrame, signal_date: pd.Timestamp, lookback: int, sample_count: int) -> np.ndarray:
    signal_i = int(qfq.index[qfq["date"] == pd.Timestamp(signal_date)][0])
    hist = qfq.iloc[signal_i - lookback + 1 : signal_i + 1].reset_index(drop=True)
    future = qfq.iloc[signal_i + 1 : signal_i + 1 + PRED_LEN].reset_index(drop=True)
    set_seed(SEED)
    return _forecast_samples(predictor, hist[PRICE_COLS], hist["date"], future["date"], sample_count)


def official_mean_close(predictor, qfq: pd.DataFrame, signal_date: pd.Timestamp, lookback: int, sample_count: int) -> float:
    from model import KronosPredictor

    assert isinstance(predictor, KronosPredictor)
    signal_i = int(qfq.index[qfq["date"] == pd.Timestamp(signal_date)][0])
    hist = qfq.iloc[signal_i - lookback + 1 : signal_i + 1].reset_index(drop=True)
    future = qfq.iloc[signal_i + 1 : signal_i + 1 + PRED_LEN].reset_index(drop=True)
    set_seed(SEED)
    pred = predictor.predict(
        df=hist[PRICE_COLS],
        x_timestamp=hist["date"],
        y_timestamp=future["date"],
        pred_len=PRED_LEN,
        T=TEMPERATURE,
        top_k=TOP_K,
        top_p=TOP_P,
        sample_count=sample_count,
        verbose=False,
    )
    return float(pred["close"].iloc[-1])


def run_inference(predictor, panels: dict, slots: pd.DataFrame, out_dir: Path, log) -> None:
    needed = slots[slots["window"].isin(["selection", "confirmation", "revalidation"])]
    specs = {(120, 5), (120, 20), (60, 5), (240, 5)}
    for lookback, samples in sorted(specs):
        path = out_dir / f"preds_lb{lookback}_s{samples}.csv"
        done = set()
        rows = []
        if path.exists():
            old = pd.read_csv(path, dtype={"code": str})
            old["code"] = old["code"].str.zfill(6)
            done = set(zip(old["code"], old["signal_date"]))
            rows.extend(old.to_dict("records"))
            log(f"resume lb{lookback} s{samples} existing {len(done)}")
        pending = []
        for slot in needed.itertuples(index=False):
            key = (slot.code, pd.Timestamp(slot.signal_date).strftime("%Y-%m-%d"))
            if key not in done:
                pending.append(slot)
        log(f"infer lb{lookback} s{samples} pending {len(pending)}")
        for index, slot in enumerate(pending, start=1):
            qfq = panels[slot.code]["qfq"]
            samples_close = predict_slot(predictor, qfq, pd.Timestamp(slot.signal_date), lookback, samples)
            row = {
                "code": slot.code,
                "signal_date": pd.Timestamp(slot.signal_date).strftime("%Y-%m-%d"),
                "lookback": lookback,
                "samples": samples,
                "origin_close": float(slot.signal_close_qfq),
            }
            for sample_i, value in enumerate(samples_close):
                row[f"pred_close_{sample_i}"] = float(value)
            rows.append(row)
            if index % 25 == 0 or index == len(pending):
                pd.DataFrame(rows).to_csv(path, index=False)
                log(f"infer lb{lookback} s{samples} {index}/{len(pending)}")
        pd.DataFrame(rows).to_csv(path, index=False)
