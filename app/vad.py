"""人聲偵測（Silero VAD，CPU 執行）與切段。

切段原則：只在沒人講話的地方切，每段不超過辨識模型的上限，
中間若有超過 GAP_SPLIT_S 秒的空白就另起一段，避免把長時間的音樂或靜音送進模型。
"""
import os
from typing import Callable

import numpy as np

from .config import SAMPLE_RATE

GAP_SPLIT_S = 3.0
CHUNK_PAD_S = 0.3

PRESETS = {
    "normal": dict(threshold=0.5, min_speech_duration_ms=250, min_silence_duration_ms=350, speech_pad_ms=150),
    # 耳語、ASMR：門檻調低，避免把小聲的句子當成沒聲音
    "sensitive": dict(threshold=0.3, neg_threshold=0.15, min_speech_duration_ms=120,
                      min_silence_duration_ms=500, speech_pad_ms=250),
}


def speech_segments(wav: np.ndarray, sensitive: bool, max_speech_s: float,
                    on_progress: Callable[[float], None] | None = None,
                    check: Callable[[], None] | None = None) -> list[list[float]]:
    import torch
    from silero_vad import get_speech_timestamps, load_silero_vad

    torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))
    model = load_silero_vad()

    def cb(percent):
        if check:
            check()
        if on_progress:
            on_progress(percent / 100.0)

    params = PRESETS["sensitive" if sensitive else "normal"]
    ts = get_speech_timestamps(
        torch.from_numpy(wav), model, sampling_rate=SAMPLE_RATE, return_seconds=True,
        max_speech_duration_s=max_speech_s, progress_tracking_callback=cb, **params,
    )
    return [[float(t["start"]), float(t["end"])] for t in ts]


MIN_COVERAGE = 0.25


def coverage(segments: list[list[float]], total_s: float) -> float:
    if total_s <= 0:
        return 0.0
    return sum(e - s for s, e in segments) / total_s


def fixed_chunks(total_s: float, max_chunk_s: float) -> list[list[float]]:
    """整段平均切。用在 VAD 幾乎抓不到人聲的情況（例如配樂很滿的歌曲）。"""
    step = max(10.0, max_chunk_s)
    out, t = [], 0.0
    while t < total_s:
        out.append([round(t, 3), round(min(total_s, t + step), 3)])
        t += step
    return out


def build_chunks(segments: list[list[float]], total_s: float, max_chunk_s: float) -> list[list[float]]:
    chunks: list[list[float]] = []
    cur = None
    for start, end in segments:
        if cur is not None and end - cur[0] <= max_chunk_s and start - cur[1] <= GAP_SPLIT_S:
            cur[1] = end
            continue
        if cur is not None:
            chunks.append(cur)
        cur = [start, end]
    if cur is not None:
        chunks.append(cur)

    padded = []
    prev_end = 0.0
    for start, end in chunks:
        s = max(prev_end, start - CHUNK_PAD_S, 0.0)
        e = min(total_s, end + CHUNK_PAD_S)
        padded.append([round(s, 3), round(e, 3)])
        prev_end = e
    return padded
