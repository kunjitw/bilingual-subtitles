"""語音辨識功能測試：確認模型能在本機 GPU 跑，並記錄顯存與耗時。

用法：
  python -s tests/smoke_asr.py <影片> <Japanese|English|Chinese> <qwen|anime>

辨識模型與對齊器依序載入，同一時間只有一個模型在顯存中。
ffmpeg 用 app/config.py 的設定（可以用環境變數 VS_FFMPEG 指定）；結果存到系統暫存資料夾的 vs-smoke。
"""
import gc
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.config import FFMPEG, MODELS_DIR  # noqa: E402

MODELS = MODELS_DIR / "asr"
OUT = Path(tempfile.gettempdir()) / "vs-smoke"
SR = 16000


def extract_wav(media: Path) -> np.ndarray:
    with tempfile.TemporaryDirectory() as td:
        wav_path = Path(td) / "a.wav"
        subprocess.run(
            [FFMPEG, "-y", "-loglevel", "error", "-i", str(media), "-vn", "-ac", "1", "-ar", str(SR), str(wav_path)],
            check=True,
        )
        wav, _ = sf.read(wav_path, dtype="float32")
    return wav


def free_gpu():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def gpu_peak_mb() -> int:
    return int(torch.cuda.max_memory_allocated() / 1024**2)


def run_qwen(wav: np.ndarray, language: str) -> dict:
    from qwen_asr import Qwen3ASRModel, Qwen3ForcedAligner

    report = {}

    # 階段 1：辨識
    t0 = time.time()
    model = Qwen3ASRModel.from_pretrained(
        str(MODELS / "Qwen3-ASR-1.7B"),
        dtype=torch.bfloat16,
        device_map="cuda:0",
        attn_implementation="sdpa",
        max_inference_batch_size=8,
        max_new_tokens=2048,
    )
    t_load = time.time()
    res = model.transcribe(audio=(wav, SR), language=language)[0]
    t_done = time.time()
    text = res.text
    report["asr"] = {
        "load_s": round(t_load - t0, 1),
        "infer_s": round(t_done - t_load, 1),
        "peak_vram_mb": gpu_peak_mb(),
        "chars": len(text),
    }
    del model
    free_gpu()

    # 階段 2：對齊（辨識模型已卸載）
    t0 = time.time()
    aligner = Qwen3ForcedAligner.from_pretrained(
        str(MODELS / "Qwen3-ForcedAligner-0.6B"),
        dtype=torch.bfloat16,
        device_map="cuda:0",
        attn_implementation="sdpa",
    )
    t_load = time.time()
    items = aligner.align(audio=(wav, SR), text=text, language=language)[0].items
    t_done = time.time()
    report["align"] = {
        "load_s": round(t_load - t0, 1),
        "infer_s": round(t_done - t_load, 1),
        "peak_vram_mb": gpu_peak_mb(),
        "units": len(items),
        "first": [(i.text, i.start_time, i.end_time) for i in items[:8]],
        "last": [(i.text, i.start_time, i.end_time) for i in items[-3:]],
    }
    del aligner
    free_gpu()
    return report, text


def run_anime(wav: np.ndarray, language: str) -> dict:
    from transformers import pipeline

    t0 = time.time()
    pipe = pipeline(
        "automatic-speech-recognition",
        model=str(MODELS / "anime-whisper"),
        device="cuda",
        torch_dtype=torch.float16,
        chunk_length_s=30.0,
        batch_size=16,
    )
    t_load = time.time()
    out = pipe({"raw": wav, "sampling_rate": SR}, generate_kwargs={"language": language.lower(), "task": "transcribe"})
    t_done = time.time()
    text = out["text"]
    report = {
        "asr": {
            "load_s": round(t_load - t0, 1),
            "infer_s": round(t_done - t_load, 1),
            "peak_vram_mb": gpu_peak_mb(),
            "chars": len(text),
        }
    }
    del pipe
    free_gpu()
    return report, text


def main():
    if len(sys.argv) < 4:
        raise SystemExit(__doc__)
    media, language, engine = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
    if not media.is_file():
        raise SystemExit(f"找不到影片：{media}")
    if not (Path(FFMPEG).is_file() or shutil.which(FFMPEG)):
        raise SystemExit(f"找不到 ffmpeg：{FFMPEG}。請把環境變數 VS_FFMPEG 設成 ffmpeg.exe 的完整路徑。")
    OUT.mkdir(parents=True, exist_ok=True)
    wav = extract_wav(media)
    free_gpu()

    runner = run_qwen if engine == "qwen" else run_anime
    report, text = runner(wav, language)
    report["audio_s"] = round(len(wav) / SR, 1)

    stem = f"{media.stem}_{engine}"
    (OUT / f"{stem}.txt").write_text(text, encoding="utf-8")
    (OUT / f"{stem}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"ok -> {OUT / stem}.json")


if __name__ == "__main__":
    main()
