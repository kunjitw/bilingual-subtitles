"""辨識 / 對齊的各個階段（run_asr、run_align、run_verify、run_realign）。

兩種跑法共用這裡的程式，只差在模型從哪裡來（models 參數）：
  常駐：app/speech_worker.py 把模型留在顯卡上，跨步驟、跨影片沿用（預設，由 app/speech.py 管理）
  單次：python -s -m app.asr_child ...，每次只載入一個模型，處理完就結束程序，顯存完整歸還
        （環境變數 VS_SPEECH_WORKER=0 時用，也是常駐程序出問題時的退路）
進度以 JSON 行輸出到 stdout，結果寫入工作資料夾的檢查點檔，中斷後可續跑。

用法：
  python -s -m app.asr_child asr     <workdir> <engine> <language> <batch>
  python -s -m app.asr_child align   <workdir> <language> <batch>
  python -s -m app.asr_child verify  <workdir> <cues.json> <out.json> <language> <batch> [<only.json>]
  python -s -m app.asr_child realign <workdir> <cues.json> <verify.json> <fix.json> <touched.json> <language> <batch>

verify / realign 是時間軸健康檢查（演算法在 app/health.py），檔名都是工作資料夾裡的相對路徑。
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import soundfile as sf

warnings.filterwarnings("ignore")

from .config import ALIGNER, ASR_ENGINES, LANGUAGES, SAMPLE_RATE, WHISPER_LANG  # noqa: E402
from .cues import normalize_asr_text  # noqa: E402

# Whisper 系模型在沒人聲時常冒出的句子，只過濾整句完全相同的情況
WHISPER_HALLUCINATIONS = {
    "ご視聴ありがとうございました", "ご視聴ありがとうございました。", "チャンネル登録よろしくお願いします",
    "最後までご視聴頂きありがとうございました", "おやすみなさい。おやすみなさい。",
}

STAGES = ("asr", "align", "verify", "realign")
# 命令列（argv[0] 是階段名稱）裡批次大小在第幾個
STAGE_BATCH_ARG = {"asr": 4, "align": 3, "verify": 5, "realign": 7}


def stage_model(argv: list[str]) -> str:
    """這個階段要用哪個模型：qwen、anime、whisper（辨識引擎）或 aligner。檢查時間軸一律用 Qwen3-ASR。"""
    stage = argv[0]
    if stage == "asr":
        return argv[2]
    if stage == "verify":
        return "qwen"
    if stage in ("align", "realign"):
        return "aligner"
    raise ValueError(f"unknown stage {stage}")


def stage_batch(argv: list[str]) -> int:
    return int(argv[STAGE_BATCH_ARG[argv[0]]])


def with_batch(argv: list[str], bs: int) -> list[str]:
    out = list(argv)
    out[STAGE_BATCH_ARG[argv[0]]] = str(int(bs))
    return out


def emit(**kw):
    print(json.dumps(kw, ensure_ascii=False), flush=True)


def load_json(path: Path, default):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def save_json(path: Path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def load_chunks(workdir: Path):
    wav, sr = sf.read(workdir / "audio.wav", dtype="float32")
    assert sr == SAMPLE_RATE
    chunks = load_json(workdir / "chunks.json", [])
    return wav, chunks


def slice_chunk(wav, chunk):
    return wav[int(chunk[0] * SAMPLE_RATE):int(chunk[1] * SAMPLE_RATE)]


# ---------- 載入模型 ----------

def load_qwen_asr(bs: int, max_new_tokens: int):
    import torch
    from qwen_asr import Qwen3ASRModel

    return Qwen3ASRModel.from_pretrained(
        str(ASR_ENGINES["qwen"]["path"]), dtype=torch.bfloat16, device_map="cuda:0", attn_implementation="sdpa",
        max_inference_batch_size=bs, max_new_tokens=max_new_tokens,
    )


def load_aligner():
    import torch
    from qwen_asr import Qwen3ForcedAligner

    return Qwen3ForcedAligner.from_pretrained(
        str(ALIGNER["path"]), dtype=torch.bfloat16, device_map="cuda:0", attn_implementation="sdpa",
    )


def load_whisper(engine_key: str):
    """Whisper 系（anime-whisper、Whisper large-v3）的 transformers pipeline；批次大小在每次呼叫時給。"""
    import torch
    from transformers import pipeline

    return pipeline(
        "automatic-speech-recognition", model=str(ASR_ENGINES[engine_key]["path"]), device="cuda:0",
        torch_dtype=torch.float16, chunk_length_s=30.0,
    )


class OneShot:
    """單次子程序用：要用的時候才載入，程序結束就釋放。常駐程序換成自己的（app/speech_worker.py 的 Resident）。"""

    def emit(self, **kw):
        emit(**kw)

    def check(self):
        """每一批開始前呼叫；常駐程序在這裡檢查取消。"""

    def qwen(self, bs: int, max_new_tokens: int):
        return load_qwen_asr(bs, max_new_tokens)

    def aligner(self):
        return load_aligner()

    def whisper(self, engine_key: str):
        return load_whisper(engine_key)


# ---------- 辨識、對齊 ----------

def run_asr(workdir: Path, engine_key: str, lang: str, bs: int, models=None):
    models = models or OneShot()
    wav, chunks = load_chunks(workdir)
    out_path = workdir / "asr.json"
    texts = load_json(out_path, {})
    todo = [i for i in range(len(chunks)) if str(i) not in texts]
    models.emit(p=(len(chunks) - len(todo)) / max(1, len(chunks)), stage="載入辨識模型")
    if not todo:
        return

    if engine_key == "qwen":
        # 同一個模型物件辨識、檢查都用：批次大小、max_new_tokens 在取得模型時設好
        model = models.qwen(bs, 2048)

        def transcribe(wavs):
            res = model.transcribe(audio=[(w, SAMPLE_RATE) for w in wavs], language=LANGUAGES[lang]["asr"])
            return [r.text for r in res]
    else:
        from qwen_asr.inference.utils import detect_and_fix_repetitions

        pipe = models.whisper(engine_key)
        whisper_lang = WHISPER_LANG.get(lang, "japanese")

        def transcribe(wavs):
            outs = pipe(
                [{"raw": w, "sampling_rate": SAMPLE_RATE} for w in wavs], batch_size=bs,
                generate_kwargs={"language": whisper_lang, "task": "transcribe", "no_repeat_ngram_size": 5},
            )
            result = []
            for o in outs:
                t = detect_and_fix_repetitions(o["text"].strip())
                result.append("" if t in WHISPER_HALLUCINATIONS else t)
            return result

    done = len(chunks) - len(todo)
    for k in range(0, len(todo), bs):
        models.check()
        batch = todo[k:k + bs]
        outs = transcribe([slice_chunk(wav, chunks[i]) for i in batch])
        for i, t in zip(batch, outs):
            texts[str(i)] = normalize_asr_text(t, lang)
        save_json(out_path, texts)
        done += len(batch)
        models.emit(p=done / len(chunks), stage="語音辨識")


def run_align(workdir: Path, lang: str, bs: int, models=None):
    models = models or OneShot()
    wav, chunks = load_chunks(workdir)
    texts = load_json(workdir / "asr.json", {})
    out_path = workdir / "align.json"
    aligned = load_json(out_path, {})
    todo = [i for i in range(len(chunks)) if texts.get(str(i), "").strip() and str(i) not in aligned]
    total = sum(1 for i in range(len(chunks)) if texts.get(str(i), "").strip())
    models.emit(p=(total - len(todo)) / max(1, total), stage="載入對齊模型")
    if not todo:
        return

    aligner = models.aligner()
    done = total - len(todo)
    lang_name = LANGUAGES[lang]["asr"]
    for k in range(0, len(todo), bs):
        models.check()
        batch = todo[k:k + bs]
        results = aligner.align(
            audio=[(slice_chunk(wav, chunks[i]), SAMPLE_RATE) for i in batch],
            text=[texts[str(i)] for i in batch],
            language=lang_name,
        )
        for i, r in zip(batch, results):
            offset = chunks[i][0]
            aligned[str(i)] = [[it.text, round(it.start_time + offset, 3), round(it.end_time + offset, 3)] for it in r.items]
        save_json(out_path, aligned)
        done += len(batch)
        models.emit(p=done / max(1, total), stage="對齊時間軸")


# ---------- 時間軸健康檢查 ----------

def window_clip(wav, start: float, end: float):
    from . import health

    s, e = health.clip_window(start, end, len(wav) / SAMPLE_RATE)
    clip = wav[int(s * SAMPLE_RATE):int(e * SAMPLE_RATE)] if e > s else wav[:0]
    # 字幕時間超出音訊結尾時片段會太短，補靜音到 0.5 秒，模型才吃得下
    min_len = SAMPLE_RATE // 2
    if len(clip) < min_len:
        clip = np.pad(clip, (0, min_len - len(clip)))
    return clip


def run_verify(workdir: Path, cues_name: str, out_name: str, lang: str, bs: int, only_name: str | None = None,
               models=None):
    """重新辨識每一行字幕的時間窗，輸出 {行號: recall}。only_name 指定只檢查哪些行（list，或含 touched 的 dict）。
    檢查一律用 Qwen3-ASR，不管原本的字幕是哪個模型辨識的。"""
    from . import health

    models = models or OneShot()
    wav, sr = sf.read(workdir / "audio.wav", dtype="float32")
    assert sr == SAMPLE_RATE
    cues = load_json(workdir / cues_name, [])
    out_path = workdir / out_name
    rec = load_json(out_path, {})
    if only_name:
        only = load_json(workdir / only_name, [])
        only = only.get("touched", []) if isinstance(only, dict) else only
        idx = sorted({int(i) for i in only if 0 <= int(i) < len(cues)})
    else:
        idx = list(range(len(cues)))
    todo = [i for i in idx if str(i) not in rec]
    models.emit(p=(len(idx) - len(todo)) / max(1, len(idx)), stage="載入辨識模型")
    if not todo:
        return

    # 一行字幕通常只有幾秒，256 個 token 很夠用，也省下生成時預留的顯存
    model = models.qwen(bs, 256)
    lang_name = LANGUAGES[lang]["asr"]
    done = len(idx) - len(todo)
    for batch in health.verify_batches(todo, cues, len(wav) / SAMPLE_RATE, bs):
        models.check()
        res = model.transcribe(
            audio=[(window_clip(wav, cues[i]["start"], cues[i]["end"]), SAMPLE_RATE) for i in batch],
            language=lang_name,
        )
        for i, r in zip(batch, res):
            rec[str(i)] = round(health.recall(cues[i]["text"], r.text), 3)
        save_json(out_path, rec)
        done += len(batch)
        models.emit(p=done / len(idx), stage="檢查時間軸")


def run_realign(workdir: Path, cues_name: str, verify_name: str, fix_name: str, touched_name: str, lang: str, bs: int,
                models=None):
    """把錨點之間有對不上的區段重新對齊，輸出重新對齊後的字幕和改到的行。"""
    from . import health

    models = models or OneShot()
    wav, sr = sf.read(workdir / "audio.wav", dtype="float32")
    assert sr == SAMPLE_RATE
    total = len(wav) / SAMPLE_RATE
    cues = load_json(workdir / cues_name, [])
    rec = {int(k): v for k, v in load_json(workdir / verify_name, {}).items()}
    regions = [r for r in health.plan_regions(cues, rec, total) if health.alignable(r)]
    units_path = workdir / health.units_name(fix_name)
    units = load_json(units_path, {})
    todo = [r for r in regions if health.region_key(r) not in units]
    models.emit(p=(len(regions) - len(todo)) / max(1, len(regions)), stage="載入對齊模型")

    if todo:
        aligner = models.aligner()
        lang_name = LANGUAGES[lang]["asr"]
        done = len(regions) - len(todo)
        for batch in health.align_batches(todo, bs):
            models.check()
            results = aligner.align(
                audio=[(wav[int(s * SAMPLE_RATE):int(e * SAMPLE_RATE)], SAMPLE_RATE) for _, _, s, e in batch],
                text=[health.region_text(cues, lo, hi, lang) for lo, hi, _, _ in batch],
                language=lang_name,
            )
            for region, r in zip(batch, results):
                offset = region[2]
                units[health.region_key(region)] = [
                    [it.text, round(it.start_time + offset, 3), round(it.end_time + offset, 3)] for it in r.items
                ]
            save_json(units_path, units)
            done += len(batch)
            models.emit(p=done / len(regions), stage="修正時間軸")

    fixed, touched = health.apply_regions(cues, regions, units)
    save_json(workdir / fix_name, fixed)
    save_json(workdir / touched_name, {"regions": regions, "touched": touched})


def dispatch(argv: list[str], models=None):
    stage, workdir = argv[0], Path(argv[1])
    if stage == "asr":
        run_asr(workdir, argv[2], argv[3], int(argv[4]), models=models)
    elif stage == "align":
        run_align(workdir, argv[2], int(argv[3]), models=models)
    elif stage == "verify":
        run_verify(workdir, argv[2], argv[3], argv[4], int(argv[5]), argv[6] if len(argv) > 6 else None, models=models)
    elif stage == "realign":
        run_realign(workdir, argv[2], argv[3], argv[4], argv[5], argv[6], int(argv[7]), models=models)
    else:
        raise SystemExit(f"unknown stage {stage}")


def main():
    dispatch(sys.argv[1:])
    emit(p=1.0)


if __name__ == "__main__":
    main()
