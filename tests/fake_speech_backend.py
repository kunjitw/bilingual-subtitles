"""常駐語音程序（app/speech_worker.py）測試用的假模型和假顯存，不需要顯卡、不 import torch。

用法：工作程序的環境變數 VS_SPEECH_BACKEND=tests.fake_speech_backend，
     VS_FAKE_SPEECH 是一個 JSON：
       total_mb      假顯卡的總顯存（預設 16303）
       fail_start    有值時啟動就失敗，訊息是這個字串（模擬 CUDA 用不了）
       per_item_mb   推論時每一筆多用的顯存 {"qwen": 420, "verify": 50, "aligner": 250, "anime": 160}
       events        事件紀錄檔（一行一個 JSON：load、unload、batch、oom）
       control       控制檔路徑，每一批開始前讀：
                       delay_s       每一批花幾秒
                       crash_on      跑到這個模型（qwen、aligner、anime、whisper）的推論時程序直接結束
                       hang_on       跑到這個模型的推論時卡住（不看取消）
                       fatal_on      跑到這個模型的推論時丟 CUDA 壞掉的錯誤
                       oom_always    這個模型的推論一定 OOM
模型常駐量、載入多用的量照 config.SPEECH_VRAM。
"""
import gc
import json
import os
import sys
import time
import types
from pathlib import Path
from types import SimpleNamespace

from app.config import SPEECH_VRAM

# whisper 系的辨識會 import qwen_asr 的去重複函式；真的 qwen_asr 會連帶載入 torch、transformers（很慢），換成假的
if "qwen_asr" not in sys.modules:
    _pkg = types.ModuleType("qwen_asr")
    _inference = types.ModuleType("qwen_asr.inference")
    _utils = types.ModuleType("qwen_asr.inference.utils")
    _utils.detect_and_fix_repetitions = lambda text: text
    _pkg.inference, _inference.utils = _inference, _utils
    sys.modules.update({"qwen_asr": _pkg, "qwen_asr.inference": _inference, "qwen_asr.inference.utils": _utils})

CFG = json.loads(os.environ.get("VS_FAKE_SPEECH") or "{}")
PER_ITEM = {"qwen": 420, "verify": 50, "aligner": 250, "anime": 160, "whisper": 400, **CFG.get("per_item_mb", {})}


class FakeOOM(RuntimeError):
    pass


class FakeFatal(RuntimeError):
    pass


def event(**kw):
    path = CFG.get("events")
    if path:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(kw, ensure_ascii=False) + "\n")


def control() -> dict:
    path = CFG.get("control")
    try:
        return json.loads(Path(path).read_text(encoding="utf-8")) if path else {}
    except (OSError, ValueError):
        return {}


class Backend:
    def __init__(self):
        if CFG.get("fail_start"):
            raise RuntimeError(CFG["fail_start"])
        self.total_mb = int(CFG.get("total_mb", 16303))
        self.allocated = 0
        self.peak = 0
        self.limit = None
        global BACKEND
        BACKEND = self

    # 記帳
    def _alloc(self, mb: int, what: str):
        if self.limit is not None and self.allocated + mb > self.limit:
            event(ev="oom", what=what, allocated=self.allocated, need=mb, limit=self.limit)
            raise FakeOOM(f"CUDA out of memory while {what}")
        self.allocated += mb
        self.peak = max(self.peak, self.allocated)

    def _free(self, mb: int):
        self.allocated -= mb

    # speech_worker 要的介面
    def load_qwen(self, bs, max_new_tokens):
        return FakeModel(self, "qwen")

    def load_aligner(self):
        return FakeModel(self, "aligner")

    def load_whisper(self, engine_key):
        return FakeModel(self, engine_key)

    def is_oom(self, e):
        return isinstance(e, FakeOOM)

    def is_fatal(self, e):
        return isinstance(e, FakeFatal)

    def release(self):
        gc.collect()

    def set_limit(self, limit_mb):
        self.limit = int(limit_mb) if limit_mb else None

    def reset_peak(self):
        self.peak = self.allocated

    def allocated_mb(self):
        return self.allocated

    def memory(self):
        return {"reserved_mb": self.allocated, "allocated_mb": self.allocated, "peak_reserved_mb": self.peak,
                "peak_allocated_mb": self.peak}


BACKEND = None


class FakeModel:
    """假的 Qwen3-ASR（transcribe）、對齊模型（align）、whisper pipeline（直接呼叫）。"""

    def __init__(self, backend: Backend, key: str):
        self.backend = backend
        self.key = key
        self.mb = SPEECH_VRAM["resident_mb"][key]
        self.max_inference_batch_size = None
        self.max_new_tokens = None
        backend._alloc(SPEECH_VRAM["load_extra_mb"][key] + self.mb, f"loading {key}")
        backend._free(SPEECH_VRAM["load_extra_mb"][key])
        self._alive = True
        event(ev="load", model=key)

    def __del__(self):
        if getattr(self, "_alive", False):
            self.backend._free(self.mb)
            event(ev="unload", model=self.key)

    def _infer(self, n: int, kind: str):
        c = control()
        if c.get("crash_on") == self.key:
            event(ev="crash", model=self.key)
            os._exit(9)
        if c.get("hang_on") == self.key:
            event(ev="hang", model=self.key)
            time.sleep(3600)
        if c.get("fatal_on") == self.key:
            raise FakeFatal("CUDA error: an illegal memory access was encountered")
        if c.get("delay_s"):
            time.sleep(float(c["delay_s"]))
        extra = PER_ITEM[kind] * n
        if c.get("oom_always") == self.key:
            extra = 10 ** 7
        self.backend._alloc(extra, f"{kind} x{n}")
        self.backend._free(extra)
        event(ev="batch", model=self.key, kind=kind, n=n, bs=self.max_inference_batch_size)

    def transcribe(self, audio, language=None):
        kind = "verify" if self.max_new_tokens == 256 else "qwen"
        self._infer(len(audio), kind)
        return [SimpleNamespace(text=fake_text(len(a[0]) / 16000, language)) for a in audio]

    def align(self, audio, text, language=None):
        self._infer(len(audio), "aligner")
        out = []
        for (wav, sr), t in zip(audio, text):
            dur = len(wav) / sr
            # 英文照單字、其他照字（跟真的對齊模型一樣，不含空白）
            chars = (t.split() if " " in t.strip() else [ch for ch in t if ch.strip()]) or ["x"]
            step = dur / len(chars)
            out.append(SimpleNamespace(items=[SimpleNamespace(text=ch, start_time=round(i * step, 3),
                                                              end_time=round((i + 1) * step, 3))
                                              for i, ch in enumerate(chars)]))
        return out

    def __call__(self, inputs, batch_size=None, generate_kwargs=None):
        self._infer(len(inputs), self.key)
        lang = (generate_kwargs or {}).get("language")
        return [{"text": fake_text(len(x["raw"]) / 16000, "English" if lang == "english" else lang)} for x in inputs]


WORDS = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel"]


def fake_text(seconds: float, language) -> str:
    """片段越長字越多（約每 1.5 秒一個字），同一段音訊每次都一樣。"""
    n = max(1, int(seconds / 1.5))
    if language == "English":
        return " ".join(WORDS[i % len(WORDS)] for i in range(n)) + "."
    return "".join("あいうえおかきくけこ"[i % 10] for i in range(n)) + "。"
