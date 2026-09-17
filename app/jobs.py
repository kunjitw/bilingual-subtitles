"""佇列與任務流程。

四條工作執行緒：
  gpu   執行緒：轉字幕、翻譯、轉檔，一次只跑一個任務。語音模型（辨識、對齊、檢查時間軸）放在常駐程序裡，
               顯存放得下就同時載入、跨影片沿用（app/speech.py）；翻譯的 llama-server 也跨任務沿用，兩邊放不下時先釋放另一邊
  io    執行緒：網址下載
  model 執行緒：模型下載（跟網址下載分開，下載大模型時貼 YouTube 網址也能馬上開始）
  dict  執行緒：建查字字典（app/dict_build.py），跟模型下載同時跑

模型下載可以暫停（狀態 paused，暫存檔保留），繼續時重新排隊、從中斷的地方接著下載。
"""
import json
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

import soundfile as sf

from . import cues as cue_mod
from . import config, db, gpu, media, safepath, settings, speech, syscheck, translate, vad
from .config import (ASR_ENGINES, ALIGNER, LANGUAGES, LLAMA_SERVER, MEDIA_DIR, MODEL_CATALOG, PROXY_DIR,
                     ROOT, SAKURA_STYLE, SAMPLE_RATE, SEPARATOR, THUMB_DIR, TRANSLATORS, WORK_DIR, YTDLP_CACHE_DIR,
                     child_env, model_installed, translator_vram_mb)

log = logging.getLogger("jobs")

GPU_TYPES = ("transcribe", "translate", "proxy", "titles", "glosses", "health")
IO_TYPES = ("download",)
MODEL_TYPES = ("model",)
DICT_TYPES = ("dict",)           # 建查字字典（app/dict_build.py），跟模型下載同時跑，不排在同一條執行緒
PAUSABLE_TYPES = ("model",)      # 可以暫停的任務（暫停時結束子程序，暫存檔保留）

_wake = threading.Event()
_running: dict[str, "JobContext"] = {}
_running_lock = threading.Lock()
# 把任務標成執行中時拿著；設定頁刪除模型時也拿著，刪的途中不會有任務開始載入那個模型（app/models.py）
claim_lock = threading.Lock()


def missing_model_message(label: str) -> str:
    return f"找不到模型 {label}，請到設定頁的模型管理下載"


def catalog_id(field: str, value) -> str | None:
    """型錄裡 field 等於 value 的模型 id，例如 catalog_id("engine", "qwen") → "qwen-asr"。"""
    if value is None:
        return None
    return next((mid for mid, e in MODEL_CATALOG.items() if e.get(field) == value), None)


def use_model(ctx: "JobContext", mid: str | None) -> bool:
    """任務要用某個模型之前呼叫：模型還在就登記起來（任務結束前設定頁刪不掉，見 app/models.py），回 True；
    模型已經不在回 False，由呼叫的地方決定要失敗還是不用它（例如人聲分離是選用的）。

    跟設定頁刪除模型用同一把 claim_lock：刪到一半的時候會等刪完再看，
    所以不會「確認時檔案還在、載入時被刪掉」，也不會讓套件發現檔案不見後自己從網路下載回來。
    不在型錄裡的模型不管（回 True）。
    """
    entry = MODEL_CATALOG.get(mid) if mid else None
    if entry is None:
        return True
    with claim_lock:
        if not model_installed(entry):
            return False
        ctx.uses.add(mid)
        return True


def require_model(ctx: "JobContext", mid: str | None):
    if not use_model(ctx, mid):
        raise RuntimeError(missing_model_message(MODEL_CATALOG[mid]["label"]))


def uses_of(job_id: str) -> set[str]:
    """執行中的任務自己登記要用的模型（use_model）。"""
    with _running_lock:
        ctx = _running.get(job_id)
        return set(ctx.uses) if ctx else set()


# 目前佔用顯卡的狀態，給前端顯示
gpu_state = {"job_id": None, "model": None, "stage": None}

# 連續的翻譯任務共用同一個 llama-server，不重複載入模型
_llm_lock = threading.RLock()
_llm = {"key": None, "server": None}


def translator_label(key: str, cfg: dict) -> str:
    """說明文字裡的翻譯模型名稱：用的不是預設版本時加上版本，例如 Hy-MT2-7B Q6_K。"""
    entry = config.translator_entry(key)
    variant = cfg.get("variant")
    if entry and variant and variant != entry.get("default_variant"):
        return f"{cfg['label']} {variant}"
    return cfg["label"]


def _same_llm(a: dict, b: dict) -> bool:
    return (_same_path(a.get("gguf"), b.get("gguf")) and a.get("ctx") == b.get("ctx")
            and a.get("parallel") == b.get("parallel"))


def get_llm(key: str, log_path: Path, check) -> tuple:
    """取得翻譯用的 llama-server；同一個模型（同一個版本、同樣的 ctx、parallel）就沿用，不一樣就先關掉舊的。
    版本和參數照設定（config.translator_cfg：model_variants、llm_params）。"""
    with _llm_lock:
        cfg = config.translator_cfg(key)
        server = _llm["server"]
        if _llm["key"] == key and server and server.alive() and _same_llm(getattr(server, "cfg", None) or cfg, cfg):
            return server, True
        label = translator_label(key, cfg)
        if not cfg["gguf"].is_file():
            # 模型在設定頁被刪掉了（排隊時還在）：說清楚，不要丟看不懂的「找不到檔案」
            raise RuntimeError(missing_model_message(label))
        release_llm("換模型")
        # 真的要載入新模型時才檢查顯存（已載入的模型本身就佔著顯存，沿用時不能檢查）
        need = translator_vram_mb(key, cfg=cfg)
        # 常駐的語音模型佔著顯存、翻譯模型放不下時先卸載它們（程序留著，還是不夠才結束程序）；
        # 放得下就留著，下一部影片轉字幕不用重新載入
        speech.make_room(need, f"載入 {label} 需要顯存")
        gpu.ensure_free(need, label, alternatives=lambda usable: smaller_translators(key, usable))
        server = translate.LlamaServer(key, log_path, cfg=cfg)
        server.start(check)
        _llm.update(key=key, server=server)
        return server, False


def smaller_translators(key: str, usable_mb: int) -> list[str]:
    """顯存不足的說明裡建議改用的：同一個模型放得下的最大版本，和這張卡放得下、key 能翻的語言它都能翻的其他翻譯模型。"""
    langs = set(TRANSLATORS[key]["langs"])
    out = []
    current = config.translator_cfg(key)
    entry = config.translator_entry(key)
    for v in config.variant_names(entry or {}):
        if v != current.get("variant") and translator_vram_mb(key, cfg=config.translator_cfg(key, variant=v)) <= usable_mb:
            out.append(f"{TRANSLATORS[key]['label']} {v}")
            break
    for k, tcfg in TRANSLATORS.items():
        other = config.translator_entry(k) or {}
        if k != key and langs <= set(tcfg["langs"]) and translator_vram_mb(k, other.get("size_mb")) <= usable_mb:
            out.append(tcfg["label"])
    return out


def release_llm(reason: str = ""):
    with _llm_lock:
        if _llm["server"]:
            log.info("release llama-server (%s)", reason)
            _llm["server"].stop()
        _llm.update(key=None, server=None)


class Cancelled(Exception):
    pass


class Paused(Exception):
    """使用者按了暫停（只有 PAUSABLE_TYPES 的任務）：任務改成 paused，暫存檔保留。"""


class JobContext:
    def __init__(self, job: dict):
        self.job = job
        self.id = job["id"]
        self.media_id = job["media_id"]
        self.params = job["params"]
        self.cancel_event = threading.Event()
        self.pause_event = threading.Event()
        self.workdir = WORK_DIR / self.id
        self.child: subprocess.Popen | None = None
        self._last = 0.0
        self._stage = None
        self.result = None
        self.uses: set[str] = set()   # 這個任務登記要用的模型（型錄 id），見 use_model
        self.note: str | None = None  # 途中改變做法時的說明（例如人聲分離失敗改用原音），之後的進度都會帶著
        self.speech_plan = None       # 常駐語音模型的規劃（批次大小等，plan_speech），整個任務不變
        # 這次開始時的進度：暫停、中斷後接著做的任務不是從 0 開始，網頁估剩餘時間要扣掉（resume_points）
        self.start_progress = float(job.get("progress") or 0)

    def check(self):
        if self.cancel_event.is_set():
            raise Cancelled()
        if self.pause_event.is_set():
            raise Paused()

    def progress(self, value: float, stage: str | None = None):
        if stage and self.note and self.note not in stage:
            stage = f"{stage}（{self.note}）"
        now = time.time()
        if (stage or self._stage) == self._stage and now - self._last < 0.5 and value < 1:
            return
        self._last = now
        self._stage = stage or self._stage
        fields = {"progress": round(max(0.0, min(1.0, value)), 4)}
        if stage:
            fields["stage"] = stage
            if gpu_state["job_id"] == self.id:
                gpu_state["stage"] = stage
        db.update_job(self.id, **fields)


def wake():
    _wake.set()


def cancel(job_id: str):
    job = db.get_job(job_id)
    if not job:
        return
    if job["status"] in ("queued", "paused"):
        db.update_job(job_id, status="canceled", stage="已取消", finished_at=time.time())
        db.cancel_dependents(job_id)
    elif job["status"] == "running":
        with _running_lock:
            ctx = _running.get(job_id)
        if ctx:
            ctx.cancel_event.set()
            if ctx.child and ctx.child.poll() is None:
                ctx.child.kill()


def pause(job_id: str) -> bool:
    """暫停模型下載：排隊中的直接改成 paused；執行中的結束子程序，任務自己改成 paused（暫存檔保留）。
    不能暫停（不是模型下載、已經結束）回 False。"""
    job = db.get_job(job_id)
    if not job or job["type"] not in PAUSABLE_TYPES:
        return False
    if job["status"] == "paused":
        return True
    if job["status"] == "queued" and db.move_status(job_id, "queued", status="paused", stage="已暫停"):
        return True
    with _running_lock:
        ctx = _running.get(job_id)
    if not ctx:
        return False
    ctx.pause_event.set()
    child = ctx.child
    if child and child.poll() is None:
        child.kill()
    return True


def resume(job_id: str) -> bool:
    """暫停的模型下載放回佇列，輪到時從中斷的地方接著下載。"""
    job = db.get_job(job_id)
    if not job or job["status"] != "paused":
        return False
    ok = db.move_status(job_id, "paused", status="queued", stage="排隊中（接著下載）", error=None, finished_at=None)
    if ok:
        wake()
    return ok


class RetryRefused(Exception):
    """重試被擋下。status 是 API 要回的 HTTP 狀態碼，訊息可以直接給使用者看。"""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


_retry_lock = threading.Lock()   # 檢查和改回排隊要一起做，連點兩下或兩個分頁同時按才不會排出兩個


def _active_jobs(job: dict) -> list[dict]:
    """佇列裡（排隊、執行中或暫停中）跟這個任務同一種的其他任務。"""
    return [o for o in db.list_jobs(finished_limit=0)
            if o["id"] != job["id"] and o["type"] == job["type"] and o["status"] in ("queued", "running", "paused")]


def same_active_job(job: dict) -> dict | None:
    """佇列裡（排隊或執行中）跟這個任務做同一件事的其他任務：
    同一部影片用同一個模型轉字幕；同一部影片的轉檔、下載；同一條原文的翻譯（不管用哪個模型，跟翻譯入口一樣）；
    同一條字幕的時間軸檢查；同一個模型的下載；標題翻譯；單字釋義（全部影片的任務也算涵蓋單一影片，跟 glossgen.queued_job 一樣）。
    同一部影片用別的模型轉字幕不算重複，跟「用其他模型轉字幕」一樣確認過就能排（retry_overwrites）。"""
    kind, p = job["type"], job.get("params") or {}
    src = translation_source(job) if kind == "translate" else None
    for other in _active_jobs(job):
        op = other.get("params") or {}
        if kind == "transcribe":
            same = other["media_id"] == job["media_id"] and op.get("engine") == p.get("engine")
        elif kind in ("proxy", "download"):
            same = other["media_id"] == job["media_id"]
        elif kind == "translate":
            same = bool(src) and translation_source(other) == src
        elif kind == "health":
            same = op.get("track_id") == p.get("track_id")
        elif kind == "model":
            same = op.get("model") == p.get("model") and model_job_variant(other) == model_job_variant(job)
        elif kind == "dict":
            same = op.get("lang") == p.get("lang")
        elif kind == "glosses":
            same = op.get("scope") == "library" or (p.get("scope") != "library" and other["media_id"] == job["media_id"])
        else:
            same = True
        if same:
            return other
    return None


_DUPLICATE_TEXT = {
    "transcribe": ("這部影片正在用 {model} 轉字幕", "這部影片已經在排隊用 {model} 轉字幕"),
    "translate": ("這條字幕正在翻譯", "這條字幕已經在排隊翻譯"),
    "health": ("這條字幕正在檢查時間軸", "這條字幕已經在排隊檢查時間軸"),
    "proxy": ("這部影片正在轉檔", "這部影片已經在排隊轉檔"),
    "download": ("這部影片正在下載", "這部影片已經在排隊下載"),
    "model": ("這個模型正在下載", "這個模型已經在排隊下載"),
    "dict": ("這本字典正在建立", "這本字典已經在排隊建立"),
    "titles": ("影片標題正在翻譯", "影片標題已經在排隊翻譯"),
    "glosses": ("單字釋義正在產生", "單字釋義已經在排隊"),
}


def _health_summary(track: dict) -> dict | None:
    raw = track.get("health")
    try:
        h = json.loads(raw) if isinstance(raw, str) else raw
    except ValueError:
        return None
    return h if isinstance(h, dict) else None


def model_pending(mid: str | None) -> bool:
    """模型還沒裝好，但是有下載任務在排隊、下載中、暫停，或自動安裝排好了重試（app/setup.py）。"""
    if not mid:
        return False
    try:
        from . import setup
        return setup.model_pending(mid)
    except Exception:  # noqa: BLE001
        log.warning("checking pending model downloads failed", exc_info=True)
        return False


def waiting_for_models(candidates: list[dict]) -> list[dict]:
    """排隊中的顯卡任務裡，還在等模型下載完的（app/setup.py waiting_models）。一次只查一次模型下載的任務清單。"""
    try:
        from . import setup
        pending = setup.pending_models()
        if not pending:
            return []
        return [j for j in candidates if setup.waiting_models(j, pending)]
    except Exception:  # noqa: BLE001
        log.warning("checking jobs waiting for models failed", exc_info=True)
        return []


def retry_problem(job: dict) -> str | None:
    """重試一定會再失敗的情況（影片或要處理的字幕已經刪掉、模型還沒下載），回傳說明。"""
    kind, p = job["type"], job.get("params") or {}
    if job.get("media_id") and not db.get_media(job["media_id"]):
        return "這部影片已經刪除了，不用重試，可以把這個任務移除"
    if job.get("depends_on"):
        # 前一步沒做完就排進去，會一直停在「等前一步」
        dep = db.get_job(job["depends_on"])
        if not dep:
            return "前一個步驟的任務已經移除了，這個任務沒辦法重試，可以把它移除"
        if dep["status"] in ("failed", "canceled"):
            return "前一個步驟沒有完成，請重試前一個任務，這個任務會跟著重新排入"
    if kind == "translate":
        src = translation_source(job)
        if src and not db.get_track(src):
            return "要翻譯的原文字幕已經刪除了，不用重試，可以把這個任務移除"
    if kind == "health" and not (p.get("track_id") and db.get_track(p["track_id"])):
        return "要檢查的字幕已經刪除了，不用重試，可以把這個任務移除"
    # 模型還沒下載、但是正在下載或會自動接著下載（第一次打開的自動安裝，app/setup.py）：放行，任務排隊等模型裝好
    engine = p.get("engine") if kind == "transcribe" else "qwen" if kind == "health" else None
    if (engine in ASR_ENGINES and engine not in settings.installed_engines()
            and not model_pending(catalog_id("engine", engine))):
        return f"{ASR_ENGINES[engine]['label']} 還沒下載，請到設定頁的模型管理下載後再重試"
    if kind in ("transcribe", "health") and not aligner_installed() and not model_pending(catalog_id("role", "aligner")):
        return f"{ALIGNER['label']} 還沒下載，請到設定頁的模型管理下載後再重試"
    translator = p.get("translator") if kind in ("translate", "titles") else None
    if kind == "titles" and not translator:
        translator = settings.translator_for("ja")
    if (translator in TRANSLATORS and translator not in settings.installed_translators()
            and not model_pending(catalog_id("translator", translator))):
        return f"{TRANSLATORS[translator]['label']} 還沒下載，請到設定頁的模型管理下載後再重試"
    if kind == "titles" and translator not in TRANSLATORS:
        return "還沒下載翻譯模型，請到設定頁的模型管理下載後再重試"
    if kind == "glosses":
        from . import glossgen
        if not glossgen.translator_key():
            return "產生單字釋義需要 Hy-MT2-7B 或 Hy-MT2-1.8B，請到設定頁的模型管理下載後再重試"
    return None


def retry_overwrites(job: dict) -> str | None:
    """重試會重做已經有的結果（多一條字幕或翻譯、重新檢查時間軸、重新轉檔），
    或這部影片正在用別的模型轉字幕時回傳說明，要使用者確認過（force）才排。"""
    kind, p, mid = job["type"], job.get("params") or {}, job.get("media_id")
    if kind == "transcribe" and mid and any(t["kind"] == "asr" for t in db.list_tracks(mid)):
        return "這部影片已經有字幕了，再轉一次會多一條字幕，請先確認"
    if kind == "transcribe" and mid and any(o["media_id"] == mid for o in _active_jobs(job)):
        return "這部影片正在用其他模型轉字幕，確定要同時再轉一次請先確認"
    if kind == "translate" and mid:
        src = translation_source(job)
        if src and any(t["source_track_id"] == src for t in db.list_tracks(mid)):
            return "這條字幕已經翻譯過了，再翻一次請先確認"
    if kind == "health":
        track = db.get_track(p["track_id"]) if p.get("track_id") else None
        if track and _health_summary(track):
            return "這條字幕已經檢查過時間軸，再檢查一次請先確認"
    if kind == "proxy" and mid:
        m = db.get_media(mid)
        if m and m.get("proxy_path") and Path(m["proxy_path"]).exists():
            return "這部影片已經有相容播放檔，再轉一次請先確認"
    return None


def retry(job_id: str, force: bool = False):
    """把失敗或取消的任務放回佇列（連帶因為它而取消的後續任務）。
    同一件事已經在排隊或執行就不再排；會重做已經有的結果要 force（網頁上確認過）；一定會再失敗的也先擋下。
    擋下時丟 RetryRefused。"""
    with _retry_lock:
        job = db.get_job(job_id)
        if not job:
            raise RetryRefused(404, "找不到這個任務，可能已經移除了")
        if job["status"] in ("queued", "running"):
            raise RetryRefused(409, "這個任務已經在佇列裡了")
        if job["status"] == "paused":
            raise RetryRefused(409, "這個下載暫停中，按「繼續」就會接著下載")
        if job["status"] not in ("failed", "canceled"):
            raise RetryRefused(409, "這個任務已經完成了，不用再排")
        dup = same_active_job(job)
        if dup:
            running, queued = _DUPLICATE_TEXT.get(job["type"], ("同樣的任務正在執行", "同樣的任務已經在排隊"))
            engine = (job.get("params") or {}).get("engine")
            text = (running if dup["status"] == "running" else queued).format(
                model=ASR_ENGINES.get(engine, {}).get("label") or engine or "同一個模型")
            raise RetryRefused(409, f"{text}，完成後就會出現，不用再排一次")
        problem = retry_problem(job)
        if problem:
            raise RetryRefused(400, problem)
        if not force:
            overwrite = retry_overwrites(job)
            if overwrite:
                raise RetryRefused(409, overwrite)
        db.update_job(job_id, status="queued", stage="排隊中", error=None, progress=0, finished_at=None)
        db.requeue_dependents(job_id)
    wake()


# ---------- 工作執行緒 ----------

def model_key(job: dict) -> str:
    """這個任務會用到哪一類模型，用來把同一類的任務排在一起。
    轉字幕（不管哪個辨識模型）和檢查時間軸都是語音模型（speech）：它們共用常駐程序、對齊模型和檢查用的 Qwen3-ASR，
    放得下時同時載入，連著跑不用換；翻譯類照翻譯模型分（llm:<key>）。"""
    p = job.get("params") or {}
    if job["type"] in ("transcribe", "health"):
        return "speech"
    if job["type"] in ("translate", "titles", "glosses"):
        return f"llm:{p.get('translator')}"
    return job["type"]


def speech_model_of(job: dict) -> str | None:
    """語音任務第一個要用的模型（常駐程序裡的名稱）。"""
    if job["type"] == "transcribe":
        return (job.get("params") or {}).get("engine")
    return "qwen" if job["type"] == "health" else None


# ---------- 顯卡閒置釋放、「釋放顯卡」按鈕、暫停佇列 ----------

QUEUE_PAUSED_KEY = "queue_paused"     # 存在資料庫的 settings 表（重開伺服器還是暫停），不在 settings.DEFAULTS 裡，設定頁存檔改不到
QUEUE_BUSY_MESSAGE = "佇列還有任務，要暫停佇列並釋放顯卡嗎？正在跑的任務會停下，之後按「繼續佇列」會從中斷的地方接著做"
QUEUE_WAITING_MESSAGE = "佇列還有排隊中的任務，要暫停佇列並釋放顯卡嗎？之後按「繼續佇列」才會開始"
PAUSED_STAGE = "已暫停，按「繼續佇列」接著做"
RESUME_STAGE = "從中斷的地方接著做"      # 停下來的任務（暫停、上次中斷）重新開始、還沒回報新進度前
START_STAGE = "準備中"

# 顯卡沒事做的起點（time.monotonic）；有事做時是 None。語音模型和翻譯模型共用這一個計時（idle_tick）
_idle = {"since": None}


def queue_paused() -> bool:
    """佇列暫停中：顯卡任務不會開始（「釋放顯卡」按鈕確認後暫停，按「繼續佇列」恢復）。網址、模型下載照常。"""
    return bool(settings.get(QUEUE_PAUSED_KEY))


def set_queue_paused(paused: bool):
    db.set_settings({QUEUE_PAUSED_KEY: bool(paused)})
    if not paused:
        wake()


def gpu_jobs() -> list[dict]:
    """執行中、排隊中（包括等網址下載完才能開始的）的顯卡任務。在等模型下載完的不算（等的時候照常閒置釋放顯卡）。"""
    active = [j for j in db.list_jobs(finished_limit=0) if j["status"] in ("running", "queued") and j["type"] in GPU_TYPES]
    waiting = {j["id"] for j in waiting_for_models([j for j in active if j["status"] == "queued"])}
    return [j for j in active if j["id"] not in waiting]


def gpu_work_pending() -> bool:
    """顯卡還有事要做：有執行中的顯卡任務，或佇列沒暫停、還有排隊中的顯卡任務。"""
    paused = queue_paused()
    return any(j["status"] == "running" or not paused for j in gpu_jobs())


def gpu_models_loaded() -> bool:
    return speech.WORKER.proc is not None or _llm["server"] is not None


def release_models(reason: str, force: bool = False, llm_wait_s: float = 20.0) -> list[str]:
    """結束常駐的語音模型和翻譯的 llama-server，顯存全部還回去。回傳釋放了哪些模型（名稱）。
    force：正在跑的請求也不等（「釋放顯卡」按鈕；呼叫前要先把執行中的任務標成暫停）。"""
    before = [m["label"] for m in loaded_models()]
    if force:
        speech.WORKER.kill(reason)
    else:
        speech.WORKER.stop(reason, timeout=llm_wait_s)
    # 翻譯模型正在載入時 get_llm 拿著這把鎖（任務標成暫停後半秒內就會放掉）；等不到就算了，載入的任務自己會關掉它
    if _llm_lock.acquire(timeout=llm_wait_s):
        try:
            release_llm(reason)
        finally:
            _llm_lock.release()
    return before


def idle_tick(now: float | None = None) -> bool:
    """顯卡執行緒閒著時呼叫（每 2 秒一次）。顯卡沒事做滿 config.GPU_IDLE_RELEASE_S 秒，就把顯卡上的模型全部釋放；
    中間有新任務進來就重新計時。回傳這次有沒有釋放。"""
    now = time.monotonic() if now is None else now
    if not gpu_models_loaded() or gpu_work_pending():
        _idle["since"] = None            # 沒有東西可以釋放（不用查資料庫），或還有事要做
        return False
    if _idle["since"] is None:
        _idle["since"] = now
        return False
    if now - _idle["since"] < config.GPU_IDLE_RELEASE_S:
        return False
    log.info("GPU idle for %ss, releasing models", config.GPU_IDLE_RELEASE_S)
    release_models(f"顯卡閒置 {config.GPU_IDLE_RELEASE_S} 秒")
    return True


class QueueBusy(RuntimeError):
    """「釋放顯卡」時佇列還有任務，要使用者確認暫停佇列。"""


def release_gpu(pause_queue: bool = False, wait_s: float = 30.0) -> dict:
    """「釋放顯卡」按鈕：馬上結束顯卡上所有的模型（常駐語音模型、llama-server、人聲分離之類的子程序）。
    沒有執行中的顯卡任務、佇列也不會再開始新的（空的或已暫停）：直接釋放。
    還有任務時要 pause_queue（網頁上確認過）：暫停佇列、執行中的任務停下來回到佇列（不算失敗，之後從檢查點接著做），
    再全部釋放；沒確認丟 QueueBusy。回傳釋放了哪些模型、佇列是否暫停、現在的剩餘顯存。"""
    with claim_lock:                 # 拿著這把鎖時不會有新任務開始
        active = gpu_jobs()
        paused = queue_paused()
        running = [j for j in active if j["status"] == "running"]
        if running or (active and not paused):
            if not pause_queue:
                raise QueueBusy(QUEUE_BUSY_MESSAGE if running else QUEUE_WAITING_MESSAGE)
            if not paused:
                set_queue_paused(True)
                log.info("queue paused by the release button")
        with _running_lock:
            stopping = [c for c in _running.values() if c.job["type"] in GPU_TYPES]
        for ctx in stopping:
            ctx.pause_event.set()
            child = ctx.child
            if child is not None and child.poll() is None:
                child.kill()             # 人聲分離、單次模式的辨識子程序
    released = release_models("使用者按了釋放顯卡", force=True)
    # 等執行中的任務停下來回到佇列（它可能正要載入模型），再確認一次都放掉了
    deadline = time.monotonic() + wait_s
    while stopping and time.monotonic() < deadline:
        with _running_lock:
            if not any(c.id in _running for c in stopping):
                break
        time.sleep(0.2)
    if gpu_models_loaded():
        released += [x for x in release_models("使用者按了釋放顯卡", force=True) if x not in released]
    _idle["since"] = None
    info = gpu.query(max_age=0)
    return {"released": released, "paused": queue_paused(), "stopped_jobs": [c.id for c in stopping],
            "free_mb": info["free_mb"] if info else None, "total_mb": info["total_mb"] if info else None}


def resume_queue() -> dict:
    """「繼續佇列」：暫停時停下來的任務照原來的位置開始，從檢查點接著做；模型需要時再載入。"""
    set_queue_paused(False)
    return {"paused": False}


def starting_stage(job: dict) -> str | None:
    """任務開始執行時要換掉的說明：排隊時的「已暫停，按繼續佇列接著做」「排隊中（上次中斷）」不要留到執行中
    （常駐程序重開、載入模型要十幾秒才有第一個進度）。停在中間的寫「從中斷的地方接著做」。不用換回 None。"""
    stage = job.get("stage") or ""
    if stage != PAUSED_STAGE and not stage.startswith("排隊中"):
        return None
    return RESUME_STAGE if (job.get("progress") or 0) > 0 else START_STAGE


def resume_points() -> dict[str, float]:
    """執行中的任務這次開始時的進度（job id → 0 到 1）。暫停或中斷後接著做的任務不是 0，
    網頁估剩餘時間時用「這次跑了多久、這次前進了多少」算，不會一繼續就顯示剩不到 1 分鐘。"""
    with _running_lock:
        return {jid: ctx.start_progress for jid, ctx in _running.items()}


class Worker(threading.Thread):
    def __init__(self, name: str, types: tuple):
        super().__init__(name=f"worker-{name}", daemon=True)
        self.types = types
        self.last_key = None

    def pick(self) -> dict | None:
        if self.owns_gpu and queue_paused():
            return None
        candidates = db.runnable_jobs(self.types)
        if candidates and self.owns_gpu:
            # 要用的模型還在下載（第一次打開的自動安裝）：先跳過，裝好再開始，不會用「找不到模型」失敗
            waiting = {j["id"] for j in waiting_for_models(candidates)}
            candidates = [j for j in candidates if j["id"] not in waiting]
        if not candidates:
            return None
        # 和上一個任務同一類模型的優先；語音任務裡，要用的模型已經載入的再優先。省下重新載入模型的時間
        if self.last_key and settings.get("group_by_model"):
            loaded = speech.WORKER.loaded_keys() if self.owns_gpu else set()
            candidates.sort(key=lambda j: (model_key(j) != self.last_key,
                                           model_key(j) == "speech" and speech_model_of(j) not in loaded,
                                           j["position"]))
        with claim_lock:
            if self.owns_gpu and queue_paused():      # 剛按了「釋放顯卡」
                return None
            for job in candidates:
                claimed = db.claim_job(job["id"], stage=starting_stage(job))
                if claimed:
                    return claimed
        return None

    @property
    def owns_gpu(self) -> bool:
        return "translate" in self.types

    def run(self):
        while True:
            job = self.pick()
            if job is None:
                # 只有顯卡那條執行緒做閒置釋放；下載執行緒閒著時如果也來釋放，
                # 會把顯卡執行緒正在用的 llama-server 關掉（WinError 10054 的原因）
                if self.owns_gpu:
                    # 語音模型、翻譯模型共用一個計時：顯卡沒事做滿 GPU_IDLE_RELEASE_S 秒才全部釋放，
                    # 中間加進來的影片、翻譯不用重新載入；下一個任務要的模型放不下時，由 plan_speech、get_llm 先釋放另一邊
                    try:
                        idle_tick()
                    except Exception:  # noqa: BLE001
                        log.warning("idle release check failed", exc_info=True)   # 不能讓顯卡執行緒停掉
                _wake.wait(2)
                _wake.clear()
                continue
            if self.owns_gpu:
                _idle["since"] = None
            self.execute(job)
            self.last_key = model_key(job)
            wake()

    def execute(self, job: dict):
        ctx = JobContext(job)
        with _running_lock:
            _running[ctx.id] = ctx
        handler = HANDLERS[job["type"]]
        log.info("start job %s %s", job["type"], ctx.id)
        # 顯卡任務被「釋放顯卡」停下來（佇列暫停）：回到佇列原來的位置，工作資料夾（檢查點）留著，繼續佇列後接著做
        requeue = job["type"] in GPU_TYPES and job["type"] not in PAUSABLE_TYPES
        try:
            if requeue and queue_paused():
                raise Paused()           # 認領之後、開始之前剛好按了「釋放顯卡」
            handler(ctx)
            stage = f"完成（{ctx.note}）" if ctx.note else "完成"
            db.update_job(ctx.id, status="done", progress=1, stage=stage, result=ctx.result, finished_at=time.time())
            safepath.safe_rmtree(ctx.workdir, WORK_DIR)
        except Paused:
            if requeue:
                _back_to_queue(ctx)
            else:
                # 暫停：暫存檔保留，按「繼續」重新排隊（jobs.resume）
                db.update_job(ctx.id, status="paused", stage="已暫停", finished_at=None)
        except Cancelled:
            db.update_job(ctx.id, status="canceled", stage="已取消", finished_at=time.time())
            db.cancel_dependents(ctx.id)
            # 平常取消會留著工作資料夾，重試時可以續跑；影片已經被刪掉就沒有續跑的機會，直接清掉
            if ctx.media_id and not db.get_media(ctx.media_id):
                safepath.safe_rmtree(ctx.workdir, WORK_DIR)
        except Exception as e:  # noqa: BLE001
            if requeue and ctx.pause_event.is_set() and not ctx.cancel_event.is_set():
                # 按了「釋放顯卡」：模型、子程序被結束時任務可能丟出別的錯誤（例如翻譯模型連線中斷），一樣算暫停
                _back_to_queue(ctx, e)
            else:
                log.error("job %s failed\n%s", ctx.id, traceback.format_exc())
                db.update_job(ctx.id, status="failed", stage="失敗", error=str(e)[:1000], finished_at=time.time())
                db.cancel_dependents(ctx.id)
                if ctx.media_id and not db.get_media(ctx.media_id):
                    safepath.safe_rmtree(ctx.workdir, WORK_DIR)
        finally:
            with _running_lock:
                _running.pop(ctx.id, None)
            if gpu_state["job_id"] == ctx.id:
                gpu_state.update(job_id=None, model=None, stage=None)
            ctx.child = None


def _back_to_queue(ctx: JobContext, error: Exception | None = None):
    """「釋放顯卡」停下來的顯卡任務：改回排隊中（位置不變），工作資料夾的檢查點留著，繼續佇列後從中斷的地方接著做。"""
    log.info("job %s stopped by the release button, back to the queue%s", ctx.id, f" ({error})" if error else "")
    db.update_job(ctx.id, status="queued", stage=PAUSED_STAGE, error=None, finished_at=None)


# run_child 叫起來、會佔顯存的子程序，常駐語音程序（命令列帶著 data\work\speech_worker.log），和模型下載的子程序
# （殘留的話會繼續寫 .part，跟新的下載搶同一個檔案）。下載和常駐語音程序自己也會在伺服器結束時跟著結束（watch_parent）
CHILD_MODULES = ("app.asr_child", "app.separate", "app.model_download", "app.speech_worker", "app.dict_build")


def _same_path(a, b) -> bool:
    if not a or not b:
        return False
    return os.path.normcase(os.path.normpath(str(a))) == os.path.normcase(os.path.normpath(str(b)))


def _under(arg: str, folder: str) -> bool:
    """命令列的某個參數是 folder 底下的路徑（folder 已經 normcase、normpath）。
    要在路徑分隔處斷開：data\\work2、data\\work-old 不算 data\\work 底下。"""
    try:
        p = os.path.normcase(os.path.normpath(str(arg)))
    except (TypeError, ValueError):
        return False
    return p.startswith(folder.rstrip("\\/") + os.sep)


def _module_arg(cmdline: list[str]) -> str | None:
    """python 命令列裡 -m 後面的模組名稱；-m 後面還要有參數（跟 run_child 叫起來的一樣）。"""
    for i, arg in enumerate(cmdline[:-2]):
        if arg == "-m":
            return cmdline[i + 1]
    return None


def find_orphans(llama=None, python=None, work=None) -> list:
    """找出上次留下的殘留子程序（psutil.Process 清單），不靠 PowerShell。
    比對條件要精準，免得殺到別的專案或複製出來的資料夾裡的程式：
      llama-server：執行檔完整路徑等於這個專案用的 llama-server（config.LLAMA_SERVER）
      python：執行檔等於這個伺服器用的 python（uv 的 venv 另外算它背後真正執行的 python），
              命令列是 CHILD_MODULES 裡的模組（-m app.asr_child、app.separate、app.speech_worker……），
              而且帶著這個專案的 data\\work
    而且上層程序已經不在，或上層也是符合條件的殘留（venv 的 python 跳板叫起來的真正 python）。
    還有上層程序在跑的（例如測試腳本或另一個伺服器叫起來的）不動。
    """
    import psutil
    llama = llama or LLAMA_SERVER
    pythons = [python] if python else [sys.executable, getattr(sys, "_base_executable", None)]
    work = os.path.normcase(os.path.normpath(str(work or WORK_DIR)))
    matched = {}
    for proc in psutil.process_iter(["pid", "ppid", "name"]):
        name = (proc.info.get("name") or "").lower()
        if name not in ("llama-server.exe", "python.exe"):
            continue
        try:
            exe = proc.exe()
            cmdline = proc.cmdline() if name == "python.exe" else []
        except (psutil.Error, OSError):
            continue
        if name == "llama-server.exe":
            ok = _same_path(exe, llama)
        else:
            ok = (any(_same_path(exe, p) for p in pythons if p) and _module_arg(cmdline) in CHILD_MODULES
                  and any(_under(arg, work) for arg in cmdline))
        if ok:
            matched[proc.info["pid"]] = (proc, proc.info["ppid"])
    alive = set(psutil.pids())
    orphans: dict = {}
    changed = True
    while changed:
        changed = False
        for pid, (proc, ppid) in matched.items():
            if pid not in orphans and (ppid not in alive or ppid in orphans):
                orphans[pid] = proc
                changed = True
    return [orphans[pid] for pid in sorted(orphans)]


def kill_orphans() -> list[int]:
    """上次程式被直接關掉時，留下的辨識、人聲分離子程序或 llama-server 會繼續佔顯存，啟動時先清掉。
    只清 find_orphans 找到的真正殘留，回傳結束掉的 PID。"""
    try:
        import psutil
    except ImportError:
        log.warning("psutil is not installed, skipping orphan cleanup")
        return []
    killed = []
    try:
        for proc in find_orphans():
            try:
                proc.kill()          # psutil 會先確認 PID 沒有被別的程式重複使用
                killed.append(proc.pid)
            except psutil.Error:
                continue
        if killed:
            log.info("killed orphan processes: %s", killed)
    except Exception:  # noqa: BLE001
        log.warning("orphan cleanup failed", exc_info=True)
    return killed


def discard_workdir(job_id: str) -> bool:
    """刪掉任務的工作資料夾（刪除任務、清除已完成任務時用）。執行中的任務不動。"""
    if not safepath.is_safe_name(job_id):
        return False
    with _running_lock:
        if job_id in _running:
            return False
    job = db.get_job(job_id)
    if job and job["status"] == "running":
        return False
    return safepath.safe_rmtree(WORK_DIR / job_id, WORK_DIR)


def sweep_workdirs() -> int:
    """啟動時清掉用不到的工作資料夾：找不到任務、任務已完成，或影片已經被刪掉的。
    排隊中、失敗、取消的任務會留著，重試時可以從檢查點續跑。只看 12 碼 id 的資料夾，其他檔案不動。"""
    if not WORK_DIR.is_dir():
        return 0
    count = 0
    for d in WORK_DIR.iterdir():
        if not d.is_dir() or not safepath.is_id(d.name):
            continue
        job = db.get_job(d.name)
        keep = job and job["status"] != "done" and not (job["media_id"] and not db.get_media(job["media_id"]))
        if not keep and discard_workdir(d.name):
            count += 1
    if count:
        log.info("removed %d unused work folders", count)
    return count


def start_workers():
    kill_orphans()
    db.reset_interrupted()
    try:
        sweep_workdirs()
    except Exception:  # noqa: BLE001
        log.warning("work folder cleanup failed", exc_info=True)
    Worker("gpu", GPU_TYPES).start()
    Worker("io", IO_TYPES).start()
    Worker("model", MODEL_TYPES).start()
    Worker("dict", DICT_TYPES).start()


# ---------- 子程序（辨識 / 對齊） ----------

def run_child_env() -> dict:
    """辨識、人聲分離子程序的環境變數。ffmpeg 的資料夾放在 PATH 最前面：audio-separator 只會執行 PATH 上的
    ffmpeg，不看 VS_FFMPEG（config.child_env）。"""
    return child_env(PYTHONNOUSERSITE="1", PYTHONIOENCODING="utf-8", HF_HUB_OFFLINE="1",
                     TRANSFORMERS_OFFLINE="1", TRANSFORMERS_VERBOSITY="error")


def run_child(ctx: JobContext, args: list[str], lo: float, hi: float, model_label: str,
              module: str = "app.asr_child"):
    """跑一個會用顯卡的步驟。辨識、對齊、檢查時間軸（asr_child 的階段）交給常駐語音程序（run_speech）；
    人聲分離和 VS_SPEECH_WORKER=0 時照舊開一個子程序，跑完就結束。"""
    if module == "app.asr_child" and speech.enabled():
        return run_speech(ctx, args, lo, hi, model_label)
    return run_child_process(ctx, args, lo, hi, model_label, module)


def run_child_process(ctx: JobContext, args: list[str], lo: float, hi: float, model_label: str,
                      module: str = "app.asr_child"):
    """開一個子程序跑完就結束，顯存全部歸還（人聲分離、單次模式的辨識和對齊）。"""
    release_llm("辨識任務需要顯存")  # 翻譯模型還開著的話先關掉
    gpu_state.update(job_id=ctx.id, model=model_label)
    env_cmd = [sys.executable, "-s", "-m", module, *args]
    env = run_child_env()
    # asr_child 的第一個參數是階段名稱；其他子程序（例如 app.separate）第一個參數是檔案的絕對路徑，
    # 拿來當檔名會變成寫到使用者的影片資料夾，改用模組名稱
    log_name = args[0] if module == "app.asr_child" else module.rsplit(".", 1)[-1]
    err_path = ctx.workdir / f"{log_name}.err.log"
    with open(err_path, "w", encoding="utf-8", errors="replace") as err:
        ctx.child = subprocess.Popen(
            env_cmd, cwd=str(ROOT), env=env, stdout=subprocess.PIPE, stderr=err,
            text=True, encoding="utf-8", errors="replace", creationflags=subprocess.CREATE_NO_WINDOW,
        )
        for line in ctx.child.stdout:
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            ctx.progress(lo + (hi - lo) * float(msg.get("p", 0)), msg.get("stage"))
        code = ctx.child.wait()
    ctx.child = None
    ctx.check()
    if code != 0:
        raise child_failure(err_path.read_text(encoding="utf-8", errors="replace"), model_label)


def child_failure(stderr: str, model_label: str) -> RuntimeError:
    """子程序失敗時給使用者看的原因。CUDA 用不了（驅動裝壞、CUDA_VISIBLE_DEVICES 把顯示卡藏起來）時
    torch 只會印英文，改成跟翻譯一樣的中文說明；丟 GpuUnsupported，自動歌曲模式也不會改用原音硬跑下去。"""
    tail = [l for l in stderr.strip().splitlines() if l.strip()]
    last = tail[-1] if tail else "未知錯誤"
    if syscheck.gpu_too_new("\n".join(tail[-15:])):
        return gpu.GpuUnsupported(syscheck.too_new_message(f"{model_label} "))
    if syscheck.cuda_unusable("\n".join(tail[-15:])):
        return gpu.GpuUnsupported(syscheck.no_cuda_message(f"{model_label} "))
    if "OutOfMemory" in last or "out of memory" in last.lower():
        return RuntimeError("顯存不足，辨識模型無法載入。請關閉其他使用顯卡的程式後按「重試」。")
    return RuntimeError(f"{model_label} 執行失敗：{last}")


# ---------- 常駐語音模型（app/speech.py、app/speech_worker.py） ----------

class _StopFlag:
    """取消或「釋放顯卡」暫停任一個發生，常駐程序就在下一批開始前停下這個請求（給 speech.Worker 的 cancel_event）。"""

    def __init__(self, ctx: JobContext):
        self.ctx = ctx

    def is_set(self) -> bool:
        return self.ctx.cancel_event.is_set() or self.ctx.pause_event.is_set()


def run_speech(ctx: JobContext, args: list[str], lo: float, hi: float, model_label: str):
    """在常駐語音程序裡跑 asr_child 的一個階段。模型放得下就留著給下一步、下一部影片用。
    取消：程序在下一批開始前停下來（模型留著）；等太久才結束程序。程序當掉：這個任務失敗（檢查點留著，
    重試時續跑），下次需要時自動開新的。程序根本開不起來（不是顯示卡的問題）：這次開機改用單次子程序。"""
    ctx.check()                          # 取消、「釋放顯卡」之後不要再把模型載回來
    plan = ctx.speech_plan
    if plan is None:
        release_llm("辨識任務需要顯存")   # 沒有事先規劃過：跟以前一樣先關掉翻譯模型
    gpu_state.update(job_id=ctx.id, model=model_label)

    def on_progress(msg):
        ctx.progress(lo + (hi - lo) * float(msg.get("p", 0)), msg.get("stage"))

    for attempt in (1, 2):
        try:
            reply = speech.WORKER.run(args, plan=plan, on_progress=on_progress, cancel_event=_StopFlag(ctx))
        except speech.WorkerStartError as e:
            ctx.check()
            err = child_failure(e.tail, model_label)
            if isinstance(err, gpu.GpuUnsupported) or "out of memory" in (e.tail or "").lower():
                raise err       # 顯示卡、顯存的問題：單次子程序也一樣跑不了
            # 常駐程序在這台電腦上開不起來（例如被防毒、系統設定擋住）：這次開機改用以前的單次子程序，任務照常完成
            speech.disable(f"{e}: {str(err)[:300]}")
            return run_child_process(ctx, args, lo, hi, model_label)
        except speech.WorkerError as e:
            ctx.check()
            raise child_failure(e.tail, model_label)
        ctx.check()
        if reply.get("ok") or reply.get("error") != "oom" or attempt == 2 or not _llm_vram_mb():
            break
        # 工作程序卸載了別的模型、批次降到 1 還是放不下，而翻譯模型還開著（規劃時判斷可以一起放，
        # 之後其他程式多用了顯存）：關掉翻譯模型再試一次，不讓任務失敗
        log.warning("job %s: %s out of GPU memory with the translation model loaded, releasing it and retrying",
                    ctx.id, args[0])
        release_llm("辨識時顯存不足")
        if plan is not None:
            plan.tight = True
    if not reply.get("ok"):
        if reply.get("error") == "canceled":
            raise Cancelled()
        raise child_failure(reply.get("tail") or reply.get("message") or "", model_label)
    if reply.get("oom_retries"):
        log.warning("job %s: %s ran out of GPU memory %d time(s), unloaded %s, batch %s", ctx.id, args[0],
                    reply["oom_retries"], reply.get("evicted"), reply.get("bs"))
    on_progress({"p": 1.0})


def plan_speech(ctx: JobContext, engine_key: str | None, health_on: bool, speech_s: float):
    """開始用語音模型之前呼叫一次：照現在的顯存決定這個任務每個階段的批次大小（整個任務不變，
    同一台電腦的結果才穩定）、翻譯模型要不要先關。結果放在 ctx.speech_plan 並回傳。
    VS_SPEECH_WORKER=0 時回 None，照舊在每一步之前依剩餘顯存查表（speech_batch）。"""
    if not speech.enabled():
        return None
    llm_mb = _llm_vram_mb()
    if llm_mb is None:                  # 算不出翻譯模型佔多少：跟以前一樣先關掉
        release_llm("辨識任務需要顯存")
        llm_mb = 0
    plan = speech.plan_for_job(engine_key, health_on, speech_s, llm_mb=llm_mb)
    if plan.release_llm:
        release_llm("辨識任務需要顯存")
    log.info("job %s speech plan: %s %s", ctx.id, plan.mode, plan.bs)
    ctx.speech_plan = plan
    return plan


def _llm_vram_mb() -> int | None:
    """開著的 llama-server 大約佔多少顯存（沒開 0，算不出來 None）。"""
    with _llm_lock:
        server, key = _llm["server"], _llm["key"]
        if not server or not key or not server.alive():
            return 0
        try:
            return translator_vram_mb(key, cfg=getattr(server, "cfg", None) or config.translator_cfg(key))
        except Exception:  # noqa: BLE001
            return None


def speech_batch(ctx: JobContext, stage: str, engine_key: str | None = None) -> int:
    """這個任務某個階段一次處理幾段：有規劃照規劃；沒有（VS_SPEECH_WORKER=0）照舊依剩餘顯存查表。"""
    plan = ctx.speech_plan
    if plan is not None:
        return plan.batch(stage, engine_key)
    if stage == "asr":
        engine = ASR_ENGINES[engine_key]
        return gpu.pick_batch(engine["batch_by_free_mb"], engine["label"])
    if stage == "verify":
        # 檢查的片段是一行字幕加前後 0.3 秒，通常不到 8 秒，同樣的顯存放得下好幾倍的筆數
        from . import health
        engine = ASR_ENGINES["qwen"]
        return health.verify_batch_size(gpu.pick_batch(engine["batch_by_free_mb"], engine["label"]))
    return gpu.pick_batch(ALIGNER["batch_by_free_mb"], ALIGNER["label"])


def release_speech(mid: str, label: str):
    """設定頁刪除模型前呼叫（models.delete，拿著 claim_lock、確認過沒有執行中的任務在用）：常駐程序還載著就卸載。"""
    speech.release_model(mid, f"刪除模型 {label}")


def loaded_models() -> list[dict]:
    """現在顯卡上載入的模型（給佇列上方的顯卡狀態、設定頁顯示）。"""
    out = [{**m, "kind": "speech"} for m in speech.loaded_view()]
    server, key = _llm["server"], _llm["key"]
    if server is not None and key and server.alive():
        cfg = getattr(server, "cfg", None) or {}
        label = cfg.get("label") or TRANSLATORS.get(key, {}).get("label", key)
        out.append({"key": key, "label": label, "short": label, "mb": None, "kind": "llm"})
    return out


# ---------- 任務：轉字幕 ----------

def _song_mode_wanted(params: dict) -> bool:
    """歌曲模式：任務參數 song 可以強制開（True）或關（False）；沒指定就看設定（預設自動）。"""
    if params.get("song") is False:
        return False
    return params.get("song") is True or settings.get("song_mode") != "off"


# 歌曲模式轉的字幕，字幕軌的模型名稱後面會加上這段；事後檢查時間軸靠它知道要先分離人聲
SONG_LABEL = "＋人聲分離"


SEPARATION_FALLBACK_NOTE = "人聲分離失敗，改用原音辨識"


def _separate_vocals(ctx: "JobContext", src: Path, vocals: Path) -> bool:
    """轉字幕時分離人聲，成功（或上次已經分離好）回 True。
    自動歌曲模式失敗時記 log、任務訊息加一句「人聲分離失敗，改用原音辨識」，回 False 照原音繼續；
    只有使用者強制開歌曲模式（params.song 是 True）才讓任務失敗。"""
    if vocals.exists():
        return True
    try:
        # 常駐的語音模型佔著顯存、放不下人聲分離時先卸載（8、12 GB 的卡才會用到；16 GB 放得下就留著）
        speech.make_room(SEPARATOR["vram_mb"], "人聲分離需要顯存")
        gpu.ensure_free(SEPARATOR["vram_mb"], SEPARATOR["label"])
        run_child(ctx, [str(src), str(vocals)], 0.07, 0.10, SEPARATOR["label"], module="app.separate")
        return True
    except (Cancelled, Paused, gpu.GpuUnsupported):
        raise          # 沒有能用的顯示卡：後面的辨識也跑不了，直接用這個原因失敗
    except Exception:  # noqa: BLE001
        ctx.check()    # 是被取消、「釋放顯卡」結束掉的，不是分離失敗：不要改用原音做下去
        if (ctx.params or {}).get("song") is True:
            raise
        log.warning("job %s: vocal separation failed, using the original audio", ctx.id, exc_info=True)
        safepath.safe_unlink(vocals, ctx.workdir)       # 失敗時可能留下半截的檔案
        ctx.note = SEPARATION_FALLBACK_NOTE
        ctx.progress(0.10, "偵測人聲")
        return False
    finally:
        gpu_state.update(model=None)


def aligner_installed() -> bool:
    entry = next((e for e in MODEL_CATALOG.values() if e.get("role") == "aligner"), None)
    return entry is None or model_installed(entry)


def require_aligner(ctx: "JobContext | None" = None):
    if not aligner_installed():
        raise RuntimeError(missing_model_message(ALIGNER["label"]))
    if ctx is not None:
        require_model(ctx, catalog_id("role", "aligner"))


def handle_transcribe(ctx: JobContext):
    p = ctx.params
    lang, engine_key = p["language"], p["engine"]
    # 舊任務用 profile=asmr 表示耳語內容，新任務改用 sensitive
    sensitive = bool(p.get("sensitive")) or p.get("profile") == "asmr"
    engine = ASR_ENGINES[engine_key]
    m = db.get_media(ctx.media_id)
    if not m or not m.get("path"):
        raise RuntimeError("找不到影片檔")
    src = Path(m["path"])
    if not src.exists():
        raise RuntimeError(f"影片檔不存在：{src}")
    # 排隊時模型還在、輪到時已經在設定頁刪掉了：先說清楚，不要抽完音訊才在子程序裡失敗
    if engine_key not in settings.installed_engines():
        raise RuntimeError(missing_model_message(engine["label"]))
    # 登記要用的模型（任務結束前設定頁刪不掉）。時間軸檢查在任務開始時就決定，
    # 之後使用者改設定或刪掉 Qwen3-ASR 都不會讓跑到一半的任務改變做法
    require_model(ctx, catalog_id("engine", engine_key))
    require_aligner(ctx)
    health_on = (bool(settings.get("health_check")) and "qwen" in settings.installed_engines()
                 and use_model(ctx, catalog_id("engine", "qwen")))
    ctx.workdir.mkdir(parents=True, exist_ok=True)

    wav_path = ctx.workdir / "audio.wav"
    if not wav_path.exists():
        ctx.progress(0, "抽取音訊")
        media.extract_audio(src, wav_path, m.get("duration") or 0,
                            on_progress=lambda f: ctx.progress(0.04 * f, "抽取音訊"), check=ctx.check)

    chunks_path = ctx.workdir / "chunks.json"
    song_path = ctx.workdir / "song.json"
    if not chunks_path.exists():
        ctx.progress(0.04, "偵測人聲")
        data, _ = sf.read(wav_path, dtype="float32")
        max_speech = engine["max_chunk_s"] - 2
        segs = vad.speech_segments(
            data, sensitive=sensitive, max_speech_s=max_speech,
            on_progress=lambda f: ctx.progress(0.04 + 0.03 * f, "偵測人聲"), check=ctx.check,
        )
        total_s = len(data) / SAMPLE_RATE
        cover = vad.coverage(segs, total_s)
        del data
        song = False
        # 配樂很滿的影片（歌曲、PV）在原音上幾乎偵測不到人聲。
        # 實測一首配樂很滿的日文歌：原音覆蓋率 1%，先分離人聲再偵測是 63%；
        # 辨識和對齊也改用分離後的人聲，對照正確歌詞字錯率 36.6% → 30%，也不再出現 36 秒的長行。
        vocals = ctx.workdir / "vocals.wav"
        # 人聲分離是選用的：模型不在（包括排隊時被刪掉）就照原音處理；要用時先登記，分離途中設定頁刪不掉
        if (cover < vad.MIN_COVERAGE and total_s > 30 and _song_mode_wanted(p)
                and (vocals.exists() or use_model(ctx, catalog_id("role", "separator")))
                and _separate_vocals(ctx, src, vocals)):
            voc, _ = sf.read(vocals, dtype="float32")
            vsegs = vad.speech_segments(voc, sensitive=sensitive, max_speech_s=max_speech, check=ctx.check)
            vcover = vad.coverage(vsegs, total_s)
            if vcover < vad.MIN_COVERAGE and not sensitive:
                vsegs = vad.speech_segments(voc, sensitive=True, max_speech_s=max_speech, check=ctx.check)
                vcover = vad.coverage(vsegs, total_s)
            del voc
            log.info("song mode: vad coverage %.0f%% on mix, %.0f%% on vocals", cover * 100, vcover * 100)
            if vcover > cover and vcover >= 0.08:
                # 之後的辨識、對齊、時間軸檢查都讀 audio.wav，換成人聲；原音留一份
                wav_path.replace(ctx.workdir / "audio_mix.wav")
                shutil.copyfile(vocals, wav_path)
                segs, cover, song = vsegs, vcover, True
        if not song and cover < vad.MIN_COVERAGE and not sensitive and total_s > 30:
            ctx.progress(0.08, "人聲偵測結果偏少，重新偵測")
            data, _ = sf.read(wav_path, dtype="float32")
            segs = vad.speech_segments(data, sensitive=True, max_speech_s=max_speech, check=ctx.check)
            cover = vad.coverage(segs, total_s)
            del data
        floor = 0.08 if (sensitive or song) else vad.MIN_COVERAGE
        if cover < floor:
            # 還是太少就整段都送去辨識，寧可多跑也不要漏掉（辨識模型遇到純音樂會輸出空白）
            log.info("vad coverage %.0f%% too low, using fixed chunks", cover * 100)
            chunks = vad.fixed_chunks(total_s, engine["max_chunk_s"])
        else:
            chunks = vad.build_chunks(segs, total_s, engine["max_chunk_s"])
        if not chunks:
            raise RuntimeError("整段影片沒有偵測到人聲")
        song_path.write_text(json.dumps({"song": song}), encoding="utf-8")
        chunks_path.write_text(json.dumps(chunks), encoding="utf-8")
    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    song = _read_json(song_path, {}).get("song", False)

    # 常駐語音模型：這個任務的批次大小、要不要先關翻譯模型，在這裡一次決定
    plan = plan_speech(ctx, engine_key, health_on, sum(e - s for s, e in chunks))
    if health_on and plan is not None and plan.health_skipped:
        log.warning("job %s: skipping the timing check: %s", ctx.id, plan.health_skipped)
        health_on = False

    # 有開時間軸檢查（而且 Qwen3-ASR 已下載，開頭決定的 health_on）時，辨識和對齊讓出後段進度給檢查與補正
    asr_hi, align_hi = (0.68, 0.78) if health_on else (0.80, 0.95)

    batch = speech_batch(ctx, "asr", engine_key)
    run_child(ctx, ["asr", str(ctx.workdir), engine_key, lang, str(batch)], 0.10, asr_hi, engine["label"])

    batch = speech_batch(ctx, "align")
    run_child(ctx, ["align", str(ctx.workdir), lang, str(batch)], asr_hi, align_hi, ALIGNER["label"])
    gpu_state.update(model=None)

    ctx.progress(align_hi, "產生字幕")
    texts = json.loads((ctx.workdir / "asr.json").read_text(encoding="utf-8"))
    aligned = json.loads((ctx.workdir / "align.json").read_text(encoding="utf-8")) if (ctx.workdir / "align.json").exists() else {}
    tokens = []
    for i, chunk in enumerate(chunks):
        tokens.extend(cue_mod.attach_units(texts.get(str(i), ""), aligned.get(str(i), []), chunk))
    track_lang = LANGUAGES[lang]["track"]
    cue_list = cue_mod.finalize(cue_mod.segment(tokens, lang), track_lang)
    if not cue_list:
        raise RuntimeError("沒有辨識出任何文字")
    health_summary = None
    if health_on:
        # 檢查失敗不影響字幕本身：記下來，之後可以在字幕管理裡單獨檢查
        try:
            from . import health
            fixed, health_summary = run_health(ctx, cue_list, lang, 0.78, 0.95)
            health.apply_times(cue_list, fixed, health_summary["bad"])
        except (Cancelled, Paused):
            raise
        except Exception:  # noqa: BLE001
            ctx.check()    # 「釋放顯卡」結束了模型：任務回到佇列，繼續後接著檢查，不要存一條沒檢查過的字幕
            log.warning("health check failed for job %s", ctx.id, exc_info=True)
            health_summary = None
            gpu_state.update(model=None)
        ctx.progress(0.96, "產生字幕")
    ruby_ver = None
    if lang == "ja":
        try:
            from . import furigana
            if furigana.apply(cue_list) >= 0:
                ruby_ver = furigana.version()
        except Exception:  # noqa: BLE001
            log.warning("furigana failed", exc_info=True)

    if not db.get_media(ctx.media_id):
        raise Cancelled()
    label = engine["label"] + (SONG_LABEL if song else "")
    track_id = db.add_track(ctx.media_id, track_lang, "asr", label, len(cue_list))
    cue_mod.save_cues(track_id, cue_list)
    fields = {}
    if ruby_ver:
        fields["ruby_ver"] = ruby_ver
    if health_summary:
        fields["health"] = health_summary
    if fields:
        db.update_track(track_id, **fields)
    ctx.result = {"track_id": track_id}


# ---------- 時間軸健康檢查（轉字幕與 health 任務共用） ----------

def _read_json(path: Path, default):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def _write_json(path: Path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def run_health(ctx: JobContext, cue_list: list[dict], lang: str, lo: float, hi: float) -> tuple[list[dict], dict]:
    """檢查並補正時間軸（演算法在 app/health.py，模型在常駐語音程序裡跑，見 run_child）。

    工作資料夾要先有 audio.wav。回傳 (整理後每行的 start/end/text, 摘要)。
    每一輪的進度照 檢查 10：修正 4：再檢查 3 分配，轉字幕時就是 0.78–0.88–0.92–0.95。
    """
    from . import health

    wd = ctx.workdir
    engine = ASR_ENGINES["qwen"]
    total = sf.info(str(wd / "audio.wav")).duration
    # 檢查的片段是一行字幕加前後 0.3 秒，通常不到 8 秒，同樣的顯存放得下好幾倍的筆數
    # （實際分批時還會照片段長度限制一批的總長度，見 health.verify_batches）
    asr_bs = speech_batch(ctx, "verify")
    cur = health.slim(cue_list)
    first_rec, final_rec, fixed, carry = None, {}, set(), {}
    span = (hi - lo) / health.HEALTH_ROUNDS
    rounds = 0
    for r in range(1, health.HEALTH_ROUNDS + 1):
        rounds = r
        base = lo + span * (r - 1)
        p1, p2, p3 = base + span * 10 / 17, base + span * 14 / 17, base + span
        f = health.round_files(r)
        if _read_json(wd / f["cues"], None) != cur:
            # 第一次跑，或字幕跟上次中斷時不一樣：這一輪和後面幾輪的檢查點都作廢
            for rr in range(r, health.HEALTH_ROUNDS + 1):
                for name in health.round_files(rr).values():
                    safepath.safe_unlink(wd / name, wd)
            _write_json(wd / f["cues"], cur)
        if carry:
            # 時間沒變的行沿用上一輪的 recall，verify 只會檢查缺的行
            known = _read_json(wd / f["verify"], {})
            for i, v in carry.items():
                known.setdefault(str(i), v)
            _write_json(wd / f["verify"], known)

        run_child(ctx, ["verify", str(wd), f["cues"], f["verify"], lang, str(asr_bs)], base, p1, engine["label"])
        rec = {int(k): v for k, v in _read_json(wd / f["verify"], {}).items()}
        if first_rec is None:
            first_rec = rec
        final_rec = rec
        if not health.plan_regions(cur, rec, total):
            break  # 沒有對不上的行

        al_bs = speech_batch(ctx, "realign")
        run_child(ctx, ["realign", str(wd), f["cues"], f["verify"], f["fix"], f["touched"], lang, str(al_bs)],
                  p1, p2, ALIGNER["label"])
        touched = _read_json(wd / f["touched"], {}).get("touched", [])
        if not touched:
            break

        asr_bs = speech_batch(ctx, "verify") if ctx.speech_plan is not None else asr_bs   # 爆過顯存時規劃會縮小批次
        run_child(ctx, ["verify", str(wd), f["fix"], f["verify_fix"], lang, str(asr_bs), f["touched"]],
                  p2, p3, engine["label"])
        fix = _read_json(wd / f["fix"], [])
        rec_fix = {int(k): v for k, v in _read_json(wd / f["verify_fix"], {}).items()}
        out, adopted, final_rec = health.adopt(cur, fix, rec, rec_fix, touched)
        carry = health.carry_over(cur, fix, out, final_rec, rec, rec_fix)
        cur = out
        fixed |= adopted
        if not adopted:
            break  # 這一輪沒有任何改變，再跑一輪結果也一樣
    gpu_state.update(model=None)
    ctx.progress(hi)
    return cur, health.summarize(cur, first_rec or {}, final_rec, fixed, rounds)


def _cue_lock():
    """改寫字幕檔要跟改讀音、查字補算用同一把鎖（server 透過 vocab.set_cue_lock 交給 vocab）。
    只在 vocab 已經載入時借用，不為了拿鎖去載入它。"""
    from contextlib import nullcontext
    vocab = sys.modules.get(f"{__package__}.vocab")
    return getattr(vocab, "_cue_lock", None) or nullcontext()


def _backup_cues(track_id: str):
    """第一次改時間前留一份原本的字幕檔；已經有備份就不覆蓋。"""
    from . import health
    src, bak = cue_mod.cue_path(track_id), health.backup_path(track_id)
    if src.exists() and not bak.exists():
        shutil.copyfile(src, bak)


def discard_track(track_id: str):
    """刪掉一條字幕軌：字幕檔、時間軸修正前的備份、資料列、查字索引。刪字幕、刪影片、重翻取代舊翻譯都用這個。"""
    from . import health
    if safepath.is_safe_name(track_id):
        safepath.safe_unlink(cue_mod.cue_path(track_id), cue_mod.SUBS_DIR)
        safepath.safe_unlink(health.backup_path(track_id), cue_mod.SUBS_DIR)
    db.delete_track(track_id)
    vocab = sys.modules.get(f"{__package__}.vocab")  # 伺服器一定有載入；單獨跑任務的測試沒載入就不用清索引
    if vocab:
        vocab.forget_track(track_id)


def translation_source(job: dict) -> str | None:
    """翻譯任務要翻哪一條字幕。新增影片時排的翻譯任務，要等前面的轉字幕做完才知道是哪一條。"""
    source_id = (job.get("params") or {}).get("source_track_id")
    if not source_id and job.get("depends_on"):
        dep = db.get_job(job["depends_on"])
        source_id = ((dep or {}).get("result") or {}).get("track_id")
    return source_id


# ---------- 任務：檢查已有字幕的時間軸 ----------

def handle_health(ctx: JobContext):
    from . import health

    tid = ctx.params.get("track_id")
    t = db.get_track(tid) if tid else None
    if not t:
        raise RuntimeError("找不到要檢查的字幕軌")
    if t["kind"] != "asr":
        raise RuntimeError("只有語音辨識產生的字幕可以檢查時間軸")
    lang = next((k for k, v in LANGUAGES.items() if v["track"] == t["lang"]), None)
    if not lang:
        raise RuntimeError(f"不支援這個語言：{t['lang']}")
    if "qwen" not in settings.installed_engines():
        raise RuntimeError(f"檢查時間軸需要 {ASR_ENGINES['qwen']['label']}，請到設定頁下載")
    # 歌曲模式的字幕是用分離後的人聲辨識、對齊、檢查的，事後檢查也要用同一種音訊，結果才不會互相矛盾
    song = (t.get("model") or "").endswith(SONG_LABEL)
    if song and not use_model(ctx, catalog_id("role", "separator")):
        raise RuntimeError(f"這條字幕是先分離人聲再辨識的，檢查時間軸也需要 {SEPARATOR['label']}，請到設定頁下載")
    if not use_model(ctx, catalog_id("engine", "qwen")):
        raise RuntimeError(f"檢查時間軸需要 {ASR_ENGINES['qwen']['label']}，請到設定頁下載")
    require_aligner(ctx)
    m = db.get_media(t["media_id"])
    src = Path(m["path"]) if m and m.get("path") else None
    if not src or not src.exists():
        raise RuntimeError("找不到影片檔")
    ctx.workdir.mkdir(parents=True, exist_ok=True)

    wav_path = ctx.workdir / "audio.wav"
    if not wav_path.exists():
        if song:
            vocals = ctx.workdir / "vocals.wav"
            if not vocals.exists():
                speech.make_room(SEPARATOR["vram_mb"], "人聲分離需要顯存")
                gpu.ensure_free(SEPARATOR["vram_mb"], SEPARATOR["label"])
                run_child(ctx, [str(src), str(vocals)], 0, 0.12, SEPARATOR["label"], module="app.separate")
                gpu_state.update(model=None)
            shutil.copyfile(vocals, wav_path)
        else:
            ctx.progress(0, "抽取音訊")
            media.extract_audio(src, wav_path, m.get("duration") or 0,
                                on_progress=lambda f: ctx.progress(0.05 * f, "抽取音訊"), check=ctx.check)
    before = cue_mod.load_cues(tid)
    if not before:
        raise RuntimeError("這條字幕是空的")

    plan_speech(ctx, None, True, sf.info(str(wav_path)).duration)
    new, summary = run_health(ctx, before, lang, 0.12 if song else 0.05, 0.97)
    ctx.check()
    ctx.progress(0.98, "儲存字幕")
    skipped, synced = [], 0
    with _cue_lock():
        if not db.get_track(tid):
            raise Cancelled()
        cues = cue_mod.load_cues(tid)
        if [c["text"] for c in cues] != [c["text"] for c in new]:
            raise RuntimeError("檢查期間字幕內容有變動，請重新檢查一次")
        old = [{"start": c["start"], "end": c["end"]} for c in cues]
        moved = health.times_changed(old, new)
        # 只改 start/end 和 chk；文字沒變，ruby、w 的字元位置都還有效
        if health.apply_times(cues, new, summary["bad"]):
            if moved:
                _backup_cues(tid)
            cue_mod.save_cues(tid, cues)
        if moved:
            # 翻譯綁定來源字幕：逐行翻譯的時間跟著原文改
            for x in db.list_tracks(t["media_id"]):
                if x["kind"] != "translation" or x["source_track_id"] != tid:
                    continue
                tr = cue_mod.load_cues(x["id"])
                out = health.sync_translation(old, cues, tr)
                if out is None:
                    skipped.append({"id": x["id"], "model": x["model"],
                                    "reason": "時間沒辦法逐行對應原文（整句翻譯，或原文有幾行擠在同一個時間）"})
                elif out != tr:
                    _backup_cues(x["id"])
                    cue_mod.save_cues(x["id"], out)
                    synced += 1
    summary["skipped_translations"] = skipped
    summary["synced_translations"] = synced
    db.update_track(tid, health=summary)
    ctx.result = {"track_id": tid, "fixed": summary["fixed"], "after_bad": summary["after_bad"]}


# ---------- 任務：翻譯 ----------

def handle_translate(ctx: JobContext):
    p = ctx.params
    key = p["translator"]
    cfg = TRANSLATORS[key]
    source_id = translation_source(ctx.job)
    src_track = db.get_track(source_id) if source_id else None
    if not src_track:
        raise RuntimeError("找不到要翻譯的字幕軌")
    src_cues = cue_mod.load_cues(source_id)
    src_lang = p.get("language") or ("en" if src_track["lang"] == "en" else "ja")
    mode = p.get("mode") or settings.get("translation_mode") or "line"
    units = translate.build_units(src_cues, src_lang)
    texts = [translate.unit_text(src_cues, u, src_lang) for u in units]
    ctx.workdir.mkdir(parents=True, exist_ok=True)

    # 兩種模式的檢查點分開存：逐行模式以字幕行為 key，整句模式以句子為 key
    ckpt = ctx.workdir / f"translate-{mode}.json"
    split_ckpt = ctx.workdir / "splits.json"
    done = json.loads(ckpt.read_text(encoding="utf-8")) if ckpt.exists() else {}
    splits = json.loads(split_ckpt.read_text(encoding="utf-8")) if split_ckpt.exists() else {}

    def _write(path: Path, data):
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)

    def save():
        _write(ckpt, done)

    def save_splits():
        _write(split_ckpt, splits)

    if mode == "line":
        pending = len(done) < len(src_cues)
    else:
        pending = len(done) < len(texts) or any(
            len(u) > 1 and len(done.get(str(k), "")) > translate.SPAN_MAX_CHARS and str(k) not in splits
            for k, u in enumerate(units)
        )
    if pending:
        require_model(ctx, catalog_id("translator", key))
        gpu_state.update(job_id=ctx.id, model=cfg["label"])
        ctx.progress(0, f"載入 {cfg['label']}")
        try:
            # llama-server 會跨任務沿用、一直開著 log，放在任務的工作資料夾會讓資料夾刪不掉
            server, reused = get_llm(key, WORK_DIR / "translate-llama-server.log", ctx.check)
            if mode == "line":
                runner = translate.translate_lines_sakura if key in SAKURA_STYLE else translate.translate_lines_hymt
                runner(server, src_cues, src_lang, done, lambda f: ctx.progress(0.98 * f, "逐行翻譯"), ctx.check, save)
            else:
                runner = translate.translate_sakura if key in SAKURA_STYLE else translate.translate_hymt
                runner(server, texts, src_lang, done, lambda f: ctx.progress(0.9 * f, "翻譯中"), ctx.check, save)
                # 太長的譯文要分行顯示時，交給模型按原文每一行的意思重新分段
                translate.align_splits(server, src_cues, units, done, splits,
                                       lambda f: ctx.progress(0.9 + 0.08 * f, "對齊譯文"), ctx.check, save_splits)
        finally:
            gpu_state.update(model=None)

        # 模型已經載入了，順便把還沒翻的影片標題一起翻掉
        if _llm["key"] == key:
            try:
                translate_missing_titles(_llm["server"], key, ctx.check, limit=40)
            except Cancelled:
                raise
            except Exception:  # noqa: BLE001
                log.warning("title translation failed", exc_info=True)

    ctx.progress(0.99, "產生字幕")
    if mode == "line":
        out = translate.finalize_lines(src_cues, done)
    else:
        out = translate.finalize_translation(src_cues, units, done, splits)
    # 翻譯途中影片或原文字幕被刪掉了，就不要再新增一條沒有原文的翻譯
    if not db.get_media(ctx.media_id) or not db.get_track(source_id):
        raise Cancelled()
    track_id = db.add_track(ctx.media_id, "zh-TW", "translation", cfg["label"], len(out), source_track_id=source_id)
    cue_mod.save_cues(track_id, out)
    db.update_track(track_id, mode=mode)
    # 批次重翻：新的翻譯完成後才刪掉舊的，中間不會沒有字幕
    old_id = p.get("replace_track_id")
    if old_id and old_id != track_id and db.get_track(old_id):
        discard_track(old_id)
    ctx.result = {"track_id": track_id}


# ---------- 標題翻譯 ----------

def translate_missing_titles(server, key: str, check, limit: int = 200, on_progress=None) -> int:
    langs = TRANSLATORS[key]["langs"]
    todo = [m for m in db.media_needing_title_translation(limit) if m["language"] in langs]
    for n, m in enumerate(todo, 1):
        check()
        zh = translate.translate_title(server, key, m["title"])
        if zh:
            db.update_media(m["id"], title_zh=zh)
        if on_progress:
            on_progress(n / len(todo))
    return len(todo)


def handle_titles(ctx: JobContext):
    key = ctx.params.get("translator") or settings.translator_for("ja")
    cfg = TRANSLATORS[key]
    if not db.media_needing_title_translation(1):
        return
    require_model(ctx, catalog_id("translator", key))
    gpu_state.update(job_id=ctx.id, model=cfg["label"])
    ctx.progress(0, f"載入 {cfg['label']}")
    server, _ = get_llm(key, WORK_DIR / "titles-llama-server.log", ctx.check)
    count = translate_missing_titles(server, key, ctx.check, on_progress=lambda f: ctx.progress(f, "翻譯標題"))
    ctx.result = {"titles": count}


# ---------- 任務：日文單字的中文釋義 ----------

def handle_glosses(ctx: JobContext):
    """把 JMdict 英文義項翻成繁體中文存快取。用 get_llm 沿用已載入的 Hy-MT2，不另外開 llama-server。"""
    from . import glossgen
    key = ctx.params.get("translator") or glossgen.translator_key()
    if key not in TRANSLATORS or key not in settings.installed_translators():
        key = glossgen.translator_key()
    if not key:
        raise RuntimeError("產生單字釋義需要 Hy-MT2-7B，請到設定頁下載")
    ctx.progress(0, "整理要翻譯的義項")
    items = glossgen.collect(ctx.params.get("scope", "media"), ctx.media_id)
    if not items:
        ctx.result = {"glosses": 0}
        return
    cfg = TRANSLATORS[key]
    require_model(ctx, catalog_id("translator", key))
    gpu_state.update(job_id=ctx.id, model=cfg["label"])
    ctx.progress(0, f"載入 {cfg['label']}")
    try:
        server, _ = get_llm(key, WORK_DIR / "glosses-llama-server.log", ctx.check)
        n = glossgen.run(server, items, cfg["label"], ctx.check,
                         lambda f: ctx.progress(f, f"翻譯單字釋義（共 {len(items)} 個義項）"))
    finally:
        gpu_state.update(model=None)
    ctx.result = {"glosses": n}


# ---------- 任務：轉相容播放檔 ----------

def handle_proxy(ctx: JobContext):
    release_llm("轉檔需要顯存")
    m = db.get_media(ctx.media_id)
    if not m:
        raise RuntimeError("找不到影片")
    PROXY_DIR.mkdir(parents=True, exist_ok=True)
    dst = PROXY_DIR / f"{ctx.media_id}.mp4"
    gpu_state.update(job_id=ctx.id, model="NVENC 轉檔")
    ctx.progress(0, "轉檔中")

    def on_note(note: str):
        # NVENC 不能用，改用 CPU 的 libx264：進度訊息說明會比較慢
        ctx.note = note
        gpu_state.update(model="CPU 轉檔")
        ctx.progress(0, "轉檔中")

    media.make_proxy(Path(m["path"]), dst, m.get("duration") or 0,
                     on_progress=lambda f: ctx.progress(f, "轉檔中"), check=ctx.check, on_note=on_note)
    if not db.get_media(ctx.media_id):
        # 轉檔途中影片被刪掉了，轉好的檔案沒有人會用
        safepath.safe_unlink(dst, PROXY_DIR)
        raise Cancelled()
    db.update_media(ctx.media_id, proxy_path=str(dst))


# ---------- 任務：網址下載 ----------

def _drop_download(ctx: JobContext, out_dir: Path):
    """下載途中影片被刪掉了：下載到一半或剛下載好的檔案、縮圖都清掉（影片還在的話留著，重試可以續傳）。"""
    if db.get_media(ctx.media_id):
        return
    safepath.safe_rmtree(out_dir, MEDIA_DIR)
    safepath.safe_unlink(THUMB_DIR / f"{ctx.media_id}.jpg", THUMB_DIR)


def deno_path() -> Path | None:
    """yt-dlp 解 YouTube 的 JavaScript 要用的 deno（打包計畫 P0-11）。yt-dlp[default,deno] 把 deno.exe 裝在
    Python 的 Scripts 資料夾：發佈版是 runtime\\venv\\Scripts（launcher 也記在 install-state.json）。
    找不到（例如作者的 conda env 沒裝）回 None，維持 yt-dlp 自己的預設。"""
    import sysconfig
    candidates = [config._json_path(config.INSTALL_STATE, "deno"),
                  Path(sysconfig.get_path("scripts")) / "deno.exe",
                  Path(sys.executable).parent / "deno.exe"]
    return next((p for p in candidates if p is not None and p.is_file()), None)


class _YtdlpLog:
    """yt-dlp 的警告和錯誤寫進 app.log（原本 quiet、no_warnings 整個藏起來，缺 JS 執行環境這類問題看不到）。"""

    def debug(self, msg):
        pass

    def info(self, msg):
        pass

    def warning(self, msg):
        log.warning("yt-dlp: %s", msg)

    def error(self, msg):
        log.warning("yt-dlp error: %s", msg)


def handle_download(ctx: JobContext):
    try:
        url = safepath.check_download_url(ctx.params.get("url"))
    except ValueError as e:
        raise RuntimeError(f"下載失敗：{e}")
    safepath.require_safe_name(ctx.media_id, "影片 id")
    import yt_dlp

    out_dir = MEDIA_DIR / ctx.media_id
    out_dir.mkdir(parents=True, exist_ok=True)
    cookie_copy = ctx.workdir / "cookies.txt"

    def hook(d):
        if ctx.cancel_event.is_set():
            raise Cancelled()
        if d.get("status") == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            got = d.get("downloaded_bytes") or 0
            speed = d.get("speed") or 0
            label = f"下載中 {speed / 1024 / 1024:.1f} MB/s" if speed else "下載中"
            ctx.progress(0.95 * got / total if total else 0, label)
        elif d.get("status") == "finished":
            ctx.progress(0.95, "合併影音")

    opts = {
        "outtmpl": str(out_dir / "%(title).80B [%(id)s].%(ext)s"),
        "format": "bv*[vcodec^=avc1][ext=mp4]+ba[ext=m4a]/bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/bv*+ba/b",
        "merge_output_format": "mp4",
        # 跟轉檔、抽音訊用同一個 ffmpeg（config.find_tool 找到的）
        "ffmpeg_location": str(Path(config.require_tool("ffmpeg")).parent),
        "progress_hooks": [hook],
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "logger": _YtdlpLog(),
        "windowsfilenames": True,
        # yt-dlp 預設把快取寫到使用者家目錄的 .cache，改放專案裡
        "cachedir": str(YTDLP_CACHE_DIR),
    }
    deno = deno_path()
    if deno:
        opts["js_runtimes"] = {"deno": {"path": str(deno)}}
    try:
        # 給 yt-dlp 的是 cookies.txt 的複本，它結束時回寫 cookies 不會改到使用者自己的檔案
        opts.update(settings.ytdlp_cookie_opts(cookie_copy))
    except BaseException:
        safepath.safe_rmtree(ctx.workdir, WORK_DIR)
        raise
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
    except Cancelled:
        _drop_download(ctx, out_dir)
        raise
    except Exception as e:  # noqa: BLE001
        if ctx.cancel_event.is_set():
            _drop_download(ctx, out_dir)
            raise Cancelled()
        msg = str(e)
        if "Sign in to confirm" in msg or "bot" in msg.lower() or "cookies" in msg.lower():
            raise RuntimeError(
                "YouTube 要求登入驗證。請到「設定 → 下載」選一個已登入 YouTube 的瀏覽器來共用 cookies，"
                "或匯出 cookies.txt 後指定路徑，然後重試這個任務。"
            )
        raise RuntimeError(f"下載失敗：{msg}")
    finally:
        # 工作資料夾裡只有 cookies 複本（續傳用的暫存檔在 out_dir），用完整個刪掉
        safepath.safe_rmtree(ctx.workdir, WORK_DIR)

    # 合併影音那一段 yt-dlp 不會呼叫 hook，這段時間裡的取消或刪除影片要在這裡補檢查
    if ctx.cancel_event.is_set() or not db.get_media(ctx.media_id):
        _drop_download(ctx, out_dir)
        raise Cancelled()
    downloads = info.get("requested_downloads") or []
    path = Path(downloads[0]["filepath"]) if downloads else None
    if not path or not path.exists():
        files = sorted(out_dir.glob("*.mp4"), key=lambda f: f.stat().st_mtime, reverse=True)
        path = files[0] if files else None
    if not path:
        raise RuntimeError("下載完成但找不到檔案")
    if not safepath.inside(path, out_dir):
        raise RuntimeError(f"下載的檔案不在預期的資料夾裡：{path}")
    ctx.progress(0.97, "讀取影片資訊")
    info_probe = media.probe(path)
    db.update_media(ctx.media_id, path=str(path), title=info.get("title") or path.stem, **info_probe)
    thumb = THUMB_DIR / f"{ctx.media_id}.jpg"
    if media.make_thumbnail(path, thumb, info_probe["duration"]):
        db.update_media(ctx.media_id, has_thumb=1)
    if not db.get_media(ctx.media_id):
        _drop_download(ctx, out_dir)
        raise Cancelled()


# ---------- 任務：下載模型 ----------

# 自動重試：錯誤代碼 → 每次重試前等幾秒（清單長度就是最多重試幾次，打包計畫 6.5.6）。
# rate_limit 優先等伺服器 Retry-After 說的秒數。其他代碼（空間不足、找不到、要登入……）重試也沒用，直接失敗
DOWNLOAD_RETRY_WAITS = {
    "network": (10, 30, 90), "timeout": (10, 30, 90), "github": (10, 30, 90),
    "rate_limit": (60, 120, 300), "corrupt": (0,), "file_locked": (10,),
}
# 執行中的模型下載的詳細進度（GET /api/models/downloads 用）：job id → {"done", "total", "speed", ...}
download_progress: dict[str, dict] = {}


def model_job_variant(job: dict) -> str | None:
    """模型下載任務下載的是哪個版本；舊任務沒記版本時是預設版本，沒有版本的模型回 None。"""
    p = job.get("params") or {}
    entry = MODEL_CATALOG.get(p.get("model"))
    if not entry or not entry.get("variants"):
        return None
    v = p.get("variant")
    return v if v in entry["variants"] else entry["default_variant"]


def model_job_label(entry: dict, variant: str | None) -> str:
    return f"{entry['label']} {variant}" if variant else entry["label"]


def model_download_cmd(mid: str, variant: str | None = None, work: Path | None = None) -> list[str]:
    cmd = [sys.executable, "-s", "-m", "app.model_download", mid]
    if variant:
        cmd += ["--variant", variant]
    cmd += ["--parent-pid", str(os.getpid())]
    if work:
        cmd += ["--work", str(work)]
    return cmd


def model_download_env() -> dict:
    """下載子程序的環境變數：關掉 Xet 走一般 HTTP（可以續傳，D5 選 A），逾時 60 秒。"""
    env = {**os.environ, "PYTHONNOUSERSITE": "1", "PYTHONIOENCODING": "utf-8", "HF_HUB_OFFLINE": "0",
           "HF_HUB_DISABLE_XET": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
           "HF_HUB_DOWNLOAD_TIMEOUT": os.environ.get("HF_HUB_DOWNLOAD_TIMEOUT") or "60"}
    env.pop("HF_XET_HIGH_PERFORMANCE", None)
    return env


def _download_stage(info: dict) -> str:
    from .model_download import fmt_size
    done, total = fmt_size(info.get("done") or 0), fmt_size(info.get("total") or 0)
    if info.get("checking"):
        return f"檢查已下載的部分 {done} / {total}"
    speed = info.get("speed") or 0
    return f"下載中 {done} / {total}" + (f" · {speed / 1024 / 1024:.1f} MB/s" if speed > 0 else "")


def _run_model_download(ctx: JobContext, mid: str, variant: str | None):
    """跑一次下載子程序（app/model_download.py）。失敗丟 model_download.DownloadError；取消、暫停丟 Cancelled、Paused。"""
    from .model_download import DownloadError
    info = download_progress.setdefault(ctx.id, {"done": 0, "total": 0, "speed": 0})
    info.pop("retry", None)
    err_path = ctx.workdir / "model_download.err.log"
    failure = None
    with open(err_path, "w", encoding="utf-8", errors="replace") as err:
        ctx.child = subprocess.Popen(model_download_cmd(mid, variant, ctx.workdir), cwd=str(ROOT), env=model_download_env(),
                                     stdout=subprocess.PIPE, stderr=err, text=True, encoding="utf-8", errors="replace",
                                     creationflags=subprocess.CREATE_NO_WINDOW)
        try:
            ctx.check()        # 子程序剛開起來時就按了取消、暫停：cancel()、pause() 看不到這個子程序，在這裡補結束
            for line in ctx.child.stdout:
                line = line.strip()
                if not line.startswith("{"):
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if msg.get("error"):
                    failure = msg
                    continue
                info.update(done=int(msg.get("done") or 0), total=int(msg.get("total") or 0),
                            speed=float(msg.get("speed") or 0), checking=bool(msg.get("checking")))
                total = info["total"]
                ctx.progress(min(0.99, info["done"] / total) if total else 0, _download_stage(info))
            code = ctx.child.wait()
        except BaseException:
            if ctx.child.poll() is None:
                ctx.child.kill()
            ctx.child.wait()
            raise
    ctx.child = None
    ctx.check()
    if code == 0:
        return
    if failure:
        raise DownloadError(str(failure["error"]), str(failure.get("message") or "") or None, failure.get("retry_after"))
    lines = err_path.read_text(encoding="utf-8", errors="replace").strip().splitlines()
    last = next((l for l in reversed(lines) if l.strip()), f"exit {code}")
    if last.startswith("ERROR "):
        parts = last.split(" ", 2)
        raise DownloadError(parts[1], parts[2] if len(parts) > 2 else None)
    raise DownloadError("unknown", detail=last[:300])


def _wait_before_retry(ctx: JobContext, seconds: int, attempt: int, total: int, err):
    """自動重試前倒數，期間可以取消、暫停。"""
    reason = str(err).split("。", 1)[0]
    info = download_progress.setdefault(ctx.id, {"done": 0, "total": 0, "speed": 0})
    end = time.time() + seconds
    frac = min(0.99, info["done"] / info["total"]) if info.get("total") else 0
    while True:
        ctx.check()
        left = int(end - time.time() + 0.999)
        if left <= 0:
            return
        info.update(speed=0, retry={"code": err.code, "attempt": attempt, "max_attempts": total, "retry_in_s": left})
        ctx.progress(frac, f"{reason}。{left} 秒後自動重試（第 {attempt} 次，共 {total} 次）")
        time.sleep(0.5)


def handle_model(ctx: JobContext):
    """下載模型（有版本的模型下載指定的版本）。在子程序裡下載（app/model_download.py）：取消、暫停時直接結束子程序，
    下載就停了，.part 暫存檔保留，下次接著下載。網路斷線這類錯誤自動重試（DOWNLOAD_RETRY_WAITS），
    重試完還是失敗時，任務的錯誤訊息後面帶錯誤代碼，result 也記下 error_code。"""
    from . import models
    from .model_download import DownloadError

    mid = ctx.params["model"]
    entry = MODEL_CATALOG[mid]
    variant = config.check_variant(entry, ctx.params.get("variant"))   # 舊任務沒記版本：預設版本
    label = model_job_label(entry, variant)
    ctx.progress(0, f"下載 {label}")
    ctx.workdir.mkdir(parents=True, exist_ok=True)
    attempt = 0
    try:
        while True:
            try:
                # 排隊的時候空間夠，輪到時不一定還夠
                models.check_disk(mid, variant=variant, include_queue=False)
                _run_model_download(ctx, mid, variant)
                break
            except models.ModelError as e:
                # check_disk 只會因為空間不夠擋下（模型、版本在前面已經確認過）
                err = DownloadError("disk_full" if e.status == 400 else "unknown", str(e))
            except DownloadError as e:
                err = e
            waits = DOWNLOAD_RETRY_WAITS.get(err.code, ())
            if attempt < len(waits):
                wait = err.retry_after if (err.code == "rate_limit" and err.retry_after) else waits[attempt]
                attempt += 1
                log.warning("model download %s failed (%s), retry %d/%d in %ss", label, err.code, attempt, len(waits), wait)
                _wait_before_retry(ctx, int(wait), attempt, len(waits), err)
                continue
            info = download_progress.get(ctx.id) or {}
            db.update_job(ctx.id, result={"error_code": err.code, "done_bytes": info.get("done"),
                                          "total_bytes": info.get("total")})
            raise RuntimeError(f"{err}（錯誤代碼 {err.code}）")
    finally:
        # 工作資料夾裡只有子程序的錯誤紀錄，原因已經寫進任務了
        safepath.safe_rmtree(ctx.workdir, WORK_DIR)
        download_progress.pop(ctx.id, None)
    if not (model_installed(entry, variant) if variant else model_installed(entry)):
        raise RuntimeError("下載結束但找不到模型檔，請再試一次")
    if entry.get("role") == "furigana":
        # 假名引擎只在啟動時初始化；啟動時模型不在的話，現在補初始化，不用重開程式
        from . import furigana
        try:
            furigana.reload_if_failed()
        except Exception:  # noqa: BLE001
            log.warning("furigana reload failed", exc_info=True)
    ctx.result = {"model": mid, "variant": variant} if variant else {"model": mid}


# ---------- 任務：建查字字典 ----------

def handle_dict(ctx: JobContext):
    """下載來源、建字典、換上新字典（app/dict_build.py；子程序建、這裡換檔）。"""
    from . import dict_build
    dict_build.handle_dict(ctx)


HANDLERS = {
    "transcribe": handle_transcribe,
    "translate": handle_translate,
    "proxy": handle_proxy,
    "download": handle_download,
    "model": handle_model,
    "titles": handle_titles,
    "glosses": handle_glosses,
    "health": handle_health,
    "dict": handle_dict,
}
