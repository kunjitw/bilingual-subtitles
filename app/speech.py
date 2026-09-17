"""常駐語音模型（伺服器這一邊）：什麼時候同時載入、什麼時候輪流、批次多大、什麼時候釋放。

模型本身在常駐工作程序裡（app/speech_worker.py），這裡負責：
  規劃  任務開始用語音模型之前，決定每個階段的批次大小（plan_for_job，整個任務不變）；
        每個請求送出前，照當下的顯存決定哪些已經載入的模型留著（keep_for）
  程序  開、關工作程序，送請求、收進度；取消、卡住、當掉的處理（Worker）
  釋放  顯卡沒事做滿 config.GPU_IDLE_RELEASE_S 秒（jobs.idle_tick，跟翻譯模型共用同一個計時）、
        翻譯或人聲分離需要顯存（make_room）、設定頁刪模型（release_model）、「釋放顯卡」按鈕（jobs.release_gpu，kill）

規則（數字在 config.SPEECH_VRAM、config.SPEECH_WORKER）：
  預算 = 顯卡剩餘 + 常駐程序自己佔的 − 1 GB 餘裕
  一個階段要的量 = CUDA context + 留著的模型常駐量 + max(這個階段在這個批次的推論量, 正在載入的模型多用的量)
  批次：每個階段從偏好清單（例如辨識 6、3、2、1）挑第一個「只放這個階段的模型」放得下的。
        如果所有模型一起放時批次要縮小，才用實測速度估計「同時放、批次縮小」和「輪流載入」哪個快，挑快的
  留著：這個階段要用的模型一定留著；其他已經載入的，照「這個任務接下來先用到的」優先、再來是別的任務留下的，
        放得下就留，放不下就卸載。所以卡夠大就全部同時載入、跨影片沿用；不夠就只換需要換的
  跑到一半爆顯存：工作程序先卸載用不到的模型、同樣的批次再試，再不行才把批次減半；之後這個任務只留當下要用的模型
"""
import json
import logging
import os
import queue
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import config, gpu
from .config import ALIGNER, ASR_ENGINES, MODEL_CATALOG, ROOT, WORK_DIR

log = logging.getLogger("speech")

MODULE = "app.speech_worker"
LOG_NAME = "speech_worker.log"
LOG_OLD_NAME = "speech_worker.old.log"
LOG_MAX_BYTES = 2 * 1024 * 1024
POLL_S = 0.25

# 顯示用的簡短名稱
SHORT_LABELS = {"qwen": "Qwen3-ASR", "anime": "anime-whisper", "whisper": "Whisper", "aligner": "對齊模型"}


_disabled: str | None = None     # 這次開機常駐程序開不起來的原因（disable）；None 代表可以用


def enabled() -> bool:
    """VS_SPEECH_WORKER=0，或這次開機常駐程序開不起來（disable）：改回每一步開一個子程序
    （app/asr_child.py 的命令列），用完就結束。"""
    return os.environ.get("VS_SPEECH_WORKER", "1").strip() != "0" and _disabled is None


def disable(reason: str):
    """常駐程序在這台電腦上開不起來（不是顯示卡、顯存的問題）：這次開機之後都用單次子程序，重開伺服器再試。"""
    global _disabled
    _disabled = reason or "unknown"
    log.warning("speech worker disabled until restart, using one-shot child processes: %s", _disabled)


def label_of(key: str) -> str:
    return ALIGNER["label"] if key == "aligner" else ASR_ENGINES.get(key, {}).get("label", key)


def catalog_key(mid: str) -> str | None:
    """型錄 id → 常駐程序裡的模型名稱（qwen、anime、whisper、aligner）；不是語音模型回 None。"""
    entry = MODEL_CATALOG.get(mid) or {}
    if entry.get("role") == "aligner":
        return "aligner"
    if entry.get("role") == "asr":
        return entry.get("engine")
    return None


# ---------- 規劃 ----------

def stage_key(stage: str, model: str) -> str:
    return f"asr:{model}" if stage == "asr" else stage


def job_stages(engine: str | None, health_on: bool) -> list[tuple[str, str]]:
    """任務會依序跑的階段 [(階段, 模型)]。engine 是 None：單獨檢查時間軸的任務。"""
    check = [("verify", "qwen"), ("realign", "aligner"), ("verify", "qwen")]
    if engine is None:
        return check
    return [("asr", engine), ("align", "aligner")] + (check if health_on else [])


def need_mb(models, skey: str, bs: int, loading: str | None = None, vram=None, ctx_mb=None) -> int:
    """models 都載入時，跑 skey 這個階段（批次 bs）要的顯存；loading 是這時候才要載入的模型。"""
    v = vram or config.SPEECH_VRAM
    ctx = config.SPEECH_WORKER["context_mb"] if ctx_mb is None else ctx_mb
    extra = v["stage_mb"][skey][bs]
    if loading:
        extra = max(extra, v["load_extra_mb"][loading])
    return ctx + sum(v["resident_mb"][m] for m in set(models)) + extra


class NotEnough(RuntimeError):
    def __init__(self, model: str, need: int):
        super().__init__(f"{model} needs {need} MB")
        self.model = model
        self.need = need


def keep_for(need: str, skey: str, bs: int, loaded, later, budget_mb: int, vram=None) -> set:
    """這個階段送出前，哪些模型留著：要用的一定留；其他已經載入的照優先順序（這個任務接下來先用到的、
    再來是別的任務留下的）放得下就留。回傳的集合只含 need 和已經載入的模型。"""
    loaded = set(loaded)
    keep = {need}
    loading = None if need in loaded else need
    order = [m for m in dict.fromkeys(later) if m in loaded and m != need]
    order += sorted(m for m in loaded if m not in order and m != need)
    for m in order:
        if need_mb(keep | {m}, skey, bs, loading, vram) <= budget_mb:
            keep.add(m)
    return keep


def _simulate(stages, bs: dict, budget_mb: int, loaded, vram) -> tuple[float, int, str]:
    """照 keep_for 的規則模擬整個任務：回傳 (換模型花的秒數, 載入次數, 做法)。"""
    loaded = set(loaded)
    everything = {m for _, m in stages}
    secs, loads, dropped, most = 0.0, 0, False, 0
    for i, (stage, model) in enumerate(stages):
        skey = stage_key(stage, model)
        later = [m for _, m in stages[i + 1:]]
        keep = keep_for(model, skey, bs[skey], loaded, later, budget_mb, vram)
        if model not in loaded:
            secs += vram["load_s"][model]
            loads += 1
        dropped = dropped or bool((loaded - keep) & set(later))
        loaded = keep
        most = max(most, len(loaded & everything))
    if len(everything) <= 1 or not dropped:
        mode = "同時載入"
    elif most <= 1:
        mode = "輪流載入"
    else:
        mode = "部分同時載入"
    return secs, loads, mode


@dataclass
class Plan:
    stages: list
    bs: dict
    mode: str
    budget_mb: int
    loads: int = 0
    est_s: float = 0.0
    release_llm: bool = False
    health_skipped: str | None = None   # 顯存連檢查時間軸的 Qwen3-ASR 都放不下：這次不檢查（說明）
    tight: bool = False          # 跑到一半爆過顯存：之後只留當下要用的模型
    pos: int = 0
    notes: list = field(default_factory=list)

    def batch(self, stage: str, engine: str | None = None) -> int:
        model = engine if stage == "asr" else ("qwen" if stage == "verify" else "aligner")
        return self.bs[stage_key(stage, model)]

    def later(self, stage: str, model: str) -> list[str]:
        """送出這個階段時，任務接下來還會用到的模型（照順序）；順便記下跑到哪裡。"""
        for i in range(self.pos, len(self.stages)):
            if self.stages[i] == (stage, model):
                self.pos = i + 1
                break
        return [m for _, m in self.stages[self.pos:]]


def plan_job(engine: str | None, health_on: bool, speech_s: float, budget_mb: int, loaded=(), vram=None) -> Plan:
    """決定每個階段的批次大小。連最小的批次、只放一個模型都放不下時丟 NotEnough。"""
    v = vram or config.SPEECH_VRAM
    stages = job_stages(engine, health_on)
    keys = list(dict.fromkeys((stage_key(s, m), m) for s, m in stages))
    everything = {m for _, m in stages}
    solo, together = {}, {}
    for skey, model in keys:
        prefer = v["prefer"][skey]
        solo[skey] = next((bs for bs in prefer if need_mb({model}, skey, bs, model, v) <= budget_mb), None)
        if solo[skey] is None:
            raise NotEnough(model, need_mb({model}, skey, prefer[-1], model, v))
        together[skey] = next((bs for bs in prefer if need_mb(everything, skey, bs, model, v) <= budget_mb), None)
    candidates = [solo]
    if all(together.values()) and together != solo:
        candidates.append(together)
    best = None
    for bs in candidates:
        infer = sum(v["secs_per_100s"][skey][bs[skey]] * speech_s / 100 for skey, _ in keys)
        swap_s, loads, mode = _simulate(stages, bs, budget_mb, loaded, v)
        plan = Plan(stages=stages, bs=dict(bs), mode=mode, budget_mb=budget_mb, loads=loads,
                    est_s=round(infer + swap_s, 1))
        if best is None or plan.est_s < best.est_s:
            best = plan
    return best


# ---------- 工作程序 ----------

class WorkerError(RuntimeError):
    """工作程序沒有正常回應（當掉、卡住、取消後停不下來被結束）。tail 是給 jobs.child_failure 的錯誤紀錄。"""

    def __init__(self, message: str, tail: str = ""):
        super().__init__(message)
        self.tail = tail or message


class WorkerCanceled(WorkerError):
    pass


class WorkerStartError(WorkerError):
    """工作程序開不起來（啟動時結束、回報失敗、太久沒有好），跟執行途中當掉分開：jobs.run_speech 會改用單次子程序。"""


_EOF = object()


def default_cmd(log_path: Path) -> list[str]:
    return [sys.executable, "-s", "-m", MODULE, "--parent-pid", str(os.getpid()), "--log", str(log_path)]


def default_env() -> dict:
    from . import jobs
    env = jobs.run_child_env()
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    return env


class Worker:
    """一個常駐工作程序。送請求的只有顯卡執行緒；設定頁刪模型、釋放顯存、閒置釋放會從別的執行緒來，
    全部經過 _lock（一次一個請求），等不到鎖的地方不死等。"""

    def __init__(self, cmd=None, env=None, log_path: Path | None = None, clock=time.monotonic, settings=None):
        self._cmd = cmd
        self._env = env
        self._log_path = log_path
        self._log_start = 0            # 現在這個程序的 log 從檔案的哪裡開始（log_tail）
        self.clock = clock
        self.settings = settings or config.SPEECH_WORKER
        self._lock = threading.Lock()
        self._write_lock = threading.Lock()
        self.proc: subprocess.Popen | None = None
        self._q: queue.Queue | None = None
        self.ready: dict | None = None
        self.loaded: dict[str, int] = {}
        self.reserved_mb = 0
        self.stale: set[str] = set()       # 模型檔案被刪掉、換掉，記憶體裡的不能再用（下一個請求先卸載）
        self.last_used = clock()
        self.starts = 0
        self._next_id = 0

    # ---- 狀態 ----
    @property
    def log_path(self) -> Path:
        return self._log_path or (WORK_DIR / LOG_NAME)

    def alive(self) -> bool:
        proc = self.proc            # 別的執行緒可能正好把它換掉
        return proc is not None and proc.poll() is None

    def usage_mb(self) -> int:
        """這個程序現在佔的顯存（torch reserved ＋ CUDA context）；還沒開好算 0。"""
        if not self.alive() or not self.ready:
            return 0
        return int(self.reserved_mb) + int(self.settings["context_mb"])

    def loaded_keys(self) -> set[str]:
        return set(self.loaded) - self.stale if self.alive() else set()

    def busy(self) -> bool:
        return self._lock.locked()

    def log_tail(self, limit: int = 6000) -> str:
        """現在這個程序寫的 log 最後一段（不含之前的程序留下的，免得把舊的錯誤當成這次的原因）。"""
        try:
            with open(self.log_path, "rb") as f:
                f.seek(0, os.SEEK_END)
                size = f.tell()
                start = self._log_start if self._log_start <= size else 0
                f.seek(max(start, size - limit))
                return f.read().decode("utf-8", errors="replace")
        except OSError:
            return ""

    def _open_log(self, path: Path):
        """log 接在後面寫，閒置釋放、按「釋放顯卡」、當掉重開之後，之前的紀錄（批次、爆顯存重試、錯誤）還查得到。
        超過 LOG_MAX_BYTES 時先把舊的換名成 LOG_OLD_NAME（只留一份），不會一直變大。"""
        try:
            if path.stat().st_size > LOG_MAX_BYTES:
                os.replace(path, path.with_name(LOG_OLD_NAME))
        except OSError:
            pass                      # 沒有舊檔，或舊檔還被別的程式開著：照樣接在後面寫
        err = open(path, "a", encoding="utf-8", errors="replace")
        gap = "\n" if os.fstat(err.fileno()).st_size else ""
        err.write(f"{gap}===== {time.strftime('%Y-%m-%d %H:%M:%S')} 開新的語音模型程序 =====\n")
        err.flush()
        self._log_start = os.fstat(err.fileno()).st_size
        return err

    # ---- 開、關 ----
    def _start(self):
        path = self.log_path
        path.parent.mkdir(parents=True, exist_ok=True)
        cmd = (self._cmd(path) if callable(self._cmd) else self._cmd) or default_cmd(path)
        env = (self._env() if callable(self._env) else self._env) or default_env()
        with self._open_log(path) as err:
            self.proc = subprocess.Popen(
                cmd, cwd=str(ROOT), env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err,
                text=True, encoding="utf-8", errors="replace", bufsize=1,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        q: queue.Queue = queue.Queue()
        self._q = q
        self.ready = None
        self.loaded, self.reserved_mb, self.stale = {}, 0, set()
        self.starts += 1
        self.last_used = self.clock()
        threading.Thread(target=self._read, args=(self.proc, q), name="speech-worker-stdout", daemon=True).start()
        log.info("speech worker started (pid %s)", self.proc.pid)

    @staticmethod
    def _read(proc, q):
        try:
            for line in proc.stdout:
                line = line.strip()
                if not line.startswith("{"):
                    continue
                try:
                    q.put(json.loads(line))
                except ValueError:
                    continue
        except (OSError, ValueError):
            pass
        q.put(_EOF)

    def _send(self, msg: dict):
        with self._write_lock:
            self.proc.stdin.write(json.dumps(msg) + "\n")
            self.proc.stdin.flush()

    def _forget(self):
        self.proc, self._q, self.ready = None, None, None
        self.loaded, self.reserved_mb, self.stale = {}, 0, set()

    def _kill(self, reason: str):
        proc = self.proc
        if proc is not None:
            log.warning("stopping speech worker (pid %s): %s", proc.pid, reason)
            if proc.poll() is None:
                proc.kill()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                pass
        self._forget()

    def stop(self, reason: str, timeout: float | None = None) -> bool:
        """結束工作程序，顯存全部還回去。等不到鎖（正在跑請求）回 False。"""
        if not self._lock.acquire(timeout=-1 if timeout is None else timeout):
            return False
        try:
            proc = self.proc
            if proc is None:
                return True
            if proc.poll() is None:
                log.info("releasing speech worker (pid %s): %s", proc.pid, reason)
                try:
                    self._send({"cmd": "quit"})
                    proc.wait(timeout=10)
                except (OSError, ValueError, subprocess.TimeoutExpired):
                    pass
            self._kill(reason)
            return True
        finally:
            self._lock.release()

    # ---- 請求（呼叫的地方要拿著 _lock） ----
    def _wait(self, want, on_progress=None, cancel_event=None, rid=None, stall_s=None):
        """讀回應直到 want(訊息) 為真。當掉丟 WorkerError；卡住（stall_s 秒沒有進度）就結束程序；取消後等不到就結束程序。"""
        stall_s = self.settings["stall_s"] if stall_s is None else stall_s
        last = self.clock()
        cancel_at = None
        q = self._q
        while True:
            try:
                item = q.get(timeout=POLL_S)
            except queue.Empty:
                item = None
            now = self.clock()
            if item is _EOF:
                proc = self.proc
                try:
                    code = proc.wait(timeout=5) if proc else None
                except subprocess.TimeoutExpired:
                    code = None
                message = f"語音模型的程序意外結束了（exit {code}）"
                # 最後一行放白話的原因（jobs.child_failure 拿最後一行當說明）；前面的 log 留著判斷是不是 CUDA 用不了
                tail = f"{self.log_tail()}\n{message}"
                self._forget()
                raise WorkerError(message, tail)
            if isinstance(item, dict):
                if want(item):
                    return item
                if rid is not None and item.get("id") == rid:
                    last = now
                    if isinstance(item.get("loaded"), dict):          # 跑到一半載入、卸載了模型
                        self.loaded = {k: int(v) for k, v in item["loaded"].items()}
                    if on_progress and "p" in item:
                        on_progress(item)
                    continue
            if cancel_event is not None and cancel_event.is_set():
                if rid is None:
                    raise WorkerCanceled("canceled")
                if cancel_at is None:
                    cancel_at = now
                    try:
                        self._send({"cmd": "cancel", "id": rid})
                    except (OSError, ValueError):
                        pass
                elif now - cancel_at > self.settings["cancel_wait_s"]:
                    self._kill("取消後沒有停下來")
                    raise WorkerCanceled("canceled")
            if now - last > stall_s:
                message = f"語音模型超過 {stall_s:g} 秒沒有進度，已經結束它"
                tail = f"{self.log_tail()}\n{message}"
                self._kill("太久沒有進度")
                raise WorkerError(message, tail)

    def _ensure_ready(self, cancel_event=None):
        if not self.alive():
            if self.proc is not None:
                self._forget()
            self._start()
        if self.ready:
            return
        try:
            msg = self._wait(lambda m: "ready" in m and "id" not in m, cancel_event=cancel_event,
                             stall_s=self.settings["start_timeout_s"])
        except WorkerCanceled:
            raise
        except WorkerError as e:
            raise WorkerStartError(f"語音模型的程序開不起來：{e}", e.tail) from None
        if not msg.get("ready"):
            tail = msg.get("tail") or msg.get("message") or self.log_tail()
            self._kill("啟動失敗")
            raise WorkerStartError("語音模型的程序啟動失敗", tail)
        self.ready = msg

    def _request(self, msg: dict, on_progress=None, cancel_event=None) -> dict:
        for attempt in (1, 2):
            self._ensure_ready(cancel_event)
            self._next_id += 1
            rid = self._next_id
            try:
                self._send({**msg, "id": rid})
                break
            except (OSError, ValueError):
                # 閒著的時候程序不見了：重開一次再送（請求還沒開始跑，重送是安全的）
                self._kill("送不出請求")
                if attempt == 2:
                    raise WorkerError("語音模型的程序沒有回應", self.log_tail())
        reply = self._wait(lambda m: m.get("id") == rid and "ok" in m, on_progress, cancel_event, rid)
        self.loaded = {k: int(v) for k, v in (reply.get("loaded") or {}).items()}
        self.reserved_mb = int(reply.get("reserved_mb") or 0)
        self.stale &= set(self.loaded)
        self.last_used = self.clock()
        return reply

    # ---- 對外 ----
    def run(self, argv: list[str], plan: Plan | None = None, on_progress=None, cancel_event=None,
            gpu_info=None) -> dict:
        """跑 asr_child 的一個階段。回傳工作程序的回應（ok、error、bs、loaded……）。"""
        from .asr_child import stage_batch, stage_model
        argv = [str(a) for a in argv]
        stage, model = argv[0], stage_model(argv)
        skey = stage_key(stage, model)
        bs = stage_batch(argv)
        with self._lock:
            self._ensure_ready(cancel_event)
            info = gpu.query(max_age=0) if gpu_info is None else gpu_info
            loaded = set(self.loaded) - self.stale
            later = plan.later(stage, model) if plan is not None else []
            limit = None
            if info:
                budget = int(info["free_mb"]) + self.usage_mb() - int(self.settings["margin_mb"])
                if plan is not None and plan.tight:
                    keep = {model}
                else:
                    keep = keep_for(model, skey, bs, loaded, later, budget)
                # torch 最多能用：現在佔的 ＋ 剩下的（留一點給驅動）
                limit = int(self.reserved_mb) + int(info["free_mb"]) - 150
            else:
                keep = {model}
            msg = {"cmd": "run", "argv": argv, "keep": sorted(keep), "limit_mb": limit}
            reply = self._request(msg, on_progress, cancel_event)
            if reply.get("ok"):
                self.stale.discard(model)
                if plan is not None:
                    if int(reply.get("bs") or bs) < bs:
                        plan.bs[skey] = int(reply["bs"])
                        plan.notes.append(f"{skey} 批次 {bs}→{reply['bs']}")
                    if reply.get("evicted"):
                        plan.tight = True
            return reply

    def unload(self, keys=None, timeout: float | None = None, reason: str = "") -> bool:
        """卸載模型（keys 是 None：全部），程序留著。等不到鎖回 False（keys 先記成不能再用）。"""
        if keys is not None:
            self.stale |= set(keys) & set(self.loaded)
        if not self._lock.acquire(timeout=-1 if timeout is None else timeout):
            return False
        try:
            if not self.alive() or not self.ready:
                return True
            wanted = [k for k in (self.loaded if keys is None else keys) if k in self.loaded]
            if not wanted:
                return True
            log.info("unloading %s from the speech worker: %s", wanted, reason)
            self._request({"cmd": "unload", "models": wanted})
            return True
        except WorkerError:
            return True
        finally:
            self._lock.release()

    def kill(self, reason: str, wait_s: float = 0.5) -> bool:
        """「釋放顯卡」按鈕：閒著就正常結束；正在跑請求也不等，直接結束程序（顯存馬上還回去）。
        跑到一半的請求會收到程序結束（WorkerError），任務那邊已經先標好暫停，會回到佇列、之後從檢查點接著做。
        回傳有沒有東西被結束。"""
        if self.proc is None:
            return False
        if self.stop(reason, timeout=wait_s):
            return True
        proc = self.proc
        if proc is not None and proc.poll() is None:
            log.warning("killing busy speech worker (pid %s): %s", proc.pid, reason)
            proc.kill()
        return True


WORKER = Worker()


# ---------- 給 jobs、models 用 ----------

def plan_for_job(engine: str | None, health_on: bool, speech_s: float, llm_mb: int = 0,
                 worker: Worker | None = None) -> Plan:
    """任務開始用語音模型之前呼叫：照現在的顯存決定批次大小，和翻譯模型要不要先關（release_llm）。
    llm_mb：開著的 llama-server 大約佔多少（0：沒開）。放不下時丟跟以前一樣的「顯存不足」說明。"""
    w = worker or WORKER
    info = gpu.require_supported()
    margin = int(w.settings["margin_mb"])
    free = int(info["free_mb"]) + w.usage_mb()
    loaded = w.loaded_keys()
    llm = int(llm_mb or 0)

    def attempt(budget: int) -> Plan:
        try:
            return plan_job(engine, health_on, speech_s, budget, loaded)
        except NotEnough as e:
            if engine is None or not health_on:
                raise
            # 轉字幕本身放得下、檢查時間軸的 Qwen3-ASR 放不下：字幕照做，這次不檢查（以前是檢查到一半失敗，結果一樣）
            p = plan_job(engine, False, speech_s, budget, loaded)
            p.health_skipped = f"顯存放不下檢查時間軸要用的 {label_of(e.model)}（需要約 {e.need / 1024:.1f} GB）"
            return p

    try:
        plan = attempt(free + llm - margin)
    except NotEnough as e:
        gpu.ensure_free(e.need + margin, label_of(e.model), info=dict(info, free_mb=free + llm))
        raise RuntimeError(f"顯存不足：{label_of(e.model)} 放不下")
    if llm:
        try:
            kept = attempt(free - margin)
        except NotEnough:
            kept = None
        if (kept is not None and kept.bs == plan.bs and kept.loads <= plan.loads
                and kept.health_skipped == plan.health_skipped):
            plan = kept                       # 翻譯模型留著也不影響批次、換模型次數
        else:
            plan.release_llm = True
    log.info("speech plan: %s bs=%s budget=%sMB loads=%s est=%ss release_llm=%s", plan.mode, plan.bs,
             plan.budget_mb, plan.loads, plan.est_s, plan.release_llm)
    return plan


def make_room(need_mb: int, reason: str, keep_process: bool = True, worker: Worker | None = None):
    """翻譯模型、人聲分離要載入前呼叫：剩餘顯存不夠 need_mb（再加上 margin_mb 的餘裕）時先釋放常駐的語音模型。
    餘裕跟規劃語音模型時一樣：常駐的模型把顯存佔到剛好夠時，桌面、瀏覽器一跳動就會被 Windows 搬到系統記憶體（變很慢）。
    keep_process：先只卸載模型（程序和 CUDA context 約 300 MB 留著，下一部影片不用重新 import），還是不夠才結束程序。"""
    w = worker or WORKER
    if w.proc is None:
        return
    want = int(need_mb) + int(w.settings["margin_mb"])
    info = gpu.query(max_age=0)
    if not info or int(info["free_mb"]) >= want:
        return
    if keep_process and w.loaded and w.alive():
        w.unload(None, reason=reason)
        info = gpu.query(max_age=0)
        if info and int(info["free_mb"]) >= want:
            return
    w.stop(reason)


def release_model(mid: str, reason: str, worker: Worker | None = None, timeout: float = 3.0):
    """設定頁刪除模型：常駐程序還載著它就卸載。正在跑請求等不到時，先記成不能再用，下一個請求一定先卸載它；
    正在跑的任務用到的模型 models.busy_reason 已經擋掉，所以這時候載著的只會是別的任務留下、這次沒在用的。"""
    w = worker or WORKER
    key = catalog_key(mid)
    if key is None or key not in w.loaded:
        return
    w.unload([key], timeout=timeout, reason=reason)


def loaded_view(worker: Worker | None = None) -> list[dict]:
    w = worker or WORKER
    return [{"key": k, "label": label_of(k), "short": SHORT_LABELS.get(k, k), "mb": w.loaded.get(k, 0)}
            for k in sorted(w.loaded_keys(), key=lambda k: list(SHORT_LABELS).index(k) if k in SHORT_LABELS else 99)]
