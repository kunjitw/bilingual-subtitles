"""常駐語音工作程序：辨識、對齊、時間軸檢查的模型放在同一個程序裡，跨步驟、跨影片沿用。

由 app/speech.py 叫起來、管理，一個伺服器最多一個。跟伺服器分開程序：當掉不會拖垮伺服器，
結束程序就能連 CUDA context 一起把顯存全部還回去，取消的最後手段也可以直接結束它。

溝通用 stdin / stdout，一行一個 JSON：
  請求（stdin）
    {"id": 1, "cmd": "run", "argv": [...], "keep": ["qwen", "aligner"], "limit_mb": 12000}
        argv 跟 app/asr_child 的命令列一樣（階段名稱開頭，批次大小在裡面）。
        keep 以外、已經載入的模型先卸載（這個階段要用的模型一定留著）。
        limit_mb：torch 最多能用的顯存，超過就變成接得住的 OOM，不讓 Windows 偷偷搬到系統記憶體
    {"id": 2, "cmd": "unload", "models": ["anime"]}      models 省略或 null 就全部卸載
    {"id": 3, "cmd": "status"}
    {"cmd": "cancel", "id": 1}                           執行中的請求在下一批開始前停下來
    {"cmd": "quit"}
  回應（stdout）
    {"ready": true, "pid": 123, "total_mb": 16303}       啟動完成，CUDA 可以用（失敗時 ready false 加 tail）
    {"id": 1, "p": 0.5, "stage": "語音辨識"}             進度，跟 asr_child 一樣
    {"id": 1, "ok": true, "bs": 6, ...狀態}              完成
    {"id": 1, "ok": false, "error": "oom|cuda|canceled|other", "message": "...", "tail": "...", ...狀態}
  狀態：loaded {模型: MiB}、reserved_mb、allocated_mb、peak_reserved_mb、evicted、oom_retries、load_s、run_s

顯存爆掉（OOM）時：先卸載這個階段用不到的模型，用同樣的批次再試（結果不變）；還是不夠才把批次減半
（結果可能有些微差異，會記 log）；一段都放不下才回 oom。檢查點每一批寫一次，重試從還沒做的地方接著做。
CUDA 本身壞掉（不是 OOM 的 CUDA 錯誤）時回報後結束程序，下次需要時由上層重新開。
載入過的模型檔案被換掉或刪掉（大小、修改時間不一樣）時不沿用，重新載入。
stdin 關掉（伺服器結束）或 --parent-pid 的程序結束時馬上結束。

用法：python -s -m app.speech_worker --parent-pid PID --log <data\\work\\speech_worker.log>
  --log 是上層把 stderr 導過去的檔案，命令列帶著它，下次啟動時才認得出殘留的程序（jobs.find_orphans）。
環境變數 VS_SPEECH_BACKEND=模組名稱：換成假的模型和顯存（測試用，不需要顯卡，見 tests/fake_speech_backend.py）。
"""
import argparse
import gc
import importlib
import json
import os
import queue
import sys
import threading
import time
import traceback
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
# 沒有顯存碎片：卸載模型時真的還給驅動，檢查時間軸的峰值也從 9 GB 降到 6.6 GB（實測輸出逐字相同）
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from . import asr_child  # noqa: E402
from .config import ALIGNER, ASR_ENGINES  # noqa: E402

MB = 1024 * 1024


class Canceled(Exception):
    pass


class OutOfMemory(Exception):
    pass


# ---------- 輸出 ----------

class Channel:
    """回應寫到原本的 stdout；套件自己 print 的東西（包括 C 程式直接寫 fd 1 的）改到 stderr（log 檔），不會混進協定。"""

    def __init__(self):
        self._lock = threading.Lock()
        try:
            fd = os.dup(sys.stdout.fileno())
            os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
            self._out = open(fd, "w", encoding="utf-8", errors="replace", buffering=1)
        except (OSError, ValueError, AttributeError):
            self._out = sys.stdout
        sys.stdout = sys.stderr

    def send(self, **kw):
        line = json.dumps(kw, ensure_ascii=False)
        with self._lock:
            self._out.write(line + "\n")
            self._out.flush()


def log(msg: str):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), msg, file=sys.stderr, flush=True)


def watch_parent(pid: int | None):
    """伺服器（pid）結束時這個程序也馬上結束，不會留在背景佔著顯存（跟 app/model_download.py 一樣的做法）。"""
    if not pid or os.name != "nt":
        return
    import ctypes
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.restype = ctypes.c_void_p
    kernel32.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
    kernel32.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    handle = kernel32.OpenProcess(0x00100000, 0, int(pid))       # SYNCHRONIZE
    if not handle:
        os._exit(3)

    def wait():
        kernel32.WaitForSingleObject(handle, 0xFFFFFFFF)
        os._exit(3)

    threading.Thread(target=wait, name="watch-parent", daemon=True).start()


# ---------- torch ----------

class TorchBackend:
    """真的顯卡。測試用的假版本要有一樣的方法（tests/fake_speech_backend.py）。"""

    def __init__(self):
        import torch
        self.torch = torch
        torch.zeros(1, device="cuda")                 # 先建好 CUDA context，用不了顯卡的話啟動時就知道
        self.total_mb = torch.cuda.get_device_properties(0).total_memory // MB
        # 先 import 好，第一次載入模型時不用等（跟上層做抽音訊、人聲偵測同時進行）
        import qwen_asr  # noqa: F401

    def load_qwen(self, bs: int, max_new_tokens: int):
        return asr_child.load_qwen_asr(bs, max_new_tokens)

    def load_aligner(self):
        return asr_child.load_aligner()

    def load_whisper(self, engine_key: str):
        return asr_child.load_whisper(engine_key)

    def is_oom(self, e: BaseException) -> bool:
        return isinstance(e, self.torch.OutOfMemoryError) or "out of memory" in str(e).lower()

    def is_fatal(self, e: BaseException) -> bool:
        """CUDA 本身壞掉（非法存取、裝置不見了）：這個程序的 context 不能再用。"""
        accel = getattr(self.torch, "AcceleratorError", None)
        text = str(e)
        return (accel is not None and isinstance(e, accel)) or "CUDA error" in text or "CUBLAS_STATUS" in text

    def release(self):
        gc.collect()
        self.torch.cuda.empty_cache()

    def set_limit(self, limit_mb: int | None):
        fraction = 1.0 if not limit_mb else max(0.05, min(1.0, float(limit_mb) / self.total_mb))
        self.torch.cuda.set_per_process_memory_fraction(fraction, 0)

    def reset_peak(self):
        self.torch.cuda.reset_peak_memory_stats()

    def allocated_mb(self) -> int:
        return int(self.torch.cuda.memory_allocated() // MB)

    def memory(self) -> dict:
        c = self.torch.cuda
        return {"reserved_mb": int(c.memory_reserved() // MB), "allocated_mb": int(c.memory_allocated() // MB),
                "peak_reserved_mb": int(c.max_memory_reserved() // MB),
                "peak_allocated_mb": int(c.max_memory_allocated() // MB)}


# ---------- 模型快取 ----------

def model_path(key: str) -> Path:
    return Path(ALIGNER["path"] if key == "aligner" else ASR_ENGINES[key]["path"])


def signature(key: str):
    """模型檔案的大小和修改時間；檔案被刪掉、換掉時跟載入時不一樣，就不能沿用記憶體裡的舊模型。"""
    d = model_path(key)
    try:
        files = sorted(p for p in d.iterdir() if p.suffix in (".safetensors", ".bin") or p.name == "config.json")
        return tuple((p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in files) or None
    except OSError:
        return None


class Resident:
    """已經載入的模型（qwen、anime、whisper、aligner）。也是 asr_child 各階段取得模型的地方（models 參數）。"""

    def __init__(self, backend, channel: Channel, canceled: set):
        self.backend = backend
        self.channel = channel
        self.canceled = canceled
        self.cache: dict[str, dict] = {}
        self.req_id = None
        self.load_s = 0.0

    # asr_child 的 models 介面
    def emit(self, **kw):
        self.channel.send(id=self.req_id, **kw)

    def check(self):
        if self.req_id in self.canceled:
            raise Canceled()

    def qwen(self, bs: int, max_new_tokens: int):
        model = self._get("qwen", lambda: self.backend.load_qwen(bs, max_new_tokens))
        # 辨識（2048）和檢查（256）共用同一個模型物件，每次用之前設好（qwen_asr 每次呼叫時才讀這兩個值）
        model.max_inference_batch_size = bs
        model.max_new_tokens = max_new_tokens
        return model

    def aligner(self):
        return self._get("aligner", self.backend.load_aligner)

    def whisper(self, engine_key: str):
        return self._get(engine_key, lambda: self.backend.load_whisper(engine_key))

    # 載入、卸載
    def _get(self, key: str, loader):
        entry = self.cache.get(key)
        sig = signature(key)
        if entry is not None and entry["sig"] != sig:
            log(f"model files of {key} changed, reloading")
            self.unload(key)
            entry = None
        if entry is None:
            before = self.backend.allocated_mb()
            t = time.time()
            obj = loader()
            took = time.time() - t
            self.load_s += took
            entry = {"obj": obj, "sig": sig, "mb": max(0, self.backend.allocated_mb() - before)}
            self.cache[key] = entry
            log(f"loaded {key} in {took:.1f}s ({entry['mb']} MB)")
            self.tell_loaded()
        return entry["obj"]

    def unload(self, key: str):
        entry = self.cache.pop(key, None)
        if entry is None:
            return
        del entry
        self.backend.release()
        log(f"unloaded {key}")
        self.tell_loaded()

    def tell_loaded(self):
        """請求跑到一半載入、卸載模型時馬上告訴上層（顯卡狀態顯示、「釋放顯卡」的說明用），不用等請求做完。"""
        if self.req_id is not None:
            self.channel.send(id=self.req_id, loaded=self.loaded())

    def loaded(self) -> dict:
        return {k: v["mb"] for k, v in self.cache.items()}


# ---------- 請求 ----------

def _tail(exc: BaseException) -> str:
    lines = traceback.format_exception(type(exc), exc, exc.__traceback__)
    return "".join(lines)[-4000:]


def run_stage(res: Resident, msg: dict) -> dict:
    """跑一個階段；OOM 時先卸載用不到的模型再試，再不行才把批次減半。回傳要附在回應裡的欄位。"""
    backend = res.backend
    argv = [str(a) for a in msg["argv"]]
    need = asr_child.stage_model(argv)
    keep = set(msg.get("keep") or ()) | {need}
    for key in [k for k in res.cache if k not in keep]:
        res.unload(key)
    backend.set_limit(msg.get("limit_mb"))
    backend.release()
    backend.reset_peak()
    bs = asr_child.stage_batch(argv)
    evicted, retries = [], 0
    started = time.time()
    res.load_s = 0.0
    while True:
        error = None
        try:
            asr_child.dispatch(asr_child.with_batch(argv, bs), models=res)
            break
        except Canceled:
            raise
        except Exception as e:  # noqa: BLE001
            if not backend.is_oom(e):
                raise
            error = str(e).splitlines()[0][:300] if str(e) else "out of memory"
        # 出了 except 區塊才釋放：例外（和它的 traceback）還抓著推論途中的張量
        retries += 1
        backend.release()
        others = [k for k in res.cache if k != need]
        if others:
            for key in others:
                res.unload(key)
            evicted += others
            log(f"OOM in {argv[0]} bs={bs}: unloaded {others} and retrying with the same batch ({error})")
            continue
        if bs > 1:
            log(f"OOM in {argv[0]} bs={bs}: retrying with bs={max(1, bs // 2)} ({error})")
            bs = max(1, bs // 2)
            continue
        raise OutOfMemory(error)
    backend.release()
    out = {"bs": bs, "evicted": evicted, "oom_retries": retries, "load_s": round(res.load_s, 2),
           "run_s": round(time.time() - started, 2)}
    return out


def state(res: Resident) -> dict:
    return {"loaded": res.loaded(), **res.backend.memory()}


def serve(backend, channel: Channel, requests: "queue.Queue", canceled: set):
    res = Resident(backend, channel, canceled)
    while True:
        msg = requests.get()
        rid = msg.get("id")
        cmd = msg.get("cmd")
        res.req_id = rid
        try:
            if cmd == "run":
                extra = run_stage(res, msg)
                mem = state(res)
                log(f"run {msg['argv'][0]} bs={extra['bs']} keep={msg.get('keep')} loaded={mem['loaded']} "
                    f"load={extra['load_s']}s total={extra['run_s']}s peak_reserved={mem['peak_reserved_mb']}MB "
                    f"oom_retries={extra['oom_retries']} evicted={extra['evicted']}")
                channel.send(id=rid, ok=True, **extra, **mem)
            elif cmd == "unload":
                names = msg.get("models")
                for key in list(res.cache) if names is None else names:
                    res.unload(key)
                backend.release()
                channel.send(id=rid, ok=True, **state(res))
            elif cmd == "status":
                channel.send(id=rid, ok=True, **state(res))
            else:
                channel.send(id=rid, ok=False, error="other", message=f"unknown command {cmd}", **state(res))
        except Canceled:
            backend.release()
            channel.send(id=rid, ok=False, error="canceled", message="canceled", **state(res))
        except OutOfMemory as e:
            backend.release()
            channel.send(id=rid, ok=False, error="oom", message=str(e), tail=f"torch.OutOfMemoryError: {e}",
                         **state(res))
        except BaseException as e:  # noqa: BLE001
            fatal = not isinstance(e, (Exception, SystemExit)) or backend.is_fatal(e)
            tail = _tail(e)
            log(("fatal " if fatal else "") + f"error in {cmd}:\n{tail}")
            try:
                backend.release()
                mem = state(res)
            except BaseException:  # noqa: BLE001
                mem = {"loaded": {}}
                fatal = True
            text = str(e).strip().splitlines()[-1] if str(e).strip() else type(e).__name__
            channel.send(id=rid, ok=False, error="cuda" if fatal else "other", message=text[:500], tail=tail, **mem)
            if fatal:
                os._exit(2)
        finally:
            canceled.discard(rid)
            res.req_id = None


def protocol_input():
    """請求從原本的 stdin（上層接的 pipe）讀；fd 0（STD_INPUT_HANDLE）改成 NUL。
    Windows 上一條執行緒卡在讀 pipe 時，其他執行緒對同一個 pipe 的任何查詢（例如載入 torch、CUDA 的 DLL 時
    C runtime 初始化會對標準輸入呼叫 GetFileType）都會一起卡住，程序就永遠開不起來。"""
    try:
        fd = os.dup(sys.stdin.fileno())
        nul = os.open(os.devnull, os.O_RDONLY)
        os.dup2(nul, sys.stdin.fileno())
        os.close(nul)
        return open(fd, "r", encoding="utf-8", errors="replace")
    except (OSError, ValueError, AttributeError):
        return sys.stdin


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent-pid", type=int)
    ap.add_argument("--log")
    args = ap.parse_args(argv)
    inbox = protocol_input()
    channel = Channel()
    watch_parent(args.parent_pid)
    requests: queue.Queue = queue.Queue()
    canceled: set = set()

    def read_stdin():
        for line in inbox:
            try:
                msg = json.loads(line)
            except ValueError:
                continue
            if not isinstance(msg, dict):
                continue
            if msg.get("cmd") == "cancel":
                canceled.add(msg.get("id"))
            elif msg.get("cmd") == "quit":
                os._exit(0)
            else:
                requests.put(msg)
        os._exit(0)          # stdin 關掉：伺服器已經不在了

    threading.Thread(target=read_stdin, name="stdin", daemon=True).start()
    try:
        name = os.environ.get("VS_SPEECH_BACKEND")
        backend = importlib.import_module(name).Backend() if name else TorchBackend()
    except BaseException as e:  # noqa: BLE001
        tail = _tail(e)
        log(f"startup failed:\n{tail}")
        channel.send(ready=False, error="cuda", message=str(e)[:500], tail=tail)
        os._exit(1)
    log(f"ready pid={os.getpid()} total={backend.total_mb}MB alloc_conf={os.environ.get('PYTORCH_CUDA_ALLOC_CONF')}")
    channel.send(ready=True, pid=os.getpid(), total_mb=backend.total_mb)
    serve(backend, channel, requests, canceled)


if __name__ == "__main__":
    main()
