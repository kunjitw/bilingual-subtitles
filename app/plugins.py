"""本機外掛：讓某些網址改用自己寫的下載程式，不走 yt-dlp。

外掛放在專案根目錄的 local_plugins 資料夾（.gitignore 擋掉，不會上傳），一個外掛一個 .py 檔。
位置寫死在這裡（PLUGIN_DIR），不能從網址、設定檔或環境變數指定別的地方。沒有這個資料夾時一切照舊。
不載入的檔案：檔名以 _ 開頭的（可以放外掛共用的程式）、捷徑（symlink、junction）、實際位置不在資料夾裡的。
資料夾本身是捷徑也整個不載入。外掛檔案有改動時，下次用到會重新載入，不用重開伺服器。

外掛檔案要有：
  NAME = "顯示名稱"                       選填，預設是檔名
  def match(url: str) -> bool             這個網址要不要由它處理（網址已經通過 safepath.check_download_url）
  def download(job) -> dict               把影片下載到 job.out_dir，回傳 {"path": 影片檔, "title": 標題}
可以有：
  OPTIONS = {"language": "ja", "translate": False}
      這種網址固定用的轉字幕選項，會蓋掉網頁送來的。只認 OPTION_KEYS 裡的欄位，其他的忽略。

download 拿到的 job（PluginJob）：
  job.url          網址
  job.out_dir      影片要放的資料夾（data/media/<影片 id>），刪影片時整個刪掉；回傳的檔案一定要在這裡面
  job.work_dir     暫存資料夾，任務結束就刪
  job.progress(比例 0~1, 說明)
  job.check()      使用者取消了就丟例外（任務會變成已取消）
  job.run(cmd, on_line=None, cwd=None, env=None) -> int
                   跑一個子程序（不開視窗），每行輸出呼叫 on_line(字串)，回傳結束碼；
                   取消時連同它叫起來的程序一起結束，然後丟取消的例外
"""
import importlib.util
import logging
import os
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType

from . import config, safepath

log = logging.getLogger("plugins")

PLUGIN_DIR = config.ROOT / "local_plugins"
# 外掛可以固定的轉字幕選項和型別（language 另外檢查是不是支援的語言）
OPTION_KEYS = {"language": str, "translate": bool, "sensitive": bool}


@dataclass
class Plugin:
    name: str
    file: Path
    module: ModuleType
    options: dict = field(default_factory=dict)

    def match(self, url: str) -> bool:
        try:
            return bool(self.module.match(url))
        except Exception:  # noqa: BLE001
            log.warning("plugin %s: match() failed", self.name, exc_info=True)
            return False


_lock = threading.Lock()
_cache: dict = {"key": None, "plugins": []}


def _candidates() -> list[Path]:
    """可以載入的外掛檔案。資料夾不在、是捷徑，回空清單。"""
    folder = PLUGIN_DIR
    try:
        if not folder.is_dir() or safepath.is_link(folder):
            return []
        files = sorted(folder.glob("*.py"))
    except OSError:
        return []
    out = []
    for f in files:
        if f.name.startswith("_") or not f.is_file():
            continue
        if safepath.is_link(f) or not safepath.inside(f, folder):
            log.warning("plugin %s is a link or points outside %s, skipped", f.name, folder)
            continue
        out.append(f)
    return out


def _scan_key(files: list[Path]) -> tuple:
    key = []
    for f in files:
        try:
            st = f.stat()
        except OSError:
            continue
        key.append((f.name, st.st_mtime_ns, st.st_size))
    return (str(PLUGIN_DIR), tuple(key))


def _clean_options(name: str, raw) -> dict:
    if not isinstance(raw, dict):
        return {}
    out = {}
    for k, v in raw.items():
        want = OPTION_KEYS.get(k)
        if want is None or not isinstance(v, want):
            log.warning("plugin %s: ignored option %r=%r", name, k, v)
            continue
        if k == "language" and v not in config.LANGUAGES:
            log.warning("plugin %s: unsupported language %r", name, v)
            continue
        out[k] = v
    return out


def _load(f: Path) -> Plugin | None:
    mod_name = f"vs_local_plugin_{f.stem}"
    try:
        spec = importlib.util.spec_from_file_location(mod_name, f)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = module
        spec.loader.exec_module(module)
    except Exception:  # noqa: BLE001
        sys.modules.pop(mod_name, None)
        log.warning("plugin %s failed to load", f.name, exc_info=True)
        return None
    if not callable(getattr(module, "match", None)) or not callable(getattr(module, "download", None)):
        log.warning("plugin %s has no match() or download(), skipped", f.name)
        return None
    name = getattr(module, "NAME", None)
    name = name if isinstance(name, str) and name.strip() else f.stem
    return Plugin(name=name.strip(), file=f, module=module, options=_clean_options(name, getattr(module, "OPTIONS", None)))


def load() -> list[Plugin]:
    """目前的外掛清單。檔案沒變就用上次載入的。"""
    files = _candidates()
    key = _scan_key(files)
    with _lock:
        if _cache["key"] != key:
            plugins = [p for p in (_load(f) for f in files) if p]
            if plugins:
                log.info("local plugins: %s", ", ".join(f"{p.name} ({p.file.name})" for p in plugins))
            _cache.update(key=key, plugins=plugins)
        return list(_cache["plugins"])


def find(url: str) -> Plugin | None:
    """處理這個網址的外掛；網址不合格（safepath.check_download_url）或沒有外掛要處理回 None。"""
    try:
        url = safepath.check_download_url(url)
    except ValueError:
        return None
    return next((p for p in load() if p.match(url)), None)


# ---------- 給外掛用的任務介面 ----------

def _kill_tree(proc: subprocess.Popen):
    """結束子程序和它叫起來的程序（例如下載程式再叫的 ffmpeg）。"""
    if proc.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if proc.poll() is None:
        proc.kill()


class PluginJob:
    """外掛的 download(job) 拿到的東西。ctx 是 jobs.JobContext（check、progress、cancel_event）。"""

    def __init__(self, ctx, url: str, out_dir: Path, work_dir: Path):
        self._ctx = ctx
        self.url = url
        self.out_dir = Path(out_dir)
        self.work_dir = Path(work_dir)

    def progress(self, value: float, stage: str | None = None):
        self._ctx.progress(float(value), stage)

    def check(self):
        self._ctx.check()

    def run(self, cmd: list, on_line=None, cwd=None, env=None) -> int:
        self.check()
        proc = subprocess.Popen([str(c) for c in cmd], cwd=str(cwd) if cwd else None, env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                text=True, encoding="utf-8", errors="replace",
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        lines: list[str] = []
        done = threading.Event()

        def read():
            try:
                for line in proc.stdout:
                    lines.append(line.rstrip("\r\n"))
            finally:
                done.set()

        threading.Thread(target=read, daemon=True).start()
        try:
            while True:
                finished = done.wait(0.3)
                while lines:
                    line = lines.pop(0)
                    if on_line:
                        on_line(line)
                if self._ctx.cancel_event.is_set():
                    _kill_tree(proc)
                    self.check()
                if finished:
                    break
            return proc.wait()
        finally:
            _kill_tree(proc)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
            if proc.stdout:
                proc.stdout.close()
