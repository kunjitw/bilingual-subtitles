"""建立和檢查可攜執行環境。只用 Python 標準函式庫。

start.bat 在 precheck.py 之後，用 uv 裝好的 Python 執行：
    runtime\\python\\cpython-3.12.14-windows-x86_64-none\\python.exe -I -X utf8 launcher\\setup_runtime.py

已經裝好、資料夾沒搬、uv.lock 和 runtime-manifest.json 都沒變時，不到 1 秒就結束。
要做的事（每一步都可以重跑；中途關掉視窗，重開 start.bat 會接著做）：
  1. 資料夾搬過家：改 runtime\\venv\\pyvenv.cfg 裡的路徑，重建 uv 的 Python junction
  2. runtime\\venv：用標準函式庫的 venv 建（搬家時只要改 pyvenv.cfg），再用 uv sync 裝 uv.lock 的套件
  3. torch（約 1.9 GB）：自己分段下載，可以續傳，核對 uv.lock 的 sha256 後用 uv pip install 裝
     （uv 自己下載不會續傳，uv issue #16934）
  4. yt-dlp[default,deno]：不在 uv.lock 裡（打包計畫 D8），沒裝時裝最新版；--update-ytdlp 更新
  5. VC++ 執行階段 DLL、ffmpeg、llama.cpp：照 runtime-manifest.json 下載（多個網址依序試、sha256、續傳），
     跟第 2、3 步同時在背景下載
  6. 基本測試：torch 看得到 CUDA、llama-server --list-devices、ffmpeg -version、NVENC 試編、主要套件 import
  7. 寫 runtime\\install-state.json：app/config.py 從這裡讀 ffmpeg、ffprobe、llama-server 的位置和 nvenc

參數：
  --update-ytdlp   更新 yt-dlp、yt-dlp-ejs、deno 後結束
  --repair         不管 install-state.json，全部重新檢查（repair.bat 用；環境變數 VS_REPAIR=1 也一樣）
  --repair-reset   repair.bat 用：確認沒有程式還在用 runtime（伺服器、llama-server、另一個視窗的安裝），
                   才刪掉 runtime\\venv 和 install-state.json。還在用時什麼都不刪，結束代碼 2
完整包（tools/build_release.py --full）：程式資料夾的 offline 資料夾裡附了 GitHub 上的檔案（ffmpeg、llama.cpp、cudart、
uv.lock 裡的 en-core-web-sm）。下載前先找 offline 資料夾裡同名的檔案，大小和 sha256 跟 manifest、uv.lock 一樣就直接用
（ffmpeg 這些硬連結，不行才複製；en-core-web-sm 讓 uv sync 跳過，再用 uv pip install 裝 offline 的檔案），不對或沒有才下載。
測試用的環境變數：
  VS_DOWNLOAD_CACHE  資料夾：下載前先找這裡有沒有同名而且 sha256 相同的檔案，下載完也複製一份過去
  VS_SETUP_GATE      檔案路徑：用到顯示卡的基本測試之前，等這個檔案出現（測試程式先確認顯示卡空著）
"""
import ctypes
import ctypes.wintypes as wintypes
import hashlib
import http.client
import json
import os
import shutil
import ssl
import subprocess
import sys
import threading
import time
import tomllib
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections import deque
from fnmatch import fnmatch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "launcher"
RT = ROOT / "runtime"
DOWNLOADS = RT / "downloads"
BIN = RT / "bin"
LOGS = RT / "logs"
LOG_PATH = LOGS / "install.log"
VENV = RT / "venv"
VENV_PY = VENV / "Scripts" / "python.exe"
STATE_PATH = RT / "install-state.json"
MANIFEST_PATH = LAUNCHER / "runtime-manifest.json"
LOCK_PATH = ROOT / "uv.lock"
UV = RT / "uv" / "uv.exe"
SETUP_LOCK = RT / "setup.lock"
OFFLINE = ROOT / "offline"            # 完整包附的下載檔；一般的 zip 沒有這個資料夾
SLOW_HOSTS = ("github.com", "githubusercontent.com")     # 完整包要附的檔案的來源（每條連線只有 30 到 50 KB/s）

STATE_VERSION = 1
CONNECTIONS = 6                       # GitHub 每條連線限速，實測 4 條連線約 4 倍快
SEGMENT_MIN = 32 * 1024 * 1024        # 小於這個大小只開 1 條連線
SEGMENT_SIZE = 16 * 1024 * 1024       # 切成很多小段輪流下載：某條連線變很慢時，只會卡住一小段
MAX_SEGMENTS = 64
CHUNK = 256 * 1024
ATTEMPTS = 5
BACKOFF = (5, 15, 30, 60)
USER_AGENT = "video-subtitle-setup/1"
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
DLL_NOT_FOUND = {0xC0000135, -1073741515}
VC_REDIST_URL = "https://aka.ms/vs/17/release/vc_redist.x64.exe"

# 快速路徑也要確認還在的檔案：venv 被刪到一半（例如程式開著時刪 venv、防毒刪檔）時不能直接啟動
VENV_MARKERS = ["pyvenv.cfg", "Lib/site-packages/fastapi/__init__.py", "Lib/site-packages/uvicorn/__init__.py",
                "Lib/site-packages/torch/__init__.py"]
DISK_FULL_WORDS = ("os error 112", "os error 28", "winerror 112", "no space left", "not enough space", "磁碟空間不足")

# 伺服器一定要能 import 的套件（失敗就不啟動）；其他的只影響部分功能
CORE_MODULES = ["fastapi", "uvicorn", "pydantic", "numpy", "soundfile", "psutil", "huggingface_hub", "opencc",
                "jaconv", "transformers", "qwen_asr", "silero_vad", "librosa"]
FEATURE_MODULES = ["fugashi", "unidic_lite", "pyopenjtalk", "onnxruntime", "spacy", "en_core_web_sm", "wordfreq",
                   "ckip_transformers", "audio_separator.separator", "yt_dlp"]

# start.bat 呼叫前已經跑過 launcher\env.cmd；直接執行這支程式（測試）時也要導回 runtime 裡
_ENV_DEFAULTS = {
    "UV_CACHE_DIR": RT / "cache" / "uv",
    "UV_PYTHON_INSTALL_DIR": RT / "python",
    "UV_PYTHON_BIN_DIR": RT / "python-bin",
    "UV_PYTHON_INSTALL_BIN": "0",
    "UV_PYTHON_INSTALL_REGISTRY": "0",
    "UV_TOOL_DIR": RT / "uv-tools",
    "UV_TOOL_BIN_DIR": RT / "uv-tools" / "bin",
    "UV_CREDENTIALS_DIR": RT / "uv-credentials",
    "UV_NO_CONFIG": "1",
    "UV_MANAGED_PYTHON": "1",
    "UV_PROJECT_ENVIRONMENT": VENV,
    "UV_HTTP_TIMEOUT": "120",
    "UV_HTTP_RETRIES": "5",
    "UV_SYSTEM_CERTS": "1",
    "TEMP": RT / "tmp",
    "TMP": RT / "tmp",
    "PYTHONNOUSERSITE": "1",
    "HF_HOME": RT / "cache" / "huggingface",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "TORCH_HOME": RT / "cache" / "torch",
    "CUDA_CACHE_PATH": RT / "cache" / "nv-compute",
    "NUMBA_CACHE_DIR": RT / "cache" / "numba",
    "MPLCONFIGDIR": RT / "cache" / "matplotlib",
    "XDG_CACHE_HOME": RT / "cache" / "xdg",
    "DENO_DIR": RT / "cache" / "deno",
    "DENO_NO_UPDATE_CHECK": "1",
    "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
}


class SetupError(Exception):
    """給使用者看的中文原因；start.bat 會暫停讓人看到。"""


# ---------- 輸出與 log ----------

_log_lock = threading.Lock()


def log(message: str):
    try:
        LOGS.mkdir(parents=True, exist_ok=True)
        with _log_lock, open(LOG_PATH, "a", encoding="utf-8") as f:
            for line in str(message).rstrip().splitlines() or [""]:
                f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + line + "\n")
    except OSError:
        pass


def _width(text: str) -> int:
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in text)


def _fit(text: str, cols: int) -> str:
    out, w = [], 0
    for c in text:
        cw = 2 if unicodedata.east_asian_width(c) in "WF" else 1
        if w + cw > cols:
            break
        out.append(c)
        w += cw
    return "".join(out)


def mb(n: float) -> str:
    return f"{n / 1048576:,.0f} MB"


def duration(seconds: float) -> str:
    seconds = int(max(0, seconds))
    if seconds < 60:
        return f"{seconds} 秒"
    if seconds < 3600:
        return f"{seconds // 60} 分 {seconds % 60:02d} 秒"
    return f"{seconds // 3600} 小時 {seconds % 3600 // 60:02d} 分"


class Console:
    """一般訊息逐行印；下載、安裝進度合成一行，用 \\r 原地更新（輸出不是主控台時每 20 秒印一行）。"""

    def __init__(self):
        self.lock = threading.RLock()
        self.tasks: dict[str, str] = {}
        try:
            self.tty = sys.stdout.isatty()
        except (AttributeError, ValueError):
            self.tty = False
        self.shown = 0
        self.last_plain = 0.0
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        if self._thread is None:
            self._thread = threading.Thread(target=self._loop, name="status", daemon=True)
            self._thread.start()

    def stop(self):
        self._stop.set()
        with self.lock:
            self.tasks.clear()
            self._clear()

    def say(self, text: str = ""):
        with self.lock:
            self._clear()
            print(text, flush=True)
        log("SAY " + text)

    def task(self, key: str, text: str):
        with self.lock:
            self.tasks[key] = text

    def end(self, key: str):
        with self.lock:
            self.tasks.pop(key, None)

    def _clear(self):
        if self.tty and self.shown:
            sys.stdout.write("\r" + " " * self.shown + "\r")
            sys.stdout.flush()
            self.shown = 0

    def _loop(self):
        while not self._stop.wait(0.5):
            with self.lock:
                if not self.tasks:
                    self._clear()
                    continue
                line = "｜".join(self.tasks.values())
                if self.tty:
                    cols = max(40, shutil.get_terminal_size((80, 25)).columns - 2)
                    line = _fit(line, cols)
                    w = _width(line)
                    sys.stdout.write("\r" + line + " " * max(0, self.shown - w))
                    sys.stdout.flush()
                    self.shown = max(w, 1)
                elif time.time() - self.last_plain >= 20:
                    print("  " + line, flush=True)
                    self.last_plain = time.time()


# ---------- 小工具 ----------

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(4 * 1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def read_json(path: Path) -> dict:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def write_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    replace_retry(tmp, path)


def replace_retry(src: Path, dst: Path, tries: int = 10, wait: float = 1.0):
    """改名。防毒、備份、索引工具剛好開著檔案時（WinError 5、32）等一下再試。"""
    for i in range(tries):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if i + 1 >= tries:
                raise
            time.sleep(wait)


def disk_full(e: BaseException) -> bool:
    return isinstance(e, OSError) and (getattr(e, "winerror", None) in (39, 112) or e.errno == 28)


def os_problem(e: BaseException) -> str | None:
    """寫檔、解壓縮失敗（OSError）時給使用者看的原因；不是這幾種回 None。"""
    if not isinstance(e, OSError):
        return None
    if disk_full(e):
        try:
            free = f"，{os.path.splitdrive(str(ROOT))[0]} 只剩 {shutil.disk_usage(ROOT).free / 1024 ** 3:.1f} GB"
        except OSError:
            free = ""
        return f"安裝到一半磁碟空間不夠了{free}。清出空間後重新打開 start.bat，會從中斷的地方接著裝。"
    if getattr(e, "winerror", None) in (32, 33):
        return ("安裝時有檔案被其他程式鎖住（多半是防毒軟體在掃描，或另一個 Bilingual Subtitles 視窗還開著）。"
                "等一下再重新打開 start.bat。")
    if isinstance(e, PermissionError):
        return ("安裝時沒辦法寫入程式資料夾（可能被防毒軟體擋住）。"
                "請把整個資料夾搬到 D:\\BilingualSubtitles 這類位置，再重新打開 start.bat。")
    return None


def rel(path: Path) -> str:
    """寫進 install-state 的路徑：相對程式根目錄（資料夾搬家後照樣能用；app/config.py 會接回根目錄）。"""
    try:
        return str(Path(path).relative_to(ROOT))
    except ValueError:
        return str(path)


def inside(path: Path, parent: Path) -> bool:
    try:
        Path(path).resolve().relative_to(Path(parent).resolve())
        return True
    except (ValueError, OSError):
        return False


def remove_tree(path: Path, parent: Path):
    """只刪 parent 底下的資料夾，不跟進 junction、捷徑。"""
    path = Path(path)
    if not path.exists() and not path.is_symlink():
        return
    if not inside(path.parent, parent) or path == parent:
        raise SetupError(f"不刪除 {path}：不在 {parent} 裡面")
    if path.is_symlink() or _is_junction(path):
        os.rmdir(path) if path.is_dir() else path.unlink()
        return

    def readonly(func, p, exc):
        # 唯讀屬性的檔案（有些套件的資料檔）：拿掉唯讀再刪一次
        if isinstance(exc, PermissionError) and os.path.exists(p):
            os.chmod(p, 0o666)
            func(p)
        else:
            raise exc

    shutil.rmtree(path, onexc=readonly)


def _is_junction(path: Path) -> bool:
    try:
        return bool(getattr(os.path, "isjunction", lambda p: False)(path))
    except OSError:
        return False


def file_version(path: Path) -> tuple | None:
    """Windows 檔案內容的版本（14.44.35211.0 → (14, 44, 35211, 0)）；讀不到回 None。"""
    try:
        ver = ctypes.WinDLL("version")
    except OSError:
        return None
    ver.GetFileVersionInfoSizeW.argtypes = [ctypes.c_wchar_p, ctypes.POINTER(wintypes.DWORD)]
    ver.GetFileVersionInfoSizeW.restype = wintypes.DWORD
    ver.GetFileVersionInfoW.argtypes = [ctypes.c_wchar_p, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p]
    ver.GetFileVersionInfoW.restype = wintypes.BOOL
    ver.VerQueryValueW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.POINTER(ctypes.c_void_p),
                                   ctypes.POINTER(wintypes.UINT)]
    ver.VerQueryValueW.restype = wintypes.BOOL
    handle = wintypes.DWORD()
    size = ver.GetFileVersionInfoSizeW(str(path), ctypes.byref(handle))
    if not size:
        return None
    buf = ctypes.create_string_buffer(size)
    if not ver.GetFileVersionInfoW(str(path), 0, size, buf):
        return None
    ptr, length = ctypes.c_void_p(), wintypes.UINT()
    if not ver.VerQueryValueW(buf, "\\", ctypes.byref(ptr), ctypes.byref(length)) or not ptr.value:
        return None

    class Fixed(ctypes.Structure):
        _fields_ = [("sig", wintypes.DWORD), ("struc", wintypes.DWORD), ("ms", wintypes.DWORD), ("ls", wintypes.DWORD)]

    info = Fixed.from_address(ptr.value)
    return info.ms >> 16, info.ms & 0xFFFF, info.ls >> 16, info.ls & 0xFFFF


def child_env(**extra) -> dict:
    env = dict(os.environ)
    for key, value in _ENV_DEFAULTS.items():
        env.setdefault(key, str(value))
    for key in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "CONDA_PREFIX"):
        env.pop(key, None)
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["NO_COLOR"] = "1"
    env.update({k: str(v) for k, v in extra.items()})
    return env


# ---------- 設定檔 ----------

def load_manifest() -> dict:
    data = read_json(MANIFEST_PATH)
    if not data:
        raise SetupError(f"讀不到 {rel(MANIFEST_PATH)}，請重新解壓縮程式")
    return data


def base_python(manifest: dict) -> Path:
    return RT / "python" / manifest["python"]["dir"] / "python.exe"


def fingerprint(manifest: dict) -> dict:
    return {
        "uv_lock": sha256_file(LOCK_PATH) if LOCK_PATH.is_file() else None,
        "manifest": sha256_file(MANIFEST_PATH),
        "python": manifest["python"]["version"],
    }


def lock_torch() -> dict:
    """uv.lock 裡 torch 的版本、win_amd64 cp312 wheel 的網址和 sha256。"""
    lock = tomllib.loads(LOCK_PATH.read_text(encoding="utf-8"))
    for pkg in lock.get("package", []):
        if pkg.get("name") != "torch":
            continue
        for wheel in pkg.get("wheels", []):
            url = wheel.get("url", "")
            if "cp312" in url and "win_amd64" in url:
                name = urllib.parse.unquote(url.rsplit("/", 1)[-1])
                return {"version": pkg["version"], "name": name, "urls": [url],
                        "sha256": wheel["hash"].split(":", 1)[1], "size": wheel.get("size")}
    raise SetupError("uv.lock 裡找不到 torch 的 Windows 版，請重新解壓縮程式")


def slow_host(url: str) -> bool:
    host = (urllib.parse.urlsplit(url).hostname or "").lower()
    return any(host == h or host.endswith("." + h) for h in SLOW_HOSTS)


def lock_slow_wheels(lock_path: Path | None = None) -> list[dict]:
    """uv.lock 裡放在 GitHub 上、這台電腦會裝的 wheel（現在只有 en-core-web-sm）。完整包把它們附在 offline 資料夾。
    每一項 {"package", "version", "name", "urls", "sha256", "size"（uv.lock 沒寫時是 None）}；讀不到 uv.lock 回空的。"""
    try:
        lock = tomllib.loads(Path(lock_path or LOCK_PATH).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    out = []
    for pkg in lock.get("package", []):
        for wheel in pkg.get("wheels", []):
            url = wheel.get("url", "")
            name = urllib.parse.unquote(url.rsplit("/", 1)[-1])
            usable = name.endswith("-none-any.whl") or ("win_amd64" in name and ("cp312" in name or "abi3" in name
                                                                                 or "-py3-" in name))
            if slow_host(url) and usable and wheel.get("hash", "").startswith("sha256:"):
                out.append({"package": pkg["name"], "version": pkg["version"], "name": name, "urls": [url],
                            "sha256": wheel["hash"].split(":", 1)[1].lower(), "size": wheel.get("size")})
    return out


def slow_runtime_files(manifest: dict, lock_path: Path | None = None) -> list[dict]:
    """安裝執行環境時要從 GitHub 下載的檔案：manifest 裡網址在 GitHub 上的，加上 uv.lock 裡在 GitHub 上的 wheel。"""
    files = [f for section in manifest.values() if isinstance(section, dict)
             for f in section.get("files") or [] if any(slow_host(u) for u in f["urls"])]
    return files + lock_slow_wheels(lock_path)


def offline_complete(manifest: dict, folder: Path | None = None, lock_path: Path | None = None) -> bool:
    """完整包：GitHub 上的執行環境檔在 offline 資料夾裡都有（只看檔名和大小，sha256 安裝時才核對）。
    precheck.py 用它決定要不要檢查 GitHub 連不連得到。"""
    files = slow_runtime_files(manifest, lock_path)
    return bool(files) and all(offline_file(f, folder) is not None for f in files)


# ---------- 下載 ----------

class _NoRange(Exception):
    pass


def _open(url: str, start: int | None = None, end: int | None = None, timeout: float = 60):
    headers = {"User-Agent": USER_AGENT, "Accept-Encoding": "identity"}
    if start is not None:
        headers["Range"] = f"bytes={start}-" if end is None else f"bytes={start}-{end}"
    return urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=timeout)


def remote_size(url: str) -> int | None:
    with _open(url, 0, 0, timeout=30) as r:
        if r.status == 206:
            total = (r.headers.get("Content-Range") or "").rsplit("/", 1)[-1]
            return int(total) if total.isdigit() else None
        length = r.headers.get("Content-Length") or ""
        return int(length) if length.isdigit() else None


def _cache_dir() -> Path | None:
    raw = (os.environ.get("VS_DOWNLOAD_CACHE") or "").strip().strip('"')
    return Path(raw) if raw else None


def _from_cache(name: str, sha: str, size: int | None, final: Path) -> bool:
    cache = _cache_dir()
    src = cache / name if cache else None
    if not src or not src.is_file() or (size and src.stat().st_size != size):
        return False
    if sha256_file(src) != sha:
        return False
    final.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, final)
    log(f"copied {name} from VS_DOWNLOAD_CACHE")
    return True


def offline_file(item: dict, folder: Path | None = None) -> Path | None:
    """完整包 offline\\ 裡同名、大小也對的檔案（還沒核對 sha256）；沒有回 None。"""
    src = Path(folder or OFFLINE) / item["name"]
    try:
        if not src.is_file() or (item.get("size") and src.stat().st_size != item["size"]):
            return None
    except OSError:
        return None
    return src


# 這次執行已經核對過 sha256 的檔案 {路徑: (大小, 修改時間, sha256)}：offline 的檔案放好後 download() 不用再算一次
_verified: dict[str, tuple] = {}


def _mark_verified(path: Path, sha: str):
    try:
        st = path.stat()
        _verified[os.path.normcase(str(path))] = (st.st_size, st.st_mtime_ns, sha.lower())
    except OSError:
        pass


def _is_verified(path: Path, sha: str) -> bool:
    try:
        st = path.stat()
    except OSError:
        return False
    return _verified.get(os.path.normcase(str(path))) == (st.st_size, st.st_mtime_ns, sha.lower())


def _from_offline(item: dict, final: Path, con: Console, key: str, label: str) -> bool:
    """offline\\ 裡的檔案 sha256 跟 manifest 一樣：硬連結（不行就複製）到 final。不對或沒有回 False，照常下載。"""
    src = offline_file(item)
    if src is None:
        return False
    name, sha = item["name"], item["sha256"].lower()
    tmp = final.with_name(final.name + ".offline")
    try:
        con.task(key, f"{label} 核對 offline 資料夾裡的檔案")
        got = sha256_file(src)
        if got != sha:
            log(f"offline {name}: sha256 {got} does not match {sha}, downloading instead")
            return False
        final.parent.mkdir(parents=True, exist_ok=True)
        tmp.unlink(missing_ok=True)
        try:
            os.link(src, tmp)
        except OSError:
            shutil.copyfile(src, tmp)
        replace_retry(tmp, final)
    except OSError as e:
        log(f"could not use offline {name}: {type(e).__name__}: {e}")
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        return False
    finally:
        con.end(key)
    _mark_verified(final, sha)
    log(f"using {name} from the offline folder")
    return True


def _to_cache(final: Path):
    cache = _cache_dir()
    if cache and not (cache / final.name).exists():
        try:
            cache.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(final, cache / final.name)
        except OSError as e:
            log(f"could not copy {final.name} to VS_DOWNLOAD_CACHE: {e}")


class Progress:
    def __init__(self, con: Console, key: str, label: str, total: int):
        self.con, self.key, self.label, self.total = con, key, label, total
        self.samples: deque = deque(maxlen=20)

    def update(self, done: int, note: str = ""):
        now = time.time()
        self.samples.append((now, done))
        while len(self.samples) > 2 and now - self.samples[0][0] > 8:
            self.samples.popleft()
        speed = 0.0
        if len(self.samples) >= 2 and now - self.samples[0][0] > 0.5:
            speed = (done - self.samples[0][1]) / (now - self.samples[0][0])
        text = f"{self.label} {done / 1048576:,.0f} / {mb(self.total)}"
        if speed > 0:
            text += f"，{speed / 1048576:.1f} MB/s，約剩 {duration((self.total - done) / speed)}"
        self.con.task(self.key, text + note)


def _segments(size: int, n: int) -> list:
    step = -(-size // n)
    return [[start, min(start + step, size) - 1, 0] for start in range(0, size, step)]


def _valid_segments(segs, size: int) -> bool:
    try:
        pos = 0
        for s, e, d in segs:
            if s != pos or e < s or not 0 <= d <= e - s + 1:
                return False
            pos = e + 1
        return pos == size
    except (TypeError, ValueError):
        return False


def download_status(item: dict, dest: Path = DOWNLOADS) -> tuple[int, int | None]:
    """(已經下載的位元組, 總大小)。已經下載好的算全部；.part 下載到一半的照 .part.json 的分段紀錄。
    總大小不知道時（torch 在 uv.lock 裡沒有寫大小）用 .part.json 記下的，還沒開始下載就是 None。"""
    name, sha, size = item["name"], item["sha256"].lower(), item.get("size")
    final = dest / name
    if final.is_file():
        return final.stat().st_size, final.stat().st_size
    meta = read_json(dest / (name + ".part.json"))
    msize, segs, part = meta.get("size"), meta.get("segments"), dest / (name + ".part")
    if isinstance(msize, int) and msize > 0 and (not size or msize == size) and meta.get("sha256") == sha \
            and part.is_file() and part.stat().st_size == msize and _valid_segments(segs, msize):
        return sum(s[2] for s in segs), msize
    return 0, size


def remaining_bytes(item: dict, dest: Path = DOWNLOADS) -> int:
    """這個檔案還要下載多少（已經下載好、或 .part 下載到一半的部分不算）；大小不知道時回 0。"""
    done, total = download_status(item, dest)
    return max(0, (total or 0) - done)


def _fetch(url: str, part: Path, meta_path: Path, size: int, sha: str, progress: Progress):
    """分段下載到 part（預先配置好大小），進度存在 meta_path，中斷後從各段停下的地方接著下載。"""
    meta = read_json(meta_path)
    segs = meta.get("segments")
    if not (meta.get("size") == size and meta.get("sha256") == sha and part.is_file()
            and part.stat().st_size == size and _valid_segments(segs, size)):
        n = min(MAX_SEGMENTS, max(CONNECTIONS, -(-size // SEGMENT_SIZE))) if size >= SEGMENT_MIN else 1
        segs = _segments(size, n)
        with open(part, "wb") as f:
            f.truncate(size)
    save_errors = [0]

    def save():
        # 進度檔暫時被別的程式開著（防毒、備份、索引工具）時不要中斷下載：下次再存，最多少記幾秒的進度
        try:
            write_json(meta_path, {"size": size, "sha256": sha, "segments": segs})
        except OSError as e:
            save_errors[0] += 1
            if save_errors[0] <= 3:
                log(f"could not save {meta_path.name} (will retry): {type(e).__name__}: {e}")
    save()
    stop = threading.Event()
    errors: list = []
    pending = deque(seg for seg in segs if seg[0] + seg[2] <= seg[1])
    pending_lock = threading.Lock()

    def fetch_segment(seg):
        pos = seg[0] + seg[2]
        with _open(url, pos, seg[1]) as r:
            if r.status != 206:
                raise _NoRange()
            with open(part, "r+b") as f:
                f.seek(pos)
                while not stop.is_set():
                    want = min(CHUNK, seg[1] - (seg[0] + seg[2]) + 1)
                    if want <= 0:
                        break
                    data = r.read(want)
                    if not data:
                        raise ConnectionError("連線提早結束")
                    f.write(data)
                    seg[2] += len(data)

    def work():
        try:
            while not stop.is_set():
                with pending_lock:
                    if not pending:
                        return
                    seg = pending.popleft()
                fetch_segment(seg)
        except BaseException as e:  # noqa: BLE001
            errors.append(e)
            stop.set()

    threads = [threading.Thread(target=work, daemon=True) for _ in range(min(CONNECTIONS, len(pending)))]
    for t in threads:
        t.start()
    last_save = time.time()
    try:
        while any(t.is_alive() for t in threads):
            time.sleep(0.5)
            progress.update(sum(s[2] for s in segs))
            if time.time() - last_save >= 2:
                save()
                last_save = time.time()
    except KeyboardInterrupt:
        stop.set()          # 關視窗、Ctrl+C：程序本身就要結束了，不等連線
        raise
    except BaseException:
        # 這裡出錯時下載執行緒一定要先停下來，不然下一輪會有兩批連線同時寫同一個 .part
        stop.set()
        for t in threads:
            t.join()
        save()
        raise
    for t in threads:
        t.join()
    save()
    progress.update(sum(s[2] for s in segs))
    if errors:
        raise errors[0]
    if sum(s[2] for s in segs) != size:
        raise ConnectionError("下載不完整")


def _fetch_whole(url: str, part: Path, size: int, progress: Progress):
    """伺服器不支援分段時：一條連線從頭下載。"""
    done = 0
    with _open(url) as r, open(part, "wb") as f:
        while True:
            data = r.read(CHUNK)
            if not data:
                break
            f.write(data)
            done += len(data)
            progress.update(done)
    if done != size:
        raise ConnectionError("下載不完整")


def network_message(label: str, error) -> str:
    tail = "已經下載的部分會保留，重新打開 start.bat 會接著下載。"
    if isinstance(error, ssl.SSLError) or "CERTIFICATE" in str(error).upper():
        return (f"下載 {label} 失敗：連線被攔截（常見於防毒軟體的 HTTPS 掃描或公司、學校網路）。"
                f"暫時關掉防毒的網頁掃描，或換個網路再試。{tail}")
    if isinstance(error, urllib.error.HTTPError):
        if error.code in (403, 404, 410):
            return f"下載 {label} 失敗：下載網址已經失效（HTTP {error.code}），請跟作者拿新版的程式。"
        return f"下載 {label} 失敗：伺服器回應錯誤（HTTP {error.code}），等一下再試。{tail}"
    if disk_full(error):
        return f"下載 {label} 失敗：磁碟空間不夠。清出空間後重新打開 start.bat。"
    if isinstance(error, SetupError):
        return str(error)
    return f"下載 {label} 失敗：網路中斷或太慢（{error}）。{tail}"


def download(item: dict, label: str, con: Console, dest: Path = DOWNLOADS, key: str | None = None) -> Path:
    """下載 manifest 的一個檔案：已經有、sha256 對就直接用；完整包 offline\\ 裡有、sha256 對也直接用；
    網址依序試，每輪失敗等一下再試，最多 ATTEMPTS 輪。"""
    name, sha = item["name"], item["sha256"].lower()
    size = item.get("size")
    key = key or name
    dest.mkdir(parents=True, exist_ok=True)
    final = dest / name
    if final.is_file():
        if _is_verified(final, sha) or ((not size or final.stat().st_size == size) and sha256_file(final) == sha):
            return final
        final.unlink()
    if _from_offline(item, final, con, key, label):
        return final
    if _from_cache(name, sha, size, final):
        return final
    part = dest / (name + ".part")
    meta = dest / (name + ".part.json")
    last = None
    try:
        for attempt in range(ATTEMPTS):
            for url in item["urls"]:
                try:
                    total = size or remote_size(url)
                    if not total:
                        raise SetupError(f"下載 {label} 失敗：拿不到檔案大小")
                    progress = Progress(con, key, label, total)
                    log(f"download {name} from {url} (attempt {attempt + 1})")
                    try:
                        _fetch(url, part, meta, total, sha, progress)
                    except _NoRange:
                        log(f"{url} does not support Range, downloading in one piece")
                        meta.unlink(missing_ok=True)
                        _fetch_whole(url, part, total, progress)
                    con.task(key, f"{label} 核對檔案")
                    got = sha256_file(part)
                    if got != sha:
                        log(f"sha256 mismatch for {name}: got {got}, want {sha}; deleting")
                        part.unlink(missing_ok=True)
                        meta.unlink(missing_ok=True)
                        last = SetupError(f"下載 {label} 失敗：檔案內容不對（sha256 不符），已經刪掉，重新打開 start.bat 會重新下載")
                        continue
                    replace_retry(part, final)
                    try:
                        meta.unlink(missing_ok=True)
                    except OSError:
                        pass
                    _to_cache(final)
                    log(f"downloaded {name}")
                    return final
                except (OSError, urllib.error.URLError, http.client.HTTPException, ValueError, SetupError) as e:
                    last = e
                    log(f"download {name} failed: {type(e).__name__}: {e}")
                    if disk_full(e):
                        raise SetupError(network_message(label, e))
            if attempt + 1 < ATTEMPTS:
                wait = BACKOFF[min(attempt, len(BACKOFF) - 1)]
                for left in range(wait, 0, -1):
                    con.task(key, f"{label}：連線中斷，{left} 秒後重試（第 {attempt + 2} 次）")
                    time.sleep(1)
    finally:
        con.end(key)
    raise SetupError(network_message(label, last))


# ---------- 子程序 ----------

def run_logged(cmd: list, con: Console, key: str, label: str, timeout: float | None = None, cwd: Path = ROOT,
               env: dict | None = None) -> tuple:
    """執行指令，完整輸出寫進 install.log，畫面上只顯示經過時間；回傳 (結束代碼, 最後 30 行)。"""
    log("RUN " + " ".join(f'"{c}"' if " " in str(c) else str(c) for c in cmd))
    started = time.time()
    tail: deque = deque(maxlen=30)
    current = {"text": ""}
    try:
        proc = subprocess.Popen([str(c) for c in cmd], cwd=str(cwd), env=child_env(**(env or {})), stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, creationflags=NO_WINDOW)
    except OSError as e:
        log(f"could not start: {e}")
        return -1, [str(e)]

    def reader():
        for raw in iter(proc.stdout.readline, b""):
            line = raw.decode("utf-8", errors="replace").rstrip()
            if not line:
                continue
            tail.append(line)
            log("  | " + line)
            if line.lstrip().startswith("Downloading "):
                current["text"] = "下載 " + line.strip()[len("Downloading "):]

    t = threading.Thread(target=reader, daemon=True)
    t.start()
    while proc.poll() is None:
        extra = f"，{current['text']}" if current["text"] else ""
        con.task(key, f"{label} 經過 {duration(time.time() - started)}{extra}")
        if timeout and time.time() - started > timeout:
            proc.kill()
            log("killed: timeout")
            break
        time.sleep(0.5)
    proc.wait()
    t.join(timeout=5)
    con.end(key)
    log(f"exit {proc.returncode} after {time.time() - started:.1f}s")
    return proc.returncode, list(tail)


def run_capture(cmd: list, timeout: float, env: dict | None = None) -> tuple:
    """(結束代碼, stdout+stderr 文字)；執行不了回 (None, 原因)。"""
    try:
        res = subprocess.run([str(c) for c in cmd], capture_output=True, timeout=timeout, env=env or child_env(),
                             stdin=subprocess.DEVNULL, creationflags=NO_WINDOW, cwd=str(ROOT))
    except subprocess.TimeoutExpired:
        return None, f"逾時（{timeout:.0f} 秒）"
    except OSError as e:
        return None, f"{type(e).__name__}: {e}"
    text = (res.stdout or b"").decode("utf-8", errors="replace") + (res.stderr or b"").decode("utf-8", errors="replace")
    return res.returncode, text


def uv(args: list, con: Console, key: str, label: str, retries=(10, 30, 90), env: dict | None = None) -> list:
    """執行 uv，失敗時照 retries 等一下再試；最後還是失敗丟 SetupError。"""
    for attempt in range(len(retries) + 1):
        code, tail = run_logged([UV, *args], con, key, label, env=env)
        if code == 0:
            return tail
        if any(w in line.lower() for line in tail for w in DISK_FULL_WORDS):
            # 磁碟滿了：重試也沒用，不要讓使用者多等幾分鐘
            raise SetupError(f"{label}失敗：{os_problem(OSError(28, 'disk full'))}")
        if attempt < len(retries):
            wait = retries[attempt]
            for left in range(wait, 0, -1):
                con.task(key, f"{label}失敗，{left} 秒後重試（第 {attempt + 2} 次）")
                time.sleep(1)
    detail = "\n".join(tail[-8:])
    hint = "連線被攔截（防毒軟體的 HTTPS 掃描或公司網路）" if "certificate" in detail.lower() else "多半是網路中斷"
    raise SetupError(f"{label}失敗，{hint}。重新打開 start.bat 會接著裝。\n最後的訊息：\n{detail}")


# ---------- 步驟 ----------

def repair_moved(manifest: dict, old_root: str, con: Console):
    """資料夾搬過家：venv 的 pyvenv.cfg 指到舊路徑、uv 的 cpython-3.12 junction 指到舊資料夾。"""
    con.say(f"程式資料夾換過位置（原本在 {old_root}），正在修正路徑…")
    started = time.time()
    fix_pyvenv_cfg(manifest)
    if UV.is_file():
        code, _ = run_logged([UV, "python", "install", manifest["python"]["version"]], con, "junction", "修正 Python 路徑")
        if code != 0:
            log("uv python install failed while fixing the junction (not fatal)")
    log(f"moved from {old_root} to {ROOT}, fixed in {time.time() - started:.2f}s")


def fix_pyvenv_cfg(manifest: dict) -> bool:
    """把 pyvenv.cfg 的 home、executable、command 改成目前位置；有改回 True。"""
    cfg = VENV / "pyvenv.cfg"
    if not cfg.is_file():
        return False
    base = base_python(manifest)
    lines = cfg.read_text(encoding="utf-8", errors="replace").splitlines()
    wanted = {"home": str(base.parent), "executable": str(base),
              "command": f"{base} -m venv --without-pip {VENV}"}
    out, changed = [], False
    for line in lines:
        key = line.split("=", 1)[0].strip().lower()
        if key in wanted and "=" in line:
            new = f"{key} = {wanted[key]}"
            changed |= new != line
            out.append(new)
        else:
            out.append(line)
    if changed:
        cfg.write_text("\n".join(out) + "\n", encoding="utf-8")
        log(f"rewrote {cfg}")
    return changed


def venv_works() -> bool:
    if not VENV_PY.is_file():
        return False
    code, text = run_capture([VENV_PY, "-I", "-c", "import sys; print(sys.prefix)"], timeout=60)
    return code == 0


def ensure_venv(manifest: dict, con: Console) -> bool:
    """回傳 True 表示這次新建了 venv。用標準函式庫的 venv 建：Scripts\\python.exe 讀 pyvenv.cfg 的 home，
    資料夾搬家時只要改文字檔（uv 建的 venv 把 Python 路徑寫死在 python.exe 裡，搬家就要整個重建）。"""
    if venv_works():
        return False
    if VENV.exists():
        fix_pyvenv_cfg(manifest)
        if venv_works():
            return False
        con.say("runtime\\venv 壞掉了，重新建立…")
        remove_tree(VENV, RT)
    base = base_python(manifest)
    code, text = run_capture([base, "-I", "-m", "venv", "--without-pip", VENV], timeout=300)
    log(f"venv create exit {code}: {text}")
    if code != 0 or not venv_works():
        raise SetupError(f"建立 runtime\\venv 失敗：{text.strip()[-300:]}")
    return True


def sync_packages(con: Console):
    con.say("安裝 Python 套件（約 500 MB，不含 torch；已經裝好的會跳過）")
    skip, local = offline_wheels(con)
    args = ["sync", "--frozen", "--inexact", "--no-install-package", "torch"]
    for w in skip:
        args += ["--no-install-package", w["package"]]
    uv(args, con, "sync", "Python 套件")
    for w, path in local:
        uv(["pip", "install", "--python", VENV_PY, "--no-deps", "--no-cache", "--offline", "--reinstall-package",
            w["package"], path], con, "offline-wheel", f"安裝 {w['package']}")
        got = installed_version(w["package"])
        if got != w["version"]:
            raise SetupError(f"{w['package']} 沒有裝好（裝到的版本 {got}），請執行 repair.bat")


def offline_wheels(con: Console) -> tuple[list, list]:
    """uv.lock 裡放在 GitHub 上的 wheel（en-core-web-sm，12 MB 在 GitHub 上要等好幾分鐘）。
    完整包 offline 資料夾裡有、sha256 對的：uv sync 跳過它，之後用 uv pip install 裝 offline 的檔案。
    已經從檔案裝好的（之前用過 offline，之後 offline 資料夾刪掉了）也跳過，不然 uv sync 會當成來源不一樣，重新從 GitHub 下載。
    回傳 (uv sync 要跳過的, 要從 offline 裝的 [(項目, 路徑)])；沒有 offline 資料夾、也沒用過時兩個都是空的，跟以前一樣。"""
    skip, local = [], []
    for w in lock_slow_wheels():
        installed = installed_version(w["package"]) == w["version"]
        src = offline_file(w)
        if installed and (src is not None or installed_from_file(w["package"])):
            skip.append(w)
            continue
        if src is None:
            continue
        con.task("offline-wheel", f"核對 offline 資料夾裡的 {w['package']}")
        try:
            got = sha256_file(src)
        except OSError as e:
            got = f"{type(e).__name__}: {e}"
        finally:
            con.end("offline-wheel")
        if got == w["sha256"]:
            log(f"using {w['name']} from the offline folder")
            skip.append(w)
            local.append((w, src))
        else:
            log(f"offline {w['name']}: sha256 {got} does not match {w['sha256']}, downloading instead")
            con.say(f"offline 資料夾裡的 {w['package']} 壞了，改從網路下載")
    return skip, local


def _dist_info(dist: str) -> Path | None:
    site = VENV / "Lib" / "site-packages"
    norm = dist.lower().replace("-", "_")
    for d in site.glob("*.dist-info"):
        name = d.name[: -len(".dist-info")].partition("-")[0]
        if name.lower().replace("-", "_") == norm and (d / "RECORD").is_file():
            return d
    return None


def installed_version(dist: str) -> str | None:
    d = _dist_info(dist)
    return d.name[: -len(".dist-info")].partition("-")[2] if d else None


def installed_from_file(dist: str) -> bool:
    """這個套件是從本機的檔案裝的（direct_url.json 的網址是 file:）。"""
    d = _dist_info(dist)
    return bool(d) and str(read_json(d / "direct_url.json").get("url", "")).startswith("file:")


def ensure_torch(con: Console) -> dict:
    torch = lock_torch()
    have = installed_version("torch")
    if have == torch["version"]:
        return {"version": have}
    done, total = download_status(torch)
    if not done:
        con.say(f"下載 torch {torch['version']}（約 1.9 GB，可以續傳）")
    elif total and done < total:
        con.say(f"接著下載 torch {torch['version']}（還剩 {mb(total - done)}）")
    wheel = download(torch, "torch", con, key="torch")
    con.say("安裝 torch")
    # --no-cache：不在 runtime\cache\uv 另外解開一份（2.8 GB）。每次 repair 重新下載的 wheel 檔案時間不同，
    # uv 會當成新的東西再解開一份，舊的沒人用也不會清掉
    uv(["pip", "install", "--python", VENV_PY, "--no-deps", "--no-cache", "--reinstall-package", "torch", wheel],
       con, "torch-install", "安裝 torch")
    got = installed_version("torch")
    if got != torch["version"]:
        raise SetupError(f"torch 沒有裝好（裝到的版本 {got}），請執行 repair.bat")
    try:
        wheel.unlink()   # 已經解開在 venv 裡，壓縮檔留著佔 1.9 GB
    except OSError:
        pass
    return {"version": got}


YTDLP_RETRY_AFTER = 3600      # yt-dlp 沒裝好時，隔多久以後的啟動再試一次（離線時不要每次啟動都卡住）


def ensure_ytdlp(manifest: dict, con: Console, update: bool = False, quick: bool = False) -> dict:
    """yt-dlp 不在 uv.lock 裡（打包計畫 D8）。沒裝好只影響網址下載，不擋啟動。
    quick：啟動時補裝，連線逾時設短、不重試，離線時頂多多等幾十秒。"""
    have = installed_version("yt-dlp")
    deno = VENV / "Scripts" / "deno.exe"
    if have and deno.is_file() and not update:
        return {"version": have, "deno": rel(deno), "ok": True}
    req = manifest["yt_dlp"]["requirement"]
    con.say("更新 YouTube 下載元件（yt-dlp）" if update else "安裝 YouTube 下載元件（yt-dlp）")
    args = ["pip", "install", "--python", VENV_PY, req]
    if update:
        args += ["--upgrade-package", "yt-dlp", "--upgrade-package", "yt-dlp-ejs", "--upgrade-package", "deno"]
    env = {"UV_HTTP_TIMEOUT": "20", "UV_HTTP_RETRIES": "1"} if quick else None
    try:
        uv(args, con, "ytdlp", "YouTube 下載元件", retries=() if quick else (10,), env=env)
    except SetupError as e:
        log(f"yt-dlp install failed: {e}")
        con.say("注意：YouTube 下載元件沒裝好，網址下載暫時不能用，其他功能不受影響。之後打開 start.bat 會再試。")
        return {"version": installed_version("yt-dlp"), "ok": False, "tried_at": time.time()}
    return {"version": installed_version("yt-dlp"), "deno": rel(deno) if deno.is_file() else None, "ok": True}


def ytdlp_retry_due(state: dict) -> bool:
    info = state.get("yt_dlp") or {}
    return info.get("ok") is False and time.time() - float(info.get("tried_at") or 0) > YTDLP_RETRY_AFTER


def _extract(archive: Path, target: Path, keep: list, strip_top: bool):
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            if info.is_dir():
                continue
            parts = info.filename.replace("\\", "/").split("/")
            if strip_top:
                parts = parts[1:]
            if not parts or any(p in ("", ".", "..") for p in parts) or ":" in parts[0]:
                continue
            name = "/".join(parts)
            if keep and not any(fnmatch(name, pat) for pat in keep):
                continue
            dest = target.joinpath(*parts)
            dest.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as src, open(dest, "wb") as out:
                shutil.copyfileobj(src, out, 1024 * 1024)


def native_installed(entry: dict) -> Path | None:
    final = BIN / entry["dir"]
    marker = read_json(final / ".installed.json")
    want = sorted(f["sha256"] for f in entry["files"])
    if marker.get("files") == want and all((final / p).is_file() for p in entry["exe"].values()):
        return final
    return None


def install_native(entry: dict, label: str, archives: list, con: Console) -> Path:
    final = BIN / entry["dir"]
    tmp = BIN / (entry["dir"] + ".tmp")
    remove_tree(tmp, BIN)
    tmp.mkdir(parents=True)
    con.task("extract", f"解壓縮 {label}")
    try:
        for arc in archives:
            _extract(arc, tmp, entry.get("keep") or [], entry.get("strip_top_folder", False))
    finally:
        con.end("extract")
    missing = [p for p in entry["exe"].values() if not (tmp / p).is_file()]
    if missing:
        raise SetupError(f"{label} 的壓縮檔裡找不到 {', '.join(missing)}，請跟作者拿新版的程式")
    write_json(tmp / ".installed.json", {"version": entry["version"], "files": sorted(f["sha256"] for f in entry["files"])})
    try:
        remove_tree(final, BIN)
        os.replace(tmp, final)
    except OSError as e:
        raise SetupError(f"沒辦法更新 {rel(final)}（{e}）。可能有程式正在使用它，關掉其他 Bilingual Subtitles 視窗後重新打開 start.bat")
    for arc in archives:
        try:
            arc.unlink()
        except OSError:
            pass
    return final


def msvc_dir(manifest: dict, con: Console, archive: Path | None = None) -> Path:
    m = manifest["msvc_runtime"]
    target = BIN / f"msvc-runtime-{m['version']}"
    if all((target / n).is_file() for n in m["dlls"]):
        return target
    archive = archive or download(m["files"][0], "VC++ 執行階段", con)
    tmp = BIN / (target.name + ".tmp")
    remove_tree(tmp, BIN)
    tmp.mkdir(parents=True)
    with zipfile.ZipFile(archive) as z:
        for n in m["dlls"]:
            with z.open(m["wheel_dir"] + n) as src, open(tmp / n, "wb") as out:
                shutil.copyfileobj(src, out)
    remove_tree(target, BIN)
    os.replace(tmp, target)
    try:
        archive.unlink()
    except OSError:
        pass
    return target


def place_dlls(src: Path, dst: Path, names: list) -> list:
    """把 VC++ DLL 放到 dst（python.exe、llama-server.exe 旁邊）：沒有或版本比較舊才複製。
    Python 3.8 以後擴充模組的相依 DLL 不會從 PATH 找，一定要放在執行檔旁邊。"""
    placed = []
    for n in names:
        s, d = src / n, dst / n
        if d.is_file():
            have, new = file_version(d), file_version(s)
            if have and new and have >= new:
                continue
        try:
            shutil.copyfile(s, d)
        except PermissionError:
            # 這個 DLL 正在被使用（例如執行這支程式的 python.exe 載入的 vcruntime140.dll）：改名後再放新的
            old = d.with_name(d.name + f".old{int(time.time())}")
            os.replace(d, old)
            shutil.copyfile(s, d)
        placed.append(n)
    for old in dst.glob("*.dll.old*"):
        try:
            old.unlink()
        except OSError:
            pass
    if placed:
        log(f"placed {placed} into {dst}")
    return placed


# ---------- 基本測試 ----------

def _looks_like_dll_problem(text: str, code) -> bool:
    low = (text or "").lower()
    return code in DLL_NOT_FOUND or code == 0xC000007B or "dll load failed" in low or "winerror 126" in low \
        or "winerror 1114" in low or "[winerror 126]" in low


def test_python() -> dict:
    code, text = run_capture([VENV_PY, "-s", LAUNCHER / "selftest.py"], timeout=900)
    result = None
    for line in reversed(text.splitlines()):
        if line.startswith("SELFTEST "):
            try:
                result = json.loads(line[len("SELFTEST "):])
            except ValueError:
                pass
            break
    if result is None:
        return {"ok": False, "error": text.strip()[-800:] or f"exit {code}", "exit": code,
                "dll_problem": _looks_like_dll_problem(text, code)}
    log("selftest: " + json.dumps(result, ensure_ascii=False))
    return result


def test_llama(exe: Path) -> dict:
    code, text = run_capture([exe, "--list-devices"], timeout=120)
    devices = [l.strip() for l in text.splitlines() if l.strip().startswith("CUDA")]
    log(f"llama-server --list-devices exit {code}: {text.strip()[-600:]}")
    return {"ok": code == 0 and bool(devices), "exit": code, "devices": devices,
            "dll_problem": _looks_like_dll_problem(text, code), "output": text.strip()[-400:]}


def test_ffmpeg(ffmpeg: Path, ffprobe: Path) -> dict:
    code, text = run_capture([ffmpeg, "-hide_banner", "-version"], timeout=60)
    pcode, ptext = run_capture([ffprobe, "-hide_banner", "-version"], timeout=60)
    first = text.strip().splitlines()[0] if text.strip() else ""
    return {"ok": code == 0 and pcode == 0, "version": first, "exit": code, "ffprobe_exit": pcode,
            "dll_problem": _looks_like_dll_problem(text + ptext, code if code else pcode)}


def test_nvenc(ffmpeg: Path) -> dict:
    # 跟 app/media.py 的 nvenc_works 同樣的參數
    code, text = run_capture([ffmpeg, "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
                              "color=c=black:s=640x360:r=30:d=0.2", "-c:v", "h264_nvenc", "-preset", "p5", "-cq", "21",
                              "-pix_fmt", "yuv420p", "-f", "null", "-"], timeout=60)
    log(f"nvenc test exit {code}: {text.strip()[-400:]}")
    return {"ok": code == 0, "exit": code, "output": text.strip()[-300:]}


def quick_check(paths: dict, state: dict, con: Console) -> dict | None:
    """只搬了資料夾、其他都沒變：確認 venv、torch、ffmpeg、llama-server 在新位置開得起來，沿用上次的測試結果。
    有任何一項不行回 None，改跑完整測試。"""
    con.task("tests", "確認搬家後的執行環境")
    try:
        code, text = run_capture([VENV_PY, "-s", "-c", "import torch, fastapi, uvicorn; print(torch.__version__)"],
                                 timeout=300)
        ff = test_ffmpeg(paths["ffmpeg"], paths["ffprobe"])
        lcode, ltext = run_capture([paths["llama_server"], "--version"], timeout=60)
    finally:
        con.end("tests")
    log(f"quick check after move: python {code} {text.strip()[-200:]}; ffmpeg {ff['ok']}; llama {lcode}")
    if code != 0 or not ff["ok"] or lcode != 0:
        return None
    tests = dict(state["tests"])
    tests["moved_check"] = {"at": time.time(), "from": state.get("root")}
    return tests


MIN_DRIVER = 580


def versions_text(torch: dict) -> str:
    """驅動、PyTorch、翻譯程式各自的 CUDA 版本，放在顯示卡相關的錯誤後面，一眼看得出哪裡對不上。
    規則：驅動要支援到 CUDA 13（580 版以上）；驅動比較新沒關係（NVIDIA 驅動往下相容）。"""
    drv, drv_cuda = "未知", "讀不到"
    try:
        sys.path.insert(0, str(ROOT))
        from app import syscheck   # 只用標準函式庫
        info = syscheck.query() or {}
        drv = info.get("driver") or drv
        drv_cuda = info.get("cuda") or drv_cuda
    except Exception:  # noqa: BLE001
        pass
    llama_cuda = "未知"
    try:
        import re
        text = MANIFEST_PATH.read_text(encoding="utf-8")
        m = re.search(r"cudart-llama-bin-win-cuda-(\d+\.\d+)", text)
        llama_cuda = m.group(1) if m else llama_cuda
    except OSError:
        pass
    return (f"（你的驅動 {drv}，支援 CUDA {drv_cuda}；PyTorch 用 CUDA {torch.get('cuda_build') or '未知'}；"
            f"翻譯程式用 CUDA {llama_cuda}；需要驅動 {MIN_DRIVER} 版以上）")


def run_tests(paths: dict, con: Console) -> dict:
    con.task("tests", "測試顯示卡和各個元件（第一次約 1 到 2 分鐘）")
    try:
        results = {"ffmpeg": test_ffmpeg(paths["ffmpeg"], paths["ffprobe"])}
        results["nvenc"] = test_nvenc(paths["ffmpeg"]) if results["ffmpeg"]["ok"] else {"ok": False, "exit": None}
        results["llama"] = test_llama(paths["llama_server"])
        results["python"] = test_python()
    finally:
        con.end("tests")
    py = results["python"]
    fatal, warnings = [], []
    dll = any(r.get("dll_problem") for r in results.values() if isinstance(r, dict))
    torch = py.get("torch") or {}
    if not py.get("modules") and not torch:
        fatal.append("Python 套件的測試沒辦法執行：" + str(py.get("error", ""))[-300:])
    else:
        if not torch.get("ok"):
            fatal.append(f"torch 載入失敗：{torch.get('error')}")
            dll = dll or _looks_like_dll_problem(str(torch.get("error")), None)
        elif not torch.get("cuda"):
            warnings.append("torch 看不到 NVIDIA 顯示卡（CUDA），辨識會沒辦法用。請更新顯示卡驅動到 580 版以上，更新後執行 repair.bat"
                            + versions_text(torch))
        elif torch.get("arch_supported") is False:
            cap = ".".join(str(x) for x in torch.get("capability") or []) or "未知"
            warnings.append(f"這張顯示卡（{torch.get('device')}，架構 {cap}）比程式用的 PyTorch {torch.get('version')} 還新，"
                            "PyTorch 還不支援它，辨識會沒辦法用。這不是驅動的問題，要等程式出新版本")
        elif torch.get("op") is not True:
            warnings.append(f"torch 在這張顯示卡上跑不動：{torch.get('op_error')}。請先更新顯示卡驅動再執行 repair.bat"
                            + versions_text(torch))
        mods = py.get("modules") or {}
        bad_core = [m for m in CORE_MODULES if not (mods.get(m) or {}).get("ok")]
        bad_feature = [m for m in FEATURE_MODULES if not (mods.get(m) or {}).get("ok")]
        if bad_core:
            fatal.append("這些套件載入失敗：" + "、".join(f"{m}（{(mods.get(m) or {}).get('error')}）" for m in bad_core))
        if bad_feature:
            warnings.append("部分功能的套件載入失敗：" + "、".join(bad_feature) + "（詳細原因在 runtime\\logs\\install.log）")
    if not results["ffmpeg"]["ok"]:
        fatal.append("ffmpeg 開不起來。可能被防毒軟體擋掉或刪掉了，請執行 repair.bat")
    if not results["llama"]["ok"]:
        if results["llama"].get("exit") == 0:
            warnings.append("翻譯程式 llama-server 看不到 NVIDIA 顯示卡，翻譯會沒辦法用。請更新顯示卡驅動"
                            + versions_text(torch))
        else:
            warnings.append("翻譯程式 llama-server 開不起來，翻譯會沒辦法用。請先更新顯示卡驅動再執行 repair.bat"
                            + versions_text(torch))
    if not results["nvenc"]["ok"] and results["ffmpeg"]["ok"]:
        log("NVENC not available; the app will use libx264 for compatible playback files")
    if dll:
        warnings.append(f"有元件缺少微軟 Visual C++ 執行階段。請到微軟官網下載 vc_redist.x64.exe 安裝"
                        f"（{VC_REDIST_URL}），裝好後執行 repair.bat")
    return {"ok": not fatal, "fatal": fatal, "warnings": warnings, "details": results, "at": time.time()}


# ---------- 主流程 ----------

def acquire_setup_lock(con: Console):
    """同時雙擊兩次 start.bat：第二個等第一個裝完。回傳要保持開著的檔案。"""
    import msvcrt
    SETUP_LOCK.parent.mkdir(parents=True, exist_ok=True)
    f = open(SETUP_LOCK, "a+b")
    told = False
    while True:
        try:
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
            return f
        except OSError:
            if not told:
                con.say("另一個視窗正在安裝執行環境，等它完成…")
                told = True
            time.sleep(2)


def fast_ok(state: dict, fp: dict, manifest: dict) -> bool:
    if state.get("state_version") != STATE_VERSION or state.get("root") != str(ROOT):
        return False
    if state.get("fingerprint") != fp or not (state.get("tests") or {}).get("ok"):
        return False
    paths = [VENV_PY, base_python(manifest), *(VENV / m for m in VENV_MARKERS)]
    for key in ("ffmpeg", "ffprobe", "llama_server"):
        p = (state.get(key) or {}).get("path")
        if not p:
            return False
        paths.append(ROOT / p)
    return all(p.is_file() for p in paths)


def processes_using(folder: Path) -> list[tuple[int, str]]:
    """執行檔在 folder 底下、還在跑的程序（不含自己）：伺服器、語音和下載子程序、llama-server、ffmpeg、
    另一個視窗正在跑的安裝（uv.exe、Python）。只看得到同一個使用者的程序（朋友的程式就是這樣跑的）。"""
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.OpenProcess.restype = wintypes.HANDLE
    k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    k32.QueryFullProcessImageNameW.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR,
                                               ctypes.POINTER(wintypes.DWORD)]
    k32.CloseHandle.argtypes = [wintypes.HANDLE]
    count = 4096
    while True:
        pids = (wintypes.DWORD * count)()
        needed = wintypes.DWORD()
        if not k32.K32EnumProcesses(pids, ctypes.sizeof(pids), ctypes.byref(needed)):
            return []
        if needed.value < ctypes.sizeof(pids):
            break
        count *= 2
    base = os.path.normcase(str(Path(folder).resolve())).rstrip("\\") + "\\"
    found = []
    for pid in pids[: needed.value // ctypes.sizeof(wintypes.DWORD)]:
        if pid in (0, os.getpid()):
            continue
        handle = k32.OpenProcess(0x1000, False, pid)        # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            continue
        try:
            buf = ctypes.create_unicode_buffer(32768)
            size = wintypes.DWORD(32768)
            if k32.QueryFullProcessImageNameW(handle, 0, buf, ctypes.byref(size)) and \
                    os.path.normcase(buf.value).startswith(base):
                found.append((pid, buf.value))
        finally:
            k32.CloseHandle(handle)
    return found


def repair_reset() -> int:
    """repair.bat：沒有程式在用 runtime 時，刪掉 venv 和 install-state.json。0 刪好了、2 還有程式在用（什麼都沒刪）。"""
    busy = processes_using(RT)
    if busy:
        log("repair refused, still running: " + "; ".join(f"{pid} {path}" for pid, path in busy))
        print("還在執行的程式：" + "、".join(sorted({Path(p).name for _, p in busy})))
        return 2
    try:
        if STATE_PATH.exists():
            STATE_PATH.unlink()          # 先刪這個：venv 就算只刪掉一半，下次啟動也會整個重新檢查
        if VENV.exists():
            remove_tree(VENV, RT)
    except (OSError, SetupError) as e:
        log(f"repair reset failed: {type(e).__name__}: {e}")
        print(f"有些檔案刪不掉（{e}）。")
        return 2
    log("repair: deleted runtime\\venv and install-state.json")
    return 0


def update_ytdlp_only(manifest: dict, con: Console) -> int:
    state = read_json(STATE_PATH)
    if not VENV_PY.is_file():
        con.say("執行環境還沒裝好，請先打開 start.bat")
        return 1
    info = ensure_ytdlp(manifest, con, update=True)
    if info.get("ok"):
        if state:
            state["yt_dlp"] = info
            state["deno"] = {"path": info.get("deno")}
            state["updated_at"] = time.time()
            write_json(STATE_PATH, state)
        con.say(f"YouTube 下載元件已經是 {info.get('version')}")
        return 0
    return 1


def plan_natives(todo: list, natives: dict, labels: dict, con: Console) -> list[str]:
    """背景下載前：完整包 offline 資料夾裡的檔案先核對 sha256、放到 runtime\\downloads（幾秒），
    畫面才說得準哪些從 offline 拿、哪些 offline 的檔案壞了、真的要下載多少。回傳要下載的項目（todo 的 key）。"""
    ready, broken = set(), []
    for k in todo:
        for f in natives[k]["files"]:
            if offline_file(f) is None:
                continue
            if _from_offline(f, DOWNLOADS / f["name"], con, "native", labels[k]):
                ready.add(f["name"])
            elif k not in broken:
                broken.append(k)
    downloading = [k for k in todo if any(f["name"] not in ready for f in natives[k]["files"])]
    offline = [k for k in todo if k not in downloading]
    if offline:
        con.say(f"使用 offline 資料夾裡的 {'、'.join(labels[k] for k in offline)}")
    for k in broken:
        con.say(f"offline 資料夾裡的 {labels[k]} 壞了，改從網路下載")
    if downloading:
        need = [f for k in downloading for f in natives[k]["files"] if f["name"] not in ready]
        total = sum(f.get("size") or 0 for f in need)
        left = sum(remaining_bytes(f, DOWNLOADS) for f in need)
        names = "、".join(labels[k] for k in downloading)
        if left < total:
            con.say(f"接著下載 {names}（還剩 {mb(left)}，共 {mb(total)}，在背景同時進行）")
        else:
            con.say(f"下載 {names}（{mb(total)}，在背景同時進行）")
    return downloading


def full_setup(manifest: dict, fp: dict, state: dict, con: Console) -> int:
    started = time.time()
    base = base_python(manifest)
    if not base.is_file():
        raise SetupError("找不到 runtime 裡的 Python，請重新打開 start.bat")
    first = not state and not VENV_PY.is_file()     # 第一次的說明 start.bat 已經印過（launcher\msg\first_run.txt）
    old_root = state.get("root")
    if old_root and old_root != str(ROOT):
        repair_moved(manifest, old_root, con)

    # 背景：下載 GitHub、PyPI 上的原生檔（GitHub 很慢，跟套件安裝同時進行）
    natives = {"msvc": manifest["msvc_runtime"], "ffmpeg": manifest["ffmpeg"], "llama": manifest["llama_cpp"]}
    msvc_ready = all((BIN / f"msvc-runtime-{natives['msvc']['version']}" / n).is_file() for n in natives["msvc"]["dlls"])
    labels = {"msvc": "VC++ 執行階段", "ffmpeg": "ffmpeg", "llama": "llama.cpp"}
    todo = [k for k in ("msvc", "ffmpeg", "llama")
            if not (msvc_ready if k == "msvc" else native_installed(natives[k]))]
    archives: dict[str, list] = {}
    bg_error: list = []

    def fetch_natives():
        try:
            for k in todo:
                archives[k] = [download(f, labels[k], con, key="native") for f in natives[k]["files"]]
        except BaseException as e:  # noqa: BLE001
            bg_error.append(e)

    bg = threading.Thread(target=fetch_natives, name="natives", daemon=True)
    downloading: list[str] = []
    if todo:
        downloading = plan_natives(todo, natives, labels, con)
        bg.start()

    created = ensure_venv(manifest, con)
    if created or state.get("fingerprint", {}).get("uv_lock") != fp["uv_lock"] or not state \
            or not all((VENV / m).is_file() for m in VENV_MARKERS if "torch" not in m):
        sync_packages(con)
    torch = ensure_torch(con)
    ytdlp = ensure_ytdlp(manifest, con)

    if todo:
        if bg.is_alive() and downloading:
            con.say(f"等 {'、'.join(labels[k] for k in downloading)} 下載完成…")
        bg.join()
        if bg_error:
            err = bg_error[0]
            raise err if isinstance(err, SetupError) else SetupError(network_message("ffmpeg、llama.cpp", err))

    msvc = msvc_dir(manifest, con, (archives.get("msvc") or [None])[0])
    ffmpeg_dir = native_installed(natives["ffmpeg"]) or install_native(natives["ffmpeg"], "ffmpeg", archives["ffmpeg"], con)
    llama_dir = native_installed(natives["llama"]) or install_native(natives["llama"], "llama.cpp", archives["llama"], con)
    dlls = manifest["msvc_runtime"]["dlls"]
    place_dlls(msvc, base.parent, dlls)
    place_dlls(msvc, llama_dir, dlls)

    paths = {"ffmpeg": ffmpeg_dir / natives["ffmpeg"]["exe"]["ffmpeg"],
             "ffprobe": ffmpeg_dir / natives["ffmpeg"]["exe"]["ffprobe"],
             "llama_server": llama_dir / natives["llama"]["exe"]["llama_server"]}
    moved_only = bool(old_root and old_root != str(ROOT) and not created and not todo
                      and state.get("fingerprint") == fp and (state.get("tests") or {}).get("ok"))
    tests = quick_check(paths, state, con) if moved_only else None
    if tests is None:
        gate = (os.environ.get("VS_SETUP_GATE") or "").strip()
        if gate:
            con.say("（測試模式）等測試程式確認顯示卡空著…")
            while not Path(gate).exists():
                time.sleep(1)
        con.say("測試顯示卡和各個元件…")
        tests = run_tests(paths, con)

    # 舊版本的 ffmpeg、llama.cpp 資料夾（manifest 換版本後留下的）
    for key, current in (("ffmpeg", ffmpeg_dir), ("llama_server", llama_dir)):
        old = (state.get(key) or {}).get("dir")
        if old and tests["ok"] and (ROOT / old).resolve() != current.resolve() and inside(ROOT / old, BIN):
            try:
                remove_tree(ROOT / old, BIN)
            except (OSError, SetupError) as e:
                log(f"could not remove old {old}: {e}")

    now = time.time()
    new_state = {
        "state_version": STATE_VERSION,
        "root": str(ROOT),
        "fingerprint": fp,
        "installed_at": state.get("installed_at") or now,
        "updated_at": now,
        "python": {"version": manifest["python"]["version"], "path": rel(base)},
        "venv": {"path": rel(VENV), "python": rel(VENV_PY)},
        "torch": torch,
        "yt_dlp": ytdlp,
        "msvc_runtime": {"version": manifest["msvc_runtime"]["version"], "dir": rel(msvc)},
        "ffmpeg": {"path": rel(paths["ffmpeg"]), "dir": rel(ffmpeg_dir), "version": natives["ffmpeg"]["version"]},
        "ffprobe": {"path": rel(paths["ffprobe"])},
        "llama_server": {"path": rel(paths["llama_server"]), "dir": rel(llama_dir), "build": natives["llama"]["build"]},
        "deno": {"path": ytdlp.get("deno")},
        "nvenc": bool(tests["details"]["nvenc"]["ok"]),
        "tests": tests,
    }
    write_json(STATE_PATH, new_state)
    for w in tests["warnings"]:
        con.say("注意：" + w)
    if not tests["ok"]:
        for f in tests["fatal"]:
            con.say("沒辦法啟動：" + f)
        con.say(f"詳細紀錄在 {LOG_PATH}")
        return 1
    if first or old_root != str(ROOT) or created:
        con.say(f"執行環境準備好了（花了 {duration(time.time() - started)}）。")
    log(f"setup finished in {time.time() - started:.1f}s")
    return 0


def main(argv: list) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    if os.name != "nt":
        print("這個程式只支援 Windows。")
        return 1
    if "--repair-reset" in argv:
        return repair_reset()
    update = "--update-ytdlp" in argv
    repair = "--repair" in argv or os.environ.get("VS_REPAIR") == "1"
    con = Console()
    try:
        manifest = load_manifest()
        fp = fingerprint(manifest)
        state = read_json(STATE_PATH)
        if not update and not repair and fast_ok(state, fp, manifest):
            if not ytdlp_retry_due(state):
                return 0
            # 其他都裝好了，只有 yt-dlp 上次沒裝成功：補裝一次，成不成功都照常啟動
            con.start()
            state["yt_dlp"] = ensure_ytdlp(manifest, con, quick=True)
            if state["yt_dlp"].get("ok"):
                state["deno"] = {"path": state["yt_dlp"].get("deno")}
            write_json(STATE_PATH, state)
            return 0
        RT.mkdir(parents=True, exist_ok=True)
        for d in (LOGS, DOWNLOADS, BIN, RT / "tmp"):
            d.mkdir(parents=True, exist_ok=True)
        con.start()
        lock = acquire_setup_lock(con)
        try:
            log(f"==== setup start: root={ROOT} update={update} repair={repair} argv={argv}")
            if update:
                return update_ytdlp_only(manifest, con)
            state = read_json(STATE_PATH)      # 等鎖的時候另一個視窗可能已經裝好了
            if not repair and fast_ok(state, fp, manifest):
                return 0
            if not UV.is_file():
                raise SetupError("找不到 runtime\\uv\\uv.exe，請重新打開 start.bat")
            return full_setup(manifest, fp, state, con)
        finally:
            lock.close()
    except SetupError as e:
        con.stop()
        log(f"SetupError: {e}")
        print()
        print(str(e))
        print(f"詳細紀錄在 {LOG_PATH}")
        return 1
    except KeyboardInterrupt:
        con.stop()
        print("\n已經中斷。重新打開 start.bat 會接著做。")
        return 1
    except Exception as e:  # noqa: BLE001
        con.stop()
        import traceback
        log("unexpected error:\n" + traceback.format_exc())
        print()
        known = os_problem(e)       # 解壓縮、寫檔時磁碟滿了、檔案被鎖住、沒有權限
        if known:
            print(known)
            print(f"詳細紀錄在 {LOG_PATH}")
            return 1
        print(f"安裝時發生沒預料到的錯誤：{type(e).__name__}: {e}")
        print(f"請把 {LOG_PATH} 傳給作者")
        return 1
    finally:
        con.stop()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
