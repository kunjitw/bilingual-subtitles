"""下載模型的子程序（設定頁按「下載」後，佇列的 model 任務叫起來的）。

放在自己的程序裡跑，取消、暫停時直接結束這個程序，下載就真的停了。

怎麼下載（打包計畫 P0-10、6.6.4，D5 選 A）：
  - 一律走一般 HTTP（不用 Xet）。每個檔案先寫到 <檔名>.part，用 Range 從 .part 的大小接著下載，
    程式關掉重開、按重試、暫停後繼續，都從中斷的地方接著下載。
  - 每個檔案可以有好幾個網址（之後可以加鏡像），前一個連不上或找不到就換下一個。
  - 下載完核對大小；有 sha256 的也核對（邊下載邊算，接著下載時先把 .part 已有的部分算進去），對了才改成正式檔名。
  - 整個模型（有版本的模型是那個版本）都下載完，寫完成標記 .complete.json（config.COMPLETE_MARKER）。
  - Hugging Face 整個 repo 的模型（hf、onnx、parts）先用 huggingface_hub 的唯讀 API 查檔案清單、大小、sha256，
    記在 .download.json（固定 revision），續傳時照它下載，不會中途換成新版本的檔案。

用法：python -s -m app.model_download <型錄 id> [--variant 版本] [--parent-pid PID] [--work 工作資料夾]
  --parent-pid：伺服器的 PID。伺服器被直接關掉時這個程序跟著結束，不會留在背景繼續寫 .part。
  --work：任務的工作資料夾，只用來讓下次啟動時認出殘留的程序（jobs.find_orphans）。
進度：stdout 每 0.5 秒一行 JSON {"done": 位元組, "total": 位元組, "speed": 每秒位元組, "file": 檔名}；
     正在核對已下載的部分時多一個 "checking": true。
失敗時 stdout 最後一行是 {"error": 代碼, "message": 說明, "retry_after": 秒數}，
stderr 最後一行是 "ERROR <代碼> <說明>"。代碼見 ERROR_TEXT，任務用它決定要不要自動重試。
"""
import argparse
import errno
import hashlib
import http.client
import json
import os
import shutil
import socket
import ssl
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import deque
from pathlib import Path

from . import config, safepath
from .config import COMPLETE_MARKER, DOWNLOAD_MARKER, MODEL_CATALOG, PART_SUFFIX

MIB = 1024 * 1024
GIB = 1024 * MIB
CHUNK = MIB
TIMEOUT = int(os.environ.get("HF_HUB_DOWNLOAD_TIMEOUT") or 60)
USER_AGENT = "video-subtitle"
DISK_MARGIN = 64 * MIB          # 子程序最後一道檢查；任務開始前 models.check_disk 已經要求多留 1 GB
QUICK_WAITS = (2, 4, 6, 10, 15)  # 同一次下載中連線斷掉、但有收到資料時，隔幾秒馬上重連（最多幾次）

# 錯誤代碼和給使用者看的說明（打包計畫 6.5.6）
# 自動重試用完、任務已經失敗時也會原樣顯示，所以不寫「會自動重試」這類之後才會發生的事；
# 自動重試等待中的倒數文字只取第一句（jobs._wait_before_retry）
ERROR_TEXT = {
    "network": "連不到{source}。已下載的部分會保留，網路恢復後按「重試」。",
    "timeout": "網路太慢或中斷，已下載 {done} / {total}，按「重試」會接著下載。",
    "ssl": "連線被攔截（常見於防毒軟體的 HTTPS 掃描或公司網路）。換個網路，或暫時關掉網頁掃描再試。",
    "rate_limit": "下載次數太多被暫時限制。過幾分鐘再按「重試」。",
    "not_found": "模型檔在{source}上找不到（可能改名或下架了），請更新程式。",
    "gated": "這個模型要登入 Hugging Face 同意授權才能下載。",
    "disk_full": "磁碟空間不夠：還需要 {need}，目前只剩 {free}。清出空間後按「重試」。",
    "no_permission": "沒辦法寫入 models 資料夾，請把程式資料夾搬到 D:\\BilingualSubtitles 這類位置。",
    "file_locked": "檔案被其他程式鎖住（多半是防毒軟體在掃描）。等一下再按「重試」。",
    "corrupt": "下載的檔案內容不對（sha256 核對不符），已經刪掉。按「重試」會重新下載。",
    "changed": "下載來源上的檔案跟程式記錄的不一樣（可能更新過），請更新程式。",
    "github": "連不到下載來源，歌曲用的人聲分離模型下載不了，其他功能不受影響。",
    "unknown": "下載失敗：{detail}",
}
TIMEOUT_NO_SIZE = "網路太慢或中斷，連不上{source}。已下載的部分會保留，按「重試」會接著下載。"


def fmt_size(num_bytes: int) -> str:
    """跟 models.fmt_size 一樣：1024 進位，1 GB 以上一位小數，以下用 MB。"""
    if num_bytes >= GIB - MIB // 2:
        return f"{num_bytes / GIB:.1f} GB"
    if num_bytes <= 0:
        return "0 MB"
    return f"{max(1, round(num_bytes / MIB))} MB"


def error_text(code: str, template: str | None = None, **fields) -> str:
    """ERROR_TEXT 填上細節。來源是英文名稱（Hugging Face）時前後加空格，中文（下載來源）不加。
    timeout 還不知道下載了多少時（例如查檔案清單就逾時），不顯示「? / ?」。"""
    values = {"source": "Hugging Face", "done": "?", "total": "?", "need": "?", "free": "?", "detail": code, **fields}
    if values["source"].isascii():
        values["source"] = f" {values['source']} "
    if template is None and code == "timeout" and "?" in (values["done"], values["total"]):
        template = TIMEOUT_NO_SIZE
    text = (template or ERROR_TEXT.get(code, ERROR_TEXT["unknown"])).format(**values)
    return text.replace(" 。", "。").replace(" ，", "，")


class DownloadError(Exception):
    """下載失敗。code 是 ERROR_TEXT 的代碼，訊息可以直接給使用者看。"""

    def __init__(self, code: str, message: str | None = None, retry_after: int | None = None, **fields):
        super().__init__(message or error_text(code, **fields))
        self.code = code
        self.retry_after = retry_after


class Truncated(OSError):
    """連線提早結束，收到的比伺服器說的少。"""


# ---------- 進度 ----------

class Progress:
    """整個模型的下載進度：base 是已經完成的檔案，current 是正在下載的檔案目前的大小。"""

    def __init__(self, total: int, out=None, interval: float = 0.5):
        self.total = total
        self.base = 0
        self.current = 0
        self.file = ""
        self.checking = False
        self.out = out or sys.stdout
        self.interval = interval
        self._last = 0.0
        self._samples: deque = deque()

    @property
    def done(self) -> int:
        return self.base + self.current

    def start_file(self, name: str, have: int):
        self.file, self.current = name, have
        self._samples.clear()           # 已經在磁碟上的部分不算進速度
        self.emit(force=True)

    def add(self, n: int):
        self.current += n
        self.emit()

    def finish_file(self):
        self.base += self.current
        self.current = 0

    def speed(self) -> float:
        if len(self._samples) < 2:
            return 0.0
        (t0, d0), (t1, d1) = self._samples[0], self._samples[-1]
        return (d1 - d0) / (t1 - t0) if t1 > t0 else 0.0

    def emit(self, force: bool = False):
        now = time.monotonic()
        self._samples.append((now, self.done))
        while len(self._samples) > 2 and now - self._samples[0][0] > 5:
            self._samples.popleft()
        if not force and now - self._last < self.interval:
            return
        self._last = now
        line = {"done": self.done, "total": self.total, "speed": round(self.speed()), "file": self.file}
        if self.checking:
            line["checking"] = True
        try:
            print(json.dumps(line, ensure_ascii=False), file=self.out, flush=True)
        except (OSError, ValueError):
            pass


class _NullProgress(Progress):
    def __init__(self):
        super().__init__(0)

    def emit(self, force: bool = False):
        pass


# ---------- 錯誤分類 ----------

def _source(url: str | None) -> str:
    host = (urllib.parse.urlsplit(url or "").hostname or "").lower()
    hf_host = (urllib.parse.urlsplit(os.environ.get("HF_ENDPOINT") or "").hostname or "").lower()
    hf = host.endswith(("huggingface.co", "hf.co")) or not host or (hf_host and host == hf_host)
    return "Hugging Face" if hf else "下載來源"


def _retry_after(headers) -> int | None:
    try:
        value = (headers or {}).get("Retry-After")
        return max(1, int(float(value))) if value else None
    except (TypeError, ValueError):
        return None


def _server_error_text(source: str, status: int) -> str:
    name = f"{source} " if source.isascii() else source
    return f"{name}暫時有問題（HTTP {status}）。已下載的部分會保留，等一下按「重試」。"


HF_NOT_FOUND_CODES = ("RepoNotFound", "EntryNotFound", "RevisionNotFound")


def http_error(status: int, headers=None, url: str | None = None) -> DownloadError:
    """HTTP 錯誤碼 → 錯誤代碼。Hugging Face 用 X-Error-Code 標頭說明原因：GatedRepo 才是要登入同意授權；
    repo 不存在（改名、下架、改成私人）回的是 401 而不是 404，也沒有 X-Error-Code（2026-09-17 實測）。"""
    source = _source(url)
    try:
        hf_code = ((headers or {}).get("X-Error-Code") or "").strip()
    except AttributeError:
        hf_code = ""
    if hf_code == "GatedRepo":
        return DownloadError("gated")
    if status == 404 or hf_code in HF_NOT_FOUND_CODES:
        return DownloadError("not_found", source=source)
    if status == 401 and source == "Hugging Face":
        return DownloadError("not_found", source=source)
    if status == 429:
        return DownloadError("rate_limit", retry_after=_retry_after(headers) or 60)
    if status in (401, 403):
        # GitHub、鏡像拒絕（常見是暫時被限流）：跟伺服器錯誤一樣自動重試，不叫使用者去登入 Hugging Face
        name = f"{source} " if source.isascii() else source
        return DownloadError("network", f"{name}拒絕下載（HTTP {status}）。已下載的部分會保留，等一下按「重試」；"
                                        "一直這樣的話請更新程式。")
    return DownloadError("network", _server_error_text(source, status))


def _open_error(path) -> int | None:
    """試著用「可讀寫、不擋別人」的方式打開 path，回傳 Windows 錯誤碼：32 被別的程式鎖住（現在打得開也算，
    代表剛才只是暫時被鎖住）、5 沒有權限（包括檔案不存在、在資料夾裡建不了檔案）。不是 Windows 或沒有路徑回 None。"""
    if os.name != "nt" or not path:
        return None
    import ctypes
    from ctypes import wintypes
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateFileW.restype = wintypes.HANDLE
    k32.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD,
                                wintypes.DWORD, wintypes.HANDLE]
    k32.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = k32.CreateFileW(str(path), 0x80000000 | 0x40000000, 0x1 | 0x2 | 0x4, None, 3, 0x80, None)
    if handle in (None, wintypes.HANDLE(-1).value):
        err = ctypes.get_last_error()
        return 32 if err in (32, 33) else 5
    k32.CloseHandle(handle)
    return 32


def classify(exc: BaseException, url: str | None = None, progress: Progress | None = None) -> DownloadError:
    """把各種例外換成 DownloadError（urllib、huggingface_hub 和 requests 的、寫檔的 OSError）。"""
    if isinstance(exc, DownloadError):
        return exc
    if isinstance(exc, urllib.error.HTTPError):
        return http_error(exc.code, exc.headers, url)
    # huggingface_hub 的例外（查檔案清單時）：GatedRepoError 是 RepositoryNotFoundError 的子類別，要先認
    hub = type(exc).__name__
    if hub == "GatedRepoError":
        return DownloadError("gated")
    if hub in ("RepositoryNotFoundError", "RevisionNotFoundError", "EntryNotFoundError", "RemoteEntryNotFoundError"):
        return DownloadError("not_found", source=_source(url))
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    if isinstance(status, int) and status >= 400:
        return http_error(status, getattr(response, "headers", None), url)
    reason = exc.reason if isinstance(exc, urllib.error.URLError) else exc
    chain, cur = [], exc
    while cur is not None and len(chain) < 8:
        chain.append(cur)
        cur = cur.__cause__ or cur.__context__
    names = " ".join(type(e).__name__ for e in chain + [reason])
    done = fmt_size(progress.done) if progress else "?"
    total = fmt_size(progress.total) if progress and progress.total else "?"
    if isinstance(reason, ssl.SSLError) or "SSLError" in names or "CertificateError" in names:
        return DownloadError("ssl")
    if isinstance(reason, (socket.timeout, TimeoutError)) or "Timeout" in names:
        return DownloadError("timeout", done=done, total=total, source=_source(url))
    if isinstance(exc, OSError) and not isinstance(exc, (urllib.error.URLError, ConnectionError, Truncated)):
        winerror = getattr(exc, "winerror", None)
        if winerror is None and isinstance(exc, PermissionError):
            # Python 的 open() 把「被別的程式開著」（ERROR_SHARING_VIOLATION）也變成 errno 13、沒有 winerror，
            # 直接問 Windows 這個檔案現在打不打得開，才分得出是被鎖住還是真的沒有權限
            winerror = _open_error(exc.filename)
        if winerror in (32, 33):
            return DownloadError("file_locked")
        if winerror in (39, 112) or exc.errno == errno.ENOSPC:
            return DownloadError("disk_full", "磁碟空間不夠，下載到一半寫不進去。清出空間後按「重試」。")
        if winerror == 5 or isinstance(exc, PermissionError) or exc.errno == errno.EACCES:
            return DownloadError("no_permission")
    if isinstance(exc, (urllib.error.URLError, ConnectionError, http.client.HTTPException, Truncated)) \
            or "ConnectionError" in names or "ProtocolError" in names:
        return DownloadError("network", source=_source(url))
    detail = " ".join(f"{type(exc).__name__}: {exc}".split())[:300]
    return DownloadError("unknown", detail=detail)


# ---------- 單一檔案 ----------

def _part_of(dst: Path) -> Path:
    return dst.with_name(dst.name + PART_SUFFIX)


def _truncate(path: Path):
    with open(path, "wb"):
        pass


def _discard(path: Path):
    """刪掉壞掉的暫存檔（走 safepath，只刪模型資料夾裡的）；不能刪就清空，下次從頭下載。"""
    if path.exists() and not safepath.safe_unlink(path, path.parent):
        _truncate(path)


def _hash_file(path: Path, upto: int, hasher, progress: Progress):
    """接著下載前，把 .part 已經有的部分算進 sha256。"""
    progress.checking = True
    progress.emit(force=True)
    left = upto
    with open(path, "rb") as f:
        while left > 0:
            chunk = f.read(min(8 * MIB, left))
            if not chunk:
                break
            hasher.update(chunk)
            left -= len(chunk)
            progress.emit()
    progress.checking = False


def _content_range(value: str | None) -> tuple[int | None, int | None]:
    """"bytes 100-199/1000" → (100, 1000)。"""
    try:
        unit, rest = (value or "").split(" ", 1)
        span, total = rest.split("/", 1)
        return int(span.split("-", 1)[0]), (None if total.strip() == "*" else int(total))
    except (ValueError, AttributeError):
        return None, None


def _stream(url: str, part: Path, have: int, size: int | None, state: dict, progress: Progress) -> int:
    """從 have 的位置下載到 .part 的尾巴。回傳下載後 .part 的大小。伺服器不支援續傳時從頭下載。
    state["hasher"] 是要跟著更新的 sha256（沒有要核對時 None），從頭下載時換成新的。"""
    headers = {"User-Agent": USER_AGENT, "Accept-Encoding": "identity"}
    if have:
        headers["Range"] = f"bytes={have}-"
    try:
        r = urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=TIMEOUT)
    except urllib.error.HTTPError as e:
        if e.code == 416 and have:
            # 範圍超出遠端檔案：.part 比遠端的檔案還大（或遠端換了檔案），從頭下載
            _truncate(part)
            raise _Restart()
        raise
    with r:
        start, remote_total = 0, None
        if r.status == 206:
            start, remote_total = _content_range(r.headers.get("Content-Range"))
            if start != have:
                _truncate(part)
                raise _Restart()
        elif r.status == 200:
            length = r.headers.get("Content-Length")
            remote_total = int(length) if length and length.isdigit() else None
            if have:
                # 伺服器不理 Range：從頭下載
                _truncate(part)
                have = 0
                if state.get("hasher") is not None:
                    state["hasher"] = hashlib.sha256()
                progress.start_file(progress.file, 0)
        else:
            raise DownloadError("network", _server_error_text(_source(url), r.status))
        if size is not None and remote_total is not None and remote_total != size:
            raise DownloadError("changed")
        length = r.headers.get("Content-Length")
        expected = int(length) if length and length.isdigit() else None
        got = 0
        hasher = state.get("hasher")
        with open(part, "ab" if have else "wb") as f:
            while True:
                chunk = r.read(CHUNK)
                if not chunk:
                    break
                f.write(chunk)
                if hasher is not None:
                    hasher.update(chunk)
                got += len(chunk)
                progress.add(len(chunk))
        if expected is not None and got != expected:
            raise Truncated(f"只收到 {got:,} / {expected:,} bytes，連線提早中斷")
        return have + got


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def remote_sha256(url: str | None) -> str | None:
    """遠端檔案現在的 sha256：Hugging Face 的 resolve 網址在轉址前帶 X-Linked-ETag（LFS 檔就是 sha256）。
    拿不到（不是 Hugging Face、連不上、不是 LFS 檔）回 None。只在下載完 sha256 對不上時用。"""
    if not url:
        return None
    opener = urllib.request.build_opener(_NoRedirect)
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": USER_AGENT})
    try:
        with opener.open(req, timeout=30) as r:
            headers = r.headers
    except urllib.error.HTTPError as e:          # 不跟著轉址時，302 會變成 HTTPError，標頭還在
        headers = e.headers
    except (OSError, ValueError, http.client.HTTPException):
        return None
    if headers is None:
        return None
    for key in ("X-Linked-ETag", "ETag"):
        value = (headers.get(key) or "").strip()
        value = value[2:] if value.startswith("W/") else value
        value = value.strip('"').lower()
        if len(value) == 64 and all(c in "0123456789abcdef" for c in value):
            return value
    return None


class _Restart(Exception):
    """.part 跟遠端對不上，已經清空，要從頭下載。"""


def _download_url(url: str, part: Path, size: int | None, want_sha: str | None, progress: Progress):
    """把一個網址下載到 .part（接著已有的部分），回傳 sha256 物件（沒有要核對時 None）。"""
    quick = 0
    restarts = 0
    state = {"hasher": None}
    hashed = -1                 # sha256 已經算到 .part 的哪裡（-1：要重算）
    while True:
        have = part.stat().st_size if part.exists() else 0
        if size is not None and have > size:
            _truncate(part)
            have = 0
        if want_sha and hashed != have:
            state["hasher"] = hashlib.sha256()
            if have:
                # 核對已下載的部分時，進度照樣顯示已經下載的量（不要先歸零，看起來像從頭下載）
                progress.current = have
                _hash_file(part, have, state["hasher"], progress)
            hashed = have
        progress.start_file(progress.file, have)
        if size is not None and have == size:
            return state["hasher"]
        before = have
        try:
            after = _stream(url, part, have, size, state, progress)
            if size is None or after == size:
                return state["hasher"]
            raise Truncated(f"只收到 {after:,} / {size:,} bytes")
        except _Restart:
            restarts += 1
            hashed = -1
            if restarts > 2:
                raise DownloadError("changed")
            continue
        except DownloadError:
            raise
        except Exception as e:  # noqa: BLE001
            err = classify(e, url, progress)
            now = part.stat().st_size if part.exists() else 0
            # 寫進 .part 的每一段都已經算進 sha256（progress.current 跟著一起加）：對得上就不用重算
            hashed = now if (want_sha and progress.current == now) else -1
            if err.code in ("network", "timeout") and now > before and quick < len(QUICK_WAITS):
                time.sleep(QUICK_WAITS[quick])
                quick += 1
                continue
            raise err from e


def fetch_file(urls: list[str], dst: Path, size: int | None = None, sha256: str | None = None,
               progress: Progress | None = None) -> dict:
    """下載一個檔案到 dst：先寫 dst.part，核對大小和 sha256，對了才改名。已經有正式檔名、大小也對就跳過。
    回傳 {"size": 位元組, "sha256": 核對過的 sha256 或 None}。"""
    progress = progress or _NullProgress()
    part = _part_of(dst)
    if dst.is_file() and not part.exists() and (size is None or dst.stat().st_size == size):
        progress.start_file(dst.name, dst.stat().st_size)
        progress.finish_file()
        return {"size": dst.stat().st_size, "sha256": None}
    dst.parent.mkdir(parents=True, exist_ok=True)
    progress.file = dst.name
    last = None
    used = urls[-1] if urls else None
    for i, url in enumerate(urls):
        try:
            hasher = _download_url(url, part, size, sha256, progress)
            used = url
            break
        except DownloadError as e:
            last = e
            # 這個網址找不到、要登入、連不上：還有別的網址（鏡像）就換下一個
            if e.code in ("not_found", "gated", "network", "timeout", "ssl") and i + 1 < len(urls):
                continue
            raise
    else:
        raise last or DownloadError("not_found")
    got = part.stat().st_size
    if size is not None and got != size:
        raise DownloadError("network", source=_source(used))
    digest = hasher.hexdigest() if hasher is not None else None
    if sha256 and digest != sha256.lower():
        _discard(part)
        # 大小對、內容不對：遠端換成同樣大小的新檔案時，重下幾次都一樣，要說「請更新程式」而不是「重試」
        remote = remote_sha256(used)
        if remote and remote != sha256.lower():
            raise DownloadError("changed")
        raise DownloadError("corrupt")
    if got == 0:
        _discard(part)
        raise DownloadError("corrupt", "收到的檔案是空的，已經刪掉。按「重試」會重新下載。")
    os.replace(part, dst)
    progress.finish_file()
    return {"size": got, "sha256": digest}


def fetch(urls: list[str], dst: Path):
    """舊的介面：下載一個檔案，失敗丟 RuntimeError（訊息可以給使用者看）。"""
    try:
        fetch_file(urls, dst)
    except DownloadError as e:
        raise RuntimeError(f"{dst.name} 下載失敗：{e}") from e


# ---------- 下載計畫 ----------

def hf_url(repo: str, revision: str, path: str) -> str:
    endpoint = (os.environ.get("HF_ENDPOINT") or "https://huggingface.co").rstrip("/")
    return f"{endpoint}/{repo}/resolve/{urllib.parse.quote(revision, safe='')}/{urllib.parse.quote(path, safe='/')}"


def list_repo(repo: str, allow) -> tuple[str, list[dict]]:
    """Hugging Face repo 的檔案清單（唯讀 API）：(commit sha, [{"path", "size", "sha256"}])，照 allow 過濾。"""
    os.environ["HF_HUB_OFFLINE"] = "0"
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    from huggingface_hub import HfApi
    from huggingface_hub.utils import filter_repo_objects
    info = HfApi().model_info(repo, files_metadata=True, timeout=TIMEOUT)
    keep = set(filter_repo_objects([s.rfilename for s in info.siblings], allow_patterns=allow))
    files = []
    for s in info.siblings:
        if s.rfilename not in keep:
            continue
        lfs = s.lfs
        sha = (lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)) if lfs else None
        files.append({"path": s.rfilename, "size": s.size, "sha256": sha})
    return info.sha, files


def _write_json(path: Path, data: dict):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def build_plan(mid: str, variant: str | None = None) -> dict:
    """要下載哪些檔案：{"dir", "items": [{"rel", "urls", "size", "sha256"}], "support", "record", "pending"}。
    record 會寫進完成標記；pending 是多檔模型要寫的 .download.json（續傳時沿用同一個 revision 和檔案清單）。"""
    entry = MODEL_CATALOG[mid]
    view = config.variant_view(entry, variant)
    d = Path(entry["dir"])
    kind = entry["kind"]
    if kind == "gguf":
        items = [{"rel": view["file"], "urls": list(view.get("urls") or [hf_url(view["repo"], "main", view["file"])]),
                  "size": view.get("bytes"), "sha256": view.get("sha256")}]
        return {"dir": d, "items": items, "support": [], "record": {"repo": view["repo"]}, "pending": None}
    if kind == "ckpt":
        names = entry.get("files") or [entry["file"]]
        items = [{"rel": n, "urls": list(entry["urls"][n]), "size": (entry.get("file_bytes") or {}).get(n),
                  "sha256": (entry.get("file_sha256") or {}).get(n)} for n in names]
        support = [{"rel": n, "urls": list(u)} for n, u in (entry.get("support") or {}).items()]
        return {"dir": d, "items": items, "support": support, "record": {}, "pending": None}
    parts = entry.get("parts") or [{"repo": entry["repo"], "sub": ""}]
    allow = entry.get("allow")
    saved = config._read_json(d / DOWNLOAD_MARKER)
    ok = (saved.get("model") == mid and saved.get("allow") == allow
          and [(p.get("repo"), p.get("sub")) for p in saved.get("parts") or []] == [(p["repo"], p["sub"]) for p in parts])
    if not ok:
        listed = []
        for p in parts:
            sha, files = list_repo(p["repo"], allow)
            listed.append({"repo": p["repo"], "sub": p["sub"], "revision": sha, "files": files})
        saved = {"version": 1, "model": mid, "allow": allow, "started_at": time.time(), "parts": listed}
    items = []
    for p in saved["parts"]:
        for f in p["files"]:
            rel = f"{p['sub']}/{f['path']}" if p["sub"] else f["path"]
            items.append({"rel": rel, "urls": [hf_url(p["repo"], p["revision"], f["path"])],
                          "size": f.get("size"), "sha256": f.get("sha256")})
    record = {"revisions": {p["sub"] or ".": {"repo": p["repo"], "revision": p["revision"]} for p in saved["parts"]}}
    return {"dir": d, "items": items, "support": [], "record": record, "pending": saved}


def _check_space(d: Path, need: int):
    probe = d
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    free = shutil.disk_usage(probe).free
    if free < need + DISK_MARGIN:
        raise DownloadError("disk_full", need=fmt_size(need + DISK_MARGIN), free=fmt_size(free))


def write_complete(entry: dict, variant: str | None, record: dict):
    """寫完成標記：同一個檔案裡其他模型、其他版本的紀錄保留。"""
    path = Path(entry["dir"]) / COMPLETE_MARKER
    data = config._read_json(path)
    data["version"] = 1
    models = data.setdefault("models", {})
    key = entry.get("id") or entry.get("label")
    if variant:
        slot = models.setdefault(key, {})
        slot.setdefault("variants", {})[variant] = record
    else:
        models[key] = record
    _write_json(path, data)


def run(mid: str, variant: str | None = None, progress_out=None) -> dict:
    entry = MODEL_CATALOG[mid]
    variant = config.check_variant(entry, variant)
    plan = build_plan(mid, variant)
    d = plan["dir"]
    d.mkdir(parents=True, exist_ok=True)
    if plan["pending"] is not None and config._read_json(d / DOWNLOAD_MARKER) != plan["pending"]:
        _write_json(d / DOWNLOAD_MARKER, plan["pending"])
    items = plan["items"]
    total = sum(i["size"] or 0 for i in items)
    progress = Progress(total, out=progress_out)
    need = 0
    for it in items:
        dst = d / it["rel"]
        if it["size"] is None or (dst.is_file() and dst.stat().st_size == it["size"]):
            continue
        part = _part_of(dst)
        need += max(0, it["size"] - (part.stat().st_size if part.exists() else 0))
    _check_space(d, need)
    files, hashes = {}, {}
    for it in items:
        try:
            got = fetch_file(it["urls"], d / it["rel"], it["size"], it["sha256"], progress)
        except DownloadError as e:
            if e.code == "network" and entry["kind"] == "ckpt":
                raise DownloadError("github") from e
            raise
        files[it["rel"]] = got["size"]
        if got["sha256"]:
            hashes[it["rel"]] = got["sha256"]
        if entry["kind"] == "gguf":
            # 舊版用 huggingface_hub 下載到一半留下的暫存檔（.cache\huggingface\download\*.incomplete），現在用不到了
            for p in config.hf_legacy_files(d, it["rel"]):
                if p.name.endswith(".incomplete"):
                    _discard(p)
    # 套件自己的小檔案（例如 audio-separator 的模型清單）：先抓好，第一次用的時候就不用連網
    for it in plan["support"]:
        if not (d / it["rel"]).is_file():
            try:
                fetch_file(it["urls"], d / it["rel"])
            except DownloadError as e:
                raise DownloadError("github" if e.code == "network" else e.code, None if e.code == "network" else str(e)) from e
    progress.emit(force=True)
    record = {**plan["record"], "files": files, "sha256": hashes, "completed_at": round(time.time())}
    write_complete(entry, variant, record)
    if plan["pending"] is not None:
        _discard(d / DOWNLOAD_MARKER)
    return record


# ---------- 伺服器被關掉時跟著結束 ----------

def watch_parent(pid: int | None):
    """伺服器（pid）結束時這個程序也馬上結束：關掉黑色視窗後不會留在背景繼續下載、佔著 .part。"""
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


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--variant")
    ap.add_argument("--parent-pid", type=int)
    ap.add_argument("--work")
    args = ap.parse_args(argv)
    watch_parent(args.parent_pid)
    try:
        if args.model not in MODEL_CATALOG:
            raise DownloadError("unknown", detail=f"沒有 {args.model} 這個模型")
        run(args.model, args.variant)
    except BaseException as e:  # noqa: BLE001
        if isinstance(e, SystemExit):
            raise
        err = e if isinstance(e, DownloadError) else (
            DownloadError("unknown", detail=str(e)) if isinstance(e, ValueError) else classify(e))
        msg = " ".join(str(err).split())[:500]
        print(json.dumps({"error": err.code, "message": msg, "retry_after": err.retry_after}, ensure_ascii=False),
              flush=True)
        print(f"ERROR {err.code} {msg}", file=sys.stderr, flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
