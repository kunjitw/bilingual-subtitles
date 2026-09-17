"""重複啟動與 port。

單一實例鎖：data\\server.lock。只有會跑佇列的伺服器才鎖（VS_NO_WORKERS=1 的測試伺服器不鎖），
拿到鎖之後才會把上次沒跑完的任務放回佇列（jobs.start_workers → db.reset_interrupted），
所以同一個 data 資料夾同時啟動兩次，只有一個跑佇列，執行中的任務不會被另一個改回排隊。
拿到鎖的伺服器把實際的網址寫進 data\\server.json，第二個啟動的程式照這個檔打開瀏覽器後結束。

port：先自己綁好 socket 再交給 uvicorn（綁不到就還沒動到佇列）。
  被占用（WinError 10048）：是同一份程式（同一個 data 資料夾，看 /api/meta 的 data_id）就打開瀏覽器後結束；
  不是（別的程式，或另一份 Bilingual Subtitles，例如作者的開發版開著時試裝發佈版）就改試下一個（8766 到 8790）。
  被 Windows 保留（WinError 10013，開了 Hyper-V、WSL 常見）：也換下一個。
  使用者用 --port 明確指定時不換 port，直接說明原因後結束。
"""
import errno
import hashlib
import json
import os
import socket
import time
import urllib.request

from . import config

APP_ID = "video-subtitle"          # /api/meta 回傳的 app，用來認出「是自己的程式」。改名後沿用舊值，跟舊版才認得彼此
PORT_TRIES = 25                    # 預設 port 被占用時往後再試幾個
LOCK_NAME = "server.lock"
INFO_NAME = "server.json"

_lock_file = None


class StartupStop(Exception):
    """啟動到一半要結束：code 是結束代碼，url 不是 None 時代表要打開這個網址（已經在執行的那一個）。"""

    def __init__(self, message: str, code: int = 1, url: str | None = None):
        super().__init__(message)
        self.code = code
        self.url = url


# ---------- 單一實例鎖 ----------

def lock_path():
    return config.DATA_DIR / LOCK_NAME


def info_path():
    return config.DATA_DIR / INFO_NAME


def acquire_lock() -> bool:
    """拿到鎖回 True（已經拿著也是 True）；另一個程式拿著回 False。程式結束（包括當掉）時系統會自動放開。"""
    global _lock_file
    if _lock_file is not None:
        return True
    path = lock_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    f = open(path, "a+b")
    try:
        f.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        f.close()
        return False
    _lock_file = f
    return True


def lock_held() -> bool:
    return _lock_file is not None


def release_lock():
    global _lock_file
    f, _lock_file = _lock_file, None
    if f is None:
        return
    try:
        f.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
    except OSError:
        pass
    f.close()


def write_info(host: str, port: int, url: str):
    """拿著鎖的伺服器寫下自己的網址（覆寫舊的；結束時不刪，讀的人會確認對方還在不在）。"""
    data = {"pid": os.getpid(), "host": host, "port": port, "url": url, "started_at": time.time()}
    tmp = info_path().with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, info_path())


def read_info() -> dict:
    try:
        data = json.loads(info_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


# ---------- port ----------

def local_only(host: str) -> bool:
    """綁這個位址時只有這台電腦連得到（127.x、::1、localhost）。"""
    host = (host or "").strip().lower()
    return host.startswith("127.") or host in ("::1", "localhost")


def browser_url(host: str, port: int) -> str:
    """在這台電腦上開網頁用的網址。0.0.0.0、:: 都有聽 127.0.0.1（:: 綁成同時收 IPv4、IPv6，見 bind_socket）；
    ::1 只收 IPv6，要用 [::1]。"""
    host = (host or "").strip()
    if host in ("", "0.0.0.0", "::", "127.0.0.1", "localhost"):
        return f"http://127.0.0.1:{port}/"
    return f"http://[{host}]:{port}/" if ":" in host else f"http://{host}:{port}/"


def data_id() -> str:
    """這份程式的代號：data 資料夾路徑的雜湊（/api/meta 會回傳，不直接給路徑）。"""
    path = os.path.normcase(os.path.abspath(str(config.DATA_DIR)))
    return hashlib.sha256(path.encode("utf-8")).hexdigest()[:16]


def _meta(host: str, port: int, timeout: float) -> dict | None:
    url = browser_url(host, port) + "api/meta"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            meta = json.load(r)
    except (OSError, ValueError):
        return None
    return meta if isinstance(meta, dict) else None


def is_ours(host: str, port: int, timeout: float = 2.0) -> bool:
    """這個 port 上是不是 Bilingual Subtitles（打 /api/meta）。舊版沒有 app 欄位，用它特有的欄位認。"""
    meta = _meta(host, port, timeout)
    return meta is not None and (meta.get("app") == APP_ID or
                                 {"engines", "translators", "health_version"} <= set(meta))


def same_data(host: str, port: int, timeout: float = 2.0) -> bool:
    """port 上的 Bilingual Subtitles 是不是用同一個 data 資料夾（同一份程式）。舊版沒有 data_id，當成不是。"""
    meta = _meta(host, port, timeout)
    return meta is not None and meta.get("data_id") == data_id()


def wait_ready(host: str, port: int, timeout: float = 120) -> bool:
    """等到網頁真的連得上（啟動時要先清殘留程序、開資料庫，可能要幾秒）。"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if is_ours(host, port, timeout=1.0):
            return True
        time.sleep(0.3)
    return False


def wait_existing(timeout: float = 60) -> str | None:
    """另一個程式拿著鎖：照 data\\server.json 找到它的網址，等它連得上。它剛啟動、還沒寫檔時會一直重讀。"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        info = read_info()
        port = info.get("port")
        if isinstance(port, int) and is_ours(info.get("host") or "127.0.0.1", port, timeout=1.0):
            return info.get("url") or browser_url(info.get("host") or "127.0.0.1", port)
        time.sleep(0.5)
    return None


def _in_use(e: OSError) -> bool:
    return getattr(e, "winerror", None) == 10048 or e.errno == errno.EADDRINUSE


def _forbidden(e: OSError) -> bool:
    return getattr(e, "winerror", None) == 10013 or e.errno == errno.EACCES


def _exclusive_socket(family) -> socket.socket:
    s = socket.socket(family, socket.SOCK_STREAM)
    opt = getattr(socket, "SO_EXCLUSIVEADDRUSE", None)
    if opt is not None:
        # Windows：不讓別的程式在同一個 port 綁更精確的位址搶走連線
        s.setsockopt(socket.SOL_SOCKET, opt, 1)
    return s


def bind_socket(host: str, port: int) -> socket.socket:
    """綁好（還沒 listen）的 socket；綁不到丟 OSError。
    要綁 127.0.0.1 時先試 0.0.0.0：別的程式綁在 0.0.0.0 同一個 port 時，Windows 還是會讓我們綁 127.0.0.1，
    結果兩個程式同時在聽同一個 port，所以這種情況也要算被占用。"""
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    wildcard = "::" if family == socket.AF_INET6 else "0.0.0.0"
    if host != wildcard:
        probe = _exclusive_socket(family)
        try:
            probe.bind((wildcard, port))
        finally:
            probe.close()
    sock = _exclusive_socket(family)
    try:
        if host == "::":
            # Windows 的 IPv6 socket 預設只收 IPv6：:: 要讓 127.0.0.1 和區網的 IPv4 也連得到
            sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
        sock.bind((host, port))
    except OSError:
        sock.close()
        raise
    sock.set_inheritable(False)
    return sock


def open_port(host: str, port: int, explicit: bool) -> tuple[socket.socket, int]:
    """綁 port，回傳 (socket, 實際的 port)；要結束時丟 StartupStop。"""
    ports = [port] if explicit else [port + i for i in range(PORT_TRIES + 1)]
    for p in ports:
        try:
            return bind_socket(host, p), p
        except OSError as e:
            if _in_use(e):
                if is_ours(host, p):
                    url = browser_url(host, p)
                    if not same_data(host, p):
                        # 另一份 Bilingual Subtitles（別的資料夾）：不是這一份，打開它的網頁會搞混，當成被占用
                        if explicit:
                            raise StartupStop(f"port {p} 已經有另一份 Bilingual Subtitles（別的資料夾）在執行。"
                                              "請換一個 port，或拿掉 --port 讓程式自己找")
                        continue
                    if explicit:
                        raise StartupStop(f"port {p} 已經有 Bilingual Subtitles 在執行（{url}），這次不再啟動")
                    raise StartupStop(f"Bilingual Subtitles 已經在執行：{url}", code=0, url=url)
                if explicit:
                    raise StartupStop(f"port {p} 已經被其他程式占用。請換一個 port，或拿掉 --port 讓程式自己找")
                continue
            if _forbidden(e):
                if explicit:
                    raise StartupStop(f"port {p} 被 Windows 保留，不能使用（開了 Hyper-V 或 WSL 時常見）。請換一個 port")
                continue
            raise StartupStop(f"沒辦法在 {host}:{p} 開網頁伺服器：{e}。請確認 VS_HOST 設定的位址是這台電腦的")
    raise StartupStop(f"port {ports[0]} 到 {ports[-1]} 都被其他程式占用或被 Windows 保留。"
                      "請關掉占用的程式，或用 --port 指定別的 port")
