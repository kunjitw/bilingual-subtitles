"""刪檔前的安全檢查，以及給 yt-dlp 的網址檢查。

專案裡所有刪除檔案、資料夾的地方都走這裡：
  safe_unlink(path, root)   刪一個檔案
  safe_rmtree(path, root)   刪一整個資料夾

root 是程式自己寫死的資料夾（例如 SUBS_DIR、MEDIA_DIR、WORK_DIR），path 可能是從資料庫或網址參數組出來的。
刪之前會檢查：
  1. root 本身必須是專案的資料夾（data/media、data/work、data/subs、data/thumbs、data/proxy、data/dict、models）
     或在它們底下。這些位置照 app/config.py 的設定（VS_DATA_DIR、VS_MODELS_DIR 改過就跟著改）。
     單元測試要刪暫存資料夾裡的東西，要先把那個資料夾加進 TEST_ROOTS（只認系統暫存資料夾底下的子資料夾），測完拿掉。
  2. path 解析 .. 和捷徑之後，一定要在 root 裡面，而且不能就是 root 本身。
  3. 要刪的那一項本身不能是捷徑（symlink 或 junction）。
不符合就不刪，寫一筆 log，回傳 False。
"""
import ipaddress
import logging
import os
import re
import shutil
import socket
import tempfile
import unicodedata
from pathlib import Path
from urllib.parse import unquote, urlsplit

log = logging.getLogger("safepath")

# 專案產生的 id：uuid4 的前 12 碼（db.new_id）
ID_RE = re.compile(r"[0-9a-f]{12}")
# 拿來當檔名的 id（字幕軌、任務）：只允許英數字、底線、連字號，擋掉 .. \ / : 這類能跳出資料夾的字元
NAME_RE = re.compile(r"[0-9A-Za-z_-]{1,64}")
# Windows 的裝置名稱，當檔名會變成開裝置而不是開檔案
_DEVICE_NAMES = {"con", "prn", "aux", "nul", *(f"com{i}" for i in range(10)), *(f"lpt{i}" for i in range(10))}


def is_id(value) -> bool:
    """是不是專案產生的 12 碼 id。網址參數裡的 id 在查資料庫、組路徑之前先用這個擋。"""
    return isinstance(value, str) and ID_RE.fullmatch(value) is not None


def is_safe_name(value) -> bool:
    """能不能直接拿來當檔名的一部分（不會跳出資料夾、不是裝置名稱）。"""
    return (isinstance(value, str) and NAME_RE.fullmatch(value) is not None
            and value.lower() not in _DEVICE_NAMES)


def require_safe_name(value, what: str = "id") -> str:
    if not is_safe_name(value):
        raise ValueError(f"不正確的{what}：{value!r}")
    return value


# ---------- 路徑 ----------

def _resolve(p) -> Path | None:
    try:
        return Path(p).resolve()
    except (OSError, RuntimeError, ValueError, TypeError):
        return None


def _key(p: Path) -> str:
    return os.path.normcase(os.path.normpath(str(p)))


def _within(target: Path, base: Path, allow_equal: bool = False) -> bool:
    """兩個都已經 resolve 過；target 在 base 裡面（allow_equal 時也可以就是 base）。不分大小寫。"""
    t, b = _key(target), _key(base)
    try:
        if os.path.commonpath([t, b]) != b:
            return False
    except ValueError:  # 不同磁碟
        return False
    return allow_equal or t != b


def is_link(p) -> bool:
    """是不是 symlink 或 junction。讀不到屬性時當成是，寧可不刪。"""
    p = Path(p)
    try:
        return p.is_symlink() or p.is_junction()
    except OSError:
        return True


def inside(path, root) -> bool:
    """path 解析 .. 和捷徑之後是不是在 root 裡面（不含 root 本身）。"""
    target, base = _resolve(path), _resolve(root)
    return bool(target and base and _within(target, base))


def _allowed_bases() -> list[Path]:
    from . import config  # 放在函式裡：tools 或子程序只用到這個模組時不用先載入設定
    bases = [config.MEDIA_DIR, config.WORK_DIR, config.SUBS_DIR, config.THUMB_DIR, config.PROXY_DIR,
             config.DICT_DIR, config.MODELS_DIR]
    return [b for b in (_resolve(x) for x in bases) if b]


# 單元測試明確註冊的根目錄（測試的 setup 加入、teardown 拿掉）。
# 只有在系統暫存資料夾底下的子資料夾才算數，不小心加了別的位置（或暫存資料夾本身）也不會開放。
TEST_ROOTS: list = []


def _test_roots() -> list[Path]:
    if not TEST_ROOTS:
        return []
    temp = _resolve(tempfile.gettempdir())
    if not temp:
        return []
    return [r for r in (_resolve(x) for x in list(TEST_ROOTS)) if r and _within(r, temp)]


def _root_ok(base: Path) -> bool:
    if any(_within(base, b, allow_equal=True) for b in _allowed_bases()):
        return True
    return any(_within(base, r, allow_equal=True) for r in _test_roots())


def _checked(path, root, kind: str) -> Path | None:
    """回傳可以刪的那一項（上層資料夾已解析，最後一段保持原樣，才看得出它本身是不是捷徑）；不能刪就回 None。"""
    base = _resolve(root)
    if base is None or not _root_ok(base):
        log.warning("拒絕刪除%s：%s 不是專案允許刪除的資料夾（目標 %s）", kind, root, path)
        return None
    p = Path(path)
    parent = _resolve(p.parent)
    # Windows 會忽略結尾的空白和點，「.. 」「...」實際上就是 .. 或 .
    if parent is None or p.name.rstrip(" .") == "":
        log.warning("拒絕刪除%s：路徑不正確 %s", kind, path)
        return None
    item = parent / p.name
    if not _within(item, base):
        log.warning("拒絕刪除%s：%s 不在 %s 裡面", kind, path, root)
        return None
    if is_link(item):
        log.warning("拒絕刪除%s：%s 是捷徑（symlink 或 junction）", kind, path)
        return None
    real = _resolve(item)
    if real is None or not _within(real, base):
        log.warning("拒絕刪除%s：%s 實際指到 %s，不在 %s 裡面", kind, path, real, root)
        return None
    return item


def safe_unlink(path, root) -> bool:
    """刪一個檔案。檔案本來就不在也算成功；被拒絕或刪不掉回 False。"""
    item = _checked(path, root, "檔案")
    if item is None:
        return False
    if item.is_dir():
        log.warning("拒絕刪除檔案：%s 是資料夾", path)
        return False
    try:
        item.unlink(missing_ok=True)
        return True
    except OSError as e:
        log.warning("刪除檔案失敗 %s：%s", item, e)
        return False


def safe_rmtree(path, root) -> bool:
    """刪一整個資料夾。資料夾本來就不在也算成功；被拒絕回 False。

    裡面如果有捷徑，shutil.rmtree 只會拿掉捷徑本身，不會跟進去刪捷徑指到的地方。
    個別檔案刪不掉（例如正被播放）只記 log，不中斷。
    """
    item = _checked(path, root, "資料夾")
    if item is None:
        return False
    if not item.exists():
        return True
    if not item.is_dir():
        log.warning("拒絕刪除資料夾：%s 不是資料夾", path)
        return False

    def onexc(_func, failed, exc):
        log.warning("刪除失敗 %s：%s", failed, exc)

    shutil.rmtree(item, onexc=onexc)
    return True


# ---------- 下載網址 ----------

def _host_ip(host: str):
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        ip = None
        # 127.1、2130706433、0x7f.0.0.1 這類舊寫法，連線時一樣會被當成 IP
        if re.fullmatch(r"[0-9a-fA-FxX.]+", host):
            try:
                ip = ipaddress.ip_address(socket.inet_aton(host))
            except OSError:
                ip = None
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
        ip = ip.ipv4_mapped
    return ip


def check_download_url(url) -> str:
    """給 yt-dlp 下載的網址：只收 http、https，不能有空白，不能以 - 開頭，不能直接指到本機或區網位址。

    回傳去掉前後空白的網址；不合格丟 ValueError（訊息可以直接顯示給使用者）。
    只檢查網址上寫的主機名稱，不做 DNS 查詢（有些代理軟體會把網站解析成保留位址），也管不到網站自己的轉址。
    """
    url = (url or "").strip() if isinstance(url, str) else ""
    if not url or url.startswith("-"):
        raise ValueError("請輸入完整網址（http 或 https 開頭）")
    if any(c.isspace() or ord(c) < 32 or ord(c) == 127 for c in url):
        raise ValueError("網址裡不能有空白或換行")
    try:
        parts = urlsplit(url)
        host = parts.hostname
    except ValueError:
        raise ValueError("網址格式不正確")
    if parts.scheme.lower() not in ("http", "https") or not host:
        raise ValueError("請輸入完整網址（http 或 https 開頭）")
    # 先還原成連線時實際用的主機名稱再檢查：%2e、全形數字、①、「。」這類寫法都會被當成一般的點和數字
    host = unicodedata.normalize("NFKC", unquote(host))
    host = re.sub("[。．｡]", ".", host)
    try:
        host = host.encode("idna").decode("ascii")
    except UnicodeError:
        raise ValueError("網址格式不正確")
    host = host.rstrip(".").lower()
    if not host:
        raise ValueError("請輸入完整網址（http 或 https 開頭）")
    if host == "localhost" or host.endswith(".localhost"):
        raise ValueError("不能下載本機或區網裡的網址")
    ip = _host_ip(host)
    if ip is not None and (ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved
                           or ip.is_multicast or ip.is_unspecified):
        raise ValueError("不能下載本機或區網裡的網址")
    return url
