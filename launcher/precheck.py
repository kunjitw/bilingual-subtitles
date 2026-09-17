"""啟動前的電腦檢查。只用 Python 標準函式庫，在建立執行環境以前就能跑。

start.bat 用 uv 裝好的 Python 執行：python -I -X utf8 launcher\\precheck.py
結果寫到 runtime\\precheck.json（之後導覽的「檢查電腦」也會讀）。
有「無法繼續」的項目就印出中文原因、結束代碼 1；只有「注意」時照常繼續。

已經裝好的電腦每次啟動也會跑，所以只做很快的檢查（顯示卡用 NVML、磁碟、記憶體、路徑），
網路只在還沒裝好時檢查；完整包的 offline 資料夾有 GitHub 上的執行環境檔時不檢查 GitHub。

測試用的環境變數：
  VS_FAKE_SYSTEM      JSON 檔，裡面的 "gpu" 當作顯示卡資訊（跟 app/syscheck.py 一樣，null 代表沒有 NVIDIA）
  VS_TEST_ALLOW_TEMP  設成 1 時允許放在暫存資料夾底下（自動測試用）
"""
import ctypes
import ctypes.wintypes as wintypes
import json
import os
import platform
import shutil
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RT = ROOT / "runtime"
OUT_PATH = RT / "precheck.json"
STATE_PATH = RT / "install-state.json"

MAX_PATH_LEN = 80
BAD_CHARS = "%;!^&"      # 會弄壞 cmd 或 PATH；中文等非英文字元另外擋（見 check_location）
GIB = 1024 ** 3
MIN_DRIVER = 580
MIN_CUDA_TEXT = "13.0"    # torch cu130、llama.cpp CUDA 13；驅動 580 起支援
MIN_ARCH = (7, 5)
MIN_VRAM_MB = 7700

sys.path.insert(0, str(ROOT))
try:
    from app import syscheck   # 只用標準函式庫（ctypes 讀 NVML）
except Exception:  # noqa: BLE001
    syscheck = None


def item(items: list, id_: str, status: str, value: str = "", message: str = ""):
    items.append({"id": id_, "status": status, "value": value, "message": message})


# ---------- 各項檢查 ----------

def check_os(items: list):
    ver = sys.getwindowsversion()
    arch = platform.machine()
    value = f"Windows build {ver.build}（{arch}）"
    if arch.upper() not in ("AMD64", "X86_64"):
        item(items, "os", "fail", value, "這個程式只支援 64 位元（x64）的 Windows。")
    elif ver.build < 19041:
        item(items, "os", "fail", value, "Windows 版本太舊。請先更新到 Windows 10 22H2 或 Windows 11。")
    elif ver.build < 22000:
        item(items, "os", "warn", value, "Windows 10 已經停止支援，程式還是可以用，建議之後升級到 Windows 11。")
    else:
        item(items, "os", "ok", value)


def _driver_major(driver) -> int | None:
    try:
        return int(str(driver).split(".")[0])
    except (TypeError, ValueError):
        return None


def check_gpu(items: list) -> dict | None:
    info = syscheck.query() if syscheck else None
    if not info:
        item(items, "gpu", "fail", "沒有偵測到",
             "這台電腦沒有偵測到 NVIDIA 顯示卡。轉字幕和翻譯需要 NVIDIA RTX 20 系列以後的顯示卡。"
             "如果你有 NVIDIA 顯示卡，請先到 NVIDIA 官網或 NVIDIA App 安裝最新驅動，再打開 start.bat。")
        return None
    name = info.get("name") or "NVIDIA"
    total = info.get("total_mb") or 0
    driver = info.get("driver") or "未知"
    cuda = info.get("cuda_version")
    cuda_text = f"{cuda // 1000}.{cuda % 1000 // 10}" if cuda else "讀不到"
    # 顯示使用者的驅動和 CUDA 版本，旁邊寫這個程式需要多少，不合的時候一眼看得出差在哪
    value = f"{name}，{total / 1024:.0f} GB，驅動 {driver}，CUDA {cuda_text}（需要 {MIN_CUDA_TEXT} 以上）"
    major = _driver_major(driver)
    arch = info.get("arch")
    if (cuda and cuda < 13000) or (not cuda and major is not None and major < MIN_DRIVER):
        item(items, "gpu", "fail", value,
             f"顯示卡驅動太舊（目前 {driver}，支援 CUDA {cuda_text}）。"
             f"這個程式需要驅動 {MIN_DRIVER} 版以上（支援 CUDA {MIN_CUDA_TEXT}）。"
             "請到 NVIDIA 官網或 NVIDIA App 更新驅動，更新完重新開機，再打開 start.bat。")
    elif arch and tuple(arch[:2]) < MIN_ARCH:
        item(items, "gpu", "fail", value,
             f"這張顯示卡（{name}）太舊了。程式用的 CUDA 13 從 RTX 20 系列開始支援，GTX 10 系列和更早的不能用。")
    else:
        notes = []
        max_major = getattr(syscheck, "MAX_TESTED_ARCH_MAJOR", 12)
        if arch and arch[0] > max_major:
            notes.append(f"這張顯示卡（架構 {arch[0]}.{arch[1]}）比程式用的 PyTorch 還新，可能還不支援。"
                         "安裝完的基本測試會確認，不支援的話要等程式出新版本。")
        if arch and tuple(arch[:2]) == MIN_ARCH:
            notes.append("RTX 20 和 GTX 16 系列可以用，但辨識會比 RTX 30 系列以後的卡慢，也還沒有實際測試過。")
        if total and total < MIN_VRAM_MB:
            notes.append(f"顯存只有 {total / 1024:.0f} GB，目前的辨識和翻譯模型要 8 GB 以上才放得下。")
        if major is not None and major < 595:
            notes.append("驅動版本可以用，但翻譯程式只在 595 版以上測過。遇到翻譯開不起來時，請先更新驅動。")
        item(items, "gpu", "warn" if notes else "ok", value, "".join(notes))
    return info


def _volume_info(path: Path) -> tuple:
    """(檔案系統名稱, 磁碟類型)；讀不到回 (None, None)。"""
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    buf = ctypes.create_unicode_buffer(300)
    if not k32.GetVolumePathNameW(str(path), buf, 300):
        return None, None
    root = buf.value
    fs = ctypes.create_unicode_buffer(64)
    ok = k32.GetVolumeInformationW(ctypes.c_wchar_p(root), None, 0, None, None, None, fs, 64)
    drive_type = k32.GetDriveTypeW(ctypes.c_wchar_p(root))
    return (fs.value if ok else None), drive_type


def _under(path: str, parent: str | None) -> bool:
    if not parent:
        return False
    p = os.path.normcase(os.path.abspath(path)).rstrip("\\") + "\\"
    q = os.path.normcase(os.path.abspath(parent)).rstrip("\\") + "\\"
    return p.startswith(q)


def check_location(items: list, installed: bool):
    root = str(ROOT)
    low = root.lower()
    bad = sorted({c for c in root if c in BAD_CHARS})
    env = os.environ
    temps = [env.get("VS_ORIG_TEMP"), env.get("VS_ORIG_TMP"), os.path.join(env.get("SystemRoot", r"C:\Windows"), "Temp")]
    onedrives = [env.get(k) for k in ("OneDrive", "OneDriveConsumer", "OneDriveCommercial")]
    program_files = [env.get(k) for k in ("ProgramFiles", "ProgramFiles(x86)", "ProgramW6432")]
    if len(root) > MAX_PATH_LEN:
        item(items, "path", "fail", root,
             f"程式資料夾的路徑太長了（{len(root)} 個字，上限 {MAX_PATH_LEN}）。請把整個資料夾搬到比較短的位置，"
             "例如 D:\\BilingualSubtitles，再重新打開 start.bat。")
    elif bad:
        item(items, "path", "fail", root,
             f"程式資料夾的路徑裡有 {' '.join(bad)} 這些符號，會讓安裝出錯。請把資料夾改名或搬到 D:\\BilingualSubtitles 這類位置。")
    elif any(_under(root, p) for p in onedrives) or "\\onedrive" in low:
        item(items, "path", "fail", root,
             "程式放在 OneDrive 資料夾裡。OneDrive 會把幾十 GB 的模型上傳到雲端，也可能鎖住檔案。"
             "請把整個資料夾搬到 D:\\BilingualSubtitles 這類位置。")
    elif any(_under(root, p) for p in program_files):
        item(items, "path", "fail", root,
             "程式放在 Program Files 裡，一般權限沒辦法寫入。請把整個資料夾搬到 D:\\BilingualSubtitles 這類位置。")
    elif env.get("VS_TEST_ALLOW_TEMP") != "1" and (any(_under(root, p) for p in temps) or "\\appdata\\local\\temp" in low):
        item(items, "path", "fail", root,
             "程式還在暫存資料夾裡（可能是直接從壓縮檔打開）。暫存資料夾會被系統清掉，"
             "請先把整個壓縮檔解壓縮到 D:\\BilingualSubtitles 這類位置，再從那裡打開 start.bat。")
    elif any(ord(c) > 127 for c in root):
        # 2026-09-17 實測：路徑有中文時，qwen-asr 用的 nagisa（DyNet）讀不到自己的模型檔，辨識整個不能用
        item(items, "path", "fail", root,
             "程式資料夾的路徑裡有中文或其他非英文字元，辨識用的套件會讀不到檔案。"
             "請把整個資料夾搬到只有英文、數字的位置，例如 D:\\BilingualSubtitles，再重新打開 start.bat。")
    else:
        item(items, "path", "ok", root)

    try:
        fs, drive_type = _volume_info(ROOT)
    except OSError:
        fs, drive_type = None, None
    fs_value = f"{fs or '未知'}"
    if fs and fs.upper() in ("FAT32", "FAT", "EXFAT"):
        item(items, "filesystem", "fail", fs_value,
             f"這個磁碟是 {fs} 格式，放不下大模型，也沒辦法建立執行環境。請把程式放到 NTFS 格式的磁碟（一般的 C:、D: 都是）。")
    elif drive_type == 4:
        item(items, "filesystem", "fail", fs_value + "，網路磁碟",
             "程式放在網路磁碟上，速度太慢也容易出錯。請把整個資料夾搬到這台電腦的磁碟，例如 D:\\BilingualSubtitles。")
    else:
        item(items, "filesystem", "ok", fs_value)

    # 實際寫入測試（受控資料夾存取、唯讀位置）
    try:
        (RT / "tmp").mkdir(parents=True, exist_ok=True)
        probe = RT / "tmp" / f"write-test-{os.getpid()}.tmp"
        probe.write_bytes(b"ok")
        probe.unlink()
        item(items, "writable", "ok", "可以寫入")
    except OSError as e:
        item(items, "writable", "fail", str(e),
             "沒辦法寫入程式資料夾（可能被防毒軟體的「受控資料夾存取」擋住）。請把整個資料夾搬到 D:\\BilingualSubtitles 這類位置。")

    anchor = ROOT
    while not anchor.exists() and anchor.parent != anchor:
        anchor = anchor.parent
    try:
        free = shutil.disk_usage(anchor).free
    except OSError:
        item(items, "disk", "unknown", "讀不到")
        return
    drive = os.path.splitdrive(root)[0] or root[:2]
    value = f"{drive} 還剩 {free / GIB:.0f} GB"
    if not installed and free < 10 * GIB:
        item(items, "disk", "fail", value, f"{drive} 只剩 {free / GIB:.0f} GB，安裝執行環境至少要 10 GB。清出空間後再打開 start.bat。")
    elif not installed and free < 30 * GIB:
        item(items, "disk", "warn", value, f"{drive} 只剩 {free / GIB:.0f} GB。執行環境裝得下，但模型大多要 5 到 12 GB，可能放不下常用的組合。")
    elif installed and free < 5 * GIB:
        item(items, "disk", "warn", value, f"{drive} 只剩 {free / GIB:.0f} GB，處理影片時可能不夠用。")
    else:
        item(items, "disk", "ok", value)


class _MemoryStatus(ctypes.Structure):
    _fields_ = [("dwLength", wintypes.DWORD), ("dwMemoryLoad", wintypes.DWORD),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


def check_memory(items: list):
    st = _MemoryStatus()
    st.dwLength = ctypes.sizeof(_MemoryStatus)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
        item(items, "memory", "unknown", "讀不到")
        return
    gb = st.ullTotalPhys / GIB
    value = f"{gb:.0f} GB"
    # 8 GB 的電腦常常只讀到 7.4 到 7.9 GB（內顯共用）
    if gb < 7.0:
        item(items, "memory", "fail", value, f"記憶體只有 {gb:.0f} GB，處理影片時很容易當掉，至少要 8 GB。")
    elif gb < 15.0:
        item(items, "memory", "warn", value, f"記憶體 {gb:.0f} GB 可以用，但處理很長的歌曲影片時可能不夠，先關掉其他程式。")
    else:
        item(items, "memory", "ok", value)


def check_sac(items: list):
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\CI\Policy") as key:
            state, _ = winreg.QueryValueEx(key, "VerifiedAndReputablePolicyState")
    except OSError:
        item(items, "sac", "ok", "關閉")
        return
    if state in (1, 2):
        item(items, "sac", "warn", "開啟" if state == 1 else "評估模式",
             "這台電腦開著「智慧型應用程式控制」，Windows 可能會擋下程式裡沒有簽章的檔案（例如 python.exe、llama-server.exe）。"
             "如果安裝或啟動時出現「已封鎖」，請把畫面拍下來傳給作者。")
    else:
        item(items, "sac", "ok", "關閉")


def _reachable(url: str, timeout: float = 10) -> bool:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "video-subtitle-precheck/1", "Range": "bytes=0-0"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status < 500
    except urllib.error.HTTPError as e:
        return e.code < 500
    except OSError:
        return False


def github_needed() -> bool:
    """完整包的 offline 資料夾已經有 GitHub 上的執行環境檔（ffmpeg、llama.cpp、cudart、en-core-web-sm）時，安裝不用連 GitHub。
    判斷不了（讀不到 manifest 之類）時當成要連。"""
    try:
        here = str(Path(__file__).resolve().parent)
        if here not in sys.path:
            sys.path.insert(0, here)
        import setup_runtime
        manifest = json.loads((ROOT / "launcher" / "runtime-manifest.json").read_text(encoding="utf-8"))
        return not setup_runtime.offline_complete(manifest, ROOT / "offline", ROOT / "uv.lock")
    except Exception:  # noqa: BLE001
        return True


def check_network(items: list, github: bool = True):
    targets = {"pypi": "https://pypi.org/simple/uv/", "pytorch": "https://download.pytorch.org/whl/cu130/",
               "github": "https://github.com/", "huggingface": "https://huggingface.co/api/models/Qwen/Qwen3-ASR-1.7B"}
    if not github:
        del targets["github"]
    result: dict = {}
    threads = [threading.Thread(target=lambda k=k, u=u: result.__setitem__(k, _reachable(u)), daemon=True)
               for k, u in targets.items()]
    for t in threads:
        t.start()
    for t in threads:
        t.join(15)
    value = "、".join(f"{k} {'可以連' if result.get(k) else '連不到'}" for k in targets)
    if not github:
        value += "、github 不用連（offline 資料夾裡有檔案）"
    if not result.get("pypi") and not result.get("pytorch"):
        item(items, "network", "fail", value,
             "連不到下載來源（pypi.org、download.pytorch.org）。請確認網路正常；公司或學校網路可能擋住了，換個網路再試。")
    elif (github and not result.get("github")) or not result.get("pypi") or not result.get("pytorch"):
        missing = [k for k in ("pypi", "pytorch", "github") if k in targets and not result.get(k)]
        item(items, "network", "fail", value,
             f"連不到 {'、'.join(missing)}，執行環境沒辦法下載完整。請確認網路正常，或換個網路再試。")
    elif not result.get("huggingface"):
        item(items, "network", "warn", value, "連不到 Hugging Face，之後沒辦法下載模型。公司或學校網路可能擋住了。")
    else:
        item(items, "network", "ok", value)


# ---------- 主流程 ----------

def installed_here() -> bool:
    try:
        state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return isinstance(state, dict) and (state.get("tests") or {}).get("ok") is True


def run() -> dict:
    started = time.time()
    installed = installed_here()
    items: list = []
    check_os(items)
    check_location(items, installed)
    gpu = check_gpu(items)
    check_memory(items)
    check_sac(items)
    if not installed:
        check_network(items, github=github_needed())
    overall = "fail" if any(i["status"] == "fail" for i in items) else \
        "warn" if any(i["status"] == "warn" for i in items) else "ok"
    return {"version": 1, "checked_at": time.time(), "seconds": round(time.time() - started, 3),
            "installed": installed, "overall": overall, "gpu": gpu, "items": items}


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    result = run()
    try:
        RT.mkdir(parents=True, exist_ok=True)
        tmp = OUT_PATH.with_name(OUT_PATH.name + ".tmp")
        tmp.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, OUT_PATH)
    except OSError:
        pass
    fails = [i for i in result["items"] if i["status"] == "fail"]
    warns = [i for i in result["items"] if i["status"] == "warn"]
    # 裝好以後每次啟動都會檢查，「注意」只在第一次安裝時顯示（之後在導覽的「檢查電腦」看）
    if not result["installed"]:
        gpu = next((i for i in result["items"] if i["id"] == "gpu" and i["status"] != "fail"), None)
        if gpu:
            print(f"顯示卡：{gpu['value']}")
        for i in warns:
            print("注意：" + i["message"])
    if fails:
        print()
        for i in fails:
            print("沒辦法繼續：" + i["message"])
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
