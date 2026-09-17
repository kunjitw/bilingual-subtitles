"""顯示卡檢查：型號、顯存、驅動版本、驅動支援的 CUDA 版本、架構（compute capability）。

先用驅動內建的 nvml.dll（ctypes 呼叫，不用另外裝套件，也不會在顯卡上建立 CUDA context），
讀不到再改用 nvidia-smi。只看第 0 張卡：NVML、nvidia-smi 都照 PCI 順序編號，
跟 llama-server 的 --device CUDA0、子程序的 CUDA_DEVICE_ORDER=PCI_BUS_ID 是同一張。

模擬別台電腦（測試、導覽截圖）：環境變數 VS_FAKE_SYSTEM 指到一個 JSON 檔，
裡面的 "gpu" 就是 query() 要回傳的內容，null 代表沒有 NVIDIA 顯示卡。程式裡可以用 set_fake()。
"""
import ctypes
import json
import os
import re
import subprocess
import threading
import time

# 公開版的 torch 是 cu130、llama.cpp 是 CUDA 13：驅動要支援 CUDA 13.0，也就是 580 版以上
MIN_CUDA_VERSION = 13000          # NVML 的寫法：13.0 → 13000
MIN_DRIVER = 580
MIN_ARCH = (7, 5)                 # RTX 20、GTX 16 系列；torch cu130 支援 sm_75 以上
MIB = 1024 * 1024

_UNSET = object()
_fake = _UNSET


class _MemoryV1(ctypes.Structure):
    _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong), ("used", ctypes.c_ulonglong)]


class _MemoryV2(ctypes.Structure):
    # nvmlMemory_v2_t：used 不含驅動保留的部分，跟 nvidia-smi 的 memory.used 一樣
    _fields_ = [("version", ctypes.c_uint), ("total", ctypes.c_ulonglong), ("reserved", ctypes.c_ulonglong),
                ("free", ctypes.c_ulonglong), ("used", ctypes.c_ulonglong)]


class _PciInfo(ctypes.Structure):
    _fields_ = [("busIdLegacy", ctypes.c_char * 16), ("domain", ctypes.c_uint), ("bus", ctypes.c_uint),
                ("device", ctypes.c_uint), ("pciDeviceId", ctypes.c_uint), ("pciSubSystemId", ctypes.c_uint),
                ("busId", ctypes.c_char * 32)]


_lock = threading.Lock()
_nvml = {"lib": None, "handle": None, "static": None, "retry_at": 0.0}
_smi_static: dict = {}


def set_fake(value=_UNSET):
    """測試用：之後的 query() 直接回傳 value（None 代表沒有 NVIDIA 顯示卡）；不給參數就恢復讀真的顯示卡。"""
    global _fake
    _fake = value


def _fake_from_env():
    path = os.environ.get("VS_FAKE_SYSTEM")
    if not path:
        return _UNSET
    try:
        with open(path, encoding="utf-8-sig") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return _UNSET
    return data.get("gpu") if isinstance(data, dict) else _UNSET


def _mb(value: int) -> int:
    return int(round(value / MIB))


def _cuda_text(version: int | None) -> str | None:
    if not version:
        return None
    return f"{version // 1000}.{version % 1000 // 10}"


def _nvml_dll_paths() -> list[str]:
    system_root = os.environ.get("SystemRoot", r"C:\Windows")
    program_files = os.environ.get("ProgramFiles", r"C:\Program Files")
    return [os.path.join(system_root, "System32", "nvml.dll"),
            os.path.join(program_files, "NVIDIA Corporation", "NVSMI", "nvml.dll")]


def _optional(lib, func: str, *args) -> bool:
    """呼叫 NVML 的選用函式，成功回 True；舊版 nvml.dll 沒有這個函式也回 False。"""
    fn = getattr(lib, func, None)
    return fn is not None and fn(*args) == 0


def _nvml_open():
    """載入 nvml.dll 並初始化，拿到第 0 張卡的 handle 和不會變的資訊。失敗回 None，30 秒內不再試。"""
    if _nvml["handle"] is not None:
        return _nvml
    if time.time() < _nvml["retry_at"]:
        return None
    _nvml["retry_at"] = time.time() + 30
    lib = None
    for path in _nvml_dll_paths():
        if os.path.isfile(path):
            try:
                lib = ctypes.CDLL(path)
                break
            except OSError:
                continue
    if lib is None:
        return None
    # 很舊的驅動（R418 以前）的 nvml.dll 缺少下面某些函式：必要的缺了就改用 nvidia-smi，
    # 選用的（CUDA 版本、架構、PCI）缺了就當作讀不到，驅動版本檢查照樣會說「驅動太舊」
    init = getattr(lib, "nvmlInit_v2", None)
    if init is None or init() != 0:
        return None
    try:
        count = ctypes.c_uint()
        handle = ctypes.c_void_p()
        if lib.nvmlDeviceGetCount_v2(ctypes.byref(count)) != 0 or count.value < 1 \
                or lib.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(handle)) != 0:
            lib.nvmlShutdown()
            return None
        name = ctypes.create_string_buffer(96)
        lib.nvmlDeviceGetName(handle, name, 96)
        driver = ctypes.create_string_buffer(96)
        lib.nvmlSystemGetDriverVersion(driver, 96)
    except AttributeError:
        lib.nvmlShutdown()
        return None
    cuda = ctypes.c_int()
    cuda_version = cuda.value if _optional(lib, "nvmlSystemGetCudaDriverVersion_v2", ctypes.byref(cuda)) else None
    major, minor = ctypes.c_int(), ctypes.c_int()
    arch = None
    if _optional(lib, "nvmlDeviceGetCudaComputeCapability", handle, ctypes.byref(major), ctypes.byref(minor)):
        arch = [major.value, minor.value]
    pci = _PciInfo()
    bus_id = None
    if _optional(lib, "nvmlDeviceGetPciInfo_v3", handle, ctypes.byref(pci)):
        bus_id = pci.busId.decode(errors="replace")
    _nvml.update(lib=lib, handle=handle, static={
        "name": name.value.decode(errors="replace"), "driver": driver.value.decode(errors="replace"),
        "cuda_version": cuda_version, "cuda": _cuda_text(cuda_version),
        "arch": arch, "compute_cap": f"{arch[0]}.{arch[1]}" if arch else None,
        "count": count.value, "pci_bus_id": bus_id,
    })
    return _nvml


def _nvml_query() -> dict | None:
    nv = _nvml_open()
    if nv is None:
        return None
    lib, handle = nv["lib"], nv["handle"]
    mem2 = _MemoryV2()
    mem2.version = ctypes.sizeof(_MemoryV2) | (2 << 24)
    try:
        ok_v2 = lib.nvmlDeviceGetMemoryInfo_v2(handle, ctypes.byref(mem2)) == 0
    except AttributeError:          # 很舊的驅動沒有 v2
        ok_v2 = False
    if ok_v2:
        total, reserved, free, used = mem2.total, mem2.reserved, mem2.free, mem2.used
    else:
        mem = _MemoryV1()
        if lib.nvmlDeviceGetMemoryInfo(handle, ctypes.byref(mem)) != 0:
            # 顯示卡被拔掉、驅動重新載入：下次重新初始化
            _nvml.update(handle=None, static=None, retry_at=0.0)
            return None
        total, reserved, free, used = mem.total, 0, mem.free, mem.used
    return {**nv["static"], "total_mb": _mb(total), "reserved_mb": _mb(reserved), "used_mb": _mb(used),
            "free_mb": _mb(free), "source": "nvml"}


def _num(text: str) -> int | None:
    text = (text or "").strip()
    return int(float(text)) if re.fullmatch(r"\d+(\.\d+)?", text) else None


def nvidia_smi_query(timeout: float = 5) -> dict | None:
    """用 nvidia-smi 讀第 0 張卡。舊驅動不認得的欄位（compute_cap、memory.reserved）會拿掉再試一次。"""
    fields = ["name", "memory.total", "memory.used", "memory.free", "driver_version", "compute_cap",
              "memory.reserved", "pci.bus_id"]
    for attempt in (fields, fields[:5]):
        try:
            out = subprocess.run(["nvidia-smi", f"--query-gpu={','.join(attempt)}", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout,
                                 creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except (OSError, subprocess.SubprocessError):
            return None
        lines = [l for l in out.stdout.strip().splitlines() if l.strip()]
        if out.returncode == 0 and lines:
            break
    else:
        return None
    row = dict(zip(attempt, [x.strip() for x in lines[0].split(",")]))
    total, used, free = _num(row.get("memory.total")), _num(row.get("memory.used")), _num(row.get("memory.free"))
    if total is None or used is None or free is None:
        return None
    cap = row.get("compute_cap")
    arch = [int(x) for x in cap.split(".")] if cap and re.fullmatch(r"\d+\.\d+", cap) else None
    if "cuda" not in _smi_static:
        # 驅動支援的 CUDA 版本只在 nvidia-smi 的表頭（CUDA Version: 13.2）
        try:
            head = subprocess.run(["nvidia-smi"], capture_output=True, text=True, encoding="utf-8", errors="replace",
                                  timeout=timeout, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout
            m = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", head)
            _smi_static["cuda"] = (int(m.group(1)) * 1000 + int(m.group(2)) * 10) if m else None
        except (OSError, subprocess.SubprocessError):
            _smi_static["cuda"] = None
    cuda_version = _smi_static["cuda"]
    return {"name": row.get("name"), "driver": row.get("driver_version"), "cuda_version": cuda_version,
            "cuda": _cuda_text(cuda_version), "arch": arch, "compute_cap": cap if arch else None,
            "count": len(lines), "pci_bus_id": row.get("pci.bus_id"),
            "total_mb": total, "reserved_mb": _num(row.get("memory.reserved")) or 0, "used_mb": used, "free_mb": free,
            "source": "nvidia-smi"}


def query() -> dict | None:
    """第 0 張 NVIDIA 顯示卡的資訊；讀不到（沒有 NVIDIA 顯示卡、沒裝驅動）回 None。

    total_mb、used_mb、free_mb、reserved_mb 是 MiB。used + free 是模型最多能用的量（總量扣掉驅動保留的）。
    driver 是驅動版本字串（595.79），cuda_version 是驅動支援的 CUDA 版本（13.2 → 13020），
    arch 是架構 [major, minor]，source 是 nvml 或 nvidia-smi。
    """
    fake = _fake if _fake is not _UNSET else _fake_from_env()
    if fake is not _UNSET:
        return dict(fake) if isinstance(fake, dict) else None
    with _lock:
        try:
            info = _nvml_query()
        except (OSError, AttributeError):      # nvml.dll 壞掉或版本太舊：改用 nvidia-smi
            info = None
    return info or nvidia_smi_query()


def hidden_by_env(info: dict | None = None, env=None) -> str | None:
    """環境變數 CUDA_VISIBLE_DEVICES 把顯示卡藏起來時，回傳它的值（辨識、翻譯的子程序會看不到顯示卡）。
    CUDA 從第一個值開始讀，遇到不合法的就停：空的、負數，或編號超過顯示卡數量，就一張都看不到。"""
    raw = (os.environ if env is None else env).get("CUDA_VISIBLE_DEVICES")
    if raw is None:
        return None
    first = raw.split(",")[0].strip()
    count = (info or {}).get("count")
    if not first or first.startswith("-") or (first.isdigit() and isinstance(count, int) and int(first) >= count):
        return raw
    return None


def _driver_major(driver) -> int | None:
    m = re.match(r"\s*(\d+)", str(driver or ""))
    return int(m.group(1)) if m else None


def problem_code(info: dict | None) -> tuple[str, str] | None:
    """這張卡不能跑 GPU 任務時回 (代碼, 給使用者看的原因)；可以用回 None。
    代碼：no_gpu 讀不到 NVIDIA 顯示卡、driver_old 驅動太舊、arch_old 架構太舊、gpu_hidden 被 CUDA_VISIBLE_DEVICES 藏起來。
    第一次打開的自動安裝（app/setup.py）照代碼顯示簡短的說明。"""
    if not info:
        return ("no_gpu", "讀不到 NVIDIA 顯示卡。辨識、翻譯都要用 NVIDIA 顯示卡（RTX 20、GTX 16 系列或更新），"
                          "請確認電腦有這類顯示卡，而且已經裝好 NVIDIA 驅動")
    cuda_version = info.get("cuda_version")
    driver = _driver_major(info.get("driver"))
    too_old = (cuda_version < MIN_CUDA_VERSION) if cuda_version else (driver is not None and driver < MIN_DRIVER)
    if too_old:
        now = info.get("driver") or "未知"
        cuda = f"，支援到 CUDA {_cuda_text(cuda_version)}" if cuda_version else ""
        return ("driver_old",
                f"顯示卡驅動太舊（目前 {now}{cuda}）。這個程式需要驅動 {MIN_DRIVER} 版以上"
                f"（支援 CUDA {_cuda_text(MIN_CUDA_VERSION)}），請到 NVIDIA 官網更新驅動，再按「重試」")
    arch = info.get("arch")
    if arch and tuple(arch[:2]) < MIN_ARCH:
        return ("arch_old", f"這張顯示卡（{info.get('name') or 'NVIDIA'}）太舊，程式沒辦法用它："
                            f"架構 {arch[0]}.{arch[1]} 低於 7.5。需要 RTX 20、GTX 16 系列或更新的 NVIDIA 顯示卡")
    hidden = hidden_by_env(info)
    if hidden is not None:
        return ("gpu_hidden", f"環境變數 CUDA_VISIBLE_DEVICES={hidden} 把顯示卡藏起來了，程式用不到顯示卡。"
                              "請到 Windows 的環境變數設定拿掉它，再重新啟動程式")
    return None


def problem(info: dict | None) -> str | None:
    """這張卡不能跑 GPU 任務的原因（給使用者看）；可以用回 None。"""
    code = problem_code(info)
    return code[1] if code else None


def driver_major(driver) -> int | None:
    """驅動版本字串的主版號（595.79 → 595）。"""
    return _driver_major(driver)


# 子程序（torch、llama-server）找不到能用的 CUDA 裝置時會印的字樣（小寫比對）
NO_CUDA_MARKERS = ("no cuda gpus are available", "found no nvidia driver", "nvidia driver on your system is too old",
                   "cuda driver version is insufficient", "no cuda-capable device", "cuda unknown error",
                   "invalid device", "failed to initialize cuda", "no usable gpu")


# 顯示卡比程式用的 PyTorch、llama.cpp 還新（新一代架構還沒被編進去）時的錯誤字樣
GPU_TOO_NEW_MARKERS = ("no kernel image is available", "is not compatible with the current pytorch installation",
                       "unsupported gpu architecture")
# torch 2.14 cu130 編到 sm_120（RTX 50 系列）；架構大版本比這個新的顯示卡，要等 PyTorch 支援、程式更新
MAX_TESTED_ARCH_MAJOR = 12


def gpu_too_new(text: str) -> bool:
    low = (text or "").lower()
    return any(k in low for k in GPU_TOO_NEW_MARKERS)


def too_new_message(what: str) -> str:
    return (f"{what}沒辦法用這張顯示卡：顯示卡比程式用的 PyTorch 還新，PyTorch 還不支援它。"
            "這不是驅動的問題，要等程式出新版本")


def cuda_unusable(text: str) -> bool:
    low = (text or "").lower()
    return any(k in low for k in NO_CUDA_MARKERS)


def no_cuda_message(what: str) -> str:
    return (f"{what}沒辦法用顯示卡：找不到可以用的 NVIDIA 顯示卡（CUDA）。"
            f"請確認顯示卡驅動已經裝好、版本在 {MIN_DRIVER} 以上，再按「重試」")
