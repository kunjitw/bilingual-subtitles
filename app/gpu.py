"""顯卡狀態查詢與載入前的顯存檢查。資訊來自 app/syscheck.py（NVML，讀不到再用 nvidia-smi）。"""
import threading
import time

from . import syscheck

_cache = {"t": 0.0, "info": None}
_lock = threading.Lock()
_NOW = object()


class GpuUnsupported(RuntimeError):
    """沒有 NVIDIA 顯示卡、驅動太舊或架構太舊。訊息可以直接給使用者看。"""


def query(max_age=2.0) -> dict | None:
    with _lock:
        if time.time() - _cache["t"] < max_age:
            return _cache["info"]
        try:
            info = syscheck.query()
        except Exception:  # noqa: BLE001
            info = None
        _cache.update(t=time.time(), info=info)
        return info


def usable_mb(info: dict | None) -> int | None:
    """整張卡上模型最多能拿到的顯存：used + free（總顯存扣掉驅動保留的，跟 models.gpu_usable_mb 同一個算法）。"""
    if not info:
        return None
    return int(info["used_mb"]) + int(info["free_mb"])


def fits(need_mb: int, info=_NOW) -> bool | None:
    """整張卡放不放得下（不看現在其他程式用掉多少）；讀不到顯卡時回 None。"""
    usable = usable_mb(query() if info is _NOW else info)
    return None if usable is None else need_mb <= usable


def require_supported(info=_NOW) -> dict:
    """GPU 任務開始前檢查：沒有 NVIDIA 顯示卡、驅動太舊、架構太舊就丟 GpuUnsupported 並說明原因，
    不讓任務在 CPU 上硬跑，也不會跑到一半才壞掉。回傳目前的顯卡資訊。"""
    info = query(max_age=0) if info is _NOW else info
    reason = syscheck.problem(info)
    if reason:
        raise GpuUnsupported(reason)
    return info


def pick_batch(table: list[tuple[int, int]], label: str) -> int:
    """依目前剩餘顯存挑一次處理的段數；連最小的都放不下就報錯。"""
    info = require_supported()
    for need, batch in table:
        if info["free_mb"] >= need:
            return batch
    ensure_free(table[-1][0], label, info=info)
    return table[-1][1]


def ensure_free(need_mb: int, label: str, alternatives=None, info=_NOW):
    """剩餘顯存不夠就直接報錯，不讓驅動偷偷借用系統記憶體。

    整張卡本來就放不下（需要的比 used + free 還多）時，關程式、重試都沒用，改說這張卡放不下，請換小一點的模型；
    alternatives(usable_mb) 可以回傳放得下的模型名稱，寫進說明裡。只有「關掉其他程式就夠」時才叫人重試。
    """
    info = require_supported(info)
    usable = usable_mb(info)
    if need_mb > usable:
        names = list(alternatives(usable)) if alternatives else []
        hint = f"請到設定頁的模型管理改用 {'、'.join(names)}" if names else "請到設定頁的模型管理換成小一點的模型"
        raise RuntimeError(
            f"這張顯示卡放不下 {label}：需要約 {need_mb / 1024:.1f} GB，整張卡最多只能給模型用 {usable / 1024:.1f} GB，"
            f"重試也不會成功。{hint}。"
        )
    if info["free_mb"] < need_mb:
        raise RuntimeError(
            f"顯存不足：{label} 需要約 {need_mb / 1024:.1f} GB，目前只剩 {info['free_mb'] / 1024:.1f} GB。"
            f"請關閉其他使用顯卡的程式後按「重試」。"
        )
