"""第一次打開時自動安裝需要的模型和字典。

什麼時候開始：伺服器啟動時，拿到單一實例鎖的那個伺服器呼叫 startup()，不等網頁打開；網頁只顯示進度（GET /api/setup）。
新使用者：資料庫 settings 表沒有 setup，而且沒有影片、任務紀錄、模型檔、字典（existing_install）。判定一次就記住，
  之後刪光模型也不會再自動裝。擁有者的電腦（已經有模型、影片、字典）記成 existing，不會出現安裝畫面、不會下載任何東西。
不自動裝（off，不寫資料庫）：程式資料夾有 .dev 而且 VS_SETUP_AUTO 不是 1、VS_SETUP_AUTO=0、VS_NO_WORKERS=1。
選模型（choose）：預算 = used + free − config.SPEECH_WORKER["margin_mb"]，翻譯模型需要的顯存一律用
  config.translator_vram_mb（載入前實際檢查的同一個數字）。所有人都裝 tsqyomi、對齊器、Qwen3-ASR-1.7B；
  翻譯 Hy-MT2-7B 挑放得下的最好版本（8 GB 級 Q6_K、12 GB 以上 Q8_0），都放不下再用 Hy-MT2-1.8B。
  沒有 NVIDIA 顯示卡、驅動或架構太舊、顯存小於 8 GB、磁碟不夠：blocked，一個檔案都不下載，也不改模型設定。
排任務：模型下載照 tsqyomi → 對齊器 → Qwen3-ASR → 翻譯模型排進 model 執行緒（一次一個）；
  日文、英文字典排進 dict 執行緒（app/dict_build.py），跟模型下載同時跑。
中斷接續：下載的 .part 留著。監看執行緒（setup-monitor）在網路恢復後自動重試失敗的項目；磁碟滿了轉成 blocked，
  空間夠了自動接著裝；重開程式時失敗的項目立刻重試一次；使用者取消的項目不再自動排。
等模型的任務：安裝中新增的影片排隊等模型裝好（waiting_models；jobs.Worker.pick 先跳過、jobs.gpu_jobs 不算）。
狀態存在 settings 表的 setup（跟 jobs.QUEUE_PAUSED_KEY 一樣不在 settings.DEFAULTS 裡，PUT /api/settings 改不到），
只有這個模組會寫。API 在下面的 router（server.py include_router）。
"""
import hashlib
import json
import logging
import math
import os
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

from fastapi import APIRouter
from pydantic import BaseModel

from . import config, db, dict_build, gpu, instance, jobs, models, settings, syscheck
from .config import MODEL_CATALOG

log = logging.getLogger("setup")

SETTING_KEY = "setup"
MIB = 1024 * 1024
GIB = 1024 * MIB

RESERVE_MB = config.SPEECH_WORKER["margin_mb"]     # 1024，留給桌面、瀏覽器
TIGHT_MB = 500                  # 在「預算 − 500」以內算放得下，超過但還在預算內算很緊
MIN_TOTAL_MB = 7700             # 跟 launcher/precheck.py 的 MIN_VRAM_MB 一樣
DRIVER_WARN_BELOW = 595         # 580 到 594 版能用，但提醒翻譯開不起來時先更新驅動
BASE_MODELS = ("tsqyomi", "aligner", "qwen-asr")    # 也是下載順序
TRANSLATOR_ID, TRANSLATOR_KEY = "hymt2-7b", "hymt"
QUALITY_ORDER = ("Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M")
LOW_PARAMS = {"ctx": 4096, "parallel": 2}           # 每個 slot 一樣是 2048 token（打包計畫 6.4.2）
FALLBACK_ID, FALLBACK_KEY = "hymt2-1.8b", "hymt-mini"
DICTS = ("ja", "en")
DISK_WARN_BYTES = 10 * GIB      # 裝完剩不到這麼多時提醒

# 監看執行緒
TICK_S = 2
DISK_RECHECK_S = 60
NET_CODES = ("network", "timeout", "github")
NET_WAITS = (30, 60, 120)       # 連不上時隔多久再測一次（之後每 120 秒，不限次數）
NET_CHECK_TIMEOUT_S = 10
FIRST_WAIT_S = {"rate_limit": 300, "file_locked": 60, "unknown": 60}
RETRY_LIMIT = {"file_locked": 3, "corrupt": 1, "corrupt_source": 1, "unknown": 2}
NO_RETRY_CODES = ("ssl", "no_permission", "gated", "not_found", "changed", "no_7z", "dict_locked")

# 剩餘時間和進度
WARMUP_S = 10                   # 下載剛開始的這幾秒速度還不準（連線剛建立），先不估剩餘時間
SPEED_HOLD_S = 60               # 速度掉到 0（換下一個檔案、下一個模型）時沿用上一個值多久；下載都完成了就不沿用
BUILD_BPS = 8 * MIB             # 總進度裡字典建置一秒折成幾個位元組（研究時實測 Hugging Face 約 8 MB/s）

USE_TEXT = {"asr": "語音辨識", "aligner": "時間軸", "furigana": "日文假名", "translator": "翻譯",
            "dict_ja": "日文字典", "dict_en": "英文字典", "dict_zh": "中文辭典"}
STATE_RANK = {"downloading": 0, "waiting": 1, "net_wait": 2, "paused": 3, "failed": 4}

_lock = threading.RLock()        # 讀、改、存 setup 狀態
_plan_lock = threading.Lock()    # 偵測和排任務一次只做一個


class SetupError(Exception):
    """API 的錯誤：{"detail": 訊息, "code": 代碼, 其他欄位}。"""

    def __init__(self, http_status: int, code: str, message: str, **extra):
        super().__init__(message)
        self.status = http_status
        self.code = code
        self.extra = extra               # 例如 plan、status（回應裡照原樣帶著）


# ---------- 狀態 ----------

def load() -> dict | None:
    value = db.get_settings().get(SETTING_KEY)
    return value if isinstance(value, dict) else None


def _save(st: dict) -> dict:
    db.set_settings({SETTING_KEY: st})
    return st


def _new_state(status: str, reason: str) -> dict:
    return {"version": 1, "status": status, "reason": reason, "decided_at": time.time(), "dismissed": False,
            "gpu": None, "choice": None, "items": [], "block": None, "applied": {},
            "started_at": None, "finished_at": None}


def _update(**fields) -> dict | None:
    with _lock:
        st = load()
        if st is None:
            return None
        st.update(fields)
        return _save(st)


def auto_off(env=None, root=None) -> bool:
    """這個程式資料夾不自動安裝：VS_SETUP_AUTO=0，或有 .dev（擁有者的開發模式）而且 VS_SETUP_AUTO 不是 1。"""
    env = os.environ if env is None else env
    auto = (env.get("VS_SETUP_AUTO") or "").strip()
    if auto == "0":
        return True
    return auto != "1" and os.path.exists(os.path.join(str(root or config.ROOT), ".dev"))


def workers_running() -> bool:
    """這個伺服器有在跑佇列（拿到單一實例鎖、沒設 VS_NO_WORKERS）。"""
    return os.environ.get("VS_NO_WORKERS") != "1" and instance.lock_held()


def existing_install() -> str | None:
    """已經在用的電腦回傳理由（影片、任務、模型、字典），全新的回 None。全部唯讀，不遞迴算大小。"""
    if db._one("SELECT 1 AS x FROM media LIMIT 1"):
        return "media"
    if db._one("SELECT 1 AS x FROM jobs LIMIT 1"):
        return "jobs"
    for mid, entry in MODEL_CATALOG.items():
        try:
            if config.model_installed(entry):
                return f"model:{mid}"
            for path, is_dir in config.model_targets(entry):
                if (path.is_dir() and any(path.iterdir())) if is_dir else path.exists():
                    return f"model:{mid}"
        except OSError:
            return f"model:{mid}"            # 讀不到就當成有東西，寧可不自動裝
    for name in dict_build.DB_NAMES.values():
        if (dict_build.DICT_DIR / name).exists():
            return "dict"
    return None


def _install_blocked(st: dict | None) -> bool:
    """安裝途中磁碟滿了轉成的 blocked（項目已經排了，空間夠了接著裝），不是一開始就不能裝。"""
    return bool(st and st.get("status") == "blocked" and st.get("items")
                and (st.get("block") or {}).get("code") == "disk")


def startup():
    """伺服器拿到單一實例鎖、開好工作執行緒之後呼叫。只在這裡判斷新舊使用者。"""
    if os.environ.get("VS_NO_WORKERS") == "1":
        return
    if auto_off():
        # 不判斷新舊使用者、不自動排任務。但設定頁按「重新偵測並安裝建議模型」還沒裝完就重開的，照樣要監看：
        # 失敗的項目自動重試、磁碟空間夠了接著裝、全部裝好標記完成（不然會一直停在「安裝中」）
        st = load()
        if st and (st.get("status") == "installing" or _install_blocked(st)):
            log.info("first-run setup is off for this folder, but an install is still running: watching it")
            MONITOR.start(boot=True)
        else:
            log.info("first-run setup is off for this folder")
        return
    with _lock:
        st = load()
        if st is None:
            why = existing_install()
            if why:
                _save(_new_state("done", "existing"))
                log.info("existing install (%s): no first-run setup", why)
                return
            st = _save(_new_state("pending", "new"))
            log.info("new install: starting first-run setup")
    status = st.get("status")
    if status == "done":
        return
    MONITOR.start(boot=True)
    if status == "pending" or (status == "blocked" and not _install_blocked(st)):
        threading.Thread(target=plan_and_start, args=(st.get("reason") or "new",), name="setup-plan",
                         daemon=True).start()


# ---------- 選模型 ----------

def gpu_view(info: dict | None) -> dict | None:
    if not info:
        return None
    code = syscheck.problem_code(info)
    major = syscheck.driver_major(info.get("driver"))
    arch = info.get("arch")
    total = int(info.get("total_mb") or 0)
    return {"name": info.get("name"), "gb": round(total / 1024), "total_mb": total, "usable_mb": gpu.usable_mb(info),
            "driver": info.get("driver"), "cuda": info.get("cuda"),
            "arch": f"{arch[0]}.{arch[1]}" if arch and len(arch) >= 2 else None,
            "driver_warn": bool(major is not None and syscheck.MIN_DRIVER <= major < DRIVER_WARN_BELOW
                                and not (code and code[0] == "driver_old"))}


def _short_gpu_name(name) -> str:
    text = str(name or "NVIDIA")
    for prefix in ("NVIDIA ", "GeForce "):
        if text.startswith(prefix):
            text = text[len(prefix):]
    return text or "NVIDIA"


def _block(code: str, info: dict | None = None, **values) -> dict:
    info = info or {}
    if code == "driver_old":
        values = {"driver": info.get("driver"), "cuda": info.get("cuda"), **values}
    elif code == "arch_old":
        arch = info.get("arch") or []
        values = {"name": _short_gpu_name(info.get("name")),
                  "arch": f"{arch[0]}.{arch[1]}" if len(arch) >= 2 else None, **values}
    elif code == "vram_small":
        total = int(info.get("total_mb") or 0)
        values = {"gb": round(total / 1024), "total_mb": total, "usable_mb": gpu.usable_mb(info) if info else None,
                  **values}
    elif code == "gpu_hidden":
        values = {"value": syscheck.hidden_by_env(info), **values}
    block = {"code": code, "values": values}
    block.update(block_text(block))
    return block


def block_text(block: dict) -> dict:
    """不能自動安裝時畫面上的兩段話：原因、怎麼辦。"""
    code, v = block.get("code"), block.get("values") or {}
    if code == "no_gpu":
        return {"reason": "沒有找到 NVIDIA 顯示卡。轉字幕和翻譯要用 NVIDIA 顯示卡，顯存 8 GB 以上。",
                "fix": "有 NVIDIA 顯示卡的話，先裝好驅動再按「重新檢查」。"}
    if code == "driver_old":
        min_cuda = syscheck._cuda_text(syscheck.MIN_CUDA_VERSION)
        return {"reason": f"顯示卡驅動太舊（目前 {v.get('driver') or '未知'}，支援 CUDA {v.get('cuda') or '讀不到'}）。"
                          f"這個程式需要驅動 {syscheck.MIN_DRIVER} 版以上（支援 CUDA {min_cuda}）。",
                "fix": "到 NVIDIA 官網或 NVIDIA App 更新驅動，再按「重新檢查」。"}
    if code == "arch_old":
        return {"reason": f"這張顯示卡（{v.get('name') or 'NVIDIA'}）太舊，程式用不了。",
                "fix": "需要 RTX 20、GTX 16 系列或更新的 NVIDIA 顯示卡。"}
    if code == "vram_small":
        return {"reason": f"顯存只有 {v.get('gb')} GB，模型放不下。", "fix": "需要顯存 8 GB 以上的 NVIDIA 顯示卡。"}
    if code == "gpu_hidden":
        return {"reason": "環境變數 CUDA_VISIBLE_DEVICES 把顯示卡藏起來了。",
                "fix": "到 Windows 的環境變數拿掉它，再重新打開程式。"}
    if code == "disk":
        # 跟網頁（app.js 的 setupBlockText）同一種寫法：一律 GB 一位小數，要清出的空間無條件進位（清出這麼多一定夠）
        return {"reason": f"{v.get('drive') or '磁碟'} 空間不夠：要 {_gb(v.get('need_bytes'))}，"
                          f"只剩 {_gb(v.get('free_bytes'))}。",
                "fix": f"清出 {_gb(v.get('short_bytes'), up=True)} 以上再按「重新檢查」。"}
    return {"reason": "檢查電腦時出錯了。", "fix": "按「重新檢查」再試一次。一直這樣的話，把 data\\app.log 傳給作者。"}


def _gb(num_bytes, up: bool = False) -> str:
    """不能裝的畫面上的大小：1024 進位、GB 一位小數；up 時無條件進位。"""
    tenths = (max(0, int(num_bytes or 0)) / GIB) * 10          # 跟 app.js 同樣的算法，分界值也一樣
    tenths = math.ceil(tenths) if up else math.floor(tenths + 0.5)
    return f"{tenths / 10:.1f} GB"


def _choice(mid: str, variant: str | None, params: dict, vram: int, budget: int, fit: str) -> dict:
    entry = MODEL_CATALOG[mid]
    base = config.TRANSLATORS[entry["translator"]]
    return {"id": mid, "label": entry["label"], "variant": variant, "default_variant": entry.get("default_variant"),
            "ctx": params["ctx"], "parallel": params["parallel"],
            "low_params": (params["ctx"], params["parallel"]) != (base["ctx"], base["parallel"]),
            "vram_mb": int(vram), "budget_mb": int(budget), "fit": fit}


def choose(info: dict | None) -> tuple[dict | None, dict | None]:
    """照這張顯示卡挑翻譯模型：回 (choice, None)，不能裝時回 (None, block)。"""
    code = syscheck.problem_code(info)
    if code:
        return None, _block(code[0], info)
    usable = gpu.usable_mb(info)
    budget = usable - RESERVE_MB
    asr_need = max(config.model_vram_mb(MODEL_CATALOG["qwen-asr"]) or 0,
                   config.model_vram_mb(MODEL_CATALOG["aligner"]) or 0)
    if int(info.get("total_mb") or 0) < MIN_TOTAL_MB or budget < asr_need:
        return None, _block("vram_small", info)
    base = config.TRANSLATORS[TRANSLATOR_KEY]
    params_list = [{"ctx": base["ctx"], "parallel": base["parallel"]}, dict(LOW_PARAMS)]
    entry = MODEL_CATALOG[TRANSLATOR_ID]
    for v in QUALITY_ORDER:
        if v not in (entry.get("variants") or {}):
            continue
        needs = [(p, config.translator_vram_mb(TRANSLATOR_KEY,
                                               cfg=config.translator_cfg(TRANSLATOR_KEY, variant=v, params=p)))
                 for p in params_list]
        ok = [(p, n) for p, n in needs if n <= budget - TIGHT_MB]
        if ok:
            return _choice(TRANSLATOR_ID, v, ok[0][0], ok[0][1], budget, "ok"), None       # 先試預設參數
        fit = [(p, n) for p, n in needs if n <= budget]
        if fit:
            return _choice(TRANSLATOR_ID, v, fit[-1][0], fit[-1][1], budget, "tight"), None  # 很緊時用省顯存的參數
    cfg = config.translator_cfg(FALLBACK_KEY, params={})
    need = config.translator_vram_mb(FALLBACK_KEY, MODEL_CATALOG[FALLBACK_ID].get("size_mb"), cfg=cfg)
    if need <= budget:
        fit = "ok" if need <= budget - TIGHT_MB else "tight"
        return _choice(FALLBACK_ID, None, {"ctx": cfg["ctx"], "parallel": cfg["parallel"]}, need, budget, fit), None
    return None, _block("vram_small", info)


# ---------- 項目 ----------

def _model_item(mid: str, variant: str | None) -> dict:
    return {"key": f"model:{mid}" + (f":{variant}" if variant else ""), "kind": "model", "id": mid,
            "variant": variant, "job_id": None}


def _dict_item(lang: str) -> dict:
    return {"key": f"dict:{lang}", "kind": "dict", "lang": lang, "job_id": None}


def planned_items(choice: dict) -> list[dict]:
    return ([_model_item(mid, None) for mid in BASE_MODELS] + [_model_item(choice["id"], choice["variant"])]
            + [_dict_item(lang) for lang in DICTS])


def item_role(item: dict) -> str:
    if item["kind"] == "dict":
        return f"dict_{item['lang']}"
    return MODEL_CATALOG[item["id"]]["role"]


def item_label(item: dict) -> str:
    """模型是名稱（Hy-MT2-7B Q6_K）；字典是「日文字典」「英文字典」，來源名稱（JMdict、ECDICT）另外放在 source。"""
    if item["kind"] == "dict":
        return dict_build.LABELS[item["lang"]]
    return short_label(item["id"], item.get("variant"))


def short_label(mid: str, variant: str | None = None) -> str:
    """畫面上的模型名稱：tsqyomi（假名同形異音）只留 tsqyomi；有版本的加上版本。"""
    name = MODEL_CATALOG[mid]["label"].split("（")[0].strip()
    return f"{name} {variant}" if variant else name


def item_size(item: dict) -> int:
    if item["kind"] == "dict":
        return dict_build.source_bytes(item["lang"])
    entry = MODEL_CATALOG[item["id"]]
    if item.get("variant"):
        return int(entry["variants"][item["variant"]]["bytes"])
    return int(entry["size_mb"]) * MIB


def item_installed(item: dict) -> bool:
    if item["kind"] == "dict":
        return dict_build.ready(item["lang"])
    entry = MODEL_CATALOG.get(item["id"])
    if not entry:
        return False
    return config.model_installed(entry, item["variant"]) if item.get("variant") else config.model_installed(entry)


def item_need_bytes(item: dict) -> int:
    """還要下載多少（已經下載一部分的模型接著下載）。"""
    if item_installed(item):
        return 0
    if item["kind"] == "dict":
        return dict_build.source_bytes(item["lang"])
    return models.download_need_bytes(MODEL_CATALOG[item["id"]], item.get("variant"))


def _drive(path) -> str:
    full = os.path.abspath(str(path))
    return os.path.splitdrive(full)[0].upper() or os.path.dirname(full)


def disk_check(items: list[dict]) -> dict:
    """每顆磁碟要放得下還沒裝的模型（扣掉 .part 已經有的）和字典建置的峰值，再多留 models.DISK_MARGIN。
    回傳不夠的那顆磁碟；都夠時回裝完剩最少的那顆。"""
    groups: dict = {}
    for it in items:
        if item_installed(it):
            continue
        if it["kind"] == "dict":
            path, need = dict_build.DICT_DIR, dict_build.NEED_BYTES[it["lang"]]
        else:
            entry = MODEL_CATALOG[it["id"]]
            path, need = entry["dir"], models.download_need_bytes(entry, it.get("variant"))
        g = groups.setdefault(_drive(path), {"path": path, "need": 0})
        g["need"] += need
    if not groups:
        groups[_drive(config.MODELS_DIR)] = {"path": config.MODELS_DIR, "need": 0}
    best = None
    for drive, g in groups.items():
        free = int(models.disk_free(g["path"]))
        need = g["need"] + (models.DISK_MARGIN if g["need"] else 0)
        view = {"ok": free >= need, "drive": drive, "need_bytes": need, "free_bytes": free,
                "after_bytes": free - g["need"], "short_bytes": max(0, need - free),
                "warn": free - g["need"] < DISK_WARN_BYTES}
        if not view["ok"]:
            return view
        if best is None or view["after_bytes"] < best["after_bytes"]:
            best = view
    return best


def _disk_block(view: dict) -> dict:
    return _block("disk", None, drive=view["drive"], need_bytes=view["need_bytes"], free_bytes=view["free_bytes"],
                  short_bytes=view["short_bytes"])


def settings_changes(choice: dict) -> tuple[dict, list[dict]]:
    """選好的模型要寫進設定的值（一律經過 settings.save 驗證），和跟現在設定不同的說明。"""
    values = settings.all_values()
    new, changes = {}, []
    if choice.get("variant"):
        chosen = dict(values.get("model_variants") or {})
        before = settings.chosen_variant(choice["id"])
        chosen[choice["id"]] = choice["variant"]
        new["model_variants"] = chosen
        if before != choice["variant"]:
            changes.append({"key": "model_variants", "text": f"翻譯改用 {choice['variant']} 版"})
    if choice["id"] == TRANSLATOR_ID:
        params = dict(values.get("llm_params") or {})
        had = params.get(TRANSLATOR_KEY)
        if choice.get("low_params"):
            params[TRANSLATOR_KEY] = dict(LOW_PARAMS)
            if had != LOW_PARAMS:
                changes.append({"key": "llm_params", "text": "翻譯改成一次翻比較少行"})
        else:
            params.pop(TRANSLATOR_KEY, None)
            if had:
                changes.append({"key": "llm_params", "text": "翻譯參數改回預設"})
        new["llm_params"] = params
    if choice["id"] == FALLBACK_ID:
        default = dict(values.get("default_translator") or {})
        before = (default.get("ja"), default.get("en"))
        default.update(ja=FALLBACK_KEY, en=FALLBACK_KEY)
        new["default_translator"] = default
        if before != (FALLBACK_KEY, FALLBACK_KEY):
            changes.append({"key": "default_translator", "text": "翻譯改用 Hy-MT2-1.8B"})
    return new, changes


def make_plan(info: dict | None) -> dict:
    """偵測結果算出要裝什麼：gpu、choice、block、items（含 installed）、disk、changes、settings。不寫任何東西。"""
    g = gpu_view(info)
    choice, block = choose(info)
    if block:
        return {"gpu": g, "choice": None, "block": block, "items": [], "disk": None, "changes": [], "settings": {}}
    items = planned_items(choice)
    for it in items:
        it["installed"] = item_installed(it)
    disk = disk_check(items)
    if not disk["ok"]:
        block = _disk_block(disk)
    new_settings, changes = settings_changes(choice)
    return {"gpu": g, "choice": choice, "block": block, "items": items, "disk": disk, "changes": changes,
            "settings": new_settings}


def plan_signature(plan: dict) -> str:
    c = plan.get("choice") or {}
    data = {"choice": [c.get("id"), c.get("variant"), c.get("ctx"), c.get("parallel")] if c else None,
            "items": [[it["key"], bool(it.get("installed"))] for it in plan.get("items") or []],
            "changes": [f"{x['key']}:{x['text']}" for x in plan.get("changes") or []],
            "disk_ok": bool((plan.get("disk") or {}).get("ok")),
            "block": (plan.get("block") or {}).get("code")}
    return hashlib.sha1(json.dumps(data, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def plan_view(plan: dict) -> dict:
    """GET /api/setup/plan：設定頁「重新偵測並安裝建議模型」的確認框內容。"""
    items = []
    for it in plan["items"]:
        items.append({"key": it["key"], "kind": it["kind"], "label": item_label(it), "use": USE_TEXT[item_role(it)],
                      "role": item_role(it), "installed": bool(it.get("installed")), "need_bytes": item_need_bytes(it)})
    download = sum(i["need_bytes"] for i in items)
    return {"signature": plan_signature(plan), "gpu": plan["gpu"], "choice": plan["choice"], "block": plan["block"],
            "disk": plan["disk"], "items": items, "download_bytes": download, "changes": plan["changes"],
            "nothing": not plan["block"] and all(i["installed"] for i in items) and not plan["changes"]}


# ---------- 開始安裝 ----------

def _active_jobs() -> list[dict]:
    return [j for j in db.list_jobs(finished_limit=0) if j["status"] in ("queued", "running", "paused")]


def _job_variant(item: dict) -> str | None:
    entry = MODEL_CATALOG.get(item.get("id")) or {}
    return item.get("variant") if entry.get("variants") else None


def _enqueue_item(item: dict, active: list[dict]) -> str:
    """把一個項目排進佇列：同一個模型和版本（或同一本字典）已經在佇列裡（含暫停中）就沿用那個任務。回傳 job_id。"""
    if item["kind"] == "model":
        same = next((j for j in active if j["type"] == "model" and (j["params"] or {}).get("model") == item["id"]
                     and jobs.model_job_variant(j) == _job_variant(item)), None)
        if same:
            return same["id"]
        entry = MODEL_CATALOG[item["id"]]
        params = {"model": item["id"], "label": jobs.model_job_label(entry, item.get("variant")), "setup": True}
        if item.get("variant"):
            params["variant"] = item["variant"]
        return db.add_job(None, "model", params)
    same = next((j for j in active if j["type"] == "dict" and (j["params"] or {}).get("lang") == item["lang"]), None)
    if same:
        return same["id"]
    return db.add_job(None, "dict", {"lang": item["lang"], "label": dict_build.LABELS[item["lang"]], "setup": True,
                                     "force_download": False})


def enqueue(items: list[dict]) -> list[dict]:
    """照順序排任務，已經裝好的跳過。回傳要存進狀態的項目（had：開始前就已經有了）。"""
    active = _active_jobs()
    out = []
    for it in items:
        keep = {k: it[k] for k in ("key", "kind", "id", "variant", "lang") if k in it}
        keep["had"] = item_installed(it)
        keep["job_id"] = None if keep["had"] else _enqueue_item(it, active)
        out.append(keep)
    return out


def _start(plan: dict, reason: str) -> dict:
    applied = {}
    if plan["settings"]:
        settings.save(plan["settings"])
        applied = plan["settings"]
    items = enqueue(plan["items"])
    with _lock:
        st = load() or _new_state("installing", reason)
        if st.get("status") == "blocked" and reason == "new":
            st["dismissed"] = False      # 之前不能裝、按了關閉，後來（例如更新驅動後重開）可以裝了：再顯示一次進度頁
        st.update(status="installing", reason=reason, gpu=plan["gpu"], choice=plan["choice"], items=items, block=None,
                  applied=applied, started_at=time.time(), finished_at=None)
        _save(st)
    log.info("first-run setup (%s): %s, installing %s", reason, (plan["choice"] or {}).get("variant")
             or (plan["choice"] or {}).get("id"), [i["key"] for i in items if not i["had"]])
    MONITOR.reset()
    MONITOR.start()
    jobs.wake()
    return st


def _set_blocked(reason: str, g: dict | None, block: dict, choice: dict | None = None) -> dict:
    with _lock:
        st = load() or _new_state("blocked", reason)
        st.update(status="blocked", reason=reason, gpu=g, choice=choice, block=block, items=[], applied={},
                  started_at=None, finished_at=None)
        _save(st)
    log.warning("first-run setup blocked: %s %s", block.get("code"), block.get("values"))
    return st


class Busy(Exception):
    """另一個偵測或安裝正在進行。"""


def plan_and_start(reason: str = "new", raise_busy: bool = False) -> dict | None:
    """偵測顯示卡和磁碟，能裝就寫設定、排任務（status installing），不能裝就存 blocked。"""
    if not _plan_lock.acquire(blocking=False):
        if raise_busy:
            raise Busy()
        return load()
    try:
        try:
            plan = make_plan(gpu.query(max_age=0))
        except Exception:  # noqa: BLE001
            log.error("first-run setup: checking the computer failed", exc_info=True)
            return _set_blocked(reason, None, _block("detect_failed"))
        if plan["block"]:
            return _set_blocked(reason, plan["gpu"], plan["block"], plan["choice"])
        try:
            return _start(plan, reason)
        except Exception:  # noqa: BLE001
            log.error("first-run setup: starting the install failed", exc_info=True)
            return _set_blocked(reason, plan["gpu"], _block("detect_failed"), plan["choice"])
    finally:
        _plan_lock.release()


# ---------- 監看執行緒：自動重試、磁碟、完成 ----------

def net_ok(url: str, timeout: float = NET_CHECK_TIMEOUT_S) -> bool:
    """連得到這個主機（任何 HTTP 回應都算，包括 404）。"""
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "video-subtitle"})
    try:
        with urllib.request.urlopen(req, timeout=timeout):
            return True
    except urllib.error.HTTPError:
        return True
    except Exception:  # noqa: BLE001
        return False


def probe_url(item: dict) -> str:
    if item["kind"] == "model":
        endpoint = (os.environ.get("HF_ENDPOINT") or "https://huggingface.co").rstrip("/")
        return f"{endpoint}/api/models/Qwen/Qwen3-ASR-1.7B"
    first = dict_build.sources(item["lang"])[0]["urls"][0]
    parts = urllib.parse.urlsplit(first)
    return f"{parts.scheme}://{parts.netloc}/"


class Monitor:
    def __init__(self):
        self.event = threading.Event()
        self.thread: threading.Thread | None = None
        self.lock = threading.Lock()
        self.sched: dict[str, dict] = {}      # 項目 key → 自動重試的排程
        self.boot = False
        self.next_disk_check = 0.0
        self.speed = {"value": 0.0, "t": None, "nonzero_at": None, "since": None}

    def start(self, boot: bool = False):
        with self.lock:
            if boot:
                self.boot = True
            if self.thread is None or not self.thread.is_alive():
                self.thread = threading.Thread(target=self._loop, name="setup-monitor", daemon=True)
                self.thread.start()
        self.event.set()

    def wake(self):
        self.event.set()

    def reset(self):
        self.sched.clear()
        self.speed.update(value=0.0, t=None, nonzero_at=None, since=None)

    def forget(self, key: str):
        self.sched.pop(key, None)

    def _loop(self):
        while True:
            active = False
            try:
                active = self.tick()
            except Exception:  # noqa: BLE001
                log.warning("setup monitor failed", exc_info=True)
            self.event.wait(TICK_S if active else 60)
            self.event.clear()

    def pending_ids(self) -> set[str]:
        """監看執行緒排好了會自動重試的模型（還沒放棄的）。"""
        return {key.split(":")[1] for key, s in list(self.sched.items())
                if key.startswith("model:") and not s.get("give_up")}

    def schedule_of(self, key: str, job: dict | None) -> dict | None:
        s = self.sched.get(key)
        if s and job and s.get("sig") == (job["id"], job.get("finished_at")):
            return s
        return None

    # ----- 每 2 秒 -----

    def tick(self, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        st = load()
        if not st or st.get("status") not in ("installing", "blocked"):
            self.boot = False
            if st and st.get("status") == "done":
                self.sched.clear()
            return False
        if st["status"] == "blocked":
            self.boot = False
            if _install_blocked(st) and now >= self.next_disk_check:
                self.next_disk_check = now + DISK_RECHECK_S
                self.resume_after_disk(st)
            return True
        boot, self.boot = self.boot, False
        items = st.get("items") or []
        finished, disk_full, inst_speed = True, False, 0.0
        for it in items:
            job = db.get_job(it["job_id"]) if it.get("job_id") else None
            if item_installed(it):
                self.sched.pop(it["key"], None)
                continue
            status = job["status"] if job else None
            if status in ("queued", "running", "paused"):
                finished = False
                if status == "running":
                    inst_speed += float((jobs.download_progress.get(job["id"]) or {}).get("speed") or 0)
                continue
            if status != "failed":
                self.sched.pop(it["key"], None)           # 使用者取消、移除，或裝好後又刪掉：不再自動排
                continue
            finished = False
            code = (job.get("result") or {}).get("error_code") or "unknown"
            if code == "disk_full":
                disk_full = True
                self.sched[it["key"]] = {"sig": (job["id"], job.get("finished_at")), "job_id": job["id"],
                                         "code": code, "count": 0, "give_up": False, "disk": True, "next_at": now}
                continue
            s = self._schedule(it, job, code, now, boot)
            if not s["give_up"] and now >= s["next_at"]:
                self._attempt(it, job, s, now, boot)
        self._sample_speed(inst_speed, now)
        if disk_full:
            view = disk_check(items)
            with _lock:
                cur = load()
                if cur and cur.get("status") == "installing":
                    cur.update(status="blocked", block=_disk_block(view))
                    _save(cur)
            self.next_disk_check = now + DISK_RECHECK_S
            log.warning("first-run setup: disk is full, waiting for free space")
            return True
        if finished:
            self.finish()
            return False
        return True

    def _schedule(self, item: dict, job: dict, code: str, now: float, boot: bool) -> dict:
        sig = (job["id"], job.get("finished_at"))
        s = self.sched.get(item["key"])
        if s is not None and s.get("sig") == sig:
            return s
        count = s["count"] if (s and s.get("job_id") == job["id"] and s.get("code") == code) else 0
        s = {"sig": sig, "job_id": job["id"], "code": code, "count": count, "step": 0, "give_up": False,
             "next_at": now}
        # 不認得的代碼照 unknown：60 秒後重試，最多 2 次；網路類和 rate_limit 不限次數
        kind = code if (code in RETRY_LIMIT or code in NET_CODES or code == "rate_limit") else "unknown"
        if code in NO_RETRY_CODES:
            s["give_up"] = True
        elif not boot:                           # 重開程式時：可以重試的項目立刻重試一次，不等退避
            limit = RETRY_LIMIT.get(kind)
            if limit is not None and count >= limit:
                s["give_up"] = True
            s["next_at"] = now + FIRST_WAIT_S.get(kind, 0)
        self.sched[item["key"]] = s
        return s

    def _attempt(self, item: dict, job: dict, s: dict, now: float, boot: bool):
        if s["code"] in NET_CODES and not boot and not net_ok(probe_url(item)):
            s["next_at"] = now + NET_WAITS[min(s["step"], len(NET_WAITS) - 1)]
            s["step"] += 1
            return
        s["count"] += 1
        try:
            jobs.retry(job["id"])
            log.info("first-run setup: retrying %s (%s, attempt %d)", item["key"], s["code"], s["count"])
        except jobs.RetryRefused as e:
            dup = jobs.same_active_job(job)
            if dup:
                set_item_job(item["key"], dup["id"])
                self.sched.pop(item["key"], None)
            else:
                log.warning("first-run setup: retry of %s refused: %s", item["key"], e)
                s["give_up"] = True
        jobs.wake()

    def resume_after_disk(self, st: dict | None = None) -> bool:
        """安裝途中磁碟滿了：空間夠了就重試因為空間不夠失敗的項目，回到 installing。不夠時更新畫面上的數字。"""
        st = st or load()
        if not _install_blocked(st):
            return False
        items = st.get("items") or []
        view = disk_check(items)
        if not view["ok"]:
            _update(block=_disk_block(view))
            return False
        for it in items:
            job = db.get_job(it["job_id"]) if it.get("job_id") else None
            if job and job["status"] == "failed" and (job.get("result") or {}).get("error_code") == "disk_full":
                try:
                    jobs.retry(job["id"])
                except jobs.RetryRefused as e:
                    dup = jobs.same_active_job(job)
                    if dup:
                        set_item_job(it["key"], dup["id"])
                    else:
                        log.warning("first-run setup: retry of %s refused: %s", it["key"], e)
                self.sched.pop(it["key"], None)
        _update(status="installing", block=None)
        log.info("first-run setup: disk has space again, continuing")
        jobs.wake()
        self.event.set()
        return True

    def finish(self):
        with _lock:
            st = load()
            if not st or st.get("status") != "installing":
                return
            st.update(status="done", finished_at=time.time())
            _save(st)
        self.sched.clear()
        log.info("first-run setup finished")
        vocab = sys.modules.get(f"{__package__}.vocab")
        if vocab is not None:
            try:
                vocab.kick()
            except Exception:  # noqa: BLE001
                log.warning("vocab kick failed", exc_info=True)

    def _sample_speed(self, inst: float, now: float):
        """下載速度的 10 秒指數平均；速度掉到 0 時，SPEED_HOLD_S 內沿用上一個值。
        下載剛開始（或停了很久又開始）的前 WARMUP_S 秒直接用最新的速度，不跟連線剛建立時偏低的速度平均。"""
        sp = self.speed
        dt = max(0.001, now - sp["t"]) if sp["t"] is not None else 0.0
        if inst > 0:
            if sp["nonzero_at"] is None or now - sp["nonzero_at"] > SPEED_HOLD_S or sp["since"] is None:
                sp.update(value=inst, since=now)
            elif now - sp["since"] < WARMUP_S:
                sp["value"] = inst
            else:
                sp["value"] += (1 - math.exp(-dt / 10)) * (inst - sp["value"])
            sp["nonzero_at"] = now
        elif sp["nonzero_at"] is None or now - sp["nonzero_at"] > SPEED_HOLD_S:
            sp.update(value=0.0, since=None)
        sp["t"] = now

    def speed_bps(self, now: float | None = None) -> float:
        """算總剩餘時間用的下載速度。下載剛開始的前 WARMUP_S 秒還不準，回 0（畫面寫「計算中」）。"""
        now = time.monotonic() if now is None else now
        sp = self.speed
        if sp["value"] <= 0 or sp["since"] is None or now - sp["since"] < WARMUP_S:
            return 0.0
        return sp["value"]


MONITOR = Monitor()


def set_item_job(key: str, job_id: str):
    with _lock:
        st = load()
        if not st:
            return
        for it in st.get("items") or []:
            if it["key"] == key:
                it["job_id"] = job_id
        _save(st)


# ---------- 畫面用的狀態 ----------

def _part_done(item: dict, size: int) -> int:
    if item["kind"] != "model":
        return 0
    try:
        return max(0, min(size, size - models.download_need_bytes(MODEL_CATALOG[item["id"]], item.get("variant"))))
    except (OSError, KeyError):
        return 0


def _dict_eta(lang: str, info: dict, state: str) -> int:
    """這本字典大約還要幾秒。建立中照 dict_build.eta_s（下載看速度，建置看這台電腦的快慢）；
    排隊中、還不知道速度時用 dict_build.EST_S。"""
    if state == "waiting":
        return dict_build.EST_S[lang]
    if state not in ("downloading", "building", "checking", "retry_wait"):
        return 0
    eta = dict_build.eta_s(lang, info)
    return dict_build.EST_S[lang] if eta is None else eta


def _warm(job: dict | None) -> bool:
    """任務開始下載超過 WARMUP_S 秒了（剛開始的速度還不準，先不顯示剩餘時間）。"""
    started = (job or {}).get("started_at")
    return not started or time.time() - started >= WARMUP_S


def item_view(item: dict, now: float | None = None) -> dict:
    now = time.monotonic() if now is None else now
    role = item_role(item)
    size = item_size(item)
    job = db.get_job(item["job_id"]) if item.get("job_id") else None
    v = {"key": item["key"], "kind": item["kind"], "role": role, "use": USE_TEXT[role], "label": item_label(item),
         "size_bytes": size, "job_id": item.get("job_id"), "had": bool(item.get("had")),
         "state": "waiting", "done_bytes": 0, "speed_bps": 0, "eta_s": None, "retry": None, "error": None}
    if item["kind"] == "model":
        v.update(id=item["id"], variant=item.get("variant"))
    else:
        v.update(lang=item["lang"], source=dict_build.SOURCE_LABELS[item["lang"]])
    info = (jobs.download_progress.get(job["id"]) if job else None) or {}
    status = job["status"] if job else None
    if item_installed(item):
        v.update(state="done", done_bytes=size)
        if item["kind"] == "dict":
            v["progress"] = 1.0
        return v
    if status == "queued":
        v.update(state="waiting", done_bytes=_part_done(item, size))
        if item["kind"] == "dict":
            v["eta_s"] = _dict_eta(item["lang"], info, "waiting")
    elif status == "running":
        speed = float(info.get("speed") or 0)
        done = int(info.get("done") or 0)
        if item["kind"] == "model":
            state = "retry_wait" if info.get("retry") else ("checking" if info.get("checking") else "downloading")
            done = min(size, done or _part_done(item, size))
            v.update(state=state, done_bytes=done, speed_bps=int(speed),
                     eta_s=int((size - done) / speed) if speed > 0 and state == "downloading" and _warm(job) else None)
        else:
            stage = info.get("stage") or "download"
            if info.get("retry"):
                state = "retry_wait"
            elif stage == "download":
                state = "checking" if info.get("checking") else "downloading"
            else:
                state = "building"
            v.update(state=state, stage=stage, done_bytes=size if stage != "download" else min(size, done),
                     speed_bps=int(speed))
            if stage != "download":
                v["progress"] = round(float(info.get("progress") or 0), 4)     # 整本字典的進度（建置階段）
            v["eta_s"] = _dict_eta(item["lang"], info, state)
        if info.get("retry"):
            # 任務裡自動重試的倒數（jobs._wait_before_retry）：{"code", "attempt", "max_attempts", "retry_in_s"}
            v["retry"] = {**info["retry"], "in_s": info["retry"].get("retry_in_s")}
    elif status == "paused":
        v.update(state="paused", done_bytes=_part_done(item, size))
    elif status == "failed":
        code = (job.get("result") or {}).get("error_code") or "unknown"
        v["done_bytes"] = _part_done(item, size)
        v["error"] = {"code": code, "message": job.get("error")}
        s = MONITOR.schedule_of(item["key"], job)
        auto = code not in NO_RETRY_CODES and code != "disk_full"
        if s is not None and s.get("give_up"):
            auto = False
        if auto:
            wait = max(0, int((s["next_at"] - now) + 0.999)) if s else None
            if code in NET_CODES or code == "rate_limit":
                v["state"] = "net_wait"
                v["error"]["next_check_s"] = wait
            else:
                v["state"] = "retry_wait"
                v["retry"] = {"code": code, "retry_in_s": wait, "in_s": wait}
        else:
            v["state"] = "failed"
    else:
        v["state"] = "canceled"            # 使用者取消、移除了任務，或裝好後又刪掉
    return v


def ready_view(st: dict | None) -> dict:
    cat = MODEL_CATALOG
    choice = (st or {}).get("choice")
    if choice and choice.get("id") in cat:
        translate = config.model_installed(cat[choice["id"]])
    else:
        translate = bool(settings.installed_translators())
    return {"transcribe": config.model_installed(cat["qwen-asr"]) and config.model_installed(cat["aligner"]),
            "translate": translate, "dict_ja": dict_build.ready("ja"), "dict_en": dict_build.ready("en")}


def _build_weight(v: dict) -> tuple[float, float]:
    """字典建置折成進度的位元組：(總共, 已經做了)。建置的秒數（作者電腦上）× BUILD_BPS，照建置做了幾成算。"""
    if v["kind"] != "dict":
        return 0.0, 0.0
    weight = dict_build.build_total_s(v["lang"]) * BUILD_BPS
    if v["state"] == "done":
        return weight, weight
    if v["state"] == "building":
        return weight, weight * dict_build.build_fraction(v["lang"], float(v.get("progress") or 0))
    return weight, 0.0


def _totals(views: list[dict], now: float) -> dict:
    """總進度那一行：done_bytes / total_bytes 是真的下載量；速度、剩餘時間；percent 另外把字典建置的時間算進去，
    模型都下載完、英文字典還在建時進度條才不會停在 99%。"""
    counted = [v for v in views if not v["had"] and v["state"] != "canceled"]
    total = sum(v["size_bytes"] for v in counted)
    done = sum(min(v["size_bytes"], v["done_bytes"]) for v in counted)
    left = sum(v["size_bytes"] - min(v["size_bytes"], v["done_bytes"]) for v in counted)
    unfinished = [v for v in counted if v["state"] != "done"]
    # 都下載完了（只剩字典在建）就不再沿用上一個速度，畫面不會還寫著 MB/s
    speed = MONITOR.speed_bps(now) if left else 0.0
    if not left:
        eta_bytes = 0
    elif speed > 0:
        eta_bytes = int(left / speed)
    else:
        eta_bytes = None                     # 剛開始下載、等網路、暫停：不知道要多久
    dict_eta = sum(int(v.get("eta_s") or 0) for v in counted if v["kind"] == "dict" and v["state"] != "done")
    eta = None if eta_bytes is None else max(eta_bytes, dict_eta)
    weights = [_build_weight(v) for v in counted]
    work_total = total + sum(w for w, _ in weights)
    work_done = done + sum(d for _, d in weights)
    percent = int(work_done * 100 / work_total) if work_total else 100
    if unfinished:
        percent = min(percent, 99)
    net_wait = any(v["state"] == "net_wait" for v in views)
    return {"done_bytes": done, "total_bytes": total, "speed_bps": int(speed), "eta_s": eta, "percent": percent,
            "net_wait": net_wait}


def status_view(st: dict | None = None, now: float | None = None) -> dict:
    """GET /api/setup：完整狀態，不寫任何東西。"""
    st = load() if st is None else st
    workers = workers_running()
    off = auto_off()
    if st is None:
        return {"status": "off" if (off or not workers) else "pending", "reason": None, "dismissed": False,
                "workers": workers, "off": off, "gpu": None, "choice": None, "block": None, "disk": None,
                "totals": None, "ready": None, "items": [], "started_at": None, "finished_at": None}
    now = time.monotonic() if now is None else now
    status = st.get("status")
    views = [item_view(it, now) for it in st.get("items") or []]
    out = {"status": status, "reason": st.get("reason"), "dismissed": bool(st.get("dismissed")), "workers": workers,
           "off": off, "gpu": st.get("gpu"), "choice": st.get("choice"), "block": st.get("block"), "disk": None,
           "totals": None, "ready": None, "items": views, "started_at": st.get("started_at"),
           "finished_at": st.get("finished_at")}
    if st.get("choice"):
        choice = dict(st["choice"])
        if choice.get("id") in MODEL_CATALOG:
            choice["label"] = MODEL_CATALOG[choice["id"]]["label"]
        out["choice"] = choice
    if views:
        out["totals"] = _totals(views, now)
        if status in ("installing", "blocked") or (status == "done" and st.get("reason") != "existing"):
            try:
                disk = disk_check(st["items"])
                out["disk"] = {k: disk[k] for k in ("drive", "need_bytes", "free_bytes", "after_bytes", "warn")}
            except OSError:
                pass
    if status != "done" or st.get("reason") != "existing":
        out["ready"] = ready_view(st)
    return out


def summary_view() -> dict:
    """GET /api/state 的 setup 摘要（右上角的小狀態用）。安裝中才算進度，其他時候很便宜。"""
    st = load()
    if st is None:
        return {"status": "off" if (auto_off() or not workers_running()) else "pending", "reason": None,
                "dismissed": False, "percent": None, "eta_s": None, "net_wait": False, "problem": False,
                "blocked": False, "ready": None}
    status = st.get("status")
    out = {"status": status, "reason": st.get("reason"), "dismissed": bool(st.get("dismissed")), "percent": None,
           "eta_s": None, "net_wait": False, "problem": False, "blocked": status == "blocked", "ready": None}
    if status in ("installing", "blocked"):
        full = status_view(st)
        totals = full["totals"] or {}
        out.update(percent=totals.get("percent"), eta_s=totals.get("eta_s"), net_wait=bool(totals.get("net_wait")),
                   problem=any(v["state"] == "failed" for v in full["items"]),
                   ready={k: full["ready"][k] for k in ("transcribe", "translate")} if full["ready"] else None)
    return out


# ---------- 等模型的任務 ----------

def pending_models(job_list: list[dict] | None = None) -> set[str]:
    """還沒裝好、但是正在下載或會自動接著下載的模型 id：有 model 任務在排隊、下載中或暫停（任何版本、使用者自己按的也算），
    或監看執行緒排好了自動重試。"""
    job_list = db.list_jobs(finished_limit=0) if job_list is None else job_list
    ids = {(j.get("params") or {}).get("model") for j in job_list
           if j["type"] == "model" and j["status"] in ("queued", "running", "paused")}
    ids |= MONITOR.pending_ids()
    return {mid for mid in ids if mid in MODEL_CATALOG and not config.model_installed(MODEL_CATALOG[mid])}


def model_pending(mid: str | None) -> bool:
    return bool(mid) and mid in pending_models()


def required_models(job: dict) -> list[str]:
    """任務一定要用的模型（型錄 id）：轉字幕是辨識模型和對齊器，檢查時間軸是 Qwen3-ASR 和對齊器，翻譯類是翻譯模型。"""
    kind, p = job["type"], job.get("params") or {}
    if kind == "transcribe":
        ids = [jobs.catalog_id("engine", p.get("engine")), jobs.catalog_id("role", "aligner")]
    elif kind == "health":
        ids = [jobs.catalog_id("engine", "qwen"), jobs.catalog_id("role", "aligner")]
    elif kind in ("translate", "titles", "glosses"):
        key = p.get("translator") or (settings.translator_for("ja") if kind == "titles" else None)
        ids = [jobs.catalog_id("translator", key)]
    else:
        ids = []
    return [i for i in ids if i]


def waiting_models(job: dict, pending: set[str] | None = None) -> list[str]:
    """任務需要、還沒裝好、而且會裝好的模型；空的代表不用等。"""
    pending = pending_models() if pending is None else pending
    if not pending:
        return []
    return [mid for mid in required_models(job) if mid in pending]


def _model_wait(mid: str, job_list: list[dict]) -> tuple[str, int | None]:
    mine = [j for j in job_list if j["type"] == "model" and (j.get("params") or {}).get("model") == mid
            and j["status"] in ("queued", "running", "paused")]
    running = next((j for j in mine if j["status"] == "running"), None)
    if running:
        info = jobs.download_progress.get(running["id"]) or {}
        retry = info.get("retry")
        if retry and retry.get("code") in NET_CODES:
            return "net_wait", None
        speed, done, total = float(info.get("speed") or 0), int(info.get("done") or 0), int(info.get("total") or 0)
        ok = speed > 0 and total and not info.get("checking") and _warm(running)
        return "downloading", (int((total - done) / speed) if ok else None)
    if any(j["status"] == "queued" for j in mine):
        return "waiting", None
    if mine:
        return "paused", None
    codes = [s.get("code") for key, s in list(MONITOR.sched.items()) if key.split(":")[1:2] == [mid]]
    return ("net_wait" if any(c in NET_CODES or c == "rate_limit" for c in codes) else "failed"), None


def annotate_waiting(job_list: list[dict]) -> list[dict]:
    """/api/state 的 jobs：排隊中、在等模型的顯卡任務加上 waiting（等哪些模型、最慢那個的狀態、大約還要多久）。"""
    queued = [j for j in job_list if j["status"] == "queued" and j["type"] in jobs.GPU_TYPES]
    if not queued:
        return job_list
    pending = pending_models(job_list)
    if not pending:
        return job_list
    for j in queued:
        mids = waiting_models(j, pending)
        if not mids:
            continue
        states = [_model_wait(mid, job_list) for mid in mids]
        worst = max(states, key=lambda s: STATE_RANK.get(s[0], 0))
        etas = [e for _, e in states]
        j["waiting"] = {"models": mids, "labels": [short_label(m) for m in mids], "state": worst[0],
                        "eta_s": max(etas) if etas and all(e is not None for e in etas) else None}
    return job_list


# ---------- API ----------

router = APIRouter()


class InstallReq(BaseModel):
    signature: str = ""


class DictBuildReq(BaseModel):
    force_download: bool = False


def _require_workers():
    if not workers_running():
        raise SetupError(503, "no_workers", "這個視窗不處理佇列，請用最先打開的那個")


def _block_error(block: dict, **extra) -> SetupError:
    code = "disk" if block.get("code") == "disk" else "blocked"
    return SetupError(400, code, f"{block.get('reason', '')}{block.get('fix', '')}", **extra)


def _find_item(key: str) -> dict:
    st = load()
    item = next((it for it in (st or {}).get("items") or [] if it["key"] == key), None)
    if item is None:
        raise SetupError(404, "unknown_item", "找不到這一項")
    return item


@router.get("/api/setup")
def api_status():
    return status_view()


@router.post("/api/setup/dismiss")
def api_dismiss():
    with _lock:
        st = load()
        if st is not None and not st.get("dismissed"):
            st["dismissed"] = True
            _save(st)
    return {"ok": True}


@router.post("/api/setup/recheck")
def api_recheck():
    _require_workers()
    st = load()
    if not st or st.get("status") not in ("pending", "blocked"):
        raise SetupError(409, "not_blocked", "現在不用重新檢查")
    if _install_blocked(st):
        if not MONITOR.resume_after_disk(st):
            raise _block_error(load()["block"], status=status_view())
        return status_view()
    try:
        result = plan_and_start(st.get("reason") or "new", raise_busy=True)
    except Busy:
        raise SetupError(409, "busy", "正在檢查，請稍等")
    if result and result.get("status") == "blocked":
        raise _block_error(result["block"], status=status_view())
    return status_view()


@router.get("/api/setup/plan")
def api_plan():
    try:
        return plan_view(make_plan(gpu.query(max_age=0)))
    except Exception:  # noqa: BLE001
        log.error("setup plan: checking the computer failed", exc_info=True)
        raise SetupError(500, "detect_failed", "檢查電腦時出錯了")


@router.post("/api/setup/install")
def api_install(req: InstallReq | None = None):
    _require_workers()
    st = load()
    if st and st.get("status") == "installing":
        raise SetupError(409, "busy", "已經在安裝了")
    if not _plan_lock.acquire(blocking=False):
        raise SetupError(409, "busy", "已經在安裝了")
    try:
        try:
            plan = make_plan(gpu.query(max_age=0))
        except Exception:  # noqa: BLE001
            log.error("setup install: checking the computer failed", exc_info=True)
            raise SetupError(500, "detect_failed", "檢查電腦時出錯了")
        view = plan_view(plan)
        if not req or req.signature != view["signature"]:
            raise SetupError(409, "plan_changed", "顯示卡或檔案有變，請再看一次", plan=view)
        if plan["block"]:
            raise _block_error(plan["block"], plan=view)
        if all(i["installed"] for i in view["items"]):
            if plan["settings"] and plan["changes"]:
                settings.save(plan["settings"])
            return {"ok": True, "nothing": True, "changes": plan["changes"]}
        _start(plan, "redetect")
    finally:
        _plan_lock.release()
    return status_view()


@router.post("/api/setup/items/{key}/retry")
def api_item_retry(key: str):
    _require_workers()
    item = _find_item(key)
    if item_installed(item):
        raise SetupError(409, "active", "這一項已經裝好了")
    job = db.get_job(item["job_id"]) if item.get("job_id") else None
    if job and job["status"] in ("queued", "running"):
        raise SetupError(409, "active", "這一項已經在下載")
    if job and job["status"] == "paused":
        raise SetupError(409, "active", "這一項暫停中，按「繼續」接著下載")
    if item["kind"] == "model":
        try:
            models.check_disk(item["id"], variant=_job_variant(item))
        except models.ModelError as e:
            raise SetupError(e.status, "disk" if e.status == 400 else "retry_refused", str(e))
    else:
        short = dict_build.disk_problem(item["lang"])
        if short:
            raise SetupError(400, "disk", f"{_drive(dict_build.DICT_DIR)} 空間不夠：要 {models.fmt_size(short[0])}，"
                                          f"只剩 {models.fmt_size(short[1])}")
    job_id = None
    if job and job["status"] in ("failed", "canceled"):
        try:
            jobs.retry(job["id"])
            job_id = job["id"]
        except jobs.RetryRefused as e:
            dup = jobs.same_active_job(job)
            if dup:
                job_id = dup["id"]
            elif e.status == 400:
                raise SetupError(400, "retry_refused", str(e))
    if job_id is None:
        job_id = _enqueue_item(item, _active_jobs())
    MONITOR.forget(key)
    with _lock:
        st = load()
        for it in st.get("items") or []:
            if it["key"] == key:
                it["job_id"] = job_id
        if st.get("status") == "done":
            st.update(status="installing", finished_at=None)
        _save(st)
    MONITOR.start()
    jobs.wake()
    return status_view()


@router.post("/api/setup/items/{key}/resume")
def api_item_resume(key: str):
    _require_workers()
    item = _find_item(key)
    job = db.get_job(item["job_id"]) if item.get("job_id") else None
    if not job or job["status"] != "paused" or not jobs.resume(job["id"]):
        raise SetupError(409, "not_paused", "這一項沒有暫停")
    MONITOR.wake()
    return status_view()


@router.post("/api/dicts/{lang}/build")
def api_dict_build(lang: str, req: DictBuildReq | None = None):
    """設定頁、單字頁的〔建立〕〔重建〕：排一個 dict 任務。中文辭典也用這個（CKIP 模型另外用模型下載）。"""
    if lang not in dict_build.LANGS:
        raise SetupError(400, "bad_lang", "沒有這種字典")
    _require_workers()
    same = [j for j in db.list_jobs(finished_limit=0) if j["type"] == "dict" and j["status"] in ("queued", "running")
            and (j.get("params") or {}).get("lang") == lang]
    if same:
        text = "這本字典正在建立" if any(j["status"] == "running" for j in same) else "這本字典已經在排隊建立"
        raise SetupError(409, "duplicate", text)
    short = dict_build.disk_problem(lang)
    if short:
        raise SetupError(400, "disk", f"{_drive(dict_build.DICT_DIR)} 空間不夠：要 {models.fmt_size(short[0])}，"
                                      f"只剩 {models.fmt_size(short[1])}")
    job_id = db.add_job(None, "dict", {"lang": lang, "label": dict_build.LABELS[lang], "setup": False,
                                       "force_download": bool(req and req.force_download)})
    jobs.wake()
    return {"ok": True, "job_id": job_id}


def dict_job_view(lang: str, job_list: list[dict] | None = None) -> dict | None:
    """/api/dicts 每本字典的 job：排隊或建立中的任務；沒有時是最近一次失敗的（字典還沒建好時才列）。"""
    job_list = db.list_jobs() if job_list is None else job_list
    mine = [j for j in job_list if j["type"] == "dict" and (j.get("params") or {}).get("lang") == lang]
    job = next((j for j in mine if j["status"] == "running"), None) or next(
        (j for j in mine if j["status"] == "queued"), None)
    if job is None and not dict_build.ready(lang):
        failed = [j for j in mine if j["status"] == "failed"]
        job = max(failed, key=lambda j: j.get("finished_at") or 0) if failed else None
    if job is None:
        return None
    info = jobs.download_progress.get(job["id"]) or {}
    out = {"id": job["id"], "status": job["status"], "progress": job.get("progress") or 0, "stage": job.get("stage"),
           "step": info.get("stage")}
    if job["status"] == "failed":
        out["error"] = {"code": (job.get("result") or {}).get("error_code"), "message": job.get("error")}
    return out
