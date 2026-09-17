"""模型管理：佔用空間、顯存門檻、刪除前的影響說明、刪除、下載前的磁碟檢查。

設定頁的模型管理用這裡（server.py 的 /api/settings、/api/models/...），下載任務開始前也用這裡檢查磁碟。

刪除只動 config.model_targets 列出的、這個模型自己的檔案，全部經過 safepath：
  hf、onnx、gguf  models/<分類>/<模型> 整個資料夾（不能是 models 本身，也不能是 asr、gguf 這種分類資料夾）
  ckpt           models/sep 是共用資料夾，只刪同名的 .ckpt、.yaml 和下載到一半的 .part
  parts          models/ckip/<子模型> 各自的資料夾
沒安裝但留著下載到一半的檔案（殘檔）也用同一套刪除。

有版本（量化）的翻譯模型（config 型錄的 variants）：各版本放在同一個資料夾。
  指定版本刪除時只刪那個版本的 gguf、.part 和舊版 huggingface_hub 留下的紀錄、暫存檔，完成標記裡拿掉那個版本；
  刪的是最後一個還有檔案的版本時，整個資料夾一起刪。沒指定版本就是整個模型。

跟佇列的配合：
  正在跑的任務用到的模型不能刪（409）。「用到」包含任務參數一定會用的，以及任務途中決定要用、自己登記的
  （jobs.use_model：轉字幕時的時間軸檢查、人聲分離）。
  刪除時拿著 jobs.claim_lock，刪的途中不會有新任務開始，任務也不能登記要用模型；
  排隊中的任務輪到時，會在載入模型前發現模型不在，用「找不到模型」失敗，不會自動下載。
  翻譯模型：llama-server 會跨任務沿用，閒置但還載著這個 gguf 時先關掉再刪
  （正在翻譯的任務一定是「執行中」，前面就擋掉了，所以只會關到沒人在用的）。
  辨識、對齊模型：常駐語音程序（app/speech.py）會跨任務沿用，還載著時先卸載再刪；
  它正在跑別的任務、等不到時先記成不能再用，下一個請求一定先卸載（它也會發現檔案變了，不會沿用舊的）。
  先刪權重（config.model_key_files），刪不掉就整個停下來，模型保持完整；權重刪掉後剩下刪不掉的算殘檔。
"""
import contextlib
import json
import logging
import os
import shutil
import threading
import time
from pathlib import Path

from . import config, db, jobs, safepath, settings
from .config import (ASR_ENGINES, COMPLETE_MARKER, DOWNLOAD_MARKER, HF_LOCAL_CACHE, LANGUAGES, MODEL_CATALOG,
                     MODELS_DIR, PART_SUFFIX, TRANSLATORS, model_disk_bytes, model_installed, model_key_files,
                     model_targets, model_vram_mb, tree_bytes)

log = logging.getLogger("models")

MIB = 1024 * 1024
GIB = 1024 * MIB
DISK_MARGIN = GIB                 # 下載完磁碟至少還要留這麼多
GROUP_DIRS = {"asr", "gguf", "furigana", "ckip", "sep"}   # models 底下的分類資料夾，不能整個刪

_delete_lock = threading.Lock()


class ModelError(RuntimeError):
    """訊息可以直接顯示給使用者；status 是 HTTP 狀態碼。"""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


def fmt_size(num_bytes: int) -> str:
    """1024 進位（跟 Windows 檔案總管一樣）。1 GB 以上一位小數，以下用 MB。"""
    if num_bytes >= GIB - MIB // 2:
        return f"{num_bytes / GIB:.1f} GB"
    if num_bytes <= 0:
        return "0 MB"
    return f"{max(1, round(num_bytes / MIB))} MB"


def _entry(mid: str) -> dict:
    entry = MODEL_CATALOG.get(mid)
    if not entry:
        raise ModelError(404, "沒有這個模型")
    return entry


def check_variant(entry: dict, variant: str | None, default: bool = False) -> str | None:
    """API 帶來的版本名稱：沒有版本的模型帶了版本、或不認得的版本，丟 ModelError 400。
    有版本的模型沒帶版本：default=True（下載）時用設定選的版本（沒選用預設版本），否則回 None（整個模型）。"""
    if not variant:
        if default and entry.get("variants"):
            chosen = config._user_choices()[0].get(entry.get("id"))
            return chosen if chosen in entry["variants"] else entry["default_variant"]
        return None
    try:
        return config.check_variant(entry, variant)
    except ValueError as e:
        raise ModelError(400, str(e))


def active_variant(entry: dict) -> str | None:
    """有版本的模型現在要用哪個版本（設定選的，沒下載就改用已下載的）；沒有版本回 None。
    跟 settings.variant_for 一樣，但資料庫還沒開時也能用（當成沒選）。"""
    if not entry.get("variants"):
        return None
    return config.pick_variant(entry, config._user_choices()[0].get(entry.get("id")))


def _installed(entry: dict, variant: str | None) -> bool:
    return model_installed(entry, variant) if variant else model_installed(entry)


def _label(entry: dict, variant: str | None) -> str:
    return f"{entry['label']} {variant}" if variant else entry["label"]


def _mid(field: str, value) -> str | None:
    if value is None:
        return None
    return next((mid for mid, e in MODEL_CATALOG.items() if e.get(field) == value), None)


# ---------- 空間與顯存 ----------

def disk_free(path=None) -> int:
    p = Path(path or MODELS_DIR)
    while not p.exists() and p.parent != p:
        p = p.parent
    return shutil.disk_usage(p).free


def storage_summary() -> dict:
    used = tree_bytes(MODELS_DIR) if MODELS_DIR.is_dir() else 0
    free = disk_free(MODELS_DIR)
    return {"dir": str(MODELS_DIR), "drive": MODELS_DIR.drive or str(MODELS_DIR.anchor),
            "used_bytes": used, "used": fmt_size(used), "free_bytes": free, "free": fmt_size(free)}


def gpu_usable_mb(info: dict | None) -> int | None:
    """顯卡上模型最多能拿到的顯存：總顯存扣掉驅動程式保留的部分。

    nvidia-smi 的 memory.free = total − reserved − used，所以 used + free 就是扣掉保留之後的量
    （例如 16 GB 的卡：16303 總共、306 保留，used + free = 15997）。舊驅動沒有分開回報保留量時就等於總顯存。
    載入前檢查（gpu.ensure_free、gpu.pick_batch）比的是 free，free 永遠不會超過這個數字，
    所以門檻比它大的模型在這張卡上一定跑不起來。不用當下的 free 比，因為那會隨其他程式開關變動。
    """
    if not info:
        return None
    return int(info["used_mb"]) + int(info["free_mb"])


def license_limit(text: str | None) -> str | None:
    """型錄 license 文字有沒有限制：開頭是「非商用」回 "noncommercial"，「授權不明」回 "unknown"，其他回 None。
    設定頁用它決定授權文字要不要用醒目的顏色。沒寫授權的也當成不明。"""
    text = (text or "").strip()
    if not text or text.startswith("授權不明"):
        return "unknown"
    if text.startswith("非商用"):
        return "noncommercial"
    return None


def variant_row(mid: str, name: str, gpu_info: dict | None, active: str | None = None) -> dict:
    """設定頁模型列底下每個版本的一行：大小、顯存、放不放得下、有沒有裝好、是不是正在用的版本。"""
    entry = MODEL_CATALOG[mid]
    view = config.variant_view(entry, name)
    installed = model_installed(entry, name)
    disk = model_disk_bytes(entry, name)
    size = int(view["bytes"])
    vram = model_vram_mb(entry, name)
    usable = gpu_usable_mb(gpu_info)
    return {
        "name": name, "installed": installed,
        "active": installed and name == active, "default": name == entry.get("default_variant"),
        "size_bytes": size, "size": fmt_size(size),
        "disk_bytes": disk, "disk": fmt_size(disk),
        "leftover": not installed and disk > 0,
        "vram_mb": vram, "vram": fmt_size(vram * MIB) if vram else None,
        # 門檻是照「gguf 大小 + overhead」估計的，還沒實測（型錄 vram: estimate）
        "vram_estimate": view.get("vram") != "measured",
        "fits": None if (vram is None or not usable) else vram <= usable,
        "repo": view.get("repo"), "file": view["file"], "path": str(entry["dir"] / view["file"]),
        "license_url": view.get("license_url") or entry.get("license_url"),
    }


def row(mid: str, gpu_info: dict | None) -> dict:
    """設定頁每個模型列要顯示的空間、顯存、授權資訊。gpu_info 是 gpu.query() 的結果（讀不到顯卡時 None）。
    有版本的模型：大小、顯存是現在要用的版本的，variants 是每個版本的一行（variant_row）。"""
    entry = MODEL_CATALOG[mid]
    installed = model_installed(entry)
    disk = model_disk_bytes(entry)
    active = active_variant(entry)
    size = int(config.variant_view(entry, active)["bytes"]) if active else int(entry["size_mb"]) * MIB
    # 載入前實際檢查的門檻（config.model_vram_mb）；在 CPU 上跑的是 None
    vram = model_vram_mb(entry, active)
    usable = gpu_usable_mb(gpu_info)
    if entry.get("parts"):
        path = entry["dir"]
    elif entry.get("kind") == "ckpt":
        path = entry["dir"] / entry["file"]
    else:
        path = entry["dir"]
    extra = {}
    if entry.get("variants"):
        extra = {"variant": active,
                 "variants": [variant_row(mid, n, gpu_info, active) for n in config.variant_names(entry)]}
    return {
        **extra,
        "installed": installed,
        "disk_bytes": disk, "disk": fmt_size(disk),
        "size_bytes": size, "size": fmt_size(size),
        "leftover": not installed and disk > 0,
        "vram_mb": vram, "vram": fmt_size(vram * MIB) if vram else None,
        # 辨識、對齊依剩餘顯存決定一次處理幾段（batch_by_free_mb），門檻是最低需求，顯存越大越快
        "vram_min": entry.get("role") in ("asr", "aligner"),
        # 跟整張卡能用的顯存比（不是當下剩餘的）：整張卡都不夠就一定跑不了
        "fits": None if (vram is None or not usable) else vram <= usable,
        "path": str(path),
        # 授權（config.MODEL_CATALOG 的 license、license_url）；型錄沒寫的顯示「授權不明」
        "license": entry.get("license") or "授權不明",
        "license_url": entry.get("license_url"),
        "license_limit": license_limit(entry.get("license")),
    }


# ---------- 任務用到哪些模型 ----------

def _song_track(track_id) -> bool:
    t = db.get_track(track_id) if track_id else None
    return bool(t) and (t.get("model") or "").endswith(jobs.SONG_LABEL)


def _gloss_key(params: dict, without=()) -> str | None:
    """handle_glosses 實際會用的翻譯模型（指定的不在就換 glossgen.translator_key）。"""
    from . import glossgen
    key = params.get("translator")
    installed = settings.installed_translators(without=without) if without else settings.installed_translators()
    if key in TRANSLATORS and key in installed:
        return key
    return glossgen.translator_key(without)


def running_uses(job: dict) -> set[str]:
    """執行中的任務用到的模型（型錄 id），這些不能刪。

    兩個來源合起來：
      1. 任務參數一定會用到的（轉字幕的辨識模型和對齊模型、翻譯模型……），任務剛開始還沒登記時也擋得住。
      2. 任務跑的途中才決定要不要用的（轉字幕時的時間軸檢查、人聲分離），由任務自己在決定的當下登記
         （jobs.use_model，跟刪除共用 claim_lock）。不看「現在的」設定去猜，任務開始後使用者改設定也不會算錯。
    """
    p = job.get("params") or {}
    t = job["type"]
    out = set(jobs.uses_of(job["id"]))
    if t == "transcribe":
        out |= {_mid("engine", p.get("engine")), _mid("role", "aligner")}
    elif t == "health":
        out |= {_mid("engine", "qwen"), _mid("role", "aligner")}
        if _song_track(p.get("track_id")):
            out.add(_mid("role", "separator"))
    elif t == "translate":
        out.add(_mid("translator", p.get("translator")))
    elif t == "titles":
        out.add(_mid("translator", p.get("translator") or settings.translator_for("ja")))
    elif t == "glosses":
        out.add(_mid("translator", _gloss_key(p)))
    out.discard(None)
    return out


def _other_installed_variants(entry: dict, variant: str | None) -> list[str]:
    if not variant:
        return []
    return [n for n in config.installed_variants(entry) if n != variant]


def in_use_variants(entry: dict) -> set[str]:
    """有版本的模型，任務可能正在用的版本：現在設定要用的版本，和 llama-server 還載著的那個版本。"""
    out = {active_variant(entry)}
    server = jobs._llm.get("server")
    gguf = (getattr(server, "cfg", None) or {}).get("gguf") if jobs._llm.get("key") == entry.get("translator") else None
    for n in config.variant_names(entry):
        if gguf and jobs._same_path(gguf, entry["dir"] / entry["variants"][n]["file"]):
            out.add(n)
    out.discard(None)
    return out


def queued_failing(mid: str, variant: str | None = None) -> list[dict]:
    """排隊中的任務裡，刪掉這個模型（或版本）後輪到時會失敗的。
    刪的是版本、而且還有別的版本裝好時，任務會改用別的版本，不會失敗。"""
    entry = _entry(mid)
    if _other_installed_variants(entry, variant):
        return []
    role = entry.get("role")
    tkey = entry.get("translator")
    out = []
    for j in db.list_jobs():
        if j["status"] != "queued":
            continue
        p = j.get("params") or {}
        t = j["type"]
        fails = False
        if t == "transcribe":
            fails = role == "aligner" or mid == _mid("engine", p.get("engine"))
        elif t == "health":
            fails = (role == "aligner" or entry.get("engine") == "qwen"
                     or (role == "separator" and _song_track(p.get("track_id"))))
        elif t == "translate" and tkey:
            fails = p.get("translator") == tkey
        elif t == "titles" and tkey:
            if p.get("translator"):                  # server.translate_titles 排的都有指定
                fails = p["translator"] == tkey
            else:                                    # 沒指定時輪到才用日文的預設（刪掉後改用後備的）
                after = settings.translator_for("ja", without={tkey})
                fails = (settings.translator_for("ja") == tkey
                         and after not in settings.installed_translators(without={tkey}))
        elif t == "glosses" and tkey:
            fails = _gloss_key(p) == tkey and _gloss_key(p, without={tkey}) is None
        if fails:
            out.append(j)
    return out


def busy_reason(mid: str, variant: str | None = None) -> str | None:
    """現在不能刪的原因（下載中、正在用），可以刪回 None。
    指定版本時：同一個版本在下載佇列裡（含暫停）不能刪；別的版本正在下載時也先不刪（兩邊都要改完成標記）；
    正在跑的任務用到這個模型時，只擋可能正在用的版本（in_use_variants）。"""
    entry = _entry(mid)
    for j in db.list_jobs():
        status = j["status"]
        if status not in ("queued", "running", "paused"):
            continue
        if j["type"] == "model" and (j.get("params") or {}).get("model") == mid:
            job_variant = jobs.model_job_variant(j)
            if variant is None or job_variant is None or job_variant == variant:
                if status == "running":
                    return "這個模型正在下載，請先取消下載任務"
                if status == "paused":
                    return "這個模型的下載暫停中，請先取消下載任務"
                return "這個模型在下載佇列裡，請先取消下載任務"
            if status == "running":
                return f"正在下載 {entry['label']} 的其他版本，等下載完或暫停後再刪"
            continue
        if status == "running" and mid in running_uses(j):
            if variant is None or variant in in_use_variants(entry) or not _other_installed_variants(entry, variant):
                return "正在用這個模型，等任務跑完再刪"
    if entry.get("role") == "segmenter":
        from . import vocab_zh
        if vocab_zh.status().get("running"):
            return "正在用這個模型整理中文單字，等一下再刪"
    return None


# ---------- 刪除前的說明 ----------

def _group_lines(by_after: dict, found: str, none: str, names_suffix=None) -> list[str]:
    lines = []
    for after, langs in by_after.items():
        names = "、".join(LANGUAGES[x]["label"] for x in langs)
        if names_suffix:
            names += names_suffix(langs)
        lines.append(found.format(names=names, after=after) if after else none.format(names=names))
    return lines


def effects(mid: str, variant: str | None = None) -> list[str]:
    """刪掉這個已安裝的模型（或版本）會怎樣，照目前的預設模型和後備邏輯實際算。
    刪的是版本、而且還有別的版本裝好時，只會換版本；刪的是最後一個版本就跟刪掉整個模型一樣。"""
    entry = _entry(mid)
    others = _other_installed_variants(entry, variant)
    if others:
        active = active_variant(entry)
        if active != variant:
            return ["目前沒有用這個版本，刪掉不影響翻譯"]
        after = config.pick_variant(entry, config._user_choices()[0].get(mid), without={variant})
        return [f"現在用的就是這個版本，刪掉後改用 {after}"]
    role = entry.get("role")
    lines = []
    if role == "asr":
        eng = entry["engine"]
        by_after: dict = {}
        for lang in LANGUAGES:
            if eng not in settings.installed_engines(lang) or settings.engine_for(lang) != eng:
                continue
            after = settings.engine_for(lang, without={eng})
            ok = after in settings.installed_engines(lang, without={eng})
            by_after.setdefault(ASR_ENGINES[after]["label"] if ok else None, []).append(lang)
        lines += _group_lines(by_after, "{names}影片預設用它辨識，刪掉後改用 {after}",
                              "{names}影片預設用它辨識，刪掉後沒有辨識模型，要再下載才能轉字幕")
        if not lines:
            lines.append("目前沒有語言預設用它辨識")
        if eng == "qwen":
            lines.append("檢查時間軸也要用它，刪掉後轉完字幕不會再檢查，也不能手動檢查" if settings.get("health_check")
                         else "檢查時間軸要用它，刪掉後不能手動檢查時間軸")
    elif role == "aligner":
        lines.append("所有影片都靠它產生字幕時間軸，刪掉後沒辦法轉字幕，也不能檢查時間軸")
    elif role == "translator":
        key = entry["translator"]
        by_after = {}
        for lang in LANGUAGES:
            if key not in settings.installed_translators(lang) or settings.translator_for(lang) != key:
                continue
            after = settings.translator_for(lang, without={key})
            ok = after in settings.installed_translators(lang, without={key})
            by_after.setdefault(TRANSLATORS[after]["label"] if ok else None, []).append(lang)
        # 翻譯影片標題用的是日文的預設翻譯模型（server.translate_titles）
        lines += _group_lines(by_after, "{names}預設用它翻譯，刪掉後改用 {after}",
                              "{names}預設用它翻譯，刪掉後沒有翻譯模型，只能轉字幕不能翻譯",
                              names_suffix=lambda langs: "影片和影片標題" if "ja" in langs else "影片")
        if not lines:
            lines.append("目前沒有語言預設用它翻譯")
        lines += _gloss_lines(key)
    elif role == "furigana":
        lines.append("日文假名要用它：重新啟動程式後，新的日文字幕不會標假名（已經標好的會保留）")
    elif role == "separator":
        if settings.get("song_mode") != "off":
            lines.append("刪掉後歌曲、配樂很滿的影片不會先分離人聲，辨識可能比較不準")
        n = sum(1 for t in db.list_tracks() if t["kind"] == "asr" and (t.get("model") or "").endswith(jobs.SONG_LABEL))
        if n:
            lines.append(f"有 {n} 條字幕是分離人聲後做的，刪掉後這幾條不能再檢查時間軸")
    elif role == "segmenter":
        lines.append("單字頁的中文分頁要用它，刪掉後不能用，重新下載就會恢復")
    return lines


def _gloss_lines(key: str) -> list[str]:
    from . import dict_ja, glossgen
    if key not in ("hymt", "hymt-mini") or not dict_ja.ready() or glossgen.translator_key() != key:
        return []
    after = glossgen.translator_key(without={key})
    if not after:
        return ["刪掉後沒辦法再產生日文單字的中文解釋"]
    label = TRANSLATORS[after]["label"]
    # 翻譯完自動產生只在用 Hy-MT2-7B 翻譯時才排（glossgen.maybe_enqueue_after_pipeline）
    if key == "hymt" and settings.get("auto_glosses"):
        return [f"翻譯完不會再自動產生日文單字的中文解釋，手動產生時改用 {label}"]
    return [f"日文單字的中文解釋改用 {label} 產生"]


def impact(mid: str, variant: str | None = None) -> dict:
    """刪除確認視窗的內容。variant：只刪這個版本。"""
    entry = _entry(mid)
    variant = check_variant(entry, variant)
    installed = _installed(entry, variant)
    lines, queued = [], 0
    if installed:
        lines = effects(mid, variant)
        queued = len(queued_failing(mid, variant))
        if queued:
            lines.append(f"佇列裡有 {queued} 個排隊中的任務要用它，刪掉後這些任務會失敗")
    targets, _whole = delete_targets(entry, variant)
    disk = _targets_bytes(targets)
    return {
        "id": mid, "variant": variant, "label": _label(entry, variant), "installed": installed,
        "disk_bytes": disk, "disk": fmt_size(disk),
        "lines": lines, "queued": queued, "busy": busy_reason(mid, variant),
        "paths": [str(p) for p, _ in targets if p.exists()],
    }


# ---------- 刪除 ----------

def _real(p) -> Path | None:
    try:
        return Path(p).resolve()
    except (OSError, RuntimeError, ValueError, TypeError):
        return None


def _same(a, b) -> bool:
    return a is not None and b is not None and os.path.normcase(str(a)) == os.path.normcase(str(b))


def delete_targets(entry: dict, variant: str | None) -> tuple[list[tuple[Path, bool]], bool]:
    """要刪哪些：(model_targets, 是不是整個模型)。指定版本、但別的版本都沒有檔案（沒裝也沒下載到一半）時，
    整個資料夾一起刪（連同完成標記、舊版 huggingface_hub 的 .cache）。"""
    if variant and entry.get("variants"):
        others = [n for n in config.variant_names(entry) if n != variant and model_disk_bytes(entry, n) > 0]
        if others:
            return model_targets(entry, variant), False
    return model_targets(entry), True


def _targets_bytes(targets) -> int:
    total = 0
    for path, is_dir in targets:
        try:
            if is_dir:
                if path.is_dir() and not path.is_symlink():
                    total += tree_bytes(path)
            elif path.is_file() and not path.is_symlink():
                total += path.stat().st_size
        except OSError:
            continue
    return total


def _checked_targets(entry: dict, targets=None) -> list[tuple[Path, bool]]:
    """要刪的每一項都要在 models 裡、至少在分類資料夾底下一層、不是分類資料夾、不是捷徑，
    而且只能是型錄 dir 本身（hf、onnx、gguf）或 dir 底下直接的一項（ckpt 的檔案、parts 的子資料夾、版本的 gguf），
    或 dir\\.cache\\huggingface\\download 底下直接的一項（版本的舊下載紀錄），
    files 裡寫了 ..\\別的模型 這種也擋掉。有一項不對就整個不刪。"""
    root = _real(MODELS_DIR)
    own = _real(entry["dir"])
    own_cache = _real(Path(entry["dir"]) / HF_LOCAL_CACHE)
    out = []
    for path, is_dir in (model_targets(entry) if targets is None else targets):
        path = Path(path)
        real = _real(path)
        ok = root is not None and real is not None and safepath.inside(path, MODELS_DIR)
        if ok:
            rel = Path(os.path.relpath(real, root)).parts
            ok = len(rel) >= 2 and ".." not in rel and not (is_dir and path.name.lower() in GROUP_DIRS)
        if ok:
            parent = _real(path.parent)
            ok = _same(real, own) if path == Path(entry["dir"]) else (
                _same(parent, own) or (not is_dir and _same(parent, own_cache)))
        if ok and path.exists() and safepath.is_link(path):
            ok = False
        if not ok:
            log.warning("refused to delete model files %s (%s)", path, entry.get("label"))
            raise ModelError(400, "模型資料夾的位置不對，沒有刪除")
        out.append((path, is_dir))
    return out


def _left_files(targets) -> int:
    n = 0
    for path, is_dir in targets:
        if is_dir:
            if path.is_dir():
                n += sum(len(files) for _, _, files in os.walk(path))
        elif path.exists():
            n += 1
    return n


RETRY_WAITS = (0.5, 1.0, 1.5)      # 剛關掉的 llama-server 有時要一下子才放開檔案


def _key_files(entry: dict, targets, variant: str | None = None) -> list[Path]:
    """決定「已安裝」的檔案，而且一定在已經檢查過的刪除範圍裡（不在範圍裡的不碰）。"""
    out = []
    for f in model_key_files(entry, variant):
        for path, is_dir in targets:
            if (_same(f, path) if not is_dir else safepath.inside(f, path)) and f not in out:
                out.append(f)
    return out


def _unlink_all(files: list[Path]) -> list[Path]:
    """刪掉這些檔案，刪不掉的等一下再試；回傳最後還在的。"""
    for wait in (*RETRY_WAITS, None):
        for f in files:
            safepath.safe_unlink(f, MODELS_DIR)
        files = [f for f in files if f.exists()]
        if not files or wait is None:
            return files
        time.sleep(wait)
    return files


def _remove(entry: dict, targets, variant: str | None = None) -> bool:
    """先刪模型權重（決定「已安裝」的檔案），刪不掉就停下來回 False，其他檔案一個都不動，模型保持完整能用。
    權重刪掉之後再刪其他檔案；這時候就算有檔案被鎖住，模型也已經顯示成沒安裝，剩下的算殘檔，可以再按「清掉」。"""
    if _unlink_all(_key_files(entry, targets, variant)):
        return False
    for wait in (*RETRY_WAITS, None):
        for path, is_dir in targets:
            if is_dir:
                safepath.safe_rmtree(path, MODELS_DIR)
            else:
                safepath.safe_unlink(path, MODELS_DIR)
        if not _left_files(targets) or wait is None:
            break
        time.sleep(wait)
    return True


def _release_llm_holding(key: str, label: str, gguf: Path | None = None):
    """llama-server 還載著這個翻譯模型（閒置中）就關掉，放開 gguf 檔案。gguf：只刪一個版本時，載著的是那個檔案才關。

    只有 _llm["key"] 是這個模型時才需要 _llm_lock。別的翻譯模型正在載入時 get_llm 會一直拿著這把鎖（最久 300 秒），
    所以不死等：等到鎖就自己關；等不到但 key 已經換掉（get_llm 換模型時會先關掉舊的）就不用做了。
    呼叫前已經拿著 claim_lock、也確認過沒有執行中的任務用到這個模型，所以這段時間不會有人重新載入它。
    """
    def holding() -> bool:
        if jobs._llm["key"] != key:
            return False
        if gguf is None:
            return True
        loaded = (getattr(jobs._llm.get("server"), "cfg", None) or {}).get("gguf")
        return loaded is None or jobs._same_path(loaded, gguf)

    while holding():
        if jobs._llm_lock.acquire(timeout=0.2):
            try:
                if holding():
                    jobs.release_llm(f"刪除模型 {label}")
            finally:
                jobs._llm_lock.release()
            return


def _forget_markers(entry: dict, variant: str | None):
    """刪掉模型（或版本）後，完成標記裡拿掉它的紀錄（同一個檔案裡其他模型、版本的保留），
    沒有其他紀錄就刪掉標記檔；整個模型刪掉時，下載到一半的紀錄 .download.json 也刪。"""
    d = Path(entry["dir"])
    path = d / COMPLETE_MARKER
    data = config._read_json(path)
    models = data.get("models")
    key = entry.get("id") or entry.get("label")
    if isinstance(models, dict) and key in models:
        if variant and isinstance(models[key], dict):
            (models[key].get("variants") or {}).pop(variant, None)
            if not models[key].get("variants"):
                models.pop(key, None)
        else:
            models.pop(key, None)
        if models:
            tmp = path.with_name(path.name + ".tmp")
            tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
            os.replace(tmp, path)
        else:
            safepath.safe_unlink(path, MODELS_DIR)
    if not variant and config._read_json(d / DOWNLOAD_MARKER).get("model") == key:
        safepath.safe_unlink(d / DOWNLOAD_MARKER, MODELS_DIR)


def delete(mid: str, variant: str | None = None) -> dict:
    """刪掉模型（或指定的版本）自己的檔案（已安裝的模型或下載到一半的殘檔都一樣）。"""
    entry = _entry(mid)
    variant = check_variant(entry, variant)
    raw_targets, whole = delete_targets(entry, variant)
    targets = _checked_targets(entry, raw_targets)
    key_variant = None if whole else variant
    label = _label(entry, variant)
    before = _targets_bytes(targets)
    was_installed = _installed(entry, variant)
    with contextlib.ExitStack() as stack:
        stack.enter_context(_delete_lock)
        stack.enter_context(jobs.claim_lock)        # 刪的途中不會有新任務開始，任務也不能在這時候登記要用模型
        reason = busy_reason(mid, variant)
        if reason:
            raise ModelError(409, reason)
        if entry.get("role") == "segmenter":
            from . import vocab_zh
            if not vocab_zh._run_lock.acquire(timeout=3):
                raise ModelError(409, "正在用這個模型整理中文單字，等一下再刪")
            stack.callback(vocab_zh._run_lock.release)
        if entry.get("translator"):
            gguf = Path(entry["dir"]) / entry["variants"][variant]["file"] if key_variant else None
            _release_llm_holding(entry["translator"], label, gguf)
        if entry.get("role") in ("asr", "aligner"):
            # 常駐語音程序還載著它（上一個任務留下的）就先卸載，不然刪掉檔案後記憶體裡的舊模型還會被拿來用
            jobs.release_speech(mid, label)
        weights_gone = _remove(entry, targets, key_variant)
        if weights_gone:
            _forget_markers(entry, key_variant)
        left = _left_files(targets)
    after = _targets_bytes(targets)
    if _installed(entry, variant):
        log.warning("model %s: weights are locked, nothing deleted", label)
        raise ModelError(409, "模型檔案正被其他程式開著，沒有刪除，模型還能照常使用。"
                              "關掉開著它的程式後再刪一次")
    if left:
        log.warning("model %s: %d files left (weights removed: %s)", label, left, weights_gone)
        head = "模型已經不能用了，但" if was_installed else ""
        raise ModelError(409, f"{head}還有 {left} 個檔案刪不掉，可能正被其他程式開著，佔用 {fmt_size(after)}。"
                              "關掉開著它的程式後按「清掉」")
    log.info("deleted model %s (%s)", label, fmt_size(before))
    return {"ok": True, "freed": fmt_size(max(0, before - after))}


# ---------- 下載前檢查磁碟 ----------

def download_need_bytes(entry: dict, variant: str | None = None) -> int:
    """還要下載多少（已經下載一部分的會接著下載）。有版本的模型沒指定版本時算預設版本。"""
    if entry.get("variants"):
        view = config.variant_view(entry, variant)
        if model_installed(entry, view["variant"]):
            return 0
        part = Path(entry["dir"]) / (view["file"] + PART_SUFFIX)
        try:
            have = part.stat().st_size if part.is_file() else 0
        except OSError:
            have = 0
        return max(0, int(view["bytes"]) - have)
    return max(0, int(entry["size_mb"]) * MIB - model_disk_bytes(entry))


def check_disk(mid: str, include_queue: bool = True, variant: str | None = None):
    """磁碟剩餘空間要放得下這個模型、佇列裡其他還沒下載完的模型（含暫停中的），再多留 DISK_MARGIN；不夠丟 ModelError。"""
    entry = _entry(mid)
    variant = check_variant(entry, variant, default=True)
    need = download_need_bytes(entry, variant)
    others = 0
    if include_queue:
        for j in db.list_jobs():
            other = (j.get("params") or {}).get("model")
            if j["type"] != "model" or j["status"] not in ("queued", "running", "paused") or other not in MODEL_CATALOG:
                continue
            other_variant = jobs.model_job_variant(j)
            if other == mid and other_variant == variant:
                continue
            others += download_need_bytes(MODEL_CATALOG[other], other_variant)
    free = disk_free(entry["dir"])
    if free >= need + others + DISK_MARGIN:
        return
    drive = Path(entry["dir"]).drive or str(MODELS_DIR)
    msg = f"磁碟空間不夠：{_label(entry, variant)} 要 {fmt_size(need)}"
    if others:
        msg += f"，佇列裡其他模型還要 {fmt_size(others)}"
    msg += f"，{drive} 只剩 {fmt_size(free)}。下載完至少要留 {fmt_size(DISK_MARGIN)}，請先清出空間再下載"
    raise ModelError(400, msg)
