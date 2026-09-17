"""使用者設定：預設模型、yt-dlp cookies、佇列行為。存在資料庫的 settings 表。"""
import shutil
from pathlib import Path

from . import db
from .config import (ASR_ENGINES, MODEL_CATALOG, TRANSLATORS, default_engine, default_translator,
                     llm_params_problem, model_installed, pick_variant)

DEFAULTS = {
    "default_engine": {},      # {"ja": "qwen"}，沒設就用內建預設
    "default_translator": {},  # {"ja": "hymt"}
    "cookies_browser": "",     # "" | chrome | edge | firefox | brave | vivaldi | opera
    "cookies_file": "",        # cookies.txt 路徑，優先於瀏覽器
    "group_by_model": True,    # 佇列盡量把同一個模型的任務排在一起
    "translation_mode": "line",  # line 逐行對照（每行各自翻、保持原文語序）｜ sentence 整句通順
    "auto_glosses": True,      # 日文影片用 Hy-MT2 翻譯完，順便產生單字的中文釋義
    "health_check": True,      # 轉完字幕後用 Qwen3-ASR 檢查時間軸，對不上的段落重新對齊（app/health.py）
    "song_mode": "auto",       # auto：原音幾乎偵測不到人聲（歌曲、配樂很滿）時先分離人聲｜ off 不分離
    # 讓同一個網路的手機、平板連線。關著時只開本機 127.0.0.1，區網連進來一律回 403（關掉馬上生效）；
    # 打開要重新啟動程式才會開始聽區網（server.bind_host）。有設 VS_HOST 環境變數時綁哪裡照它
    "lan_access": False,
    # 翻譯模型用哪個版本（量化）：{"hymt2-7b": "Q6_K"}，key 是型錄 id。沒設用型錄的 default_variant；
    # 選的版本沒下載時改用已下載的其他版本（config.pick_variant）
    "model_variants": {},
    # llama-server 的參數：{"hymt": {"ctx": 4096, "parallel": 2}}，key 是翻譯模型 key。沒設用 config.TRANSLATORS 的
    "llm_params": {},
}


def all_values() -> dict:
    return {**DEFAULTS, **db.get_settings()}


def get(key: str):
    return all_values().get(key)


def peek(key: str, db_path=None):
    """資料庫還沒開（伺服器啟動前決定要綁哪個位址）時唯讀地讀一個設定；讀不到回預設值。"""
    import json
    import sqlite3
    path = Path(db_path or db.DB_PATH)
    if not path.is_file():
        return DEFAULTS.get(key)
    try:
        # 不用 mode=ro：WAL 模式的資料庫在 -shm 檔不在時，唯讀連線會打不開
        con = sqlite3.connect(path, timeout=5)
        try:
            row = con.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
        finally:
            con.close()
        return json.loads(row[0]) if row else DEFAULTS.get(key)
    except (sqlite3.Error, ValueError):
        return DEFAULTS.get(key)


COOKIE_BROWSERS = ("", "chrome", "edge", "firefox", "brave", "vivaldi", "opera")
COOKIES_MAX_BYTES = 20 * 1024 * 1024


def cookies_file_problem(path: str) -> str | None:
    """cookies.txt 路徑有問題就回傳說明，沒問題回 None。"""
    p = Path(path)
    if not p.is_absolute():
        return "cookies.txt 請填完整路徑，例如 E:\\cookies.txt"
    if not p.is_file():
        return "找不到這個 cookies.txt 檔案"
    if p.stat().st_size > COOKIES_MAX_BYTES:
        return "這個檔案太大，看起來不是 cookies.txt"
    return None


def save(values: dict):
    """只存認得的設定；格式不對丟 ValueError（訊息可以直接顯示給使用者）。"""
    clean = {k: v for k, v in values.items() if k in DEFAULTS}
    if "cookies_browser" in clean and clean["cookies_browser"] not in COOKIE_BROWSERS:
        raise ValueError("不支援這個瀏覽器")
    if "lan_access" in clean and not isinstance(clean["lan_access"], bool):
        raise ValueError("區網連線的設定只能是開或關")
    if "model_variants" in clean:
        chosen = clean["model_variants"]
        if not isinstance(chosen, dict):
            raise ValueError("模型版本的設定格式不對")
        for mid, name in chosen.items():
            entry = MODEL_CATALOG.get(mid)
            if not entry or not entry.get("variants"):
                raise ValueError(f"{mid} 沒有可以選的版本")
            if name not in entry["variants"]:
                raise ValueError(f"{entry['label']} 沒有 {name} 這個版本")
    if "llm_params" in clean:
        params = clean["llm_params"]
        if not isinstance(params, dict):
            raise ValueError("翻譯模型參數的設定格式不對")
        for key, value in params.items():
            problem = llm_params_problem(key, value)
            if problem:
                raise ValueError(problem)
    if "cookies_file" in clean:
        raw = clean["cookies_file"]
        path = raw.strip().strip('"').strip("'").strip() if isinstance(raw, str) else None
        if path is None:
            raise ValueError("cookies.txt 路徑格式不對")
        if path:
            problem = cookies_file_problem(path)
            if problem:
                raise ValueError(problem)
        clean["cookies_file"] = path
    if clean:
        db.set_settings(clean)


def _pick(mapping: dict, language: str):
    # 舊版設定用「ja:general」當 key，一併讀進來
    return mapping.get(language) or mapping.get(f"{language}:general")


def installed_engines(language: str | None = None, without=()) -> list[str]:
    """已下載的辨識模型。without：當成已經刪掉的引擎 key（算「刪掉後改用哪個」用）。"""
    out = []
    for key, cfg in ASR_ENGINES.items():
        entry = next((e for e in MODEL_CATALOG.values() if e.get("engine") == key), None)
        if key in without or (entry and not model_installed(entry)):
            continue
        if language and language not in cfg["langs"]:
            continue
        out.append(key)
    return out


def variant_for(mid: str, without=()) -> str | None:
    """有版本的模型現在要用哪個版本（設定 model_variants 選的，沒下載就改用已下載的）；沒有版本的模型回 None。
    without：當成已經刪掉的版本。"""
    entry = MODEL_CATALOG.get(mid)
    if not entry or not entry.get("variants"):
        return None
    return pick_variant(entry, (all_values().get("model_variants") or {}).get(mid), without)


def chosen_variant(mid: str) -> str | None:
    """設定裡選的版本（不管有沒有下載），沒選回預設版本；沒有版本的模型回 None。下載沒指定版本時用這個。"""
    entry = MODEL_CATALOG.get(mid)
    if not entry or not entry.get("variants"):
        return None
    chosen = (all_values().get("model_variants") or {}).get(mid)
    return chosen if chosen in entry["variants"] else entry["default_variant"]


def installed_translators(language: str | None = None, without=()) -> list[str]:
    out = []
    for key, cfg in TRANSLATORS.items():
        entry = next((e for e in MODEL_CATALOG.values() if e.get("translator") == key), None)
        if key in without or (entry and not model_installed(entry)):
            continue
        if language and language not in cfg["langs"]:
            continue
        out.append(key)
    return out


def _installed(fn, language, without):
    # 沒有 without 時照原本的呼叫方式（測試會把 installed_* 換成只收 language 的假函式）
    return fn(language, without=without) if without else fn(language)


def engine_for(language: str, profile: str | None = None, without=()) -> str:
    chosen = _pick(all_values()["default_engine"], language) or default_engine(language)
    available = _installed(installed_engines, language, without)
    if chosen in available:
        return chosen
    fallback = default_engine(language)
    if fallback in available:
        return fallback
    return available[0] if available else chosen


def translator_for(language: str, profile: str | None = None, without=()) -> str | None:
    if language == "zh":
        return None
    chosen = _pick(all_values()["default_translator"], language) or default_translator(language)
    available = _installed(installed_translators, language, without)
    if chosen in available:
        return chosen
    fallback = default_translator(language)
    if fallback in available:
        return fallback
    return available[0] if available else chosen


def ytdlp_cookie_opts(copy_to: Path) -> dict:
    """給 yt-dlp 的 cookies 設定，解決 YouTube 的機器人驗證。

    yt-dlp 下載結束時會把 cookies 寫回 cookiefile 指定的檔案（整檔重寫，路徑不存在還會新建），
    所以不把使用者的 cookies.txt 直接交給它，而是先複製一份到 copy_to（任務的工作資料夾），給它複本。
    複本用完由呼叫端刪掉。瀏覽器 cookies 是 yt-dlp 自己複製到暫存資料夾讀，不會寫回瀏覽器。
    """
    values = all_values()
    if values.get("cookies_file"):
        src = str(values["cookies_file"])
        problem = cookies_file_problem(src)
        if problem:
            raise RuntimeError(f"{problem}：{src}，請到「設定 → 下載影片」重新指定")
        copy_to.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, copy_to)
        return {"cookiefile": str(copy_to)}
    if values.get("cookies_browser") in COOKIE_BROWSERS[1:]:
        return {"cookiesfrombrowser": (values["cookies_browser"],)}
    return {}
