"""網頁伺服器。啟動：python -s -m app.server"""
import json
import logging
import mimetypes
import os
import re
import subprocess
import threading
import webbrowser
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from . import config, db, gpu, health, instance, jobs, media, models, netinfo, plugins, safepath, syscheck
from . import cues as cue_mod
from . import series as series_mod
from . import settings as settings_mod
from . import vocab, vocab_api
from . import dict_build, setup   # 第一次打開的自動安裝、建查字字典
from .config import (ALIGNER, ASR_ENGINES, LANGUAGES, LOG_PATH, MEDIA_DIR, MODEL_CATALOG, MODELS_DIR,  # noqa: F401
                     PORT, PROXY_DIR, THUMB_DIR, TRACK_LANG_LABELS, TRANSLATORS, VIDEO_EXTS, WEB_DIR, ensure_dirs)

log = logging.getLogger("server")

# Windows 登錄檔裡的 MIME 類型可能被別的軟體改掉（例如 .js 變成 text/plain），瀏覽器就不執行、畫面壞掉。
# StaticFiles 靠 mimetypes 判斷，啟動時先註冊正確的類型
for _mime, _ext in (("text/css", ".css"), ("text/javascript", ".js"), ("text/javascript", ".mjs"),
                    ("application/json", ".json"), ("image/svg+xml", ".svg"),
                    ("application/manifest+json", ".webmanifest")):
    mimetypes.add_type(_mime, _ext)

# main() 實際綁的位址和 port（設定頁顯示區網網址用）；不是從 main() 啟動時是 None
RUNTIME = {"host": None, "port": None}


@asynccontextmanager
async def lifespan(_app):
    ensure_dirs()
    db.init()
    # 程式資料夾搬家（或 data 換位置）後，上傳、網址下載的影片路徑還指到舊位置：改成現在的 data 資料夾
    moved = db.relocate_media_paths(config.MEDIA_DIR, config.PROXY_DIR)
    if moved:
        log.info("data folder moved: updated %d video paths", moved)
    vocab.init()
    for problem in config.tool_problems():
        log.warning(problem)
    # 開發測試用：第二個伺服器只提供網頁和 API，不處理佇列，避免跟正式的搶顯卡。
    # 會跑佇列的伺服器要先拿到 data 資料夾的單一實例鎖，拿到之後才把上次中斷的任務放回佇列；
    # 拿不到代表同一個 data 資料夾已經有別的伺服器在跑佇列，這個只提供網頁，不動佇列
    if os.environ.get("VS_NO_WORKERS") != "1":
        if instance.acquire_lock():
            dict_build.finish_pending()      # 上次建好但換不掉的字典換上去、清掉建到一半的暫存檔（還沒有人開字典）
            jobs.start_workers()
            setup.startup()                  # 新使用者：偵測顯示卡、自動排模型下載和建字典（不等網頁打開）
        else:
            log.warning("another server is running the queue for %s; this one only serves the web page", config.DATA_DIR)
    # 假名引擎在背景初始化（約 1 秒），失敗也不影響其他功能
    from . import furigana
    threading.Thread(target=furigana.init, name="furigana-init", daemon=True).start()
    # 查字：背景載入字典，並把還沒切詞索引的字幕補算（只用 CPU）
    vocab.kick()
    yield


app = FastAPI(lifespan=lifespan)


@app.middleware("http")
async def no_cache_static(request: Request, call_next):
    response = await call_next(request)
    if request.url.path == "/" or request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-cache"
    return response


def _allowed_host_names() -> set[str]:
    import socket
    names = {"localhost", "testserver"}   # testserver：FastAPI TestClient 的主機名稱
    host = socket.gethostname().lower()
    names |= {host, f"{host}.local"}
    names |= {h.strip().lower() for h in os.environ.get("VS_ALLOWED_HOSTS", "").split(",") if h.strip()}
    return names


_ALLOWED_HOST_NAMES = _allowed_host_names()


def host_allowed(host_header: str) -> bool:
    """網址列的主機是 IP、localhost 或這台電腦的名稱才接受。
    擋 DNS rebinding：惡意網站把自己的網域解析到這台電腦，瀏覽器會把它當成同一個網站，Origin 檢查擋不住。
    其他名稱（例如路由器給的 xxx.lan）可以用環境變數 VS_ALLOWED_HOSTS 加，逗號分隔。"""
    import ipaddress
    host = (host_header or "").strip().lower()
    if not host:
        return True
    if host.startswith("["):                      # [::1]:8765
        host = host[1:host.find("]")] if "]" in host else host
    elif host.count(":") == 1:
        host = host.rsplit(":", 1)[0]
    host = host.rstrip(".")
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return host in _ALLOWED_HOST_NAMES


@app.middleware("http")
async def reject_unknown_host(request: Request, call_next):
    if not host_allowed(request.headers.get("host", "")):
        return JSONResponse({"detail": "請用 IP 位址或 localhost 開啟這個網頁"}, status_code=403)
    return await call_next(request)


# 手機看得到這一頁，代表伺服器已經在聽區網，打開開關馬上生效，不用重新啟動
LAN_OFF_MESSAGE = "這台電腦沒有開放手機、平板連線。請在電腦上打開設定頁的「讓同一個網路的手機、平板連線」，再重新整理這一頁"


def _ip(host: str | None):
    import ipaddress
    try:
        ip = ipaddress.ip_address((host or "").split("%", 1)[0])
    except ValueError:
        return None
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
        ip = ip.ipv4_mapped
    return ip


def is_local_client(host: str | None, server_host: str | None = None) -> bool:
    """連線是不是從這台電腦來的：127.0.0.1、::1，或來源位址就是這條連線連進來的本機位址
    （VS_HOST 設成區網 IP 時，在這台電腦上開 http://192.168.x.x:port/，來源和目的地是同一個 IP；
    別台電腦沒辦法用這台的 IP 完成 TCP 連線）。testclient 是 FastAPI TestClient 的假位址，
    真正的連線一定是 IP（uvicorn 關掉了 proxy_headers，別人沒辦法用 X-Forwarded-For 冒充）。"""
    if host == "testclient":
        return True
    ip = _ip(host)
    if ip is None:
        return False
    if ip.is_loopback:
        return True
    local = _ip(server_host)
    return local is not None and not local.is_unspecified and ip == local


@app.middleware("http")
async def reject_lan_when_off(request: Request, call_next):
    """區網開關關著時，不是從這台電腦來的連線一律回 403。每次都讀設定，關掉馬上生效，不用重新啟動。"""
    client = request.client.host if request.client else None
    server_addr = request.scope.get("server") or (None, None)
    # 讀資料庫可能要等佇列的寫入放開鎖，放到執行緒裡，不卡住其他連線
    if not is_local_client(client, server_addr[0]) and not await run_in_threadpool(settings_mod.get, "lan_access"):
        if request.url.path.startswith("/api/"):
            return JSONResponse({"detail": LAN_OFF_MESSAGE}, status_code=403)
        return HTMLResponse(f'<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width">'
                            f'<p style="font:16px sans-serif;margin:2em">{LAN_OFF_MESSAGE}</p>', status_code=403)
    return await call_next(request)


@app.middleware("http")
async def reject_cross_site_writes(request: Request, call_next):
    """不設密碼，但擋掉「別的網站的網頁」偷偷送來的改資料請求（例如瀏覽惡意網頁時被自動送出的表單）。
    瀏覽器送 POST、PUT、PATCH、DELETE 時會帶 Origin，跟網址列的主機不同就拒絕；
    自己的網頁（本機或區網裝置打開的都一樣）和沒有 Origin 的程式呼叫不受影響。"""
    if request.method not in ("GET", "HEAD", "OPTIONS"):
        origin = request.headers.get("origin")
        if origin is not None:
            from urllib.parse import urlsplit
            try:
                origin_host = urlsplit(origin).netloc.lower()
            except ValueError:
                origin_host = ""
            if not origin_host or origin_host != request.headers.get("host", "").lower():
                return JSONResponse({"detail": "不接受其他網站送來的請求"}, status_code=403)
    return await call_next(request)


app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
app.include_router(vocab_api.router)   # 查字與單字頁
app.include_router(setup.router)       # 第一次打開的自動安裝、建字典（/api/setup、/api/dicts/{lang}/build）


@app.get("/")
def index():
    return FileResponse(WEB_DIR / "index.html")


# ---------- 設定資訊 ----------

def _pending_models() -> set:
    """還沒裝好、但是正在下載或會自動接著下載的模型 id（app/setup.py）。"""
    try:
        return setup.pending_models()
    except Exception:  # noqa: BLE001
        log.warning("checking pending model downloads failed", exc_info=True)
        return set()


@app.get("/api/meta")
def meta():
    # pending：還沒裝好但正在下載的模型（新增影片時可以選，影片先排隊，裝好自動開始）
    pending = _pending_models()
    return {
        "app": instance.APP_ID,   # 重複啟動時靠這個認出 port 上是不是自己的程式（app/instance.py）
        "data_id": instance.data_id(),   # 同一台電腦上有兩份程式時，分辨 port 上的是不是這一份（路徑的雜湊）
        "languages": {k: v["label"] for k, v in LANGUAGES.items()},
        "track_langs": TRACK_LANG_LABELS,
        "engines": {k: {"label": v["label"], "langs": v["langs"], "installed": k in settings_mod.installed_engines(),
                        "pending": jobs.catalog_id("engine", k) in pending}
                    for k, v in ASR_ENGINES.items()},
        "translators": {k: {"label": v["label"], "langs": v["langs"], "installed": k in settings_mod.installed_translators(),
                            "pending": jobs.catalog_id("translator", k) in pending}
                        for k, v in TRANSLATORS.items()},
        "defaults": {
            lang: {"engine": settings_mod.engine_for(lang), "translator": settings_mod.translator_for(lang)}
            for lang in LANGUAGES
        },
        "health_version": health.HEALTH_VERSION,
    }


# ---------- 設定與模型 ----------

def _job_view(job: dict | None) -> dict | None:
    if not job:
        return None
    return {"id": job["id"], "status": job["status"], "progress": job["progress"], "stage": job["stage"],
            "variant": jobs.model_job_variant(job)}


def _catalog_view(gpu_info: dict | None = None):
    combos = _combos_view()
    model_jobs = [j for j in db.list_jobs() if j["type"] == "model" and j["status"] in ("queued", "running", "paused")]
    out = []
    for mid, e in MODEL_CATALOG.items():
        role = e["role"]
        if role == "aligner":
            used_by = ["所有語言"]
        elif role == "furigana":
            used_by = ["日文假名"]
        elif role == "segmenter":
            used_by = ["中文單字"]
        elif role == "separator":
            used_by = ["歌曲"] if settings_mod.get("song_mode") != "off" else []
        elif role == "asr":
            used_by = [c["label"] for c in combos if c["engine"] == e.get("engine")]
        else:
            used_by = [c["label"] for c in combos if c["translator"] == e.get("translator")]
        mine = [j for j in model_jobs if (j["params"] or {}).get("model") == mid]
        # 同一個模型有好幾個下載任務（不同版本）時，執行中的優先
        job = next((j for j in mine if j["status"] == "running"), mine[0] if mine else None)
        row = models.row(mid, gpu_info)
        for v in row.get("variants") or []:
            v["job"] = _job_view(next((j for j in mine if jobs.model_job_variant(j) == v["name"]), None))
        out.append({
            "id": mid, "label": e["label"], "role": role, "kind": e["kind"], "repo": e["repo"],
            "file": e.get("file"), "langs": e["langs"], "note": e["note"],
            "engine": e.get("engine"), "translator": e.get("translator"),
            **row,
            "used_by": used_by,
            "job": _job_view(job),
        })
    return out


def _combos_view():
    out = []
    for lang in LANGUAGES:
        for _ in [lang]:
            out.append({
                "key": lang,
                "language": lang,
                "label": LANGUAGES[lang]["label"],
                "engine": settings_mod.engine_for(lang),
                "translator": settings_mod.translator_for(lang),
                "engines": [{"key": k, "label": ASR_ENGINES[k]["label"]} for k in settings_mod.installed_engines(lang)],
                "translators": ([{"key": k, "label": TRANSLATORS[k]["label"]} for k in settings_mod.installed_translators(lang)]
                                if lang != "zh" else []),
            })
    return out


def _network_view() -> dict:
    """設定頁區網開關底下的狀態：現在有沒有在聽區網、要不要重新啟動、手機可以連的位址。"""
    lan = bool(settings_mod.get("lan_access"))
    host, port = RUNTIME["host"], RUNTIME["port"]
    listening_lan = host is not None and not instance.local_only(host)
    # VS_HOST 設成只開本機（127.0.0.1、::1）時照它綁，開關打開、重新啟動都沒用，要拿掉 VS_HOST
    host_env_local = bool(config.HOST_ENV) and instance.local_only(config.HOST_ENV)
    if not (lan or listening_lan) or host_env_local:
        addresses = []
    elif listening_lan and host not in ("0.0.0.0", "::"):
        addresses = [f"[{host}]" if ":" in host else host]     # VS_HOST 指定了一個區網位址：只有它連得到
    else:
        addresses = netinfo.lan_addresses()                     # 排除虛擬網卡後的位址（app/netinfo.py）
    return {
        "lan_access": lan,
        "listening_lan": listening_lan,
        "restart_needed": lan and host is not None and not listening_lan and not config.HOST_ENV,
        "host_env": config.HOST_ENV,
        "host_env_local": host_env_local,
        "port": port,
        # 開關關著又沒在聽區網、或 VS_HOST 只開本機時不用列
        "addresses": addresses,
    }


@app.get("/api/settings")
def read_settings():
    info = gpu.query()
    return {"values": settings_mod.all_values(), "catalog": _catalog_view(info), "combos": _combos_view(),
            "storage": models.storage_summary(), "gpu_usable_mb": models.gpu_usable_mb(info),
            "gpu_problem": syscheck.problem(info), "network": _network_view()}


@app.put("/api/settings")
def write_settings(body: dict):
    try:
        settings_mod.save(body)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"ok": True}


def _translation_mode(t: dict) -> str:
    """舊資料沒記錄模式：逐行模式每句譯文的起訖時間都和原文某一句相同，整句模式會跨行。"""
    if t.get("mode"):
        return t["mode"]
    src = db.get_track(t["source_track_id"]) if t.get("source_track_id") else None
    if not src:
        return "unknown"
    spans = {(round(c["start"], 2), round(c["end"], 2)) for c in cue_mod.load_cues(src["id"])}
    mine = [(round(c["start"], 2), round(c["end"], 2)) for c in cue_mod.load_cues(t["id"])]
    return "line" if mine and all(s in spans for s in mine) else "sentence"


def _outdated_translations() -> list[dict]:
    out = []
    for t in db.list_tracks():
        if t["kind"] != "translation" or not t.get("source_track_id") or not db.get_track(t["source_track_id"]):
            continue
        if _translation_mode(t) == "sentence":
            out.append(t)
    return out


@app.get("/api/translations/outdated")
def outdated_translations():
    items = _outdated_translations()
    cues = sum(db.get_track(t["source_track_id"])["cue_count"] for t in items)
    return {"count": len(items), "cues": cues, "estimate_s": int(cues * 0.15),
            "pending": any(j["type"] == "translate" and j["status"] in ("queued", "running")
                           and (j["params"] or {}).get("replace_track_id") for j in db.list_jobs())}


def _require_translator(key: str | None):
    """翻譯模型要已經下載才排任務，不然要等輪到才在佇列裡失敗。"""
    if key not in TRANSLATORS:
        raise HTTPException(400, "還沒下載翻譯模型，請到設定頁的模型管理下載")
    # 正在下載（第一次打開的自動安裝）：任務先排隊，模型裝好自動開始
    if key not in settings_mod.installed_translators() and not jobs.model_pending(jobs.catalog_id("translator", key)):
        raise HTTPException(400, f"{TRANSLATORS[key]['label']} 還沒下載，請到設定頁的模型管理下載")


@app.post("/api/translations/redo")
def redo_translations():
    """把所有舊的整句模式翻譯，改用逐行對照重翻（完成後才刪掉舊的）。"""
    items = _outdated_translations()
    queued = {(j["params"] or {}).get("replace_track_id") for j in db.list_jobs() if j["status"] in ("queued", "running")}
    installed = settings_mod.installed_translators()
    plan, missing = [], []
    for t in items:
        if t["id"] in queued:
            continue
        src = db.get_track(t["source_track_id"])
        lang = "en" if src["lang"] == "en" else "ja"
        # 用原本翻它的模型；那個模型已經刪掉的話改用這個語言的預設模型（translator_for 會挑已下載的）
        translator = next((k for k, v in TRANSLATORS.items() if v["label"] == t["model"]), None)
        if translator not in installed:
            translator = settings_mod.translator_for(lang)
        if translator not in installed:
            missing.append(TRANSLATORS.get(translator, {}).get("label") or "翻譯模型")
            continue
        plan.append((t, src, lang, translator))
    # 有任何一條沒辦法翻就整批都不排，免得只排了一半
    if missing:
        raise HTTPException(400, f"{'、'.join(dict.fromkeys(missing))} 還沒下載，請到設定頁的模型管理下載後再重翻")
    count = 0
    for t, src, lang, translator in plan:
        db.add_job(t["media_id"], "translate", {
            "translator": translator, "language": lang, "source_track_id": src["id"],
            "mode": "line", "replace_track_id": t["id"],
        })
        count += 1
    jobs.wake()
    return {"ok": True, "count": count}


@app.post("/api/titles/translate")
def translate_titles():
    pending = db.media_needing_title_translation()
    if not pending:
        return {"ok": True, "count": 0}
    if any(j["type"] == "titles" and j["status"] in ("queued", "running") for j in db.list_jobs()):
        raise HTTPException(409, "標題翻譯已經在佇列裡")
    translator = settings_mod.translator_for("ja")
    _require_translator(translator)
    db.add_job(None, "titles", {"translator": translator, "label": f"{len(pending)} 個影片標題"})
    jobs.wake()
    return {"ok": True, "count": len(pending)}


# ---------- 時間軸健康檢查 ----------

def _health_of(t: dict) -> dict | None:
    raw = t.get("health")
    if not raw:
        return None
    try:
        h = json.loads(raw) if isinstance(raw, str) else raw
    except ValueError:
        return None
    return h if isinstance(h, dict) else None


def _health_queued() -> set:
    """已經在佇列裡（排隊或執行中）的檢查任務對應的字幕軌。"""
    return {(j["params"] or {}).get("track_id") for j in db.list_jobs()
            if j["type"] == "health" and j["status"] in ("queued", "running")}


def _health_current(t: dict) -> bool:
    h = _health_of(t)
    return bool(h) and h.get("v") == health.HEALTH_VERSION


def _health_todo() -> tuple[list[dict], set]:
    """還沒用目前版本檢查過、影片檔還在、也還沒排進佇列的辨識字幕。"""
    queued = _health_queued()
    media_ok = {m["id"] for m in db.list_media() if m["path"] and Path(m["path"]).exists()}
    todo = [t for t in db.list_tracks() if t["kind"] == "asr" and t["media_id"] in media_ok
            and t["id"] not in queued and not _health_current(t)]
    return todo, queued


def _require_qwen():
    """檢查時間軸要用 Qwen3-ASR 和對齊模型（jobs.run_health）。"""
    if "qwen" not in settings_mod.installed_engines():
        raise HTTPException(400, f"檢查時間軸需要 {ASR_ENGINES['qwen']['label']}，請到設定頁下載")
    if not jobs.aligner_installed():
        raise HTTPException(400, f"檢查時間軸需要 {ALIGNER['label']}，請到設定頁下載")


class HealthReq(BaseModel):
    force: bool = False


@app.post("/api/tracks/{tid}/health")
def track_health(tid: str, req: HealthReq | None = None):
    t = _track_or_404(tid)
    if t["kind"] != "asr":
        raise HTTPException(400, "只有語音辨識產生的字幕可以檢查時間軸")
    if tid in _health_queued():
        raise HTTPException(409, "這條字幕已經在檢查時間軸的佇列裡")
    if _health_current(t) and not (req and req.force):
        raise HTTPException(409, "這條字幕已經檢查過時間軸，確定要再檢查一次請先確認（force）")
    m = db.get_media(t["media_id"])
    if not m or not m["path"] or not Path(m["path"]).exists():
        raise HTTPException(400, "找不到影片檔，沒辦法檢查時間軸")
    _require_qwen()
    db.add_job(t["media_id"], "health", {"track_id": tid})
    jobs.wake()
    return {"ok": True}


@app.post("/api/health/all")
def health_all():
    """把所有還沒用目前版本檢查過的辨識字幕排進佇列。"""
    todo, _ = _health_todo()
    if todo:
        _require_qwen()
    for t in todo:
        db.add_job(t["media_id"], "health", {"track_id": t["id"]})
    jobs.wake()
    return {"ok": True, "count": len(todo)}


@app.get("/api/health/pending")
def health_pending():
    todo, queued = _health_todo()
    lines = sum(t["cue_count"] or 0 for t in todo)
    # 預估時間只是估計：每 100 行約 20 秒（health.SECONDS_PER_LINE），實際看影片和顯卡
    return {"count": len(todo), "cues": lines, "estimate_s": int(lines * health.SECONDS_PER_LINE),
            "queued": len(queued)}


def _model_jobs(statuses=("queued", "running", "paused")) -> list[dict]:
    return [j for j in db.list_jobs() if j["type"] == "model" and j["status"] in statuses]


@app.post("/api/models/{mid}/download")
def download_model(mid: str, variant: str | None = None):
    """下載模型。有版本的模型用 variant 指定版本，沒指定用設定選的版本（沒選用預設版本）。
    同一個版本的下載暫停中時，改成繼續下載。"""
    if mid not in MODEL_CATALOG:
        raise HTTPException(404, "沒有這個模型")
    entry = MODEL_CATALOG[mid]
    try:
        variant = models.check_variant(entry, variant, default=True)
    except models.ModelError as e:
        raise HTTPException(e.status, str(e))
    same = [j for j in _model_jobs() if (j["params"] or {}).get("model") == mid and jobs.model_job_variant(j) == variant]
    paused = next((j for j in same if j["status"] == "paused"), None)
    if paused and len(same) == 1:
        jobs.resume(paused["id"])
        return {"ok": True, "resumed": True}
    if same:
        raise HTTPException(409, "這個模型已經在下載佇列裡")
    try:
        # 磁碟要放得下這個模型和佇列裡其他還沒下載完的模型，再多留一點
        models.check_disk(mid, variant=variant)
    except models.ModelError as e:
        raise HTTPException(e.status, str(e))
    params = {"model": mid, "label": jobs.model_job_label(entry, variant)}
    if variant:
        params["variant"] = variant
    db.add_job(None, "model", params)
    jobs.wake()
    return {"ok": True}


@app.get("/api/models/downloads")
def model_downloads():
    """模型下載的總進度（導覽的下載步驟用）：每個任務下載了多少、速度、剩多久、錯誤代碼。
    失敗的任務只列還沒解決的：同一個模型（版本）之後已經裝好、或已經有比較新的下載任務，就不列。"""
    items, done_all, total_all, speed_all = [], 0, 0, 0.0
    all_model_jobs = _model_jobs(("queued", "running", "paused", "failed", "done", "canceled"))
    for j in all_model_jobs:
        if j["status"] not in ("queued", "running", "paused", "failed"):
            continue
        p = j["params"] or {}
        entry = MODEL_CATALOG.get(p.get("model"))
        if not entry:
            continue
        variant = jobs.model_job_variant(j)
        if j["status"] == "failed":
            installed = config.model_installed(entry, variant) if variant else config.model_installed(entry)
            newer = any(o["id"] != j["id"] and (o["params"] or {}).get("model") == p.get("model")
                        and jobs.model_job_variant(o) == variant and o["created_at"] > j["created_at"]
                        for o in all_model_jobs)
            if installed or newer:
                continue
        info = jobs.download_progress.get(j["id"]) or {}
        view = config.variant_view(entry, variant)
        total = info.get("total") or int(view.get("bytes") or view["size_mb"] * models.MIB)
        done = info.get("done") if info else int(total * (j["progress"] or 0))
        speed = float(info.get("speed") or 0)
        item = {"job_id": j["id"], "id": p["model"], "variant": variant, "label": p.get("label") or entry["label"],
                "status": j["status"], "done_mb": done // models.MIB, "total_mb": total // models.MIB,
                "speed_mbps": round(speed / models.MIB, 1),
                "eta_s": int((total - done) / speed) if speed > 0 else None}
        if info.get("retry"):
            item["retry"] = info["retry"]
        if j["status"] == "failed":
            item["error"] = {"code": (j.get("result") or {}).get("error_code"), "message": j.get("error")}
        else:
            done_all, total_all, speed_all = done_all + done, total_all + total, speed_all + speed
        items.append(item)
    return {"active": any(i["status"] in ("queued", "running") for i in items),
            "done_mb": done_all // models.MIB, "total_mb": total_all // models.MIB,
            "speed_mbps": round(speed_all / models.MIB, 1),
            "eta_s": int((total_all - done_all) / speed_all) if speed_all > 0 else None, "items": items}


@app.post("/api/models/downloads/pause")
def pause_model_downloads():
    """全部模型下載暫停（暫存檔保留）。"""
    n = sum(1 for j in _model_jobs(("queued", "running")) if jobs.pause(j["id"]))
    return {"ok": True, "count": n}


@app.post("/api/models/downloads/resume")
def resume_model_downloads():
    n = sum(1 for j in _model_jobs(("paused",)) if jobs.resume(j["id"]))
    return {"ok": True, "count": n}


@app.get("/api/models/{mid}/impact")
def model_impact(mid: str, variant: str | None = None):
    """刪除確認視窗的內容：刪掉會影響什麼、排隊中會失敗的任務數、現在能不能刪。variant：只刪這個版本。"""
    try:
        return models.impact(mid, variant)
    except models.ModelError as e:
        raise HTTPException(e.status, str(e))


@app.delete("/api/models/{mid}")
def delete_model(mid: str, variant: str | None = None):
    """刪掉模型自己的檔案，下載到一半的殘檔也用這個清（規則見 app/models.py）。variant：只刪這個版本。"""
    try:
        return models.delete(mid, variant)
    except models.ModelError as e:
        raise HTTPException(e.status, str(e))


@app.get("/api/state")
def state():
    tracks_by_media: dict[str, list] = {}
    for t in db.list_tracks():
        # 時間軸檢查摘要：前端只需要數字，對不上的行號已經標在字幕檔的 chk 上，不跟著每次輪詢傳
        h = _health_of(t)
        t["health"] = {k: v for k, v in h.items() if k != "bad"} if h else None
        tracks_by_media.setdefault(t["media_id"], []).append(t)
    items = []
    all_media = db.list_media()
    for m in all_media:
        items.append({
            "id": m["id"], "title": m["title"], "title_zh": m.get("title_zh"),
            "source": m["source"], "path": m["path"], "url": m["url"],
            "duration": m["duration"], "language": m["language"], "profile": m["profile"],
            "vcodec": m["vcodec"], "width": m["width"], "height": m["height"],
            "playable": bool(m["playable"]), "has_proxy": bool(m["proxy_path"] and Path(m["proxy_path"]).exists()),
            "has_thumb": bool(m["has_thumb"]), "created_at": m["created_at"],
            "tracks": tracks_by_media.get(m["id"], []),
            "series": series_mod.loads(m.get("series_info")),
        })
    # 依作品分的播放列表（作品 → 季 → 集）。有字幕：有語音辨識產生的字幕
    subtitled = {mid for mid, ts in tracks_by_media.items() if any(t["kind"] == "asr" for t in ts)}
    series_tree = series_mod.tree(all_media, subtitled)
    info = gpu.query()
    job_list = db.list_jobs()
    # 暫停、中斷後接著做的任務這次開始時的進度（jobs.resume_points），網頁估剩餘時間用
    starts = jobs.resume_points()
    for j in job_list:
        if j["status"] == "running" and starts.get(j["id"]):
            j["resumed_from"] = starts[j["id"]]
    try:
        setup.annotate_waiting(job_list)       # 在等模型下載完的任務加上 waiting
        setup_summary = setup.summary_view()   # 右上角的安裝狀態
    except Exception:  # noqa: BLE001
        log.warning("first-run setup status failed", exc_info=True)
        setup_summary = None
    return {
        "media": items,
        "series": series_tree,
        "jobs": job_list,
        "setup": setup_summary,
        # loaded：顯卡上載入著的模型（常駐的語音模型、llama-server），顯卡沒事做 idle_release_s 秒後自動釋放
        # queue_paused：按「釋放顯卡」暫停了佇列（顯卡任務不會開始，按「繼續佇列」恢復）
        "gpu": {**jobs.gpu_state, "info": info, "loaded": jobs.loaded_models(),
                "idle_release_s": config.GPU_IDLE_RELEASE_S, "queue_paused": jobs.queue_paused()},
    }


class ReleaseGpuReq(BaseModel):
    pause_queue: bool = False    # 佇列還有任務時，網頁上確認過「暫停佇列並釋放顯卡」才帶 true


@app.post("/api/gpu/release")
def release_gpu(req: ReleaseGpuReq | None = None):
    """「釋放顯卡」：馬上結束顯卡上所有的模型（例如要玩遊戲）。佇列還有任務而且沒帶 pause_queue 時不動任何東西，
    回 needs_confirm（網頁問使用者要不要暫停佇列）；帶了就暫停佇列、執行中的任務停下來回到佇列，再全部釋放。"""
    try:
        return {"ok": True, **jobs.release_gpu(pause_queue=bool(req and req.pause_queue))}
    except jobs.QueueBusy as e:
        return {"ok": False, "needs_confirm": True, "message": str(e)}


@app.post("/api/queue/resume")
def queue_resume():
    """「繼續佇列」：「釋放顯卡」暫停的佇列恢復，停下來的任務從檢查點接著做。"""
    return {"ok": True, **jobs.resume_queue()}


# ---------- 新增影片 ----------

class TranscribeOptions(BaseModel):
    language: str
    translate: bool = False
    engine: str | None = None
    translator: str | None = None
    sensitive: bool = False  # 小聲或耳語較多：人聲偵測用高靈敏度
    force: bool = False


class AddMedia(TranscribeOptions):
    source: str
    path: str | None = None
    url: str | None = None


def _resolve_options(opt: TranscribeOptions) -> tuple[str, str | None]:
    if opt.language not in LANGUAGES:
        raise HTTPException(400, "請選擇影片語言")
    engine = opt.engine or settings_mod.engine_for(opt.language)
    if engine not in ASR_ENGINES or opt.language not in ASR_ENGINES[engine]["langs"]:
        raise HTTPException(400, f"{ASR_ENGINES.get(engine, {}).get('label', engine)} 不支援這個語言")
    # 模型正在下載（第一次打開的自動安裝）時放行：任務排隊，模型裝好自動開始（jobs.Worker.pick 先跳過）
    if engine not in settings_mod.installed_engines() and not jobs.model_pending(jobs.catalog_id("engine", engine)):
        raise HTTPException(400, f"{ASR_ENGINES[engine]['label']} 還沒下載，請到設定頁下載")
    if not jobs.aligner_installed() and not jobs.model_pending(jobs.catalog_id("role", "aligner")):
        raise HTTPException(400, f"時間軸對齊模型 {ALIGNER['label']} 還沒下載，請到設定頁下載")
    translator = None
    if opt.translate and opt.language != "zh":
        translator = opt.translator or settings_mod.translator_for(opt.language)
        if translator not in TRANSLATORS or opt.language not in TRANSLATORS[translator]["langs"]:
            raise HTTPException(400, "這個翻譯模型不支援這個語言")
        if (translator not in settings_mod.installed_translators()
                and not jobs.model_pending(jobs.catalog_id("translator", translator))):
            raise HTTPException(400, f"{TRANSLATORS[translator]['label']} 還沒下載，請到設定頁下載")
    return engine, translator


def _enqueue_pipeline(media_id: str, opt: TranscribeOptions, depends_on: str | None = None):
    engine, translator = _resolve_options(opt)
    db.update_media(media_id, language=opt.language)
    t_job = db.add_job(media_id, "transcribe",
                       {"language": opt.language, "engine": engine, "sensitive": opt.sensitive},
                       depends_on=depends_on)
    last_job = t_job
    if translator:
        last_job = db.add_job(media_id, "translate", {"translator": translator, "language": opt.language},
                              depends_on=t_job)
    # 日文影片用 Hy-MT2 翻譯時，翻完順便產生單字的中文釋義（沿用已載入的模型）
    from . import glossgen
    glossgen.maybe_enqueue_after_pipeline(media_id, opt.language, translator, last_job)
    jobs.wake()


def _make_thumb_async(media_id: str, path: Path, duration: float):
    def run():
        thumb = THUMB_DIR / f"{media_id}.jpg"
        if media.make_thumbnail(path, thumb, duration):
            if db.get_media(media_id):
                db.update_media(media_id, has_thumb=1)
            else:  # 縮圖還沒做好影片就被刪掉了
                safepath.safe_unlink(thumb, THUMB_DIR)
    threading.Thread(target=run, daemon=True).start()


def _uploads_dir() -> Path:
    return MEDIA_DIR / "uploads"


def _upload_folder(raw) -> Path | None:
    """上傳的檔案所在的資料夾，一定要是 data/media/uploads/<12 碼 id>；路徑不是這個樣子就回 None。
    回傳的是解析過 .. 和捷徑的真實位置。"""
    try:
        p = Path(raw).resolve()
        uploads = _uploads_dir().resolve()
    except (OSError, RuntimeError, ValueError, TypeError):
        return None
    folder = p.parent
    if not safepath.is_id(folder.name) or os.path.normcase(str(folder.parent)) != os.path.normcase(str(uploads)):
        return None
    return folder


def _folder_used_by_other_media(folder: Path, mid: str | None = None) -> bool:
    """資料夾裡有沒有其他影片的檔案（有的話不能整個刪掉）。"""
    return any(m["id"] != mid and m["path"] and safepath.inside(m["path"], folder) for m in db.list_media())


def _discard_upload(folder: Path | None):
    """加入失敗時把剛上傳的那份刪掉，不然會一直留在 data/media/uploads 裡。"""
    if folder is not None and not _folder_used_by_other_media(folder):
        safepath.safe_rmtree(folder, MEDIA_DIR)


def _require_media_tools(*names: str):
    """處理影片要用 ffmpeg、ffprobe；找不到丟 config.ToolMissing，API 回 503 加中文說明（見 tool_missing）。
    先查 ffmpeg：使用者設錯 VS_FFMPEG 時，看到的是那個設定的說明。轉字幕、轉檔只用 ffmpeg，不用 ffprobe。"""
    for name in names or ("ffmpeg", "ffprobe"):
        config.require_tool(name)


@app.post("/api/media/precheck")
def precheck_media(opt: TranscribeOptions):
    """上傳影片前先檢查語言、需要的模型（辨識、對齊、翻譯）和 ffmpeg，不用等大檔案傳完才知道缺東西。"""
    _require_media_tools()
    _resolve_options(opt)
    return {"ok": True}


@app.get("/api/url-plugin")
def url_plugin(url: str = ""):
    """這個網址是不是由本機外掛處理（app/plugins.py）。網頁用來自動選好外掛固定的選項（例如語言、不翻譯）；
    expand 為 True 時網頁再用 /api/url-plugin/list 列出整部作品讓使用者勾。"""
    plugin = plugins.find(url)
    if not plugin:
        return {"plugin": None, "options": {}, "expand": False}
    return {"plugin": plugin.name, "options": plugin.options, "expand": plugin.can_expand}


def _plugin_listing(url: str):
    """(外掛, 網址, 清單)；網址不是能展開的外掛網址回 400，外掛讀不到清單照外掛的狀態碼。"""
    try:
        url = safepath.check_download_url(url)
    except ValueError as e:
        raise HTTPException(400, str(e))
    plugin = plugins.find(url)
    if not plugin or not plugin.can_expand:
        raise HTTPException(400, "這個網址不能列出集數")
    try:
        return plugin, url, plugins.expand(plugin, url)
    except plugins.PluginError as e:
        raise HTTPException(e.status, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/api/url-plugin/list")
def url_plugin_list(url: str = ""):
    """外掛把網址展開成的清單（例如整部作品的每一季、每一集），每一集標出是不是已經在播放列表裡。"""
    plugin, url, listing = _plugin_listing(url)
    have = {}
    for m in db.list_media():
        if m["url"] and plugin.match(m["url"]):
            have.setdefault(plugins.item_key(plugin, m["url"]), m["id"])
    current = plugins.item_key(plugin, url)
    groups = []
    for g in listing["groups"]:
        items = []
        for it in g["items"]:
            key = plugins.item_key(plugin, it["url"])
            items.append({**it, "in_library": key in have, "media_id": have.get(key), "current": key == current})
        groups.append({"key": g["key"], "title": g["title"], "note": g["note"], "checked": g["checked"],
                       "language": g["options"].get("language") or plugin.options.get("language"), "items": items})
    return {"plugin": plugin.name, "title": listing["title"], "groups": groups}


# ---------- 外掛設定（只寫不讀：回傳的只有有沒有設定、什麼時候設定、瀏覽器型號） ----------

@app.get("/api/plugins/settings")
def plugin_settings():
    return {"plugins": plugins.all_settings()}


def _plugin_or_404(pid: str):
    plugin = plugins.by_id(pid)
    if not plugin or not plugin.settings:
        raise HTTPException(404, "找不到這個外掛")
    return plugin


@app.put("/api/plugins/{pid}/settings")
async def save_plugin_settings(pid: str, request: Request):
    # 自己讀 body：格式錯誤時 FastAPI 的 422 會把送來的內容放回錯誤訊息，這裡不回傳也不記錄填的內容
    try:
        body = json.loads(await request.body() or b"{}")
    except ValueError:
        raise HTTPException(400, "格式不對")
    if not isinstance(body, dict):
        raise HTTPException(400, "格式不對")

    def save():
        plugin = _plugin_or_404(pid)
        try:
            return plugins.save_settings(plugin, body.get("values"), body.get("user_agent"))
        except plugins.PluginError as e:
            raise HTTPException(e.status, str(e))

    return {"ok": True, "plugin": await run_in_threadpool(save)}


class UserAgentReq(BaseModel):
    user_agent: str = ""


@app.put("/api/plugins/{pid}/user-agent")
def plugin_user_agent(pid: str, req: UserAgentReq):
    """網頁打開時自動送的瀏覽器 UA（外掛要的才送，只在跟存的不一樣時送）。規則見 plugins.update_user_agent。"""
    plugin = _plugin_or_404(pid)
    try:
        return {"ok": True, **plugins.update_user_agent(plugin, req.user_agent)}
    except plugins.PluginError as e:
        raise HTTPException(e.status, str(e))


@app.delete("/api/plugins/{pid}/settings")
def clear_plugin_settings(pid: str):
    plugin = _plugin_or_404(pid)
    try:
        return {"ok": True, "plugin": plugins.clear_settings(plugin)}
    except plugins.PluginError as e:
        raise HTTPException(e.status, str(e))


@app.post("/api/media")
def add_media(req: AddMedia):
    if req.source == "local":
        # 本機路徑可以指到整台電腦的任何檔案，區網裡任何裝置都能用，所以不再開放；以前加入的本機影片照常播放
        raise HTTPException(400, "已經不能用電腦上的檔案路徑加入影片，請改用「上傳」或貼 YouTube 網址")
    if req.source == "upload":
        # 只收 /api/upload 剛存好的檔案：data/media/uploads/<12 碼 id>/檔名
        raw = (req.path or "").strip().strip('"').strip("'")
        folder = _upload_folder(raw) if raw else None
        if folder is None:
            raise HTTPException(400, "找不到上傳的檔案，請重新上傳")
        path = folder / Path(raw).name
        if not path.is_file() or safepath.is_link(path) or not safepath.inside(path, _uploads_dir()):
            raise HTTPException(400, "找不到上傳的檔案，請重新上傳")
        if _folder_used_by_other_media(folder):
            raise HTTPException(409, "這個檔案已經在播放列表裡，可以直接在影片頁面按「轉字幕」")
        try:
            _require_media_tools()
            _resolve_options(req)
            try:
                info = media.probe(path)
            except ValueError as e:
                raise HTTPException(400, str(e))
        except Exception:
            _discard_upload(folder)
            raise
        mid = db.add_media(title=path.stem, source="upload", path=str(path), **info)
        _make_thumb_async(mid, path, info["duration"])
        _enqueue_pipeline(mid, req)
        return {"id": mid}
    if req.source == "url":
        try:
            url = safepath.check_download_url(req.url)
        except ValueError as e:
            raise HTTPException(400, str(e))
        # 本機外掛處理的網址：外掛固定的選項（例如語言、不翻譯）蓋掉網頁送來的
        plugin = plugins.find(url)
        if plugin:
            _apply_options(req, plugin.options)
        _require_media_tools()
        _resolve_options(req)
        return {"id": _add_url_media(url, req)}
    raise HTTPException(400, "未知的來源")


def _apply_options(opt: TranscribeOptions, fixed: dict):
    """外掛固定的選項蓋掉網頁送來的。換了語言時，網頁選的辨識、翻譯模型是給原本的語言的，改用新語言的預設。"""
    if fixed.get("language", opt.language) != opt.language:
        opt.engine = opt.translator = None
    for key, value in fixed.items():
        setattr(opt, key, value)


def _add_url_media(url: str, opt: TranscribeOptions, title: str | None = None, series: dict | None = None) -> str:
    """加一部網址影片：先下載，再照 opt 轉字幕（選項已經檢查過）。series：外掛標的作品、季、集（app/series.py）。"""
    mid = db.add_media(title=title or url, source="url", url=url, playable=1, series_info=series_mod.dumps(series))
    d_job = db.add_job(mid, "download", {"url": url})
    _enqueue_pipeline(mid, opt, depends_on=d_job)
    return mid


class AddBatch(TranscribeOptions):
    url: str                 # 貼的網址（外掛從它展開清單）
    items: list[str]         # 勾選的每一集的網址，要在清單裡


@app.post("/api/media/batch")
def add_media_batch(req: AddBatch):
    """外掛展開的清單裡勾選的集數一次加入。每一集的選項：網頁送來的 → 外掛固定的 → 那一組固定的（例如配音版的語言）。
    全部檢查過才開始加，不會只加一半。"""
    plugin, _url, listing = _plugin_listing(req.url)
    wanted = set(req.items)
    if not wanted:
        raise HTTPException(400, "請至少勾一集")
    if len(wanted) > plugins.MAX_ITEMS:
        raise HTTPException(400, "一次勾太多了")
    known = {it["url"]: (it, g) for g in listing["groups"] for it in g["items"]}
    if any(u not in known for u in wanted):
        raise HTTPException(400, "清單已經變了，請重新貼一次網址")
    base = req.model_dump(include=set(TranscribeOptions.model_fields))
    plan, checked = [], {}
    for g in listing["groups"]:            # 照清單的順序排進佇列
        opt = TranscribeOptions(**base)
        _apply_options(opt, plugin.options)
        _apply_options(opt, g["options"])
        sig = json.dumps(opt.model_dump(), sort_keys=True)
        for it in g["items"]:
            if it["url"] in wanted:
                if sig not in checked:
                    checked[sig] = _resolve_options(opt)
                plan.append((it, opt))
    _require_media_tools()
    ids = [_add_url_media(it["url"], opt, it["title"] or None, it.get("series")) for it, opt in plan]
    return {"ok": True, "ids": ids, "count": len(ids)}


@app.put("/api/upload")
async def upload(request: Request, name: str):
    safe = re.sub(r'[\\/:*?"<>|\x00-\x1f]+', "_", Path(name).name).strip(" .") or "upload.mp4"
    suffix = Path(safe).suffix
    if suffix.lower() not in VIDEO_EXTS:
        raise HTTPException(400, "不支援這種檔案格式")
    # 檔名太長會超過 Windows 路徑長度上限，只留前 100 個字
    safe = (Path(safe).stem[:100].rstrip(" .") or "upload") + suffix
    folder = _uploads_dir() / db.new_id()
    folder.mkdir(parents=True, exist_ok=True)
    dst = folder / safe
    try:
        with open(dst, "wb") as f:
            async for chunk in request.stream():
                f.write(chunk)
    except BaseException as e:
        # 上傳到一半斷線或取消：不留半截的檔案
        safepath.safe_rmtree(folder, MEDIA_DIR)
        if isinstance(e, OSError):
            raise HTTPException(400, f"檔案存不進去：{e.strerror or e}")
        raise
    return {"path": str(dst)}


# ---------- 影片操作 ----------

def _media_or_404(mid):
    m = db.get_media(mid) if safepath.is_id(mid) else None
    if not m:
        raise HTTPException(404, "找不到影片")
    return m


@app.post("/api/media/{mid}/transcribe")
def transcribe(mid: str, opt: TranscribeOptions):
    m = _media_or_404(mid)
    if not m["path"]:
        raise HTTPException(400, "影片還沒下載完成")
    _require_media_tools("ffmpeg")
    if not opt.force:
        # 已經有字幕或已在佇列裡就先擋下，避免白白佔用顯卡
        if any(t["kind"] == "asr" for t in db.list_tracks(mid)):
            raise HTTPException(409, "這部影片已經有字幕了")
        if any(j["type"] == "transcribe" and j["status"] in ("queued", "running") for j in db.jobs_for_media(mid)):
            raise HTTPException(409, "這部影片已經在轉字幕佇列裡")
    _enqueue_pipeline(mid, opt)
    return {"ok": True}


class Rename(BaseModel):
    title: str


@app.patch("/api/media/{mid}")
def rename(mid: str, body: Rename):
    _media_or_404(mid)
    title = body.title.strip()
    if not title:
        raise HTTPException(400, "名稱不能是空的")
    db.update_media(mid, title=title)
    return {"ok": True}


@app.post("/api/media/{mid}/proxy")
def proxy(mid: str):
    m = _media_or_404(mid)
    if not m["path"]:
        raise HTTPException(400, "影片還沒下載完成")
    if any(j["type"] == "proxy" and j["status"] in ("queued", "running") for j in db.jobs_for_media(mid)):
        raise HTTPException(409, "已經在轉檔佇列裡")
    _require_media_tools("ffmpeg")
    db.add_job(mid, "proxy", {})
    jobs.wake()
    return {"ok": True}


@app.post("/api/media/{mid}/reveal")
def reveal(mid: str):
    m = _media_or_404(mid)
    if m["path"] and Path(m["path"]).exists():
        subprocess.Popen(["explorer", "/select,", m["path"]])
    return {"ok": True}


def _owned_media_folder(m: dict) -> Path | None:
    """刪除影片時可以整個刪掉的資料夾：只有程式自己建的才算。
      網址下載：data/media/<影片 id>
      上傳：data/media/uploads/<12 碼 id>（用檔案真實位置判斷，路徑裡的 .. 和大小寫騙不過）
    本機路徑加入的原始影片永遠回 None。資料夾裡還有別部影片的檔案也回 None。"""
    mid = m["id"]
    if not safepath.is_id(mid):
        return None
    if m["source"] == "url":
        folder = MEDIA_DIR / mid
    elif m["source"] == "upload" and m["path"]:
        folder = _upload_folder(m["path"])
        if folder is None:
            log.warning("media %s: upload path is not inside data/media/uploads/<id>, keeping files: %s", mid, m["path"])
            return None
    else:
        return None
    if _folder_used_by_other_media(folder, mid):
        log.warning("media %s: folder %s also holds another video, keeping files", mid, folder)
        return None
    return folder


@app.delete("/api/media/{mid}")
def delete_media(mid: str):
    m = _media_or_404(mid)
    job_ids = [j["id"] for j in db.jobs_for_media(mid)]
    for j in db.jobs_for_media(mid):
        if j["status"] in ("queued", "running"):
            jobs.cancel(j["id"])
    for t in db.list_tracks(mid):
        jobs.discard_track(t["id"])
    safepath.safe_unlink(THUMB_DIR / f"{mid}.jpg", THUMB_DIR)
    # 相容播放檔只會在 data/proxy/<影片 id>.mp4，轉到一半的暫存檔也一起清
    safepath.safe_unlink(PROXY_DIR / f"{mid}.mp4", PROXY_DIR)
    safepath.safe_unlink(PROXY_DIR / f"{mid}.tmp.mp4", PROXY_DIR)
    if m["proxy_path"]:
        safepath.safe_unlink(m["proxy_path"], PROXY_DIR)
    # 只刪除程式自己下載或上傳的檔案，本機加入的原始影片不動
    folder = _owned_media_folder(m)
    if folder is not None:
        safepath.safe_rmtree(folder, MEDIA_DIR)
    db.delete_media(mid)
    # 已經結束的任務留下的工作資料夾（執行中的任務會在自己結束時清掉）
    for jid in job_ids:
        if not db.get_job(jid):
            jobs.discard_workdir(jid)
    return {"ok": True}


@app.get("/media/{mid}/video")
def video(mid: str):
    m = _media_or_404(mid)
    path = m["proxy_path"] if m["proxy_path"] and Path(m["proxy_path"]).exists() else m["path"]
    if not path or not Path(path).exists():
        raise HTTPException(404, "影片檔不存在")
    mime = mimetypes.guess_type(path)[0] or "video/mp4"
    if Path(path).suffix.lower() in (".mov", ".m4v"):
        mime = "video/mp4"
    return FileResponse(path, media_type=mime)


@app.get("/media/{mid}/thumb.jpg")
def thumb(mid: str):
    if not safepath.is_id(mid):
        raise HTTPException(404)
    p = THUMB_DIR / f"{mid}.jpg"
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(p, media_type="image/jpeg")


# ---------- 字幕軌 ----------

def _track_or_404(tid):
    t = db.get_track(tid) if safepath.is_id(tid) else None
    if not t:
        raise HTTPException(404, "找不到字幕軌")
    return t


_ruby_lock = threading.Lock()
vocab.set_cue_lock(_ruby_lock)   # 查字補算 w 時也會改寫字幕檔，跟改讀音共用同一把鎖


def _refresh_ruby(t: dict, force: bool = False) -> list[dict]:
    """日文字幕的假名版本跟目前引擎（含使用者辭典）不同時，重算並存檔。"""
    from . import furigana
    data = cue_mod.load_cues(t["id"])
    if t["lang"] != "ja":
        return data
    if not force and t.get("ruby_ver") == furigana.version():
        return data
    if not furigana.available(10):
        return data
    with _ruby_lock:
        fresh = db.get_track(t["id"])
        if not force and fresh and fresh.get("ruby_ver") == furigana.version():
            return cue_mod.load_cues(t["id"])
        data = cue_mod.load_cues(t["id"])
        if furigana.apply(data) >= 0:
            cue_mod.save_cues(t["id"], data)
            db.update_track(t["id"], ruby_ver=furigana.version())
    return data


@app.get("/api/tracks/{tid}/cues")
def track_cues(tid: str):
    t = _track_or_404(tid)
    data = _refresh_ruby(t)
    # 日文、英文辨識字幕補上查字用的詞單位（cue["w"]），版本或文字不同就重算；失敗不影響字幕顯示
    try:
        return vocab.ensure(t, data)
    except Exception:  # noqa: BLE001
        log.warning("vocab ensure failed for %s", tid, exc_info=True)
        return data


def _download_name(m, label, ext):
    base = re.sub(r'[\\/:*?"<>|]+', "_", m["title"])[:80]
    return f"{base}.{label}.{ext}"


@app.get("/api/tracks/{tid}/download")
def track_download(tid: str, fmt: str = "srt"):
    t = _track_or_404(tid)
    m = _media_or_404(t["media_id"])
    data = cue_mod.load_cues(tid)
    body = cue_mod.to_vtt(data) if fmt == "vtt" else cue_mod.to_srt(data)
    name = _download_name(m, t["lang"], "vtt" if fmt == "vtt" else "srt")
    return Response(body, media_type="text/plain; charset=utf-8",
                    headers={"Content-Disposition": f"attachment; filename*=UTF-8''{_quote(name)}"})


@app.get("/api/media/{mid}/bilingual")
def bilingual(mid: str, top: str, bottom: str, fmt: str = "srt"):
    m = _media_or_404(mid)
    # 兩條字幕都要是這部影片的，不能拿網址參數去讀別的檔案
    for tid in (top, bottom):
        if _track_or_404(tid)["media_id"] != mid:
            raise HTTPException(404, "找不到字幕軌")
    merged = cue_mod.merge_bilingual(cue_mod.load_cues(top), cue_mod.load_cues(bottom))
    body = cue_mod.to_vtt(merged) if fmt == "vtt" else cue_mod.to_srt(merged)
    name = _download_name(m, "雙語", "vtt" if fmt == "vtt" else "srt")
    return Response(body, media_type="text/plain; charset=utf-8",
                    headers={"Content-Disposition": f"attachment; filename*=UTF-8''{_quote(name)}"})


def _quote(s):
    from urllib.parse import quote
    return quote(s)


class TranslateReq(BaseModel):
    translator: str
    force: bool = False


@app.post("/api/tracks/{tid}/translate")
def translate_track(tid: str, req: TranslateReq):
    t = _track_or_404(tid)
    lang = next((k for k, v in LANGUAGES.items() if v["track"] == t["lang"]), None)
    if lang not in ("ja", "en"):
        raise HTTPException(400, "只有日文和英文字幕可以翻譯")
    if req.translator not in TRANSLATORS or lang not in TRANSLATORS[req.translator]["langs"]:
        raise HTTPException(400, "這個翻譯模型不支援這個語言")
    # 已經在翻譯或排隊的就算確認過（force）也不再排一次，完成後自然會出現；新增影片時排的翻譯要從前一個任務的結果找原文
    if any(j["type"] == "translate" and j["status"] in ("queued", "running") and jobs.translation_source(j) == tid
           for j in db.jobs_for_media(t["media_id"])):
        raise HTTPException(409, "這條字幕正在翻譯或已經在佇列裡，完成後就會出現")
    # 模型沒下載的話排進去一定失敗；放在「已經翻譯過」前面，免得使用者確認完才看到這個
    _require_translator(req.translator)
    if not req.force and any(x["source_track_id"] == tid for x in db.list_tracks(t["media_id"])):
        raise HTTPException(409, "這條字幕已經翻譯過了")
    db.add_job(t["media_id"], "translate", {"translator": req.translator, "language": lang, "source_track_id": tid})
    jobs.wake()
    return {"ok": True}


@app.post("/api/tracks/{tid}/furigana")
def add_furigana(tid: str):
    """強制重算整條日文字幕的假名（保留手動改過的讀音）。"""
    t = _track_or_404(tid)
    if t["lang"] != "ja":
        raise HTTPException(400, "只有日文字幕需要假名")
    data = _refresh_ruby(t, force=True)
    return {"ok": True, "cues": len(data), "with_ruby": sum(1 for c in data if c.get("ruby"))}


class RubyEdit(BaseModel):
    s: int
    e: int
    rt: str


@app.put("/api/tracks/{tid}/cues/{index}/ruby")
def edit_ruby(tid: str, index: int, body: RubyEdit):
    """手動修改某一句某一段的讀音；rt 為空字串代表這裡不要標。"""
    from . import furigana
    t = _track_or_404(tid)
    with _ruby_lock:
        data = cue_mod.load_cues(tid)
        if not 0 <= index < len(data):
            raise HTTPException(404, "找不到這句字幕")
        try:
            furigana.set_user_span(data[index], body.s, body.e, body.rt.strip())
        except ValueError as e:
            raise HTTPException(400, str(e))
        cue_mod.save_cues(tid, data)
    return {"ok": True, "cue": data[index], "track": t["id"]}


@app.get("/api/tracks/{tid}/cues/{index}/lookup")
def lookup_ruby(tid: str, index: int, s: int, e: int):
    from . import furigana
    _track_or_404(tid)
    data = cue_mod.load_cues(tid)
    if not 0 <= index < len(data):
        raise HTTPException(404, "找不到這句字幕")
    return furigana.lookup(data[index]["text"], s, e)


class DictWord(BaseModel):
    surface: str
    reading: str


@app.post("/api/furigana/dict")
def add_dict_word(body: DictWord):
    """把名詞加進使用者辭典，之後所有日文字幕打開時會自動套用。"""
    from . import furigana
    try:
        furigana.add_user_word(body.surface, body.reading)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"ok": True, "version": furigana.version()}


@app.delete("/api/tracks/{tid}")
def delete_track(tid: str):
    t = _track_or_404(tid)
    # 翻譯綁定來源字幕：刪原文時連同從它翻出來的翻譯一起刪，也取消還在排隊的翻譯任務
    doomed = [tid] + [x["id"] for x in db.list_tracks(t["media_id"]) if x["source_track_id"] == tid]
    for j in db.jobs_for_media(t["media_id"]):
        # 新增影片時排的翻譯任務參數裡沒有來源字幕，要從前面的轉字幕任務結果找
        if j["type"] == "translate" and j["status"] in ("queued", "running") \
                and jobs.translation_source(j) == tid:
            jobs.cancel(j["id"])
        # 這條字幕的時間軸檢查也不用跑了
        if j["type"] == "health" and j["status"] in ("queued", "running") \
                and (j["params"] or {}).get("track_id") == tid:
            jobs.cancel(j["id"])
    for x in doomed:
        jobs.discard_track(x)
    return {"ok": True, "deleted": doomed}


# ---------- 佇列 ----------

@app.post("/api/jobs/{jid}/cancel")
def job_cancel(jid: str):
    jobs.cancel(jid)
    return {"ok": True}


@app.post("/api/jobs/{jid}/pause")
def job_pause(jid: str):
    """暫停模型下載（暫存檔保留，按繼續從中斷的地方接著下載）。"""
    if not jobs.pause(jid):
        raise HTTPException(400, "只有排隊中或下載中的模型下載可以暫停")
    return {"ok": True}


@app.post("/api/jobs/{jid}/resume")
def job_resume(jid: str):
    if not jobs.resume(jid):
        raise HTTPException(400, "這個任務沒有暫停")
    return {"ok": True}


class RetryReq(BaseModel):
    force: bool = False   # 會重做已經有的結果（字幕、翻譯、時間軸檢查、相容播放檔），網頁上確認過才帶


@app.post("/api/jobs/{jid}/retry")
def job_retry(jid: str, req: RetryReq | None = None):
    # 規則在 jobs.retry：同一件事已經在佇列裡回 409；有結果沒確認回 409；模型沒下載、字幕已刪除回 400
    try:
        jobs.retry(jid, force=bool(req and req.force))
    except jobs.RetryRefused as e:
        raise HTTPException(e.status, str(e))
    return {"ok": True}


class Move(BaseModel):
    direction: str


@app.post("/api/jobs/{jid}/move")
def job_move(jid: str, body: Move):
    db.move_job(jid, "up" if body.direction == "up" else "down")
    return {"ok": True}


@app.delete("/api/jobs/{jid}")
def job_delete(jid: str):
    if not safepath.is_id(jid):
        raise HTTPException(404, "找不到任務")
    db.delete_job(jid)
    # 任務刪掉就不會再重試，留著的工作資料夾（抽好的音訊、檢查點）一起清掉；執行中的任務不會被刪
    if not db.get_job(jid):
        jobs.discard_workdir(jid)
    return {"ok": True}


@app.post("/api/jobs/clear")
def job_clear():
    for jid in db.clear_finished():
        jobs.discard_workdir(jid)
    return {"ok": True}


@app.exception_handler(HTTPException)
async def http_error(_request, exc: HTTPException):
    return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)


@app.exception_handler(setup.SetupError)
async def setup_error(_request, exc: setup.SetupError):
    # 自動安裝的 API：detail 是給使用者看的訊息，code 給前端判斷，其他欄位（例如 plan、status）照原樣帶著
    return JSONResponse({"detail": str(exc), "code": exc.code, **exc.extra}, status_code=exc.status)


@app.exception_handler(config.ToolMissing)
async def tool_missing(_request, exc: config.ToolMissing):
    # 找不到 ffmpeg、ffprobe：不是 500，給使用者看得懂的說明
    return JSONResponse({"detail": str(exc)}, status_code=503)


def bind_host() -> str:
    """要綁的網路位址：有設 VS_HOST 環境變數就照它；沒設時看設定頁的區網開關，打開綁 0.0.0.0，關著只開本機。"""
    if config.HOST_ENV:
        return config.HOST_ENV
    return "0.0.0.0" if settings_mod.peek("lan_access") is True else "127.0.0.1"


def main():
    import argparse
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--port", type=int, default=None,
                        help=f"指定 port（預設 {PORT}，被占用時會自己換；有指定時被占用就直接結束）")
    args = parser.parse_args()

    ensure_dirs()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.FileHandler(LOG_PATH, encoding="utf-8"), logging.StreamHandler()],
    )

    def open_browser(url):
        if not args.no_browser:
            webbrowser.open(url)

    # 1. 會跑佇列的伺服器先拿單一實例鎖：已經有一個在跑，就打開它的網頁後結束，佇列完全不動
    workers = os.environ.get("VS_NO_WORKERS") != "1"
    if workers and not instance.acquire_lock():
        print("Bilingual Subtitles 已經開著了，正在找它的網頁……")
        url = instance.wait_existing()
        if url:
            print(f"Bilingual Subtitles 已經在執行：{url}")
            open_browser(url)
            return
        print(f"另一個 Bilingual Subtitles 程式正在使用資料夾 {config.DATA_DIR}，但連不上它的網頁。"
              "請先關掉它（或在工作管理員結束殘留的 python.exe），再重新啟動")
        sys.exit(1)

    # 2. 綁 port（被占用、被 Windows 保留時換下一個；--port 明確指定時不換）
    host = bind_host()
    explicit = args.port is not None
    base = args.port if explicit else PORT
    try:
        sock, port = instance.open_port(host, base, explicit)
    except instance.StartupStop as stop:
        print(stop)
        if stop.url:
            open_browser(stop.url)
        sys.exit(stop.code)
    RUNTIME.update(host=host, port=port)
    url = instance.browser_url(host, port)
    if workers:
        instance.write_info(host, port, url)

    print(f"Bilingual Subtitles 已啟動：{url}")
    if port != base:
        print(f"  port {base} 不能用（被其他程式占用或被 Windows 保留），改用 {port}")
    if host in ("0.0.0.0", "::"):
        for ip in netinfo.lan_addresses():
            print(f"  同一個網路的手機、平板：http://{ip}:{port}/")
    elif not config.HOST_ENV:
        print("  現在只有這台電腦能開。要讓手機、平板連線，請到設定頁打開區網開關後重新啟動")
    elif instance.local_only(host) and settings_mod.peek("lan_access") is True:
        print(f"  區網開關是開的，但環境變數 VS_HOST={host} 讓程式只開本機。要讓手機、平板連線，請拿掉 VS_HOST 後重新啟動")
    for problem in config.tool_problems():
        print(f"  注意：{problem}")
    # 3. 網頁真的連得上才開瀏覽器（啟動時要清殘留程序、開資料庫）
    if not args.no_browser:
        threading.Thread(target=lambda: instance.wait_ready(host, port) and webbrowser.open(url),
                         name="open-browser", daemon=True).start()
    # proxy_headers 關掉：沒有反向代理，連線來源就是真正的 IP（區網開關的判斷靠它）
    server = uvicorn.Server(uvicorn.Config(app, host=host, port=port, log_level="warning", proxy_headers=False))
    server.run(sockets=[sock])


if __name__ == "__main__":
    main()
