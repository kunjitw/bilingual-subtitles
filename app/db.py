"""SQLite 存取：影片、字幕軌、佇列任務。"""
import json
import sqlite3
import threading
import time
import uuid

from .config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS media (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    source TEXT NOT NULL,
    path TEXT,
    url TEXT,
    duration REAL,
    vcodec TEXT,
    acodec TEXT,
    width INTEGER,
    height INTEGER,
    playable INTEGER DEFAULT 1,
    proxy_path TEXT,
    has_thumb INTEGER DEFAULT 0,
    language TEXT,
    profile TEXT DEFAULT 'general',
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS tracks (
    id TEXT PRIMARY KEY,
    media_id TEXT NOT NULL,
    lang TEXT NOT NULL,
    kind TEXT NOT NULL,
    model TEXT,
    source_track_id TEXT,
    cue_count INTEGER DEFAULT 0,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    media_id TEXT,
    type TEXT NOT NULL,
    params TEXT NOT NULL DEFAULT '{}',
    result TEXT,
    status TEXT NOT NULL DEFAULT 'queued',
    progress REAL DEFAULT 0,
    stage TEXT,
    error TEXT,
    depends_on TEXT,
    position REAL NOT NULL,
    created_at REAL NOT NULL,
    started_at REAL,
    finished_at REAL
);
CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_tracks_media ON tracks(media_id);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status, position);
"""

UPSTREAM_FAILED = "前一個步驟沒有完成"

_lock = threading.RLock()
_conn: sqlite3.Connection | None = None


def init():
    global _conn
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    _conn = sqlite3.connect(DB_PATH, check_same_thread=False, isolation_level=None)
    _conn.row_factory = sqlite3.Row
    _conn.execute("PRAGMA journal_mode=WAL")
    _conn.executescript(SCHEMA)
    # 舊資料庫補欄位
    cols = {r["name"] for r in _conn.execute("PRAGMA table_info(media)").fetchall()}
    if "title_zh" not in cols:
        _conn.execute("ALTER TABLE media ADD COLUMN title_zh TEXT")
    tcols = {r["name"] for r in _conn.execute("PRAGMA table_info(tracks)").fetchall()}
    if "ruby_ver" not in tcols:
        _conn.execute("ALTER TABLE tracks ADD COLUMN ruby_ver TEXT")
    if "mode" not in tcols:
        _conn.execute("ALTER TABLE tracks ADD COLUMN mode TEXT")
    if "health" not in tcols:
        # 時間軸健康檢查的摘要（JSON），見 app/health.py
        _conn.execute("ALTER TABLE tracks ADD COLUMN health TEXT")


def new_id() -> str:
    return uuid.uuid4().hex[:12]


def _rows(sql, args=()):
    with _lock:
        return [dict(r) for r in _conn.execute(sql, args).fetchall()]


def _one(sql, args=()):
    rows = _rows(sql, args)
    return rows[0] if rows else None


def _exec(sql, args=()):
    with _lock:
        _conn.execute(sql, args)


def _update(table, row_id, fields):
    if not fields:
        return
    cols = ", ".join(f"{k} = ?" for k in fields)
    _exec(f"UPDATE {table} SET {cols} WHERE id = ?", (*fields.values(), row_id))


# ---------- media ----------

def add_media(**fields) -> str:
    mid = new_id()
    fields = {"id": mid, "created_at": time.time(), **fields}
    cols = ", ".join(fields)
    marks = ", ".join("?" for _ in fields)
    _exec(f"INSERT INTO media ({cols}) VALUES ({marks})", tuple(fields.values()))
    return mid


def get_media(mid):
    return _one("SELECT * FROM media WHERE id = ?", (mid,))


def find_media_by_path(path: str):
    return _one("SELECT * FROM media WHERE lower(path) = lower(?)", (path,))


def list_media():
    return _rows("SELECT * FROM media ORDER BY created_at DESC")


def update_media(mid, **fields):
    _update("media", mid, fields)


def relocate_media_paths(media_dir, proxy_dir) -> int:
    """程式資料夾搬家（或 data 換位置）後，資料庫裡的絕對路徑還指到舊位置。只改程式自己放的檔案：
      上傳：<舊 data>\\media\\uploads\\<12 碼 id>\\<檔名>
      網址下載：<舊 data>\\media\\<影片 id>\\<檔名>
      相容播放檔：<舊 data>\\proxy\\<影片 id>.mp4
    舊位置沒有檔案、新位置有，才改成新位置。本機路徑加入的影片不動。回傳改了幾筆。"""
    import re
    from pathlib import Path, PureWindowsPath

    is_id = re.compile(r"^[0-9a-f]{12}$").match

    def moved(raw, candidate):
        return candidate if raw and not Path(raw).exists() and candidate is not None and candidate.is_file() else None

    changed = 0
    for m in list_media():
        fields = {}
        parts = PureWindowsPath(m["path"]).parts if m["path"] else ()
        new = None
        if m["source"] == "upload" and len(parts) >= 4 and parts[-4].lower() == "media" \
                and parts[-3].lower() == "uploads" and is_id(parts[-2]):
            new = moved(m["path"], Path(media_dir) / "uploads" / parts[-2] / parts[-1])
        elif m["source"] == "url" and len(parts) >= 3 and parts[-3].lower() == "media" and parts[-2] == m["id"]:
            new = moved(m["path"], Path(media_dir) / m["id"] / parts[-1])
        if new is not None:
            fields["path"] = str(new)
        pparts = PureWindowsPath(m["proxy_path"]).parts if m.get("proxy_path") else ()
        if len(pparts) >= 2 and pparts[-2].lower() == "proxy" and pparts[-1].lower() == f"{m['id']}.mp4":
            new_proxy = moved(m["proxy_path"], Path(proxy_dir) / pparts[-1])
            if new_proxy is not None:
                fields["proxy_path"] = str(new_proxy)
        if fields:
            update_media(m["id"], **fields)
            changed += 1
    return changed


def media_needing_title_translation(limit=200):
    return _rows(
        "SELECT id, title, language FROM media WHERE (title_zh IS NULL OR title_zh = '') "
        "AND language IN ('ja', 'en') AND path IS NOT NULL ORDER BY created_at DESC LIMIT ?",
        (limit,),
    )


def delete_media(mid):
    with _lock:
        _conn.execute("DELETE FROM tracks WHERE media_id = ?", (mid,))
        _conn.execute("DELETE FROM jobs WHERE media_id = ? AND status NOT IN ('running')", (mid,))
        _conn.execute("DELETE FROM media WHERE id = ?", (mid,))


# ---------- tracks ----------

def add_track(media_id, lang, kind, model, cue_count, source_track_id=None) -> str:
    tid = new_id()
    _exec(
        "INSERT INTO tracks (id, media_id, lang, kind, model, source_track_id, cue_count, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (tid, media_id, lang, kind, model, source_track_id, cue_count, time.time()),
    )
    return tid


def get_track(tid):
    return _one("SELECT * FROM tracks WHERE id = ?", (tid,))


def update_track(tid, **fields):
    if isinstance(fields.get("health"), dict):
        fields["health"] = json.dumps(fields["health"], ensure_ascii=False)
    _update("tracks", tid, fields)


def list_tracks(media_id=None):
    if media_id:
        return _rows("SELECT * FROM tracks WHERE media_id = ? ORDER BY created_at", (media_id,))
    return _rows("SELECT * FROM tracks ORDER BY created_at")


def delete_track(tid):
    _exec("DELETE FROM tracks WHERE id = ?", (tid,))


# ---------- jobs ----------

def _decode_job(row):
    if row is None:
        return None
    row["params"] = json.loads(row["params"] or "{}")
    row["result"] = json.loads(row["result"]) if row.get("result") else None
    return row


def add_job(media_id, job_type, params, depends_on=None) -> str:
    jid = new_id()
    with _lock:
        top = _conn.execute("SELECT COALESCE(MAX(position), 0) FROM jobs").fetchone()[0]
        _conn.execute(
            "INSERT INTO jobs (id, media_id, type, params, depends_on, position, created_at, stage) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (jid, media_id, job_type, json.dumps(params, ensure_ascii=False), depends_on, top + 1, time.time(), "排隊中"),
        )
    return jid


def get_job(jid):
    return _decode_job(_one("SELECT * FROM jobs WHERE id = ?", (jid,)))


def list_jobs(finished_limit=40):
    # 暫停中的模型下載（paused）還沒結束，跟排隊、執行中的一起列，不受 finished_limit 限制
    active = _rows("SELECT * FROM jobs WHERE status IN ('running', 'queued', 'paused') ORDER BY position")
    done = _rows(
        "SELECT * FROM jobs WHERE status NOT IN ('running', 'queued', 'paused') ORDER BY finished_at DESC LIMIT ?",
        (finished_limit,),
    )
    return [_decode_job(r) for r in active + done]


def jobs_for_media(media_id):
    return [_decode_job(r) for r in _rows("SELECT * FROM jobs WHERE media_id = ? ORDER BY position", (media_id,))]


def update_job(jid, **fields):
    for key in ("params", "result"):
        if key in fields and not isinstance(fields[key], str) and fields[key] is not None:
            fields[key] = json.dumps(fields[key], ensure_ascii=False)
    _update("jobs", jid, fields)


def move_status(jid: str, from_status: str, **fields) -> bool:
    """任務狀態還是 from_status 時才改（跟工作執行緒搶同一個任務時不會改錯），有改到回 True。"""
    for key in ("params", "result"):
        if key in fields and not isinstance(fields[key], str) and fields[key] is not None:
            fields[key] = json.dumps(fields[key], ensure_ascii=False)
    cols = ", ".join(f"{k} = ?" for k in fields)
    with _lock:
        cur = _conn.execute(f"UPDATE jobs SET {cols} WHERE id = ? AND status = ?", (*fields.values(), jid, from_status))
        return cur.rowcount > 0


def runnable_jobs(types) -> list[dict]:
    """可以開始跑的任務（相依的前一步已完成），依佇列順序。
    前一步的任務已經不在（舊版「清除已結束」刪掉的）也算可以開始：不然會永遠排不到，顯卡也永遠不會閒置釋放。
    做得下去就照做（例如網址下載完的轉字幕），缺前一步的結果時任務自己會失敗並說明原因。"""
    marks = ", ".join("?" for _ in types)
    rows = _rows(
        f"SELECT j.* FROM jobs j LEFT JOIN jobs d ON j.depends_on = d.id "
        f"WHERE j.status = 'queued' AND j.type IN ({marks}) "
        f"AND (j.depends_on IS NULL OR d.status = 'done' OR d.id IS NULL) "
        f"ORDER BY j.position",
        tuple(types),
    )
    return [_decode_job(r) for r in rows]


def claim_job(job_id: str, stage: str | None = None) -> dict | None:
    """把任務標成執行中；已經被別人拿走就回 None。stage：同時換掉排隊時的說明（同一個 UPDATE，網頁不會看到
    「執行中」配上「已暫停」「排隊中」）。"""
    with _lock:
        cur = _conn.execute(
            "UPDATE jobs SET status = 'running', started_at = ?, error = NULL, stage = COALESCE(?, stage) "
            "WHERE id = ? AND status = 'queued'",
            (time.time(), stage, job_id),
        )
        if cur.rowcount == 0:
            return None
    return get_job(job_id)


def cancel_dependents(jid):
    """前一個任務失敗或取消時，連帶取消後續任務。"""
    for row in _rows("SELECT id FROM jobs WHERE depends_on = ? AND status = 'queued'", (jid,)):
        update_job(row["id"], status="canceled", stage="已取消", error=UPSTREAM_FAILED, finished_at=time.time())
        cancel_dependents(row["id"])


def requeue_dependents(jid):
    for row in _rows("SELECT id FROM jobs WHERE depends_on = ? AND status = 'canceled' AND error = ?", (jid, UPSTREAM_FAILED)):
        update_job(row["id"], status="queued", stage="排隊中", error=None, progress=0, finished_at=None)
        requeue_dependents(row["id"])


def move_job(jid, direction: str):
    with _lock:
        job = _one("SELECT * FROM jobs WHERE id = ?", (jid,))
        if not job or job["status"] != "queued":
            return
        if direction == "up":
            other = _one("SELECT * FROM jobs WHERE status = 'queued' AND position < ? ORDER BY position DESC LIMIT 1", (job["position"],))
        else:
            other = _one("SELECT * FROM jobs WHERE status = 'queued' AND position > ? ORDER BY position ASC LIMIT 1", (job["position"],))
        if other:
            _conn.execute("UPDATE jobs SET position = ? WHERE id = ?", (other["position"], jid))
            _conn.execute("UPDATE jobs SET position = ? WHERE id = ?", (job["position"], other["id"]))


def _detach_dependents(dep_filter: str = "", args=()):
    """要刪掉的已完成任務（d，dep_filter 可以再限定），後面還有依賴它的任務（例如轉字幕完成、翻譯還在排隊）：
    把後面任務需要的結果（翻譯要翻哪一條字幕）記到它自己的參數裡，再拿掉依賴。呼叫的地方要拿著 _lock。"""
    rows = _conn.execute(
        "SELECT k.id, k.type, k.params, d.result FROM jobs k JOIN jobs d ON k.depends_on = d.id "
        f"WHERE d.status = 'done' {dep_filter}", args).fetchall()
    for kid in rows:
        params = json.loads(kid["params"] or "{}")
        result = json.loads(kid["result"]) if kid["result"] else {}
        if kid["type"] == "translate" and not params.get("source_track_id") and (result or {}).get("track_id"):
            params["source_track_id"] = result["track_id"]
        _conn.execute("UPDATE jobs SET depends_on = NULL, params = ? WHERE id = ?",
                      (json.dumps(params, ensure_ascii=False), kid["id"]))


def delete_job(jid):
    with _lock:
        _detach_dependents("AND d.id = ?", (jid,))
        _conn.execute("DELETE FROM jobs WHERE id = ? AND status NOT IN ('running')", (jid,))


def clear_finished() -> list[str]:
    """刪掉已結束的任務，回傳刪掉的 id（呼叫端要一起清工作資料夾）。
    還在排隊的後續任務（例如翻譯）不會因為前一步的任務被刪掉而永遠排不到（_detach_dependents）。"""
    with _lock:
        ids = [r["id"] for r in _conn.execute("SELECT id FROM jobs WHERE status IN ('done', 'canceled', 'failed')")]
        _detach_dependents()
        _conn.execute("DELETE FROM jobs WHERE status IN ('done', 'canceled', 'failed')")
    return ids


# ---------- settings ----------

def get_settings() -> dict:
    out = {}
    for row in _rows("SELECT key, value FROM settings"):
        try:
            out[row["key"]] = json.loads(row["value"])
        except json.JSONDecodeError:
            pass
    return out


def set_settings(values: dict):
    with _lock:
        for k, v in values.items():
            _conn.execute("INSERT INTO settings (key, value) VALUES (?, ?) "
                          "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                          (k, json.dumps(v, ensure_ascii=False)))


def reset_interrupted():
    """程式重開時，把上次沒跑完的任務放回佇列（會從檢查點續跑）。"""
    _exec("UPDATE jobs SET status = 'queued', stage = '排隊中（上次中斷）' WHERE status = 'running'")
