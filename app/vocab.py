"""單字：字幕切詞索引、詞表（滑鼠查字用）、單字頁統計、已會／忽略狀態。

資料放在 data/library.db（這裡自己建表，不動 db.py 的 SCHEMA）：
  lexemes      每個詞的顯示原形、讀音、分組（跨影片共用）
  track_vocab  每條字幕軌裡每個詞出現幾次、在哪幾句
  vocab_index  每條字幕軌的索引版本與字幕文字 sha1，任何一個不同就重算
  word_status  使用者標的狀態：1 不會、2 已會、3 忽略（沒有資料列 = 還沒標）
  sense_zh     JMdict 英文義項的中文翻譯快取（glosses 任務產生，永久保存）

只統計辨識出來的原文字幕（tracks.kind = 'asr'）。同一部影片有好幾條同語言的辨識字幕時，只算最新那條。
"""
import functools
import hashlib
import json
import logging
import re
import threading
import time
from collections import defaultdict
from contextlib import nullcontext

from . import cues as cue_mod
from . import db
from . import dict_en, dict_ja

log = logging.getLogger("vocab")

WORD_STATUS_SQL = """
CREATE TABLE IF NOT EXISTS word_status (
    key        TEXT PRIMARY KEY,
    status     INTEGER NOT NULL CHECK (status IN (1, 2, 3)),
    updated_at REAL NOT NULL
) WITHOUT ROWID
"""

SCHEMA = f"""
CREATE TABLE IF NOT EXISTS lexemes (
    key        TEXT PRIMARY KEY,
    lang       TEXT NOT NULL,
    display    TEXT NOT NULL,
    reading    TEXT,
    grp        TEXT NOT NULL DEFAULT 'word',   -- word | name | filler | number | affix | unknown | function
    pos        TEXT,
    entry_id   INTEGER,
    is_phrase  INTEGER NOT NULL DEFAULT 0,
    first_seen REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_lex_lang ON lexemes(lang, grp);
{WORD_STATUS_SQL};
CREATE TABLE IF NOT EXISTS track_vocab (
    track_id   TEXT NOT NULL,
    key        TEXT NOT NULL,
    count      INTEGER NOT NULL,
    first_cue  INTEGER NOT NULL,
    first_time REAL NOT NULL,
    cues       TEXT NOT NULL,
    PRIMARY KEY (track_id, key)
) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS idx_tv_key ON track_vocab(key);
CREATE TABLE IF NOT EXISTS vocab_index (
    track_id   TEXT PRIMARY KEY,
    lang       TEXT NOT NULL,
    version    TEXT NOT NULL,
    text_sha1  TEXT NOT NULL,
    units      INTEGER NOT NULL,
    counted    INTEGER NOT NULL,
    built_at   REAL NOT NULL
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS sense_zh (
    src        TEXT PRIMARY KEY,
    zh         TEXT NOT NULL,
    model      TEXT NOT NULL,
    created_at REAL NOT NULL
) WITHOUT ROWID;
"""

TRACK_LANG = {"ja": "ja", "en": "en", "zh": "zh-TW"}
LANG_OF_TRACK = {v: k for k, v in TRACK_LANG.items()}
STATUS_UNSURE, STATUS_KNOWN, STATUS_IGNORE = 1, 2, 3     # 0（沒有資料列）＝未標記
STATUSES = (STATUS_UNSURE, STATUS_KNOWN, STATUS_IGNORE)

_cue_lock = None                       # server 的 _ruby_lock：改寫字幕檔的人都要拿這把鎖
_build_locks: dict[str, threading.Lock] = defaultdict(threading.Lock)
_bg = {"thread": None, "wake": threading.Event(), "busy": False, "error": None}


def init():
    with db._lock:
        db._conn.executescript(SCHEMA)
    migrate_status()


_STATUS_CHECK = re.compile(r"\bstatus\s+IN\s*\(([^)]*)\)", re.IGNORECASE)


def _status_outdated(c) -> bool:
    """看 sqlite_master 裡的建表語法：CHECK 還沒包含全部狀態就要升級。只讀，資料庫被別的程式鎖住也不影響。"""
    row = c.execute("SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'word_status'").fetchone()
    m = _STATUS_CHECK.search(row[0] or "") if row else None
    if not m:
        return False
    allowed = {v.strip() for v in m.group(1).split(",")}
    return not {str(s) for s in STATUSES} <= allowed


def migrate_status():
    """舊資料庫的 word_status 只收 2、3，補上 1（不會）。重複執行沒有影響，舊資料會保留。
    已經升級過就只讀建表語法、不寫入，每次啟動不會去搶寫入鎖。"""
    with db._lock:
        c = db._conn
        if not _status_outdated(c):
            return
        c.execute("BEGIN IMMEDIATE")
        try:
            if not _status_outdated(c):                        # 拿到寫入鎖之前，另一個程序可能剛好升級完
                c.execute("ROLLBACK")
                return
            c.execute("DROP TABLE IF EXISTS word_status_old")   # 上次升到一半就中斷的話
            c.execute("ALTER TABLE word_status RENAME TO word_status_old")
            c.execute(WORD_STATUS_SQL)
            c.execute("INSERT OR IGNORE INTO word_status (key, status, updated_at) "
                      "SELECT key, status, updated_at FROM word_status_old WHERE status IN (1, 2, 3)")
            c.execute("DROP TABLE word_status_old")
            c.execute("COMMIT")
        except BaseException:
            if c.in_transaction:
                c.execute("ROLLBACK")
            raise
        log.info("word_status 已升級：加入「不會」狀態")


def set_cue_lock(lock):
    global _cue_lock
    _cue_lock = lock


# ---------- 版本與索引 ----------

def dict_ready(lang: str) -> bool:
    if lang == "ja":
        return dict_ja.ready()
    if lang == "en":
        return dict_en.ready() and dict_en.nlp_available()
    if lang == "zh":
        from . import vocab_zh
        return vocab_zh.ready()
    return False


def index_version(lang: str) -> str:
    if lang == "ja":
        return f"ja:{dict_ja.SEGMENT_RULES}:{dict_ja.TOKENIZER_VERSION}:{dict_ja.meta('built_at')}"
    if lang == "en":
        return f"en:{dict_en.SEGMENT_RULES}:{dict_en.MODEL_VERSION}:{dict_en.meta('built_at')}"
    from . import vocab_zh
    return f"zh:{vocab_zh.VERSION}"


def text_sha1(cues: list[dict]) -> str:
    return hashlib.sha1("\n".join(c.get("text", "") for c in cues).encode("utf-8")).hexdigest()


def _index_row(track_id: str):
    return db._one("SELECT version, text_sha1, units FROM vocab_index WHERE track_id = ?", (track_id,))


def is_fresh(t: dict, cues: list[dict] | None = None) -> bool:
    lang = LANG_OF_TRACK.get(t["lang"])
    row = _index_row(t["id"])
    if not row or row["version"] != index_version(lang):
        return False
    cues = cue_mod.load_cues(t["id"]) if cues is None else cues
    if row["text_sha1"] != text_sha1(cues):
        return False
    # 字幕檔被別的流程改寫過、文字沒變但 w 不見了
    if lang in ("ja", "en") and row["units"] and not any("w" in c for c in cues):
        return False
    return True


def annotate(cues: list[dict], lang: str) -> dict:
    meta: dict = {}
    if lang == "ja":
        prev = None
        for c in cues:
            # 把上一行也給 segment，接起 と｜いう、かも｜しれない、見せて｜いた、そう｜いう 這種被斷行切開的說法
            # （そう｜いう 會就地把上一行 w 裡的 そう 改成 そういう、標成不計數）
            units = dict_ja.segment(c.get("text", ""), meta, prev=prev and prev.get("text", ""),
                                    prev_brk=prev and prev.get("brk"), prev_units=prev and prev.get("w"))
            if units:
                c["w"] = units
            else:
                c.pop("w", None)
            prev = c
        # 同一個詞在這部片裡多數是什麼用法就算哪一組（お 偶爾被切成一般詞，不能因此整個算成單字）
        for m in meta.values():
            votes = m.pop("votes", None)
            if votes:
                m["grp"] = max(votes, key=lambda g: (votes[g], g == "word"))
    elif lang == "en":
        dict_en.annotate(cues, meta)
    return meta


def ensure(t: dict, cues: list[dict] | None = None) -> list[dict]:
    """ja/en 辨識字幕保證帶 w；版本或文字不同就重算、存檔、更新單字索引。回傳字幕。"""
    lang = LANG_OF_TRACK.get(t.get("lang"))
    if cues is None:
        cues = cue_mod.load_cues(t["id"])
    if t.get("kind") != "asr" or lang not in ("ja", "en") or not dict_ready(lang):
        return cues
    if is_fresh(t, cues):
        return cues
    with _build_locks[t["id"]]:                      # 兩個請求同時進來只重算一次
        if is_fresh(t):
            return cue_mod.load_cues(t["id"])
        started = time.time()
        with (_cue_lock or nullcontext()):
            cues = cue_mod.load_cues(t["id"])
            if not cues or not db.get_track(t["id"]):
                return cues
            meta = annotate(cues, lang)
            cue_mod.save_cues(t["id"], cues)
        index_track(t["id"], cues, lang, meta)
        log.info("vocab index %s (%s, %d cues) %.2fs", t["id"], lang, len(cues), time.time() - started)
    return cues


def index_track(track_id: str, cues: list[dict], lang: str, meta: dict, keys_per_cue: list | None = None):
    """keys_per_cue：中文沒有 w，由呼叫端直接給每句的詞 key。"""
    agg: dict = {}                                      # key -> [次數, 首句, 首句時間, [cue 索引]]
    for i, c in enumerate(cues):
        if keys_per_cue is not None:
            keys = keys_per_cue[i]
        else:
            # 日文跨行的詞只算一次，另一行那一段（w[4] 是 1）不計數，見 dict_ja 開頭的說明
            keys = [w[2] for w in c.get("w", []) if not (len(w) > 4 and w[4])] + [ph[0] for ph in c.get("ph", [])]
        for k in keys:
            a = agg.setdefault(k, [0, i, c.get("start", 0), []])
            a[0] += 1
            if not a[3] or a[3][-1] != i:
                a[3].append(i)
    counted = sum(a[0] for k, a in agg.items() if meta[k]["grp"] == "word" and not meta[k]["phrase"])
    now = time.time()
    with db._lock:
        c = db._conn
        c.execute("BEGIN")
        try:
            c.executemany(
                """INSERT INTO lexemes (key, lang, display, reading, grp, pos, entry_id, is_phrase, first_seen)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(key) DO UPDATE SET
                     grp = excluded.grp,
                     display = CASE WHEN excluded.entry_id IS NOT NULL THEN excluded.display ELSE lexemes.display END,
                     reading = CASE WHEN excluded.entry_id IS NOT NULL THEN excluded.reading ELSE lexemes.reading END,
                     pos = COALESCE(excluded.pos, lexemes.pos)""",
                [(k, lang, m["display"], m.get("reading"), m["grp"], m.get("pos"), m.get("entry_id"),
                  m.get("phrase", 0), now) for k, m in meta.items() if k in agg])
            c.execute("DELETE FROM track_vocab WHERE track_id = ?", (track_id,))
            c.executemany("INSERT INTO track_vocab VALUES (?, ?, ?, ?, ?, ?)",
                          [(track_id, k, a[0], a[1], a[2], json.dumps(a[3])) for k, a in agg.items()])
            c.execute("INSERT OR REPLACE INTO vocab_index VALUES (?, ?, ?, ?, ?, ?, ?)",
                      (track_id, lang, index_version(lang), text_sha1(cues), sum(a[0] for a in agg.values()),
                       counted, now))
            c.execute("COMMIT")
        except Exception:
            c.execute("ROLLBACK")
            raise


def forget_track(track_id: str):
    """刪除字幕軌時清掉索引；lexemes、word_status、sense_zh 是全域資料，不刪。"""
    with db._lock:
        db._conn.execute("DELETE FROM track_vocab WHERE track_id = ?", (track_id,))
        db._conn.execute("DELETE FROM vocab_index WHERE track_id = ?", (track_id,))


def counted_tracks(lang: str, media_id: str | None = None) -> list[dict]:
    """單字頁統計用的字幕軌：每部影片最新一條這個語言的辨識字幕。"""
    sql = ("SELECT t.* FROM tracks t WHERE t.kind = 'asr' AND t.lang = ? AND t.created_at = ("
           "SELECT MAX(t2.created_at) FROM tracks t2 WHERE t2.media_id = t.media_id AND t2.kind = 'asr' AND t2.lang = t.lang)")
    args = [TRACK_LANG[lang]]
    if media_id:
        sql += " AND t.media_id = ?"
        args.append(media_id)
    return db._rows(sql, args)


# ---------- 背景補算 ----------

def pending_tracks(lang: str | None = None) -> list[dict]:
    out = []
    for lg in ([lang] if lang else ["ja", "en"]):
        if not dict_ready(lg):
            continue
        for t in counted_tracks(lg):
            try:
                if not is_fresh(t):
                    out.append(t)
            except Exception:  # noqa: BLE001
                log.warning("vocab freshness check failed for %s", t["id"], exc_info=True)
    return out


def _bg_loop():
    while True:
        _bg["wake"].wait(600)
        _bg["wake"].clear()
        _bg["busy"] = True
        try:
            dict_ja.warm()
            for t in pending_tracks():
                if not db.get_track(t["id"]):
                    continue
                try:
                    ensure(t)
                except Exception as e:  # noqa: BLE001
                    _bg["error"] = f"{type(e).__name__}: {e}"
                    log.warning("vocab index failed for %s", t["id"], exc_info=True)
            # 先把片庫裡日文詞的條目摘要讀進快取，第一次打開單字頁不用等
            if dict_ja.ready():
                for r in db._rows("SELECT DISTINCT entry_id FROM lexemes WHERE lang = 'ja' AND entry_id IS NOT NULL"):
                    dict_ja.summary(r["entry_id"])
            from . import vocab_zh
            vocab_zh.run_pending()
        except Exception:  # noqa: BLE001
            log.warning("vocab background loop failed", exc_info=True)
        finally:
            _bg["busy"] = False


def kick():
    """請背景執行緒把還沒索引、或版本不同的字幕補算。"""
    if _bg["thread"] is None:
        _bg["thread"] = threading.Thread(target=_bg_loop, name="vocab-index", daemon=True)
        _bg["thread"].start()
    _bg["wake"].set()


# ---------- 釋義 ----------

def sense_zh_map(srcs) -> dict:
    srcs = list(dict.fromkeys(s for s in srcs if s))
    out = {}
    for k in range(0, len(srcs), 500):
        part = srcs[k:k + 500]
        for r in db._rows(f"SELECT src, zh FROM sense_zh WHERE src IN ({','.join('?' * len(part))}) AND zh != ''", part):
            out[r["src"]] = r["zh"]
    return out


def _gloss_ja(summ: dict, zh: dict, n: int = 2) -> tuple[str, str]:
    senses = summ["senses"][:n]
    got = [zh[s] for s in senses if s in zh]
    if got:
        return "；".join(got), "mt"
    # 還沒翻成中文：先顯示英文，每個義項只取前三個說法
    return " / ".join("; ".join(s.split("; ")[:3]) for s in senses), "en"


def _lex_info(lang: str, key: str, display: str, entry_id, pos: str | None, zh: dict | None = None) -> dict:
    """短釋義、等級、常用度（詞表和單字頁共用）。"""
    if lang == "ja":
        if entry_id and dict_ja.ready():
            s = dict_ja.summary(entry_id)
            g, gs = _gloss_ja(s, zh or {})
            return {"g": g, "gs": gs, "lv": f"N{s['lv']}" if s["lv"] else "", "common": s["rank"] <= 50,
                    "misc": s["misc"][:3]}
        return {"g": "", "gs": "", "lv": ""}
    if lang == "en":
        if dict_en.ready():
            return dict_en.summary(display, pos or "")
        return {"g": "", "gs": "", "lv": ""}
    from . import vocab_zh
    return vocab_zh.gloss(display)


def track_payload(track_id: str) -> dict:
    """整條字幕的詞表：滑鼠查字時只查這個，不打 API。"""
    t = db.get_track(track_id)
    lang = LANG_OF_TRACK.get(t["lang"]) if t else None
    if not t or lang not in ("ja", "en"):
        return {"lang": lang, "lex": {}}
    rows = db._rows(
        "SELECT tv.key, tv.count, l.display, l.reading, l.grp, l.pos, l.entry_id, l.is_phrase, "
        "COALESCE(ws.status, 0) AS st FROM track_vocab tv JOIN lexemes l ON l.key = tv.key "
        "LEFT JOIN word_status ws ON ws.key = tv.key WHERE tv.track_id = ?", (track_id,))
    zh = {}
    if lang == "ja" and dict_ja.ready():
        zh = sense_zh_map(s for r in rows if r["entry_id"] for s in dict_ja.summary(r["entry_id"])["senses"][:2])
    lex = {}
    for r in rows:
        item = {"d": r["display"], "r": r["reading"], "grp": r["grp"], "ph": r["is_phrase"], "st": r["st"],
                "n": r["count"]}
        item.update(_lex_info(lang, r["key"], r["display"], r["entry_id"], r["pos"], zh))
        lex[r["key"]] = item
    return {"lang": lang, "track": track_id, "lex": lex}


# ---------- 狀態 ----------

def set_status(key: str, status: int):
    """status：0 清掉標記、1 不會、2 已會、3 忽略。"""
    if status in STATUSES:
        db._exec("INSERT INTO word_status (key, status, updated_at) VALUES (?, ?, ?) "
                 "ON CONFLICT(key) DO UPDATE SET status = excluded.status, updated_at = excluded.updated_at",
                 (key, status, time.time()))
    else:
        db._exec("DELETE FROM word_status WHERE key = ?", (key,))


# ---------- 單字頁 ----------

def library_vocab(lang: str, media_id: str | None = None) -> dict:
    tracks = counted_tracks(lang, media_id)
    ready = dict_ready(lang)
    version = index_version(lang) if ready else ""
    fresh = [t for t in tracks if (row := _index_row(t["id"])) and row["version"] == version]
    stale = len(tracks) - len(fresh) if ready else 0
    if stale:
        kick()
    ids = [t["id"] for t in fresh]
    media_of = {t["id"]: t["media_id"] for t in fresh}
    agg: dict = {}
    for k in range(0, len(ids), 400):
        part = ids[k:k + 400]
        for r in db._rows(
                f"SELECT tv.track_id, tv.key, tv.count, tv.first_time, l.display, l.reading, l.grp, l.pos, "
                f"l.entry_id, l.is_phrase, COALESCE(ws.status, 0) AS st "
                f"FROM track_vocab tv JOIN lexemes l ON l.key = tv.key LEFT JOIN word_status ws ON ws.key = tv.key "
                f"WHERE tv.track_id IN ({','.join('?' * len(part))})", part):
            a = agg.get(r["key"])
            if a is None:
                a = agg[r["key"]] = {"k": r["key"], "d": r["display"], "r": r["reading"], "grp": r["grp"],
                                     "pos": r["pos"], "eid": r["entry_id"], "ph": r["is_phrase"], "st": r["st"],
                                     "n": 0, "v": 0}
            a["n"] += r["count"]
            a["v"] += 1
            mid = media_of[r["track_id"]]
            if media_id or "m" not in a:
                a["m"], a["t"] = mid, r["first_time"]
    zh = {}
    if lang == "ja" and dict_ja.ready():
        zh = sense_zh_map(s for a in agg.values() if a["eid"] for s in dict_ja.summary(a["eid"])["senses"][:2])
    items = []
    missing_zh = 0
    for a in agg.values():
        info = _lex_info(lang, a["k"], a["d"], a["eid"], a["pos"], zh)
        if lang == "ja" and info.get("gs") == "en" and a["grp"] == "word":
            missing_zh += 1
        items.append({"k": a["k"], "d": a["d"], "r": a["r"], "grp": a["grp"], "ph": a["ph"], "st": a["st"],
                      "n": a["n"], "v": a["v"], "g": info.get("g", ""), "gs": info.get("gs", ""),
                      "lv": info.get("lv", ""), "common": info.get("common", False), "t": round(a.get("t") or 0, 2)})
    items.sort(key=lambda x: (-x["n"], -x["v"], x["d"]))
    return {"lang": lang, "media": media_id or None, "tracks": len(tracks), "indexed": len(fresh),
            "pending": stale, "busy": _bg["busy"], "missing_zh": missing_zh, "items": items,
            "dict_ready": dict_ready(lang)}


@functools.lru_cache(maxsize=64)
def _cues_cached(track_id: str, mtime: float) -> list[dict]:
    return cue_mod.load_cues(track_id)


def cues_of(track_id: str) -> list[dict]:
    p = cue_mod.cue_path(track_id)
    return _cues_cached(track_id, p.stat().st_mtime) if p.exists() else []


def _translation_line(media_id: str, src_track_id: str, cue: dict) -> str:
    tr = [t for t in db.list_tracks(media_id) if t["kind"] == "translation" and t["source_track_id"] == src_track_id]
    if not tr:
        return ""
    cues = cues_of(tr[-1]["id"])
    mid = (cue["start"] + cue["end"]) / 2
    lo, hi = 0, len(cues) - 1
    while lo <= hi:
        m = (lo + hi) // 2
        if mid < cues[m]["start"]:
            hi = m - 1
        elif mid >= cues[m]["end"]:
            lo = m + 1
        else:
            return cues[m]["text"]
    return ""


def _highlights(lang: str, key: str, cue: dict, display: str) -> list:
    if lang == "zh":
        text, out, p = cue.get("text", ""), [], 0
        while display and (i := text.find(display, p)) >= 0:
            out.append([i, i + len(display)])
            p = i + len(display)
        return out
    spans = [[w[0], w[1]] for w in cue.get("w", []) if w[2] == key]
    for ph in cue.get("ph", []):
        if ph[0] == key:
            spans += ph[1]
    return sorted(spans)


def examples(key: str, lang: str, media_id: str | None = None, per_video: int = 4, limit_videos: int = 12,
             exclude: tuple | None = None) -> list[dict]:
    """某個詞在影片裡出現的句子。exclude=(track_id, cue_idx) 用來排除目前這句。"""
    tracks = {t["id"]: t for t in counted_tracks(lang, media_id)}
    if not tracks:
        return []
    lex = db._one("SELECT display FROM lexemes WHERE key = ?", (key,))
    display = lex["display"] if lex else ""
    rows = db._rows("SELECT track_id, count, cues FROM track_vocab WHERE key = ? ORDER BY count DESC", (key,))
    out = []
    for r in rows:
        t = tracks.get(r["track_id"])
        if not t:
            continue
        m = db.get_media(t["media_id"])
        if not m:
            continue
        cues = cues_of(t["id"])
        items = []
        for i in json.loads(r["cues"]):
            if exclude and exclude == (t["id"], i):
                continue
            if i >= len(cues):
                continue
            c = cues[i]
            item = {"i": i, "start": c["start"], "end": c["end"], "text": c.get("text", ""),
                    "hl": _highlights(lang, key, c, display), "zh": _translation_line(m["id"], t["id"], c)}
            if c.get("ruby"):
                item["ruby"] = c["ruby"]        # 小視窗播放時照樣標假名
            items.append(item)
            if len(items) >= per_video:
                break
        if items:
            out.append({"media_id": m["id"], "title": m["title"], "title_zh": m.get("title_zh"), "track_id": t["id"],
                        "count": r["count"], "items": items})
        if len(out) >= limit_videos:
            break
    return out


def library_languages() -> dict:
    """單字頁的語言切換：各語言有幾部影片。"""
    out = {}
    for lang, tl in TRACK_LANG.items():
        out[lang] = db._one("SELECT COUNT(DISTINCT media_id) AS n FROM tracks WHERE kind = 'asr' AND lang = ?", (tl,))["n"]
    return out


# ---------- 釘選卡 ----------

def lang_of_key(key: str) -> str:
    return {"jm": "ja", "ja": "ja", "en": "en", "zh": "zh"}.get(key.split(":", 1)[0], "")


def _cue_at(track_id: str, cue_idx: int):
    if not track_id or cue_idx is None or cue_idx < 0:
        return None
    cues = cues_of(track_id)
    return cues[cue_idx] if cue_idx < len(cues) else None


def lexeme_detail(key: str, track_id: str = "", cue_idx: int = -1, k: int = -1) -> dict:
    lang = lang_of_key(key)
    lex = db._one("SELECT * FROM lexemes WHERE key = ?", (key,)) or {}
    st = db._one("SELECT status FROM word_status WHERE key = ?", (key,))
    cue = _cue_at(track_id, cue_idx)
    w = cue["w"][k] if cue and 0 <= k < len(cue.get("w", [])) else None
    out = {"key": key, "lang": lang, "st": st["status"] if st else 0, "d": lex.get("display") or key.split(":", 1)[-1],
           "r": lex.get("reading"), "grp": lex.get("grp"), "ph": lex.get("is_phrase", 0)}

    if lang == "ja" and key.startswith("jm:") and dict_ja.ready():
        eid = int(key[3:])
        info = dict_ja.entry_info(eid)
        word, reading = dict_ja.headword(eid)
        srcs = [dict_ja.sense_src(s) for s in info.data["s"]]
        zh = sense_zh_map(srcs)
        s = dict_ja.summary(eid)
        out.update(
            d=word, r=reading, lv=f"N{s['lv']}" if s["lv"] else "", common=s["rank"] <= 50,
            accent=dict_ja.accent(word, reading),
            forms=[x["t"] for x in info.data["k"] if not set(x.get("inf") or []) & {"sK"}],
            readings=[x["t"] for x in info.data["r"] if not set(x.get("inf") or []) & {"sk"}],
            senses=[{"pos": sn.get("pos", []), "misc": sn.get("misc", []), "field": sn.get("field", []),
                     "dial": sn.get("dial", []), "inf": sn.get("inf", []), "en": src, "zh": zh.get(src, "")}
                    for sn, src in zip(info.data["s"], srcs)],
        )
        if cue and w:
            alts = []
            for c in dict_ja.scan(cue.get("text", ""), w[0], limit=5)["cands"]:
                if c["entry_id"] == eid:
                    continue
                cs = dict_ja.summary(c["entry_id"])
                g, gs = _gloss_ja(cs, sense_zh_map(cs["senses"][:1]), 1)
                alts.append({"key": c["key"], "d": cs["d"], "r": cs["r"], "g": g, "gs": gs,
                             "surface": cue["text"][w[0]:w[0] + c["len"]]})
            out["alts"] = alts[:3]
    elif lang == "ja":
        out.update(senses=[], pos=lex.get("pos"))
    elif lang == "en" and dict_en.ready():
        display = out["d"]
        surface = cue["text"][w[0]:w[1]] if cue and w else display
        upos = (w[3] if w and len(w) > 3 else "") or lex.get("pos") or ""
        e = dict_en.entry(display) or (dict_en.entry(display.lower()) if display != display.lower() else None)
        out.update(lv=dict_en.cefr(display.lower()), z=dict_en.zipf(display.lower()), upos=upos,
                   sp=dict_en.SPOKEN.get(display.lower(), ""))
        if e:
            out.update(word=e["word"], pn=e["phonetic"], lines=dict_en.sort_lines(e["lines"], "" if out["ph"] else upos),
                       collins=e["collins"], oxford=e["oxford"], toefl=e["toefl"], ielts=e["ielts"])
        else:
            out.update(lines=[])
        if not out["ph"] and surface.lower() != display.lower():
            se = dict_en.entry(surface) or dict_en.entry(surface.lower())
            if se and se["lines"]:
                out["surface"] = {"word": se["word"], "lines": dict_en.sort_lines(se["lines"], upos)[:3]}
    elif lang == "zh":
        from . import vocab_zh
        out.update(vocab_zh.detail(out["d"]))

    t = db.get_track(track_id) if track_id else None
    media_id = t["media_id"] if t else None
    if lang:
        this = examples(key, lang, media_id, per_video=3, exclude=(track_id, cue_idx)) if media_id else []
        others = [g for g in examples(key, lang, per_video=1, limit_videos=8) if g["media_id"] != media_id]
        stats = db._one(
            "SELECT COUNT(*) AS v, COALESCE(SUM(tv.count), 0) AS n FROM track_vocab tv JOIN vocab_index vi "
            "ON vi.track_id = tv.track_id WHERE tv.key = ?", (key,))
        out.update(examples=this[0]["items"] if this else [], others=others[:4],
                   total_videos=stats["v"] if stats else 0, total_count=stats["n"] if stats else 0)
    return out


def scan_payload(track_id: str, cue_idx: int, pos: int) -> dict:
    """Shift 即時查詢：從游標的字元位置做最長比對（只有日文）。"""
    cue = _cue_at(track_id, cue_idx)
    if not cue or not dict_ja.ready() or not 0 <= pos < len(cue.get("text", "")):
        return {"len": 0, "cands": []}
    res = dict_ja.scan(cue["text"], pos)
    zh = sense_zh_map(s for c in res["cands"] for s in dict_ja.summary(c["entry_id"])["senses"][:2])
    for c in res["cands"]:
        s = dict_ja.summary(c["entry_id"])
        g, gs = _gloss_ja(s, zh)
        c.update(d=s["d"], r=s["r"], g=g, gs=gs, lv=f"N{s['lv']}" if s["lv"] else "", common=s["rank"] <= 50,
                 misc=s["misc"][:3], surface=cue["text"][pos:pos + c["len"]])
        st = db._one("SELECT status FROM word_status WHERE key = ?", (c["key"],))
        c["st"] = st["status"] if st else 0
    return res
