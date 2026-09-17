"""日文單字的中文釋義：把 JMdict 的英文義項交給 Hy-MT2 翻成繁體中文，存進 sense_zh 永久快取。

只在佇列的 GPU 執行緒（glosses 任務）裡跑，用 jobs.get_llm 取得已載入的 llama-server；
查字的時候只讀快取，絕對不載入模型。

以英文原文當主鍵：同樣的英文只翻一次，JMdict 更新後英文沒變就不用重翻。
"""
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor

from . import db, dict_ja, settings, vocab
from .cues import to_taiwan

log = logging.getLogger("glossgen")

PROMPT = "将以下文本翻译为中文，注意只需要输出翻译后的结果，不要额外解释：\n\n{}"
MAX_SENSES = 5          # 每個條目只自動翻前幾個義項，其餘在釘選卡顯示英文
CHUNK = 64              # 每翻完一段就寫入一次，取消時最多浪費一段
_KANA = re.compile(r"[ぁ-ゖァ-ヺ]")


def _params():
    from .translate import HYMT_PARAMS
    return dict(HYMT_PARAMS, temperature=0.2, max_tokens=160)


GLOSS_TRANSLATORS = ("hymt", "hymt-mini")   # 優先順序：Hy-MT2-7B，再來 Hy-MT2-1.8B


def translator_key(without=()) -> str | None:
    """產生釋義用哪個模型：已下載、而且這張顯示卡放得下的 Hy-MT2-7B，不行就用 Hy-MT2-1.8B。
    兩個都放不下時還是回傳已下載的第一個（任務會說明這張卡放不下）；讀不到顯卡時不看顯存。
    without：當成已經刪掉的翻譯模型 key。"""
    from . import gpu
    from .config import translator_vram_mb
    installed = settings.installed_translators(without=without) if without else settings.installed_translators()
    candidates = [k for k in GLOSS_TRANSLATORS if k in installed]
    if not candidates:
        return None
    try:
        info = gpu.query(max_age=10)
        fitting = [k for k in candidates if gpu.fits(translator_vram_mb(k), info) is not False]
    except Exception:  # noqa: BLE001
        fitting = candidates
    return (fitting or candidates)[0]


def collect(scope: str, media_id: str | None) -> list[str]:
    """還沒有中文的英文義項（去重）。scope：media 這部影片、library 全部日文影片。"""
    if not dict_ja.ready():
        return []
    tracks = vocab.counted_tracks("ja", media_id if scope == "media" else None)
    if scope == "media" and not media_id:
        return []
    for t in tracks:
        try:
            vocab.ensure(t)                       # 字幕剛轉好還沒索引時先補算（CPU，一條 1 秒內）
        except Exception:  # noqa: BLE001
            log.warning("vocab ensure failed for %s", t["id"], exc_info=True)
    ids = [t["id"] for t in tracks]
    eids = set()
    for k in range(0, len(ids), 400):
        part = ids[k:k + 400]
        eids |= {r["entry_id"] for r in db._rows(
            f"SELECT DISTINCT l.entry_id FROM track_vocab tv JOIN lexemes l ON l.key = tv.key "
            f"WHERE tv.track_id IN ({','.join('?' * len(part))}) AND l.entry_id IS NOT NULL", part)}
    todo = list(dict.fromkeys(src for eid in sorted(eids)
                              for src in dict_ja.summary(eid)["senses"][:MAX_SENSES] if src))
    have = set()
    for k in range(0, len(todo), 500):
        part = todo[k:k + 500]
        have |= {r["src"] for r in db._rows(f"SELECT src FROM sense_zh WHERE src IN ({','.join('?' * len(part))})", part)}
    return [s for s in todo if s not in have]


def pending_count(scope: str = "library", media_id: str | None = None) -> int:
    try:
        return len(collect(scope, media_id))
    except Exception:  # noqa: BLE001
        log.warning("count pending glosses failed", exc_info=True)
        return 0


def clean(out: str, src: str) -> str:
    from .translate import _clean_output
    text = _clean_output(out or "")
    text = text.strip().rstrip("。.").strip("「」“”\"' ").rstrip("。.")
    # 模型偶爾回日文、或把說明也寫出來：不合格就存空字串，之後顯示英文，不會一直重問
    if not text or _KANA.search(text) or len(text) > max(40, len(src) * 2):
        return ""
    return to_taiwan(text).strip()


def run(server, items: list[str], model: str, check, on_progress) -> int:
    params = _params()

    def ask(src: str) -> str:
        return clean(server.chat([{"role": "user", "content": PROMPT.format(src)}], params), src)

    done = 0
    pool = ThreadPoolExecutor(max(1, server.cfg.get("parallel", 2)))
    try:
        for s in range(0, len(items), CHUNK):
            chunk = items[s:s + CHUNK]
            futures = [pool.submit(ask, it) for it in chunk]
            results = []
            for src, fut in zip(chunk, futures):
                check()
                results.append((src, fut.result()))
            write(results, model)                     # 只在這條執行緒寫 SQLite
            done += len(chunk)
            on_progress(done / len(items))
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    return done


def write(results: list[tuple[str, str]], model: str):
    now = time.time()
    with db._lock:
        c = db._conn
        c.execute("BEGIN")
        try:
            c.executemany("INSERT OR REPLACE INTO sense_zh (src, zh, model, created_at) VALUES (?, ?, ?, ?)",
                          [(src, zh, model, now) for src, zh in results])
            c.execute("COMMIT")
        except Exception:
            c.execute("ROLLBACK")
            raise


def queued_job(scope: str, media_id: str | None):
    for j in db.list_jobs():
        if j["type"] == "glosses" and j["status"] in ("queued", "running"):
            p = j["params"] or {}
            if p.get("scope") == "library" or (scope == "media" and j["media_id"] == media_id):
                return j
    return None


def enqueue(scope: str, media_id: str | None, depends_on: str | None = None) -> str | None:
    key = translator_key()
    if not key or queued_job(scope, media_id):
        return None
    label = "日文單字釋義（全部影片）" if scope == "library" else "日文單字釋義"
    return db.add_job(media_id if scope == "media" else None, "glosses",
                      {"translator": key, "scope": scope, "label": label}, depends_on=depends_on)


def maybe_enqueue_after_pipeline(media_id: str, language: str, translator: str | None, last_job: str):
    """轉字幕並用 Hy-MT2 翻譯的日文影片，翻完順便產生單字釋義（模型已經載入，不用多載一次）。"""
    try:
        if language != "ja" or translator != "hymt" or not settings.get("auto_glosses") or not dict_ja.ready():
            return None
        return enqueue("media", media_id, depends_on=last_job)
    except Exception:  # noqa: BLE001
        log.warning("enqueue glosses failed", exc_info=True)
        return None
