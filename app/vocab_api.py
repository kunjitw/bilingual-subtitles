"""查字與單字頁的 API（server.py 用 include_router 掛上）。

查字時只讀字典和快取，絕對不載入任何顯卡模型；中文釋義由佇列的 glosses 任務產生。
"""
import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from . import db, dict_en, dict_ja, glossgen, jobs, safepath, settings, vocab, vocab_zh
from .config import DICT_DIR

log = logging.getLogger("vocab_api")
router = APIRouter()


def _track(tid: str) -> dict:
    t = db.get_track(tid) if safepath.is_id(tid) else None
    if not t:
        raise HTTPException(404, "找不到字幕軌")
    return t


@router.get("/api/tracks/{tid}/vocab")
def track_vocab(tid: str):
    t = _track(tid)
    try:
        vocab.ensure(t)
    except Exception:  # noqa: BLE001
        log.warning("vocab ensure failed for %s", tid, exc_info=True)
    return vocab.track_payload(tid)


@router.get("/api/lexeme")
def lexeme(key: str, track: str = "", cue: int = -1, k: int = -1):
    if not key or ":" not in key:
        raise HTTPException(400, "缺少要查的詞")
    # track 只能是資料庫裡的字幕軌，不能拿網址參數去讀別的檔案；找不到就當成沒帶
    if track and not (safepath.is_id(track) and db.get_track(track)):
        track = ""
    return vocab.lexeme_detail(key, track, cue, k)


@router.get("/api/dict/scan")
def dict_scan(track: str, cue: int, pos: int):
    _track(track)
    return vocab.scan_payload(track, cue, pos)


class StatusBody(BaseModel):
    key: str
    status: int


@router.put("/api/lexeme/status")
def lexeme_status(body: StatusBody):
    if not body.key or ":" not in body.key or body.status not in (0, 1, 2, 3):
        raise HTTPException(400, "狀態不正確")
    vocab.set_status(body.key, body.status)
    return {"ok": True, "key": body.key, "status": body.status}


@router.get("/api/vocab")
def library_vocab(lang: str = "ja", media: str = ""):
    if lang not in vocab.TRACK_LANG:
        raise HTTPException(400, "不支援這個語言")
    if lang == "zh" and vocab_zh.ready():
        vocab.kick()
    data = vocab.library_vocab(lang, media or None)
    if lang == "zh":
        data["zh"] = vocab_zh.status()
    return data


@router.get("/api/vocab/examples")
def vocab_examples(key: str, lang: str, media: str = ""):
    if lang not in vocab.TRACK_LANG:
        raise HTTPException(400, "不支援這個語言")
    return {"groups": vocab.examples(key, lang, media or None, per_video=4, limit_videos=12)}


@router.get("/api/vocab/langs")
def vocab_langs():
    return vocab.library_languages()


def _size_mb(path) -> int:
    return int(path.stat().st_size / 1024 / 1024) if path.exists() else 0


def _build_info(lang: str, job_list: list) -> dict:
    """建字典的任務（app/dict_build.py）：job 是排隊或建立中的任務（沒有時是字典還沒建好時最近一次失敗的），
    est_s 是作者電腦上建置大約要幾秒（不含下載），download_bytes 是要下載的來源大小。"""
    from . import dict_build, setup
    return {"job": setup.dict_job_view(lang, job_list), "est_s": dict_build.EST_S[lang],
            "download_bytes": dict_build.source_bytes(lang)}


@router.get("/api/dicts")
def dicts():
    job_list = db.list_jobs()
    return {
        "dir": str(DICT_DIR),
        "ja": {"ready": dict_ja.ready(), "built_at": dict_ja.meta("built_at"), "source_date": dict_ja.meta("jmdict_date"),
               "entries": dict_ja.meta("entries"), "size_mb": _size_mb(dict_ja.DB), **_build_info("ja", job_list)},
        "en": {"ready": dict_en.ready(), "nlp": dict_en.nlp_available(), "built_at": dict_en.meta("built_at"),
               "entries": dict_en.meta("entries"), "size_mb": _size_mb(dict_en.DB), **_build_info("en", job_list)},
        "zh": {**vocab_zh.status(), "size_mb": _size_mb(vocab_zh.REF_DB), **_build_info("zh", job_list)},
        "glosses": {"translator": glossgen.translator_key(), "auto": bool(settings.get("auto_glosses")),
                    "queued": bool(glossgen.queued_job("library", None))},
    }


@router.get("/api/glosses/pending")
def glosses_pending(media: str = ""):
    scope = "media" if media else "library"
    return {"count": glossgen.pending_count(scope, media or None), "translator": glossgen.translator_key(),
            "queued": bool(glossgen.queued_job(scope, media or None))}


class GlossReq(BaseModel):
    scope: str = "library"
    media_id: str | None = None


@router.post("/api/glosses")
def glosses_enqueue(body: GlossReq):
    scope = "media" if body.scope == "media" else "library"
    if not dict_ja.ready():
        raise HTTPException(400, "日文字典還沒建立")
    if not glossgen.translator_key():
        raise HTTPException(400, "產生中文釋義需要 Hy-MT2-7B，請先到模型管理下載")
    if scope == "media" and not db.get_media(body.media_id or ""):
        raise HTTPException(404, "找不到影片")
    if glossgen.queued_job(scope, body.media_id):
        raise HTTPException(409, "單字釋義任務已經在佇列裡")
    jid = glossgen.enqueue(scope, body.media_id)
    jobs.wake()
    return {"ok": True, "job_id": jid}
