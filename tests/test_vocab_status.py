"""單字狀態（不會／已會／忽略）與例句酬載的檢查：python -s tests/test_vocab_status.py

不碰正式資料：片庫是系統暫存資料夾裡的資料庫和字幕（一部假的日文影片），日文字典用 tests/fake_jmdict.py 的迷你字典，
測完整個刪掉。不啟動佇列（VS_NO_WORKERS=1），不會碰顯卡。
分頁是前端做的（理由見 web/vocab.js 的單字頁區塊），所以這裡只驗狀態相關的 API。
"""
import json
import os
import shutil
import sqlite3
import sys
import tempfile
import time
from pathlib import Path

os.environ["VS_NO_WORKERS"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient  # noqa: E402

from app import cues as cue_mod  # noqa: E402
from app import db, server, vocab  # noqa: E402

import fake_jmdict  # noqa: E402

OLD_SCHEMA = ("CREATE TABLE word_status (key TEXT PRIMARY KEY, "
              "status INTEGER NOT NULL CHECK (status IN (2, 3)), updated_at REAL NOT NULL) WITHOUT ROWID")


def check_migration() -> dict:
    """舊資料庫（只收 2、3）升級後要能存 1，而且舊資料還在、重複執行沒事。"""
    tmp_dir = Path(tempfile.mkdtemp(prefix="vs-vocab-"))
    tmp = tmp_dir / "old.db"
    con = sqlite3.connect(tmp)
    con.execute(OLD_SCHEMA)
    con.executemany("INSERT INTO word_status VALUES (?, ?, ?)", [("jm:1", 2, 1.0), ("jm:2", 3, 2.0)])
    con.commit()
    con.close()

    saved = db._conn
    db._conn = sqlite3.connect(tmp, check_same_thread=False, isolation_level=None)
    db._conn.row_factory = sqlite3.Row
    try:
        vocab.init()
        vocab.init()                                    # 重複執行不應該出事
        kept = {r["key"]: r["status"] for r in db._rows("SELECT key, status FROM word_status")}
        assert kept == {"jm:1": 2, "jm:2": 3}, kept     # 舊的已會、忽略要保留
        vocab.set_status("jm:3", vocab.STATUS_UNSURE)   # 新狀態存得進去
        assert db._one("SELECT status FROM word_status WHERE key = 'jm:3'")["status"] == 1
        left = [r["name"] for r in db._rows("SELECT name FROM sqlite_master WHERE name LIKE 'word_status%'")]
        assert left == ["word_status"], left            # 不留暫存表
        sql = db._one("SELECT sql FROM sqlite_master WHERE name = 'word_status'")["sql"]
    finally:
        db._conn.close()
        db._conn = saved
        shutil.rmtree(tmp_dir, ignore_errors=True)
    assert "IN (1, 2, 3)" in sql, sql
    return {"kept": kept, "check": True}


def check_locked_startup() -> dict:
    """資料庫被別的連線鎖住寫入時（例如資料庫瀏覽工具開著沒存）：
    已經升級過的資料庫，vocab.init() 要照常跑完、不寫入；還沒升級的會丟錯，但連線不能停在交易中。"""
    tmp_dir = Path(tempfile.mkdtemp(prefix="vs-vocab-lock-"))
    saved = db._conn
    out = {}

    def connect(path):
        con = sqlite3.connect(path, check_same_thread=False, isolation_level=None, timeout=0.5)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA journal_mode=WAL")
        return con

    try:
        # 1) 新版資料庫
        new_db = tmp_dir / "new.db"
        db._conn = connect(new_db)
        vocab.init()
        holder = sqlite3.connect(new_db, isolation_level=None)
        holder.execute("BEGIN IMMEDIATE")
        holder.execute("INSERT INTO word_status VALUES ('jm:9', 2, 1.0)")
        try:
            t0 = time.time()
            vocab.init()
            out["migrated_locked_seconds"] = round(time.time() - t0, 3)
            assert not db._conn.in_transaction
        finally:
            holder.execute("ROLLBACK")
            holder.close()
        db._conn.close()

        # 2) 舊版資料庫（只收 2、3）
        old_db = tmp_dir / "old.db"
        db._conn = connect(old_db)
        db._conn.execute(OLD_SCHEMA)
        db._conn.executescript(vocab.SCHEMA)           # 其他表先建好，word_status 維持舊版
        holder = sqlite3.connect(old_db, isolation_level=None)
        holder.execute("BEGIN IMMEDIATE")
        try:
            try:
                vocab.migrate_status()
                raise AssertionError("被鎖住時應該升級不了")
            except sqlite3.OperationalError as err:
                out["old_locked_error"] = str(err)
            assert not db._conn.in_transaction
        finally:
            holder.execute("ROLLBACK")
            holder.close()
        vocab.migrate_status()                          # 鎖放掉之後就升級得了
        sql = db._one("SELECT sql FROM sqlite_master WHERE name = 'word_status'")["sql"]
        assert "IN (1, 2, 3)" in sql, sql
        out["old_upgraded_after_unlock"] = True
    finally:
        try:
            db._conn.close()
        except Exception:
            pass
        db._conn = saved
        shutil.rmtree(tmp_dir, ignore_errors=True)
    return out


def check_status_api(client: TestClient) -> dict:
    lib = client.get("/api/vocab?lang=ja")
    assert lib.status_code == 200, lib.text[:200]
    items = lib.json()["items"]
    assert items, "日文詞表是空的"
    target = next(x for x in items if x["grp"] == "word")
    key, before = target["k"], target["st"]
    seen = {}
    try:
        for st in (1, 2, 3, 0):
            r = client.put("/api/lexeme/status", json={"key": key, "status": st})
            assert r.status_code == 200, (st, r.text[:200])
            row = db._one("SELECT status FROM word_status WHERE key = ?", (key,))
            assert (row["status"] if row else 0) == st, (st, row)          # 0 是刪掉資料列
            again = next(x for x in client.get("/api/vocab?lang=ja").json()["items"] if x["k"] == key)
            assert again["st"] == st, (st, again["st"])
            detail = client.get(f"/api/lexeme?key={key}")
            assert detail.json()["st"] == st, (st, detail.json()["st"])
            seen[st] = True
        # 這條字幕軌的詞表也要看得到狀態
        client.put("/api/lexeme/status", json={"key": key, "status": 1})
        tv = db._one("SELECT track_id FROM track_vocab WHERE key = ?", (key,))
        lex = client.get(f"/api/tracks/{tv['track_id']}/vocab").json()["lex"]
        assert lex[key]["st"] == 1, lex[key]
    finally:
        client.put("/api/lexeme/status", json={"key": key, "status": before})
    assert client.put("/api/lexeme/status", json={"key": key, "status": 9}).status_code == 400
    assert client.put("/api/lexeme/status", json={"key": key, "status": -1}).status_code == 400
    assert client.put("/api/lexeme/status", json={"key": "nocolon", "status": 1}).status_code == 400
    restored = db._one("SELECT status FROM word_status WHERE key = ?", (key,))
    assert (restored["status"] if restored else 0) == before
    return {"key": key, "statuses_ok": sorted(seen), "restored": before}


def check_examples(client: TestClient) -> dict:
    """小視窗播放要用到的欄位：start、end、高亮位置，日文還要有假名。"""
    out = {}
    for lang in ("ja", "en"):
        items = client.get(f"/api/vocab?lang={lang}").json()["items"]
        if not items:
            continue
        key = next(x["k"] for x in items if x["grp"] == "word")
        groups = client.get(f"/api/vocab/examples?key={key}&lang={lang}").json()["groups"]
        assert groups, (lang, key)
        first = groups[0]["items"][0]
        for field in ("i", "start", "end", "text", "hl"):
            assert field in first, (lang, field, first)
        assert first["end"] > first["start"], first
        assert first["hl"] and all(0 <= a < b <= len(first["text"]) for a, b in first["hl"]), first
        out[lang] = {"key": key, "videos": len(groups), "ruby": "ruby" in first,
                     "sample": first["text"][:40], "hl": first["hl"]}
    return out


def setup_library(tmp: Path):
    """暫存片庫：一部日文影片、一條辨識字幕（兩句，帶詞位置和假名）和它的逐行翻譯，單字索引先建好。"""
    db.DB_PATH = tmp / "test.db"
    db.init()
    vocab.init()
    cue_mod.SUBS_DIR = tmp / "subs"
    eat, me = f"jm:{fake_jmdict.EAT}", f"jm:{fake_jmdict.I_ME}"
    mid = db.add_media(title="test", source="local", path=str(tmp / "video.mp4"), duration=10.0, language="ja")
    tid = db.add_track(mid, "ja", "asr", "Qwen3-ASR-1.7B", 2)
    cues = [{"start": 0.0, "end": 1.5, "text": "私は食べる", "w": [[0, 1, me], [2, 5, eat]],
             "ruby": [[0, 1, "わたし"], [2, 3, "た"]]},
            {"start": 2.0, "end": 3.5, "text": "もう食べる", "w": [[2, 5, eat]], "ruby": [[2, 3, "た"]]}]
    cue_mod.save_cues(tid, cues)
    vocab.index_track(tid, cues, "ja", {k: dict(v) for k, v in fake_jmdict.META.items()})
    zh = db.add_track(mid, "zh-TW", "translation", "Hy-MT2-7B", 2, source_track_id=tid)
    cue_mod.save_cues(zh, [{"start": 0.0, "end": 1.5, "text": "我吃"}, {"start": 2.0, "end": 3.5, "text": "已經吃了"}])


def main():
    report = {"migration": check_migration(), "locked_startup": check_locked_startup()}
    tmp = Path(tempfile.mkdtemp(prefix="vs-vocab-api-"))
    saved = (db.DB_PATH, db._conn, cue_mod.SUBS_DIR)
    restore_dict = fake_jmdict.install()
    try:
        setup_library(tmp)
        client = TestClient(server.app)
        report["status_api"] = check_status_api(client)
        report["examples"] = check_examples(client)
    finally:
        if db._conn is not None and db._conn is not saved[1]:
            db._conn.close()
        db.DB_PATH, db._conn, cue_mod.SUBS_DIR = saved
        restore_dict()
        shutil.rmtree(tmp, ignore_errors=True)
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
