"""glosses 任務的單元測試（假的 llama-server、記憶體裡的資料庫，不碰顯卡也不動正式資料）：
python -s tests/test_glosses.py

日文字典用 tests/fake_jmdict.py 在暫存資料夾建的迷你字典，不用先建 data/dict/jmdict.db。
"""
import os
import sqlite3
import sys
import time
from pathlib import Path

os.environ["VS_NO_WORKERS"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import db, dict_ja, glossgen, jobs, settings, vocab  # noqa: E402

import fake_jmdict  # noqa: E402


class FakeServer:
    cfg = {"parallel": 3}

    def __init__(self):
        self.calls = []

    def chat(self, messages, params):
        prompt = messages[-1]["content"]
        self.calls.append(prompt)
        if "to eat" in prompt:
            return "吃；维持生计"
        if "I; me" in prompt:
            return "我"
        return "テストです"          # 回日文：不合格，要存成空字串


def setup_memory_db():
    conn = sqlite3.connect(":memory:", check_same_thread=False, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.executescript(db.SCHEMA)
    for col in ("title_zh TEXT",):
        conn.execute(f"ALTER TABLE media ADD COLUMN {col}")
    for col in ("ruby_ver TEXT", "mode TEXT"):
        conn.execute(f"ALTER TABLE tracks ADD COLUMN {col}")
    db._conn = conn
    vocab.init()
    now = time.time()
    conn.execute("INSERT INTO media (id, title, source, created_at, language) VALUES ('m1', 'test', 'local', ?, 'ja')", (now,))
    conn.execute("INSERT INTO tracks (id, media_id, lang, kind, model, cue_count, created_at) "
                 "VALUES ('t1', 'm1', 'ja', 'asr', 'x', 2, ?)", (now,))
    meta = {"jm:1358280": {"display": "食べる", "reading": "たべる", "grp": "word", "entry_id": 1358280, "pos": "動詞", "phrase": 0},
            "jm:1311110": {"display": "私", "reading": "わたし", "grp": "word", "entry_id": 1311110, "pos": "代名詞", "phrase": 0}}
    cues = [{"start": 0, "end": 1, "text": "私", "w": [[0, 1, "jm:1311110"]]},
            {"start": 1, "end": 2, "text": "食べる", "w": [[0, 3, "jm:1358280"]]}]
    vocab.index_track("t1", cues, "ja", meta)


def test_collect_run_and_cache():
    setup_memory_db()
    items = glossgen.collect("media", "m1")
    senses = dict_ja.summary(1358280)["senses"][:glossgen.MAX_SENSES] + dict_ja.summary(1311110)["senses"][:5]
    assert set(items) == set(senses), items
    server = FakeServer()
    progress = []
    n = glossgen.run(server, items, "Hy-MT2-7B", lambda: None, progress.append)
    assert n == len(items) and progress[-1] == 1.0
    zh = vocab.sense_zh_map(senses)
    assert zh[dict_ja.summary(1358280)["senses"][0]] == "吃；維持生計", zh     # 簡體要轉成台灣繁體
    assert zh[dict_ja.summary(1311110)["senses"][0]] == "我"
    # 不合格的輸出存成空字串，不會一直重問
    assert glossgen.collect("media", "m1") == []
    payload = vocab.track_payload("t1")
    assert payload["lex"]["jm:1358280"]["gs"] == "mt" and payload["lex"]["jm:1358280"]["g"].startswith("吃"), payload


def test_handle_glosses_uses_shared_llm():
    setup_memory_db()
    fake = FakeServer()
    used = {}

    def fake_get_llm(key, log_path, check):
        used["key"] = key
        return fake, True

    orig_get, orig_installed, orig_model_installed = jobs.get_llm, settings.installed_translators, jobs.model_installed
    jobs.get_llm = fake_get_llm
    settings.installed_translators = lambda language=None: ["hymt", "sakura"]
    jobs.model_installed = lambda entry: True          # 模型是假的，當成已經下載（沒下載模型的電腦也能跑）
    try:
        assert jobs.model_key({"type": "glosses", "params": {"translator": "hymt"}}) == "llm:hymt"
        assert "glosses" in jobs.GPU_TYPES and jobs.HANDLERS["glosses"] is jobs.handle_glosses
        ctx = jobs.JobContext({"id": "j1", "media_id": "m1", "params": {"translator": "hymt", "scope": "media"}})
        jobs.handle_glosses(ctx)
        assert used["key"] == "hymt", used
        assert ctx.result["glosses"] > 0 and fake.calls, ctx.result
        # 第二次全部命中快取，不會再叫模型
        fake.calls.clear()
        used.clear()
        ctx = jobs.JobContext({"id": "j2", "media_id": "m1", "params": {"translator": "hymt", "scope": "media"}})
        jobs.handle_glosses(ctx)
        assert ctx.result == {"glosses": 0} and not fake.calls and "key" not in used
    finally:
        jobs.get_llm, settings.installed_translators, jobs.model_installed = orig_get, orig_installed, orig_model_installed


def test_clean_rejects_bad_output():
    assert glossgen.clean("「吃飯」。", "to eat") == "吃飯"
    assert glossgen.clean("食べる", "to eat") == ""
    assert glossgen.clean("", "to eat") == ""


if __name__ == "__main__":
    restore_dict = fake_jmdict.install()
    try:
        for name, fn in list(globals().items()):
            if name.startswith("test_"):
                fn()
                print("ok", name)
    finally:
        restore_dict()
