"""測試用的迷你 JMdict：在系統暫存資料夾建一個只有兩個條目的 jmdict.db，讓 app.dict_ja 改讀它。
不用先執行 tools/build_dict_ja.py，也不會讀寫 data/dict。

    restore = fake_jmdict.install()
    try:
        ...
    finally:
        restore()
"""
import json
import shutil
import sqlite3
import tempfile
from pathlib import Path

from app import dict_ja

ROOT = Path(__file__).resolve().parents[1]
RULES = ROOT / "tools" / "yomitan" / "yomitan_ja_transforms.json"

# 跟 tools/build_dict_ja.py 建出來的表一樣
SCHEMA = """
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE entry (id INTEGER PRIMARY KEY, data TEXT NOT NULL);
CREATE TABLE form (text TEXT NOT NULL, entry_id INTEGER NOT NULL, kana INTEGER NOT NULL, pri TEXT, rank INTEGER NOT NULL);
CREATE TABLE jlpt (entry_id INTEGER PRIMARY KEY, level INTEGER NOT NULL);
CREATE TABLE accent (term TEXT NOT NULL, reading TEXT NOT NULL, pattern TEXT NOT NULL, PRIMARY KEY (term, reading)) WITHOUT ROWID;
"""

EAT, I_ME = 1358280, 1311110          # 條目編號用 JMdict 的 ent_seq，內容只留測試用得到的欄位
ENTRIES = {
    EAT: {"k": [{"t": "食べる", "pri": ["ichi1", "nf25"], "inf": []}],
          "r": [{"t": "たべる", "pri": ["ichi1", "nf25"], "inf": []}],
          "s": [{"pos": ["v1", "vt"], "misc": [], "gloss": ["to eat"]},
                {"pos": ["v1", "vt"], "misc": [], "gloss": ["to live on", "to subsist on"]}]},
    I_ME: {"k": [{"t": "私", "pri": ["ichi1", "nf01"], "inf": []}],
           "r": [{"t": "わたし", "pri": ["ichi1", "nf01"], "inf": []}],
           "s": [{"pos": ["pn"], "misc": [], "gloss": ["I", "me"]}]},
}

# 字幕的 w、詞表的 meta 用這兩個詞時可以直接拿
META = {
    f"jm:{EAT}": {"display": "食べる", "reading": "たべる", "grp": "word", "entry_id": EAT, "pos": "動詞", "phrase": 0},
    f"jm:{I_ME}": {"display": "私", "reading": "わたし", "grp": "word", "entry_id": I_ME, "pos": "代名詞", "phrase": 0},
}


def install():
    """建好迷你字典並讓 dict_ja 改讀它，回傳還原用的函式（會關掉連線、刪掉暫存資料夾）。"""
    tmp = Path(tempfile.mkdtemp(prefix="vs-jmdict-"))
    path = tmp / "jmdict.db"
    con = sqlite3.connect(path)
    try:
        con.executescript(SCHEMA)
        for eid, d in ENTRIES.items():
            con.execute("INSERT INTO entry VALUES (?, ?)", (eid, json.dumps(d, ensure_ascii=False)))
            for kana, forms in ((0, d["k"]), (1, d["r"])):
                for f in forms:
                    con.execute("INSERT INTO form VALUES (?, ?, ?, ?, ?)", (f["t"], eid, kana, ",".join(f["pri"]), 25))
            con.execute("INSERT INTO jlpt VALUES (?, 5)", (eid,))
        con.executemany("INSERT INTO meta VALUES (?, ?)", [("schema_version", "1"), ("built_at", "test")])
        con.commit()
    finally:
        con.close()

    saved = (dict_ja.DB, dict_ja.RULES_PATH)
    dict_ja.close_all()
    dict_ja.DB, dict_ja.RULES_PATH = path, RULES

    def restore():
        dict_ja.close_all()                          # Windows 上連線開著就刪不掉檔案
        dict_ja.DB, dict_ja.RULES_PATH = saved
        shutil.rmtree(tmp, ignore_errors=True)

    return restore
