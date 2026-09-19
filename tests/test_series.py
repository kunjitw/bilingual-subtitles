"""依作品分的播放列表（app/series.py）：資料庫補欄位、標記的清理、作品 → 季 → 集 的排序。
資料庫在暫存資料夾，不碰正式資料。

python -s tests/test_series.py
"""
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_safety import BASE, Env  # noqa: E402  （也會設好 VS_NO_WORKERS）

from fastapi.testclient import TestClient  # noqa: E402

from app import db, series, server  # noqa: E402


def run(fn):
    env = Env()
    try:
        fn(env)
    finally:
        env.close()


def test_migration_adds_column_once_and_keeps_rows():
    def body(env):
        old = env.tmp / "old.db"
        con = sqlite3.connect(old)
        # 加這個功能以前的 media 表（沒有 series_info，也沒有 title_zh）
        con.execute("CREATE TABLE media (id TEXT PRIMARY KEY, title TEXT NOT NULL, source TEXT NOT NULL, path TEXT, "
                    "url TEXT, duration REAL, vcodec TEXT, acodec TEXT, width INTEGER, height INTEGER, "
                    "playable INTEGER DEFAULT 1, proxy_path TEXT, has_thumb INTEGER DEFAULT 0, language TEXT, "
                    "profile TEXT DEFAULT 'general', created_at REAL NOT NULL)")
        con.execute("INSERT INTO media (id, title, source, url, created_at) VALUES ('aaaaaaaaaaaa', '舊影片', 'url', "
                    "'https://www.youtube.com/watch?v=x', 1)")
        con.commit()
        con.close()
        env.patch(db, "DB_PATH", old)
        for _ in range(2):                         # 跑兩次也不會出錯（第二次不再加）
            db._conn.close()
            db.init()
        cols = [r["name"] for r in db._conn.execute("PRAGMA table_info(media)").fetchall()]
        assert cols.count("series_info") == 1 and "title_zh" in cols, cols
        m = db.get_media("aaaaaaaaaaaa")
        assert m["title"] == "舊影片" and m["series_info"] is None
        # 沒有標記的影片不會出現在依作品的列表
        assert series.tree(db.list_media()) == []
    run(body)


def test_clean_keeps_known_fields_only():
    assert series.clean(None) is None and series.clean("x") is None
    assert series.clean({"season": "第一季"}) is None                 # 沒有作品名
    assert series.clean({"series": "   "}) is None
    got = series.clean({"series": "  小熊 \n 日記 ", "season": "第二季", "season_order": 2, "episode": 3,
                        "version": "中文配音", "path": "C:/x", "episode_label": None})
    assert got == {"series": "小熊 日記", "season": "第二季", "season_order": 2.0, "episode": 3.0,
                   "episode_label": "3", "version": "中文配音"}, got
    got = series.clean({"series": "x" * 500, "episode": float("nan"), "season_order": True, "episode_label": 12.5})
    assert len(got["series"]) == 120 and "episode" not in got and "season_order" not in got
    assert got["episode_label"] == "12.5"
    assert series.loads("{broken") is None and series.loads(None) is None
    assert series.loads(series.dumps({"series": "a", "episode": 1})) == {"series": "a", "episode": 1.0,
                                                                        "episode_label": "1"}


def _row(mid, created, info):
    return {"id": mid, "created_at": created, "series_info": series.dumps(series.clean(info)) if info else None}


def test_tree_groups_and_orders():
    rows = [
        _row("yt1", 50, None),                                                         # YouTube：沒有標記
        _row("s2e2", 10, {"series": "小熊", "season": "第二季", "season_order": 2, "episode": 2}),
        _row("s1e10", 11, {"series": "小熊", "season": "第一季", "season_order": 1, "episode": 10}),
        _row("s1e2", 12, {"series": "小熊", "season": "第一季", "season_order": 1, "episode": 2}),
        _row("s1zh1", 13, {"series": "小熊", "season": "第一季", "season_order": 1, "episode": 1,
                           "version": "中文配音"}),
        _row("s1e1", 14, {"series": "小熊", "season": "第一季", "season_order": 1, "episode": 1}),
        _row("sp", 15, {"series": "小熊", "season": "特別篇", "episode_label": "SP"}),     # 沒有季數、集數
        _row("s2e1", 16, {"series": "小熊", "season": "第二季", "season_order": 2, "episode": 1}),
        _row("s1e1b", 17, {"series": "小熊", "season": "第一季", "season_order": 1, "episode": 1}),  # 同一集加了兩次
        _row("other1", 5, {"series": "別部", "season": "第一季", "season_order": 1, "episode": 1}),
        _row("new1", 30, {"series": "最新的", "episode": 1}),
    ]
    t = series.tree(rows, subtitled={"s1e1", "s1e2", "s2e1", "other1"})
    # 作品：最近加入的在前
    assert [w["title"] for w in t] == ["最新的", "小熊", "別部"], [w["title"] for w in t]
    bear = t[1]
    assert [s["title"] for s in bear["seasons"]] == ["第一季", "第一季 中文配音", "第二季", "特別篇"]
    s1, s1zh, s2, sp = bear["seasons"]
    assert s1["items"] == ["s1e1", "s1e1b", "s1e2", "s1e10"]            # 照集數，不是照字串或加入時間
    assert s1zh["items"] == ["s1zh1"] and s2["items"] == ["s2e1", "s2e2"] and sp["items"] == ["sp"]
    assert (s1["count"], s1["subtitled"]) == (4, 2)
    assert (bear["count"], bear["subtitled"]) == (8, 3)
    assert t[0]["seasons"][0]["title"] == "其他"                         # 沒有季名
    assert "yt1" not in [i for w in t for s in w["seasons"] for i in s["items"]]
    # key 分得出不同季、不同版本
    assert len({s["key"] for s in bear["seasons"]}) == 4


def test_state_includes_series_tree():
    def body(env):
        a = env.add_media("url", url="https://www.youtube.com/watch?v=a")
        b = env.add_media("url", url="https://x.example/2",
                          series_info=series.dumps({"series": "作品", "season": "第一季", "season_order": 1, "episode": 2}))
        time.sleep(0.01)
        c = env.add_media("url", url="https://x.example/1",
                          series_info=series.dumps({"series": "作品", "season": "第一季", "season_order": 1, "episode": 1}))
        env.add_track(c)
        st = TestClient(server.app).get("/api/state").json()
        by_id = {m["id"]: m for m in st["media"]}
        assert by_id[a]["series"] is None and by_id[b]["series"]["episode"] == 2
        assert st["series"] == [{"key": "作品", "title": "作品", "count": 2, "subtitled": 1, "seasons": [
            {"key": "作品\u241f第一季\u241f", "title": "第一季", "count": 2, "subtitled": 1, "items": [c, b]}]}], st["series"]
    run(body)


if __name__ == "__main__":
    started = time.time()
    try:
        for name, fn in list(globals().items()):
            if name.startswith("test_") and callable(fn):
                fn()
                print("ok", name)
    finally:
        if BASE.exists() and not any(BASE.iterdir()):
            BASE.rmdir()
    print(f"all passed in {time.time() - started:.1f}s")
