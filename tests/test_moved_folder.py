"""程式資料夾搬家、同一台電腦有兩份程式時的測試，不用網路、不用顯示卡。

- 搬家後資料庫裡上傳、網址下載的影片路徑還指到舊位置：啟動時改成現在的 data 資料夾（影片照常播放、刪除時連檔案一起刪）
- 另一份 Bilingual Subtitles（別的資料夾）占著 8765：不是打開它的網頁就結束，而是換下一個 port

暫存檔放在系統暫存資料夾的 moved-folder-test，測完刪掉。

python -s tests/test_moved_folder.py
"""
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PY = sys.executable
BASE = Path(tempfile.gettempdir()) / "moved-folder-test"

from app import db, instance  # noqa: E402


def fresh(name: str) -> Path:
    d = BASE / name
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    return d


def _make_library(data: Path) -> dict:
    """舊位置的 data：上傳、網址下載、相容播放檔、本機路徑加入的影片各一部。"""
    data.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(data / "library.db", isolation_level=None)
    con.executescript(db.SCHEMA)
    ids = {"upload": "aaaaaaaaaaa1", "url": "bbbbbbbbbbb2", "local": "ccccccccccc3"}
    up = data / "media" / "uploads" / "0123456789ab" / "片段 (1).mp4"
    dl = data / "media" / ids["url"] / "Title [xyz].mp4"
    proxy = data / "proxy" / f"{ids['url']}.mp4"
    local = BASE / "elsewhere" / "local.mp4"
    for p, body in ((up, b"upload"), (dl, b"url"), (proxy, b"proxy"), (local, b"local")):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(body)
    now = time.time()
    rows = [(ids["upload"], "upload", str(up), None), (ids["url"], "url", str(dl), str(proxy)),
            (ids["local"], "local", str(BASE / "gone" / "media" / "uploads" / "0123456789ab" / "片段 (1).mp4"), None)]
    for mid, source, path, proxy_path in rows:
        con.execute("INSERT INTO media (id, title, source, path, proxy_path, created_at, playable) VALUES (?, ?, ?, ?, ?, ?, 1)",
                    (mid, mid, source, path, proxy_path, now))
    con.close()
    return ids


def test_relocate_media_paths_after_folder_move():
    old_root = fresh("move") / "BilingualSubtitles old"
    ids = _make_library(old_root / "data")
    new_root = old_root.with_name("BilingualSubtitles moved (2)")
    old_root.rename(new_root)
    data = new_root / "data"
    saved = (db.DB_PATH, db._conn)
    db.DB_PATH = data / "library.db"
    try:
        db.init()
        assert db.relocate_media_paths(data / "media", data / "proxy") == 2
        up, dl = db.get_media(ids["upload"]), db.get_media(ids["url"])
        assert up["path"] == str(data / "media" / "uploads" / "0123456789ab" / "片段 (1).mp4"), up
        assert dl["path"] == str(data / "media" / ids["url"] / "Title [xyz].mp4"), dl
        assert dl["proxy_path"] == str(data / "proxy" / f"{ids['url']}.mp4"), dl
        # 本機路徑加入的影片不動（就算形狀很像，新位置也有同名檔案）
        assert "gone" in db.get_media(ids["local"])["path"]
        # 第二次啟動沒有要改的
        assert db.relocate_media_paths(data / "media", data / "proxy") == 0
        # 檔案在舊位置還在（沒有搬家，只是換了 VS_DATA_DIR）：不改
        other = fresh("same") / "data"
        other_ids = _make_library(other)
        db._conn.close()
        db.DB_PATH = other / "library.db"
        db.init()
        assert db.relocate_media_paths(data / "media", data / "proxy") == 0
        assert db.get_media(other_ids["upload"])["path"].startswith(str(other))
    finally:
        if db._conn is not None:
            db._conn.close()
        db.DB_PATH, db._conn = saved


def test_moved_server_plays_and_deletes_uploaded_video():
    """真的用搬家後的 data 開伺服器（TestClient，不跑佇列）：影片播得到，刪除時上傳的檔案一起刪掉。"""
    old_root = fresh("move-server") / "BilingualSubtitles"
    ids = _make_library(old_root / "data")
    new_root = old_root.with_name("BilingualSubtitles moved")
    old_root.rename(new_root)
    data = new_root / "data"
    script = (
        "import json, sys\n"
        "from fastapi.testclient import TestClient\n"
        "from app import server\n"
        "out = {}\n"
        "with TestClient(server.app) as c:\n"
        f"    out['video'] = c.get('/media/{ids['upload']}/video').status_code\n"
        f"    out['proxy'] = c.get('/media/{ids['url']}/video').content.decode()\n"
        "    state = c.get('/api/state').json()\n"
        f"    out['has_proxy'] = [m['has_proxy'] for m in state['media'] if m['id'] == '{ids['url']}'][0]\n"
        f"    out['delete'] = c.delete('/api/media/{ids['upload']}').status_code\n"
        "print(json.dumps(out))\n")
    env = {k: v for k, v in os.environ.items() if not k.startswith("VS_")}
    env.update(PYTHONNOUSERSITE="1", PYTHONIOENCODING="utf-8", HF_HUB_OFFLINE="1", VS_NO_WORKERS="1",
               VS_DATA_DIR=str(data), VS_MODELS_DIR=str(new_root / "models"))
    res = subprocess.run([PY, "-s", "-c", script], cwd=str(ROOT), env=env, capture_output=True, text=True,
                         encoding="utf-8", timeout=180)
    assert res.returncode == 0, res.stderr[-2000:]
    out = json.loads(res.stdout.strip().splitlines()[-1])
    assert out == {"video": 200, "proxy": "proxy", "has_proxy": True, "delete": 200}, out
    assert not (data / "media" / "uploads" / "0123456789ab").exists()        # 以前會留下，log 寫 keeping files


def test_another_copy_on_the_port_is_skipped():
    """另一份 Bilingual Subtitles（不同 data 資料夾，或沒有 data_id 的舊版）占著 port：換下一個；同一份才打開它的網頁。不真的綁 port。"""
    real = (instance.bind_socket, instance.is_ours, instance.same_data)
    tried = []

    def fake_bind(host, port):
        tried.append(port)
        if port == 8765:
            raise OSError(None, "位址已被使用", None, 10048)
        return "sock"
    try:
        instance.bind_socket, instance.is_ours = fake_bind, lambda host, port, timeout=2.0: True
        instance.same_data = lambda host, port, timeout=2.0: False
        assert instance.open_port("127.0.0.1", 8765, explicit=False) == ("sock", 8766) and tried == [8765, 8766]
        try:
            instance.open_port("127.0.0.1", 8765, explicit=True)
        except instance.StartupStop as stop:
            assert stop.code == 1 and "另一份 Bilingual Subtitles" in str(stop) and stop.url is None, str(stop)
        else:
            raise AssertionError("明確指定的 port 被另一份占著時應該結束")
        instance.same_data = lambda host, port, timeout=2.0: True
        try:
            instance.open_port("127.0.0.1", 8765, explicit=False)
        except instance.StartupStop as stop:
            assert stop.code == 0 and stop.url == "http://127.0.0.1:8765/", str(stop)
        else:
            raise AssertionError("同一份程式已經在跑時應該打開它的網頁後結束")
    finally:
        instance.bind_socket, instance.is_ours, instance.same_data = real
    # data_id 跟著 data 資料夾：不同資料夾不同、大小寫不同算同一個
    from app import config
    saved = config.DATA_DIR
    try:
        config.DATA_DIR = Path(r"D:\BilingualSubtitles\data")
        a = instance.data_id()
        config.DATA_DIR = Path(r"d:\bilingualsubtitles\DATA")
        assert instance.data_id() == a and len(a) == 16
        config.DATA_DIR = Path(r"E:\Other\BilingualSubtitles\data")
        assert instance.data_id() != a
    finally:
        config.DATA_DIR = saved


if __name__ == "__main__":
    started = time.time()
    try:
        for name, fn in list(globals().items()):
            if name.startswith("test_") and callable(fn):
                fn()
                print("ok", name)
    finally:
        shutil.rmtree(BASE, ignore_errors=True)
    print(f"all passed in {time.time() - started:.1f}s")
