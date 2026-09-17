"""刪檔、寫檔、外部程式的安全測試。不用顯卡，也不碰正式資料：
資料庫、data/media、data/subs 這些路徑常數全部換到系統暫存資料夾的 safety-test 底下，測完整個刪掉。

python -s tests/test_safety.py
"""
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

os.environ["VS_NO_WORKERS"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
_PRESET_ENV = {k for k in ("HF_HOME", "HF_HUB_CACHE", "HF_XET_CACHE", "HF_MODULES_CACHE", "TORCH_HOME", "MPLCONFIGDIR")
               if k in os.environ}
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import HTTPException  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app import config, db, jobs, media, models, safepath, server, settings, translate, vocab, vocab_api  # noqa: E402
from app import cues as cue_mod  # noqa: E402
from app import health  # noqa: E402

BASE = Path(tempfile.gettempdir()) / "safety-test"
PING = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "PING.EXE"


def touch(p: Path, text: str = "x") -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


def junction(link: Path, target: Path):
    import _winapi
    link.parent.mkdir(parents=True, exist_ok=True)
    _winapi.CreateJunction(str(target), str(link))


def expect_http(status: int, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except HTTPException as e:
        assert e.status_code == status, (e.status_code, e.detail)
        return e
    raise AssertionError(f"預期 HTTP {status}，結果沒有丟例外")


class Env:
    """假的專案資料夾和資料庫，全部在 %TEMP%\\safety-test\\vs-safety-xxxx 底下。"""

    def __init__(self):
        BASE.mkdir(parents=True, exist_ok=True)
        self.tmp = Path(tempfile.mkdtemp(prefix="vs-safety-", dir=BASE))
        self.proj = self.tmp / "proj"
        self.data = self.proj / "data"
        self.media = self.data / "media"
        self.subs = self.data / "subs"
        self.thumbs = self.data / "thumbs"
        self.proxy = self.data / "proxy"
        self.work = self.data / "work"
        self.models = self.proj / "models"
        self.outside = self.tmp / "outside"   # 專案外：使用者自己的檔案
        for d in (self.media / "uploads", self.subs, self.thumbs, self.proxy, self.work, self.models, self.outside):
            d.mkdir(parents=True, exist_ok=True)
        # 假的專案資料夾可以當刪除的 root（safepath 只認明確註冊、在暫存資料夾底下的）；outside 也在裡面，
        # 所以每個測試還是要傳正確的 root（env.media、env.work……），檢查的是「不會跑出 root」
        safepath.TEST_ROOTS.append(self.proj)
        self._saved = []
        self._saved_conn = db._conn
        self.patch(db, "DB_PATH", self.tmp / "test.db")
        db.init()
        vocab.init()
        for mod in (server, jobs):
            self.patch(mod, "MEDIA_DIR", self.media)
            self.patch(mod, "THUMB_DIR", self.thumbs)
            self.patch(mod, "PROXY_DIR", self.proxy)
        self.patch(server, "MODELS_DIR", self.models)
        self.patch(cue_mod, "SUBS_DIR", self.subs)
        self.patch(jobs, "WORK_DIR", self.work)
        self._levels = []
        for name in ("safepath", "server", "jobs", "models"):
            logger = logging.getLogger(name)
            self._levels.append((logger, logger.level))
            logger.setLevel(logging.CRITICAL)   # 拒絕刪除的 log 是預期的，不要洗版

    def patch(self, obj, name, value):
        self._saved.append((obj, name, getattr(obj, name)))
        setattr(obj, name, value)

    def close(self):
        for obj, name, value in reversed(self._saved):
            setattr(obj, name, value)
        for logger, level in self._levels:
            logger.setLevel(level)
        db._conn.close()
        db._conn = self._saved_conn
        safepath.TEST_ROOTS.remove(self.proj)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def add_media(self, source, path=None, **kw):
        return db.add_media(title="test", source=source, path=str(path) if path else None, **kw)

    def add_track(self, mid, text="hi"):
        tid = db.add_track(mid, "ja", "asr", "test", 1)
        cue_mod.save_cues(tid, [{"start": 0, "end": 1, "text": text}])
        touch(health.backup_path(tid), "[]")
        return tid


def run(fn):
    env = Env()
    try:
        fn(env)
    finally:
        env.close()


# ---------- 1. safe_unlink / safe_rmtree ----------

def test_safe_rmtree_refuses_outside_links_and_root():
    def body(env: Env):
        root = env.media
        victim = touch(env.outside / "victim" / "diary.txt")
        assert not safepath.safe_rmtree(env.outside / "victim", root)                      # 專案外
        assert not safepath.safe_rmtree(Path(str(root) + r"\x\..\..\..\outside\victim"), root)  # 用 .. 繞出去
        assert not safepath.safe_rmtree(root, root)                                          # root 本身
        assert not safepath.safe_rmtree(root / "..", root)
        (root / "sub").mkdir()
        for tail in (".. ", "...", ". ", " "):                  # Windows 會把結尾的空白和點去掉
            assert not safepath.safe_rmtree(Path(str(root / "sub") + "\\" + tail), root), tail
            assert not safepath.safe_unlink(Path(str(root / "sub") + "\\" + tail), root), tail
        assert victim.exists() and (root / "sub").exists()

        # junction 本身、以及經過 junction 的路徑都不刪
        junction(root / "jx", env.outside / "victim")
        assert not safepath.safe_rmtree(root / "jx", root)
        assert not safepath.safe_unlink(root / "jx" / "diary.txt", root)
        touch(env.outside / "victim" / "sub" / "notes.txt")
        assert not safepath.safe_rmtree(root / "jx" / "sub", root)
        assert victim.exists() and (env.outside / "victim" / "sub" / "notes.txt").exists()
        assert (root / "jx").exists()

        # symlink（要開發人員模式或系統管理員才能建，建不了就跳過這段）
        try:
            os.symlink(env.outside / "victim", root / "sx", target_is_directory=True)
        except OSError:
            print("  （沒有建立 symlink 的權限，跳過 symlink 檢查）")
        else:
            assert not safepath.safe_rmtree(root / "sx", root)
            assert victim.exists()

        # 資料夾裡面藏了 junction：資料夾照刪，但不會跟進去刪專案外的東西
        real = root / "real"
        touch(real / "a.mp4")
        junction(real / "link", env.outside / "victim")
        assert safepath.safe_rmtree(real, root)
        assert not real.exists() and victim.exists()

        # 大小寫不同但確實在裡面：可以刪
        touch(root / "CaseDir" / "x.mp4")
        assert safepath.safe_rmtree(Path(str(root / "CaseDir").upper()), root)
        assert not (root / "CaseDir").exists()

        # root 不是專案允許刪除的資料夾：一律拒絕（目標故意用不存在的名字，萬一檢查壞掉也不會刪到東西）
        desktop = Path.home() / "Desktop"
        assert not safepath.safe_rmtree(desktop / "vs-safety-no-such-folder-7f3a", desktop)
        assert not safepath.safe_unlink(desktop / "vs-safety-no-such-file-7f3a.txt", desktop)
        temp = Path(tempfile.gettempdir())               # 暫存資料夾本身也不能當 root
        assert not safepath.safe_rmtree(temp / "vs-safety-no-such-folder-7f3a", temp)
        assert not safepath.safe_rmtree(Path("E:/vs-safety-no-such-folder-7f3a"), Path("E:/"))

        # safe_unlink：專案外拒絕、資料夾拒絕、裡面的檔案可以刪、本來就不在算成功
        assert not safepath.safe_unlink(victim, root)
        assert not safepath.safe_unlink(root / "uploads", root)
        f = touch(root / "ok.txt")
        assert safepath.safe_unlink(f, root) and not f.exists()
        assert safepath.safe_unlink(root / "missing.txt", root)
        assert victim.exists()
    run(body)


def test_refusal_is_logged():
    def body(env: Env):
        records = []

        class Grab(logging.Handler):
            def emit(self, record):
                records.append(record.getMessage())

        logger = logging.getLogger("safepath")
        handler = Grab()
        logger.addHandler(handler)
        env.patch(logger, "propagate", False)
        logger.setLevel(logging.WARNING)
        try:
            assert not safepath.safe_rmtree(env.outside, env.media)
        finally:
            logger.setLevel(logging.CRITICAL)
            logger.removeHandler(handler)
        assert any("拒絕刪除" in r for r in records), records
    run(body)


def test_cue_path_rejects_traversal_ids():
    for bad in ["..\\x", "../x", "C:\\Users\\x", "a.b", "", "NUL", "con", "x y", "a:b", "x" * 65, None]:
        try:
            cue_mod.cue_path(bad)
        except ValueError:
            continue
        raise AssertionError(f"cue_path 應該拒絕 {bad!r}")
    assert cue_mod.cue_path("0123456789ab").name == "0123456789ab.json"
    assert health.backup_path("abc").name == "abc.bak.json"


# ---------- 2. 刪除影片 ----------

def test_delete_media_never_touches_files_outside_project():
    def body(env: Env):
        victim_dir = env.outside / "victim"
        clip = touch(victim_dir / "clip.mp4")
        diary = touch(victim_dir / "diary.txt")
        local_video = touch(env.outside / "camera_clip.MP4")

        # a. 舊資料裡被竄改的上傳路徑：用 .. 繞到專案外
        crafted = env.add_media("upload", str(env.media) + r"\..\..\..\outside\victim\clip.mp4")
        server.delete_media(crafted)
        assert clip.exists() and diary.exists()
        assert db.get_media(crafted) is None                     # 資料列照樣刪掉，只是不動檔案

        # b. 本機路徑加入的原始影片：永遠不刪
        local = env.add_media("local", local_video)
        server.delete_media(local)
        assert local_video.exists()

        # c. 用大小寫不同的路徑指到別部影片的下載檔
        other = env.add_media("url", url="https://www.youtube.com/watch?v=x")
        other_file = touch(env.media / other / "download.mp4")
        db.update_media(other, path=str(other_file))
        sneaky = env.add_media("upload", str(other_file).upper())
        server.delete_media(sneaky)
        assert other_file.exists()

        # d. 直接放在 uploads 底下的檔案：不能把整個 uploads 刪掉
        keep = touch(env.media / "uploads" / "0123456789ab" / "other.mp4")
        env.add_media("upload", keep)
        loose = env.add_media("upload", touch(env.media / "uploads" / "loose.mp4"))
        server.delete_media(loose)
        assert keep.exists()

        # e. uploads/<id> 其實是指到專案外的 junction
        touch(env.outside / "victimB" / "clip.mp4")
        notes = touch(env.outside / "victimB" / "notes.txt")
        junction(env.media / "uploads" / "aaaaaaaaaaaa", env.outside / "victimB")
        jmid = env.add_media("upload", env.media / "uploads" / "aaaaaaaaaaaa" / "clip.mp4")
        server.delete_media(jmid)
        assert notes.exists()

        # f. 正常的上傳影片：它自己的資料夾、字幕、備份、縮圖、相容播放檔、工作資料夾都要清掉
        up_file = touch(env.media / "uploads" / "bbbbbbbbbbbb" / "clip.mp4")
        up = env.add_media("upload", up_file)
        tid = env.add_track(up)
        thumb = touch(env.thumbs / f"{up}.jpg")
        proxy = touch(env.proxy / f"{up}.mp4")
        proxy_tmp = touch(env.proxy / f"{up}.tmp.mp4")
        db.update_media(up, proxy_path=str(proxy))
        jid = db.add_job(up, "transcribe", {})
        db.update_job(jid, status="failed")
        workdir = touch(env.work / jid / "audio.wav").parent
        server.delete_media(up)
        assert not up_file.parent.exists()
        assert not cue_mod.cue_path(tid).exists() and not health.backup_path(tid).exists()
        assert not thumb.exists() and not proxy.exists() and not proxy_tmp.exists()
        assert not workdir.exists()
        assert db.get_track(tid) is None and db.get_media(up) is None

        # g. 正常的網址影片：data/media/<影片 id> 整個清掉
        url_mid = env.add_media("url", url="https://www.youtube.com/watch?v=y")
        dl = touch(env.media / url_mid / "video [y].mp4")
        db.update_media(url_mid, path=str(dl))
        server.delete_media(url_mid)
        assert not dl.parent.exists()

        # 全部做完，專案外的東西一個都沒少
        assert clip.exists() and diary.exists() and local_video.exists() and notes.exists() and other_file.exists()
    run(body)


def test_add_media_rejects_local_paths_and_crafted_uploads():
    def body(env: Env):
        outside_video = touch(env.outside / "movie.mp4")
        before = len(db.list_media())
        expect_http(400, server.add_media, server.AddMedia(source="local", path=str(outside_video), language="ja"))
        for raw in (str(env.media) + r"\..\..\..\outside\movie.mp4",      # .. 繞出去
                    str(outside_video),                                    # 直接給專案外的路徑
                    str(touch(env.media / "uploads" / "loose.mp4")),       # 不在 uploads/<id> 裡
                    str(touch(env.media / "uploads" / "not-an-id" / "a.mp4"))):
            expect_http(400, server.add_media, server.AddMedia(source="upload", path=raw, language="ja"))
        assert len(db.list_media()) == before and outside_video.exists()

        # 已經在播放列表裡的上傳檔：回 409，檔案不能被當成失敗的上傳刪掉
        dup = touch(env.media / "uploads" / "cccccccccccc" / "a.mp4")
        env.add_media("upload", dup)
        expect_http(409, server.add_media, server.AddMedia(source="upload", path=str(dup), language="ja"))
        assert dup.exists()

        # 剛上傳但加入失敗（例如沒選語言）：那份上傳要刪掉，不留在 uploads 裡
        fresh = touch(env.media / "uploads" / "dddddddddddd" / "b.mp4")
        expect_http(400, server.add_media, server.AddMedia(source="upload", path=str(fresh), language="xx"))
        assert not fresh.parent.exists()
    run(body)


def test_upload_endpoint_stays_inside_uploads():
    def body(env: Env):
        client = TestClient(server.app)
        uploads = env.media / "uploads"
        for name in (r"..\..\evil.mp4", "../../evil.mp4", r"C:\Windows\evil.mp4", "C:evil.mp4",
                     r"\\server\share\v.mp4", "a.mp4::$DATA", "clip.mp4. . ", "..", "CON.mp4"):
            r = client.put("/api/upload", params={"name": name}, content=b"data")
            if r.status_code != 200:
                assert r.status_code == 400, (name, r.status_code)
                continue
            p = Path(r.json()["path"])
            assert p.exists() and safepath.inside(p, uploads) and safepath.is_id(p.parent.name), (name, p)
            assert p.parent.parent == uploads, (name, p)
        assert client.put("/api/upload", params={"name": "notes.txt"}, content=b"x").status_code == 400
        r = client.put("/api/upload", params={"name": "長" * 300 + ".mp4"}, content=b"x")   # 檔名太長
        assert r.status_code == 200 and len(Path(r.json()["path"]).name) <= 104, r.status_code
        assert not (env.media / "evil.mp4").exists() and not (env.data / "evil.mp4").exists()
        assert client.get("/api/fs").status_code in (404, 405)          # 瀏覽整台電腦的 API 已經拿掉
    run(body)


def test_cross_site_writes_are_rejected():
    def body(env: Env):
        client = TestClient(server.app)              # TestClient 的主機名稱是 testserver
        jid = db.add_job(None, "titles", {})
        db.update_job(jid, status="done")
        for origin in ("http://evil.example", "null", "http://testserver.evil.example"):
            r = client.post("/api/jobs/clear", headers={"Origin": origin})
            assert r.status_code == 403, (origin, r.status_code)
            assert client.delete(f"/api/jobs/{jid}", headers={"Origin": origin}).status_code == 403
        assert db.get_job(jid) is not None
        assert client.post("/api/jobs/clear", headers={"Origin": "http://testserver"}).status_code == 200
        assert db.get_job(jid) is None
        assert client.get("/api/meta", headers={"Origin": "http://evil.example"}).status_code == 200   # 讀取不受影響
        assert client.post("/api/jobs/clear").status_code == 200                                     # 程式呼叫沒有 Origin

        # DNS rebinding：Host 和 Origin 都是攻擊者的網域，兩邊一樣也要擋，讀取也擋
        evil = {"Host": "evil.example:8765", "Origin": "http://evil.example:8765"}
        jid = db.add_job(None, "titles", {})
        db.update_job(jid, status="done")
        assert client.post("/api/jobs/clear", headers=evil).status_code == 403
        assert client.get("/api/state", headers={"Host": "evil.example:8765"}).status_code == 403
        assert db.get_job(jid) is not None
        for host in ("127.0.0.1:8765", "localhost:8765", "192.168.1.20:8765", "[::1]:8765", "LOCALHOST."):
            assert server.host_allowed(host), host
        for host in ("evil.example", "127.0.0.1.nip.io:8765", "localhost.evil.example"):
            assert not server.host_allowed(host), host
    run(body)


# ---------- 3. 字幕、模型、唯讀 API 的路徑穿越 ----------

def test_track_and_model_ids_cannot_escape():
    def body(env: Env):
        secret = touch(env.outside / "secret.json", '[{"start": 0, "end": 1, "text": "secret"}]')
        client = TestClient(server.app)
        for bad in ("..%5C..%5C..%5Coutside%5Csecret", "..%2F..%2Foutside", "C:%5CUsers"):
            assert client.delete(f"/api/tracks/{bad}").status_code == 404
        expect_http(404, server.delete_track, r"..\..\..\outside\secret")
        expect_http(404, server.thumb, r"..\..\outside\x")
        mid = env.add_media("upload", touch(env.media / "uploads" / "eeeeeeeeeeee" / "a.mp4"))
        tid = env.add_track(mid)
        expect_http(404, server.bilingual, mid, str(env.outside / "secret"), tid)
        expect_http(404, server.bilingual, mid, tid, r"..\..\..\outside\secret")
        other = env.add_media("url", url="https://youtu.be/z")
        expect_http(404, server.bilingual, other, tid, tid)              # 別部影片的字幕也不行

        # 查字 API：track 參數不是資料庫裡的字幕軌就當成沒帶
        seen = {}
        env.patch(vocab, "lexeme_detail", lambda key, track, cue, k: seen.update(track=track) or {})
        vocab_api.lexeme(key="jm:1", track=r"..\..\..\outside\secret", cue=0, k=0)
        assert seen["track"] == ""
        vocab_api.lexeme(key="jm:1", track=tid, cue=0, k=0)
        assert seen["track"] == tid

        # 刪字幕：只刪自己的檔案
        server.delete_track(tid)
        assert not cue_mod.cue_path(tid).exists() and secret.exists()

        # 刪模型（規則在 app/models.py）：每個模型都能刪，但只刪它自己的檔案
        expect_http(404, server.delete_model, r"..\..\outside")
        outside_model = touch(env.outside / "big_model.bin").parent
        group = touch(env.models / "asr" / "keep" / "model.safetensors")
        own = touch(env.models / "gguf" / "Fake-7B" / "fake.gguf")
        other_gguf = touch(env.models / "gguf" / "Other" / "other.gguf")
        sep = env.models / "sep"                                   # 共用資料夾
        shared = [touch(sep / "download_checks.json"), touch(sep / "other.ckpt")]
        mine = [touch(sep / "fake.ckpt"), touch(sep / "fake.yaml"), touch(sep / "fake.ckpt.part")]
        env.patch(models, "MODELS_DIR", env.models)
        env.patch(models, "MODEL_CATALOG", {
            "evil": {"role": "x", "label": "evil", "kind": "hf", "dir": outside_model},
            "group": {"role": "x", "label": "group", "kind": "hf", "dir": env.models / "asr"},
            "shared": {"role": "x", "label": "shared", "kind": "hf", "dir": sep},
            "escape": {"role": "x", "label": "escape", "kind": "ckpt", "dir": sep, "file": "x.ckpt",
                       "files": [r"..\gguf\Other\other.gguf"]},
            "fake": {"role": "x", "label": "fake", "kind": "gguf", "file": "fake.gguf", "dir": own.parent},
            "ckpt": {"role": "x", "label": "ckpt", "kind": "ckpt", "dir": sep, "file": "fake.ckpt",
                     "files": ["fake.ckpt", "fake.yaml"]},
        })
        for bad in ("evil", "group", "shared", "escape"):
            expect_http(400, server.delete_model, bad)
        assert (env.outside / "big_model.bin").exists() and group.exists() and own.exists() and other_gguf.exists()
        assert all(p.exists() for p in shared + mine)
        assert server.delete_model("fake")["ok"] and not own.parent.exists()
        assert server.delete_model("ckpt")["ok"] and not any(p.exists() for p in mine)
        assert all(p.exists() for p in shared) and other_gguf.exists() and group.exists()
    run(body)


# ---------- 4. 下載網址與 cookies ----------

def test_download_url_rules():
    bad = ["", "-o x", "--exec calc https://www.youtube.com/watch?v=x", " -https://youtu.be/x",
           "file:///C:/Windows/win.ini", "ftp://example.com/v.mp4", "javascript:alert(1)",
           "http://127.0.0.1:8765/", "http://localhost/x", "http://192.168.1.5/v.mp4", "http://10.0.0.2/",
           "http://[::1]/", "http://2130706433/", "http://0x7f.0.0.1/", "http://169.254.169.254/latest",
           "https://youtu.be/a b", "https://youtu.be/a\nb", None,
           # 看起來不像本機、連線時卻會變成 127.0.0.1 的寫法
           "http://①②⑦.0.0.1:8765/", "http://127。0。0。1:8765/", "http://１２７.0.0.1:8765/",
           "http://127.0.0.1．:8765/", "http://127.0.0.1%2e:8765/", "http://LOCALHOST。:8765/",
           "http://%31%32%37.0.0.1/", "http://ｌｏｃａｌｈｏｓｔ/"]
    for url in bad:
        try:
            safepath.check_download_url(url)
        except ValueError:
            continue
        raise AssertionError(f"應該拒絕 {url!r}")
    for url in ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", " https://youtu.be/dQw4w9WgXcQ ",
                "https://www.bilibili.com/video/BV1xx411c7mD", "http://example.com/video.mp4"):
        assert safepath.check_download_url(url) == url.strip()


def test_add_media_and_download_job_reject_bad_urls():
    def body(env: Env):
        before = len(db.list_media())
        for url in ("--exec calc", "-https://youtu.be/x", "http://127.0.0.1:8765/api/state"):
            expect_http(400, server.add_media, server.AddMedia(source="url", url=url, language="ja"))
        assert len(db.list_media()) == before
        mid = db.new_id()
        ctx = jobs.JobContext({"id": db.new_id(), "media_id": mid, "params": {"url": "--exec calc"}})
        try:
            jobs.handle_download(ctx)
        except RuntimeError as e:
            assert "下載失敗" in str(e)
        else:
            raise AssertionError("handle_download 應該拒絕 - 開頭的網址")
        assert not (env.media / mid).exists()
    run(body)


def test_cookies_file_is_copied_not_rewritten():
    def body(env: Env):
        import yt_dlp
        original = ("# Netscape HTTP Cookie File\n# 我自己的註解\n"
                    ".youtube.com\tTRUE\t/\tTRUE\t2147483647\tPREF\tf6=40000000\n")
        cookies = touch(env.outside / "cookies.txt", original)
        before = cookies.read_bytes()

        for value in (str(env.outside / "missing.txt"), "relative\\cookies.txt", str(env.outside)):
            try:
                settings.save({"cookies_file": value})
            except ValueError:
                continue
            raise AssertionError(f"應該拒絕 cookies_file={value!r}")
        expect_http(400, server.write_settings, {"cookies_file": str(env.outside / "missing.txt")})
        try:
            settings.save({"cookies_browser": "chrome:C:\\evil"})
            raise AssertionError("應該拒絕不認得的瀏覽器")
        except ValueError:
            pass
        assert not (env.outside / "missing.txt").exists()

        settings.save({"cookies_file": f'"{cookies}"'})           # 從檔案總管複製的路徑常帶引號
        assert settings.get("cookies_file") == str(cookies)
        copy = env.work / "0123456789ab" / "cookies.txt"
        opts = settings.ytdlp_cookie_opts(copy)
        assert opts == {"cookiefile": str(copy)} and copy.exists()
        with yt_dlp.YoutubeDL({**opts, "quiet": True, "no_warnings": True, "cachedir": False}) as ydl:
            assert any(c.name == "PREF" for c in ydl.cookiejar)   # cookies 照樣讀得到
        # yt-dlp 結束時會回寫 cookiefile：寫到的是複本，使用者的檔案一個位元組都沒變
        assert cookies.read_bytes() == before
        assert copy.read_text(encoding="utf-8") != original

        cookies.unlink()
        try:
            settings.ytdlp_cookie_opts(env.work / "x" / "cookies.txt")
            raise AssertionError("cookies.txt 不見了應該回報錯誤")
        except RuntimeError as e:
            assert "cookies.txt" in str(e)
        assert not cookies.exists()
    run(body)


class FakeYDL:
    """假的 yt-dlp：寫一個下載檔，下載途中使用者把影片刪掉。"""
    seen = {}

    def __init__(self, opts):
        self.opts = opts
        FakeYDL.seen["opts"] = opts
        cookiefile = opts.get("cookiefile")
        FakeYDL.seen["cookie_copy_exists"] = bool(cookiefile) and Path(cookiefile).exists()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def extract_info(self, url, download):
        out = Path(self.opts["outtmpl"]).parent
        f = touch(out / "video [abc].mp4")
        on_extract = FakeYDL.seen.get("on_extract")
        if on_extract:
            on_extract()
        return {"requested_downloads": [{"filepath": str(f)}], "title": "t"}


def test_download_uses_cookie_copy_and_cleans_up_when_media_deleted():
    def body(env: Env):
        import yt_dlp
        cookies = touch(env.outside / "cookies.txt", "# Netscape HTTP Cookie File\n")
        before = cookies.read_bytes()
        settings.save({"cookies_file": str(cookies)})
        mid = env.add_media("url", url="https://youtu.be/abc")
        jid = db.add_job(mid, "download", {"url": "https://youtu.be/abc"})
        FakeYDL.seen.clear()
        FakeYDL.seen["on_extract"] = lambda: server.delete_media(mid)    # 合併影音時使用者刪掉影片
        env.patch(yt_dlp, "YoutubeDL", FakeYDL)
        ctx = jobs.JobContext(db.get_job(jid))
        try:
            jobs.handle_download(ctx)
        except jobs.Cancelled:
            pass
        else:
            raise AssertionError("影片被刪掉後下載任務應該以取消結束")
        opts = FakeYDL.seen["opts"]
        assert Path(opts["cookiefile"]) != cookies and safepath.inside(opts["cookiefile"], env.work)
        assert FakeYDL.seen["cookie_copy_exists"]
        assert safepath.inside(opts["cachedir"], config.DATA_DIR)          # yt-dlp 快取在專案裡
        assert not (env.media / mid).exists()                               # 下載好的檔案清掉了
        assert not ctx.workdir.exists()                                     # cookies 複本也清掉了
        assert cookies.read_bytes() == before
    run(body)


# ---------- 5. 工作資料夾與孤兒檔 ----------

def test_workdirs_are_cleaned_with_jobs():
    def body(env: Env):
        mid = env.add_media("url", url="https://youtu.be/w")

        def job(status):
            jid = db.add_job(mid, "transcribe", {})
            db.update_job(jid, status=status)
            touch(env.work / jid / "audio.wav")
            return jid

        failed, running, done, canceled = job("failed"), job("running"), job("done"), job("canceled")
        server.job_delete(failed)
        server.job_delete(running)                    # 執行中的任務刪不掉，資料夾也不動
        assert not (env.work / failed).exists() and (env.work / running).exists()
        expect_http(404, server.job_delete, r"..\..\outside")
        server.job_clear()
        assert not (env.work / done).exists() and not (env.work / canceled).exists()
        assert (env.work / running).exists()

        # 啟動時的清理：沒有任務、已完成、影片被刪掉的清掉；排隊中和失敗的留著續跑
        orphan = touch(env.work / "ffffffffffff" / "audio.wav").parent
        keep_failed = job("failed")
        keep_queued = job("queued")
        gone_media = env.add_media("url", url="https://youtu.be/g")
        stale = db.add_job(gone_media, "transcribe", {})
        db.update_job(stale, status="failed")
        touch(env.work / stale / "audio.wav")
        db.delete_media(gone_media)
        log_file = touch(env.work / "titles-llama-server.log")
        other_dir = touch(env.work / "notes" / "keep.txt").parent
        jobs.sweep_workdirs()
        assert not orphan.exists() and not (env.work / stale).exists()
        assert (env.work / keep_failed).exists() and (env.work / keep_queued).exists()
        assert log_file.exists() and other_dir.exists()
    run(body)


def test_canceled_job_of_deleted_media_drops_workdir():
    def body(env: Env):
        mid = env.add_media("url", url="https://youtu.be/c")
        jid = db.add_job(mid, "transcribe", {})

        def handler(ctx):
            touch(ctx.workdir / "audio.wav")
            db.delete_media(mid)                      # 轉字幕途中影片被刪掉
            raise jobs.Cancelled()

        env.patch(jobs, "HANDLERS", {**jobs.HANDLERS, "transcribe": handler})
        jobs.Worker("test", ("transcribe",)).execute(db.get_job(jid))
        assert not (env.work / jid).exists()

        # 影片還在的取消：資料夾留著，重試時可以續跑
        mid2 = env.add_media("url", url="https://youtu.be/d")
        jid2 = db.add_job(mid2, "transcribe", {})

        def handler2(ctx):
            touch(ctx.workdir / "audio.wav")
            raise jobs.Cancelled()

        env.patch(jobs, "HANDLERS", {**jobs.HANDLERS, "transcribe": handler2})
        jobs.Worker("test", ("transcribe",)).execute(db.get_job(jid2))
        assert (env.work / jid2 / "audio.wav").exists()
    run(body)


def test_proxy_temp_and_result_are_cleaned():
    def body(env: Env):
        # ffmpeg 失敗（來源不存在）：轉到一半的暫存檔不留
        dst = env.proxy / "abcdefabcdef.mp4"
        tmp = touch(env.proxy / "abcdefabcdef.tmp.mp4")
        try:
            media.make_proxy(env.outside / "no-such-video.mp4", dst, 10)
        except (RuntimeError, OSError):
            pass
        assert not tmp.exists() and not dst.exists()

        # 轉檔途中影片被刪掉：轉好的檔案刪掉
        src = touch(env.outside / "src.mp4")
        mid = env.add_media("local", src)
        jid = db.add_job(mid, "proxy", {})

        def fake_make_proxy(s, d, duration, on_progress=None, check=None, **kw):
            touch(d)
            db.delete_media(mid)

        env.patch(media, "make_proxy", fake_make_proxy)
        env.patch(jobs, "release_llm", lambda reason="": None)
        try:
            jobs.handle_proxy(jobs.JobContext(db.get_job(jid)))
        except jobs.Cancelled:
            pass
        assert not (env.proxy / f"{mid}.mp4").exists() and src.exists()
    run(body)


def test_track_delete_cancels_pipeline_translation_and_replace_cleans_backup():
    def body(env: Env):
        mid = env.add_media("url", url="https://youtu.be/t")
        tid = env.add_track(mid)
        t_job = db.add_job(mid, "transcribe", {"language": "ja"})
        db.update_job(t_job, status="done", result={"track_id": tid})
        tr_job = db.add_job(mid, "translate", {"translator": "hymt", "language": "ja"}, depends_on=t_job)
        assert jobs.translation_source(db.get_job(tr_job)) == tid
        server.delete_track(tid)
        assert db.get_job(tr_job)["status"] == "canceled"
        assert not cue_mod.cue_path(tid).exists() and not health.backup_path(tid).exists()

        # 重翻取代舊翻譯時走同一個函式：字幕檔、備份、索引一起清
        old = env.add_track(mid)
        db._conn.execute("INSERT INTO track_vocab VALUES (?, ?, ?, ?, ?, ?)", (old, "jm:1", 1, 0, 0.0, "[0]"))
        jobs.discard_track(old)
        assert not cue_mod.cue_path(old).exists() and not health.backup_path(old).exists()
        assert db.get_track(old) is None
        assert db._one("SELECT * FROM track_vocab WHERE track_id = ?", (old,)) is None
    run(body)


# ---------- 6. 子程序 ----------

def test_llama_server_is_stopped_when_loading_is_cancelled():
    def body(env: Env):
        real_popen = subprocess.Popen
        started = {}

        def fake_popen(args, *a, **kw):
            if str(args[0]) == str(translate.LLAMA_SERVER):
                # 假的 llama-server：一直活著、永遠不回 /health
                args = [sys.executable, "-c", "import time; time.sleep(120)"]
                proc = real_popen(args, *a, **kw)
                started["proc"] = proc
                return proc
            return real_popen(args, *a, **kw)

        cancel = threading.Event()

        def check():
            if cancel.is_set():
                raise jobs.Cancelled()

        srv = translate.LlamaServer("hymt", env.work / "llama.log")
        threading.Timer(1.0, cancel.set).start()
        env.patch(translate.subprocess, "Popen", fake_popen)
        try:
            srv.start(check)
        except jobs.Cancelled:
            pass
        else:
            raise AssertionError("start() 應該因為取消而丟例外")
        finally:
            setattr(translate.subprocess, "Popen", real_popen)
        proc = started["proc"]
        try:
            assert proc.poll() is not None, "取消載入後 llama-server 應該被關掉"
        finally:
            if proc.poll() is None:
                proc.kill()
    run(body)


def _start_orphan(args: list) -> int:
    """用一個馬上結束的 python 叫起程式，讓它變成「上層程序已經不在」的殘留程序。回傳它的 PID。"""
    code = ("import subprocess, sys, json; "
            "p = subprocess.Popen(json.loads(sys.argv[1]), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, "
            "creationflags=0x08000000); print(p.pid)")
    import json
    out = subprocess.run([sys.executable, "-s", "-c", code, json.dumps([str(a) for a in args])],
                         capture_output=True, text=True, timeout=60, creationflags=subprocess.CREATE_NO_WINDOW)
    return int(out.stdout.strip().splitlines()[-1])


def test_orphan_cleanup_matches_only_this_project():
    """殘留程序改用 psutil 找（jobs.find_orphans），不靠 PowerShell；比對條件跟以前一樣嚴格。"""
    import psutil

    def body(env: Env):
        if not PING.exists():
            print("  （找不到 PING.EXE，跳過實際程序比對）")
            return
        mine = env.tmp / "bilingual-subtitles" / "bin" / "llama.cpp" / "llama-server.exe"
        sibling = env.tmp / "bilingual-subtitles - 複製" / "bin" / "llama.cpp" / "llama-server.exe"
        for exe in (mine, sibling):
            exe.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(PING, exe)
        work = env.tmp / "bilingual-subtitles" / "data" / "work"
        other_work = env.tmp / "bilingual-subtitles - 複製" / "data" / "work"
        nothing = env.tmp / "no-such-python.exe"
        # 假的 python 子程序：真的 python，但 -c 只是等 60 秒；-c 後面的參數都只是 sys.argv，
        # 命令列看起來跟 run_child 叫起來的 -m app.asr_child 一樣
        wait = "import time; time.sleep(60)"
        pids = []
        try:
            orphan_llama = _start_orphan([mine, "-n", "60", "127.0.0.1"])
            sibling_llama = _start_orphan([sibling, "-n", "60", "127.0.0.1"])
            orphan_py = _start_orphan([sys.executable, "-s", "-c", wait, "-m", "app.asr_child", "asr", work / "x"])
            other_py = _start_orphan([sys.executable, "-s", "-c", wait, "-m", "app.asr_child", "asr", other_work / "x"])
            server_py = _start_orphan([sys.executable, "-s", "-c", wait, "-m", "app.server", "x", work / "x"])
            # 名字以 work 開頭的其他資料夾（data\work2、data\work-old）不是本專案的 data\work
            prefix_py = _start_orphan([sys.executable, "-s", "-c", wait, "-m", "app.asr_child", "asr",
                                       work.parent / "work2" / "x"])
            sep_py = _start_orphan([sys.executable, "-s", "-c", wait, "-m", "app.separate", env.tmp / "song.mp4",
                                    work / "y" / "vocals.wav"])
            pids = [orphan_llama, sibling_llama, orphan_py, other_py, server_py, prefix_py, sep_py]
            time.sleep(1.0)
            found = {p.pid for p in jobs.find_orphans(llama=mine, python=nothing, work=work)}
            assert orphan_llama in found, (pids, found)
            assert sibling_llama not in found, "複製出來的兄弟資料夾裡的 llama-server 不應該被清"
            found = {p.pid for p in jobs.find_orphans(llama=env.tmp / "no-such-llama.exe", python=sys.executable,
                                                      work=work)}
            assert orphan_py in found, (pids, found)
            assert other_py not in found, "別的資料夾的 data\\work 不應該被清"
            assert server_py not in found, "只清 app.asr_child 和 app.separate"
            assert prefix_py not in found, "data\\work2 不是 data\\work 底下"
            assert sep_py in found, "人聲分離的輸出檔在 data\\work 底下，要認得"
            assert jobs._under(work / "a" / "b.wav", os.path.normcase(str(work)))
            assert not jobs._under(str(work) + "-old", os.path.normcase(str(work)))

            # 還有上層程序在跑的 llama-server（這個測試自己叫起來的）不算殘留
            live = subprocess.Popen([str(mine), "-n", "60", "127.0.0.1"], stdout=subprocess.DEVNULL,
                                    creationflags=subprocess.CREATE_NO_WINDOW)
            pids.append(live.pid)
            time.sleep(0.5)
            found = {p.pid for p in jobs.find_orphans(llama=mine, python=nothing, work=work)}
            assert live.pid not in found and orphan_llama in found, (live.pid, found)
        finally:
            for pid in pids:     # 只結束這個測試自己叫起來的假程序
                try:
                    psutil.Process(pid).kill()
                except psutil.Error:
                    pass
            time.sleep(0.5)

        assert jobs._module_arg(["python.exe", "-s", "-m", "app.separate"]) is None     # -m 後面沒有參數
        assert jobs._same_path(str(mine).upper(), mine) and not jobs._same_path(mine, sibling)
    run(body)


# ---------- 7. 快取位置 ----------

def test_package_caches_point_into_project():
    for key in config._CACHE_ENV:
        if key in _PRESET_ENV or (key.startswith("HF_") and "HF_HOME" in _PRESET_ENV):
            continue      # 使用者自己設過的就照他的
        assert safepath.inside(os.environ[key], config.CACHE_DIR), (key, os.environ.get(key))


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
