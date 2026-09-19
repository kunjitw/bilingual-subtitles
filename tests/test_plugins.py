"""本機外掛（app/plugins.py）的測試：用假外掛，外掛資料夾和資料庫都在暫存資料夾。

python -s tests/test_plugins.py
"""
import logging
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_safety import BASE, Env, expect_http, junction, touch  # noqa: E402  （也會設好 VS_NO_WORKERS）

from app import db, jobs, media, plugins, server  # noqa: E402

FAKE = '''
NAME = "假外掛"
OPTIONS = {"language": "ja", "translate": False, "engine": "x", "sensitive": "yes"}
CALLS = []

def match(url):
    return url.startswith("https://fake.example/")

def download(job):
    CALLS.append(job.url)
    job.progress(0.5, "下載中")
    job.check()
    f = job.out_dir / "episode.mp4"
    f.write_bytes(b"fake video")
    return {"path": str(f), "title": "  第 2 集  "}
'''


def write_plugin(folder: Path, name: str, code: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    p = folder / name
    p.write_text(textwrap.dedent(code), encoding="utf-8")
    return p


def run(fn):
    env = Env()
    env.plugins = env.tmp / "local_plugins"
    env.patch(plugins, "PLUGIN_DIR", env.plugins)
    plugins._cache.update(key=None, plugins=[])
    logger = logging.getLogger("plugins")
    level = logger.level
    logger.setLevel(logging.CRITICAL)        # 載入失敗、選項不對的 log 是預期的
    try:
        fn(env)
    finally:
        logger.setLevel(level)
        plugins._cache.update(key=None, plugins=[])
        env.close()


def fake_probe(env):
    env.patch(media, "probe", lambda path: {"duration": 10.0, "vcodec": "h264", "acodec": "aac", "width": 640,
                                            "height": 360, "playable": 1})
    env.patch(media, "make_thumbnail", lambda *a, **k: False)


def capture_options(env):
    """_resolve_options 需要裝好的模型；這裡只記下送進去的選項。"""
    seen = []

    def resolve(opt):
        seen.append(opt.model_dump())
        return "fake-engine", ("fake-translator" if opt.translate and opt.language != "zh" else None)

    env.patch(server, "_resolve_options", resolve)
    env.patch(server, "_require_media_tools", lambda *names: None)
    return seen


# ---------- 載入 ----------

def test_no_plugin_folder_changes_nothing():
    def body(env):
        assert not env.plugins.exists()
        assert plugins.load() == [] and plugins.find("https://fake.example/1") is None
        assert server.url_plugin("https://fake.example/1") == {"plugin": None, "options": {}}
        seen = capture_options(env)
        res = server.add_media(server.AddMedia(source="url", url="https://fake.example/1", language="en",
                                               translate=True))
        assert seen[-1]["language"] == "en" and seen[-1]["translate"] is True
        m = db.get_media(res["id"])
        assert m["source"] == "url" and m["language"] == "en"
        types = [j["type"] for j in db.jobs_for_media(res["id"])]
        assert "download" in types and "translate" in types
    run(body)


def test_load_rules():
    def body(env):
        write_plugin(env.plugins, "fake.py", FAKE)
        write_plugin(env.plugins, "_helper.py", "def match(u): return True\ndef download(j): pass\n")
        write_plugin(env.plugins, "nodownload.py", "def match(u): return True\n")
        write_plugin(env.plugins, "broken.py", "raise RuntimeError('boom')\n")
        write_plugin(env.plugins, "notpy.txt", "def match(u): return True\ndef download(j): pass\n")
        loaded = plugins.load()
        assert [p.name for p in loaded] == ["假外掛"], loaded
        # 只留認得、型別正確的選項
        assert loaded[0].options == {"language": "ja", "translate": False}
        assert plugins.find("https://fake.example/ep/1").name == "假外掛"
        assert plugins.find("https://other.example/") is None

        # 檔案改了會重新載入；沒有 NAME 用檔名；不支援的語言不收
        time.sleep(0.05)
        write_plugin(env.plugins, "fake.py", "OPTIONS = {'language': 'xx'}\ndef match(u): return False\n"
                                             "def download(j): pass\n")
        loaded = plugins.load()
        assert [p.name for p in loaded] == ["fake"] and loaded[0].options == {}
        assert plugins.find("https://fake.example/ep/1") is None
    run(body)


def test_match_errors_are_ignored():
    def body(env):
        write_plugin(env.plugins, "bad.py", "def match(u): raise ValueError('x')\ndef download(j): pass\n")
        write_plugin(env.plugins, "good.py", FAKE)
        assert plugins.find("https://fake.example/1").name == "假外掛"
    run(body)


def test_links_are_not_loaded():
    def body(env):
        real = env.outside / "plugins"
        write_plugin(real, "fake.py", FAKE)
        junction(env.plugins, real)                      # 外掛資料夾本身是 junction：整個不載入
        assert plugins.load() == []
        os.rmdir(env.plugins)                            # 拿掉 junction（不會動到 real）
        assert (real / "fake.py").exists()
        env.plugins.mkdir()
        try:
            os.symlink(real / "fake.py", env.plugins / "linked.py")
        except OSError:
            print("  （沒有建立 symlink 的權限，跳過 symlink 檢查）")
        else:
            assert plugins.load() == []                  # 捷徑指到資料夾外的外掛不載入
    run(body)


def test_plugin_folder_is_fixed():
    # 位置寫死在專案根目錄，不看環境變數或設定
    assert plugins.PLUGIN_DIR == Path(__file__).resolve().parents[1] / "local_plugins"


# ---------- 網址規則不能被繞過 ----------

def test_url_rules_still_apply():
    def body(env):
        write_plugin(env.plugins, "all.py", "CALLS = []\ndef match(u): return True\n"
                                            "def download(j):\n    CALLS.append(j.url)\n")
        for url in ("--exec calc", "-https://fake.example/1", "http://127.0.0.1:8765/api/state",
                    "file:///C:/Windows/win.ini", "http://localhost/x"):
            assert plugins.find(url) is None, url
            assert server.url_plugin(url)["plugin"] is None
            expect_http(400, server.add_media, server.AddMedia(source="url", url=url, language="ja"))
            mid = db.new_id()
            ctx = jobs.JobContext({"id": db.new_id(), "media_id": mid, "params": {"url": url}})
            try:
                jobs.handle_download(ctx)
            except RuntimeError as e:
                assert "下載失敗" in str(e)
            else:
                raise AssertionError(f"handle_download 應該拒絕 {url!r}")
            assert not (env.media / mid).exists()
        assert plugins.load()[0].module.CALLS == []
    run(body)


# ---------- 新增影片、下載 ----------

def test_plugin_options_override_request():
    def body(env):
        write_plugin(env.plugins, "fake.py", FAKE)
        seen = capture_options(env)
        assert server.url_plugin("https://fake.example/1") == {
            "plugin": "假外掛", "options": {"language": "ja", "translate": False}}
        res = server.add_media(server.AddMedia(source="url", url="https://fake.example/1", language="en",
                                               translate=True, engine="whisper-en", translator="t-en"))
        opt = seen[-1]
        assert opt["language"] == "ja" and opt["translate"] is False
        assert opt["engine"] is None and opt["translator"] is None   # 換了語言，網頁選的模型不用
        m = db.get_media(res["id"])
        assert m["source"] == "url" and m["language"] == "ja"
        js = db.jobs_for_media(res["id"])
        assert [j["type"] for j in js if j["type"] in ("download", "transcribe", "translate")] == ["download", "transcribe"]
        assert next(j for j in js if j["type"] == "transcribe")["params"]["language"] == "ja"

        # 同一個語言：網頁選的辨識模型留著
        server.add_media(server.AddMedia(source="url", url="https://fake.example/2", language="ja",
                                         engine="anime", translate=True))
        assert seen[-1]["engine"] == "anime" and seen[-1]["translate"] is False

        # 其他網址照舊
        server.add_media(server.AddMedia(source="url", url="https://www.youtube.com/watch?v=x", language="en",
                                         translate=True))
        assert seen[-1]["language"] == "en" and seen[-1]["translate"] is True
    run(body)


def test_plugin_download_is_managed_like_url_download():
    def body(env):
        write_plugin(env.plugins, "fake.py", FAKE)
        fake_probe(env)
        mid = env.add_media("url", url="https://fake.example/1")
        jid = db.add_job(mid, "download", {"url": "https://fake.example/1"})
        ctx = jobs.JobContext(db.get_job(jid))
        jobs.handle_download(ctx)
        m = db.get_media(mid)
        assert m["title"] == "第 2 集" and Path(m["path"]) == env.media / mid / "episode.mp4"
        assert m["width"] == 640 and not ctx.workdir.exists()
        assert plugins.load()[0].module.CALLS == ["https://fake.example/1"]
        # 刪影片時連同下載的檔案一起刪（跟網址下載一樣）
        server.delete_media(mid)
        assert not (env.media / mid).exists()
    run(body)


def test_plugin_file_must_be_inside_media_folder():
    def body(env):
        victim = touch(env.outside / "mine.mp4")
        write_plugin(env.plugins, "evil.py", f'''
            def match(u): return True
            def download(job): return {{"path": {str(victim)!r}, "title": "x"}}
        ''')
        write_plugin(env.plugins, "nothing.py", "def match(u): return u.endswith('/none')\ndef download(j): return None\n")
        fake_probe(env)
        for url in ("https://fake.example/none", "https://fake.example/1"):
            mid = env.add_media("url", url=url)
            ctx = jobs.JobContext(db.get_job(db.add_job(mid, "download", {"url": url})))
            try:
                jobs.handle_download(ctx)
            except RuntimeError:
                pass
            else:
                raise AssertionError("外掛回傳的檔案不在影片資料夾裡應該失敗")
            assert db.get_media(mid)["path"] is None
            server.delete_media(mid)
        assert victim.exists()
    run(body)


def test_plugin_failure_message_and_cancel():
    def body(env):
        write_plugin(env.plugins, "fail.py", '''
            def match(u): return u.endswith("/fail")
            def download(job): raise RuntimeError("需要登入")
        ''')
        write_plugin(env.plugins, "gone.py", '''
            import sys
            def match(u): return u.endswith("/gone")
            def download(job):
                (job.out_dir / "part.mp4").write_bytes(b"x")
                sys.modules["app.server"].delete_media(job._ctx.media_id)   # 下載途中使用者刪掉影片
                job.check()
        ''')
        mid = env.add_media("url", url="https://fake.example/fail")
        ctx = jobs.JobContext(db.get_job(db.add_job(mid, "download", {"url": "https://fake.example/fail"})))
        try:
            jobs.handle_download(ctx)
        except RuntimeError as e:
            assert "需要登入" in str(e) and "fail" in str(e)
        else:
            raise AssertionError("外掛失敗應該讓任務失敗")

        mid = env.add_media("url", url="https://fake.example/gone")
        jid = db.add_job(mid, "download", {"url": "https://fake.example/gone"})
        ctx = jobs.JobContext(db.get_job(jid))
        with jobs._running_lock:
            jobs._running[ctx.id] = ctx         # 讓 delete_media 的取消找得到這個任務
        try:
            db.update_job(jid, status="running")
            jobs.handle_download(ctx)
        except jobs.Cancelled:
            pass
        else:
            raise AssertionError("影片被刪掉後應該以取消結束")
        finally:
            with jobs._running_lock:
                jobs._running.pop(ctx.id, None)
        assert not (env.media / mid).exists() and not ctx.workdir.exists()
    run(body)


def test_job_run_streams_lines_and_kills_on_cancel():
    def body(env):
        mid = env.add_media("url", url="https://fake.example/1")
        ctx = jobs.JobContext({"id": db.new_id(), "media_id": mid, "params": {}})
        job = plugins.PluginJob(ctx, "https://fake.example/1", env.media / mid, env.work / ctx.id)
        lines = []
        code = job.run([sys.executable, "-c", "print('一'); print('二'); raise SystemExit(3)"],
                       on_line=lines.append, env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        assert code == 3 and lines == ["一", "二"], (code, lines)

        # 取消：連同孫程序一起結束
        pid_file = env.tmp / "grandchild.pid"
        script = ("import subprocess, sys, time;"
                  f"p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']);"
                  f"open({str(pid_file)!r}, 'w').write(str(p.pid));"
                  "print('started', flush=True); time.sleep(60)")

        def on_line(line):
            if line == "started":
                ctx.cancel_event.set()

        started = time.time()
        try:
            job.run([sys.executable, "-c", script], on_line=on_line)
        except jobs.Cancelled:
            pass
        else:
            raise AssertionError("取消時應該丟 Cancelled")
        assert time.time() - started < 20
        gpid = int(pid_file.read_text())
        time.sleep(0.5)
        out = subprocess.run(["tasklist", "/FI", f"PID eq {gpid}", "/NH"], capture_output=True, text=True).stdout
        assert str(gpid) not in out, out
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
