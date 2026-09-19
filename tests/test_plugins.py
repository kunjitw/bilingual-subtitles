"""本機外掛（app/plugins.py）的測試：用假外掛，外掛資料夾和資料庫都在暫存資料夾。

python -s tests/test_plugins.py
"""
import json
import logging
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_safety import BASE, Env, expect_http, junction, touch  # noqa: E402  （也會設好 VS_NO_WORKERS）

from fastapi.testclient import TestClient  # noqa: E402

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


class Capture(logging.Handler):
    """收集所有 log（檢查 cookie 這類內容不會被記下來）。"""

    def __init__(self):
        super().__init__(logging.DEBUG)
        self.lines = []

    def emit(self, record):
        text = record.getMessage()
        if record.exc_info:
            text += "\n" + logging.Formatter().formatException(record.exc_info)
        self.lines.append(text)


def run(fn):
    env = Env()
    env.plugins = env.tmp / "local_plugins"
    env.settings_dir = env.tmp / "plugin_settings"
    env.patch(plugins, "PLUGIN_DIR", env.plugins)
    env.patch(plugins, "SETTINGS_DIR", env.settings_dir)
    plugins._cache.update(key=None, plugins=[])
    logger = logging.getLogger("plugins")
    level = logger.level
    logger.setLevel(logging.CRITICAL)        # 載入失敗、選項不對的 log 是預期的
    env.log = Capture()
    root = logging.getLogger()
    root_level = root.level
    root.addHandler(env.log)
    try:
        fn(env)
    finally:
        root.removeHandler(env.log)
        root.setLevel(root_level)
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
        assert server.url_plugin("https://fake.example/1") == {"plugin": None, "options": {}, "expand": False}
        assert server.plugin_settings() == {"plugins": []}
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
            "plugin": "假外掛", "options": {"language": "ja", "translate": False}, "expand": False}
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


# ---------- 外掛設定：只寫不讀 ----------

SECRET = "SECRET-cookie-7f3a9"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36"

SETTINGS_PLUGIN = '''
import json
from pathlib import Path

NAME = "假登入外掛"
SETTINGS = {"title": "假登入", "intro": "貼上登入資料", "user_agent": True,
            "fields": [{"key": "token", "label": "Token", "type": "textarea", "placeholder": "a=1; b=2",
                        "help": "要用同一個瀏覽器", "steps": ["打開網站", "", "複製", 5]},
                       {"key": "bad key!", "label": "壞掉的欄位"},
                       {"key": "note", "type": "weird"}]}

def match(url):
    return url.startswith("https://login.example/")

def download(job):
    raise RuntimeError("not used")

def save_settings(values, ctx):
    if values.get("token", "").startswith("boom"):
        raise RuntimeError("cannot save " + values["token"])
    if values.get("token", "").startswith("bad"):
        raise ValueError("這串不像登入資料")
    folder = Path(ctx.dir)
    for k, v in values.items():
        (folder / f"{k}.txt").write_text(v, encoding="utf-8")
    (folder / "ua.txt").write_text(ctx.user_agent or "", encoding="utf-8")

def settings_status(ctx):
    folder = Path(ctx.dir)
    out = {}
    for key in ("token", "note"):
        p = folder / f"{key}.txt"
        # 故意把內容也放進去：程式不能轉給網頁
        out[key] = {"set": p.exists(), "saved_at": p.stat().st_mtime if p.exists() else None,
                    "value": p.read_text(encoding="utf-8") if p.exists() else None}
    ua = folder / "ua.txt"
    return {"fields": out, "user_agent": ua.read_text(encoding="utf-8") if ua.exists() else None,
            "raw": out, "problem": None}

def clear_settings(ctx):
    for p in Path(ctx.dir).glob("*.txt"):
        p.unlink()
'''


def test_settings_fields_and_write_only():
    def body(env):
        write_plugin(env.plugins, "login.py", SETTINGS_PLUGIN)
        for name in ("plugins", "server"):
            logging.getLogger(name).setLevel(logging.DEBUG)
        logging.getLogger().setLevel(logging.DEBUG)
        client = TestClient(server.app)

        view = client.get("/api/plugins/settings").json()["plugins"]
        assert len(view) == 1, view
        p = view[0]
        assert p["id"] == "login" and p["title"] == "假登入" and p["user_agent"] is True
        # 名稱不合規則的欄位丟掉，不認得的型別當 text，空的、不是字串的步驟丟掉
        assert [f["key"] for f in p["fields"]] == ["token", "note"]
        assert p["fields"][0]["steps"] == ["打開網站", "複製"] and p["fields"][1]["type"] == "text"
        assert not any(f["set"] for f in p["fields"]) and p["browser"] is None

        url = "/api/plugins/login/settings"
        r = client.put(url, json={"values": {"token": f"  {SECRET}  "}, "user_agent": UA})
        assert r.status_code == 200, r.text
        assert SECRET not in r.text and "Chrome/140" not in r.text and "value" not in r.text, r.text
        got = r.json()["plugin"]
        assert got["fields"][0]["set"] is True and got["fields"][0]["saved_at"] > 0
        assert got["fields"][1]["set"] is False
        assert got["browser"] == "Chrome 140（Windows）", got["browser"]
        # 外掛自己存的檔案在（暫存的）設定資料夾，內容已經去掉頭尾空白
        assert (env.settings_dir / "login" / "token.txt").read_text(encoding="utf-8") == SECRET
        assert (env.settings_dir / "login" / "ua.txt").read_text(encoding="utf-8") == UA

        # 讀回來也只有狀態
        for path in ("/api/plugins/settings", "/api/state", url):
            r = client.get(path)
            assert SECRET not in r.text and "Chrome/140" not in r.text, path

        # 錯誤：沒有的欄位、空的、沒有 UA、太長、不是 JSON、不是物件、沒有這個外掛
        bad = [({"values": {"password": "x"}, "user_agent": UA}, 400),
               ({"values": {"token": "   "}, "user_agent": UA}, 400),
               ({"values": {"token": "x"}}, 400),
               ({"values": {"token": "x"}, "user_agent": "a\nb"}, 400),
               ({"values": {"token": "x" * (plugins.MAX_VALUE_CHARS + 1)}, "user_agent": UA}, 400),
               ({"values": {"token": 5}, "user_agent": UA}, 400),
               ({"values": ["token"], "user_agent": UA}, 400),
               ([1, 2], 400)]
        for payload, status in bad:
            r = client.put(url, json=payload)
            assert r.status_code == status, (payload, r.status_code, r.text)
        r = client.put(url, content=f'{{"values": {{"token": "{SECRET}"'.encode(), headers={"Content-Type": "application/json"})
        assert r.status_code == 400 and SECRET not in r.text, r.text
        assert client.put("/api/plugins/nothere/settings", json={"values": {"token": "x"}, "user_agent": UA}).status_code == 404
        assert client.put("/api/plugins/..%5Clogin/settings", json={"values": {"token": "x"}, "user_agent": UA}).status_code == 404

        # 外掛的 ValueError 訊息給使用者看；其他例外（訊息可能帶內容）不回傳也不記錄
        r = client.put(url, json={"values": {"token": "bad-" + SECRET}, "user_agent": UA})
        assert r.status_code == 400 and "這串不像登入資料" in r.json()["detail"]
        r = client.put(url, json={"values": {"token": "boom-" + SECRET}, "user_agent": UA})
        assert r.status_code == 500 and SECRET not in r.text

        # 別的網站送來的、區網開關關著時手機送來的，都擋掉，檔案不變
        r = client.put(url, json={"values": {"token": "evil"}, "user_agent": UA}, headers={"Origin": "http://evil.example"})
        assert r.status_code == 403
        phone = TestClient(server.app, client=("192.168.1.20", 50123))
        assert phone.put(url, json={"values": {"token": "evil"}, "user_agent": UA}).status_code == 403
        assert phone.get("/api/plugins/settings").status_code == 403
        assert (env.settings_dir / "login" / "token.txt").read_text(encoding="utf-8") == SECRET

        # 清除
        r = client.delete(url)
        assert r.status_code == 200 and r.json()["plugin"]["fields"][0]["set"] is False
        assert not (env.settings_dir / "login" / "token.txt").exists()

        assert any("settings saved" in line for line in env.log.lines) \
            and any("save_settings() failed" in line for line in env.log.lines), env.log.lines   # 真的有收到 log
        assert not any(SECRET in line or "Chrome/140" in line for line in env.log.lines), \
            [line for line in env.log.lines if SECRET in line]
    run(body)


def test_settings_folder_can_be_moved_and_links_refused():
    def body(env):
        write_plugin(env.plugins, "login.py", SETTINGS_PLUGIN)
        p = plugins.by_id("login")
        assert plugins.settings_dir(p) == env.settings_dir / "login"
        assert plugins.by_id("..") is None and plugins.by_id("con") is None
        # 設定資料夾是捷徑（junction）：不寫
        real = env.outside / "elsewhere"
        real.mkdir(parents=True)
        junction(env.settings_dir / "login", real)
        try:
            plugins.save_settings(p, {"token": SECRET}, UA)
        except plugins.PluginError as e:
            assert e.status == 500
        else:
            raise AssertionError("設定資料夾是捷徑時應該拒絕")
        assert not any(real.iterdir())
        os.rmdir(env.settings_dir / "login")
    run(body)


def test_browser_label():
    cases = {
        UA: "Chrome 140（Windows）",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:143.0) Gecko/20100101 Firefox/143.0": "Firefox 143（Windows）",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 "
        "Safari/537.36 Edg/140.0.0.0": "Edge 140（Windows）",
        "Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) "
        "Version/18.0 Mobile/15E148 Safari/604.1": "Safari 18（iPhone）",
        "Mozilla/5.0 (Linux; Android 14) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0 Mobile Safari/537.36":
            "Chrome 140（Android）",
        "": None, None: None, "curl/8": "其他瀏覽器",
    }
    for ua, want in cases.items():
        assert plugins.browser_label(ua) == want, (ua, plugins.browser_label(ua))


def test_cookie_header_formats():
    ch = plugins.cookie_header
    # Cookie 標頭：可以帶「cookie:」開頭、換行、多餘的分號
    assert ch("cookie: a=1; b=two==; \n c=3;;", ("site.example",)) == "a=1; b=two==; c=3"
    assert ch("  A=1;B=2  ") == "A=1; B=2"
    # 擴充套件匯出的 JSON：只留這個網域（和子網域）的，拿掉不要的名稱，同名用後面的
    exported = json.dumps([
        {"domain": ".site.example", "name": "sid", "value": "s1", "path": "/", "httpOnly": True},
        {"domain": "www.site.example", "name": "uid", "value": "u1"},
        {"domain": "site.example", "name": "noisy", "value": "x"},
        {"domain": ".evil.example", "name": "evil", "value": "e"},
        {"domain": "notsite.example", "name": "near", "value": "n"},
        {"name": "nodomain", "value": "d"},
        {"domain": ".site.example", "name": "sid", "value": "s2"},
        {"domain": ".site.example", "name": "bad name", "value": "v"},
        {"domain": ".site.example", "name": "semi", "value": "a;b"},
        "junk", {"domain": ".site.example", "value": "no name"},
    ])
    assert ch(exported, ("site.example",), drop=("noisy",)) == "uid=u1; sid=s2"
    assert ch(json.dumps({"cookies": [{"domain": "site.example", "name": "k", "value": 5}]}), ("site.example",)) == "k=5"
    # cookies.txt（Netscape 格式）
    netscape = ("# Netscape HTTP Cookie File\n"
                ".site.example\tTRUE\t/\tTRUE\t0\tsid\tabc\n"
                "#HttpOnly_.site.example\tTRUE\t/\tTRUE\t0\thid\tdef\n"
                ".evil.example\tTRUE\t/\tFALSE\t0\tevil\tx\n")
    assert ch(netscape, ("site.example",)) == "sid=abc; hid=def"
    for bad in ("", "   ", "[1, 2", "just words", None, 5, "[]"):
        assert ch(bad, ("site.example",)) == "", bad


AUTO_UA_PLUGIN = SETTINGS_PLUGIN.replace('"user_agent": True,', '"user_agent": True, "auto_user_agent": True,') + '''

def save_user_agent(ctx):
    (Path(ctx.dir) / "ua.txt").write_text(ctx.user_agent, encoding="utf-8")
'''


def test_user_agent_is_sent_on_page_load_rules():
    def body(env):
        write_plugin(env.plugins, "login.py", AUTO_UA_PLUGIN)
        client = TestClient(server.app)
        ua_file = env.settings_dir / "login" / "ua.txt"
        p = client.get("/api/plugins/settings").json()["plugins"][0]
        assert p["auto_user_agent"] is True and p["ua_fp"] is None

        url = "/api/plugins/login/user-agent"
        new = UA.replace("Chrome/140", "Chrome/141")
        # 還沒存過：直接存
        r = client.put(url, json={"user_agent": UA})
        assert r.status_code == 200 and r.json()["updated"] is True and ua_file.read_text(encoding="utf-8") == UA
        p = client.get("/api/plugins/settings").json()["plugins"][0]
        assert p["ua_fp"] == plugins.ua_fingerprint(UA) and "Chrome/140" not in json.dumps(p)
        # 一樣的：不動；同一種瀏覽器的新版本：更新；別種瀏覽器或別的系統（手機）：不動
        assert client.put(url, json={"user_agent": UA}).json()["updated"] is False
        assert client.put(url, json={"user_agent": new}).json()["updated"] is True
        assert ua_file.read_text(encoding="utf-8") == new
        phone = "Mozilla/5.0 (Linux; Android 14) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0 Mobile Safari/537.36"
        firefox = "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:143.0) Gecko/20100101 Firefox/143.0"
        for other in (phone, firefox):
            assert client.put(url, json={"user_agent": other}).json()["updated"] is False
        assert ua_file.read_text(encoding="utf-8") == new
        for bad in ("", "a\nb", "x" * (plugins.MAX_UA_CHARS + 1)):
            assert client.put(url, json={"user_agent": bad}).status_code == 400
        assert client.put(url, json={"user_agent": new}, headers={"Origin": "http://evil.example"}).status_code == 403
        # 沒有要 UA 的外掛：404
        write_plugin(env.plugins, "login.py", SETTINGS_PLUGIN)
        assert client.put(url, json={"user_agent": UA}).status_code == 404
        # 網頁的雜湊算法要跟這裡一樣（web/app.js 的 uaFingerprint）
        assert plugins.ua_fingerprint("abc") == "1a47e90b" and plugins.ua_fingerprint("中") == "b846d3f4"
    run(body)


# ---------- 展開成清單、一次加入勾選的集數 ----------

LIST_PLUGIN = '''
NAME = "假作品"
OPTIONS = {"language": "ja", "translate": False}
CALLS = []

def match(url):
    return url.startswith("https://anime.example/ep/")

def item_key(url):
    return url.rstrip("/").rsplit("/", 1)[-1].split("?")[0]

def download(job):
    f = job.out_dir / "ep.mp4"
    f.write_bytes(b"x")
    return {"path": str(f), "title": "下載後的標題",
            "series": {"series": "下載時的作品", "season": "第一季", "episode": 9}}

def ep(n, season, order, version=None):
    info = {"series": "假動畫", "season": season, "season_order": order, "episode": n}
    if version:
        info["version"] = version
    return info

def expand(url, ctx):
    CALLS.append(url)
    if url.endswith("/fail"):
        raise RuntimeError("網站沒有回應")
    if url.endswith("/empty"):
        return {"title": "空", "groups": [{"title": "x", "items": [{"url": "file:///C:/x"}]}]}
    return {"title": "假動畫", "groups": [
        {"title": "第一季", "items": [
            {"url": "https://anime.example/ep/101", "label": "1", "title": "假動畫 [1]", "series": ep(1, "第一季", 1)},
            {"url": "https://anime.example/ep/102", "label": "2", "title": "假動畫 [2]", "series": ep(2, "第一季", 1)},
            {"url": "file:///C:/Windows/win.ini", "label": "壞"},
            {"url": "https://other.example/ep/9", "label": "別人的"},
            {"url": "https://anime.example/ep/101", "label": "重複"},
            {"url": "https://anime.example/ep/103", "label": "3", "title": "假動畫 [3]", "series": ep(3, "第一季", 1)}]},
        {"title": "第一季 中文配音", "note": "配音版", "checked": False,
         "options": {"language": "zh", "bogus": 1},
         "items": [{"url": "https://anime.example/ep/201", "label": "1", "title": "假動畫 [1] 配音",
                    "series": ep(1, "第一季", 1, "中文配音")}]},
        {"title": "沒有集數", "items": []}]}
'''


def test_expand_lists_groups_and_library():
    def body(env):
        write_plugin(env.plugins, "work.py", LIST_PLUGIN)
        client = TestClient(server.app)
        assert server.url_plugin("https://anime.example/ep/102")["expand"] is True
        # 已經在播放列表的那一集（網址寫法不同也算）
        have = env.add_media("url", url="https://anime.example/ep/102?from=share")

        r = client.get("/api/url-plugin/list", params={"url": "https://anime.example/ep/103"})
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["plugin"] == "假作品" and data["title"] == "假動畫"
        g1, g2 = data["groups"]
        assert [it["label"] for it in g1["items"]] == ["1", "2", "3"]      # 壞網址、別的外掛的、重複的丟掉
        assert g1["checked"] is True and g1["language"] == "ja"
        assert g2["checked"] is False and g2["language"] == "zh" and g2["note"] == "配音版"
        assert [it["in_library"] for it in g1["items"]] == [False, True, False]
        assert g1["items"][1]["media_id"] == have
        assert [it["current"] for it in g1["items"]] == [False, False, True]

        for url, status in (("https://anime.example/ep/fail", 502), ("https://anime.example/ep/empty", 404),
                            ("https://fake.example/1", 400), ("file:///C:/x", 400), ("http://127.0.0.1:8765/", 400)):
            r = client.get("/api/url-plugin/list", params={"url": url})
            assert r.status_code == status, (url, r.status_code, r.text)
        assert "網站沒有回應" in client.get("/api/url-plugin/list", params={"url": "https://anime.example/ep/fail"}).json()["detail"]
    run(body)


def test_batch_add_uses_group_options_and_order():
    def body(env):
        write_plugin(env.plugins, "work.py", LIST_PLUGIN)
        seen = capture_options(env)
        client = TestClient(server.app)
        base = {"language": "en", "translate": True, "engine": "whisper-en", "sensitive": True,
                "url": "https://anime.example/ep/101"}

        # 沒勾、清單裡沒有的網址：什麼都不加
        for items in ([], ["https://anime.example/ep/999"], ["https://anime.example/ep/101", "file:///C:/x"]):
            r = client.post("/api/media/batch", json={**base, "items": items})
            assert r.status_code == 400, (items, r.text)
        assert db.list_media() == []

        # 勾的順序亂也照清單的順序加
        items = ["https://anime.example/ep/201", "https://anime.example/ep/103", "https://anime.example/ep/101"]
        r = client.post("/api/media/batch", json={**base, "items": items})
        assert r.status_code == 200, r.text
        ids = r.json()["ids"]
        assert r.json()["count"] == 3
        ms = [db.get_media(i) for i in ids]
        assert [m["url"] for m in ms] == [items[2], items[1], items[0]]
        assert [m["title"] for m in ms] == ["假動畫 [1]", "假動畫 [3]", "假動畫 [1] 配音"]
        assert [m["language"] for m in ms] == ["ja", "ja", "zh"]
        # 外掛固定的（日文、不翻譯）蓋掉網頁送的；配音那組再蓋成中文；換語言時網頁選的模型不用
        assert all(o["translate"] is False and o["engine"] is None and o["sensitive"] is True for o in seen)
        for m in ms:
            js = db.jobs_for_media(m["id"])
            assert [j["type"] for j in js if j["type"] in ("download", "transcribe", "translate")] == ["download", "transcribe"]
        # 作品、季、集存起來了（依作品分的播放列表）
        from app import series
        infos = [series.loads(m["series_info"]) for m in ms]
        assert infos[0] == {"series": "假動畫", "season": "第一季", "season_order": 1.0, "episode": 1.0,
                            "episode_label": "1"}, infos[0]
        assert infos[2]["version"] == "中文配音"

        # 一般的單集加入也還是可以
        r = client.post("/api/media", json={"source": "url", "url": "https://anime.example/ep/102", "language": "en"})
        assert r.status_code == 200 and db.get_media(r.json()["id"])["series_info"] is None
    run(body)


def test_download_result_series_fills_only_missing():
    def body(env):
        write_plugin(env.plugins, "work.py", LIST_PLUGIN)
        fake_probe(env)
        from app import series
        mids = []
        for url, info in (("https://anime.example/ep/301", None),
                          ("https://anime.example/ep/302", {"series": "清單的作品", "episode": 2})):
            mid = env.add_media("url", url=url, series_info=series.dumps(info))
            ctx = jobs.JobContext(db.get_job(db.add_job(mid, "download", {"url": url})))
            jobs.handle_download(ctx)
            mids.append(mid)
        assert series.loads(db.get_media(mids[0])["series_info"])["series"] == "下載時的作品"
        assert series.loads(db.get_media(mids[1])["series_info"])["series"] == "清單的作品"
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
