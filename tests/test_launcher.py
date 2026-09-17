"""啟動器（launcher\\、start.bat、tools\\build_release.py）的測試，不用網路、不用顯示卡。

- 發佈 zip 的白名單：data、models、bin、runtime、private、tests、.dev、__pycache__ 一定不會被放進去
- runtime-manifest.json、get_uv.cmd、.python-version、start.bat、env.cmd 彼此一致
- 下載：本機假伺服器測分段下載、中斷後續傳、第一個網址壞掉改用下一個、sha256 不符、伺服器不支援 Range、
  進度檔被別的程式開著時不會重複下載、還剩多少的計算
- 磁碟滿了的說明、repair 在程式還開著時不刪東西、start.bat 在非英文路徑一開始就停下
- install-state.json 的格式 app/config.py 讀得懂（find_tool、LLAMA_SERVER、nvenc）
- 資料夾搬家後 pyvenv.cfg 的修正、快速路徑的判斷、zip 解壓縮的篩選
- precheck.py 的路徑、顯示卡判斷（用 syscheck.set_fake 假造）

暫存檔放在環境變數 VS_TEST_TMP 指定的資料夾，沒設就用系統暫存資料夾，測完刪掉。

python -s tests/test_launcher.py
"""
import hashlib
import http.server
import json
import os
import shutil
import sys
import tempfile
import threading
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "launcher"))
sys.path.insert(0, str(ROOT / "tools"))

import build_release  # noqa: E402
import precheck  # noqa: E402
import setup_runtime as sr  # noqa: E402
from app import config, syscheck  # noqa: E402

BASE = Path(os.environ.get("VS_TEST_TMP") or tempfile.gettempdir()) / "launcher-test"
# setup_runtime、precheck 預設寫到專案的 runtime\：測試時全部導到暫存資料夾，專案裡不能多出 runtime
PROJECT_RUNTIME_EXISTED = (ROOT / "runtime").exists()
sr.LOGS = BASE / "logs"
sr.LOG_PATH = sr.LOGS / "install.log"
precheck.RT = BASE / "precheck-runtime"
precheck.OUT_PATH = precheck.RT / "precheck.json"
precheck.STATE_PATH = precheck.RT / "install-state.json"


def fresh(name: str) -> Path:
    d = BASE / name
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    return d


class Quiet(sr.Console):
    def __init__(self):
        super().__init__()
        self.tty = False
        self.lines = []

    def say(self, text=""):
        self.lines.append(text)


# ---------- 發佈 zip ----------

def test_release_whitelist_never_includes_private_things():
    root = fresh("release-root")
    for rel in ["start.bat", "repair.bat", "README.md", "LICENSE", "THIRD_PARTY_NOTICES.md", "pyproject.toml",
                "uv.lock", ".python-version", "app/server.py", "app/__init__.py", "web/index.html",
                "launcher/setup_runtime.py", "launcher/precheck.py", "launcher/env.cmd", "launcher/get_uv.cmd",
                "launcher/runtime-manifest.json", "launcher/msg/install_failed.txt", "tools/build_dict_ja.py",
                # 不能放進去的
                ".dev", "data/library.db", "data/app.log", "models/gguf/x.gguf", "bin/llama.cpp/llama-server.exe",
                "runtime/install-state.json", "private/docs/plan.md", "tests/samples/a.mp4", "tests/test_x.py",
                "docs/internal/x.md", "app/__pycache__/server.cpython-312.pyc", "tools/build_release.py",
                "web/__pycache__/x.js", "launcher/__pycache__/x.pyc", "DJI_0001.MP4", "app/cookies.txt"]:
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        data = b"@echo off\r\n" if rel.endswith((".bat", ".cmd")) else b"[project]\nversion = \"9.9.9\"\n" \
            if rel == "pyproject.toml" else b"x"
        p.write_bytes(data)
    files = build_release.collect(root)
    names = {rel for _, rel in files}
    for bad in (".dev", "data/library.db", "models/gguf/x.gguf", "runtime/install-state.json", "private/docs/plan.md",
                "tests/samples/a.mp4", "tests/test_x.py", "app/__pycache__/server.cpython-312.pyc",
                "tools/build_release.py", "DJI_0001.MP4", "web/__pycache__/x.js", "launcher/__pycache__/x.pyc"):
        assert bad not in names, bad
    assert "README.txt" in names and "app/server.py" in names and "launcher/msg/install_failed.txt" in names
    build_release.check(files)
    out = fresh("release-out")
    zp = build_release.build(out, root=root)
    with zipfile.ZipFile(zp) as z:
        listed = z.namelist()
    assert all(n.startswith("BilingualSubtitles/") for n in listed)
    assert not any(part in n.lower() for n in listed for part in ("/data/", "/models/", "/runtime/", "/private/",
                                                                     "/tests/", "__pycache__", "/.dev"))
    assert zp.name == "BilingualSubtitles-9.9.9.zip" and zp.with_name(zp.name + ".sha256").is_file()


def test_release_rejects_forbidden_even_if_whitelisted():
    assert build_release.forbidden("app/__pycache__/x.py")
    assert build_release.forbidden("web/data/x.js")
    assert build_release.forbidden(".dev")
    assert build_release.forbidden("tools/yomitan/tests/samples/x.json")
    assert build_release.forbidden("web/movie.mp4")
    assert build_release.forbidden("app/server.py") is None
    root = fresh("release-lf")
    (root / "start.bat").write_bytes(b"@echo off\ngoto x\n")
    try:
        build_release.check([(root / "start.bat", "start.bat")])
    except build_release.ReleaseError as e:
        assert "CRLF" in str(e)
    else:
        raise AssertionError("LF bat should be rejected")


def _fake_release_root(name: str) -> Path:
    root = fresh(name)
    for rel in ["start.bat", "repair.bat", "README.md", "LICENSE", "THIRD_PARTY_NOTICES.md", "pyproject.toml",
                "uv.lock", ".python-version", "app/server.py", "web/index.html", "launcher/setup_runtime.py",
                "launcher/precheck.py", "launcher/env.cmd", "launcher/get_uv.cmd", "launcher/msg/install_failed.txt",
                ".dev", "data/library.db", "models/x.gguf", "runtime/install-state.json", "private/bundle-cache/x.zip",
                "tests/samples/a.mp4", "offline/sneaky.txt"]:
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"@echo off\r\n" if rel.endswith((".bat", ".cmd")) else
                      b"[project]\nversion = \"9.9.9\"\n" if rel == "pyproject.toml" else b"x")
    return root


SM_WHEEL = "en_core_web_sm-3.8.0-py3-none-any.whl"


def _fake_lock(path: Path, wheel: bytes) -> str:
    """假的 uv.lock：GitHub 上的 wheel（要附）、PyPI 上的 wheel（不附）、GitHub 上但這台電腦不會裝的 Linux wheel（不附）。"""
    sha = hashlib.sha256(wheel).hexdigest()
    gh = "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0"
    path.write_text(
        'version = 1\n\n[[package]]\nname = "en-core-web-sm"\nversion = "3.8.0"\n'
        f'source = {{ url = "{gh}/{SM_WHEEL}" }}\nwheels = [\n    {{ url = "{gh}/{SM_WHEEL}", hash = "sha256:{sha}" }},\n]\n\n'
        '[[package]]\nname = "fastapi"\nversion = "1.0"\nsource = { registry = "https://pypi.org/simple" }\nwheels = [\n'
        f'    {{ url = "https://files.pythonhosted.org/packages/fastapi-1.0-py3-none-any.whl", hash = "sha256:{"a" * 64}", size = 10 }},\n]\n\n'
        '[[package]]\nname = "gh-linux"\nversion = "1.0"\nwheels = [\n'
        f'    {{ url = "https://github.com/x/releases/download/y/gh_linux-1.0-cp312-cp312-manylinux_2_17_x86_64.whl", hash = "sha256:{"b" * 64}" }},\n]\n',
        encoding="utf-8")
    return sha


def test_full_release_adds_offline_files_with_sha256_list():
    """完整包：一般包的白名單檔案，加上 offline\\ 裡清單列出、sha256 核對過的檔案；快取缺檔或檔案不對就不打包。"""
    import gzip
    import io
    from app import dict_build
    root = _fake_release_root("full-root")
    wheel = os.urandom(20_000)
    _fake_lock(root / "uv.lock", wheel)
    big = os.urandom(300_000)
    csv = b"word,level\nx,A1\n"
    jm_xml = io.BytesIO()
    with gzip.GzipFile(fileobj=jm_xml, mode="wb", mtime=0) as f:
        f.write(b"<JMdict>\n" + b"<entry><ent_seq>1</ent_seq></entry>\n" * 20 + b"</JMdict>\n")
    jm = jm_xml.getvalue()
    manifest = {
        "manifest_version": 1,
        "uv": {"files": [{"name": "uv.whl", "urls": ["https://files.pythonhosted.org/uv.whl"], "sha256": "a" * 64, "size": 1}]},
        "python": {"version": "3.12.14", "dir": "x"},
        "ffmpeg": {"version": "n8", "build": "test", "source": "https://github.com/FFmpeg/FFmpeg/commit/x",
                   "files": [{"name": "ff.zip", "urls": ["https://github.com/BtbN/FFmpeg-Builds/releases/download/x/ff.zip"],
                              "sha256": hashlib.sha256(big).hexdigest().upper(), "size": len(big)}]},
        "llama_cpp": {"version": "b1", "files": [
            {"name": "ll.zip", "urls": ["https://objects.githubusercontent.com/ll.zip"],
             "sha256": hashlib.sha256(big[::-1]).hexdigest(), "size": len(big)}]},
    }
    (root / "launcher" / "runtime-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    sources = {"ja": [{"name": "JMdict_e.gz", "daily": True, "bytes": None, "sha256": None, "urls": ["https://www.edrdg.org/x"]}],
               "en": [{"name": "cefr.csv", "bytes": len(csv), "sha256": hashlib.sha256(csv).hexdigest(),
                       "urls": ["https://raw.githubusercontent.com/x/cefr.csv"]}],
               "zh": [{"name": "zh.json.xz", "bytes": 1, "sha256": "b" * 64, "urls": ["https://raw.githubusercontent.com/x"]}]}
    items = build_release.offline_items(root, dict_sources=sources)
    # GitHub 上的執行檔、uv.lock 裡 GitHub 上的 wheel、日文英文字典的來源；PyPI 上的 uv、中文辭典不附
    assert [i["name"] for i in items] == ["ff.zip", "ll.zip", SM_WHEEL, "JMdict_e.gz", "cefr.csv"], items
    assert items[2]["for"] == "python" and items[2]["size"] is None and items[2]["sha256"] == hashlib.sha256(wheel).hexdigest()
    assert build_release.forbidden("offline/ff.zip") and build_release.forbidden("offline/x.txt")
    cache = fresh("full-cache")
    out = fresh("full-out")
    old_min = dict_build.JMDICT_MIN_ENTRIES
    dict_build.JMDICT_MIN_ENTRIES = 10
    try:
        # 快取缺檔、--no-fetch：報錯，不產生 zip
        (cache / SM_WHEEL).write_bytes(wheel)
        (cache / "ff.zip").write_bytes(big)
        (cache / "ll.zip").write_bytes(big[::-1])
        (cache / "JMdict_e.gz").write_bytes(jm)
        try:
            build_release.build(out, root=root, full=True, cache=cache, items=items, fetch=False)
        except build_release.ReleaseError as e:
            assert "cefr.csv" in str(e) and "ff.zip" not in str(e), e
        else:
            raise AssertionError("缺檔時不能打包")
        # sha256 不對（大小一樣）
        (cache / "cefr.csv").write_bytes(csv)
        (cache / "ll.zip").write_bytes(big)
        try:
            build_release.build(out, root=root, full=True, cache=cache, items=items, fetch=False)
        except build_release.ReleaseError as e:
            assert "ll.zip" in str(e), e
        else:
            raise AssertionError("sha256 不對時不能打包")
        assert list(out.iterdir()) == []
        (cache / "ll.zip").write_bytes(big[::-1])
        zp = build_release.build(out, root=root, full=True, cache=cache, items=items, fetch=False)
    finally:
        dict_build.JMDICT_MIN_ENTRIES = old_min
    assert zp.name == "BilingualSubtitles-9.9.9-full.zip"
    digest = zp.with_name(zp.name + ".sha256").read_text(encoding="utf-8").split()[0]
    assert digest == hashlib.sha256(zp.read_bytes()).hexdigest()
    normal = build_release.build(out, root=root)
    with zipfile.ZipFile(normal) as z:
        normal_names = set(z.namelist())
    with zipfile.ZipFile(zp) as z:
        names = set(z.namelist())
        listing = json.loads(z.read("BilingualSubtitles/offline/files.json"))
        offline = {n for n in names if n.startswith("BilingualSubtitles/offline/")}
        assert names - offline == normal_names, names ^ normal_names
        assert offline == {f"BilingualSubtitles/offline/{n}" for n in ("files.json", "README.txt", "LICENSE-ECDICT.txt",
                                                              "LICENSE-llama.cpp.txt", "ff.zip", "ll.zip", SM_WHEEL,
                                                              "JMdict_e.gz", "cefr.csv")}, offline
        assert "Copyright (c) 2023-2026 The ggml authors" in z.read("BilingualSubtitles/offline/LICENSE-llama.cpp.txt").decode("utf-8")
        assert [f["name"] for f in listing["files"]] == ["ff.zip", "ll.zip", SM_WHEEL, "JMdict_e.gz", "cefr.csv"]
        for f in listing["files"]:
            content = z.read(f"BilingualSubtitles/offline/{f['name']}")
            assert hashlib.sha256(content).hexdigest() == f["sha256"] and len(content) == f["size"], f
        assert listing["files"][2]["for"] == "python" and listing["files"][2]["size"] == len(wheel)
        assert listing["files"][3]["sha256"] == hashlib.sha256(jm).hexdigest() and listing["files"][3]["for"] == "dict:ja"
        assert z.getinfo("BilingualSubtitles/offline/ff.zip").compress_type == zipfile.ZIP_STORED
        assert z.getinfo(f"BilingualSubtitles/offline/{SM_WHEEL}").compress_type == zipfile.ZIP_STORED
        readme = z.read("BilingualSubtitles/offline/README.txt").decode("utf-8")
        assert "GPL" in readme and "THIRD_PARTY_NOTICES" in readme and "—" not in readme
    for bad in ("/.dev", "/data/", "/models/", "/runtime/", "/private/", "/tests/", "sneaky"):
        assert not any(bad in n for n in names), bad


def test_real_project_offline_items():
    items = build_release.offline_items()
    names = [i["name"] for i in items]
    manifest = json.loads((ROOT / "launcher" / "runtime-manifest.json").read_text(encoding="utf-8"))
    for key in ("ffmpeg", "llama_cpp"):
        for f in manifest[key]["files"]:
            assert f["name"] in names, f["name"]
    assert manifest["uv"]["files"][0]["name"] not in names and manifest["msvc_runtime"]["files"][0]["name"] not in names
    from app import dict_build
    for lang in ("ja", "en"):
        for s in dict_build.SOURCES[lang]:
            assert s["name"] in names, s["name"]
    assert "dict-revised.json.xz" not in names
    # uv.lock 裡放在 GitHub 上的 en-core-web-sm（2026-09-17 測試：完整包安裝時有 4 分 57 秒在等它）
    wheels = [i for i in items if i["for"] == "python"]
    assert [w["name"] for w in wheels] == [SM_WHEEL], wheels
    assert wheels[0]["sha256"] == "1932429db727d4bff3deed6b34cfc05df17794f4a52eeb26cf8928f7c1a0fb85"
    assert all(i["sha256"] and (i["size"] or i["for"] == "python") for i in items if not i.get("daily"))
    # 執行環境要的 GitHub 檔案，完整包全部都附（precheck 靠這份清單決定要不要檢查 GitHub）
    assert {f["name"] for f in sr.slow_runtime_files(manifest)} <= set(names)


def test_real_project_release_list_is_clean():
    files = build_release.collect()
    build_release.check(files)
    names = [rel for _, rel in files]
    assert ".dev" not in names and not any(n.startswith(("data/", "models/", "bin/", "runtime/", "private/", "tests/"))
                                           for n in names)


# ---------- 設定檔彼此一致 ----------

def test_manifest_matches_get_uv_python_version_and_start_bat():
    manifest = json.loads((ROOT / "launcher" / "runtime-manifest.json").read_text(encoding="utf-8"))
    uv = manifest["uv"]
    get_uv = (ROOT / "launcher" / "get_uv.cmd").read_text(encoding="utf-8")
    assert f'set "UVVER={uv["version"]}"' in get_uv
    assert f'set "UVSHA={uv["files"][0]["sha256"]}"' in get_uv
    assert f'set "UVURL={uv["files"][0]["urls"][0]}"' in get_uv
    assert uv["files"][0]["name"] in get_uv
    py = manifest["python"]
    assert (ROOT / ".python-version").read_text(encoding="utf-8").strip() == py["version"]
    start = (ROOT / "start.bat").read_text(encoding="utf-8")
    assert f"python\\{py['dir']}\\python.exe" in start
    assert f"python install {py['version']}" in start
    for f in [manifest[k]["files"] for k in ("uv", "msvc_runtime", "ffmpeg", "llama_cpp")]:
        for item in f:
            assert len(item["sha256"]) == 64 and item["size"] > 0 and item["urls"], item
    assert manifest["llama_cpp"]["build"] == 10985


def test_env_cmd_matches_setup_runtime_defaults():
    text = (ROOT / "launcher" / "env.cmd").read_text(encoding="utf-8")
    values = {}
    for line in text.splitlines():
        line = line.strip()
        if line.lower().startswith('set "') and "=" in line:
            k, _, v = line[5:].rstrip('"').partition("=")
            values[k] = v
    for key, want in sr._ENV_DEFAULTS.items():
        assert key in values, key
        got = values[key].replace("%RT%", str(sr.RT))
        assert os.path.normcase(got) == os.path.normcase(str(want)), (key, got, want)
    assert "VS_HOST" not in values     # 區網開關要能生效
    assert "set \"PATH=%RT%\\venv\\Scripts;%PATH%\"" in text


def test_bat_files_are_crlf_and_ascii_except_fallback():
    for rel in ("start.bat", "repair.bat", "launcher/env.cmd", "launcher/get_uv.cmd"):
        data = (ROOT / rel).read_bytes()
        assert data.count(b"\n") == data.count(b"\r\n"), rel
        lines = data.decode("utf-8").split("\r\n")
        non_ascii = [i for i, l in enumerate(lines) if any(ord(c) > 127 for c in l)]
        # 只有檔案最後、找不到 launcher 資料夾時的那一行說明可以有中文
        assert len(non_ascii) <= 1 and all(i > len(lines) - 12 for i in non_ascii), (rel, non_ascii)
    # start.bat 停下來等按鍵時（開發模式也一樣）顯示中文「按任意鍵關閉這個視窗」，不顯示英文的 Press any key to continue
    start = (ROOT / "start.bat").read_text(encoding="utf-8").splitlines()
    pauses = [i for i, l in enumerate(start) if l.strip().split(" ")[-1] == "pause" or l.strip().startswith("pause")]
    assert pauses and all(start[i].strip() == "pause >nul" for i in pauses), [start[i] for i in pauses]
    for i in pauses:
        assert "press_any_key.txt" in start[i - 1] or "按任意鍵關閉這個視窗" in start[i - 1], start[i - 1]
    assert (ROOT / "launcher" / "msg" / "press_any_key.txt").read_text(encoding="utf-8").strip() == "按任意鍵關閉這個視窗。"
    for p in (ROOT / "launcher" / "msg").glob("*.txt"):
        data = p.read_bytes()
        assert not data.startswith(b"\xef\xbb\xbf"), p.name
        data.decode("utf-8")
        assert "—" not in data.decode("utf-8"), p.name


def test_lock_has_torch_cu130_and_no_ytdlp():
    torch = sr.lock_torch()
    assert torch["version"] == "2.14.0+cu130"
    assert torch["name"] == "torch-2.14.0+cu130-cp312-cp312-win_amd64.whl"
    assert len(torch["sha256"]) == 64 and torch["urls"][0].startswith("https://download")
    lock = (ROOT / "uv.lock").read_text(encoding="utf-8")
    assert 'name = "yt-dlp"' not in lock
    assert 'name = "psutil"' in lock and 'name = "onnxruntime"' in lock


# ---------- install-state 跟 app/config.py ----------

def test_install_state_format_is_what_config_reads():
    root = fresh("state")
    ff = root / "runtime" / "bin" / "ffmpeg-x" / "bin" / "ffmpeg.exe"
    ff.parent.mkdir(parents=True)
    ff.write_bytes(b"x")
    state = {"ffmpeg": {"path": str(ff), "dir": "runtime\\bin\\ffmpeg-x", "version": "n8"},
             "ffprobe": {"path": "runtime\\bin\\ffmpeg-x\\bin\\ffprobe.exe"},
             "llama_server": {"path": "runtime\\bin\\llama.cpp-b10985\\llama-server.exe", "build": 10985},
             "deno": {"path": "runtime\\venv\\Scripts\\deno.exe"}, "nvenc": False}
    found = config.find_tool("ffmpeg", env={}, root=root, state=state, which=lambda n: None, legacy_dir=root / "none")
    assert found["path"] == str(ff) and found["source"] == "install-state", found
    assert config._json_path(state, "llama_server") == config.ROOT / "runtime" / "bin" / "llama.cpp-b10985" / "llama-server.exe"
    assert config._json_path(state, "ffprobe") == config.ROOT / "runtime" / "bin" / "ffmpeg-x" / "bin" / "ffprobe.exe"
    assert config._json_path(state, "deno") == config.ROOT / "runtime" / "venv" / "Scripts" / "deno.exe"
    assert config._json_path({"deno": {"path": None}}, "deno") is None
    # setup_runtime.rel 寫的是相對根目錄的路徑
    assert sr.rel(sr.ROOT / "runtime" / "bin" / "x.exe") == "runtime\\bin\\x.exe"


# ---------- 下載 ----------

class FileServer:
    """本機假下載伺服器：/data（支援 Range）、/norange（不支援 Range）、/missing（404）。
    drop_after：前幾次請求在送出這麼多位元組後直接斷線（模擬網路中斷）。"""

    def __init__(self, data: bytes):
        self.data = data
        self.drop_budget = 0
        self.drop_after = 0
        self.delay = 0.0          # 每送 64 KB 停這麼久（模擬慢速連線）
        self.sent = 0
        self.requests = []
        outer = self

        class H(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def do_GET(self):
                outer.requests.append((self.path, self.headers.get("Range")))
                if self.path == "/missing":
                    self.send_response(404)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                body = outer.data
                rng = self.headers.get("Range") if self.path == "/data" else None
                start, end = 0, len(body) - 1
                if rng:
                    a, _, b = rng[len("bytes="):].partition("-")
                    start = int(a)
                    end = int(b) if b else len(body) - 1
                    self.send_response(206)
                    self.send_header("Content-Range", f"bytes {start}-{end}/{len(body)}")
                else:
                    self.send_response(200)
                chunk = body[start:end + 1]
                self.send_header("Content-Length", str(len(chunk)))
                self.end_headers()
                drop = outer.drop_budget > 0
                if drop:
                    outer.drop_budget -= 1
                    chunk = chunk[:outer.drop_after]
                try:
                    for i in range(0, len(chunk), 65536):
                        self.wfile.write(chunk[i:i + 65536])
                        outer.sent += len(chunk[i:i + 65536])
                        if outer.delay:
                            time.sleep(outer.delay)
                except OSError:
                    return
                if drop:
                    self.close_connection = True

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


def _with_fast_retries(fn):
    old = (sr.ATTEMPTS, sr.BACKOFF, sr.SEGMENT_MIN)
    sr.ATTEMPTS, sr.BACKOFF, sr.SEGMENT_MIN = 2, (1,), 1024 * 1024
    try:
        return fn()
    finally:
        sr.ATTEMPTS, sr.BACKOFF, sr.SEGMENT_MIN = old


def test_download_segmented_resume_fallback_and_sha():
    data = os.urandom(24 * 1024 * 1024 + 123)
    sha = hashlib.sha256(data).hexdigest()
    srv = FileServer(data)
    con = Quiet()
    try:
        def run():
            # 1. 第一個網址 404，改用第二個；分段下載（6 條連線）結果正確
            d = fresh("dl1")
            item = {"name": "a.bin", "urls": [srv.url + "/missing", srv.url + "/data"], "sha256": sha, "size": len(data)}
            p = sr.download(item, "測試", con, dest=d)
            assert p.read_bytes() == data and not (d / "a.bin.part").exists() and not (d / "a.bin.part.json").exists()
            ranges = [r for path, r in srv.requests if path == "/data"]
            assert len(ranges) == sr.CONNECTIONS, ranges

            # 2. 中斷：每條連線送 600KB 就斷，兩輪都失敗 → SetupError，.part 和進度檔留著
            d = fresh("dl2")
            item = {"name": "b.bin", "urls": [srv.url + "/data"], "sha256": sha, "size": len(data)}
            srv.drop_budget, srv.drop_after = 10 ** 6, 600_000
            try:
                sr.download(item, "測試", con, dest=d)
            except sr.SetupError as e:
                assert "接著下載" in str(e), e
            else:
                raise AssertionError("should fail while the server keeps dropping")
            meta = json.loads((d / "b.bin.part.json").read_text(encoding="utf-8"))
            done = sum(s[2] for s in meta["segments"])
            assert 0 < done < len(data), done
            # 3. 網路恢復：從停下的地方接著下載，送出的位元組數只有剩下的部分
            srv.drop_budget = 0
            srv.sent = 0
            p = sr.download(item, "測試", con, dest=d)
            assert p.read_bytes() == data
            assert srv.sent == len(data) - done, (srv.sent, len(data), done)

            # 4. 已經有正確的檔案：不再下載
            srv.sent = 0
            assert sr.download(item, "測試", con, dest=d) == p and srv.sent == 0

            # 5. sha256 不符：刪掉暫存檔並報錯
            d = fresh("dl3")
            bad = {"name": "c.bin", "urls": [srv.url + "/data"], "sha256": "0" * 64, "size": len(data)}
            try:
                sr.download(bad, "測試", con, dest=d)
            except sr.SetupError as e:
                assert "sha256" in str(e)
            else:
                raise AssertionError("sha mismatch should fail")
            assert not (d / "c.bin").exists() and not (d / "c.bin.part").exists()

            # 6. 伺服器不支援 Range：改成一條連線從頭下載
            d = fresh("dl4")
            item = {"name": "d.bin", "urls": [srv.url + "/norange"], "sha256": sha, "size": len(data)}
            assert sr.download(item, "測試", con, dest=d).read_bytes() == data

            # 7. 不知道大小（torch 在 uv.lock 裡沒寫大小）：先問伺服器
            d = fresh("dl5")
            item = {"name": "e.bin", "urls": [srv.url + "/data"], "sha256": sha, "size": None}
            assert sr.download(item, "測試", con, dest=d).read_bytes() == data

            # 8. VS_DOWNLOAD_CACHE：先從快取資料夾複製
            cache = fresh("dlcache")
            (cache / "f.bin").write_bytes(data)
            os.environ["VS_DOWNLOAD_CACHE"] = str(cache)
            try:
                d = fresh("dl6")
                srv.sent = 0
                item = {"name": "f.bin", "urls": [srv.url + "/data"], "sha256": sha, "size": len(data)}
                assert sr.download(item, "測試", con, dest=d).read_bytes() == data and srv.sent == 0
            finally:
                del os.environ["VS_DOWNLOAD_CACHE"]

        _with_fast_retries(run)

        # 9. 切成比連線數多的小段（16 MB 一段）：6 條連線輪流拿，每段各一次請求，結果正確
        old_size = sr.SEGMENT_SIZE
        sr.SEGMENT_SIZE = 1024 * 1024
        try:
            d = fresh("dl7")
            srv.requests.clear()
            item = {"name": "g.bin", "urls": [srv.url + "/data"], "sha256": sha, "size": len(data)}
            assert _with_fast_retries(lambda: sr.download(item, "測試", con, dest=d)).read_bytes() == data
            assert len([1 for path, _ in srv.requests if path == "/data"]) == 25, len(srv.requests)
        finally:
            sr.SEGMENT_SIZE = old_size
    finally:
        srv.close()


def test_locked_progress_file_does_not_start_a_second_batch_of_connections():
    """.part.json 暫時被別的程式開著、不允許刪除（防毒、備份、索引工具）：存進度失敗不能中斷下載，
    不然下一輪會有兩批連線同時下載同一段、最後 .part 改名時 WinError 32（2026-09-17 測試發現）。"""
    data = os.urandom(24 * 1024 * 1024)
    sha = hashlib.sha256(data).hexdigest()
    srv = FileServer(data)
    srv.delay = 0.04          # 每條連線約 1.6 MB/s，6 條連線下載 24 MB 約 2.5 秒
    con = Quiet()
    d = fresh("dl-locked")
    meta = d / "h.bin.part.json"
    held = {}

    def hold():
        end = time.time() + 10
        while not meta.exists() and time.time() < end:
            time.sleep(0.02)
        time.sleep(0.3)
        with open(meta, "rb"):      # Python 的 open 不允許刪除、改名：os.replace 到這個檔案會失敗
            held["at"] = time.time()
            time.sleep(2.0)

    th = threading.Thread(target=hold, daemon=True)
    try:
        th.start()
        item = {"name": "h.bin", "urls": [srv.url + "/data"], "sha256": sha, "size": len(data)}
        started = time.time()
        p = _with_fast_retries(lambda: sr.download(item, "測試", con, dest=d))
        th.join(5)
        assert held, "測試沒有鎖到進度檔"
        assert p.read_bytes() == data and not (d / "h.bin.part").exists()
        assert srv.sent == len(data), (srv.sent, len(data))          # 每個位元組只下載一次
        assert len([1 for path, _ in srv.requests if path == "/data"]) == sr.CONNECTIONS, srv.requests
        assert time.time() - started < 20
    finally:
        srv.close()


def test_offline_folder_used_only_when_sha256_matches():
    """完整包的 offline\\：sha256 跟 manifest 一樣就直接用，一個位元組都不下載；不對、大小不對、沒有才照常下載。
    沒有 offline 資料夾（一般的 zip）時跟以前一樣。"""
    data = os.urandom(3 * 1024 * 1024 + 7)
    sha = hashlib.sha256(data).hexdigest()
    srv = FileServer(data)
    con = Quiet()
    old_offline = sr.OFFLINE
    try:
        def get(name, offline_bytes):
            srv.sent = 0
            srv.requests.clear()
            off = fresh("offline")
            if offline_bytes is not None:
                (off / name).write_bytes(offline_bytes)
            sr.OFFLINE = off
            d = fresh("dl-offline")
            item = {"name": name, "urls": [srv.url + "/data"], "sha256": sha.upper(), "size": len(data)}
            p = sr.download(item, "測試", con, dest=d)
            assert p == d / name and p.read_bytes() == data
            assert not list(d.glob("*.offline")) and not (d / (name + ".part")).exists()
            return off

        def run():
            # 1. sha256 對：直接用，不連線；offline 的檔案留著（硬連結或複製）
            off = get("a.zip", data)
            assert srv.sent == 0 and srv.requests == [], srv.requests
            assert (off / "a.zip").read_bytes() == data
            assert sr.offline_file({"name": "a.zip", "size": len(data)}) == off / "a.zip"
            # 2. 內容不對（大小一樣）：照常下載，offline 的壞檔不動
            bad = bytearray(data)
            bad[12345] ^= 0xFF
            off = get("b.zip", bytes(bad))
            assert srv.sent == len(data) and (off / "b.zip").read_bytes() == bytes(bad)
            # 3. 大小不對：不核對 sha256，照常下載
            get("c.zip", data[:-1])
            assert srv.sent == len(data)
            assert sr.offline_file({"name": "c.zip", "size": len(data)}) is None
            # 4. offline 資料夾裡沒有這個檔案
            get("d.zip", None)
            assert srv.sent == len(data)
            # 5. 沒有 offline 資料夾：跟以前一樣下載
            sr.OFFLINE = BASE / "no-such-offline"
            d = fresh("dl-offline")
            srv.sent = 0
            item = {"name": "e.zip", "urls": [srv.url + "/data"], "sha256": sha, "size": len(data)}
            assert sr.download(item, "測試", con, dest=d).read_bytes() == data and srv.sent == len(data)
            assert sr.offline_file(item) is None
        _with_fast_retries(run)
    finally:
        sr.OFFLINE = old_offline
        srv.close()
    # 專案本身（開發資料夾）沒有 offline：setup_runtime 預設找的是程式資料夾的 offline
    assert old_offline == sr.ROOT / "offline"


def test_native_messages_say_what_comes_from_offline():
    """完整包：畫面要說清楚哪些從 offline 拿、哪些 offline 的檔案壞了改下載、真的要下載多少。
    （2026-09-17 測試：ffmpeg、llama.cpp 都從 offline 拿，畫面卻寫「下載 VC++ 執行階段、ffmpeg、llama.cpp（2 MB）」；
    offline 的 ffmpeg 壞掉、實際下載 76 MB 時也一樣寫 2 MB，也沒說 offline 的檔案壞了。）"""
    old_offline, old_downloads = sr.OFFLINE, sr.DOWNLOADS
    vc, ff, ll, cu = os.urandom(2_100_000), os.urandom(3_000_000), os.urandom(1000), os.urandom(2000)

    def entry(name, data):
        return {"name": name, "urls": ["http://127.0.0.1:9/" + name], "sha256": hashlib.sha256(data).hexdigest(),
                "size": len(data)}
    natives = {"msvc": {"files": [entry("vc.whl", vc)]}, "ffmpeg": {"files": [entry("ff.zip", ff)]},
               "llama": {"files": [entry("ll.zip", ll), entry("cu.zip", cu)]}}
    labels = {"msvc": "VC++ 執行階段", "ffmpeg": "ffmpeg", "llama": "llama.cpp"}
    todo = ["msvc", "ffmpeg", "llama"]
    try:
        off = fresh("plan-offline")
        for name, data in (("ff.zip", ff), ("ll.zip", ll), ("cu.zip", cu)):
            (off / name).write_bytes(data)
        sr.OFFLINE = off
        # 1. ffmpeg、llama.cpp 都在 offline：只有 VC++（PyPI）要下載
        sr.DOWNLOADS = fresh("plan-dl")
        con = Quiet()
        assert sr.plan_natives(todo, natives, labels, con) == ["msvc"]
        assert con.lines == ["使用 offline 資料夾裡的 ffmpeg、llama.cpp", "下載 VC++ 執行階段（2 MB，在背景同時進行）"], con.lines
        # 放好的檔案下載時直接用（網址連不到也沒關係）
        assert sr.download(natives["ffmpeg"]["files"][0], "ffmpeg", con, dest=sr.DOWNLOADS).read_bytes() == ff
        # 2. offline 的 ffmpeg 壞了：說出來，要下載的大小算上 ffmpeg
        bad = bytearray(ff)
        bad[5] ^= 1
        (off / "ff.zip").write_bytes(bytes(bad))
        sr.DOWNLOADS = fresh("plan-dl")
        con = Quiet()
        assert sr.plan_natives(todo, natives, labels, con) == ["msvc", "ffmpeg"]
        assert con.lines == ["使用 offline 資料夾裡的 llama.cpp", "offline 資料夾裡的 ffmpeg 壞了，改從網路下載",
                             "下載 VC++ 執行階段、ffmpeg（5 MB，在背景同時進行）"], con.lines
        assert not (sr.DOWNLOADS / "ff.zip").exists()
        # 3. 沒有 offline 資料夾（一般的 zip）：跟以前一樣
        sr.OFFLINE = BASE / "no-such-offline"
        sr.DOWNLOADS = fresh("plan-dl")
        con = Quiet()
        assert sr.plan_natives(todo, natives, labels, con) == todo
        assert con.lines == ["下載 VC++ 執行階段、ffmpeg、llama.cpp（5 MB，在背景同時進行）"], con.lines
    finally:
        sr.OFFLINE, sr.DOWNLOADS = old_offline, old_downloads


def test_offline_wheel_skips_github_in_uv_sync():
    """完整包：uv.lock 裡放在 GitHub 上的 en-core-web-sm 從 offline 裝，uv sync 不下載它
    （2026-09-17 測試：12 MB 的它讓 uv sync 卡了 4 分 57 秒）。offline 壞了、沒有 offline 時照常交給 uv sync。"""
    root = fresh("wheel")
    old = _patch_paths(root)
    old_lock, old_offline, old_uv = sr.LOCK_PATH, sr.OFFLINE, sr.uv
    wheel = os.urandom(5000)
    sr.LOCK_PATH = root / "uv.lock"
    _fake_lock(sr.LOCK_PATH, wheel)
    site = sr.VENV / "Lib" / "site-packages"
    calls = []

    def installed(url):
        d = site / "en_core_web_sm-3.8.0.dist-info"
        d.mkdir(parents=True, exist_ok=True)
        (d / "RECORD").write_text("", encoding="utf-8")
        (d / "direct_url.json").write_text(json.dumps({"url": url}), encoding="utf-8")

    def fake_uv(args, con, key, label, retries=(10, 30, 90), env=None):
        calls.append([str(a) for a in args])
        if args[0] == "pip":
            installed(Path(args[-1]).resolve().as_uri())
        return []

    def run(offline_bytes, installed_url=None):
        calls.clear()
        shutil.rmtree(site, ignore_errors=True)
        site.mkdir(parents=True)
        if installed_url:
            installed(installed_url)
        off = fresh("wheel-offline")
        if offline_bytes is not None:
            (off / SM_WHEEL).write_bytes(offline_bytes)
        sr.OFFLINE = off
        con = Quiet()
        sr.sync_packages(con)
        return con.lines, off

    base = ["sync", "--frozen", "--inexact", "--no-install-package", "torch"]
    sr.uv = fake_uv
    try:
        assert [w["name"] for w in sr.lock_slow_wheels()] == [SM_WHEEL]       # PyPI、Linux 的 wheel 不算
        # 1. offline 的 sha256 對：uv sync 跳過它，再用 uv pip install 裝 offline 的檔案（--offline，不連網）
        lines, off = run(wheel)
        assert calls[0] == base + ["--no-install-package", "en-core-web-sm"], calls
        assert calls[1][0] == "pip" and "--offline" in calls[1] and "--no-deps" in calls[1], calls
        assert calls[1][-1] == str(off / SM_WHEEL) and len(calls) == 2, calls
        assert not any("壞了" in line for line in lines), lines
        # 2. offline 的檔案壞了：照常 uv sync（從 GitHub 下載），畫面說 offline 的檔案壞了
        lines, _ = run(wheel[:-1] + b"x")
        assert calls == [base], calls
        assert "offline 資料夾裡的 en-core-web-sm 壞了，改從網路下載" in lines, lines
        # 3. 沒有 offline（一般的 zip）：跟以前一樣
        run(None)
        assert calls == [base], calls
        # 4. 之前從 offline 裝好了、後來 offline 資料夾刪掉（uv.lock 改版要重新 sync）：不要讓 uv sync 從 GitHub 重新下載
        run(None, installed_url=(root / "old" / SM_WHEEL).as_uri())
        assert calls == [base + ["--no-install-package", "en-core-web-sm"]], calls
        # 5. 一般的 zip 從 GitHub 裝好的：跟以前一樣交給 uv sync
        run(None, installed_url="https://github.com/explosion/spacy-models/releases/download/x/" + SM_WHEEL)
        assert calls == [base], calls
    finally:
        sr.LOCK_PATH, sr.OFFLINE, sr.uv = old_lock, old_offline, old_uv
        _restore(old)
    # offline_complete：manifest 和 uv.lock 裡 GitHub 上的檔案都在（大小也對）才算齊全
    manifest = {"ffmpeg": {"files": [{"name": "ff.zip", "urls": ["https://github.com/x/ff.zip"], "sha256": "a" * 64,
                                      "size": 3}]},
                "uv": {"files": [{"name": "uv.whl", "urls": ["https://files.pythonhosted.org/uv.whl"], "sha256": "b" * 64,
                                  "size": 1}]}}
    off = fresh("complete")
    lock = root / "uv.lock"
    assert not sr.offline_complete(manifest, off, lock)
    (off / "ff.zip").write_bytes(b"abc")
    assert not sr.offline_complete(manifest, off, lock)
    (off / SM_WHEEL).write_bytes(wheel)
    assert sr.offline_complete(manifest, off, lock)
    (off / "ff.zip").write_bytes(b"ab")
    assert not sr.offline_complete(manifest, off, lock)


def test_uv_disk_full_stops_without_retries_and_os_errors_are_explained():
    calls = []

    def fake_run(cmd, con, key, label, timeout=None, cwd=None, env=None):
        calls.append(cmd)
        return 2, ["error: Failed to install: torch", "Caused by: failed to write to file `x`: 磁碟空間不足。 (os error 112)"]

    old_run, old_sleep = sr.run_logged, sr.time.sleep
    sr.run_logged = fake_run
    sr.time.sleep = lambda s: calls.append(("sleep", s))
    try:
        sr.uv(["sync"], Quiet(), "sync", "Python 套件")
    except sr.SetupError as e:
        assert "磁碟空間不夠" in str(e) and "網路" not in str(e), e
    else:
        raise AssertionError("磁碟滿了應該失敗")
    finally:
        sr.run_logged, sr.time.sleep = old_run, old_sleep
    assert len(calls) == 1, calls                                        # 不重試、不等
    assert "磁碟空間不夠" in sr.os_problem(OSError(None, "磁碟空間不足。", None, 112))
    assert "鎖住" in sr.os_problem(OSError(None, "being used", None, 32))
    assert "沒辦法寫入" in sr.os_problem(PermissionError(13, "denied"))
    assert sr.os_problem(ValueError("x")) is None
    # 解壓縮時磁碟滿了（OSError 不是 SetupError）：main 印出白話的原因，不是「沒預料到的錯誤」
    import contextlib
    import io
    old_manifest = sr.load_manifest

    def boom():
        raise OSError(None, "磁碟空間不足。", None, 112)
    sr.load_manifest = boom
    try:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = sr.main([])
    finally:
        sr.load_manifest = old_manifest
    assert code == 1 and "磁碟空間不夠" in out.getvalue() and "沒預料到" not in out.getvalue(), out.getvalue()


def test_remaining_bytes_counts_partial_downloads():
    d = fresh("remain")
    item = {"name": "x.zip", "sha256": "AB" * 32, "size": 1000}
    assert sr.remaining_bytes(item, d) == 1000
    (d / "x.zip.part").write_bytes(b"\0" * 1000)
    sr.write_json(d / "x.zip.part.json", {"size": 1000, "sha256": "ab" * 32, "segments": [[0, 499, 300], [500, 999, 500]]})
    assert sr.remaining_bytes(item, d) == 200
    # 大小不知道（torch 在 uv.lock 裡沒寫大小）：用 .part.json 記下的大小
    assert sr.download_status({**item, "size": None}, d) == (800, 1000)
    assert sr.download_status({**item, "name": "y.zip", "size": None}, d) == (0, None)
    (d / "x.zip").write_bytes(b"x")
    assert sr.remaining_bytes(item, d) == 0


def test_repair_reset_refuses_while_something_runs_from_runtime():
    import contextlib
    import io
    import subprocess
    root = fresh("repair")
    old = _patch_paths(root)
    try:
        (sr.VENV / "Scripts").mkdir(parents=True)
        (sr.VENV / "pyvenv.cfg").write_text("home = x\n", encoding="utf-8")
        sr.STATE_PATH.write_text("{}", encoding="utf-8")
        exe = sr.RT / "bin" / "fake" / "ping.exe"          # 代替 runtime 裡還在跑的伺服器、llama-server
        exe.parent.mkdir(parents=True)
        shutil.copyfile(Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "ping.exe", exe)
        proc = subprocess.Popen([str(exe), "-n", "60", "127.0.0.1"], stdout=subprocess.DEVNULL,
                                creationflags=subprocess.CREATE_NO_WINDOW)
        try:
            end = time.time() + 10
            while time.time() < end and not any(pid == proc.pid for pid, _ in sr.processes_using(sr.RT)):
                time.sleep(0.1)
            assert any(pid == proc.pid for pid, _ in sr.processes_using(sr.RT))
            assert not any(pid == proc.pid for pid, _ in sr.processes_using(root / "elsewhere"))
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = sr.repair_reset()
            assert code == 2 and "ping.exe" in out.getvalue(), (code, out.getvalue())
            assert (sr.VENV / "pyvenv.cfg").is_file() and sr.STATE_PATH.is_file()     # 什麼都沒刪
        finally:
            proc.kill()
            proc.wait(10)
        with contextlib.redirect_stdout(io.StringIO()):
            assert sr.repair_reset() == 0
        assert not sr.VENV.exists() and not sr.STATE_PATH.exists()
    finally:
        _restore(old)


def test_start_bat_stops_on_non_english_path_before_anything_is_created():
    """路徑有中文、é 這類字元：start.bat 在下載 uv、Python 以前就停下（以前要先下載 38 MB 才由 precheck 擋，
    é 還會讓 tar.exe 失敗、顯示成網路中斷）。只複製 start.bat 和訊息檔，沒有 get_uv.cmd，就算判斷失效也不會下載。"""
    import subprocess
    base = fresh("b")
    for name in ("中文 (測)", "café"):
        d = base / name / "V"
        if len(str(d)) > 80:        # 先被「路徑太長」擋下，測不到這一項
            print(f"  （暫存資料夾路徑太長，跳過 {name}）")
            continue
        (d / "launcher" / "msg").mkdir(parents=True)
        shutil.copyfile(ROOT / "start.bat", d / "start.bat")
        for m in (ROOT / "launcher" / "msg").glob("*.txt"):
            shutil.copyfile(m, d / "launcher" / "msg" / m.name)
        env = {**os.environ, "VS_TEST_ALLOW_TEMP": "1"}
        info = subprocess.STARTUPINFO()
        info.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        info.wShowWindow = 0
        # 要有主控台 chcp 才能用：開一個隱藏的新主控台
        out = subprocess.run(f'cmd.exe /d /s /c ""{d / "start.bat"}""', stdin=subprocess.DEVNULL, capture_output=True,
                             env=env, timeout=60, startupinfo=info, creationflags=subprocess.CREATE_NEW_CONSOLE)
        text = out.stdout.decode("utf-8", errors="replace")
        assert out.returncode == 1 and "非英文字元" in text, (name, out.returncode, text)
        assert "按任意鍵關閉這個視窗" in text and "Press any key" not in text, text
        assert not (d / "runtime").exists(), name


# ---------- 搬家、快速路徑、解壓縮 ----------

def _patch_paths(root: Path):
    old = {k: getattr(sr, k) for k in ("ROOT", "RT", "VENV", "VENV_PY", "BIN", "STATE_PATH")}
    sr.ROOT, sr.RT = root, root / "runtime"
    sr.VENV = sr.RT / "venv"
    sr.VENV_PY = sr.VENV / "Scripts" / "python.exe"
    sr.BIN = sr.RT / "bin"
    sr.STATE_PATH = sr.RT / "install-state.json"
    return old


def _restore(old):
    for k, v in old.items():
        setattr(sr, k, v)


def test_fix_pyvenv_cfg_after_move():
    root = fresh("moved 資料夾 (新)")
    old = _patch_paths(root)
    try:
        manifest = {"python": {"version": "3.12.14", "dir": "cpython-3.12.14-windows-x86_64-none"}}
        cfg = sr.VENV / "pyvenv.cfg"
        cfg.parent.mkdir(parents=True)
        cfg.write_text("home = D:\\old place\\runtime\\python\\cpython-3.12.14-windows-x86_64-none\n"
                       "include-system-site-packages = false\nversion = 3.12.14\n"
                       "executable = D:\\old place\\runtime\\python\\cpython-3.12.14-windows-x86_64-none\\python.exe\n"
                       "command = D:\\old\\python.exe -m venv D:\\old\\runtime\\venv\n", encoding="utf-8")
        assert sr.fix_pyvenv_cfg(manifest) is True
        text = cfg.read_text(encoding="utf-8")
        base = root / "runtime" / "python" / "cpython-3.12.14-windows-x86_64-none"
        assert f"home = {base}\n" in text and f"executable = {base}\\python.exe" in text
        assert "include-system-site-packages = false" in text and "old place" not in text
        assert sr.fix_pyvenv_cfg(manifest) is False     # 第二次沒有要改的
    finally:
        _restore(old)


def test_fast_path_requires_same_root_fingerprint_and_files():
    root = fresh("fast")
    old = _patch_paths(root)
    try:
        manifest = {"python": {"version": "3.12.14", "dir": "cpython-3.12.14-windows-x86_64-none"}}
        fp = {"uv_lock": "a", "manifest": "b", "python": "3.12.14"}
        for rel in ("runtime/venv/Scripts/python.exe", "runtime/python/cpython-3.12.14-windows-x86_64-none/python.exe",
                    "runtime/bin/ff/ffmpeg.exe", "runtime/bin/ff/ffprobe.exe", "runtime/bin/ll/llama-server.exe",
                    *(f"runtime/venv/{m}" for m in sr.VENV_MARKERS)):
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            (root / rel).write_bytes(b"x")
        state = {"state_version": sr.STATE_VERSION, "root": str(root), "fingerprint": fp, "tests": {"ok": True},
                 "ffmpeg": {"path": "runtime\\bin\\ff\\ffmpeg.exe"}, "ffprobe": {"path": "runtime\\bin\\ff\\ffprobe.exe"},
                 "llama_server": {"path": "runtime\\bin\\ll\\llama-server.exe"}}
        assert sr.fast_ok(state, fp, manifest)
        # venv 被刪到一半（程式開著時刪 venv 只剩 python.exe、pyvenv.cfg 不見）：不能走快速路徑直接啟動
        for m in ("pyvenv.cfg", "Lib/site-packages/torch/__init__.py"):
            (sr.VENV / m).rename(sr.VENV / (m + ".gone"))
            assert not sr.fast_ok(state, fp, manifest), m
            (sr.VENV / (m + ".gone")).rename(sr.VENV / m)
        assert sr.fast_ok(state, fp, manifest)
        assert not sr.fast_ok({**state, "root": "D:\\elsewhere"}, fp, manifest)
        assert not sr.fast_ok(state, {**fp, "uv_lock": "changed"}, manifest)
        assert not sr.fast_ok({**state, "tests": {"ok": False}}, fp, manifest)
        (root / "runtime/bin/ll/llama-server.exe").unlink()
        assert not sr.fast_ok(state, fp, manifest)
        # yt-dlp 上次沒裝好：一小時內不重試（離線時不要每次啟動都卡住），之後才補裝
        assert not sr.ytdlp_retry_due({"yt_dlp": {"ok": True}})
        assert not sr.ytdlp_retry_due({"yt_dlp": {"ok": False, "tried_at": time.time() - 60}})
        assert sr.ytdlp_retry_due({"yt_dlp": {"ok": False, "tried_at": time.time() - 7200}})
    finally:
        _restore(old)


def test_extract_keeps_only_listed_files_and_blocks_traversal():
    root = fresh("extract")
    arc = root / "a.zip"
    with zipfile.ZipFile(arc, "w") as z:
        z.writestr("ffmpeg-n8/bin/ffmpeg.exe", b"1")
        z.writestr("ffmpeg-n8/bin/avcodec-62.dll", b"2")
        z.writestr("ffmpeg-n8/bin/ffplay.exe", b"3")
        z.writestr("ffmpeg-n8/doc/x.html", b"4")
        z.writestr("ffmpeg-n8/LICENSE.txt", b"5")
        z.writestr("ffmpeg-n8/../../evil.dll", b"6")
    out = root / "out"
    sr._extract(arc, out, ["bin/ffmpeg.exe", "bin/*.dll", "LICENSE.txt"], strip_top=True)
    got = sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file())
    assert got == ["LICENSE.txt", "bin/avcodec-62.dll", "bin/ffmpeg.exe"], got
    assert not (root / "evil.dll").exists() and not (BASE / "evil.dll").exists()


def test_file_version_reads_python_dll():
    v = sr.file_version(Path(sys.base_prefix) / "python312.dll") or sr.file_version(Path(sys.executable))
    assert v is None or (isinstance(v, tuple) and len(v) == 4)
    vc = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "vcruntime140.dll"
    if vc.is_file():
        assert sr.file_version(vc)[0] == 14


# ---------- precheck ----------

def _items(fn, *args):
    items = []
    fn(items, *args)
    return {i["id"]: i for i in items}


def test_precheck_gpu_rules():
    cases = [
        (None, "fail", "沒有偵測到 NVIDIA"),
        ({"name": "GTX 1080", "driver": "595.79", "cuda_version": 13020, "arch": [6, 1], "total_mb": 8192}, "fail", "太舊"),
        ({"name": "RTX 3060", "driver": "552.22", "cuda_version": 12040, "arch": [8, 6], "total_mb": 12288}, "fail", "驅動太舊"),
        ({"name": "RTX 2060", "driver": "595.79", "cuda_version": 13020, "arch": [7, 5], "total_mb": 6144}, "warn", "RTX 20"),
        ({"name": "RTX 4060", "driver": "581.10", "cuda_version": 13000, "arch": [8, 9], "total_mb": 8188}, "warn", "595"),
        ({"name": "RTX 4070", "driver": "595.79", "cuda_version": 13020, "arch": [8, 9], "total_mb": 12282}, "ok", ""),
    ]
    try:
        for fake, status, words in cases:
            syscheck.set_fake(fake)
            got = _items(precheck.check_gpu)["gpu"]
            assert got["status"] == status and words in got["message"], (fake, got)
    finally:
        syscheck.set_fake()


def test_precheck_location_rules():
    old_root, old_rt = precheck.ROOT, precheck.RT
    saved = {k: os.environ.get(k) for k in ("OneDrive", "VS_ORIG_TEMP", "VS_TEST_ALLOW_TEMP")}
    tmp = fresh("loc")
    try:
        def check(path: str, **env):
            for k in saved:
                os.environ.pop(k, None)
            os.environ.update(env)
            precheck.ROOT = Path(path)
            precheck.RT = tmp / "runtime"
            return _items(precheck.check_location, False)

        long_path = "E:\\" + "a" * 78
        assert len(long_path) == 81 and check(long_path)["path"]["status"] == "fail"
        assert check("E:\\" + "a" * 77)["path"]["status"] == "ok"
        assert check("E:\\Video%Sub")["path"]["status"] == "fail"
        assert check("E:\\a;b")["path"]["status"] == "fail"
        assert check("E:\\a&b")["path"]["status"] == "fail"
        assert "OneDrive" in check("C:\\Users\\x\\OneDrive\\BilingualSubtitles")["path"]["message"]
        assert "OneDrive" in check("C:\\Users\\x\\Docs\\BilingualSubtitles", OneDrive="C:\\Users\\x\\Docs")["path"]["message"]
        assert "暫存" in check("C:\\Users\\x\\AppData\\Local\\Temp\\Temp1_BilingualSubtitles.zip\\BilingualSubtitles")["path"]["message"]
        assert "暫存" in check("E:\\tmpdir\\BilingualSubtitles", VS_ORIG_TEMP="E:\\tmpdir")["path"]["message"]
        assert check("C:\\Users\\x\\AppData\\Local\\Temp\\t\\BilingualSubtitles", VS_TEST_ALLOW_TEMP="1")["path"]["status"] == "ok"
        # 中文路徑：nagisa（DyNet）讀不到模型檔，擋下；空白、括號可以
        assert "中文" in check("E:\\影片 字幕 (測試)")["path"]["message"]
        assert check("E:\\Video Sub (test)")["path"]["status"] == "ok"
        assert check("E:\\Video Sub (test)")["writable"]["status"] == "ok"
    finally:
        precheck.ROOT, precheck.RT = old_root, old_rt
        for k, v in saved.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v


def test_precheck_skips_github_when_offline_has_the_runtime_files():
    """完整包的 offline 資料夾有 GitHub 上的執行環境檔時，precheck 不連 GitHub，GitHub 連不到也照常安裝
    （2026-09-17 測試：只讓 GitHub 連不到，完整包的 precheck 結束代碼 1「連不到 github」）。"""
    urls = []
    old_reachable, old_root = precheck._reachable, precheck.ROOT

    def fake_reachable(url, timeout=10):
        urls.append(url)
        return "github" not in url
    root = fresh("precheck-offline")
    (root / "launcher").mkdir()
    wheel = os.urandom(3000)
    _fake_lock(root / "uv.lock", wheel)
    ff = os.urandom(1000)
    manifest = {"ffmpeg": {"files": [{"name": "ff.zip", "urls": ["https://github.com/x/ff.zip"],
                                      "sha256": hashlib.sha256(ff).hexdigest(), "size": len(ff)}]}}
    (root / "launcher" / "runtime-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    precheck._reachable = fake_reachable
    try:
        precheck.ROOT = root
        # 一般的 zip（沒有 offline）：要檢查 GitHub，連不到就不能繼續
        assert precheck.github_needed()
        got = _items(precheck.check_network, precheck.github_needed())["network"]
        assert got["status"] == "fail" and "github" in got["message"] and any("github" in u for u in urls), got
        # offline 只有一部分：還是要連
        (root / "offline").mkdir()
        (root / "offline" / "ff.zip").write_bytes(ff)
        assert precheck.github_needed()
        # 完整包：不連 GitHub，GitHub 連不到也照常繼續
        (root / "offline" / SM_WHEEL).write_bytes(wheel)
        urls.clear()
        assert not precheck.github_needed()
        got = _items(precheck.check_network, precheck.github_needed())["network"]
        assert got["status"] == "ok" and not any("github.com" in u for u in urls), (got, urls)
        assert "github 不用連" in got["value"], got
        # pypi 連不到時一樣擋下
        precheck._reachable = lambda url, timeout=10: "pypi" not in url and "github" not in url
        got = _items(precheck.check_network, False)["network"]
        assert got["status"] == "fail" and "pypi" in got["message"] and "github" not in got["message"], got
        # 開發資料夾沒有 offline：要連
        precheck.ROOT = ROOT
        assert precheck.github_needed()
    finally:
        precheck._reachable, precheck.ROOT = old_reachable, old_root


def test_start_bat_pauses_when_the_server_crashes_even_with_negative_exit_codes():
    """伺服器的結束代碼不是 0 就停下來顯示中文提示。負的結束代碼（程式當掉的 0xC0000005、工作管理員的 -1）以前被
    「if errorlevel 1」漏掉，視窗直接關掉（2026-09-17 測試發現）。Ctrl+C（0xC000013A）是正常關掉，不停。
    用開發模式（.dev）跑一個假的 app.server，一般模式是同樣的判斷。"""
    import subprocess
    d = fresh("crash")
    shutil.copyfile(ROOT / "start.bat", d / "start.bat")
    (d / ".dev").write_text("", encoding="utf-8")
    (d / "launcher" / "msg").mkdir(parents=True)
    for m in (ROOT / "launcher" / "msg").glob("*.txt"):
        shutil.copyfile(m, d / "launcher" / "msg" / m.name)
    (d / "app").mkdir()
    (d / "app" / "__init__.py").write_text("", encoding="utf-8")
    (d / "app" / "server.py").write_text("import os, sys\nprint('fake server', flush=True)\n"
                                         "sys.exit(int(os.environ['VS_FAKE_EXIT']))\n", encoding="utf-8")
    info = subprocess.STARTUPINFO()
    info.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    info.wShowWindow = 0
    for code, pauses in ((0, False), (1, True), (-1073741819, True), (-1, True), (-1073741510, False)):
        env = {**os.environ, "VS_PYTHON": sys.executable, "VS_FAKE_EXIT": str(code)}
        out = subprocess.run(f'cmd.exe /d /s /c ""{d / "start.bat"}""', stdin=subprocess.DEVNULL, capture_output=True,
                             env=env, timeout=60, startupinfo=info, creationflags=subprocess.CREATE_NEW_CONSOLE)
        text = out.stdout.decode("utf-8", errors="replace")
        assert "fake server" in text, (code, text, out.stderr)
        assert ("按任意鍵關閉這個視窗" in text) == pauses and "Press any key" not in text, (code, text)
        assert out.returncode == (1 if pauses else 0), (code, out.returncode, text)
    start = (ROOT / "start.bat").read_text(encoding="utf-8")
    assert "goto server_failed" in start and "goto dev_stopped" in start
    assert "if errorlevel 1 goto server_failed" not in start and "if errorlevel 1 goto dev_stopped" not in start
    # precheck、setup_runtime 當掉或被結束（負的結束代碼）時也要停下，不能接著啟動沒裝好的伺服器
    lines = start.splitlines()
    for script in ("precheck.py", "setup_runtime.py"):
        i = next(i for i, line in enumerate(lines) if line.startswith('"%BASEPY%"') and script in line)
        assert lines[i + 1] == 'if not "%errorlevel%"=="0" goto fail_pause', (script, lines[i + 1])


def test_precheck_runs_quickly_on_this_machine():
    t = time.time()
    result = precheck.run()
    ids = {i["id"] for i in result["items"]}
    assert {"os", "path", "filesystem", "writable", "disk", "gpu", "memory", "sac"} <= ids, ids
    if result["installed"]:
        assert time.time() - t < 2


if __name__ == "__main__":
    started = time.time()
    try:
        for name, fn in list(globals().items()):
            if name.startswith("test_") and callable(fn):
                fn()
                print("ok", name)
    finally:
        shutil.rmtree(BASE, ignore_errors=True)
    assert PROJECT_RUNTIME_EXISTED or not (ROOT / "runtime").exists(), "測試在專案裡建立了 runtime 資料夾"
    print(f"all passed in {time.time() - started:.1f}s")
