"""模型下載改造（打包計畫 P0-10、P1-15、P1-17）的測試：續傳、中斷、hash 不符、錯誤代碼、完成標記、
版本（量化）的安裝判斷和刪除、暫停和繼續、自動重試、下載有自己的工作執行緒、擁有者現有模型的判斷不變。

不連網、不用顯卡：下載的來源是這個測試在 127.0.0.1 開的假 HTTP 伺服器；模型和資料庫都放在系統暫存資料夾的
model-download-test 底下，測完整個刪掉。會唯讀地看真正的 models 資料夾（確認擁有者裝好的模型還是被認成已安裝），
不在裡面寫入或刪除任何東西。

python -s tests/test_model_download.py
"""
import hashlib
import http.server
import json
import logging
import os
import re
import shutil
import ssl
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
from pathlib import Path

os.environ["VS_NO_WORKERS"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import config, db, jobs, model_download as md, models, safepath, server, settings, translate  # noqa: E402

BASE = Path(tempfile.gettempdir()) / "model-download-test"
MIB = 1024 * 1024
REAL_MODELS = config.MODELS_DIR


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ---------- 假的下載伺服器 ----------

class FakeFiles(http.server.ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self):
        super().__init__(("127.0.0.1", 0), _Handler)
        self.files = {}          # 路徑 → {"data", "range", "cut", "status", "delay", "retry_after"}
        self.log = []            # (路徑, Range 標頭, 狀態碼, 這次送出的位元組數)
        self.thread = threading.Thread(target=self.serve_forever, daemon=True)
        self.thread.start()

    def url(self, path: str) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}{path}"

    def add(self, path: str, data: bytes, **opts):
        self.files[path] = {"data": data, **opts}
        return self.url(path)

    def ranges(self, path: str) -> list:
        return [r for p, r, _s, _n in self.log if p == path]

    def close(self):
        self.shutdown()
        self.server_close()


class _Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def do_GET(self):
        srv: FakeFiles = self.server
        cfg = srv.files.get(self.path.split("?")[0])
        rng = self.headers.get("Range")
        if cfg is None:
            return self._status(404, rng)
        statuses = cfg.get("status") or []
        if statuses:
            code = statuses.pop(0)
            if code != 200:
                return self._status(code, rng, cfg.get("retry_after"))
        data = cfg["data"]
        start = 0
        m = re.match(r"bytes=(\d+)-$", rng or "")
        if m and cfg.get("range", True):
            start = int(m.group(1))
            if start >= len(data):
                return self._status(416, rng)
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{len(data) - 1}/{len(data)}")
        else:
            self.send_response(200)
        body = data[start:]
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        cuts = cfg.get("cut") or []
        cut = cuts.pop(0) if cuts else None       # 這次只送這麼多就斷線
        limit = len(body) if cut is None else min(cut, len(body))
        sent = 0
        try:
            while sent < limit:
                chunk = body[sent:min(limit, sent + 64 * 1024)]
                self.wfile.write(chunk)
                sent += len(chunk)
                if cfg.get("delay"):
                    time.sleep(cfg["delay"])
        except (ConnectionError, OSError):
            pass
        srv.log.append((self.path, rng, 206 if start else 200, sent))
        if cut is not None:
            self.close_connection = True

    def do_HEAD(self):
        # 像 Hugging Face 的 resolve 網址：轉址前帶 X-Linked-ETag（遠端檔案現在的 sha256）；不記進 log
        cfg = self.server.files.get(self.path.split("?")[0])
        if cfg is not None and cfg.get("etag"):
            self.send_response(302)
            self.send_header("Location", "/cdn" + self.path)
            self.send_header("X-Linked-ETag", f'"{cfg["etag"]}"')
        else:
            self.send_response(404 if cfg is None else 200)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _status(self, code, rng, retry_after=None):
        self.server.log.append((self.path, rng, code, 0))
        self.send_response(code)
        if retry_after:
            self.send_header("Retry-After", str(retry_after))
        cfg = self.server.files.get(self.path.split("?")[0]) or {}
        for k, v in (cfg.get("headers") or {}).items():
            self.send_header(k, v)
        self.send_header("Content-Length", "0")
        self.end_headers()


# ---------- 暫存資料夾、資料庫 ----------

class Env:
    def __init__(self):
        BASE.mkdir(parents=True, exist_ok=True)
        self.tmp = Path(tempfile.mkdtemp(prefix="vs-dl-", dir=BASE))
        self.models = self.tmp / "models"
        self.models.mkdir()
        self.work = self.tmp / "work"
        self.work.mkdir()
        safepath.TEST_ROOTS.append(self.tmp)
        self._saved = []
        self._saved_conn = db._conn
        self.patch(db, "DB_PATH", self.tmp / "test.db")
        db.init()
        self.patch(models, "MODELS_DIR", self.models)
        self.patch(models, "RETRY_WAITS", (0.05,))
        self.patch(jobs, "WORK_DIR", self.work)
        self.patch(md, "QUICK_WAITS", ())
        self.added = []
        self.srv = FakeFiles()
        self._levels = []
        for name in ("models", "safepath", "jobs"):
            logger = logging.getLogger(name)
            self._levels.append((logger, logger.level))
            logger.setLevel(logging.CRITICAL)

    def patch(self, obj, name, value):
        self._saved.append((obj, name, getattr(obj, name)))
        setattr(obj, name, value)

    def add_model(self, mid: str, entry: dict) -> dict:
        """假的型錄項目加進共用的 MODEL_CATALOG（config、jobs、models、settings 都是同一個 dict），測完拿掉。"""
        entry = {"id": mid, "label": mid, "langs": ["ja"], "note": "", "repo": "fake/repo", **entry}
        for v in (entry.get("variants") or {}).values():
            v["size_mb"] = v["bytes"] // MIB
        if entry.get("variants"):
            entry.setdefault("default_variant", next(iter(entry["variants"])))
            entry["size_mb"] = entry["variants"][entry["default_variant"]]["size_mb"]
        assert mid not in config.MODEL_CATALOG
        config.MODEL_CATALOG[mid] = entry
        self.added.append(mid)
        return entry

    def close(self):
        for mid in self.added:
            config.MODEL_CATALOG.pop(mid, None)
        self.srv.close()
        for obj, name, value in reversed(self._saved):
            setattr(obj, name, value)
        for logger, level in self._levels:
            logger.setLevel(level)
        db._conn.close()
        db._conn = self._saved_conn
        safepath.TEST_ROOTS.remove(self.tmp)
        shutil.rmtree(self.tmp, ignore_errors=True)


def run(fn):
    env = Env()
    try:
        fn(env)
    finally:
        env.close()


def expect_download_error(code, fn, *args, **kwargs) -> md.DownloadError:
    try:
        fn(*args, **kwargs)
    except md.DownloadError as e:
        assert e.code == code, (e.code, str(e))
        return e
    raise AssertionError(f"預期錯誤代碼 {code}")


# ---------- 單一檔案：續傳、中斷、hash ----------

def test_resume_uses_range_from_part():
    def body(env: Env):
        data = os.urandom(3 * MIB + 123)
        url = env.srv.add("/a.gguf", data)
        dst = env.models / "a.gguf"
        (env.models / "a.gguf.part").write_bytes(data[:MIB + 7])        # 上次下載到一半
        got = md.fetch_file([url], dst, len(data), sha(data))
        assert got == {"size": len(data), "sha256": sha(data)}, got
        assert dst.read_bytes() == data and not (env.models / "a.gguf.part").exists()
        assert env.srv.log == [("/a.gguf", f"bytes={MIB + 7}-", 206, len(data) - MIB - 7)], env.srv.log
    run(body)


def test_interrupted_download_keeps_part_and_continues_where_it_stopped():
    def body(env: Env):
        data = os.urandom(2 * MIB + 5)
        url = env.srv.add("/b.gguf", data, cut=[MIB + 300])               # 第一次送到一半就斷線
        dst = env.models / "b.gguf"
        err = expect_download_error("network", md.fetch_file, [url], dst, len(data), sha(data))
        assert "已下載的部分會保留" in str(err), err
        part = env.models / "b.gguf.part"
        assert part.stat().st_size == MIB + 300 and not dst.exists()
        md.fetch_file([url], dst, len(data), sha(data))                   # 重開：從斷掉的地方接著下載
        assert dst.read_bytes() == data
        assert env.srv.ranges("/b.gguf") == [None, f"bytes={MIB + 300}-"], env.srv.log
        # 同一次下載裡斷線、但有收到資料：馬上重連接著下載，不用等使用者按重試
        env.patch(md, "QUICK_WAITS", (0, 0))
        url2 = env.srv.add("/c.gguf", data, cut=[700 * 1024])
        md.fetch_file([url2], env.models / "c.gguf", len(data), sha(data))
        assert (env.models / "c.gguf").read_bytes() == data
        assert env.srv.ranges("/c.gguf") == [None, f"bytes={700 * 1024}-"], env.srv.log
    run(body)


def test_hash_mismatch_deletes_part():
    def body(env: Env):
        data = os.urandom(MIB)
        bad = bytearray(data)
        bad[1234] ^= 0xFF
        url = env.srv.add("/d.gguf", bytes(bad))
        dst = env.models / "d.gguf"
        err = expect_download_error("corrupt", md.fetch_file, [url], dst, len(data), sha(data))
        assert "重新下載" in str(err)
        assert not dst.exists() and not (env.models / "d.gguf.part").exists()
        # 接著下載時 .part 裡原本的部分也要算進 sha256：前半段壞掉的，下載完一樣抓得到
        (env.models / "d.gguf.part").write_bytes(bytes(bad[:MIB // 2]))
        env.srv.add("/d.gguf", data)
        expect_download_error("corrupt", md.fetch_file, [url], dst, len(data), sha(data))
        assert not dst.exists()
    run(body)


def test_server_without_range_support_starts_over():
    def body(env: Env):
        data = os.urandom(MIB + 11)
        url = env.srv.add("/e.bin", data, range=False)
        (env.models / "e.bin.part").write_bytes(b"x" * 5000)
        md.fetch_file([url], env.models / "e.bin", len(data), sha(data))
        assert (env.models / "e.bin").read_bytes() == data
    run(body)


def test_http_errors_have_codes_and_mirrors_are_tried():
    def body(env: Env):
        data = os.urandom(2048)
        dst = env.models / "f.bin"
        e = expect_download_error("not_found", md.fetch_file, [env.srv.url("/missing")], dst, len(data))
        assert "找不到" in str(e) and "下載來源上找不到" in str(e), e
        # 要登入同意授權：Hugging Face 用 X-Error-Code: GatedRepo 標明
        env.srv.add("/gated", data, status=[401], headers={"X-Error-Code": "GatedRepo"})
        assert "登入 Hugging Face" in str(expect_download_error("gated", md.fetch_file, [env.srv.url("/gated")], dst))
        # 沒有 GatedRepo 的 403（GitHub、鏡像拒絕或限流）：不叫人登入 Hugging Face，當成暫時的連線問題（會自動重試）
        env.srv.add("/forbidden", data, status=[403])
        e = expect_download_error("network", md.fetch_file, [env.srv.url("/forbidden")], dst)
        assert "HTTP 403" in str(e) and "Hugging Face" not in str(e), e
        e = md.http_error(403, None, "https://github.com/TRvlvr/model_repo/releases/download/v1/x.ckpt")
        assert e.code == "network" and "登入" not in str(e), e
        # Hugging Face 上 repo 改名、下架：回 401 而且沒有 X-Error-Code（2026-09-17 實測），是「找不到」不是「要登入」
        e = md.http_error(401, {"X-Error-Message": "Invalid username or password."},
                          "https://huggingface.co/tencent/renamed/resolve/main/a.gguf")
        assert e.code == "not_found" and "改名或下架" in str(e), e
        assert md.http_error(404, {"X-Error-Code": "EntryNotFound"}, "https://huggingface.co/x/y").code == "not_found"
        env.srv.add("/limit", data, status=[429], retry_after=7)
        assert expect_download_error("rate_limit", md.fetch_file, [env.srv.url("/limit")], dst).retry_after == 7
        env.srv.add("/busy", data, status=[503])
        e = expect_download_error("network", md.fetch_file, [env.srv.url("/busy")], dst)
        assert "HTTP 503" in str(e), e
        # 第一個網址（鏡像）找不到，換下一個
        good = env.srv.add("/good", data)
        md.fetch_file([env.srv.url("/missing"), good], dst, len(data), sha(data))
        assert dst.read_bytes() == data
        # 遠端檔案大小跟型錄不同：程式記錄過期，不重試
        env.srv.add("/changed", os.urandom(4096))
        expect_download_error("changed", md.fetch_file, [env.srv.url("/changed")], env.models / "g.bin", 2048)
        # 連不上
        port = env.srv.server_address[1]
        env.srv.close()
        e = expect_download_error("network", md.fetch_file, [f"http://127.0.0.1:{port}/x"], env.models / "h.bin")
        assert "連不到下載來源" in str(e), e
        env.srv = FakeFiles()
        assert "連不到 Hugging Face。" in md.error_text("network")
    run(body)


def test_classify_os_and_ssl_errors():
    def oserr(winerror, cls=OSError):
        e = cls(0, "x")
        e.winerror = winerror
        return e
    assert md.classify(oserr(112)).code == "disk_full"
    assert md.classify(OSError(28, "No space left on device")).code == "disk_full"
    assert md.classify(oserr(5, PermissionError)).code == "no_permission"
    assert md.classify(oserr(32, PermissionError)).code == "file_locked"
    assert md.classify(TimeoutError("timed out")).code == "timeout"
    assert md.classify(urllib.error.URLError(ssl.SSLCertVerificationError(1, "certificate verify failed"))).code == "ssl"
    assert md.classify(urllib.error.URLError(ConnectionRefusedError(10061, "refused"))).code == "network"
    assert md.classify(ConnectionResetError(10054, "reset")).code == "network"

    class FakeResponse:
        status_code = 404
        headers = {}

    class HubError(Exception):
        response = FakeResponse()
    assert md.classify(HubError("Entry Not Found")).code == "not_found"
    assert md.classify(ValueError("??")).code == "unknown"

    # huggingface_hub 查檔案清單時的例外：GatedRepoError 才是要登入；repo 不存在（401）是找不到
    class RepositoryNotFoundError(Exception):
        response = type("R", (), {"status_code": 401, "headers": {}})()

    class GatedRepoError(RepositoryNotFoundError):
        pass
    assert md.classify(RepositoryNotFoundError("401 Client Error")).code == "not_found"
    assert md.classify(GatedRepoError("401 Client Error")).code == "gated"
    # 查檔案清單就逾時（還不知道下載了多少）：不要顯示「? / ?」
    e = md.classify(TimeoutError("timed out"))
    assert e.code == "timeout" and "?" not in str(e) and "Hugging Face" in str(e), e
    p = md.Progress(2 * MIB, out=open(os.devnull, "w"))
    p.base = MIB
    assert "已下載 1 MB / 2 MB" in str(md.classify(TimeoutError("timed out"), progress=p))
    # 任務失敗後原樣顯示的訊息：不能寫「會自動重試」「會刪掉重新下載」這類之後不會發生的事
    for code in ("rate_limit", "corrupt", "file_locked"):
        text = md.error_text(code)
        assert "自動" not in text and "會刪掉" not in text and "按「重試」" in text, (code, text)


def test_locked_file_is_file_locked_not_no_permission():
    """.part 被防毒這類程式用不共用的方式開著：open() 丟的 PermissionError 沒有 winerror，
    以前被當成「沒有寫入權限、請搬資料夾」，也不會自動重試。唯讀檔案才是沒有權限。"""
    import ctypes
    from ctypes import wintypes

    def body(env: Env):
        part = env.models / "x.gguf.part"
        part.write_bytes(b"abc")
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.CreateFileW.restype = wintypes.HANDLE
        k32.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD,
                                    wintypes.DWORD, wintypes.HANDLE]
        handle = k32.CreateFileW(str(part), 0x80000000, 0, None, 3, 0x80, None)      # GENERIC_READ、不共用
        assert handle and handle != wintypes.HANDLE(-1).value
        try:
            try:
                open(part, "ab")
            except PermissionError as e:
                assert getattr(e, "winerror", None) is None
                err = md.classify(e)
            else:
                raise AssertionError("檔案被鎖住時應該打不開")
            assert err.code == "file_locked" and "防毒" in str(err), err
        finally:
            k32.CloseHandle(wintypes.HANDLE(handle))
        os.chmod(part, 0o444)
        try:
            try:
                open(part, "ab")
            except PermissionError as e:
                assert md.classify(e).code == "no_permission"
            else:
                raise AssertionError("唯讀檔案應該打不開")
        finally:
            os.chmod(part, 0o666)
        assert "file_locked" in jobs.DOWNLOAD_RETRY_WAITS               # 被鎖住會自動重試一次
    run(body)


def test_hash_mismatch_with_new_remote_file_is_changed_not_corrupt():
    """大小一樣、sha256 不對：遠端換了新檔案（X-Linked-ETag 跟型錄不同）是 changed，重下幾次都一樣；
    傳輸途中壞掉（遠端的 sha256 跟型錄一樣）才是 corrupt。"""
    def body(env: Env):
        data, new = os.urandom(MIB), os.urandom(MIB)
        url = env.srv.add("/h.gguf", new, etag=sha(new))
        e = expect_download_error("changed", md.fetch_file, [url], env.models / "h.gguf", len(data), sha(data))
        assert "更新程式" in str(e) and not (env.models / "h.gguf.part").exists()
        bad = bytearray(data)
        bad[5] ^= 1
        url2 = env.srv.add("/i.gguf", bytes(bad), etag=sha(data))
        expect_download_error("corrupt", md.fetch_file, [url2], env.models / "i.gguf", len(data), sha(data))
        assert md.remote_sha256(env.srv.url("/nothing")) is None
    run(body)


def test_resume_check_shows_downloaded_amount_not_zero():
    """接著下載前核對 .part 已有的部分時，進度是已經下載的量，不是先歸零（看起來像從頭下載）。"""
    import io

    def body(env: Env):
        data = os.urandom(3 * MIB)
        url = env.srv.add("/p.gguf", data)
        (env.models / "p.gguf.part").write_bytes(data[:2 * MIB])
        out = io.StringIO()
        progress = md.Progress(len(data), out=out, interval=0)
        md.fetch_file([url], env.models / "p.gguf", len(data), sha(data), progress)
        lines = [json.loads(line) for line in out.getvalue().splitlines()]
        checking = [line for line in lines if line.get("checking")]
        assert checking and all(line["done"] == 2 * MIB for line in checking), checking
        assert min(line["done"] for line in lines) >= 2 * MIB, lines[:3]
        assert jobs._download_stage({"done": 2 * MIB, "total": 3 * MIB, "checking": True}) == "檢查已下載的部分 2 MB / 3 MB"
    run(body)


# ---------- 整個模型：完成標記、版本 ----------

def _gguf_entry(env: Env, name="fake-mt", files=None) -> tuple[dict, dict]:
    files = files or {"Q8": os.urandom(MIB + 1), "Q4": os.urandom(600 * 1024)}
    variants = {}
    for v, data in files.items():
        url = env.srv.add(f"/{name}/{v}.gguf", data)
        variants[v] = {"file": f"{name}-{v}.gguf", "bytes": len(data), "sha256": sha(data), "urls": [url],
                       "vram": "estimate"}
    entry = env.add_model(name, {"role": "x", "kind": "gguf", "dir": env.models / "gguf" / name, "variants": variants})
    return entry, files


def test_variant_download_writes_marker_and_installed_follows_it():
    def body(env: Env):
        entry, files = _gguf_entry(env)
        d = entry["dir"]
        assert not config.model_installed(entry) and config.pick_variant(entry) == "Q8"
        rec = md.run("fake-mt", "Q4", progress_out=open(os.devnull, "w"))
        assert rec["files"] == {"fake-mt-Q4.gguf": len(files["Q4"])} and rec["sha256"]["fake-mt-Q4.gguf"] == sha(files["Q4"])
        marker = json.loads((d / config.COMPLETE_MARKER).read_text(encoding="utf-8"))
        assert set(marker["models"]["fake-mt"]["variants"]) == {"Q4"}, marker
        assert config.model_installed(entry, "Q4") and not config.model_installed(entry, "Q8")
        assert config.model_installed(entry) and config.installed_variants(entry) == ["Q4"]
        # 預設版本沒下載：改用已下載的
        assert config.pick_variant(entry) == "Q4" and config.pick_variant(entry, "Q8") == "Q4"
        # 有標記就照標記：檔案大小不對（被截斷、被換掉）就不算已安裝，變成殘檔
        with open(d / "fake-mt-Q4.gguf", "ab") as f:
            f.write(b"!")
        assert not config.model_installed(entry, "Q4")
        assert models.variant_row("fake-mt", "Q4", None)["leftover"]
        # 舊資料夾沒有標記：gguf 照原本的規則，檔案在就算
        legacy = env.models / "gguf" / "fake-mt" / "fake-mt-Q8.gguf"
        legacy.write_bytes(files["Q8"])
        assert config.model_installed(entry, "Q8") and config.pick_variant(entry) == "Q8"
        # 第二個版本下載完，標記裡兩個版本都在
        (d / "fake-mt-Q4.gguf").write_bytes(files["Q4"])
        md.run("fake-mt", "Q8", progress_out=open(os.devnull, "w"))
        marker = json.loads((d / config.COMPLETE_MARKER).read_text(encoding="utf-8"))
        assert set(marker["models"]["fake-mt"]["variants"]) == {"Q4", "Q8"}, marker
    run(body)


def test_disk_space_is_checked_before_downloading():
    def body(env: Env):
        _gguf_entry(env)
        real = md.shutil.disk_usage
        env.patch(md.shutil, "disk_usage", lambda p: real(p)._replace(free=100 * 1024))
        e = expect_download_error("disk_full", md.run, "fake-mt", "Q8", progress_out=open(os.devnull, "w"))
        assert "還需要" in str(e) and "目前只剩" in str(e), e
        assert not env.srv.log
    run(body)


def test_snapshot_download_is_not_installed_until_complete():
    """多檔模型：大檔到了、小的 json 還沒到時，舊的規則會當成已安裝；有 .download.json 就不算（P1-17）。"""
    def body(env: Env):
        weights, cfg = os.urandom(MIB), b'{"a": 1}'
        entry = env.add_model("fake-asr", {"role": "x", "kind": "hf", "repo": "org/fake", "dir": env.models / "asr" / "Fake",
                                           "size_mb": 1})
        env.srv.add("/org/fake/resolve/abc123/model.safetensors", weights)
        env.srv.add("/org/fake/resolve/abc123/config.json", cfg, status=[404])
        calls = []

        def fake_list(repo, allow):
            calls.append(repo)
            return "abc123", [{"path": "model.safetensors", "size": len(weights), "sha256": sha(weights)},
                              {"path": "config.json", "size": len(cfg), "sha256": None}]
        env.patch(md, "list_repo", fake_list)
        os.environ["HF_ENDPOINT"] = env.srv.url("")
        try:
            expect_download_error("not_found", md.run, "fake-asr", progress_out=open(os.devnull, "w"))
            d = entry["dir"]
            assert (d / "model.safetensors").is_file() and (d / config.DOWNLOAD_MARKER).is_file()
            assert config._safetensors_complete(d) and not config.model_installed(entry)
            md.run("fake-asr", progress_out=open(os.devnull, "w"))
        finally:
            os.environ.pop("HF_ENDPOINT", None)
        assert calls == ["org/fake"], calls                     # 接著下載時用記下來的 revision 和清單，不再查一次
        assert not (d / config.DOWNLOAD_MARKER).exists() and config.model_installed(entry)
        rec = config.complete_record(entry)
        assert rec["files"] == {"model.safetensors": len(weights), "config.json": len(cfg)}, rec
        assert rec["revisions"]["."]["revision"] == "abc123"
        # 刪掉一個小檔案：標記說不完整，不算已安裝
        (d / "config.json").unlink()
        assert not config.model_installed(entry)
    run(body)


def test_variant_delete_only_removes_that_variant():
    def body(env: Env):
        entry, files = _gguf_entry(env)
        d = entry["dir"]
        for v in ("Q8", "Q4"):
            md.run("fake-mt", v, progress_out=open(os.devnull, "w"))
        # 舊版 huggingface_hub 留下的 Q4 紀錄和暫存檔，刪 Q4 時一起清
        cache = d / config.HF_LOCAL_CACHE
        cache.mkdir(parents=True)
        (cache / "fake-mt-Q4.gguf.metadata").write_text("x")
        import base64
        prefix = base64.urlsafe_b64encode(hashlib.sha1(b"fake-mt-Q4.gguf.metadata").digest()).decode()
        legacy_part = cache / f"{prefix}.etag123.incomplete"
        legacy_part.write_bytes(b"half")
        assert legacy_part in config.hf_legacy_files(d, "fake-mt-Q4.gguf")
        other_meta = cache / "fake-mt-Q8.gguf.metadata"
        other_meta.write_text("x")
        (d / "fake-mt-Q4.gguf.part").write_bytes(b"old")
        info = models.impact("fake-mt", "Q4")
        assert info["installed"] and info["label"] == "fake-mt Q4" and info["busy"] is None, info
        assert info["lines"] == ["目前沒有用這個版本，刪掉不影響翻譯"], info
        assert models.delete("fake-mt", "Q4")["ok"]
        assert not (d / "fake-mt-Q4.gguf").exists() and not (d / "fake-mt-Q4.gguf.part").exists()
        assert not legacy_part.exists() and other_meta.exists()
        assert (d / "fake-mt-Q8.gguf").read_bytes() == files["Q8"] and config.model_installed(entry, "Q8")
        marker = json.loads((d / config.COMPLETE_MARKER).read_text(encoding="utf-8"))
        assert set(marker["models"]["fake-mt"]["variants"]) == {"Q8"}, marker
        # 刪最後一個版本：整個資料夾一起刪
        info = models.impact("fake-mt", "Q8")
        assert info["paths"] == [str(d)], info
        assert models.delete("fake-mt", "Q8")["ok"] and not d.exists()
        # 不認得的版本
        try:
            models.delete("fake-mt", "Q2")
        except models.ModelError as e:
            assert e.status == 400 and "沒有 Q2" in str(e)
        else:
            raise AssertionError("不認得的版本應該擋下")
    run(body)


def test_variant_busy_rules():
    def body(env: Env):
        entry, _ = _gguf_entry(env)
        md.run("fake-mt", "Q8", progress_out=open(os.devnull, "w"))
        jid = db.add_job(None, "model", {"model": "fake-mt", "variant": "Q4", "label": "fake-mt Q4"})
        assert models.busy_reason("fake-mt", "Q4") == "這個模型在下載佇列裡，請先取消下載任務"
        assert models.busy_reason("fake-mt", "Q8") is None                    # 排隊中的是別的版本
        assert "下載佇列" in models.busy_reason("fake-mt")                      # 整個模型
        db.update_job(jid, status="running")
        assert "其他版本" in models.busy_reason("fake-mt", "Q8")               # 正在下載別的版本：先不刪
        db.update_job(jid, status="paused")
        assert models.busy_reason("fake-mt", "Q4") == "這個模型的下載暫停中，請先取消下載任務"
        assert models.busy_reason("fake-mt", "Q8") is None
        # 下載需要的空間：暫停的也算
        assert models.download_need_bytes(entry, "Q4") == entry["variants"]["Q4"]["bytes"]
        assert models.download_need_bytes(entry, "Q8") == 0
        env.patch(models, "disk_free", lambda path=None: 2 * 1024 * MIB)
        models.check_disk("fake-mt", variant="Q8")
        db.update_job(jid, status="canceled")

        # 翻譯任務正在跑：只擋可能正在用的版本（設定要用的、llama-server 載著的），其他版本可以刪
        data = {"Q8": os.urandom(1000), "Q4": os.urandom(800)}
        tr = env.add_model("busy-mt", {"role": "translator", "translator": "fake-tr", "kind": "gguf",
                                       "dir": env.models / "gguf" / "busy-mt", "variants": {
                                           v: {"file": f"busy-{v}.gguf", "bytes": len(b), "sha256": sha(b),
                                               "urls": [env.srv.add(f"/busy/{v}", b)]} for v, b in data.items()}})
        for v in data:
            md.run("busy-mt", v, progress_out=open(os.devnull, "w"))
        tj = db.add_job(None, "translate", {"translator": "fake-tr"})
        db.update_job(tj, status="running")
        assert models.active_variant(tr) == "Q8" and models.in_use_variants(tr) == {"Q8"}
        assert models.busy_reason("busy-mt", "Q4") is None
        assert models.busy_reason("busy-mt", "Q8") == "正在用這個模型，等任務跑完再刪"
        assert models.busy_reason("busy-mt") == "正在用這個模型，等任務跑完再刪"
        assert models.delete("busy-mt", "Q4")["ok"] and config.installed_variants(tr) == ["Q8"]
        # 只剩一個版本：刪掉就沒有翻譯模型了，一樣要擋
        assert models.busy_reason("busy-mt", "Q8") == "正在用這個模型，等任務跑完再刪"
        db.update_job(tj, status="done")
    run(body)


def test_parts_model_delete_removes_markers():
    """CKIP 這種好幾個 repo 組成的模型：完成標記、下載紀錄在上層資料夾，整個模型刪掉時一起刪。"""
    def body(env: Env):
        entry = env.add_model("fake-parts", {"role": "x", "kind": "hf", "dir": env.models / "ckip-fake", "size_mb": 1,
                                             "parts": [{"repo": "org/a", "sub": "a"}, {"repo": "org/b", "sub": "b"}],
                                             "files": ["a/w.bin", "b/w.bin"]})
        blobs = {"org/a": os.urandom(3000), "org/b": os.urandom(2000)}
        for repo, b in blobs.items():
            env.srv.add(f"/{repo}/resolve/r1/w.bin", b)
        env.patch(md, "list_repo", lambda repo, allow: ("r1", [{"path": "w.bin", "size": len(blobs[repo]), "sha256": sha(blobs[repo])}]))
        os.environ["HF_ENDPOINT"] = env.srv.url("")
        try:
            md.run("fake-parts", progress_out=open(os.devnull, "w"))
        finally:
            os.environ.pop("HF_ENDPOINT", None)
        d = entry["dir"]
        assert config.model_installed(entry) and (d / config.COMPLETE_MARKER).is_file()
        assert set(config.complete_record(entry)["files"]) == {"a/w.bin", "b/w.bin"}
        (d / config.DOWNLOAD_MARKER).write_text(json.dumps({"model": "fake-parts"}), encoding="utf-8")
        assert config.model_installed(entry)                  # 有完成標記時照標記，留著的下載紀錄不影響
        assert models.delete("fake-parts")["ok"]
        assert not (d / "a").exists() and not (d / "b").exists()
        assert not (d / config.COMPLETE_MARKER).exists() and not (d / config.DOWNLOAD_MARKER).exists()
    run(body)


# ---------- 任務：子程序、暫停、繼續、自動重試、取消 ----------

WRAPPER = r"""
import json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from app import config, safepath
spec = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
entry = spec["entry"]
entry["dir"] = Path(entry["dir"])
config.MODEL_CATALOG[spec["id"]] = entry
safepath.TEST_ROOTS.append(Path(spec["root"]))
from app import model_download
model_download.QUICK_WAITS = ()
model_download.main(sys.argv[3:])
"""


def _use_wrapper(env: Env, entry: dict):
    spec = env.tmp / f"{entry['id']}.json"
    spec.write_text(json.dumps({"id": entry["id"], "root": str(env.tmp), "entry": {**entry, "dir": str(entry["dir"])}},
                               ensure_ascii=False), encoding="utf-8")
    real = jobs.model_download_cmd

    def cmd(mid, variant=None, work=None):
        return [sys.executable, "-s", "-c", WRAPPER, str(ROOT), str(spec)] + real(mid, variant, work)[4:]
    env.patch(jobs, "model_download_cmd", cmd)


def _start(job_id: str) -> threading.Thread:
    job = db.claim_job(job_id)
    assert job, db.get_job(job_id)
    th = threading.Thread(target=jobs.Worker("model", jobs.MODEL_TYPES).execute, args=(job,), daemon=True)
    th.start()
    return th


def _wait(cond, timeout=30.0, what="條件"):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return
        time.sleep(0.05)
    raise AssertionError(f"等不到{what}")


def test_job_pause_resume_continues_from_part():
    def body(env: Env):
        data = os.urandom(6 * MIB)
        entry, _ = _gguf_entry(env, "slow-mt", {"Q8": data})
        env.srv.files["/slow-mt/Q8.gguf"]["delay"] = 0.04            # 約 1.5 MB/s，才來得及在下載途中暫停
        _use_wrapper(env, entry)
        part = entry["dir"] / "slow-mt-Q8.gguf.part"
        jid = db.add_job(None, "model", {"model": "slow-mt", "variant": "Q8", "label": "slow-mt Q8"})
        th = _start(jid)
        _wait(lambda: part.exists() and part.stat().st_size > MIB, what="下載到 1 MB")
        _wait(lambda: (jobs._running.get(jid) and jobs._running[jid].child) is not None, what="子程序")
        child = jobs._running[jid].child
        assert "下載中" in (db.get_job(jid)["stage"] or ""), db.get_job(jid)
        assert jobs.pause(jid)
        th.join(10)
        assert not th.is_alive()
        t0 = time.time()
        _wait(lambda: child.poll() is not None, 5, "子程序結束")
        assert time.time() - t0 < 5
        job = db.get_job(jid)
        assert job["status"] == "paused" and job["stage"] == "已暫停", job
        paused_at = part.stat().st_size
        assert MIB < paused_at < len(data), paused_at
        assert _raises_retry_refused(jid)                        # 暫停中不能按重試，要按繼續
        # 繼續：放回佇列，從 .part 接著下載
        assert jobs.resume(jid) and db.get_job(jid)["status"] == "queued"
        env.srv.files["/slow-mt/Q8.gguf"]["delay"] = 0
        th = _start(jid)
        th.join(30)
        job = db.get_job(jid)
        assert job["status"] == "done" and job["result"] == {"model": "slow-mt", "variant": "Q8"}, job
        assert (entry["dir"] / "slow-mt-Q8.gguf").read_bytes() == data
        ranges = env.srv.ranges("/slow-mt/Q8.gguf")
        assert ranges[0] is None and len(ranges) == 2, ranges
        resumed_from = int(ranges[1].split("=")[1].rstrip("-"))
        assert resumed_from >= paused_at - MIB and resumed_from > 0, (resumed_from, paused_at)
        assert config.model_installed(entry, "Q8") and jid not in jobs.download_progress
    run(body)


def _raises_retry_refused(jid) -> bool:
    try:
        jobs.retry(jid)
    except jobs.RetryRefused as e:
        return e.status == 409 and "繼續" in str(e)
    return False


def test_job_retries_network_errors_and_reports_codes():
    def body(env: Env):
        env.patch(jobs, "DOWNLOAD_RETRY_WAITS", {"network": (0, 0), "corrupt": (0,)})
        data = os.urandom(MIB // 2)
        entry, _ = _gguf_entry(env, "flaky-mt", {"Q8": data, "Q4": data[:1000]})
        _use_wrapper(env, entry)
        env.srv.files["/flaky-mt/Q8.gguf"]["status"] = [503, 503]       # 兩次伺服器錯誤後才正常
        jid = db.add_job(None, "model", {"model": "flaky-mt", "variant": "Q8", "label": "flaky-mt Q8"})
        _start(jid).join(60)
        job = db.get_job(jid)
        assert job["status"] == "done", job
        assert [s for p, _r, s, _n in env.srv.log if p == "/flaky-mt/Q8.gguf"] == [503, 503, 200], env.srv.log
        # 找不到：不重試，任務失敗，訊息帶錯誤代碼，result 記下代碼
        env.srv.files.pop("/flaky-mt/Q4.gguf")
        jid = db.add_job(None, "model", {"model": "flaky-mt", "variant": "Q4", "label": "flaky-mt Q4"})
        _start(jid).join(60)
        job = db.get_job(jid)
        assert job["status"] == "failed" and "找不到" in job["error"] and "錯誤代碼 not_found" in job["error"], job
        assert job["result"]["error_code"] == "not_found", job
        assert len([1 for p, *_ in env.srv.log if p == "/flaky-mt/Q4.gguf"]) == 1
        view = server.model_downloads()
        failed = [i for i in view["items"] if i["job_id"] == jid][0]
        assert failed["error"]["code"] == "not_found" and failed["variant"] == "Q4", failed
        # hash 不符：刪掉重新下載一次，還是不對就失敗
        bad = bytearray(data[:1000])
        bad[0] ^= 1
        env.srv.add("/flaky-mt/Q4.gguf", bytes(bad))
        jid = db.add_job(None, "model", {"model": "flaky-mt", "variant": "Q4", "label": "flaky-mt Q4"})
        _start(jid).join(60)
        job = db.get_job(jid)
        assert job["status"] == "failed" and "錯誤代碼 corrupt" in job["error"], job
        assert len([1 for p, *_ in env.srv.log if p == "/flaky-mt/Q4.gguf"]) == 3        # 404 一次 + 下載兩次
        assert not (entry["dir"] / "flaky-mt-Q4.gguf.part").exists()
    run(body)


def test_job_cancel_stops_child_and_keeps_part():
    def body(env: Env):
        data = os.urandom(6 * MIB)
        entry, _ = _gguf_entry(env, "cancel-mt", {"Q8": data})
        env.srv.files["/cancel-mt/Q8.gguf"]["delay"] = 0.04
        _use_wrapper(env, entry)
        part = entry["dir"] / "cancel-mt-Q8.gguf.part"
        jid = db.add_job(None, "model", {"model": "cancel-mt", "variant": "Q8", "label": "cancel-mt Q8"})
        th = _start(jid)
        _wait(lambda: part.exists() and part.stat().st_size > MIB // 2, what="開始下載")
        child = jobs._running[jid].child
        jobs.cancel(jid)
        th.join(10)
        _wait(lambda: child.poll() is not None, 5, "子程序結束")
        assert db.get_job(jid)["status"] == "canceled"
        size = part.stat().st_size
        time.sleep(0.5)
        assert part.stat().st_size == size                  # 取消後沒有東西繼續寫
        assert models.variant_row("cancel-mt", "Q8", None)["leftover"]
    run(body)


def test_download_endpoint_variants_and_pause_all():
    def body(env: Env):
        entry, _ = _gguf_entry(env, "api-mt")
        env.patch(models, "disk_free", lambda path=None: 100 * 1024 * MIB)
        assert server.download_model("api-mt", "Q4") == {"ok": True}
        job = [j for j in db.list_jobs() if j["type"] == "model"][0]
        assert job["params"] == {"model": "api-mt", "label": "api-mt Q4", "variant": "Q4"}, job
        from fastapi import HTTPException
        for bad in ("Q2",):
            try:
                server.download_model("api-mt", bad)
            except HTTPException as e:
                assert e.status_code == 400
            else:
                raise AssertionError("不認得的版本應該擋下")
        try:
            server.download_model("api-mt", "Q4")
        except HTTPException as e:
            assert e.status_code == 409
        # 暫停全部 → 同一個版本再按下載 = 繼續
        assert server.pause_model_downloads()["count"] == 1
        assert db.get_job(job["id"])["status"] == "paused"
        assert server.download_model("api-mt", "Q4") == {"ok": True, "resumed": True}
        assert db.get_job(job["id"])["status"] == "queued"
        # 沒指定版本：用設定選的版本（沒選用預設版本）
        assert server.download_model("api-mt") == {"ok": True}
        assert [j["params"].get("variant") for j in db.list_jobs() if j["type"] == "model"] == ["Q4", "Q8"]
        settings.save({"model_variants": {"api-mt": "Q4"}})
        view = {m["id"]: m for m in server._catalog_view(None)}["api-mt"]
        assert [v["job"]["variant"] for v in view["variants"]] == ["Q8", "Q4"], view["variants"]
        assert jobs.pause(job["id"]) and not jobs.pause("nope")
        jobs.cancel(job["id"])
        assert db.get_job(job["id"])["status"] == "canceled"
    run(body)


def test_model_downloads_hides_failures_that_were_resolved():
    """/api/models/downloads 的失敗任務：同一個模型（版本）之後裝好了、或已經有比較新的下載任務，就不再列出來。"""
    def body(env: Env):
        _gguf_entry(env, "stale-mt")
        j1 = db.add_job(None, "model", {"model": "stale-mt", "variant": "Q4", "label": "stale-mt Q4"})
        db.update_job(j1, status="failed", error="連不到 Hugging Face。（錯誤代碼 network）", result={"error_code": "network"})
        assert [i["job_id"] for i in server.model_downloads()["items"]] == [j1]
        md.run("stale-mt", "Q4", progress_out=open(os.devnull, "w"))           # 之後用另一個任務下載好了
        assert server.model_downloads()["items"] == []
        # 別的版本失敗、還沒裝：照樣列；按「下載」建立新任務後，只列新的
        time.sleep(0.01)
        j2 = db.add_job(None, "model", {"model": "stale-mt", "variant": "Q8", "label": "stale-mt Q8"})
        db.update_job(j2, status="failed", error="x（錯誤代碼 timeout）", result={"error_code": "timeout"})
        assert [i["job_id"] for i in server.model_downloads()["items"]] == [j2]
        time.sleep(0.01)
        j3 = db.add_job(None, "model", {"model": "stale-mt", "variant": "Q8", "label": "stale-mt Q8"})
        assert [(i["job_id"], i["status"]) for i in server.model_downloads()["items"]] == [(j3, "queued")]
        db.update_job(j3, status="canceled")
    run(body)


def test_model_downloads_have_their_own_worker():
    def body(env: Env):
        assert "model" not in jobs.IO_TYPES and jobs.MODEL_TYPES == ("model",) and "download" in jobs.IO_TYPES
        mj = db.add_job(None, "model", {"model": "x"})
        db.update_job(mj, status="running")
        dj = db.add_job(None, "download", {"url": "https://youtu.be/x"})
        picked = jobs.Worker("io", jobs.IO_TYPES).pick()
        assert picked and picked["id"] == dj                   # 模型下載中，網址下載照樣馬上開始
        assert jobs.Worker("model", jobs.MODEL_TYPES).pick() is None
        assert "app.model_download" in jobs.CHILD_MODULES      # 殘留的下載子程序啟動時會清掉
        cmd = jobs.model_download_cmd("hymt2-7b", "Q6_K", env.work / "abc")
        assert cmd[-6:] == ["--variant", "Q6_K", "--parent-pid", str(os.getpid()), "--work", str(env.work / "abc")]
        assert jobs._module_arg(cmd) == "app.model_download"
        e = jobs.model_download_env()
        assert e["HF_HUB_DISABLE_XET"] == "1" and "HF_XET_HIGH_PERFORMANCE" not in e and e["HF_HUB_OFFLINE"] == "0"
    run(body)


def test_child_exits_when_server_process_dies():
    """伺服器被直接關掉時，下載子程序跟著結束（不會留在背景繼續寫 .part）。"""
    parent = subprocess.Popen([sys.executable, "-s", "-c", "import time; time.sleep(120)"],
                              creationflags=subprocess.CREATE_NO_WINDOW)
    child = subprocess.Popen([sys.executable, "-s", "-c",
                              "import sys, time; sys.path.insert(0, sys.argv[1]);"
                              "from app.model_download import watch_parent; watch_parent(int(sys.argv[2])); time.sleep(120)",
                              str(ROOT), str(parent.pid)], creationflags=subprocess.CREATE_NO_WINDOW)
    try:
        time.sleep(1.5)
        assert child.poll() is None
        parent.kill()
        t0 = time.time()
        child.wait(10)
        assert time.time() - t0 < 5 and child.returncode == 3, child.returncode
    finally:
        for p in (parent, child):
            if p.poll() is None:
                p.kill()


# ---------- 設定：版本、llama.cpp 參數 ----------

def test_settings_validate_variants_and_llm_params():
    def body(env: Env):
        for bad in ({"model_variants": {"hymt2-7b": "Q2_K"}}, {"model_variants": {"qwen-asr": "Q8_0"}},
                    {"model_variants": ["Q8_0"]}, {"llm_params": {"nope": {"ctx": 4096}}},
                    {"llm_params": {"hymt": {"ctx": 4096, "parallel": 8}}}, {"llm_params": {"hymt": {"ctx": "4096"}}},
                    {"llm_params": {"hymt": {"ctx": 4096, "batch": 2}}}):
            try:
                settings.save(bad)
            except ValueError:
                continue
            raise AssertionError(f"應該擋下 {bad}")
        settings.save({"model_variants": {"hymt2-7b": "Q6_K"}, "llm_params": {"hymt": {"ctx": 4096, "parallel": 2}}})
        values = settings.all_values()
        assert values["model_variants"] == {"hymt2-7b": "Q6_K"} and values["llm_params"]["hymt"]["parallel"] == 2
        assert settings.DEFAULTS["model_variants"] == {} and settings.DEFAULTS["llm_params"] == {}
    run(body)


def test_translator_cfg_follows_settings_and_llm_is_reloaded_when_changed():
    def body(env: Env):
        real = config.MODEL_CATALOG["hymt2-7b"]
        fake = {**real, "dir": env.models / "gguf" / "Hy-MT2-7B"}
        catalog = {**config.MODEL_CATALOG, "hymt2-7b": fake}
        env.patch(config, "MODEL_CATALOG", catalog)
        d = fake["dir"]
        d.mkdir(parents=True)
        (d / real["variants"]["Q6_K"]["file"]).write_bytes(b"q6" * 1000)
        cfg = config.translator_cfg("hymt")
        assert cfg["variant"] == "Q6_K" and cfg["gguf"] == d / "HY-MT2-7B-Q6_K.gguf"     # 預設版本沒下載：用已下載的
        assert (cfg["ctx"], cfg["parallel"], cfg["overhead_mb"]) == (8192, 4, 1200)
        assert config.translator_vram_mb("hymt", cfg=cfg) == 0 + 1200                      # 已下載：用檔案大小
        not_yet = config.translator_cfg("hymt", variant="Q4_K_M")
        assert config.translator_vram_mb("hymt", cfg=not_yet) == 4410 + 1200               # 沒下載：用型錄大小
        settings.save({"llm_params": {"hymt": {"ctx": 4096, "parallel": 2}}})
        cfg = config.translator_cfg("hymt")
        assert (cfg["ctx"], cfg["parallel"], cfg["overhead_mb"]) == (4096, 2, 700)
        assert config.llm_overhead_mb("hymt", 16384) == 1200 + 1024                        # 表上沒有：照 KV cache 估
        assert config.translator_cfg("hymt", variant="Q8_0", params={})["ctx"] == 8192

        loaded = []

        class FakeServer:
            def __init__(self, key, log_path, cfg=None):
                self.cfg = cfg
                self.stopped = False
                loaded.append(self)

            def start(self, check):
                pass

            def alive(self):
                return not self.stopped

            def stop(self):
                self.stopped = True

        env.patch(translate, "LlamaServer", FakeServer)
        env.patch(jobs.gpu, "ensure_free", lambda *a, **k: None)
        saved = dict(jobs._llm)
        try:
            s1, reused = jobs.get_llm("hymt", env.work / "l.log", lambda: None)
            assert not reused and s1.cfg["variant"] == "Q6_K" and s1.cfg["ctx"] == 4096
            assert jobs.get_llm("hymt", env.work / "l.log", lambda: None) == (s1, True)
            # 下載了 Q8_0、改選 Q8_0：換成新的 llama-server
            (d / real["variants"]["Q8_0"]["file"]).write_bytes(b"q8" * 1000)
            settings.save({"model_variants": {"hymt2-7b": "Q8_0"}})
            s2, reused = jobs.get_llm("hymt", env.work / "l.log", lambda: None)
            assert not reused and s1.stopped and s2.cfg["variant"] == "Q8_0"
            # 顯存不足的建議：同一個模型放得下的版本排第一
            env.patch(config, "_user_choices", lambda: ({}, {}))
            names = jobs.smaller_translators("hymt", 7500)
            assert names[0] == "Hy-MT2-7B Q6_K", names
            # 刪除版本時只關掉載著那個版本的 llama-server
            models._release_llm_holding("hymt", "x", d / "HY-MT2-7B-Q6_K.gguf")
            assert jobs._llm["server"] is s2 and not s2.stopped
            models._release_llm_holding("hymt", "x", d / "HY-MT2-7B-Q8_0.gguf")
            assert s2.stopped and jobs._llm["key"] is None
        finally:
            jobs._llm.update(saved)
    run(body)


# ---------- 擁有者現有的模型（唯讀） ----------

def test_owner_installed_models_behave_as_before():
    """作者電腦上已經裝好的 Hy-MT2-7B Q8_0、Sakura-14B Q6_K：一樣被認成已安裝、是預設用的版本，
    gguf 路徑、ctx、parallel、顯存門檻都跟改之前一樣。只讀檔案資訊，不寫入。沒有這些模型的電腦跳過。"""
    cat = config.MODEL_CATALOG
    old = {  # 改之前 config.TRANSLATORS 的值
        "hymt": ("gguf/Hy-MT2-7B/HY-MT2-7B-Q8_0.gguf", 8192, 4, 1200, "hymt2-7b", "Q8_0", 7612),
        "sakura": ("gguf/Sakura-14B/sakura-14b-qwen2.5-v1.0-q6k.gguf", 4096, 2, 700, "sakura-14b", "Q6_K", 11563),
        "galtransl": ("gguf/GalTransl-7B/Sakura-Galtransl-7B-v3.7.gguf", 4096, 2, 700, "galtransl-7b", "Q6_K", 5964),
        "hymt-mini": ("gguf/Hy-MT2-1.8B/Hy-MT2-1.8B-Q8_0.gguf", 8192, 4, 900, "hymt2-1.8b", None, 1820),
    }
    for key, (rel, ctx, parallel, overhead, mid, variant, size_mb) in old.items():
        cfg = config.translator_cfg(key, params={}) if variant is None else config.translator_cfg(key, variant=variant, params={})
        assert config.TRANSLATORS[key]["gguf"] == REAL_MODELS / rel
        assert (cfg["ctx"], cfg["parallel"], cfg["overhead_mb"]) == (ctx, parallel, overhead), (key, cfg)
        assert abs(cat[mid]["size_mb"] - size_mb) <= 1
        entry = cat[mid]
        installed = (REAL_MODELS / rel).is_file()
        if variant:
            assert entry["default_variant"] == variant and cfg["gguf"] == REAL_MODELS / rel
            assert config.check_variant(entry, None) == variant
        if not installed:
            print(f"  （沒有安裝 {rel}，只檢查設定）")
            continue
        assert config.model_installed(entry)
        if variant:
            assert config.model_installed(entry, variant) and config.pick_variant(entry) == variant
            assert config.installed_variants(entry)[0] == variant
            size = (REAL_MODELS / rel).stat().st_size
            assert size == entry["variants"][variant]["bytes"], (rel, size)
        assert config.translator_vram_mb(key, cfg=cfg) == int((REAL_MODELS / rel).stat().st_size / MIB) + overhead
        # 整個模型的刪除範圍還是自己的資料夾
        assert config.model_targets(entry) == [(entry["dir"], True)]
    if (REAL_MODELS / old["hymt"][0]).is_file():
        assert config.translator_vram_mb("hymt", cfg=config.translator_cfg("hymt", variant="Q8_0", params={})) == 8812
    if (REAL_MODELS / old["sakura"][0]).is_file():
        assert config.translator_vram_mb("sakura", cfg=config.translator_cfg("sakura", variant="Q6_K", params={})) == 12262
    # 其他模型沒有完成標記，照原本的規則判斷（這台電腦上的結果跟改之前的規則一樣）
    for mid, entry in cat.items():
        if entry.get("variants") or config.complete_record(entry) is not None:
            continue
        d = entry["dir"]
        if entry.get("files"):
            legacy = all((d / f).is_file() for f in entry["files"])
        elif entry.get("kind") == "gguf":
            legacy = (d / entry["file"]).is_file()
        else:
            legacy = d.is_dir() and config._safetensors_complete(d)
        assert config.model_installed(entry) == legacy, mid


if __name__ == "__main__":
    started = time.time()
    try:
        for name, fn in list(globals().items()):
            if name.startswith("test_") and callable(fn):
                t0 = time.time()
                fn()
                print(f"ok {name} ({time.time() - t0:.1f}s)")
    finally:
        if BASE.exists() and not any(BASE.iterdir()):
            BASE.rmdir()
    print(f"all passed in {time.time() - started:.1f}s")
