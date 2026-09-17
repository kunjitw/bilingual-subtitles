"""設定頁模型管理（app/models.py）的測試。不用顯卡、不連網，也絕對不動 models 底下真正的模型檔：
要刪的模型全部是假的型錄項目，檔案放在系統暫存資料夾的 models-test 底下，測完整個刪掉。
（顯存門檻、假名引擎重新初始化這兩項會唯讀地讀真正的模型檔。）

python -s tests/test_models.py
"""
import http.server
import logging
import os
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path

os.environ["VS_NO_WORKERS"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import HTTPException  # noqa: E402

from app import config, db, jobs, model_download, models, safepath, server, settings  # noqa: E402

BASE = Path(tempfile.gettempdir()) / "models-test"
MIB = 1024 * 1024


def touch(p: Path, data: bytes = b"x") -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    return p


class Env:
    """假的 models 資料夾、假的型錄、暫存資料庫。"""

    def __init__(self, catalog_fn):
        BASE.mkdir(parents=True, exist_ok=True)
        self.tmp = Path(tempfile.mkdtemp(prefix="vs-models-", dir=BASE))
        self.models = self.tmp / "models"
        self.models.mkdir()
        safepath.TEST_ROOTS.append(self.tmp)       # 這個暫存資料夾裡的東西可以刪（測完拿掉）
        self._saved = []
        self._saved_conn = db._conn
        self.patch(db, "DB_PATH", self.tmp / "test.db")
        db.init()
        self.catalog = catalog_fn(self.models)
        for mod in (models, jobs, settings):
            self.patch(mod, "MODEL_CATALOG", self.catalog)
        self.patch(models, "MODELS_DIR", self.models)
        self.patch(models, "RETRY_WAITS", (0.05,))
        # 保險：真的 models 資料夾不能出現在這個測試要刪的任何路徑裡
        real = str(config.MODELS_DIR.resolve()).lower()
        for e in self.catalog.values():
            assert not str(Path(e["dir"]).resolve()).lower().startswith(real), e["dir"]
        self._levels = []
        for name in ("models", "safepath", "jobs"):
            logger = logging.getLogger(name)
            self._levels.append((logger, logger.level))
            logger.setLevel(logging.CRITICAL)

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
        safepath.TEST_ROOTS.remove(self.tmp)
        shutil.rmtree(self.tmp, ignore_errors=True)


def run(catalog_fn, fn):
    env = Env(catalog_fn)
    try:
        fn(env)
    finally:
        env.close()


def expect_model_error(status, fn, *args):
    try:
        fn(*args)
    except models.ModelError as e:
        assert e.status == status, (e.status, str(e))
        return str(e)
    raise AssertionError(f"預期 ModelError {status}")


def hf_catalog(m: Path):
    return {
        "one": {"role": "x", "label": "OneFile", "kind": "hf", "dir": m / "asr" / "One", "size_mb": 1, "langs": ["ja"]},
        "shard": {"role": "x", "label": "Sharded", "kind": "hf", "dir": m / "asr" / "Shard", "size_mb": 1, "langs": ["ja"]},
        "sep": {"role": "separator", "label": "FakeSep", "kind": "ckpt", "dir": m / "sep", "file": "f.ckpt",
                "files": ["f.ckpt", "f.yaml"], "size_mb": 1, "langs": ["ja"]},
    }


# ---------- 顯示用的數字 ----------

def test_fmt_size_units():
    assert models.fmt_size(0) == "0 MB"
    assert models.fmt_size(77 * MIB) == "77 MB"
    assert models.fmt_size(1000 * MIB) == "1000 MB"
    assert models.fmt_size(1024 * MIB) == "1.0 GB"
    assert models.fmt_size(4485 * MIB) == "4.4 GB"
    assert models.fmt_size(12262 * MIB) == "12.0 GB"


def test_fits_compares_with_whole_card_minus_reserved():
    # 門檻是實際檢查用的數字；Sakura-14B 是 gguf 大小 + 700，大約 12262 MiB
    # （已經下載時兩邊都用檔案大小；還沒下載時都用型錄的 size_mb，沒下載模型的電腦也能跑）
    entry = config.MODEL_CATALOG["sakura-14b"]
    need = config.model_vram_mb(entry)
    assert need == config.translator_vram_mb("sakura", entry["size_mb"])
    card12 = {"name": "12GB", "total_mb": 12288, "used_mb": 700, "free_mb": 11330}   # 保留 258
    card16 = {"name": "16GB", "total_mb": 16303, "used_mb": 1858, "free_mb": 14140}
    assert models.gpu_usable_mb(card12) == 12030 and models.gpu_usable_mb(None) is None
    assert models.row("sakura-14b", card12)["fits"] is False          # 總顯存 12288 看起來夠，其實放不下
    assert models.row("sakura-14b", card16)["fits"] is True
    assert models.row("sakura-14b", card12)["vram_mb"] == need
    assert models.row("hymt2-7b", {"total_mb": 8188, "used_mb": 400, "free_mb": 7538})["fits"] is False
    assert models.row("qwen-asr", {"total_mb": 8188, "used_mb": 400, "free_mb": 7538})["fits"] is True
    tsq = models.row("tsqyomi", card12)
    assert tsq["vram"] is None and tsq["fits"] is None                 # CPU 模型
    assert models.row("qwen-asr", None)["fits"] is None                # 讀不到顯卡


def test_catalog_licenses():
    # 每個模型都寫了授權和寫著授權條件的網址；非商用、授權不明的設定頁要看得出來
    for mid, e in config.MODEL_CATALOG.items():
        assert e.get("license") and (e.get("license_url") or "").startswith("https://"), mid
    limits = {mid: models.license_limit(e["license"]) for mid, e in config.MODEL_CATALOG.items()}
    for mid in ("sakura-14b", "galtransl-7b", "anime-whisper"):
        assert limits[mid] == "noncommercial", (mid, limits)
    assert limits["bs-roformer"] == "unknown", limits
    assert limits["qwen-asr"] is None and limits["hymt2-7b"] is None and limits["ckip"] is None, limits
    assert models.license_limit(None) == "unknown" and models.license_limit("MIT") is None
    r = models.row("sakura-14b", None)
    assert r["license"].startswith("非商用") and r["license_limit"] == "noncommercial" and r["license_url"], r


# ---------- 刪除：檔案被鎖住 ----------

def test_locked_weights_leave_model_intact():
    def body(env: Env):
        d = env.catalog["one"]["dir"]
        files = [touch(d / n) for n in ("added_tokens.json", "config.json", "merges.txt", "model.safetensors")]
        handle = open(d / "model.safetensors", "rb")          # 沒有 FILE_SHARE_DELETE，刪不掉
        try:
            msg = expect_model_error(409, models.delete, "one")
            assert "沒有刪除" in msg, msg
            assert all(f.exists() for f in files), "權重刪不掉時其他檔案都不能動"
            assert config.model_installed(env.catalog["one"])
        finally:
            handle.close()
        assert models.delete("one")["ok"] and not d.exists()
    run(hf_catalog, body)


def test_weights_gone_but_other_file_locked_becomes_leftover():
    def body(env: Env):
        d = env.catalog["one"]["dir"]
        touch(d / "model.safetensors", b"w" * 2048)
        touch(d / "tokenizer.json")
        handle = open(d / "tokenizer.json", "rb")
        try:
            msg = expect_model_error(409, models.delete, "one")
            assert "不能用" in msg and "清掉" in msg, msg
            r = models.row("one", None)
            assert not r["installed"] and r["leftover"], r
            assert models.impact("one")["installed"] is False
        finally:
            handle.close()
        assert models.delete("one")["ok"] and not d.exists()          # 清掉殘檔
    run(hf_catalog, body)


def test_sharded_model_with_one_locked_shard_is_not_shown_installed():
    def body(env: Env):
        d = env.catalog["shard"]["dir"]
        touch(d / "config.json")
        touch(d / "model-00001-of-00002.safetensors")
        touch(d / "model-00002-of-00002.safetensors")
        assert config.model_installed(env.catalog["shard"])
        handle = open(d / "model-00002-of-00002.safetensors", "rb")
        try:
            expect_model_error(409, models.delete, "shard")
            assert not config.model_installed(env.catalog["shard"])
            assert models.row("shard", None)["leftover"]
        finally:
            handle.close()
        assert models.delete("shard")["ok"] and not d.exists()
    run(hf_catalog, body)


# ---------- 刪除：跟任務的配合 ----------

def _running_ctx(job_type, params):
    jid = db.add_job(None, job_type, params)
    db.update_job(jid, status="running")
    ctx = jobs.JobContext(db.get_job(jid))
    with jobs._running_lock:
        jobs._running[ctx.id] = ctx
    return ctx


def _drop_ctx(ctx):
    with jobs._running_lock:
        jobs._running.pop(ctx.id, None)
    db.update_job(ctx.id, status="done")


def test_model_registered_by_running_job_cannot_be_deleted():
    def body(env: Env):
        sep = env.catalog["sep"]
        for n in sep["files"]:
            touch(sep["dir"] / n)
        checks = touch(sep["dir"] / "download_checks.json")
        # 轉字幕任務在跑，但還沒決定要不要分離人聲：可以刪（任務之後會照原音處理）
        ctx = _running_ctx("transcribe", {"engine": "none", "language": "ja"})
        try:
            assert models.busy_reason("sep") is None
            # 決定要分離人聲、登記之後就不能刪，跟當下的設定無關
            assert jobs.use_model(ctx, "sep") is True
            settings.save({"song_mode": "off"})
            assert models.busy_reason("sep") == "正在用這個模型，等任務跑完再刪"
            expect_model_error(409, models.delete, "sep")
            assert all((sep["dir"] / n).exists() for n in sep["files"])
        finally:
            _drop_ctx(ctx)
        assert models.delete("sep")["ok"]
        assert not any((sep["dir"] / n).exists() for n in sep["files"]) and checks.exists()
    run(hf_catalog, body)


def test_use_model_waits_for_delete_and_reports_missing():
    def body(env: Env):
        sep = env.catalog["sep"]
        for n in sep["files"]:
            touch(sep["dir"] / n)
        started = threading.Event()
        orig = models._remove

        def slow_remove(entry, targets, *rest):
            started.set()
            time.sleep(0.8)
            return orig(entry, targets, *rest)

        env.patch(models, "_remove", slow_remove)
        t = threading.Thread(target=models.delete, args=("sep",))
        t.start()
        started.wait(5)
        ctx = jobs.JobContext({"id": "use-model-test", "media_id": None, "params": {}})
        t0 = time.time()
        # 刪除途中登記：等刪完才回答，而且回答「模型不在」，不會拿到刪到一半的檔案
        assert jobs.use_model(ctx, "sep") is False
        assert time.time() - t0 > 0.3 and not ctx.uses
        t.join()
        try:
            jobs.require_model(ctx, "sep")
        except RuntimeError as e:
            assert str(e) == "找不到模型 FakeSep，請到設定頁的模型管理下載", e
        else:
            raise AssertionError("模型不在應該失敗")
    run(hf_catalog, body)


def gguf_catalog(m: Path):
    return {
        "fa": {"role": "translator", "translator": "fa", "label": "FakeA", "kind": "gguf", "file": "fa.gguf",
               "dir": m / "gguf" / "FA", "size_mb": 1, "langs": ["ja"]},
    }


class FakeServer:
    def __init__(self):
        self.stopped = False

    def stop(self):
        self.stopped = True


def test_idle_llama_server_is_released_and_other_load_does_not_block():
    def body(env: Env):
        gguf = touch(env.catalog["fa"]["dir"] / "fa.gguf")
        env.patch(settings, "installed_translators", lambda language=None, without=(): [])
        # 1. 別的翻譯模型正在載入（拿著 _llm_lock），這個模型沒有載入：不用等那邊載完
        hold = threading.Event()
        release = threading.Event()

        def loader():
            with jobs._llm_lock:
                hold.set()
                release.wait(10)

        th = threading.Thread(target=loader)
        th.start()
        hold.wait(5)
        try:
            t0 = time.time()
            assert models.delete("fa")["ok"]
            assert time.time() - t0 < 2, time.time() - t0
            assert not gguf.exists()
        finally:
            release.set()
            th.join()

        # 2. llama-server 閒置但還載著這個模型：先關掉再刪
        gguf = touch(env.catalog["fa"]["dir"] / "fa.gguf")
        fake = FakeServer()
        saved = dict(jobs._llm)
        jobs._llm.update(key="fa", server=fake)
        try:
            assert models.delete("fa")["ok"]
            assert fake.stopped and jobs._llm["key"] is None and not gguf.exists()
        finally:
            jobs._llm.update(saved)
    run(gguf_catalog, body)


# ---------- 下載 ----------

def test_download_is_refused_when_disk_is_full():
    def body(env: Env):
        env.patch(models, "disk_free", lambda path=None: 200 * MIB)
        big = {"role": "x", "label": "Big", "kind": "gguf", "file": "big.gguf", "dir": env.models / "gguf" / "Big",
               "size_mb": 5000, "langs": ["ja"]}
        env.catalog["big"] = big
        env.patch(server, "MODEL_CATALOG", env.catalog)
        try:
            server.download_model("big")
        except HTTPException as e:
            assert e.status_code == 400 and "磁碟空間不夠" in e.detail and "4.9 GB" in e.detail, e.detail
        else:
            raise AssertionError("空間不夠應該拒絕")
        assert not any(j["type"] == "model" for j in db.list_jobs())
    run(hf_catalog, body)


class _Truncating(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Length", "1000" if self.path == "/short" else "100")
        self.end_headers()
        self.wfile.write(b"a" * 100)
        self.wfile.flush()

    def log_message(self, *args):
        pass


def test_fetch_rejects_truncated_download():
    tmp = Path(tempfile.mkdtemp(prefix="vs-fetch-", dir=BASE if BASE.exists() else None))
    httpd = http.server.HTTPServer(("127.0.0.1", 0), _Truncating)
    th = threading.Thread(target=httpd.serve_forever, daemon=True)
    th.start()
    saved_waits = model_download.QUICK_WAITS
    model_download.QUICK_WAITS = (0,)
    try:
        url = f"http://127.0.0.1:{httpd.server_address[1]}"
        dst = tmp / "m.ckpt"
        try:
            model_download.fetch([f"{url}/short"], dst)
        except RuntimeError as e:
            # 連線中斷算網路錯誤（代碼 network），已下載的部分留在 .part，之後接著下載（細節見 tests/test_model_download.py）
            assert "連不到下載來源" in str(e) and "已下載的部分會保留" in str(e), e
        else:
            raise AssertionError("收到的比 Content-Length 少應該算失敗")
        assert not dst.exists() and (tmp / "m.ckpt.part").stat().st_size == 100
        model_download.fetch([f"{url}/full"], dst)       # 伺服器不支援續傳：從頭下載
        assert dst.stat().st_size == 100 and not (tmp / "m.ckpt.part").exists()
    finally:
        model_download.QUICK_WAITS = saved_waits
        httpd.shutdown()
        httpd.server_close()
        shutil.rmtree(tmp, ignore_errors=True)


# ---------- 設定頁資料 ----------

def test_settings_view_has_space_vram_and_roles():
    env_tmp = Path(tempfile.mkdtemp(prefix="vs-models-view-", dir=BASE if BASE.exists() else None))
    saved = (db.DB_PATH, db._conn)
    db.DB_PATH = env_tmp / "t.db"
    db.init()
    try:
        info = {"name": "fake", "total_mb": 12288, "used_mb": 700, "free_mb": 11330}
        view = {m["id"]: m for m in server._catalog_view(info)}
        assert set(view) == set(config.MODEL_CATALOG)
        for m in view.values():
            assert "core" not in m and m["size"] and m["disk"] is not None, m
            assert (m["vram"] is None) == (m["role"] in ("furigana", "segmenter")), m
        assert view["bs-roformer"]["used_by"] == ["歌曲"] and view["ckip"]["used_by"] == ["中文單字"]
        assert view["sakura-14b"]["fits"] is False
        impact = server.model_impact("aligner")
        assert any("沒辦法轉字幕" in line for line in impact["lines"]) or not impact["installed"]
        st = models.storage_summary()
        assert st["dir"] == str(config.MODELS_DIR) and st["free"].endswith("GB")
    finally:
        db._conn.close()
        db.DB_PATH, db._conn = saved
        shutil.rmtree(env_tmp, ignore_errors=True)


def test_furigana_reinit_after_download():
    """啟動時 tsqyomi 不在（初始化失敗），下載好之後不用重開程式就能用。真正的模型資料夾只唯讀地讀。"""
    from app import furigana
    real = furigana.TSQYOMI_DIR
    if not (real / "model.onnx").is_file():
        print("  （沒有安裝 tsqyomi，跳過）")
        return
    empty = Path(tempfile.mkdtemp(prefix="vs-tsq-", dir=BASE if BASE.exists() else None))
    logger = logging.getLogger("furigana")
    level = logger.level
    logger.setLevel(logging.CRITICAL)
    saved_csv = furigana.USER_CSV
    furigana.USER_CSV = empty / "no-user-dict.csv"      # 不去重建 data 裡的使用者辭典
    try:
        furigana.TSQYOMI_DIR = empty
        furigana.init()
        assert not furigana.available(1)
        furigana.TSQYOMI_DIR = real
        furigana.reload_if_failed()
        assert furigana.available(1), furigana._state["error"]
        assert any(r[2] == "かんじ" for r in furigana.ruby_for_text("漢字"))
    finally:
        furigana.TSQYOMI_DIR = real
        furigana.USER_CSV = saved_csv
        logger.setLevel(level)
        shutil.rmtree(empty, ignore_errors=True)


def test_furigana_without_model_logs_one_line():
    """新安裝還沒下載 tsqyomi：啟動視窗只多一行說明，不印整段 Traceback。"""
    from app import furigana
    saved_dir, saved_state, was_ready = furigana.TSQYOMI_DIR, dict(furigana._state), furigana._ready.is_set()
    empty = Path(tempfile.mkdtemp(prefix="vs-tsq-none-", dir=BASE if BASE.exists() else None))
    records = []
    handler = logging.Handler()
    handler.emit = records.append
    logger = logging.getLogger("furigana")
    level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        furigana.TSQYOMI_DIR = empty
        furigana.init()
        assert not furigana.available(1) and furigana._state["error"] == "tsqyomi 模型還沒下載"
        assert len(records) == 1 and records[0].exc_info is None and records[0].levelno == logging.INFO, records
        assert "tsqyomi" in records[0].getMessage()
    finally:
        logger.removeHandler(handler)
        logger.setLevel(level)
        furigana.TSQYOMI_DIR = saved_dir
        furigana._state.clear()
        furigana._state.update(saved_state)
        if not was_ready:
            furigana._ready.clear()
        shutil.rmtree(empty, ignore_errors=True)


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
