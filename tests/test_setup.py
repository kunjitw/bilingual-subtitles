"""第一次打開的自動安裝（app/setup.py）的測試：各級顯示卡選哪個翻譯模型（含分界值）、不能裝的情況和說明、
新舊使用者的判斷（擁有者的資料不觸發、.dev、判定一次就記住）、排任務的順序和設定、磁碟不夠、監看執行緒
（等網路、自動重試、放棄、取消、磁碟滿了接著裝、重開時立刻重試、裝好標記完成）、安裝中新增的影片排隊等模型、
API 的錯誤代碼，最後用假的 Hugging Face 和假的字典來源從頭裝一次（中途斷網再恢復，從 .part 接著下載）。

不連網、不用顯卡：顯示卡用假的（syscheck.set_fake），data、models 都在系統暫存資料夾的 setup-test 底下
（VS_DATA_DIR、VS_MODELS_DIR），測完整個刪掉。會唯讀地看真正的 models 資料夾（確認擁有者的電腦判成已經在用），
前後比對檔案清單，不在裡面寫入任何東西。

python -s tests/test_setup.py
"""
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

BASE = Path(tempfile.gettempdir()) / "setup-test"
RUN = BASE / f"run-{os.getpid()}"
os.environ["VS_DATA_DIR"] = str(RUN / "data")
os.environ["VS_MODELS_DIR"] = str(RUN / "models")
os.environ["VS_SETUP_AUTO"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_ENDPOINT"] = "http://127.0.0.1:9"              # 連不上：不會意外連到真的 Hugging Face
os.environ["VS_DICT_SOURCE_BASE"] = "http://127.0.0.1:9"
for _k in ("VS_NO_WORKERS", "VS_FAKE_SYSTEM", "CUDA_VISIBLE_DEVICES"):
    os.environ.pop(_k, None)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi.testclient import TestClient  # noqa: E402

import fake_sources as fk  # noqa: E402
from app import config, db, dict_build, gpu, jobs, models, safepath, server, settings, setup, syscheck  # noqa: E402
from app import model_download as md  # noqa: E402

REAL_MODELS = ROOT / "models"
REAL_NET_OK = setup.net_ok
REAL_WORKERS_RUNNING = setup.workers_running
MIB = 1024 * 1024
GIB = 1024 * MIB
PY = sys.executable


# ---------- 假的顯示卡 ----------

def card(total_mb, usable_mb=None, used_mb=400, **kw):
    reserved = total_mb - usable_mb if usable_mb is not None else 300
    info = {"name": f"NVIDIA GeForce Fake {total_mb // 1024}GB", "driver": "595.79", "cuda_version": 13020, "cuda": "13.2",
            "arch": [8, 9], "compute_cap": "8.9", "count": 1, "total_mb": total_mb, "reserved_mb": reserved,
            "used_mb": used_mb, "free_mb": total_mb - reserved - used_mb, "source": "fake"}
    info.update(kw)
    return info


def fake_gpu(value):
    syscheck.set_fake(value)
    gpu._cache.update(t=0.0, info=None)


def real_gpu():
    syscheck.set_fake()
    gpu._cache.update(t=0.0, info=None)


# ---------- 暫存資料夾、資料庫 ----------

class Env:
    def __init__(self):
        self.models, self.dict, self.work = config.MODELS_DIR, config.DICT_DIR, config.WORK_DIR
        for d in (self.models, self.dict, self.work):
            assert str(d).startswith(str(BASE)), d                       # 絕對不能動到真的 models、data
            shutil.rmtree(d, ignore_errors=True)
            d.mkdir(parents=True, exist_ok=True)
        self._saved = []
        self._saved_conn = db._conn
        self.patch(db, "DB_PATH", RUN / "data" / f"test-{time.time_ns()}.db")
        db.init()
        from app import vocab
        self.patch(vocab, "kick", lambda: None)
        self.patch(models, "disk_free", lambda path=None: 500 * GIB)
        self.patch(setup, "MONITOR", setup.Monitor())
        self.patch(setup.MONITOR, "start", lambda boot=False: None)     # 監看執行緒由測試自己呼叫 tick
        self.patch(setup, "workers_running", lambda: True)
        self.patch(setup, "net_ok", lambda url, timeout=10: True)
        self.patch(jobs, "DOWNLOAD_RETRY_WAITS", {})
        self.envs = {}
        fake_gpu(card(8188, 7988))
        self._levels = []
        for name in ("setup", "jobs", "models", "safepath", "dict_build", "model_download", "furigana"):
            logger = logging.getLogger(name)
            self._levels.append((logger, logger.level))
            logger.setLevel(logging.CRITICAL)

    def patch(self, obj, name, value):
        self._saved.append((obj, name, getattr(obj, name)))
        setattr(obj, name, value)

    def setenv(self, key, value):
        self.envs.setdefault(key, os.environ.get(key))
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value

    def close(self):
        for obj, name, value in reversed(self._saved):
            setattr(obj, name, value)
        for key, value in self.envs.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        for logger, level in self._levels:
            logger.setLevel(level)
        real_gpu()
        jobs.download_progress.clear()
        db._conn.close()
        db._conn = self._saved_conn
        for d in (self.models, self.dict, self.work):
            shutil.rmtree(d, ignore_errors=True)


def run(fn):
    env = Env()
    try:
        fn(env)
    finally:
        env.close()


def fake_install(mid: str, variant: str | None = None):
    """在暫存的 models 放一個「下載完成」的假模型（檔案加完成標記，跟 model_download.run 寫的一樣）。"""
    entry = config.MODEL_CATALOG[mid]
    d = Path(entry["dir"])
    if variant:
        rels = [entry["variants"][variant]["file"]]
    elif entry.get("files"):
        rels = list(entry["files"])
    elif entry.get("kind") == "gguf":
        rels = [entry["file"]]
    else:
        rels = ["model.safetensors"]
    for rel in rels:
        (d / rel).parent.mkdir(parents=True, exist_ok=True)
        (d / rel).write_bytes(b"x" * 10)
    md.write_complete(entry, variant, {"files": {rel: 10 for rel in rels}})


def fake_dict(lang: str):
    d = config.DICT_DIR
    (d / dict_build.DB_NAMES[lang]).write_bytes(b"x")
    if lang == "ja":
        (d / dict_build.RULES_NAME).write_text("{}", encoding="utf-8")


def wait(cond, timeout=30.0, what="條件"):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return
        time.sleep(0.05)
    raise AssertionError(f"等不到{what}")


def status() -> str | None:
    return (setup.load() or {}).get("status")


def item(key: str) -> dict:
    return next(v for v in setup.status_view()["items"] if v["key"] == key)


# ---------- 選模型 ----------

def test_choose_tiers_match_the_spec():
    def body(env: Env):
        table = [  # (total, usable) → (版本, ctx, parallel, 需要, 判定)
            ((8188, 7988), ("Q6_K", 4096, 2, 6578, "tight")),
            ((10240, 9990), ("Q8_0", 4096, 2, 8312, "ok")),
            ((12282, 12032), ("Q8_0", 8192, 4, 8812, "ok")),
            ((16303, 15997), ("Q8_0", 8192, 4, 8812, "ok")),
            ((24564, 24114), ("Q8_0", 8192, 4, 8812, "ok")),
        ]
        for (total, usable), (variant, ctx, parallel, need, fit) in table:
            choice, block = setup.choose(card(total, usable))
            assert block is None, (total, block)
            got = (choice["variant"], choice["ctx"], choice["parallel"], choice["vram_mb"], choice["fit"])
            assert got == (variant, ctx, parallel, need, fit), (total, got)
            assert choice["budget_mb"] == usable - 1024 and choice["id"] == "hymt2-7b"
            params = {"ctx": ctx, "parallel": parallel}
            assert need == config.translator_vram_mb("hymt", cfg=config.translator_cfg("hymt", variant, params))
            assert choice["low_params"] == (ctx == 4096)
    run(body)


def test_choose_boundaries():
    def body(env: Env):
        cases = {
            7023: None, 7024: ("Q5_K_M", 4096, "tight"), 7345: ("Q5_K_M", 4096, "tight"),
            7346: ("Q5_K_M", 4096, "ok"), 7601: ("Q5_K_M", 4096, "ok"), 7602: ("Q6_K", 4096, "tight"),
            8101: ("Q6_K", 4096, "tight"), 8102: ("Q6_K", 4096, "ok"), 8601: ("Q6_K", 4096, "ok"),
            8602: ("Q6_K", 8192, "ok"), 9335: ("Q6_K", 8192, "ok"), 9336: ("Q8_0", 4096, "tight"),
            9835: ("Q8_0", 4096, "tight"), 9836: ("Q8_0", 4096, "ok"), 10335: ("Q8_0", 4096, "ok"),
            10336: ("Q8_0", 8192, "ok"),
        }
        for usable, want in cases.items():
            choice, block = setup.choose(card(max(8188, usable + 256), usable))    # 只看 usable（total 夠 7700）
            if want is None:
                assert choice is None and block["code"] == "vram_small", (usable, choice, block)
                continue
            assert block is None, (usable, block)
            got = (choice["variant"], choice["ctx"], choice["fit"])
            assert got == want, (usable, got, want)
            params = {"ctx": choice["ctx"], "parallel": choice["parallel"]}
            assert choice["vram_mb"] == config.translator_vram_mb(
                "hymt", cfg=config.translator_cfg("hymt", choice["variant"], params))
            assert choice["ctx"] // choice["parallel"] == 2048            # 每個 slot 一樣長
        # 型錄或門檻改了、7B 都放不下時：改用 Hy-MT2-1.8B，設定也改成用它
        env.patch(setup, "QUALITY_ORDER", ())
        choice, block = setup.choose(card(8188, 7988))
        assert block is None and (choice["id"], choice["variant"], choice["vram_mb"], choice["fit"]) == (
            "hymt2-1.8b", None, 2720, "ok"), choice
        new, changes = setup.settings_changes(choice)
        assert new == {"default_translator": {"ja": "hymt-mini", "en": "hymt-mini"}}, new
        assert [c["text"] for c in changes] == ["翻譯改用 Hy-MT2-1.8B"]
    run(body)


def test_blocked_cards_have_codes_and_short_texts():
    def body(env: Env):
        cases = [
            (None, "no_gpu", "沒有找到 NVIDIA 顯示卡"),
            (card(12282, 12032, driver="552.22", cuda_version=12040, cuda="12.4"), "driver_old", "目前 552.22"),
            (card(8192, 7990, name="NVIDIA GeForce GTX 1080", arch=[6, 1]), "arch_old", "這張顯示卡（GTX 1080）太舊"),
            (card(6144, 5950), "vram_small", "顯存只有 6 GB"),
            (card(7700, 7000), "vram_small", "顯存只有 8 GB"),               # 預算 5976 < 辨識要的 6000
        ]
        for info, code, text in cases:
            choice, block = setup.choose(info)
            assert choice is None and block["code"] == code, (code, block)
            assert text in block["reason"] and block["fix"], block
            assert "—" not in block["reason"] + block["fix"]
        env.setenv("CUDA_VISIBLE_DEVICES", "-1")
        choice, block = setup.choose(card(12282, 12032))
        assert block["code"] == "gpu_hidden" and "CUDA_VISIBLE_DEVICES" in block["reason"], block
        env.setenv("CUDA_VISIBLE_DEVICES", None)
        # 驅動 580 到 594：不擋，只提醒
        info = card(8188, 7988, driver="585.10", cuda_version=13000, cuda="13.0")
        choice, block = setup.choose(info)
        assert block is None and setup.gpu_view(info)["driver_warn"] is True
        assert setup.gpu_view(card(8188, 7988))["driver_warn"] is False
        assert setup.gpu_view(card(8188, 7988)) == {
            "name": "NVIDIA GeForce Fake 7GB", "gb": 8, "total_mb": 8188, "usable_mb": 7988, "driver": "595.79",
            "cuda": "13.2", "arch": "8.9", "driver_warn": False}
        assert syscheck.problem_code(None)[0] == "no_gpu" and syscheck.problem(None) == syscheck.problem_code(None)[1]
        # 不能裝：狀態存 blocked，一個任務都不排，設定不動
        fake_gpu(None)
        with setup._lock:
            setup._save(setup._new_state("pending", "new"))
        st = setup.plan_and_start("new")
        assert st["status"] == "blocked" and st["block"]["code"] == "no_gpu" and db.list_jobs() == []
        assert settings.all_values()["model_variants"] == {}
    run(body)


# ---------- 新舊使用者 ----------

def test_existing_install_detection_and_off_switches():
    def body(env: Env):
        assert setup.existing_install() is None
        mid = db.add_media(title="x", source="url", url="https://youtu.be/x")
        assert setup.existing_install() == "media"
        db.delete_media(mid)
        jid = db.add_job(None, "download", {})
        db.update_job(jid, status="failed")
        assert setup.existing_install() == "jobs"
        db.delete_job(jid)
        assert setup.existing_install() is None
        cat = config.MODEL_CATALOG
        for path, want in [
            (cat["qwen-asr"]["dir"] / "x.safetensors", "model:qwen-asr"),
            (cat["hymt2-7b"]["dir"] / "HY-MT2-7B-Q6_K.gguf.part", "model:hymt2-7b"),          # 只有下載到一半的
            (cat["bs-roformer"]["dir"] / (cat["bs-roformer"]["file"] + ".part"), "model:bs-roformer"),
            (config.DICT_DIR / "jmdict.db", "dict"),
        ]:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"x")
            assert setup.existing_install() == want, (path, setup.existing_install())
            path.unlink()
        assert setup.existing_install() is None
        (cat["aligner"]["dir"]).mkdir(parents=True, exist_ok=True)       # 空的資料夾不算
        assert setup.existing_install() is None
        # .dev 和 VS_SETUP_AUTO
        dev = RUN / "devroot"
        dev.mkdir(parents=True, exist_ok=True)
        (dev / ".dev").write_text("", encoding="utf-8")
        assert setup.auto_off(env={}, root=dev) is True
        assert setup.auto_off(env={"VS_SETUP_AUTO": "1"}, root=dev) is False
        assert setup.auto_off(env={"VS_SETUP_AUTO": "0"}, root=RUN) is True
        assert setup.auto_off(env={}, root=RUN) is False
        if (ROOT / ".dev").exists():
            assert setup.auto_off(env={}) is True                           # 擁有者的開發資料夾
        # off：什麼都不寫
        env.setenv("VS_SETUP_AUTO", "0")
        setup.startup()
        assert setup.load() is None and setup.status_view()["status"] == "off"
        env.setenv("VS_SETUP_AUTO", "1")
        env.setenv("VS_NO_WORKERS", "1")
        setup.startup()
        assert setup.load() is None
        env.setenv("VS_NO_WORKERS", None)
        # 已經在用：記成 done / existing，不排任務，也不會出現進度頁
        fake_dict("ja")
        setup.startup()
        st = setup.load()
        assert (st["status"], st["reason"]) == ("done", "existing") and db.list_jobs() == []
        assert setup.summary_view()["status"] == "done" and setup.status_view()["ready"] is None
        # 判定一次就記住：之後刪掉字典也不會自動裝
        (config.DICT_DIR / "jmdict.db").unlink()
        setup.startup()
        assert setup.load()["reason"] == "existing" and db.list_jobs() == []
    run(body)


OWNER_SCRIPT = r"""
import json, sys, time
sys.path.insert(0, sys.argv[1])
from app import db, setup, syscheck
db.init()
syscheck.set_fake({"name": "Fake", "driver": "595.79", "cuda_version": 13020, "cuda": "13.2", "arch": [8, 9],
                   "count": 1, "total_mb": 8188, "reserved_mb": 200, "used_mb": 400, "free_mb": 7588})
setup.MONITOR.start = lambda boot=False: None
setup.startup()
time.sleep(1.0)
print(json.dumps({"why": setup.existing_install(), "state": setup.load(), "jobs": len(db.list_jobs())}))
"""


def test_owner_models_folder_is_existing_and_untouched():
    """真的 models 資料夾（唯讀）＋暫存 data：判成已經在用，不排任務；開發資料夾沒設 VS_SETUP_AUTO 時整個關掉。"""
    if not REAL_MODELS.is_dir() or not any(REAL_MODELS.iterdir()):
        print("  （這台電腦沒有 models，跳過）")
        return
    before = fk.rel_listing(REAL_MODELS)
    data = RUN / "owner-data"
    base_env = {k: v for k, v in os.environ.items() if not k.startswith("VS_")}
    base_env.update(PYTHONNOUSERSITE="1", PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1",
                    VS_DATA_DIR=str(data), VS_MODELS_DIR=str(REAL_MODELS), VS_DICT_SOURCE_BASE="http://127.0.0.1:9",
                    HF_ENDPOINT="http://127.0.0.1:9")
    try:
        out = subprocess.run([PY, "-s", "-c", OWNER_SCRIPT, str(ROOT)], cwd=str(ROOT), capture_output=True, text=True,
                             encoding="utf-8", errors="replace", timeout=180, env={**base_env, "VS_SETUP_AUTO": "1"})
        got = json.loads(out.stdout.strip().splitlines()[-1])
        assert got["why"] and got["why"].startswith("model:"), got
        assert (got["state"]["status"], got["state"]["reason"], got["jobs"]) == ("done", "existing", 0), got
        if (ROOT / ".dev").exists():
            shutil.rmtree(data, ignore_errors=True)
            out = subprocess.run([PY, "-s", "-c", OWNER_SCRIPT, str(ROOT)], cwd=str(ROOT), capture_output=True,
                                 text=True, encoding="utf-8", errors="replace", timeout=180, env=base_env)
            got = json.loads(out.stdout.strip().splitlines()[-1])
            assert got["state"] is None and got["jobs"] == 0, got           # .dev、沒設 VS_SETUP_AUTO：off，不寫資料庫
    finally:
        shutil.rmtree(data, ignore_errors=True)
    assert fk.rel_listing(REAL_MODELS) == before                             # models 沒有任何變動


def test_off_folder_still_watches_an_unfinished_install():
    """有 .dev（off）的資料夾在設定頁按「重新偵測並安裝建議模型」，沒裝完就重開：還是要開監看執行緒，
    不然裝好了也一直停在「安裝中」、失敗的項目不會自動重試（2026-09-17 擁有者保護測試發現）。"""
    def body(env: Env):
        started = []
        env.patch(setup.MONITOR, "start", lambda boot=False: started.append(boot))
        env.setenv("VS_SETUP_AUTO", "0")
        setup.startup()
        assert started == [] and setup.load() is None                   # 沒有安裝在跑：什麼都不做、不寫資料庫
        ids = _start_install(env)
        setup._update(reason="redetect")
        started.clear()
        n = len(db.list_jobs())
        setup.startup()                                                 # 重開程式
        time.sleep(0.3)
        assert started == [True] and len(db.list_jobs()) == n and status() == "installing"
        # 監看執行緒接手：裝好了就標記完成
        for mid, variant in (("tsqyomi", None), ("aligner", None), ("qwen-asr", None), ("hymt2-7b", "Q6_K")):
            fake_install(mid, variant)
        fake_dict("ja")
        fake_dict("en")
        for jid in ids.values():
            db.update_job(jid, status="done")
        assert setup.MONITOR.tick(now=1.0) is False and status() == "done"
        started.clear()
        setup.startup()
        assert started == []                                            # 裝完了：不再監看
        # 安裝途中磁碟滿了轉成的 blocked：一樣要監看（空間夠了接著裝）；一開始就不能裝的 blocked 不用
        setup._update(status="blocked", block={"code": "disk", "values": {}})
        setup.startup()
        assert started == [True]
        started.clear()
        setup._update(items=[])
        setup.startup()
        assert started == [] and status() == "blocked"
        env.setenv("VS_NO_WORKERS", "1")                                # 不跑佇列的伺服器：不監看
        setup._update(status="installing", items=[{"key": "dict:ja", "kind": "dict", "lang": "ja", "job_id": None}])
        setup.startup()
        assert started == []
    run(body)


def test_progress_speed_and_eta():
    """總進度那一行（2026-09-17 乾淨安裝實測的問題）：剛開始下載的幾秒不估剩餘時間（曾經報 5 小時）；
    都下載完、只剩字典在建時不再沿用下載速度；字典建置算進進度，不會停在 99% 好幾分鐘。"""
    def body(env: Env):
        # 每一項：任務剛開始下載的幾秒不顯示剩餘時間（實測剛開始時對齊模型寫「約 47 分鐘」，實際半分鐘）
        ids = _start_install(env)
        jid = ids["model:aligner"]
        assert db.claim_job(jid)
        jobs.download_progress[jid] = {"done": 10 * MIB, "total": 1755 * MIB, "speed": 7 * MIB}
        assert item("model:aligner")["state"] == "downloading" and item("model:aligner")["eta_s"] is None
        db.update_job(jid, started_at=time.time() - 30)
        assert item("model:aligner")["eta_s"] == int((1755 * MIB - 10 * MIB) / (7 * MIB)), item("model:aligner")
        jobs.download_progress[jid]["checking"] = True
        assert item("model:aligner")["eta_s"] is None
        jobs.download_progress.pop(jid)
        # 總速度：剛開始的幾秒不算
        mon = setup.Monitor()
        env.patch(mon, "start", lambda boot=False: None)
        env.patch(setup, "MONITOR", mon)
        now = 1000.0
        mon._sample_speed(0.8 * MIB, now)                               # 連線剛建立，速度偏低
        assert mon.speed_bps(now) == 0 and mon.speed_bps(now + 9) == 0
        mon._sample_speed(40 * MIB, now + 4)
        mon._sample_speed(60 * MIB, now + 8)
        mon._sample_speed(61 * MIB, now + 9.5)
        assert mon.speed_bps(now + 9.5) == 0
        assert mon.speed_bps(now + 10) == 61 * MIB                      # 暖機完：用最新的速度，不跟一開始的 0.8 平均
        mon._sample_speed(59 * MIB, now + 12)
        assert 59 * MIB < mon.speed_bps(now + 12) < 61 * MIB
        mon._sample_speed(0, now + 14)                                  # 換下一個模型：沿用
        assert mon.speed_bps(now + 14) > 59 * MIB
        mon._sample_speed(0, now + 80)                                  # 停了超過 60 秒：歸零，再開始時重新暖機
        assert mon.speed_bps(now + 80) == 0
        mon._sample_speed(30 * MIB, now + 82)
        assert mon.speed_bps(now + 85) == 0 and mon.speed_bps(now + 92) == 30 * MIB

        def model(size, done, state):
            return {"kind": "model", "had": False, "state": state, "size_bytes": size, "done_bytes": done,
                    "speed_bps": 0, "eta_s": None}

        def dictv(lang, state, eta=None, progress=None):
            v = {"kind": "dict", "lang": lang, "had": False, "state": state, "size_bytes": 50 * MIB,
                 "done_bytes": 0 if state in ("waiting", "downloading") else 50 * MIB, "speed_bps": 0, "eta_s": eta}
            if progress is not None:
                v["progress"] = progress
            return v

        t = now + 200
        mon._sample_speed(60 * MIB, t - 20)
        mon._sample_speed(60 * MIB, t)
        # 還在下載模型：剩下的位元組 ÷ 速度，跟字典的時間取大的
        views = [model(8 * GIB, 2 * GIB, "downloading"), dictv("ja", "done", progress=1.0), dictv("en", "downloading", eta=200)]
        tot = setup._totals(views, t)
        assert tot["speed_bps"] == 60 * MIB and tot["eta_s"] == max(int((6 * GIB + 50 * MIB) / (60 * MIB)), 200), tot
        # 模型都下載完、英文字典在建：速度 0（畫面不再寫 MB/s），剩餘時間是字典的，進度照建置做了幾成往上走
        before = None
        for p in (0.20, 0.47, 0.80, 0.95):
            views = [model(8 * GIB, 8 * GIB, "done"), dictv("ja", "done", progress=1.0), dictv("en", "building", eta=90, progress=p)]
            tot = setup._totals(views, t)
            assert tot["speed_bps"] == 0 and tot["eta_s"] == 90 and tot["done_bytes"] == tot["total_bytes"], tot
            assert tot["percent"] <= 99 and (before is None or tot["percent"] > before), (p, tot, before)
            before = tot["percent"]
        views[2] = dictv("en", "building", eta=90, progress=0.20)
        assert setup._totals(views, t)["percent"] <= 93                  # 建置剛開始時還有明顯的一段
        views[2] = dictv("en", "done", progress=1.0)
        assert setup._totals(views, t)["percent"] == 100
        # 剛開始下載（速度還在暖機）：不估剩餘時間，畫面寫「計算中」
        fresh = setup.Monitor()
        env.patch(fresh, "start", lambda boot=False: None)
        env.patch(setup, "MONITOR", fresh)
        fresh._sample_speed(0.8 * MIB, t)
        tot = setup._totals([model(8 * GIB, 0, "downloading"), dictv("en", "waiting", eta=300)], t + 2)
        assert tot["eta_s"] is None and tot["speed_bps"] == 0, tot
    run(body)


def test_disk_text_matches_the_web_page():
    """不能裝的畫面：後端的說明（按「重新檢查」時的訊息）跟網頁算的數字一樣，要清出的空間無條件進位。
    2026-09-17 實測：要 14,858,226,336 bytes、剩 8 GB，網頁寫 5.9 GB、重新檢查的訊息寫 5.8 GB。"""
    need, free = 14_858_226_336, 8 * GIB
    b = setup._block("disk", None, drive="C:", need_bytes=need, free_bytes=free, short_bytes=need - free)
    assert b["reason"] == "C: 空間不夠：要 13.8 GB，只剩 8.0 GB。" and b["fix"] == "清出 5.9 GB 以上再按「重新檢查」。", b
    assert setup._gb(GIB, up=True) == "1.0 GB" and setup._gb(GIB + 1, up=True) == "1.1 GB" and setup._gb(0) == "0.0 GB"
    assert setup._gb(500 * MIB) == "0.5 GB" and setup._gb(512 * MIB) == "0.5 GB"


# ---------- 排任務 ----------

def _model_jobs():
    return sorted([j for j in db.list_jobs(finished_limit=0) if j["type"] == "model"], key=lambda j: j["position"])


def test_new_user_8gb_enqueues_in_order_and_decision_sticks():
    def body(env: Env):
        setup.startup()
        wait(lambda: status() == "installing", what="開始安裝")
        st = setup.load()
        assert st["reason"] == "new" and not st["dismissed"] and st["started_at"]
        assert (st["choice"]["variant"], st["choice"]["ctx"], st["choice"]["fit"]) == ("Q6_K", 4096, "tight")
        keys = [i["key"] for i in st["items"]]
        assert keys == ["model:tsqyomi", "model:aligner", "model:qwen-asr", "model:hymt2-7b:Q6_K", "dict:ja", "dict:en"], keys
        mj = _model_jobs()
        assert [(j["params"]["model"], j["params"].get("variant"), j["params"]["setup"]) for j in mj] == [
            ("tsqyomi", None, True), ("aligner", None, True), ("qwen-asr", None, True), ("hymt2-7b", "Q6_K", True)]
        assert mj[3]["params"]["label"] == "Hy-MT2-7B Q6_K"
        dj = [j for j in db.list_jobs(finished_limit=0) if j["type"] == "dict"]
        assert sorted((j["params"]["lang"], j["params"]["setup"], j["params"]["force_download"]) for j in dj) == [
            ("en", True, False), ("ja", True, False)]
        values = settings.all_values()
        assert values["model_variants"] == {"hymt2-7b": "Q6_K"} and values["llm_params"] == {"hymt": {"ctx": 4096, "parallel": 2}}
        assert st["applied"]["model_variants"] == {"hymt2-7b": "Q6_K"}
        assert config.translator_cfg("hymt")["variant"] == "Q6_K" and config.translator_cfg("hymt")["ctx"] == 4096
        n = len(db.list_jobs())
        setup.startup()                                     # 重開程式：不會再排一次
        time.sleep(0.3)
        assert len(db.list_jobs()) == n and status() == "installing"
        fake_install("qwen-asr")                            # 判定記住了：之後有模型也照樣裝完
        setup.startup()
        time.sleep(0.3)
        assert status() == "installing" and setup.load()["reason"] == "new" and len(db.list_jobs()) == n
        again = setup.enqueue(setup.planned_items(st["choice"]))       # 已經在佇列裡的沿用同一個任務
        by_key = {i["key"]: i for i in again}
        assert by_key["model:qwen-asr"]["had"] and by_key["model:qwen-asr"]["job_id"] is None
        assert by_key["model:aligner"]["job_id"] == st["items"][1]["job_id"] and len(db.list_jobs()) == n
        # 畫面的狀態
        view = setup.status_view()
        assert view["gpu"]["gb"] == 8 and view["choice"]["label"] == "Hy-MT2-7B" and view["disk"]["drive"]
        assert [v["state"] for v in view["items"]] == ["waiting", "waiting", "done", "waiting", "waiting", "waiting"]
        assert [v["use"] for v in view["items"]] == ["日文假名", "時間軸", "語音辨識", "翻譯", "日文字典", "英文字典"]
        assert [v["label"] for v in view["items"]][:4] == ["tsqyomi", "Qwen3-ForcedAligner-0.6B", "Qwen3-ASR-1.7B", "Hy-MT2-7B Q6_K"]
        assert view["ready"] == {"transcribe": False, "translate": False, "dict_ja": False, "dict_en": False}
        assert 0 <= view["totals"]["percent"] < 100 and view["totals"]["eta_s"] is None
        summary = setup.summary_view()
        assert summary["status"] == "installing" and summary["percent"] == view["totals"]["percent"]
        assert not summary["problem"] and not summary["net_wait"] and summary["ready"] == {"transcribe": False, "translate": False}
        setup.api_dismiss()
        assert setup.load()["dismissed"] is True
    run(body)


def test_12gb_uses_q8_default_params_and_keeps_other_llm_params():
    def body(env: Env):
        fake_gpu(card(12282, 12032))
        settings.save({"llm_params": {"hymt": {"ctx": 4096, "parallel": 2}, "sakura": {"ctx": 4096, "parallel": 2}}})
        plan = setup.make_plan(gpu.query(max_age=0))
        assert [c["text"] for c in plan["changes"]] == ["翻譯參數改回預設"], plan["changes"]    # Q8_0 本來就是預設版本
        setup.startup()
        wait(lambda: status() == "installing", what="開始安裝")
        st = setup.load()
        assert (st["choice"]["variant"], st["choice"]["ctx"], st["choice"]["parallel"]) == ("Q8_0", 8192, 4)
        assert st["items"][3]["key"] == "model:hymt2-7b:Q8_0"
        values = settings.all_values()
        assert values["llm_params"] == {"sakura": {"ctx": 4096, "parallel": 2}}, values["llm_params"]
        assert values["model_variants"] == {"hymt2-7b": "Q8_0"}
    run(body)


def test_disk_not_enough_blocks_then_recheck_installs():
    def body(env: Env):
        env.patch(models, "disk_free", lambda path=None: 5 * GIB)
        setup.startup()
        wait(lambda: status() == "blocked", what="blocked")
        st = setup.load()
        b = st["block"]
        assert b["code"] == "disk" and b["values"]["free_bytes"] == 5 * GIB, b
        need = (77 + 1755 + 4485) * MIB + 6164482720 + (200 + 750) * MIB + GIB
        assert b["values"]["need_bytes"] == need and b["values"]["short_bytes"] == need - 5 * GIB
        # 差 8.84 GB：要清出的空間無條件進位（清出 8.8 GB 其實不夠），跟網頁 app.js 的寫法一樣
        assert b["reason"].endswith("空間不夠：要 13.8 GB，只剩 5.0 GB。") and b["fix"] == "清出 8.9 GB 以上再按「重新檢查」。", b
        assert db.list_jobs() == [] and settings.all_values()["model_variants"] == {}
        assert setup.summary_view()["blocked"] is True
        try:
            setup.api_recheck()
        except setup.SetupError as e:
            assert (e.status, e.code) == (400, "disk") and "status" in e.extra
        else:
            raise AssertionError("空間還是不夠")
        env.patch(models, "disk_free", lambda path=None: 500 * GIB)
        view = setup.api_recheck()
        assert view["status"] == "installing" and len(_model_jobs()) == 4
        # 裝完剩不到 10 GB：提醒，不擋
        env.patch(models, "disk_free", lambda path=None: 18 * GIB)
        assert setup.status_view()["disk"]["warn"] is True
    run(body)


# ---------- 監看執行緒 ----------

def _start_install(env: Env) -> dict:
    with setup._lock:
        setup._save(setup._new_state("pending", "new"))
    st = setup.plan_and_start("new")
    assert st["status"] == "installing", st
    return {i["key"]: i["job_id"] for i in st["items"]}


def _fail(jid, code, at):
    db.update_job(jid, status="failed", stage="失敗", error=f"x（錯誤代碼 {code}）", result={"error_code": code},
                  finished_at=at)


def test_monitor_waits_for_network_retries_and_finishes():
    def body(env: Env):
        ids = _start_install(env)
        mon = setup.MONITOR
        net = {"ok": False, "urls": []}
        env.patch(setup, "net_ok", lambda url, timeout=10: net["urls"].append(url) or net["ok"])
        # 網路斷了：等網路，連得上才重試（同一個任務，從 .part 接著下載）
        q = ids["model:qwen-asr"]
        _fail(q, "network", 1000.0)
        assert mon.tick(now=100.0)
        v = item("model:qwen-asr")
        assert v["state"] == "net_wait" and v["error"]["code"] == "network", v
        assert net["urls"] == ["http://127.0.0.1:9/api/models/Qwen/Qwen3-ASR-1.7B"]
        assert "qwen-asr" in setup.pending_models() and setup.summary_view()["net_wait"]
        assert setup.item_view(setup.load()["items"][2], now=110.0)["error"]["next_check_s"] == 20
        mon.tick(now=131.0)
        assert db.get_job(q)["status"] == "failed" and len(net["urls"]) == 2
        net["ok"] = True
        mon.tick(now=150.0)
        assert db.get_job(q)["status"] == "failed"               # 還沒到下次檢查的時間（60 秒）
        mon.tick(now=192.0)
        assert db.get_job(q)["status"] == "queued" and item("model:qwen-asr")["state"] == "waiting"
        # 字典連不上時測的是字典來源的主機
        d = ids["dict:en"]
        _fail(d, "timeout", 1001.0)
        net["ok"] = False
        mon.tick(now=200.0)
        assert net["urls"][-1] == "http://127.0.0.1:9/" and item("dict:en")["state"] == "net_wait"
        # 要登入、找不到：不自動重試
        a = ids["model:aligner"]
        _fail(a, "gated", 1002.0)
        mon.tick(now=210.0)
        mon.tick(now=5000.0)
        assert db.get_job(a)["status"] == "failed" and item("model:aligner")["state"] == "failed"
        assert "aligner" not in setup.pending_models() and setup.summary_view()["problem"]
        # 檔案被鎖住：60 秒後重試，最多 3 次
        t = ids["model:hymt2-7b:Q6_K"]
        at, now = 2000.0, 6000.0
        for attempt in range(3):
            _fail(t, "file_locked", at + attempt)
            mon.tick(now=now)
            assert db.get_job(t)["status"] == "failed" and item("model:hymt2-7b:Q6_K")["state"] == "retry_wait"
            now += 61
            mon.tick(now=now)
            assert db.get_job(t)["status"] == "queued", attempt
        _fail(t, "file_locked", at + 10)
        mon.tick(now=now + 1)
        mon.tick(now=now + 1000)
        assert db.get_job(t)["status"] == "failed" and item("model:hymt2-7b:Q6_K")["state"] == "failed"
        # 使用者取消、移除：不再自動排
        db.update_job(ids["model:tsqyomi"], status="canceled", stage="已取消", finished_at=1.0)
        db.delete_job(ids["dict:ja"])
        mon.tick(now=now + 1001)
        assert item("model:tsqyomi")["state"] == "canceled" and item("dict:ja")["state"] == "canceled"
        # 重開程式：失敗、可以重試的項目立刻重試一次（不測網路、不等）
        boot = setup.Monitor()
        env.patch(setup, "MONITOR", boot)
        _fail(q, "network", 3000.0)
        boot.boot = True
        boot.tick(now=1.0)
        assert db.get_job(q)["status"] == "queued" and db.get_job(a)["status"] == "failed"
        # 全部裝好（取消的算結束）：標記完成
        for mid, variant in (("aligner", None), ("qwen-asr", None), ("hymt2-7b", "Q6_K")):
            fake_install(mid, variant)
        db.update_job(q, status="done")
        fake_dict("en")
        assert status() == "installing"
        assert boot.tick(now=2.0) is False
        st = setup.load()
        assert st["status"] == "done" and st["finished_at"] and setup.pending_models() == set()
        view = setup.status_view()
        assert [v["state"] for v in view["items"]] == ["canceled", "done", "done", "done", "canceled", "done"]
        assert view["totals"]["percent"] == 100 and view["ready"]["transcribe"] and view["ready"]["translate"]
    run(body)


def test_monitor_disk_full_blocks_and_continues_when_space_is_back():
    def body(env: Env):
        ids = _start_install(env)
        mon = setup.MONITOR
        q = ids["model:qwen-asr"]
        _fail(q, "disk_full", 10.0)
        env.patch(models, "disk_free", lambda path=None: 3 * GIB)
        mon.tick(now=100.0)
        st = setup.load()
        assert st["status"] == "blocked" and st["block"]["code"] == "disk" and st["items"], st
        assert "qwen-asr" in setup.pending_models()                   # 空間夠了會接著裝，影片照樣排隊等
        mon.tick(now=120.0)                                           # 還沒到 60 秒
        mon.tick(now=161.0)                                           # 檢查了，還是不夠：數字更新
        assert status() == "blocked" and db.get_job(q)["status"] == "failed"
        env.patch(models, "disk_free", lambda path=None: 400 * GIB)
        mon.tick(now=162.0)
        assert status() == "blocked"
        mon.tick(now=222.0)
        assert status() == "installing" and db.get_job(q)["status"] == "queued" and setup.load()["block"] is None
        # 使用者按「重新檢查」也可以馬上接著裝
        _fail(q, "disk_full", 20.0)
        mon.tick(now=300.0)
        assert status() == "blocked"
        assert setup.api_recheck()["status"] == "installing" and db.get_job(q)["status"] == "queued"
    run(body)


# ---------- 安裝中新增的影片 ----------

def test_new_videos_wait_for_models_being_installed():
    def body(env: Env):
        env.patch(server, "_require_media_tools", lambda *names: None)
        fake_install("aligner")
        client = TestClient(server.app)
        add = {"source": "url", "url": "https://youtu.be/abcdefghijk", "language": "ja"}
        r = client.post("/api/media", json=add)
        assert r.status_code == 400 and r.json()["detail"] == "Qwen3-ASR-1.7B 還沒下載，請到設定頁下載", r.json()
        mj = db.add_job(None, "model", {"model": "qwen-asr", "label": "Qwen3-ASR-1.7B", "setup": True})
        r = client.post("/api/media", json={**add, "translate": True})
        assert r.status_code == 400 and "Hy-MT2-7B 還沒下載" in r.json()["detail"]
        tj_model = db.add_job(None, "model", {"model": "hymt2-7b", "variant": "Q6_K", "label": "Hy-MT2-7B Q6_K", "setup": True})
        r = client.post("/api/media", json={**add, "translate": True})
        assert r.status_code == 200, r.json()
        media_id = r.json()["id"]
        by_type = {j["type"]: j for j in db.jobs_for_media(media_id)}
        db.update_job(by_type["download"]["id"], status="done")
        state = client.get("/api/state").json()
        jobs_by_id = {j["id"]: j for j in state["jobs"]}
        tr = jobs_by_id[by_type["transcribe"]["id"]]
        assert tr["waiting"] == {"models": ["qwen-asr"], "labels": ["Qwen3-ASR-1.7B"], "state": "waiting", "eta_s": None}, tr
        tl = jobs_by_id[by_type["translate"]["id"]]
        assert tl["waiting"]["labels"] == ["Hy-MT2-7B"] and "waiting" not in jobs_by_id[mj]
        assert "setup" in state
        meta = client.get("/api/meta").json()
        assert meta["engines"]["qwen"] == {**meta["engines"]["qwen"], "installed": False, "pending": True}
        assert meta["translators"]["hymt"]["pending"] and not meta["translators"]["sakura"]["pending"]
        assert meta["engines"]["anime"]["pending"] is False
        # 顯卡執行緒先跳過，閒置釋放也不算它
        worker = jobs.Worker("gpu", jobs.GPU_TYPES)
        assert worker.pick() is None
        assert by_type["transcribe"]["id"] not in {j["id"] for j in jobs.gpu_jobs()}
        # 下載中：顯示最慢那一個的狀態和剩多久
        db.update_job(mj, status="running")
        jobs.download_progress[mj] = {"done": 100 * MIB, "total": 1100 * MIB, "speed": MIB}
        tr = {j["id"]: j for j in client.get("/api/state").json()["jobs"]}[by_type["transcribe"]["id"]]
        assert tr["waiting"]["state"] == "downloading" and tr["waiting"]["eta_s"] == 1000, tr
        jobs.pause(tj_model)
        tl = {j["id"]: j for j in client.get("/api/state").json()["jobs"]}[by_type["translate"]["id"]]
        assert tl["waiting"]["state"] == "paused"
        jobs.download_progress.pop(mj)
        # 重試：模型在下載中就放行，沒有在下載才擋
        failed = db.add_job(media_id, "transcribe", {"language": "ja", "engine": "qwen"})
        db.update_job(failed, status="failed")
        assert jobs.retry_problem(db.get_job(failed)) is None
        db.update_job(mj, status="canceled")
        assert "Qwen3-ASR-1.7B 還沒下載" in jobs.retry_problem(db.get_job(failed))
        # 裝好了：拿得到
        fake_install("qwen-asr")
        picked = worker.pick()
        assert picked and picked["type"] == "transcribe", picked
    run(body)


# ---------- API ----------

def _expect(resp, status_code, code):
    body = resp.json()
    assert resp.status_code == status_code and body.get("code") == code and body.get("detail"), (resp.status_code, body)
    return body


def test_api_error_codes():
    def body(env: Env):
        c = TestClient(server.app)
        assert c.get("/api/setup").json()["status"] == "pending"            # 還沒判斷（這個測試沒有跑 startup）
        # 沒有顯示卡
        fake_gpu(None)
        plan = c.get("/api/setup/plan").json()
        assert plan["block"]["code"] == "no_gpu" and plan["items"] == [] and plan["signature"]
        _expect(c.post("/api/setup/install", json={"signature": "nope"}), 409, "plan_changed")
        assert "plan" in c.post("/api/setup/install", json={"signature": "nope"}).json()
        b = _expect(c.post("/api/setup/install", json={"signature": plan["signature"]}), 400, "blocked")
        assert "沒有找到 NVIDIA 顯示卡" in b["detail"]
        # 磁碟不夠
        fake_gpu(card(8188, 7988))
        env.patch(models, "disk_free", lambda path=None: 2 * GIB)
        plan = c.get("/api/setup/plan").json()
        assert plan["block"]["code"] == "disk" and plan["download_bytes"] > 11.9 * GIB
        b = _expect(c.post("/api/setup/install", json={"signature": plan["signature"]}), 400, "disk")
        assert "空間不夠" in b["detail"]
        env.patch(models, "disk_free", lambda path=None: 500 * GIB)
        # 設定頁的重新偵測：確認框內容、開始安裝
        plan = c.get("/api/setup/plan").json()
        assert plan["block"] is None and not plan["nothing"] and plan["download_bytes"] > 11.9 * GIB
        assert [x["text"] for x in plan["changes"]] == ["翻譯改用 Q6_K 版", "翻譯改成一次翻比較少行"]
        assert [(i["key"], i["installed"]) for i in plan["items"]][3] == ("model:hymt2-7b:Q6_K", False)
        r = c.post("/api/setup/install", json={"signature": plan["signature"]})
        assert r.status_code == 200 and r.json()["status"] == "installing" and r.json()["reason"] == "redetect", r.json()
        _expect(c.post("/api/setup/install", json={"signature": plan["signature"]}), 409, "busy")
        _expect(c.post("/api/setup/recheck"), 409, "not_blocked")
        # 項目
        st = setup.load()
        ids = {i["key"]: i["job_id"] for i in st["items"]}
        _expect(c.post("/api/setup/items/model%3Anope/retry"), 404, "unknown_item")
        _expect(c.post("/api/setup/items/model%3Aqwen-asr/retry"), 409, "active")
        _expect(c.post("/api/setup/items/model%3Aqwen-asr/resume"), 409, "not_paused")
        assert jobs.pause(ids["model:qwen-asr"])
        r = c.post("/api/setup/items/model%3Aqwen-asr/resume")
        assert r.status_code == 200 and db.get_job(ids["model:qwen-asr"])["status"] == "queued"
        _fail(ids["model:qwen-asr"], "network", 5.0)
        real_retry = jobs.retry
        env.patch(jobs, "retry", lambda jid, force=False: (_ for _ in ()).throw(jobs.RetryRefused(400, "這個不能重試")))
        b = _expect(c.post("/api/setup/items/model%3Aqwen-asr/retry"), 400, "retry_refused")
        assert b["detail"] == "這個不能重試"
        env.patch(jobs, "retry", real_retry)
        env.patch(models, "disk_free", lambda path=None: 1 * GIB)
        _expect(c.post("/api/setup/items/model%3Aqwen-asr/retry"), 400, "disk")
        env.patch(models, "disk_free", lambda path=None: 500 * GIB)
        r = c.post("/api/setup/items/model%3Aqwen-asr/retry")
        assert r.status_code == 200 and db.get_job(ids["model:qwen-asr"])["status"] == "queued"
        db.delete_job(ids["model:aligner"])                                   # 使用者把任務移除了：重新下載排一個新的
        assert item("model:aligner")["state"] == "canceled"
        r = c.post("/api/setup/items/model%3Aaligner/retry")
        new_id = next(i["job_id"] for i in setup.load()["items"] if i["key"] == "model:aligner")
        assert r.status_code == 200 and new_id != ids["model:aligner"] and db.get_job(new_id)["status"] == "queued"
        # 字典
        _expect(c.post("/api/dicts/xx/build", json={}), 400, "bad_lang")
        _expect(c.post("/api/dicts/ja/build", json={}), 409, "duplicate")
        r = c.post("/api/dicts/zh/build", json={"force_download": True})
        assert r.status_code == 200 and db.get_job(r.json()["job_id"])["params"] == {
            "lang": "zh", "label": "中文辭典", "setup": False, "force_download": True}
        _expect(c.post("/api/dicts/zh/build", json={}), 409, "duplicate")
        jobs.cancel(r.json()["job_id"])
        env.patch(models, "disk_free", lambda path=None: 100 * MIB)
        _expect(c.post("/api/dicts/zh/build", json={}), 400, "disk")
        env.patch(models, "disk_free", lambda path=None: 500 * GIB)
        dicts = c.get("/api/dicts").json()
        assert dicts["zh"]["est_s"] == 25 and dicts["ja"]["job"]["status"] == "queued" and dicts["en"]["download_bytes"]
        # 不跑佇列的伺服器
        env.patch(setup, "workers_running", lambda: False)
        _expect(c.post("/api/setup/install", json={"signature": "x"}), 503, "no_workers")
        _expect(c.post("/api/dicts/ja/build", json={}), 503, "no_workers")
        assert c.get("/api/setup").json()["workers"] is False
        env.patch(setup, "workers_running", lambda: True)
        # 偵測出錯
        env.patch(gpu, "query", lambda max_age=2.0: (_ for _ in ()).throw(RuntimeError("boom")))
        _expect(c.get("/api/setup/plan"), 500, "detect_failed")
        with setup._lock:
            setup._save(setup._new_state("pending", "new"))
        _expect(c.post("/api/setup/recheck"), 400, "blocked")
        assert setup.load()["block"]["code"] == "detect_failed"
        env.patch(gpu, "query", lambda max_age=2.0: None)
        b = _expect(c.post("/api/setup/recheck"), 400, "blocked")
        assert b["status"]["block"]["code"] == "no_gpu"
        setup._plan_lock.acquire()
        try:
            _expect(c.post("/api/setup/recheck"), 409, "busy")
        finally:
            setup._plan_lock.release()
        assert c.post("/api/setup/dismiss").json() == {"ok": True} and setup.load()["dismissed"] is True
        # 真正的 VS_NO_WORKERS 判斷
        env.setenv("VS_NO_WORKERS", "1")
        assert REAL_WORKERS_RUNNING() is False
        env.setenv("VS_NO_WORKERS", None)
        # 什麼都裝好了：沒有要做的事
        env.patch(gpu, "query", lambda max_age=2.0: card(8188, 7988))
        setup._update(status="done")
        for mid in ("tsqyomi", "aligner", "qwen-asr"):
            fake_install(mid)
        fake_install("hymt2-7b", "Q6_K")
        fake_dict("ja")
        fake_dict("en")
        # 假的 gguf 只有 10 bytes，算顯存時照檔案大小會變成用預設參數；先把參數改成一樣，才是「沒有要做的事」
        settings.save({"llm_params": {}})
        plan = c.get("/api/setup/plan").json()
        assert plan["nothing"] and plan["download_bytes"] == 0, plan
        assert c.post("/api/setup/install", json={"signature": plan["signature"]}).json()["nothing"] is True
    run(body)


# ---------- 從頭裝一次（假的 Hugging Face、假的字典來源） ----------

class StoppableMonitor(setup.Monitor):
    def __init__(self):
        super().__init__()
        self.stopped = threading.Event()

    def _loop(self):
        while not self.stopped.is_set():
            try:
                self.tick()
            except Exception:  # noqa: BLE001
                logging.getLogger("setup").warning("monitor tick failed", exc_info=True)
            self.event.wait(setup.TICK_S)
            self.event.clear()


class TestWorker(jobs.Worker):
    def __init__(self, name, types, stop: threading.Event):
        super().__init__(name, types)
        self.stop = stop

    def run(self):
        while not self.stop.is_set():
            job = self.pick()
            if job is None:
                self.stop.wait(0.2)
                continue
            self.execute(job)


def test_end_to_end_with_fake_servers_and_network_drop():
    def body(env: Env):
        # 假的 Hugging Face：hf 模型照 revision 下載，gguf 照 main
        hf = fk.FakeFiles()
        lists, rev = {}, "0123456789abcdef0123456789abcdef01234567"

        def repo_files(repo, files, delay=0.0):
            listed = []
            for path, data in files.items():
                hf.add(f"/{repo}/resolve/{rev}/{path}", data, delay=delay)
                listed.append({"path": path, "size": len(data), "sha256": fk.sha(data)})
            lists[repo] = [rev, listed]

        repo_files("tsukumijima/tsqyomi-models", {"v4/model.onnx": os.urandom(300_000), "v4/tokenizer.json": b"{}",
                                                  "v4/metadata.json": b"{}", "README.md": b"x"})
        repo_files("Qwen/Qwen3-ForcedAligner-0.6B", {"config.json": b"{}", "model.safetensors": os.urandom(500_000)})
        asr = os.urandom(3 * MIB)
        repo_files("Qwen/Qwen3-ASR-1.7B", {"config.json": b"{}", "model.safetensors": asr}, delay=0.06)
        gguf = os.urandom(MIB)
        hf.add("/tencent/Hy-MT2-7B-GGUF/resolve/main/HY-MT2-7B-Q6_K.gguf", gguf)
        model_spec = fk.write_spec(RUN, "model-spec.json", {
            "lists": lists, "variants": {"hymt2-7b": {"Q6_K": {"bytes": len(gguf), "sha256": fk.sha(gguf)}}}})
        env.setenv("HF_ENDPOINT", hf.base)
        env.patch(jobs, "model_download_cmd", fk.model_child_cmd(model_spec))
        # 假的字典來源
        dsrv = fk.FakeFiles()
        sources = {"ja": fk.ja_sources(40)}
        if fk.can_make_7z():
            sources["en"] = {"stardict.7z": fk.make_7z({"stardict.csv": fk.stardict_csv(50)}), **fk.en_extra_sources()}
        else:
            env.patch(setup, "DICTS", ("ja",))
            print("  （沒有 py7zr 也沒有 7-Zip，只裝日文字典）")
        patched = {}
        for lang, files in sources.items():
            for name, data in files.items():
                dsrv.add(f"/{name}", data, etag='"e1"')
            for s in dict_build.SOURCES[lang]:
                if s["name"] in files and not s.get("daily"):
                    patched.setdefault(lang, {})[s["name"]] = {"bytes": len(files[s["name"]]), "sha256": fk.sha(files[s["name"]])}
        saved_sources = {lang: [dict(s) for s in dict_build.SOURCES[lang]] for lang in dict_build.SOURCES}
        for lang, files in patched.items():
            for s in dict_build.SOURCES[lang]:
                s.update(files.get(s["name"], {}))
        env.setenv("VS_DICT_SOURCE_BASE", dsrv.base)
        dict_spec = fk.write_spec(RUN, "dict-spec.json", {"min_entries": {"ja": 10, "en": 10}, "jmdict_min": 10,
                                                          "ecdict_rows": 53, "sources": patched})
        env.patch(dict_build, "child_cmd", fk.dict_child_cmd(dict_spec))
        # 監看執行緒和工作執行緒真的開起來（測完停掉）
        env.patch(setup, "TICK_S", 0.2)
        env.patch(setup, "NET_WAITS", (0.5,))
        env.patch(setup, "net_ok", REAL_NET_OK)                       # 真的去連假伺服器，斷掉時連不上
        mon = StoppableMonitor()
        env.patch(setup, "MONITOR", mon)
        stop = threading.Event()
        workers = [TestWorker("model", jobs.MODEL_TYPES, stop), TestWorker("dict", jobs.DICT_TYPES, stop)]
        for w in workers:
            w.start()
        try:
            setup.startup()
            wait(lambda: status() == "installing", what="開始安裝")
            part = config.MODEL_CATALOG["qwen-asr"]["dir"] / "model.safetensors.part"
            wait(lambda: part.exists() and part.stat().st_size > MIB, timeout=60, what="Qwen3-ASR 下載到一半")
            hf.stop()                                                 # 斷網
            wait(lambda: any(v["state"] == "net_wait" for v in setup.status_view()["items"]), timeout=60,
                 what="等網路恢復")
            assert setup.summary_view()["net_wait"] is True
            got_part = part.stat().st_size
            assert MIB < got_part < len(asr)
            time.sleep(1.0)
            hf.start()                                                # 網路恢復：不用按任何東西就接著裝
            wait(lambda: status() == "done", timeout=120, what="全部裝好")
        finally:
            stop.set()
            mon.stopped.set()
            mon.event.set()
            for w in workers:
                w.join(30)
            hf.stop()
            dsrv.stop()
            for lang, rows in saved_sources.items():
                for s, row in zip(dict_build.SOURCES[lang], rows):
                    s.clear()
                    s.update(row)
        ranges = hf.ranges(f"/Qwen/Qwen3-ASR-1.7B/resolve/{rev}/model.safetensors")
        assert ranges[0] is None and any(r and int(r.split("=")[1].rstrip("-")) > 0 for r in ranges[1:]), ranges
        assert (config.MODEL_CATALOG["qwen-asr"]["dir"] / "model.safetensors").read_bytes() == asr
        view = setup.status_view()
        assert all(v["state"] == "done" for v in view["items"]), [(v["key"], v["state"]) for v in view["items"]]
        assert view["totals"]["percent"] == 100 and view["ready"]["transcribe"] and view["ready"]["translate"]
        assert dict_build.ready("ja") and (not fk.can_make_7z() or dict_build.ready("en"))
        assert config.model_installed(config.MODEL_CATALOG["hymt2-7b"], "Q6_K")
        assert [j["status"] for j in db.list_jobs(finished_limit=0) if j["status"] != "done"] == []
    run(body)


if __name__ == "__main__":
    started = time.time()
    RUN.mkdir(parents=True, exist_ok=True)
    try:
        for name, fn in list(globals().items()):
            if name.startswith("test_") and callable(fn):
                t0 = time.time()
                fn()
                print(f"ok {name} ({time.time() - t0:.1f}s)")
        print(f"all passed in {time.time() - started:.1f}s")
    finally:
        shutil.rmtree(RUN, ignore_errors=True)
        if BASE.exists() and not any(BASE.iterdir()):
            BASE.rmdir()
