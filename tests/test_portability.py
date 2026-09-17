"""可攜性修正的測試：ffmpeg 位置、子程序 PATH、人聲分離失敗改用原音、llama-server 的 --device、
顯示卡檢查、顯存不足的說明、翻譯模型挑放得下的、區網開關、位置環境變數、相容播放檔改用 libx264、MIME 類型、
safepath 的測試根目錄。

不碰正式資料庫：要用資料庫的都換到系統暫存資料夾的 portability-test 底下，測完整個刪掉。
會真的執行的外部程式只有：ffmpeg（CPU 產生 2 秒的測試片段、用 libx264 轉檔）、nvidia-smi 和 nvml.dll（唯讀），
以及 CUDA_VISIBLE_DEVICES=-1 的 llama-server 和 torch（看不到顯示卡，一開始就結束，不會載入模型、不佔顯存）。

python -s tests/test_portability.py
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

os.environ["VS_NO_WORKERS"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import soundfile as sf  # noqa: E402
import numpy as np  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app import config, db, glossgen, gpu, jobs, media, netinfo, safepath, server, settings, syscheck, translate  # noqa: E402
from app.config import SAMPLE_RATE  # noqa: E402

BASE = Path(tempfile.gettempdir()) / "portability-test"
PY = sys.executable


def touch(p: Path, data: bytes = b"x") -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    return p


class Env:
    """暫存資料庫與假的 data 資料夾（server、jobs 的路徑常數換過去），暫存資料夾註冊成可以刪的測試根目錄。"""

    def __init__(self):
        BASE.mkdir(parents=True, exist_ok=True)
        self.tmp = Path(tempfile.mkdtemp(prefix="vs-port-", dir=BASE))
        self.media = self.tmp / "data" / "media"
        self.work = self.tmp / "data" / "work"
        for d in (self.media / "uploads", self.work):
            d.mkdir(parents=True, exist_ok=True)
        safepath.TEST_ROOTS.append(self.tmp)
        self._saved = []
        self._saved_conn = db._conn
        self.patch(db, "DB_PATH", self.tmp / "test.db")
        db.init()
        self.patch(server, "MEDIA_DIR", self.media)
        self.patch(jobs, "WORK_DIR", self.work)
        self._levels = []
        for name in ("safepath", "server", "jobs", "media"):
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


def run(fn):
    env = Env()
    try:
        fn(env)
    finally:
        env.close()


def fake_gpu(value):
    """之後的顯示卡查詢都回傳 value（None 代表沒有 NVIDIA 顯示卡），並清掉 gpu.query 的快取。"""
    syscheck.set_fake(value)
    gpu._cache.update(t=0.0, info=None)


def real_gpu():
    syscheck.set_fake()
    gpu._cache.update(t=0.0, info=None)


def card(total_mb, used_mb=400, reserved_mb=300, **kw):
    info = {"name": f"Fake {total_mb // 1024}GB", "driver": "595.79", "cuda_version": 13020, "cuda": "13.2",
            "arch": [8, 6], "compute_cap": "8.6", "count": 1, "total_mb": total_mb, "reserved_mb": reserved_mb,
            "used_mb": used_mb, "free_mb": total_mb - reserved_mb - used_mb, "source": "fake"}
    info.update(kw)
    return info


# ---------- ffmpeg、ffprobe 的位置 ----------

def test_find_tool_order():
    tmp = Path(tempfile.mkdtemp(prefix="vs-tool-", dir=BASE if BASE.exists() else None))
    try:
        root = tmp / "root"
        state_exe = touch(tmp / "runtime-bin" / "ffmpeg.exe")
        bin_exe = touch(root / "bin" / "ffmpeg" / "ffmpeg.exe")
        path_exe = touch(tmp / "on-path" / "ffmpeg.exe")
        legacy = tmp / "legacy"
        legacy_exe = touch(legacy / "ffmpeg.exe")
        env_exe = touch(tmp / "env" / "ffmpeg.exe")
        which = lambda name: str(path_exe)                         # noqa: E731
        state = {"ffmpeg": {"path": str(state_exe)}}

        def find(env=None, state=state, which=which):
            return config.find_tool("ffmpeg", env=env or {}, root=root, state=state, which=which, legacy_dir=legacy)

        assert find({"VS_FFMPEG": str(env_exe)})["path"] == str(env_exe)
        assert find()["source"] == "install-state" and find()["path"] == str(state_exe)
        assert find(state={})["path"] == str(bin_exe)
        bin_exe.unlink()
        touch(root / "bin" / "ffmpeg" / "bin" / "ffmpeg.exe")          # 解壓後多一層 bin 也找得到
        assert find(state={})["source"] == "bin"
        shutil.rmtree(root / "bin")
        assert find(state={})["path"] == str(path_exe)
        assert find(state={}, which=lambda name: None)["path"] == str(legacy_exe)   # 作者電腦原本的位置最後才找
        legacy_exe.unlink()
        missing = find(state={}, which=lambda name: None)
        assert missing["path"] is None and missing["problem"] == "找不到 ffmpeg，請重新解壓縮程式或執行 repair.bat"

        # 環境變數指到不存在的檔案：算找不到，不會偷偷改用 PATH 上的
        bad = find({"VS_FFMPEG": str(tmp / "nope" / "ffmpeg.exe")})
        assert bad["path"] is None and bad["problem"].startswith("找不到 ffmpeg") and "VS_FFMPEG" in bad["problem"], bad
        # install-state 可以寫相對路徑（以程式根目錄為準），也可以寫在 paths 底下
        rel = touch(root / "runtime" / "bin" / "ffmpeg-8.1" / "ffprobe.exe")
        got = config.find_tool("ffprobe", env={}, root=root, which=lambda n: None, legacy_dir=legacy,
                               state={"paths": {"ffprobe": str(rel)}})
        assert got["path"] == str(rel), got

        # 只設 VS_FFMPEG：ffprobe 先找它旁邊（比 install-state、PATH 優先），旁邊沒有才照原本的順序
        tools = tmp / "tools"
        only_ff = touch(tools / "ffmpeg.exe")
        probe_side = touch(tools / "ffprobe.exe")
        got = config.find_tool("ffprobe", env={"VS_FFMPEG": str(only_ff)}, root=root, which=lambda n: str(path_exe),
                               legacy_dir=legacy, state={"paths": {"ffprobe": str(rel)}})
        assert got["path"] == str(probe_side) and got["source"] == "VS_FFMPEG", got
        probe_side.unlink()
        got = config.find_tool("ffprobe", env={"VS_FFMPEG": str(only_ff)}, root=root, which=lambda n: None,
                               legacy_dir=legacy, state={})
        assert got["path"] is None and "VS_FFMPEG" in got["problem"] and "VS_FFPROBE" in got["problem"], got
        # 反過來只設 VS_FFPROBE 也一樣
        touch(tools / "ffprobe.exe")
        got = config.find_tool("ffmpeg", env={"VS_FFPROBE": str(tools / "ffprobe.exe")}, root=root,
                               which=lambda n: None, legacy_dir=legacy, state={})
        assert got["path"] == str(only_ff), got
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # 這台電腦（作者的）一定還找得到，而且 yt-dlp 用同一個
    assert config.require_tool("ffmpeg") and config.require_tool("ffprobe")
    assert config.FFMPEG == config.TOOLS["ffmpeg"]["path"]


def test_missing_ffmpeg_is_503_with_chinese_message_not_500():
    def body(env: Env):
        real_ffmpeg, real_ffprobe = config.TOOLS["ffmpeg"], config.TOOLS["ffprobe"]
        missing = config.find_tool("ffmpeg", env={"VS_FFMPEG": str(env.tmp / "nope" / "ffmpeg.exe")})
        env.patch(config, "TOOLS", {"ffmpeg": missing, "ffprobe": config.TOOLS["ffprobe"]})
        env.patch(server, "_resolve_options", lambda opt: ("qwen", None))     # 不管模型有沒有下載
        client = TestClient(server.app, raise_server_exceptions=False)

        r = client.post("/api/media/precheck", json={"language": "ja"})
        assert r.status_code == 503 and r.json()["detail"].startswith("找不到 ffmpeg"), (r.status_code, r.text)

        clip = touch(env.media / "uploads" / "0123456789ab" / "clip.mp4")
        r = client.post("/api/media", json={"source": "upload", "path": str(clip), "language": "ja"})
        assert r.status_code == 503 and "找不到 ffmpeg" in r.json()["detail"], (r.status_code, r.text)
        assert not clip.parent.exists(), "加入失敗的上傳檔要清掉"
        r = client.post("/api/media", json={"source": "url", "url": "https://youtu.be/x", "language": "ja"})
        assert r.status_code == 503 and not db.list_media()

        # 任務裡：抽音訊直接失敗，原因是同一句中文
        try:
            media.extract_audio(clip, env.work / "a.wav", 1)
        except config.ToolMissing as e:
            assert str(e) == missing["problem"]
        else:
            raise AssertionError("找不到 ffmpeg 應該丟 ToolMissing")
        assert media.make_thumbnail(clip, env.work / "t.jpg", 1) is False

        # ffprobe 在啟動後才被刪掉：一樣是 503，不是 500
        gone = {"path": str(env.tmp / "deleted" / "ffprobe.exe"), "source": "PATH", "problem": None}
        env.patch(config, "TOOLS", {"ffmpeg": real_ffmpeg, "ffprobe": gone})
        r = client.post("/api/media/precheck", json={"language": "ja"})
        assert r.status_code == 503 and "ffprobe" in r.json()["detail"], r.text
        # 轉字幕、轉檔只用 ffmpeg：缺 ffprobe 不擋
        mid = db.add_media(title="t", source="upload", path=str(touch(env.tmp / "have.mp4")), duration=1.0)
        assert client.post(f"/api/media/{mid}/proxy").status_code == 200
        env.patch(server, "_enqueue_pipeline", lambda *a, **k: None)
        assert client.post(f"/api/media/{mid}/transcribe", json={"language": "ja"}).status_code == 200

        # VS_FFMPEG 設錯、ffprobe 也找不到：先說 VS_FFMPEG 的問題（使用者真正要改的地方）
        probe_missing = config.find_tool("ffprobe", env={"VS_FFMPEG": str(env.tmp / "nope" / "ffmpeg.exe")},
                                         which=lambda n: None, legacy_dir=env.tmp / "no-legacy", state={},
                                         root=env.tmp / "no-root")
        env.patch(config, "TOOLS", {"ffmpeg": missing, "ffprobe": probe_missing})
        r = client.post("/api/media/precheck", json={"language": "ja"})
        assert r.status_code == 503 and "VS_FFMPEG" in r.json()["detail"], r.text
        env.patch(config, "TOOLS", {"ffmpeg": missing, "ffprobe": real_ffprobe})
        mid2 = db.add_media(title="t2", source="upload", path=str(touch(env.tmp / "have2.mp4")), duration=1.0)
        r = client.post(f"/api/media/{mid2}/proxy")
        assert r.status_code == 503 and "VS_FFMPEG" in r.json()["detail"], r.text
    run(body)


# ---------- 子程序的 PATH、人聲分離失敗改用原音 ----------

def test_child_env_puts_ffmpeg_folder_first():
    saved = config.TOOLS
    try:
        ff = r"D:\BilingualSubtitles\runtime\bin\ffmpeg-8.1\ffmpeg.exe"
        config.TOOLS = {"ffmpeg": {"path": ff}, "ffprobe": {"path": r"D:\BilingualSubtitles\runtime\bin\ffmpeg-8.1\ffprobe.exe"}}
        base = {"PATH": r"C:\Windows\System32;D:\BilingualSubtitles\runtime\bin\ffmpeg-8.1;C:\other", "KEEP": "1"}
        env = config.child_env(base, HF_HUB_OFFLINE="1")
        parts = env["PATH"].split(os.pathsep)
        assert parts[0] == r"D:\BilingualSubtitles\runtime\bin\ffmpeg-8.1" and parts.count(parts[0]) == 1, parts
        assert env["VS_FFMPEG"] == ff and env["KEEP"] == "1" and env["HF_HUB_OFFLINE"] == "1"
        assert env["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    finally:
        config.TOOLS = saved
    env = jobs.run_child_env()
    first = env["PATH"].split(os.pathsep)[0]
    assert os.path.normcase(first) == os.path.normcase(str(Path(config.require_tool("ffmpeg")).parent)), first
    assert env["PYTHONNOUSERSITE"] == "1" and env["HF_HUB_OFFLINE"] == "1"


def test_separator_child_finds_ffmpeg_without_it_on_path():
    """audio-separator 只會執行 PATH 上的 ffmpeg：PATH 拿掉 ffmpeg 後，子程序環境（run_child_env）還是找得到。
    只執行 ffmpeg -version，不載入模型。"""
    ff_dir = os.path.normcase(str(Path(config.require_tool("ffmpeg")).parent))
    stripped = os.pathsep.join(p for p in os.environ.get("PATH", "").split(os.pathsep)
                               if p and "ffmpeg" not in p.lower() and os.path.normcase(p) != ff_dir)
    code = "import shutil, subprocess; p = shutil.which('ffmpeg'); print(p); subprocess.run([p, '-version'], check=True, capture_output=True)"
    saved = os.environ.get("PATH")
    os.environ["PATH"] = stripped
    try:
        without = subprocess.run([PY, "-s", "-c", "import shutil; print(shutil.which('ffmpeg'))"],
                                 env={**os.environ}, capture_output=True, text=True, timeout=60)
        assert without.stdout.strip() == "None", without.stdout
        with_fix = subprocess.run([PY, "-s", "-c", code], env=jobs.run_child_env(), capture_output=True, text=True,
                                  timeout=60)
        assert with_fix.returncode == 0, with_fix.stderr
    finally:
        os.environ["PATH"] = saved


class _StopAfterChunks(Exception):
    pass


def _song_job(env: Env, song_param):
    """轉字幕任務跑到切段完成為止：原音偵測不到人聲，人聲分離的子程序失敗。"""
    mid = db.add_media(title="song", source="local", path=str(touch(env.tmp / "song.mp4")), duration=40,
                       language="ja")
    params = {"language": "ja", "engine": "qwen", "sensitive": False}
    if song_param is not None:
        params["song"] = song_param
    jid = db.add_job(mid, "transcribe", params)
    ctx = jobs.JobContext(db.get_job(jid))
    ctx.workdir.mkdir(parents=True, exist_ok=True)
    sf.write(ctx.workdir / "audio.wav", np.zeros(SAMPLE_RATE * 40, dtype=np.float32), SAMPLE_RATE)
    calls = []

    def fake_run_child(c, args, lo, hi, label, module="app.asr_child"):
        calls.append(module)
        if module == "app.separate":
            touch(Path(args[1]), b"half")                # 失敗前留下半截的檔案
            raise RuntimeError("BS-RoFormer（人聲分離） 執行失敗：FileNotFoundError: ffmpeg")
        raise AssertionError("不應該跑到辨識")

    def stop(*args, **kw):
        raise _StopAfterChunks()

    env.patch(jobs, "run_child", fake_run_child)
    env.patch(jobs.gpu, "ensure_free", lambda *a, **k: None)
    env.patch(jobs.gpu, "pick_batch", stop)
    env.patch(jobs, "plan_speech", stop)          # 切段完成後第一件用到語音模型的事
    env.patch(jobs.vad, "speech_segments", lambda *a, **k: [])
    env.patch(jobs, "use_model", lambda c, m: True)
    env.patch(jobs, "require_model", lambda c, m: None)
    env.patch(jobs, "require_aligner", lambda c=None: None)
    env.patch(settings, "installed_engines", lambda language=None, without=(): ["qwen"])
    settings.save({"health_check": False, "song_mode": "auto"})
    return ctx, calls


def test_auto_song_mode_falls_back_to_original_audio():
    def body(env: Env):
        ctx, calls = _song_job(env, None)
        try:
            jobs.handle_transcribe(ctx)
        except _StopAfterChunks:
            pass
        assert calls == ["app.separate"], calls
        assert ctx.note == "人聲分離失敗，改用原音辨識"
        stage = db.get_job(ctx.id)["stage"]
        assert "人聲分離失敗，改用原音辨識" in stage, stage
        assert json.loads((ctx.workdir / "song.json").read_text("utf-8")) == {"song": False}
        assert (ctx.workdir / "chunks.json").exists() and not (ctx.workdir / "vocals.wav").exists()
        # 完成時的狀態也留著這句
        db.update_job(ctx.id, status="running")
        env.patch(jobs, "HANDLERS", {**jobs.HANDLERS, "transcribe": lambda c: setattr(c, "note", jobs.SEPARATION_FALLBACK_NOTE)})
        jobs.Worker("t", ("transcribe",)).execute(db.get_job(ctx.id))
        assert db.get_job(ctx.id)["stage"] == "完成（人聲分離失敗，改用原音辨識）"
    run(body)


def test_song_mode_without_usable_gpu_fails_with_gpu_reason():
    """沒有能用的顯示卡時不要說「人聲分離失敗」，直接用顯示卡的原因失敗（後面的辨識也跑不了）。"""
    def body(env: Env):
        ctx, calls = _song_job(env, None)

        def no_gpu(*a, **k):
            raise gpu.GpuUnsupported(syscheck.problem(None))

        env.patch(jobs.gpu, "ensure_free", no_gpu)
        try:
            jobs.handle_transcribe(ctx)
        except gpu.GpuUnsupported as e:
            assert "讀不到 NVIDIA 顯示卡" in str(e), e
        else:
            raise AssertionError("沒有顯示卡應該直接失敗")
        assert ctx.note is None and calls == []

        # 顯示卡讀得到，但人聲分離的子程序用不了 CUDA（torch 印英文）：一樣直接用顯示卡的原因失敗，不改用原音硬跑
        env.patch(jobs.gpu, "ensure_free", lambda *a, **k: None)

        def child_no_cuda(c, args, lo, hi, label, module="app.asr_child"):
            calls.append(module)
            raise jobs.child_failure("Traceback (most recent call last):\nRuntimeError: No CUDA GPUs are available\n", label)

        env.patch(jobs, "run_child", child_no_cuda)
        try:
            jobs.handle_transcribe(ctx)
        except gpu.GpuUnsupported as e:
            assert "沒辦法用顯示卡" in str(e) and "No CUDA" not in str(e), e
        else:
            raise AssertionError("子程序用不了 CUDA 應該直接失敗")
        assert ctx.note is None and calls == ["app.separate"], calls
    run(body)


def test_child_process_cuda_errors_are_explained():
    """辨識、對齊、人聲分離的子程序用不了 CUDA 時，torch 只印英文；改成跟翻譯一樣的中文說明。"""
    for tail in ("Traceback (most recent call last):\n  File \"x.py\", line 1\nRuntimeError: No CUDA GPUs are available\n",
                 "AssertionError: Found no NVIDIA driver on your system. Please check that you have an NVIDIA GPU\n",
                 "RuntimeError: CUDA error: no CUDA-capable device is detected\nCUDA kernel errors might be reported\n"):
        e = jobs.child_failure(tail, "Qwen3-ASR-1.7B")
        assert isinstance(e, gpu.GpuUnsupported), tail
        assert str(e).startswith("Qwen3-ASR-1.7B 沒辦法用顯示卡：") and "580" in str(e), str(e)
    oom = jobs.child_failure("torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB\n", "X")
    assert not isinstance(oom, gpu.GpuUnsupported) and str(oom).startswith("顯存不足"), oom
    other = jobs.child_failure("warning\nValueError: bad chunk\n\n", "Qwen3-ASR-1.7B")
    assert type(other) is RuntimeError and str(other) == "Qwen3-ASR-1.7B 執行失敗：ValueError: bad chunk", other
    assert str(jobs.child_failure("", "X")) == "X 執行失敗：未知錯誤"
    assert translate.NO_CUDA_MESSAGE.startswith("翻譯模型沒辦法用顯示卡：")

    # 真的 torch 看不到顯示卡時印的訊息也認得出來（CUDA_VISIBLE_DEVICES=-1，不會用到顯示卡）
    code = "import torch; torch.zeros(1, device='cuda')"
    out = subprocess.run([PY, "-s", "-c", code], env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1"},
                         capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300,
                         creationflags=subprocess.CREATE_NO_WINDOW)
    assert out.returncode != 0, out.stdout
    assert isinstance(jobs.child_failure(out.stderr, "X"), gpu.GpuUnsupported), out.stderr[-800:]


def test_cuda_visible_devices_hiding_the_card_fails_fast():
    """CUDA_VISIBLE_DEVICES 把顯示卡藏起來（-1、空的、編號超過張數）：任務一開始檢查顯存時就說明，不等子程序載入 torch。"""
    one = card(8188)
    assert syscheck.hidden_by_env(one, env={}) is None
    for value in ("-1", "", " ", "1", "-1,0"):
        assert syscheck.hidden_by_env(one, env={"CUDA_VISIBLE_DEVICES": value}) == value, value
    for value in ("0", "0,1", "GPU-2b7c0a52-1111-2222-3333-444455556666"):
        assert syscheck.hidden_by_env(one, env={"CUDA_VISIBLE_DEVICES": value}) is None, value
    saved = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    try:
        fake_gpu(one)
        reason = syscheck.problem(syscheck.query())
        assert reason and "CUDA_VISIBLE_DEVICES=-1" in reason and "重新啟動程式" in reason, reason
        try:
            gpu.ensure_free(1000, "測試模型")
        except gpu.GpuUnsupported as e:
            assert "CUDA_VISIBLE_DEVICES" in str(e)
        else:
            raise AssertionError("顯示卡被藏起來時 GPU 任務應該直接失敗")
        assert "讀不到 NVIDIA 顯示卡" in syscheck.problem(None)          # 沒有顯示卡時先說沒有顯示卡
    finally:
        if saved is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = saved
        real_gpu()


def test_forced_song_mode_fails_the_job():
    def body(env: Env):
        ctx, calls = _song_job(env, True)
        try:
            jobs.handle_transcribe(ctx)
        except RuntimeError as e:
            assert "人聲分離" in str(e), e
        else:
            raise AssertionError("使用者強制開歌曲模式時，分離失敗要讓任務失敗")
        assert ctx.note is None and not (ctx.workdir / "chunks.json").exists()
    run(body)


# ---------- llama-server 只用 CUDA0 ----------

def test_llama_server_uses_cuda0_and_explains_missing_cuda():
    def body(env: Env):
        real_popen = subprocess.Popen
        seen = {}

        def fake_popen(args, *a, **kw):
            seen["args"], seen["env"] = [str(x) for x in args], kw.get("env") or {}
            fake = [PY, "-c", "print('ggml_cuda_init: failed to initialize CUDA: no CUDA-capable device is detected');"
                              "print('error: invalid device: CUDA0'); raise SystemExit(1)"]
            return real_popen(fake, *a, **kw)

        env.patch(translate.subprocess, "Popen", fake_popen)
        srv = translate.LlamaServer("hymt-mini", env.work / "llama.log")
        t0 = time.time()
        try:
            srv.start(lambda: None)
        except RuntimeError as e:
            assert str(e) == translate.NO_CUDA_MESSAGE, e
        else:
            raise AssertionError("沒有 CUDA 時應該失敗")
        assert time.time() - t0 < 10
        i = seen["args"].index("--device")
        assert seen["args"][i + 1] == "CUDA0", seen["args"]
        assert seen["env"].get("CUDA_DEVICE_ORDER") == "PCI_BUS_ID"
        # 其他失敗原因照舊
        assert translate.load_error_message("ggml_backend_cuda_buffer_type_alloc_buffer: out of memory").startswith("顯存不足")
        assert translate.load_error_message("error: something else\n") == "翻譯模型載入失敗：error: something else"
    run(body)


def test_real_llama_server_fails_fast_without_cuda():
    """真的 llama-server、CUDA_VISIBLE_DEVICES=-1：看不到顯示卡，檢查 --device 時就結束，不會載入模型、不佔顯存。"""
    if not config.LLAMA_SERVER.is_file():
        print("  （沒有 llama-server，跳過）")
        return

    def body(env: Env):
        saved = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        try:
            srv = translate.LlamaServer("hymt-mini", env.work / "llama-nocuda.log")
            t0 = time.time()
            try:
                srv.start(lambda: None)
            except RuntimeError as e:
                elapsed = time.time() - t0
                assert str(e) == translate.NO_CUDA_MESSAGE, (str(e), (env.work / "llama-nocuda.log").read_text("utf-8"))
            else:
                srv.stop()
                raise AssertionError("CUDA_VISIBLE_DEVICES=-1 時應該失敗")
            assert elapsed < 15, elapsed
            assert not srv.alive()
            print(f"  （llama-server 沒有 CUDA 時 {elapsed:.1f} 秒內失敗）")
        finally:
            if saved is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = saved
    run(body)


# ---------- 顯示卡、驅動、架構檢查 ----------

def test_gpu_jobs_fail_with_reason_on_unsupported_cards():
    try:
        cases = {
            "沒有 NVIDIA": (None, "讀不到 NVIDIA 顯示卡"),
            "驅動 552": (card(8188, driver="552.22", cuda_version=12040, cuda="12.4"), "驅動太舊"),
            "舊驅動讀不到 CUDA 版本": (card(8188, driver="566.14", cuda_version=None, cuda=None), "驅動太舊"),
            "GTX 1080": (card(8192, name="NVIDIA GeForce GTX 1080", arch=[6, 1], compute_cap="6.1"), "架構 6.1 低於 7.5"),
        }
        for name, (info, words) in cases.items():
            fake_gpu(info)
            assert words in (syscheck.problem(syscheck.query()) or ""), (name, syscheck.problem(info))
            for call in (lambda: gpu.ensure_free(1000, "測試模型"), lambda: gpu.pick_batch([(6000, 2), (3000, 1)], "測試模型")):
                try:
                    call()
                except gpu.GpuUnsupported as e:
                    assert words in str(e), (name, str(e))
                else:
                    raise AssertionError(f"{name}：GPU 任務應該直接失敗")
        # RTX 2060（7.5）、8GB 卡：可以用
        fake_gpu(card(6144, name="NVIDIA GeForce RTX 2060", arch=[7, 5], compute_cap="7.5"))
        assert syscheck.problem(syscheck.query()) is None
        fake_gpu(card(8188, used_mb=200))
        assert gpu.pick_batch([(6000, 2), (3000, 1)], "測試模型") == 2
        gpu.ensure_free(3000, "測試模型")

        # VS_FAKE_SYSTEM：用 JSON 檔模擬別台電腦
        real_gpu()
        BASE.mkdir(parents=True, exist_ok=True)
        fixture = BASE / "fake-system.json"
        fixture.write_text(json.dumps({"gpu": None}), encoding="utf-8")
        os.environ["VS_FAKE_SYSTEM"] = str(fixture)
        try:
            assert syscheck.query() is None
            fixture.write_text(json.dumps({"gpu": card(12288)}), encoding="utf-8")
            assert syscheck.query()["total_mb"] == 12288
        finally:
            os.environ.pop("VS_FAKE_SYSTEM", None)
            fixture.unlink()
    finally:
        real_gpu()


def test_syscheck_matches_nvidia_smi():
    """本機 NVML 讀到的跟 nvidia-smi 一致（顯存用量兩次讀取之間會變，允許一點差距）。"""
    if shutil.which("nvidia-smi") is None:
        print("  （沒有 nvidia-smi，跳過）")
        return
    real_gpu()
    nvml = syscheck.query()
    smi = syscheck.nvidia_smi_query()
    assert nvml and smi, (nvml, smi)
    if nvml["source"] != "nvml":
        print("  （NVML 讀不到，已經改用 nvidia-smi）")
    for key in ("name", "total_mb", "driver", "compute_cap", "cuda", "reserved_mb", "pci_bus_id"):
        assert nvml[key] == smi[key], (key, nvml[key], smi[key])
    for key in ("used_mb", "free_mb"):
        assert abs(nvml[key] - smi[key]) < 600, (key, nvml[key], smi[key])
    assert syscheck.problem(nvml) is None
    print(f"  （{nvml['name']}，{nvml['total_mb']} MiB，驅動 {nvml['driver']}，CUDA {nvml['cuda']}，架構 {nvml['compute_cap']}）")


class _OldNvml:
    """模擬很舊的驅動的 nvml.dll：包住真的 nvml.dll，但少了 missing 裡的函式（跟 ctypes 找不到函式時一樣丟 AttributeError）。"""

    def __init__(self, real, missing):
        self._real, self._missing = real, set(missing)
        self.open_count = 0          # 成功的 nvmlInit_v2 減掉 nvmlShutdown，測完要補關

    def __getattr__(self, name):
        if name in self._missing:
            raise AttributeError(name)
        fn = getattr(self._real, name)
        if name in ("nvmlInit_v2", "nvmlShutdown"):
            def counted(*args):
                code = fn(*args)
                if code == 0:
                    self.open_count += 1 if name == "nvmlInit_v2" else -1
                return code
            return counted
        return fn


def test_old_nvml_dll_without_newer_functions():
    """R418 以前的驅動：nvml.dll 缺 CUDA 版本、架構這些函式時不丟例外；缺必要的函式時改用 nvidia-smi。只讀。"""
    dll = next((p for p in syscheck._nvml_dll_paths() if os.path.isfile(p)), None)
    if dll is None or shutil.which("nvidia-smi") is None:
        print("  （沒有 nvml.dll 或 nvidia-smi，跳過）")
        return
    real_cdll = syscheck.ctypes.CDLL
    saved = dict(syscheck._nvml)
    opened = []

    def use(missing):
        def fake(path, *a, **k):
            lib = _OldNvml(real_cdll(path, *a, **k), missing)
            opened.append(lib)
            return lib
        syscheck.ctypes.CDLL = fake
        syscheck._nvml.update(lib=None, handle=None, static=None, retry_at=0.0)

    try:
        real_gpu()
        use({"nvmlSystemGetCudaDriverVersion_v2", "nvmlDeviceGetCudaComputeCapability", "nvmlDeviceGetPciInfo_v3",
             "nvmlDeviceGetMemoryInfo_v2"})
        info = syscheck.query()
        assert info and info["source"] == "nvml" and info["cuda_version"] is None and info["arch"] is None, info
        assert info["total_mb"] > 0 and info["driver"], info
        # 讀不到 CUDA 版本時看驅動版本（這台是新驅動，所以沒有問題；舊驅動會說驅動太舊）
        assert syscheck.problem(info) is None
        assert "驅動太舊" in syscheck.problem(dict(info, driver="418.81"))

        use({"nvmlInit_v2"})
        info = syscheck.query()
        assert info and info["source"] == "nvidia-smi", info
        use({"nvmlDeviceGetCount_v2"})
        info = syscheck.query()
        assert info and info["source"] == "nvidia-smi", info
    finally:
        syscheck.ctypes.CDLL = real_cdll
        for lib in opened:                           # 這個測試開的 NVML 自己關掉（NVML 會計數，不動原本的）
            for _ in range(lib.open_count):
                lib._real.nvmlShutdown()
        syscheck._nvml.clear()
        syscheck._nvml.update(saved)
        real_gpu()


# ---------- 顯存不足的說明、翻譯模型挑放得下的 ----------

def test_vram_message_says_card_too_small_instead_of_retry():
    try:
        fake_gpu(card(8188, used_mb=400, reserved_mb=250))          # used + free = 7938
        try:
            gpu.ensure_free(8812, "Hy-MT2-7B", alternatives=lambda usable: jobs.smaller_translators("hymt", usable))
        except RuntimeError as e:
            msg = str(e)
            assert "這張顯示卡放不下 Hy-MT2-7B" in msg and "重試也不會成功" in msg and "Hy-MT2-1.8B" in msg, msg
            assert "按「重試」" not in msg and "Sakura-14B" not in msg, msg
        else:
            raise AssertionError("整張卡放不下應該報錯")
        fake_gpu(card(16303, used_mb=9000, reserved_mb=306))        # 整張卡放得下，只是現在被佔用
        try:
            gpu.ensure_free(8812, "Hy-MT2-7B")
        except RuntimeError as e:
            assert "顯存不足" in str(e) and "重試" in str(e), e
        else:
            raise AssertionError("剩餘顯存不夠應該報錯")
        assert gpu.fits(8812, card(8188)) is False and gpu.fits(8812, card(16303)) is True and gpu.fits(1, None) is None
    finally:
        real_gpu()


def test_gloss_translator_picks_one_that_fits():
    saved = settings.installed_translators
    settings.installed_translators = lambda language=None, without=(): [k for k in ("hymt", "hymt-mini") if k not in without]
    # 假裝已經下載：gguf 不在的話改用型錄的大小算顯存（已經下載的照舊用檔案大小），沒下載模型的電腦也能跑
    real_vram = config.translator_vram_mb
    sizes = {e["translator"]: e["size_mb"] for e in config.MODEL_CATALOG.values() if e.get("role") == "translator"}
    config.translator_vram_mb = lambda key, size_mb=None: real_vram(key, size_mb or sizes.get(key))
    try:
        need_7b = config.translator_vram_mb("hymt", 7612)
        fake_gpu(card(8188))
        assert need_7b > gpu.usable_mb(gpu.query())
        assert glossgen.translator_key() == "hymt-mini"
        fake_gpu(card(16303))
        assert glossgen.translator_key() == "hymt"
        fake_gpu(None)                                              # 讀不到顯示卡：不看顯存
        assert glossgen.translator_key() == "hymt"
        fake_gpu(card(2048))                                        # 兩個都放不下：還是給第一個，任務會說明放不下
        assert glossgen.translator_key() == "hymt"
        fake_gpu(card(8188))
        assert glossgen.translator_key(without={"hymt-mini"}) == "hymt"
    finally:
        settings.installed_translators = saved
        config.translator_vram_mb = real_vram
        real_gpu()


# ---------- 區網開關 ----------

def test_lan_switch_blocks_remote_clients_immediately():
    def body(env: Env):
        assert settings.DEFAULTS["lan_access"] is False and settings.get("lan_access") is False
        phone = TestClient(server.app, client=("192.168.1.20", 50123))
        local = TestClient(server.app, client=("127.0.0.1", 50124))
        r = phone.get("/api/meta")
        assert r.status_code == 403 and "讓同一個網路的手機、平板連線" in r.json()["detail"], r.text
        page = phone.get("/")
        assert page.status_code == 403 and "text/html" in page.headers["content-type"] and "設定" in page.text
        assert local.get("/api/meta").status_code == 200
        for host in ("::1", "::ffff:127.0.0.1", "127.0.0.5"):
            assert TestClient(server.app, client=(host, 1)).get("/api/meta").status_code == 200, host
        assert TestClient(server.app, client=("::ffff:192.168.1.20", 1)).get("/api/meta").status_code == 403

        assert local.put("/api/settings", json={"lan_access": True}).status_code == 200
        assert phone.get("/api/meta").status_code == 200 and phone.get("/").status_code == 200
        assert phone.put("/api/settings", json={"lan_access": False}).status_code == 200
        assert phone.get("/api/meta").status_code == 403                 # 關掉馬上生效
        assert local.put("/api/settings", json={"lan_access": "yes"}).status_code == 400
        assert server.is_local_client("testclient") and not server.is_local_client(None)
        assert not server.is_local_client("evil.example")
        # 手機看得到 403 頁就代表已經在聽區網：只叫人打開開關，不叫人重新啟動
        assert "重新啟動" not in server.LAN_OFF_MESSAGE and "重新整理" in server.LAN_OFF_MESSAGE

        # VS_HOST 設成區網 IP：在這台電腦上用那個 IP 開，來源和連進來的位址相同，算本機（開關關著也能開設定頁）
        same = TestClient(server.app, base_url="http://192.168.1.5:8765", client=("192.168.1.5", 50200))
        other = TestClient(server.app, base_url="http://192.168.1.5:8765", client=("192.168.1.20", 50201))
        assert same.get("/api/meta").status_code == 200 and same.get("/").status_code == 200
        assert other.get("/api/meta").status_code == 403
        assert server.is_local_client("192.168.1.5", "192.168.1.5")
        assert server.is_local_client("::ffff:192.168.1.5", "192.168.1.5")
        assert not server.is_local_client("192.168.1.20", "192.168.1.5")
        assert not server.is_local_client("0.0.0.0", "0.0.0.0") and not server.is_local_client("192.168.1.5", None)
        assert not server.is_local_client("192.168.1.5", "testserver")
    run(body)


def test_bind_host_follows_env_then_setting():
    def body(env: Env):
        env.patch(config, "HOST_ENV", None)
        assert server.bind_host() == "127.0.0.1"                        # 新安裝：只開本機
        settings.save({"lan_access": True})
        assert settings.peek("lan_access") is True and server.bind_host() == "0.0.0.0"
        env.patch(config, "HOST_ENV", "127.0.0.1")
        assert server.bind_host() == "127.0.0.1"                        # VS_HOST 優先
        assert settings.peek("lan_access", env.tmp / "no-such.db") is False
        # 設定頁的狀態：打開了但還沒重新啟動
        env.patch(config, "HOST_ENV", None)
        env.patch(server, "RUNTIME", {"host": "127.0.0.1", "port": 8765})
        view = server.read_settings()["network"]
        assert view["lan_access"] and view["restart_needed"] and not view["listening_lan"], view
        env.patch(server, "RUNTIME", {"host": "0.0.0.0", "port": 8765})
        view = server.read_settings()["network"]
        assert view["listening_lan"] and not view["restart_needed"] and view["port"] == 8765, view
        assert not view["host_env_local"]
        # VS_HOST=127.0.0.1 又打開開關：重新啟動也沒用，不能說「重新啟動就能連」
        env.patch(config, "HOST_ENV", "127.0.0.1")
        env.patch(server, "RUNTIME", {"host": "127.0.0.1", "port": 8765})
        view = server.read_settings()["network"]
        assert view["host_env_local"] and not view["restart_needed"] and not view["listening_lan"], view
        assert view["addresses"] == [], view
        # VS_HOST 指定一個區網位址：只有那個位址連得到
        env.patch(config, "HOST_ENV", "192.168.1.5")
        env.patch(server, "RUNTIME", {"host": "192.168.1.5", "port": 8765})
        view = server.read_settings()["network"]
        assert view["listening_lan"] and not view["host_env_local"] and view["addresses"] == ["192.168.1.5"], view
        env.patch(server, "RUNTIME", {"host": "::1", "port": 8765})
        assert not server.read_settings()["network"]["listening_lan"]
    run(body)


def test_lan_addresses_skip_virtual_adapters():
    # 假的網卡清單：位址用文件專用的範圍（192.0.2.x、198.51.100.x、203.0.113.x），名稱用一般寫法
    items = [
        {"name": "區域連線 2", "description": "TAP-Windows Adapter V9", "type": 53, "up": False,
         "ipv4": ["169.254.20.30"], "gateways": []},
        {"name": "WireGuard", "description": "WireGuard Tunnel", "type": 53, "up": True, "ipv4": ["100.64.0.2"], "gateways": []},
        {"name": "乙太網路", "description": "Ethernet Controller", "type": 6, "up": True,
         "ipv4": ["192.0.2.10"], "gateways": ["192.0.2.1"]},
        {"name": "VirtualBox Host-Only Network", "description": "VirtualBox Host-Only Ethernet Adapter", "type": 6,
         "up": True, "ipv4": ["198.51.100.1"], "gateways": []},
        {"name": "vEthernet (WSL)", "description": "Hyper-V Virtual Ethernet Adapter", "type": 6, "up": True,
         "ipv4": ["172.20.0.1"], "gateways": ["172.20.0.254"]},
        {"name": "Wi-Fi", "description": "Wireless Network Adapter", "type": 71, "up": True,
         "ipv4": ["203.0.113.50"], "gateways": ["203.0.113.1"]},
        {"name": "藍牙網路連線", "description": "Bluetooth Device (Personal Area Network)", "type": 6, "up": True,
         "ipv4": ["10.9.9.9"], "gateways": ["10.9.9.1"]},
    ]
    assert netinfo.pick_lan_addresses(items) == ["192.0.2.10", "203.0.113.50"]
    assert netinfo.pick_lan_addresses(items, primary="203.0.113.50") == ["203.0.113.50", "192.0.2.10"]
    # 沒有閘道的實體網卡（例如直接接手機熱點的設定不完整）：退回列出非虛擬網卡
    no_gw = [dict(a, gateways=[]) for a in items]
    assert netinfo.pick_lan_addresses(no_gw) == ["192.0.2.10", "203.0.113.50"]
    # 全部都像虛擬網卡：做不好就列出全部
    assert "198.51.100.1" in netinfo.pick_lan_addresses([items[3]])
    real = netinfo.lan_addresses()
    assert all(not ip.startswith(("127.", "169.254.")) for ip in real), real


# ---------- 位置可以用環境變數改 ----------

_PATHS_SCRIPT = r"""
import json
from app import config, safepath
from app.config import MODEL_CATALOG, model_installed
print(json.dumps({
    "data": str(config.DATA_DIR), "media": str(config.MEDIA_DIR), "db": str(config.DB_PATH),
    "models": str(config.MODELS_DIR), "cache": str(config.CACHE_DIR), "llama": str(config.LLAMA_SERVER),
    "hf_hub": __import__("os").environ.get("HF_HUB_CACHE"), "host_env": config.HOST_ENV,
    "installed": sorted(k for k, e in MODEL_CATALOG.items() if model_installed(e)),
    "media_root_ok": safepath._root_ok(safepath._resolve(config.MEDIA_DIR)),
    "models_root_ok": safepath._root_ok(safepath._resolve(config.MODELS_DIR)),
    "project_media_ok": safepath._root_ok(safepath._resolve(config.ROOT / "data" / "media")),
}))
"""


def _paths(env_extra: dict) -> dict:
    env = {k: v for k, v in os.environ.items()
           if k not in ("VS_DATA_DIR", "VS_MODELS_DIR", "VS_CACHE_DIR", "VS_LLAMA_SERVER", "VS_HOST",
                        "HF_HOME", "HF_HUB_CACHE", "HF_XET_CACHE", "HF_MODULES_CACHE", "TORCH_HOME", "MPLCONFIGDIR")}
    env.update(env_extra)
    out = subprocess.run([PY, "-s", "-c", _PATHS_SCRIPT], cwd=str(ROOT), env=env, capture_output=True, text=True,
                         encoding="utf-8", timeout=120)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_locations_follow_environment_variables():
    BASE.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="vs-paths-", dir=BASE))
    try:
        default = _paths({})
        # 沒設的時候跟以前完全一樣
        assert default["data"] == str(ROOT / "data") and default["models"] == str(ROOT / "models")
        assert default["cache"] == str(ROOT / "data" / "cache") and default["db"] == str(ROOT / "data" / "library.db")
        assert default["llama"] == str(ROOT / "bin" / "llama.cpp" / "llama-server.exe") and default["host_env"] is None
        assert default["media_root_ok"] and default["models_root_ok"] and default["project_media_ok"]

        data, cache, llama = tmp / "data", tmp / "cache", tmp / "llama" / "llama-server.exe"
        moved = _paths({"VS_DATA_DIR": str(data), "VS_MODELS_DIR": str(ROOT / "models"), "VS_CACHE_DIR": str(cache),
                        "VS_LLAMA_SERVER": str(llama)})
        assert moved["data"] == str(data) and moved["media"] == str(data / "media") and moved["db"] == str(data / "library.db")
        assert moved["cache"] == str(cache) and moved["hf_hub"] == str(cache / "huggingface" / "hub")
        assert moved["llama"] == str(llama)
        # 測試版指到專案裡同一個 models：載入判斷一樣（只讀；沒下載模型的電腦兩邊都是空的）
        assert moved["installed"] == default["installed"], moved["installed"]
        # safepath 允許刪除的根目錄跟著設定走：新的 data 可以，原本專案裡的 data 不行了
        assert moved["media_root_ok"] and moved["models_root_ok"] and not moved["project_media_ok"]

        # 模型位置寫在 data\paths.json：那裡放一個假的 tsqyomi，確認真的是去那裡找
        other_models = tmp / "models-elsewhere"
        tsq = config.MODEL_CATALOG["tsqyomi"]
        for f in tsq["files"]:
            touch(other_models / tsq["dir"].relative_to(config.MODELS_DIR) / f)
        (data).mkdir(parents=True, exist_ok=True)
        (data / "paths.json").write_text(json.dumps({"models_dir": str(other_models)}), encoding="utf-8")
        via_json = _paths({"VS_DATA_DIR": str(data)})
        assert via_json["models"] == str(other_models) and via_json["installed"] == ["tsqyomi"], via_json["installed"]
        assert via_json["models_root_ok"]
        # 環境變數比 paths.json 優先
        assert _paths({"VS_DATA_DIR": str(data), "VS_MODELS_DIR": str(ROOT / "models")})["models"] == str(ROOT / "models")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------- 相容播放檔：NVENC 不能用時改用 libx264 ----------

def _make_clip(path: Path):
    ffmpeg = config.require_tool("ffmpeg")
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc=size=160x120:rate=10",
                    "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100", "-t", "2",
                    "-c:v", "mpeg4", "-c:a", "mp3", str(path)], check=True, capture_output=True, timeout=60)


def test_proxy_falls_back_to_libx264():
    def body(env: Env):
        src = env.tmp / "clip.avi"
        _make_clip(src)
        notes, progress = [], []
        env.patch(media, "NVENC_ARGS", ["-c:v", "h264_nvenc_not_in_this_build"])   # 模擬 NVENC 不能用（不碰顯示卡）
        env.patch(media, "_nvenc_failed", {"value": False})
        dst = env.tmp / "proxy" / "aaaaaaaaaaaa.mp4"
        dst.parent.mkdir()
        media.make_proxy(src, dst, 2.0, on_progress=progress.append, on_note=notes.append)
        assert notes == ["顯示卡編碼不能用，改用 CPU 轉檔，會比較慢"], notes
        info = media.probe(dst)
        assert info["vcodec"] == "h264" and info["acodec"] == "aac" and info["playable"] == 1, info
        assert not dst.with_suffix(".tmp.mp4").exists()
        assert media._nvenc_failed["value"] is True                          # 之後的轉檔直接用 libx264

        # VS_FORCE_NO_NVENC=1：不試 NVENC
        env.patch(media, "_nvenc_failed", {"value": False})
        tried = []
        real_run = media._run_with_progress
        env.patch(media, "_run_with_progress", lambda args, *a: tried.append(args) or real_run(args, *a))
        os.environ["VS_FORCE_NO_NVENC"] = "1"
        try:
            notes.clear()
            media.make_proxy(src, dst, 2.0, on_note=notes.append)
        finally:
            os.environ.pop("VS_FORCE_NO_NVENC")
        assert len(tried) == 1 and "libx264" in tried[0] and notes, tried

        # 來源檔的問題不改用 libx264（不會白白再轉一次）
        assert not media._nvenc_problem("clip.mp4: No such file or directory")
        assert media._nvenc_problem("[h264_nvenc @ 000001] OpenEncodeSessionEx failed: unsupported device (2)")
        assert media._nvenc_problem("Cannot load nvcuda.dll")

        # 只有這部影片 NVENC 編不了（例如畫面超過 NVENC 的尺寸上限），試編一小段黑畫面是成功的：
        # 這部改用 libx264，但不記成「NVENC 不能用」，下一部照樣先試 NVENC
        env.patch(media, "_run_with_progress", real_run)
        env.patch(media, "nvenc_works", lambda ffmpeg, timeout=30: True)
        env.patch(media, "_nvenc_failed", {"value": False})
        notes.clear()
        dst.unlink()
        media.make_proxy(src, dst, 2.0, on_note=notes.append)
        assert notes == [media.X264_FILE_NOTE] and "會比較慢" in notes[0], notes
        assert media._nvenc_failed["value"] is False and media.probe(dst)["vcodec"] == "h264"
        # 試編也失敗：NVENC 整個不能用，之後直接用 libx264
        env.patch(media, "nvenc_works", lambda ffmpeg, timeout=30: False)
        media.make_proxy(src, dst, 2.0)
        assert media._nvenc_failed["value"] is True

        # 任務的進度訊息說會比較慢
        mid = db.add_media(title="p", source="local", path=str(src), duration=2.0)
        jid = db.add_job(mid, "proxy", {})
        env.patch(jobs, "PROXY_DIR", env.tmp / "proxy")
        env.patch(jobs, "release_llm", lambda reason="": None)
        stages = []
        real_progress = jobs.JobContext.progress

        def spy(self, value, stage=None):
            real_progress(self, value, stage)
            stages.append(db.get_job(self.id)["stage"])

        env.patch(jobs.JobContext, "progress", spy)
        env.patch(media, "_nvenc_failed", {"value": True})
        try:
            jobs.handle_proxy(jobs.JobContext(db.get_job(jid)))
        finally:
            jobs.gpu_state.update(job_id=None, model=None, stage=None)
        assert any("改用 CPU 轉檔，會比較慢" in (s or "") for s in stages), stages
        assert (env.tmp / "proxy" / f"{mid}.mp4").exists()
    run(body)


# ---------- MIME 類型 ----------

def test_static_files_have_correct_mime_types():
    import mimetypes
    expected = {".css": "text/css", ".js": "text/javascript", ".mjs": "text/javascript", ".json": "application/json",
                ".svg": "image/svg+xml", ".webmanifest": "application/manifest+json"}
    for ext, mime in expected.items():
        assert mimetypes.guess_type(f"x{ext}")[0] == mime, (ext, mimetypes.guess_type(f"x{ext}"))
    client = TestClient(server.app)
    assert client.get("/static/app.js").headers["content-type"].startswith("text/javascript")
    assert client.get("/static/style.css").headers["content-type"].startswith("text/css")

    def body(env: Env):
        assert client.get("/api/meta").json()["app"] == "video-subtitle"    # 重複啟動時靠它認出自己的程式
    run(body)


# ---------- safepath 只認明確註冊的測試根目錄 ----------

def test_safepath_only_accepts_registered_test_roots():
    BASE.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="vs-roots-", dir=BASE))
    logger = logging.getLogger("safepath")
    level = logger.level
    logger.setLevel(logging.CRITICAL)
    try:
        victim = touch(tmp / "root" / "a.txt")
        assert not safepath.safe_unlink(victim, tmp / "root")          # 暫存資料夾底下，但沒有註冊
        assert victim.exists()
        safepath.TEST_ROOTS.append(Path(tempfile.gettempdir()))         # 註冊暫存資料夾本身也不算
        safepath.TEST_ROOTS.append(Path.home() / "Desktop")             # 暫存資料夾以外的位置也不算
        try:
            assert not safepath.safe_unlink(victim, tmp / "root") and victim.exists()
            assert not safepath.safe_rmtree(Path.home() / "Desktop" / "vs-roots-no-such-folder-7f3a",
                                            Path.home() / "Desktop")
        finally:
            safepath.TEST_ROOTS.remove(Path(tempfile.gettempdir()))
            safepath.TEST_ROOTS.remove(Path.home() / "Desktop")
        safepath.TEST_ROOTS.append(tmp)
        try:
            assert safepath.safe_unlink(victim, tmp / "root") and not victim.exists()
        finally:
            safepath.TEST_ROOTS.remove(tmp)
        again = touch(tmp / "root" / "b.txt")
        assert not safepath.safe_unlink(again, tmp / "root") and again.exists()     # 拿掉之後又不行了
    finally:
        logger.setLevel(level)
        shutil.rmtree(tmp, ignore_errors=True)


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
