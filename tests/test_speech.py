"""常駐語音模型的測試（app/speech.py、app/speech_worker.py），不需要顯卡、不動正式資料：
python -s tests/test_speech.py

1. 規劃：同時載入、輪流載入、部分同時載入、批次大小、跟翻譯模型搶顯存、放不下時的說明
2. 工作程序：用真的 app.speech_worker 程序，模型和顯存換成假的（tests/fake_speech_backend.py）
   跨步驟跨影片沿用、輪流載入、爆顯存時卸載或降批次、取消、卡住、當掉重開、開不起來改用單次子程序、刪模型時卸載、伺服器結束
3. 任務：jobs.handle_transcribe、jobs.cancel 經過常駐程序跑完整流程；顯卡閒置 120 秒釋放（跟翻譯模型共用計時）、
   「釋放顯卡」按鈕（沒任務直接釋放、有任務先確認、暫停佇列、執行中的任務停下來回到佇列、繼續後從檢查點接著做）
"""
import json
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

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

from app import config, db, gpu, jobs, models, safepath, settings, speech  # noqa: E402
from app import cues as cue_mod  # noqa: E402
from app.config import SAMPLE_RATE  # noqa: E402

BASE = Path(tempfile.gettempdir()) / "speech-test"
V = config.SPEECH_VRAM
REAL_RELEASE_LLM = jobs.release_llm


def card(total_mb, used_mb, reserved_mb=306):
    return {"name": f"Fake {total_mb // 1024}GB", "driver": "595.79", "cuda_version": 13020, "cuda": "13.2",
            "arch": [12, 0], "compute_cap": "12.0", "count": 1, "total_mb": total_mb, "reserved_mb": reserved_mb,
            "used_mb": used_mb, "free_mb": total_mb - reserved_mb - used_mb, "source": "fake"}


class FakeGpu:
    """假顯示卡：其他程式佔 other_mb，常駐程序佔多少照它回報的算（載入、卸載後剩餘顯存會跟著變）。"""

    def __init__(self, worker, total_mb=16303, other_mb=2400):
        self.worker, self.total, self.other = worker, total_mb, other_mb

    def __call__(self, max_age=2.0):
        return card(self.total, self.other + self.worker.usage_mb())


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


class Env:
    def __init__(self):
        BASE.mkdir(parents=True, exist_ok=True)
        self.tmp = Path(tempfile.mkdtemp(prefix="vs-speech-", dir=BASE))
        self._saved = []
        self.workers = []

    def patch(self, obj, name, value):
        self._saved.append((obj, name, getattr(obj, name)))
        setattr(obj, name, value)

    def worker(self, name="w", fake=None, settings_=None, clock=None, gpu_total=16303, gpu_other=2400):
        """假模型的常駐程序（真的子程序），和跟著它變動的假顯示卡。"""
        d = self.tmp / name
        d.mkdir(parents=True, exist_ok=True)
        events, control = d / "events.jsonl", d / "control.json"
        cfg = {"events": str(events), "control": str(control), **(fake or {})}
        env = jobs.run_child_env()
        env.update(VS_SPEECH_BACKEND="tests.fake_speech_backend", VS_FAKE_SPEECH=json.dumps(cfg),
                   VS_DATA_DIR=str(d / "data"), VS_MODELS_DIR=str(d / "models"))
        s = {**config.SPEECH_WORKER, "stall_s": 30, "cancel_wait_s": 3, **(settings_ or {})}
        w = speech.Worker(env=env, log_path=d / "speech_worker.log", settings=s, clock=clock or time.monotonic)
        w.events, w.control, w.dir = events, control, d
        w.gpu = FakeGpu(w, gpu_total, gpu_other)
        self.workers.append(w)
        return w

    def close(self):
        for w in self.workers:
            if w.proc is not None and w.proc.poll() is None:
                w.proc.kill()
                w.proc.wait(10)
        for obj, name, value in reversed(self._saved):
            setattr(obj, name, value)
        shutil.rmtree(self.tmp, ignore_errors=True)


def run(body):
    env = Env()
    try:
        body(env)
    finally:
        env.close()


def events(w, ev=None, model=None):
    if not w.events.exists():
        return []
    out = [json.loads(l) for l in w.events.read_text("utf-8").splitlines() if l.strip()]
    return [e for e in out if (ev is None or e["ev"] == ev) and (model is None or e.get("model") == model)]


def set_control(w, **kw):
    w.control.write_text(json.dumps(kw), encoding="utf-8")


def ready(w):
    """先把程序開好（啟動要一兩秒，不要算進取消、卡住的計時裡）。"""
    with w._lock:
        w._ensure_ready()


def workdir(path: Path, seconds=120, chunk=30, lang_text=None):
    """工作資料夾：音訊、切好的段落；lang_text 給了就先放好辨識結果（只測對齊時用）。"""
    path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    sf.write(path / "audio.wav", (rng.standard_normal(SAMPLE_RATE * seconds) * 0.01).astype(np.float32), SAMPLE_RATE)
    chunks = [[float(s), float(min(seconds, s + chunk))] for s in range(0, seconds, chunk)]
    (path / "chunks.json").write_text(json.dumps(chunks), encoding="utf-8")
    if lang_text:
        (path / "asr.json").write_text(json.dumps({str(i): lang_text for i in range(len(chunks))}, ensure_ascii=False),
                                       encoding="utf-8")
    return path


def cues_file(path: Path, n=12, name="cues.json"):
    cl = [{"start": i * 4.0 + 0.5, "end": i * 4.0 + 3.0, "text": "あいうえお"} for i in range(n)]
    (path / name).write_text(json.dumps(cl, ensure_ascii=False), encoding="utf-8")
    return name


# ---------- 1. 規劃 ----------

def test_plan_16gb_and_12gb_load_everything_together():
    budget16 = 15997 - 2415 - 1024          # 擁有者的卡：桌面約 2.4 GB
    p = speech.plan_job("qwen", True, 480, budget16)
    assert p.mode == "同時載入" and p.bs == {"asr:qwen": 6, "align": 4, "verify": 24, "realign": 4}, p
    p = speech.plan_job("anime", True, 360, budget16, loaded={"qwen", "aligner"})
    assert p.mode == "同時載入" and p.bs["asr:anime"] == 8 and p.loads == 1, p   # 只要多載 anime-whisper
    budget12 = 12227 - 300 - 1000 - 1024    # 12 GB 卡、桌面 1 GB
    p = speech.plan_job("qwen", True, 480, budget12)
    assert p.mode == "同時載入" and p.bs["asr:qwen"] == 6, p
    p = speech.plan_job("anime", True, 360, budget12)
    assert p.mode == "同時載入" and p.bs == {"asr:anime": 8, "align": 4, "verify": 24, "realign": 4}, p


def test_plan_8gb_rotates_and_keeps_what_fits():
    budget8 = 8188 - 250 - 1000 - 1024
    p = speech.plan_job("qwen", True, 480, budget8)
    # Qwen3-ASR 和對齊模型同時放不下：輪流載入，辨識可以用 3（同時放只能 2，還沒有餘裕）
    assert p.mode == "輪流載入" and p.bs["asr:qwen"] == 3 and p.loads == 5, p
    p = speech.plan_job("anime", True, 360, budget8)
    assert p.mode == "部分同時載入" and p.bs["asr:anime"] == 8, p                 # anime-whisper＋對齊可以同時放
    p = speech.plan_job("anime", False, 360, budget8)
    assert p.mode == "同時載入" and p.loads == 2, p
    try:
        speech.plan_job("qwen", False, 100, 4000)
    except speech.NotEnough as e:
        assert e.model == "qwen" and e.need == 300 + V["resident_mb"]["qwen"] + V["stage_mb"]["asr:qwen"][1], e.need
    else:
        raise AssertionError("4 GB 的預算放不下 Qwen3-ASR")


def test_plan_smaller_batch_only_when_faster():
    budget = 12227 - 300 - 2900 - 1024      # 12 GB 卡、桌面很滿：Qwen＋對齊一起放時辨識只能用 3
    loaded = {"qwen", "aligner"}
    long = speech.plan_job("qwen", True, 480, budget, loaded)
    assert long.bs["asr:qwen"] == 6 and long.loads == 1, long      # 長片：辨識前先卸載對齊模型，批次維持 6
    short = speech.plan_job("qwen", True, 20, budget, loaded)
    assert short.bs["asr:qwen"] == 3 and short.loads == 0 and short.mode == "同時載入", short  # 短片：不換模型比較快


def test_keep_for_priorities():
    loaded = {"qwen", "aligner", "anime"}
    assert speech.keep_for("qwen", "asr:qwen", 6, loaded, ["aligner", "qwen"], 12558) == loaded
    # 放不下全部：先留接下來會用到的對齊模型，別的任務留下的 anime-whisper 卸載
    need = speech.need_mb({"qwen", "aligner"}, "asr:qwen", 6)
    assert speech.keep_for("qwen", "asr:qwen", 6, loaded, ["aligner"], need) == {"qwen", "aligner"}
    # 對齊模型放不下但比較小的 anime-whisper 放得下：放得下的就留（不用重新載入），都放不下才只留自己
    assert speech.keep_for("qwen", "asr:qwen", 6, loaded, ["aligner"], need - 1) == {"qwen", "anime"}
    small = speech.need_mb({"qwen", "anime"}, "asr:qwen", 6)
    assert speech.keep_for("qwen", "asr:qwen", 6, loaded, ["aligner"], small - 1) == {"qwen"}
    # 要載入的模型不在時，載入途中多用的量也要算
    assert speech.need_mb({"aligner"}, "align", 4, "aligner") == 300 + 1760 + 1000


def test_plan_for_job_and_translation_model():
    def body(env):
        w = env.worker()
        env.patch(gpu, "query", FakeGpu(w, 16303, 2400))
        p = speech.plan_for_job("qwen", True, 300, llm_mb=0, worker=w)
        assert not p.release_llm and p.bs["asr:qwen"] == 6
        # Hy-MT2-1.8B（約 2.7 GB）開著：一起放也不影響批次，留著
        env.patch(gpu, "query", FakeGpu(w, 16303, 2400 + 2720))
        p = speech.plan_for_job("qwen", True, 300, llm_mb=2720, worker=w)
        assert not p.release_llm and p.bs["asr:qwen"] == 6, p
        # Hy-MT2-7B（約 9.4 GB）開著：放不下，先關翻譯模型
        env.patch(gpu, "query", FakeGpu(w, 16303, 2400 + 9400))
        p = speech.plan_for_job("qwen", True, 300, llm_mb=9400, worker=w)
        assert p.release_llm and p.bs["asr:qwen"] == 6, p
        # 6 GB 卡：anime-whisper 轉字幕放得下，檢查時間軸的 Qwen3-ASR 放不下 → 字幕照做、這次不檢查
        env.patch(gpu, "query", FakeGpu(w, 6144, 600, ))
        p = speech.plan_for_job("anime", True, 300, worker=w)
        assert p.health_skipped and "Qwen3-ASR" in p.health_skipped and "verify" not in p.bs, p
        try:
            speech.plan_for_job("qwen", True, 300, worker=w)
        except RuntimeError as e:
            assert "Qwen3-ASR-1.7B" in str(e) and ("顯存不足" in str(e) or "放不下" in str(e)), e
        else:
            raise AssertionError("6 GB 卡放不下 Qwen3-ASR 要說明")
        assert w.proc is None                      # 規劃不會把程序開起來
    run(body)


# ---------- 2. 工作程序 ----------

def test_worker_coloads_and_reuses_across_videos():
    def body(env):
        w = env.worker()
        env.patch(gpu, "query", w.gpu)
        for n in range(2):
            wd = workdir(env.tmp / f"job{n}")
            plan = speech.plan_for_job("qwen", True, 120, worker=w)
            assert plan.mode == "同時載入", plan
            progress = []
            r = w.run(["asr", wd, "qwen", "ja", plan.batch("asr", "qwen")], plan=plan, on_progress=progress.append)
            assert r["ok"] and r["bs"] == 6 and len(json.loads((wd / "asr.json").read_text("utf-8"))) == 4, r
            assert progress[0]["stage"] == "載入辨識模型" and progress[-1] == {"id": progress[-1]["id"], "p": 1.0,
                                                                        "stage": "語音辨識"}, progress
            assert w.run(["align", wd, "ja", plan.batch("align")], plan=plan)["ok"]
            name = cues_file(wd)
            r = w.run(["verify", wd, name, "verify.json", "ja", plan.batch("verify")], plan=plan)
            assert r["ok"] and sorted(r["loaded"]) == ["aligner", "qwen"], r
            assert w.run(["realign", wd, name, "verify.json", "fix.json", "touched.json", "ja", 4], plan=plan)["ok"]
        # 兩部影片、四個階段：每個模型只載入一次，程序只開一次，沒有爆過顯存
        assert [e["model"] for e in events(w, "load")] == ["qwen", "aligner"], events(w, "load")
        assert events(w, "unload") == [] and events(w, "oom") == [] and w.starts == 1
        batches = [e for e in events(w, "batch") if e["kind"] == "verify"]
        assert batches and all(e["bs"] == 24 for e in batches), batches      # 檢查時同一個模型改成檢查用的設定
        assert [m["key"] for m in speech.loaded_view(w)] == ["qwen", "aligner"]
    run(body)


def test_worker_rotates_on_small_card():
    def body(env):
        w = env.worker(gpu_total=8188, gpu_other=1000)
        env.patch(gpu, "query", w.gpu)
        wd = workdir(env.tmp / "job")
        plan = speech.plan_for_job("qwen", True, 120, worker=w)
        assert plan.mode == "輪流載入" and plan.bs["asr:qwen"] == 3, plan
        loaded = []
        for argv in (["asr", wd, "qwen", "ja", plan.batch("asr", "qwen")], ["align", wd, "ja", 4],
                     ["verify", wd, cues_file(wd), "verify.json", "ja", 24]):
            r = w.run(argv, plan=plan)
            assert r["ok"], r
            loaded.append(sorted(r["loaded"]))
        assert loaded == [["qwen"], ["aligner"], ["qwen"]], loaded
        assert [e["ev"] + ":" + e["model"] for e in events(w) if e["ev"] in ("load", "unload")] == [
            "load:qwen", "unload:qwen", "load:aligner", "unload:aligner", "load:qwen"]
        assert events(w, "oom") == [] and w.starts == 1
    run(body)


def test_worker_oom_unloads_first_then_halves_batch():
    def body(env):
        # 規劃時以為放得下，實際上每段多用很多（例如別台卡的工作區比較大）
        w = env.worker(fake={"per_item_mb": {"qwen": 1500}}, gpu_other=6000)
        env.patch(gpu, "query", w.gpu)
        wd = workdir(env.tmp / "job", lang_text="あいうえお")
        plan = speech.plan_for_job("qwen", True, 120, worker=w)
        assert w.run(["align", wd, "ja", 4], plan=plan)["ok"]                  # 先讓對齊模型在顯卡上
        (wd / "asr.json").unlink()
        r = w.run(["asr", wd, "qwen", "ja", 6], plan=plan)
        assert r["ok"] and r["evicted"] == ["aligner"] and r["bs"] == 3 and r["oom_retries"] == 2, r
        assert plan.bs["asr:qwen"] == 3 and plan.tight, plan                   # 這個任務之後只留當下要用的
        assert len(json.loads((wd / "asr.json").read_text("utf-8"))) == 4
        (wd / "align.json").unlink()
        r = w.run(["align", wd, "ja", 4], plan=plan)
        assert r["ok"] and sorted(r["loaded"]) == ["aligner"], r

        # 一段都放不下：回報 oom，任務失敗時說顯存不足
        set_control(w, oom_always="aligner")
        (wd / "align.json").unlink()
        r = w.run(["align", wd, "ja", 4], plan=plan)
        assert not r["ok"] and r["error"] == "oom", r
        msg = str(jobs.child_failure(r["tail"], "Qwen3-ForcedAligner-0.6B"))
        assert "顯存不足" in msg, msg
        assert w.alive()                                                       # 程序還能用
        set_control(w)
        assert w.run(["align", wd, "ja", 4], plan=plan)["ok"]
    run(body)


def test_worker_cancel_stops_between_batches_and_keeps_models():
    def body(env):
        w = env.worker()
        env.patch(gpu, "query", w.gpu)
        wd = workdir(env.tmp / "job", seconds=60, chunk=5)                    # 12 段
        ready(w)
        set_control(w, delay_s=0.3)
        cancel = threading.Event()
        threading.Timer(0.8, cancel.set).start()
        t = time.monotonic()
        r = w.run(["asr", wd, "qwen", "ja", 1], cancel_event=cancel)
        assert not r["ok"] and r["error"] == "canceled" and time.monotonic() - t < 8, r
        done = len(json.loads((wd / "asr.json").read_text("utf-8")))
        assert 1 <= done < 12, done
        assert w.alive() and "qwen" in w.loaded                                # 模型留著，下次不用重新載入
        set_control(w)
        r = w.run(["asr", wd, "qwen", "ja", 1])                               # 重試：從檢查點接著做
        assert r["ok"] and len(json.loads((wd / "asr.json").read_text("utf-8"))) == 12
        assert len(events(w, "load", "qwen")) == 1 and len(events(w, "batch", "qwen")) == 12
    run(body)


def test_worker_cancel_kills_when_stuck():
    def body(env):
        w = env.worker(settings_={"cancel_wait_s": 1.0})
        env.patch(gpu, "query", w.gpu)
        wd = workdir(env.tmp / "job")
        ready(w)
        proc = w.proc
        set_control(w, hang_on="qwen")
        cancel = threading.Event()
        threading.Timer(0.5, cancel.set).start()
        t = time.monotonic()
        try:
            w.run(["asr", wd, "qwen", "ja", 6], cancel_event=cancel)
        except speech.WorkerCanceled:
            pass
        else:
            raise AssertionError("卡住又取消時要結束程序")
        assert time.monotonic() - t < 10
        assert w.proc is None and proc.poll() is not None and w.loaded == {}
        set_control(w)
        assert w.run(["asr", wd, "qwen", "ja", 6])["ok"] and w.starts == 2
    run(body)


def test_worker_stall_is_killed():
    def body(env):
        w = env.worker()
        env.patch(gpu, "query", w.gpu)
        wd = workdir(env.tmp / "job")
        ready(w)
        w.settings["stall_s"] = 1.5
        set_control(w, hang_on="aligner")
        (wd / "asr.json").write_text(json.dumps({"0": "あい"}), encoding="utf-8")
        try:
            w.run(["align", wd, "ja", 4])
        except speech.WorkerError as e:
            assert "沒有進度" in str(e), e
        else:
            raise AssertionError("卡住要結束程序")
        assert w.proc is None
    run(body)


def test_worker_crash_restarts_and_resumes():
    def body(env):
        w = env.worker()
        env.patch(gpu, "query", w.gpu)
        wd = workdir(env.tmp / "job")
        assert w.run(["asr", wd, "qwen", "ja", 6])["ok"]
        set_control(w, crash_on="aligner")
        try:
            w.run(["align", wd, "ja", 4])
        except speech.WorkerError as e:
            err = jobs.child_failure(e.tail, "Qwen3-ForcedAligner-0.6B")
            assert "意外結束" in str(e) and "Qwen3-ForcedAligner-0.6B" in str(err), (e, err)
        else:
            raise AssertionError("程序當掉要讓這一步失敗")
        assert w.proc is None and w.loaded == {}
        set_control(w)
        r = w.run(["align", wd, "ja", 4])                                     # 下次需要時自動開新的
        assert r["ok"] and w.starts == 2 and len(json.loads((wd / "asr.json").read_text("utf-8"))) == 4

        # CUDA 本身壞掉：回報後程序自己結束，下次重開
        set_control(w, fatal_on="qwen")
        proc = w.proc
        r = w.run(["verify", wd, cues_file(wd), "verify.json", "ja", 24])
        assert not r["ok"] and r["error"] == "cuda", r
        assert proc.wait(timeout=15) == 2
        set_control(w)
        assert w.run(["verify", wd, "cues.json", "verify.json", "ja", 24])["ok"] and w.starts == 3
    run(body)


def test_worker_log_keeps_earlier_processes():
    """常駐程序重開（當掉、閒置釋放、按「釋放顯卡」）時 speech_worker.log 接在後面寫，上一個程序的錯誤還查得到；
    給任務看的錯誤尾巴（log_tail）只取現在這個程序的，不會把舊的錯誤當成這次的原因。太大時換名留一份。"""
    def body(env):
        w = env.worker()
        env.patch(gpu, "query", w.gpu)
        wd = workdir(env.tmp / "job")
        assert w.run(["asr", wd, "qwen", "ja", 6])["ok"]
        set_control(w, fatal_on="aligner")
        proc = w.proc
        r = w.run(["align", wd, "ja", 4])
        assert not r["ok"] and r["error"] == "cuda", r
        assert proc.wait(timeout=15) == 2
        assert "illegal memory access" in w.log_path.read_text("utf-8")
        set_control(w)
        assert w.run(["align", wd, "ja", 4])["ok"] and w.starts == 2
        text = w.log_path.read_text("utf-8")
        assert "illegal memory access" in text and text.count("開新的語音模型程序") == 2, text
        assert "illegal memory access" not in w.log_tail() and "ready pid=" in w.log_tail()

        env.patch(speech, "LOG_MAX_BYTES", 10)
        w.stop("測試")
        assert w.run(["align", wd, "ja", 4])["ok"] and w.starts == 3
        old = w.log_path.with_name(speech.LOG_OLD_NAME)
        assert old.exists() and "illegal memory access" in old.read_text("utf-8")
        assert w.log_path.read_text("utf-8").count("開新的語音模型程序") == 1
    run(body)


def test_worker_startup_without_cuda_explains():
    def body(env):
        w = env.worker(fake={"fail_start": "RuntimeError: No CUDA GPUs are available"})
        env.patch(gpu, "query", w.gpu)
        wd = workdir(env.tmp / "job")
        try:
            w.run(["asr", wd, "qwen", "ja", 6])
        except speech.WorkerError as e:
            err = jobs.child_failure(e.tail, "Qwen3-ASR-1.7B")
            assert isinstance(err, gpu.GpuUnsupported) and "No CUDA" not in str(err), err
        else:
            raise AssertionError("CUDA 用不了要失敗")
        assert w.proc is None
    run(body)


def test_release_model_for_delete():
    def body(env):
        w = env.worker()
        env.patch(gpu, "query", w.gpu)
        wd = workdir(env.tmp / "job", lang_text="あいうえお")
        assert w.run(["align", wd, "ja", 4])["ok"]
        assert w.run(["verify", wd, cues_file(wd), "verify.json", "ja", 24])["ok"]
        assert sorted(w.loaded) == ["aligner", "qwen"]
        speech.release_model("qwen-asr", "刪除模型", worker=w)
        assert sorted(w.loaded) == ["aligner"] and events(w, "unload", "qwen")
        speech.release_model("tsqyomi", "刪除模型", worker=w)                   # 不是語音模型：不動
        assert sorted(w.loaded) == ["aligner"]

        # 正在跑別的請求（等不到）：先記成不能再用，下一個請求一定先卸載
        assert w.run(["verify", wd, "cues.json", "verify2.json", "ja", 24])["ok"]
        set_control(w, delay_s=1.0)
        (wd / "align.json").unlink()
        t = threading.Thread(target=lambda: w.run(["align", wd, "ja", 1]))
        t.start()
        time.sleep(0.8)
        speech.release_model("qwen-asr", "刪除模型", worker=w, timeout=0.2)
        assert "qwen" in w.stale and "qwen" not in [m["key"] for m in speech.loaded_view(w)]
        t.join()
        set_control(w)
        wd2 = workdir(env.tmp / "job2")
        r = w.run(["asr", wd2, "anime", "ja", 8])
        assert r["ok"] and "qwen" not in r["loaded"] and w.stale == set(), r

        # 模型檔案被換掉：不沿用記憶體裡的舊模型
        mdir = w.dir / "models" / "asr" / "anime-whisper"
        mdir.mkdir(parents=True)
        (mdir / "model.safetensors").write_bytes(b"x" * 10)
        (wd2 / "asr.json").unlink()
        assert w.run(["asr", wd2, "anime", "ja", 8])["ok"]
        (mdir / "model.safetensors").write_bytes(b"y" * 20)
        (wd2 / "asr.json").unlink()
        assert w.run(["asr", wd2, "anime", "ja", 8])["ok"]
        assert len(events(w, "load", "anime")) == 3, events(w, "load", "anime")
    run(body)


def test_models_delete_releases_speech_model():
    """設定頁刪辨識、對齊模型時先叫常駐程序卸載（假的型錄、暫存資料夾，不碰真正的 models）。"""
    def body(env):
        calls = []
        env.patch(jobs, "release_speech", lambda mid, label: calls.append((mid, label)))
        mdir = env.tmp / "models"
        d = mdir / "asr" / "FakeASR"
        d.mkdir(parents=True)
        (d / "model.safetensors").write_bytes(b"x")
        assert not str(d.resolve()).lower().startswith(str(config.MODELS_DIR.resolve()).lower())
        catalog = {"fake-asr": {"role": "asr", "engine": "qwen", "label": "FakeASR", "kind": "hf", "dir": d,
                                "size_mb": 1, "langs": ["ja"], "id": "fake-asr"}}
        conn = db._conn
        env.patch(db, "DB_PATH", env.tmp / "test.db")
        db.init()
        for mod in (models, jobs, settings):
            env.patch(mod, "MODEL_CATALOG", catalog)
        env.patch(models, "MODELS_DIR", mdir)
        safepath.TEST_ROOTS.append(env.tmp)
        try:
            models.delete("fake-asr")
        finally:
            safepath.TEST_ROOTS.remove(env.tmp)
            db._conn.close()
            db._conn = conn
        assert calls == [("fake-asr", "FakeASR")] and not d.exists(), calls
        # jobs.release_speech 本身：常駐程序沒開時什麼都不做
        assert speech.catalog_key("qwen-asr") == "qwen" and speech.catalog_key("aligner") == "aligner"
        assert speech.catalog_key("hymt2-7b") is None
    run(body)


def test_make_room_for_translation_and_separation():
    def body(env):
        w = env.worker()
        env.patch(gpu, "query", w.gpu)
        wd = workdir(env.tmp / "job", lang_text="あいうえお")
        assert w.run(["align", wd, "ja", 4])["ok"]
        assert w.run(["verify", wd, cues_file(wd), "verify.json", "ja", 24])["ok"]
        free, margin = w.gpu()["free_mb"], w.settings["margin_mb"]
        speech.make_room(free - margin, "放得下", worker=w)                    # 連餘裕都放得下：不動
        assert sorted(w.loaded) == ["aligner", "qwen"]
        speech.make_room(free - margin + 100, "人聲分離", worker=w)            # 差一點：先卸載模型，程序留著
        assert w.alive() and w.loaded == {} and w.starts == 1
        speech.make_room(w.gpu()["free_mb"] - margin + 100, "翻譯模型", worker=w)   # 卸載後還是不夠：結束程序
        assert w.proc is None
        # keep_process=False：不夠就直接結束程序
        (wd / "align.json").unlink()
        assert w.run(["align", wd, "ja", 4])["ok"] and sorted(w.loaded) == ["aligner"]
        speech.make_room(w.gpu()["free_mb"] - margin + 100, "翻譯模型", keep_process=False, worker=w)
        assert w.proc is None
    run(body)


def test_oom_with_translation_model_loaded_releases_it_and_retries():
    """規劃時判斷翻譯模型可以一起放，跑的時候還是爆顯存（例如其他程式多用了）：關掉翻譯模型再試一次，任務不失敗。"""
    def body(env):
        je = JobEnv(env)
        try:
            w = je.w
            jid = je.job("qwen", seconds=60)
            wd = jobs.WORK_DIR / jid
            (wd / "asr.json").write_text(json.dumps({str(i): "あいうえお" for i in range(3)}), encoding="utf-8")
            llm = {"mb": 2720}
            released = []

            def release(reason=""):
                released.append(reason)
                llm["mb"] = 0
                set_control(w)                  # 翻譯模型關掉之後放得下
            env.patch(jobs, "_llm_vram_mb", lambda: llm["mb"])
            env.patch(jobs, "release_llm", release)
            ctx = jobs.JobContext(db.get_job(jid))
            ctx.speech_plan = speech.plan_for_job("qwen", False, 60, llm_mb=llm["mb"], worker=w)
            assert not ctx.speech_plan.release_llm
            set_control(w, oom_always="aligner")
            jobs.run_speech(ctx, ["align", str(wd), "ja", "4"], 0.0, 1.0, "Qwen3-ForcedAligner-0.6B")
            assert released == ["辨識時顯存不足"] and (wd / "align.json").exists(), released

            # 沒有翻譯模型可以關：照樣失敗，說明顯存不足
            set_control(w, oom_always="aligner")
            (wd / "align.json").unlink()
            try:
                jobs.run_speech(ctx, ["align", str(wd), "ja", "4"], 0.0, 1.0, "Qwen3-ForcedAligner-0.6B")
            except RuntimeError as e:
                assert "顯存不足" in str(e) and len(released) == 1, (e, released)
            else:
                raise AssertionError("放不下又沒有東西可以關時要失敗")
            set_control(w)
        finally:
            jobs.gpu_state.update(job_id=None, model=None, stage=None)
            je.close()
    run(body)


def test_worker_exits_with_server():
    def body(env):
        w = env.worker()
        env.patch(gpu, "query", w.gpu)
        assert w.run(["asr", workdir(env.tmp / "job"), "qwen", "ja", 6])["ok"]
        proc = w.proc
        proc.stdin.close()                    # 伺服器結束時 stdin 會關掉
        proc.wait(timeout=15)
        # 殘留程序的認法：命令列是 -m app.speech_worker，帶著 data\work 底下的 log
        cmd = speech.default_cmd(jobs.WORK_DIR / speech.LOG_NAME)
        work = os.path.normcase(os.path.normpath(str(jobs.WORK_DIR)))
        assert jobs._module_arg(cmd) == "app.speech_worker" and jobs._module_arg(cmd) in jobs.CHILD_MODULES
        assert any(jobs._under(a, work) for a in cmd)
        assert str(os.getpid()) in cmd                                         # 伺服器 PID 給它監看

        # 伺服器被直接關掉、留下的常駐程序：下次啟動時 kill_orphans 認得（上層程序已經不在）
        import subprocess
        import psutil
        fake_work = env.tmp / "data" / "work"
        args = [sys.executable, "-s", "-c", "import time; time.sleep(60)", "-m", "app.speech_worker",
                "--parent-pid", "1", "--log", str(fake_work / speech.LOG_NAME)]
        code = ("import subprocess, sys, json; p = subprocess.Popen(json.loads(sys.argv[1]), stdout=subprocess.DEVNULL, "
                "stderr=subprocess.DEVNULL, creationflags=0x08000000); print(p.pid)")
        out = subprocess.run([sys.executable, "-s", "-c", code, json.dumps(args)], capture_output=True, text=True,
                             timeout=60, creationflags=subprocess.CREATE_NO_WINDOW)
        orphan = int(out.stdout.strip().splitlines()[-1])
        try:
            time.sleep(1.0)
            found = {p.pid for p in jobs.find_orphans(llama=env.tmp / "no-llama.exe", python=sys.executable,
                                                      work=fake_work)}
            assert orphan in found, (orphan, found)
            other = {p.pid for p in jobs.find_orphans(llama=env.tmp / "no-llama.exe", python=sys.executable,
                                                      work=env.tmp / "other" / "work")}
            assert orphan not in other
        finally:
            try:
                psutil.Process(orphan).kill()
            except psutil.Error:
                pass
    run(body)


# ---------- 3. 任務 ----------

class JobEnv:
    def __init__(self, env):
        self.env = env
        tmp = env.tmp
        env.patch(db, "DB_PATH", tmp / "test.db")
        self._conn = db._conn
        db.init()
        env.patch(cue_mod, "SUBS_DIR", tmp / "subs")
        env.patch(jobs, "WORK_DIR", tmp / "work")
        safepath.TEST_ROOTS.append(tmp)
        env.patch(settings, "installed_engines", lambda language=None, without=(): ["qwen", "anime"])
        env.patch(jobs, "model_installed", lambda entry, variant=None: True)
        env.patch(jobs, "release_llm", lambda reason="": None)
        settings.save({"health_check": True})
        self.w = env.worker()
        env.patch(speech, "WORKER", self.w)
        env.patch(gpu, "query", self.w.gpu)

    def close(self):
        db._conn.close()
        db._conn = self._conn
        safepath.TEST_ROOTS.remove(self.env.tmp)

    def job(self, engine="qwen", seconds=60, lang="en", chunk=None):
        video = self.env.tmp / f"v{time.monotonic_ns()}.mp4"
        video.write_bytes(b"fake")
        mid = db.add_media(title="t", source="local", path=str(video), duration=seconds, language=lang)
        jid = db.add_job(mid, "transcribe", {"language": lang, "engine": engine, "sensitive": False})
        wd = jobs.WORK_DIR / jid
        workdir(wd, seconds=seconds, chunk=chunk or (20 if engine == "qwen" else 10))
        (wd / "song.json").write_text(json.dumps({"song": False}), encoding="utf-8")
        return jid


def test_transcribe_jobs_share_the_worker():
    def body(env):
        je = JobEnv(env)
        quiet = logging.getLogger("jobs")
        level = quiet.level
        quiet.setLevel(logging.ERROR)
        try:
            w = je.w
            for engine in ("qwen", "qwen", "anime"):
                jid = je.job(engine)
                db.update_job(jid, status="running")
                jobs.Worker("gpu", jobs.GPU_TYPES).execute(db.get_job(jid))
                job = db.get_job(jid)
                assert job["status"] == "done", job
                t = db.get_track(job["result"]["track_id"])
                assert t["cue_count"] > 0 and json.loads(t["health"])["checked"] == t["cue_count"], t
            # 三部影片：程序開一次；Qwen3-ASR、對齊模型、anime-whisper 各載入一次
            assert w.starts == 1, w.starts
            assert sorted(e["model"] for e in events(w, "load")) == ["aligner", "anime", "qwen"], events(w, "load")
            assert sorted(m["key"] for m in jobs.loaded_models()) == ["aligner", "anime", "qwen"]
            assert jobs.gpu_state["job_id"] is None

            # 執行中取消：程序在下一批前停下來，任務改成已取消，模型留著
            set_control(w, delay_s=0.4)
            jid = je.job("qwen", seconds=240, chunk=10)                     # 24 段，辨識分 4 批
            db.update_job(jid, status="running")
            th = threading.Thread(target=lambda: jobs.Worker("gpu", jobs.GPU_TYPES).execute(db.get_job(jid)))
            th.start()
            for _ in range(100):
                if (jobs.WORK_DIR / jid / "asr.json").exists():
                    break
                time.sleep(0.1)
            jobs.cancel(jid)
            th.join(20)
            assert not th.is_alive() and db.get_job(jid)["status"] == "canceled", db.get_job(jid)
            assert w.alive() and w.starts == 1 and "qwen" in w.loaded
        finally:
            quiet.setLevel(level)
            je.close()
    run(body)


def test_worker_that_cannot_start_falls_back_to_one_shot_children():
    """常駐程序在這台電腦上開不起來（不是顯示卡的問題）：這一步改用單次子程序，這次開機之後也都用單次子程序；
    CUDA 用不了時照樣說明原因，不改做法。"""
    def body(env):
        je = JobEnv(env)
        env.patch(speech, "_disabled", None)
        quiet = logging.getLogger("speech")
        level = quiet.level
        quiet.setLevel(logging.ERROR)
        try:
            jid = je.job("qwen")
            ctx = jobs.JobContext(db.get_job(jid))
            calls = []
            env.patch(jobs, "run_child_process",
                      lambda ctx, args, lo, hi, label, module="app.asr_child": calls.append((args[0], module)))

            cuda = env.worker("cuda", fake={"fail_start": "RuntimeError: No CUDA GPUs are available"})
            env.patch(speech, "WORKER", cuda)
            try:
                jobs.run_child(ctx, ["asr", str(ctx.workdir), "qwen", "en", "6"], 0.1, 0.5, "Qwen3-ASR-1.7B")
            except gpu.GpuUnsupported:
                pass
            else:
                raise AssertionError("CUDA 用不了要說明原因")
            assert speech.enabled() and calls == []

            broken = env.worker("broken", fake={"fail_start": "OSError: [WinError 5] 存取被拒。"})
            env.patch(speech, "WORKER", broken)
            jobs.run_child(ctx, ["asr", str(ctx.workdir), "qwen", "en", "6"], 0.1, 0.5, "Qwen3-ASR-1.7B")
            assert calls == [("asr", "app.asr_child")] and not speech.enabled() and broken.starts == 1, calls
            jobs.run_child(ctx, ["align", str(ctx.workdir), "en", "4"], 0.5, 0.8, "Qwen3-ForcedAligner-0.6B")
            assert calls[-1] == ("align", "app.asr_child") and broken.starts == 1, calls   # 不再試著開常駐程序
            assert jobs.plan_speech(ctx, "qwen", True, 60) is None
        finally:
            quiet.setLevel(level)
            jobs.gpu_state.update(job_id=None, model=None, stage=None)
            je.close()
    run(body)


class FakeLlm:
    """假的 llama-server（jobs._llm["server"]）。"""

    def __init__(self, label="FakeMT"):
        self.cfg = {"label": label}
        self.stopped = False

    def alive(self):
        return not self.stopped

    def stop(self):
        self.stopped = True


def gpu_env(env):
    """JobEnv 加上真的 release_llm、乾淨的閒置計時和佇列暫停狀態；回傳 (JobEnv, 還原函式)。"""
    je = JobEnv(env)
    env.patch(jobs, "release_llm", REAL_RELEASE_LLM)
    env.patch(jobs, "_idle", {"since": None})
    saved = dict(jobs._llm)

    def restore():
        jobs._llm.clear()
        jobs._llm.update(saved)
        jobs.gpu_state.update(job_id=None, model=None, stage=None)
        je.close()
    return je, restore


def test_idle_clock_releases_speech_and_translation_models():
    """顯卡沒事做滿 120 秒：常駐語音模型和 llama-server 一起釋放；中間有新任務進來就重新計時。"""
    def body(env):
        je, restore = gpu_env(env)
        try:
            w = je.w
            assert config.GPU_IDLE_RELEASE_S == 120 and config.SPEECH_WORKER["idle_release_s"] == 120
            assert not jobs.idle_tick(now=0) and jobs._idle["since"] is None     # 什麼都沒載入：不計時
            assert w.run(["asr", workdir(env.tmp / "job"), "qwen", "ja", 6])["ok"]
            llm = FakeLlm()
            jobs._llm.update(key="hymt", server=llm)
            assert not jobs.idle_tick(now=1000)                                   # 開始計時
            assert not jobs.idle_tick(now=1119) and w.alive() and not llm.stopped
            jid = je.job("qwen")                                                  # 第 119 秒加進新任務
            assert not jobs.idle_tick(now=1119.5) and jobs._idle["since"] is None
            assert not jobs.idle_tick(now=1400) and w.alive() and not llm.stopped  # 還在排隊：不釋放
            db.update_job(jid, status="done")
            assert not jobs.idle_tick(now=1401)                                   # 做完了：從這裡重新計時
            assert not jobs.idle_tick(now=1520) and w.alive()
            assert jobs.idle_tick(now=1521)
            assert w.proc is None and llm.stopped and jobs._llm["server"] is None
            assert not jobs.gpu_models_loaded() and jobs.loaded_models() == []
            assert not jobs.idle_tick(now=2000)                                   # 已經放掉了，不會重複釋放
        finally:
            restore()
    run(body)


def test_idle_clock_counts_waiting_jobs_but_not_a_paused_queue():
    """等網址下載完才能開始的轉字幕也算還有事做；佇列暫停時排隊的不算（執行中的還是算）。
    暫停存在資料庫裡，重開伺服器還是暫停。"""
    def body(env):
        je, restore = gpu_env(env)
        try:
            mid = db.add_media(title="t", source="url", path=None, duration=10, language="ja")
            dl = db.add_job(mid, "download", {"url": "https://example.com/v"})
            db.update_job(dl, status="running")
            assert not jobs.gpu_work_pending()                                    # 網址下載不用顯卡
            tr = db.add_job(mid, "transcribe", {"language": "ja", "engine": "qwen"}, depends_on=dl)
            assert jobs.gpu_work_pending() and not [j for j in db.runnable_jobs(jobs.GPU_TYPES) if j["id"] == tr]
            jobs.set_queue_paused(True)
            assert not jobs.gpu_work_pending()
            db.update_job(tr, status="running")
            assert jobs.gpu_work_pending()
            db.update_job(tr, status="queued")
            db._conn.close()                                                      # 重開伺服器（重新開資料庫）
            db.init()
            assert jobs.queue_paused()
            db.update_job(dl, status="done")
            worker = jobs.Worker("gpu", jobs.GPU_TYPES)
            assert worker.pick() is None                                          # 暫停中：顯卡任務不會開始
            assert jobs.resume_queue() == {"paused": False} and not jobs.queue_paused()
            assert worker.pick()["id"] == tr
            # 設定頁存檔改不到暫停狀態
            settings.save({"queue_paused": True})
            assert not jobs.queue_paused()
        finally:
            restore()
    run(body)


def test_release_button_without_and_with_queued_jobs():
    """沒有任務：馬上全部釋放。還有排隊中的任務：沒確認就不動任何東西；確認後暫停佇列再釋放；
    佇列已經暫停、沒有執行中的任務：不用再問。"""
    def body(env):
        je, restore = gpu_env(env)
        try:
            w = je.w
            assert w.run(["asr", workdir(env.tmp / "job"), "qwen", "ja", 6])["ok"]
            llm = FakeLlm()
            jobs._llm.update(key="hymt", server=llm)
            t = time.monotonic()
            r = jobs.release_gpu()
            assert time.monotonic() - t < 5
            assert r["released"] == ["Qwen3-ASR-1.7B", "FakeMT"] and not r["paused"] and r["stopped_jobs"] == [], r
            assert r["free_mb"] == w.gpu()["free_mb"] and w.proc is None and llm.stopped and jobs.loaded_models() == []

            assert w.run(["asr", workdir(env.tmp / "job2"), "qwen", "ja", 6])["ok"]
            jid = je.job("qwen")
            for status, message in (("running", jobs.QUEUE_BUSY_MESSAGE), ("queued", jobs.QUEUE_WAITING_MESSAGE)):
                db.update_job(jid, status=status)     # 有執行中的任務才說「正在跑的任務會停下」
                try:
                    jobs.release_gpu()
                except jobs.QueueBusy as e:
                    assert str(e) == message, (status, str(e))
                else:
                    raise AssertionError("佇列還有任務要先確認")
            assert w.alive() and not jobs.queue_paused() and db.get_job(jid)["status"] == "queued"

            r = jobs.release_gpu(pause_queue=True)
            assert r["paused"] and r["released"] == ["Qwen3-ASR-1.7B"] and w.proc is None, r
            assert db.get_job(jid)["status"] == "queued"
            assert jobs.release_gpu()["paused"]                                   # 已經暫停：不用再問
            jobs.resume_queue()
            assert not jobs.queue_paused()
        finally:
            restore()
    run(body)


def test_release_button_stops_running_job_and_resume_continues_from_checkpoint():
    """執行中的轉字幕按「釋放顯卡」：幾秒內停下來、顯存放掉，任務回到佇列（不算失敗）；
    繼續佇列後從檢查點接著做，做完的段落不重做。"""
    def body(env):
        je, restore = gpu_env(env)
        quiet = logging.getLogger("jobs")
        level = quiet.level
        quiet.setLevel(logging.ERROR)
        try:
            w = je.w
            set_control(w, delay_s=0.4)
            jid = je.job("qwen", seconds=240, chunk=10)                         # 24 段，辨識分 4 批
            asr = jobs.WORK_DIR / jid / "asr.json"
            worker = jobs.Worker("gpu", jobs.GPU_TYPES)
            job = worker.pick()
            assert job["id"] == jid
            th = threading.Thread(target=worker.execute, args=(job,))
            th.start()
            for _ in range(200):
                if asr.exists():
                    break
                time.sleep(0.05)
            assert asr.exists()
            t = time.monotonic()
            r = jobs.release_gpu(pause_queue=True)
            took = time.monotonic() - t
            th.join(20)
            assert not th.is_alive() and took < 10, took
            job = db.get_job(jid)
            assert job["status"] == "queued" and job["stage"] == jobs.PAUSED_STAGE and not job["error"], job
            assert r["paused"] and r["stopped_jobs"] == [jid] and w.proc is None and not jobs.gpu_models_loaded(), r
            done_before = len(json.loads(asr.read_text("utf-8")))
            assert 1 <= done_before < 24, done_before
            assert worker.pick() is None                                          # 暫停中不會把模型載回來
            assert w.proc is None

            jobs.resume_queue()
            set_control(w)
            job = worker.pick()
            assert job["id"] == jid
            worker.execute(job)
            job = db.get_job(jid)
            assert job["status"] == "done", job
            # 辨識一共 4 批：被打斷的那一批沒寫進檢查點，重做一次；做完的不重做
            assert len([e for e in events(w, "batch") if e["kind"] == "qwen"]) == 4, events(w, "batch")
        finally:
            quiet.setLevel(level)
            restore()
    run(body)


def test_release_button_interrupts_translation_without_failing_it():
    """翻譯中按「釋放顯卡」：llama-server 被關掉時翻譯會丟連線錯誤，一樣算暫停、回到佇列，不算失敗。"""
    def body(env):
        je, restore = gpu_env(env)
        quiet = logging.getLogger("jobs")
        level = quiet.level
        quiet.setLevel(logging.ERROR)
        try:
            llm = FakeLlm("Hy-MT2-7B")
            jobs._llm.update(key="hymt", server=llm)
            started = threading.Event()

            def fake_translate(ctx):
                started.set()
                while True:
                    if llm.stopped:
                        raise RuntimeError("翻譯模型的程序意外結束了，請按「重試」")
                    time.sleep(0.05)
            env.patch(jobs, "HANDLERS", {**jobs.HANDLERS, "translate": fake_translate})
            mid = db.add_media(title="t", source="local", path=str(env.tmp / "v.mp4"), duration=10, language="ja")
            jid = db.add_job(mid, "translate", {"translator": "hymt"})
            worker = jobs.Worker("gpu", jobs.GPU_TYPES)
            job = worker.pick()
            th = threading.Thread(target=worker.execute, args=(job,))
            th.start()
            assert started.wait(10)
            r = jobs.release_gpu(pause_queue=True)
            th.join(10)
            job = db.get_job(jid)
            assert job["status"] == "queued" and not job["error"] and r["released"] == ["Hy-MT2-7B"], (job, r)
            assert llm.stopped and jobs._llm["server"] is None and r["stopped_jobs"] == [jid]
        finally:
            quiet.setLevel(level)
            restore()
    run(body)


def test_release_api_asks_first_and_rejects_other_sites():
    """/api/gpu/release、/api/queue/resume：佇列有任務時回 needs_confirm；別的網站送來的一律擋掉。"""
    from fastapi.testclient import TestClient

    from app import server

    def body(env):
        je, restore = gpu_env(env)
        try:
            client = TestClient(server.app)
            je.job("qwen")
            r = client.post("/api/gpu/release", json={})
            assert r.status_code == 200 and r.json() == {"ok": False, "needs_confirm": True,
                                                         "message": jobs.QUEUE_WAITING_MESSAGE}, r.text
            evil = {"Origin": "http://evil.example"}
            assert client.post("/api/gpu/release", json={"pause_queue": True}, headers=evil).status_code == 403
            assert client.post("/api/queue/resume", headers=evil).status_code == 403
            assert not jobs.queue_paused()
            r = client.post("/api/gpu/release", json={"pause_queue": True}).json()
            assert r["ok"] and r["paused"] and r["released"] == [] and isinstance(r["free_mb"], int), r
            g = client.get("/api/state").json()["gpu"]
            assert g["queue_paused"] is True and g["loaded"] == [] and g["idle_release_s"] == 120, g
            assert client.post("/api/queue/resume").json() == {"ok": True, "paused": False}
            assert client.get("/api/state").json()["gpu"]["queue_paused"] is False
        finally:
            restore()
    run(body)


def test_clear_finished_does_not_strand_later_steps():
    """「清除已結束」「移除」刪掉已完成的前一步：後面排隊的翻譯、轉字幕照樣會開始（翻譯記住要翻哪一條字幕），
    不會永遠停在排隊、讓顯卡永遠不閒置釋放。舊版已經刪掉前一步的任務也排得到。"""
    def body(env):
        je, restore = gpu_env(env)
        try:
            w = je.w
            mid = db.add_media(title="t", source="url", path=str(env.tmp / "v.mp4"), duration=10, language="ja")
            asr = db.add_job(mid, "transcribe", {"language": "ja", "engine": "qwen"})
            tr = db.add_job(mid, "translate", {"translator": "hymt", "language": "ja"}, depends_on=asr)
            gl = db.add_job(mid, "glosses", {"translator": "hymt", "scope": "media"}, depends_on=tr)
            db.update_job(asr, status="done", result={"track_id": "track0000001"})
            dl = db.add_job(mid, "download", {"url": "https://example.com/v"})
            tc = db.add_job(mid, "transcribe", {"language": "ja", "engine": "anime"}, depends_on=dl)
            db.update_job(dl, status="done")
            failed = db.add_job(mid, "proxy", {})
            after_failed = db.add_job(mid, "health", {"track_id": "track0000001"}, depends_on=failed)
            db.update_job(failed, status="failed")
            db.cancel_dependents(failed)
            assert {j["id"] for j in db.runnable_jobs(jobs.GPU_TYPES)} == {tr, tc}

            deleted = set(db.clear_finished())
            assert deleted == {asr, dl, failed, after_failed}, deleted
            job = db.get_job(tr)
            assert job["depends_on"] is None and job["params"]["source_track_id"] == "track0000001", job
            assert jobs.translation_source(job) == "track0000001"
            assert db.get_job(tc)["depends_on"] is None and db.get_job(gl)["depends_on"] == tr
            assert [j["id"] for j in db.runnable_jobs(jobs.GPU_TYPES)] == [tr, tc]

            # 「移除」單一個已完成的任務也一樣
            db.update_job(tr, status="done", result={"track_id": "track0000002"})
            db.delete_job(tr)
            assert db.get_job(gl)["depends_on"] is None and gl in {j["id"] for j in db.runnable_jobs(jobs.GPU_TYPES)}

            # 舊版刪掉前一步留下來的：也排得到，不會卡住
            orphan = db.add_job(mid, "transcribe", {"language": "ja", "engine": "qwen"}, depends_on="bbbbbbbbbbbb")
            assert orphan in {j["id"] for j in db.runnable_jobs(jobs.GPU_TYPES)}

            # 都做完以後閒置 120 秒照常釋放
            assert w.run(["asr", workdir(env.tmp / "job"), "qwen", "ja", 6])["ok"]
            for jid in (tc, gl, orphan):
                db.update_job(jid, status="done")
            assert not jobs.gpu_work_pending()
            assert not jobs.idle_tick(now=0) and not jobs.idle_tick(now=119) and jobs.idle_tick(now=121)
            assert w.proc is None
        finally:
            restore()
    run(body)


def test_resumed_job_stage_and_remaining_time():
    """暫停或中斷後接著做：開始執行時不再顯示「已暫停，按繼續佇列接著做」「排隊中」；
    /api/state 帶著這次開始時的進度（resumed_from），網頁估剩餘時間只算這次前進的部分。"""
    from fastapi.testclient import TestClient

    from app import server

    def body(env):
        je, restore = gpu_env(env)
        try:
            client = TestClient(server.app)
            seen = {}

            def fake_transcribe(ctx):
                job = db.get_job(ctx.id)
                st = client.get("/api/state").json()
                mine = next(j for j in st["jobs"] if j["id"] == ctx.id)
                seen[ctx.id] = (job["status"], job["stage"], jobs.resume_points().get(ctx.id), mine.get("resumed_from"))
                ctx.result = {"track_id": None}
            env.patch(jobs, "HANDLERS", {**jobs.HANDLERS, "transcribe": fake_transcribe})
            mid = db.add_media(title="t", source="local", path=str(env.tmp / "v.mp4"), duration=10, language="ja")
            paused = db.add_job(mid, "transcribe", {"language": "ja", "engine": "qwen"})
            db.update_job(paused, stage=jobs.PAUSED_STAGE, progress=0.53)
            fresh = db.add_job(mid, "transcribe", {"language": "ja", "engine": "anime"})
            worker = jobs.Worker("gpu", jobs.GPU_TYPES)
            for expected in (jobs.RESUME_STAGE, jobs.START_STAGE):
                job = worker.pick()                        # 標成執行中的同一個 UPDATE 就換掉說明
                assert job["status"] == "running" and job["stage"] == expected, job
                worker.execute(job)
            assert seen[paused] == ("running", jobs.RESUME_STAGE, 0.53, 0.53), seen
            assert seen[fresh] == ("running", jobs.START_STAGE, 0.0, None), seen
            assert jobs.resume_points() == {} and db.get_job(paused)["status"] == "done"
        finally:
            restore()
    run(body)


def test_plan_speech_decides_on_translation_model():
    """jobs.plan_speech：翻譯模型開著時，一起放不影響批次就留著，不然先關；算不出佔多少也先關。"""
    def body(env):
        w = env.worker()
        env.patch(speech, "WORKER", w)
        released = []
        env.patch(jobs, "release_llm", lambda reason="": released.append(reason))

        class Server:
            cfg = {"gguf": env.tmp / "x.gguf"}

            def alive(self):
                return True

        saved = dict(jobs._llm)
        jobs._llm.update(key="hymt-mini", server=Server())
        try:
            ctx = jobs.JobContext({"id": "plan", "media_id": None, "params": {}})
            env.patch(jobs, "translator_vram_mb", lambda key, size_mb=None, cfg=None: 2720)
            env.patch(gpu, "query", FakeGpu(w, 16303, 2400 + 2720))
            plan = jobs.plan_speech(ctx, "qwen", True, 300)
            assert released == [] and ctx.speech_plan is plan and plan.bs["asr:qwen"] == 6
            env.patch(jobs, "translator_vram_mb", lambda key, size_mb=None, cfg=None: 9400)
            env.patch(gpu, "query", FakeGpu(w, 16303, 2400 + 9400))
            jobs.plan_speech(ctx, "qwen", True, 300)
            assert len(released) == 1

            def broken(key, size_mb=None, cfg=None):
                raise OSError("gguf gone")
            env.patch(jobs, "translator_vram_mb", broken)
            env.patch(gpu, "query", FakeGpu(w, 16303, 2400))
            jobs.plan_speech(ctx, "qwen", True, 300)
            assert len(released) == 2
        finally:
            jobs._llm.clear()
            jobs._llm.update(saved)
    run(body)


def test_queue_prefers_jobs_whose_model_is_loaded():
    def body(env):
        conn = db._conn
        env.patch(db, "DB_PATH", env.tmp / "test.db")
        db.init()
        try:
            mid = db.add_media(title="t", source="local", path=str(env.tmp / "v.mp4"), duration=10, language="ja")
            anime = db.add_job(mid, "transcribe", {"language": "ja", "engine": "anime"})
            qwen = db.add_job(mid, "transcribe", {"language": "ja", "engine": "qwen"})
            w = env.worker()
            env.patch(speech, "WORKER", w)

            class Alive:
                pid = 0

                def poll(self):
                    return None

                def kill(self):
                    pass

                def wait(self, timeout=None):
                    return 0
            w.proc, w.loaded = Alive(), {"qwen": 3900, "aligner": 1760}
            worker = jobs.Worker("gpu", jobs.GPU_TYPES)
            worker.last_key = "speech"
            assert worker.pick()["id"] == qwen            # Qwen3-ASR 已經載入：先跑它
            assert worker.pick()["id"] == anime
            w.proc, w.loaded = None, {}
        finally:
            db._conn.close()
            db._conn = conn
    run(body)


def test_legacy_mode_keeps_old_behaviour():
    saved = os.environ.get("VS_SPEECH_WORKER")
    os.environ["VS_SPEECH_WORKER"] = "0"
    try:
        assert not speech.enabled()
        ctx = jobs.JobContext({"id": "legacy", "media_id": None, "params": {}})
        assert jobs.plan_speech(ctx, "qwen", True, 100) is None and ctx.speech_plan is None
        seen = []
        saved_pick = gpu.pick_batch
        gpu.pick_batch = lambda table, label: seen.append(label) or 3
        try:
            assert jobs.speech_batch(ctx, "asr", "qwen") == 3
            assert jobs.speech_batch(ctx, "verify") == 12            # 跟以前一樣：辨識批次 × 4
            assert jobs.speech_batch(ctx, "realign") == 3
        finally:
            gpu.pick_batch = saved_pick
        assert seen == ["Qwen3-ASR-1.7B", "Qwen3-ASR-1.7B", "Qwen3-ForcedAligner-0.6B"]
    finally:
        if saved is None:
            os.environ.pop("VS_SPEECH_WORKER", None)
        else:
            os.environ["VS_SPEECH_WORKER"] = saved


if __name__ == "__main__":
    started = time.time()
    try:
        for name, fn in list(globals().items()):
            if name.startswith("test_") and callable(fn):
                t = time.time()
                fn()
                print(f"ok {name} ({time.time() - t:.1f}s)")
    finally:
        if BASE.exists() and not any(BASE.iterdir()):
            BASE.rmdir()
    print(f"all passed in {time.time() - started:.1f}s")
