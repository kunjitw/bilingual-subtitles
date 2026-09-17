"""佇列重試擋重複、重做已有結果要確認、模型沒下載先擋的測試。
不用顯卡、不碰正式資料：資料庫和字幕資料夾換到系統暫存資料夾的 retry-test 底下，測完整個刪掉；
模型有沒有下載用假的函式決定，不看 models 資料夾。

python -s tests/test_retry_guard.py
"""
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

from fastapi.testclient import TestClient  # noqa: E402

from app import cues as cue_mod  # noqa: E402
from app import db, jobs, server, settings  # noqa: E402

BASE = Path(tempfile.gettempdir()) / "retry-test"


class Env:
    """暫存資料庫＋暫存字幕資料夾；預設所有模型都下載好了，個別測試再改。"""

    def __init__(self):
        BASE.mkdir(parents=True, exist_ok=True)
        self.tmp = Path(tempfile.mkdtemp(prefix="vs-retry-", dir=BASE))
        self._saved = []
        self._saved_conn = db._conn
        self.patch(db, "DB_PATH", self.tmp / "test.db")
        db.init()
        self.patch(cue_mod, "SUBS_DIR", self.tmp / "subs")
        (self.tmp / "subs").mkdir()
        self.patch(jobs, "PROXY_DIR", self.tmp / "proxy")
        (self.tmp / "proxy").mkdir()
        self.engines = ["qwen", "anime", "whisper"]
        self.translators = ["hymt", "hymt-mini", "sakura", "galtransl"]
        self.aligner = True
        self.patch(settings, "installed_engines", lambda language=None, without=(): list(self.engines))
        self.patch(settings, "installed_translators", lambda language=None, without=(): list(self.translators))
        self.patch(jobs, "aligner_installed", lambda: self.aligner)
        self.client = TestClient(server.app)   # 不用 with，不會跑 lifespan（不啟動佇列、不載入字典）
        self._levels = []
        for name in ("server", "jobs"):
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
        shutil.rmtree(self.tmp, ignore_errors=True)

    # ---- 假資料 ----
    def media(self, **kw):
        fields = {"title": "test", "source": "upload", "path": str(self.tmp / "v.mp4"), "language": "ja", **kw}
        return db.add_media(**fields)

    def track(self, mid, kind="asr", lang="ja", source=None, model="Qwen3-ASR-1.7B", cues=((0, 1), (1, 2))):
        tid = db.add_track(mid, lang, kind, model, len(cues), source)
        cue_mod.save_cues(tid, [{"start": s, "end": e, "text": "x"} for s, e in cues])
        return tid

    def job(self, mid, kind, params, status="failed", depends_on=None, result=None):
        jid = db.add_job(mid, kind, params, depends_on=depends_on)
        fields = {"status": status}
        if status in ("failed", "canceled", "done"):
            fields["finished_at"] = time.time()
        if status == "failed":
            fields["error"] = "測試用的失敗"
        if result is not None:
            fields["result"] = result
        db.update_job(jid, **fields)
        return jid

    def retry(self, jid, force=None):
        body = None if force is None else {"force": force}
        return self.client.post(f"/api/jobs/{jid}/retry", json=body) if body is not None \
            else self.client.post(f"/api/jobs/{jid}/retry")


def run(fn):
    env = Env()
    try:
        fn(env)
    finally:
        env.close()


def status_of(jid):
    return db.get_job(jid)["status"]


def active_count():
    return sum(1 for j in db.list_jobs(finished_limit=0) if j["status"] in ("queued", "running"))


def expect(res, code, text=None):
    assert res.status_code == code, (res.status_code, res.text)
    if text is not None:
        assert text in res.json()["detail"], res.json()
    return res


# ---------- 重試：同一件事已經在佇列裡 ----------

def test_retry_refuses_when_same_work_is_already_queued_or_running():
    def body(env: Env):
        mid = env.media()
        src = env.track(mid)
        other_media = env.media()
        cases = []
        # 轉字幕：同一部影片、同一個模型
        cases.append((env.job(mid, "transcribe", {"language": "ja", "engine": "qwen"}),
                      env.job(mid, "transcribe", {"language": "ja", "engine": "qwen"}, status="running"),
                      "正在用 Qwen3-ASR-1.7B 轉字幕"))
        # 翻譯：同一條原文，換了模型也算
        cases.append((env.job(mid, "translate", {"translator": "hymt", "language": "ja", "source_track_id": src}),
                      env.job(mid, "translate", {"translator": "sakura", "language": "ja", "source_track_id": src},
                              status="queued"),
                      "這條字幕已經在排隊翻譯"))
        # 時間軸檢查：同一條字幕
        cases.append((env.job(mid, "health", {"track_id": src}),
                      env.job(mid, "health", {"track_id": src}, status="queued"), "這條字幕已經在排隊檢查時間軸"))
        cases.append((env.job(other_media, "proxy", {}), env.job(other_media, "proxy", {}, status="running"),
                      "這部影片正在轉檔"))
        cases.append((env.job(other_media, "download", {"url": "https://youtu.be/x"}, status="canceled"),
                      env.job(other_media, "download", {"url": "https://youtu.be/x"}, status="queued"),
                      "這部影片已經在排隊下載"))
        cases.append((env.job(None, "model", {"model": "qwen-asr"}), env.job(None, "model", {"model": "qwen-asr"}, status="running"),
                      "這個模型正在下載"))
        cases.append((env.job(None, "titles", {"translator": "hymt"}), env.job(None, "titles", {"translator": "hymt"}, status="queued"),
                      "影片標題已經在排隊翻譯"))
        # 單字釋義：全部影片的任務也涵蓋單一影片
        cases.append((env.job(mid, "glosses", {"translator": "hymt", "scope": "media"}),
                      env.job(None, "glosses", {"translator": "hymt", "scope": "library"}, status="queued"),
                      "單字釋義已經在排隊"))
        before = active_count()
        for failed, _active, text in cases:
            expect(env.retry(failed), 409, text)
            expect(env.retry(failed, force=True), 409, text)   # 確認過也不能重複排
            assert status_of(failed) in ("failed", "canceled"), db.get_job(failed)
        assert active_count() == before

        # 佇列裡的做完或取消後就可以重試
        for failed, active, _text in cases:
            db.update_job(active, status="canceled", finished_at=time.time())
        for failed, _active, _text in cases:
            res = env.retry(failed, force=True)
            assert res.status_code == 200, (db.get_job(failed)["type"], res.text)
            assert status_of(failed) == "queued"
    run(body)


def test_retry_of_pipeline_translation_matches_by_source_track():
    """新增影片時排的翻譯任務沒有 source_track_id，要從前一個轉字幕任務的結果找原文。"""
    def body(env: Env):
        mid = env.media()
        src = env.track(mid)
        t_job = env.job(mid, "transcribe", {"language": "ja", "engine": "qwen"}, status="done", result={"track_id": src})
        pipeline = env.job(mid, "translate", {"translator": "hymt", "language": "ja"}, depends_on=t_job)
        manual = env.job(mid, "translate", {"translator": "hymt", "language": "ja", "source_track_id": src}, status="running")
        expect(env.retry(pipeline, force=True), 409, "這條字幕正在翻譯")
        db.update_job(manual, status="done", finished_at=time.time())
        env.track(mid, kind="translation", lang="zh-TW", source=src, model="Hy-MT2-7B")
        expect(env.retry(pipeline), 409, "已經翻譯過了")
        expect(env.retry(pipeline, force=True), 200)
    run(body)


def test_retry_with_other_asr_model_needs_confirm():
    """同一部影片正在用別的模型轉字幕：跟「用其他模型轉字幕」一樣，確認過（force）才排。"""
    def body(env: Env):
        mid = env.media()
        failed = env.job(mid, "transcribe", {"language": "ja", "engine": "whisper"})
        env.job(mid, "transcribe", {"language": "ja", "engine": "qwen"}, status="running")
        expect(env.retry(failed), 409, "正在用其他模型轉字幕")
        assert status_of(failed) == "failed"
        expect(env.retry(failed, force=True), 200)
        assert status_of(failed) == "queued"
    run(body)


def test_double_click_and_two_tabs_only_queue_once():
    def body(env: Env):
        mid = env.media()
        src = env.track(mid)
        a = env.job(mid, "translate", {"translator": "hymt", "language": "ja", "source_track_id": src})
        b = env.job(mid, "translate", {"translator": "hymt", "language": "ja", "source_track_id": src}, status="canceled")
        results = []
        barrier = threading.Barrier(4)

        def go(jid):
            barrier.wait()
            try:
                jobs.retry(jid)
                results.append((jid, "ok"))
            except jobs.RetryRefused as e:
                results.append((jid, e.status))
        threads = [threading.Thread(target=go, args=(j,)) for j in (a, a, b, b)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert sum(1 for _, r in results if r == "ok") == 1, results
        assert all(r in ("ok", 409) for _, r in results), results
        assert active_count() == 1
    run(body)


# ---------- 重試：會重做已經有的結果要先確認 ----------

def test_retry_that_redoes_existing_results_needs_force():
    def body(env: Env):
        mid = env.media()
        src = env.track(mid)
        tr = env.job(mid, "translate", {"translator": "hymt", "language": "ja", "source_track_id": src})
        tc = env.job(mid, "transcribe", {"language": "ja", "engine": "qwen"}, status="canceled")
        hc = env.job(mid, "health", {"track_id": src})
        px = env.job(mid, "proxy", {})
        # 還沒有結果：直接排
        for jid in (tr, hc, px):
            expect(env.retry(jid), 200)
            db.update_job(jid, status="failed", finished_at=time.time())
        # 已經有結果：沒確認回 409，確認過才排
        env.track(mid, kind="translation", lang="zh-TW", source=src, model="Hy-MT2-7B")
        db.update_track(src, health={"v": "2", "bad": [], "at": 1})
        proxy = env.tmp / "proxy" / f"{mid}.mp4"
        proxy.write_bytes(b"x")
        db.update_media(mid, proxy_path=str(proxy))
        for jid, text in ((tr, "已經翻譯過了"), (tc, "已經有字幕了"), (hc, "已經檢查過時間軸"), (px, "已經有相容播放檔")):
            expect(env.retry(jid), 409, text)
            expect(env.retry(jid, force=False), 409, text)
            assert status_of(jid) in ("failed", "canceled")
            expect(env.retry(jid, force=True), 200)
            assert status_of(jid) == "queued"
    run(body)


def test_retry_requeues_dependents_and_refuses_finished_or_missing_jobs():
    def body(env: Env):
        mid = env.media()
        dl = env.job(mid, "download", {"url": "https://youtu.be/x"})
        tc = env.job(mid, "transcribe", {"language": "ja", "engine": "qwen"}, status="queued", depends_on=dl)
        db.cancel_dependents(dl)
        assert status_of(tc) == "canceled"
        # 只重試後面那一步會卡在「等前一步」：擋下並說要重試前一個
        expect(env.retry(tc), 400, "請重試前一個任務")
        orphan = env.job(mid, "translate", {"translator": "hymt", "language": "ja"}, status="canceled", depends_on="bbbbbbbbbbbb")
        expect(env.retry(orphan), 400, "前一個步驟的任務已經移除了")
        expect(env.retry(dl), 200)
        assert status_of(dl) == "queued" and status_of(tc) == "queued"
        expect(env.retry(dl), 409, "已經在佇列裡")
        done = env.job(mid, "proxy", {}, status="done")
        expect(env.retry(done), 409, "已經完成")
        expect(env.retry("aaaaaaaaaaaa"), 404)
    run(body)


# ---------- 重試：一定會再失敗的先擋下 ----------

def test_retry_refuses_missing_models_and_deleted_inputs():
    def body(env: Env):
        mid = env.media()
        src = env.track(mid)
        tc = env.job(mid, "transcribe", {"language": "ja", "engine": "whisper"})
        tr = env.job(mid, "translate", {"translator": "sakura", "language": "ja", "source_track_id": src})
        hc = env.job(mid, "health", {"track_id": src})
        ti = env.job(None, "titles", {"translator": "hymt"})
        gl = env.job(mid, "glosses", {"translator": "hymt", "scope": "media"})
        env.engines = ["qwen"]
        env.translators = []
        expect(env.retry(tc, force=True), 400, "Whisper large-v3 還沒下載")
        expect(env.retry(tr, force=True), 400, "Sakura-14B 還沒下載")
        expect(env.retry(ti, force=True), 400, "Hy-MT2-7B 還沒下載")
        expect(env.retry(gl, force=True), 400, "單字釋義需要")
        env.translators = ["hymt-mini"]
        expect(env.retry(gl), 200)            # 釋義會改用 Hy-MT2-1.8B
        env.engines = []
        expect(env.retry(hc, force=True), 400, "Qwen3-ASR-1.7B 還沒下載")
        env.engines = ["qwen", "whisper"]
        env.aligner = False
        expect(env.retry(tc, force=True), 400, "還沒下載")
        expect(env.retry(hc, force=True), 400, "還沒下載")
        env.aligner = True
        for jid in (tc, tr, hc, ti):
            assert status_of(jid) == "failed"
        # 字幕或影片已經刪掉
        db.delete_track(src)
        expect(env.retry(hc, force=True), 400, "字幕已經刪除")
        env.translators = ["sakura"]
        expect(env.retry(tr, force=True), 400, "原文字幕已經刪除")
        gone = env.media()
        px = env.job(gone, "proxy", {}, status="running")
        db.delete_media(gone)     # 刪影片時執行中的任務會留下來，之後才變成失敗
        db.update_job(px, status="failed", finished_at=time.time())
        expect(env.retry(px, force=True), 400, "影片已經刪除")
    run(body)


# ---------- 排翻譯前先確認模型下載了 ----------

def test_translate_track_requires_downloaded_model():
    def body(env: Env):
        mid = env.media()
        src = env.track(mid)
        env.translators = ["hymt"]
        expect(env.client.post(f"/api/tracks/{src}/translate", json={"translator": "sakura"}), 400, "Sakura-14B 還沒下載")
        assert active_count() == 0
        expect(env.client.post(f"/api/tracks/{src}/translate", json={"translator": "hymt"}), 200)
        assert active_count() == 1
        expect(env.client.post(f"/api/tracks/{src}/translate", json={"translator": "hymt"}), 409, "正在翻譯")
        db.update_job(db.list_jobs(finished_limit=0)[0]["id"], status="done", finished_at=time.time())
        env.track(mid, kind="translation", lang="zh-TW", source=src, model="Hy-MT2-7B")
        # 已經翻譯過而且模型沒下載：先說模型的事，免得使用者確認完才知道
        expect(env.client.post(f"/api/tracks/{src}/translate", json={"translator": "sakura"}), 400, "還沒下載")
        expect(env.client.post(f"/api/tracks/{src}/translate", json={"translator": "hymt"}), 409, "已經翻譯過了")
        expect(env.client.post(f"/api/tracks/{src}/translate", json={"translator": "hymt", "force": True}), 200)
    run(body)


def test_redo_and_title_translation_require_downloaded_model():
    def body(env: Env):
        mid = env.media(title="タイトル")
        src = env.track(mid, cues=((0, 1), (1, 2)))
        env.track(mid, kind="translation", lang="zh-TW", source=src, model="Hy-MT2-7B", cues=((0, 2),))  # 整句模式的舊翻譯
        env.translators = []
        expect(env.client.post("/api/translations/redo"), 400, "Hy-MT2-7B 還沒下載")
        expect(env.client.post("/api/titles/translate"), 400, "Hy-MT2-7B 還沒下載")
        assert active_count() == 0
        # 原本的模型刪掉了、還有別的：改用已下載的
        env.translators = ["hymt-mini"]
        res = expect(env.client.post("/api/translations/redo"), 200)
        assert res.json()["count"] == 1
        job = next(j for j in db.list_jobs(finished_limit=0) if j["type"] == "translate")
        assert job["params"]["translator"] == "hymt-mini", job
        res = expect(env.client.post("/api/titles/translate"), 200)
        assert res.json()["count"] == 1
    run(body)


def test_upload_precheck_reports_missing_models_before_upload():
    def body(env: Env):
        post = lambda **kw: env.client.post("/api/media/precheck", json=kw)  # noqa: E731
        expect(post(language="xx"), 400, "請選擇影片語言")
        env.engines = []
        expect(post(language="ja"), 400, "還沒下載")
        env.engines = ["qwen"]
        env.aligner = False
        expect(post(language="ja"), 400, "還沒下載")
        env.aligner = True
        env.translators = []
        expect(post(language="ja", translate=True), 400, "還沒下載")
        expect(post(language="ja", translate=False), 200)
        expect(post(language="zh", translate=True), 200)      # 中文不用翻譯模型
        env.translators = ["hymt"]
        expect(post(language="ja", translate=True), 200)
        assert not db.list_media() and not db.list_jobs()
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
