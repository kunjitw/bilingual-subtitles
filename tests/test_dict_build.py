"""字典建置（app/dict_build.py）的測試：建日文、英文、中文字典，解壓 stardict.7z（py7zr，開發環境沒有時用 7-Zip），
下載續傳、sha256、JMdict 換了 ETag、gzip 太大，換檔（連線開著、被鎖住、下次啟動換上），佇列任務的子程序
（優先權、TEMP、取消、錯誤代碼、壞掉的來源自動重下一次），沒有 py7zr 時的處理，命令列工具。

不連網、不用顯卡：data、models 都在系統暫存資料夾的 dict-build-test 底下（VS_DATA_DIR、VS_MODELS_DIR），測完整個刪掉；
下載來源是 127.0.0.1 上的假伺服器，沒指定時 VS_DICT_SOURCE_BASE 指到連不上的 127.0.0.1:9，不會意外連到外面。

python -s tests/test_dict_build.py
python -s tests/test_dict_build.py --real     手動：用 data\\dict\\src 的真實來源（唯讀複製）在暫存資料夾建日文、英文、中文字典，
                                             記錄時間和大小；來源的 sha256 對不上時會連網下載固定版本
"""
import io
import json
import logging
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

BASE = Path(tempfile.gettempdir()) / "dict-build-test"
RUN = BASE / f"run-{os.getpid()}"
os.environ["VS_DATA_DIR"] = str(RUN / "data")
os.environ["VS_MODELS_DIR"] = str(RUN / "models")
os.environ["VS_NO_WORKERS"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["VS_DICT_SOURCE_BASE"] = "http://127.0.0.1:9"      # 預設連不上：測試不會意外下載真的來源
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import fake_sources as fk  # noqa: E402
from app import config, db, dict_build, dict_en, dict_ja, jobs, safepath, vocab_zh  # noqa: E402
from app import model_download as md  # noqa: E402
from app.model_download import DownloadError  # noqa: E402

REAL_DICT = ROOT / "data" / "dict"
MIB = 1024 * 1024


class Env:
    def __init__(self):
        self.dict = config.DICT_DIR
        assert str(self.dict).startswith(str(BASE)), self.dict          # 絕對不能動到真的 data\dict
        shutil.rmtree(self.dict, ignore_errors=True)
        self.dict.mkdir(parents=True, exist_ok=True)
        self.src = self.dict / "src"
        self.src.mkdir()
        config.WORK_DIR.mkdir(parents=True, exist_ok=True)
        self._saved = []
        self._saved_conn = db._conn
        self.patch(db, "DB_PATH", RUN / "data" / f"test-{time.time_ns()}.db")
        db.init()
        self.patch(dict_build, "MIN_ENTRIES", {"ja": 10, "en": 10, "zh": 10})
        self.patch(dict_build, "JMDICT_MIN_ENTRIES", 10)
        self.patch(dict_build, "SWAP_WAITS", (0.05,))
        self.patch(md, "QUICK_WAITS", ())
        self.patch(jobs, "DOWNLOAD_RETRY_WAITS", {"corrupt": (0,)})
        from app import vocab
        self.patch(vocab, "kick", lambda: None)   # 換上字典時會叫查字索引重算，測試的資料庫沒有那些表
        self.sources = {}                  # 換過大小、sha256 的來源（也要帶進子程序）
        self.srv = None
        self._levels = []
        for name in ("dict_build", "safepath", "jobs"):
            logger = logging.getLogger(name)
            self._levels.append((logger, logger.level))
            logger.setLevel(logging.CRITICAL)

    def patch(self, obj, name, value):
        self._saved.append((obj, name, getattr(obj, name)))
        setattr(obj, name, value)

    def set_sources(self, lang: str, files: dict[str, bytes], write: bool = True):
        """來源的大小和 sha256 換成假檔案的；write 時直接放進 data\\dict\\src。"""
        for s in dict_build.SOURCES[lang]:
            if s["name"] in files and not s.get("daily"):
                self._saved.append((s, None, dict(s)))
                s.update(bytes=len(files[s["name"]]), sha256=fk.sha(files[s["name"]]))
                self.sources.setdefault(lang, {})[s["name"]] = {"bytes": s["bytes"], "sha256": s["sha256"]}
        if write:
            for name, data in files.items():
                (self.src / name).write_bytes(data)

    def serve(self, files: dict[str, bytes], **opts) -> fk.FakeFiles:
        if self.srv is None:
            self.srv = fk.FakeFiles()
            os.environ["VS_DICT_SOURCE_BASE"] = self.srv.base
        for name, data in files.items():
            self.srv.add(f"/{name}", data, **opts.get(name, {}))
        return self.srv

    def spec(self, **extra) -> Path:
        spec = {"test_roots": [], "min_entries": dict_build.MIN_ENTRIES, "jmdict_min": dict_build.JMDICT_MIN_ENTRIES,
                "sources": self.sources, **extra}
        return fk.write_spec(RUN, f"spec-{time.time_ns()}.json", spec)

    def close(self):
        for obj, name, value in reversed(self._saved):
            if name is None:
                obj.clear()
                obj.update(value)
            else:
                setattr(obj, name, value)
        for logger, level in self._levels:
            logger.setLevel(level)
        if self.srv:
            self.srv.stop()
        os.environ["VS_DICT_SOURCE_BASE"] = "http://127.0.0.1:9"
        dict_ja.close_all()
        dict_en.close_all()
        vocab_zh.close_all()
        db._conn.close()
        db._conn = self._saved_conn
        shutil.rmtree(self.dict, ignore_errors=True)


def run(fn):
    env = Env()
    try:
        fn(env)
    finally:
        env.close()


def expect_error(code, fn, *args, **kwargs) -> DownloadError:
    try:
        fn(*args, **kwargs)
    except DownloadError as e:
        assert e.code == code, (e.code, str(e))
        return e
    raise AssertionError(f"預期錯誤代碼 {code}")


def progress_lines(out: io.StringIO) -> list[dict]:
    return [json.loads(l) for l in out.getvalue().splitlines() if l.startswith("{")]


# ---------- 建置 ----------

def test_build_ja_and_swap_with_open_connection():
    def body(env: Env):
        files = fk.ja_sources(60)
        env.set_sources("ja", files)
        out = io.StringIO()
        result = dict_build.build("ja", out=out)
        assert result["lang"] == "ja" and result["entries"] == 62, result
        lines = progress_lines(out)
        stages = [l["stage"] for l in lines]
        assert stages[0] == "download" and "build" in stages and stages[-1] == "finish", stages
        values = [l["progress"] for l in lines]
        assert values == sorted(values) and values[-1] == 1.0 and 0.05 < max(v for l, v in zip(lines, values)
                                                                              if l["stage"] == "download") <= 0.10
        tmp, ok = env.dict / "jmdict.db.tmp", env.dict / "jmdict.db.tmp.ok"
        assert tmp.is_file() and ok.is_file() and not (env.dict / "jmdict.db").exists()
        assert json.loads(ok.read_text("utf-8"))["entries"] == 62
        con = sqlite3.connect(tmp)
        try:
            tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
            assert tables == {"meta", "entry", "form", "jlpt", "accent"}, tables
            meta = dict(con.execute("SELECT key, value FROM meta"))
            assert meta["jmdict_date"] == "2026-09-17" and meta["entries"] == "62" and meta["built_at"]
            assert con.execute("SELECT level FROM jlpt WHERE entry_id = 1358280").fetchone() == (5,)
            assert con.execute("SELECT pattern FROM accent WHERE term = '食べる'").fetchone() == ("2",)
        finally:
            con.close()
        # 第一次換上：之後用 dict_ja 開著連線查字
        from app import vocab
        before_version = vocab.index_version("ja")
        assert dict_build.swap("ja", kick=False)["entries"] == 62
        assert dict_ja.ready() and not tmp.exists() and not ok.exists()
        assert dict_ja.entries_for("食べる") and dict_ja.meta("entries") == "62"
        v1 = vocab.index_version("ja")
        assert v1 != before_version
        time.sleep(1.1)                                   # built_at 到秒，重建後要不一樣
        dict_build.build("ja", out=io.StringIO())
        assert dict_ja.con().execute("SELECT COUNT(*) FROM entry").fetchone()[0] == 62   # 舊檔的連線開著
        dict_build.swap("ja", kick=False)
        assert vocab.index_version("ja") != v1 and dict_ja.entries_for("私")
        assert (env.dict / dict_build.RULES_NAME).read_bytes() == dict_build.RULES_SRC.read_bytes()
    run(body)


def _en_files(rows=300, archive=None):
    csv_bytes = fk.stardict_csv(rows)
    return csv_bytes, {"stardict.7z": archive if archive is not None else fk.make_7z({"stardict.csv": csv_bytes}),
                       **fk.en_extra_sources()}


def test_build_en_and_zh():
    def body(env: Env):
        if not fk.can_make_7z():
            print("  （沒有 py7zr 也沒有 7-Zip，跳過英文字典）")
        else:
            _csv, files = _en_files()
            env.set_sources("en", files)
            env.patch(dict_build, "ECDICT_ROWS", 303)
            out = io.StringIO()
            result = dict_build.build("en", out=out)
            assert result["entries"] == 303, result          # give、give up（片語，第一個詞常見）、laser 加上 300 個假的
            stages = [l["stage"] for l in progress_lines(out)]
            assert "extract" in stages and "build" in stages
            assert not (env.src / "_ecdict").exists() and (env.src / "stardict.7z").is_file()   # 解出的 csv 刪掉、7z 留著
            dict_build.swap("en", kick=False)
            assert dict_en.ready()
            assert "雷射" in dict_en.entry("laser")["lines"][0], dict_en.entry("laser")
            assert dict_en.cefr("laser") == "B2" and dict_en.meta("cefr") == "2"
            print(f"  英文用 {'py7zr' if fk.py7zr_available() else '7-Zip'} 解壓")
        files = {"dict-revised.json.xz": fk.moedict_xz(30)}
        env.set_sources("zh", files)
        assert dict_build.build("zh", out=io.StringIO())["entries"] == 31
        dict_build.swap("zh", kick=False)
        assert vocab_zh.in_dict("辭典") and vocab_zh.detail("辭典")["bopomofo"] == "ㄘˊ ㄉㄧㄢˇ"
        # 換檔時連中文辭典的連線也關掉
        files2 = {"dict-revised.json.xz": fk.moedict_xz(40)}
        env.set_sources("zh", files2)
        dict_build.build("zh", out=io.StringIO())
        dict_build.swap("zh", kick=False)
        assert vocab_zh.in_dict("詞35")
    run(body)


def test_bad_archives_are_refused():
    def body(env: Env):
        if not fk.can_make_7z():
            print("  （沒有 py7zr 也沒有 7-Zip，跳過）")
            return
        csv_bytes = fk.stardict_csv(3000)
        cases = {
            "wrong name": fk.make_7z({"x.csv": csv_bytes}),
            "two files": fk.make_7z({"stardict.csv": csv_bytes, "extra.txt": b"x"}),
            "in a folder": fk.make_7z({"sub/stardict.csv": csv_bytes}),
        }
        good = fk.make_7z({"stardict.csv": csv_bytes})
        bad = bytearray(good)
        bad[len(bad) // 2] ^= 0xFF                        # 壓縮資料中間改壞一個位元組
        cases["corrupt byte"] = bytes(bad)
        cases["truncated"] = good[: len(good) // 2]
        for name, archive in cases.items():
            _csv, files = _en_files(archive=archive)
            env.set_sources("en", files)
            err = expect_error("corrupt_source", dict_build.build, "en", out=io.StringIO())
            assert "按「重試」會重新下載" in str(err), (name, str(err))
            assert not (env.src / "stardict.7z").exists(), name              # 壞掉的來源刪掉，重試時重新下載
            assert not (env.src / "_ecdict.tmp").exists() and not (env.src / "_ecdict").exists(), name
            assert not (env.dict / "en_zh.db.tmp").exists() and not (env.dict / "en_zh.db.tmp.ok").exists(), name
        # 解壓後太大
        env.patch(dict_build, "ECDICT_MAX_BYTES", 1000)
        _csv, files = _en_files(archive=good)
        env.set_sources("en", files)
        err = expect_error("corrupt_source", dict_build.build, "en", out=io.StringIO())
        assert "太大" in str(err)
    run(body)


def test_without_py7zr_uses_7zip_or_says_run_start_bat():
    def body(env: Env):
        if not fk.can_make_7z():
            print("  （沒有 py7zr 也沒有 7-Zip，跳過）")
            return
        _csv, files = _en_files()
        env.set_sources("en", files)
        saved = sys.modules.get("py7zr", "missing")
        sys.modules["py7zr"] = None                       # import py7zr 會失敗
        try:
            env.patch(dict_build, "SEVEN_ZIP", RUN / "no-7z.exe")
            err = expect_error("no_7z", dict_build.build, "en", out=io.StringIO())
            assert str(err) == "缺少 py7zr，請重新執行 start.bat"
            assert (env.src / "stardict.7z").is_file()     # 來源沒壞，留著
            if fk.SEVEN_ZIP.is_file():
                env.patch(dict_build, "SEVEN_ZIP", fk.SEVEN_ZIP)
                env.patch(dict_build, "ECDICT_ROWS", 303)
                assert dict_build.build("en", out=io.StringIO())["entries"] == 303
            else:
                print("  （沒有 7-Zip，只測了缺少 py7zr 的訊息）")
        finally:
            if saved == "missing":
                sys.modules.pop("py7zr", None)
            else:
                sys.modules["py7zr"] = saved
    run(body)


# ---------- 下載 ----------

def test_downloads_resume_check_sha_and_jmdict_etag():
    def body(env: Env):
        files = fk.ja_sources(80)
        env.set_sources("ja", files, write=False)
        jm = files["JMdict_e.gz"]
        srv = env.serve(files, **{"JMdict_e.gz": {"etag": '"v1"', "cut": [len(jm) // 2]}})
        # 下載到一半斷線：.part 和 ETag 紀錄留著
        expect_error("network", dict_build.build, "ja", out=io.StringIO())
        part = env.src / "JMdict_e.gz.part"
        assert part.is_file() and part.stat().st_size == len(jm) // 2
        assert json.loads((env.src / "JMdict_e.gz.part.json").read_text("utf-8"))["etag"] == '"v1"'
        # 重開：ETag 一樣，從 .part 接著下載
        dict_build.build("ja", out=io.StringIO())
        assert srv.ranges("/JMdict_e.gz") == [None, f"bytes={len(jm) // 2}-"], srv.log
        assert not (env.src / "JMdict_e.gz.part.json").exists() and (env.src / "JMdict_e.gz").read_bytes() == jm
        assert len(srv.ranges("/accents.txt")) == 1
        # 已經有、驗得過的來源不重下載；--force-download 只重抓每天更新的 JMdict
        n = len(srv.log)
        dict_build.build("ja", out=io.StringIO())
        assert [x for x in srv.log[n:] if x[0] == "GET"] == []
        dict_build.build("ja", force=True, out=io.StringIO())
        assert [x[1] for x in srv.log[n:] if x[0] == "GET"] == ["/JMdict_e.gz"], srv.log[n:]
        # 伺服器上的 JMdict 換了（ETag 不同）：上次下載到一半的 .part 丟掉從頭下載
        (env.src / "JMdict_e.gz").unlink()
        part.write_bytes(jm[:1000])
        (env.src / "JMdict_e.gz.part.json").write_text(json.dumps({"etag": '"v1"', "size": len(jm)}), encoding="utf-8")
        jm2 = fk.gz(fk.jmdict_xml(90))
        srv.add("/JMdict_e.gz", jm2, etag='"v2"')
        n = len(srv.log)
        dict_build.build("ja", out=io.StringIO())
        assert srv.ranges("/JMdict_e.gz")[-1] is None and (env.src / "JMdict_e.gz").read_bytes() == jm2
        # 固定版本的來源內容不對（大小一樣）：corrupt，暫存檔刪掉
        (env.src / "accents.txt").unlink()
        bad = bytearray(files["accents.txt"])
        bad[0] ^= 1
        srv.add("/accents.txt", bytes(bad))
        expect_error("corrupt", dict_build.build, "ja", out=io.StringIO())
        assert not (env.src / "accents.txt.part").exists() and not (env.src / "accents.txt").exists()
        # 找不到：not_found
        srv.files.pop("/accents.txt")
        expect_error("not_found", dict_build.build, "ja", out=io.StringIO())
        srv.add("/accents.txt", files["accents.txt"])
        # JMdict 解壓後太大、條目太少：corrupt_source，下載的檔案刪掉
        env.patch(dict_build, "JMDICT_MAX_BYTES", 5000)
        (env.src / "JMdict_e.gz").unlink()
        err = expect_error("corrupt_source", dict_build.build, "ja", out=io.StringIO())
        assert "太大" in str(err) and not (env.src / "JMdict_e.gz").exists()
        env.patch(dict_build, "JMDICT_MAX_BYTES", 256 * MIB)
        env.patch(dict_build, "JMDICT_MIN_ENTRIES", 1000)
        expect_error("corrupt_source", dict_build.build, "ja", out=io.StringIO())
        assert not (env.src / "JMdict_e.gz").exists()
        # 截斷的 gzip
        env.patch(dict_build, "JMDICT_MIN_ENTRIES", 10)
        (env.src / "JMdict_e.gz").write_bytes(jm2[: len(jm2) // 2])
        srv.add("/JMdict_e.gz", jm2, etag='"v2"')
        n = len(srv.log)
        dict_build.build("ja", out=io.StringIO())          # 已經有的檔案驗不過：刪掉重新下載
        gets = [x[1] for x in srv.log[n:] if x[0] == "GET"]
        # accents.txt 在前面壞掉時刪了，這次一起補下載；其他來源已經有、sha256 對，不重下載
        assert gets == ["/JMdict_e.gz", "/accents.txt"], gets
        cut = env.src / "cut.gz"
        cut.write_bytes(jm2[: len(jm2) // 2])
        expect_error("corrupt_source", dict_build.check_jmdict, cut)
    run(body)


def test_offline_folder_is_used_when_sha256_matches():
    """完整包的 offline\\：sha256 對就直接用、一個請求都不發；不對、沒有才下載那一個；沒有 offline 資料夾時照常下載。"""
    def body(env: Env):
        files = fk.ja_sources(40)
        env.set_sources("ja", files, write=False)
        srv = env.serve(files, **{"JMdict_e.gz": {"etag": '"v1"'}})
        off = RUN / "offline"
        shutil.rmtree(off, ignore_errors=True)
        off.mkdir(parents=True)
        for name, data in files.items():
            (off / name).write_bytes(data)
        jm = files["JMdict_e.gz"]

        def listing(sha=None):
            (off / dict_build.OFFLINE_LIST).write_text(json.dumps({"format": 1, "files": [
                {"name": "JMdict_e.gz", "size": len(jm), "sha256": sha or fk.sha(jm), "for": "dict:ja"}]}),
                encoding="utf-8")

        def gets(n):
            return sorted(x[1] for x in srv.log[n:] if x[0] in ("GET", "HEAD"))

        def reset_src():
            shutil.rmtree(env.src, ignore_errors=True)
            env.src.mkdir()

        # 1. 沒有 offline 資料夾（一般的 zip）：全部照常下載
        env.patch(dict_build, "OFFLINE_DIR", RUN / "no-offline")
        n = len(srv.log)
        assert dict_build.build("ja", out=io.StringIO())["entries"] == 42
        assert "/accents.txt" in gets(n) and "/JMdict_e.gz" in gets(n) and len([x for x in gets(n) if x != "/JMdict_e.gz"]) == 6

        # 2. offline 全部對：不發任何請求，來源放進 data\dict\src，offline 的檔案留著
        env.patch(dict_build, "OFFLINE_DIR", off)
        listing()
        reset_src()
        n = len(srv.log)
        assert dict_build.build("ja", out=io.StringIO())["entries"] == 42
        assert gets(n) == [], srv.log[n:]
        for name, data in files.items():
            assert (env.src / name).read_bytes() == data and (off / name).read_bytes() == data, name
        assert not list(env.src.glob("*.offline"))

        # 3. sha256 不對（大小一樣）、缺檔、JMdict 在清單裡的 sha256 不對：只下載那幾個，offline 的壞檔不動
        bad = bytearray(files["accents.txt"])
        bad[0] ^= 1
        (off / "accents.txt").write_bytes(bytes(bad))
        (off / "jlpt_n3.csv").unlink()
        listing("0" * 64)
        reset_src()
        n = len(srv.log)
        dict_build.build("ja", out=io.StringIO())
        assert gets(n) == ["/JMdict_e.gz", "/JMdict_e.gz", "/accents.txt", "/jlpt_n3.csv"], gets(n)   # JMdict 是 HEAD + GET
        assert (env.src / "accents.txt").read_bytes() == files["accents.txt"] and (off / "accents.txt").read_bytes() == bytes(bad)

        # 4. 清單裡沒有 JMdict、sha256 對但內容驗不過（條目太少）：下載
        (off / dict_build.OFFLINE_LIST).unlink()
        reset_src()
        n = len(srv.log)
        dict_build.build("ja", out=io.StringIO())
        assert "/JMdict_e.gz" in gets(n)
        few = fk.gz(fk.jmdict_xml(0))
        (off / "JMdict_e.gz").write_bytes(few)
        (off / dict_build.OFFLINE_LIST).write_text(json.dumps({"files": [
            {"name": "JMdict_e.gz", "size": len(few), "sha256": fk.sha(few)}]}), encoding="utf-8")
        (env.src / "JMdict_e.gz").unlink()
        n = len(srv.log)
        dict_build.build("ja", out=io.StringIO())
        assert gets(n) == ["/JMdict_e.gz", "/JMdict_e.gz"] and (env.src / "JMdict_e.gz").read_bytes() == jm, gets(n)

        # 5. --force-download 要最新的 JMdict：不用 offline 的
        (off / "JMdict_e.gz").write_bytes(jm)
        listing()
        n = len(srv.log)
        dict_build.build("ja", force=True, out=io.StringIO())
        assert gets(n) == ["/JMdict_e.gz", "/JMdict_e.gz"], gets(n)
        shutil.rmtree(off, ignore_errors=True)
    run(body)


# ---------- 換檔 ----------

def test_locked_dictionary_is_swapped_at_next_start():
    def body(env: Env):
        env.set_sources("ja", fk.ja_sources(30))
        dict_build.build("ja", out=io.StringIO())
        dict_build.swap("ja", kick=False)
        dict_build.build("ja", out=io.StringIO())
        tmp, ok = env.dict / "jmdict.db.tmp", env.dict / "jmdict.db.tmp.ok"
        holder = sqlite3.connect(env.dict / "jmdict.db")      # 別的程式開著舊檔
        holder.execute("SELECT COUNT(*) FROM entry").fetchone()
        try:
            try:
                dict_build.swap("ja", kick=False)
            except dict_build.DictLocked as e:
                assert "重新打開程式後會自動換上" in str(e)
            else:
                raise AssertionError("檔案被鎖住時應該換不掉")
            assert tmp.is_file() and ok.is_file()
            assert dict_build.finish_pending() == []            # 還鎖著：留著
            assert tmp.is_file() and ok.is_file()
        finally:
            holder.close()
        new_size = tmp.stat().st_size
        assert dict_build.finish_pending() == ["ja"]
        assert not tmp.exists() and not ok.exists() and (env.dict / "jmdict.db").stat().st_size == new_size
        # 沒有 .ok 的 tmp（建到一半被結束）、解壓到一半的資料夾：刪掉
        (env.dict / "en_zh.db.tmp").write_bytes(b"half")
        (env.dict / "zh_dict.db.tmp.ok").write_text("{}", encoding="utf-8")
        (env.dict / "yomitan_ja_transforms.json.tmp").write_text("{}", encoding="utf-8")
        (env.src / "_ecdict.tmp").mkdir()
        (env.src / "_ecdict.tmp" / "stardict.csv").write_bytes(b"x")
        assert dict_build.finish_pending() == []
        assert not (env.dict / "en_zh.db.tmp").exists() and not (env.dict / "zh_dict.db.tmp.ok").exists()
        assert not (env.dict / "yomitan_ja_transforms.json.tmp").exists() and not (env.src / "_ecdict.tmp").exists()
        assert dict_ja.ready()
    run(body)


# ---------- 佇列任務（子程序） ----------

def _start_job(lang: str, **params) -> tuple[str, threading.Thread]:
    jid = db.add_job(None, "dict", {"lang": lang, "label": dict_build.LABELS[lang], "setup": True, **params})
    job = db.claim_job(jid)
    th = threading.Thread(target=jobs.Worker("dict", jobs.DICT_TYPES).execute, args=(job,), daemon=True)
    th.start()
    return jid, th


def _wait(cond, timeout=60.0, what="條件"):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return
        time.sleep(0.05)
    raise AssertionError(f"等不到{what}")


def _etilqs() -> set:
    temp = Path(os.environ.get("TEMP") or tempfile.gettempdir())
    try:
        return {p.name for p in temp.iterdir() if p.name.startswith("etilqs_")}
    except OSError:
        return set()


def test_dict_job_runs_child_below_normal_and_swaps():
    def body(env: Env):
        import psutil
        files = fk.ja_sources(40)
        env.set_sources("ja", files, write=False)
        env.serve(files)
        assert "app.dict_build" in jobs.CHILD_MODULES and jobs.HANDLERS["dict"] is jobs.handle_dict
        assert jobs.DICT_TYPES == ("dict",) and "dict" not in jobs.MODEL_TYPES
        work = config.WORK_DIR / "abc"
        child_env = dict_build.child_env(work)
        assert child_env["TEMP"] == child_env["TMP"] == str(work / "tmp") and (work / "tmp").is_dir()
        assert child_env["PYTHONNOUSERSITE"] == "1"
        safepath.safe_rmtree(work, config.WORK_DIR)
        cmd = dict_build.child_cmd("ja", True, work)
        assert cmd[:6] == [sys.executable, "-s", "-m", "app.dict_build", "ja", "--parent-pid"], cmd
        assert cmd[-3:] == ["--work", str(work), "--force-download"] and jobs._module_arg(cmd) == "app.dict_build"
        env.patch(dict_build, "child_cmd", fk.dict_child_cmd(env.spec(build_delay=2.0)))
        before = _etilqs()
        jid, th = _start_job("ja")
        _wait(lambda: (jobs._running.get(jid) and jobs._running[jid].child) is not None, what="子程序")
        child = jobs._running[jid].child
        _wait(lambda: (jobs.download_progress.get(jid) or {}).get("stage") == "build", what="建置階段")
        assert psutil.Process(child.pid).nice() == psutil.BELOW_NORMAL_PRIORITY_CLASS
        assert "建立字典" in (db.get_job(jid)["stage"] or ""), db.get_job(jid)
        info = dict(jobs.download_progress.get(jid) or {})           # 記下開始建置的時間，剩餘時間照建置的快慢算
        assert info.get("build_t0") and info.get("build_p0") is not None and dict_build.eta_s("ja", info) is not None, info
        th.join(60)
        job = db.get_job(jid)
        assert job["status"] == "done" and job["result"]["entries"] == 42 and job["result"]["lang"] == "ja", job
        assert dict_ja.ready() and jid not in jobs.download_progress and not (config.WORK_DIR / jid).exists()
        assert _etilqs() <= before, _etilqs() - before              # VACUUM 的暫存檔沒寫到真正的 TEMP
        # 同一本字典不能重複排
        jid2 = db.add_job(None, "dict", {"lang": "ja"})
        dup = db.add_job(None, "dict", {"lang": "ja"})
        assert jobs.same_active_job(db.get_job(dup))["id"] == jid2
        assert jobs._DUPLICATE_TEXT["dict"] == ("這本字典正在建立", "這本字典已經在排隊建立")
        db.update_job(jid2, status="canceled")
        db.update_job(dup, status="canceled")
    run(body)


def test_eta_uses_download_speed_and_build_pace():
    """剩餘時間：下載看速度，建置看這台電腦實際的快慢。2026-09-17 乾淨安裝實測時舊算法把 GitHub 下載慢外推成整本字典的時間，
    報 17 分鐘、實際 3 分鐘；重開程式後接著建又報 36 秒、實際 2 分多鐘。"""
    en_total = dict_build.build_total_s("en")
    assert dict_build.build_ref_s("en", 0.15) == 0 and dict_build.build_ref_s("en", 1.0) == en_total
    assert abs(dict_build.build_ref_s("en", 0.25) - 8.0) < 1e-6
    assert abs(dict_build.build_fraction("en", 0.30) - 13.0 / en_total) < 1e-9
    for lang, points in dict_build.BUILD_PROFILE.items():               # 表格照進度排好、時間不會倒退
        assert points[0][0] == dict_build.DOWNLOAD_END[lang] and points[-1][0] == 1.0, lang
        assert all(a[0] < b[0] and a[1] <= b[1] for a, b in zip(points, points[1:])), lang
    # 下載中（實測：49 MB 下載到 33 MB，GitHub 0.28 MB/s，實際還要約 225 秒）
    info = {"stage": "download", "progress": 0.10, "done": 33 * MIB, "total": 49 * MIB, "speed": 0.28 * MIB}
    eta = dict_build.eta_s("en", info, now=100.0)
    assert eta == int(16 / 0.28 + en_total + 0.5) and 150 <= eta <= 300, eta
    assert dict_build.eta_s("en", {**info, "speed": 0.1 * MIB, "speed_avg": 0.28 * MIB}) == eta   # 用平均速度，不跟著跳
    assert dict_build.eta_s("en", {**info, "speed": 0}) is None                  # 還不知道速度
    assert dict_build.eta_s("en", {**info, "checking": True}) is None
    assert dict_build.eta_s("en", {**info, "retry": {"retry_in_s": 5}}) is None
    assert dict_build.eta_s("en", {"stage": "download", "done": 0, "total": 0, "speed": 0}) is None
    assert dict_build.eta_s("en", {**info, "done": 49 * MIB}) == int(en_total + 0.5)
    # 建置中：跟作者電腦一樣快時就是表上剩下的秒數；慢兩倍時大約兩倍
    t0 = 1000.0
    build = {"stage": "build", "progress": 0.47, "build_t0": t0, "build_p0": 0.15}
    ref = dict_build.build_ref_s("en", 0.47)
    assert abs(dict_build.eta_s("en", build, now=t0 + ref) - (en_total - ref)) <= 1
    slow = dict_build.eta_s("en", build, now=t0 + 2 * ref)
    assert 1.8 * (en_total - ref) <= slow <= 2.0 * (en_total - ref), slow
    # 下載花了很久也不影響：開始建置的時間是 build_t0，不是任務開始的時間
    assert dict_build.eta_s("en", {**build, "progress": 0.2235}, now=t0 + 4) < 200
    # 重開程式後從中間接著建（來源、解好的 csv 都在）：這次開始建的時間和進度，不會只剩幾十秒
    restart = {"stage": "build", "progress": 0.2394, "build_t0": t0, "build_p0": 0.15}
    assert 110 <= dict_build.eta_s("en", restart, now=t0 + 5) <= 165
    # 還沒記到開始時間（舊的進度）：照表
    assert dict_build.eta_s("en", {"stage": "finish", "progress": 0.99}) == int(en_total - dict_build.build_ref_s("en", 0.99) + 0.5)


def test_reporter_average_speed_counts_only_fetched_bytes():
    """估剩餘時間用的平均速度：只算這次真的收到的位元組（接著下載的 .part 不算），停下來的時間也算進去。"""
    out = io.StringIO()
    clock = {"t": 100.0}
    real = time.monotonic
    time.monotonic = lambda: clock["t"]
    try:
        r = dict_build.Reporter("en", out=out)
        r.total = 60 * MIB
        r.start_file("stardict.7z", 40 * MIB)                      # 已經下載到 40 MB 的 .part
        assert r.avg_speed() == 0 and r.fetched == 0
        r.add(MIB)
        clock["t"] = 101.0
        r.add(MIB)
        assert r.avg_speed() == r.speed()                             # 剛開始的 2 秒用最近的速度
        clock["t"] = 110.0                                            # 停了 9 秒才又收到
        r.add(MIB)
        assert abs(r.avg_speed() - 3 * MIB / 10) < 1 and r.done == 43 * MIB
        line = json.loads(out.getvalue().strip().splitlines()[-1])
        assert line["stage"] == "download" and line["avg_speed"] == round(3 * MIB / 10), line
    finally:
        time.monotonic = real


def test_dict_job_errors_retries_and_cancel():
    def body(env: Env):
        files = fk.ja_sources(40)
        env.set_sources("ja", files, write=False)
        srv = env.serve({k: v for k, v in files.items() if k != "accents.txt"})
        env.patch(dict_build, "child_cmd", fk.dict_child_cmd(env.spec()))
        # 找不到：不重試，錯誤代碼記進 result
        jid, th = _start_job("ja")
        th.join(60)
        job = db.get_job(jid)
        assert job["status"] == "failed" and "錯誤代碼 not_found" in job["error"], job
        assert job["result"]["error_code"] == "not_found" and not (config.WORK_DIR / jid).exists()
        # 來源壞掉（JMdict 條目太少）：自動重新下載一次，還是不對才失敗
        srv.add("/accents.txt", files["accents.txt"])
        srv.add("/JMdict_e.gz", fk.gz(fk.jmdict_xml(0)), etag='"x"')
        (env.src / "JMdict_e.gz").unlink(missing_ok=True)           # 上一個任務已經下載好的正常版本
        env.patch(dict_build, "JMDICT_MIN_ENTRIES", 5)
        env.patch(dict_build, "child_cmd", fk.dict_child_cmd(env.spec()))
        n = len(srv.log)
        jid, th = _start_job("ja")
        th.join(60)
        job = db.get_job(jid)
        assert job["status"] == "failed" and job["result"]["error_code"] == "corrupt_source", job
        assert len([x for x in srv.log[n:] if x[:2] == ("GET", "/JMdict_e.gz")]) == 2, srv.log[n:]
        # 取消：子程序結束，沒有半成品
        srv.add("/JMdict_e.gz", files["JMdict_e.gz"])
        env.patch(dict_build, "child_cmd", fk.dict_child_cmd(env.spec(build_delay=30)))
        jid, th = _start_job("ja")
        _wait(lambda: (jobs.download_progress.get(jid) or {}).get("stage") == "build", what="建置階段")
        child = jobs._running[jid].child
        jobs.cancel(jid)
        th.join(20)
        _wait(lambda: child.poll() is not None, 10, "子程序結束")
        assert db.get_job(jid)["status"] == "canceled"
        assert not (env.dict / "jmdict.db").exists() and not (env.dict / "jmdict.db.tmp.ok").exists()
        (env.dict / "jmdict.db.tmp").write_bytes(b"half")            # 被結束時留下的半成品
        dict_build.finish_pending()
        assert not (env.dict / "jmdict.db.tmp").exists()
    run(body)


def test_command_line_tools_still_work():
    def body(env: Env):
        env.set_sources("zh", {"dict-revised.json.xz": fk.moedict_xz(20)})
        out = io.StringIO()
        saved = sys.stdout
        sys.stdout = out
        try:
            dict_build.cli("zh", [])
        finally:
            sys.stdout = saved
        text = out.getvalue()
        assert "完成：" in text and "21 筆" in text, text
        assert (env.dict / "zh_dict.db").is_file() and not (env.src / "tmp").exists()
        for lang in ("ja", "en", "zh"):
            r = subprocess.run([sys.executable, "-s", str(ROOT / "tools" / f"build_dict_{lang}.py"), "--help"],
                               capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120,
                               env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
            assert r.returncode == 0 and "--download" in r.stdout, (lang, r.stdout, r.stderr)
    run(body)


# ---------- 手動：真的資料 ----------

def real_build():
    """用 data\\dict\\src 的真實來源（唯讀複製到暫存資料夾）建三本字典，記錄時間、大小、筆數。"""
    real_src = REAL_DICT / "src"
    assert real_src.is_dir(), "沒有 data\\dict\\src"
    dst = config.DICT_DIR / "src"
    shutil.rmtree(config.DICT_DIR, ignore_errors=True)
    dst.mkdir(parents=True)
    names = {s["name"] for lang in dict_build.LANGS for s in dict_build.SOURCES[lang]}
    for p in real_src.iterdir():
        if p.name in names:
            shutil.copyfile(p, dst / p.name)
    os.environ.pop("VS_DICT_SOURCE_BASE", None)
    tmp_dir = config.DICT_DIR / "src" / "tmp"
    tmp_dir.mkdir()
    os.environ["TEMP"] = os.environ["TMP"] = str(tmp_dir)
    import psutil
    for lang in ("ja", "en", "zh"):
        proc = psutil.Process()
        peak = {"rss": 0}
        stop = threading.Event()

        def watch():
            while not stop.is_set():
                peak["rss"] = max(peak["rss"], proc.memory_info().rss)
                time.sleep(0.05)
        threading.Thread(target=watch, daemon=True).start()
        t0 = time.time()
        result = dict_build.build(lang, out=io.StringIO())
        dict_build.swap(lang, kick=False)
        stop.set()
        path = config.DICT_DIR / dict_build.DB_NAMES[lang]
        print(json.dumps({"lang": lang, "seconds": round(time.time() - t0, 1), "entries": result["entries"],
                          "bytes": path.stat().st_size, "peak_rss_mib": peak["rss"] // MIB,
                          "py7zr": fk.py7zr_available()}), flush=True)
    shutil.rmtree(config.DICT_DIR, ignore_errors=True)


if __name__ == "__main__":
    started = time.time()
    RUN.mkdir(parents=True, exist_ok=True)
    try:
        if "--real" in sys.argv:
            real_build()
        else:
            for name, fn in list(globals().items()):
                if name.startswith("test_") and callable(fn):
                    t0 = time.time()
                    fn()
                    print(f"ok {name} ({time.time() - t0:.1f}s)")
            print(f"all passed in {time.time() - started:.1f}s")
    finally:
        dict_ja.close_all()
        dict_en.close_all()
        vocab_zh.close_all()
        shutil.rmtree(RUN, ignore_errors=True)
        if BASE.exists() and not any(BASE.iterdir()):
            BASE.rmdir()
