"""自動安裝、字典建置測試共用的假東西：假的下載伺服器、假的字典來源檔、子程序的包裝腳本。

不連網：伺服器開在 127.0.0.1，可以停掉再從同一個 port 開起來（模擬斷網）。
"""
import gzip
import hashlib
import http.server
import io
import json
import lzma
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEVEN_ZIP = Path(r"C:\Program Files\7-Zip\7z.exe")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ---------- 假的下載伺服器 ----------

class FakeFiles:
    """files：路徑 → {"data", "etag", "range"(預設 True), "cut": [這次只送多少就斷線], "status": [先回這些狀態碼], "delay"}。
    log：(方法, 路徑, Range 標頭, 狀態碼, 送出的位元組)。stop() 之後 start() 會用同一個 port。"""

    def __init__(self):
        self.files: dict[str, dict] = {}
        self.log: list[tuple] = []
        self.port = 0
        self.httpd = None
        self.down = False
        self.start()

    def start(self):
        owner = self
        self.down = False

        class Handler(_Handler):
            server_owner = owner

        httpd = http.server.ThreadingHTTPServer(("127.0.0.1", self.port), Handler)
        httpd.daemon_threads = True
        self.port = httpd.server_address[1]
        self.httpd = httpd
        threading.Thread(target=httpd.serve_forever, daemon=True).start()

    def stop(self):
        self.down = True                  # 傳到一半的連線也中斷（像斷網）
        if self.httpd is not None:
            self.httpd.shutdown()
            self.httpd.server_close()
            self.httpd = None

    close = stop

    @property
    def base(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def url(self, path: str) -> str:
        return self.base + path

    def add(self, path: str, data: bytes, **opts) -> str:
        self.files[path] = {"data": data, **opts}
        return self.url(path)

    def ranges(self, path: str, method: str = "GET") -> list:
        return [r for m, p, r, _s, _n in self.log if p == path and m == method]


class _Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_owner: FakeFiles = None

    def log_message(self, *args):
        pass

    def _cfg(self):
        return self.server_owner.files.get(self.path.split("?")[0])

    def do_HEAD(self):
        cfg = self._cfg()
        self.server_owner.log.append(("HEAD", self.path, None, 404 if cfg is None else 200, 0))
        self.send_response(404 if cfg is None else 200)
        if cfg is not None:
            self.send_header("Content-Length", str(len(cfg["data"])))
            if cfg.get("etag"):
                self.send_header("ETag", cfg["etag"])
        else:
            self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        owner = self.server_owner
        cfg = self._cfg()
        rng = self.headers.get("Range")
        if cfg is None:
            return self._status(404, rng)
        statuses = cfg.get("status") or []
        if statuses:
            code = statuses.pop(0)
            if code != 200:
                return self._status(code, rng)
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
        if cfg.get("etag"):
            self.send_header("ETag", cfg["etag"])
        self.end_headers()
        cuts = cfg.get("cut") or []
        cut = cuts.pop(0) if cuts else None
        limit = len(body) if cut is None else min(cut, len(body))
        sent = 0
        try:
            while sent < limit and not owner.down:
                chunk = body[sent:min(limit, sent + 64 * 1024)]
                self.wfile.write(chunk)
                sent += len(chunk)
                if cfg.get("delay"):
                    time.sleep(cfg["delay"])
        except (ConnectionError, OSError):
            pass
        owner.log.append(("GET", self.path, rng, 206 if start else 200, sent))
        if cut is not None or sent < limit:
            self.close_connection = True

    def _status(self, code, rng):
        self.server_owner.log.append(("GET", self.path, rng, code, 0))
        self.send_response(code)
        self.send_header("Content-Length", "0")
        self.end_headers()


# ---------- 假的字典來源 ----------

JMDICT_HEAD = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE JMdict [
<!ELEMENT JMdict (entry*)>
<!ENTITY v1 "Ichidan verb">
<!ENTITY vt "transitive verb">
<!ENTITY n "noun (common) (futsuumeishi)">
<!ENTITY pn "pronoun">
]>
<!-- JMdict created: 2026-09-17 -->
<JMdict>
"""

KANA = "あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわ"


def jmdict_xml(n: int = 50) -> bytes:
    """JMdict_e 的樣子：兩個真的條目（食べる、私，跟 fake_jmdict 一樣）加上 n 個假的名詞。"""
    parts = [JMDICT_HEAD,
             "<entry><ent_seq>1358280</ent_seq><k_ele><keb>食べる</keb><ke_pri>ichi1</ke_pri><ke_pri>nf25</ke_pri></k_ele>"
             "<r_ele><reb>たべる</reb><re_pri>ichi1</re_pri></r_ele><sense><pos>&v1;</pos><pos>&vt;</pos>"
             "<gloss>to eat</gloss></sense></entry>\n",
             "<entry><ent_seq>1311110</ent_seq><k_ele><keb>私</keb><ke_pri>ichi1</ke_pri></k_ele>"
             "<r_ele><reb>わたし</reb></r_ele><sense><pos>&pn;</pos><gloss>I</gloss><gloss>me</gloss></sense></entry>\n"]
    for i in range(n):
        a, b, c = KANA[i % len(KANA)], KANA[(i // len(KANA)) % len(KANA)], KANA[(i // 2500) % len(KANA)]
        parts.append(f"<entry><ent_seq>{5000000 + i}</ent_seq><r_ele><reb>{c}{b}{a}ん</reb></r_ele>"
                     f"<sense><pos>&n;</pos><gloss>fake word {i}</gloss></sense></entry>\n")
    parts.append("</JMdict>\n")
    return "".join(parts).encode("utf-8")


def gz(data: bytes) -> bytes:
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0) as f:
        f.write(data)
    return buf.getvalue()


def ja_sources(n: int = 50) -> dict[str, bytes]:
    out = {"JMdict_e.gz": gz(jmdict_xml(n)),
           "accents.txt": "食べる\tたべる\t2\n私\tわたし\t0\n".encode("utf-8")}
    for level in range(1, 6):
        seq = 1358280 if level == 5 else 1311110 if level == 4 else 5000000 + level
        out[f"jlpt_n{level}.csv"] = f"jmdict_seq,kana,kanji,waller_definition\n{seq},x,y,z\n".encode("utf-8")
    return out


ECDICT_HEADER = "word,phonetic,definition,translation,pos,collins,oxford,tag,bnc,frq,exchange,detail,audio\n"


def stardict_csv(n: int = 300) -> bytes:
    rows = [ECDICT_HEADER,
            'give,giv,to give,"vt. 给予\\nvi. 让步",v,5,1,zk gk,100,120,d:given/p:gave/0:give,,\n',
            'give up,,,"放弃",,0,0,,0,0,,,\n',
            'laser,leɪzə,,"n. 镭射",n,3,1,cet4,5000,4000,,,\n']
    for i in range(n):
        letters = "".join(chr(97 + (i // 26 ** k) % 26) for k in range(3))      # 只能是英文字母（build_en 的 WORD）
        rows.append(f"fake{letters},wɜːd,,\"n. 单词 {i}\",n,0,0,,{i},{i},,,\n")
    return "".join(rows).encode("utf-8")


def en_extra_sources() -> dict[str, bytes]:
    return {"cefrj-vocabulary-profile-1.5.csv": "headword,pos,CEFR\ngive,verb,A1\nlaser,noun,B2\n".encode("utf-8"),
            "octanove-vocabulary-profile-c1c2-1.0.csv": "headword,pos,CEFR\nlaser,noun,C1\n".encode("utf-8")}


def moedict_xz(n: int = 30) -> bytes:
    data = [{"title": "辭典", "heteronyms": [{"bopomofo": "ㄘˊ ㄉㄧㄢˇ", "definitions": [{"type": "名", "def": "工具書。"}]}]}]
    for i in range(n):
        data.append({"title": f"詞{i}", "heteronyms": [{"bopomofo": "ㄘˊ", "definitions": [{"def": f"說明 {i}。"}]}]})
    return lzma.compress(json.dumps(data, ensure_ascii=False).encode("utf-8"))


def py7zr_available() -> bool:
    try:
        import py7zr  # noqa: F401
        return True
    except ImportError:
        return False


def make_7z(members: dict[str, bytes]) -> bytes:
    """做一個 7z：有 py7zr 用 py7zr，沒有用 7-Zip。都沒有丟 RuntimeError（呼叫的測試會跳過）。"""
    tmp = Path(tempfile.mkdtemp(prefix="vs-7z-"))
    try:
        out = tmp / "out.7z"
        if py7zr_available():
            import py7zr
            with py7zr.SevenZipFile(out, "w") as z:
                for name, data in members.items():
                    z.writestr(data, name)
        elif SEVEN_ZIP.is_file():
            src = tmp / "src"
            for name, data in members.items():
                p = src / name
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(data)
            names = list(members)
            subprocess.run([str(SEVEN_ZIP), "a", "-bd", "-y", str(out), *names], cwd=str(src), check=True,
                           stdout=subprocess.DEVNULL, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        else:
            raise RuntimeError("no 7z tool")
        return out.read_bytes()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def can_make_7z() -> bool:
    return py7zr_available() or SEVEN_ZIP.is_file()


# ---------- 子程序的包裝：把測試用的設定帶進子程序 ----------

DICT_WRAPPER = r"""
import json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
spec = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
from app import dict_build, safepath
for r in spec.get("test_roots", []):
    safepath.TEST_ROOTS.append(Path(r))
dict_build.MIN_ENTRIES.update(spec.get("min_entries") or {})
if spec.get("jmdict_min") is not None:
    dict_build.JMDICT_MIN_ENTRIES = spec["jmdict_min"]
if spec.get("ecdict_rows"):
    dict_build.ECDICT_ROWS = spec["ecdict_rows"]
if spec.get("seven_zip"):
    dict_build.SEVEN_ZIP = Path(spec["seven_zip"])
for lang, files in (spec.get("sources") or {}).items():
    for s in dict_build.SOURCES[lang]:
        if s["name"] in files:
            s.update(files[s["name"]])
if spec.get("no_py7zr"):
    sys.modules["py7zr"] = None
if spec.get("print_temp"):
    import tempfile
    print(json.dumps({"temp": tempfile.gettempdir()}), flush=True)
if spec.get("build_delay"):
    import time
    real = dict_build.parse_jmdict
    def slow(path):
        time.sleep(spec["build_delay"])
        return real(path)
    dict_build.parse_jmdict = slow
dict_build.main(sys.argv[3:])
"""

MODEL_WRAPPER = r"""
import json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
spec = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
from app import config, model_download
for mid, variants in (spec.get("variants") or {}).items():
    for name, v in variants.items():
        config.MODEL_CATALOG[mid]["variants"][name].update(v)
lists = spec.get("lists") or {}
def fake_list(repo, allow):
    from huggingface_hub.utils import filter_repo_objects
    sha, files = lists[repo]
    keep = set(filter_repo_objects([f["path"] for f in files], allow_patterns=allow))
    return sha, [f for f in files if f["path"] in keep]
model_download.list_repo = fake_list
model_download.QUICK_WAITS = ()
model_download.main(sys.argv[3:])
"""


def write_spec(folder: Path, name: str, spec: dict) -> Path:
    path = Path(folder) / name
    path.write_text(json.dumps(spec, ensure_ascii=False), encoding="utf-8")
    return path


def dict_child_cmd(spec_path: Path):
    """換掉 dict_build.child_cmd：子程序先套用測試設定再跑 dict_build.main。"""
    from app import dict_build
    real = getattr(dict_build.child_cmd, "real", dict_build.child_cmd)      # 已經換過一次時用原本的

    def cmd(lang, force=False, work=None):
        return [sys.executable, "-s", "-c", DICT_WRAPPER, str(ROOT), str(spec_path)] + real(lang, force, work)[4:]
    cmd.real = real
    return cmd


def model_child_cmd(spec_path: Path):
    from app import jobs
    real = getattr(jobs.model_download_cmd, "real", jobs.model_download_cmd)

    def cmd(mid, variant=None, work=None):
        return [sys.executable, "-s", "-c", MODEL_WRAPPER, str(ROOT), str(spec_path)] + real(mid, variant, work)[4:]
    cmd.real = real
    return cmd


def env_python() -> str:
    return sys.executable


def rel_listing(folder: Path) -> list[tuple]:
    """資料夾裡每個檔案的 (相對路徑, 大小, 修改時間)，比對前後有沒有被改過。"""
    out = []
    if not Path(folder).is_dir():
        return out
    for root, _dirs, files in os.walk(folder):
        for name in files:
            p = Path(root) / name
            try:
                st = p.stat()
            except OSError:
                continue
            out.append((str(p.relative_to(folder)), st.st_size, st.st_mtime_ns))
    return sorted(out)
