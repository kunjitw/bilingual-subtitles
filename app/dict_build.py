"""查字用的字典：下載原始資料、解壓、建 sqlite、換上新字典（打包計畫 P0-9）。

日文 jmdict.db、英文 en_zh.db 第一次打開時自動建（app/setup.py 排進佇列的 dict 任務）；
中文 zh_dict.db 在設定頁按按鈕才建。開發者也可以用命令列建：python -s tools/build_dict_ja.py（cli）。

佇列任務（jobs 的 dict 執行緒，handle_dict）：
  1. 子程序 python -s -m app.dict_build <ja|en|zh> [--parent-pid PID] [--work DIR] [--force-download]
     下載、解壓、建置、驗證，寫出 data\\dict\\<名稱>.db.tmp 和 <名稱>.db.tmp.ok。
     伺服器被關掉時子程序跟著結束（model_download.watch_parent）。TEMP 指到任務的工作資料夾，VACUUM 的暫存檔不寫到程式資料夾以外。
  2. 子程序成功結束後，在伺服器行程裡換檔（swap）：先關掉查字的連線，再 os.replace。換不掉（檔案被鎖住）時任務失敗，
     tmp 和 .ok 留著，下次啟動時 finish_pending 換上。
下載：來源檔放在 data\\dict\\src\\，用 model_download.fetch_file（.part 續傳、換鏡像、核對大小和 sha256）。
  JMdict 每天更新，沒辦法固定 sha256：用 ETag 決定 .part 能不能接著下載，下載完用 gzip 完整解一次來驗證。
  測試用：環境變數 VS_DICT_SOURCE_BASE=http://127.0.0.1:port，每個來源改從 <base>/<檔名> 下載。
  完整包（tools/build_release.py --full）：程式資料夾的 offline\\ 附了日文、英文字典的來源。下載前先找那裡，
  sha256 對就直接用（硬連結，不行才複製）；JMdict 的 sha256 寫在 offline\\files.json，還要 check_jmdict 驗得過。
  不對或沒有才照原本的網址下載。--force-download 要的是最新的 JMdict，不用 offline 的。
進度：stdout 每 0.5 秒一行 JSON {"stage": "download|extract|build|finish", "progress": 0.43, "done", "total", "speed"}。
失敗：stdout 最後一行 {"error": 代碼, "message": 說明}，stderr 最後一行 "ERROR <代碼> <說明>"（跟 app/model_download.py 一樣）。

出處（授權和標示方式見 THIRD_PARTY_NOTICES.md）：
  JMdict_e.gz        EDRDG，CC BY-SA 4.0
  jlpt_n1~5.csv      stephenmk/yomitan-jlpt-vocab，CC BY-SA 4.0（原始資料 Jonathan Waller，CC BY）
  accents.txt        mifunetoshiro/kanjium，CC BY-SA 4.0（要標示作者）
  stardict.7z        skywind3000/ECDICT，MIT
  cefrj-vocabulary-profile-1.5.csv、octanove-vocabulary-profile-c1c2-1.0.csv   openlanguageprofiles/olp-en-cefrj
  dict-revised.json.xz  g0v/moedict-data（教育部《重編國語辭典修訂本》，CC BY-ND 3.0 TW）
  去活用規則 tools/yomitan/yomitan_ja_transforms.json（Yomitan，GPL-3.0-or-later）會一起複製到 data\\dict。
"""
import argparse
import csv
import gzip
import hashlib
import json
import logging
import lzma
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
import traceback
import urllib.request
import xml.etree.ElementTree as ET
import zlib
from pathlib import Path

from . import config, safepath
from . import model_download as md
from .config import ROOT
from .model_download import DownloadError

log = logging.getLogger("dict_build")

MIB = 1024 * 1024
DICT_DIR = config.DICT_DIR          # 測試會換掉；函式裡都在呼叫當下讀
LANGS = ("ja", "en", "zh")
DB_NAMES = {"ja": "jmdict.db", "en": "en_zh.db", "zh": "zh_dict.db"}
COUNT_TABLE = {"ja": "entry", "en": "dict", "zh": "dict"}
LABELS = {"ja": "日文字典", "en": "英文字典", "zh": "中文辭典"}
SOURCE_LABELS = {"ja": "JMdict", "en": "ECDICT", "zh": "萌典"}
RULES_NAME = "yomitan_ja_transforms.json"
RULES_SRC = ROOT / "tools" / "yomitan" / RULES_NAME
SCHEMA_VERSION = "1"
OFFLINE_DIR = ROOT / "offline"      # 完整包附的下載檔（一般的 zip 沒有）；測試會換掉，函式裡都在呼叫當下讀
OFFLINE_LIST = "files.json"

# 建好後檢查的筆數下限（2026-09-17 實測：日文 218,785、英文 2,714,667、中文 161,134）
MIN_ENTRIES = {"ja": 150_000, "en": 2_000_000, "zh": 100_000}
JMDICT_MIN_ENTRIES = 150_000        # 下載的 JMdict_e.gz 至少要有這麼多 <entry>
JMDICT_MAX_BYTES = 256 * MIB        # 解壓後的上限（實測 63 MB）
JMDICT_APPROX_BYTES = 10_600_000    # 還沒問過伺服器時估大小用（2026-09-17 是 10,574,053）
ECDICT_MAX_BYTES = 400 * MIB        # stardict.csv 解壓後的上限（實測 232,668,349）
ECDICT_ROWS = 3_402_564             # stardict.csv 的列數，算進度用
# 建置時磁碟最多要多少（來源、解壓的 csv、db.tmp、VACUUM 暫存）
NEED_BYTES = {"ja": 200 * MIB, "en": 750 * MIB, "zh": 60 * MIB}
# 還沒開始建（排隊中、還不知道下載速度）時估的秒數（下載以外），抓寬一點，慢的電腦也不會差太多；設定頁也顯示這個
EST_S = {"ja": 20, "en": 300, "zh": 25}
# 作者電腦上（執行環境的 Python，2026-09-17 乾淨安裝實測）建置時整本字典的進度 → 從下載完算起大約過了幾秒。
# 英文：解壓 3 秒、第一輪 10 秒、第二輪（OpenCC）約 145 秒、VACUUM 和檢查 6 秒；日文約 9 秒；中文約 5 秒。
# 建置中的剩餘時間照這張表，再乘上這台電腦實際的快慢（eta_s）；總進度也用它把建置時間折成進度（app/setup.py）。
BUILD_PROFILE = {
    "ja": ((0.10, 0.0), (0.55, 4.0), (0.85, 6.0), (0.90, 7.0), (1.0, 9.0)),
    "en": ((0.15, 0.0), (0.20, 3.0), (0.30, 13.0), (0.92, 158.0), (1.0, 164.0)),
    "zh": ((0.15, 0.0), (0.55, 2.5), (0.90, 4.0), (1.0, 5.0)),
}
ETA_PRIOR_S = 10.0                  # 剛開始建置時快慢還看不準，先偏向作者電腦的速度
SEVEN_ZIP = Path(r"C:\Program Files\7-Zip\7z.exe")   # 開發模式沒裝 py7zr 時才用
SWAP_WAITS = (0.5, 1.0, 1.5, 3.0)
DOWNLOAD_END = {"ja": 0.10, "en": 0.15, "zh": 0.15}  # 下載佔整體進度的比例
STAGE_TEXT = {"download": "下載", "extract": "解壓縮", "build": "建立字典", "finish": "檢查字典"}

NO_7Z_TEXT = "缺少 py7zr，請重新執行 start.bat"
LOCKED_TEXT = "{label}正在使用中，換不掉。重新打開程式後會自動換上。"

# ---------- 來源（固定 commit 的網址，2026-09-17 下載核對過大小和 sha256） ----------

RAW = "https://raw.githubusercontent.com"
_K = "mifunetoshiro/kanjium/8a0cdaa16d64a281a2048de2eee2ec5e3a440fa6"
_J = "stephenmk/yomitan-jlpt-vocab/b062d4e38c4bdd0950ae1d4ec55f04b176182e03"
_E = "skywind3000/ECDICT/bc015ed2e24a7abef49fc6dbbb7fe32c1dadaf8b"
_O = "openlanguageprofiles/olp-en-cefrj/d4e45b75b38f27b30dfc5c44d8c571aec7e7092f"
_M = "g0v/moedict-data/a6dc997417507eb510fc29822bc514de2c92728c"
SOURCES = {
    "ja": [
        # 每天更新：沒有固定大小和 sha256。https 為主（ftp.edrdg.org 的憑證不符，不能用 https），http 備用
        {"name": "JMdict_e.gz", "daily": True, "bytes": None, "sha256": None,
         "urls": ["https://www.edrdg.org/pub/Nihongo/JMdict_e.gz", "http://ftp.edrdg.org/pub/Nihongo/JMdict_e.gz"]},
        {"name": "accents.txt", "urls": [f"{RAW}/{_K}/data/source_files/raw/accents.txt"], "bytes": 3226405,
         "sha256": "8bd0dd127dab32ceec94cb03ab1ba6b68858ea73421dfa1731af2f373deb4f20"},
        {"name": "jlpt_n1.csv", "urls": [f"{RAW}/{_J}/original_data/n1.csv"], "bytes": 199103,
         "sha256": "7a58f0584e9ec2b0299ccb9b109f3bd03e08b90d129a714307c0a1cb72f07b32"},
        {"name": "jlpt_n2.csv", "urls": [f"{RAW}/{_J}/original_data/n2.csv"], "bytes": 95356,
         "sha256": "42a4413326e857d701dc9659b2711637ab612304a339fe0d09f4f0f042d3a214"},
        {"name": "jlpt_n3.csv", "urls": [f"{RAW}/{_J}/original_data/n3.csv"], "bytes": 88623,
         "sha256": "bd4d68c59cfee861351e1bdf9d99818e434392201eabbed4d3657225fe53a3ac"},
        {"name": "jlpt_n4.csv", "urls": [f"{RAW}/{_J}/original_data/n4.csv"], "bytes": 24772,
         "sha256": "a14dcc7fdc02259b22331a486c6ff66df74c16472c8aeb5a0ebe7e5fa8ee8eb4"},
        {"name": "jlpt_n5.csv", "urls": [f"{RAW}/{_J}/original_data/n5.csv"], "bytes": 24010,
         "sha256": "07dc6f197b51cc076c65c3c9f23218d10c2b67f76545f5c1c4d9e3750495533a"},
    ],
    "en": [
        {"name": "stardict.7z", "urls": [f"{RAW}/{_E}/stardict.7z"], "bytes": 51577449,
         "sha256": "f370a0ecb58ada758d9dfe739db1667fd4ed87ed3055a4a7cb6c7054ecdf83d6"},
        {"name": "cefrj-vocabulary-profile-1.5.csv", "urls": [f"{RAW}/{_O}/cefrj-vocabulary-profile-1.5.csv"],
         "bytes": 233214, "sha256": "b0dd3c635f1c9a4fdf1490c7e5b7c48e8bbe55b652ad0c9860a95f98e10ae498"},
        {"name": "octanove-vocabulary-profile-c1c2-1.0.csv",
         "urls": [f"{RAW}/{_O}/octanove-vocabulary-profile-c1c2-1.0.csv"],
         "bytes": 46462, "sha256": "18c33a407f2f89f7b8de9671c6d45fe3ea0bce45e7d2d7dcaab48d73e0f7b380"},
    ],
    "zh": [
        {"name": "dict-revised.json.xz", "urls": [f"{RAW}/{_M}/dict-revised.json.xz"], "bytes": 14739512,
         "sha256": "5cc4ec0efd7e549621edf9b46d261989230c1729d9dc2c8e2056f4b21c8a93da"},
    ],
}


def sources(lang: str) -> list[dict]:
    """這本字典要下載的來源（複本）。設了 VS_DICT_SOURCE_BASE 時網址換成 <base>/<檔名>。"""
    base = (os.environ.get("VS_DICT_SOURCE_BASE") or "").strip().rstrip("/")
    out = []
    for s in SOURCES[lang]:
        s = dict(s)
        s["urls"] = [f"{base}/{s['name']}"] if base else list(s["urls"])
        out.append(s)
    return out


def source_bytes(lang: str) -> int:
    """要下載的來源大小（JMdict 用估計值）。"""
    return sum(s["bytes"] or JMDICT_APPROX_BYTES for s in SOURCES[lang])


def db_path(lang: str, dict_dir=None) -> Path:
    return Path(dict_dir or DICT_DIR) / DB_NAMES[lang]


def ready(lang: str, dict_dir=None) -> bool:
    """字典已經建好、可以查（日文還要有去活用規則，跟 dict_ja.ready 一樣）。"""
    d = Path(dict_dir or DICT_DIR)
    if lang == "ja":
        return (d / DB_NAMES["ja"]).is_file() and (d / RULES_NAME).is_file()
    return (d / DB_NAMES[lang]).is_file()


def _corrupt(message: str) -> DownloadError:
    return DownloadError("corrupt_source", message)


def _unlink(path: Path, root: Path) -> bool:
    return safepath.safe_unlink(path, root) if path.exists() else True


def _rmtree(path: Path, root: Path) -> bool:
    return safepath.safe_rmtree(path, root) if path.exists() else True


# ---------- 進度 ----------

class Reporter(md.Progress):
    """整個字典建置的進度。下載階段沿用 model_download.Progress 算位元組和速度，之後照 set() 給的比例。
    text=True（命令列）時印給人看的文字，不印 JSON。"""

    def __init__(self, lang: str, out=None, text: bool = False, interval: float = 0.5):
        super().__init__(0, out=out, interval=interval)
        self.lang = lang
        self.stage = "download"
        self.value = 0.0
        self.text = text
        self._said = None
        self.fetched = 0                 # 這次真的從網路收到的位元組（不算已經有的檔案、接著下載的 .part）
        self.fetch_t0 = None

    def add(self, n: int):
        if self.fetch_t0 is None:
            self.fetch_t0 = time.monotonic()
        self.fetched += n
        super().add(n)

    def avg_speed(self) -> float:
        """開始收到資料以來的平均下載速度。GitHub 有時候忽快忽慢、停十幾秒，最近 5 秒的速度拿來估剩餘時間會跳來跳去；
        2026-09-17 實測三次安裝，用平均速度估的誤差最小。剛開始的 2 秒還是用最近的速度。"""
        if self.fetch_t0 is None:
            return 0.0
        elapsed = time.monotonic() - self.fetch_t0
        return self.fetched / elapsed if elapsed >= 2 else self.speed()

    @property
    def fraction(self) -> float:
        if self.stage == "download":
            end = DOWNLOAD_END[self.lang]
            return end * (min(1.0, self.done / self.total) if self.total else 0.0)
        return self.value

    def set(self, stage: str, value: float, force: bool = False):
        self.stage, self.value = stage, max(0.0, min(1.0, value))
        self.emit(force=force)

    def emit(self, force: bool = False):
        now = time.monotonic()
        self._samples.append((now, self.done))
        while len(self._samples) > 2 and now - self._samples[0][0] > 5:
            self._samples.popleft()
        if self.text:
            part = (min(1.0, self.done / self.total) if self.total else 0.0) if self.stage == "download" else self.fraction
            key = (self.stage, int(part * 10))
            if key != self._said:
                self._said = key
                extra = (f" {self.done / MIB:.0f} / {self.total / MIB:.0f} MB"
                         if self.stage == "download" and self.total else "")
                self._print(f"{STAGE_TEXT.get(self.stage, self.stage)}{extra}（{self.fraction * 100:.0f}%）")
            return
        if not force and now - self._last < self.interval:
            return
        self._last = now
        line = {"stage": self.stage, "progress": round(self.fraction, 4), "done": self.done, "total": self.total,
                "speed": round(self.speed()), "avg_speed": round(self.avg_speed())}
        if self.checking:
            line["checking"] = True
        self._print(json.dumps(line, ensure_ascii=False))

    def _print(self, text: str):
        try:
            print(text, file=self.out, flush=True)
        except (OSError, ValueError):
            pass


def build_ref_s(lang: str, progress: float) -> float:
    """整本字典的進度到 progress 時，作者電腦上從下載完算起大約過了幾秒（BUILD_PROFILE 線性內插）。"""
    points = BUILD_PROFILE[lang]
    if progress <= points[0][0]:
        return 0.0
    for (p0, s0), (p1, s1) in zip(points, points[1:]):
        if progress <= p1:
            return s0 + (s1 - s0) * (progress - p0) / (p1 - p0)
    return points[-1][1]


def build_total_s(lang: str) -> float:
    """作者電腦上建置（下載以外）大約要幾秒。"""
    return BUILD_PROFILE[lang][-1][1]


def build_fraction(lang: str, progress: float) -> float:
    """建置做了幾成（照時間算，不是照進度數字）：0 是剛下載完，1 是建好了。"""
    return min(1.0, build_ref_s(lang, progress) / build_total_s(lang))


def eta_s(lang: str, info: dict, now: float | None = None) -> int | None:
    """建字典的任務大約還要幾秒。info 是 jobs.download_progress 裡這個任務的進度（_run_child 寫的）。
    下載階段：剩下的位元組 ÷ 現在的速度，加上作者電腦上建置要的時間；還不知道速度（剛開始、檢查已下載的部分、
    等重試）時回 None。建置階段：作者電腦上剩下的時間 × 這台電腦的快慢（開始建置到現在實際花的時間 ÷ 作者電腦上
    同一段要的時間）。下載慢不會算進建置的時間，重開程式後從中間接著建也不會低估。"""
    now = time.monotonic() if now is None else now
    stage = info.get("stage") or "download"
    if stage == "download":
        total, done = int(info.get("total") or 0), int(info.get("done") or 0)
        speed = float(info.get("speed_avg") or info.get("speed") or 0)
        if info.get("checking") or info.get("retry") or not total:
            return None
        left = max(0, total - done)
        if left and speed <= 0:
            return None
        return int((left / speed if left else 0) + build_total_s(lang) + 0.5)
    p = float(info.get("progress") or 0)
    ref_left = max(0.0, build_total_s(lang) - build_ref_s(lang, p))
    t0 = info.get("build_t0")
    if t0 is None:
        return int(ref_left + 0.5)
    ref_done = max(0.0, build_ref_s(lang, p) - build_ref_s(lang, float(info.get("build_p0") or 0)))
    factor = (max(0.0, now - t0) + ETA_PRIOR_S) / (ref_done + ETA_PRIOR_S)
    return int(ref_left * min(8.0, max(0.25, factor)) + 0.5)


# ---------- 下載 ----------

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * MIB), b""):
            h.update(chunk)
    return h.hexdigest()


class _KeepHead(urllib.request.HTTPRedirectHandler):
    """urllib 跟著轉址時會把 HEAD 改成 GET，這裡保持 HEAD。"""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None:
            new.method = req.get_method()
        return new


def remote_head(urls: list[str]) -> dict:
    """問伺服器檔案現在的大小和 ETag（跟著轉址）：{"url", "size", "etag"}。連不上丟 DownloadError。"""
    opener = urllib.request.build_opener(_KeepHead)
    last = None
    for i, url in enumerate(urls):
        req = urllib.request.Request(url, method="HEAD",
                                     headers={"User-Agent": md.USER_AGENT, "Accept-Encoding": "identity"})
        try:
            with opener.open(req, timeout=md.TIMEOUT) as r:
                length = r.headers.get("Content-Length")
                etag = (r.headers.get("ETag") or "").strip()
                return {"url": url, "size": int(length) if length and length.isdigit() else None, "etag": etag or None}
        except Exception as e:  # noqa: BLE001
            last = md.classify(e, url)
            if last.code in ("not_found", "gated", "network", "timeout", "ssl") and i + 1 < len(urls):
                continue
            raise last from e
    raise last or DownloadError("not_found", source="下載來源")


def check_jmdict(path: Path):
    """下載的 JMdict_e.gz 完整解一次：gzip 核對得過、解壓後不超過上限、根元素是 JMdict、條目夠多。不合格丟 corrupt_source。"""
    total = entries = 0
    root = False
    tail_entry = tail_root = b""
    try:
        with gzip.open(path, "rb") as f:
            while True:
                chunk = f.read(4 * MIB)
                if not chunk:
                    break
                total += len(chunk)
                if total > JMDICT_MAX_BYTES:
                    raise _corrupt("JMdict 解壓後太大，不像字典檔，已經刪掉。按「重試」會重新下載。")
                buf = tail_entry + chunk
                entries += buf.count(b"<entry>")
                tail_entry = buf[-6:]            # 比 <entry> 短一個字：跨兩段的也數得到，也不會重複數
                if not root:
                    root = b"<JMdict>" in tail_root + chunk
                    tail_root = (tail_root + chunk)[-7:]
    except DownloadError:
        raise
    except (OSError, EOFError, zlib.error) as e:
        raise _corrupt("JMdict 下載的檔案壞掉了，已經刪掉。按「重試」會重新下載。") from e
    if not root or entries < JMDICT_MIN_ENTRIES:
        raise _corrupt("JMdict 的內容不對，已經刪掉。按「重試」會重新下載。")
    return entries


def _fetch_daily(s: dict, dst: Path, head: dict, reporter: Reporter, root: Path):
    """每天更新的檔案（JMdict）：ETag 或大小跟上次下載到一半時不同，.part 就不能接著用。"""
    part = dst.with_name(dst.name + md.PART_SUFFIX)
    meta_path = dst.with_name(dst.name + md.PART_SUFFIX + ".json")
    saved = config._read_json(meta_path)
    if part.exists() and (not head["etag"] or saved.get("etag") != head["etag"] or saved.get("size") != head["size"]):
        _unlink(part, root)
    meta_path.write_text(json.dumps({"etag": head["etag"], "size": head["size"], "url": head["url"]}), encoding="utf-8")
    urls = [head["url"]] + [u for u in s["urls"] if u != head["url"]]
    md.fetch_file(urls, dst, head["size"], None, reporter)
    try:
        check_jmdict(dst)
    except DownloadError:
        _unlink(dst, root)
        raise
    finally:
        _unlink(meta_path, root)


def offline_source(s: dict) -> Path | None:
    """完整包 offline\\ 裡這個來源的檔案，大小和 sha256 都對才回傳。固定版本的照 SOURCES，JMdict 照 offline\\files.json。"""
    d = Path(OFFLINE_DIR)
    p = d / s["name"]
    if not p.is_file():
        return None
    sha, size = s.get("sha256"), s.get("bytes")
    if not sha:
        listed = config._read_json(d / OFFLINE_LIST).get("files")
        entry = next((f for f in listed if isinstance(f, dict) and f.get("name") == s["name"]), None) \
            if isinstance(listed, list) else None
        sha, size = (entry or {}).get("sha256"), (entry or {}).get("size")
    if not isinstance(sha, str) or len(sha) != 64:
        return None
    try:
        if (size and p.stat().st_size != size) or sha256_file(p) != sha.lower():
            log.warning("offline %s does not match its sha256, downloading instead", s["name"])
            return None
    except OSError:
        return None
    return p


def _place_offline(src: Path, dst: Path, root: Path) -> bool:
    """offline 的檔案放到來源資料夾：硬連結（不佔空間），不行才複製。失敗回 False，照常下載。"""
    tmp = dst.with_name(dst.name + ".offline")
    try:
        _unlink(tmp, root)
        try:
            os.link(src, tmp)
        except OSError:
            shutil.copyfile(src, tmp)
        os.replace(tmp, dst)
    except OSError as e:
        log.warning("could not use offline %s: %s", src.name, e)
        _unlink(tmp, root)
        return False
    log.info("using %s from the offline folder", src.name)
    return True


def download_sources(lang: str, src_dir: Path, reporter: Reporter, force: bool = False, root: Path | None = None):
    """把這本字典的來源下載到 src_dir。已經有、而且大小和 sha256 都對的跳過；JMdict 已經有、驗得過就沿用，force 才重抓。
    完整包 offline\\ 裡有、sha256 對的直接用（force 時 JMdict 不用 offline 的）。"""
    root = Path(root or DICT_DIR)
    src_dir.mkdir(parents=True, exist_ok=True)
    plan, total = [], 0
    for s in sources(lang):
        dst = src_dir / s["name"]
        if s.get("daily"):
            if dst.is_file() and force:
                _unlink(dst, root)
            if not dst.is_file() and not force:
                off = offline_source(s)
                if off is not None:
                    _place_offline(off, dst, root)
            if dst.is_file():
                try:
                    check_jmdict(dst)
                    plan.append((s, dst, "have", None))
                    total += dst.stat().st_size
                    continue
                except DownloadError:
                    _unlink(dst, root)
            head = remote_head(s["urls"])
            plan.append((s, dst, "daily", head))
            total += head["size"] or JMDICT_APPROX_BYTES
            continue
        if dst.is_file():
            if dst.stat().st_size == s["bytes"] and sha256_file(dst) == s["sha256"]:
                plan.append((s, dst, "have", None))
                total += s["bytes"]
                continue
            _unlink(dst, root)          # 舊版或壞掉的檔案：重新下載固定版本
        off = offline_source(s)
        if off is not None and _place_offline(off, dst, root):
            plan.append((s, dst, "have", None))
            total += s["bytes"]
            continue
        plan.append((s, dst, "fetch", None))
        total += s["bytes"]
    reporter.total = total
    for s, dst, how, head in plan:
        if how == "have":
            reporter.start_file(dst.name, dst.stat().st_size)
            reporter.finish_file()
        elif how == "daily":
            _fetch_daily(s, dst, head, reporter, root)
        else:
            md.fetch_file(s["urls"], dst, s["bytes"], s["sha256"], reporter)
    reporter.emit(force=True)


# ---------- 解壓 stardict.7z ----------

def extract_ecdict(src_dir: Path, reporter: Reporter, root: Path | None = None) -> Path:
    """stardict.7z 解出 stardict.csv：先解到 _ecdict.tmp，成功才改名成 _ecdict。壓縮檔壞掉丟 corrupt_source 並刪掉它。"""
    root = Path(root or DICT_DIR)
    archive = src_dir / "stardict.7z"
    done_dir, tmp_dir = src_dir / "_ecdict", src_dir / "_ecdict.tmp"
    csv_path = done_dir / "stardict.csv"
    reporter.set("extract", DOWNLOAD_END["en"], force=True)
    if csv_path.is_file():
        return csv_path
    _rmtree(tmp_dir, root)
    _rmtree(done_dir, root)
    try:
        import py7zr
    except ImportError:
        py7zr = None
    try:
        if py7zr is not None:
            _extract_py7zr(py7zr, archive, tmp_dir)
        elif SEVEN_ZIP.is_file():
            _extract_7zexe(archive, tmp_dir)
        else:
            raise DownloadError("no_7z", NO_7Z_TEXT)
    except DownloadError as e:
        _rmtree(tmp_dir, root)
        if e.code == "corrupt_source":
            _unlink(archive, root)
        raise
    os.replace(tmp_dir, done_dir)
    reporter.set("extract", 0.20, force=True)
    return csv_path


def _extract_py7zr(py7zr, archive: Path, tmp_dir: Path):
    from py7zr import exceptions as zx
    try:
        with py7zr.SevenZipFile(archive, "r") as z:
            infos = z.list()
            if [i.filename for i in infos] != ["stardict.csv"] or not infos[0].is_file:
                raise _corrupt("ECDICT 壓縮檔裡的檔案不對，已經刪掉。按「重試」會重新下載。")
            size = int(infos[0].uncompressed or 0)
            if size > ECDICT_MAX_BYTES:
                raise _corrupt("ECDICT 解壓後太大，不像字典檔，已經刪掉。按「重試」會重新下載。")
            tmp_dir.mkdir(parents=True, exist_ok=True)
            z.extract(path=tmp_dir, targets=["stardict.csv"])
    except DownloadError:
        raise
    except (zx.ArchiveError, lzma.LZMAError, zlib.error, EOFError, ValueError, KeyError) as e:
        raise _corrupt("ECDICT 的壓縮檔壞掉了，已經刪掉。按「重試」會重新下載。") from e
    out = tmp_dir / "stardict.csv"
    if not out.is_file() or out.stat().st_size != size:
        raise _corrupt("ECDICT 的壓縮檔壞掉了，已經刪掉。按「重試」會重新下載。")


def _extract_7zexe(archive: Path, tmp_dir: Path):
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    listing = subprocess.run([str(SEVEN_ZIP), "l", "-slt", "-ba", str(archive)], capture_output=True, text=True,
                             encoding="utf-8", errors="replace", creationflags=flags)
    paths = [l[7:].strip() for l in listing.stdout.splitlines() if l.startswith("Path = ")]
    sizes = [int(l[7:].strip() or 0) for l in listing.stdout.splitlines() if l.startswith("Size = ")]
    if listing.returncode != 0 or paths != ["stardict.csv"]:
        raise _corrupt("ECDICT 的壓縮檔壞掉了，已經刪掉。按「重試」會重新下載。")
    if sum(sizes) > ECDICT_MAX_BYTES:
        raise _corrupt("ECDICT 解壓後太大，不像字典檔，已經刪掉。按「重試」會重新下載。")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    r = subprocess.run([str(SEVEN_ZIP), "x", "-y", "-bd", f"-o{tmp_dir}", str(archive)], stdout=subprocess.DEVNULL,
                       stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace", creationflags=flags)
    files = sorted(p.relative_to(tmp_dir).as_posix() for p in tmp_dir.rglob("*") if p.is_file())
    if r.returncode != 0 or files != ["stardict.csv"]:
        raise _corrupt("ECDICT 的壓縮檔壞掉了，已經刪掉。按「重試」會重新下載。")


# ---------- 建置：日文（原本的 tools/build_dict_ja.py） ----------

JA_SCHEMA = """
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE entry (id INTEGER PRIMARY KEY, data TEXT NOT NULL);
CREATE TABLE form (text TEXT NOT NULL, entry_id INTEGER NOT NULL, kana INTEGER NOT NULL, pri TEXT, rank INTEGER NOT NULL);
CREATE TABLE jlpt (entry_id INTEGER PRIMARY KEY, level INTEGER NOT NULL);
CREATE TABLE accent (term TEXT NOT NULL, reading TEXT NOT NULL, pattern TEXT NOT NULL, PRIMARY KEY (term, reading)) WITHOUT ROWID;
"""


def rank(pris: list[str]) -> int:
    """越小越常用：nfXX 取 XX（1~48），ichi1/news1/spec1/gai1 為 50，*2 為 80，沒標記 999。"""
    r = 999
    for p in pris:
        if p.startswith("nf"):
            r = min(r, int(p[2:]))
        elif p in ("ichi1", "news1", "spec1", "gai1"):
            r = min(r, 50)
        elif p in ("ichi2", "news2", "spec2", "gai2"):
            r = min(r, 80)
    return r


def parse_jmdict(path: Path):
    with gzip.open(path, "rb") as f:
        raw = f.read().decode("utf-8")
    date = (re.search(r"JMdict created: (\d{4}-\d{2}-\d{2})", raw) or [None, ""])[1]
    # JMdict 用 DTD entity 表示詞性（&v1; &n;），ElementTree 不會展開，先換成代碼本身
    ents = dict(re.findall(r'<!ENTITY ([\w\-]+) "([^"]*)">', raw))
    raw = re.sub(r"<!DOCTYPE JMdict \[.*?\]>", "", raw, flags=re.S)
    raw = re.sub(r"&([\w\-]+);", lambda m: m.group(1) if m.group(1) in ents else m.group(0), raw)
    root = ET.fromstring(raw)
    del raw
    entries, forms = [], []
    for e in root.iter("entry"):
        eid = int(e.findtext("ent_seq"))
        k = [{"t": x.findtext("keb"), "pri": [p.text for p in x.findall("ke_pri")],
              "inf": [p.text for p in x.findall("ke_inf")]} for x in e.findall("k_ele")]
        r = []
        for x in e.findall("r_ele"):
            item = {"t": x.findtext("reb"), "pri": [p.text for p in x.findall("re_pri")],
                    "inf": [p.text for p in x.findall("re_inf")]}
            restr = [p.text for p in x.findall("re_restr")]
            if restr:
                item["restr"] = restr
            if x.find("re_nokanji") is not None:
                item["nokanji"] = 1
            r.append(item)
        senses, pos = [], []
        for s in e.findall("sense"):
            p = [x.text for x in s.findall("pos")] or pos   # 沒寫 pos 的 sense 沿用上一個
            pos = p
            sense = {"pos": p, "misc": [x.text for x in s.findall("misc")],
                     "field": [x.text for x in s.findall("field")],
                     "gloss": [g.text for g in s.findall("gloss") if g.text]}
            for tag, key in (("dial", "dial"), ("s_inf", "inf"), ("stagk", "stagk"), ("stagr", "stagr")):
                vals = [x.text for x in s.findall(tag) if x.text]
                if vals:
                    sense[key] = vals
            senses.append(sense)
        entries.append((eid, json.dumps({"k": k, "r": r, "s": senses}, ensure_ascii=False, separators=(",", ":"))))
        for x in k:
            forms.append((x["t"], eid, 0, ",".join(x["pri"]), rank(x["pri"])))
        for x in r:
            forms.append((x["t"], eid, 1, ",".join(x["pri"]), rank(x["pri"])))
    return date, entries, forms


def _connect_new(path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(path)
    con.execute("PRAGMA journal_mode=OFF")
    con.execute("PRAGMA synchronous=OFF")
    return con


def _insert_batches(con, sql: str, rows: list, reporter: Reporter, lo: float, hi: float, batch: int = 20000):
    for i in range(0, len(rows), batch):
        con.executemany(sql, rows[i:i + batch])
        reporter.set("build", lo + (hi - lo) * min(1.0, (i + batch) / max(1, len(rows))))


def build_ja(src: Path, tmp: Path, reporter: Reporter, dict_dir: Path) -> int:
    reporter.set("build", 0.10, force=True)
    date, entries, forms = parse_jmdict(src / "JMdict_e.gz")
    reporter.set("build", 0.55, force=True)
    con = _connect_new(tmp)
    try:
        con.executescript(JA_SCHEMA)
        _insert_batches(con, "INSERT INTO entry VALUES (?, ?)", entries, reporter, 0.55, 0.68)
        _insert_batches(con, "INSERT INTO form VALUES (?, ?, ?, ?, ?)", forms, reporter, 0.68, 0.80)
        con.execute("CREATE INDEX ix_form_text ON form(text)")
        con.execute("CREATE INDEX ix_form_entry ON form(entry_id)")
        reporter.set("build", 0.85)

        # JLPT：同一個條目出現在好幾級時取較簡單的（數字大的）
        ids = {eid for eid, _ in entries}
        jlpt, missing = {}, 0
        for n in range(1, 6):
            with open(src / f"jlpt_n{n}.csv", encoding="utf-8", newline="") as f:
                for row in csv.DictReader(f):
                    try:
                        eid = int(row["jmdict_seq"])
                    except (KeyError, ValueError, TypeError):
                        continue
                    if eid not in ids:
                        missing += 1
                        continue
                    jlpt[eid] = max(jlpt.get(eid, 0), n)
        con.executemany("INSERT INTO jlpt VALUES (?, ?)", jlpt.items())

        # 重音：詞\t讀音（假名詞留空）\t重音（可能是 0,2）
        acc = {}
        with open(src / "accents.txt", encoding="utf-8") as f:
            for line in f:
                p = line.rstrip("\n").split("\t")
                if len(p) >= 3 and p[0] and p[2]:
                    acc[(p[0], p[1] or p[0])] = p[2]
        con.executemany("INSERT INTO accent VALUES (?, ?, ?)", [(t, r, v) for (t, r), v in acc.items()])
        log.info("JMdict %s: %d entries, %d forms, JLPT %d (%d unmatched), accents %d",
                 date, len(entries), len(forms), len(jlpt), missing, len(acc))
        con.executemany("INSERT INTO meta VALUES (?, ?)", [
            ("schema_version", SCHEMA_VERSION), ("jmdict_date", date), ("built_at", time.strftime("%Y-%m-%d %H:%M:%S")),
            ("entries", str(len(entries))),
        ])
        con.commit()
        reporter.set("build", 0.90, force=True)
        con.execute("VACUUM")
    finally:
        con.close()
    shutil.copyfile(RULES_SRC, dict_dir / f"{RULES_NAME}.tmp")
    return len(entries)


# ---------- 建置：英文（原本的 tools/build_dict_en.py） ----------

EN_SCHEMA = """
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE dict (word TEXT NOT NULL COLLATE NOCASE, phonetic TEXT, pos TEXT, zh TEXT, tag TEXT,
                   bnc INTEGER, frq INTEGER, collins INTEGER, oxford INTEGER, exchange TEXT, lemma TEXT, multi INTEGER);
CREATE TABLE cefr (word TEXT PRIMARY KEY COLLATE NOCASE, level TEXT NOT NULL) WITHOUT ROWID;
"""
LEVELS = ["A1", "A2", "B1", "B2", "C1", "C2"]
# OpenCC s2twp 偶爾會轉出香港或大陸用語，發現一個加一個
TW_FIX = {"鐳射": "雷射", "的士": "計程車"}
WORD = re.compile(r"^[A-Za-z][A-Za-z'\-\.]*$")


def keep(word: str, common_heads: set) -> bool:
    if len(word) > 40:
        return False
    parts = word.split(" ")
    if len(parts) == 1:
        return bool(WORD.match(word))
    # 片語：2~4 個純字母詞，而且第一個詞是常見詞（有詞頻、考試標籤或柯林斯星級）
    return 2 <= len(parts) <= 4 and all(WORD.match(p) for p in parts) and parts[0].lower() in common_heads


def lemma_from_exchange(ex: str) -> str:
    for part in ex.split("/"):
        if part.startswith("0:"):
            return part[2:]
    return ""


def build_en(csv_path: Path, src: Path, tmp: Path, reporter: Reporter) -> tuple[int, int]:
    from opencc import OpenCC

    csv.field_size_limit(1 << 30)
    cc = OpenCC("s2twp")
    reporter.set("build", 0.20, force=True)
    common = set()
    with csv_path.open(encoding="utf-8", newline="") as f:
        for i, r in enumerate(csv.DictReader(f), 1):
            if " " not in r["word"] and (r["frq"] not in ("", "0") or r["bnc"] not in ("", "0")
                                         or r["tag"] or r["collins"] not in ("", "0")):
                common.add(r["word"].lower())
            if i % 20000 == 0:
                reporter.set("build", 0.20 + 0.10 * min(1.0, i / ECDICT_ROWS))
    reporter.set("build", 0.30, force=True)

    con = _connect_new(tmp)
    try:
        con.executescript(EN_SCHEMA)
        rows, n = [], 0
        with csv_path.open(encoding="utf-8", newline="") as f:
            for i, r in enumerate(csv.DictReader(f), 1):
                if i % 20000 == 0:
                    reporter.set("build", 0.30 + 0.62 * min(1.0, i / ECDICT_ROWS))
                if not r["translation"].strip() or not keep(r["word"], common):
                    continue
                # CSV 裡的換行是字面的反斜線加 n
                zh = cc.convert(r["translation"].replace("\\n", "\n").strip())
                for a, b in TW_FIX.items():
                    zh = zh.replace(a, b)
                rows.append((r["word"], r["phonetic"], r["pos"], zh, r["tag"], int(r["bnc"] or 0), int(r["frq"] or 0),
                             int(r["collins"] or 0), int(r["oxford"] or 0), r["exchange"],
                             lemma_from_exchange(r["exchange"]), 1 if " " in r["word"] else 0))
                if len(rows) >= 50000:
                    con.executemany("INSERT INTO dict VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", rows)
                    n += len(rows)
                    rows.clear()
        con.executemany("INSERT INTO dict VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", rows)
        n += len(rows)
        con.execute("CREATE INDEX ix_word ON dict(word COLLATE NOCASE)")

        cefr = {}
        for name in ("cefrj-vocabulary-profile-1.5.csv", "octanove-vocabulary-profile-c1c2-1.0.csv"):
            with open(src / name, encoding="utf-8", newline="") as f:
                for r in csv.DictReader(f):
                    lv = (r.get("CEFR") or "").strip()
                    for hw in (r.get("headword") or "").split("/"):
                        hw = hw.strip().lower()
                        if hw and lv in LEVELS and (hw not in cefr or LEVELS.index(lv) < LEVELS.index(cefr[hw])):
                            cefr[hw] = lv
        con.executemany("INSERT INTO cefr VALUES (?, ?)", cefr.items())
        con.executemany("INSERT INTO meta VALUES (?, ?)", [
            ("schema_version", SCHEMA_VERSION), ("built_at", time.strftime("%Y-%m-%d %H:%M:%S")),
            ("entries", str(n)), ("cefr", str(len(cefr))),
        ])
        con.commit()
        reporter.set("build", 0.92, force=True)
        con.execute("VACUUM")
    finally:
        con.close()
    log.info("ECDICT: %d rows, CEFR %d", n, len(cefr))
    return n, len(cefr)


# ---------- 建置：中文（原本的 tools/build_dict_zh.py） ----------

def build_zh(src: Path, tmp: Path, reporter: Reporter) -> int:
    reporter.set("build", 0.15, force=True)
    with lzma.open(src / "dict-revised.json.xz") as f:
        data = json.load(f)
    reporter.set("build", 0.55, force=True)
    rows = {}
    for e in data:
        word = e.get("title", "")
        if not word or "{[" in word:          # 缺字用圖檔表示的詞條跳過
            continue
        hets = e.get("heteronyms") or []
        defs = []
        for h in hets:
            for d in h.get("definitions") or []:
                text = re.sub(r"\s+", "", d.get("def", ""))
                if text:
                    pos = d.get("type")
                    defs.append(f"［{pos}］{text}" if pos else text)
        if not defs or word in rows:
            continue
        rows[word] = (word, (hets[0].get("bopomofo") if hets else "") or "", json.dumps(defs, ensure_ascii=False))
    del data
    con = _connect_new(tmp)
    try:
        con.executescript("""
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE dict (word TEXT PRIMARY KEY, bopomofo TEXT, defs TEXT NOT NULL) WITHOUT ROWID;
        """)
        _insert_batches(con, "INSERT INTO dict VALUES (?, ?, ?)", list(rows.values()), reporter, 0.55, 0.90)
        con.executemany("INSERT INTO meta VALUES (?, ?)", [("built_at", time.strftime("%Y-%m-%d %H:%M:%S")),
                                                           ("entries", str(len(rows))),
                                                           ("source", "教育部《重編國語辭典修訂本》（萌典 JSON）")])
        con.commit()
        reporter.set("build", 0.90, force=True)
        con.execute("VACUUM")
    finally:
        con.close()
    return len(rows)


# ---------- 整個流程 ----------

def verify(lang: str, path: Path) -> int:
    """建好的 db：quick_check 要是 ok、筆數要夠。回傳筆數。"""
    try:
        con = sqlite3.connect(path)
        try:
            check = con.execute("PRAGMA quick_check").fetchone()[0]
            count = con.execute(f"SELECT COUNT(*) FROM {COUNT_TABLE[lang]}").fetchone()[0]
        finally:
            con.close()
    except sqlite3.Error as e:
        raise DownloadError("unknown", f"{LABELS[lang]}建好後打不開：{e}") from e
    if check != "ok":
        raise DownloadError("unknown", f"{LABELS[lang]}建好後檢查沒過：{check}")
    if count < MIN_ENTRIES[lang]:
        raise DownloadError("unknown", f"{LABELS[lang]}建好後只有 {count:,} 筆，比預期少很多，沒有換上")
    return count


def build(lang: str, force: bool = False, out=None, text: bool = False, dict_dir=None) -> dict:
    """下載來源、建出 <名稱>.db.tmp、驗證、寫 .ok。不換檔（伺服器行程裡的 swap 或命令列的 cli 才換）。"""
    d = Path(dict_dir or DICT_DIR)
    src = d / "src"
    d.mkdir(parents=True, exist_ok=True)
    reporter = Reporter(lang, out=out, text=text)
    download_sources(lang, src, reporter, force, root=d)
    name = DB_NAMES[lang]
    tmp, ok = d / f"{name}.tmp", d / f"{name}.tmp.ok"
    rules_tmp = d / f"{RULES_NAME}.tmp"
    for p in (ok, tmp) + ((rules_tmp,) if lang == "ja" else ()):
        if not _unlink(p, d):
            raise DownloadError("file_locked")
    try:
        if lang == "ja":
            entries = build_ja(src, tmp, reporter, d)
        elif lang == "en":
            csv_path = extract_ecdict(src, reporter, root=d)
            entries, _ = build_en(csv_path, src, tmp, reporter)
        else:
            entries = build_zh(src, tmp, reporter)
        reporter.set("finish", 0.99, force=True)
        entries = verify(lang, tmp)
        info = {"built_at": time.strftime("%Y-%m-%d %H:%M:%S"), "entries": entries, "schema_version": SCHEMA_VERSION}
        ok_tmp = ok.with_name(ok.name + ".part")
        ok_tmp.write_text(json.dumps(info), encoding="utf-8")
        os.replace(ok_tmp, ok)
    except BaseException:
        for p in (tmp, ok) + ((rules_tmp,) if lang == "ja" else ()):
            _unlink(p, d)
        raise
    if lang == "en":
        _rmtree(src / "_ecdict", d)          # 解出來的 csv（222 MB）用完就刪，來源的 7z 留著，之後可以離線重建
    reporter.set("finish", 1.0, force=True)
    return {"lang": lang, "entries": entries, "size_bytes": tmp.stat().st_size}


# ---------- 換上新字典 ----------

_swap_lock = threading.Lock()


class DictLocked(RuntimeError):
    """字典檔被開著，換不掉。建好的 tmp 和 .ok 留著，下次啟動時換上。"""


def _close_connections(lang: str):
    """Windows 上 sqlite 連線開著時不能 os.replace：先關掉查字的連線、清掉快取。"""
    try:
        if lang == "ja":
            from . import dict_ja
            dict_ja.close_all()
            rules = getattr(dict_ja, "_rules", None)
            if rules is not None and hasattr(rules, "cache_clear"):
                rules.cache_clear()          # 去活用規則也一起換，下次用時重新讀
        elif lang == "en":
            from . import dict_en
            dict_en.close_all()
        else:
            from . import vocab_zh
            vocab_zh.close_all()
    except Exception:  # noqa: BLE001
        log.warning("closing %s dictionary connections failed", lang, exc_info=True)


def pending_swap(lang: str, dict_dir=None) -> bool:
    d = Path(dict_dir or DICT_DIR)
    name = DB_NAMES[lang]
    return (d / f"{name}.tmp").is_file() and (d / f"{name}.tmp.ok").is_file()


def swap(lang: str, dict_dir=None, waits=None, kick: bool = True) -> dict:
    """把建好的 <名稱>.db.tmp 換成正式檔（日文先換去活用規則）。一直換不掉丟 DictLocked，檔案都留著。
    waits：換不掉時每次等幾秒再試（沒給用 SWAP_WAITS）。"""
    waits = SWAP_WAITS if waits is None else waits
    d = Path(dict_dir or DICT_DIR)
    name = DB_NAMES[lang]
    tmp, ok = d / f"{name}.tmp", d / f"{name}.tmp.ok"
    if not (tmp.is_file() and ok.is_file()):
        raise RuntimeError(f"找不到建好的{LABELS[lang]}，請重新建立")
    info = config._read_json(ok)
    moves = ([(d / f"{RULES_NAME}.tmp", d / RULES_NAME)] if lang == "ja" else []) + [(tmp, d / name)]
    with _swap_lock:
        for src, dst in moves:
            if not src.exists():
                continue
            for wait in (*waits, None):
                _close_connections(lang)
                try:
                    os.replace(src, dst)
                    break
                except PermissionError as e:
                    if wait is None:
                        log.warning("dictionary %s is locked, keeping %s for the next start", dst, src)
                        raise DictLocked(LOCKED_TEXT.format(label=LABELS[lang])) from e
                    time.sleep(wait)
        _unlink(ok, d)
        _close_connections(lang)
    log.info("dictionary %s replaced (%s entries)", d / name, info.get("entries"))
    if kick:
        vocab = sys.modules.get(f"{__package__}.vocab")
        if vocab is not None:
            try:
                vocab.kick()                 # 字幕的查字索引照新字典重算
            except Exception:  # noqa: BLE001
                log.warning("vocab kick failed", exc_info=True)
    try:
        size = (d / name).stat().st_size
    except OSError:
        size = None
    return {"lang": lang, "entries": info.get("entries"), "size_bytes": size}


def finish_pending(dict_dir=None) -> list[str]:
    """伺服器啟動時（還沒有人開字典之前）呼叫：上次建好但換不掉的字典換上去；沒建完的 tmp、解壓到一半的資料夾刪掉。"""
    d = Path(dict_dir or DICT_DIR)
    if not d.is_dir():
        return []
    done = []
    for lang, name in DB_NAMES.items():
        tmp, ok = d / f"{name}.tmp", d / f"{name}.tmp.ok"
        if tmp.is_file() and ok.is_file():
            try:
                swap(lang, d, waits=(0.5,), kick=False)
                done.append(lang)
            except Exception:  # noqa: BLE001
                log.warning("could not finish the %s dictionary", lang, exc_info=True)
            continue
        for p in (tmp, ok):
            _unlink(p, d)
        if lang == "ja":
            _unlink(d / f"{RULES_NAME}.tmp", d)
    _rmtree(d / "src" / "_ecdict.tmp", d)
    _rmtree(d / "src" / "tmp", d)
    return done


# ---------- 佇列任務（伺服器行程） ----------

def child_cmd(lang: str, force: bool = False, work: Path | None = None) -> list[str]:
    cmd = [sys.executable, "-s", "-m", "app.dict_build", lang, "--parent-pid", str(os.getpid())]
    if work:
        cmd += ["--work", str(work)]
    if force:
        cmd.append("--force-download")
    return cmd


def child_env(work: Path) -> dict:
    """跟模型下載一樣的環境變數，TEMP、TMP 指到任務的工作資料夾（VACUUM 的暫存檔寫在這裡）。"""
    from . import jobs
    env = jobs.model_download_env()
    tmp = Path(work) / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    env["TEMP"] = env["TMP"] = str(tmp)
    return env


def disk_problem(lang: str, dict_dir=None) -> tuple[int, int] | None:
    """建這本字典的空間不夠時回 (需要的位元組（含餘裕）, 剩下的位元組)，夠回 None。"""
    from .models import DISK_MARGIN, disk_free
    need = NEED_BYTES[lang] + DISK_MARGIN
    free = disk_free(Path(dict_dir or DICT_DIR))
    return None if free >= need else (need, free)


def stage_text(info: dict) -> str:
    stage = info.get("stage")
    if stage == "download":
        if info.get("checking"):
            return "檢查已下載的部分"
        return f"下載 {round((info.get('done') or 0) / MIB)} / {round((info.get('total') or 0) / MIB)} MB"
    if stage == "extract":
        return "解壓縮"
    if stage == "finish":
        return "檢查字典"
    return f"建立字典 {int((info.get('progress') or 0) * 100)}%"


def _run_child(ctx, lang: str, force: bool):
    from . import jobs
    info = jobs.download_progress.setdefault(ctx.id, {"done": 0, "total": 0, "speed": 0})
    info.pop("retry", None)
    err_path = ctx.workdir / "dict_build.err.log"
    failure = None
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0)
    with open(err_path, "w", encoding="utf-8", errors="replace") as err:
        ctx.child = subprocess.Popen(child_cmd(lang, force, ctx.workdir), cwd=str(ROOT), env=child_env(ctx.workdir),
                                     stdout=subprocess.PIPE, stderr=err, text=True, encoding="utf-8",
                                     errors="replace", creationflags=flags)
        try:
            ctx.check()
            for line in ctx.child.stdout:
                line = line.strip()
                if not line.startswith("{"):
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if msg.get("error"):
                    failure = msg
                    continue
                if "stage" not in msg:
                    continue
                if msg["stage"] == "download":
                    info.pop("build_t0", None)       # 重試時從下載重新開始
                    info.pop("build_p0", None)
                    # 估剩餘時間用平均速度（Reporter.avg_speed），畫面上顯示的速度還是最近 5 秒的
                    info["speed_avg"] = float(msg.get("avg_speed") or 0)
                elif "build_t0" not in info:
                    # 開始建置的時間和進度：剩餘時間照建置實際的快慢算（eta_s），不把下載的時間算進去
                    info.update(build_t0=time.monotonic(), build_p0=float(msg.get("progress") or 0))
                info.update(stage=msg["stage"], progress=float(msg.get("progress") or 0),
                            done=int(msg.get("done") or 0), total=int(msg.get("total") or 0),
                            speed=float(msg.get("speed") or 0) if msg["stage"] == "download" else 0,
                            checking=bool(msg.get("checking")))
                info.pop("retry", None)
                ctx.progress(min(0.99, info["progress"]), stage_text(info))
            code = ctx.child.wait()
        except BaseException:
            if ctx.child.poll() is None:
                ctx.child.kill()
            ctx.child.wait()
            raise
    ctx.child = None
    ctx.check()
    if code == 0:
        return
    if failure:
        raise DownloadError(str(failure["error"]), str(failure.get("message") or "") or None, failure.get("retry_after"))
    lines = err_path.read_text(encoding="utf-8", errors="replace").strip().splitlines()
    last = next((l for l in reversed(lines) if l.strip()), f"exit {code}")
    if last.startswith("ERROR "):
        parts = last.split(" ", 2)
        raise DownloadError(parts[1], parts[2] if len(parts) > 2 else None)
    raise DownloadError("unknown", detail=last[:300])


def retry_waits() -> dict:
    """任務裡自動重試的等待秒數：跟模型下載一樣，來源檔壞掉（corrupt_source）重新下載一次。"""
    from . import jobs
    return {**jobs.DOWNLOAD_RETRY_WAITS, "corrupt_source": (0,)}


def handle_dict(ctx):
    """dict 任務：子程序建好字典，這裡換上。網路這類錯誤自動重試，失敗時任務的錯誤訊息帶錯誤代碼，result 也記下。"""
    from . import db, jobs
    lang = (ctx.params or {}).get("lang")
    if lang not in LANGS:
        raise RuntimeError("沒有這種字典")
    force = bool(ctx.params.get("force_download"))
    ctx.progress(0, f"準備{LABELS[lang]}")
    ctx.workdir.mkdir(parents=True, exist_ok=True)
    waits_by_code = retry_waits()
    attempt = 0
    try:
        while True:
            try:
                short = disk_problem(lang)
                if short:
                    raise DownloadError("disk_full", need=md.fmt_size(short[0]), free=md.fmt_size(short[1]))
                _run_child(ctx, lang, force)
                break
            except DownloadError as e:
                err = e
            force = False                    # 重試時不再重抓這次已經下載好的來源
            waits = waits_by_code.get(err.code, ())
            if attempt < len(waits):
                wait = err.retry_after if (err.code == "rate_limit" and err.retry_after) else waits[attempt]
                attempt += 1
                log.warning("dictionary %s failed (%s), retry %d/%d in %ss", lang, err.code, attempt, len(waits), wait)
                jobs._wait_before_retry(ctx, int(wait), attempt, len(waits), err)
                continue
            info = jobs.download_progress.get(ctx.id) or {}
            db.update_job(ctx.id, result={"lang": lang, "error_code": err.code, "stage": info.get("stage")})
            raise RuntimeError(f"{err}（錯誤代碼 {err.code}）")
        ctx.check()
        info = jobs.download_progress.setdefault(ctx.id, {})
        info.update(stage="finish", progress=0.995, speed=0)
        ctx.progress(0.995, "換上新字典")
        try:
            result = swap(lang)
        except DictLocked as e:
            db.update_job(ctx.id, result={"lang": lang, "error_code": "dict_locked", "stage": "finish"})
            raise RuntimeError(f"{e}（錯誤代碼 dict_locked）") from e
    finally:
        safepath.safe_rmtree(ctx.workdir, jobs.WORK_DIR)
        jobs.download_progress.pop(ctx.id, None)
    ctx.result = result


# ---------- 命令列 ----------

def cli(lang: str, argv=None):
    """tools/build_dict_*.py 用：在自己的行程裡建好、直接換上。伺服器開著、字典換不掉時留著新檔，下次啟動時換上。"""
    ap = argparse.ArgumentParser(description=f"建立{LABELS[lang]}（{DB_NAMES[lang]}）")
    ap.add_argument("--download", "--force-download", dest="force", action="store_true",
                    help="重新下載每天更新的來源（JMdict）")
    args = ap.parse_args(argv)
    d = Path(DICT_DIR)
    t0 = time.time()
    tmp_dir = d / "src" / "tmp"                 # VACUUM 的暫存檔放這裡，用完刪掉
    tmp_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TEMP"] = os.environ["TMP"] = str(tmp_dir)
    try:
        try:
            result = build(lang, force=args.force, text=True)
        except DownloadError as e:
            print(f"失敗：{e}（錯誤代碼 {e.code}）")
            sys.exit(1)
        try:
            swap(lang, waits=(0.5,), kick=False)
        except DictLocked:
            print(f"字典檔正在被伺服器使用，換不掉。新檔留在 {d / (DB_NAMES[lang] + '.tmp')}，請關閉伺服器後再跑一次。")
            sys.exit(1)
        path = d / DB_NAMES[lang]
        print(f"完成：{path}（{result['entries']:,} 筆，{path.stat().st_size / 1e6:.0f} MB，{time.time() - t0:.0f} 秒）")
    finally:
        _rmtree(tmp_dir, d)


# ---------- 子程序 ----------

def _classify(e: BaseException) -> DownloadError:
    if isinstance(e, DownloadError):
        return e
    if isinstance(e, sqlite3.OperationalError) and "full" in str(e).lower():
        return DownloadError("disk_full", "磁碟空間不夠，字典寫不進去。清出空間後按「重試」。")
    if isinstance(e, ImportError) and "py7zr" in str(e):
        return DownloadError("no_7z", NO_7Z_TEXT)
    return md.classify(e)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("lang", choices=LANGS)
    ap.add_argument("--parent-pid", type=int)
    ap.add_argument("--work")
    ap.add_argument("--force-download", action="store_true")
    args = ap.parse_args(argv)
    md.watch_parent(args.parent_pid)
    try:
        result = build(args.lang, force=args.force_download)
        print(json.dumps({"stage": "finish", "progress": 1.0, "result": result}, ensure_ascii=False), flush=True)
    except BaseException as e:  # noqa: BLE001
        if isinstance(e, SystemExit):
            raise
        err = _classify(e)
        if err.code == "unknown":
            traceback.print_exc(file=sys.stderr)
        msg = " ".join(str(err).split())[:500]
        print(json.dumps({"error": err.code, "message": msg, "retry_after": err.retry_after}, ensure_ascii=False),
              flush=True)
        print(f"ERROR {err.code} {msg}", file=sys.stderr, flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
