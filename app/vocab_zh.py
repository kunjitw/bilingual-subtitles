"""中文單字統計：CKIP Transformers bert-base 斷詞＋詞性，只在 CPU 上跑，結果存進單字索引（快取）。

模型放在 models/ckip/（bert-base-chinese-ws、bert-base-chinese-pos，約 776 MB，GPL-3.0），
在設定頁的模型管理下載、刪除（型錄 config.MODEL_CATALOG["ckip"]，路徑要跟這裡的 WS_DIR、POS_DIR 一致）。
斷詞在子程序裡跑，並設 CUDA_VISIBLE_DEVICES 為空，保證不碰顯卡；跑完子程序結束，記憶體也還回去。
中文解釋用教育部《重編國語辭典修訂本》（萌典 JSON，CC BY-ND 3.0 TW），建法見 tools/build_dict_zh.py。

只統計中文影片的辨識字幕（tracks.kind = 'asr'、lang = 'zh-TW'），日文、英文影片的中文翻譯不算。
執行子程序：python -s -m app.vocab_zh <輸入 JSON> <輸出 JSON>
"""
import functools
import importlib.util
import json
import logging
import os
import re
import sqlite3
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

from .config import DICT_DIR, MODELS_DIR, ROOT

log = logging.getLogger("vocab_zh")

MODEL_DIR = MODELS_DIR / "ckip"
WS_DIR = MODEL_DIR / "bert-base-chinese-ws"
POS_DIR = MODEL_DIR / "bert-base-chinese-pos"
REF_DB = DICT_DIR / "zh_dict.db"
VERSION = "ckip-bert-base:1"    # 過濾規則改了就加一

# CKIP 詞性：這些是功能詞、量詞、代名詞、數詞，不算單字
DROP_POS = {"DE", "SHI", "T", "Di", "Caa", "Cab", "Cba", "Cbb", "P", "Nep", "Neqa", "Neqb", "Nes", "Neu", "Nf",
            "Ng", "Nh", "I", "DM", "Dfa", "Dk", "FW"}
HAN = re.compile(r"^[㐀-鿿豈-﫿]{2,}$")                      # 只收兩字以上的純漢字詞
NUM_PREFIX = re.compile(r"^[一二三四五六七八九十百千萬兩幾第]")    # 五樓、三點、兩百塊
COMPLEMENTS = ("起來", "出來", "下來", "上去", "進去", "過來", "回來", "完", "到", "掉", "好", "給")
STOP = {"欸", "齁", "蠻", "想說", "等一下", "就是", "然後", "所以", "因為", "可是", "但是", "其實", "真的", "這樣",
        "那樣", "什麼", "怎麼", "為什麼", "一下", "一些", "一個", "有點", "的話", "之後", "之前", "時候", "東西"}

_run_lock = threading.Lock()
_state = {"running": None, "error": None}


def models_ready() -> bool:
    return ((WS_DIR / "pytorch_model.bin").exists() and (POS_DIR / "pytorch_model.bin").exists()
            and importlib.util.find_spec("ckip_transformers") is not None)


def ready() -> bool:
    return models_ready()


def status() -> dict:
    return {"models": models_ready(), "dict": REF_DB.exists(), "running": _state["running"], "error": _state["error"]}


# ---------- 辭典 ----------

_local = threading.local()
_all_cons: list = []
_all_lock = threading.Lock()
_gen = [0]


def _ref():
    c = getattr(_local, "con", None)
    if c is None or getattr(_local, "gen", -1) != _gen[0]:
        c = sqlite3.connect(REF_DB.as_uri() + "?mode=ro", uri=True, check_same_thread=False)
        with _all_lock:
            _all_cons.append(c)
        _local.con, _local.gen = c, _gen[0]
    return c


def close_all():
    """換辭典檔前呼叫（app/dict_build.py 的 swap）：Windows 上檔案開著就不能 os.replace。
    所有執行緒的連線都關掉，快取清掉，下次查詢時重新開。"""
    with _all_lock:
        _gen[0] += 1
        for c in _all_cons:
            try:
                c.close()
            except Exception:  # noqa: BLE001
                pass
        _all_cons.clear()
    _entry.cache_clear()


@functools.lru_cache(maxsize=50_000)
def _entry(word: str):
    if not REF_DB.exists():
        return None
    row = _ref().execute("SELECT bopomofo, defs FROM dict WHERE word = ?", (word,)).fetchone()
    return (row[0], json.loads(row[1])) if row else None


def in_dict(word: str) -> bool:
    return _entry(word) is not None


def gloss(word: str) -> dict:
    e = _entry(word)
    if not e or not e[1]:
        return {"g": "", "gs": "", "lv": ""}
    first = re.sub(r"\s+", "", e[1][0])
    first = re.split(r"(?<=。)|如：|《", first)[0] or first     # 列表只放第一句，出處和例句留給完整釋義
    return {"g": first[:60] + ("…" if len(first) > 60 else ""), "gs": "moe", "lv": ""}


def detail(word: str) -> dict:
    e = _entry(word)
    if not e:
        return {"defs": [], "bopomofo": ""}
    return {"defs": e[1][:12], "bopomofo": e[0] or ""}


# ---------- 斷詞與過濾 ----------

def base_form(word: str) -> str:
    """封起來→封、處理完→處理：去掉補語後的詞在辭典裡、原詞不在才還原。"""
    if in_dict(word):
        return word
    for c in COMPLEMENTS:
        if word.endswith(c) and len(word) > len(c):
            base = word[: -len(c)]
            if in_dict(base):
                return base
    return word


def filter_words(words: list[list[str]], tags: list[list[str]]):
    """回傳 (每句的 key 清單, meta)。"""
    meta, keys_per_cue = {}, []
    for ws, ps in zip(words, tags):
        keys = []
        for w, p in zip(ws, ps):
            if p in DROP_POS or p.endswith("CATEGORY") or not HAN.match(w) or NUM_PREFIX.match(w) or w in STOP:
                continue
            w = base_form(w)
            if len(w) < 2:
                continue
            key = f"zh:{w}"
            grp = "name" if p in ("Nb", "Nc") and not in_dict(w) else "word"
            m = meta.setdefault(key, {"lang": "zh", "display": w, "reading": None, "grp": grp, "entry_id": None,
                                      "pos": p, "phrase": 0})
            if grp == "word":
                m["grp"] = "word"
            keys.append(key)
        keys_per_cue.append(keys)
    return keys_per_cue, meta


def segment_texts(texts: list[str]) -> tuple[list, list]:
    """在子程序裡跑 CKIP（CPU），回傳 (words, tags)。"""
    with tempfile.TemporaryDirectory(prefix="vs-ckip-") as tmp:
        src, dst = Path(tmp) / "in.json", Path(tmp) / "out.json"
        src.write_text(json.dumps(texts, ensure_ascii=False), encoding="utf-8")
        env = {**os.environ, "PYTHONNOUSERSITE": "1", "PYTHONIOENCODING": "utf-8", "CUDA_VISIBLE_DEVICES": "",
               "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TRANSFORMERS_VERBOSITY": "error"}
        proc = subprocess.run([sys.executable, "-s", "-m", "app.vocab_zh", str(src), str(dst)], cwd=str(ROOT), env=env,
                              capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=3600,
                              creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if proc.returncode != 0 or not dst.exists():
            tail = (proc.stderr or "").strip().splitlines()
            raise RuntimeError("中文斷詞失敗：" + (tail[-1] if tail else f"exit {proc.returncode}"))
        data = json.loads(dst.read_text(encoding="utf-8"))
    return data["words"], data["tags"]


def index_track(t: dict):
    from . import cues as cue_mod
    from . import vocab
    cues = cue_mod.load_cues(t["id"])
    texts = [c.get("text", "") for c in cues]
    words, tags = segment_texts(texts)
    keys_per_cue, meta = filter_words(words, tags)
    vocab.index_track(t["id"], cues, "zh", meta, keys_per_cue)


def run_pending():
    """背景執行緒呼叫：把還沒斷詞、或字幕文字改過的中文字幕補算。"""
    if not ready():
        return
    from . import db, vocab
    with _run_lock:
        # 設定頁刪除模型時會拿著這把鎖刪檔，等到鎖時模型可能已經不在了
        if not ready():
            return
        for t in vocab.counted_tracks("zh"):
            if vocab.is_fresh(t) or not db.get_track(t["id"]):
                continue
            _state["running"] = t["id"]
            try:
                index_track(t)
                _state["error"] = None
            except Exception as e:  # noqa: BLE001
                _state["error"] = str(e)[:300]
                log.warning("zh segment failed for %s", t["id"], exc_info=True)
            finally:
                _state["running"] = None


def _child(src: str, dst: str):
    import torch
    torch.set_num_threads(max(2, (os.cpu_count() or 8) // 2))   # 留一半 CPU 給其他工作
    from ckip_transformers.nlp import CkipPosTagger, CkipWordSegmenter
    texts = json.loads(Path(src).read_text(encoding="utf-8"))
    ws = CkipWordSegmenter(model_name=str(WS_DIR), device=-1)
    pos = CkipPosTagger(model_name=str(POS_DIR), device=-1)
    idx = [i for i, t in enumerate(texts) if t.strip()]
    words_nz = ws([texts[i] for i in idx], batch_size=64, max_length=128, show_progress=False)
    tags_nz = pos(words_nz, batch_size=64, max_length=128, show_progress=False)
    words, tags = [[] for _ in texts], [[] for _ in texts]
    for i, w, p in zip(idx, words_nz, tags_nz):
        words[i], tags[i] = w, p
    Path(dst).write_text(json.dumps({"words": words, "tags": tags}, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    _child(sys.argv[1], sys.argv[2])
