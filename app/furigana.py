"""日文假名標註：pyopenjtalk-plus（OpenJTalk 前端）+ tsqyomi 同形異音模型 + 使用者辭典，全部在 CPU 上跑。

cue["ruby"] = [[起, 迄, 平假名, 旗標], ...]，起迄是 cue["text"] 的字元位置，文字本身不改。
旗標：""  一般
      "h" tsqyomi 管的同形異音詞（這個詞有別的讀法，前端可列出候選）
      "g" 送假名對不上，整個詞一組
      "u" 使用者手動改過（重算時保留；rt 為空字串代表「這裡不要標」）
"""
import json
import logging
import re
import threading

import jaconv

from .config import DATA_DIR, MODELS_DIR

log = logging.getLogger("furigana")

TSQYOMI_DIR = MODELS_DIR / "furigana" / "tsqyomi" / "v4"
USER_CSV = DATA_DIR / "user_dict.csv"
USER_DIC = DATA_DIR / "user_dict.dic"
# 套件版本、演算法有改就把 r 的數字加一，舊字幕會在下次讀取時自動重算
ENGINE = "ojtplus0.4.1p9-tsq4-r1"

KANJI = r"[㐀-䶿一-鿿豈-﫿\U00020000-\U0003FFFF々〆ヵヶ〇]"
_KANJI_RUN = re.compile(f"({KANJI}+)")
_HAS_KANJI = re.compile(KANJI)
_HAS_DIGIT = re.compile(r"[0-9０-９]")
_HIRA = re.compile(r"^[ぁ-ゖー]+$")
# 讀音允許出現的字：漢字、假名、數字、小數點與千分位。% ℃ 這類有讀音的符號不在內，整組跳過
_SPAN_OK = re.compile(rf"^(?:{KANJI}|[0-9０-９.．,，ぁ-ヿー])+$")
_SILENT = set(" 　()（）[]［］「」『』【】〈〉《》、。，．・…!?！？\"'“”‘’~〜♪♡")
_NUM_PUNCT = set(".．,，")
_E_ROW = set("えけせてねへめれげぜでべぺぇ")
_O_ROW = set("おこそとのほもよろごぞどぼぽょぉ")

_lock = threading.RLock()
_ready = threading.Event()
_state = {"error": None, "homographs": {}, "dict_hash": ""}


# ---------- 初始化與使用者辭典 ----------

def init():
    """啟動時在背景執行緒呼叫一次（約 1 到 2.5 秒）。失敗只記錄，字幕照常顯示，只是沒有假名。"""
    if not all((TSQYOMI_DIR / f).is_file() for f in ("model.onnx", "tokenizer.json", "metadata.json")):
        # 還沒下載模型是正常情況（新安裝），只記一行，不要在啟動視窗印一大段錯誤；下載後 reload_if_failed 會再試
        _state["error"] = "tsqyomi 模型還沒下載"
        log.info("日文假名要等 tsqyomi 模型下載好才會有（設定頁的模型管理）")
        _ready.set()
        return
    try:
        import pyopenjtalk
        from pyopenjtalk import tsqyomi
        with _lock:
            _apply_user_dict(pyopenjtalk)
            # 明確指定 CPU：之後就算 env 裝了 onnxruntime-gpu 也不會偷用顯存
            tsqyomi.load_model(onnx_providers=["CPUExecutionProvider"], model_dir=TSQYOMI_DIR,
                               allow_provider_fallback=False)
            meta = json.loads((TSQYOMI_DIR / "metadata.json").read_text(encoding="utf-8"))
            _state["homographs"] = {k: list(v) for k, v in meta["class_index_by_surface_and_pronunciation"].items()}
            pyopenjtalk.g2p_mapping("暖機", use_tsqyomi=True)
            _state["error"] = None
    except Exception as e:  # noqa: BLE001
        _state["error"] = f"{type(e).__name__}: {e}"
        log.warning("furigana init failed", exc_info=True)
    finally:
        _ready.set()


def reload_if_failed():
    """在設定頁下載完 tsqyomi 後呼叫（jobs.handle_model）。啟動時模型不在、初始化失敗的話，
    現在再初始化一次，不用重開程式就有假名；本來就正常的不動。還沒初始化過（啟動中）也不動。"""
    if _ready.is_set() and _state["error"]:
        log.info("tsqyomi downloaded, retrying furigana init")
        init()


def available(timeout: float = 30) -> bool:
    return _ready.wait(timeout) and not _state["error"]


def version() -> str:
    return f"{ENGINE}:{_state['dict_hash']}"


def _apply_user_dict(pyopenjtalk):
    import hashlib
    raw = USER_CSV.read_bytes() if USER_CSV.exists() else b""
    if not raw.strip():
        if _state["dict_hash"]:
            pyopenjtalk.unset_user_dict()
        _state["dict_hash"] = ""
        return
    pyopenjtalk.mecab_dict_index(str(USER_CSV), str(USER_DIC))   # 幾十毫秒，會印進度條到 stdout
    # is_reading_protected：不讓 tsqyomi 蓋掉自己指定的讀音
    pyopenjtalk.update_global_jtalk_with_user_dict([{"dic_path": str(USER_DIC), "is_reading_protected": True}])
    _state["dict_hash"] = hashlib.sha1(raw).hexdigest()[:10]


def add_user_word(surface: str, reading: str):
    """加一個名詞到使用者辭典並立即生效。動詞、形容詞不要加（會壞掉活用）。"""
    surface, kata = surface.strip(), jaconv.hira2kata(reading.strip())
    if not _HAS_KANJI.search(surface) or re.search(r'[,\s"]', surface) or not re.fullmatch(r"[ァ-ヶー]+", kata):
        raise ValueError("詞要含漢字、不能有逗號或空白，讀音只能是假名")
    mora = len(re.sub(r"[ァィゥェォャュョヮ]", "", kata))
    line = f"{surface},,,100,名詞,一般,*,*,*,*,{surface},{kata},{kata},0/{mora},*"
    import pyopenjtalk
    with _lock:
        rows = USER_CSV.read_text(encoding="utf-8").splitlines() if USER_CSV.exists() else []
        rows = [r for r in rows if r.strip() and not r.startswith(surface + ",")] + [line]
        USER_CSV.write_text("\n".join(rows) + "\n", encoding="utf-8")
        _apply_user_dict(pyopenjtalk)


# ---------- 讀音 → 字元位置 ----------

def split_okurigana(surface: str, reading: str):
    """('取り扱い', 'とりあつかい') → [('取','と'), ('り',None), ('扱','あつか'), ('い',None)]；對不上回 None"""
    parts = [p for p in _KANJI_RUN.split(surface) if p]
    rx = "".join("(.+?)" if _HAS_KANJI.match(p) else "(" + re.escape(jaconv.kata2hira(p)) + ")" for p in parts)
    m = re.fullmatch(rx, reading)
    if not m:
        return None
    return [(p, m.group(i + 1) if _HAS_KANJI.match(p) else None) for i, p in enumerate(parts)]


def _emit(out: list, text: str, a: int, b: int, kana: str, flag: str):
    while a < b and text[a] in _SILENT:
        a += 1
    while b > a and text[b - 1] in _SILENT:
        b -= 1
    surf = text[a:b]
    if not surf or not _HIRA.match(kana) or not _SPAN_OK.match(surf):
        return
    if _HAS_DIGIT.search(surf):
        if _HAS_KANJI.search(surf):          # 3本、10月、2026年 標；單純的 2026 不標
            out.append([a, b, kana, flag])
        return
    if not _HAS_KANJI.search(surf):
        return
    parts = split_okurigana(surf, kana)
    if parts is None:
        out.append([a, b, kana, "g"])
        return
    cur = a
    for base, rt in parts:
        if rt and rt != jaconv.kata2hira(base):
            out.append([cur, cur + len(base), rt, flag])
        cur += len(base)


def ruby_for_text(text: str) -> list[list]:
    if not text or not (_HAS_KANJI.search(text) or _HAS_DIGIT.search(text)) or not available(0):
        return []
    import pyopenjtalk
    with _lock:
        feats = pyopenjtalk.g2p_mapping(text, use_tsqyomi=True, revert_long_vowels=True, revert_yotsugana=True)
    groups = []  # [起, 迄, 片假名讀音, 數字組還開著]
    for f in feats:
        a, b = f["char_span"]
        if f["is_unknown"]:
            groups.append([a, b, None, False])
            continue
        is_num = f["pos_group1"] == "数" or (b - a == 1 and text[a:b] in _NUM_PUNCT)
        pron = f["pron"].replace("’", "")
        prev = groups[-1] if groups else None
        if prev and prev[3] and (is_num or f["pos_group2"] == "助数詞"):
            # 數字和助數詞併成一組；讀音用 pron 才有 3本=サンボン、6本=ロッポン
            prev[1] = max(prev[1], b)
            prev[2] = (prev[2] or "") + pron
            prev[3] = is_num
            continue
        if b <= a:                       # OpenJTalk 把 2026 展開成 二千二十六 時補出的零寬「十」
            if prev and prev[3]:
                prev[2] = (prev[2] or "") + pron
            continue
        # 一般詞用 read：pron 會把 実は 變 ジツワ、という 變 トユウ
        groups.append([a, b, pron if is_num else f["read"], is_num])
    out, pos = [], 0
    for a, b, rd, _ in groups:
        a = max(a, pos)
        if b <= a:
            continue
        pos = b
        if rd:
            flag = "h" if text[a:b] in _state["homographs"] else ""
            _emit(out, text, a, b, jaconv.kata2hira(rd), flag)
    return out


def apply(cues: list[dict]) -> int:
    """就地重算整軌假名，保留使用者改過的段落。回傳有假名的句數；引擎不能用時回 -1 且不動資料。"""
    if not available():
        return -1
    count = 0
    for c in cues:
        user = [r for r in c.get("ruby") or [] if len(r) > 3 and r[3] == "u"]
        auto = [r for r in ruby_for_text(c.get("text", "")) if not any(r[0] < u[1] and u[0] < r[1] for u in user)]
        spans = sorted(user + auto)
        if spans:
            c["ruby"] = spans
            count += 1
        else:
            c.pop("ruby", None)
    return count


def set_user_span(cue: dict, s: int, e: int, rt: str):
    text = cue.get("text", "")
    if not (0 <= s < e <= len(text)) or (rt and not _HIRA.match(rt)):
        raise ValueError("位置或讀音不正確")
    keep = [r for r in cue.get("ruby") or [] if not (r[0] < e and s < r[1])]
    cue["ruby"] = sorted(keep + [[s, e, rt, "u"]])


# ---------- 讀音候選（給前端的修改彈窗） ----------

def _pron_to_kana(p: str) -> str:
    out = []
    for ch in jaconv.kata2hira(p):
        if ch == "ー" and out:
            ch = "い" if out[-1] in _E_ROW else "う" if out[-1] in _O_ROW else "ー"
        out.append(ch)
    return "".join(out)


def lookup(text: str, s: int, e: int) -> dict:
    """某一段的所屬詞、詞性、同形異音候選（已切成這一段的讀音）。"""
    if not available(5):
        return {"surface": text[s:e], "pos": None, "alternatives": []}
    import pyopenjtalk
    with _lock:
        feats = pyopenjtalk.g2p_mapping(text, use_tsqyomi=True, revert_long_vowels=True, revert_yotsugana=True)
    tok = next((f for f in feats if f["char_span"][0] <= s and e <= f["char_span"][1]), None)
    a, b = tok["char_span"] if tok else (s, e)
    surface, alts = text[a:b], []
    for cand in _state["homographs"].get(surface, []):
        parts = split_okurigana(surface, _pron_to_kana(cand))
        cur = a
        for base, rt in parts or []:
            if cur == s and cur + len(base) == e and rt and rt not in alts:
                alts.append(rt)
            cur += len(base)
    return {"surface": surface, "pos": tok["pos"] if tok else None, "alternatives": alts}
