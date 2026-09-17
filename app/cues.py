"""字幕組裝：對齊結果 → 帶標點的字詞 → 字幕行；以及 SRT / VTT 匯出。"""
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from . import safepath
from .config import SUBS_DIR

SENTENCE_END = set("。！？!?…♪")
SOFT_BREAK = set("、，,;；：:")
CLOSERS = "\"'”’)]）」』】"
EN_ABBREV = {"mr", "mrs", "ms", "dr", "prof", "st", "vs", "jr", "sr", "inc", "ltd", "no", "vol", "fig", "approx"}
# 超過長度必須斷行時，英文優先斷在這些字前面，日文優先斷在助詞後面
EN_BREAK_BEFORE = {"and", "but", "or", "so", "because", "that", "which", "who", "when", "while", "where", "if",
                   "although", "though", "than", "to", "in", "on", "at", "for", "with", "from", "of", "about",
                   "after", "before", "until", "since", "as", "like", "into", "through"}
JA_BREAK_AFTER = {"は", "が", "を", "に", "で", "と", "も", "へ", "から", "まで", "けど", "ので", "のに", "て", "ば", "たら"}
GAP_BREAK_S = 0.6
MAX_CUE_S = 7.0
MIN_CUE_S = 0.8
READ_TAIL_S = 0.3

LIMITS = {
    # soft：長度超過就在逗號處斷行；hard：強制斷行
    "cjk": {"soft": 12, "hard": 24},
    "latin": {"soft": 40, "hard": 80},
}

_opencc = None


def to_taiwan(text: str) -> str:
    global _opencc
    if _opencc is None:
        from opencc import OpenCC
        _opencc = OpenCC("s2twp")
    return _opencc.convert(text)


@dataclass
class Token:
    text: str
    start: float
    end: float


def _kept(ch: str) -> bool:
    # 與 Qwen3-ForcedAligner 的 clean_token 規則一致：字母、數字、撇號
    return ch == "'" or unicodedata.category(ch)[0] in "LN"


def attach_units(text: str, units: list, chunk: list[float]) -> list[Token]:
    """把對齊器的字詞（已去標點）對回原文，讓每個字詞帶著後面的標點與空白。"""
    if not text.strip():
        return []
    kept_pos = [i for i, ch in enumerate(text) if _kept(ch)]
    kept_str = "".join(text[i] for i in kept_pos)
    spans = []  # (text_start, start_s, end_s)
    p = 0
    for utext, s, e in units:
        n = len(utext)
        if n == 0:
            continue
        if kept_str[p:p + n] == utext:
            j = p
        else:
            j = kept_str.find(utext, p)
            if j == -1 or j - p > 40:
                continue
        spans.append((kept_pos[j], float(s), float(e)))
        p = j + n

    if not spans:
        return _even_tokens(text, chunk)

    # 對齊結果明顯不合理（時間跑出這段範圍，或一大段文字被擠在極短時間內）就不採用
    first, last = spans[0][1], max(s[2] for s in spans)
    chunk_len = max(0.1, chunk[1] - chunk[0])
    out_of_range = last > chunk[1] + 1.0 or first < chunk[0] - 1.0
    squeezed = len(kept_str) > 40 and (last - first) < 0.2 * chunk_len and (last - first) < len(kept_str) * 0.05
    if out_of_range or squeezed:
        return _even_tokens(text, chunk)

    tokens = []
    for k, (pos, s, e) in enumerate(spans):
        begin = 0 if k == 0 else pos
        stop = spans[k + 1][0] if k + 1 < len(spans) else len(text)
        tokens.append(Token(text[begin:stop], s, max(e, s)))
    return tokens


def _even_tokens(text: str, chunk: list[float]) -> list[Token]:
    """對齊失敗時的備援：把文字平均分配到整段時間。"""
    words = text.split() if " " in text.strip() else list(text)
    if not words:
        return []
    dur = (chunk[1] - chunk[0]) / len(words)
    sep = " " if " " in text.strip() else ""
    return [Token(w + sep, chunk[0] + i * dur, chunk[0] + (i + 1) * dur) for i, w in enumerate(words)]


def _visible_len(tokens: list[Token]) -> int:
    return len("".join(t.text for t in tokens).strip())


def _tail(text: str) -> str:
    return text.rstrip().rstrip(CLOSERS)


def _ends_sentence(text: str, lang: str) -> bool:
    tail = _tail(text)
    if not tail:
        return False
    if tail[-1] in SENTENCE_END:
        return True
    if lang == "en" and tail[-1] == ".":
        words = tail.rstrip(".").split()
        word = words[-1].strip("\"'(") if words else ""
        # 縮寫（Mr.、U.S.）與單一大寫字母不算句尾
        if word.lower() in EN_ABBREV or re.search(r"\.\w", tail) or (len(word) == 1 and word.isupper()):
            return False
        return True
    return False


def _soft_break(text: str) -> bool:
    tail = _tail(text)
    return bool(tail) and tail[-1] in SOFT_BREAK


def normalize_asr_text(text: str, lang: str) -> str:
    """英文辨識結果偶爾句號後面沒空格（situation.For），補上才切得開句子。"""
    if lang == "en":
        text = re.sub(r"(?<=[a-z])([.!?])(?=[A-Z])", r"\1 ", text)
        text = re.sub(r"(?<=[a-zA-Z])([,;:])(?=[A-Za-z])", r"\1 ", text)
    elif lang == "ja":
        # 日文不用空格：去掉日文字旁邊的空格，以及單一英文字母之間的空格（J R → JR）
        text = re.sub(r"\s+(?=[^\x00-\x7F])|(?<=[^\x00-\x7F])\s+", "", text)
        joined = None
        while joined != text:
            joined = text
            text = re.sub(r"(?<![A-Za-z])([A-Z])\s+(?=[A-Z](?![a-z]))", r"\1", text)
    return text


def _word(tok: Token) -> str:
    return "".join(ch for ch in tok.text if _kept(ch)).lower()


def _boundary_bonus(toks: list[Token], k: int, lang: str) -> float:
    """在 toks[k-1] 和 toks[k] 之間斷開有多自然，越大越好。"""
    left, right = toks[k - 1], toks[k]
    if _soft_break(left.text) or _ends_sentence(left.text, lang):
        return 0.5
    bonus = 0.0
    if lang == "en" and _word(right) in EN_BREAK_BEFORE:
        bonus = 0.25
    elif lang == "ja" and _word(left) in JA_BREAK_AFTER:
        bonus = 0.25
    if right.start - left.end > 0.25:
        bonus += 0.2
    return bonus


def _split_balanced(toks: list[Token], lang: str, soft: int, hard: int) -> list[list[Token]]:
    """一句話太長時，挑最平均又最自然的位置切成兩半，必要時再往下切。"""
    total_len = _visible_len(toks)
    total_s = toks[-1].end - toks[0].start
    fits = total_len <= hard and total_s <= MAX_CUE_S
    if len(toks) < 2:
        return [toks]
    min_side = 8 if lang == "en" else 3
    best_k, best_score = None, None
    for k in range(1, len(toks)):
        ll, lr = _visible_len(toks[:k]), _visible_len(toks[k:])
        balance = abs(ll - lr) / max(1, total_len)
        if total_s > MAX_CUE_S:
            tl = toks[k - 1].end - toks[0].start
            balance = max(balance, abs(tl - (total_s - tl)) / max(0.1, total_s))
        score = balance - _boundary_bonus(toks, k, lang)
        if min(ll, lr) < min_side:
            score += 1.0
        if best_score is None or score < best_score:
            best_k, best_score = k, score
    if fits:
        # 沒超過上限但偏長：中間附近剛好有逗號才分兩行
        if total_len <= soft * 1.5 or best_k is None:
            return [toks]
        left = toks[best_k - 1]
        ll = _visible_len(toks[:best_k])
        if not _soft_break(left.text) or not (0.3 <= ll / total_len <= 0.7):
            return [toks]
    return _split_balanced(toks[:best_k], lang, soft, hard) + _split_balanced(toks[best_k:], lang, soft, hard)


def segment(tokens: list[Token], lang: str) -> list[dict]:
    """切成字幕行。每行記錄斷開的原因 brk：eos 句尾、gap 停頓、soft 逗號、hard 長度、end 結尾。"""
    kind = "latin" if lang == "en" else "cjk"
    soft, hard = LIMITS[kind]["soft"], LIMITS[kind]["hard"]

    sentences: list[tuple[list[Token], str]] = []
    cur: list[Token] = []
    for tok in tokens:
        if cur and tok.start - cur[-1].end >= GAP_BREAK_S:
            sentences.append((cur, "gap"))
            cur = []
        cur.append(tok)
        if _ends_sentence(tok.text, lang):
            sentences.append((cur, "eos"))
            cur = []
    if cur:
        sentences.append((cur, "end"))

    out = []
    for toks, reason in sentences:
        pieces = _split_balanced(toks, lang, soft, hard)
        for k, piece in enumerate(pieces):
            text = "".join(t.text for t in piece).strip()
            if not text:
                continue
            if k == len(pieces) - 1:
                brk = reason
            else:
                brk = "soft" if _soft_break(piece[-1].text) else "hard"
            out.append({"start": piece[0].start, "end": piece[-1].end, "text": text, "brk": brk})
    return out


# 翻譯模型偶爾直接輸出日文漢字，OpenCC 對不上詞彙時會轉錯
ZH_FIXES = {"臺風": "颱風", "台風": "颱風", "臺灣": "台灣"}


def tidy_text(text: str, lang: str) -> str:
    text = " ".join(text.split())
    if lang == "zh-TW":
        for wrong, right in ZH_FIXES.items():
            text = text.replace(wrong, right)
    if lang in ("zh-TW", "ja"):
        text = text.rstrip("、，,。")
        text = text.lstrip("、，,。")
    return text


TINY_CUE_S = 0.5


def _merge_tiny(cues: list[dict], lang: str) -> list[dict]:
    """對齊器偶爾把幾個字擠在很短的時間裡，這種一閃而過的行併進下一句。"""
    hard = LIMITS["latin" if lang == "en" else "cjk"]["hard"]
    sep = "" if lang == "ja" else " "
    few_chars = 3 if lang == "en" else 2

    def tiny(c, nxt):
        gap = nxt["start"] - c["end"]
        flash = c["end"] - c["start"] < TINY_CUE_S and gap < 0.3
        lonely = len(c["text"]) <= few_chars and gap < 0.8
        return (flash or lonely) and len(c["text"]) + len(nxt["text"]) + 1 <= hard

    out: list[dict] = []
    i = 0
    while i < len(cues):
        c = dict(cues[i])
        while i + 1 < len(cues) and tiny(c, cues[i + 1]):
            nxt = cues[i + 1]
            c = {**nxt, "start": c["start"], "text": f"{c['text']}{sep}{nxt['text']}"}
            i += 1
        out.append(c)
        i += 1
    return out


def finalize(cues: list[dict], lang: str) -> list[dict]:
    """轉台灣繁體、整理標點、修正時間（最短顯示時間、結尾留一點閱讀時間、不重疊）。"""
    result = []
    for c in cues:
        text = c["text"]
        if lang == "zh-TW":
            text = to_taiwan(text)
        text = tidy_text(text, lang)
        if text:
            item = {"start": round(c["start"], 3), "end": round(c["end"], 3), "text": text}
            if c.get("brk"):
                item["brk"] = c["brk"]
            result.append(item)
    result = _merge_tiny(result, lang)
    for i in range(1, len(result)):
        if result[i]["start"] < result[i - 1]["start"]:
            result[i]["start"] = result[i - 1]["start"] + 0.05
    for i, c in enumerate(result):
        c["end"] = display_end(c["start"], c["end"], result[i + 1]["start"] if i + 1 < len(result) else None)
        c["start"] = round(c["start"], 3)
    return result


def display_end(start: float, last: float, next_start: float | None) -> float:
    """一行字幕顯示到什麼時候：最後一個字結束後留 READ_TAIL_S 閱讀時間、至少顯示 MIN_CUE_S，
    但不蓋到下一行（下一行開始前 0.02 秒）。時間軸檢查重新對齊的行也用這個規則。"""
    desired = max(last + READ_TAIL_S, start + MIN_CUE_S)
    limit = next_start - 0.02 if next_start is not None else last + 2.0
    return round(max(round(start, 3) + 0.05, min(desired, limit)), 3)


# ---------- 存取與匯出 ----------

def cue_path(track_id: str) -> Path:
    """字幕檔的位置。track_id 只能是 id 字元，擋掉 ..、絕對路徑這類會跳出 data/subs 的寫法（讀、寫、刪都經過這裡）。"""
    safepath.require_safe_name(track_id, "字幕軌 id")
    return SUBS_DIR / f"{track_id}.json"


def save_cues(track_id: str, cues: list[dict]):
    """先寫暫存檔再替換，寫到一半關掉程式也不會把字幕檔弄壞。"""
    SUBS_DIR.mkdir(parents=True, exist_ok=True)
    path = cue_path(track_id)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(cues, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def load_cues(track_id: str) -> list[dict]:
    p = cue_path(track_id)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else []


def _ts(sec: float, sep: str) -> str:
    ms = int(round(sec * 1000))
    h, ms = divmod(ms, 3600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def to_srt(cues: list[dict]) -> str:
    return "\n".join(
        f"{i}\n{_ts(c['start'], ',')} --> {_ts(c['end'], ',')}\n{c['text']}\n" for i, c in enumerate(cues, 1)
    )


def to_vtt(cues: list[dict]) -> str:
    body = "\n".join(f"{_ts(c['start'], '.')} --> {_ts(c['end'], '.')}\n{c['text']}\n" for c in cues)
    return "WEBVTT\n\n" + body


def merge_bilingual(top: list[dict], bottom: list[dict]) -> list[dict]:
    """以第一軌的時間為準，把第二軌中時間重疊最多的句子接在下一行。"""
    merged = []
    j = 0
    for c in top:
        texts = []
        while j < len(bottom) and bottom[j]["end"] <= c["start"]:
            j += 1
        k = j
        while k < len(bottom) and bottom[k]["start"] < c["end"]:
            overlap = min(c["end"], bottom[k]["end"]) - max(c["start"], bottom[k]["start"])
            if overlap > 0.5 * min(c["end"] - c["start"], bottom[k]["end"] - bottom[k]["start"]):
                texts.append(bottom[k]["text"])
            k += 1
        line2 = " ".join(texts)
        merged.append({"start": c["start"], "end": c["end"], "text": c["text"] + ("\n" + line2 if line2 else "")})
    return merged
