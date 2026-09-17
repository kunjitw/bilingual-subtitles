"""英文查字：spaCy 分析整句（詞性、原形、片語）＋ ECDICT 台灣繁體（data/dict/en_zh.db）＋ CEFR ＋ wordfreq。

cue["w"]  = [[起, 迄, key, UPOS], ...]            key：en:<原形小寫>；專有名詞保留大小寫（en:Tokyo）
cue["ph"] = [[片語 key, [[起, 迄], ...]], ...]      可以不連續，例如 give it up；key 例如 en:give up

滑鼠單位用自己的 regex 切（gonna、don't、Swift's 各是一個），再用字元位置對回 spaCy 的 token。
spaCy 分析時把被切成好幾行的同一句話接起來，詞性比逐行準。
"""
import functools
import importlib.metadata
import re
import sqlite3
import threading
from collections import Counter

from .config import DICT_DIR

DB = DICT_DIR / "en_zh.db"

try:
    MODEL_VERSION = (f"spacy{importlib.metadata.version('spacy')}"
                     f"-sm{importlib.metadata.version('en_core_web_sm')}")
except importlib.metadata.PackageNotFoundError:
    MODEL_VERSION = "spacy-missing"
SEGMENT_RULES = 3   # 切詞、分組、片語規則改了就加一

UNIT = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)*|['’](?:cause|em|til)\b", re.I)
# 口語縮寫：保留原字當詞條，另外附展開說明
SPOKEN = {"gonna": "going to", "wanna": "want to", "gotta": "got to", "kinda": "kind of", "sorta": "sort of",
          "lemme": "let me", "gimme": "give me", "dunno": "don't know", "outta": "out of", "lotta": "lot of",
          "y'all": "you all", "ain't": "am not / is not / have not", "'cause": "because", "cuz": "because",
          "gotcha": "got you", "ya": "you", "'em": "them", "tryna": "trying to", "finna": "fixing to（正要）"}
FUNCTION_UPOS = {"PRON", "DET", "AUX", "CCONJ", "SCONJ", "ADP", "PART"}
# 這幾個當本動詞時 spaCy 會標 VERB，但在字幕裡絕大多數是助動詞用法，一律不算單字
ALWAYS_FUNCTION = {"be", "have", "do", "will", "would", "shall", "should", "can", "could", "may", "might", "must"}
SKIP_UPOS = {"NUM", "PUNCT", "SYM", "SPACE"}
PARTICLES = {"up", "down", "out", "off", "on", "in", "over", "away", "back", "around", "about", "along", "through",
             "by", "across", "ahead", "apart", "aside", "forward", "together", "after", "for", "with", "into", "onto",
             "from", "to", "of", "at"}
# 規則比對容易誤判、而且另有正確寫法的組合
PHRASE_BLOCK = {"get to", "use to", "have to", "go to", "be to", "do to", "accord to"}
FIXED_SURFACE = {"according to", "used to", "supposed to", "got to", "going to", "based on", "due to"}

POS_PREFIX = {"VERB": ("v.", "vt.", "vi."), "AUX": ("v.", "vt.", "vi.", "aux."), "NOUN": ("n.", "pl."),
              "PROPN": ("n.",), "ADJ": ("a.", "adj."), "ADV": ("adv.", "ad."), "ADP": ("prep.",),
              "PRON": ("pron.",), "CCONJ": ("conj.",), "SCONJ": ("conj.",), "INTJ": ("int.", "interj."),
              "DET": ("det.", "art.", "a."), "NUM": ("num.",)}

_nlp = None
_nlp_lock = threading.Lock()   # spaCy 不能多執行緒同時用
_local = threading.local()
_all_cons: list = []
_all_lock = threading.Lock()
_gen = [0]


def ready() -> bool:
    return DB.exists()


def nlp_available() -> bool:
    return MODEL_VERSION != "spacy-missing"


def con() -> sqlite3.Connection:
    c = getattr(_local, "con", None)
    if c is None or getattr(_local, "gen", -1) != _gen[0]:
        c = sqlite3.connect(DB.as_uri() + "?mode=ro", uri=True, check_same_thread=False)
        with _all_lock:
            _all_cons.append(c)
        _local.con, _local.gen = c, _gen[0]
    return c


def close_all():
    with _all_lock:
        _gen[0] += 1
        for c in _all_cons:
            try:
                c.close()
            except Exception:  # noqa: BLE001
                pass
        _all_cons.clear()
    for f in (meta, has_entry, entry, cefr, zipf):
        f.cache_clear()


@functools.lru_cache(maxsize=16)
def meta(key: str) -> str:
    if not ready():
        return ""
    row = con().execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return row[0] if row else ""


def nlp():
    global _nlp
    if _nlp is None:
        import spacy
        _nlp = spacy.load("en_core_web_sm", disable=["ner"])
    return _nlp


# ---------- 字典 ----------

@functools.lru_cache(maxsize=200_000)
def has_entry(word: str) -> bool:
    return con().execute("SELECT 1 FROM dict WHERE word = ? LIMIT 1", (word,)).fetchone() is not None


@functools.lru_cache(maxsize=50_000)
def entry(word: str) -> dict | None:
    row = con().execute(
        "SELECT word, phonetic, zh, collins, oxford, lemma, tag FROM dict WHERE word = ? "
        "ORDER BY (word = ? COLLATE BINARY) DESC LIMIT 1", (word, word)).fetchone()
    if not row:
        return None
    lines = [ln.strip() for ln in (row[2] or "").split("\n") if ln.strip()]
    tags = (row[6] or "").split()
    return {"word": row[0], "phonetic": row[1] or "", "lines": lines, "collins": row[3] or 0,
            "oxford": bool(row[4]), "base": row[5] or "", "toefl": "toefl" in tags, "ielts": "ielts" in tags}


def sort_lines(lines: list[str], upos: str = "") -> list[str]:
    """句子裡的詞性排前面（give 的 vt. 放在 n. 前），[網路]、[計] 這類放最後。"""
    want = POS_PREFIX.get(upos, ())
    return sorted(lines, key=lambda ln: (ln.startswith("["), not ln.startswith(want) if want else False))


@functools.lru_cache(maxsize=1)
def _cefr_map() -> dict:
    return dict(con().execute("SELECT word, level FROM cefr"))


@functools.lru_cache(maxsize=50_000)
def cefr(word: str) -> str:
    return _cefr_map().get(word.lower(), "")


@functools.lru_cache(maxsize=50_000)
def zipf(word: str) -> float:
    if " " in word:
        return 0.0      # 片語的 zipf 是用單字推估的，不準，不顯示
    try:
        from wordfreq import zipf_frequency
        return round(zipf_frequency(word, "en"), 2)
    except Exception:  # noqa: BLE001
        return 0.0


def display_of(key: str) -> str:
    return key.split(":", 1)[1] if ":" in key else key


def summary(word: str, upos: str = "", max_lines: int = 4) -> dict:
    """詞表和小卡用：依詞性排序後的前幾行 ECDICT 釋義。"""
    e = entry(word) or (entry(word.lower()) if word != word.lower() else None)
    low = word.lower()
    out = {"lv": cefr(low), "z": zipf(low)}
    if low in SPOKEN:
        out["sp"] = SPOKEN[low]
    if e:
        lines = sort_lines(e["lines"], upos)
        out.update(zl=lines[:max_lines], pn=e["phonetic"], g=lines[0] if lines else "", gs="ecdict" if lines else "")
    else:
        out.update(zl=[], pn="", g="", gs="")
    return out


# ---------- 分析字幕 ----------

def _phrases(doc, max_gap: int = 3) -> list:
    toks = list(doc)
    out, used = [], set()
    for i, t in enumerate(toks):
        if t.pos_ != "VERB" or i in used:
            continue
        lem = t.lemma_.lower()
        for j in range(i + 1, min(len(toks), i + 2 + max_gap)):
            tj = toks[j]
            low = tj.text.lower()
            if low in PARTICLES:
                if low == "to" and tj.pos_ == "PART":
                    break                  # 不定詞的 to（need to do）不算片語
                gap = toks[i + 1:j]
                # 不相鄰（give it up）時，一定要 parser 標成 prt，中間也只能是受詞
                if gap and not (tj.dep_ == "prt" and tj.head.i == t.i
                                and all(g.pos_ in ("PRON", "DET", "NOUN", "PROPN", "ADJ", "NUM") for g in gap)):
                    break
                if not gap and f"{t.text.lower()} {low}" in FIXED_SURFACE:   # according to / used to 不還原
                    out.append((f"{t.text.lower()} {low}", [i, j]))
                    used.update((i, j))
                    break
                nxt = toks[j + 1].text.lower() if j + 1 < len(toks) and not gap else ""
                if nxt in PARTICLES and has_entry(f"{lem} {low} {nxt}"):        # come up with / run out of
                    out.append((f"{lem} {low} {nxt}", [i, j, j + 1]))
                    used.update((i, j, j + 1))
                elif f"{lem} {low}" not in PHRASE_BLOCK and has_entry(f"{lem} {low}"):
                    out.append((f"{lem} {low}", [i, j]))
                    used.update((i, j))
                break
            if tj.pos_ in ("VERB", "AUX", "PUNCT", "SCONJ", "CCONJ"):
                break
    return out


def annotate(cues: list[dict], meta_out: dict):
    """就地為英文字幕加上 w / ph，meta_out[key] 收集顯示用的原形、分組、最常見的詞性。"""
    from .translate import build_units
    units = build_units(cues, "en")
    texts, offsets = [], []
    for u in units:
        parts, pos, starts = [], 0, []
        for i in u:
            starts.append(pos)
            parts.append(cues[i].get("text", ""))
            pos += len(parts[-1]) + 1
        texts.append(" ".join(parts))
        offsets.append(starts)
    pos_count: dict[str, Counter] = {}
    grp_votes: dict[str, Counter] = {}
    with _nlp_lock:
        docs = list(nlp().pipe(texts, batch_size=128))
    for u, starts, doc in zip(units, offsets, docs):
        by_char = {}
        for t in doc:
            by_char.setdefault(t.idx, t)
        for i, base in zip(u, starts):
            cue = cues[i]
            text = cue.get("text", "")
            words = []
            for m in UNIT.finditer(text):
                s, e = m.span()
                surface = m.group(0).replace("’", "'")
                low = surface.lower()
                head = by_char.get(base + s)
                upos = head.pos_ if head is not None else ""
                if low in SPOKEN:
                    key, grp, upos = f"en:{low}", "word", ""
                elif head is None:
                    key, grp = f"en:{low}", "word"
                elif upos in SKIP_UPOS:
                    continue
                elif upos == "PROPN" or (head.tag_ in ("NNP", "NNPS") and s > 0):
                    key, grp, upos = f"en:{head.text}", "name", "PROPN"
                else:
                    lemma = head.lemma_.lower()
                    if not lemma.isascii() or not lemma.strip() or lemma == "-pron-":
                        lemma = low
                    key = f"en:{lemma}"
                    if upos == "INTJ":
                        grp = "filler"
                    elif lemma in ALWAYS_FUNCTION or (upos in FUNCTION_UPOS and zipf(lemma) >= 5.5):
                        grp = "function"
                    elif upos == "X":
                        grp = "unknown"
                    else:
                        grp = "word"
                words.append([s, e, key, upos])
                meta_out.setdefault(key, {"lang": "en", "display": display_of(key), "reading": None,
                                          "grp": grp, "entry_id": None, "pos": upos, "phrase": 0})
                grp_votes.setdefault(key, Counter())[grp] += 1
                if upos:
                    pos_count.setdefault(key, Counter())[upos] += 1
            phrases = []
            lo, hi = base, base + len(text)
            for phrase, idxs in _phrases(doc):
                spans = [[doc[k].idx - base, doc[k].idx + len(doc[k].text) - base] for k in idxs]
                if all(lo <= doc[k].idx and doc[k].idx + len(doc[k].text) <= hi for k in idxs):
                    key = f"en:{phrase}"
                    phrases.append([key, spans])
                    meta_out.setdefault(key, {"lang": "en", "display": phrase, "reading": None, "grp": "word",
                                              "entry_id": None, "pos": "VERB", "phrase": 1})
            if words:
                cue["w"] = words
            else:
                cue.pop("w", None)
            if phrases:
                cue["ph"] = phrases
            else:
                cue.pop("ph", None)
    for key, cnt in pos_count.items():
        meta_out[key]["pos"] = cnt.most_common(1)[0][0]
    # 分組看這部片裡多數的用法（of 偶爾被標成 ADV，不能因此整個算成單字）
    for key, votes in grp_votes.items():
        meta_out[key]["grp"] = votes.most_common(1)[0][0]


def warm():
    if ready() and nlp_available():
        with _nlp_lock:
            nlp()
