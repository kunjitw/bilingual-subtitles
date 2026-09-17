"""日文查字：JMdict（data/dict/jmdict.db）＋ Yomitan 去活用規則 ＋ UniDic 斷詞。

出處：去活用的 _rules() 和 deinflect() 改寫自 Yomitan（https://github.com/yomidevs/yomitan）
  ext/js/language/language-transformer.js 的 LanguageTransformer（_getConditionFlagsMap、transform），
  Copyright (C) 2024-2026 Yomitan Authors，授權 GPL-3.0-or-later。
  改了什麼：從 JavaScript 改寫成 Python；規則改讀 tools/yomitan/dump_yomitan.mjs 匯出的 JSON，不直接執行 Yomitan 的 JS；
  只支援字尾和整個詞的規則，用字尾字串查表取代逐條比對正規表示式，遇到真正的正規表示式就報錯；
  判斷循環時只比對活用名稱和文字（Yomitan 還會比對規則編號），另外活用鏈超過 8 層就不再往下拆。
  scan() 從游標往後由長到短查詞、同長度再比活用鏈長短的排序，也是照 Yomitan 查詞的做法。
  規則資料本身（tools/yomitan/yomitan_ja_transforms.json）也是 GPL-3.0-or-later，見 tools/yomitan/README.md。

流程：fugashi（unidic-lite）先切出 token，每個實詞 token 往後做最長比對（只試結束在 token 邊界上的長度），
每個長度都先去活用再查 JMdict，詞性要吻合才算。查到的就是一個滑鼠可以查的詞單位。
另外幾條規則（SEGMENT_RULES 9，理由見各段註解）：
- 助詞、助動詞開頭的說法（という、っていう、かもしれない、について…）只收 EXPRESSIONS 白名單裡的條目，
  其他助詞照舊不包；實詞的最長比對吞掉了說法的開頭（しなきゃ｜いけない）時，實詞讓出那一段。
- て形後面的補助動詞（ていく、てくる、てみる、てくれる、てもらう…）用 TE_AUX 補的去活用規則併進前面的動詞。
  這些規則只給切詞用，Shift 即時查詢（scan）照舊只用 Yomitan 的規則。
- 跨過 UniDic 切點的假合併（さんと→三都、いいん→委員、説明すると→すると、ところで、今日は→こんにちは）放棄那個長度改試短的。
- 同長度排序多看 UniDic 詞性（副詞 そう、感嘆詞 あの）和口語縮約（めっちゃ 不是 めく）；口語的 行った 當 行く。
- 這一行開頭接著上一行的說法（と｜いう、かも｜しれない、そう｜いう）照兩行合起來的字串決定 key（_carry_over）。

cue["w"] = [[起, 迄, key, 活用鏈(沒有就省略)], ...]，起迄是 cue["text"] 的字元位置
跨行的詞只在一行計數，另一行那一段是 [起, 迄, key, 活用鏈, 1]：滑鼠照樣查得到，但不重複計數。
見せて｜いた 算在上一行（這一行的 いた 標 1）；そう｜いう 算在這一行（上一行的 そう 改成 そういう、標 1）。
key：jm:<JMdict ent_seq>；JMdict 查不到的用 ja:<UniDic 語彙素>|<詞性>

假名標註（furigana.py）現在用 pyopenjtalk，切點跟 UniDic 不一定一樣；前端遇到跨詞的 ruby 就不包那個詞。
這裡的 Tagger 自己建，跟假名標註無關。
"""
import functools
import importlib.metadata
import json
import re
import sqlite3
import threading
from collections import defaultdict, namedtuple
from dataclasses import dataclass

import jaconv

from .config import DICT_DIR

DB = DICT_DIR / "jmdict.db"
RULES_PATH = DICT_DIR / "yomitan_ja_transforms.json"

try:
    TOKENIZER_VERSION = "unidic-lite-" + importlib.metadata.version("unidic-lite")
except importlib.metadata.PackageNotFoundError:
    TOKENIZER_VERSION = "unidic-lite"
# 切詞、分組規則改了就加一，舊字幕下次讀取時會自動重算
SEGMENT_RULES = 9

HALF2FULL = str.maketrans({chr(c): chr(c + 0xFEE0) for c in range(0x21, 0x7F)})   # JMdict 寫 ドＳ、ＡＳＭＲ
META_CHARS = set("\\[](){}.*+?|$^")
SKIP_POS1 = {"助詞", "助動詞", "補助記号", "空白"}   # 不包 span、不算單字
GLUE_HEAD = {"動詞", "形容詞", "形状詞"}
MAX_LEN = 16
_DIGITS = re.compile(r"^[0-9０-９.,．，]+$")
CONTINUE_BREAKS = {"hard", "soft"}   # 上一行的 brk 是這些時，句子還沒講完（見 translate.build_units）


@dataclass(frozen=True, slots=True)
class Expr:
    no_iu_kanji: bool = False     # 寫成 言う 的留給動詞 言う（Jiten 的做法：って言って 是真的在說）
    prev_pos: tuple = ()          # 前一個 token 的詞性要是這些（ゆったり｜として 的 と 不是 として）
    kana_only: bool = False       # 寫漢字時多半是本動詞（対岸に渡って 是「渡過去」）


_IU = Expr(no_iu_kanji=True)
_NOUNISH = ("名詞", "代名詞", "接尾辞")
# 助詞、助動詞開頭、學習者要當成一個詞學的說法（JMdict ent_seq）。
# 依據：BCCWJ 長単位把複合辞當一個單位（における、として、によって），jpdb、Jiten、ichiran 也都把這些列成獨立的詞。
# 用白名單而不是全部放開：從助詞開始比對會冒出 になった→担う、てい→体、をして、にして 這類假詞（片庫實測）。
EXPRESSIONS = {
    # と／って＋いう 類：只收寫假名、沒有活用的（と言います、って言って 是動詞 言う）
    1922760: _IU,   # という
    2757880: _IU,   # っていう
    2206220: _IU,   # というか
    2122590: _IU,   # っていうか
    2136030: _IU,   # というと
    1008850: _IU,   # といえば
    1008500: _IU,   # というのは
    2009080: _IU,   # というのも
    1008840: _IU,   # というわけ
    2136300: _IU,   # ということは
    2540200: _IU,   # といった
    2136190: _IU,   # といっても
    2037320: _IU,   # とはいえ
    2857085: _IU,   # ってことは
    # 複合助詞
    1008590: Expr(prev_pos=_NOUNISH),               # として
    2256430: Expr(prev_pos=_NOUNISH),               # としては
    1008600: Expr(prev_pos=_NOUNISH + ("助動詞",)),  # としても
    1009780: Expr(),   # について
    1009640: Expr(),   # によって
    1009660: Expr(),   # による
    2076730: Expr(),   # により
    1009670: Expr(),   # によると
    1009600: Expr(),   # にとって
    1009800: Expr(),   # に対して
    1009810: Expr(),   # に対する
    1009740: Expr(),   # に関して
    1009550: Expr(),   # において
    1009560: Expr(),   # における
    2013910: Expr(kana_only=True),   # にわたって
    2669380: Expr(kana_only=True),   # にわたり
    1009720: Expr(),   # に加えて
    2630530: Expr(),   # に従って
    2644330: Expr(),   # に向けて
    1009760: Expr(),   # に基づいて
    1009700: Expr(),   # に応じて
    2100520: Expr(),   # を通して
    1008830: Expr(),   # とともに
    1612900: Expr(),   # にもかかわらず
    2026640: Expr(),   # にしても
    2028250: Expr(),   # からすると
    2087300: Expr(),   # かどうか
    # 推量、否定、義務
    1002970: Expr(),   # かもしれない
    1002975: Expr(),   # かもしれません
    2100710: Expr(),   # かもしれん
    2026790: Expr(),   # しかない
    2755350: Expr(),   # じゃない
    2823770: Expr(),   # ではない
    2394660: Expr(),   # ないといけない
    2831357: Expr(),   # なきゃいけない
    2831358: Expr(),   # なくちゃいけない
    2255320: Expr(),   # なくてはいけない
    1632350: Expr(),   # なければならない（なければいけない 也是這個條目）
}
# 同一個說法的禮貌、口語形併到基本形（單字頁上是同一個詞），值是 (基本形條目, 補在活用鏈最裡面的說明)
EXPR_ALIAS = {1002975: (1002970, ("-ます",)),   # かもしれません → かもしれない
              2100710: (1002970, ())}           # かもしれん → かもしれない

# て形後面的補助動詞：Yomitan 的日文規則只有 -いる、-おく、-しまう、-ちゃう、-ちまう，這些補在這裡（不改 Yomitan 的 JSON）。
# 清單參考 ichiran dict-grammar.lisp 和 Jiten 的合併規則，JMdict 也把這些義項標成「after the -te form of a verb」。
# 名稱: (補助動詞的寫法, 補助動詞本身的活用類別, UniDic 語彙素)
# 補助動詞照慣例寫假名；寫漢字的（訪れて見て、歩いて行きます）多半是真的「看」「去」，留給本動詞。
# 下さる、頂く 寫漢字時幾乎都是補助用法，照收。
# UniDic 也要切成「動詞＋て＋這個補助動詞」才算（初めて｜来ました 的 初めて 是副詞，不是 初める＋てくる；
# なって｜き｜ました 的 き 是 来る，不是 てく）
TE_AUX = {
    "-くる": (("くる",), "vk", {"来る"}),
    "-いく": (("いく", "く"), "v5", {"行く", "てく"}),       # 撮ってく 是 撮っていく 的口語
    "-みる": (("みる",), "v1", {"見る"}),
    "-くれる": (("くれる",), "v1", {"呉れる"}),
    "-もらう": (("もらう",), "v5", {"貰う"}),
    "-あげる": (("あげる",), "v1", {"上げる"}),
    "-ある": (("ある",), "v5", {"有る"}),
    "-いただく": (("いただく", "頂く"), "v5", {"頂く"}),
    "-くださる": (("くださる", "下さる"), "v5", {"下さる"}),
    # てやる 不收：集中して｜やってる 的 やる 多半是「做」，片庫實測誤併比真的「てやる」多
}
# 口語縮約（ては→ちゃ、ければ→きゃ）：句子裡的字串本身就是條目時，不要硬拆成別的詞（めっちゃ 不是 めく）
SLANG_CHAIN = {"-ちゃ", "-ゃ"}
# UniDic 詞性很可靠、而且 JMdict 同音條目常常只差在詞性的幾類（副詞 そう 不是樣態的 そう）
POS_JM = {"副詞": ("adv", "adv-to", "n-adv"), "感動詞": ("int",), "接続詞": ("conj",), "連体詞": ("adj-pn", "adj-f")}
# 前面有修飾語時不該跟後面助詞併成一個條目的形式名詞，值是會被擋掉的條目詞性：
# しゃべる｜とき｜に→時に、あなたの｜ところ｜で→ところで、広がる｜こと｜に→殊に。
# こと 後面的 exp（ことができる、ことから、ことによって）是真的說法，不擋
FORMAL_NOUNS = {"所": {"conj", "adv", "prt", "int", "adj-pn", "exp"}, "時": {"conj", "adv", "prt", "int", "adj-pn", "exp"},
                "事": {"conj", "adv", "prt", "int"}}
GREETINGS = {1289400, 1289480}      # 今日は→こんにちは、今晩は→こんばんは
DEMONSTRATIVES = ("そう", "こう", "ああ", "どう")   # 跨行接 そう｜いう 這類

_local = threading.local()
_all_cons: list = []
_all_lock = threading.Lock()
_gen = [0]
_forms = None
_forms_lock = threading.Lock()
_tagger = None
_tag_lock = threading.Lock()   # MeCab tagger 不能多執行緒同時用（API 執行緒池 + 背景索引）


def ready() -> bool:
    return DB.exists() and RULES_PATH.exists()


def con() -> sqlite3.Connection:
    c = getattr(_local, "con", None)
    if c is None or getattr(_local, "gen", -1) != _gen[0]:
        c = sqlite3.connect(DB.as_uri() + "?mode=ro", uri=True, check_same_thread=False)
        with _all_lock:
            _all_cons.append(c)
        _local.con, _local.gen = c, _gen[0]
    return c


def close_all():
    """重建字典前呼叫。Windows 上檔案開著就不能 os.replace。"""
    global _forms
    with _all_lock:
        _gen[0] += 1
        for c in _all_cons:
            try:
                c.close()
            except Exception:  # noqa: BLE001
                pass
        _all_cons.clear()
    _forms = None
    for f in (entries_for, entry_info, meta, deinflect, headword, jlpt_level, summary, _expr_heads, _units_alone):
        f.cache_clear()


@functools.lru_cache(maxsize=16)
def meta(key: str) -> str:
    if not ready():
        return ""
    row = con().execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return row[0] if row else ""


# ---------- 斷詞 ----------

def get_tagger():
    global _tagger
    if _tagger is None:
        import fugashi
        import unidic_lite
        # 一定要明確指定：fugashi.Tagger() 在裝了 unidic 完整版時會自動改用它，切點會不一樣
        _tagger = fugashi.Tagger('-d "%s"' % unidic_lite.DICDIR.replace("\\", "/"))
    return _tagger


@dataclass(frozen=True, slots=True)
class Tok:
    s: int
    e: int
    surface: str
    pos1: str
    pos2: str
    lemma: str | None       # 語彙素（為る、居る），對條目用；已去掉 -cake 這種尾巴
    orth_base: str | None   # 顯示用辭書形（する、いる）
    kana_base: str | None   # 辭書形讀音（平假名）；UniDic 的慣例讀音，私 會是 わたくし
    kana: str | None        # 這個字形實際的讀音（平假名），私 是 わたし
    accent: str | None
    pos3: str = ""          # 安定 是「サ変形状詞可能」、実際 是「副詞可能」


@functools.lru_cache(maxsize=20000)
def tokens(text: str) -> tuple:
    with _tag_lock:
        words = [(w.surface, w.feature) for w in get_tagger()(text)]
    out, cursor = [], 0
    for surface, f in words:
        s = text.find(surface, cursor) if surface else -1
        if s < 0:
            continue
        cursor = s + len(surface)
        lemma = f.lemma.split("-")[0] if f.lemma else None
        out.append(Tok(s, cursor, surface, f.pos1 or "", f.pos2 or "", lemma, f.orthBase,
                       jaconv.kata2hira(f.kanaBase) if f.kanaBase else None,
                       jaconv.kata2hira(f.kana) if f.kana else None,
                       f.aType if f.aType and f.aType != "*" else None, f.pos3 or ""))
    return tuple(out)


# ---------- 去活用（Yomitan 規則） ----------

@functools.lru_cache(maxsize=4)
def _rules(aux: bool = True):
    """aux：加上 TE_AUX 的補助動詞規則（切詞用）。Shift 即時查詢不看 token，加了會亂猜
    （初めてきました→初める＋ていく，てきます 和 てくる 撞形），所以 scan() 用 aux=False。"""
    data = json.loads(RULES_PATH.read_text(encoding="utf-8"))
    flags, bit, pending = {}, 0, list(data["conditions"].items())
    while pending:                      # 對應 Yomitan LanguageTransformer._getConditionFlagsMap
        rest = []
        for name, c in pending:
            subs = c["subConditions"]
            if not subs:
                flags[name] = 1 << bit
                bit += 1
            elif all(s in flags for s in subs):
                v = 0
                for s in subs:
                    v |= flags[s]
                flags[name] = v
            else:
                rest.append((name, c))
        if len(rest) == len(pending):
            raise RuntimeError("Yomitan 規則的條件互相參照，無法展開")
        pending = rest

    def fl(names):
        v = 0
        for n in names:
            v |= flags[n]
        return v

    transforms = dict(data["transforms"])
    for tid, (forms_, cls, _lemmas) in (TE_AUX.items() if aux else ()):
        transforms[tid] = {"rules": [{"pattern": f"{te}{f}$", "to": te, "in": [cls], "out": ["-て"]}
                                     for f in forms_ for te in ("て", "で")]}
    suffix, whole = defaultdict(list), defaultdict(list)
    for tid, t in transforms.items():
        for r in t["rules"]:
            pat = r["pattern"]
            body = pat[1:-1] if pat.startswith("^") else pat[:-1]
            if not pat.endswith("$") or META_CHARS & set(body):
                raise RuntimeError(f"Yomitan 規則出現真正的 regex，要改寫 deinflect：{pat}")
            item = (tid, r["to"], fl(r["in"]), fl(r["out"]))
            (whole if pat.startswith("^") else suffix)[body].append(item)
    return flags, dict(suffix), dict(whole), max(map(len, suffix))


@functools.lru_cache(maxsize=50_000)
def deinflect(text: str, aux: bool = True) -> tuple:
    """((候選原形, 條件旗標, 活用鏈), ...)，第一個是原字串；活用鏈由外而內，例如 ('-た','negative','potential or passive')
    aux=False 只用 Yomitan 原本的規則（見 _rules）。"""
    _, suffix, whole, max_suffix = _rules(aux)
    out, traces, i = [(text, 0, ())], [()], 0
    while i < len(out):
        (cur, cond, chain), trace = out[i], traces[i]
        i += 1
        if len(chain) > 8:
            continue
        hits = [(rule, cur[:len(cur) - k] + rule[1])
                for k in range(1, min(max_suffix, len(cur)) + 1) for rule in suffix.get(cur[-k:], ())]
        hits += [(rule, rule[1]) for rule in whole.get(cur, ())]
        for (tid, _to, cin, cout), new in hits:
            if (cond and not cond & cin) or (tid, cur) in trace:
                continue
            out.append((new, cout, chain + (tid,)))
            traces.append(trace + ((tid, cur),))
    return tuple(out)


def forms() -> frozenset:
    global _forms
    if _forms is None:
        with _forms_lock:
            if _forms is None:
                _forms = frozenset(r[0] for r in con().execute("SELECT DISTINCT text FROM form"))
    return _forms


@functools.lru_cache(maxsize=200_000)
def entries_for(form: str) -> tuple:
    full, fs = form.translate(HALF2FULL), forms()
    if form not in fs and full not in fs:
        return ()
    return tuple(con().execute("SELECT entry_id, MIN(rank) FROM form WHERE text IN (?, ?) GROUP BY entry_id",
                               (form, full)))


Info = namedtuple("Info", "pos_flags kanji kana data")


def _pos_flags(pos_list) -> int:
    flags = _rules()[0]
    v = 0
    for p in pos_list:
        if p == "v1" or p.startswith("v1-"):
            v |= flags["v1"]
        elif p.startswith("v5"):
            v |= flags["v5"]
        elif p == "vk":
            v |= flags["vk"]
        elif p.startswith("vs"):
            v |= flags["vs"]
        elif p == "vz":
            v |= flags["vz"]
        elif p in ("adj-i", "adj-ix"):
            v |= flags["adj-i"]
    return v


@functools.lru_cache(maxsize=50_000)
def entry_info(eid: int) -> Info:
    row = con().execute("SELECT data FROM entry WHERE id = ?", (eid,)).fetchone()
    d = json.loads(row[0]) if row else {"k": [], "r": [], "s": []}
    flags = _pos_flags(p for s in d["s"] for p in s["pos"])
    if eid in EXPRESSIONS and any(r["t"].endswith("ない") for r in d["r"]):
        flags |= _rules()[0]["adj-i"]      # かもしれない、なきゃいけない 在 JMdict 只標 exp，照樣會 かもしれなくて 這樣活用
    return Info(flags, frozenset(k["t"] for k in d["k"]), frozenset(r["t"] for r in d["r"]), d)


def candidates(text: str, start: int, lengths, keep=None, aux: bool = True) -> list:
    """由長到短試 lengths，某個長度查到就停。回傳 [(長度, 原形, 條目, 常用度, 活用鏈), ...]
    keep(候選) 回傳 False 的不算查到（這個長度全被擋掉就繼續試短的）。aux 見 _rules。"""
    for L in sorted(set(lengths), reverse=True):
        found = []
        for base, cond, chain in deinflect(text[start:start + L], aux):
            for eid, rank in entries_for(base):
                if cond and not entry_info(eid).pos_flags & cond:
                    continue                    # 去活用後詞性要吻合：食べ→食べる 只收一段動詞
                c = (L, base, eid, rank, chain)
                if keep is None or keep(c):
                    found.append(c)
        if found:
            return found
    return []


_KANA_ONLY = re.compile(r"^[ぁ-ゖァ-ヺー・]+$")


def usually_kana(eid: int) -> bool:
    """平常寫假名的詞：沒有正常的漢字寫法（只剩 sK、rK 這種），或義項標了 uk。"""
    d = entry_info(eid).data
    ks = [k for k in d["k"] if not (_BAD_K & set(k.get("inf") or []))]
    return not ks or any("uk" in (s.get("misc") or []) for s in d["s"][:3])


def kana_mismatch(surface: str, eid: int) -> bool:
    """句子裡寫假名，條目卻是平常寫漢字的詞。"""
    return bool(_KANA_ONLY.match(surface)) and not usually_kana(eid)


def lemma_matches(lemma: str | None, info: Info) -> bool:
    if not lemma:
        return False
    return lemma in info.kana or lemma in info.kanji or any(k.startswith(lemma) for k in info.kanji)


def scan(text: str, pos: int, max_len: int = MAX_LEN, limit: int = 3) -> dict:
    """給 Shift 即時查詢用：不看 token 邊界，Yomitan 原本的做法（不加 TE_AUX 規則）。"""
    cands = candidates(text, pos, range(1, min(max_len, len(text) - pos) + 1), aux=False)
    cands.sort(key=lambda c: (-c[0], kana_mismatch(text[pos:pos + c[0]], c[2]), len(c[4]), c[3]))
    seen, out = set(), []
    for L, base, eid, rank, chain in cands:
        if eid in seen:
            continue
        seen.add(eid)
        out.append({"key": f"jm:{eid}", "len": L, "base": base, "chain": list(chain), "entry_id": eid})
        if len(out) >= limit:
            break
    return {"len": out[0]["len"] if out else 0, "cands": out}


def lookup_base(orth_base, lemma, kana_base):
    for form in (orth_base, lemma):
        ids = [e for e, _ in entries_for(form)] if form else []
        if ids:
            return next((e for e in ids if kana_base in entry_info(e).kana), ids[0])
    return None


# ---------- 條目資訊 ----------

_BAD_K = {"rK", "sK", "iK", "oK", "ik"}
_BAD_R = {"sk", "ok", "ik", "gikun"}


def _pri_rank(pris) -> int:
    r = 999
    for p in pris or []:
        if p.startswith("nf"):
            r = min(r, int(p[2:]))
        elif p in ("ichi1", "news1", "spec1", "gai1"):
            r = min(r, 50)
        elif p in ("ichi2", "news2", "spec2", "gai2"):
            r = min(r, 80)
    return r


@functools.lru_cache(maxsize=50_000)
def headword(eid: int) -> tuple[str, str | None]:
    """(顯示用寫法, 讀音)。通常寫假名（uk）的詞用假名；同類寫法有好幾個時挑最常用的（皆→みんな）。"""
    d = entry_info(eid).data
    ks = [k for k in d["k"] if not (_BAD_K & set(k.get("inf") or []))]   # 只剩 sK/rK 寫法的詞當成假名詞（の、乃）
    ks = [k["t"] for k in sorted(ks, key=lambda k: _pri_rank(k.get("pri")))]
    rs = [r for r in d["r"] if not (_BAD_R & set(r.get("inf") or []))] or d["r"]
    rs = sorted(rs, key=lambda r: _pri_rank(r.get("pri")))
    first_misc = set(d["s"][0].get("misc") or []) if d["s"] else set()
    if not ks or "uk" in first_misc:
        word = rs[0]["t"] if rs else (ks[0] if ks else "")
        return word, None
    word = ks[0]
    reading = next((r["t"] for r in rs if not r.get("nokanji") and (not r.get("restr") or word in r["restr"])),
                   rs[0]["t"] if rs else None)
    return word, reading


@functools.lru_cache(maxsize=50_000)
def jlpt_level(eid: int) -> int:
    row = con().execute("SELECT level FROM jlpt WHERE entry_id = ?", (eid,)).fetchone()
    return row[0] if row else 0


def best_rank(eid: int) -> int:
    row = con().execute("SELECT MIN(rank) FROM form WHERE entry_id = ?", (eid,)).fetchone()
    return row[0] if row and row[0] is not None else 999


def accent(term: str, reading: str | None) -> str:
    row = con().execute("SELECT pattern FROM accent WHERE term = ? AND reading = ?", (term, reading or term)).fetchone()
    return row[0] if row else ""


def sense_src(sense: dict) -> str:
    """一個義項的英文原文，也是中文翻譯快取（sense_zh）的主鍵。"""
    return "; ".join(sense.get("gloss") or [])


@functools.lru_cache(maxsize=50_000)
def summary(eid: int) -> dict:
    """詞表和小卡用的條目摘要（不含中文翻譯，中文由 vocab 從快取補）。"""
    info = entry_info(eid)
    word, reading = headword(eid)
    misc = []
    for s in info.data["s"][:2]:
        for m in s.get("misc") or []:
            if m not in misc:
                misc.append(m)
    return {
        "d": word, "r": reading, "lv": jlpt_level(eid), "rank": best_rank(eid), "misc": misc,
        "senses": [sense_src(s) for s in info.data["s"]],
    }


def reading_for(eid, base, kana_base):
    info = entry_info(eid)
    if base in info.kana:
        return None
    if kana_base in info.kana:
        return kana_base
    return info.data["r"][0]["t"] if info.data["r"] else None


# ---------- 切出可查的詞單位 ----------

def _group(t: Tok) -> str:
    if t.lemma is None:
        return "unknown"          # 多半是外文或辨識錯字
    if t.pos2 == "固有名詞":
        return "name"
    if t.pos1 == "感動詞":
        return "filler"
    if t.pos2 == "数詞":
        return "number"
    if t.pos1 in ("接頭辞", "接尾辞"):
        return "affix"
    return "word"


_HIRA_ONLY = re.compile(r"^[ぁ-ゖー]+$")


def _span(toks, n: int, end: int) -> tuple:
    k = n
    while k + 1 < len(toks) and toks[k + 1].s < end:
        k += 1
    return toks[n:k + 1]


def _entry_pos(eid: int) -> set:
    return {p for s in entry_info(eid).data["s"] for p in s["pos"]}


def _false_merge(text: str, toks, n: int, c) -> bool:
    """最長比對跨過 UniDic 的切點、併成意思不相干的條目（片庫實測的假合併）：這個候選不算，改試短的。"""
    L, _base, eid, _rank, chain = c
    t = toks[n]
    span = _span(toks, n, t.s + L)
    # 0. 補助動詞的去活用（TE_AUX）要 UniDic 也切成 動詞＋て＋補助動詞，
    #    而且主動詞就是 UniDic 切出來的那個動詞（どこで｜だっ｜て｜いける 的 だっ 是 UniDic 切錯，不是 だる＋ていく）
    if any(x in TE_AUX for x in chain):
        if any(x in TE_AUX and not _aux_structure(span, TE_AUX[x][2]) for x in chain):
            return True
        if t.pos1 == "動詞" and not _head_verb_matches(t, toks[n - 1] if n else None, eid):
            return True
    if len(span) < 2:
        return False
    surface = text[t.s:t.s + L]
    # 1. 整段寫平假名、跨好幾個 token、條目平常寫漢字：さん｜と→三都、いい｜ん→委員、よく｜そう→浴槽、なる｜と→鳴門
    #    有活用的只有 UniDic 也說開頭是動詞、形容詞才算（わかん｜ない 照舊，よく｜そう 不是 浴す 的意志形）
    #    接頭、接尾辞組成的（みな｜さん、やつ｜ら）和片假名外來語（グーグル｜マップ）照舊
    #    UniDic 各 token 的語彙素接起來就是條目的漢字寫法時是同一個詞（その｜あと→其の後、あと｜で→後で）
    #    裡面有 UniDic 認不出來的 token（め｜かく｜し 的 め 是記号）時切點本身不可靠，照舊合併（めかくし→目隠し）
    if (_HIRA_ONLY.match(surface) and span[0].pos1 != "接頭辞" and span[-1].pos1 != "接尾辞"
            and not any(x.pos1 == "記号" for x in span)
            and (not chain or span[0].pos1 not in ("動詞", "形容詞", "接尾辞")) and not usually_kana(eid)
            and "".join(x.surface if x.pos1 in SKIP_POS1 else (x.lemma or x.surface) for x in span)
            not in entry_info(eid).kanji):
        return True
    if chain:
        # 1b. 有活用、整段平假名，後面卻接到名詞、副詞這類另一個詞，開頭的語彙素又對不上條目：
        #     ある｜こう→歩こう、いい｜そう→言いそう、それ｜そう→逸れる（やって｜きました、しよう｜と｜する 後面是動詞，照舊）
        return bool(_HIRA_ONLY.match(surface) and not lemma_matches(span[0].lemma, entry_info(eid))
                    and any(x.pos1 not in ("助詞", "助動詞", "接尾辞", "動詞", "形容詞") for x in span[1:]))
    pos = _entry_pos(eid)
    has_prt = any(x.pos1 == "助詞" for x in span[1:])
    # 2. UniDic 說是動詞＋助詞，條目卻只是接續詞、助詞：説明｜する｜と→すると（句首的 すると UniDic 本來就是一個 token）
    #    UniDic 自己都讀不通的動詞不算（どこで｜だっ｜て 的 だっ 讀音對不上 立つ，だって 是助詞）
    if span[0].pos1 == "動詞" and has_prt and pos <= {"conj", "prt", "int"} and not _guessed_verb(span[0]):
        return True
    # 3. 前面有修飾語的形式名詞＋助詞，併成接續詞、副詞：あなたの｜ところ｜で→ところで、しゃべる｜とき｜に→時に
    blocked = FORMAL_NOUNS.get(span[0].lemma) if span[0].pos1 == "名詞" else None
    if (blocked and has_prt and pos <= blocked and n > 0
            and (toks[n - 1].pos1 in ("動詞", "形容詞", "助動詞", "連体詞") or toks[n - 1].surface == "の")):
        return True
    # 4. 寫漢字、UniDic 切成 今日｜は 的 今日は 幾乎都是「今天」：辨識出來的招呼語會寫假名 こんにちは。
    #    只有後面接另一句招呼（今日はこんばんは）或前面是 皆さん（皆さん今日は）時才當打招呼；
    #    後面接 まあ、あの 這種填充詞（今日はまあいろんな）、或這一行就停在 今日は（ということで今日は｜…）都是「今天」
    if eid in GREETINGS and not _greeting_context(toks, n, t.s + L):
        return True
    # 5. これ｜これ 是講了兩次，不是「如此這般」
    if len(span) == 2 and span[0].surface == span[1].surface and span[0].pos1 == "代名詞":
        return True
    return False


def _greeting_context(toks, n: int, end: int) -> bool:
    after = next((x for x in toks if x.s >= end and x.pos1 not in ("補助記号", "空白")), None)
    if after is not None:
        return after.pos1 == "感動詞" and after.pos2 != "フィラー"
    prev = next((x for x in reversed(toks[:n]) if x.pos1 not in ("補助記号", "空白")), None)
    return prev is not None and (prev.lemma in ("皆", "皆さん") or (prev.pos1 == "接尾辞" and prev.lemma in ("さん", "様")))


def _head_verb_matches(t: Tok, prev: Tok | None, eid: int) -> bool:
    """併了補助動詞的條目要是開頭那個動詞 token 本身：語彙素或辭書形相同（撮っ 的語彙素是 取る、辭書形是 撮る），
    或條目寫法以這個 token 開頭（話し｜かけて→話しかける）。"""
    info = entry_info(eid)
    forms_ = info.kanji | info.kana
    return (lemma_matches(_lemma(t, prev), info) or t.orth_base in forms_
            or any(f.startswith(t.surface) for f in forms_))


def _guessed_verb(t: Tok) -> bool:
    """UniDic 硬把不認得的字串當成動詞：讀音跟辭書形對不起來（だっ 被當成 立つ，讀 たつ）。"""
    return (t.pos1 == "動詞" and bool(t.kana) and bool(t.kana_base) and t.lemma not in ("為る", "来る")
            and not t.kana_base.startswith(t.kana[:max(1, len(t.kana) - 1)]))


def _aux_structure(span, lemmas: set) -> bool:
    for i, x in enumerate(span):
        if "てく" in lemmas and x.pos1 == "助動詞" and x.surface in ("てく", "でく"):   # UniDic 把 過ぎてく 的 てく 當一個助動詞
            if i > 0 and span[i - 1].pos1 in ("動詞", "助動詞"):
                return True
        if (x.pos1 == "動詞" and x.lemma in lemmas and i >= 2 and span[i - 1].pos1 == "助詞"
                and span[i - 1].surface in ("て", "で") and span[i - 2].pos1 in ("動詞", "助動詞")):
            return True
    return False


def _pos_mismatch(toks, n: int, L: int, eid: int) -> int:
    """整個詞就是一個 token、UniDic 說是副詞／感動詞／接續詞／連體詞時，條目沒有對應詞性（0 吻合，越大越不吻合）。
    UniDic 常把句首的感嘆詞 ああ 標成副詞（ああなるほど、ああ一番）：副詞後面沒接動詞、助動詞時，
    感嘆詞條目排在只有副詞的條目前面（嗚呼 不是「那樣」的 ああ；然う 本身兩種詞性都有，不受影響）。"""
    t = toks[n]
    want = POS_JM.get(t.pos1)
    if not want or L != t.e - t.s:
        return 0
    pos = _entry_pos(eid)
    if t.pos1 == "副詞" and t.surface == "そう" and n and toks[n - 1].e == t.s and (
            (toks[n - 1].pos1 == "接尾辞" and toks[n - 1].surface == "さ") or "形状詞可能" in toks[n - 1].pos3):
        # 緊接在 よさ、なさ、不安定 這種後面的 そう 是樣態（UniDic 標成副詞是標錯；実際そうでした 的 実際 是副詞可能，照舊）
        return 0 if "aux" in pos else 2
    if t.pos1 == "副詞" and not _before_predicate(toks, n):
        return 0 if "int" in pos else (1 if pos & set(want) else 2)
    return 0 if pos & set(want) else 2


def _before_predicate(toks, n: int) -> bool:
    """下一個 token 是動詞、助動詞（ああ｜なる｜ほど 的 なる 是 なるほど 的一部分，不算）。"""
    nxt = toks[n + 1] if n + 1 < len(toks) else None
    if nxt is None or nxt.pos1 not in ("動詞", "助動詞"):
        return False
    return not (nxt.surface == "なる" and n + 2 < len(toks) and toks[n + 2].surface == "ほど")


def _lemma(t: Tok, prev: Tok | None) -> str | None:
    """UniDic 把口語的 行った、行って 常標成 行う（一人で行った、何回も行った）；前面不是 を 就當 行く。"""
    if t.lemma == "行う" and t.surface.startswith("行っ") and not (prev is not None and prev.surface == "を"):
        return "行く"
    return t.lemma


def _order(text: str, toks, n: int):
    t, prev = toks[n], (toks[n - 1] if n else None)
    lemma = _lemma(t, prev)

    def key(c):
        L, base, eid, rank, ch = c
        info = entry_info(eid)
        # 同長度的排序：
        # 0. JMdict 本身就有的說法優先於「動詞＋補助動詞」（持っていく、やってみる 是條目）
        # 1. UniDic 語彙素是條目的寫法（行った→行く 不是 行う，という→言う 不是 いい，いる→居る 不是 射る）
        # 2. 句子裡寫假名時，優先挑平常就寫假名的條目（なんか→何か 不是 軟化，そう→然う 不是 僧）
        # 3. 字串本身是條目時，不拆成口語縮約（めっちゃ 不是 めく＋ちゃ）
        # 4. 常用詞優先（UniDic 把 私 一律讀 わたくし，不能只看讀音）
        # 5. 讀音吻合（方：ほう／かた）> UniDic 詞性吻合（副詞 そう）> 不是關西腔變化 > 活用鏈短 > 常用度
        return (-L, sum(x in TE_AUX for x in ch), not lemma_matches(lemma, info),
                kana_mismatch(text[t.s:t.s + L], eid), bool(SLANG_CHAIN.intersection(ch)),
                rank > 50, t.kana not in info.kana, t.kana_base not in info.kana, _pos_mismatch(toks, n, L, eid),
                any("kansai" in x for x in ch), len(ch), rank)
    return key


@functools.lru_cache(maxsize=1)
def _expr_heads() -> tuple:
    return tuple(sorted({f[:2] for eid in EXPRESSIONS for f in entry_info(eid).kanji | entry_info(eid).kana}))


def _expression(text: str, toks, n: int, ends) -> tuple | None:
    """從第 n 個 token（助詞、助動詞）開始的白名單說法，回傳候選 (長度, 原形, 條目, 常用度, 活用鏈)。"""
    t = toks[n]
    if not text.startswith(_expr_heads(), t.s):
        return None
    prev = toks[n - 1] if n > 0 else None

    def keep(c):
        L, _base, eid, _rank, chain = c
        rule = EXPRESSIONS.get(eid)
        if rule is None:
            return False
        if rule.no_iu_kanji and (chain or "言" in text[t.s:t.s + L]):
            return False                        # と言います、って言って、といって 是動詞 言う
        if rule.kana_only and not _KANA_ONLY.match(text[t.s:t.s + L]):
            return False
        return not rule.prev_pos or (prev is not None and prev.pos1 in rule.prev_pos)

    cands = candidates(text, t.s, [e - t.s for e in ends if 1 < e - t.s <= MAX_LEN], keep)
    return min(cands, key=lambda c: (len(c[4]), c[3])) if cands else None


def _inner_expression(text: str, toks, n: int, end: int, ends) -> int | None:
    """實詞範圍 [toks[n].s, end) 裡面有助詞、助動詞開頭的說法、而且說法超出這個範圍時，回傳實詞要讓出的位置。
    例如 しなきゃ｜いけない → し｜なきゃいけない。"""
    for m in range(n + 1, len(toks)):
        x = toks[m]
        if x.s >= end:
            break
        if x.pos1 in SKIP_POS1:
            e = _expression(text, toks, m, ends)
            if e and x.s + e[0] > end:
                return x.s
    return None


def _note(meta_out: dict, key: str, display, reading, grp: str, eid, pos: str):
    m = meta_out.setdefault(key, {"lang": "ja", "display": display, "reading": reading, "grp": grp,
                                  "entry_id": eid, "pos": pos, "phrase": 0})
    votes = m.setdefault("votes", {})                    # 分組看多數用法，由 vocab.annotate 最後決定
    votes[grp] = votes.get(grp, 0) + 1


def _segment(text: str, toks, meta_out: dict, p: int = 0, until: int | None = None) -> list:
    """until：只切到開頭在這個位置之前的詞單位（接上一行時只需要跨過行尾的那一個）。"""
    ends = sorted({t.e for t in toks})
    units = []
    for n, t in enumerate(toks):
        if until is not None and t.s >= until:
            break
        if t.s < p or not t.surface.strip() or _DIGITS.match(t.surface):
            continue
        chain = ()
        if t.pos1 in SKIP_POS1:
            found = _expression(text, toks, n, ends)
            if not found:
                continue                                     # 一般的助詞、助動詞不包 span、不算單字
            L, base, eid, _, chain = found
            if eid in EXPR_ALIAS:
                eid, extra = EXPR_ALIAS[eid]
                chain = tuple(chain) + extra
            cands = [found]
        else:
            lengths = [e - t.s for e in ends if 0 < e - t.s <= MAX_LEN]

            def keep(c, n=n):
                return not _false_merge(text, toks, n, c)

            cands = candidates(text, t.s, lengths, keep)
            stop = len(text)
            if cands:
                cut = _inner_expression(text, toks, n, t.s + cands[0][0], ends)
                if cut is not None:
                    stop = cut
                    cands = candidates(text, t.s, [x for x in lengths if x <= cut - t.s], keep)
                    # 讓出去之後剩下的常是動詞未然形（こなさ、やら），查到的條目要跟 UniDic 語彙素一致，不然改用語彙素查
                    if t.pos1 == "動詞":
                        cands = [c for c in cands if lemma_matches(t.lemma, entry_info(c[2]))]
            if cands:
                L, base, eid, _, chain = min(cands, key=_order(text, toks, n))
        if cands:
            end = t.s + L
            info = entry_info(eid)
            display, reading = headword(eid)
            if reading is None and base != display and base in info.kana:
                display = base                               # 平常寫假名的詞，照句子裡的寫法（みんな 不顯示成 みな）
        else:
            if t.pos1 == "記号":
                continue
            end, k = t.e, n
            while (t.pos1 in GLUE_HEAD and k + 1 < len(toks) and toks[k + 1].s == end and toks[k + 1].e <= stop
                   and (toks[k + 1].pos1 == "助動詞" or toks[k + 1].surface in ("て", "で", "ば"))):
                k += 1
                end = toks[k].e                     # 言っ+たら 併成一個滑鼠範圍
            eid = lookup_base(t.orth_base, t.lemma, t.kana_base)   # 第二次機會：イっ→行く
            if eid:
                display, reading = headword(eid)
            else:
                display = t.orth_base or t.surface
                reading = t.kana_base if t.kana_base and t.kana_base != display else None
        key = f"jm:{eid}" if eid else f"ja:{t.lemma or t.surface}|{t.pos1}"
        # 跨好幾個 token 的詞（お客様、国家公務員）：裡面只要有一般詞就算 word，不然照第一個 token 分組
        span_groups = [_group(x) for x in toks[n:] if x.s < end and x.pos1 not in SKIP_POS1]
        grp = "word" if "word" in span_groups else _group(t)
        _note(meta_out, key, display, reading, grp, eid, t.pos1)
        units.append([t.s, end, key, list(chain)] if chain else [t.s, end, key])
        p = end
    return units


_CONT_POS = ("動詞", "助動詞", "助詞")


def _continues_line(x: Tok) -> bool:
    """這一行開頭可以接上一行的 token：動詞、助動詞、助詞，和 ない、いい 這類非自立的形容詞。"""
    return x.pos1 in _CONT_POS or (x.pos1 == "形容詞" and x.pos2 == "非自立可能")


@functools.lru_cache(maxsize=4096)
def _units_alone(text: str) -> tuple:
    """一行自己切出來的詞單位 (起, 迄, key)，接下一行時判斷上一行算過哪些詞用。"""
    return tuple((u[0], u[1], u[2]) for u in _segment(text, tokens(text), {}))


def _crossing(prev: str, text: str, toks, pre: tuple, start: int, continues: bool, prev_units: list | None):
    """把上一行＋這一行接起來、從上一行的 start 開始切，找跨過行尾的詞單位並檢查能不能接。
    整行接起來給 MeCab（なきゃ｜なって｜いう 接起來才會切成 な｜って｜いう），但只切 start 之後、行尾之前開頭的詞。
    回傳 (這一行的詞單位, meta) 或 None。prev_units 見 segment。"""
    joined, cut = prev + text, len(prev)
    jt = tokens.__wrapped__(joined)                  # 接起來的字串每次都不同，不放進 tokens 的快取
    if not any(x.s == cut for x in jt):
        return None
    scratch: dict = {}
    units = [u for u in _segment(joined, jt, scratch, start, until=cut) if u[1] > cut]
    if not units:
        return None
    u = units[0]
    end = u[1] - cut
    if end not in {x.e for x in toks} or not all(_continues_line(x) for x in toks if x.s < end):
        return None                                  # 這一行被接走的部分只能是接在後面的東西（そう本当に 的 そう 不接）
    key, chain = u[2], (u[3] if len(u) > 3 else [])
    if any(x[0] == u[0] and x[2] == key for x in pre):
        # 上一行已經算過同一個詞（見せて｜いた 的 見せる）：尾巴標 1 不重複計數；只在句子明顯沒講完時接
        return ([0, end, key, chain, 1], None) if continues else None
    # 其他從上一行實詞開始的說法（上に｜立って→上に立つ、とって｜も→とても）容易接錯，
    # 只收助詞開頭的白名單說法（と｜いう）和 そう｜いう 這類；這兩種不管 brk 都接，因為 いう、しれない 不會自己開頭一句話
    unit = ([0, end, key, chain] if chain else [0, end, key]), scratch[key]
    head = next((x for x in jt if x.s == u[0]), None)
    if head is not None and head.pos1 in SKIP_POS1:
        return unit
    if head is None or head.surface not in DEMONSTRATIVES or text[:1] not in ("い", "言"):
        return None
    # そう｜いう：上一行的 そう 本來是自己一個詞單位（然う），要改成同一個 key、標 1 不計數，不然同一個 そういう
    # 會算成 然う＋そういう 兩個詞。只在拿得到上一行詞單位（prev_units）、而且上一行最後一個詞剛好就是 そう 時接
    old = next((x for x in prev_units or () if x[0] == u[0] and x[1] == cut), None)
    if old is None:
        return None
    old[2:] = [key, list(chain), 1]
    return unit


def _carry_over(prev: str, text: str, toks, meta_out: dict, continues: bool, prev_units: list | None) -> list | None:
    """上一行話沒講完就被切斷時，這一行開頭接著上一行的詞：と｜いう方が、かも｜しれない、見せて｜いた、そう｜いう。
    把兩行接起來切，找跨過行尾的那個詞單位，回傳它在這一行的部分；上一行自己的詞單位只有 そう｜いう 會改（見 _crossing）。
    continues：上一行的 brk 是 hard、soft（句子明顯沒講完）。不是的話（停頓 gap、辨識斷句 eos）只接助詞開頭的說法
    和 そう｜いう，因為「いう」「しれない」不會自己開始一句話，但 見せて｜いた 這種有可能是兩句。"""
    if not toks or not _continues_line(toks[0]):
        return None
    ptoks = tokens(prev)
    if not ptoks:
        return None
    last = ptoks[-1]
    if last.e != len(prev) or not (last.pos1 in SKIP_POS1 or last.surface in DEMONSTRATIVES):
        return None
    pre = _units_alone(prev)
    k = len(ptoks)
    while k > 0 and ptoks[k - 1].pos1 in SKIP_POS1:
        k -= 1
    run = ptoks[k].s if k < len(ptoks) else len(prev)
    # 先從上一行最後一個詞開始接（見せて｜いた、そう｜いう）；接不上再只從結尾那串助詞開始（どちらか｜と｜いう）
    for start in dict.fromkeys((min(pre[-1][0] if pre else len(prev), run), run)):
        if start >= len(prev) or len(prev) - start > MAX_LEN:
            continue
        found = _crossing(prev, text, toks, pre, start, continues, prev_units)
        if found:
            unit, m = found
            if m is not None:
                grp = max(m["votes"], key=m["votes"].get)
                _note(meta_out, unit[2], m["display"], m["reading"], grp, m["entry_id"], m["pos"])
            return unit
    return None


def segment(text: str, meta_out: dict, prev: str | None = None, prev_brk: str | None = None,
            prev_units: list | None = None) -> list:
    """一句日文 → 詞單位。meta_out[key] 收集顯示用的原形、讀音、分組。
    prev、prev_brk：上一行的文字和它的 brk（見 translate.build_units），用來接起被斷行切開的說法。
    prev_units：上一行已經切好的詞單位（cue["w"]）；接 そう｜いう 時會就地改掉上一行的 そう，沒給就不接這種。"""
    toks = tokens(text)
    head = _carry_over(prev, text, toks, meta_out, prev_brk in CONTINUE_BREAKS, prev_units) if prev else None
    if not head:
        return _segment(text, toks, meta_out)
    return [head] + _segment(text, toks, meta_out, head[1])


def warm():
    """伺服器啟動時在背景先載入，第一次打開影片不用多等。"""
    if ready():
        get_tagger()
        forms()
        _rules()
