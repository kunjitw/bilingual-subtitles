"""日文查字的黃金案例：python -s tests/test_dict_ja.py

需要先建好 data/dict/jmdict.db（tools/build_dict_ja.py）。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import dict_ja  # noqa: E402


def unit_for(text: str, surface: str):
    meta = {}
    units = dict_ja.segment(text, meta)
    s = text.index(surface)
    for u in units:
        if u[0] == s:
            return u, meta[u[2]], units
    raise AssertionError(f"{surface} 沒有對到詞單位：{units}")


def test_inflection_chain():
    u, m, _ = unit_for("昨日はケーキを食べられなかった。", "食べられなかった")
    assert u[1] - u[0] == len("食べられなかった"), u
    assert u[2] == "jm:1358280", u
    assert u[3] == ["-た", "negative", "potential or passive"], u
    assert m["display"] == "食べる"


def test_expressions_and_ambiguous():
    cases = [
        ("国家公務員の彼は、つい手を出してしまった。", "手を出してしまった", "jm:1896030"),
        ("友達と一緒に京都へ行った。", "行った", "jm:1578850"),
        ("寝落ちしちゃってもいいよ", "寝落ち", "jm:2816820"),
        ("ASMRでドＳなお姉さん", "ドＳ", "jm:2646770"),
        ("国家公務員の彼は", "国家公務員", "jm:1657290"),
        ("そう言ったら怒られる", "言ったら", "jm:1587040"),
    ]
    for text, surface, key in cases:
        u, _, units = unit_for(text, surface)
        assert u[2] == key, (surface, u, units)


def test_base_forms():
    _, m, _ = unit_for("先生に怒られる", "怒られる")
    assert m["display"] == "怒る", m
    _, m, _ = unit_for("先生がおっしゃいます", "おっしゃいます")
    assert m["display"] == "仰る" or m["reading"] in (None, "おっしゃる") or m["display"] == "おっしゃる", m


def test_kana_homophones():
    # 句子裡寫假名的詞不要對到平常寫漢字的同音詞
    for text, surface, bad in [("なんかすごいね", "なんか", "軟化"), ("そうなんですよ", "そう", "僧"),
                               ("先生がいう通りです", "いう", "いい")]:
        _, m, units = unit_for(text, surface)
        assert m["display"] != bad, (surface, m, units)
    _, m, _ = unit_for("みんなで行こう", "みんな")
    assert m["display"] == "みんな", m
    _, m, _ = unit_for("私は学生です", "私")
    assert m["reading"] == "わたし", m


def test_particles_not_wrapped():
    meta = {}
    units = dict_ja.segment("私は東京に行きます。", meta)
    text = "私は東京に行きます。"
    surfaces = [text[u[0]:u[1]] for u in units]
    assert "は" not in surfaces and "に" not in surfaces and "。" not in surfaces, surfaces


def test_scan():
    res = dict_ja.scan("食べられなかった", 0)
    assert res["len"] == 8 and res["cands"], res
    # 寫假名的の：助詞條目（漢字只有 sK 寫法）要排在 野 前面，顯示成假名
    res = dict_ja.scan("チヤクシの観光", 4)
    assert res["cands"][0]["key"] == "jm:1469800", res
    assert dict_ja.headword(1469800) == ("の", None)


# ---------- SEGMENT_RULES 8：說法、補助動詞、假合併、跨行（例句取自擁有者回報和片庫） ----------

IU, TOIU, TTEIU = "jm:1587040", "jm:1922760", "jm:2757880"


def units_of(text: str, prev: str | None = None, prev_brk: str | None = None) -> list:
    """[(字面, key, 活用鏈…, 不計數旗標…), ...]"""
    units = dict_ja.segment(text, {}, prev=prev, prev_brk=prev_brk)
    return [(text[u[0]:u[1]], u[2]) + tuple(u[3:]) for u in units]


def pairs_of(text: str, **kw) -> list:
    return [u[:2] for u in units_of(text, **kw)]


def test_to_iu_is_its_own_word():
    # 擁有者回報「言う」的例句：と／って＋いう 是自己的詞，不再算成 言う
    for text, surface, key in [
        ("大原ヘンリーさんという方が書いたですね", "という", TOIU),
        ("こんなことやっていていいのかというね", "という", TOIU),
        ("ちょっと耳受け付けませんねっていう方も", "っていう", TTEIU),
        ("ということで今回は一人暮らしをしたことが", "という", TOIU),
        ("これはやめとけっていう物件の条件を", "っていう", TTEIU),
        ("ちゃんと適正というか違和感のない価格で", "というか", "jm:2206220"),
    ]:
        pairs = pairs_of(text)
        assert (surface, key) in pairs and all(k != IU for _, k in pairs), (text, pairs)
    # さん｜と 不能併成 三都
    assert ("さん", "jm:1005340") in pairs_of("大原ヘンリーさんという方が書いたですね")
    # 真的在說：寫漢字或有活用的照舊是 言う
    for text, surface in [("そう言ったら怒られる", "言ったら"), ("ありがとうといって帰った", "いって")]:
        assert (surface, IU) in pairs_of(text), (text, pairs_of(text))


def test_carry_over_line_breaks():
    # と、って 在上一行（擁有者的例子：…っていう方が多いのではないでしょうか 被切成兩行）
    first = units_of("いう方が多いのではないでしょうか", prev="そろそろ新生活の準備を始めなきゃなって", prev_brk="hard")[0]
    assert first == ("いう", TTEIU), first
    assert units_of("いうことで", prev="と", prev_brk="gap")[0] == ("いう", TOIU)
    assert units_of("いう方が多いのではないでしょうか")[0] == ("いう", IU)       # 沒有上一行就照舊
    first = units_of("しれない窓ガラス叩き割られて", prev="空き室に入られるかも", prev_brk="hard")[0]
    assert first == ("しれない", "jm:1002970"), first
    # 見せて｜いた：算在 見せる 上、標 1 不重複計數；停頓（gap）時可能是兩句，不接
    first = units_of("いた飲食店エリアには", prev="多くの人で賑わいを見せて", prev_brk="hard")[0]
    assert first[:2] == ("いた", "jm:1259210") and first[-1] == 1, first
    first = units_of("いた飲食店エリアには", prev="多くの人で賑わいを見せて", prev_brk="gap")[0]
    assert first[1] != "jm:1259210" and len(first) < 5, first


def test_te_form_auxiliaries():
    for text, surface, key, aux in [
        ("動画を撮っていきたいと思っています", "撮っていきたい", "jm:1298790", "-いく"),
        ("いよいよ市場らしい雰囲気になってきました", "なってきました", "jm:1375610", "-くる"),
        ("見てみてください", "見てみて", "jm:1259290", "-みる"),
        ("こういう絵馬に書いてある願いを", "書いてある", "jm:1343950", "-ある"),
        ("友達が教えてくれた", "教えてくれた", "jm:1236900", "-くれる"),
    ]:
        u = next((u for u in units_of(text) if u[0] == surface), None)
        assert u and u[1] == key and aux in u[2], (text, units_of(text))
    assert pairs_of("持っていきます") == [("持っていきます", "jm:1315700")]      # JMdict 本身有的說法優先
    assert pairs_of("やってみたら") == [("やってみたら", "jm:2059360")]
    assert ("来ました", "jm:1547720") in pairs_of("今日初めて来ましたが")        # 初めて 是副詞，不是 初める
    assert ("見て", "jm:1259290") in pairs_of("実際に今回訪れて見ても")           # 寫漢字的 見る 是真的在看


def test_grammar_expressions():
    for text, surface, key in [
        ("分かりづらいところもあったかもしれないんですけど", "かもしれない", "jm:1002970"),
        ("それが今一番辛いところなのかもしれません", "かもしれません", "jm:1002970"),   # 禮貌形併到基本形
        ("好きでさやに似合うかもしれなくて", "かもしれなくて", "jm:1002970"),
        ("長かったんじゃない?", "じゃない", "jm:2755350"),
        ("スーパーではないんだけど", "ではない", "jm:2823770"),
        ("バイノーラルマイクを使って提出しなきゃいけない", "なきゃいけない", "jm:2831357"),
        ("ま僕からすると中国の人って", "からすると", "jm:2028250"),
        ("クロモン市場との比較対象として", "として", "jm:1008590"),
        ("対応によってヤエスの地での店舗再開となりました", "によって", "jm:1009640"),
    ]:
        assert (surface, key) in pairs_of(text), (text, pairs_of(text))
    assert all(k != "jm:1008590" for _, k in pairs_of("ゆったりとしている"))     # ゆったり｜と｜している
    assert ("しなきゃ", "jm:1157170") not in pairs_of("提出しなきゃいけない")


def test_false_merges_and_homographs():
    for text, bad in [("ゲームとかでよくそういうの", "jm:1547480"),        # よく｜そう→浴槽
                      ("が良ければいいんじゃないですかね", "jm:1156100"),  # いい｜ん→委員
                      ("観光スポットとかになると", "jm:1753350"),          # なる｜と→鳴門
                      ("そこで今日は完全在宅ワークの", "jm:1289400"),      # 今日は→こんにちは
                      ("いや私あなたのところで骨を埋める気だった", "jm:1343110"),
                      ("軽く説明すると", "jm:1006280"),
                      ("うーんめっちゃ寒い", "jm:1012440")]:               # めっちゃ→めく
        assert all(k != bad for _, k in pairs_of(text)), (text, pairs_of(text))
    assert ("そういう", "jm:1394680") in pairs_of("ゲームとかでよくそういうの")
    assert ("めっちゃ", "jm:2183440") in pairs_of("うーんめっちゃ寒い")
    assert ("今日は", "jm:1289400") in pairs_of("皆さん今日は")               # 真的在打招呼
    assert ("ところで", "jm:1343110") in pairs_of("ところで、元気？")
    assert ("そのあと", "jm:1006860") in pairs_of("そのあとシャンプーする")    # 其の＋後 是同一個詞
    assert ("そう", "jm:2137720") in pairs_of("そうなったときクロモン市場が")  # 副詞 そう，不是樣態
    assert ("行った", "jm:1578850") in pairs_of("一人で行った方が楽")         # UniDic 標 行う，其實是 行く
    assert ("行った", "jm:1589060") in pairs_of("調査を行った")


def test_carry_over_tail_not_counted():
    """vocab.annotate 會把上一行給 segment；接在上一行同一個詞後面的尾巴不進單字索引。"""
    import sqlite3
    from app import db, vocab
    cues = [{"start": 0, "end": 1, "text": "多くの人で賑わいを見せて", "brk": "hard"},
            {"start": 1, "end": 2, "text": "いた飲食店エリアには", "brk": "eos"},
            {"start": 2, "end": 3, "text": "と", "brk": "gap"},
            {"start": 3, "end": 4, "text": "いうことで", "brk": "eos"}]
    meta = vocab.annotate(cues, "ja")
    assert cues[1]["w"][0][2] == "jm:1259210" and cues[1]["w"][0][4] == 1, cues[1]
    assert cues[3]["w"][0][2] == TOIU, cues[3]
    saved = db._conn
    db._conn = sqlite3.connect(":memory:", check_same_thread=False, isolation_level=None)
    db._conn.row_factory = sqlite3.Row
    try:
        db._conn.executescript(vocab.SCHEMA)
        vocab.index_track("t1", cues, "ja", meta)
        row = db._one("SELECT count, cues FROM track_vocab WHERE track_id = 't1' AND key = 'jm:1259210'")
        assert row["count"] == 1 and row["cues"] == "[0]", dict(row)
        assert db._one("SELECT count FROM track_vocab WHERE key = ?", (TOIU,))["count"] == 1
    finally:
        db._conn.close()
        db._conn = saved


# ---------- SEGMENT_RULES 9：驗證時找到的回歸（例句取自片庫） ----------

SOUIU, SOU_ADV, SOU_LOOKS = "jm:1394680", "jm:2137720", "jm:1006610"


def test_rules9_regressions():
    # UniDic 把句首的感嘆詞 ああ 標成副詞：後面沒接動詞時是 嗚呼，不是「那樣」的 ああ
    for text in ("ああなるほどね", "ああ一番", "ああ全然辛い"):
        assert ("ああ", "jm:1565440") in pairs_of(text), (text, pairs_of(text))
    assert ("ああやって", "jm:2772390") in pairs_of("ああやって")
    # め 被 UniDic 當記号時切點不可靠，めかくし 照舊是 目隠し
    assert ("めかくし", "jm:1535310") in pairs_of("荒波でめかくしてサーフィン")
    # だっ 被 UniDic 誤切成動詞 立つ：不能併成 だる＋ていく，だって 也不能變成 だる
    pairs = pairs_of("どこでだっていけるんです")
    assert all(k != "jm:2867372" for _, k in pairs) and ("いける", "jm:1578850") in pairs, pairs
    # 寫漢字的 今日は 後面接填充詞、或停在行尾，是「今天」；後面接另一句招呼才是打招呼
    for text in ("ということで今日はまあいろんなね州出身の", "はいということで今日は", "今日は"):
        assert ("今日", "jm:1579110") in pairs_of(text) and all(k != "jm:1289400" for _, k in pairs_of(text)), text
    assert ("今日は", "jm:1289400") in pairs_of("今日はこんばんは豆こと申します")
    # 緊接在 不安定、よさ 後面的 そう 是樣態；実際そうでした 的 そう 是副詞
    assert ("そう", SOU_LOOKS) in pairs_of("不安定そうっていうイメージあると思います")
    assert ("そう", SOU_LOOKS) in pairs_of("城内で泳いでる人たち気持ちよさそうすぎるな")
    assert ("そう", SOU_ADV) in pairs_of("僕も実際そうでした")


def test_scan_without_te_aux():
    """Shift 即時查詢不用 TE_AUX 規則：てきます 不會被猜成 ていく。"""
    for text in ("初めてきました", "なってきました", "見てきます"):
        res = dict_ja.scan(text, 0, limit=5)
        assert res["cands"] and all(x not in dict_ja.TE_AUX for c in res["cands"] for x in c["chain"]), (text, res)
    assert any(c["base"] == "初めて" for c in dict_ja.scan("初めてきました", 0, limit=5)["cands"])


def test_carry_over_demonstrative_counted_once():
    """そう｜いう 跨行：上一行的 そう 改成 そういう、標 1 不計數，這一行的 いう 算一次；停頓（gap）也接。"""
    import sqlite3
    from app import db, vocab
    cues = [{"start": 0, "end": 1, "text": "政治話がすごく好きで、ああそう", "brk": "gap"},
            {"start": 1, "end": 2, "text": "いうイメージあるね", "brk": "eos"}]
    meta = vocab.annotate(cues, "ja")
    last = cues[0]["w"][-1]
    assert last[2] == SOUIU and last[4] == 1, cues[0]
    assert cues[1]["w"][0][:3] == [0, 2, SOUIU] and len(cues[1]["w"][0]) < 5, cues[1]
    saved = db._conn
    db._conn = sqlite3.connect(":memory:", check_same_thread=False, isolation_level=None)
    db._conn.row_factory = sqlite3.Row
    try:
        db._conn.executescript(vocab.SCHEMA)
        vocab.index_track("t1", cues, "ja", meta)
        assert db._one("SELECT count, cues FROM track_vocab WHERE key = ?", (SOUIU,))["count"] == 1
        assert db._one("SELECT count FROM track_vocab WHERE key = ?", (SOU_ADV,)) is None
    finally:
        db._conn.close()
        db._conn = saved
    # 沒給上一行的詞單位（prev_units）就不接，免得同一個詞算兩次
    assert units_of("いうイメージあるね", prev="好きで、ああそう", prev_brk="hard")[0][1] == IU


if __name__ == "__main__":
    if not dict_ja.ready():
        sys.exit("這支測試要用真正的日文字典，請先執行 python -s tools/build_dict_ja.py 建好 data/dict")
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
