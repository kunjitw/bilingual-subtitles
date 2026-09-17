"""字幕組裝的單元測試：python -s -m pytest tests/test_cues.py  或  python -s tests/test_cues.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import translate  # noqa: E402
from app.cues import Token, attach_units, finalize, normalize_asr_text, segment  # noqa: E402


def make_tokens(words, start=0.0, step=0.3, pauses=None):
    pauses = pauses or {}
    toks, t = [], start
    for i, w in enumerate(words):
        t += pauses.get(i, 0.0)
        toks.append(Token(w, t, t + step * 0.9))
        t += step
    return toks


def test_japanese_long_sentence_is_balanced():
    words = ["来", "れ", "ない", "し", "と", "いう", "そういう", "状況", "が", "続い", "てる", "か", "と", "思い", "ます。"]
    cues = segment(make_tokens(words), "ja")
    texts = [c["text"] for c in cues]
    assert "".join(texts) == "".join(words)
    assert all(len(t) >= 3 for t in texts), texts
    assert cues[-1]["brk"] == "eos"


def test_english_breaks_at_sentence_and_prefers_conjunction():
    words = ("Then we evaluate or appraise the situation and whether it helps or hinders our goals. "
             "That process has four steps.").split(" ")
    toks = make_tokens([w + " " for w in words])
    cues = segment(toks, "en")
    assert cues[-1]["text"] == "That process has four steps."
    first = [c for c in cues if c["brk"] != "eos"]
    assert first and first[-1]["text"].split()[-1] not in ("our", "the"), [c["text"] for c in cues]


def test_english_normalization_and_abbrev():
    assert normalize_asr_text("the situation.For example", "en") == "the situation. For example"
    assert normalize_asr_text("in the U.S. today", "en") == "in the U.S. today"
    assert normalize_asr_text("J R の方 では", "ja") == "JRの方では"
    toks = make_tokens([w + " " for w in "Mr. Smith went to the U.S. last year.".split()])
    assert len(segment(toks, "en")) == 1


def test_attach_units_keeps_punctuation():
    text = "Hello, world. It's U.S. time!"
    units = [["Hello", 0, 0.4], ["world", 0.5, 0.9], ["It's", 1.2, 1.4], ["US", 1.5, 1.8], ["time", 1.9, 2.2]]
    toks = attach_units(text, units, [0, 3])
    assert "".join(t.text for t in toks) == text


def test_finalize_no_overlap_and_merges_tiny():
    cues = [
        {"start": 0.0, "end": 1.5, "text": "這本書總共兩冊", "brk": "soft"},
        {"start": 1.6, "end": 1.8, "text": "和附錄", "brk": "hard"},
        {"start": 1.7, "end": 3.0, "text": "我是先看附錄的", "brk": "eos"},
    ]
    out = finalize(cues, "zh-TW")
    assert len(out) == 2
    assert out[1]["text"] == "和附錄 我是先看附錄的"
    for a, b in zip(out, out[1:]):
        assert a["end"] <= b["start"]


def test_lonely_short_cue_merges_forward():
    cues = [
        {"start": 10.0, "end": 11.0, "text": "天氣是冷的", "brk": "gap"},
        {"start": 11.9, "end": 12.7, "text": "有", "brk": "gap"},
        {"start": 12.8, "end": 13.8, "text": "一點一點變化的", "brk": "eos"},
    ]
    out = finalize(cues, "zh-TW")
    assert [c["text"] for c in out] == ["天氣是冷的", "有 一點一點變化的"]


def test_translation_units_and_distribution():
    src = [
        {"start": 0, "end": 2, "text": "Psychologists use this tool to identify where and how to intervene", "brk": "hard"},
        {"start": 2, "end": 4, "text": "in the process that forms our emotions.", "brk": "eos"},
        {"start": 4, "end": 6, "text": "That process has four steps.", "brk": "eos"},
    ]
    units = translate.build_units(src, "en")
    assert units == [[0, 1], [2]]
    done = {"0": "心理學家利用這一工具來識別在情緒形成過程中，應在何處以及如何進行干預，這是很長的一句話", "1": "這一過程包含四個步驟"}
    out = translate.finalize_translation(src, units, done)
    assert out[-1]["text"] == "這一過程包含四個步驟"
    assert len(out) == 3 and all(c["text"] for c in out)
    short = translate.finalize_translation(src, units, {"0": "心理學家用這工具找出介入點", "1": "包含四個步驟"})
    assert short[0]["start"] == 0 and short[0]["end"] == 4


def test_short_translation_pieces_are_merged():
    pieces = [
        {"start": 0, "end": 2, "text": "這一過程的每個階段都為我們提供了有意識地"},
        {"start": 2, "end": 4, "text": "而該"},
        {"start": 4, "end": 6, "text": "過程模型概述了我們可以嘗試的策略"},
    ]
    out = translate._merge_short_pieces(pieces)
    assert len(out) == 2 and out[0]["end"] == 4
    first_short = translate._merge_short_pieces([{"start": 0, "end": 1, "text": "而"}, {"start": 1, "end": 3, "text": "這是很長的句子"}])
    assert first_short == [{"start": 0, "end": 3, "text": "而，這是很長的句子"}]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
