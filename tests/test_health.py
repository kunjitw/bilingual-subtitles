"""時間軸健康檢查的單元測試（假的辨識與對齊模型、暫存資料夾裡的資料庫，不碰顯卡也不動正式資料）：
python -s tests/test_health.py

1. app/health.py 的演算法：norm / recall、區段規劃、對齊結果分回各行、採用規則、時間整理、翻譯同步
2. app/asr_child.py 的 verify / realign：用假模型跑完整流程
3. app/jobs.py 的 handle_health 與 handle_transcribe：子程序改成在同一個程序裡呼叫 asr_child.dispatch
"""
import json
import logging
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

os.environ["VS_NO_WORKERS"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

from app import asr_child, db, gpu, health, jobs, safepath, settings  # noqa: E402
from app import cues as cue_mod  # noqa: E402
from app.config import SAMPLE_RATE  # noqa: E402

SCALE = 2 ** 24  # 假音訊的第 n 個樣本值是 n / 2^24（float32 可以精確表示），從片段第一個樣本就知道它從哪裡切出來


# ---------- 1. 演算法 ----------

def cue(start, end, text, **kw):
    return {"start": start, "end": end, "text": text, **kw}


def test_norm_and_recall():
    assert health.norm("「こんにちは、世界！」") == "こんにちは世界"
    assert health.norm("Hello, World - OK?") == "helloworldok"
    assert health.recall("こんにちは世界", "えっと、こんにちは世界です") == 1.0
    assert health.recall("Hello, world.", "hello world and more") == 1.0
    assert abs(health.recall("東京大阪", "東京") - 0.5) < 1e-9
    assert health.recall("", "なにか") == 1.0          # 沒有字可以比的行不算對不上
    assert health.recall("大阪", "") == 0.0
    # 依序比對：順序亂掉的字不全算
    assert health.recall("abcd", "dcba") < 0.5


def test_anchor_and_bad_rules():
    assert health.is_anchor("大阪", 0.8) and not health.is_anchor("大", 1.0)
    assert not health.is_anchor("大阪", 0.79)
    assert health.is_bad("大阪城", 0.59) and not health.is_bad("大阪城", 0.6)
    assert not health.is_bad("大阪", 0.0)              # 兩個字以下不判斷
    assert not health.is_bad("大阪城", None)


def test_plan_regions_between_anchors():
    cl = [cue(i * 3, i * 3 + 2, t) for i, t in enumerate(
        ["一二三", "四五六", "七八九", "甲乙丙", "丁戊己", "庚辛壬", "子丑寅", "卯辰巳"])]
    rec = {0: 1.0, 1: 0.9, 2: 0.7, 3: 0.2, 4: 0.65, 5: 0.95, 6: 0.1, 7: 0.3}
    regions = health.plan_regions(cl, rec, 30.0)
    # 行 3 對不上：往前到錨點 1、往後到錨點 5，中間非錨點的 2、4 一起重對
    assert regions[0] == (2, 4, cl[1]["end"], cl[5]["start"]), regions
    # 行 6、7 都對不上而且後面沒有錨點：區段到音訊結尾
    assert regions[1] == (6, 7, cl[5]["end"], 30.0), regions
    assert len(regions) == 2
    # 開頭就對不上：從 0 秒開始
    r0 = health.plan_regions(cl, {**rec, 0: 0.0, 1: 0.5, 2: 0.9}, 30.0)[0]
    assert r0 == (0, 1, 0.0, cl[2]["start"]), r0


def test_plan_regions_short_lines_are_not_anchors():
    cl = [cue(0, 2, "東京駅"), cue(3, 4, "え"), cue(5, 7, "大阪城"), cue(8, 10, "名古屋")]
    rec = {0: 1.0, 1: 1.0, 2: 0.0, 3: 1.0}
    # 一個字的行 recall 再高也不當錨點，所以區段往前延伸到行 0
    assert health.plan_regions(cl, rec, 12.0) == [(1, 2, 2, 8)]


def test_plan_regions_long_region_shrinks():
    cl = [cue(0, 2, "東京駅"), cue(50, 52, "大阪城"), cue(53, 55, "名古屋"), cue(150, 152, "博多駅")]
    rec = {0: 1.0, 1: 0.1, 2: 0.5, 3: 1.0}
    (lo, hi, s, e), = health.plan_regions(cl, rec, 200.0)
    assert (lo, hi) == (1, 2)
    # 原本 2–150 秒超過 90 秒：縮到第 1–2 行前後 10 秒
    assert (s, e) == (40, 65), (s, e)
    # 縮完不能超出錨點
    cl2 = [cue(42, 45, "東京駅"), cue(50, 52, "大阪城"), cue(150, 152, "博多駅")]
    (_, _, s2, e2), = health.plan_regions(cl2, {0: 1.0, 1: 0.0, 2: 1.0}, 200.0)
    assert (s2, e2) == (45, 62), (s2, e2)


def test_plan_regions_odd_timing_is_not_anchor():
    """時間窗長得不合理、或跟前一行同時開始的行，recall 再高也不當錨點。"""
    cl = [cue(9, 11, "福岡博多"), cue(13.0, 13.05, "札幌仙台"), cue(13.0, 13.05, "横浜川崎"),
          cue(13.0, 13.05, "神戸姫路"), cue(13.0, 26.95, "奈良広島"), cue(29, 31, "金沢富山")]
    rec = {0: 1.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 1.0, 5: 1.0}
    assert health.odd_timing(cl, 4) and health.odd_timing(cl, 2) and not health.odd_timing(cl, 1)
    assert not health.odd_timing(cl, 0) and not health.odd_timing(cl, 5)
    # 行 4 的窗有 14 秒，什麼都聽得到：不能當錨點，區段要一路延伸到行 5，聲音才涵蓋 13–29 秒
    assert health.plan_regions(cl, rec, 40.0) == [(1, 4, 11, 29)]
    # 跟前一行同時開始的行（行 2、3）也不當錨點
    rec2 = {0: 1.0, 1: 0.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0}
    assert health.plan_regions(cl[:4] + [cue(20, 22, "奈良広島"), cl[5]], rec2, 40.0) == [(1, 3, 11, 20)]
    # 長度門檻看字數：10 個字講 5 秒不算可疑
    assert not health.odd_timing([cue(0, 5, "あ" * 10), cue(6, 11, "い" * 10)], 1)


def test_alignable_and_batches():
    assert not health.alignable((0, 0, 1.0, 1.2))
    assert not health.alignable((0, 0, 0.0, 200.0))
    assert health.alignable((0, 0, 0.0, 90.0))
    regions = [(0, 0, 0, 50), (1, 1, 60, 70), (2, 2, 80, 90), (3, 3, 100, 190)]
    batches = health.align_batches(regions, 2)            # 每批（最長 × 筆數）不超過 120 秒
    assert [[r[0] for r in b] for b in batches] == [[1, 2], [0], [3]], batches
    for b in batches:
        assert len(b) * max(r[3] - r[2] for r in b) <= 120 or len(b) == 1


def test_verify_batches_limit_length_not_just_count():
    """同一批的片段會被補齊到最長那筆的長度，所以一行很長的字幕不能跟一大堆短行同一批。"""
    assert health.verify_batch_size(6) == 24 and health.verify_batch_size(9) == 32
    cl = [cue(i * 5, i * 5 + 2, "測試") for i in range(20)]
    cl[7].update(start=100.0, end=220.0)                  # 對齊失敗留下的長行
    batches = health.verify_batches(list(range(20)), cl, 300.0, 8)
    assert sorted(i for b in batches for i in b) == list(range(20))
    assert [b for b in batches if 7 in b] == [[7]], batches   # 長行自己一批
    budget = 8 * health.ASR_CHUNK_S / health.VERIFY_SCALE
    for b in batches:
        longest = max(cl[i]["end"] - cl[i]["start"] + 2 * health.PAD_S for i in b)
        assert len(b) <= 8 and (len(b) == 1 or len(b) * longest <= budget), b
    # 都是短行時就照筆數分批
    assert [len(b) for b in health.verify_batches(list(range(20)), cl[:7] + cl[8:] + [cl[7]], 300.0, 8)][:2] == [8, 8]


def test_distribute_units_to_lines():
    # 日文：一個字一個單位，標點不算
    units = [["こ", 1.0, 1.1], ["ん", 1.1, 1.2], ["に", 1.2, 1.3], ["ち", 2.0, 2.1], ["は", 2.1, 2.5]]
    spans = health.distribute(units, ["こんに、", "…", "ちは！"])
    assert spans == [(1.0, 1.3), None, (2.0, 2.5)], spans
    # 英文：一個字一個單位，字元數加總；太短的行至少 0.3 秒
    units = [["It's", 5.0, 5.2], ["a", 5.2, 5.25], ["test", 5.3, 5.6], ["OK", 6.0, 6.05]]
    spans = health.distribute(units, ["It's a test.", "OK!"])
    assert spans == [(5.0, 5.6), (6.0, 6.3)], spans
    # 單位不夠分：後面的行保留原時間
    assert health.distribute([["ab", 0, 1]], ["ab", "cd"]) == [(0.0, 1.0), None]


def test_distribute_word_across_lines():
    """對齊器的一個詞跨兩行時（nagisa 把「したいな」+「いろんな」斷成「ない」），
    照字數比例切開那個詞的時間，後面的行不會跟著偏。"""
    texts = ["利用したいな", "いろんなお店", "回りきれない"]
    # 每個字 0.1 秒、行與行之間停 1 秒；「な」+「い」被當成一個詞，時間跨過停頓
    t, char_times = 0.0, []
    for text in texts:
        for _ in text:
            char_times.append((round(t, 3), round(t + 0.1, 3)))
            t += 0.1
        t += 1.0
    words = ["利用", "し", "たい", "ない", "ろんな", "お", "店", "回り", "きれ", "ない"]
    units, p = [], 0
    for w in words:
        units.append([w, char_times[p][0], char_times[p + len(w) - 1][1]])
        p += len(w)
    spans = health.distribute(units, texts)
    # 跨行的詞 0.5–1.7 秒（「な」0.5–0.6、「い」1.6–1.7）照字數對半分在 1.1 秒
    assert spans[0] == (0.0, 1.1), spans
    assert spans[1] == (1.1, char_times[11][1]), spans
    # 第三行完全不受影響
    assert spans[2] == (char_times[12][0], char_times[17][1]), spans


def test_apply_regions():
    cl = [cue(0, 2, "東京駅"), cue(5, 7, "大阪城"), cue(8, 10, "名古屋")]
    region = (1, 1, 2, 8)
    units = {health.region_key(region): [["大", 3.0, 3.2], ["阪", 3.2, 3.4], ["城", 3.4, 4.0]]}
    fixed, touched = health.apply_regions(cl, [region], units)
    assert touched == [1]
    # 結束時間跟 cues.finalize 一樣留 0.3 秒閱讀時間
    assert (fixed[1]["start"], fixed[1]["end"]) == (3.0, 4.3)
    assert cl[1]["start"] == 5  # 不改原本的 list
    # 沒有對齊結果的區段不動
    fixed, touched = health.apply_regions(cl, [region], {})
    assert touched == [] and fixed == cl
    # 很短的行至少顯示 0.8 秒，但不蓋到下一行
    cl2 = [cue(0, 2, "東京駅"), cue(5, 7, "え"), cue(8, 10, "名古屋")]
    units = {health.region_key(region): [["え", 3.0, 3.1]]}
    assert health.apply_regions(cl2, [region], units)[0][1] == cue(3.0, 3.8, "え")
    units = {health.region_key(region): [["え", 7.5, 7.6]]}
    assert health.apply_regions(cl2, [region], units)[0][1] == cue(7.5, 7.98, "え")


def test_apply_regions_punctuation_line_between_aligned():
    """區段中間夾一行整行標點：擺到前後已對齊的行中間，不能用它的舊時間把後面對齊好的行推走。"""
    cl = [cue(1.0, 3.0, "東京大阪"), cue(9.0, 9.9, "名古屋市"), cue(10.0, 10.4, "……"),
          cue(10.5, 12.4, "福岡博多"), cue(13.0, 15.0, "札幌仙台")]
    region = (1, 3, 3.0, 13.0)
    units = [[ch, round(5.0 + k * 0.5, 3), round(5.45 + k * 0.5, 3)] for k, ch in enumerate("名古屋市")]
    units += [[ch, round(7.5 + k * 0.5, 3), round(7.95 + k * 0.5, 3)] for k, ch in enumerate("福岡博多")]
    fixed, touched = health.apply_regions(cl, [region], {health.region_key(region): units})
    assert touched == [1, 3]
    assert [(c["start"], c["end"]) for c in fixed[1:4]] == [(5.0, 7.25), (7.25, 7.5), (7.5, 9.75)], fixed
    # 採用：兩行都對上了，中間那行跟著擺在兩行之間
    out, adopted, _ = health.adopt(cl, fixed, {0: 1.0, 1: 0.0, 2: 1.0, 3: 0.0, 4: 1.0}, {1: 1.0, 3: 1.0}, touched)
    assert adopted == {1, 3}, adopted
    assert [(c["start"], c["end"]) for c in out[1:4]] == [(5.0, 7.25), (7.25, 7.5), (7.5, 9.75)], out


def test_adopt_requires_gain():
    cl = [cue(0, 2, "東京駅"), cue(5, 7, "大阪城"), cue(10, 12, "名古屋"), cue(15, 17, "博多駅")]
    fixed = [dict(c) for c in cl]
    fixed[1].update(start=3.0, end=4.5)
    fixed[2].update(start=8.0, end=9.5)
    rec = {0: 1.0, 1: 0.5, 2: 0.2, 3: 1.0}
    rec_fix = {1: 0.6, 2: 0.29}                        # 行 1 剛好進步 0.1 → 採用；行 2 只進步 0.09 → 不採用
    out, adopted, current = health.adopt(cl, fixed, rec, rec_fix, [1, 2])
    assert adopted == {1}, adopted
    assert (out[1]["start"], out[1]["end"]) == (3.0, 4.5)
    assert (out[2]["start"], out[2]["end"]) == (10, 12)
    assert current[1] == 0.6 and current[2] == 0.2
    # 沒有重新檢查到的行不能採用
    out, adopted, _ = health.adopt(cl, fixed, rec, {}, [1, 2])
    assert adopted == set() and out == cl


def test_adopt_keeps_order():
    cl = [cue(0, 2, "東京駅"), cue(20, 22, "大阪城"), cue(24, 26, "名古屋"), cue(30, 32, "博多駅")]
    fixed = [dict(c) for c in cl]
    fixed[1].update(start=5.0, end=7.0)
    fixed[2].update(start=8.0, end=10.0)
    rec = {0: 1.0, 1: 0.0, 2: 0.0, 3: 1.0}
    # 兩行都進步：不管處理順序，最後都要一起採用
    out, adopted, _ = health.adopt(cl, fixed, rec, {1: 1.0, 2: 1.0}, [2, 1])
    assert adopted == {1, 2}, adopted
    # 只有行 2 進步：行 1 原本就對不上，時間不可信，不能擋行 2；
    # 行 2 採用後行 1 順序不合，改放它自己重新對齊的時間（還是標成對不上）
    out, adopted, current = health.adopt(cl, fixed, rec, {1: 0.0, 2: 1.0}, [1, 2])
    assert adopted == {2}, adopted
    assert [(c["start"], c["end"]) for c in out] == [(0, 2), (5.0, 7.0), (8.0, 10.0), (30, 32)], out
    assert current[1] == 0.0 and health.is_bad(cl[1]["text"], current[1])
    # 行 1 原本對得上（recall 0.7，只是不夠當錨點）：它的時間可信，行 2 跑到它前面就不採用
    rec_ok = {**rec, 1: 0.7}
    out, adopted, _ = health.adopt(cl, fixed, rec_ok, {1: 0.7, 2: 1.0}, [1, 2])
    assert adopted == set(), adopted
    starts = [c["start"] for c in out]
    assert starts == sorted(starts)


def test_adopt_chain_not_blocked_by_one_failure():
    """對齊器把一串行擠在同一個時間，重新對齊後只有中間一行驗證不過：其他行照樣採用，
    沒採用的那行擺到前後行中間，不會留在原地讓前面整串變成閃一下的行。"""
    names = ["東京大阪", "名古屋市", "福岡博多", "札幌仙台", "横浜川崎", "神戸姫路", "奈良広島", "金沢富山", "松江鳥取"]
    truth = [(4 * i + 1.0, 4 * i + 3.25) for i in range(len(names))]
    cl = [cue(s, e, t) for (s, e), t in zip(truth, names)]
    for i in range(2, 8):
        cl[i].update(start=9.0, end=9.05)
    fixed = [cue(s, e, t) for (s, e), t in zip(truth, names)]
    rec = {i: (0.0 if 2 <= i < 8 else 1.0) for i in range(len(names))}
    rec_fix = {2: 1.0, 3: 1.0, 4: 0.0, 5: 1.0, 6: 1.0, 7: 1.0}
    out, adopted, current = health.adopt(cl, fixed, rec, rec_fix, list(range(2, 8)))
    assert adopted == {2, 3, 5, 6, 7}, adopted
    assert [(c["start"], c["end"]) for c in out] == truth, out
    assert current[4] == 0.0 and all(current[i] == 1.0 for i in adopted)
    # 行 4 的重新對齊時間也放不進去（跑到行 5 後面）：夾在前後行之間
    fixed[4].update(start=30.0, end=31.0)
    out, adopted, current = health.adopt(cl, fixed, rec, rec_fix, list(range(2, 8)))
    assert adopted == {2, 3, 5, 6, 7}, adopted
    assert (out[4]["start"], out[4]["end"]) == (15.25, 16.05), out[4]
    for a, b in zip(out, out[1:]):
        assert a["start"] <= b["start"] and a["end"] <= b["start"], (a, b)


def test_adopt_long_window_line_does_not_block():
    """時間窗長得不合理的行 recall 很高，但時間不可信：不能擋前面修好的行；順序不合時改放它重新對齊的時間。"""
    cl = [cue(9, 11, "福岡博多"), cue(13.0, 13.05, "札幌仙台"), cue(13.0, 13.05, "横浜川崎"),
          cue(13.0, 13.05, "神戸姫路"), cue(13.0, 26.95, "奈良広島"), cue(29, 31, "金沢富山")]
    fixed = [cue(9, 11, "福岡博多"), cue(13, 15.3, "札幌仙台"), cue(17, 19.3, "横浜川崎"),
             cue(21, 23.3, "神戸姫路"), cue(25, 27.3, "奈良広島"), cue(29, 31, "金沢富山")]
    rec = {0: 1.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 1.0, 5: 1.0}
    rec_fix = {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0}
    out, adopted, current = health.adopt(cl, fixed, rec, rec_fix, [1, 2, 3, 4])
    assert adopted == {1, 2, 3}, adopted            # 行 4 的 recall 沒有進步，不算修正
    assert [(c["start"], c["end"]) for c in out] == [(c["start"], c["end"]) for c in fixed], out
    assert current[4] == 1.0


def test_tidy_times():
    cl = [cue(0, 5, "a"), cue(3, 4, "b"), cue(3, 3, "c"), cue(10, 9, "d")]
    health.tidy_times(cl)
    for a, b in zip(cl, cl[1:]):
        assert a["end"] <= b["start"], cl
        assert a["start"] <= b["start"], cl
    assert all(c["end"] > c["start"] for c in cl), cl
    assert cl[0] == {"start": 0.0, "end": 3.0, "text": "a"}
    # 沒問題的資料不動
    ok = [cue(0, 1.5, "a"), cue(2, 3, "b")]
    assert health.tidy_times([dict(c) for c in ok]) == ok


def test_apply_times_and_summary():
    target = [cue(0, 1, "東京駅", ruby=[[0, 2, "とうきょう"]], w=[[0, 3, "k"]]), cue(2, 3, "大阪城", chk=0), cue(4, 5, "名古屋")]
    new = [cue(0, 1, "東京駅"), cue(2.5, 3.5, "大阪城"), cue(4, 5, "名古屋")]
    assert health.apply_times(target, new, [2])
    assert target[0]["ruby"] == [[0, 2, "とうきょう"]] and target[0]["w"] == [[0, 3, "k"]]
    assert "chk" not in target[1] and (target[1]["start"], target[1]["end"]) == (2.5, 3.5)
    assert target[2]["chk"] == 0 and "chk" not in target[0]
    assert not health.apply_times(target, new, [2])   # 再套一次沒有變化
    s = health.summarize(new, {0: 1.0, 1: 0.1, 2: 0.2}, {0: 1.0, 1: 0.9, 2: 0.2}, {1}, 1)
    assert (s["checked"], s["before_bad"], s["fixed"], s["after_bad"], s["bad"]) == (3, 2, 1, 1, [2]), s
    assert s["v"] == health.HEALTH_VERSION


def test_carry_over():
    before = [cue(0, 1, "a"), cue(2, 3, "b"), cue(4, 5, "c")]
    fixed = [cue(0, 1, "a"), cue(2.2, 3.5, "b"), cue(4, 5, "c")]
    after = [cue(0, 1, "a"), cue(2.2, 3.5, "b"), cue(4, 4.5, "c")]   # 行 2 被整理過，時間不同
    rec, rec_fix = {0: 1.0, 1: 0.5, 2: 0.8}, {1: 0.9}
    assert health.carry_over(before, fixed, after, {0: 1.0, 1: 0.9, 2: 0.8}, rec, rec_fix) == {0: 1.0, 1: 0.9}
    # 沒採用（recall 還是舊的）但時間被改成重新對齊的時間：量的時間不一樣，不能沿用
    moved = [cue(0, 1, "a"), cue(2.2, 3.5, "b"), cue(4, 5, "c")]
    assert health.carry_over(before, fixed, moved, {0: 1.0, 1: 0.5, 2: 0.8}, rec, rec_fix) == {0: 1.0, 2: 0.8}


def test_sync_translation():
    old = [cue(0, 1, "a"), cue(2, 3, "b"), cue(4, 5, "c"), cue(6, 7, "d")]
    new = [cue(0, 1, "a"), cue(2.4, 3.2, "b"), cue(4, 5, "c"), cue(6, 7, "d")]
    same = [cue(0, 1, "甲"), cue(2, 3, "乙"), cue(4, 5, "丙"), cue(6, 7, "丁", extra=1)]
    out = health.sync_translation(old, new, same)
    assert [(c["start"], c["end"]) for c in out] == [(0, 1), (2.4, 3.2), (4, 5), (6, 7)]
    assert out[3]["extra"] == 1 and same[1]["start"] == 2        # 其他欄位保留，不改原本的 list
    # 逐行翻譯遇到空白譯文少一行：照時間對回原文
    dropped = [cue(0, 1, "甲"), cue(2, 3, "乙"), cue(6, 7, "丁")]
    out = health.sync_translation(old, new, dropped)
    assert [(c["start"], c["end"]) for c in out] == [(0, 1), (2.4, 3.2), (6, 7)]
    # 整句翻譯跨行：對不回原文，不動
    sentence = [cue(0, 3, "甲乙"), cue(4, 7, "丙丁")]
    assert health.sync_translation(old, new, sentence) is None
    # 舊資料小數點誤差在 0.01 秒內視為相同
    assert health.sync_translation(old, new, [cue(2.004, 2.996, "乙")])[0]["start"] == 2.4
    assert health.sync_translation(old, new[:3], same) is None
    # 原文有幾行時間完全相同（對齊器擠在一起），譯文又少一行：分不出少的是哪一行，不動
    old2 = [cue(5, 6, "A"), cue(10.0, 10.05, "B"), cue(10.0, 10.05, "C"), cue(12, 13, "D")]
    new2 = [cue(5, 6, "A"), cue(8, 9, "B"), cue(10, 11, "C"), cue(12, 13, "D")]
    tr2 = [cue(5, 6, "譯A"), cue(10.0, 10.05, "譯C"), cue(12, 13, "譯D")]
    assert health.sync_translation(old2, new2, tr2) is None
    # 同樣時間的行譯文都在，或少的是別的行：對應關係確定，照樣同步
    full = [cue(5, 6, "譯A"), cue(10.0, 10.05, "譯B"), cue(10.0, 10.05, "譯C"), cue(12, 13, "譯D")]
    assert [c["start"] for c in health.sync_translation(old2, new2, full)] == [5, 8, 10, 12]
    assert [c["start"] for c in health.sync_translation(old2, new2, full[1:])] == [8, 10, 12]


def test_file_names():
    f = health.round_files(1)
    assert f["units"] == "health1_fix.units.json" == health.units_name(f["fix"])
    assert len(set(f.values())) == len(f)
    assert health.backup_path("abc").name == "abc.bak.json"


# ---------- 假模型 ----------

class Obj:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def ramp_wav(path: Path, seconds: float):
    n = int(seconds * SAMPLE_RATE)
    sf.write(str(path), (np.arange(n, dtype=np.float64) / SCALE).astype(np.float32), SAMPLE_RATE, subtype="FLOAT")


def clip_start(clip) -> float:
    return round(float(clip[0]) * SCALE) / SAMPLE_RATE


class FakeASR:
    """聽到的內容：跟片段重疊超過一半長度的真實句子。"""

    def __init__(self, truth):
        self.truth = truth  # [(text, start, end)]
        self.clips = 0

    def transcribe(self, audio, language):
        out = []
        for clip, sr in audio:
            assert sr == SAMPLE_RATE
            s = clip_start(clip)
            e = s + len(clip) / SAMPLE_RATE
            heard = [t for t, ts, te in self.truth if min(e, te) - max(s, ts) >= 0.5 * (te - ts)]
            out.append(Obj(text=" ".join(heard)))
            self.clips += 1
        return out


class FakeAligner:
    """每個字（英文是每個單字）回傳它真正的時間；wrong 裡的字故意回傳錯的時間。"""

    def __init__(self, units, wrong=None):
        self.times = {u: (s, e) for u, s, e in units}
        self.wrong = wrong or {}
        self.calls = 0

    def align(self, audio, text, language):
        self.calls += 1
        out = []
        for (clip, _), t in zip(audio, text):
            s0 = clip_start(clip)
            if language == "English":
                toks = ["".join(ch for ch in w if cue_mod._kept(ch)) for w in t.split()]
            else:
                toks = [ch for ch in t if cue_mod._kept(ch)]
            items = []
            for tok in toks:
                if not tok:
                    continue
                s, e = self.wrong.get(tok) or self.times[tok]
                items.append(Obj(text=tok, start_time=round(s - s0, 3), end_time=round(e - s0, 3)))
            out.append(Obj(items=items))
        return out


# 日文測試資料：12 行、每行 4 個字（全部不重複），第 i 行真正在 [4i+1, 4i+3]
JA_LINES = ["東京大阪", "名古屋市", "福岡博多", "札幌仙台", "横浜川崎", "神戸姫路",
            "奈良広島", "金沢富山", "松江鳥取", "長野熊本", "秋田青森", "那覇沖縄"]
JA_SECONDS = 50.0


def ja_truth():
    lines = [(t, 4 * i + 1.0, 4 * i + 3.0) for i, t in enumerate(JA_LINES)]
    units = []
    for t, s, e in lines:
        step = (e - s) / len(t)
        units += [(ch, round(s + k * step, 3), round(s + (k + 1) * step - 0.05, 3)) for k, ch in enumerate(t)]
    return lines, units


def ja_wrong():
    """行 9 的字對齊器會對到錯的時間（跟偏掉的字幕一樣），修不好。"""
    return {ch: (38.5 + k * 0.5, 38.9 + k * 0.5) for k, ch in enumerate(JA_LINES[9])}


def ja_cues():
    """行 5 和行 9 往後偏 1.5 秒：時間窗裡只聽得到那行的 40%，辨識結果是空的。"""
    lines, _ = ja_truth()
    cl = [cue(s, e, t) for t, s, e in lines]
    for i in (5, 9):
        cl[i]["start"] += 1.5
        cl[i]["end"] += 1.5
    return cl


def install_fakes(truth_lines, units, wrong=None):
    fakes = {"asr": FakeASR(truth_lines), "aligner": FakeAligner(units, wrong), "loads": []}

    def load_qwen(bs, max_new_tokens):
        fakes["loads"].append(("qwen", bs))
        return fakes["asr"]

    def load_aligner():
        fakes["loads"].append(("aligner", None))
        return fakes["aligner"]

    asr_child.load_qwen_asr = load_qwen
    asr_child.load_aligner = load_aligner
    return fakes


# ---------- 2. 子程序 stage ----------

def test_child_stages_with_fake_models():
    tmp = Path(tempfile.mkdtemp(prefix="vs-health-"))
    orig = (asr_child.load_qwen_asr, asr_child.load_aligner, asr_child.emit)
    emitted = []
    asr_child.emit = lambda **kw: emitted.append(kw)
    try:
        lines, units = ja_truth()
        fakes = install_fakes(lines, units, ja_wrong())
        ramp_wav(tmp / "audio.wav", JA_SECONDS)
        (tmp / "cues.json").write_text(json.dumps(ja_cues(), ensure_ascii=False), encoding="utf-8")

        asr_child.dispatch(["verify", str(tmp), "cues.json", "verify.json", "ja", "4"])
        rec = {int(k): v for k, v in json.loads((tmp / "verify.json").read_text("utf-8")).items()}
        assert len(rec) == 12 and rec[5] == 0 and rec[9] == 0 and rec[4] == 1.0, rec
        assert emitted[0]["stage"] == "載入辨識模型" and emitted[-1] == {"p": 1.0, "stage": "檢查時間軸"}
        assert all("stage" in e for e in emitted)

        # 檢查點：再跑一次不會載入模型
        fakes["loads"].clear()
        asr_child.dispatch(["verify", str(tmp), "cues.json", "verify.json", "ja", "4"])
        assert fakes["loads"] == []

        asr_child.dispatch(["realign", str(tmp), "cues.json", "verify.json", "fix.json", "touched.json", "ja", "2"])
        touched = json.loads((tmp / "touched.json").read_text("utf-8"))
        assert touched["touched"] == [5, 9], touched
        assert [r[:2] for r in touched["regions"]] == [[5, 5], [9, 9]]
        fix = json.loads((tmp / "fix.json").read_text("utf-8"))
        assert (fix[5]["start"], fix[5]["end"]) == (21.0, 23.25), fix[5]
        assert (fix[9]["start"], fix[9]["end"]) == (38.5, 40.7), fix[9]
        assert (tmp / "fix.units.json").exists()
        assert any(e.get("stage") == "修正時間軸" for e in emitted)

        emitted.clear()
        asr_child.dispatch(["verify", str(tmp), "fix.json", "verify_fix.json", "ja", "4", "touched.json"])
        rec_fix = {int(k): v for k, v in json.loads((tmp / "verify_fix.json").read_text("utf-8")).items()}
        assert set(rec_fix) == {5, 9} and rec_fix[5] == 1.0 and rec_fix[9] == 0, rec_fix

        cl = json.loads((tmp / "cues.json").read_text("utf-8"))
        out, adopted, current = health.adopt(cl, fix, rec, rec_fix, touched["touched"])
        assert adopted == {5}
        assert (out[5]["start"], out[5]["end"]) == (21.0, 23.25)
        assert (out[9]["start"], out[9]["end"]) == (38.5, 40.5)
        assert current[5] == 1.0 and current[9] == 0
    finally:
        asr_child.load_qwen_asr, asr_child.load_aligner, asr_child.emit = orig
        shutil.rmtree(tmp, ignore_errors=True)


def test_realign_skips_tiny_region():
    """兩個錨點之間只差 0.4 秒：這種區段沒辦法對齊，直接跳過，也不用載入對齊模型。"""
    tmp = Path(tempfile.mkdtemp(prefix="vs-health-tiny-"))
    orig = (asr_child.load_qwen_asr, asr_child.load_aligner, asr_child.emit)
    asr_child.emit = lambda **kw: None
    try:
        truth = [("東京大阪", 1.0, 3.0), ("名古屋市", 10.0, 12.0), ("福岡博多", 3.4, 6.0)]
        units = [(ch, 10.0 + k * 0.4, 10.3 + k * 0.4) for k, ch in enumerate("東京大阪名古屋市福岡博多")]
        fakes = install_fakes(truth, units)
        ramp_wav(tmp / "audio.wav", 13.0)
        cl = [cue(1, 3, "東京大阪"), cue(3.1, 3.3, "名古屋市"), cue(3.4, 6, "福岡博多")]
        (tmp / "cues.json").write_text(json.dumps(cl, ensure_ascii=False), encoding="utf-8")

        asr_child.dispatch(["verify", str(tmp), "cues.json", "verify.json", "ja", "4"])
        rec = {int(k): v for k, v in json.loads((tmp / "verify.json").read_text("utf-8")).items()}
        assert (rec[0], rec[1], rec[2]) == (1.0, 0.0, 1.0), rec
        # 區段規劃得出來，只是太短不能對齊
        assert health.plan_regions(cl, rec, 13.0) == [(1, 1, 3.0, 3.4)]

        fakes["loads"].clear()
        asr_child.dispatch(["realign", str(tmp), "cues.json", "verify.json", "fix.json", "touched.json", "ja", "2"])
        touched = json.loads((tmp / "touched.json").read_text("utf-8"))
        assert touched == {"regions": [], "touched": []}, touched
        assert fakes["loads"] == [], fakes["loads"]
        assert json.loads((tmp / "fix.json").read_text("utf-8")) == cl

        # only 也可以直接給行號清單（不是含 touched 的物件）
        (tmp / "only.json").write_text(json.dumps([0, 2]), encoding="utf-8")
        asr_child.dispatch(["verify", str(tmp), "cues.json", "part.json", "ja", "4", "only.json"])
        assert set(json.loads((tmp / "part.json").read_text("utf-8"))) == {"0", "2"}
    finally:
        asr_child.load_qwen_asr, asr_child.load_aligner, asr_child.emit = orig
        shutil.rmtree(tmp, ignore_errors=True)


def test_window_clip_pads_short_clips():
    wav = np.zeros(SAMPLE_RATE * 2, dtype=np.float32)
    assert len(asr_child.window_clip(wav, 5.0, 6.0)) == SAMPLE_RATE // 2       # 超出結尾
    assert len(asr_child.window_clip(wav, 0.1, 1.0)) == int(1.3 * SAMPLE_RATE)  # 開頭不會變負的


# ---------- 3. 任務 ----------

class Env:
    """暫存資料夾裡的資料庫、字幕、工作資料夾，子程序改成直接呼叫 asr_child.dispatch。"""

    def __init__(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="vs-health-job-"))
        safepath.TEST_ROOTS.append(self.tmp)        # 檢查點作廢時要刪工作資料夾裡的檔案（測完拿掉）
        self.saved = (db.DB_PATH, db._conn, cue_mod.SUBS_DIR, jobs.WORK_DIR, jobs.run_child, gpu.pick_batch,
                      settings.installed_engines, asr_child.load_qwen_asr, asr_child.load_aligner, asr_child.emit,
                      jobs.model_installed, jobs.plan_speech)
        db.DB_PATH = self.tmp / "test.db"
        db.init()
        cue_mod.SUBS_DIR = self.tmp / "subs"
        jobs.WORK_DIR = self.tmp / "work"
        gpu.pick_batch = lambda table, label: 2
        # 不規劃常駐語音模型（不讀顯卡）：批次大小走舊的查表（上面的假 pick_batch）；常駐的部分在 test_speech.py
        jobs.plan_speech = lambda ctx, engine, health_on, speech_s: None
        settings.installed_engines = lambda language=None: ["qwen", "anime"]
        jobs.model_installed = lambda entry: True          # 模型都是假的，當成已經下載（沒下載模型的電腦也能跑）
        self.calls = []
        self.progress = []
        self.separate_seconds = 0.0
        env = self

        def fake_run_child(ctx, args, lo, hi, label, module="app.asr_child"):
            if module == "app.separate":
                # 假的人聲分離：直接寫出一段假音訊
                env.calls.append(("separate", round(lo, 4), round(hi, 4), label))
                ramp_wav(Path(args[1]), env.separate_seconds)
                return
            env.calls.append((args[0], round(lo, 4), round(hi, 4), label))

            def emit(**kw):
                ctx.progress(lo + (hi - lo) * float(kw.get("p", 0)), kw.get("stage"))
                env.progress.append(lo + (hi - lo) * float(kw.get("p", 0)))
            asr_child.emit = emit
            asr_child.dispatch(args)
            emit(p=1.0)
            ctx.check()

        jobs.run_child = fake_run_child

    def close(self):
        (db.DB_PATH, conn, cue_mod.SUBS_DIR, jobs.WORK_DIR, jobs.run_child, gpu.pick_batch,
         settings.installed_engines, asr_child.load_qwen_asr, asr_child.load_aligner, asr_child.emit,
         jobs.model_installed, jobs.plan_speech) = self.saved
        db._conn.close()
        db._conn = conn
        safepath.TEST_ROOTS.remove(self.tmp)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def media(self, seconds, lang):
        path = self.tmp / "video.mp4"
        path.write_bytes(b"fake")
        return db.add_media(title="test", source="local", path=str(path), duration=seconds, language=lang)

    def context(self, media_id, job_type, params):
        jid = db.add_job(media_id, job_type, params)
        ctx = jobs.JobContext(db.get_job(jid))
        ctx.workdir.mkdir(parents=True, exist_ok=True)
        return ctx


def test_handle_health_updates_track_and_translations():
    env = Env()
    try:
        lines, units = ja_truth()
        install_fakes(lines, units, ja_wrong())
        mid = env.media(JA_SECONDS, "ja")
        cl = ja_cues()
        cl[5]["ruby"] = [[0, 2, "こうべ"]]
        cl[5]["w"] = [[0, 2, "jm:1"]]
        cl[3]["chk"] = 0                                    # 舊的旗標要清掉
        tid = db.add_track(mid, "ja", "asr", "anime-whisper", len(cl))
        cue_mod.save_cues(tid, cl)
        # 逐行翻譯（有記 mode）、少一行的逐行翻譯（舊資料沒記 mode）、整句翻譯
        line_tr = [cue(c["start"], c["end"], f"譯{i}") for i, c in enumerate(cl)]
        t_line = db.add_track(mid, "zh-TW", "translation", "Hy-MT2-7B", len(line_tr), source_track_id=tid)
        db.update_track(t_line, mode="line")
        cue_mod.save_cues(t_line, line_tr)
        dropped = [c for i, c in enumerate(line_tr) if i != 2]
        t_drop = db.add_track(mid, "zh-TW", "translation", "Hy-MT2-7B", len(dropped), source_track_id=tid)
        cue_mod.save_cues(t_drop, dropped)
        sentence = [cue(cl[i]["start"], cl[i + 1]["end"], f"句{i}") for i in range(0, 12, 2)]
        t_sent = db.add_track(mid, "zh-TW", "translation", "Hy-MT2-7B", len(sentence), source_track_id=tid)
        db.update_track(t_sent, mode="sentence")
        cue_mod.save_cues(t_sent, sentence)

        # 檢查時間軸跟轉字幕一樣是語音模型的任務，排在一起（共用常駐程序裡的 Qwen3-ASR、對齊模型）
        assert jobs.model_key({"type": "health", "params": {"track_id": tid}}) == "speech"
        assert jobs.model_key({"type": "transcribe", "params": {"engine": "anime"}}) == "speech"
        assert "health" in jobs.GPU_TYPES and jobs.HANDLERS["health"] is jobs.handle_health

        ctx = env.context(mid, "health", {"track_id": tid})
        ramp_wav(ctx.workdir / "audio.wav", JA_SECONDS)    # 先放好音訊，不跑 ffmpeg
        jobs.handle_health(ctx)

        assert [c[0] for c in env.calls] == ["verify", "realign", "verify"], env.calls
        assert env.calls[0][3] == "Qwen3-ASR-1.7B" and env.calls[1][3] == "Qwen3-ForcedAligner-0.6B"
        assert env.progress == sorted(env.progress) and 0 <= env.progress[0] and env.progress[-1] <= 1

        out = cue_mod.load_cues(tid)
        assert (out[5]["start"], out[5]["end"]) == (21.0, 23.25), out[5]
        assert out[5]["ruby"] == [[0, 2, "こうべ"]] and out[5]["w"] == [[0, 2, "jm:1"]]
        assert out[9].get("chk") == 0 and (out[9]["start"], out[9]["end"]) == (38.5, 40.5)
        assert [i for i, c in enumerate(out) if "chk" in c] == [9]
        for a, b in zip(out, out[1:]):
            assert a["end"] <= b["start"]

        bak = json.loads(health.backup_path(tid).read_text("utf-8"))
        assert bak[5]["start"] == 22.5 and bak[3].get("chk") == 0          # 備份是原本的檔案

        tr = cue_mod.load_cues(t_line)
        assert [(c["start"], c["end"]) for c in tr] == [(c["start"], c["end"]) for c in out]
        tr = cue_mod.load_cues(t_drop)
        assert len(tr) == 11 and (tr[4]["start"], tr[4]["end"]) == (21.0, 23.25)
        assert health.backup_path(t_line).exists()
        assert cue_mod.load_cues(t_sent) == sentence

        h = json.loads(db.get_track(tid)["health"])
        assert (h["checked"], h["before_bad"], h["fixed"], h["after_bad"], h["bad"]) == (12, 2, 1, 1, [9]), h
        assert h["v"] == health.HEALTH_VERSION and h["synced_translations"] == 2
        assert [x["id"] for x in h["skipped_translations"]] == [t_sent]
        assert ctx.result == {"track_id": tid, "fixed": 1, "after_bad": 1}

        # 再檢查一次：時間不再變動，備份不覆蓋，只剩行 9 標著
        env.calls.clear()
        env.progress.clear()
        ctx = env.context(mid, "health", {"track_id": tid})
        ramp_wav(ctx.workdir / "audio.wav", JA_SECONDS)
        jobs.handle_health(ctx)
        again = cue_mod.load_cues(tid)
        assert [(c["start"], c["end"]) for c in again] == [(c["start"], c["end"]) for c in out]
        assert json.loads(health.backup_path(tid).read_text("utf-8"))[5]["start"] == 22.5
        h2 = json.loads(db.get_track(tid)["health"])
        assert (h2["before_bad"], h2["fixed"], h2["after_bad"]) == (1, 0, 1), h2
        assert h2["skipped_translations"] == [] and h2["at"] >= h["at"]
    finally:
        env.close()


def test_run_health_two_rounds_and_resume():
    env = Env()
    orig_rounds = health.HEALTH_ROUNDS
    try:
        lines, units = ja_truth()
        fakes = install_fakes(lines, units, ja_wrong())
        mid = env.media(JA_SECONDS, "ja")
        ctx = env.context(mid, "health", {"track_id": "x"})
        ramp_wav(ctx.workdir / "audio.wav", JA_SECONDS)

        health.HEALTH_ROUNDS = 2
        new, summary = jobs.run_health(ctx, ja_cues(), "ja", 0.0, 1.0)
        # 第二輪：行 5 採用後時間跟量的時候一樣，全部沿用 recall，不用載入模型；行 9 還是修不好就停
        assert [c[0] for c in env.calls] == ["verify", "realign", "verify", "verify", "realign", "verify"], env.calls
        assert [round(c[1], 3) for c in env.calls] == [0.0, 0.294, 0.412, 0.5, 0.794, 0.912], env.calls
        assert [x for x in fakes["loads"] if x[0] == "qwen"] == [("qwen", 8)] * 3, fakes["loads"]
        r2 = json.loads((ctx.workdir / health.round_files(2)["verify"]).read_text("utf-8"))
        assert len(r2) == 12 and r2["5"] == 1.0
        assert (summary["before_bad"], summary["fixed"], summary["after_bad"], summary["rounds"]) == (2, 1, 1, 2)
        assert (new[5]["start"], new[5]["end"]) == (21.0, 23.25)

        # 同一個工作資料夾再跑一次（例如存檔前程式被關掉）：全部從檢查點讀，不載入任何模型
        fakes["loads"].clear()
        again, summary2 = jobs.run_health(ctx, ja_cues(), "ja", 0.0, 1.0)
        assert fakes["loads"] == [], fakes["loads"]
        assert again == new and summary2["bad"] == summary["bad"]

        # 字幕跟檢查點不同：檢查點作廢重新檢查
        changed = ja_cues()
        changed[1]["end"] = 6.5
        fakes["loads"].clear()
        jobs.run_health(ctx, changed, "ja", 0.0, 1.0)
        assert ("qwen", 8) in fakes["loads"]
    finally:
        health.HEALTH_ROUNDS = orig_rounds
        env.close()


def test_handle_health_nothing_to_fix_skips_aligner():
    env = Env()
    try:
        lines, units = ja_truth()
        fakes = install_fakes(lines, units)
        mid = env.media(JA_SECONDS, "ja")
        cl = [cue(s, e, t) for t, s, e in lines]
        tid = db.add_track(mid, "ja", "asr", "Qwen3-ASR-1.7B", len(cl))
        cue_mod.save_cues(tid, cl)
        ctx = env.context(mid, "health", {"track_id": tid})
        ramp_wav(ctx.workdir / "audio.wav", JA_SECONDS)
        jobs.handle_health(ctx)
        assert [c[0] for c in env.calls] == ["verify"], env.calls
        assert ("aligner", None) not in fakes["loads"]
        assert not health.backup_path(tid).exists()
        h = json.loads(db.get_track(tid)["health"])
        assert (h["fixed"], h["after_bad"], h["bad"]) == (0, 0, [])
    finally:
        env.close()


def test_handle_health_song_track_uses_vocals():
    """歌曲模式轉的字幕：事後檢查先分離人聲再檢查，跟轉字幕當下用同一種音訊。"""
    env = Env()
    saved = (jobs.model_installed, jobs.media.extract_audio, gpu.ensure_free)
    try:
        lines, units = ja_truth()
        install_fakes(lines, units, ja_wrong())
        env.separate_seconds = JA_SECONDS
        mid = env.media(JA_SECONDS, "ja")
        cl = ja_cues()
        tid = db.add_track(mid, "ja", "asr", "Qwen3-ASR-1.7B" + jobs.SONG_LABEL, len(cl))
        cue_mod.save_cues(tid, cl)
        jobs.model_installed = lambda entry: True
        gpu.ensure_free = lambda need_mb, label: None

        def no_mix(*args, **kw):
            raise AssertionError("歌曲模式的字幕不能拿原音檢查")
        jobs.media.extract_audio = no_mix

        ctx = env.context(mid, "health", {"track_id": tid})
        jobs.handle_health(ctx)
        assert [c[0] for c in env.calls] == ["separate", "verify", "realign", "verify"], env.calls
        assert env.calls[0][1:3] == (0, 0.12) and env.calls[1][1] == 0.12, env.calls
        out = cue_mod.load_cues(tid)
        assert (out[5]["start"], out[5]["end"]) == (21.0, 23.25), out[5]

        # 人聲分離模型不在了：說清楚原因，不改用原音
        jobs.model_installed = lambda entry: False
        env.calls.clear()
        ctx = env.context(mid, "health", {"track_id": tid})
        try:
            jobs.handle_health(ctx)
        except RuntimeError as e:
            assert "人聲" in str(e), e
        else:
            raise AssertionError("separator missing should fail")
        assert env.calls == []
    finally:
        jobs.model_installed, jobs.media.extract_audio, gpu.ensure_free = saved
        env.close()


def test_run_child_log_stays_in_workdir():
    """子程序第一個參數是檔案路徑時（人聲分離），錯誤紀錄還是寫在工作資料夾，不寫到影片旁邊。
    用一個不存在的模組當子程序，馬上失敗，不會載入任何模型。"""
    tmp = Path(tempfile.mkdtemp(prefix="vs-health-child-"))
    try:
        video = tmp / "videos" / "song.mp4"
        video.parent.mkdir()
        video.write_bytes(b"fake")
        ctx = jobs.JobContext({"id": "child-log-test", "media_id": None, "params": {}})
        ctx.workdir = tmp / "work"
        ctx.workdir.mkdir()
        try:
            jobs.run_child(ctx, [str(video), str(ctx.workdir / "vocals.wav")], 0, 1, "測試子程序",
                           module="app._no_such_child")
        except RuntimeError as e:
            assert "測試子程序" in str(e), e
        else:
            raise AssertionError("missing module should fail")
        assert (ctx.workdir / "_no_such_child.err.log").exists()
        assert list(video.parent.iterdir()) == [video]
    finally:
        jobs.gpu_state.update(job_id=None, model=None, stage=None)
        shutil.rmtree(tmp, ignore_errors=True)


def test_handle_health_rejects_translation_track():
    env = Env()
    try:
        mid = env.media(40.0, "ja")
        tid = db.add_track(mid, "zh-TW", "translation", "Hy-MT2-7B", 1)
        ctx = env.context(mid, "health", {"track_id": tid})
        try:
            jobs.handle_health(ctx)
        except RuntimeError as e:
            assert "語音辨識" in str(e)
        else:
            raise AssertionError("translation track should be rejected")
    finally:
        env.close()


# 英文轉字幕：8 句，每句 3 個單字，第 k 句真正在 [4k+1, 4k+3]；對齊結果把第 3 句往後推 1.6 秒
EN_WORDS = ["alpha bravo charlie", "delta echo foxtrot", "golf hotel india", "juliet kilo lima",
            "mike november oscar", "papa quebec romeo", "sierra tango uniform", "victor whiskey xray"]


def en_truth():
    lines, units = [], []
    for k, sent in enumerate(EN_WORDS):
        s = 4 * k + 1.0
        words = sent.split()
        text = " ".join(w.capitalize() if j == 0 else w for j, w in enumerate(words)) + "."
        lines.append((text, s, s + 2.0))
        for j, w in enumerate(words):
            units.append((w.capitalize() if j == 0 else w, round(s + j * 0.66, 3), round(s + j * 0.66 + 0.6, 3)))
    return lines, units


def run_transcribe(env, health_on: bool, break_health: bool = False):
    lines, units = en_truth()
    install_fakes(lines, units)
    settings.save({"health_check": health_on})
    mid = env.media(40.0, "en")
    ctx = env.context(mid, "transcribe", {"language": "en", "engine": "anime", "sensitive": False})
    ramp_wav(ctx.workdir / "audio.wav", 40.0)
    (ctx.workdir / "chunks.json").write_text(json.dumps([[0.0, 40.0]]), encoding="utf-8")
    shifted = [[t, s + 1.6, e + 1.6] if t in ("Golf", "hotel", "india") else [t, s, e] for t, s, e in units]
    real_dispatch = asr_child.dispatch

    def dispatch(args):
        # 轉字幕本身的辨識和對齊直接寫結果（這裡只測檢查補正的串接）
        wd = Path(args[1])
        if args[0] == "asr":
            (wd / "asr.json").write_text(json.dumps({"0": " ".join(t for t, _, _ in lines)}), encoding="utf-8")
        elif args[0] == "align":
            (wd / "align.json").write_text(json.dumps({"0": shifted}), encoding="utf-8")
        elif break_health:
            raise RuntimeError("假裝檢查時顯存不夠")
        else:
            real_dispatch(args)

    asr_child.dispatch = dispatch
    try:
        jobs.handle_transcribe(ctx)
    finally:
        asr_child.dispatch = real_dispatch
    return ctx


def test_transcribe_runs_health_check():
    env = Env()
    try:
        ctx = run_transcribe(env, True)
        assert [c[:3] for c in env.calls] == [
            ("asr", 0.1, 0.68), ("align", 0.68, 0.78), ("verify", 0.78, 0.88),
            ("realign", 0.88, 0.92), ("verify", 0.92, 0.95)], env.calls
        assert env.calls[2][3] == "Qwen3-ASR-1.7B"      # 原本用 anime-whisper，檢查一樣用 Qwen3-ASR
        tid = ctx.result["track_id"]
        out = cue_mod.load_cues(tid)
        assert len(out) == 8 and out[2]["text"] == "Golf hotel india."
        assert (out[2]["start"], out[2]["end"]) == (9.0, 11.22), out[2]
        assert not any("chk" in c for c in out)
        h = json.loads(db.get_track(tid)["health"])
        assert (h["before_bad"], h["fixed"], h["after_bad"]) == (1, 1, 0), h
    finally:
        env.close()


def test_transcribe_without_health_check():
    env = Env()
    try:
        ctx = run_transcribe(env, False)
        assert [c[:3] for c in env.calls] == [("asr", 0.1, 0.8), ("align", 0.8, 0.95)], env.calls
        t = db.get_track(ctx.result["track_id"])
        assert t["health"] is None
        assert cue_mod.load_cues(t["id"])[2]["start"] == 10.6
    finally:
        env.close()


def test_transcribe_survives_health_failure():
    """檢查途中失敗（例如顯存不夠）時，字幕照常存檔，只是沒有檢查摘要。"""
    env = Env()
    quiet = logging.getLogger("jobs")           # 這個測試故意讓檢查失敗，不用印出堆疊
    level = quiet.level
    quiet.setLevel(logging.CRITICAL)
    try:
        ctx = run_transcribe(env, True, break_health=True)
        t = db.get_track(ctx.result["track_id"])
        assert t["health"] is None
        out = cue_mod.load_cues(t["id"])
        assert len(out) == 8 and out[2]["start"] == 10.6      # 保留對齊器給的（偏掉的）時間
        assert not any("chk" in c for c in out)
    finally:
        quiet.setLevel(level)
        env.close()


def test_settings_default():
    assert settings.DEFAULTS["health_check"] is True


if __name__ == "__main__":
    started = time.time()
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
    print(f"all passed in {time.time() - started:.1f}s")
