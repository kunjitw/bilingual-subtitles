"""字幕時間軸健康檢查與局部補正：演算法部分（只用 CPU，可以單元測試）。

做法（開發時用實驗腳本在實際影片上測過有效）：
1. 用 Qwen3-ASR 重新辨識每一行字幕的時間窗 [start-0.3, end+0.3]，
   算 recall = 字幕文字（去標點空白、小寫）依序出現在辨識結果裡的比例。
2. recall 高的行當錨點（時間窗長得不合理、或跟前一行同時開始的不算）；
   對不上的行往前後找最近的錨點，錨點之間的行組成一個區段。
3. 只拿區段的聲音、對那幾行的文字重新強制對齊，依每行保留的字元數把對齊結果分回各行，
   結束時間跟 cues.finalize 一樣留閱讀時間。
4. 被重新對齊的行再檢查一次，recall 至少進步 MIN_GAIN 才採用新時間。
5. 整理時間：依序、不重疊、每行 end > start。

模型在 app/asr_child.py 的 verify / realign 階段裡跑（平常在常駐語音程序 app/speech_worker.py 裡，
跟轉字幕共用已經載入的模型），流程串接在 app/jobs.py 的 run_health。
"""
import difflib
import re
import time
from pathlib import Path

from . import cues as cue_mod

# 演算法或門檻改了就改版本：前端和「檢查所有尚未檢查的字幕」用它判斷要不要重新檢查
# 2：錨點排除時間可疑的行、跨行的斷詞不再讓後面的行偏掉、採用規則不被對不上的鄰行卡住、修正的行留閱讀時間
HEALTH_VERSION = "2"
# 補正輪數：第二輪的效果還在驗證，先跑一輪
HEALTH_ROUNDS = 1

PAD_S = 0.3              # 檢查時間窗前後各多聽一點
ANCHOR_RECALL = 0.8      # 錨點：recall 夠高
ANCHOR_MIN_CHARS = 2     #       而且至少 2 個字（太短的行容易碰巧對上）
BAD_RECALL = 0.6         # 對不上：recall 太低
BAD_MIN_CHARS = 3        #         而且至少 3 個字（一兩個字的行辨識不穩，不當成問題）
MIN_GAIN = 0.1           # 新時間的 recall 至少要進步這麼多才採用
REGION_MAX_S = 90.0      # 區段超過這個長度就縮到那幾行前後 REGION_MARGIN_S 秒
REGION_MARGIN_S = 10.0
REGION_MIN_S = 0.5       # 區段太短沒辦法對齊
LONG_MIN_S = 4.0         # 一行的時間窗超過 max(LONG_MIN_S, 字數 × LONG_PER_CHAR_S) 就算長得不合理：
LONG_PER_CHAR_S = 0.6    # 窗大到什麼都聽得到，recall 高也不代表時間對（現有資料 1 萬多行裡約 35 行）
SAME_START_S = 0.01      # 跟前一行的開始時間差不到這麼多，是對齊器把好幾行擠在同一個時間
# Qwen3-ForcedAligner 官方流程一次最多送 180 秒（qwen_asr 的 MAX_FORCE_ALIGN_INPUT_SECONDS），更長的區段跳過
ALIGN_MAX_S = 180.0
FIX_MIN_S = 0.3          # 重新對齊後每行至少這麼長
ORDER_GAP_S = 0.2        # 採用新時間時，跟前後行的開始時間至少差這麼多，避免順序亂掉
MIN_CUE_S = 0.05         # 整理時間時每行最短長度
MIN_CLIP_S = 0.5         # 送進模型的片段最短長度（太短的會補靜音，見 asr_child.window_clip）
ALIGN_CHUNK_S = 60.0     # 對齊器的 batch 表是用 60 秒的段落量的
ASR_CHUNK_S = 60.0       # 辨識的 batch 表也是用 60 秒的段落量的
VERIFY_SCALE = 4         # 檢查的片段是一行字幕，比辨識的段落短很多，同樣顯存放得下幾倍的筆數
VERIFY_BATCH_MAX = 32    # 一批再多也不超過這個數，免得一批跑太久都沒有進度
# 估計值：每 100 行約 20 秒（檢查全部行加上補正），給前端顯示預估時間用
SECONDS_PER_LINE = 0.2

_NORM_RE = re.compile(r"[\s、。，,.!?！？「」『』（）()…・〜~ー-]")


# ---------- 比對 ----------

def norm(text: str) -> str:
    """去掉空白與常見標點、轉小寫，中日英都適用。"""
    return _NORM_RE.sub("", text or "").lower()


def recall(cue_text: str, heard: str) -> float:
    """字幕文字有多少比例依序出現在辨識結果裡（difflib 的 matching blocks 總長 / 字幕字數）。"""
    a, b = norm(cue_text), norm(heard)
    if not a:
        return 1.0
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    return sum(m.size for m in sm.get_matching_blocks()) / len(a)


def is_anchor(text: str, rec: float | None) -> bool:
    return rec is not None and rec >= ANCHOR_RECALL and len(norm(text)) >= ANCHOR_MIN_CHARS


def is_bad(text: str, rec: float | None) -> bool:
    return rec is not None and rec < BAD_RECALL and len(norm(text)) >= BAD_MIN_CHARS


def odd_timing(cues: list[dict], i: int) -> bool:
    """這行的時間本身就可疑：時間窗比文字需要的長很多，或跟前一行同一個時間開始。"""
    c = cues[i]
    if c["end"] - c["start"] > max(LONG_MIN_S, len(norm(c["text"])) * LONG_PER_CHAR_S):
        return True
    return i > 0 and abs(c["start"] - cues[i - 1]["start"]) < SAME_START_S


def strong_anchor(cues: list[dict], rec: dict[int, float], i: int) -> bool:
    """可以當區段邊界的錨點：對得上，而且時間不可疑（窗太長時 recall 高只代表窗夠大）。"""
    return is_anchor(cues[i]["text"], rec.get(i)) and not odd_timing(cues, i)


def settled(cues: list[dict], rec: dict[int, float], i: int) -> bool:
    """原本的時間可以拿來擋鄰行採用新時間：對得上、字數夠判斷，時間本身也不可疑。"""
    r = rec.get(i)
    return (r is not None and r >= BAD_RECALL and len(norm(cues[i]["text"])) >= BAD_MIN_CHARS
            and not odd_timing(cues, i))


def clip_window(start: float, end: float, total: float) -> tuple[float, float]:
    """一行字幕要拿去辨識的時間窗（前後各加 PAD_S，限制在音訊範圍內）。"""
    return max(0.0, start - PAD_S), min(total, end + PAD_S)


# ---------- 規劃區段 ----------

def plan_regions(cues: list[dict], rec: dict[int, float], total: float) -> list[tuple[int, int, float, float]]:
    """找出要重新對齊的區段：兩個錨點之間只要有一行對不上，就把錨點之間的行整段重對。

    回傳 [(lo, hi, start, end)]：lo..hi 是區段裡的行（含），start/end 是要拿去對齊的聲音範圍。
    """
    n = len(cues)

    def strong(i):
        return strong_anchor(cues, rec, i)

    regions, i = [], 0
    while i < n:
        if not is_bad(cues[i]["text"], rec.get(i)):
            i += 1
            continue
        lo = i
        while lo > 0 and not strong(lo - 1):
            lo -= 1
        hi = i
        while hi + 1 < n and not strong(hi + 1):
            hi += 1
        start = cues[lo - 1]["end"] if lo > 0 else 0.0
        end = cues[hi + 1]["start"] if hi + 1 < n else total
        # 區段太長時縮到這幾行附近
        if end - start > REGION_MAX_S:
            start = max(start, cues[lo]["start"] - REGION_MARGIN_S)
            end = min(end, cues[hi]["end"] + REGION_MARGIN_S)
        regions.append((lo, hi, round(start, 3), round(end, 3)))
        i = hi + 1
    return regions


def alignable(region) -> bool:
    return REGION_MIN_S <= region[3] - region[2] <= ALIGN_MAX_S


def region_key(region) -> str:
    lo, hi, s, e = region
    return f"{lo}-{hi}-{s:.3f}-{e:.3f}"


def region_text(cues: list[dict], lo: int, hi: int, lang: str) -> str:
    """送去對齊的文字。每種語言都用空白隔開各行：日文行尾的「。、」已經被 tidy_text 去掉，
    直接相連時 nagisa 常把上一行結尾和下一行開頭斷成同一個詞（「したいな」+「いろんな」→「ない」），
    中文和日文裡的英文也會黏在一起（the + most → themost）。對齊器清理時會把空白丟掉。"""
    return " ".join(cues[k]["text"] for k in range(lo, hi + 1))


def batch_by_length(items: list, length, budget: float, count_max: int | None = None) -> list[list]:
    """照長度排序後分批。

    同一批裡短的片段會被補齊到最長那筆的長度，所以限制的是（最長的 × 筆數）不超過 budget 秒；
    count_max 是額外的筆數上限。單獨一筆就超過 budget 的自己一批。
    """
    out, cur = [], []
    for it in sorted(items, key=length):
        if cur and ((count_max and len(cur) >= count_max) or (len(cur) + 1) * length(it) > budget):
            out.append(cur)
            cur = []
        cur.append(it)
    if cur:
        out.append(cur)
    return out


def align_batches(regions: list, bs: int) -> list[list]:
    """區段長短不一，一批的長度不超過 bs × 60 秒。"""
    return batch_by_length(regions, lambda r: r[3] - r[2], max(1, bs) * ALIGN_CHUNK_S)


def verify_batch_size(bs: int) -> int:
    """檢查時一次辨識幾行：辨識模型一次幾段 × VERIFY_SCALE，上限 VERIFY_BATCH_MAX。"""
    return max(1, min(VERIFY_BATCH_MAX, bs * VERIFY_SCALE))


def verify_batches(idx: list[int], cues: list[dict], total: float, bs: int) -> list[list[int]]:
    """把要檢查的行分批。大部分的行只有幾秒，但偶爾有很長的行（對齊失敗留下的），
    一批會被補齊到最長那筆的長度，所以除了筆數也要限制總長度。"""
    def secs(i):
        s, e = clip_window(cues[i]["start"], cues[i]["end"], total)
        return max(MIN_CLIP_S, e - s)

    return batch_by_length(idx, secs, max(1, bs) * ASR_CHUNK_S / VERIFY_SCALE, max(1, bs))


# ---------- 對齊結果分回各行 ----------

def distribute(units: list, texts: list[str]) -> list[tuple[float, float] | None]:
    """units 是對齊器的 [文字, 開始, 結束]（已去標點），照累計的字元位置分給各行：
    回傳每行 (第一個字開始, 最後一個字結束)。

    對齊器的一個詞偶爾會跨兩行，這時照字數比例切開它的時間，每行分到的還是剛好自己那幾個字，
    不會多吃下一行的字、讓後面每一行跟著往後偏。
    分不到任何單位的行（例如整行都是標點）回傳 None。
    """
    out, p, used = [], 0, 0  # used：units[p] 已經分給前面的行幾個字
    for text in texts:
        need = sum(1 for ch in text if cue_mod._kept(ch))
        first = last = None
        while need > 0 and p < len(units):
            word, s, e = units[p][0], float(units[p][1]), float(units[p][2])
            size = len(word)
            take = min(need, size - used)
            if take > 0:
                first = s + (e - s) * used / size if first is None else first
                last = s + (e - s) * (used + take) / size
                need -= take
                used += take
            if used >= size:
                p, used = p + 1, 0
        if first is None:
            out.append(None)
        else:
            out.append((round(first, 3), round(max(last, first + FIX_MIN_S), 3)))
    return out


def apply_regions(cues: list[dict], regions: list, units_by_key: dict) -> tuple[list[dict], list[int]]:
    """把每個區段的對齊結果套回字幕，回傳 (重新對齊後的字幕, 改到的行)。

    結束時間跟 cues.finalize 用同一套顯示規則（閱讀時間、最短顯示時間、不蓋到下一行），
    重新檢查時的時間窗才跟原本的檢查同樣條件。
    分不到單位的行（整行都是標點）不留在舊時間，擺到前後已對齊的行中間，免得整理時間時把已對齊的行推走。
    """
    n = len(cues)
    fixed = [dict(c) for c in cues]
    touched = []
    for region in regions:
        units = units_by_key.get(region_key(region))
        if units is None:
            continue
        lo, hi, r_start, r_end = region
        rows = range(lo, hi + 1)
        spans = dict(zip(rows, distribute(units, [cues[k]["text"] for k in rows])))
        aligned = [k for k in rows if spans[k]]
        if not aligned:
            continue
        after = cues[hi + 1]["start"] if hi + 1 < n else r_end
        for idx, k in enumerate(aligned):
            start, last = spans[k]
            nxt = spans[aligned[idx + 1]][0] if idx + 1 < len(aligned) else after
            fixed[k]["start"], fixed[k]["end"] = start, cue_mod.display_end(start, last, nxt)
            touched.append(k)
        for k in rows:
            if spans[k]:
                continue
            gap_start = fixed[k - 1]["end"] if k > lo else r_start
            gap_end = next((spans[j][0] for j in range(k + 1, hi + 1) if spans[j]), after)
            start = min(gap_start, gap_end)
            fixed[k]["start"] = round(start, 3)
            fixed[k]["end"] = round(min(start + max(0.0, cues[k]["end"] - cues[k]["start"]), gap_end), 3)
    tidy_times(fixed)
    return fixed, sorted(set(touched))


# ---------- 採用與整理 ----------

def tidy_times(cues: list[dict]) -> list[dict]:
    """依序、不重疊、每行 end > start。只動需要動的行（原地修改，也回傳同一個 list）。"""
    for k, c in enumerate(cues):
        c["start"], c["end"] = round(float(c["start"]), 3), round(float(c["end"]), 3)
        if k > 0:
            prev = cues[k - 1]
            if c["start"] < prev["start"]:  # 保險：正常資料不會發生
                c["start"] = prev["start"]
            if prev["end"] > c["start"]:
                prev["end"] = round(max(c["start"], prev["start"] + MIN_CUE_S), 3)
                if prev["end"] > c["start"]:
                    c["start"] = prev["end"]
        if c["end"] < c["start"] + MIN_CUE_S:
            c["end"] = round(c["start"] + MIN_CUE_S, 3)
    return cues


def adopt(cues: list[dict], fixed: list[dict], rec: dict[int, float], rec_fix: dict[int, float],
          touched: list[int]) -> tuple[list[dict], set[int], dict[int, float]]:
    """重新對齊的行比原本更對得上（進步至少 MIN_GAIN）才採用新時間。

    採用後開始時間要跟前後行保持順序；前後行也要改的話，多掃幾次讓它們有機會一起採用。
    但原本的時間就不可信的鄰行（重新對齊過卻對不上、字太少沒辦法判斷、時間本身可疑，
    或整行都是標點）不拿來擋：對齊器把一串行擠在一起時，只要中間一行辨識寫法不同，
    就會讓前面修好的行全部不能採用。這種行採用完再處理：順序不合的，
    放得進前後行中間就用它重新對齊的時間，放不進去就夾在前後行之間，照樣依 recall 標記。
    回傳 (整理後的字幕, 採用的行, 每行目前的 recall)。
    """
    n = len(cues)
    out = [dict(c) for c in cues]
    current = dict(rec)
    cand = {i for i in touched
            if i in rec_fix and rec.get(i) is not None and rec_fix[i] - rec[i] >= MIN_GAIN - 1e-9}
    no_text = {i for i, c in enumerate(cues) if not any(cue_mod._kept(ch) for ch in c["text"])}
    loose = {i for i in set(touched) | no_text if not settled(cues, rec, i)}
    adopted: set[int] = set()

    def neighbor(i, step):
        """往前或往後最近一個能拿來比順序的行（跳過還沒採用、時間又不可信的行）。"""
        j = i + step
        while 0 <= j < n and j in loose and j not in adopted:
            j += step
        return j if 0 <= j < n else None

    progress = True
    while progress:
        progress = False
        for i in sorted(cand - adopted):
            s = fixed[i]["start"]
            p, q = neighbor(i, -1), neighbor(i, 1)
            if p is not None and s < out[p]["start"] + ORDER_GAP_S:
                continue
            if q is not None and s + ORDER_GAP_S > out[q]["start"]:
                continue
            out[i]["start"], out[i]["end"] = s, fixed[i]["end"]
            current[i] = rec_fix[i]
            adopted.add(i)
            progress = True

    changed = set(adopted)
    for i in sorted(loose - adopted):
        prev = out[i - 1] if i > 0 else None
        q = neighbor(i, 1)
        lo_t = prev["start"] if prev else 0.0
        next_start = out[q]["start"] if q is not None else float("inf")
        # 開始時間至少要比後面那行早 MIN_CUE_S，整理時間時才不會把後面的行往後推
        hi_t = next_start - MIN_CUE_S
        s = out[i]["start"]
        covered = prev is not None and i - 1 in changed and prev["end"] > s + 1e-9
        if lo_t <= s <= hi_t and not covered:
            continue
        f = fixed[i]
        if lo_t <= f["start"] <= hi_t:
            out[i]["start"], out[i]["end"] = f["start"], f["end"]
            if i in rec_fix:
                current[i] = rec_fix[i]
            if i in cand:
                adopted.add(i)
        else:
            # 被擠在一起的行原本只有 0.05 秒，照 cues.finalize 至少顯示 MIN_CUE_S
            dur = max(cue_mod.MIN_CUE_S, out[i]["end"] - s)
            start = max(lo_t, min(prev["end"] if prev else max(0.0, hi_t - dur), hi_t))
            out[i]["start"], out[i]["end"] = round(start, 3), round(min(start + dur, next_start), 3)
            if i not in no_text:
                current[i] = 0.0  # 位置是推出來的，沒有檢查過，照對不上標記
        changed.add(i)
    tidy_times(out)
    return out, adopted, current


def carry_over(before: list[dict], fixed: list[dict], after: list[dict], current: dict[int, float],
               rec: dict[int, float], rec_fix: dict[int, float]) -> dict[int, float]:
    """下一輪可以沿用的 recall：量的時候的時間跟整理後的時間完全相同的行。"""
    out = {}
    for i, c in enumerate(after):
        if i not in current:
            continue

        def same(x):
            return x["start"] == c["start"] and x["end"] == c["end"]

        if (i in rec_fix and current[i] == rec_fix[i] and same(fixed[i])) or \
                (i in rec and current[i] == rec[i] and same(before[i])):
            out[i] = current[i]
    return out


# ---------- 摘要與寫回 ----------

def slim(cues: list[dict]) -> list[dict]:
    """檢查只需要時間和文字，寫進工作資料夾的檔案也比較小。"""
    return [{"start": round(float(c["start"]), 3), "end": round(float(c["end"]), 3), "text": c["text"]} for c in cues]


def summarize(cues: list[dict], first_rec: dict[int, float], final_rec: dict[int, float], fixed: set[int],
              rounds: int) -> dict:
    bad = [i for i, c in enumerate(cues) if is_bad(c["text"], final_rec.get(i))]
    return {
        "v": HEALTH_VERSION,
        "at": round(time.time(), 3),
        "rounds": rounds,
        "checked": len(cues),
        "before_bad": sum(1 for i, c in enumerate(cues) if is_bad(c["text"], first_rec.get(i))),
        "fixed": len(fixed),
        "after_bad": len(bad),
        "bad": bad,
    }


def times_changed(old: list[dict], new: list[dict]) -> bool:
    return any(a["start"] != b["start"] or a["end"] != b["end"] for a, b in zip(old, new))


def apply_times(target: list[dict], new: list[dict], bad: list[int]) -> bool:
    """把新時間寫回完整的字幕（ruby、w 等欄位不動），清掉舊的 chk 再依結果標上。回傳有沒有改變。"""
    bad_set = set(bad)
    changed = False
    for i, (c, n) in enumerate(zip(target, new)):
        if c["start"] != n["start"] or c["end"] != n["end"]:
            c["start"], c["end"] = n["start"], n["end"]
            changed = True
        if i in bad_set:
            if c.get("chk") != 0:
                c["chk"] = 0
                changed = True
        elif "chk" in c:
            del c["chk"]
            changed = True
    return changed


def _same_time(a: dict, b: dict) -> bool:
    return abs(a["start"] - b["start"]) <= 0.01 and abs(a["end"] - b["end"]) <= 0.01


def _ambiguous(src: list[dict], tr: list[dict]) -> bool:
    """原文有連續幾行時間相同，譯文裡同樣時間的行卻比較少：少掉的是哪一行分不出來。"""
    i = 0
    while i < len(src):
        j = i + 1
        while j < len(src) and _same_time(src[j], src[i]):
            j += 1
        if j - i > 1 and 0 < sum(1 for t in tr if _same_time(t, src[i])) < j - i:
            return True
        i = j
    return False


def sync_translation(old_src: list[dict], new_src: list[dict], tr: list[dict]) -> list[dict] | None:
    """逐行翻譯的每一行時間都跟原文某一行相同，原文時間改了就跟著改。

    行數相同時逐行對應；逐行翻譯遇到空白譯文會少幾行，這時照原本的時間依序對回原文。
    有任何一行對不回原文（整句翻譯會跨行）就回傳 None，不動這條翻譯。
    譯文比原文少、原文又有好幾行時間完全相同（對齊器擠在一起的）時，對應關係不確定，也回傳 None。
    """
    if len(old_src) != len(new_src):
        return None
    if len(tr) < len(old_src) and _ambiguous(old_src, tr):
        return None
    idx, j = [], 0
    for c in tr:
        while j < len(old_src) and not _same_time(old_src[j], c):
            j += 1
        if j == len(old_src):
            return None
        idx.append(j)
        j += 1
    out = [dict(c) for c in tr]
    for c, i in zip(out, idx):
        c["start"], c["end"] = new_src[i]["start"], new_src[i]["end"]
    return out


def backup_path(track_id: str) -> Path:
    """第一次修正時間前的字幕備份：data/subs/<track_id>.bak.json。"""
    return cue_mod.cue_path(track_id).with_suffix(".bak.json")


def round_files(r: int) -> dict[str, str]:
    """第 r 輪在工作資料夾裡用到的檔名。"""
    p = f"health{r}_"
    files = {"cues": p + "cues.json", "verify": p + "verify.json", "fix": p + "fix.json",
             "touched": p + "touched.json", "verify_fix": p + "verify_fix.json"}
    files["units"] = units_name(files["fix"])
    return files


def units_name(fix_name: str) -> str:
    """realign 子程序存對齊結果檢查點的檔名。"""
    return Path(fix_name).stem + ".units.json"
