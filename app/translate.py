"""翻譯：以 llama-server 載入 GGUF（所有層都放 GPU，放不下就報錯），逐句或分批翻譯。

輸出統一先要簡體中文，再由 OpenCC s2twp 轉成台灣繁體與台灣用語。
"""
import json
import re
import socket
import subprocess
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

from . import syscheck
from .config import LLAMA_SERVER, SAKURA_STYLE, TRANSLATORS, child_env
from .cues import tidy_text, to_taiwan

SAKURA_SYSTEM = ("你是一个轻小说翻译模型，可以流畅通顺地以日本轻小说的风格将日文翻译成简体中文，"
                 "并联系上下文正确使用人称代词，不擅自添加原文中没有的代词。")
HYMT_PARAMS = dict(temperature=0.7, top_p=0.6, top_k=20, repeat_penalty=1.05, max_tokens=512)
SAKURA_PARAMS = dict(temperature=0.1, top_p=0.3, frequency_penalty=0.1, max_tokens=1024)
HYMT_CONTEXT_LINES = 3
SAKURA_BATCH = 8
UNIT_MAX_CUES = 4
UNIT_MAX_S = 15.0
SPAN_MAX_CHARS = 42  # 整句譯文不超過這個長度時，就在整句期間持續顯示，不切開
CUT_CHARS = set("，、。；：！？,.;:!? ")
# 譯文長度超過「原文長度 × 倍率 + 常數」視為把上下文也翻進去了，改用不帶上下文重翻
LEAK_LIMIT = {"en": (1.1, 8), "ja": (1.6, 8)}


# llama-server 找不到 CUDA 裝置時（--device CUDA0 讓它直接結束，不會默默改用 CPU）的說明；
# 字樣跟辨識子程序共用（syscheck.NO_CUDA_MARKERS）
NO_CUDA_MESSAGE = syscheck.no_cuda_message("翻譯模型")


def llama_server_args(cfg: dict, port: int) -> list[str]:
    # --device CUDA0：只用第一張 NVIDIA 顯示卡（CUDA_DEVICE_ORDER=PCI_BUS_ID），沒有 CUDA 時直接失敗，不在 CPU 上跑
    return [str(LLAMA_SERVER), "-m", str(cfg["gguf"]), "--device", "CUDA0", "-ngl", "all", "--fit", "off",
            "-c", str(cfg["ctx"]), "-np", str(cfg["parallel"]), "-fa", "on", "--jinja",
            "--host", "127.0.0.1", "--port", str(port)]


def load_error_message(tail: str) -> str:
    """llama-server 載入失敗時，把它的輸出翻成使用者看得懂的中文。"""
    low = tail.lower()
    if syscheck.gpu_too_new(tail):
        return syscheck.too_new_message("翻譯模型")
    if syscheck.cuda_unusable(tail):
        return NO_CUDA_MESSAGE
    if "out of memory" in low or "failed to allocate" in low:
        return "顯存不足，翻譯模型無法完整載入 GPU。請關閉其他使用顯卡的程式後按「重試」。"
    lines = [l for l in tail.strip().splitlines() if l.strip()]
    return "翻譯模型載入失敗：" + lines[-1] if lines else "翻譯模型載入失敗"


class LlamaServer:
    def __init__(self, key: str, log_path: Path, cfg: dict | None = None):
        # cfg：config.translator_cfg 的結果（用哪個版本的 gguf、ctx、parallel）；沒給用預設
        self.cfg = cfg or TRANSLATORS[key]
        self.log_path = log_path
        self.proc = None
        self.port = None

    def start(self, check: Callable[[], None]):
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            self.port = s.getsockname()[1]
        self.log = open(self.log_path, "w", encoding="utf-8", errors="replace")
        try:
            self.proc = subprocess.Popen(
                llama_server_args(self.cfg, self.port), env=child_env(CUDA_DEVICE_ORDER="PCI_BUS_ID"),
                stdout=self.log, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW,
            )
        except FileNotFoundError:
            self.log.close()
            raise RuntimeError("找不到翻譯程式 llama-server，請重新解壓縮程式或執行 repair.bat")
        try:
            deadline = time.time() + 300
            while time.time() < deadline:
                check()
                if self.proc.poll() is not None:
                    self.log.flush()
                    tail = self.log_path.read_text(encoding="utf-8", errors="replace")[-1500:]
                    raise RuntimeError(load_error_message(tail))
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=2) as r:
                        if r.status == 200:
                            return
                except (urllib.error.URLError, ConnectionError, TimeoutError):
                    pass
                time.sleep(0.5)
            raise RuntimeError("翻譯模型載入逾時")
        except BaseException:
            # 取消、逾時或載入失敗：已經開起來的 llama-server 要關掉，不然沒人管它，會一直佔著顯存
            self.stop()
            raise

    def alive(self) -> bool:
        return bool(self.proc) and self.proc.poll() is None

    def chat(self, messages: list[dict], params: dict) -> str:
        body = json.dumps({"messages": messages, "stream": False, **params}).encode()
        last_error = None
        for attempt in range(3):
            if not self.alive():
                raise RuntimeError("翻譯模型的程序意外結束了，請按「重試」")
            req = urllib.request.Request(
                f"http://127.0.0.1:{self.port}/v1/chat/completions", data=body,
                headers={"Content-Type": "application/json"},
            )
            try:
                with urllib.request.urlopen(req, timeout=600) as r:
                    data = json.load(r)
                return (data["choices"][0]["message"]["content"] or "").strip()
            except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
                last_error = e
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"翻譯模型連線失敗：{last_error}")

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        if getattr(self, "log", None):
            self.log.close()


def translate_title(server: LlamaServer, key: str, title: str) -> str:
    """影片標題翻成繁體中文。標題通常帶【】、emoji、feat. 之類的，要求保留這些格式。"""
    if key in SAKURA_STYLE:
        out = server.chat(
            [{"role": "system", "content": SAKURA_SYSTEM},
             {"role": "user", "content": "将下面的日文文本翻译成中文：" + title}],
            SAKURA_PARAMS,
        )
    else:
        prompt = ("将以下视频标题翻译为简体中文，保留原有的符号、括号、表情符号、英文专有名词与格式，"
                  f"只需要输出翻译后的标题，不要额外解释：\n\n{title}")
        out = server.chat([{"role": "user", "content": prompt}], HYMT_PARAMS)
    out = _clean_output(out)
    return to_taiwan(out).strip() if out else ""


def _clean_output(text: str) -> str:
    text = text.replace("<think>", "").replace("</think>", "").strip()
    return " ".join(text.splitlines()).strip()


def translate_hymt(server: LlamaServer, texts: list[str], lang: str, done: dict, on_progress, check, save):
    ratio, extra = LEAK_LIMIT.get(lang, (1.6, 8))

    def prompt_for(i: int, with_context: bool) -> str:
        src = texts[i]
        context = [t for t in texts[max(0, i - HYMT_CONTEXT_LINES):i] if t.strip()]
        if with_context and context:
            return ("\n".join(context) + "\n参考上面的信息，把下面的文本翻译成简体中文，"
                    "注意不需要翻译上文，也不要额外解释：\n" + src)
        return f"将以下文本翻译为简体中文，注意只需要输出翻译后的结果，不要额外解释：\n\n{src}"

    def one(i: int) -> str:
        out = _clean_output(server.chat([{"role": "user", "content": prompt_for(i, True)}], HYMT_PARAMS))
        # 帶上下文時偶爾會把上文一起翻出來，長度異常就改用不帶上下文的方式重翻
        if not out or len(out) > len(texts[i]) * ratio + extra:
            out = _clean_output(server.chat([{"role": "user", "content": prompt_for(i, False)}], HYMT_PARAMS))
        return out

    todo = [i for i in range(len(texts)) if str(i) not in done]
    _run_parallel(todo, one, server.cfg["parallel"], done, len(texts), on_progress, check, save)


def translate_sakura(server: LlamaServer, texts: list[str], lang: str, done: dict, on_progress, check, save):
    todo = [i for i in range(len(texts)) if str(i) not in done]
    batches = [todo[k:k + SAKURA_BATCH] for k in range(0, len(todo), SAKURA_BATCH)]

    def ask(lines: list[str]) -> str:
        return server.chat(
            [{"role": "system", "content": SAKURA_SYSTEM},
             {"role": "user", "content": "将下面的日文文本翻译成中文：" + "\n".join(lines)}],
            SAKURA_PARAMS,
        )

    def one_batch(idx: list[int]) -> list[str]:
        lines = [" ".join(texts[i].splitlines()) for i in idx]
        out = [l.strip() for l in ask(lines).strip().split("\n")]
        if len(out) == len(lines):
            return out
        # 行數對不上就逐行重翻，確保每句字幕都對得到
        return [_clean_output(ask([line])) for line in lines]

    finished = len(texts) - len(todo)
    pool = ThreadPoolExecutor(server.cfg["parallel"])
    try:
        futures = [(idx, pool.submit(one_batch, idx)) for idx in batches]
        for idx, fut in futures:
            check()
            for i, t in zip(idx, fut.result()):
                done[str(i)] = t
            finished += len(idx)
            save()
            on_progress(finished / max(1, len(texts)))
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


def _run_parallel(todo, fn, workers, done, total, on_progress, check, save):
    finished = total - len(todo)
    pool = ThreadPoolExecutor(workers)
    try:
        futures = [(i, pool.submit(fn, i)) for i in todo]
        for n, (i, fut) in enumerate(futures, 1):
            check()
            done[str(i)] = fut.result()
            finished += 1
            if n % 20 == 0 or n == len(futures):
                save()
            on_progress(finished / max(1, total))
    finally:
        save()
        pool.shutdown(wait=False, cancel_futures=True)


# ---------- 逐行對照翻譯（預設） ----------
#
# 每一行字幕都有自己的譯文，順序跟原文一樣，方便對照學習。
# 同一句話被切成好幾行時，整句一起送給模型當上下文，但要求逐行輸出、每行只翻該行。

LINE_RE = re.compile(r"^\s*(\d+)\s*[.．、:：)）]\s*(.*)$")


def _parse_numbered(text: str, n: int) -> list[str] | None:
    found: dict[int, str] = {}
    for ln in _clean_lines(text):
        m = LINE_RE.match(ln)
        if m:
            idx = int(m.group(1))
            if 1 <= idx <= n and idx not in found:
                found[idx] = m.group(2).strip()
    if len(found) != n or any(not found[i] for i in range(1, n + 1)):
        return None
    return [found[i] for i in range(1, n + 1)]


def _hymt_lines(server: LlamaServer, lines: list[str], context: list[str]) -> list[str] | None:
    n = len(lines)
    ctx_block = ("上文（仅供理解，不要翻译）：\n" + "\n".join(context) + "\n\n") if context else ""
    numbered = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(lines))
    prompt = (
        f"{ctx_block}下面是影片字幕连续的 {n} 行，合起来是一句话。请逐行翻译为简体中文：\n"
        f"1. 每一行只翻译该行原文的内容，保持原文的语序，不要把其他行的内容挪到这一行\n"
        f"2. 输出必须正好 {n} 行，每行以编号开头，格式为「编号. 译文」\n"
        f"3. 只输出译文，不要额外解释\n\n{numbered}"
    )
    try:
        out = server.chat([{"role": "user", "content": prompt}], {**HYMT_PARAMS, "temperature": 0.3})
    except RuntimeError:
        raise
    except Exception:  # noqa: BLE001
        return None
    return _parse_numbered(out, n)


def translate_lines_hymt(server: LlamaServer, cues: list[dict], lang: str, done: dict, on_progress, check, save):
    """Hy-MT2 逐行翻譯；done 以字幕行的索引為 key。"""
    ratio, extra = LEAK_LIMIT.get(lang, (1.6, 8))
    units = build_units(cues, lang)
    todo = [u for u in units if any(str(i) not in done for i in u)]
    texts = [c["text"] for c in cues]

    def context_for(first: int) -> list[str]:
        return [t for t in texts[max(0, first - HYMT_CONTEXT_LINES):first] if t.strip()]

    def single(i: int) -> str:
        src = texts[i]
        ctx = context_for(i)
        prompt = (("\n".join(ctx) + "\n参考上面的信息，把下面的文本翻译成简体中文，注意不需要翻译上文，也不要额外解释：\n" + src)
                  if ctx else f"将以下文本翻译为简体中文，注意只需要输出翻译后的结果，不要额外解释：\n\n{src}")
        out = _clean_output(server.chat([{"role": "user", "content": prompt}], HYMT_PARAMS))
        if not out or len(out) > len(src) * ratio + extra:
            out = _clean_output(server.chat(
                [{"role": "user", "content": f"将以下文本翻译为简体中文，注意只需要输出翻译后的结果，不要额外解释：\n\n{src}"}],
                HYMT_PARAMS))
        return out

    def one(unit: list[int]) -> dict[int, str]:
        if len(unit) == 1:
            return {unit[0]: single(unit[0])}
        lines = [texts[i] for i in unit]
        out = _hymt_lines(server, lines, context_for(unit[0]))
        total_src = sum(len(t) for t in lines)
        if out and len("".join(out)) <= total_src * ratio + extra * len(unit):
            return dict(zip(unit, out))
        # 模型沒照格式輸出：退回整句翻譯，再請它依原文每行的意思分段
        whole = _clean_output(server.chat(
            [{"role": "user", "content": f"将以下文本翻译为简体中文，注意只需要输出翻译后的结果，不要额外解释：\n\n{unit_text(cues, unit, lang)}"}],
            HYMT_PARAMS))
        parts = split_by_lines(server, lines, whole) if whole else None
        if parts:
            return dict(zip(unit, parts))
        return {i: single(i) for i in unit}

    finished = sum(1 for i in range(len(cues)) if str(i) in done)
    pool = ThreadPoolExecutor(server.cfg["parallel"])
    try:
        futures = [pool.submit(one, u) for u in todo]
        for n, fut in enumerate(futures, 1):
            check()
            for i, t in fut.result().items():
                done[str(i)] = t
            finished = sum(1 for i in range(len(cues)) if str(i) in done)
            if n % 10 == 0 or n == len(futures):
                save()
            on_progress(finished / max(1, len(cues)))
    finally:
        save()
        pool.shutdown(wait=False, cancel_futures=True)


def translate_lines_sakura(server: LlamaServer, cues: list[dict], lang: str, done: dict, on_progress, check, save):
    """Sakura 本來就是多行輸入、逐行輸出；分批時以整句為界，不把一句話切到兩批。"""
    units = build_units(cues, lang)
    batches, cur = [], []
    for u in units:
        pending = [i for i in u if str(i) not in done]
        if not pending:
            continue
        cur.extend(u)
        if len(cur) >= SAKURA_BATCH:
            batches.append(cur)
            cur = []
    if cur:
        batches.append(cur)

    def ask(lines: list[str]) -> str:
        return server.chat(
            [{"role": "system", "content": SAKURA_SYSTEM},
             {"role": "user", "content": "将下面的日文文本翻译成中文：" + "\n".join(lines)}],
            SAKURA_PARAMS,
        )

    def one(idx: list[int]) -> list[str]:
        lines = [" ".join(cues[i]["text"].splitlines()) for i in idx]
        out = [ln.strip() for ln in ask(lines).strip().split("\n")]
        if len(out) == len(lines) and all(out):
            return out
        return [_clean_output(ask([line])) for line in lines]

    pool = ThreadPoolExecutor(server.cfg["parallel"])
    try:
        futures = [(idx, pool.submit(one, idx)) for idx in batches]
        for idx, fut in futures:
            check()
            for i, t in zip(idx, fut.result()):
                done[str(i)] = t
            save()
            on_progress(sum(1 for i in range(len(cues)) if str(i) in done) / max(1, len(cues)))
    finally:
        save()
        pool.shutdown(wait=False, cancel_futures=True)


def finalize_lines(cues: list[dict], done: dict) -> list[dict]:
    out = []
    for i, c in enumerate(cues):
        text = tidy_text(to_taiwan(done.get(str(i), "")), "zh-TW")
        if text:
            out.append({"start": c["start"], "end": c["end"], "text": text})
    return out


# ---------- 以句子為單位翻譯（整句通順模式） ----------

def build_units(cues: list[dict], lang: str) -> list[list[int]]:
    """把被切成好幾行的同一句話合回來，整句翻譯比翻半句準。"""
    units, cur = [], []
    for i, c in enumerate(cues):
        cur.append(i)
        brk = c.get("brk")
        if brk is None:
            tail = c["text"].rstrip().rstrip("\"'”’)」』")
            ends = lang != "en" or tail[-1:] in (".", "!", "?")
        else:
            ends = brk in ("eos", "gap", "end")
        span = cues[cur[-1]]["end"] - cues[cur[0]]["start"]
        if ends or len(cur) >= UNIT_MAX_CUES or span > UNIT_MAX_S:
            units.append(cur)
            cur = []
    if cur:
        units.append(cur)
    return units


def unit_text(cues: list[dict], unit: list[int], lang: str) -> str:
    if lang == "en":
        return " ".join(cues[i]["text"] for i in unit)
    parts = []
    for k, i in enumerate(unit):
        parts.append(cues[i]["text"])
        if k < len(unit) - 1 and cues[i].get("brk") == "soft":
            parts.append("、")
    return "".join(parts)


SPLIT_PARAMS = dict(temperature=0.2, top_p=0.6, max_tokens=1024)


def split_by_lines(server: LlamaServer, source_lines: list[str], translation: str) -> list[str] | None:
    """請模型把整句譯文按照原文每一行的意思重新分段。

    中日文語序不同，用字數比例切會對錯行（日文的時間副詞在中文常跑到句尾），
    所以交給模型做語意對齊；切不出來就回 None，讓呼叫端退回比例切法。
    """
    n = len(source_lines)
    numbered = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(source_lines))
    prompt = (
        f"下面是一句話被切成 {n} 行的原文，以及這句話完整的中文翻譯。\n"
        f"請把譯文重新分成 {n} 段，讓每一段對應同編號那一行原文的意思（原文與譯文語序不同沒關係）。\n"
        f"不要增加或刪減任何內容，不要輸出編號或解釋，直接輸出 {n} 行：\n\n"
        f"原文：\n{numbered}\n\n譯文：\n{translation}"
    )
    try:
        out = server.chat([{"role": "user", "content": prompt}], SPLIT_PARAMS)
    except Exception:  # noqa: BLE001
        return None
    lines = [re.sub(r"^\s*\d+\s*[.、)）]\s*", "", ln).strip() for ln in _clean_lines(out)]
    if len(lines) != n or any(not ln for ln in lines):
        return None
    ratio = len("".join(lines)) / max(1, len(translation))
    if not 0.6 <= ratio <= 1.6:
        return None
    return lines


def _clean_lines(text: str) -> list[str]:
    text = text.replace("<think>", "").replace("</think>", "")
    return [ln.strip() for ln in text.strip().splitlines() if ln.strip()]


def align_splits(server: LlamaServer, src_cues: list[dict], units: list[list[int]], done: dict,
                 splits: dict, on_progress, check, save):
    """把過長、需要分行顯示的譯文交給模型做語意對齊。"""
    todo = []
    for u, idx in enumerate(units):
        text = done.get(str(u), "")
        if len(idx) > 1 and len(text) > SPAN_MAX_CHARS and str(u) not in splits:
            todo.append(u)
    if not todo:
        return

    def one(u: int):
        return split_by_lines(server, [src_cues[i]["text"] for i in units[u]], done[str(u)])

    pool = ThreadPoolExecutor(server.cfg["parallel"])
    try:
        futures = [(u, pool.submit(one, u)) for u in todo]
        for n, (u, fut) in enumerate(futures, 1):
            check()
            result = fut.result()
            splits[str(u)] = result if result else []
            if n % 10 == 0 or n == len(futures):
                save()
            on_progress(n / len(futures))
    finally:
        save()
        pool.shutdown(wait=False, cancel_futures=True)


def _best_cut(text: str, target: int, lo: int, hi: int) -> int:
    if lo >= hi:
        return max(0, min(len(text), lo))
    window = max(4, int((hi - lo) * 0.35))
    best = None
    for j in range(lo, hi + 1):
        if text[j - 1] in CUT_CHARS and abs(j - target) <= window:
            if best is None or abs(j - target) < abs(best - target):
                best = j
    return best if best is not None else max(lo, min(hi, target))


def finalize_translation(src_cues: list[dict], units: list[list[int]], done: dict,
                         splits: dict | None = None) -> list[dict]:
    splits = splits or {}
    out = []
    for u, idx in enumerate(units):
        text = tidy_text(to_taiwan(done.get(str(u), "")), "zh-TW")
        if not text:
            continue
        first, last = src_cues[idx[0]], src_cues[idx[-1]]
        if len(idx) == 1 or len(text) <= SPAN_MAX_CHARS:
            out.append({"start": first["start"], "end": last["end"], "text": text})
            continue
        aligned = splits.get(str(u))
        if aligned and len(aligned) == len(idx):
            pieces = [{"start": src_cues[i]["start"], "end": src_cues[i]["end"],
                       "text": tidy_text(to_taiwan(aligned[k]), "zh-TW")} for k, i in enumerate(idx)]
            out.extend(_merge_short_pieces(pieces))
            continue
        # 譯文太長：依原文每一行的長度比例，在標點附近切開分配
        weights = [max(1, len(src_cues[i]["text"])) for i in idx]
        total = sum(weights)
        pieces, pos, acc = [], 0, 0
        for k in range(len(idx) - 1):
            acc += weights[k]
            remaining = len(idx) - k - 1
            cut = _best_cut(text, round(len(text) * acc / total), pos + 2, len(text) - 2 * remaining)
            pieces.append(text[pos:cut])
            pos = cut
        pieces.append(text[pos:])
        unit_out: list[dict] = []
        for k, i in enumerate(idx):
            piece = tidy_text(pieces[k].strip(" ，、。；："), "zh-TW")
            c = src_cues[i]
            unit_out.append({"start": c["start"], "end": c["end"], "text": piece})
        out.extend(_merge_short_pieces(unit_out))
    return out


def _merge_short_pieces(pieces: list[dict], min_chars: int = 4) -> list[dict]:
    """切出來太短的片段（例如兩個字）併到相鄰片段，避免一閃而過的破碎譯文。"""
    merged: list[dict] = []
    carry = None
    for p in pieces:
        if carry:
            p = {**p, "start": carry["start"], "text": (carry["text"] + "，" + p["text"]).strip("，")}
            carry = None
        if len(p["text"]) < min_chars:
            if merged:
                merged[-1]["end"] = p["end"]
                if p["text"]:
                    merged[-1]["text"] += "，" + p["text"]
            else:
                carry = p
            continue
        merged.append(p)
    if carry:
        merged.append(carry)
    return [m for m in merged if m["text"]]
