"""翻譯模型功能測試：以 llama-server 全部載入 GPU，確認輸出格式與顯存用量。

用法：
  python -s tests/smoke_llm.py hymt
  python -s tests/smoke_llm.py sakura

llama-server 和模型的位置用 app/config.py 的設定；結果存到系統暫存資料夾的 vs-smoke。
"""
import json
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.config import LLAMA_SERVER, MODELS_DIR  # noqa: E402

SERVER = Path(LLAMA_SERVER)
OUT = Path(tempfile.gettempdir()) / "vs-smoke"
PORT = 18089

MODELS = {
    "hymt": MODELS_DIR / "gguf" / "Hy-MT2-7B" / "HY-MT2-7B-Q8_0.gguf",
    "sakura": MODELS_DIR / "gguf" / "Sakura-14B" / "sakura-14b-qwen2.5-v1.0-q6k.gguf",
}

SAKURA_SYSTEM = "你是一个轻小说翻译模型，可以流畅通顺地以日本轻小说的风格将日文翻译成简体中文，并联系上下文正确使用人称代词，不擅自添加原文中没有的代词。"


def vram_used_mb() -> int:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True
    )
    return int(out.strip().splitlines()[0])


def chat(messages, **params) -> tuple[str, float]:
    body = json.dumps({"messages": messages, "stream": False, **params}).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions", data=body, headers={"Content-Type": "application/json"}
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=300) as r:
        data = json.load(r)
    return data["choices"][0]["message"]["content"], time.time() - t0


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in MODELS:
        raise SystemExit(__doc__)
    name = sys.argv[1]
    if not SERVER.is_file():
        raise SystemExit(f"找不到 llama-server：{SERVER}")
    if not MODELS[name].is_file():
        raise SystemExit(f"找不到模型檔：{MODELS[name]}，請先到設定頁下載")
    OUT.mkdir(parents=True, exist_ok=True)
    before = vram_used_mb()
    proc = subprocess.Popen(
        [str(SERVER), "-m", str(MODELS[name]), "-ngl", "all", "--fit", "off", "-c", "4096", "-np", "1",
         "--host", "127.0.0.1", "--port", str(PORT), "--jinja", "-fa", "on"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
    )
    log = []
    try:
        t0 = time.time()
        while True:
            if proc.poll() is not None:
                log.extend(proc.stdout.readlines())
                raise RuntimeError("llama-server exited:\n" + "".join(log[-30:]))
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2) as r:
                    if r.status == 200:
                        break
            except Exception:
                time.sleep(0.5)
        load_s = time.time() - t0
        loaded = vram_used_mb()

        results = {"load_s": round(load_s, 1), "vram_before_mb": before, "vram_loaded_mb": loaded, "cases": []}
        if name == "hymt":
            params = dict(temperature=0.7, top_p=0.6, top_k=20, repeat_penalty=1.05, max_tokens=512)
            cases = [
                ("en", "简体中文", "I used to think being productive meant doing more, but lately I've realized it's about doing what matters."),
                ("en", "繁体中文", "I used to think being productive meant doing more, but lately I've realized it's about doing what matters."),
                ("ja", "简体中文", "明日は全国的に強い風が吹く見込みで、交通機関への影響に注意が必要です。"),
                ("ja", "繁体中文", "明日は全国的に強い風が吹く見込みで、交通機関への影響に注意が必要です。"),
            ]
            for lang, target, src in cases:
                prompt = f"将以下文本翻译为{target}，注意只需要输出翻译后的结果，不要额外解释：\n\n{src}"
                text, sec = chat([{"role": "user", "content": prompt}], **params)
                results["cases"].append({"target": target, "src": src, "out": text, "sec": round(sec, 2)})
        else:
            params = dict(temperature=0.1, top_p=0.3, frequency_penalty=0.1, max_tokens=512)
            src = "ねえ、今日もお疲れさま。\nゆっくり休んでね。\n明日も一緒にがんばろう。"
            text, sec = chat(
                [{"role": "system", "content": SAKURA_SYSTEM}, {"role": "user", "content": "将下面的日文文本翻译成中文：" + src}],
                **params,
            )
            results["cases"].append({"src": src, "out": text, "sec": round(sec, 2)})

        (OUT / f"llm_{name}.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"ok -> {OUT / f'llm_{name}.json'}")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    main()
