"""下載中文斷詞用的 CKIP Transformers 模型到 models/ckip/（bert-base 斷詞＋詞性，約 776 MB，GPL-3.0）。

用法（在專案根目錄，python 用 conda env subtitle 裡的）：
  python -s tools/download_ckip.py

需要先裝套件：python -s -m pip install ckip-transformers==0.3.4
ckiplab 的 main 分支只有 pytorch_model.bin，allow_patterns 不能只寫 safetensors，不然載入會失敗。
模型只在 CPU 上跑（app/vocab_zh.py 用子程序並關掉 CUDA），不會佔顯卡。
"""
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import app.config  # noqa: E402,F401  載入時會把 Hugging Face 的下載快取改到 data/cache，不寫到使用者家目錄

from app.config import MODELS_DIR  # noqa: E402
TARGET = MODELS_DIR / "ckip"   # 跟程式同一個位置（VS_MODELS_DIR、data\paths.json 改過也一樣）
MODELS = ["bert-base-chinese-ws", "bert-base-chinese-pos"]


def main():
    os.environ["HF_HUB_OFFLINE"] = "0"
    os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
    from huggingface_hub import snapshot_download
    for name in MODELS:
        t0 = time.time()
        dst = TARGET / name
        snapshot_download(f"ckiplab/{name}", local_dir=str(dst), allow_patterns=["*.json", "vocab.txt", "pytorch_model.bin"])
        size = sum(f.stat().st_size for f in dst.rglob("*") if f.is_file()) / 1e6
        print(f"{name}：{size:.0f} MB，{time.time() - t0:.0f} 秒", flush=True)
    if not all((TARGET / n / "pytorch_model.bin").exists() for n in MODELS):
        sys.exit("下載不完整，請再跑一次")
    print("完成")


if __name__ == "__main__":
    main()
