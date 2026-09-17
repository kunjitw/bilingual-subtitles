"""人聲分離子程序（歌曲、配樂很滿的影片用）。

在自己的程序裡跑，跑完就結束，顯存完整歸還。
把影片或音訊的人聲抽出來，存成 16k 單聲道 wav，之後的人聲偵測、辨識、對齊都用它。

用法：python -s -m app.separate <輸入檔> <輸出wav>
進度以 JSON 行輸出到 stdout，跟 asr_child 一致。
"""
import json
import logging
import subprocess
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

from .config import SAMPLE_RATE, SEPARATOR, ToolMissing, require_tool  # noqa: E402
from .safepath import safe_unlink  # noqa: E402


def emit(**kw):
    print(json.dumps(kw, ensure_ascii=False), flush=True)


def _require_model_files():
    # audio-separator 找不到模型檔會自己從網路下載；模型在設定頁被刪掉時要直接失敗，不能偷偷載回來。
    # 上層任務在叫起這個子程序前已經登記用到這個模型（jobs.use_model），設定頁在任務結束前刪不掉；這裡是第二道防線
    missing = [f for f in SEPARATOR["files"] if not (SEPARATOR["dir"] / f).is_file()]
    if missing:
        raise SystemExit(f"找不到模型 {SEPARATOR['label']}，請到設定頁的模型管理下載")


def run(src: Path, dst: Path):
    _require_model_files()
    from audio_separator.separator import Separator

    # 上層任務把找到的 ffmpeg 位置傳下來（config.child_env），找不到時說清楚
    ffmpeg = require_tool("ffmpeg")
    tmp = dst.parent / "sep_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    # 分離模型吃 44.1k 立體聲
    st = tmp / "mix.st44.wav"
    emit(p=0.02, stage="準備音訊")
    subprocess.run([ffmpeg, "-y", "-v", "error", "-i", str(src), "-map", "0:a:0", "-vn",
                    "-ac", "2", "-ar", "44100", "-c:a", "pcm_s16le", str(st)], check=True,
                   creationflags=subprocess.CREATE_NO_WINDOW)

    emit(p=0.06, stage="載入人聲分離模型")
    sep = Separator(model_file_dir=str(SEPARATOR["dir"]), output_dir=str(tmp), output_format="WAV",
                    output_single_stem="Vocals", sample_rate=44100, use_soundfile=True,
                    log_level=logging.WARNING)
    _require_model_files()   # 準備音訊可能花了一兩分鐘，載入前再確認一次
    # 載入時它一定會讀 download_checks.json（模型清單），沒有就連網下載。設定頁下載模型時會一起抓好，
    # 手動放模型檔的電腦第一次用才會缺；離線時說清楚，不要只丟 ConnectionError
    checks = SEPARATOR["dir"] / "download_checks.json"
    had_checks = checks.is_file()
    try:
        sep.load_model(model_filename=SEPARATOR["file"])
    except Exception as e:  # noqa: BLE001
        if not had_checks and not checks.is_file():
            raise SystemExit(f"人聲分離第一次使用要連網下載模型清單 download_checks.json，下載失敗（{type(e).__name__}）。"
                             "請確認網路連線後按「重試」") from e
        raise

    emit(p=0.12, stage="分離人聲")
    files = sep.separate(str(st), custom_output_names={"Vocals": "vocals"})
    produced = tmp / "vocals.wav"
    if not produced.exists():
        cand = [tmp / f for f in files] + list(tmp.glob("vocals*"))
        produced = next(p for p in cand if Path(p).exists())

    emit(p=0.95, stage="輸出")
    # 先寫到暫存名稱，寫完才換成正式的：中途被結束（取消、「釋放顯卡」）不會留下半截的人聲檔，續跑時被當成已經分離好
    part = tmp / "vocals.out.wav"
    subprocess.run([ffmpeg, "-y", "-v", "error", "-i", str(produced), "-ac", "1", "-ar", str(SAMPLE_RATE),
                    "-c:a", "pcm_s16le", str(part)], check=True, creationflags=subprocess.CREATE_NO_WINDOW)
    part.replace(dst)
    for f in tmp.glob("*.wav"):
        safe_unlink(f, tmp)
    emit(p=1.0)


def main():
    try:
        run(Path(sys.argv[1]), Path(sys.argv[2]))
    except ToolMissing as e:
        raise SystemExit(str(e))


if __name__ == "__main__":
    main()
