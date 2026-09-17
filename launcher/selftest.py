"""執行環境的基本測試，用 runtime\\venv 的 Python 執行（setup_runtime.py 呼叫）。

最後一行印 SELFTEST {JSON}：torch 能不能載入、看不看得到 CUDA、在顯示卡上做一次小運算，以及主要套件能不能 import。
顯示卡上的小運算只佔幾百 MB 顯存，程式結束就還回去。
"""
import importlib
import json
import sys
import time
import warnings

MODULES = ["fastapi", "uvicorn", "pydantic", "numpy", "soundfile", "psutil", "huggingface_hub", "opencc", "jaconv",
           "transformers", "qwen_asr", "silero_vad", "librosa", "fugashi", "unidic_lite", "pyopenjtalk",
           "onnxruntime", "spacy", "en_core_web_sm", "wordfreq", "ckip_transformers", "audio_separator.separator",
           "yt_dlp"]


def _err(e: BaseException) -> str:
    return f"{type(e).__name__}: {e}"[:500]


def main() -> dict:
    out = {"ok": True, "python": sys.version.split()[0], "executable": sys.executable, "modules": {}}
    t0 = time.time()
    try:
        import torch
        info = {"ok": True, "version": torch.__version__, "cuda_build": torch.version.cuda}
        try:
            info["cuda"] = bool(torch.cuda.is_available())
            if info["cuda"]:
                major, minor = torch.cuda.get_device_capability(0)
                info["device"] = torch.cuda.get_device_name(0)
                info["capability"] = [major, minor]
                # CUDA 的相容規則：為 X.Y 編的程式可以跑在同一個大版本、小版本更大的卡上（sm_86 可以跑 RTX 40 的 8.9）
                archs = [a for a in torch.cuda.get_arch_list() if a.startswith("sm_")]
                info["arch_list"] = archs
                info["arch_supported"] = any(int(a[3:-1]) == major and int(a[-1]) <= minor for a in archs if a[3:].isdigit())
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        x = torch.arange(4, device="cuda", dtype=torch.float32)
                        info["op"] = float((x * 2).sum().item()) == 12.0
                        del x
                        torch.cuda.empty_cache()
                except Exception as e:  # noqa: BLE001
                    info["op"] = False
                    info["op_error"] = _err(e)
        except Exception as e:  # noqa: BLE001
            info["cuda"] = False
            info["cuda_error"] = _err(e)
    except BaseException as e:  # noqa: BLE001
        info = {"ok": False, "error": _err(e), "winerror": getattr(e, "winerror", None)}
        out["ok"] = False
    info["seconds"] = round(time.time() - t0, 2)
    out["torch"] = info
    for name in MODULES:
        t = time.time()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                importlib.import_module(name)
            out["modules"][name] = {"ok": True, "seconds": round(time.time() - t, 2)}
        except BaseException as e:  # noqa: BLE001
            out["modules"][name] = {"ok": False, "error": _err(e)}
            out["ok"] = False
    out["seconds"] = round(time.time() - t0, 2)
    return out


if __name__ == "__main__":
    result = main()
    sys.stdout.flush()
    print("SELFTEST " + json.dumps(result, ensure_ascii=False), flush=True)
