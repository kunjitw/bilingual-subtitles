"""路徑、模型與預設組合。"""
import json
import os
import re
import shutil
import stat
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEB_DIR = ROOT / "web"


def _env_path(name: str, env=None) -> Path | None:
    """環境變數設的路徑；沒設或空字串回 None。相對路徑以程式根目錄為準。"""
    raw = ((os.environ if env is None else env).get(name) or "").strip().strip('"')
    if not raw:
        return None
    p = Path(os.path.expandvars(raw)).expanduser()
    return p if p.is_absolute() else ROOT / p


def _read_json(path: Path) -> dict:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _json_path(data: dict, key: str) -> Path | None:
    """JSON 裡寫的路徑：{"ffmpeg": "..."}、{"ffmpeg": {"path": "..."}} 或 {"paths": {"ffmpeg": "..."}} 都可以。"""
    value = data.get(key)
    if value is None and isinstance(data.get("paths"), dict):
        value = data["paths"].get(key)
    if isinstance(value, dict):
        value = value.get("path")
    if not isinstance(value, str) or not value.strip():
        return None
    p = Path(os.path.expandvars(value.strip())).expanduser()
    return p if p.is_absolute() else ROOT / p


# 打包版的執行環境（launcher 寫的），開發環境沒有這個檔
RUNTIME_DIR = ROOT / "runtime"
INSTALL_STATE_PATH = RUNTIME_DIR / "install-state.json"
INSTALL_STATE = _read_json(INSTALL_STATE_PATH)

# 位置可以用環境變數改，優先順序：環境變數 → data\paths.json（只有模型位置）→ 預設值
DATA_DIR = _env_path("VS_DATA_DIR") or ROOT / "data"
MEDIA_DIR = DATA_DIR / "media"
WORK_DIR = DATA_DIR / "work"
SUBS_DIR = DATA_DIR / "subs"
THUMB_DIR = DATA_DIR / "thumbs"
PROXY_DIR = DATA_DIR / "proxy"
DB_PATH = DATA_DIR / "library.db"
LOG_PATH = DATA_DIR / "app.log"
PATHS_JSON = DATA_DIR / "paths.json"  # 選用：{"models_dir": "D:\\BilingualSubtitlesModels"}，模型放在別的磁碟時用
MODELS_DIR = _env_path("VS_MODELS_DIR") or _json_path(_read_json(PATHS_JSON), "models_dir") or ROOT / "models"
DICT_DIR = DATA_DIR / "dict"  # 查字用的字典（可刪掉重建，建法見 tools/build_dict_*.py）
CACHE_DIR = _env_path("VS_CACHE_DIR") or DATA_DIR / "cache"  # 套件的下載快取（可以整個刪掉，需要時會重建）
YTDLP_CACHE_DIR = CACHE_DIR / "yt-dlp"

# 套件預設把快取寫到使用者家目錄（~/.cache），改放到專案裡。
# 只影響這個程式和它叫起來的子程序；使用者自己設過的環境變數優先，設了 HF_HOME 就整組 Hugging Face 照他的。
# 登入 Hugging Face 的 token 還是在原本的位置（沒有改 HF_HOME）。
_CACHE_ENV = {
    "HF_HUB_CACHE": CACHE_DIR / "huggingface" / "hub",
    "HF_XET_CACHE": CACHE_DIR / "huggingface" / "xet",
    "HF_MODULES_CACHE": CACHE_DIR / "huggingface" / "modules",
    "TORCH_HOME": CACHE_DIR / "torch",
    "MPLCONFIGDIR": CACHE_DIR / "matplotlib",
}
for _key, _dir in _CACHE_ENV.items():
    if _key.startswith("HF_") and os.environ.get("HF_HOME"):
        continue
    os.environ.setdefault(_key, str(_dir))

# ---------- 外部程式 ----------

LEGACY_FFMPEG_DIR = Path(r"C:\Program Files\FFMPEG\bin")   # 開發時作者電腦上的位置，找不到別的才用


class ToolMissing(RuntimeError):
    """找不到 ffmpeg、ffprobe。訊息可以直接給使用者看；API 回 503。"""


def find_tool(name: str, env=None, root: Path = ROOT, state: dict | None = None, which=shutil.which,
              legacy_dir: Path = LEGACY_FFMPEG_DIR) -> dict:
    """找 ffmpeg 或 ffprobe，依序：環境變數 VS_FFMPEG / VS_FFPROBE → runtime\\install-state.json →
    專案的 bin\\ffmpeg → PATH → C:\\Program Files\\FFMPEG\\bin。
    環境變數有設就只看它（指到不存在的檔案就算找不到，不會偷偷改用別的）。
    回傳 {"path": 完整路徑或 None, "source": 從哪裡找到的, "problem": 找不到時給使用者看的說明}。"""
    env = os.environ if env is None else env
    state = INSTALL_STATE if state is None else state
    exe = f"{name}.exe"
    var = f"VS_{name.upper()}"
    raw = (env.get(var) or "").strip().strip('"')
    if raw:
        if Path(raw).is_file():
            return {"path": str(Path(raw)), "source": var, "problem": None}
        return {"path": None, "source": var,
                "problem": f"找不到 {name}：環境變數 {var} 設定的位置 {raw} 沒有這個檔案，請改正或拿掉這個設定"}
    # 只設了另一個（例如 VS_FFMPEG）：ffmpeg、ffprobe 通常放在同一個資料夾，先找它旁邊
    other = "ffprobe" if name == "ffmpeg" else "ffmpeg"
    other_var = f"VS_{other.upper()}"
    other_raw = (env.get(other_var) or "").strip().strip('"')
    candidates = [(other_var, Path(other_raw).parent / exe if other_raw else None),
                  ("install-state", _json_path(state, name)),
                  ("bin", root / "bin" / "ffmpeg" / exe), ("bin", root / "bin" / "ffmpeg" / "bin" / exe)]
    found = which(name)
    candidates.append(("PATH", Path(found) if found else None))
    candidates.append(("legacy", Path(legacy_dir) / exe))
    for source, p in candidates:
        if p is not None and p.is_file():
            return {"path": str(p), "source": source, "problem": None}
    if other_raw and Path(other_raw).is_file():
        return {"path": None, "source": None,
                "problem": f"找不到 {name}：環境變數 {other_var} 指到的資料夾裡沒有 {exe}。"
                           f"請把 {exe} 放到同一個資料夾，或用環境變數 {var} 指定它的位置"}
    return {"path": None, "source": None, "problem": f"找不到 {name}，請重新解壓縮程式或執行 repair.bat"}


# 啟動時找一次並記下結果（server 啟動時把缺的寫進 log）
TOOLS = {"ffmpeg": find_tool("ffmpeg"), "ffprobe": find_tool("ffprobe")}


def require_tool(name: str) -> str:
    """要執行 ffmpeg、ffprobe 前呼叫：回傳完整路徑，找不到丟 ToolMissing。"""
    info = TOOLS.get(name) or {}
    if info.get("path") and Path(info["path"]).is_file():
        return info["path"]
    raise ToolMissing(info.get("problem") or f"找不到 {name}，請重新解壓縮程式或執行 repair.bat")


def tool_problems() -> list[str]:
    out = []
    for name in TOOLS:
        try:
            require_tool(name)
        except ToolMissing as e:
            out.append(str(e))
    return out


# 舊的寫法：找不到時是「預期的位置」，執行時會丟 FileNotFoundError（新程式碼用 require_tool）
FFMPEG = TOOLS["ffmpeg"]["path"] or (os.environ.get("VS_FFMPEG") or str(LEGACY_FFMPEG_DIR / "ffmpeg.exe"))
FFPROBE = TOOLS["ffprobe"]["path"] or (os.environ.get("VS_FFPROBE") or str(LEGACY_FFMPEG_DIR / "ffprobe.exe"))


def child_env(base=None, **extra) -> dict:
    """給子程序的環境變數：ffmpeg 的資料夾放在 PATH 最前面（audio-separator 只會執行 PATH 上的 ffmpeg），
    找到的 ffmpeg、ffprobe 位置也傳下去，子程序不用再找一次，一定跟這個程式用同一個。
    多張顯示卡時 CUDA 裝置照 PCI 順序編號，跟 NVML、nvidia-smi 的第 0 張一致。"""
    env = dict(os.environ if base is None else base)
    dirs = []
    for name in ("ffmpeg", "ffprobe"):
        path = (TOOLS.get(name) or {}).get("path")
        if path:
            env[f"VS_{name.upper()}"] = path
            d = str(Path(path).parent)
            if d not in dirs:
                dirs.append(d)
    if dirs:
        rest = [p for p in env.get("PATH", "").split(os.pathsep)
                if p and os.path.normcase(os.path.normpath(p)) not in {os.path.normcase(os.path.normpath(d)) for d in dirs}]
        env["PATH"] = os.pathsep.join(dirs + rest)
    env.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    env.update({k: str(v) for k, v in extra.items()})
    return env


LLAMA_SERVER = _env_path("VS_LLAMA_SERVER") or _json_path(INSTALL_STATE, "llama_server") \
    or ROOT / "bin" / "llama.cpp" / "llama-server.exe"

# 綁哪個網路位址：有設 VS_HOST 就照它；沒設時看設定頁的區網開關（lan_access），關著只開本機 127.0.0.1
HOST_ENV = (os.environ.get("VS_HOST") or "").strip() or None
PORT = int(os.environ.get("VS_PORT", "8765"))

SAMPLE_RATE = 16000

VIDEO_EXTS = {".mp4", ".m4v", ".mov", ".mkv", ".webm", ".avi", ".flv", ".ts", ".m2ts", ".wmv", ".mpg", ".mpeg",
              ".mp3", ".m4a", ".wav", ".flac", ".ogg", ".opus", ".aac"}

# 語言：asr 是 Qwen3-ASR / 對齊器使用的名稱，track 是字幕軌語言代碼
LANGUAGES = {
    "zh": {"label": "中文", "asr": "Chinese", "track": "zh-TW"},
    "ja": {"label": "日文", "asr": "Japanese", "track": "ja"},
    "en": {"label": "英文", "asr": "English", "track": "en"},
}

TRACK_LANG_LABELS = {"zh-TW": "繁體中文", "ja": "日文", "en": "英文"}

PROFILES = {
    "general": "一般",
    "asmr": "ASMR／耳語",
}

_SEP_FILE = "model_bs_roformer_ep_317_sdr_12.9755"

# 模型型錄：可在設定頁下載、刪除、指定預設。刪除、算佔用只動 model_targets 列出的這個模型自己的檔案。
# kind：hf    Hugging Face repo 整包下載（allow 過濾），放在自己的資料夾
#       onnx  同 hf，files 列出載入時一定要有的檔案
#       gguf  repo 裡的單一檔案，放在自己的資料夾
#       ckpt  從網址下載的單檔模型，放在共用資料夾（models/sep），files 是它自己的檔案，urls 是下載來源
# parts：一個模型由好幾個 repo 組成（CKIP 斷詞＋詞性），各自下載到 dir 底下自己的資料夾
# size_mb：要下載的檔案合計（MiB），2026-09-16 照 allow、file 過濾對過 Hugging Face 和 GitHub 上的實際大小
# license：設定頁顯示的授權，2026-09-17 照模型官方頁面確認；license_url 是寫著授權條件的那一頁。
#          限制商用的開頭寫「非商用」，作者沒有公開授權的開頭寫「授權不明」，設定頁會用不同顏色標出來
#          （models.license_limit）。完整說明見 THIRD_PARTY_NOTICES.md
MODEL_CATALOG = {
    "qwen-asr": {
        "role": "asr", "engine": "qwen", "label": "Qwen3-ASR-1.7B", "kind": "hf",
        "repo": "Qwen/Qwen3-ASR-1.7B", "dir": MODELS_DIR / "asr" / "Qwen3-ASR-1.7B",
        "size_mb": 4485, "langs": ["zh", "ja", "en"],
        "note": "中英日通用，一般影片的預設辨識模型",
        "license": "Apache-2.0", "license_url": "https://huggingface.co/Qwen/Qwen3-ASR-1.7B",
    },
    "anime-whisper": {
        "role": "asr", "engine": "anime", "label": "anime-whisper", "kind": "hf",
        "repo": "litagin/anime-whisper", "dir": MODELS_DIR / "asr" / "anime-whisper",
        "size_mb": 2887, "langs": ["ja"],
        "note": "動畫與 Galgame 語音微調，ASMR、耳語的預設模型",
        # 模型卡寫 MIT，但訓練資料（Galgame_Speech_ASR_16kHz，源自 OOPPEENN/Galgame_Dataset）規定用它訓練的模型不得商用
        "license": "非商用（模型 MIT，訓練資料禁止商用）",
        "license_url": "https://huggingface.co/datasets/litagin/Galgame_Speech_ASR_16kHz",
    },
    "whisper-large-v3": {
        "role": "asr", "engine": "whisper", "label": "Whisper large-v3", "kind": "hf",
        "repo": "openai/whisper-large-v3", "dir": MODELS_DIR / "asr" / "whisper-large-v3",
        "allow": ["*.json", "*.txt", "model.safetensors"],
        "size_mb": 2949, "langs": ["zh", "ja", "en"],
        "note": "老牌通用模型，專有名詞較穩，可當對照組",
        "license": "Apache-2.0", "license_url": "https://huggingface.co/openai/whisper-large-v3",
    },
    "aligner": {
        "role": "aligner", "label": "Qwen3-ForcedAligner-0.6B", "kind": "hf",
        "repo": "Qwen/Qwen3-ForcedAligner-0.6B", "dir": MODELS_DIR / "asr" / "Qwen3-ForcedAligner-0.6B",
        "size_mb": 1755, "langs": ["zh", "ja", "en"],
        "note": "產生字幕時間軸，所有辨識模型共用",
        "license": "Apache-2.0", "license_url": "https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B",
    },
    "tsqyomi": {
        "role": "furigana", "label": "tsqyomi（假名同形異音）", "kind": "onnx",
        "repo": "tsukumijima/tsqyomi-models", "dir": MODELS_DIR / "furigana" / "tsqyomi",
        "allow": ["v4/*"], "files": ["v4/model.onnx", "v4/tokenizer.json", "v4/metadata.json"],
        "size_mb": 77, "langs": ["ja"],
        "note": "搭配 OpenJTalk 在 CPU 上產生假名，判斷 辛い、人気、方 這類多種讀法的詞",
        "license": "MIT", "license_url": "https://huggingface.co/tsukumijima/tsqyomi-models",
    },
    # 翻譯模型的版本（量化）：variants 列出可以下載的版本，同一個模型的版本放在同一個資料夾，各自下載、刪除。
    #   bytes、sha256：2026-09-17 用 huggingface_hub 唯讀 API（model_info files_metadata=True）查到的實際值；
    #                  size_mb 由 bytes 算（無條件捨去成 MiB，跟下載後用檔案大小算顯存的算法一樣）
    #   vram：顯存門檻的依據。measured 實測過；estimate 是照打包計畫 6.4.2 用「gguf 大小 + overhead」估計的
    #   repo 沒寫的用模型的 repo。預設用 default_variant；設定 model_variants 可以改（找不到那個版本時用已下載的）
    "hymt2-7b": {
        "role": "translator", "translator": "hymt", "label": "Hy-MT2-7B", "kind": "gguf",
        "repo": "tencent/Hy-MT2-7B-GGUF",
        "dir": MODELS_DIR / "gguf" / "Hy-MT2-7B", "langs": ["ja", "en"],
        "default_variant": "Q8_0",
        "variants": {
            "Q8_0": {"file": "HY-MT2-7B-Q8_0.gguf", "bytes": 7981928896, "vram": "measured",
                     "sha256": "58b3ad55dd6f6fa08c695cddc34fb5f8f708a844f78ae10508071914b0ed67c0"},
            "Q6_K": {"file": "HY-MT2-7B-Q6_K.gguf", "bytes": 6164482720, "vram": "estimate",
                     "sha256": "88ef0aba59952a4cfe4be36cb5baf797dbb370bc60e9dcbd7297036021e52831"},
            # 官方沒有 Q5_K_M，用 unsloth 量化的
            "Q5_K_M": {"repo": "unsloth/Hy-MT2-7B-GGUF", "file": "Hy-MT2-7B-Q5_K_M.gguf", "bytes": 5371236128,
                       "vram": "estimate", "license_url": "https://huggingface.co/unsloth/Hy-MT2-7B-GGUF",
                       "sha256": "b37a1b004e82670aef7adb018bfadb297d0a897e6c74c0eb61db352fcd82d1e9"},
            "Q4_K_M": {"file": "Hy-MT2-7B-Q4_K_M.gguf", "bytes": 4624648896, "vram": "estimate",
                       "sha256": "9f96256500f3fc1ab4d64336b58f52a949a95ad7516b0c229476eef782f9f77b"},
        },
        "note": "翻譯品質最好，英→中與日文新聞的預設",
        "license": "Apache-2.0", "license_url": "https://huggingface.co/tencent/Hy-MT2-7B-GGUF",
    },
    "sakura-14b": {
        "role": "translator", "translator": "sakura", "label": "Sakura-14B", "kind": "gguf",
        "repo": "SakuraLLM/Sakura-14B-Qwen2.5-v1.0-GGUF",
        "dir": MODELS_DIR / "gguf" / "Sakura-14B", "langs": ["ja"],
        "default_variant": "Q6_K",
        # 顯存：設定頁顯示的是載入前檢查的門檻 translator_vram_mb（Q6_K：11562 + 700 = 12262 MiB）。
        # 實測載入後佔用 12033 MiB（tests/smoke_llm.py sakura，nvidia-smi 14815 − 2782，-c 4096），比門檻略少
        "variants": {
            "Q6_K": {"file": "sakura-14b-qwen2.5-v1.0-q6k.gguf", "bytes": 12124683584, "vram": "measured",
                     "sha256": "2c1fc22a43c15cc42cf1443822115b979c2f9143671f5369df0158d9fc184b77"},
            "Q4_K_M": {"file": "sakura-14b-qwen2.5-v1.0-q4km.gguf", "bytes": 8988110144, "vram": "estimate",
                       "sha256": "c87697cd9c7898464426cb7a1ec5e220755affaa08096766e8d20de1853c2063"},
            "IQ4_XS": {"file": "sakura-14b-qwen2.5-v1.0-iq4xs.gguf", "bytes": 8186195264, "vram": "estimate",
                       "sha256": "34af88f99c113418d0665d3ceede767c9a12040c9e7c4bb5e87cdb1b1e06e94a"},
        },
        "note": "輕小說與 Galgame 語料，ASMR 的預設翻譯模型",
        # SakuraLLM 所有模型禁止任何形式的商用；公開發布用它翻的譯文時，要標明是機器翻譯和模型版本
        "license": "非商用（CC BY-NC-SA 4.0）",
        "license_url": "https://huggingface.co/SakuraLLM/Sakura-14B-Qwen2.5-v1.0-GGUF",
    },
    "hymt2-1.8b": {
        "role": "translator", "translator": "hymt-mini", "label": "Hy-MT2-1.8B", "kind": "gguf",
        "repo": "tencent/Hy-MT2-1.8B-GGUF", "file": "Hy-MT2-1.8B-Q8_0.gguf",
        "bytes": 1908528192, "sha256": "5c3fe0b1408a5ceb0143184ef247b11b579c525f4b02b060e6c851bb76fef1a4",
        "dir": MODELS_DIR / "gguf" / "Hy-MT2-1.8B", "size_mb": 1820, "langs": ["ja", "en"],
        "note": "同一家的小模型，載入與翻譯都快很多，品質略降",
        "license": "Apache-2.0", "license_url": "https://huggingface.co/tencent/Hy-MT2-1.8B-GGUF",
    },
    "galtransl-7b": {
        "role": "translator", "translator": "galtransl", "label": "Sakura-GalTransl-7B", "kind": "gguf",
        "repo": "SakuraLLM/Sakura-GalTransl-7B-v3.7",
        "dir": MODELS_DIR / "gguf" / "GalTransl-7B", "langs": ["ja"],
        "default_variant": "Q6_K",
        "variants": {
            # 檔名沒寫量化；計畫表寫 Q6_K，大小也符合 7.6B 參數的 Q6_K（約 6.56 bit）
            "Q6_K": {"file": "Sakura-Galtransl-7B-v3.7.gguf", "bytes": 6254196608, "vram": "estimate",
                     "sha256": "e1ae01b1735cfdcd00a3c7e2e5e06ff3a2756cc0d7592f4b4f5945638a8629f2"},
            "IQ4_XS": {"file": "Sakura-Galtransl-7B-v3.7-IQ4_XS.gguf", "bytes": 4250296192, "vram": "estimate",
                       "sha256": "8f515bf4769f279a7fcf43e57446455a9d4de7f65b1bc9eddee76717e1ff7919"},
        },
        "note": "Galgame 專用、比 Sakura-14B 小一半，顯存吃緊時的替代",
        "license": "非商用（CC BY-NC-SA 4.0）",
        "license_url": "https://huggingface.co/SakuraLLM/Sakura-GalTransl-7B-v3.7",
    },
    "bs-roformer": {
        "role": "separator", "label": "BS-RoFormer（人聲分離）", "kind": "ckpt",
        "repo": f"audio-separator（{_SEP_FILE}）",
        "file": f"{_SEP_FILE}.ckpt", "files": [f"{_SEP_FILE}.ckpt", f"{_SEP_FILE}.yaml"],
        # 跟 audio-separator 自己下載時用的來源一樣（yaml 它先試 UVR，找不到再用 audio-separator 的）
        "urls": {
            f"{_SEP_FILE}.ckpt": [
                f"https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/{_SEP_FILE}.ckpt"],
            f"{_SEP_FILE}.yaml": [
                f"https://github.com/nomadkaraoke/python-audio-separator/releases/download/model-configs/{_SEP_FILE}.yaml",
                f"https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/mdx_model_data/mdx_c_configs/{_SEP_FILE}.yaml"],
        },
        # audio-separator 載入模型時一定要讀的模型清單（它自己的快取，不屬於這個模型，刪除模型時保留）。
        # 沒有這個檔案時它會連網下載，下載模型時順便抓好，之後離線也能用
        "support": {"download_checks.json": [
            "https://raw.githubusercontent.com/TRvlvr/application_data/main/filelists/download_checks.json"]},
        # 下載完核對大小（2026-09-17 對過 GitHub 回的 Content-Length）；GitHub release 沒有公布 sha256
        "file_bytes": {f"{_SEP_FILE}.ckpt": 639331213, f"{_SEP_FILE}.yaml": 2273},
        "dir": MODELS_DIR / "sep", "size_mb": 610, "langs": ["zh", "ja", "en"],
        "note": "配樂很滿的歌曲或 PV，先把人聲抽出來再辨識，實測歌曲字錯率 36.6% 降到 30%",
        # viperx 訓練、放在 UVR 的模型庫（TRvlvr/model_repo），那個 repo 和作者都沒有公開授權
        "license": "授權不明（作者沒有公開授權）",
        "license_url": "https://github.com/TRvlvr/model_repo/releases/tag/all_public_uvr_models",
    },
    "ckip": {
        "role": "segmenter", "label": "CKIP 中文斷詞", "kind": "hf",
        "repo": "ckiplab/bert-base-chinese-ws、bert-base-chinese-pos",
        # app/vocab_zh.py 從 models/ckip/<名稱> 載入；main 分支只有 pytorch_model.bin，allow 不能只寫 safetensors
        "parts": [{"repo": "ckiplab/bert-base-chinese-ws", "sub": "bert-base-chinese-ws"},
                  {"repo": "ckiplab/bert-base-chinese-pos", "sub": "bert-base-chinese-pos"}],
        "allow": ["*.json", "vocab.txt", "pytorch_model.bin"],
        "files": ["bert-base-chinese-ws/pytorch_model.bin", "bert-base-chinese-pos/pytorch_model.bin"],
        "dir": MODELS_DIR / "ckip", "size_mb": 776, "langs": ["zh"],
        "note": "中文字幕的斷詞和詞性，單字頁的中文分頁要用，在 CPU 上跑",
        "license": "GPL-3.0", "license_url": "https://huggingface.co/ckiplab/bert-base-chinese-ws",
    },
}

# 人聲分離（歌曲模式）
SEPARATOR = {
    "label": "BS-RoFormer（人聲分離）",
    "dir": MODELS_DIR / "sep",
    "file": f"{_SEP_FILE}.ckpt",
    "files": [f"{_SEP_FILE}.ckpt", f"{_SEP_FILE}.yaml"],
    "vram_mb": 4000,
}

# 下載中的暫存檔：<檔名>.part，下載完、核對過才改名（app/model_download.py）
PART_SUFFIX = ".part"
MIB = 1024 * 1024

# 下載完成標記（放在模型的資料夾裡）：下載程式核對過大小（有 sha256 的也核對過）才寫，內容是檔案清單和大小。
# ckpt 的共用資料夾、同一個資料夾裡的多個版本都記在同一個檔案，用模型 id 和版本分開：
#   {"version": 1, "models": {"qwen-asr": {"files": {"model.safetensors": 123, ...}, ...},
#                             "hymt2-7b": {"variants": {"Q6_K": {"files": {...}, ...}}}}}
COMPLETE_MARKER = ".complete.json"
# 多個檔案的模型（hf、onnx、parts）下載中的紀錄：固定的 revision 和檔案清單，續傳時照它下載。
# 有這個檔、又還沒有完成標記時，就算檔案看起來到齊（例如大檔到了、小的 json 還沒）也不算已安裝
DOWNLOAD_MARKER = ".download.json"

for _mid, _e in MODEL_CATALOG.items():
    _e["id"] = _mid
    for _v in (_e.get("variants") or {}).values():
        _v["size_mb"] = _v["bytes"] // MIB
    if _e.get("variants"):
        # 預設版本的大小，給沒有指定版本的地方用（例如「要下載多少」的預設）
        _e["size_mb"] = _e["variants"][_e["default_variant"]]["size_mb"]

_SHARD_RE = re.compile(r"^(.*)-(\d{5})-of-(\d{5})\.safetensors$")


def _safetensors_complete(d: Path) -> bool:
    """有 safetensors，而且分片（model-00001-of-00002）都到齊。"""
    names = {f.name for f in d.glob("*.safetensors")}
    for name in names:
        m = _SHARD_RE.match(name)
        if m:
            prefix, total = m.group(1), m.group(3)
            if any(f"{prefix}-{i:05d}-of-{total}.safetensors" not in names for i in range(1, int(total) + 1)):
                return False
    return bool(names)


# ---------- 版本（量化） ----------

def variant_names(entry: dict) -> list[str]:
    return list(entry.get("variants") or {})


def check_variant(entry: dict, variant: str | None) -> str | None:
    """確認版本名稱：沒有版本的模型回 None（指定了版本就是錯的）；有版本的模型沒指定時回預設版本。
    不認得的版本丟 ValueError（訊息可以直接給使用者看）。"""
    variants = entry.get("variants")
    if not variants:
        if variant:
            raise ValueError(f"{entry.get('label', '這個模型')} 沒有其他版本")
        return None
    if not variant:
        return entry.get("default_variant") or next(iter(variants))
    if variant not in variants:
        raise ValueError(f"{entry.get('label', '這個模型')} 沒有 {variant} 這個版本")
    return variant


def variant_view(entry: dict, variant: str | None = None) -> dict:
    """型錄項目套上某個版本（repo、file、bytes、size_mb、sha256、vram、license_url 換成那個版本的），回傳複本。
    沒有版本的模型原樣複製。"""
    name = check_variant(entry, variant)
    if name is None:
        return dict(entry)
    return {**entry, **entry["variants"][name], "variant": name}


def _marker_key(entry: dict) -> str:
    return entry.get("id") or entry.get("label") or str(entry["dir"])


def complete_record(entry: dict, variant: str | None = None) -> dict | None:
    """完成標記裡這個模型（有版本的模型是那個版本）的紀錄，沒有回 None。"""
    rec = ((_read_json(Path(entry["dir"]) / COMPLETE_MARKER).get("models") or {}).get(_marker_key(entry)))
    if isinstance(rec, dict) and entry.get("variants"):
        rec = (rec.get("variants") or {}).get(variant)
    return rec if isinstance(rec, dict) else None


def _record_ok(base: Path, rec: dict) -> bool:
    """紀錄裡的檔案都在，大小也跟下載完時一樣。"""
    files = rec.get("files")
    if not isinstance(files, dict) or not files:
        return False
    for rel, size in files.items():
        try:
            p = Path(base) / rel
            if not p.is_file() or (isinstance(size, int) and p.stat().st_size != size):
                return False
        except (OSError, ValueError):
            return False
    return True


def download_pending(entry: dict) -> bool:
    """多檔模型有沒有下載到一半的紀錄（.download.json）。"""
    return _read_json(Path(entry["dir"]) / DOWNLOAD_MARKER).get("model") == _marker_key(entry)


def _variant_installed(entry: dict, name: str) -> bool:
    rec = complete_record(entry, name)
    if rec is not None:
        return _record_ok(entry["dir"], rec)
    # 舊資料夾沒有標記：gguf 是單一檔案，下載完才會出現正式檔名（Hugging Face 和這個程式都是）
    return (Path(entry["dir"]) / entry["variants"][name]["file"]).is_file()


def installed_variants(entry: dict) -> list[str]:
    return [n for n in variant_names(entry) if _variant_installed(entry, n)]


def pick_variant(entry: dict, chosen: str | None = None, without=()) -> str | None:
    """要用哪個版本：選的版本（設定 model_variants）已下載就用它；不然用預設版本；再不然用已下載的第一個
    （照型錄順序，品質高的在前）；都沒下載時回選的或預設的版本。沒有版本的模型回 None。
    without：當成已經刪掉的版本（算「刪掉後改用哪個」用）。"""
    variants = entry.get("variants")
    if not variants:
        return None
    default = entry.get("default_variant") or next(iter(variants))
    chosen = chosen if chosen in variants else None
    installed = [n for n in variants if n not in without and _variant_installed(entry, n)]
    for name in (chosen, default):
        if name and name in installed:
            return name
    if installed:
        return installed[0]
    return chosen or default


def model_installed(entry: dict, variant: str | None = None) -> bool:
    """模型（有版本的模型可以指定版本）下載完整、可以用。
    有完成標記（.complete.json）就照標記：列出的檔案都在、大小也對。沒有標記的舊資料夾照原本的規則：
    列出的 files 都在；gguf 看檔案在不在；其他看 safetensors 分片到齊。多檔模型下載到一半（有 .download.json）不算。
    有版本的模型沒指定版本時，任何一個版本裝好就算。"""
    if entry.get("variants"):
        names = [variant] if variant else variant_names(entry)
        return any(_variant_installed(entry, n) for n in names if n in entry["variants"])
    rec = complete_record(entry)
    if rec is not None:
        return _record_ok(entry["dir"], rec)
    if download_pending(entry):
        return False
    d = entry["dir"]
    if entry.get("files"):
        return all((d / f).is_file() for f in entry["files"])
    if entry.get("kind") == "gguf":
        return (d / entry["file"]).is_file()
    return d.is_dir() and _safetensors_complete(d)


def model_key_files(entry: dict, variant: str | None = None) -> list[Path]:
    """決定「已安裝」的那幾個檔案（模型權重），model_installed 看的就是這些。
    刪除時先刪它們：刪不掉就停下來，其他檔案不動，模型保持完整能用；刪掉了，就算其他小檔案被鎖住刪不掉，
    設定頁也會正確顯示成「沒安裝、留著殘檔」，不會出現看起來已安裝、其實設定檔已經不見的模型。
    有版本的模型：指定版本時是那個版本的 gguf，沒指定時是所有版本的。"""
    d = entry["dir"]
    if entry.get("variants"):
        names = [variant] if variant else variant_names(entry)
        return [d / entry["variants"][n]["file"] for n in names]
    if entry.get("files"):
        return [d / f for f in entry["files"]]
    if entry.get("kind") == "gguf":
        return [d / entry["file"]]
    return sorted(d.glob("*.safetensors")) if d.is_dir() else []


HF_LOCAL_CACHE = Path(".cache") / "huggingface" / "download"   # 舊版用 huggingface_hub 下載時留下的紀錄和暫存檔


def hf_legacy_files(d: Path, filename: str) -> list[Path]:
    """舊版用 huggingface_hub 下載某個檔案時，在 <資料夾>\\.cache\\huggingface\\download 留下的東西：
    <檔名>.metadata、<檔名>.lock，和下載到一半的 <sha1(檔名.metadata) 的 base64>.<etag>.incomplete。"""
    import base64
    import hashlib
    base = Path(d) / HF_LOCAL_CACHE
    out = [base / f"{filename}.metadata", base / f"{filename}.lock"]
    prefix = base64.urlsafe_b64encode(hashlib.sha1(f"{filename}.metadata".encode()).digest()).decode()
    if base.is_dir():
        out += sorted(p for p in base.iterdir() if p.name.startswith(prefix + ".") and p.name.endswith(".incomplete"))
    return out


def model_targets(entry: dict, variant: str | None = None) -> list[tuple[Path, bool]]:
    """這個模型自己的檔案或資料夾：[(路徑, 是不是資料夾)]。刪除、算佔用、找殘檔都只看這些。

    hf、onnx、gguf 是它自己的資料夾（含 Hugging Face 下載到一半的 .cache）；parts 是各個子資料夾；
    ckpt 放在共用資料夾，只算同名的檔案和下載中的 .part，audio-separator 的 download_checks.json 這類快取不算。
    有版本的模型指定版本時，只算那個版本的 gguf、.part 和舊版 huggingface_hub 留下的紀錄、暫存檔。
    """
    d = entry["dir"]
    if entry.get("variants") and variant:
        name = entry["variants"][variant]["file"]
        return ([(d / name, False), (d / (name + PART_SUFFIX), False)]
                + [(p, False) for p in hf_legacy_files(d, name)])
    if entry.get("parts"):
        return [(d / p["sub"], True) for p in entry["parts"]]
    if entry.get("kind") == "ckpt":
        names = entry.get("files") or [entry["file"]]
        return [(d / n, False) for n in names] + [(d / (n + PART_SUFFIX), False) for n in names]
    return [(d, True)]


def tree_bytes(path: Path) -> int:
    total = 0
    for root, dirs, files in os.walk(path):   # 不跟進捷徑
        for name in files:
            try:
                st = os.lstat(os.path.join(root, name))
            except OSError:
                continue
            if not stat.S_ISLNK(st.st_mode):
                total += st.st_size
    return total


def model_disk_bytes(entry: dict, variant: str | None = None) -> int:
    """這個模型（或它的某個版本）自己的檔案實際佔用的位元組（沒下載完的也算）。"""
    total = 0
    for path, is_dir in model_targets(entry, variant):
        try:
            if is_dir:
                if path.is_dir() and not path.is_symlink():
                    total += tree_bytes(path)
            elif path.is_file() and not path.is_symlink():
                total += path.stat().st_size
        except OSError:
            continue
    return total


# ---------- 翻譯模型：版本、llama.cpp 參數、顯存門檻 ----------

def translator_entry(key: str) -> dict | None:
    """翻譯模型 key（hymt、sakura……）對應的型錄項目。"""
    return next((e for e in MODEL_CATALOG.values() if e.get("translator") == key), None)


LLM_CTX_RANGE = (1024, 32768)
LLM_PARALLEL_RANGE = (1, 8)
LLM_MIN_SLOT_CTX = 1024          # 每個 slot（-c ÷ -np）至少要有這麼多 token，Sakura 一批 8 行才放得下


def llm_params_problem(key: str, params) -> str | None:
    """設定 llm_params 裡一個翻譯模型的參數有問題就回說明，沒問題回 None。"""
    label = TRANSLATORS[key]["label"] if key in TRANSLATORS else key
    if key not in TRANSLATORS:
        return f"沒有 {key} 這個翻譯模型"
    if not isinstance(params, dict) or set(params) - {"ctx", "parallel"}:
        return f"{label} 的參數只能設 ctx 和 parallel"
    ctx = params.get("ctx", TRANSLATORS[key]["ctx"])
    parallel = params.get("parallel", TRANSLATORS[key]["parallel"])
    if type(ctx) is not int or not LLM_CTX_RANGE[0] <= ctx <= LLM_CTX_RANGE[1]:
        return f"{label} 的 ctx 要是 {LLM_CTX_RANGE[0]} 到 {LLM_CTX_RANGE[1]} 的整數"
    if type(parallel) is not int or not LLM_PARALLEL_RANGE[0] <= parallel <= LLM_PARALLEL_RANGE[1]:
        return f"{label} 的 parallel 要是 {LLM_PARALLEL_RANGE[0]} 到 {LLM_PARALLEL_RANGE[1]} 的整數"
    if ctx // parallel < LLM_MIN_SLOT_CTX:
        return f"{label} 的 ctx ÷ parallel 至少要 {LLM_MIN_SLOT_CTX}"
    return None


def llm_overhead_mb(key: str, ctx: int) -> int:
    """gguf 以外要多留的顯存（KV cache、計算用的暫存）。表上有的 ctx 用表上的數字（overhead_mb 是預設 ctx 的，
    overhead_by_ctx 是其他 ctx 的）；表上沒有的從最接近的一格照每個 token 的 KV cache 加減（估計）。"""
    base = TRANSLATORS[key]
    table = {int(base["ctx"]): int(base["overhead_mb"])}
    table.update({int(c): int(mb) for c, mb in (base.get("overhead_by_ctx") or {}).items()})
    ctx = int(ctx)
    if ctx in table:
        return table[ctx]
    near = min(table, key=lambda c: (abs(c - ctx), -c))
    return max(200, int(table[near] + base.get("kv_mb_per_token", 0.125) * (ctx - near) + 0.999))


def _user_choices() -> tuple[dict, dict]:
    """設定頁存的 model_variants、llm_params。資料庫還沒開（子程序、部分測試）時當成沒設。"""
    try:
        from . import settings
        values = settings.all_values()
    except Exception:  # noqa: BLE001
        return {}, {}
    variants, params = values.get("model_variants"), values.get("llm_params")
    return (variants if isinstance(variants, dict) else {}), (params if isinstance(params, dict) else {})


def translator_cfg(key: str, variant: str | None = None, params: dict | None = None) -> dict:
    """翻譯模型實際要用的設定（TRANSLATORS[key] 的複本）：
      gguf、variant、size_mb、vram_basis  換成要用的版本（沒指定就照設定 model_variants 和已下載的版本挑，pick_variant）
      ctx、parallel                      套上設定 llm_params（有問題的值不用）
      overhead_mb                        照 ctx 算（llm_overhead_mb）"""
    base = TRANSLATORS[key]
    cfg = dict(base)
    chosen_variants, chosen_params = ({}, {}) if (variant and params is not None) else _user_choices()
    entry = translator_entry(key)
    if entry and entry.get("variants"):
        name = variant if variant in entry["variants"] else pick_variant(entry, chosen_variants.get(entry["id"]))
        v = entry["variants"][name]
        cfg.update(gguf=Path(entry["dir"]) / v["file"], variant=name, size_mb=v["size_mb"], vram_basis=v.get("vram"))
    p = chosen_params.get(key) if params is None else params
    if isinstance(p, dict) and not llm_params_problem(key, p):
        cfg.update(ctx=p.get("ctx", base["ctx"]), parallel=p.get("parallel", base["parallel"]))
    cfg["overhead_mb"] = llm_overhead_mb(key, cfg["ctx"])
    return cfg


def translator_vram_mb(key: str, size_mb: int | None = None, cfg: dict | None = None) -> int:
    """翻譯模型載入前檢查的顯存：gguf 檔案大小加上 overhead_mb（jobs.get_llm 用同一個算法）。
    cfg 是 translator_cfg 的結果（哪個版本、ctx），沒給就用目前設定要用的版本。
    檔案還沒下載時用那個版本在型錄上的大小；沒有版本的模型用傳進來的 size_mb（型錄的 size_mb）。"""
    cfg = cfg or translator_cfg(key)
    if cfg["gguf"].is_file():
        size_mb = int(cfg["gguf"].stat().st_size / 1024 / 1024)
    elif cfg.get("size_mb"):
        size_mb = cfg["size_mb"]
    return int(size_mb or 0) + cfg["overhead_mb"]


def model_vram_mb(entry: dict, variant: str | None = None) -> int | None:
    """載入前檢查的顯存門檻，跟實際檢查用的是同一個數字；在 CPU 上跑的回 None。
      辨識、對齊：batch_by_free_mb 最後一格（gpu.pick_batch 連一段都放不下就報錯的門檻）
      翻譯：translator_vram_mb（jobs.get_llm）；有版本的模型沒指定版本時是目前要用的版本
      人聲分離：SEPARATOR["vram_mb"]（jobs.handle_transcribe、handle_health）
    """
    role = entry.get("role")
    if role == "asr":
        return ASR_ENGINES[entry["engine"]]["batch_by_free_mb"][-1][0]
    if role == "aligner":
        return ALIGNER["batch_by_free_mb"][-1][0]
    if role == "translator":
        key = entry["translator"]
        if key not in TRANSLATORS:
            return None
        return translator_vram_mb(key, entry.get("size_mb"), cfg=translator_cfg(key, variant=variant))
    if role == "separator":
        return SEPARATOR["vram_mb"]
    return None   # furigana（onnxruntime CPU）、segmenter（CKIP 子程序關掉 CUDA）


# vram_mb 為載入前檢查的剩餘顯存下限（實測峰值再加餘裕）
ASR_ENGINES = {
    # batch_by_free_mb：依剩餘顯存決定一次處理幾段（實測 4 段 120 秒約吃 11GB）
    "qwen": {
        "label": "Qwen3-ASR-1.7B",
        "path": MODELS_DIR / "asr" / "Qwen3-ASR-1.7B",
        "langs": ["zh", "ja", "en"],
        # 60 秒：對齊誤差不會在一段裡累積太久，辨識的上下文也還夠
        "max_chunk_s": 60,
        "batch_by_free_mb": [(12000, 6), (8500, 3), (6000, 1)],
    },
    "anime": {
        "label": "anime-whisper",
        "path": MODELS_DIR / "asr" / "anime-whisper",
        "langs": ["ja"],
        "family": "whisper",
        "max_chunk_s": 28,
        "batch_by_free_mb": [(6000, 8), (4000, 4), (3000, 1)],
    },
    "whisper": {
        "label": "Whisper large-v3",
        "path": MODELS_DIR / "asr" / "whisper-large-v3",
        "langs": ["zh", "ja", "en"],
        "family": "whisper",
        "max_chunk_s": 28,
        "batch_by_free_mb": [(7000, 8), (5000, 4), (3500, 1)],
    },
}

WHISPER_LANG = {"zh": "chinese", "ja": "japanese", "en": "english"}

ALIGNER = {
    "label": "Qwen3-ForcedAligner-0.6B",
    "path": MODELS_DIR / "asr" / "Qwen3-ForcedAligner-0.6B",
    "batch_by_free_mb": [(7000, 4), (5000, 2), (3500, 1)],
}

# 顯卡沒有事做（沒有執行中的顯卡任務，佇列裡也沒有會開始的）滿這麼多秒，就把顯卡上的模型全部釋放：
# 常駐的語音模型（app/speech_worker.py）和翻譯的 llama-server。這不是專門的伺服器，平常要讓使用者正常用電腦；
# 重開語音模型約 15 秒（import 9 秒＋載入 5 秒），翻譯模型約 10 秒。中間有新任務進來就重新計時（jobs.idle_tick）
GPU_IDLE_RELEASE_S = 120

# 常駐語音模型（app/speech_worker.py、app/speech.py）：辨識、對齊、時間軸檢查的模型放在同一個常駐程序裡，
# 顯存放得下就同時載入、跨影片沿用，放不下就在同一個程序裡輪流載入。環境變數 VS_SPEECH_WORKER=0 改回每一步開一個子程序
SPEECH_WORKER = {
    "idle_release_s": GPU_IDLE_RELEASE_S,   # 顯示用；閒置釋放由 jobs.idle_tick 統一計時
    "margin_mb": 1024,        # 算預算時留給桌面、瀏覽器播放影片的餘裕（這些程式的顯存會跳動）
    "context_mb": 300,        # 每個程序的 CUDA context（torch 統計不到，實測 200 到 260）
    "stall_s": 600,           # 這麼久沒有任何進度就當成卡住，結束程序（一批最長不到 20 秒）
    "cancel_wait_s": 15,      # 取消後等這麼久還沒停下來就直接結束程序（辨識一批最長約 9 秒）
    # 開程序到 import 完、CUDA 可以用，實測 9 到 15 秒；這麼久還沒好就當成開不起來，這次開機改用單次子程序
    "start_timeout_s": 180,
}

# 語音模型的顯存（MiB），2026-09-17 在 RTX 5070 Ti（torch 2.14 cu130、expandable_segments）實測：
#   resident_mb   載入後常駐的量（torch allocated）
#   load_extra_mb 載入途中比常駐多用的量
#   stage_mb      各階段、各批次大小推論時比常駐多用的量（實測最大值加 25%）；whisper 沒實測，照舊表估計
#   prefer        批次大小的偏好順序，從大的開始挑第一個放得下的
#   secs_per_100s 每 100 秒語音要幾秒（只用來比較「同時載入但批次縮小」和「輪流載入」哪個快）
#   load_s        載入一次要幾秒
SPEECH_VRAM = {
    "resident_mb": {"qwen": 3900, "aligner": 1760, "anime": 1450, "whisper": 3000},
    "load_extra_mb": {"qwen": 600, "aligner": 50, "anime": 150, "whisper": 400},
    "stage_mb": {
        "asr:qwen": {8: 3300, 6: 2500, 3: 1250, 2: 850, 1: 600},
        "asr:anime": {8: 1300, 4: 700, 1: 200},
        "asr:whisper": {8: 3700, 4: 1900, 1: 600},
        "align": {4: 1000, 2: 600, 1: 600},
        "realign": {4: 1000, 2: 600, 1: 600},
        "verify": {24: 1200, 12: 650, 8: 500, 4: 250},
    },
    "prefer": {
        "asr:qwen": [6, 3, 2, 1], "asr:anime": [8, 4, 1], "asr:whisper": [8, 4, 1],
        "align": [4, 2, 1], "realign": [4, 2, 1], "verify": [24, 12, 8, 4],
    },
    "secs_per_100s": {
        "asr:qwen": {8: 3.5, 6: 3.6, 3: 6.0, 2: 7.8, 1: 10.5},
        "asr:anime": {8: 1.0, 4: 1.2, 1: 2.5}, "asr:whisper": {8: 3.0, 4: 3.6, 1: 7.5},
        "align": {4: 0.25, 2: 0.25, 1: 0.3}, "realign": {4: 0.1, 2: 0.1, 1: 0.1},
        "verify": {24: 1.9, 12: 2.8, 8: 4.0, 4: 6.7},
    },
    "load_s": {"qwen": 3.3, "aligner": 1.8, "anime": 1.4, "whisper": 2.5},
}

# 翻譯模型。key 不能改（任務參數、設定、glossgen 都用 key）。
# gguf 是預設版本的檔案；有版本的模型實際用哪個檔案、ctx、parallel 由 translator_cfg 照設定決定。
# ctx、parallel 是 llama-server 的 -c（所有 slot 共用的總量）、-np；設定 llm_params 可以改。
# overhead_mb 是預設 ctx 時 gguf 以外要多留的顯存；overhead_by_ctx 是其他 ctx 的（打包計畫 6.4.2，估計）；
# kv_mb_per_token 是每個 token 的 KV cache（層數 × KV head × 維度 × 2 × f16 的 2 bytes），表上沒有的 ctx 照它加減
TRANSLATORS = {
    "hymt": {
        "label": "Hy-MT2-7B",
        "gguf": MODELS_DIR / "gguf" / "Hy-MT2-7B" / "HY-MT2-7B-Q8_0.gguf",
        "langs": ["ja", "en"],
        "ctx": 8192,
        "parallel": 4,
        "overhead_mb": 1200,
        "overhead_by_ctx": {4096: 700},      # ctx 4096、np 2：每個 slot 一樣 2048 token
        "kv_mb_per_token": 0.125,            # 32 層 × 8 × 128
    },
    "sakura": {
        "label": "Sakura-14B",
        "gguf": MODELS_DIR / "gguf" / "Sakura-14B" / "sakura-14b-qwen2.5-v1.0-q6k.gguf",
        "langs": ["ja"],
        "ctx": 4096,
        "parallel": 2,
        "overhead_mb": 700,
        "kv_mb_per_token": 0.1875,           # 48 層 × 8 × 128
    },
    "hymt-mini": {
        "label": "Hy-MT2-1.8B",
        "gguf": MODELS_DIR / "gguf" / "Hy-MT2-1.8B" / "Hy-MT2-1.8B-Q8_0.gguf",
        "langs": ["ja", "en"],
        "ctx": 8192,
        "parallel": 4,
        "overhead_mb": 900,
        "kv_mb_per_token": 0.0625,           # 32 層 × 4 × 128
    },
    "galtransl": {
        "label": "Sakura-GalTransl-7B",
        "gguf": MODELS_DIR / "gguf" / "GalTransl-7B" / "Sakura-Galtransl-7B-v3.7.gguf",
        "langs": ["ja"],
        "ctx": 4096,
        "parallel": 2,
        "overhead_mb": 700,
        "kv_mb_per_token": 0.0547,           # 28 層 × 4 × 128
    },
}

SAKURA_STYLE = ("sakura", "galtransl")  # 用 Sakura 那套 prompt 與分批方式


def default_engine(language: str, profile: str | None = None) -> str:
    """內建預設：不分內容類型，使用者遇到特殊內容（動畫、耳語）再自己換模型。"""
    return "qwen"


def default_translator(language: str, profile: str | None = None) -> str | None:
    return None if language == "zh" else "hymt"


def nvenc_allowed() -> bool:
    """相容播放檔要不要先試 NVENC：測試開關 VS_FORCE_NO_NVENC=1 或安裝時測過 NVENC 不能用就不試。"""
    return os.environ.get("VS_FORCE_NO_NVENC") != "1" and INSTALL_STATE.get("nvenc") is not False


def ensure_dirs():
    for d in (DATA_DIR, MEDIA_DIR, WORK_DIR, SUBS_DIR, THUMB_DIR, PROXY_DIR, DICT_DIR):
        d.mkdir(parents=True, exist_ok=True)
