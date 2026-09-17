"""把程式打包成發佈用的 zip。

一般包（打包計畫 D9）：只有程式碼，執行環境第一次啟動時下載。
完整包（--full）：一般包再加上 offline 資料夾，裡面是放在 GitHub 上、朋友那邊下載很慢的檔案
（runtime-manifest.json 裡的 ffmpeg、llama.cpp、cudart，uv.lock 裡的 en-core-web-sm，和日文、英文字典的原始資料）。
安裝時 launcher/setup_runtime.py、app/dict_build.py 先找 offline 資料夾，sha256 對就直接用。
PyPI、PyTorch 官網、Hugging Face 上的東西照常下載，不放進 zip。

用白名單挑程式檔案，不是黑名單：沒列在下面的檔案一律不放。另外再擋一次絕對不能出現的東西
（data、models、bin、runtime、private、tests、docs、.dev 開發標記、__pycache__、影片、資料庫、log），
白名單不小心寫錯也放不進去。offline 資料夾的檔案不走白名單，只放 offline_items() 列出的、sha256 核對過的檔案。

用法：
    python -s tools/build_release.py --out <輸出資料夾> [--name BilingualSubtitles]
    python -s tools/build_release.py --out <輸出資料夾> --full [--cache 資料夾] [--connections 32] [--no-fetch]
    python -s tools/build_release.py --fetch-only [--cache 資料夾]      只把完整包要附的檔案下載到快取

輸出 <輸出資料夾>\\BilingualSubtitles-<版本>.zip（完整包是 BilingualSubtitles-<版本>-full.zip）和 .sha256；zip 裡最上層是 BilingualSubtitles\\ 資料夾。
版本讀 pyproject.toml 的 version。README.md 在 zip 裡改名成 README.txt（朋友雙擊就能用記事本打開）。
完整包的檔案來源是快取資料夾（預設 private\\bundle-cache，被 .gitignore 擋掉）：沒有或 sha256 不對就下載
（setup_runtime 的多連線、續傳、sha256 下載器），下載紀錄寫在快取資料夾的 fetch-log.json。
JMdict 每天更新、沒有固定的 sha256：快取裡有、驗得過就沿用（要換新的就刪掉快取裡的 JMdict_e.gz），打包時算 sha256 寫進 offline\\files.json。
"""
import argparse
import fnmatch
import hashlib
import json
import sys
import time
import tomllib
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# (來源, zip 裡的名稱)；來源是相對根目錄的路徑
ROOT_FILES = [
    ("start.bat", "start.bat"),
    ("repair.bat", "repair.bat"),
    ("README.md", "README.txt"),
    ("LICENSE", "LICENSE"),
    ("THIRD_PARTY_NOTICES.md", "THIRD_PARTY_NOTICES.md"),
    ("pyproject.toml", "pyproject.toml"),
    ("uv.lock", "uv.lock"),
    (".python-version", ".python-version"),
]
OPTIONAL_ROOT_FILES = [("VERSION", "VERSION")]

# (資料夾, 要的檔案樣式, 要不要往下找子資料夾)
TREES = [
    ("app", ["*.py"], False),
    ("web", ["*.html", "*.js", "*.css", "*.svg", "*.png", "*.ico", "*.webmanifest", "*.woff2"], True),
    ("launcher", ["*.py", "*.cmd", "runtime-manifest.json"], False),
    ("launcher/msg", ["*.txt"], False),
    ("tools", ["build_dict_en.py", "build_dict_ja.py", "build_dict_zh.py", "download_ckip.py"], False),
    ("tools/yomitan", ["*"], True),
    ("licenses", ["*"], True),
]

# 不管白名單怎麼寫都不能放的
FORBIDDEN_PARTS = {"data", "models", "bin", "runtime", "private", "tests", "docs", "__pycache__", ".git", ".venv",
                   "venv", "node_modules", ".pytest_cache", ".vscode", ".idea", "offline"}
FORBIDDEN_NAMES = {".dev", ".env", "thumbs.db", "desktop.ini", "install-state.json", "precheck.json", "server.json",
                   "server.lock", "paths.json"}
FORBIDDEN_PATTERNS = ["*.pyc", "*.pyo", "*.db", "*.db-*", "*.sqlite", "*.log", "*.tmp", "*.bak", "*.bak.json",
                      "*.gguf", "*.safetensors", "*.onnx", "*.ckpt", "*.pt", "*.pth", "*.bin", "*.exe", "*.dll",
                      "*.mp4", "*.mkv", "*.webm", "*.mov", "*.avi", "*.wav", "*.mp3", "*.m4a", "*.flac", "*.srt",
                      "*.vtt", "*.ass", "*cookies*", "*.part", "*.whl", "*.zip", "*.7z"]
MAX_FILE_BYTES = 2 * 1024 * 1024
CRLF_SUFFIXES = {".bat", ".cmd"}

# ---------- 完整包 ----------

OFFLINE = "offline"                 # zip 裡的資料夾名稱，跟 setup_runtime.OFFLINE、dict_build.OFFLINE_DIR 一致
OFFLINE_LIST = "files.json"
OFFLINE_README = "README.txt"
CACHE_DIR = ROOT / "private" / "bundle-cache"
FETCH_LOG = "fetch-log.json"
FULL_DICT_LANGS = ("ja", "en")      # 第一次打開自動建的字典（app/setup.py）；中文辭典要按按鈕才建，不附
FETCH_CONNECTIONS = 32              # GitHub release 每條連線只有 30 到 50 KB/s（2026-09-17 實測）
STORED_SUFFIXES = {".zip", ".7z", ".gz", ".xz", ".whl"}      # 已經壓縮過的檔案不再壓一次

MIT_TEXT = """MIT License

Copyright (c) {holder}

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
# MIT 授權要求附上授權全文，原本的壓縮檔裡沒有（llama.cpp 的 zip 只有 LLVM OpenMP 的授權、stardict.7z 只有 csv）
OFFLINE_LICENSES = {
    "LICENSE-llama.cpp.txt": ("llama.cpp（https://github.com/ggml-org/llama.cpp）的授權，"
                              "llama-b10985-bin-win-cuda-13.4-x64.zip 來自這個專案。", "2023-2026 The ggml authors"),
    "LICENSE-ECDICT.txt": ("ECDICT（https://github.com/skywind3000/ECDICT）的授權，stardict.7z 來自這個專案。",
                           "2025 Linwei"),
}


def license_text(name: str) -> str:
    intro, holder = OFFLINE_LICENSES[name]
    return f"{intro}\n\n" + MIT_TEXT.format(holder=holder)


class ReleaseError(Exception):
    pass


def forbidden(rel: str) -> str | None:
    parts = rel.replace("\\", "/").split("/")
    low = [p.lower() for p in parts]
    for p in low[:-1]:
        if p in FORBIDDEN_PARTS:
            return f"資料夾 {p}"
    name = low[-1]
    if name in FORBIDDEN_NAMES:
        return f"檔名 {name}"
    for pat in FORBIDDEN_PATTERNS:
        if fnmatch.fnmatch(name, pat):
            return f"樣式 {pat}"
    return None


def collect(root: Path = ROOT) -> list[tuple[Path, str]]:
    files: list[tuple[Path, str]] = []
    for src, dst in ROOT_FILES:
        p = root / src
        if not p.is_file():
            raise ReleaseError(f"缺少必要的檔案：{src}")
        files.append((p, dst))
    for src, dst in OPTIONAL_ROOT_FILES:
        if (root / src).is_file():
            files.append((root / src, dst))
    for folder, patterns, recursive in TREES:
        base = root / folder
        if not base.is_dir():
            continue
        candidates = base.rglob("*") if recursive else base.glob("*")
        for p in sorted(candidates):
            if not p.is_file() or p.is_symlink():
                continue
            rel = p.relative_to(root).as_posix()
            if not any(fnmatch.fnmatch(p.name, pat) for pat in patterns):
                continue
            if forbidden(rel):      # 例如 web\__pycache__、tools\yomitan\node_modules：直接跳過
                continue
            files.append((p, rel))
    seen = set()
    out = []
    for p, rel in files:
        if rel in seen:
            continue
        seen.add(rel)
        out.append((p, rel))
    return out


def check(files: list[tuple[Path, str]]):
    problems = []
    for p, rel in files:
        why = forbidden(rel)
        if why:
            problems.append(f"{rel}：不能放進發佈檔（{why}）")
        size = p.stat().st_size
        if size > MAX_FILE_BYTES:
            problems.append(f"{rel}：檔案太大（{size / 1048576:.1f} MB），確認是不是不該放的東西")
        if p.suffix.lower() in CRLF_SUFFIXES:
            data = p.read_bytes()
            if data.count(b"\n") != data.count(b"\r\n"):
                problems.append(f"{rel}：換行不是 CRLF（cmd 用 LF 換行時 goto、call 會出錯）")
    names = {rel for _, rel in files}
    for must in ("start.bat", "launcher/setup_runtime.py", "launcher/precheck.py", "launcher/env.cmd",
                 "launcher/get_uv.cmd", "launcher/runtime-manifest.json", "launcher/msg/install_failed.txt",
                 "app/server.py", "web/index.html", "uv.lock"):
        if must not in names:
            problems.append(f"少了 {must}")
    if problems:
        raise ReleaseError("打包檢查沒通過：\n  " + "\n  ".join(problems))


def version(root: Path = ROOT) -> str:
    data = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    return data["project"]["version"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _import_path(folder: Path):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))


def _setup_runtime():
    _import_path(ROOT / "launcher")
    import setup_runtime
    return setup_runtime


def slow_host(url: str) -> bool:
    return _setup_runtime().slow_host(url)


def offline_items(root: Path = ROOT, dict_sources: dict | None = None) -> list[dict]:
    """完整包要附的檔案：安裝執行環境時要從 GitHub 下載的檔案（runtime-manifest.json 的 ffmpeg、llama.cpp、cudart，
    uv.lock 的 en-core-web-sm，跟 setup_runtime.slow_runtime_files 同一份清單），和第一次打開自動建的字典（日文、英文）的原始資料。
    每一項 {"name", "urls", "sha256"（JMdict 是 None）, "size"（JMdict、uv.lock 沒寫大小的 wheel 是 None）,
    "for": "runtime"、"python" 或 "dict:<語言>"}。"""
    sr = _setup_runtime()
    manifest = json.loads((root / "launcher" / "runtime-manifest.json").read_text(encoding="utf-8"))
    items = []
    for f in sr.slow_runtime_files(manifest, root / "uv.lock"):
        items.append({"name": f["name"], "urls": list(f["urls"]), "sha256": f["sha256"].lower(), "size": f.get("size"),
                      "for": "python" if f.get("package") else "runtime"})
    if dict_sources is None:
        _import_path(ROOT)
        from app import dict_build
        dict_sources = dict_build.SOURCES
    for lang in FULL_DICT_LANGS:
        for s in dict_sources[lang]:
            items.append({"name": s["name"], "urls": list(s["urls"]), "sha256": (s.get("sha256") or "").lower() or None,
                          "size": s.get("bytes"), "for": f"dict:{lang}", "daily": bool(s.get("daily"))})
    names = [i["name"] for i in items]
    dup = {n for n in names if names.count(n) > 1}
    if dup:
        raise ReleaseError(f"完整包的檔名重複：{', '.join(sorted(dup))}")
    return items


def _cached_ok(item: dict, path: Path) -> str | None:
    """快取裡的檔案對不對：對的話回 sha256，不對回 None。JMdict 沒有固定的 sha256，改成完整解一次 gzip 驗證。"""
    if not path.is_file():
        return None
    if item.get("sha256"):
        if item.get("size") and path.stat().st_size != item["size"]:
            return None
        got = sha256_file(path)
        return got if got == item["sha256"] else None
    _import_path(ROOT)
    from app import dict_build
    from app.model_download import DownloadError
    try:
        dict_build.check_jmdict(path)
    except DownloadError:
        return None
    return sha256_file(path)


def _download_plain(item: dict, cache: Path) -> tuple[Path, str]:
    """沒有固定 sha256 的檔案（JMdict）：一條連線下載到 .part，驗得過才改名。回傳 (路徑, 用的網址)。"""
    final = cache / item["name"]
    part = cache / (item["name"] + ".part")
    last = None
    for url in item["urls"]:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "video-subtitle-release/1",
                                                       "Accept-Encoding": "identity"})
            with urllib.request.urlopen(req, timeout=60) as r, open(part, "wb") as out:
                while True:
                    chunk = r.read(1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
            if not _cached_ok(item, part):
                raise ReleaseError(f"{item['name']} 下載下來驗不過")
            part.replace(final)
            return final, url
        except Exception as e:  # noqa: BLE001
            last = e
            part.unlink(missing_ok=True)
    raise ReleaseError(f"{item['name']} 下載失敗：{last}")


def fetch_cache(items: list[dict], cache: Path = CACHE_DIR, connections: int = FETCH_CONNECTIONS,
                download: bool = True) -> dict[str, str]:
    """確認快取裡每個檔案都在、sha256 對；沒有或不對就下載（download=False 時直接報錯）。回傳 {檔名: sha256}。
    下載紀錄（網址、大小、秒數）加到快取資料夾的 fetch-log.json。"""
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    sums, missing = {}, []
    for item in items:
        got = _cached_ok(item, cache / item["name"])
        if got:
            sums[item["name"]] = got
        else:
            missing.append(item)
    if not missing:
        return sums
    if not download:
        raise ReleaseError("快取裡缺少或 sha256 不對：" + "、".join(i["name"] for i in missing)
                           + f"（快取資料夾 {cache}）")
    sr = _setup_runtime()
    # 下載器的 log 寫到快取資料夾，不能在專案裡建 runtime\
    sr.LOGS = cache / "logs"
    sr.LOG_PATH = sr.LOGS / "fetch.log"
    sr.CONNECTIONS = max(1, int(connections))
    sr.SEGMENT_SIZE = 4 * 1024 * 1024
    sr.SEGMENT_MIN = 1024 * 1024        # 12 MB 的 en-core-web-sm 也分段（一條連線要 5 分鐘）
    sr.MAX_SEGMENTS = 256
    con = sr.Console()
    con.start()
    log_path = cache / FETCH_LOG
    records = json.loads(log_path.read_text(encoding="utf-8")) if log_path.is_file() else []
    try:
        for item in missing:
            started = time.time()
            size_text = f"{item['size'] / 1048576:,.1f} MB" if item.get("size") else "大小要問伺服器"
            print(f"下載 {item['name']}（{size_text}）", flush=True)
            if item.get("sha256"):
                path = sr.download({"name": item["name"], "urls": item["urls"], "sha256": item["sha256"],
                                    "size": item.get("size")}, item["name"], con, dest=cache, key=item["name"])
                url = item["urls"][0] if len(item["urls"]) == 1 else _used_url(sr.LOG_PATH, item["name"])
            else:
                path, url = _download_plain(item, cache)
            seconds = time.time() - started
            size = path.stat().st_size
            sums[item["name"]] = sha256_file(path)
            record = {"name": item["name"], "url": url, "size": size, "sha256": sums[item["name"]],
                      "seconds": round(seconds, 1), "mb_per_s": round(size / 1048576 / max(seconds, 0.001), 2),
                      "connections": sr.CONNECTIONS if item.get("sha256") else 1,
                      "at": time.strftime("%Y-%m-%d %H:%M:%S")}
            records.append(record)
            log_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
            con.say(f"  完成 {item['name']}：{size / 1048576:,.1f} MB，{seconds:.0f} 秒")
    except sr.SetupError as e:
        raise ReleaseError(str(e)) from e
    finally:
        con.stop()
    return sums


def _used_url(log_path: Path, name: str) -> str | None:
    try:
        lines = [l for l in log_path.read_text(encoding="utf-8").splitlines() if f"download {name} from " in l]
    except OSError:
        return None
    return lines[-1].split(" from ", 1)[1].rsplit(" (attempt", 1)[0] if lines else None


def offline_list(items: list[dict], sums: dict[str, str], cache: Path) -> dict:
    files = []
    for item in items:
        path = Path(cache) / item["name"]
        entry = {"name": item["name"], "size": path.stat().st_size, "sha256": sums[item["name"]], "for": item["for"],
                 "source": item["urls"][0]}
        if item.get("daily"):
            entry["date"] = time.strftime("%Y-%m-%d", time.localtime(path.stat().st_mtime))
        files.append(entry)
    return {"format": 1,
            "note": "完整包附的下載檔。安裝時核對 sha256，對了就直接用，不對或沒有才從原本的網址下載。",
            "files": files}


def offline_readme(root: Path = ROOT) -> str:
    manifest = json.loads((root / "launcher" / "runtime-manifest.json").read_text(encoding="utf-8"))
    ff, ll = manifest["ffmpeg"], manifest["llama_cpp"]
    return (
        "這裡是完整包附的安裝檔，安裝時直接拿來用，不用再從 GitHub 慢慢下載。\n"
        "程式和字典都裝好以後，可以刪掉整個 offline 資料夾。\n"
        "\n"
        "授權（詳細見上一層的 THIRD_PARTY_NOTICES.md）：\n"
        f"- ffmpeg {ff['version']}（{ff['build']}）：GPL-3.0-or-later。\n"
        f"  原始碼 {ff['source']}\n"
        "  建置腳本和相依函式庫的版本 https://github.com/BtbN/FFmpeg-Builds\n"
        f"- llama.cpp {ll['version']}：MIT，全文在 LICENSE-llama.cpp.txt。\n"
        "- cudart-llama-bin（NVIDIA CUDA 執行階段檔案）：依 NVIDIA CUDA EULA 可隨程式散布，\n"
        "  https://docs.nvidia.com/cuda/eula/index.html\n"
        "- en_core_web_sm（spaCy 英文模型，Explosion）：MIT。wheel 裡有授權全文 LICENSE，\n"
        "  和訓練資料的授權說明 LICENSES_SOURCES\n"
        "- JMdict_e.gz：EDRDG，CC BY-SA 4.0，https://www.edrdg.org/edrdg/licence.html\n"
        "- jlpt_n1~5.csv：Stephen Kraus（yomitan-jlpt-vocab），CC BY-SA 4.0；原始資料 Jonathan Waller，CC BY\n"
        "- accents.txt：Kanjium（Uros O.），CC BY-SA 4.0\n"
        "- stardict.7z：ECDICT，MIT，全文在 LICENSE-ECDICT.txt\n"
        "- cefrj-vocabulary-profile-1.5.csv：The CEFR-J Wordlist Version 1.5. Compiled by Yukio Tono,\n"
        "  Tokyo University of Foreign Studies. Retrieved from http://www.cefr-j.org/download.html on 1/20/2020.\n"
        "- octanove-vocabulary-profile-c1c2-1.0.csv：Octanove Labs，CC BY-SA 4.0\n"
        "CC BY-SA 4.0 全文：https://creativecommons.org/licenses/by-sa/4.0/\n"
    )


def _zip_bytes(z: zipfile.ZipFile, arcname: str, data: bytes, compress: bool = True):
    info = zipfile.ZipInfo(arcname, date_time=(2026, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED if compress else zipfile.ZIP_STORED
    info.external_attr = 0o644 << 16
    z.writestr(info, data)


def _zip_file(z: zipfile.ZipFile, arcname: str, path: Path, compress: bool):
    info = zipfile.ZipInfo(arcname, date_time=(2026, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED if compress else zipfile.ZIP_STORED
    info.external_attr = 0o644 << 16
    info.file_size = path.stat().st_size
    with open(path, "rb") as src, z.open(info, "w", force_zip64=info.file_size > 0x7FFFFFFF) as out:
        while True:
            chunk = src.read(8 * 1024 * 1024)
            if not chunk:
                break
            out.write(chunk)


def build(out_dir: Path, name: str = "BilingualSubtitles", root: Path = ROOT, full: bool = False, cache: Path | None = None,
          items: list[dict] | None = None, connections: int = FETCH_CONNECTIONS, fetch: bool = True) -> Path:
    files = collect(root)
    check(files)
    out_dir = Path(out_dir)
    if out_dir.resolve() == root.resolve() or root.resolve() in out_dir.resolve().parents:
        raise ReleaseError("輸出資料夾不能在專案裡面，請指定專案外的資料夾")
    offline = []
    if full:
        cache = Path(cache or CACHE_DIR)
        items = offline_items(root) if items is None else items
        sums = fetch_cache(items, cache, connections, download=fetch)
        offline = [(cache / i["name"], f"{OFFLINE}/{i['name']}") for i in items]
        listing = offline_list(items, sums, cache)
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / f"{name}-{version(root)}{'-full' if full else ''}.zip"
    tmp = zip_path.with_name(zip_path.name + ".tmp")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for p, rel in sorted(files, key=lambda x: x[1]):
            _zip_bytes(z, f"{name}/{rel}", p.read_bytes())
        if full:
            _zip_bytes(z, f"{name}/{OFFLINE}/{OFFLINE_LIST}",
                       json.dumps(listing, ensure_ascii=False, indent=2).encode("utf-8"))
            _zip_bytes(z, f"{name}/{OFFLINE}/{OFFLINE_README}", offline_readme(root).replace("\n", "\r\n").encode("utf-8"))
            for lic in OFFLINE_LICENSES:
                _zip_bytes(z, f"{name}/{OFFLINE}/{lic}", license_text(lic).replace("\n", "\r\n").encode("utf-8"))
            for p, rel in offline:
                _zip_file(z, f"{name}/{rel}", p, compress=p.suffix.lower() not in STORED_SUFFIXES)
    if full:
        # 打包時快取的檔案被換掉（另一個視窗在下載）：再核對一次 zip 裡的內容
        with zipfile.ZipFile(tmp) as z:
            for i in items:
                h = hashlib.sha256()
                with z.open(f"{name}/{OFFLINE}/{i['name']}") as f:
                    for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
                        h.update(chunk)
                if h.hexdigest() != sums[i["name"]]:
                    tmp.unlink(missing_ok=True)
                    raise ReleaseError(f"{i['name']} 打包時內容變了，請重新打包")
    tmp.replace(zip_path)
    digest = sha256_file(zip_path)
    zip_path.with_name(zip_path.name + ".sha256").write_text(f"{digest}  {zip_path.name}\n", encoding="utf-8")
    total = sum(p.stat().st_size for p, _ in files + offline)
    print(f"{zip_path}")
    print(f"  {len(files) + len(offline)} 個檔案，原始 {total / 1024:,.0f} KB，壓縮後 {zip_path.stat().st_size / 1024:,.0f} KB")
    print(f"  sha256 {digest}")
    return zip_path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="打包發佈用的 zip（白名單）")
    parser.add_argument("--out", help="輸出資料夾（要在專案外面）")
    parser.add_argument("--name", default="BilingualSubtitles", help="zip 檔名和裡面最上層資料夾的名稱")
    parser.add_argument("--list", action="store_true", help="只列出會放進去的檔案，不打包")
    parser.add_argument("--full", action="store_true", help="完整包：附上 GitHub 上的下載檔（offline 資料夾）")
    parser.add_argument("--cache", default=str(CACHE_DIR), help="完整包的檔案快取資料夾")
    parser.add_argument("--connections", type=int, default=FETCH_CONNECTIONS, help="下載時每個檔案的連線數")
    parser.add_argument("--no-fetch", action="store_true", help="快取裡缺檔案時直接報錯，不下載")
    parser.add_argument("--fetch-only", action="store_true", help="只把完整包要附的檔案下載到快取，不打包")
    args = parser.parse_args(argv)
    try:
        if args.list:
            files = collect()
            check(files)
            for _, rel in files:
                print(rel)
            if args.full:
                for item in offline_items():
                    print(f"{OFFLINE}/{item['name']}")
            return 0
        if args.fetch_only:
            items = offline_items()
            sums = fetch_cache(items, Path(args.cache), args.connections)
            for item in items:
                p = Path(args.cache) / item["name"]
                print(f"{item['name']}\t{p.stat().st_size}\t{sums[item['name']]}")
            return 0
        if not args.out:
            parser.error("要指定 --out")
        build(Path(args.out), args.name, full=args.full, cache=Path(args.cache), connections=args.connections,
              fetch=not args.no_fetch)
    except ReleaseError as e:
        print(e)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
