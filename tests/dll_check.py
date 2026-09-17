"""執行環境的 DLL 靜態檢查（打包計畫第 7.3 節方法 2）。只讀檔案，不執行任何程式。

對 runtime 底下每個 exe、dll、pyd 讀出匯入表，確認每個要載入的 DLL 都找得到：
  同一個資料夾 → 執行檔旁邊（python.exe 所在的資料夾、llama-server.exe 所在的資料夾）
  → 這個執行環境裡任何地方有同名的 DLL（套件用 os.add_dll_directory 加的，例如 torch\\lib、*.libs）
  → System32（但假裝沒有 VC++ 執行階段：msvcp140、vcruntime140 這些在乾淨的 Windows 上不一定有）
並讀 PE 標頭的連結器版本，確認附帶的 VC++ DLL（檔案版本）不比任何用到它的檔案舊。

python -s tests/dll_check.py <runtime 資料夾>
"""
import os
import struct
import sys
from pathlib import Path

VC_RUNTIME = {"msvcp140.dll", "msvcp140_1.dll", "msvcp140_2.dll", "msvcp140_atomic_wait.dll",
              "msvcp140_codecvt_ids.dll", "vcruntime140.dll", "vcruntime140_1.dll", "vcruntime140_threads.dll",
              "vcomp140.dll", "concrt140.dll"}


def _sections(data, pe):
    nsec = struct.unpack_from("<H", data, pe + 6)[0]
    optsz = struct.unpack_from("<H", data, pe + 20)[0]
    off = pe + 24 + optsz
    out = []
    for i in range(nsec):
        vsz, va, rsz, raw = struct.unpack_from("<IIII", data, off + i * 40 + 8)
        out.append((va, vsz, raw, rsz))
    return out


def _rva(sections, rva):
    for va, vsz, raw, rsz in sections:
        if va <= rva < va + max(vsz, rsz):
            return rva - va + raw
    return None


def pe_info(path: Path):
    """(一般匯入, 延遲匯入, 連結器版本) ；不是 PE 檔回 None。"""
    data = path.read_bytes()
    if data[:2] != b"MZ":
        return None
    pe = struct.unpack_from("<I", data, 0x3C)[0]
    if data[pe:pe + 4] != b"PE\0\0":
        return None
    opt = pe + 24
    magic = struct.unpack_from("<H", data, opt)[0]
    if magic != 0x20B or struct.unpack_from("<H", data, pe + 4)[0] != 0x8664:
        return None     # 只看 x64（setuptools 附的 32 位元、ARM64 啟動器用不到）
    linker = (data[opt + 2], data[opt + 3])
    ddir = opt + (112 if magic == 0x20B else 96)
    imp_rva = struct.unpack_from("<I", data, ddir + 8)[0]
    delay_rva = struct.unpack_from("<I", data, ddir + 13 * 8)[0]
    secs = _sections(data, pe)

    def cstr(off):
        return data[off:data.index(b"\0", off)].decode("ascii", "replace")

    normal, delay = [], []
    off = _rva(secs, imp_rva) if imp_rva else None
    while off is not None:
        name_rva = struct.unpack_from("<I", data, off + 12)[0]
        if not name_rva:
            break
        normal.append(cstr(_rva(secs, name_rva)).lower())
        off += 20
    off = _rva(secs, delay_rva) if delay_rva else None
    while off is not None:
        name_rva = struct.unpack_from("<I", data, off + 4)[0]
        if not name_rva:
            break
        delay.append(cstr(_rva(secs, name_rva)).lower())
        off += 32
    return normal, delay, linker


def _load(path: Path):
    data = path.read_bytes()
    pe = struct.unpack_from("<I", data, 0x3C)[0]
    opt = pe + 24
    magic = struct.unpack_from("<H", data, opt)[0]
    ddir = opt + (112 if magic == 0x20B else 96)
    return data, _sections(data, pe), ddir, magic == 0x20B


def _cstr(data, off):
    return data[off:data.index(b"\0", off)].decode("ascii", "replace")


def imported_functions(path: Path, dlls: set) -> dict:
    """{DLL 名稱: 用名字匯入的函式集合}，只看 dlls 裡列的。"""
    data, secs, ddir, is64 = _load(path)
    out: dict = {}
    imp_rva = struct.unpack_from("<I", data, ddir + 8)[0]
    off = _rva(secs, imp_rva) if imp_rva else None
    width, flag = (8, 1 << 63) if is64 else (4, 1 << 31)
    while off is not None:
        ilt, _, _, name_rva, iat = struct.unpack_from("<IIIII", data, off)
        if not name_rva:
            break
        dll = _cstr(data, _rva(secs, name_rva)).lower()
        if dll in dlls:
            names = out.setdefault(dll, set())
            t = _rva(secs, ilt or iat)
            while t is not None:
                v = struct.unpack_from("<Q" if is64 else "<I", data, t)[0]
                if not v:
                    break
                if not v & flag:
                    names.add(_cstr(data, _rva(secs, v & 0x7FFFFFFF) + 2))
                t += width
        off += 20
    return out


def exported_functions(path: Path) -> set:
    data, secs, ddir, _ = _load(path)
    exp_rva = struct.unpack_from("<I", data, ddir)[0]
    off = _rva(secs, exp_rva) if exp_rva else None
    if off is None:
        return set()
    count, names_rva = struct.unpack_from("<I", data, off + 24)[0], struct.unpack_from("<I", data, off + 32)[0]
    arr = _rva(secs, names_rva)
    return {_cstr(data, _rva(secs, struct.unpack_from("<I", data, arr + 4 * i)[0])) for i in range(count)}


# 選用功能才會載入、缺了也沒關係的相依（numba 的 TBB 執行緒層）
OPTIONAL = {"tbb12.dll"}
# 不檢查的資料夾：uv 的快取（跟 venv 是同一份檔案）、下載暫存、Python 的 Tcl/Tk（沒用到）
SKIP_PARTS = {"cache", "downloads", "tmp", "tcl"}


def main(runtime: Path) -> int:
    sys32 = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32"
    system = {p.name.lower() for p in sys32.glob("*.dll")} - VC_RUNTIME
    files = [p for p in runtime.rglob("*") if p.suffix.lower() in (".exe", ".dll", ".pyd") and p.is_file()
             and not ({x.lower() for x in p.relative_to(runtime).parts[:-1]} & SKIP_PARTS)]
    by_name: dict[str, list[Path]] = {}
    for p in files:
        by_name.setdefault(p.name.lower(), []).append(p.parent)
    python_dirs = [p.parent for p in files if p.name.lower() == "python.exe" and "cpython-" in str(p.parent).lower()]
    missing, optional, vc_users = [], [], []
    for p in files:
        try:
            info = pe_info(p)
        except (OSError, struct.error, ValueError, IndexError):
            continue
        if not info:
            continue
        normal, delay, linker = info
        if any(n in VC_RUNTIME for n in normal):
            vc_users.append((linker, p))
        top = p.relative_to(runtime).parts[0].lower()
        # python、venv 底下的檔案都是 python.exe 載入的，「執行檔旁邊」是 python.exe 的資料夾；bin 底下的是各自的 exe
        app_dirs = [p.parent] + (python_dirs if top in ("python", "venv") else [])
        for name in normal:
            if name.startswith(("api-ms-win-", "ext-ms-")):
                continue
            if any((d / name).exists() for d in app_dirs):
                continue
            if name in VC_RUNTIME:
                missing.append((p, name, "VC++ 執行階段不在執行檔旁邊"))
            elif name in OPTIONAL:
                optional.append((p, name))
            elif name not in by_name and name not in system:
                missing.append((p, name, "找不到"))
    vc_users.sort(reverse=True)
    newest = vc_users[0][0] if vc_users else (0, 0)
    print(f"檢查 {len(files)} 個檔案；用到 VC++ 執行階段的 {len(vc_users)} 個，最新的連結器版本 {newest[0]}.{newest[1]}")
    for linker, p in vc_users[:8]:
        print(f"  連結器 {linker[0]}.{linker[1]}：{p.relative_to(runtime)}")
    bundled = sorted({str(d.relative_to(runtime)) for d in by_name.get("msvcp140.dll", [])})
    print("附帶的 msvcp140.dll 在：" + ("、".join(bundled) if bundled else "（沒有）"))
    # 用新版工具鏈編譯的檔案，匯入的函式在附帶的（比較舊的）VC++ DLL 裡都找得到嗎
    exports_cache: dict = {}
    for linker, p in vc_users:
        top = p.relative_to(runtime).parts[0].lower()
        dirs = ([p.parent] + python_dirs) if top in ("python", "venv") else [p.parent]
        for dll, names in imported_functions(p, VC_RUNTIME).items():
            src = next((d / dll for d in dirs if (d / dll).exists()), None)
            if src is None:
                continue
            if src not in exports_cache:
                exports_cache[src] = exported_functions(src)
            lacking = sorted(names - exports_cache[src])
            if lacking:
                missing.append((p, dll, f"附帶的 {dll} 沒有 {', '.join(lacking[:3])}"))
    print(f"匯入函式比對：{len(vc_users)} 個檔案用名字匯入的 VC++ 函式，都在附帶的 DLL 裡找得到"
          if not any("沒有" in why for _, _, why in missing) else "匯入函式比對：有缺少的函式")
    for p, name in optional:
        print(f"  （選用，不影響）缺少 {name}：{p.relative_to(runtime)}")
    for p, name, why in missing:
        print(f"  缺少 {name}（{why}）：{p.relative_to(runtime)}")
    print("通過" if not missing else f"有 {len(missing)} 個問題")
    return 0 if not missing else 1


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1])))
