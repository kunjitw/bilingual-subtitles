"""本機外掛：讓某些網址改用自己寫的下載程式，不走 yt-dlp。

外掛放在專案根目錄的 local_plugins 資料夾（.gitignore 擋掉，不會上傳），一個外掛一個 .py 檔。
位置寫死在這裡（PLUGIN_DIR），不能從網址、設定檔或環境變數指定別的地方。沒有這個資料夾時一切照舊。
不載入的檔案：檔名以 _ 開頭的（可以放外掛共用的程式）、捷徑（symlink、junction）、實際位置不在資料夾裡的。
資料夾本身是捷徑也整個不載入。外掛檔案有改動時，下次用到會重新載入，不用重開伺服器。

外掛檔案要有：
  NAME = "顯示名稱"                       選填，預設是檔名
  def match(url: str) -> bool             這個網址要不要由它處理（網址已經通過 safepath.check_download_url）
  def download(job) -> dict               把影片下載到 job.out_dir，回傳 {"path": 影片檔, "title": 標題}
可以有：
  OPTIONS = {"language": "ja", "translate": False}
      這種網址固定用的轉字幕選項，會蓋掉網頁送來的。只認 OPTION_KEYS 裡的欄位，其他的忽略。

download 拿到的 job（PluginJob）：
  job.url          網址
  job.out_dir      影片要放的資料夾（data/media/<影片 id>），刪影片時整個刪掉；回傳的檔案一定要在這裡面
  job.work_dir     暫存資料夾，任務結束就刪
  job.settings_dir 這個外掛的設定資料夾（見下面「設定」），可能還不存在
  job.progress(比例 0~1, 說明)
  job.check()      使用者取消了就丟例外（任務會變成已取消）
  job.run(cmd, on_line=None, cwd=None, env=None) -> int
                   跑一個子程序（不開視窗），每行輸出呼叫 on_line(字串)，回傳結束碼；
                   取消時連同它叫起來的程序一起結束，然後丟取消的例外

設定（選填，例如登入用的 cookie，在設定頁的「外掛」填）：
  SETTINGS = {"title": "標題", "intro": "一句說明", "user_agent": True,
              "fields": [{"key": "cookie", "label": "Cookie", "type": "textarea",
                          "placeholder": "...", "help": "...", "steps": ["第一步", "第二步"]}]}
      type 是 text 或 textarea。user_agent 為 True 時，存檔會一起送瀏覽器的 User-Agent。
  def save_settings(values: dict, ctx)    values 只有這次有填的欄位（字串）；ctx.dir 設定資料夾、ctx.user_agent
  def settings_status(ctx) -> dict        {"fields": {"cookie": {"set": True, "saved_at": 時間戳}},
                                           "user_agent": 存的 UA, "problem": "給使用者看的問題（選填）"}
  def clear_settings(ctx)                 選填，「清除」按鈕
  def save_user_agent(ctx)                選填，配 SETTINGS 的 "auto_user_agent": True：網頁打開時瀏覽器的 UA
                                          跟存的不一樣就自動送來（只接受同一種瀏覽器、同一種系統的新版本，
                                          或還沒存過），不用重貼 cookie；ctx.user_agent 是新的 UA
  填的內容只寫不讀：網頁、API 只看得到有沒有設定、什麼時候設定、瀏覽器型號（從 UA 算出來），log 不記內容。
  cookie_header(文字, 網域) 可以把貼上的 cookie（Cookie 標頭、擴充套件匯出的 JSON、cookies.txt）整理成一行。
  設定資料夾是 SETTINGS_DIR/<外掛檔名>（預設 private/plugin_settings，測試用 VS_PLUGIN_SETTINGS_DIR 換位置）。

展開成清單（選填，例如貼一集的網址，列出整部作品的每一集讓使用者勾）：
  def expand(url, ctx) -> dict   {"title": "作品名稱",
                                  "groups": [{"title": "第一季", "note": "說明（選填）", "checked": True,
                                              "options": {"language": "zh"}（選填，蓋過 OPTIONS）,
                                              "items": [{"url": ..., "label": "1", "title": "第 1 集的標題",
                                                         "series": {...}（選填，見下面「作品、季、集」）}]}]}
      checked：這一組預設要不要勾。items 的網址也要由這個外掛處理（match 為 True），不是的丟掉。
      可能會連網，外掛自己快取、不要打太多請求。失敗丟例外，訊息會顯示給使用者。
  def item_key(url) -> str | None       選填，同一集的不同網址寫法算同一個（判斷是不是已經在片庫裡）

作品、季、集（選填，依作品分的播放列表用，格式見 app/series.py）：
  {"series": "作品名", "season": "第一季", "season_order": 1, "episode": 2, "episode_label": "2", "version": "配音版"}
  可以放在 expand 的每一集（從清單加入時存起來），或 download 的回傳值 {"path", "title", "series"}
  （影片還沒有標記時才用）。
"""
import importlib.util
import logging
import os
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType

from . import config, safepath
from . import series as series_mod

log = logging.getLogger("plugins")

PLUGIN_DIR = config.ROOT / "local_plugins"
# 外掛設定（登入資料這類）放的地方：每個外掛一個資料夾。private 資料夾不進 git
SETTINGS_DIR = config._env_path("VS_PLUGIN_SETTINGS_DIR") or config.ROOT / "private" / "plugin_settings"
# 外掛可以固定的轉字幕選項和型別（language 另外檢查是不是支援的語言）
OPTION_KEYS = {"language": str, "translate": bool, "sensitive": bool}
FIELD_TYPES = ("text", "textarea")
MAX_VALUE_CHARS = 16000      # 一個設定欄位最多幾個字（cookie 通常幾千字）
MAX_UA_CHARS = 512
MAX_ITEMS = 2000             # 展開的清單最多幾集


class PluginError(Exception):
    """給使用者看的錯誤（設定、展開清單）。status 是 API 要回的 HTTP 狀態碼。"""

    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


@dataclass
class Plugin:
    name: str
    file: Path
    module: ModuleType
    options: dict = field(default_factory=dict)
    settings: dict | None = None      # 清理過的 SETTINGS；沒有設定欄位是 None

    @property
    def id(self) -> str:
        return self.file.stem

    @property
    def can_expand(self) -> bool:
        return callable(getattr(self.module, "expand", None))

    def match(self, url: str) -> bool:
        try:
            return bool(self.module.match(url))
        except Exception:  # noqa: BLE001
            log.warning("plugin %s: match() failed", self.name, exc_info=True)
            return False


_lock = threading.Lock()
_cache: dict = {"key": None, "plugins": []}


def _candidates() -> list[Path]:
    """可以載入的外掛檔案。資料夾不在、是捷徑，回空清單。"""
    folder = PLUGIN_DIR
    try:
        if not folder.is_dir() or safepath.is_link(folder):
            return []
        files = sorted(folder.glob("*.py"))
    except OSError:
        return []
    out = []
    for f in files:
        if f.name.startswith("_") or not f.is_file():
            continue
        if safepath.is_link(f) or not safepath.inside(f, folder):
            log.warning("plugin %s is a link or points outside %s, skipped", f.name, folder)
            continue
        out.append(f)
    return out


def _scan_key(files: list[Path]) -> tuple:
    key = []
    for f in files:
        try:
            st = f.stat()
        except OSError:
            continue
        key.append((f.name, st.st_mtime_ns, st.st_size))
    return (str(PLUGIN_DIR), tuple(key))


def _clean_options(name: str, raw) -> dict:
    if not isinstance(raw, dict):
        return {}
    out = {}
    for k, v in raw.items():
        want = OPTION_KEYS.get(k)
        if want is None or not isinstance(v, want):
            log.warning("plugin %s: ignored option %r=%r", name, k, v)
            continue
        if k == "language" and v not in config.LANGUAGES:
            log.warning("plugin %s: unsupported language %r", name, v)
            continue
        out[k] = v
    return out


def _short_text(v, limit: int = 400) -> str:
    return v.strip()[:limit] if isinstance(v, str) else ""


def _clean_settings(name: str, module: ModuleType) -> dict | None:
    """SETTINGS 宣告：只留認得的欄位。沒有 save_settings、settings_status 就當作沒有設定。"""
    raw = getattr(module, "SETTINGS", None)
    if raw is None:
        return None
    if not isinstance(raw, dict) or not callable(getattr(module, "save_settings", None)) \
            or not callable(getattr(module, "settings_status", None)):
        log.warning("plugin %s: SETTINGS needs a dict, save_settings() and settings_status()", name)
        return None
    fields, seen = [], set()
    for f in raw.get("fields") or []:
        if not isinstance(f, dict):
            continue
        key = f.get("key")
        if not isinstance(key, str) or not safepath.is_safe_name(key) or key in seen:
            log.warning("plugin %s: ignored settings field %r", name, key)
            continue
        seen.add(key)
        steps = [_short_text(s) for s in (f.get("steps") or []) if _short_text(s)][:12]
        fields.append({"key": key, "label": _short_text(f.get("label"), 60) or key,
                       "type": f.get("type") if f.get("type") in FIELD_TYPES else "text",
                       "placeholder": _short_text(f.get("placeholder"), 200), "help": _short_text(f.get("help")),
                       "steps": steps})
    if not fields:
        return None
    return {"title": _short_text(raw.get("title"), 60) or name, "intro": _short_text(raw.get("intro")),
            "user_agent": raw.get("user_agent") is True,
            "auto_user_agent": raw.get("auto_user_agent") is True and callable(getattr(module, "save_user_agent", None)),
            "fields": fields}


def _load(f: Path) -> Plugin | None:
    mod_name = f"vs_local_plugin_{f.stem}"
    try:
        spec = importlib.util.spec_from_file_location(mod_name, f)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = module
        spec.loader.exec_module(module)
    except Exception:  # noqa: BLE001
        sys.modules.pop(mod_name, None)
        log.warning("plugin %s failed to load", f.name, exc_info=True)
        return None
    if not callable(getattr(module, "match", None)) or not callable(getattr(module, "download", None)):
        log.warning("plugin %s has no match() or download(), skipped", f.name)
        return None
    if not safepath.is_safe_name(f.stem):
        log.warning("plugin file name %s is not usable (letters, digits, _ and - only), skipped", f.name)
        return None
    name = getattr(module, "NAME", None)
    name = name.strip() if isinstance(name, str) and name.strip() else f.stem
    return Plugin(name=name, file=f, module=module, options=_clean_options(name, getattr(module, "OPTIONS", None)),
                  settings=_clean_settings(name, module))


def load() -> list[Plugin]:
    """目前的外掛清單。檔案沒變就用上次載入的。"""
    files = _candidates()
    key = _scan_key(files)
    with _lock:
        if _cache["key"] != key:
            plugins = [p for p in (_load(f) for f in files) if p]
            if plugins:
                log.info("local plugins: %s", ", ".join(f"{p.name} ({p.file.name})" for p in plugins))
            _cache.update(key=key, plugins=plugins)
        return list(_cache["plugins"])


def find(url: str) -> Plugin | None:
    """處理這個網址的外掛；網址不合格（safepath.check_download_url）或沒有外掛要處理回 None。"""
    try:
        url = safepath.check_download_url(url)
    except ValueError:
        return None
    return next((p for p in load() if p.match(url)), None)


def by_id(pid: str) -> Plugin | None:
    return next((p for p in load() if p.id == pid), None) if safepath.is_safe_name(pid) else None


# ---------- 設定（只寫不讀） ----------

@dataclass
class SettingsContext:
    dir: Path                        # 這個外掛的設定資料夾（存檔時一定已經建好）
    user_agent: str | None = None    # 存檔時瀏覽器送來的 User-Agent（外掛有要才有）


def settings_dir(plugin: Plugin) -> Path:
    return SETTINGS_DIR / plugin.id


def _settings_ctx(plugin: Plugin, create: bool = False, user_agent: str | None = None) -> SettingsContext:
    folder = settings_dir(plugin)
    if safepath.is_link(SETTINGS_DIR) or safepath.is_link(folder):
        raise PluginError("外掛設定資料夾是捷徑，不使用", 500)
    if create:
        folder.mkdir(parents=True, exist_ok=True)
    return SettingsContext(dir=folder, user_agent=user_agent)


def _browser_parts(ua: str) -> tuple[str | None, str | None, str | None]:
    """(瀏覽器, 主版本, 系統)"""
    browser = version = None
    for pattern, label in ((r"Edg(?:e|A|iOS)?/(\d+)", "Edge"), (r"OPR/(\d+)", "Opera"), (r"Vivaldi/(\d+)", "Vivaldi"),
                           (r"Firefox/(\d+)", "Firefox"), (r"FxiOS/(\d+)", "Firefox"), (r"CriOS/(\d+)", "Chrome"),
                           (r"Chrome/(\d+)", "Chrome"), (r"Version/(\d+)[.\d]* .*Safari/", "Safari")):
        m = re.search(pattern, ua)
        if m:
            browser, version = label, m.group(1)
            break
    system = next((label for key, label in (("Windows", "Windows"), ("Android", "Android"), ("iPhone", "iPhone"),
                                             ("iPad", "iPad"), ("Mac OS X", "Mac"), ("CrOS", "ChromeOS"),
                                             ("Linux", "Linux")) if key in ua), None)
    return browser, version, system


def browser_label(ua: str | None) -> str | None:
    """User-Agent → 「Chrome 140（Windows）」這種給人看的型號，不回傳原本的字串。"""
    if not isinstance(ua, str) or not ua.strip():
        return None
    browser, version, system = _browser_parts(ua)
    browser = f"{browser} {version}" if browser else None
    if browser and system:
        return f"{browser}（{system}）"
    return browser or system or "其他瀏覽器"


def ua_fingerprint(ua: str | None) -> str | None:
    """UA 的短雜湊（FNV-1a 32 位元，UTF-8）。網頁用同樣的算法比對「存的 UA 跟現在的一不一樣」，不用拿到 UA 本身。"""
    if not isinstance(ua, str) or not ua:
        return None
    h = 0x811C9DC5
    for b in ua.encode("utf-8"):
        h = ((h ^ b) * 0x01000193) & 0xFFFFFFFF
    return f"{h:08x}"


def _same_browser_kind(a: str, b: str) -> bool:
    """同一種瀏覽器、同一種系統（版本可以不同）。"""
    pa, pb = _browser_parts(a), _browser_parts(b)
    return pa[0] is not None and pa[0] == pb[0] and pa[2] == pb[2]


def cookie_header(text, domains=(), drop=()) -> str:
    """貼上的 cookie 整理成 Cookie 標頭的一行「a=1; b=2」。看得懂三種：
      Cookie 標頭（可以帶「cookie:」開頭、換行）、
      擴充套件匯出的 JSON（[{"domain", "name", "value"}, ...]，或 {"cookies": [...]}），
      cookies.txt（Netscape 格式，一行 7 欄用 tab 分開）。
    domains：只留這些網域（和子網域）的 cookie（JSON、cookies.txt 才有網域）。drop：不要的名稱。
    同一個名稱出現兩次用後面的。看不懂回空字串。"""
    import json
    if not isinstance(text, str):
        return ""
    raw = text.strip()
    domains = tuple(d.lower().lstrip(".") for d in domains)

    def domain_ok(d) -> bool:
        if not domains:
            return True
        d = (d or "").lower().lstrip(".") if isinstance(d, str) else ""
        return bool(d) and any(d == x or d.endswith("." + x) for x in domains)

    pairs: list[tuple[str, str]] = []
    data = None
    if raw[:1] in "[{":
        try:
            data = json.loads(raw)
        except ValueError:
            data = None
    if isinstance(data, dict):
        data = data.get("cookies")
    if isinstance(data, list):
        for c in data:
            if isinstance(c, dict) and isinstance(c.get("name"), str) and isinstance(c.get("value"), (str, int, float)) \
                    and domain_ok(c.get("domain")):
                pairs.append((c["name"], str(c["value"])))
    elif "\t" in raw:
        for line in raw.splitlines():
            line = line.strip()
            if line.startswith("#HttpOnly_"):
                line = line[len("#HttpOnly_"):]
            elif not line or line.startswith("#"):
                continue
            cols = line.split("\t")
            if len(cols) >= 7 and domain_ok(cols[0]):
                pairs.append((cols[5], cols[6]))
    else:
        line = " ".join(raw.replace("\r", "\n").split("\n")).strip()
        line = re.sub(r"^\s*cookie\s*:\s*", "", line, flags=re.I)
        for part in line.split(";"):
            name, sep, value = part.strip().partition("=")
            if sep:
                pairs.append((name, value))
    out: dict[str, str] = {}
    skip = set(drop)
    for name, value in pairs:
        name, value = name.strip(), value.strip()
        if not name or name in skip or re.search(r"[\s;=,\"]", name) or re.search(r"[;\r\n]", value):
            continue
        out.pop(name, None)
        out[name] = value
    return "; ".join(f"{k}={v}" for k, v in out.items())


def settings_view(plugin: Plugin) -> dict | None:
    """設定頁顯示的狀態：欄位說明、有沒有設定、什麼時候設定、瀏覽器型號。不含填的內容。"""
    s = plugin.settings
    if not s:
        return None
    status, problem = {}, None
    try:
        raw = plugin.module.settings_status(_settings_ctx(plugin))
        status = raw if isinstance(raw, dict) else {}
    except Exception as e:  # noqa: BLE001
        # 例外的訊息可能帶到設定內容，只記種類
        log.warning("plugin %s: settings_status() failed (%s)", plugin.name, type(e).__name__)
        problem = "讀不到設定狀態"
    fields_status = status.get("fields") if isinstance(status.get("fields"), dict) else {}
    fields = []
    for f in s["fields"]:
        st = fields_status.get(f["key"]) if isinstance(fields_status.get(f["key"]), dict) else {}
        saved = st.get("saved_at")
        fields.append({**f, "set": st.get("set") is True,
                       "saved_at": float(saved) if isinstance(saved, (int, float)) and saved > 0 else None})
    if problem is None and isinstance(status.get("problem"), str) and status["problem"].strip():
        problem = status["problem"].strip()[:300]
    ua = status.get("user_agent") if isinstance(status.get("user_agent"), str) else None
    return {"id": plugin.id, "name": plugin.name, "title": s["title"], "intro": s["intro"],
            "user_agent": s["user_agent"], "auto_user_agent": s["auto_user_agent"],
            "ua_fp": ua_fingerprint(ua) if s["auto_user_agent"] else None, "fields": fields,
            "browser": browser_label(ua) if any(f["set"] for f in fields) else None,
            "problem": problem}


def _stored_user_agent(plugin: Plugin) -> str | None:
    try:
        raw = plugin.module.settings_status(_settings_ctx(plugin))
    except Exception as e:  # noqa: BLE001
        log.warning("plugin %s: settings_status() failed (%s)", plugin.name, type(e).__name__)
        raise PluginError("讀不到設定狀態", 500)
    ua = raw.get("user_agent") if isinstance(raw, dict) else None
    return ua if isinstance(ua, str) and ua else None


def update_user_agent(plugin: Plugin, user_agent) -> dict:
    """網頁打開時自動送來的 UA（SETTINGS 的 auto_user_agent）。跟存的一樣不動；存的是別種瀏覽器或別種系統
    （例如在手機上打開）也不動，免得跟 cookie 對不上。回傳 {"updated": bool}。"""
    s = plugin.settings
    if not s or not s["auto_user_agent"]:
        raise PluginError("這個外掛不用瀏覽器資訊", 404)
    ua = user_agent.strip() if isinstance(user_agent, str) else ""
    if not ua or len(ua) > MAX_UA_CHARS or any(ord(c) < 32 for c in ua):
        raise PluginError("讀不到瀏覽器資訊")
    old = _stored_user_agent(plugin)
    if old == ua or (old and not _same_browser_kind(old, ua)):
        return {"updated": False}
    try:
        plugin.module.save_user_agent(_settings_ctx(plugin, create=True, user_agent=ua))
    except Exception as e:  # noqa: BLE001
        log.warning("plugin %s: save_user_agent() failed (%s)", plugin.name, type(e).__name__)
        raise PluginError("存檔失敗", 500)
    log.info("plugin %s: browser updated to %s", plugin.name, browser_label(ua))
    return {"updated": True}


def all_settings() -> list[dict]:
    return [v for v in (settings_view(p) for p in load()) if v]


def save_settings(plugin: Plugin, values, user_agent=None) -> dict:
    """存外掛設定。values 只收宣告過的欄位、非空字串；錯誤丟 PluginError。回傳 settings_view。"""
    s = plugin.settings
    if not s:
        raise PluginError("這個外掛沒有設定", 404)
    if not isinstance(values, dict):
        raise PluginError("格式不對")
    keys = {f["key"] for f in s["fields"]}
    clean = {}
    for k, v in values.items():
        if k not in keys or not isinstance(v, str):
            raise PluginError("格式不對")
        v = v.strip()
        if not v:
            continue
        if len(v) > MAX_VALUE_CHARS:
            raise PluginError("內容太長，請確認只複製了需要的那一段")
        if "\x00" in v:
            raise PluginError("內容有看不懂的字元，請重新複製")
        clean[k] = v
    if not clean:
        raise PluginError("請先貼上內容")
    ua = None
    if s["user_agent"]:
        ua = user_agent.strip() if isinstance(user_agent, str) else ""
        if not ua or len(ua) > MAX_UA_CHARS or any(ord(c) < 32 for c in ua):
            raise PluginError("讀不到瀏覽器資訊，請用平常的瀏覽器打開這一頁再存")
    ctx = _settings_ctx(plugin, create=True, user_agent=ua)
    try:
        plugin.module.save_settings(clean, ctx)
    except ValueError as e:
        raise PluginError(str(e) or "內容不對")
    except Exception as e:  # noqa: BLE001
        log.warning("plugin %s: save_settings() failed (%s)", plugin.name, type(e).__name__)
        raise PluginError("存檔失敗", 500)
    log.info("plugin %s: settings saved (%s)", plugin.name, ", ".join(sorted(clean)))
    return settings_view(plugin)


def clear_settings(plugin: Plugin) -> dict:
    if not plugin.settings:
        raise PluginError("這個外掛沒有設定", 404)
    fn = getattr(plugin.module, "clear_settings", None)
    if not callable(fn):
        raise PluginError("這個外掛不能清除設定")
    try:
        fn(_settings_ctx(plugin))
    except Exception as e:  # noqa: BLE001
        log.warning("plugin %s: clear_settings() failed (%s)", plugin.name, type(e).__name__)
        raise PluginError("清除失敗", 500)
    log.info("plugin %s: settings cleared", plugin.name)
    return settings_view(plugin)


# ---------- 展開成清單 ----------

def item_key(plugin: Plugin, url: str) -> str:
    """同一集的識別：外掛有 item_key 用它，沒有就用網址本身。"""
    fn = getattr(plugin.module, "item_key", None)
    if callable(fn):
        try:
            key = fn(url)
            if isinstance(key, (str, int)) and str(key):
                return f"k:{key}"
        except Exception:  # noqa: BLE001
            log.warning("plugin %s: item_key() failed", plugin.name, exc_info=True)
    return f"u:{url.strip()}"


_expand_lock = threading.Lock()


def expand(plugin: Plugin, url: str) -> dict:
    """外掛的 expand(url) 清理過的結果：{"title", "groups": [{"key", "title", "note", "checked", "options",
    "items": [{"url", "label", "title"}]}]}。每一集的網址都要通過網址檢查、而且由同一個外掛處理。"""
    if not plugin.can_expand:
        raise PluginError("這個網址不能展開成清單")
    url = safepath.check_download_url(url)
    started = time.time()
    with _expand_lock:           # 一次只展開一個，避免同時對網站打很多請求
        try:
            raw = plugin.module.expand(url, _settings_ctx(plugin))
        except PluginError:
            raise
        except Exception as e:  # noqa: BLE001
            log.warning("plugin %s: expand() failed", plugin.name, exc_info=True)
            raise PluginError(f"讀不到清單：{str(e)[:300]}", 502)
    if not isinstance(raw, dict):
        raise PluginError("讀不到清單", 502)
    groups, seen, total = [], set(), 0
    for i, g in enumerate(raw.get("groups") or []):
        if not isinstance(g, dict):
            continue
        items = []
        for it in g.get("items") or []:
            if total >= MAX_ITEMS or not isinstance(it, dict):
                break
            try:
                u = safepath.check_download_url(it.get("url"))
            except ValueError:
                continue
            if u in seen or not plugin.match(u):
                continue
            seen.add(u)
            total += 1
            items.append({"url": u, "label": _short_text(it.get("label"), 20) or str(len(items) + 1),
                          "title": _short_text(it.get("title"), 200), "series": series_mod.clean(it.get("series"))})
        if items:
            groups.append({"key": f"g{i}", "title": _short_text(g.get("title"), 80) or f"第 {len(groups) + 1} 組",
                           "note": _short_text(g.get("note"), 200), "checked": g.get("checked") is not False,
                           "options": _clean_options(plugin.name, g.get("options")), "items": items})
    if not groups:
        raise PluginError("這個網址沒有可以加入的項目", 404)
    log.info("plugin %s: expanded to %d groups, %d items in %.1fs", plugin.name, len(groups), total,
             time.time() - started)
    return {"title": _short_text(raw.get("title"), 200), "groups": groups}


# ---------- 給外掛用的任務介面 ----------

def _kill_tree(proc: subprocess.Popen):
    """結束子程序和它叫起來的程序（例如下載程式再叫的 ffmpeg）。"""
    if proc.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if proc.poll() is None:
        proc.kill()


class PluginJob:
    """外掛的 download(job) 拿到的東西。ctx 是 jobs.JobContext（check、progress、cancel_event）。"""

    def __init__(self, ctx, url: str, out_dir: Path, work_dir: Path, settings_dir: Path | None = None):
        self._ctx = ctx
        self.url = url
        self.out_dir = Path(out_dir)
        self.work_dir = Path(work_dir)
        self.settings_dir = Path(settings_dir) if settings_dir else None

    def progress(self, value: float, stage: str | None = None):
        self._ctx.progress(float(value), stage)

    def check(self):
        self._ctx.check()

    def run(self, cmd: list, on_line=None, cwd=None, env=None) -> int:
        self.check()
        proc = subprocess.Popen([str(c) for c in cmd], cwd=str(cwd) if cwd else None, env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                text=True, encoding="utf-8", errors="replace",
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        lines: list[str] = []
        done = threading.Event()

        def read():
            try:
                for line in proc.stdout:
                    lines.append(line.rstrip("\r\n"))
            finally:
                done.set()

        threading.Thread(target=read, daemon=True).start()
        try:
            while True:
                finished = done.wait(0.3)
                while lines:
                    line = lines.pop(0)
                    if on_line:
                        on_line(line)
                if self._ctx.cancel_event.is_set():
                    _kill_tree(proc)
                    self.check()
                if finished:
                    break
            return proc.wait()
        finally:
            _kill_tree(proc)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
            if proc.stdout:
                proc.stdout.close()
