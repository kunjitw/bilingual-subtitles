"""用無頭 Chrome 走一遍網頁介面：截圖並收集 JavaScript 錯誤。

用法：python -s tests/ui_check.py <輸出資料夾> [media_id] [秒數] [--ja=日文影片id] [--en=英文影片id]
預設連 127.0.0.1:8765；測試伺服器用環境變數指定，例如 VS_UI_URL=http://127.0.0.1:8766/
加了 --ja / --en 會多檢查查字（滑鼠停在字上、釘選、全螢幕、逐字稿）和單字頁。
Chrome 除錯埠預設 9333，可以用 VS_CDP_PORT 換。
Chrome 依序找環境變數 VS_CHROME、Program Files 和 %LOCALAPPDATA% 底下的 Google Chrome、PATH 上的 chrome。

首次自動安裝的畫面：python -s tests/ui_check.py <輸出資料夾> --setup [--media=影片id]
  在頁面裡攔截 /api/setup、/api/state、/api/meta、/api/dicts、/api/vocab，換成假的安裝狀態，
  桌機 1600×900 和手機 412×915（觸控）各走一遍每一種狀態並截圖，不會改到伺服器的資料。
  伺服器最好用空的 data 資料夾（新使用者的樣子）；--media 給一部有影片檔的影片時，
  多檢查「進度頁開著時影片不會自己播、按空白鍵也不會播」（VS_MEDIA_URL 可以指定那部影片所在的伺服器）。
"""
import asyncio
import base64
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import websockets


def find_chrome() -> str:
    """找 chrome.exe 的位置，找不到就結束並說明怎麼指定。"""
    env = os.environ.get("VS_CHROME", "").strip().strip('"')
    if env:
        if os.path.isfile(env):
            return env
        raise SystemExit(f"環境變數 VS_CHROME 指到的檔案不存在：{env}")
    cands = []
    for var in ("ProgramFiles", "ProgramFiles(x86)", "LOCALAPPDATA"):
        base = os.environ.get(var)
        if base:
            cands.append(os.path.join(base, "Google", "Chrome", "Application", "chrome.exe"))
    cands.append(shutil.which("chrome") or "")
    for c in cands:
        if c and os.path.isfile(c):
            return c
    raise SystemExit("找不到 Chrome。請安裝 Google Chrome，或把環境變數 VS_CHROME 設成 chrome.exe 的完整路徑。")


PORT = int(os.environ.get("VS_CDP_PORT", "9333"))
APP = os.environ.get("VS_UI_URL", "http://127.0.0.1:8765/")


class CDP:
    def __init__(self, ws):
        self.ws = ws
        self.n = 0
        self.pending = {}
        self.errors = []

    async def listen(self):
        async for raw in self.ws:
            msg = json.loads(raw)
            if "id" in msg and msg["id"] in self.pending:
                self.pending.pop(msg["id"]).set_result(msg)
            elif msg.get("method") == "Runtime.exceptionThrown":
                d = msg["params"]["exceptionDetails"]
                self.errors.append(d.get("exception", {}).get("description") or d.get("text"))
            elif msg.get("method") == "Runtime.consoleAPICalled" and msg["params"]["type"] == "error":
                self.errors.append(" ".join(str(a.get("value", a.get("description"))) for a in msg["params"]["args"]))
            elif msg.get("method") == "Log.entryAdded" and msg["params"]["entry"]["level"] == "error":
                e = msg["params"]["entry"]
                if "favicon" not in e.get("url", ""):
                    self.errors.append(f"{e['text']} {e.get('url', '')}")

    async def send(self, method, **params):
        self.n += 1
        fut = asyncio.get_event_loop().create_future()
        self.pending[self.n] = fut
        await self.ws.send(json.dumps({"id": self.n, "method": method, "params": params}))
        res = await asyncio.wait_for(fut, 30)
        if "error" in res:
            raise RuntimeError(res["error"])
        return res.get("result", {})

    async def js(self, expr):
        res = await self.send("Runtime.evaluate", expression=expr, awaitPromise=True, returnByValue=True)
        if res.get("exceptionDetails"):
            self.errors.append("evaluate: " + json.dumps(res["exceptionDetails"], ensure_ascii=False)[:400])
        return res.get("result", {}).get("value")

    async def shot(self, path: Path):
        res = await self.send("Page.captureScreenshot", format="png")
        path.write_bytes(base64.b64decode(res["data"]))


async def run(out_dir: Path, media_id: str, at: float):
    with urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{PORT}/json/new?about:blank", method="PUT")) as r:
        target = json.load(r)
    async with websockets.connect(target["webSocketDebuggerUrl"], max_size=50_000_000) as ws:
        cdp = CDP(ws)
        listener = asyncio.create_task(cdp.listen())
        await cdp.send("Runtime.enable")
        await cdp.send("Log.enable")
        await cdp.send("Page.enable")
        await cdp.send("Emulation.setDeviceMetricsOverride", width=1600, height=900, deviceScaleFactor=1, mobile=False)
        await cdp.send("Page.navigate", url=f"{APP}#m={media_id}" if media_id else APP)
        await asyncio.sleep(4)

        report = {}
        if media_id:
            await cdp.js(f"""new Promise(res => {{
                const v = document.querySelector('#video');
                const go = () => {{ v.currentTime = {at}; v.addEventListener('seeked', () => setTimeout(res, 800), {{once: true}}); }};
                v.readyState >= 1 ? go() : v.addEventListener('loadedmetadata', go, {{once: true}});
                setTimeout(res, 8000);
            }})""")
            await cdp.js("showControls()")
            await asyncio.sleep(0.5)
            report["line1"] = await cdp.js("document.querySelector('#line1 span').textContent")
            report["line2"] = await cdp.js("document.querySelector('#line2 span').textContent")
            report["transcript_items"] = await cdp.js("document.querySelectorAll('#transcript li[data-i]').length")
            report["transcript_active"] = await cdp.js("document.querySelector('#transcript li.on')?.dataset.i ?? null")
            report["video_error"] = await cdp.js("document.querySelector('#video').error?.code ?? null")
            report["sel1"] = await cdp.js("document.querySelector('#sel-1').selectedOptions[0]?.textContent")
            report["sel2"] = await cdp.js("document.querySelector('#sel-2').selectedOptions[0]?.textContent")
            report["ruby_in_line"] = await cdp.js("document.querySelectorAll('#line1 ruby, #line2 ruby').length")
            report["ruby_in_transcript"] = await cdp.js("document.querySelectorAll('#transcript ruby').length")
            report["play_icon"] = await cdp.js("document.querySelector('#c-play use').getAttribute('href')")
            report["title_sub"] = await cdp.js("document.querySelector('#media-title-sub').textContent")
            await cdp.shot(out_dir / "1_player.png")

            await cdp.js("nextCue(); nextCue();")
            await asyncio.sleep(0.8)
            report["after_next_line1"] = await cdp.js("document.querySelector('#line1 span').textContent")
            await cdp.js("S.settings.style='box'; S.settings.blur2=true; applySubSettings();")
            await asyncio.sleep(0.3)
            await cdp.shot(out_dir / "2_box_blur.png")
            await cdp.js("S.settings.style='outline'; S.settings.blur2=false; applySubSettings();")

            await cdp.js("document.querySelector('#btn-files').click()")
            await asyncio.sleep(0.5)
            report["tracks_dialog_items"] = await cdp.js("document.querySelectorAll('#track-table .track-item').length")
            # 按一條還沒顯示的字幕的「顯示」：原文會換成它，或換成這條翻譯的來源
            report["slot_toggle"] = await cdp.js("""(() => {
                const btns = [...document.querySelectorAll('#track-table [data-show]')];
                const target = btns.find(b => !b.classList.contains('on'));
                if (!target) return 'only one track';
                target.click();
                const id = target.dataset.show;
                return S.trackSel[1] === id || S.trackSel[2] === id ? 'ok' : 'fail';
            })()""")
            await asyncio.sleep(0.8)
            await cdp.shot(out_dir / "3_tracks_dialog.png")
            await cdp.js("document.querySelector('#dlg-tracks').close()")

            # 字幕放到影片畫面外
            await cdp.js("setSetting('place2', 'outBottom')")
            await asyncio.sleep(0.5)
            report["outside_zone_text"] = await cdp.js("document.querySelector('#zone-outBottom').textContent.trim()")
            report["inside_zone_text"] = await cdp.js("document.querySelector('#zone-inBottom').textContent.trim()")
            await cdp.shot(out_dir / "7_outside.png")
            await cdp.js("setSetting('place1', 'outTop'); setSetting('place2', 'inBottom')")
            await asyncio.sleep(0.4)
            report["swapped_top"] = await cdp.js("document.querySelector('#zone-outTop').textContent.trim()")
            await cdp.shot(out_dir / "8_split.png")
            await cdp.js("setSetting('place1', 'inBottom'); setSetting('place2', 'inBottom')")

            # 字幕外觀面板
            await cdp.js("document.querySelector('#btn-look').click()")
            await asyncio.sleep(0.4)
            await cdp.shot(out_dir / "9_look_panel.png")
            await cdp.js("closeMenu()")

            # 已有字幕時再按轉字幕，要先跳確認
            await cdp.js("document.querySelector('#btn-transcribe').click()")
            await asyncio.sleep(0.6)
            report["confirm_open"] = await cdp.js("!!document.querySelector('#dlg-confirm[open]')")
            report["confirm_title"] = await cdp.js("document.querySelector('#confirm-title').textContent")
            await cdp.shot(out_dir / "10_confirm.png")
            await cdp.js("document.querySelector('#dlg-confirm').close()")
            await asyncio.sleep(0.3)
            report["add_dialog_blocked"] = await cdp.js("!document.querySelector('#dlg-add[open]')")

        # 設定頁
        await cdp.js("openSettings()")
        await asyncio.sleep(1.5)
        report["settings_models"] = await cdp.js("document.querySelectorAll('#model-list .model-row').length")
        report["settings_combos"] = await cdp.js("document.querySelectorAll('#combo-table .combo-row').length - 1")
        await cdp.shot(out_dir / "11_settings.png")
        await cdp.js("closeSettings()")

        await cdp.js("showTab('queue')")
        await asyncio.sleep(0.4)
        await cdp.shot(out_dir / "4_queue.png")
        await cdp.js("showTab('transcript')")

        await cdp.js("openAddDialog('add')")
        await asyncio.sleep(1.2)
        await cdp.js("document.querySelector('#lang-seg [data-v=ja]').click()")
        await asyncio.sleep(0.3)
        report["add_engine"] = await cdp.js("document.querySelector('#adv-engine').selectedOptions[0]?.textContent")
        report["add_translator"] = await cdp.js("document.querySelector('#adv-translator').selectedOptions[0]?.textContent")
        report["add_model_field_visible"] = await cdp.js("!document.querySelector('#model-field').hidden")
        await cdp.shot(out_dir / "5_add_dialog.png")
        await cdp.js("document.querySelector('#dlg-add').close()")

        if media_id:
            # 點逐字稿裡的假名，應該跳出修改讀音的彈窗，而且不會跳句
            # （查字開著時點詞是查字，改讀音在解釋卡上；這裡先關掉查字，確認原本的點假名還能用）
            await cdp.js("showTab('transcript'); S.settings.lookupOn = false; renderTranscript(); updateSubs(true)")
            await asyncio.sleep(0.3)
            report["ruby_pop"] = await cdp.js("""new Promise(res => {
                const r = document.querySelector('#transcript li.on ruby') || document.querySelector('#transcript ruby');
                if (!r) return res('no ruby');
                r.scrollIntoView({block: 'center'});
                const before = document.querySelector('#video').currentTime;
                r.click();
                setTimeout(() => res({
                    open: !document.querySelector('#ruby-pop').hidden,
                    word: document.querySelector('#rp-word').textContent,
                    cands: document.querySelector('#rp-cands').textContent,
                    seeked: Math.abs(document.querySelector('#video').currentTime - before) > 0.5,
                }), 1200);
            })""")
            await cdp.shot(out_dir / "13_ruby_pop.png")
            await cdp.js("closeRubyPop(); S.settings.lookupOn = true; saveSettings(); renderTranscript(); updateSubs(true)")

        if media_id:
            # 目前原文已經有翻譯時會先跳重新翻譯的確認框，Promise 要等人按才結束：不等它，按確定再截圖（同 pair_check）
            await cdp.js("openTranslateDialog(); 0")
            await asyncio.sleep(0.4)
            if await cdp.js("document.querySelector('#dlg-confirm').open"):
                await cdp.js("document.querySelector('#confirm-ok').click()")
                await asyncio.sleep(0.4)
            await cdp.shot(out_dir / "6_translate_dialog.png")
            await cdp.js("document.querySelector('#dlg-translate').close()")

        if LOOKUP.get("ja") or LOOKUP.get("en"):
            await lookup_checks(cdp, out_dir, report)

        # 收起左右側欄 → 重新整理 → 應該維持收起
        await cdp.js("showTab('queue'); togglePanel('left', true); togglePanel('right', true)")
        await asyncio.sleep(0.6)
        await cdp.shot(out_dir / "12_panels_hidden.png")
        await cdp.send("Page.reload")
        await asyncio.sleep(4)
        report["panels_persist"] = await cdp.js(
            "document.body.classList.contains('hide-left') && document.body.classList.contains('hide-right')")
        report["tab_persist"] = await cdp.js("S.settings.sideTab")
        await cdp.js("togglePanel('left', false); togglePanel('right', false); showTab('transcript')")

        report["errors"] = cdp.errors
        listener.cancel()
        return report


LOOKUP: dict = {}


async def mouse(cdp: CDP, kind: str, x: float, y: float, shift: bool = False):
    mods = 8 if shift else 0
    if kind == "move":
        await cdp.send("Input.dispatchMouseEvent", type="mouseMoved", x=x, y=y, modifiers=mods)
    else:
        for t in ("mousePressed", "mouseReleased"):
            await cdp.send("Input.dispatchMouseEvent", type=t, x=x, y=y, button="left", clickCount=1, modifiers=mods)


async def drag(cdp: CDP, x: float, y: float, dx: float, dy: float):
    """按住左鍵從 (x, y) 拖 (dx, dy)，分幾步移動，讓 pointermove 有機會觸發。"""
    await cdp.send("Input.dispatchMouseEvent", type="mouseMoved", x=x, y=y)
    await cdp.send("Input.dispatchMouseEvent", type="mousePressed", x=x, y=y, button="left", clickCount=1)
    for n in range(1, 6):
        await cdp.send("Input.dispatchMouseEvent", type="mouseMoved", x=x + dx * n / 5, y=y + dy * n / 5,
                       button="left", buttons=1)
    await cdp.send("Input.dispatchMouseEvent", type="mouseReleased", x=x + dx, y=y + dy, button="left", clickCount=1)


async def key(cdp: CDP, name: str, code: str, vk: int, text: str = ""):
    await cdp.send("Input.dispatchKeyEvent", type="keyDown", key=name, code=code, windowsVirtualKeyCode=vk, text=text)
    await cdp.send("Input.dispatchKeyEvent", type="keyUp", key=name, code=code, windowsVirtualKeyCode=vk)


async def wait_js(cdp: CDP, expr: str, timeout: float = 10.0):
    end = time.time() + timeout
    while time.time() < end:
        if await cdp.js(expr):
            return True
        await asyncio.sleep(0.2)
    return False


# 找一句有可查的詞（日文要有假名）的字幕，跳過去，回傳那句的索引
SEEK_WORD_CUE = """(async (needRuby, needPhrase) => {
    const cues = S.cues[1];
    let i = cues.findIndex((c, n) => n > 3 && c.w && c.w.length > 2 && (!needRuby || (c.ruby && c.ruby.length))
                                     && (!needPhrase || (c.ph && c.ph.length)));
    if (i < 0) i = cues.findIndex((c) => c.w && c.w.length);
    if (i < 0) return -1;
    video.pause();
    video.currentTime = cues[i].start + 0.15;
    await new Promise((r) => { video.addEventListener('seeked', r, {once: true}); setTimeout(r, 3000); });
    updateSubs(true);
    return i;
})"""

CENTER = """((sel, n) => {
    const el = document.querySelectorAll(sel)[n || 0];
    if (!el) return null;
    el.scrollIntoView({block: 'center'});
    const r = el.getBoundingClientRect();
    return {x: r.left + r.width / 2, y: r.top + r.height / 2, text: el.textContent};
})"""

CARD = """(() => {
    const p = document.querySelector('#wpop');
    const r = p.getBoundingClientRect();
    const c = document.querySelector('#controls').getBoundingClientRect();
    return {open: !p.hidden, pinned: p.classList.contains('pinned'), inPanel: p.classList.contains('in-panel'),
            inScreen: document.querySelector('#screen').contains(p), text: p.innerText.slice(0, 400),
            top: Math.round(r.top), bottom: Math.round(r.bottom), controlsTop: Math.round(c.top),
            fetches: (window.__vsFetch || []).slice()};
})()"""


async def lookup_checks(cdp: CDP, out_dir: Path, report: dict):
    """查字與單字頁。"""
    rep = report.setdefault("lookup", {})
    # 記下查字相關的 API 請求，確認滑鼠移動時不打 API
    await cdp.js("""(() => { if (window.__vsFetchWrapped) return; window.__vsFetchWrapped = true; window.__vsFetch = [];
        const f = window.fetch; window.fetch = (u, o) => { if (!String(u).includes('/api/state')) window.__vsFetch.push(String(u)); return f(u, o); }; })()""")
    await cdp.js("showView('player'); showTab('transcript'); togglePanel('left', false); togglePanel('right', false)")

    for lang in ("ja", "en"):
        mid = LOOKUP.get(lang)
        if not mid:
            continue
        r = rep.setdefault(lang, {})
        # 前面的檢查可能換過字幕軌：清掉記住的選擇，重新打開，第一行會是最新的辨識字幕
        await cdp.js(f"closeMedia(); store.set('tracks:{mid}', null); openMedia('{mid}')")
        loaded = await wait_js(cdp, f"S.current === '{mid}' && S.cues[1].some((c) => c.w) && !!WL._v.vocab[S.trackSel[1]]", 20)
        r["vocab_loaded"] = loaded
        i = await cdp.js(f"{SEEK_WORD_CUE}({'true' if lang == 'ja' else 'false'}, false)")
        await asyncio.sleep(0.6)
        r["cue"] = i
        r["line1"] = await cdp.js("document.querySelector('#line1 span').textContent")
        r["words_in_line"] = await cdp.js("document.querySelectorAll('#line1 .w').length")
        r["ruby_inside_words"] = await cdp.js("document.querySelectorAll('#line1 .w ruby').length")
        # 巢狀 span 不能多出底色或 padding（box 樣式）
        await cdp.js("setSetting('style', 'box')")
        await asyncio.sleep(0.2)
        r["box_nested_style"] = await cdp.js("""(() => { const w = document.querySelector('#line1 .w');
            if (!w) return null; const cs = getComputedStyle(w);
            return {padding: cs.paddingLeft, background: cs.backgroundColor, filter: cs.filter}; })()""")
        await cdp.js("setSetting('style', 'outline')")
        await asyncio.sleep(0.2)

        # 滑鼠停在字上
        await cdp.js("window.__vsFetch.length = 0")
        pos = await cdp.js(f"{CENTER}('#line1 .w', 1)") or await cdp.js(f"{CENTER}('#line1 .w', 0)")
        if not pos:
            r["error"] = "字幕上沒有可以查的詞"
            continue
        await mouse(cdp, "move", pos["x"] - 3, pos["y"])
        await mouse(cdp, "move", pos["x"], pos["y"])
        await asyncio.sleep(0.5)
        card = await cdp.js(CARD)
        r["hover_word"] = pos["text"]
        r["hover_card"] = card
        r["hover_api_calls"] = [u for u in card["fetches"] if "/api/" in u]
        r["hover_status_btns"] = await cdp.js("[...document.querySelectorAll('#wpop [data-st]')].map((b) => b.textContent)")
        await cdp.shot(out_dir / f"20_hover_{lang}.png")

        # 點一下釘選
        await mouse(cdp, "click", pos["x"], pos["y"])
        await wait_js(cdp, "!!WL._look.ctx && !!WL._look.ctx.detail", 6)
        await asyncio.sleep(0.3)
        card = await cdp.js(CARD)
        r["pinned_card"] = card
        r["pinned_paused"] = await cdp.js("video.paused")
        r["card_above_controls"] = card["bottom"] <= card["controlsTop"]
        await cdp.shot(out_dir / f"21_pinned_{lang}.png")

        # 快捷鍵 X 標已會、再按一次取消；N 標不會
        k = await cdp.js("WL._look.ctx && (WL._look.ctx.ph ? WL._look.ctx.ph[0] : WL._look.ctx.key)")
        await key(cdp, "x", "KeyX", 88, "x")
        await asyncio.sleep(0.5)
        r["status_after_x"] = await cdp.js(f"WL._v.status.get({json.dumps(k)})")
        await key(cdp, "x", "KeyX", 88, "x")
        await asyncio.sleep(0.5)
        r["status_after_x2"] = await cdp.js(f"WL._v.status.get({json.dumps(k)})")
        await key(cdp, "n", "KeyN", 78, "n")
        await asyncio.sleep(0.5)
        r["status_after_n"] = await cdp.js(f"WL._v.status.get({json.dumps(k)})")
        await key(cdp, "n", "KeyN", 78, "n")
        await asyncio.sleep(0.5)
        r["status_after_n2"] = await cdp.js(f"WL._v.status.get({json.dumps(k)})")

        # 釘選卡的例句也能開小視窗（播放器頁）
        if await cdp.js("!!document.querySelector('#wpop [data-mini]')"):
            await cdp.js("document.querySelector('#wpop [data-mini]').click()")
            await wait_js(cdp, "!document.querySelector('#mini').hidden && document.querySelector('#mini-video').readyState >= 1", 20)
            await asyncio.sleep(1.2)
            r["mini_from_card"] = await cdp.js("({shown: !document.querySelector('#mini').hidden,"
                                              " host: document.querySelector('#mini').parentElement.tagName,"
                                              " marked: document.querySelectorAll('#mini-sub mark').length,"
                                              " mainPaused: video.paused, playing: !document.querySelector('#mini-video').paused})")
            await cdp.shot(out_dir / f"21b_mini_{lang}.png")
            await cdp.js("WL.closeMini()")
            await asyncio.sleep(0.3)
        else:
            r["mini_from_card"] = "這個詞沒有其他例句"

        if lang == "ja":
            # 解釋卡上的「修改讀音」要能打開原本的改讀音彈窗
            has_btn = await cdp.js("!!document.querySelector('#wpop [data-ruby]')")
            if not has_btn:
                pos2 = await cdp.js(f"{CENTER}('#line1 .w:has(ruby)', 0)")
                if pos2:
                    await key(cdp, "Escape", "Escape", 27)
                    await mouse(cdp, "click", pos2["x"], pos2["y"])
                    await wait_js(cdp, "!!document.querySelector('#wpop [data-ruby]')", 4)
            if await cdp.js("!!document.querySelector('#wpop [data-ruby]')"):
                await cdp.js("document.querySelector('#wpop [data-ruby]').click()")
                await asyncio.sleep(1.0)
                r["ruby_pop_from_card"] = await cdp.js("({open: !document.querySelector('#ruby-pop').hidden, word: document.querySelector('#rp-word').textContent})")
                await cdp.shot(out_dir / "22_ruby_from_card.png")
                await cdp.js("closeRubyPop()")
            else:
                r["ruby_pop_from_card"] = "這句沒有假名"
        await key(cdp, "Escape", "Escape", 27)
        await asyncio.sleep(0.3)
        r["closed_by_esc"] = await cdp.js("document.querySelector('#wpop').hidden")

        if lang == "ja":
            # 按住 Shift 移到不屬於任何詞的字（助詞）上：即時查詢
            await mouse(cdp, "move", 5, 5)
            await asyncio.sleep(0.5)
            spot = await cdp.js("""(() => {
                const host = document.querySelector('#line1 span');
                const walker = document.createTreeWalker(host, NodeFilter.SHOW_TEXT);
                for (let t = walker.nextNode(); t; t = walker.nextNode()) {
                    if (t.parentElement.closest('.w, rt') || !t.textContent.trim()) continue;
                    const i = t.textContent.search(/[ぁ-ゖ]/);
                    if (i < 0) continue;
                    const range = document.createRange(); range.setStart(t, i); range.setEnd(t, i + 1);
                    const r = range.getBoundingClientRect();
                    return {x: r.left + r.width / 2, y: r.top + r.height / 2, ch: t.textContent[i]};
                }
                return null; })()""")
            if spot:
                await mouse(cdp, "move", spot["x"] - 1, spot["y"], shift=True)
                await mouse(cdp, "move", spot["x"], spot["y"], shift=True)
                await asyncio.sleep(0.9)
                r["shift_scan"] = await cdp.js("""({char: %s, open: !document.querySelector('#wpop').hidden,
                    scan: !!(WL._look.ctx && WL._look.ctx.scan), text: document.querySelector('#wpop').innerText.slice(0, 160)})""" % json.dumps(spot["ch"]))
                await cdp.shot(out_dir / "22b_shift_scan_ja.png")
                await mouse(cdp, "move", 5, 5)
                await asyncio.sleep(0.6)
            else:
                r["shift_scan"] = "這句沒有落在詞外面的假名"

        if lang == "en":
            # 片語：移到片語的其中一個字，另一個字也要一起反白
            i2 = await cdp.js(f"{SEEK_WORD_CUE}(false, true)")
            await asyncio.sleep(0.6)
            has_ph = await cdp.js("document.querySelectorAll('#line1 .w[data-ph]').length")
            if has_ph:
                p3 = await cdp.js(f"{CENTER}('#line1 .w[data-ph]', 0)")
                await mouse(cdp, "move", p3["x"] - 2, p3["y"])
                await mouse(cdp, "move", p3["x"], p3["y"])
                await asyncio.sleep(0.5)
                r["phrase"] = await cdp.js("""({cue: S.idx[1], line: document.querySelector('#line1 span').textContent,
                    hl: [...document.querySelectorAll('#line1 .w.hl')].map((x) => x.textContent),
                    card: document.querySelector('#wpop').innerText.slice(0, 200)})""")
                await cdp.shot(out_dir / "23_phrase_en.png")
                await mouse(cdp, "move", 5, 5)
                await asyncio.sleep(0.5)
            else:
                r["phrase"] = f"找不到有片語的句子（{i2}）"

        # 全螢幕：卡片要在全螢幕的元素裡面
        await cdp.send("Runtime.evaluate", expression="document.querySelector('#screen').requestFullscreen()",
                       userGesture=True, awaitPromise=True)
        await asyncio.sleep(1.0)
        await cdp.js("updateSubs(true)")
        await asyncio.sleep(0.3)
        pos = await cdp.js(f"{CENTER}('#line1 .w', 0)")
        if pos:
            await mouse(cdp, "move", pos["x"] - 3, pos["y"])
            await mouse(cdp, "move", pos["x"], pos["y"])
            await asyncio.sleep(0.6)
            r["fullscreen"] = await cdp.js("""({fs: !!document.fullscreenElement, open: !document.querySelector('#wpop').hidden,
                inside: !!document.fullscreenElement && document.fullscreenElement.contains(document.querySelector('#wpop'))})""")
            await cdp.shot(out_dir / f"24_fullscreen_{lang}.png")
            # 全螢幕時開小視窗：要放進全螢幕的元素裡才看得到
            await cdp.js("(() => { const c = S.cues[1][Math.max(0, S.idx[1])] || S.cues[1][0];"
                         " WL.openMini({media: S.current, start: c.start, end: c.end, text: c.text, hl: [], title: 'fs', lang: ''}); })()")
            await asyncio.sleep(1.0)
            r["mini_fullscreen"] = await cdp.js("({inside: !!document.fullscreenElement && document.fullscreenElement.contains(document.querySelector('#mini')),"
                                                " shown: !document.querySelector('#mini').hidden})")
            await cdp.js("WL.closeMini()")
        await cdp.js("document.exitFullscreen().catch(() => {})")
        await mouse(cdp, "move", 5, 5)
        await asyncio.sleep(0.8)

        # 逐字稿原文行也能查；點字不會跳句
        pos = await cdp.js(f"{CENTER}('#transcript li.on .a .w', 0)") or await cdp.js(f"{CENTER}('#transcript .a .w', 3)")
        if pos:
            await asyncio.sleep(0.3)
            pos = await cdp.js(f"{CENTER}('#transcript li.on .a .w', 0)") or pos
            before = await cdp.js("video.currentTime")
            await mouse(cdp, "move", pos["x"] - 3, pos["y"])
            await mouse(cdp, "move", pos["x"], pos["y"])
            await asyncio.sleep(0.6)
            r["transcript_hover"] = await cdp.js(CARD)
            await cdp.shot(out_dir / f"25_transcript_{lang}.png")
            await mouse(cdp, "click", pos["x"], pos["y"])
            await asyncio.sleep(0.6)
            r["transcript_click_seeked"] = abs((await cdp.js("video.currentTime")) - before) > 0.3
            r["transcript_pinned"] = await cdp.js("document.querySelector('#wpop').classList.contains('pinned')")
            r["translation_line_words"] = await cdp.js("document.querySelectorAll('#transcript .b .w').length")
            await key(cdp, "Escape", "Escape", 27)
            await mouse(cdp, "move", 5, 5)
            await asyncio.sleep(0.4)

    # 單字頁
    v = rep.setdefault("vocab_page", {})
    await cdp.js("S.settings.vocabLang = 'ja'; S.settings.vocabScope = ''; S.settings.vocabStatus = 'learn';"
                 " S.settings.vocabPageSize = 100; saveSettings(); showView('vocab')")
    v["ja_loaded"] = await wait_js(cdp, "document.querySelectorAll('#vb-list .vb-row').length > 0", 20)
    await asyncio.sleep(0.5)
    v["ja_rows"] = await cdp.js("document.querySelectorAll('#vb-list .vb-row').length")
    v["ja_summary"] = await cdp.js("document.querySelector('#vb-summary').textContent")
    v["ja_note"] = await cdp.js("document.querySelector('#vb-note').hidden ? '' : document.querySelector('#vb-note').innerText")
    v["ja_first_rows"] = await cdp.js("[...document.querySelectorAll('#vb-list .vb-row')].slice(0, 5).map((r) => r.innerText.replace(/\\n/g, ' | '))")
    v["status_counts"] = await cdp.js("[...document.querySelectorAll('#vb-status button')].map((b) => b.innerText.replace(/\\n/g, ' '))")
    await cdp.shot(out_dir / "26_vocab_ja.png")

    # 分頁：翻到第二頁、換每頁筆數、輸入頁碼
    first_key = await cdp.js("document.querySelector('#vb-list .vb-row').dataset.k")
    await cdp.js("document.querySelector('#vb-next').click()")
    await asyncio.sleep(0.4)
    v["page2"] = await cdp.js("({page: WL._vb.page, summary: document.querySelector('#vb-summary').textContent,"
                              " rows: document.querySelectorAll('#vb-list .vb-row').length,"
                              f" changed: document.querySelector('#vb-list .vb-row').dataset.k !== {json.dumps(first_key)},"
                              " pages: document.querySelector('#vb-pages').innerText.replace(/\\n/g, ' ')})")
    await cdp.js("(() => { const s = document.querySelector('#vb-size'); s.value = '50'; s.dispatchEvent(new Event('change')); })()")
    await asyncio.sleep(0.4)
    v["page_size_50"] = await cdp.js("({page: WL._vb.page, rows: document.querySelectorAll('#vb-list .vb-row').length,"
                                     " saved: S.settings.vocabPageSize})")
    await cdp.js("(() => { const g = document.querySelector('#vb-goto'); g.value = '3'; g.dispatchEvent(new Event('change')); })()")
    await asyncio.sleep(0.4)
    v["page_goto"] = await cdp.js("({page: WL._vb.page, summary: document.querySelector('#vb-summary').textContent})")
    await cdp.shot(out_dir / "27_vocab_pager.png")

    # 在這一頁把第一個字標成已會：它從「要學的」消失，但還停在同一頁；按提示上的復原，字回來、資料庫也清掉
    k3 = await cdp.js("document.querySelector('#vb-list .vb-row').dataset.k")
    K3 = json.dumps(k3)
    await cdp.js("document.querySelector('#vb-list .vb-row [data-st=\"2\"]').click()")
    await asyncio.sleep(0.8)
    v["mark_in_page"] = await cdp.js(f"""(async () => ({{page: WL._vb.page,
        gone: ![...document.querySelectorAll('#vb-list .vb-row')].some((r) => r.dataset.k === {K3}),
        toast: document.querySelector('.wl-toast:not([hidden])')?.innerText || '',
        srv: (await (await fetch('/api/lexeme?key=' + encodeURIComponent({K3}))).json()).st}}))()""")
    await cdp.shot(out_dir / "27b_vocab_marked.png")
    await cdp.js("document.querySelector('.wl-toast:not([hidden]) button')?.click()")
    await asyncio.sleep(0.8)
    v["undo_in_page"] = await cdp.js(f"""(async () => ({{page: WL._vb.page, st: WL._v.status.get({K3}),
        back: document.querySelector('#vb-list .vb-row').dataset.k === {K3},
        srv: (await (await fetch('/api/lexeme?key=' + encodeURIComponent({K3}))).json()).st}}))()""")
    # 保險：不管上面成不成功，都把這個字清回未標記
    await cdp.js(f"fetch('/api/lexeme/status', {{method: 'PUT', headers: {{'Content-Type': 'application/json'}},"
                 f" body: JSON.stringify({{key: {K3}, status: 0}})}}).then(() => WL._v.status.set({K3}, 0))")
    await asyncio.sleep(0.3)
    await cdp.js("document.querySelector('#vb-pages button').click()")
    await asyncio.sleep(0.3)

    # 狀態篩選：切到「不會」再切回「要學的」
    await cdp.js("[...document.querySelectorAll('#vb-status button')].find((b) => b.dataset.v === '1').click()")
    await asyncio.sleep(0.4)
    v["filter_unsure"] = await cdp.js("({filter: S.settings.vocabStatus, rows: document.querySelectorAll('#vb-list .vb-row').length,"
                                      " summary: document.querySelector('#vb-summary').textContent})")
    await cdp.js("[...document.querySelectorAll('#vb-status button')].find((b) => b.dataset.v === 'learn').click()")
    await asyncio.sleep(0.4)

    # 點單字：跳出視窗，不是就地展開
    await cdp.js("document.querySelector('#vb-list .vb-row').click()")
    opened = await wait_js(cdp, "!!document.querySelector('#dlg-word[open]') && WL._wd.groups !== null", 12)
    v["dialog_open"] = opened
    await asyncio.sleep(0.5)
    v["dialog"] = await cdp.js("({title: document.querySelector('#wd-title').textContent,"
                               " detail: document.querySelector('#wd-detail').innerText.slice(0, 120).replace(/\\n/g, ' | '),"
                               " examples: document.querySelectorAll('#wd-ex .vb-exi').length,"
                               " inline: document.querySelectorAll('#vb-list .vb-ex').length,"
                               " senses: document.querySelectorAll('#wd-detail .wp-senses li').length,"
                               " more: document.querySelector('#wd-detail [data-more]')?.textContent || '',"
                               " hscroll: (() => { const b = document.querySelector('#dlg-word .dlg-body');"
                               " return b.scrollWidth > b.clientWidth; })(),"
                               " status_btns: document.querySelectorAll('#wd-status [data-st]').length})")
    await cdp.shot(out_dir / "28_vocab_dialog.png")
    if await cdp.js("!!document.querySelector('#wd-detail [data-more]')"):
        await cdp.js("document.querySelector('#wd-detail [data-more]').click()")
        await asyncio.sleep(0.3)
        v["dialog_all_senses"] = await cdp.js("document.querySelectorAll('#wd-detail .wp-senses li').length")

    # 鍵盤：↓ 換下一個單字、N 標成不會（標完復原）
    k = await cdp.js("WL._wd.key")
    await key(cdp, "ArrowDown", "ArrowDown", 40)
    await asyncio.sleep(0.6)
    v["dialog_next"] = await cdp.js(f"({{moved: WL._wd.key !== {json.dumps(k)}, key: WL._wd.key}})")
    await key(cdp, "ArrowUp", "ArrowUp", 38)
    await asyncio.sleep(0.6)
    await key(cdp, "n", "KeyN", 78, "n")
    await asyncio.sleep(0.6)
    v["mark_unsure"] = await cdp.js(f"({{st: WL._v.status.get({json.dumps(k)}), undo: !document.querySelector('.wl-toast[hidden]')}})")
    await cdp.js(f"fetch('/api/lexeme/status', {{method: 'PUT', headers: {{'Content-Type': 'application/json'}},"
                 f" body: JSON.stringify({{key: {json.dumps(k)}, status: 0}})}}).then(() => WL._v.status.set({json.dumps(k)}, 0))")
    await asyncio.sleep(0.5)
    # 按住 N 不放：鍵盤自動重複的 keydown 不算，只標一次（不會在不會／未標記之間來回跳）
    await cdp.js("document.querySelector('#dlg-word .dlg-body').focus()")
    await cdp.send("Input.dispatchKeyEvent", type="keyDown", key="n", code="KeyN", windowsVirtualKeyCode=78, text="n")
    for _ in range(3):
        await cdp.send("Input.dispatchKeyEvent", type="keyDown", key="n", code="KeyN", windowsVirtualKeyCode=78, text="n", autoRepeat=True)
    await cdp.send("Input.dispatchKeyEvent", type="keyUp", key="n", code="KeyN", windowsVirtualKeyCode=78)
    await asyncio.sleep(0.8)
    v["hold_n"] = await cdp.js(f"WL._v.status.get({json.dumps(k)})")
    await cdp.js(f"fetch('/api/lexeme/status', {{method: 'PUT', headers: {{'Content-Type': 'application/json'}},"
                 f" body: JSON.stringify({{key: {json.dumps(k)}, status: 0}})}}).then(() => WL._v.status.set({json.dumps(k)}, 0))")
    await asyncio.sleep(0.5)

    # 小視窗播放：只播那一句
    await cdp.js("document.querySelector('#wd-ex [data-ex-mini]').click()")
    await wait_js(cdp, "!document.querySelector('#mini').hidden && document.querySelector('#mini-video').readyState >= 1", 20)
    await asyncio.sleep(1.5)
    v["mini"] = await cdp.js("({shown: !document.querySelector('#mini').hidden,"
                             " inDialog: document.querySelector('#mini').parentElement.id === 'dlg-word',"
                             " sub: document.querySelector('#mini-sub').innerText.slice(0, 60),"
                             " marked: document.querySelectorAll('#mini-sub mark').length,"
                             " start: WL._mini.ex && WL._mini.ex.start, end: WL._mini.ex && WL._mini.ex.end,"
                             " t: document.querySelector('#mini-video').currentTime,"
                             " mainPaused: video.paused, box: S.settings.miniBox})")
    await cdp.shot(out_dir / "29_vocab_mini.png")
    dur = await cdp.js("WL._mini.ex ? WL._mini.ex.end - WL._mini.ex.start : 0") or 0
    await wait_js(cdp, "document.querySelector('#mini-video').paused", dur + 8)
    await asyncio.sleep(1.0)                           # 停下來之後不會再自己往下播
    v["mini_stopped"] = await cdp.js("({paused: document.querySelector('#mini-video').paused,"
                                     " t: document.querySelector('#mini-video').currentTime,"
                                     " start: WL._mini.ex && WL._mini.ex.start, end: WL._mini.ex && WL._mini.ex.end})")

    # 拖標題列搬家、拖右下角改大小：位置和大小要記住
    before = await cdp.js("({...S.settings.miniBox})")
    head = await cdp.js("(() => { const r = document.querySelector('#mini-head').getBoundingClientRect();"
                        " return {x: r.left + 40, y: r.top + r.height / 2}; })()")
    await drag(cdp, head["x"], head["y"], -120, 60)
    await asyncio.sleep(0.2)
    grip = await cdp.js("(() => { const r = document.querySelector('.mini-grip.se').getBoundingClientRect();"
                        " return {x: r.left + 7, y: r.top + 7}; })()")
    await drag(cdp, grip["x"], grip["y"], 90, 50)
    await asyncio.sleep(0.8)
    v["mini_drag"] = await cdp.js(f"""({{before: {json.dumps(before)}, after: S.settings.miniBox,
        rect: (() => {{ const r = document.querySelector('#mini').getBoundingClientRect();
                        return [Math.round(r.left), Math.round(r.top), Math.round(r.width), Math.round(r.height)]; }})(),
        saved: JSON.parse(localStorage.getItem('vs:settings')).miniBox}})""")
    await cdp.shot(out_dir / "29b_vocab_mini_moved.png")

    # 開循環後關掉單字視窗：小視窗搬回頁面上而且繼續播
    await cdp.js("document.querySelector('#mini-loop').click()")
    await asyncio.sleep(1.0)
    await cdp.js("document.querySelector('#dlg-word').close()")
    await asyncio.sleep(1.0)
    v["mini_after_dialog"] = await cdp.js("({host: document.querySelector('#mini').parentElement.tagName,"
                                          " shown: !document.querySelector('#mini').hidden,"
                                          " playing: !document.querySelector('#mini-video').paused,"
                                          " loop: S.settings.miniLoop})")
    # 小視窗開著時再打開單字視窗：小視窗要搬進對話框（才點得到），而且不中斷
    await cdp.js("document.querySelector('#vb-list .vb-row').click()")
    await asyncio.sleep(1.0)
    v["mini_into_dialog"] = await cdp.js("({host: document.querySelector('#mini').parentElement.id,"
                                         " playing: !document.querySelector('#mini-video').paused,"
                                         " clickable: document.elementFromPoint(...(() => { const r = document.querySelector('#mini-replay').getBoundingClientRect();"
                                         " return [r.left + 5, r.top + 5]; })())?.id})")
    await cdp.js("document.querySelector('#dlg-word').close()")
    await asyncio.sleep(0.3)
    await cdp.js("document.querySelector('#mini-loop').click(); WL.closeMini()")
    await asyncio.sleep(0.4)

    # 單一影片範圍
    ja_mid = LOOKUP.get("ja")
    if ja_mid:
        await cdp.js(f"(() => {{ const s = document.querySelector('#vb-scope'); s.value = '{ja_mid}'; s.dispatchEvent(new Event('change')); }})()")
        await wait_js(cdp, f"WL._vb.data && WL._vb.data.media === '{ja_mid}'", 10)
        await asyncio.sleep(0.4)
        v["ja_one_video_rows"] = await cdp.js("document.querySelectorAll('#vb-list .vb-row').length")
        v["ja_one_video_summary"] = await cdp.js("document.querySelector('#vb-summary').textContent")
        await cdp.shot(out_dir / "30_vocab_one_video.png")
        # 從單字視窗的例句跳到影片
        await cdp.js("document.querySelector('#vb-list .vb-row').click()")
        await wait_js(cdp, "!!document.querySelector('#wd-ex [data-ex-jump]')", 10)
        target_t = await cdp.js("WL._wd.ex[0].start")
        target_text = await cdp.js("WL._wd.ex[0].text")
        await cdp.js("document.querySelector('#wd-ex [data-ex-jump]').click()")
        await asyncio.sleep(2.5)
        # 影片停著的時候，畫面上的字幕要是這個例句（不是前一句）
        v["jump"] = await cdp.js(f"({{view: document.querySelector('#vocab-page').hidden ? 'player' : 'vocab', media: S.current,"
                                 f" t: video.currentTime, target: {target_t}, dialog: !!document.querySelector('#dlg-word[open]'),"
                                 f" line_is_target: (S.cues[1][S.idx[1]] || {{}}).text === {json.dumps(target_text)}}})")
        await cdp.js("showView('vocab')")
        await cdp.js("(() => { const s = document.querySelector('#vb-scope'); s.value = ''; s.dispatchEvent(new Event('change')); })()")
        await asyncio.sleep(0.5)
    for lang in ("en", "zh"):
        await cdp.js(f"document.querySelector('#vb-lang [data-v={lang}]').click()")
        await wait_js(cdp, f"WL._vb.data && WL._vb.data.lang === '{lang}'", 60 if lang == "zh" else 15)
        await asyncio.sleep(0.8)
        v[f"{lang}_rows"] = await cdp.js("document.querySelectorAll('#vb-list .vb-row').length")
        v[f"{lang}_summary"] = await cdp.js("document.querySelector('#vb-summary').textContent")
        v[f"{lang}_note"] = await cdp.js("document.querySelector('#vb-note').hidden ? '' : document.querySelector('#vb-note').innerText")
        v[f"{lang}_first_rows"] = await cdp.js("[...document.querySelectorAll('#vb-list .vb-row')].slice(0, 5).map((r) => r.innerText.replace(/\\n/g, ' | '))")
        await cdp.shot(out_dir / f"29_vocab_{lang}.png")
    await cdp.js("S.settings.vocabLang = 'ja'; saveSettings()")

    # 設定頁的查字區塊
    await cdp.js("showView('settings')")
    await asyncio.sleep(2.5)
    await cdp.js("document.querySelector('#lookup-card').scrollIntoView({block: 'start'})")
    await asyncio.sleep(0.8)
    rep["settings_dict"] = await cdp.js("document.querySelector('#dict-status').innerText")
    rep["settings_glosses_hint"] = await cdp.js("document.querySelector('#glosses-hint').textContent")
    await cdp.shot(out_dir / "30_settings_lookup.png")
    await cdp.js("showView('player')")


# ---------- 首次自動安裝（--setup）：在頁面裡攔截 API，走過每一種狀態 ----------

MIB = 1024 ** 2
GIB = 1024 ** 3

# 蓋掉 window.fetch：安裝相關的 API 回假資料，其他照常打伺服器，/api/state 等回應再補上假的欄位。
# window.__MOCK 由測試在載入前設定，window.__calls 記下送出的 POST（檢查按鈕打了哪個 API）
SETUP_MOCK_JS = r"""(() => {
  const M0 = window.__MOCK;
  try {
    localStorage.clear();
    for (const [k, v] of Object.entries((M0 && M0.local) || {})) localStorage.setItem(k, JSON.stringify(v));
  } catch { /* 無痕 */ }
  const real = window.fetch.bind(window);
  const json = (data, status = 200) => new Response(JSON.stringify(data), {status, headers: {'Content-Type': 'application/json'}});
  window.__calls = [];
  window.fetch = async (input, opts = {}) => {
    const M = window.__MOCK;
    if (!M) return real(input, opts);
    const url = new URL(String(input), location.href);
    const path = url.pathname;
    const method = (opts.method || 'GET').toUpperCase();
    if (method !== 'GET') window.__calls.push(method + ' ' + path + (opts.body ? ' ' + opts.body : ''));
    const hasSetup = !!M.setup && M.setup !== 404;
    if (path === '/api/setup' || path.startsWith('/api/setup/')) {
      if (!hasSetup) return json({detail: 'Not Found'}, 404);
      if (path === '/api/setup') return json(M.setup);
      if (path === '/api/setup/dismiss') { M.setup.dismissed = true; if (M.summary) M.summary.dismissed = true; return json({ok: true}); }
      if (path === '/api/setup/recheck') {
        if (M.recheck) { M.setup = M.recheck.setup; M.summary = M.recheck.summary; M.recheck = null; }
        // 跟真的後端一樣：還是不能裝時回 400 blocked（磁碟不夠是 disk，detail 是後端寫的原因和怎麼辦）
        if (M.setup.status === 'blocked') {
          const disk = M.setup.block && M.setup.block.code === 'disk';
          return json({detail: disk ? 'D: 空間不夠：要 13.8 GB，只剩 8.2 GB。清出 5.6 GB 以上再按「重新檢查」。' : '還是不能裝',
            code: disk ? 'disk' : 'blocked'}, 400);
        }
        return json(M.setup);
      }
      if (path === '/api/setup/plan') return json(M.plan);
      if (path === '/api/setup/install') {
        const r = (M.install || []).shift() || {body: {ok: true, nothing: true}};
        return json(r.body, r.status || 200);
      }
      const m = path.match(/^\/api\/setup\/items\/([^/]+)\/(retry|resume)$/);
      if (m) {
        const it = (M.setup.items || []).find((i) => i.key === decodeURIComponent(m[1]));
        if (!it) return json({detail: '找不到這一項', code: 'unknown_item'}, 404);
        it.state = m[2] === 'resume' ? 'downloading' : 'waiting';
        it.error = null;
        if (M.setup.status === 'done') M.setup.status = 'installing';
        return json(M.setup);
      }
      return json({detail: 'Not Found'}, 404);
    }
    if (/^\/api\/dicts\/(ja|en|zh)\/build$/.test(path) && method === 'POST') return json({ok: true, job_id: 'mockdict0001'});
    if (path === '/api/models/ckip/download' && method === 'POST') return json({ok: true, job_id: 'mockckip0001'});
    const res = await real(input, opts);
    if (method !== 'GET' || !res.ok) return res;
    const patch = async (fn) => { const d = await res.clone().json(); fn(d); return json(d); };
    if (path === '/api/state') return patch((d) => {
      if (hasSetup) d.setup = M.summary || null;
      else if (M.setup === 404) delete d.setup;          // 模擬舊版伺服器：沒有自動安裝
      if (M.jobs) d.jobs = d.jobs.concat(M.jobs);
      if (M.media) d.media = d.media.concat(M.media);
    });
    if (path === '/api/meta' && M.meta) return patch((d) => {
      for (const [kind, key, extra] of M.meta) if (d[kind] && d[kind][key]) Object.assign(d[kind][key], extra);
    });
    if (path === '/api/dicts' && hasSetup) return patch((d) => {
      for (const k of ['ja', 'en', 'zh']) Object.assign(d[k], {job: null, est_s: {ja: 20, en: 300, zh: 25}[k]}, (M.dicts || {})[k] || {});
    });
    if (path === '/api/dicts' && M.setup === 404) return patch((d) => {
      for (const k of ['ja', 'en', 'zh']) { delete d[k].job; delete d[k].est_s; }
    });
    if (path === '/api/vocab' && M.vocab) return patch((d) => Object.assign(d, M.vocab));
    return res;
  };
})();"""


def _gpu(name, gb, total, usable, driver="595.79", warn=False):
    return {"name": name, "gb": gb, "total_mb": total, "usable_mb": usable, "driver": driver, "cuda": "13.2", "driver_warn": warn}


GPU8 = _gpu("NVIDIA GeForce RTX 4060", 8, 8188, 7988)
GPU12 = _gpu("NVIDIA GeForce RTX 4070", 12, 12282, 12032)


def _items(tier=8, **states):
    """第一次安裝的六個項目；states 用 role 指定狀態：role=(state, 其他欄位)，沒指定的是 done。"""
    tr = {8: ("model:hymt2-7b:Q6_K", "Hy-MT2-7B Q6_K", 6164482720), 12: ("model:hymt2-7b:Q8_0", "Hy-MT2-7B Q8_0", 7981928896),
          "mini": ("model:hymt2-1.8b", "Hy-MT2-1.8B", 1820 * MIB)}[tier]
    rows = [("model:tsqyomi", "furigana", "tsqyomi", 77 * MIB), ("model:aligner", "aligner", "Qwen3-ForcedAligner-0.6B", 1755 * MIB),
            ("model:qwen-asr", "asr", "Qwen3-ASR-1.7B", 4702863360), (tr[0], "translator", tr[1], tr[2]),
            ("dict:ja", "dict_ja", "日文字典", 14232322), ("dict:en", "dict_en", "英文字典", 51857125)]
    out = []
    for key, role, label, size in rows:
        state, extra = states.get(role, ("done", {}))
        kind, _, rest = key.partition(":")
        it = {"key": key, "kind": kind, "role": role, "label": label, "size_bytes": size, "job_id": "job" + rest.replace(":", "")[:9],
              "state": state, "done_bytes": size if state == "done" else 0, "speed_bps": 0, "eta_s": None, "retry": None, "error": None}
        if kind == "model":
            mid, _, variant = rest.partition(":")
            it.update(id=mid, variant=variant or None)
        else:
            it["lang"] = rest
        it.update(extra)
        out.append(it)
    return out


def _status(status="installing", tier=8, items=(), reason="new", dismissed=False, gpu=None, choice=None, block=None, disk=None, eta=None):
    if gpu is None:
        gpu = GPU12 if tier == 12 else GPU8
    if choice is None:
        choice = {"id": "hymt2-7b", "label": "Hy-MT2-7B", "variant": "Q8_0" if tier == 12 else "Q6_K", "default_variant": "Q8_0",
                  "ctx": 8192 if tier == 12 else 4096, "parallel": 4 if tier == 12 else 2, "vram_mb": 8812 if tier == 12 else 6578,
                  "budget_mb": 11008 if tier == 12 else 6964, "fit": "ok" if tier == 12 else "tight"}
    items = list(items)
    total = sum(i["size_bytes"] for i in items)
    done = sum(i["done_bytes"] or 0 for i in items)
    done_of = lambda *roles: all(i["state"] == "done" for i in items if i["role"] in roles)  # noqa: E731
    return {
        "status": status, "reason": reason, "dismissed": dismissed, "workers": True, "off": False,
        "gpu": gpu or None, "choice": choice or None, "block": block,
        "disk": disk or {"drive": "D:", "need_bytes": 15160000000, "free_bytes": 128000000000, "after_bytes": 112000000000, "warn": False},
        "totals": {"done_bytes": done, "total_bytes": total, "speed_bps": sum(i["speed_bps"] or 0 for i in items),
                   "eta_s": eta if eta is not None else (max([i["eta_s"] or 0 for i in items] + [0]) or None),
                   "percent": round(done * 100 / total) if total else 0},
        "ready": {"transcribe": done_of("asr", "aligner"), "translate": done_of("translator"),
                  "dict_ja": done_of("dict_ja"), "dict_en": done_of("dict_en")},
        "items": items, "started_at": 1789650012.3, "finished_at": 1789651900.0 if status == "done" else None,
    }


def _summary(d):
    items = d["items"]
    return {"status": d["status"], "percent": d["totals"]["percent"], "eta_s": d["totals"]["eta_s"], "dismissed": d["dismissed"],
            "net_wait": any(i["state"] == "net_wait" for i in items), "problem": any(i["state"] == "failed" for i in items),
            "blocked": d["status"] == "blocked", "ready": {k: d["ready"][k] for k in ("transcribe", "translate")}}


def _mock(setup, **kw):
    m = {"setup": setup, "summary": _summary(setup) if isinstance(setup, dict) else None}
    m.update(kw)
    return m


def setup_states() -> dict:
    """每一種要截圖的狀態。值是 (GET /api/setup 的內容, 要檢查的文字)。"""
    s = {}
    s["8gb"] = (_status(items=_items(8, asr=("downloading", {"done_bytes": 1288490188, "speed_bps": 12500000, "eta_s": 273}),
                                     translator=("waiting", {}),
                                     dict_en=("building", {"progress": 0.43, "stage": "build", "eta_s": 170, "done_bytes": 51857125})),
                        eta=760),
                {"title": "正在安裝", "notes": ["顯存 8 GB，翻譯模型用小一點的 Q6_K 版。"], "ready": "辨識和時間軸裝好就能先轉字幕。",
                 "total": "約 13 分鐘", "rows": ["好了", "好了", "1.2 / 4.4 GB · 約 5 分鐘", "等待中", "好了", "建立中 43%"],
                 "gpu": "NVIDIA GeForce RTX 4060 · 8 GB · 驅動 595.79 · CUDA 13.2（需要 13.0 以上）", "main": "先去用"})
    s["12gb"] = (_status(tier=12, items=_items(12, translator=("downloading", {"done_bytes": 3328599654, "speed_bps": 11800000, "eta_s": 395}))),
                 {"title": "正在安裝", "notes": [], "ready": "可以先轉字幕了，翻譯還在下載。", "total": "11.3 MB/s",
                  "rows": ["好了", "好了", "好了", "3.1 / 7.4 GB · 約 7 分鐘", "好了", "好了"]})
    net_err = {"code": "network", "message": "連不到 Hugging Face。已下載的部分會保留，網路恢復後會自動接著下載。", "next_check_s": 45}
    s["net_wait"] = (_status(items=_items(8, asr=("net_wait", {"done_bytes": 2 * GIB, "error": net_err}), translator=("waiting", {}),
                                           dict_en=("building", {"progress": 0.71, "stage": "build", "done_bytes": 51857125}))),
                     {"title": "正在安裝", "total": "等網路恢復", "rows": ["好了", "好了", "等網路恢復", "等待中", "好了", "建立中 71%"],
                      "badge": "安裝等網路"})
    ssl_err = {"code": "ssl", "message": "SSL 憑證驗證失敗，公司或學校的網路可能攔截了連線。"}
    s["failed"] = (_status(items=_items(8, translator=("failed", {"done_bytes": 700 * MIB, "error": ssl_err}))),
                   {"title": "正在安裝", "rows": ["好了", "好了", "好了", "連線被攔截，換個網路再試重試", "好了", "好了"],
                    "ready": "可以先轉字幕了。", "badge": "安裝有問題", "tips": [ssl_err["message"]]})
    # 模型都下載完、只剩英文字典在建：不寫下載速度，寫「建立字典」和剩多久
    s["building"] = (_status(items=_items(8, dict_en=("building", {"progress": 0.6, "stage": "build", "eta_s": 90, "done_bytes": 51857125}))),
                     {"title": "正在安裝", "total": "12.0 / 12.0 GB · 建立字典 · 約 2 分鐘",
                      "rows": ["好了", "好了", "好了", "好了", "好了", "建立中 60%"], "ready": "可以轉字幕和翻譯了，字典還在建立。"})
    s["paused"] = (_status(items=_items(8, translator=("paused", {"done_bytes": 2465792288}))),
                   {"title": "正在安裝", "rows": ["好了", "好了", "好了", "已暫停繼續", "好了", "好了"], "total": "已暫停"})
    s["states"] = (_status(gpu=_gpu("NVIDIA GeForce RTX 3060 Laptop GPU", 8, 8192, 7900, driver="581.15", warn=True),
                           choice={"id": "hymt2-1.8b", "label": "Hy-MT2-1.8B", "variant": None, "default_variant": None,
                                   "vram_mb": 2720, "budget_mb": 6876, "fit": "ok"},
                           disk={"drive": "C:", "need_bytes": 14800000000, "free_bytes": 22600000000, "after_bytes": 7 * GIB + 300 * MIB, "warn": True},
                           items=_items("mini", asr=("checking", {"done_bytes": 4702863360}),
                                        aligner=("retry_wait", {"done_bytes": 900 * MIB, "retry": 30}), translator=("waiting", {}),
                                        dict_ja=("downloading", {"done_bytes": 6 * MIB}),
                                        dict_en=("building", {"progress": 0.17, "stage": "extract", "done_bytes": 51857125}))),
                   {"title": "正在安裝", "notes": ["驅動版本比較舊，翻譯開不起來時先更新驅動。", "顯存比較小，翻譯改用小模型 Hy-MT2-1.8B。",
                                                "裝完 C: 只剩 7 GB，處理影片可能不夠。"],
                    "rows": ["好了", "連線中斷，30 秒後重試", "檢查已下載的部分", "等待中", "6 / 14 MB", "解壓縮"]})
    s["pending"] = (_status(status="pending", gpu=False, choice=False),
                    {"title": "正在準備", "sub": "檢查顯示卡和磁碟空間…", "rows": [], "main": "先去用", "busy": True, "change": False})
    s["done"] = (_status(status="done", items=_items(8)), {"title": "裝好了", "main": "開始使用", "missing": []})
    s["done_canceled"] = (_status(status="done", items=_items(8, translator=("canceled", {}))),
                          {"title": "裝好了", "main": "開始使用", "missing": ["沒裝：Hy-MT2-7B Q6_K（已取消）重新下載"]})
    blocks = {
        "no_gpu": (False, {}, "沒有找到 NVIDIA 顯示卡。轉字幕和翻譯要用 NVIDIA 顯示卡，顯存 8 GB 以上。",
                   "有 NVIDIA 顯示卡的話，先裝好驅動再按「重新檢查」。"),
        "driver_old": (dict(_gpu("NVIDIA GeForce RTX 3060", 12, 12288, 12032, driver="552.22"), cuda="12.4"),
                       {"driver": "552.22", "cuda": "12.4"},
                       "顯示卡驅動太舊（目前 552.22，支援 CUDA 12.4）。這個程式需要驅動 580 版以上（支援 CUDA 13.0）。",
                       "到 NVIDIA 官網或 NVIDIA App 更新驅動，再按「重新檢查」。"),
        "arch_old": (_gpu("NVIDIA GeForce GTX 1080", 8, 8192, 7990), {"name": "NVIDIA GeForce GTX 1080"},
                     "這張顯示卡（GTX 1080）太舊，程式用不了。", "需要 RTX 20、GTX 16 系列或更新的 NVIDIA 顯示卡。"),
        "vram_small": (_gpu("NVIDIA GeForce RTX 2060", 6, 6144, 5950), {"total_mb": 6144},
                       "顯存只有 6 GB，模型放不下。", "需要顯存 8 GB 以上的 NVIDIA 顯示卡。"),
        "gpu_hidden": (False, {}, "環境變數 CUDA_VISIBLE_DEVICES 把顯示卡藏起來了。", "到 Windows 的環境變數拿掉它，再重新打開程式。"),
        "disk": (GPU8, {"drive": "D:", "need_bytes": 14119 * MIB, "free_bytes": int(8.2 * GIB), "short_bytes": 14119 * MIB - int(8.2 * GIB)},
                 "D: 空間不夠：要 13.8 GB，只剩 8.2 GB。", "清出 5.6 GB 以上再按「重新檢查」。"),
        "detect_failed": (False, {}, "檢查電腦時出錯了。", "按「重新檢查」再試一次。一直這樣的話，把 data\\app.log 傳給作者。"),
    }
    for code, (g, values, why, how) in blocks.items():
        s["blocked_" + code] = (_status(status="blocked", gpu=g, choice=False, block={"code": code, "values": values}),
                                {"title": "沒辦法自動安裝", "why": why, "how": how, "main": "重新檢查", "badge": "沒辦法安裝"})
    return s


# 進度頁上看得到的東西
SETUP_VIEW = """(() => {
  const q = (s) => document.querySelector(s);
  const vis = (s) => { const el = q(s); return !!el && !el.hidden && el.getClientRects().length > 0; };
  const box = q('#setup');
  return {open: vis('#setup'), title: q('#setup-title').textContent, sub: vis('#setup-sub') ? q('#setup-sub').textContent : '',
    gpu: vis('#setup-gpu') ? q('#setup-gpu').textContent : '',
    notes: vis('#setup-notes') ? [...document.querySelectorAll('#setup-notes p')].map((p) => p.textContent) : [],
    total: vis('#setup-total') ? q('#setup-total-text').textContent : '',
    rows: vis('#setup-items') ? [...document.querySelectorAll('#setup-items li')].map((li) => li.querySelector('.su-right').textContent.trim()) : [],
    names: [...document.querySelectorAll('#setup-items .su-name')].map((n) => n.textContent),
    tips: [...document.querySelectorAll('#setup-items .su-right span[title]')].map((n) => n.title),
    ready: vis('#setup-ready') ? q('#setup-ready').textContent : '',
    missing: vis('#setup-missing') ? [...document.querySelectorAll('#setup-missing > div')].map((d) => d.textContent.trim()) : [],
    why: vis('#setup-block') ? q('#setup-why').textContent : '', how: vis('#setup-block') ? q('#setup-how').textContent : '',
    main: q('#setup-main').textContent, close: vis('#setup-close'), change: vis('#setup-change'), foot: vis('#setup-foot'),
    busy: vis('#setup-total') && q('#setup-total-bar').parentElement.classList.contains('busy'),
    badge: vis('#setup-badge') ? q('#setup-badge-long').textContent : '', badgeShort: q('#setup-badge-short').textContent,
    hscroll: box.scrollWidth > box.clientWidth + 1 || document.documentElement.scrollWidth > innerWidth + 1,
    below: box.scrollHeight > box.clientHeight + 1};
})()"""

_MOCK_SCRIPT = {"id": None}


async def load_mock(cdp: CDP, url: str, mock: dict):
    """換成這組假資料，重新載入頁面，等第一次輪詢做完。"""
    if _MOCK_SCRIPT["id"]:
        await cdp.send("Page.removeScriptToEvaluateOnNewDocument", identifier=_MOCK_SCRIPT["id"])
    src = "window.__MOCK = " + json.dumps(mock, ensure_ascii=False) + ";\n" + SETUP_MOCK_JS
    _MOCK_SCRIPT["id"] = (await cdp.send("Page.addScriptToEvaluateOnNewDocument", source=src))["identifier"]
    await cdp.send("Page.navigate", url="about:blank")
    await asyncio.sleep(0.3)
    await cdp.send("Page.navigate", url=url)
    await wait_js(cdp, "typeof SU !== 'undefined' && !!S.meta && (SU.decided || !SU.api) && !!S.sig", 20)
    await asyncio.sleep(0.6)


async def setup_checks(out_dir: Path, media_id: str, media_url: str) -> list:
    results = []

    def check(step, ok, **detail):
        results.append({"step": step, "ok": bool(ok), **detail})

    with urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{PORT}/json/new?about:blank", method="PUT")) as r:
        target = json.load(r)
    async with websockets.connect(target["webSocketDebuggerUrl"], max_size=50_000_000) as ws:
        cdp = CDP(ws)
        listener = asyncio.create_task(cdp.listen())
        for dom in ("Runtime", "Log", "Page"):
            await cdp.send(f"{dom}.enable")
        states = setup_states()

        async def view():
            return await cdp.js(SETUP_VIEW)

        async def shot(name):
            await cdp.shot(out_dir / f"{name}.png")

        async def shot_setup(name):
            """進度頁的截圖；內容比畫面高時，另外捲到底再截一張 _b。"""
            await cdp.js("document.querySelector('#setup').scrollTop = 0")
            await asyncio.sleep(0.2)
            await shot(name)
            if await cdp.js("(() => { const b = document.querySelector('#setup'); return b.scrollHeight > b.clientHeight + 1; })()"):
                await cdp.js("(() => { const b = document.querySelector('#setup'); b.scrollTop = b.scrollHeight; })()")
                await asyncio.sleep(0.2)
                await shot(name + "_b")
                await cdp.js("document.querySelector('#setup').scrollTop = 0")

        def expect(name, got, want):
            # total 只比對一部分（速度、剩多久）；其他要完全一樣
            same = lambda k, v: (v in (got.get(k) or "")) if k == "total" else got.get(k) == v  # noqa: E731
            bad = {k: {"want": v, "got": got.get(k)} for k, v in want.items() if k != "badge" and not same(k, v)}
            check(f"{name}：進度頁內容", got["open"] and not bad and not got["hscroll"], diff=bad, hscroll=got["hscroll"])

        for dev in ("desk", "phone"):
            if dev == "desk":
                await cdp.send("Emulation.setTouchEmulationEnabled", enabled=False)
                await cdp.send("Emulation.setDeviceMetricsOverride", width=1600, height=900, deviceScaleFactor=1, mobile=False)
            else:
                await cdp.send("Emulation.setTouchEmulationEnabled", enabled=True, maxTouchPoints=5)
                await cdp.send("Emulation.setDeviceMetricsOverride", width=412, height=915, deviceScaleFactor=1, mobile=True)
            tag = f"[{dev}]"

            # 1. 每一種狀態：自動打開、內容、截圖
            for name, (st, want) in states.items():
                await load_mock(cdp, APP, _mock(st))
                got = await view()
                expect(f"{tag} {name}", got, want)
                if dev == "phone" and name == "8gb":
                    check(f"{tag} 手機是觸控模式", await cdp.js("matchMedia('(hover: none) and (pointer: coarse)').matches"))
                if name == "done":
                    add = "新增" if dev == "phone" else "新增影片"   # 手機頂列的按鈕只寫「新增」
                    check(f"{tag} 完成畫面的說明照頂列按鈕的字", got["sub"] == f"可以開始用了。按上方「{add}」加第一部影片。", got=got["sub"])
                await shot_setup(f"setup_{name}_{dev}")
                if "badge" in want:
                    await cdp.js("closeSetup()")
                    await asyncio.sleep(0.3)
                    got2 = await view()
                    # 不能裝時按「關閉」就不顯示右上角狀態；安裝中的關掉後右上角一直顯示
                    wanted = "" if name.startswith("blocked") else want["badge"]
                    check(f"{tag} {name}：關掉後右上角狀態", got2["badge"] == wanted and not got2["open"], got=got2["badge"], short=got2["badgeShort"])
                    if name in ("net_wait", "failed"):
                        await shot(f"setup_{name}_badge_{dev}")

            # 2. 已經在用（existing）、關掉（off）、舊版伺服器（沒有 API）：都不出現進度頁和右上角狀態
            for name, mock in (("existing", _mock(_status(status="done", reason="existing", items=_items(8)))),
                               ("off", _mock(dict(_status(status="off", items=[]), off=True), summary=None)),
                               ("no_api", {"setup": 404})):
                await load_mock(cdp, APP, mock)
                await asyncio.sleep(1.2)
                got = await view()
                calls = await cdp.js("window.__calls")
                check(f"{tag} {name}：不出現進度頁和右上角狀態", not got["open"] and not got["badge"] and not calls, got=got, calls=calls)
                if name == "no_api":
                    await cdp.js("showView('settings')")
                    await asyncio.sleep(1.5)
                    card = await cdp.js("({card: document.querySelector('#setup-card').hidden, dict: document.querySelector('#dict-status').innerText,"
                                        " buttons: document.querySelectorAll('#dict-status button').length})")
                    check(f"{tag} 舊版伺服器：沒有自動安裝卡片、字典照舊說明", card["card"] and "tools" in card["dict"] and not card["buttons"], got=card)
                    await cdp.js("showView('player')")
            await shot(f"setup_existing_{dev}")

            # 3. 先去用、右上角狀態、重新整理後不再蓋上、Esc、快捷鍵
            st8 = states["8gb"][0]
            await load_mock(cdp, APP, _mock(json.loads(json.dumps(st8))))
            await cdp.js("document.querySelector('#setup-main').click()")
            await asyncio.sleep(0.4)
            got = await view()
            calls = await cdp.js("window.__calls")
            check(f"{tag} 先去用：關掉、記住、右上角顯示安裝 %", not got["open"] and calls == ["POST /api/setup/dismiss"]
                  and got["badge"] == f"安裝 {st8['totals']['percent']}%", got=got["badge"], calls=calls)
            await shot(f"setup_badge_{dev}")
            await cdp.js("document.querySelector('#setup-badge').click()")
            await asyncio.sleep(0.5)
            got = await view()
            check(f"{tag} 點右上角狀態打開進度頁", got["open"])
            await key(cdp, "Escape", "Escape", 27)
            await asyncio.sleep(0.3)
            got = await view()
            calls = await cdp.js("window.__calls")
            check(f"{tag} Esc 關掉，已經按過先去用不會再送一次", not got["open"] and calls == ["POST /api/setup/dismiss"], calls=calls)
            dismissed = json.loads(json.dumps(st8))
            dismissed["dismissed"] = True
            await load_mock(cdp, APP, _mock(dismissed))
            await asyncio.sleep(0.8)
            got = await view()
            check(f"{tag} 按過先去用，重新整理後不會自動蓋上", not got["open"] and got["badge"].startswith("安裝"), got=got["badge"])

            if dev == "desk":
                await load_mock(cdp, APP, _mock(json.loads(json.dumps(st8))))
                await key(cdp, "?", "Slash", 191, "?")
                await asyncio.sleep(0.3)
                check(f"{tag} 進度頁開著時快捷鍵不作用（? 不會打開說明）", not await cdp.js("document.querySelector('#dlg-help').open"))
                inert = await cdp.js("[...document.querySelectorAll('.topbar, .layout')].every((e) => e.inert)")
                check(f"{tag} 進度頁開著時後面按不到", inert)

                # 改選模型：關掉進度頁，到設定頁的模型管理
                await cdp.js("document.querySelector('#setup-change').click()")
                await asyncio.sleep(1.5)
                got = await view()
                check(f"{tag} 改選模型：到設定頁模型管理", not got["open"] and await cdp.js("SET.open && !document.querySelector('#settings-page').hidden")
                      and "POST /api/setup/dismiss" in await cdp.js("window.__calls"))

                # 安裝中變成裝好了：進度頁跟著換成完成畫面
                await load_mock(cdp, APP, _mock(json.loads(json.dumps(st8))))
                done = states["done"][0]
                await cdp.js(f"window.__MOCK.setup = {json.dumps(done)}; window.__MOCK.summary = {json.dumps(_summary(done))};")
                await wait_js(cdp, "document.querySelector('#setup-title').textContent === '裝好了'", 6)
                got = await view()
                check(f"{tag} 安裝中變成裝好了，畫面跟著換", got["title"] == "裝好了" and got["main"] == "開始使用", got=got)
                await cdp.js("document.querySelector('#setup-main').click()")
                await asyncio.sleep(0.3)
                check(f"{tag} 開始使用：關掉並記住", not (await view())["open"] and "POST /api/setup/dismiss" in await cdp.js("window.__calls"))

                # 背景安裝完成時跳一次提示
                await load_mock(cdp, APP, _mock(dismissed))
                await cdp.js(f"window.__MOCK.setup = {json.dumps(dict(done, dismissed=True))}; window.__MOCK.summary = {json.dumps(dict(_summary(done), dismissed=True))};")
                await wait_js(cdp, "!document.querySelector('#toast').hidden", 6)
                toast_text = await cdp.js("document.querySelector('#toast').textContent")
                got = await view()
                check(f"{tag} 關著進度頁時裝好：提示一次、右上角消失", toast_text == "模型和字典都裝好了" and not got["badge"], toast=toast_text)

            # 4. 項目的按鈕：失敗重試、暫停繼續、取消的重新下載
            for name, act, key_ in (("failed", "retry", "model:hymt2-7b:Q6_K"), ("paused", "resume", "model:hymt2-7b:Q6_K"),
                                    ("done_canceled", "retry", "model:hymt2-7b:Q6_K")):
                await load_mock(cdp, APP, _mock(json.loads(json.dumps(states[name][0]))))
                await cdp.js(f"document.querySelector('#setup [data-su-act=\"{act}\"]').click()")
                await asyncio.sleep(0.6)
                calls = await cdp.js("window.__calls")
                got = await view()
                check(f"{tag} {name}：按鈕送出 {act}", calls == [f"POST /api/setup/items/{key_.replace(':', '%3A')}/{act}"]
                      and got["open"] and got["title"] == "正在安裝", calls=calls, rows=got["rows"])

            # 5. 不能裝：重新檢查（還是不行 → 提示；通過 → 開始安裝）、關閉
            blk = states["blocked_driver_old"][0]
            ok_state = states["8gb"][0]
            await load_mock(cdp, APP, _mock(json.loads(json.dumps(blk))))
            await cdp.js("document.querySelector('#setup-main').click()")
            await wait_js(cdp, "!document.querySelector('#toast').hidden", 4)
            check(f"{tag} 重新檢查還是不行：提示", await cdp.js("document.querySelector('#toast').textContent") == "還是沒辦法安裝")
            # 磁碟不夠（後端回 400 disk）：一樣只提示「還是沒辦法安裝」，數字看畫面，不會出現第二個不同的數字
            await load_mock(cdp, APP, _mock(json.loads(json.dumps(states["blocked_disk"][0]))))
            await cdp.js("document.querySelector('#setup-main').click()")
            await wait_js(cdp, "!document.querySelector('#toast').hidden", 4)
            toast_disk = await cdp.js("document.querySelector('#toast').textContent")
            check(f"{tag} 磁碟不夠時重新檢查還是不行：提示", toast_disk == "還是沒辦法安裝" and (await view())["open"], got=toast_disk)
            await load_mock(cdp, APP, _mock(json.loads(json.dumps(blk))))
            await cdp.js(f"window.__MOCK.recheck = {{setup: {json.dumps(ok_state)}, summary: {json.dumps(_summary(ok_state))}}}")
            await cdp.js("document.querySelector('#setup-main').click()")
            await asyncio.sleep(0.8)
            got = await view()
            check(f"{tag} 重新檢查通過：換成安裝畫面", got["open"] and got["title"] == "正在安裝" and len(got["rows"]) == 6, got=got["title"])
            await load_mock(cdp, APP, _mock(json.loads(json.dumps(blk))))
            await cdp.js("document.querySelector('#setup-close').click()")
            await asyncio.sleep(0.4)
            got = await view()
            check(f"{tag} 不能裝時按關閉：進度頁和右上角都消失", not got["open"] and not got["badge"]
                  and await cdp.js("window.__calls") == ["POST /api/setup/dismiss"])

            # 6. 佇列：等模型的任務
            now = time.time()
            fake_media = [{"id": f"mock0000000{n}", "title": t, "title_zh": None, "source": "url", "path": None,
                           "url": f"https://www.youtube.com/watch?v=mock{n}", "duration": 600 + n * 60, "language": "ja", "profile": None,
                           "vcodec": None, "width": None, "height": None, "playable": False, "has_proxy": False, "has_thumb": False,
                           "created_at": now - 100 + n, "tracks": []}
                          for n, t in ((1, "東京で一番ツッコミどころが多い街を見つけた"), (2, "浅草で登山してきました"), (3, "深夜の散歩 vlog"))]

            def job(jid, typ, status, mid=None, **kw):
                j = {"id": jid, "media_id": mid, "type": typ, "status": status, "params": {}, "progress": 0, "stage": "",
                     "position": len(jid), "depends_on": None, "error": None, "result": None,
                     "created_at": now - 50, "started_at": now - 40 if status == "running" else None, "finished_at": None}
                j.update(kw)
                return j
            fake_jobs = [
                job("mockjob00001", "transcribe", "queued", "mock00000001", position=10, params={"engine": "qwen", "language": "ja"},
                    waiting={"models": ["qwen-asr"], "labels": ["Qwen3-ASR-1.7B"], "state": "downloading", "eta_s": 273}),
                job("mockjob00002", "translate", "queued", "mock00000001", position=11, params={"translator": "hymt"},
                    depends_on="mockjob00001", waiting={"models": ["hymt2-7b"], "labels": ["Hy-MT2-7B"], "state": "waiting"}),
                job("mockjob00003", "transcribe", "queued", "mock00000002", position=12, params={"engine": "qwen", "language": "ja"},
                    waiting={"models": ["qwen-asr"], "labels": ["Qwen3-ASR-1.7B"], "state": "paused"}),
                job("mockjob00004", "translate", "queued", "mock00000003", position=13, params={"translator": "hymt"},
                    waiting={"models": ["hymt2-7b"], "labels": ["Hy-MT2-7B"], "state": "failed"}),
                job("mockjob00005", "transcribe", "queued", "mock00000003", position=9, params={"engine": "qwen", "language": "ja"},
                    waiting={"models": ["qwen-asr"], "labels": ["Qwen3-ASR-1.7B"], "state": "net_wait"}),
                job("mockjob00006", "model", "running", position=1, progress=0.27, stage="下載 1.2 / 4.4 GB",
                    params={"model": "qwen-asr", "label": "Qwen3-ASR-1.7B", "setup": True}),
                job("mockjob00007", "dict", "running", position=2, progress=0.43, stage="建立字典 43%",
                    params={"lang": "en", "label": "英文字典", "setup": True}),
            ]
            meta_pending = [["engines", "qwen", {"installed": False, "pending": True}],
                            ["translators", "hymt", {"installed": False, "pending": True}]]
            await load_mock(cdp, APP, _mock(dismissed, jobs=fake_jobs, media=fake_media, meta=meta_pending))
            await cdp.js("showTab('queue', true)")
            await asyncio.sleep(0.8)
            q = await cdp.js("""[...document.querySelectorAll('#queue .job')].map((li) => ({
                state: li.querySelector('.job-state').textContent, stage: li.querySelector('.job-stage')?.textContent.trim() || '',
                kind: li.querySelector('.job-kind').textContent}))""")
            texts = {x["stage"] for x in q}
            want_q = {"等 Qwen3-ASR-1.7B 下載完，約 5 分鐘", "等 Qwen3-ASR-1.7B（下載已暫停）", "等 Hy-MT2-7B（下載失敗，看安裝進度）",
                      "等 Qwen3-ASR-1.7B（等網路恢復）"}
            check(f"{tag} 佇列顯示等模型和在等什麼", want_q <= texts and sum(x["state"] == "等模型" for x in q) == 4
                  and any(x["state"] == "等前一步" for x in q) and any(x["kind"] == "建立字典 · 英文" for x in q), got=q)
            lib = await cdp.js("[...document.querySelectorAll('#media-list .item-meta')].map((m) => m.textContent)")
            check(f"{tag} 播放列表顯示等模型", sum("等模型（轉字幕）" in t for t in lib) >= 2, got=lib)
            await shot(f"setup_queue_{dev}")
            if dev == "phone":
                await cdp.js("document.querySelector('#queue .job-stage .link-btn').scrollIntoView({block: 'center'})")
                await asyncio.sleep(0.3)
                await shot(f"setup_queue2_{dev}")
            await cdp.js("document.querySelector('#queue [data-act=\"setup\"]').click()")
            await asyncio.sleep(0.5)
            check(f"{tag} 佇列的「看安裝進度」打開進度頁", (await view())["open"])
            await cdp.js("closeSetup()")

            # 7. 新增影片：下載中的模型也能選，選到時說明會先排隊
            await cdp.js("openAddDialog('add')")
            await asyncio.sleep(0.8)
            await cdp.js("document.querySelector('#lang-seg [data-v=ja]').click()")
            await asyncio.sleep(0.5)
            add = await cdp.js("""({engine: [...document.querySelector('#adv-engine').options].map((o) => o.textContent),
                translator: [...document.querySelector('#adv-translator').options].map((o) => o.textContent),
                note: document.querySelector('#add-pending').hidden ? '' : document.querySelector('#add-pending').textContent})""")
            check(f"{tag} 新增影片：下載中的模型標（下載中），並說明會排隊",
                  "Qwen3-ASR-1.7B（預設，下載中）" in add["engine"] and "Hy-MT2-7B（預設，下載中）" in add["translator"]
                  and add["note"] == "模型還在下載，影片會先排隊，裝好自動開始。", got=add)
            await shot(f"setup_add_{dev}")
            await cdp.js("document.querySelector('#dlg-add').close()")

            # 8. 設定頁：自動安裝卡片、確認框、計畫變了、沒事做、不能裝；字典按鈕
            plan = {"signature": "sig1", "gpu": GPU8, "choice": st8["choice"], "block": None, "disk": st8["disk"],
                    "items": [{"key": "model:qwen-asr", "label": "Qwen3-ASR-1.7B", "installed": True, "need_bytes": 0},
                              {"key": "model:aligner", "label": "Qwen3-ForcedAligner-0.6B", "installed": True, "need_bytes": 0},
                              {"key": "model:tsqyomi", "label": "tsqyomi", "installed": True, "need_bytes": 0},
                              {"key": "model:hymt2-7b:Q6_K", "label": "Hy-MT2-7B Q6_K", "installed": False, "need_bytes": 6164482720},
                              {"key": "dict:ja", "label": "日文字典", "installed": True, "need_bytes": 0},
                              {"key": "dict:en", "label": "英文字典", "installed": True, "need_bytes": 0}],
                    "download_bytes": 6164482720,
                    "changes": [{"key": "model_variants", "text": "翻譯改用 Q6_K 版"}, {"key": "llm_params", "text": "翻譯改成一次翻比較少行"}]}
            plan2 = json.loads(json.dumps(plan))
            plan2["signature"] = "sig2"
            plan2["gpu"] = GPU12
            plan2["items"][3] = {"key": "model:hymt2-7b:Q8_0", "label": "Hy-MT2-7B Q8_0", "installed": False, "need_bytes": 7981928896}
            plan2["changes"] = [{"key": "model_variants", "text": "翻譯改用 Q8_0 版"}]
            done_dismissed = dict(states["done"][0], dismissed=True, reason="existing")
            dict_state = {"ja": {"ready": False, "entries": None, "size_mb": 0}, "en": {"job": {"id": "mockjob00007", "status": "running", "progress": 0.43, "stage": "build"}},
                          "zh": {"models": False, "dict": False}}
            await load_mock(cdp, APP, _mock(json.loads(json.dumps(st8 | {"dismissed": True})), dicts=dict_state,
                                            jobs=[fake_jobs[6]]))
            await cdp.js("showView('settings')")
            await asyncio.sleep(1.5)
            card = await cdp.js("""({hidden: document.querySelector('#setup-card').hidden, status: document.querySelector('#setup-card-status').textContent,
                disabled: document.querySelector('#btn-setup-install').disabled})""")
            check(f"{tag} 設定頁：安裝中顯示進度、按鈕停用", not card["hidden"] and card["status"] == f"安裝中 {st8['totals']['percent']}%，看進度"
                  and card["disabled"], got=card)
            await cdp.js("document.querySelector('#setup-card').scrollIntoView({block: 'start'})")
            await asyncio.sleep(0.3)
            await shot(f"setup_settings_installing_{dev}")
            await cdp.js("document.querySelector('#setup-card-status [data-setup-open]').click()")
            await asyncio.sleep(0.5)
            check(f"{tag} 設定頁「看進度」打開進度頁", (await view())["open"])
            await cdp.js("closeSetup()")

            # 字典卡片
            await cdp.js("document.querySelector('#dict-card').scrollIntoView({block: 'start'})")
            await asyncio.sleep(0.5)
            rows = await cdp.js("[...document.querySelectorAll('#dict-status .dict-row')].map((r) => r.innerText.replace(/\\s+/g, ' ').trim())")
            check(f"{tag} 字典卡片：沒建立、建立中、加入中文",
                  rows == ["日文 還沒建立（下載 14 MB，約 1 分鐘） 建立", "英文 建立中 43%", "中文 還沒安裝 加入中文（下載 0.8 GB）"], got=rows)
            await shot(f"setup_dicts_{dev}")
            await cdp.js("window.__calls.length = 0; document.querySelector('#dict-status [data-zh-add]').click()")
            await asyncio.sleep(1.0)
            await cdp.js("document.querySelector('#dict-status [data-dict-build=\"ja\"]').click()")
            await asyncio.sleep(1.0)
            calls = await cdp.js("window.__calls")
            check(f"{tag} 字典按鈕：加入中文依序排中文辭典和 CKIP，建立日文字典",
                  calls == ['POST /api/dicts/zh/build {"force_download":false}', "POST /api/models/ckip/download",
                            'POST /api/dicts/ja/build {"force_download":false}'], calls=calls)

            # 已經建好的字典：重建要先確認
            ready_dicts = {"ja": {"ready": True, "source_date": "2026-09-16", "entries": 218785, "size_mb": 87},
                           "en": {"ready": True, "nlp": True, "entries": 2714667, "size_mb": 235},
                           "zh": {"models": True, "dict": True, "size_mb": 23}}
            await load_mock(cdp, APP, _mock(done_dismissed, dicts=ready_dicts))
            await cdp.js("showView('settings')")
            await asyncio.sleep(1.5)
            # 手機上日文那列的文字比較長：〔重建〕還是要跟文字在同一列，不能自己掉到下一行（2026-09-17 擁有者保護測試發現）
            await cdp.js("document.querySelector('#dict-card').scrollIntoView({block: 'start'})")
            await asyncio.sleep(0.4)
            lay = await cdp.js("""[...document.querySelectorAll('#dict-status .dict-row')].map((r) => {
              const b = r.querySelector('.btn'), s = r.querySelector(':scope > span');
              if (!b || !s) return null;
              const br = b.getBoundingClientRect(), sr = s.getBoundingClientRect(), rr = r.getBoundingClientRect();
              return {text: s.textContent, same: br.top < sr.bottom && br.bottom > sr.top, inside: br.right <= rr.right + 1};
            }).filter(Boolean)""")
            check(f"{tag} 字典列的〔重建〕跟文字在同一列", len(lay) == 2 and all(x["same"] and x["inside"] for x in lay), got=lay)
            await shot(f"setup_dicts_ready_{dev}")
            rebuild = await cdp.js("!!document.querySelector('#dict-status [data-dict-build=\"ja\"][data-force]')")
            if rebuild:
                await cdp.js("document.querySelector('#dict-status [data-dict-build=\"ja\"][data-force]').click()")
                await asyncio.sleep(0.4)
                conf = await cdp.js("({open: document.querySelector('#dlg-confirm').open, title: document.querySelector('#confirm-title').textContent,"
                                    " text: document.querySelector('#confirm-text').textContent})")
                check(f"{tag} 重建字典先確認", conf == {"open": True, "title": "重建日文字典", "text": "重新下載來源再建一次，要幾分鐘。"}, got=conf)
                await cdp.js("document.querySelector('#confirm-cancel').click()")
            else:
                check(f"{tag} 重建字典先確認（測試伺服器上日文字典要是建好的，沒有重建按鈕）", False,
                      rows=await cdp.js("document.querySelector('#dict-status').innerText"))

            # 重新偵測並安裝建議模型
            await cdp.js(f"window.__MOCK.plan = {json.dumps(plan)};"
                         f" window.__MOCK.install = [{{status: 409, body: {{detail: '顯示卡或檔案有變，請再看一次', code: 'plan_changed', plan: {json.dumps(plan2)}}}}},"
                         " {body: {ok: true, nothing: true}}];")
            await cdp.js("document.querySelector('#setup-card').scrollIntoView({block: 'start'}); document.querySelector('#btn-setup-install').click()")
            await wait_js(cdp, "document.querySelector('#dlg-confirm').open", 5)
            conf = await cdp.js("({title: document.querySelector('#confirm-title').textContent, text: document.querySelector('#confirm-text').textContent,"
                                " items: [...document.querySelectorAll('#confirm-list li')].map((li) => li.textContent),"
                                " ok: document.querySelector('#confirm-ok').textContent})")
            check(f"{tag} 確認框內容", conf == {"title": "安裝建議模型", "text": "NVIDIA GeForce RTX 4060 · 8 GB",
                                              "items": ["下載 Hy-MT2-7B Q6_K（5.7 GB）", "翻譯改用 Q6_K 版", "翻譯改成一次翻比較少行",
                                                        "已經有：Qwen3-ASR-1.7B、Qwen3-ForcedAligner-0.6B、tsqyomi、日文字典、英文字典"],
                                              "ok": "開始安裝"}, got=conf)
            await shot(f"setup_confirm_{dev}")
            await cdp.js("document.querySelector('#confirm-ok').click()")
            await asyncio.sleep(0.8)
            conf2 = await cdp.js("({open: document.querySelector('#dlg-confirm').open, text: document.querySelector('#confirm-text').textContent,"
                                 " items: [...document.querySelectorAll('#confirm-list li')].map((li) => li.textContent),"
                                 " toast: document.querySelector('#toast').textContent})")
            check(f"{tag} 計畫變了：提示並重新打開確認框", conf2["open"] and conf2["text"] == "NVIDIA GeForce RTX 4070 · 12 GB"
                  and conf2["items"][0] == "下載 Hy-MT2-7B Q8_0（7.4 GB）" and conf2["toast"] == "顯示卡或檔案有變，請再看一次。", got=conf2)
            await cdp.js("document.querySelector('#confirm-ok').click()")
            await asyncio.sleep(0.8)
            calls = await cdp.js("window.__calls")
            check(f"{tag} 開始安裝帶 signature；沒事做時提示", calls[-2:] == ['POST /api/setup/install {"signature":"sig1"}', 'POST /api/setup/install {"signature":"sig2"}']
                  and await cdp.js("document.querySelector('#toast').textContent") == "建議的模型和字典都裝好了", calls=calls)
            await asyncio.sleep(3.2)
            nothing = json.loads(json.dumps(plan))
            for it in nothing["items"]:
                it["installed"] = True
            nothing["changes"] = []
            await cdp.js(f"window.__MOCK.plan = {json.dumps(nothing)}; window.__calls.length = 0; document.querySelector('#btn-setup-install').click()")
            await asyncio.sleep(0.8)
            check(f"{tag} 都裝好了：不開確認框、提示", not await cdp.js("document.querySelector('#dlg-confirm').open")
                  and await cdp.js("document.querySelector('#toast').textContent") == "建議的模型和字典都裝好了"
                  and not await cdp.js("window.__calls"))
            blocked_plan = dict(nothing, block={"code": "vram_small", "values": {"total_mb": 6144}}, gpu=_gpu("NVIDIA GeForce RTX 2060", 6, 6144, 5950))
            await asyncio.sleep(3.2)
            await cdp.js(f"window.__MOCK.plan = {json.dumps(blocked_plan)}; document.querySelector('#btn-setup-install').click()")
            await wait_js(cdp, "document.querySelector('#dlg-confirm').open", 5)
            conf3 = await cdp.js("({title: document.querySelector('#confirm-title').textContent, text: document.querySelector('#confirm-text').textContent,"
                                 " cancel: !document.querySelector('#confirm-cancel').hidden, ok: document.querySelector('#confirm-ok').textContent})")
            check(f"{tag} 不能裝：單按鈕說明", conf3 == {"title": "沒辦法自動安裝", "text": "顯存只有 6 GB，模型放不下。需要顯存 8 GB 以上的 NVIDIA 顯示卡。",
                                                   "cancel": False, "ok": "知道了"}, got=conf3)
            await shot(f"setup_confirm_blocked_{dev}")
            await cdp.js("document.querySelector('#confirm-ok').click()")
            await cdp.js("showView('player')")

            # 9. 單字頁：日文字典還沒建立 → 建立按鈕；建立中 → 說明
            await load_mock(cdp, APP, _mock(done_dismissed, vocab={"dict_ready": False, "items": [], "pending": 0, "missing_zh": 0}))
            await cdp.js("S.settings.vocabLang = 'ja'; showView('vocab')")
            await asyncio.sleep(1.5)
            note = await cdp.js("document.querySelector('#vb-note').hidden ? '' : document.querySelector('#vb-note').innerText.replace(/\\s+/g, ' ').trim()")
            check(f"{tag} 單字頁：日文字典還沒建立，有建立按鈕", note == "日文字典還沒建立。 建立", got=note)
            await shot(f"setup_vocab_{dev}")
            await cdp.js("window.__calls.length = 0; document.querySelector('#vb-note [data-dict-build]').click()")
            await asyncio.sleep(0.6)
            check(f"{tag} 單字頁建立按鈕排進佇列", await cdp.js("window.__calls") == ['POST /api/dicts/ja/build {"force_download":false}'])
            await load_mock(cdp, APP, _mock(done_dismissed, vocab={"dict_ready": False, "items": [], "pending": 0, "missing_zh": 0},
                                            jobs=[job("mockjob00008", "dict", "running", progress=0.3, params={"lang": "ja"})]))
            await cdp.js("S.settings.vocabLang = 'ja'; showView('vocab')")
            await asyncio.sleep(1.5)
            note = await cdp.js("document.querySelector('#vb-note').hidden ? '' : document.querySelector('#vb-note').innerText")
            check(f"{tag} 單字頁：建立中", note == "日文字典建立中，好了會自動更新。", got=note)
            await cdp.js("showView('player')")

        # 10. 進度頁開著時，後面的影片不會自己播，按空白鍵也不會播
        if media_id:
            await cdp.send("Emulation.setTouchEmulationEnabled", enabled=False)
            await cdp.send("Emulation.setDeviceMetricsOverride", width=1600, height=900, deviceScaleFactor=1, mobile=False)
            st = _mock(json.loads(json.dumps(states["8gb"][0])), local={"vs:last": media_id})
            await load_mock(cdp, media_url, st)
            loaded = await wait_js(cdp, f"S.current === '{media_id}' && document.querySelector('#video').readyState >= 1", 15)
            await asyncio.sleep(1.5)
            before = await cdp.js("({open: SU.open, paused: video.paused, t: video.currentTime})")
            await key(cdp, " ", "Space", 32, " ")
            await asyncio.sleep(1.0)
            after = await cdp.js("({open: SU.open, paused: video.paused, t: video.currentTime})")
            check("進度頁開著時影片不會自己播，按空白鍵也不會播", loaded and before["open"] and before["paused"] and after["paused"],
                  loaded=loaded, before=before, after=after)
            await cdp.js("closeSetup()")
            await asyncio.sleep(0.3)
            await key(cdp, " ", "Space", 32, " ")
            await asyncio.sleep(1.0)
            check("關掉進度頁後空白鍵照常播放", not await cdp.js("video.paused"))
            await cdp.js("video.pause()")

        check("沒有 JavaScript 錯誤", not cdp.errors, errors=cdp.errors)
        listener.cancel()
    return results


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = dict(a[2:].split("=", 1) if "=" in a else (a[2:], "1") for a in sys.argv[1:] if a.startswith("--"))
    for a in sys.argv[1:]:
        if a.startswith("--ja="):
            LOOKUP["ja"] = a[5:]
        elif a.startswith("--en="):
            LOOKUP["en"] = a[5:]
    if not args:
        raise SystemExit(__doc__)
    out_dir = Path(args[0])
    media_id = args[1] if len(args) > 1 else ""
    at = float(args[2]) if len(args) > 2 else 60
    chrome_exe = find_chrome()
    out_dir.mkdir(parents=True, exist_ok=True)
    profile = tempfile.mkdtemp(prefix="vs-chrome-")
    chrome = subprocess.Popen([
        chrome_exe, "--headless=new", f"--remote-debugging-port={PORT}", f"--user-data-dir={profile}",
        "--no-first-run", "--mute-audio", "--autoplay-policy=no-user-gesture-required", "about:blank",
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(40):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json/version", timeout=1)
                break
            except Exception:
                time.sleep(0.25)
        if "setup" in opts:
            sys.stdout.reconfigure(encoding="utf-8")
            results = asyncio.run(setup_checks(out_dir, opts.get("media", ""), os.environ.get("VS_MEDIA_URL", APP)))
            (out_dir / "setup_report.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
            failed = [r for r in results if not r["ok"]]
            for r in failed:
                print(json.dumps(r, ensure_ascii=False))
            print(f"\n{len(results) - len(failed)}/{len(results)} 通過")
            return
        report = asyncio.run(run(out_dir, media_id, at))
        (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print("ok")
    finally:
        chrome.kill()
        # Chrome 的暫存設定檔每個約 110 MB，不刪會一直堆在 %TEMP%
        try:
            chrome.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass
        import shutil
        shutil.rmtree(profile, ignore_errors=True)


if __name__ == "__main__":
    main()
