"""實測字幕選擇：原文選單、只列這條原文翻譯的翻譯選單、字幕管理的「顯示」、舊設定轉換、交換上下位置、
切換字幕時 subbar 排版不跳動、新翻譯自動換上後有存。

用法：python -s tests/pair_check.py [media_id] [--zh=只有中文原文的影片id] [--out=截圖資料夾]
[media_id] 要有兩條以上原文字幕，前兩條各自有翻譯；不填就從 /api/state 挑第一部符合的影片。
--zh 不填時，挑一部有中文原文、還沒有翻譯的影片；找不到就跳過中文那段。
截圖預設存到系統暫存資料夾的 vs-pair-check。
環境變數 VS_UI_URL 可指定網址，預設 http://127.0.0.1:8765/；VS_CDP_PORT 指定 Chrome 除錯埠，預設 9380
Chrome 的位置見 tests/ui_check.py 的說明（可以用 VS_CHROME 指定）。
只切換顯示和瀏覽器裡的設定，不會按刪除、轉字幕、翻譯這類會改資料的按鈕。
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ui_check import CDP, find_chrome  # noqa: E402

PORT = int(os.environ.get("VS_CDP_PORT", "9380"))
APP = os.environ.get("VS_UI_URL", "http://127.0.0.1:8765/")

SEL = """(() => {
  const q = (s) => document.querySelector(s);
  const opts = (s) => [...q(s).options].map((o) => [o.value, o.textContent]);
  return {1: S.trackSel[1], 2: S.trackSel[2], tr: S.showTr,
    ui1: q('#sel-1').value, ui2: q('#sel-2').value, opts1: opts('#sel-1'), opts2: opts('#sel-2'),
    dis1: q('#sel-1').disabled, dis2: q('#sel-2').disabled,
    saved: JSON.parse(localStorage.getItem('vs:tracks:' + S.current) || 'null')};
})()"""

ORDER = "[...document.querySelector('#zone-inBottom').children].map((e) => e.id)"
# subbar 的排版：翻譯選單寬度、subbar 高度、「字幕外觀」按鈕位置、影片區高度
LAYOUT = """(() => {
  const r = (s) => document.querySelector(s).getBoundingClientRect();
  return {w2: Math.round(r('#sel-2').width), bar: Math.round(r('.subbar').height),
          look: [Math.round(r('#btn-look').x), Math.round(r('#btn-look').y)], player: Math.round(r('.player').height)};
})()"""
SWAP = "({dis: document.querySelector('#btn-swap').disabled, title: document.querySelector('#btn-swap').title, swap: S.settings.swapLines})"
# 攔截 /api/state，window.__addNew 為 true 時替原文多加一條翻譯，模擬翻譯做好（不動伺服器）
INTERCEPT = """(() => {
  const real = window.fetch;
  window.fetch = async (url, opts) => {
    const u = String(url);
    if (u.includes('/api/tracks/newtr0000009/')) return real(u.replace('newtr0000009', '%(TA)s'), opts);
    const res = await real(url, opts);
    if (!u.endsWith('/api/state')) return res;
    const data = await res.clone().json();
    const m = data.media.find((x) => x.id === '%(MID)s');
    if (m && window.__addNew && !m.tracks.some((t) => t.id === 'newtr0000009')) {
      const base = m.tracks.find((t) => t.id === '%(TA)s');
      m.tracks.push({...base, id: 'newtr0000009', created_at: base.created_at + 100000});
    }
    return new Response(JSON.stringify(data), {status: 200, headers: {'Content-Type': 'application/json'}});
  };
})()"""
RESULTS: list = []


async def wait_js(cdp: CDP, expr: str, timeout: float = 10.0) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        if await cdp.js(expr):
            return True
        await asyncio.sleep(0.2)
    return False


async def run(mid: str, zh: str, out: Path):
    state = json.load(urllib.request.urlopen(APP + "api/state"))
    m = next(x for x in state["media"] if x["id"] == mid)
    asr = [t["id"] for t in m["tracks"] if t["kind"] == "asr"]
    trs_of = {a: [t["id"] for t in m["tracks"] if t["kind"] == "translation" and t["source_track_id"] == a] for a in asr}
    newest = {a: (trs_of[a][-1] if trs_of[a] else "") for a in asr}
    A, B = asr[0], asr[1]
    TA, TB = newest[A], newest[B]
    assert TA and TB, "兩條原文都要有翻譯"

    with urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{PORT}/json/new?about:blank", method="PUT")) as r:
        target = json.load(r)
    import websockets
    async with websockets.connect(target["webSocketDebuggerUrl"], max_size=50_000_000) as ws:
        cdp = CDP(ws)
        listener = asyncio.create_task(cdp.listen())
        await cdp.send("Runtime.enable")
        await cdp.send("Log.enable")
        await cdp.send("Page.enable")
        await cdp.send("Emulation.setDeviceMetricsOverride", width=1600, height=900, deviceScaleFactor=1, mobile=False)
        results = RESULTS

        def check(step, ok, **detail):
            results.append({"step": step, "ok": bool(ok), **detail})

        async def sel():
            return await cdp.js(SEL)

        async def choose(slot, value):
            await cdp.js(f"(() => {{ const s = document.querySelector('#sel-{slot}'); s.value = {json.dumps(value)};"
                         " s.dispatchEvent(new Event('change')); })()")
            await asyncio.sleep(1.0)
            return await sel()

        async def shot(name):
            res = await cdp.send("Page.captureScreenshot", format="png")
            (out / name).write_bytes(base64.b64decode(res["data"]))

        async def loaded(media):
            # 頁面剛開始載入時 app.js 還沒跑，先確認 S 存在，不然等待本身會被算成 JS 錯誤
            await wait_js(cdp, f"typeof S !== 'undefined' && S.current === '{media}' && S.trackSel[1] !== null", 15)
            await asyncio.sleep(2.0)

        async def seek_cue(n=8):
            await cdp.js(f"""(async () => {{ const c = S.cues[1][Math.min({n}, S.cues[1].length - 1)]; if (!c) return;
                video.pause(); video.currentTime = c.start + 0.2;
                await new Promise((r) => {{ video.addEventListener('seeked', r, {{once: true}}); setTimeout(r, 3000); }});
                updateSubs(true); }})()""")
            await asyncio.sleep(0.5)

        await cdp.send("Page.navigate", url=f"{APP}#m={mid}")
        await loaded(mid)

        # 1. 原文選單只列原文
        g = await sel()
        check("原文選單只列原文加不顯示", [v for v, _ in g["opts1"]] == [""] + asr and g["opts1"][0][1] == "不顯示",
              opts1=g["opts1"])

        # 2. 每條原文：翻譯選單只列它的翻譯，翻譯自動換成它最新的那條
        for a in asr:
            g = await choose(1, a)
            check(f"選原文 {a}：翻譯換成它最新的翻譯、選單只列它的翻譯",
                  g["1"] == a and g["2"] == newest[a] and g["ui2"] == newest[a] and not g["dis2"]
                  and [v for v, _ in g["opts2"]] == [""] + trs_of[a],
                  got={k: g[k] for k in ("1", "2", "ui2", "opts2", "dis2")})

        # 3. 逐字稿配對
        g = await choose(1, A)
        tr_info = await cdp.js("({primary: S.primary, a: document.querySelectorAll('#transcript p.a').length,"
                               " b: document.querySelectorAll('#transcript p.b').length,"
                               " cue: document.querySelector('#transcript p.a')?.dataset.cue})")
        check("逐字稿原文配翻譯", tr_info["primary"] == 1 and tr_info["a"] > 0 and tr_info["b"] > 0
              and str(tr_info["cue"]).startswith("1:"), got=tr_info)
        await seek_cue()
        await shot("1_player.png")

        # 4. 翻譯選「不顯示」後換原文，翻譯維持不顯示
        g = await choose(2, "")
        check("翻譯選不顯示", g["2"] == "" and g["tr"] is False and not g["dis2"], got=g)
        g = await choose(1, B)
        b_count = await cdp.js("document.querySelectorAll('#transcript p.b').length")
        check("翻譯不顯示時換原文，翻譯仍不顯示", g["1"] == B and g["2"] == "" and g["tr"] is False
              and [v for v, _ in g["opts2"]] == [""] + trs_of[B] and b_count == 0,
              got={k: g[k] for k in ("1", "2", "tr", "opts2")}, transcript_b=b_count)

        # 5. 原文不顯示：翻譯選單停用
        g = await choose(1, "")
        check("原文不顯示時翻譯選單停用", g["1"] == "" and g["2"] == "" and g["dis2"] and g["opts2"] == [["", "先選原文字幕"]],
              got={k: g[k] for k in ("1", "2", "dis2", "opts2")})
        await shot("2_no_original.png")
        g = await choose(1, A)
        check("再選原文：翻譯照原本的不顯示", g["1"] == A and g["2"] == "", got={k: g[k] for k in ("1", "2", "tr")})
        g = await choose(2, TA)
        g = await choose(1, "")
        g = await choose(1, B)
        check("翻譯有顯示時，原文先不顯示再選別條，翻譯跟著帶出來", g["1"] == B and g["2"] == TB and g["tr"] is True,
              got={k: g[k] for k in ("1", "2", "tr")})

        # 5b. 換原文、翻譯選不顯示時，翻譯選單寬度和 subbar 排版都不變（左右側欄都開著）
        # 預設外觀是右側欄收起、字幕在影片外，這裡明確設成兩側都開、字幕在影片內，量的是 subbar 本身
        await cdp.js("togglePanel('left', false); togglePanel('right', false); setSetting('place1', 'inBottom'); setSetting('place2', 'inBottom')")
        for w, h in ((1600, 900), (1280, 800)):
            await cdp.send("Emulation.setDeviceMetricsOverride", width=w, height=h, deviceScaleFactor=1, mobile=False)
            await asyncio.sleep(0.5)
            states = {}
            for name, slot, value in (("原文 A", 1, A), ("原文不顯示", 1, ""), ("原文 B", 1, B), ("翻譯不顯示", 2, ""), ("翻譯 TB", 2, TB)):
                await choose(slot, value)
                states[name] = await cdp.js(LAYOUT)
            panels = await cdp.js("!document.body.classList.contains('hide-left') && !document.body.classList.contains('hide-right')")
            check(f"{w}×{h} 切換字幕時 subbar 排版和影片大小不變", panels and len({json.dumps(v) for v in states.values()}) == 1,
                  got=states)
            if w == 1280:
                await shot("2b_layout_1280.png")
        await cdp.send("Emulation.setDeviceMetricsOverride", width=1600, height=900, deviceScaleFactor=1, mobile=False)
        await asyncio.sleep(0.5)

        # 6. 字幕管理
        await cdp.js("document.querySelector('#btn-files').click()")
        await asyncio.sleep(0.8)
        mgr = await cdp.js("""(() => {
          const rows = [...document.querySelectorAll('#track-table .track-item')];
          return {rows: rows.map((r) => ({name: r.querySelector('.name').textContent, child: r.classList.contains('child'),
                    sel: r.classList.contains('sel'), show: [...r.querySelectorAll('[data-show]')].map((b) => [b.dataset.show, b.textContent, b.classList.contains('on')]),
                    health: !!r.querySelector('[data-health]'), del: !!r.querySelector('[data-del]')})),
                  link: document.querySelector('#tracks-bilingual').getAttribute('href')};
        })()""")
        shows = [s for r in mgr["rows"] for s in r["show"]]
        on = sorted(s[0] for s in shows if s[2])
        check("字幕管理每列一顆顯示按鈕，顯示中的列有標示",
              len(shows) == len(m["tracks"]) and all(len(r["show"]) == 1 for r in mgr["rows"])
              and on == sorted([B, TB]) and all(s[1] == ("顯示中" if s[2] else "顯示") for s in shows)
              and sum(r["sel"] for r in mgr["rows"]) == 2,
              got=mgr["rows"])
        check("原文列有檢查時間軸按鈕", sum(r["health"] for r in mgr["rows"]) == len(asr))
        check("雙語 SRT 連結是目前的原文＋翻譯", f"top={B}&bottom={TB}" in (mgr["link"] or ""), link=mgr["link"])
        await shot("3_tracks_dialog.png")

        await cdp.js(f"document.querySelector('#track-table [data-show=\"{TA}\"]').click()")
        await asyncio.sleep(1.0)
        g = await sel()
        on = await cdp.js("[...document.querySelectorAll('#track-table [data-show].on')].map((b) => b.dataset.show)")
        check("翻譯列按顯示：原文切到它的來源", g["1"] == A and g["2"] == TA and g["ui1"] == A and g["ui2"] == TA
              and sorted(on) == sorted([A, TA]), got={k: g[k] for k in ("1", "2", "ui1", "ui2")}, on=on)
        await cdp.js(f"document.querySelector('#track-table [data-show=\"{B}\"]').click()")
        await asyncio.sleep(1.0)
        g = await sel()
        check("原文列按顯示：翻譯跟著帶", g["1"] == B and g["2"] == TB, got={k: g[k] for k in ("1", "2")})
        await cdp.js(f"document.querySelector('#track-table [data-show=\"{B}\"]').click()")
        await asyncio.sleep(0.6)
        g = await sel()
        check("按顯示中的原文不會改變", g["1"] == B and g["2"] == TB, got={k: g[k] for k in ("1", "2")})

        # 孤兒翻譯（來源已刪除）：選單不列，字幕管理照樣顯示、可以刪除，但沒有顯示按鈕。只在頁面裡模擬，不動伺服器
        orphan = await cdp.js("""(() => {
          const m = mediaById(S.current);
          const fake = {id: 'orphan0test0', kind: 'translation', lang: 'zh-TW', model: 'Hy-MT2-7B', source_track_id: 'gone00000000',
                        cue_count: 3, created_at: Date.now() / 1000, mode: 'line'};
          m.tracks.push(fake);
          syncTracks(m); renderTrackManager();
          const row = [...document.querySelectorAll('#track-table .track-item')].find((r) => r.querySelector('[data-del="orphan0test0"]'));
          const res = {row: !!row, show: row ? !!row.querySelector('[data-show]') : null,
                       sub: row ? row.querySelector('.sub').textContent : '',
                       inSel: [...document.querySelectorAll('#sel-1 option, #sel-2 option')].some((o) => o.value === fake.id),
                       sel: [S.trackSel[1], S.trackSel[2]]};
          m.tracks.splice(m.tracks.indexOf(fake), 1);
          syncTracks(m); renderTrackManager();
          return res;
        })()""")
        check("孤兒翻譯：選單不列，字幕管理有列、可刪除、沒有顯示按鈕",
              orphan["row"] and orphan["show"] is False and not orphan["inSel"] and orphan["sel"] == [B, TB], got=orphan)
        await cdp.js("document.querySelector('#dlg-tracks').close()")

        # 新產生的字幕（模擬）：在頁面裡塞一條假的字幕軌再同步，讀字幕的 API 暫時換成回傳空陣列，不打伺服器
        FRESH = """((kind, source) => {
          const m = mediaById(S.current);
          const fake = {id: 'fresh0test00', kind, lang: kind === 'asr' ? 'ja' : 'zh-TW', model: 'Hy-MT2-7B',
                        source_track_id: source, cue_count: 3, created_at: Date.now() / 1000, mode: 'line'};
          const realApi = api;
          api = async () => [];
          m.tracks.push(fake);
          syncTracks(m);
          const res = {1: S.trackSel[1], 2: S.trackSel[2], tr: S.showTr, saved: store.get('tracks:' + S.current, null),
                       opts2: [...document.querySelectorAll('#sel-2 option')].map((o) => o.value)};
          m.tracks.splice(m.tracks.indexOf(fake), 1);
          api = realApi;
          syncTracks(m);
          return res;
        })"""
        got = await cdp.js(f"{FRESH}('translation', '{B}')")
        check("目前原文的新翻譯做好：自動換成新翻譯並存起來", got["1"] == B and got["2"] == "fresh0test00" and "fresh0test00" in got["opts2"]
              and (got["saved"] or {}).get("2") == "fresh0test00", got=got)
        got = await cdp.js(f"{FRESH}('translation', '{A}')")
        check("別條原文的新翻譯做好：不動", got["1"] == B and got["2"] == TB, got=got)
        await choose(2, "")
        got = await cdp.js(f"{FRESH}('translation', '{B}')")
        check("翻譯選不顯示時新翻譯做好：不會自己打開", got["1"] == B and got["2"] == "" and got["tr"] is False, got=got)
        got = await cdp.js(f"{FRESH}('asr', null)")
        check("原文有選時新原文做好：不換原文", got["1"] == B and got["2"] == "", got=got)
        await choose(2, TB)
        await choose(1, "")
        got = await cdp.js(f"{FRESH}('asr', null)")
        check("原文沒選時新原文做好：自動選上", got["1"] == "fresh0test00" and got["2"] == "" and got["tr"] is True, got=got)
        g = await choose(1, B)
        check("模擬完回到原本的選擇", g["1"] == B and g["2"] == TB, got={k: g[k] for k in ("1", "2", "tr")})

        # 6b. 新翻譯自動換上後重新整理，還是新翻譯；頁面關著的時候翻譯做好，重新開啟也換上
        CUR = f"({{sel: [S.trackSel[1], S.trackSel[2]], ui2: document.querySelector('#sel-2').value, saved: localStorage.getItem('vs:tracks:{mid}')}})"
        sid = (await cdp.send("Page.addScriptToEvaluateOnNewDocument", source=INTERCEPT % {"TA": TA, "MID": mid}))["identifier"]
        await cdp.send("Page.reload")
        await loaded(mid)
        await choose(1, A)
        await choose(2, TA)
        await cdp.js("window.__addNew = true; poll()")
        await wait_js(cdp, "S.trackSel[2] === 'newtr0000009'", 8)
        await asyncio.sleep(1.0)
        live = await cdp.js(CUR)
        sid2 = (await cdp.send("Page.addScriptToEvaluateOnNewDocument", source="window.__addNew = true;"))["identifier"]
        await cdp.send("Page.reload")
        await loaded(mid)
        after = await cdp.js(CUR)
        check("新翻譯自動換上時有存，重新整理後還是新翻譯",
              live["sel"] == [A, "newtr0000009"] and json.loads(live["saved"] or "{}").get("2") == "newtr0000009"
              and after["sel"] == [A, "newtr0000009"] and after["ui2"] == "newtr0000009", live=live, after=after)
        await cdp.js(f"store.set('tracks:{mid}', {{1: '{A}', 2: '{TA}', tr: true}});"
                     f" store.set('known:{mid}', mediaById('{mid}').tracks.map((t) => t.id).filter((x) => x !== 'newtr0000009'))")
        await cdp.send("Page.reload")
        await loaded(mid)
        reopen = await cdp.js(CUR)
        check("頁面關著時翻譯做好，重新開啟換成新翻譯", reopen["sel"] == [A, "newtr0000009"], got=reopen)
        await cdp.send("Page.removeScriptToEvaluateOnNewDocument", identifier=sid)
        await cdp.send("Page.removeScriptToEvaluateOnNewDocument", identifier=sid2)
        await cdp.send("Page.reload")
        await loaded(mid)
        gone = await cdp.js(CUR)
        check("存的翻譯不見了就用這條原文最新的翻譯", gone["sel"] == [A, TA], got=gone)
        await choose(1, B)
        g = await choose(2, TB)
        check("回到原本的選擇", g["1"] == B and g["2"] == TB, got={k: g[k] for k in ("1", "2", "tr")})

        # 7. 翻譯對話框預設選目前原文（目前原文已經有翻譯，會先跳確認，按確定才開對話框；不等 Promise，免得卡在確認框）
        await cdp.js("openTranslateDialog(); 0")
        await asyncio.sleep(0.4)
        if await cdp.js("document.querySelector('#dlg-confirm').open"):
            await cdp.js("document.querySelector('#confirm-ok').click()")
            await asyncio.sleep(0.4)
        src = await cdp.js("document.querySelector('#tr-source').value")
        await cdp.js("document.querySelector('#dlg-translate').close()")
        check("翻譯對話框預設選目前原文", src == B, got=src)

        # 8. 查字只查原文、改讀音用原文
        await wait_js(cdp, "!!WL._v.vocab[S.trackSel[1]]", 20)
        await seek_cue()
        look = await cdp.js("""({active: WL.active(), vocab: Object.keys(WL._v.vocab),
            w1: document.querySelectorAll('#line1 .w').length, w2: document.querySelectorAll('#line2 .w').length,
            wb: document.querySelectorAll('#transcript .b .w').length, line2: document.querySelector('#line2 span').textContent,
            ruby: (() => { const r = document.querySelector('#transcript ruby'); if (!r) return null;
                           const c = rubyContext(r); return c && {slot: c.slot, track: c.trackId}; })()})""")
        check("查字只載入原文的單字、翻譯行沒有可查的詞",
              all(k in asr for k in look["vocab"]) and look["w2"] == 0 and look["wb"] == 0
              and (look["w1"] > 0 or not look["active"]) and look["line2"], got=look)
        check("改讀音對應原文字幕", look["ruby"] is None or look["ruby"] == {"slot": 1, "track": B}, got=look["ruby"])

        # 9. 交換上下位置
        await cdp.js("setSetting('place1', 'inBottom'); setSetting('place2', 'inBottom'); setSetting('swapLines', false)")
        await asyncio.sleep(0.3)
        before = await cdp.js(f"({{order: {ORDER}, dis: document.querySelector('#btn-swap').disabled,"
                              " title: document.querySelector('#btn-swap').title})")
        await cdp.js("document.querySelector('#btn-swap').click()")
        await asyncio.sleep(0.4)
        after = await cdp.js(f"({{order: {ORDER}, title: document.querySelector('#btn-swap').title,"
                             " swap: S.settings.swapLines, saved: JSON.parse(localStorage.getItem('vs:settings')).swapLines,"
                             " line: [...document.querySelector('#zone-inBottom').children].map((e) => e.textContent.trim().slice(0, 20))})")
        check("交換上下位置：同一個位置時翻譯換到上面、狀態有存",
              before["order"] == ["line1", "line2"] and not before["dis"] and "原文在上" in before["title"]
              and after["order"] == ["line2", "line1"] and after["swap"] is True and after["saved"] is True
              and "翻譯在上" in after["title"], before=before, after=after)
        await seek_cue()
        await shot("4_swapped.png")
        await cdp.js("renderTrackManager()")
        link = await cdp.js("document.querySelector('#tracks-bilingual').getAttribute('href')")
        check("交換後雙語 SRT 翻譯在上", f"top={TB}&bottom={B}" in (link or ""), link=link)
        await cdp.js("setSetting('place2', 'outBottom')")
        await asyncio.sleep(0.3)
        split = await cdp.js("({dis: document.querySelector('#btn-swap').disabled, title: document.querySelector('#btn-swap').title,"
                             " out: [...document.querySelector('#zone-outBottom').children].map((e) => e.id)})")
        check("不同位置時交換按鈕停用", split["dis"] and "不同位置" in split["title"] and split["out"] == ["line2"], got=split)
        await cdp.js("setSetting('place2', 'inBottom')")
        await cdp.send("Page.reload")
        await loaded(mid)
        g = await sel()
        rel = await cdp.js(f"({{order: {ORDER}, swap: S.settings.swapLines}})")
        check("重新整理後交換狀態和字幕選擇都還在", rel["order"] == ["line2", "line1"] and rel["swap"] is True
              and g["1"] == B and g["2"] == TB, got=rel, sel={k: g[k] for k in ("1", "2", "tr")})
        await cdp.js("document.querySelector('#btn-swap').click()")
        await asyncio.sleep(0.3)
        rel = await cdp.js(f"({{order: {ORDER}, swap: S.settings.swapLines}})")
        check("再按一次換回原文在上", rel["order"] == ["line1", "line2"] and rel["swap"] is False, got=rel)
        # 畫面上只有一行時停用，按了也不會改設定
        await choose(2, "")
        await cdp.js("document.querySelector('#btn-swap').click()")
        await asyncio.sleep(0.3)
        notr = await cdp.js(SWAP)
        await choose(2, TB)
        shown = await cdp.js(SWAP)
        await cdp.js("setSetting('hide2', true)")
        hidden = await cdp.js(SWAP)
        await cdp.js("setSetting('hide2', false)")
        check("翻譯不顯示或隱藏時交換按鈕停用，按了不改設定",
              notr["dis"] and "都顯示" in notr["title"] and notr["swap"] is False
              and not shown["dis"] and "原文在上" in shown["title"] and hidden["dis"] and "都顯示" in hidden["title"],
              notr=notr, shown=shown, hidden=hidden)

        # 10. 舊版存的選擇：頁面載入前寫進 localStorage，重新整理後要轉成新規則並存回去
        cases = [
            ("1 翻譯、2 原文", {"1": TB, "2": B}, {"1": B, "2": TB, "tr": True}),
            ("兩行都是原文", {"1": A, "2": B}, {"1": A, "2": TA, "tr": True}),
            ("1 原文、2 另一條原文的翻譯", {"1": B, "2": TA}, {"1": B, "2": TB, "tr": True}),
            ("翻譯選不顯示", {"1": A, "2": ""}, {"1": A, "2": "", "tr": False}),
            ("兩行都不顯示", {"1": "", "2": ""}, {"1": "", "2": "", "tr": False}),
            ("1 不顯示、2 原文", {"1": "", "2": B}, {"1": B, "2": "", "tr": False}),
            ("1 翻譯、2 不顯示", {"1": TA, "2": ""}, {"1": A, "2": TA, "tr": True}),
            ("字幕軌已刪除", {"1": "deleted00000", "2": "deleted00001"}, {"1": asr[-1], "2": newest[asr[-1]], "tr": True}),
            ("新格式：翻譯後來才做好", {"1": A, "2": "", "tr": True}, {"1": A, "2": TA, "tr": True}),
        ]
        for name, old, want in cases:
            script = f"localStorage.setItem('vs:tracks:{mid}', {json.dumps(json.dumps(old))});"
            sid = (await cdp.send("Page.addScriptToEvaluateOnNewDocument", source=script))["identifier"]
            await cdp.send("Page.reload")
            await loaded(mid)
            await cdp.send("Page.removeScriptToEvaluateOnNewDocument", identifier=sid)
            g = await sel()
            got = {"1": g["1"], "2": g["2"], "tr": g["tr"]}
            saved = g["saved"] or {}
            ok = got == want and g["ui1"] == want["1"] and g["ui2"] == want["2"]
            if "tr" not in old:  # 舊格式轉換後要存回新格式
                ok = ok and {k: saved.get(k) for k in ("1", "2", "tr")} == want
            extra = {}
            if want["1"] == "":
                ok = ok and g["dis2"] and g["opts2"] == [["", "先選原文字幕"]]
                extra["opts2"] = g["opts2"]
            check(f"舊設定轉換：{name}", ok, old=old, got=got, saved=saved, **extra)

        # 11. 只有中文原文的影片：翻譯選單停用
        if zh:
            zm = next(x for x in state["media"] if x["id"] == zh)
            zasr = [t["id"] for t in zm["tracks"] if t["kind"] == "asr"]
            await cdp.js(f"closeMedia(); openMedia('{zh}')")
            await loaded(zh)
            g = await sel()
            check("中文影片：原文選單只有原文、翻譯選單停用",
                  [v for v, _ in g["opts1"]] == [""] + zasr and g["1"] == zasr[-1] and g["2"] == ""
                  and g["dis2"] and g["opts2"] == [["", "這條字幕還沒有翻譯"]],
                  got={k: g[k] for k in ("1", "2", "opts1", "opts2", "dis2")})
            zswap = await cdp.js(SWAP)
            check("中文影片：沒有翻譯行，交換按鈕停用", zswap["dis"] and "都顯示" in zswap["title"], got=zswap)
            zlay = [await cdp.js(LAYOUT)]
            await seek_cue(3)
            await shot("5_zh_only.png")
            g = await choose(1, "")
            zlay.append(await cdp.js(LAYOUT))
            check("中文影片：原文不顯示時提示先選原文", g["dis2"] and g["opts2"] == [["", "先選原文字幕"]], got=g["opts2"])
            g = await choose(1, zasr[-1])
            zlay.append(await cdp.js(LAYOUT))
            check("中文影片：選回原文", g["1"] == zasr[-1] and g["2"] == "" and g["dis2"], got={k: g[k] for k in ("1", "2", "dis2")})
            check("中文影片：兩種提示文字切換時排版不變", len({json.dumps(v) for v in zlay}) == 1, got=zlay)

        check("沒有 JavaScript 錯誤", not cdp.errors, errors=cdp.errors)
        listener.cancel()
        return results


def pick_media(state: dict) -> str:
    """挑第一部「前兩條原文字幕都有翻譯」的影片。"""
    for m in state["media"]:
        asr = [t["id"] for t in m["tracks"] if t["kind"] == "asr"]
        if len(asr) < 2:
            continue
        if all(any(t["kind"] == "translation" and t["source_track_id"] == a for t in m["tracks"]) for a in asr[:2]):
            return m["id"]
    raise SystemExit("片庫裡找不到有兩條原文字幕、而且各自有翻譯的影片，請直接指定 media_id。")


def pick_zh(state: dict) -> str:
    """挑一部中文、有原文字幕、完全沒有翻譯的影片；找不到回傳空字串（跳過中文那段）。"""
    for m in state["media"]:
        kinds = [t["kind"] for t in m["tracks"]]
        if m.get("language") == "zh" and "asr" in kinds and "translation" not in kinds:
            return m["id"]
    return ""


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
    if not args or "zh" not in opts:
        try:
            state = json.load(urllib.request.urlopen(APP + "api/state", timeout=10))
        except OSError as e:
            raise SystemExit(f"連不到 {APP}（{e}）。請先開伺服器，或用 VS_UI_URL 指定網址。")
        if not args:
            args = [pick_media(state)]
        if "zh" not in opts:
            opts["zh"] = pick_zh(state)
        print(f"影片：{args[0]}，中文影片：{opts['zh'] or '（跳過）'}", flush=True)
    out = Path(opts.get("out") or Path(tempfile.gettempdir()) / "vs-pair-check")
    out.mkdir(parents=True, exist_ok=True)
    chrome_exe = find_chrome()
    profile = tempfile.mkdtemp(prefix="vs-pair-")
    chrome = subprocess.Popen([chrome_exe, "--headless=new", f"--remote-debugging-port={PORT}", f"--user-data-dir={profile}",
                               "--no-first-run", "--mute-audio", "about:blank"],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(40):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json/version", timeout=1)
                break
            except Exception:
                time.sleep(0.25)
        sys.stdout.reconfigure(encoding="utf-8")
        try:
            asyncio.run(run(args[0], opts.get("zh", ""), out))
        except Exception as e:  # 中途出錯也把已經跑完的結果印出來
            RESULTS.append({"step": f"測試中斷：{e!r}", "ok": False})
        results = RESULTS
        print(json.dumps(results, ensure_ascii=False, indent=1))
        failed = [r["step"] for r in results if not r["ok"]]
        print(f"\n{len(results) - len(failed)}/{len(results)} 通過" + (f"，失敗：{failed}" if failed else ""))
    finally:
        chrome.kill()
        chrome.wait()
        time.sleep(0.5)
        shutil.rmtree(profile, ignore_errors=True)


if __name__ == "__main__":
    main()
