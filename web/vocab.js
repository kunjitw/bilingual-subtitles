'use strict';

/* 查字（滑鼠移到字幕的字上顯示中文解釋）與單字頁
   用到 app.js 的全域：S、$、$$、esc、api、toast、video、saveSettings、saveValue、mediaById、openMedia、showView、
   updateSubs、renderTranscript、fmtTime、openRubyPop、rubyContext、titleParts、SET
   app.js 透過 window.WL 呼叫這裡（沒載入這個檔時，播放器照常運作，只是不能查字） */

(function () {
  const DEFAULTS = {
    lookupOn: true, lookupTrigger: 'hover', hoverDelay: 150, lookupPause: 'cueEnd', lookupPauseAsmr: 'off',
    vocabLang: '', vocabScope: '', vocabStatus: 'learn', vocabOther: false, vocabSort: 'count', vocabPageSize: 100,
    miniBox: null, miniLoop: false,
  };
  for (const [k, v] of Object.entries(DEFAULTS)) if (!(k in S.settings)) S.settings[k] = v;

  /* ---------- 中文標籤 ---------- */

  // Yomitan 活用名稱 → 中文（活用鏈由外而內，顯示時反過來接）
  // 鍵是 tools/yomitan/yomitan_ja_transforms.json 裡的活用名稱（出自 Yomitan，GPL-3.0-or-later，見 tools/yomitan/README.md），中文是本專案自己寫的
  const CHAIN_ZH = {
    '-た': '過去', '-ます': '禮貌', negative: '否定', '-ず': '否定（ず）', '-ぬ': '否定（ぬ）', '-ん': '否定（ん）', '-ざる': '否定（ざる）',
    passive: '被動', potential: '可能', 'potential or passive': '可能／被動', causative: '使役', 'short causative': '使役（短形）',
    imperative: '命令', volitional: '意志', 'volitional slang': '意志（口語）', '-まい': '否定意志／推測', continuative: '連用形',
    '-て': 'て形', '-ば': '假定（ば）', '-ゃ': '假定口語（ゃ）', '-ねば': '必須（ねば）', '-たら': '假定／之後（たら）', '-たり': '列舉（たり）',
    '-たい': '想要', '-そう': '看起來（そう）', '-すぎる': '過度', '-過ぎる': '過度', '-なさい': '命令（なさい）',
    '-しまう': '～てしまう', '-ちゃう': '～てしまう（ちゃう）', '-ちまう': '～てしまう（ちまう）', '-ちゃ': '～ては（ちゃ）',
    '-いる': '持續（ている）', '-おく': '預先（ておく）', '-くる': '～てくる', '-いく': '～ていく', '-みる': '試試看（てみる）',
    '-くれる': '別人為我（てくれる）', '-もらう': '請別人（てもらう）', '-あげる': '為別人（てあげる）', '-ある': '已經弄好（てある）',
    '-いただく': '請別人（ていただく）', '-くださる': '別人為我（てくださる）', '-く': '副詞形', '-さ': '名詞化（さ）', '-げ': '樣子（げ）', '-がる': '表現出（がる）',
    '-やがる': '貶意（やがる）', '-え': '口語音變', 'n-slang': '口語縮約（ん）', 'imperative negative slang': '禁止（口語）',
    '-んばかり': '幾乎要（んばかり）', '-んとする': '正要（んとする）', '-む': '意志（文言）', '-き': '文言連體（き）',
    'kansai-ben negative': '關西腔否定', 'kansai-ben -て': '關西腔て形', 'kansai-ben -た': '關西腔過去',
    'kansai-ben -たら': '關西腔たら', 'kansai-ben -たり': '關西腔たり', 'kansai-ben -く': '關西腔く',
    'kansai-ben adjective -て': '關西腔形容詞て形', 'kansai-ben adjective negative': '關西腔形容詞否定',
  };
  const POS_ZH = {
    n: '名詞', pn: '代名詞', 'adj-i': 'い形容詞', 'adj-ix': 'い形容詞', 'adj-na': 'な形容詞', 'adj-no': '名詞（の）',
    'adj-t': 'たる形容詞', 'adj-f': '連體修飾', 'adj-pn': '連體詞', adv: '副詞', 'adv-to': '副詞（と）', aux: '助動詞',
    'aux-v': '助動詞', 'aux-adj': '補助形容詞', conj: '接續詞', cop: '繫詞', ctr: '助數詞', exp: '慣用語', int: '感嘆詞',
    num: '數詞', pref: '接頭詞', suf: '接尾詞', prt: '助詞', vi: '自動詞', vt: '他動詞', vk: 'カ變動詞', vz: 'ザ變動詞',
    'n-suf': '名詞（接尾）', 'n-pref': '名詞（接頭）', 'n-adv': '副詞性名詞', 'n-t': '時間名詞', unc: '未分類',
  };
  const MISC_ZH = {
    col: '口語', sl: '俗語', 'net-sl': '網路用語', vulg: '粗俗', hon: '尊敬語', hum: '謙讓語', pol: '禮貌語', arch: '古語',
    dated: '較舊', obs: '過時', rare: '罕用', abbr: '縮寫', joc: '玩笑', derog: '貶義', fam: '親暱', male: '男性用語',
    fem: '女性用語', id: '慣用句', 'on-mim': '擬聲擬態', sens: '敏感', yoji: '四字成語', proverb: '諺語', poet: '詩語',
    form: '正式', chn: '兒語', euph: '委婉', uk: '通常寫假名',
  };
  const GRP_ZH = { name: '專有名詞', filler: '感嘆詞', number: '數字', affix: '接頭接尾', unknown: '辭典查無', function: '功能詞' };
  const UPOS_ZH = {
    NOUN: '名詞', VERB: '動詞', ADJ: '形容詞', ADV: '副詞', PRON: '代名詞', DET: '限定詞', ADP: '介系詞', AUX: '助動詞',
    CCONJ: '連接詞', SCONJ: '連接詞', PART: '小品詞', INTJ: '感嘆詞', PROPN: '專有名詞',
  };
  // 單字狀態：0 未標記、1 不會、2 已會、3 忽略
  const ST_LABEL = { 1: '不會', 2: '已會', 3: '忽略' };
  const ST_KEY = { 1: 'N', 2: 'X', 3: 'I' };
  const KEY_ST = { n: 1, x: 2, i: 3 };
  const POS_PREFIX = {
    VERB: ['v.', 'vt.', 'vi.'], AUX: ['v.', 'vt.', 'vi.', 'aux.'], NOUN: ['n.', 'pl.'], PROPN: ['n.'], ADJ: ['a.', 'adj.'],
    ADV: ['adv.', 'ad.'], ADP: ['prep.'], PRON: ['pron.'], CCONJ: ['conj.'], SCONJ: ['conj.'], INTJ: ['int.', 'interj.'],
    DET: ['det.', 'art.', 'a.'],
  };

  const posZh = (p) => POS_ZH[p] || (p.startsWith('v5') ? '五段動詞' : p.startsWith('v1') ? '一段動詞'
    : p.startsWith('vs') ? 'する動詞' : /^v[24]/.test(p) ? '古語動詞' : p);
  const chainText = (surface, base, chain) =>
    `${surface} ＝ ${[base, ...[...chain].reverse().map((t) => CHAIN_ZH[t] || t)].join(' ＋ ')}`;
  function sortLines(lines, upos) {
    const want = POS_PREFIX[upos] || [];
    const rank = (l) => (l.startsWith('[') ? 2 : 0) + (want.length && !want.some((p) => l.startsWith(p)) ? 1 : 0);
    return [...(lines || [])].sort((a, b) => rank(a) - rank(b));
  }
  // 例句：把目標詞用 <mark> 標起來，有假名資料就一起標上（高亮範圍會撐大到不切斷假名區塊）
  function sentHtml(text, spans, ruby) {
    text = String(text || '');
    const rb = (S.settings.furi !== 'off' && ruby ? ruby : [])
      .filter((r) => r[0] >= 0 && r[0] < r[1] && r[1] <= text.length).sort((a, b) => a[0] - b[0]);
    const seg = (a, b) => {
      let html = '', p = a;
      for (const [s, e, rt] of rb) {
        if (s < p || e > b) continue;
        html += esc(text.slice(p, s));
        html += rt ? `<ruby>${esc(text.slice(s, e))}<rt>${esc(rt)}</rt></ruby>` : esc(text.slice(s, e));
        p = e;
      }
      return html + esc(text.slice(p, b));
    };
    const hl = [];
    for (const span of [...(spans || [])].sort((a, b) => a[0] - b[0])) {
      let [s, e] = span;
      for (const [rs, re] of rb) if (rs < e && re > s) { s = Math.min(s, rs); e = Math.max(e, re); }
      const last = hl[hl.length - 1];
      if (last && s <= last[1]) last[1] = Math.max(last[1], e);
      else hl.push([s, e]);
    }
    let html = '', p = 0;
    for (const [s, e] of hl) {
      if (s < p || e > text.length) continue;
      html += seg(p, s) + `<mark>${seg(s, e)}</mark>`;
      p = e;
    }
    return html + seg(p, text.length);
  }
  const marked = (text, spans) => sentHtml(text, spans, null);

  /* ---------- 狀態 ---------- */

  const V = { vocab: {}, loading: {}, detail: new Map(), status: new Map(), glossDone: null };
  const LOOK = {
    el: null, timer: 0, hideTimer: 0, scanTimer: 0, pinned: false, inPanel: false, ctx: null, anchor: null,
    warmUntil: 0, pausedByLookup: false, over: false, overSlot: null,
  };
  const active = () => S.settings.lookupOn !== false;
  const stOf = (key) => V.status.get(key) || 0;
  const trackOf = (id) => (mediaById(S.current)?.tracks || []).find((t) => t.id === id);
  const lexOf = (trackId, key) => V.vocab[trackId]?.lex?.[key];

  const pop = document.createElement('div');
  pop.id = 'wpop';
  pop.hidden = true;
  $('#screen').appendChild(pop);

  function loadVocab(trackId, force = false) {
    if (!force && (V.vocab[trackId] || V.loading[trackId])) return V.loading[trackId];
    V.loading[trackId] = api('GET', `/api/tracks/${trackId}/vocab`)
      .then((v) => {
        V.vocab[trackId] = v;
        for (const [k, x] of Object.entries(v.lex || {})) V.status.set(k, x.st || 0);
        if (!pop.hidden) renderCard();
      })
      .catch(() => {})
      .finally(() => { delete V.loading[trackId]; });
    return V.loading[trackId];
  }

  function onCuesLoaded(slot, trackId) {
    if (JUMP.mid && JUMP.mid === S.current && Math.abs(video.currentTime - JUMP.at) < 0.01) {
      const at = leadTime(JUMP.t);
      if (at !== JUMP.at) {
        JUMP.at = at;
        video.currentTime = at;
        updateSubs(true);
      }
    }
    if (!trackId || !active()) return;
    const t = trackOf(trackId);
    if (t && t.kind === 'asr' && (t.lang === 'ja' || t.lang === 'en')) loadVocab(trackId);
  }

  /* ---------- 滑鼠 ---------- */

  function wordCtx(el) {
    const host = el.closest('[data-cue]');
    if (!host || !host.dataset.cue) return null;
    const [slot, i] = host.dataset.cue.split(':').map(Number);
    const cue = S.cues[slot]?.[i];
    const k = +el.dataset.k;
    const w = cue?.w?.[k];
    if (!w) return null;
    const trackId = S.trackSel[slot];
    const phIdx = el.dataset.ph != null ? +el.dataset.ph : -1;
    return {
      slot, i, k, cue, w, key: w[2], trackId, lang: trackOf(trackId)?.lang || '',
      surface: cue.text.slice(w[0], w[1]), ph: phIdx >= 0 ? cue.ph?.[phIdx] || null : null,
      inPanel: !!el.closest('#transcript'),
    };
  }

  function highlight(el, on) {
    if (!el) return;
    const host = el.closest('[data-cue]');
    const list = el.dataset.ph != null && host ? host.querySelectorAll(`.w[data-ph="${el.dataset.ph}"]`) : [el];
    list.forEach((x) => x.classList.toggle('hl', on));
  }
  const clearHighlights = () => $$('.w.hl').forEach((x) => x.classList.remove('hl'));

  function onMove(e) {
    // 手指滑過字幕或捲逐字稿不算停在字上：觸控時只用點一下（onClick）開釘選卡
    if (!active() || e.pointerType === 'touch') return;
    const line = e.target.closest('.sub-line');
    LOOK.over = !!e.target.closest('.sub-line > span, #transcript .a, #wpop');
    if (line) LOOK.overSlot = line.id === 'line2' ? 2 : 1;
    if (LOOK.pinned) return;
    const el = e.target.closest('.w');
    if (!el && e.shiftKey) scheduleScan(e);
    if (el === LOOK.el) return;
    clearTimeout(LOOK.timer);
    highlight(LOOK.el, false);
    LOOK.el = el;
    if (!el) {
      if (!e.target.closest('#wpop') && !(LOOK.ctx && LOOK.ctx.scan && e.shiftKey)) scheduleHide();
      return;
    }
    clearTimeout(LOOK.hideTimer);
    clearTimeout(LOOK.scanTimer);
    highlight(el, !e.shiftKey);
    const mode = S.settings.lookupTrigger;
    if (mode === 'click' || (mode === 'shift' && !e.shiftKey)) return;
    const delay = e.shiftKey || performance.now() < LOOK.warmUntil ? 0 : +S.settings.hoverDelay || 0;
    const wordOnly = e.shiftKey;
    LOOK.timer = setTimeout(() => { if (LOOK.el === el && el.isConnected) showCard(el, false, wordOnly); }, delay);
  }

  function onLeave() {
    LOOK.over = false;
    clearTimeout(LOOK.timer);
    clearTimeout(LOOK.scanTimer);
    highlight(LOOK.el, false);
    LOOK.el = null;
    scheduleHide();
  }

  for (const box of [$('#screen'), $('#transcript')]) {
    box.addEventListener('pointermove', onMove);
    box.addEventListener('pointerleave', onLeave);
    box.addEventListener('click', onClick);
  }
  pop.addEventListener('pointerenter', () => { LOOK.over = true; clearTimeout(LOOK.hideTimer); });
  pop.addEventListener('pointerleave', () => { LOOK.over = false; scheduleHide(); });
  $('#transcript').addEventListener('scroll', () => {
    if (!LOOK.inPanel || pop.hidden) return;
    if (LOOK.pinned) placeCard(); else hideCard();
  }, { passive: true });
  document.addEventListener('pointerdown', (e) => {
    // 小視窗和「復原」提示是從卡片開出來的，點它們不收起卡片
    if (LOOK.pinned && !e.target.closest('#wpop, .w, #ruby-pop, #mini, .wl-toast')) hideCard();
  });
  new ResizeObserver(() => placeCard()).observe($('#screen'));
  document.addEventListener('fullscreenchange', () => setTimeout(placeCard, 60));

  function scheduleHide() {
    clearTimeout(LOOK.hideTimer);
    LOOK.hideTimer = setTimeout(() => { if (!LOOK.pinned && !LOOK.over) hideCard(); }, 300);
  }

  function hideCard() {
    clearTimeout(LOOK.hideTimer);
    if (pop.hidden) return;
    pop.hidden = true;
    LOOK.pinned = false;
    LOOK.ctx = null;
    LOOK.anchor = null;
    LOOK.warmUntil = performance.now() + 400;
    clearHighlights();
  }

  function pauseMode() {
    return mediaById(S.current)?.profile === 'asmr' ? S.settings.lookupPauseAsmr : S.settings.lookupPause;
  }

  function showCard(el, pinned, wordOnly = false) {
    const c = wordCtx(el);
    if (!c) return;
    if (wordOnly) c.ph = null;
    openCard(c, el, pinned);
  }

  function openCard(c, anchor, pinned) {
    LOOK.ctx = c;
    LOOK.anchor = anchor;
    LOOK.pinned = pinned;
    LOOK.inPanel = !!c.inPanel;
    (LOOK.inPanel ? document.body : $('#screen')).appendChild(pop);
    pop.className = (pinned ? 'pinned' : '') + (LOOK.inPanel ? ' in-panel' : '');
    pop.scrollTop = 0;
    renderCard();
    pop.hidden = false;
    placeCard();
    if (pinned && anchor.classList) highlight(anchor, !c.wordOnly);
    if (!video.paused && (pinned || pauseMode() === 'now')) {
      video.pause();
      LOOK.pausedByLookup = true;
    }
    if (pinned && !c.scan) fetchDetail(c);
  }

  function fetchDetail(c) {
    const id = `${c.key}|${c.trackId}|${c.i}`;
    if (!V.detail.has(id)) {
      const q = new URLSearchParams({ key: c.key, track: c.trackId, cue: c.i, k: c.k });
      V.detail.set(id, api('GET', `/api/lexeme?${q}`).catch(() => null));
    }
    V.detail.get(id).then((d) => {
      if (LOOK.ctx !== c || !d) return;
      c.detail = d;
      // 快取的是第一次開卡時的回應，裡面的狀態可能已經過時：本地已經有這個字的狀態就以本地為準
      if (!V.status.has(d.key)) V.status.set(d.key, d.st || 0);
      renderCard();
      placeCard();
    });
  }

  function onClick(e) {
    if (!active()) return;
    const el = e.target.closest('.w');
    if (!el) return;
    e.stopPropagation();
    if (LOOK.pinned && LOOK.anchor === el) { hideCard(); return; }
    clearTimeout(LOOK.timer);
    clearHighlights();
    LOOK.el = el;
    showCard(el, true, e.shiftKey);
  }

  function placeCard() {
    const el = LOOK.anchor;
    if (!el || pop.hidden || el.isConnected === false) return; // 換句後錨點不見了就維持原位
    const m = 8, pw = pop.offsetWidth, ph = pop.offsetHeight, wr = el.getBoundingClientRect();
    if (LOOK.inPanel) {
      let top = wr.bottom + m;
      if (top + ph > innerHeight - m) top = Math.max(m, wr.top - ph - m);
      const left = Math.max(m, Math.min(wr.left + wr.width / 2 - pw / 2, innerWidth - pw - m));
      pop.style.transform = `translate(${Math.round(left)}px, ${Math.round(top)}px)`;
      return;
    }
    const scr = $('#screen').getBoundingClientRect();
    const zoneEl = el.closest('.zone, .outside');
    const zone = zoneEl ? zoneEl.getBoundingClientRect() : wr;
    const upper = zone.top + zone.height / 2 < scr.top + scr.height / 2;
    let top = upper ? zone.bottom + m : zone.top - ph - m;
    // 不要蓋住控制列：卡片底邊不超過控制列上緣
    const ctrl = $('#controls').getBoundingClientRect();
    const bottom = (ctrl.height && ctrl.top > scr.top ? Math.min(scr.bottom, ctrl.top) : scr.bottom) - m;
    top = Math.max(scr.top + m, Math.min(top, bottom - ph));
    const left = Math.max(scr.left + m, Math.min(wr.left + wr.width / 2 - pw / 2, scr.right - pw - m));
    pop.style.transform = `translate(${Math.round(left - scr.left)}px, ${Math.round(top - scr.top)}px)`;
  }

  // 「這句播完才暫停」：游標在字幕或卡片上時，播到這句結尾前一點點就停，語音不會被切斷
  function pauseCheck() {
    if (!active() || video.paused || pauseMode() !== 'cueEnd' || !(LOOK.over || !pop.hidden)) return;
    const slot = LOOK.ctx?.slot ?? LOOK.overSlot ?? S.primary;
    const cue = S.cues[slot]?.[S.idx[slot]];
    if (cue && video.currentTime >= cue.end - 0.06) {
      video.pause();
      LOOK.pausedByLookup = true;
    }
  }

  function onCueChange(slot) {
    if (LOOK.el && !LOOK.el.isConnected) LOOK.el = null;
    if (LOOK.ctx && !LOOK.pinned && !LOOK.inPanel && LOOK.ctx.slot === slot) hideCard();
  }

  /* ---------- Shift 即時查詢（日文切錯的字） ---------- */

  function textNodes(host) {
    return document.createTreeWalker(host, NodeFilter.SHOW_TEXT, {
      acceptNode: (t) => (t.parentElement.closest('rt') ? NodeFilter.FILTER_REJECT : NodeFilter.FILTER_ACCEPT),
    });
  }

  function caretCharOffset(host, x, y) {
    let node, offset;
    if (document.caretPositionFromPoint) {
      const p = document.caretPositionFromPoint(x, y);
      node = p?.offsetNode; offset = p?.offset;
    } else if (document.caretRangeFromPoint) {
      const r = document.caretRangeFromPoint(x, y);
      node = r?.startContainer; offset = r?.startOffset;
    }
    if (!node || !host.contains(node) || node.nodeType !== 3 || node.parentElement?.closest('rt')) return -1;
    // 游標在字的右半邊時，caret 會落在這個字後面：改成游標底下那個字
    if (offset > 0) {
      const range = document.createRange();
      range.setStart(node, offset - 1);
      range.setEnd(node, offset);
      const r = range.getBoundingClientRect();
      if (x >= r.left && x <= r.right) offset -= 1;
    }
    const walker = textNodes(host);
    let n = 0;
    for (let t = walker.nextNode(); t; t = walker.nextNode()) {
      if (t === node) return n + offset;
      n += t.length;
    }
    return -1;
  }

  function rangeRect(host, a, b) {
    const walker = textNodes(host);
    const range = document.createRange();
    let n = 0, started = false;
    for (let t = walker.nextNode(); t; t = walker.nextNode()) {
      if (!started && a < n + t.length) { range.setStart(t, a - n); started = true; }
      if (started && b <= n + t.length) { range.setEnd(t, b - n); return range.getBoundingClientRect(); }
      n += t.length;
    }
    return host.getBoundingClientRect();
  }

  function scheduleScan(e) {
    clearTimeout(LOOK.scanTimer);
    if (S.settings.lookupTrigger === 'click') return;
    const host = e.target.closest('[data-cue]');
    if (!host || !host.dataset.cue) return;
    const [slot, i] = host.dataset.cue.split(':').map(Number);
    const trackId = S.trackSel[slot];
    if (trackOf(trackId)?.lang !== 'ja') return;
    const x = e.clientX, y = e.clientY;
    LOOK.scanTimer = setTimeout(async () => {
      const pos = caretCharOffset(host, x, y);
      if (pos < 0) return;
      const res = await api('GET', `/api/dict/scan?track=${trackId}&cue=${i}&pos=${pos}`).catch(() => null);
      if (!res || !res.cands.length || LOOK.pinned || !host.isConnected) return;
      if (LOOK.ctx?.scan && LOOK.ctx.pos === pos && LOOK.ctx.i === i) return;
      for (const cand of res.cands) V.status.set(cand.key, cand.st || 0);
      const r = rangeRect(host, pos, pos + res.len);
      const anchor = { isConnected: true, getBoundingClientRect: () => r, closest: (sel) => host.closest(sel) };
      openCard({ scan: true, res, slot, i, pos, trackId, lang: 'ja', inPanel: !!host.closest('#transcript') }, anchor, false);
    }, 80);
  }

  /* ---------- 卡片內容 ---------- */

  function tagsHtml({ lv, common, misc, grp, upos, phrase }) {
    const tags = [];
    if (phrase) tags.push('<span class="wp-tag">片語</span>');
    if (lv) tags.push(`<span class="wp-tag lv" title="${lv.startsWith('N') ? 'JLPT 參考等級' : 'CEFR 參考等級'}">${esc(lv)}</span>`);
    if (common) tags.push('<span class="wp-tag">常用</span>');
    if (upos && UPOS_ZH[upos]) tags.push(`<span class="wp-tag">${UPOS_ZH[upos]}</span>`);
    for (const m of misc || []) if (MISC_ZH[m] && m !== 'uk') tags.push(`<span class="wp-tag">${esc(MISC_ZH[m])}</span>`);
    if (grp && GRP_ZH[grp] && grp !== 'function') tags.push(`<span class="wp-tag">${GRP_ZH[grp]}</span>`);
    return tags.join('');
  }

  function headHtml(word, reading, tags, key, lang, extra = '') {
    const st = stOf(key);
    return `<div class="wp-head"${lang === 'ja' ? ' lang="ja"' : ''}><span class="wp-word">${esc(word)}</span>`
      + `${reading ? `<span class="wp-read">${esc(reading)}</span>` : ''}${extra}${tags}`
      + `${st ? `<span class="wp-st st${st}">${ST_LABEL[st] || ''}</span>` : ''}</div>`;
  }

  const srcLabel = (gs) => (gs === 'mt' ? '機器翻譯' : gs === 'en' ? '英' : gs === 'moe' ? '教育部辭典' : '');

  function glossHtml(g, gs) {
    if (!g) return '<div class="wp-gloss muted">字典裡查不到</div>';
    const src = srcLabel(gs);
    return `<div class="wp-gloss">${esc(g)}${src ? `<span class="wp-src">${src}</span>` : ''}</div>`;
  }

  function jaHtml(c, lex) {
    const d = c.detail;
    const word = d?.d || lex.d || c.surface;
    const reading = d ? d.r : lex.r;
    const accent = d?.accent ? `<span class="wp-acc" title="重音（辭書形）">[${esc(d.accent)}]</span>` : '';
    let html = headHtml(word, reading,
      tagsHtml({ lv: d?.lv ?? lex.lv, common: d?.common ?? lex.common, misc: lex.misc, grp: lex.grp }), c.key, 'ja', accent);
    const chain = c.w[3];
    if (chain && chain.length) html += `<div class="wp-chain">${esc(chainText(c.surface, word, chain))}</div>`;
    if (LOOK.pinned && d && d.senses && d.senses.length) {
      html += '<ol class="wp-senses">' + d.senses.slice(0, 12).map((s) => {
        const labels = [...new Set((s.pos || []).map(posZh))].concat((s.misc || []).map((m) => MISC_ZH[m]).filter(Boolean));
        const main = s.zh ? `${esc(s.zh)}<div class="wp-en">${esc(s.en)}</div>` : `${esc(s.en)}<span class="wp-src">英</span>`;
        return `<li>${labels.length ? `<span class="wp-pos">${esc(labels.join('・'))}</span>` : ''}${main}</li>`;
      }).join('') + '</ol>';
      const forms = [...new Set([...(d.forms || []), ...(d.readings || [])])].filter((f) => f !== word && f !== reading);
      if (forms.length) html += `<div class="wp-meta">其他寫法：${esc(forms.slice(0, 6).join('、'))}</div>`;
      if (d.alts && d.alts.length) {
        html += `<div class="wp-meta">也可能是：${d.alts.map((a) => `${esc(a.d)}（${esc(a.g.split(' / ')[0].split('；')[0])}）`).join('、')}</div>`;
      }
    } else {
      html += glossHtml(lex.g, lex.gs);
    }
    return html;
  }

  function enHtml(c, lex) {
    const d = c.detail;
    const upos = c.w[3] || '';
    const word = lex.d || c.surface;
    const pn = d?.pn || lex.pn;
    let html = headHtml(word, pn ? `/${pn}/` : '', tagsHtml({ lv: lex.lv, grp: lex.grp, upos }), c.key, 'en');
    if (c.surface.toLowerCase() !== word.toLowerCase() && lex.grp !== 'name') {
      html += `<div class="wp-chain">${esc(c.surface)} → ${esc(word)}</div>`;
    }
    const lines = sortLines(LOOK.pinned && d?.lines ? d.lines : lex.zl, upos);
    const max = LOOK.pinned ? 10 : 3;
    html += lines.length
      ? `<ul class="wp-lines">${lines.slice(0, max).map((l) => `<li>${esc(l)}</li>`).join('')}</ul>`
      : '<div class="wp-gloss muted">字典裡查不到</div>';
    if (lex.sp) html += `<div class="wp-meta">${esc(c.surface)} ＝ ${esc(lex.sp)}</div>`;
    if (LOOK.pinned && d?.surface) {
      html += `<div class="wp-meta">「${esc(d.surface.word)}」本身：${esc(d.surface.lines.slice(0, 2).join('；'))}</div>`;
    }
    if (LOOK.pinned && d) {
      const bits = [];
      if (d.z) bits.push(`詞頻 ${d.z.toFixed(1)}`);
      if (d.collins) bits.push(`柯林斯 ${'★'.repeat(d.collins)}`);
      if (d.oxford) bits.push('牛津 3000');
      if (bits.length) html += `<div class="wp-meta">${bits.join(' · ')}</div>`;
    }
    return html;
  }

  function phraseHtml(c) {
    const key = c.ph[0];
    const lex = lexOf(c.trackId, key) || { d: key.slice(3), zl: [] };
    const lines = lex.zl || [];
    const st = LOOK.pinned ? statusButtons(key, true) : '';
    return `<div class="wp-phrase">${headHtml(lex.d, '', tagsHtml({ phrase: true }), key, 'en')}`
      + `${lines.length ? `<ul class="wp-lines">${lines.slice(0, 2).map((l) => `<li>${esc(l)}</li>`).join('')}</ul>` : ''}${st}</div>`;
  }

  function statusButtons(key, small = false) {
    const st = stOf(key);
    return `<span class="wp-stbtns${small ? ' small' : ''}">` + [1, 2, 3].map((n) =>
      `<button type="button" class="btn small${st === n ? ` on st${n}` : ''}" data-st="${n}" data-key="${esc(key)}"`
      + ` title="快捷鍵 ${ST_KEY[n]}${st === n ? '，再按一次取消' : ''}">${ST_LABEL[n]}</button>`).join('') + '</span>';
  }

  function examplesHtml(c) {
    const d = c.detail;
    if (!d) return '<div class="wp-sec"><span class="wp-label">讀取例句…</span></div>';
    LOOK.ex = [];
    // 每一句：整列點下去跳到影片，右邊的小按鈕開小視窗只播這一句
    const row = (x, mediaId, title, sub, open) => {
      const n = LOOK.ex.push({ media: mediaId, start: x.start, end: x.end, text: x.text, hl: x.hl, ruby: x.ruby,
                               zh: x.zh, title, lang: c.lang }) - 1;
      const jump = open ? ` data-open="${esc(mediaId)}" data-t="${x.start}"` : ` data-jump="${x.start}"`;
      return `<div class="wp-exrow"><button type="button" class="wp-ex"${jump}><time>${fmtTime(x.start)}</time>`
        + `<span>${sentHtml(x.text, x.hl, x.ruby)}${sub ? `<small>${esc(sub)}</small>` : ''}</span></button>`
        + `<button type="button" class="wp-mini" data-mini="${n}" title="小視窗只播這一句">▶</button></div>`;
    };
    let html = '';
    const m0 = mediaById(S.current);
    if (d.examples && d.examples.length) {
      html += '<div class="wp-sec"><div class="wp-label">本片其他例句</div>'
        + d.examples.map((x) => row(x, S.current, m0 ? titleParts(m0).main : '', x.zh, false)).join('') + '</div>';
    }
    if (d.others && d.others.length) {
      const total = Math.max(0, (d.total_videos || 0) - (d.examples && d.examples.length ? 1 : 0));
      html += `<div class="wp-sec"><div class="wp-label">其他影片${total ? `（共 ${total} 部）` : ''}</div>` + d.others.map((g) => {
        const m = mediaById(g.media_id);
        const title = m ? titleParts(m).main : g.title;
        return row(g.items[0], g.media_id, title, title, true);
      }).join('') + '</div>';
    }
    return html;
  }

  function actionsHtml(c) {
    const word = c.detail?.d || lexOf(c.trackId, c.key)?.d || c.surface;
    let rubyBtns = '';
    if (c.lang === 'ja' && LOOK.anchor?.querySelectorAll) {
      const rubies = [...LOOK.anchor.querySelectorAll('ruby[data-s], .no-ruby[data-s]')];
      rubyBtns = rubies.map((r, n) => {
        const base = c.cue.text.slice(+r.dataset.s, +r.dataset.e);
        return `<button type="button" class="btn small" data-ruby="${n}" title="修改這一段的假名讀音">修改讀音${rubies.length > 1 ? `：${esc(base)}` : ''}</button>`;
      }).join('');
    }
    const links = c.lang === 'ja'
      ? `<a href="https://jisho.org/search/${encodeURIComponent(word)}" target="_blank" rel="noreferrer">Jisho</a>`
        + `<a href="https://www.weblio.jp/content/${encodeURIComponent(word)}" target="_blank" rel="noreferrer">Weblio</a>`
      : `<a href="https://dictionary.cambridge.org/zht/${encodeURIComponent('詞典')}/${encodeURIComponent('英語-漢語-繁體')}/${encodeURIComponent(word.toLowerCase())}" target="_blank" rel="noreferrer">Cambridge</a>`;
    return `<div class="wp-actions">${statusButtons(c.key)}${rubyBtns}<span class="wp-links">${links}</span></div>`;
  }

  function scanHtml(c) {
    const [first, ...rest] = c.res.cands;
    let html = headHtml(first.d, first.r, tagsHtml({ lv: first.lv, common: first.common, misc: first.misc }), first.key, 'ja');
    if (first.chain && first.chain.length) html += `<div class="wp-chain">${esc(chainText(first.surface, first.d, first.chain))}</div>`;
    html += glossHtml(first.g, first.gs);
    if (rest.length) html += `<div class="wp-meta">也可能是：${rest.map((x) => `${esc(x.d)}（${esc(x.g.split(' / ')[0])}）`).join('、')}</div>`;
    html += footHtml('即時查詢（不算進單字統計）', first.key);
    return html;
  }

  // 還沒釘選的小卡：底下一行說明，右邊放狀態按鈕（對象跟快捷鍵一樣：片語優先）
  const footHtml = (text, key) =>
    `<div class="wp-foot"><span class="wp-meta">${esc(text)}</span>${statusButtons(key, true)}</div>`;

  function renderCard() {
    const c = LOOK.ctx;
    if (!c) return;
    if (c.scan) { pop.innerHTML = scanHtml(c); return; }
    const lex = lexOf(c.trackId, c.key) || {};
    let html = '';
    if (c.lang === 'en' && c.ph) html += phraseHtml(c);
    html += c.lang === 'en' ? enHtml(c, lex) : jaHtml(c, lex);
    if (LOOK.pinned) {
      html += examplesHtml(c) + actionsHtml(c);
    } else {
      const n = lex.n ? `本片 ${lex.n} 次` : '';
      html += footHtml([n, S.settings.lookupTrigger === 'click' ? '' : '點一下看完整解釋'].filter(Boolean).join(' · '), primaryKey(c));
    }
    pop.innerHTML = html;
  }

  pop.addEventListener('click', (e) => {
    const stBtn = e.target.closest('[data-st]');
    const jump = e.target.closest('[data-jump]');
    const open = e.target.closest('[data-open]');
    const ruby = e.target.closest('[data-ruby]');
    const miniBtn = e.target.closest('[data-mini]');
    const link = e.target.closest('a[href]');
    const c = LOOK.ctx;
    if (miniBtn) {
      openMini(LOOK.ex[+miniBtn.dataset.mini]);
    } else if (stBtn) {
      if (e.detail < 2) toggleStatus(stBtn.dataset.key, +stBtn.dataset.st);   // 連點兩下不要標了又取消
    } else if (jump) {
      video.currentTime = leadTime(+jump.dataset.jump);
      updateSubs(true);
    } else if (open) {
      hideCard();
      jumpToMedia(open.dataset.open, +open.dataset.t);
    } else if (ruby && c && LOOK.anchor) {
      const el = [...LOOK.anchor.querySelectorAll('ruby[data-s], .no-ruby[data-s]')][+ruby.dataset.ruby];
      const ctx = el && rubyContext(el);
      hideCard();
      if (ctx && ctx.trackId && ctx.index >= 0) openRubyPop(el, ctx);
    } else if (link && document.fullscreenElement) {
      // 全螢幕時開新分頁會離開全螢幕：先退出再開，不會卡在黑畫面
      e.preventDefault();
      document.exitFullscreen().finally(() => window.open(link.href, '_blank', 'noreferrer'));
    }
  });

  /* ---------- 不會／已會／忽略 ---------- */

  // 標記後的提示，附一個復原按鈕。連續標了好幾個字時，按一次復原退一步，提示會換成前一個
  // 放的位置：單字視窗開著時放進視窗底部那一列（浮在畫面上會疊到視窗），全螢幕時放進全螢幕的元素
  const undoBox = document.createElement('div');
  undoBox.className = 'wl-toast';
  undoBox.hidden = true;
  const UNDO = { timer: 0, stack: [] };
  const UNDO_MS = 8000;
  const toastHost = () => (dlgWord.open ? dlgWord.querySelector('.dlg-foot') : document.fullscreenElement || document.body);
  const undoLabel = (u) => `${u.name ? `「${u.name}」` : ''}${u.st ? `已標成${ST_LABEL[u.st]}` : '已清除標記'}`;

  undoBox.addEventListener('click', (e) => {
    if (!e.target.closest('button')) return;
    const last = UNDO.stack.pop();
    if (last) setStatus(last.key, last.prev, true);
    const next = UNDO.stack[UNDO.stack.length - 1];
    if (!next) { hideUndo(); return; }
    renderUndo(`已復原${last && last.name ? `「${last.name}」` : ''}，前一個：${undoLabel(next)}`);
  });
  // 滑鼠停在提示上時不要自己消失，免得正要按復原時剛好不見、點到底下的東西
  undoBox.addEventListener('pointerenter', () => clearTimeout(UNDO.timer));
  undoBox.addEventListener('pointerleave', () => armUndo(3000));

  function armUndo(ms) {
    clearTimeout(UNDO.timer);
    UNDO.timer = setTimeout(hideUndo, ms);
  }

  function hideUndo() {
    clearTimeout(UNDO.timer);
    undoBox.hidden = true;
    UNDO.stack = [];
  }

  function renderUndo(text) {
    undoBox.innerHTML = `<span>${esc(text)}</span><button type="button" class="btn small">復原</button>`;
    const host = toastHost();
    if (undoBox.parentElement !== host) host.prepend(undoBox);
    undoBox.hidden = false;
    armUndo(UNDO_MS);
  }

  function pushUndo(entry) {
    UNDO.stack.push(entry);
    if (UNDO.stack.length > 30) UNDO.stack.shift();
    renderUndo(undoLabel(entry));
  }

  // 提示裡要寫是哪個字（從單字列按的時候，那一列可能馬上被篩掉）
  function wordName(key) {
    const it = VB.data?.items.find((x) => x.k === key);
    if (it) return it.d;
    for (const v of Object.values(V.vocab)) if (v.lex?.[key]) return v.lex[key].d;
    if (LOOK.ctx?.scan) return LOOK.ctx.res.cands.find((x) => x.key === key)?.d || '';
    return LOOK.ctx?.detail?.key === key ? LOOK.ctx.detail.d : '';
  }

  async function setStatus(key, st, quiet = false) {
    const prev = stOf(key);
    if (prev === st) return;
    applyStatus(key, st);
    try {
      await api('PUT', '/api/lexeme/status', { key, status: st });
      if (!quiet) pushUndo({ key, prev, st, name: wordName(key) });
    } catch (err) {
      applyStatus(key, prev);
      toast(err.message, true);
    }
  }

  // 按目前這顆就清回未標記
  const toggleStatus = (key, st) => setStatus(key, stOf(key) === st ? 0 : st);

  function applyStatus(key, st) {
    V.status.set(key, st);
    for (const v of Object.values(V.vocab)) if (v.lex?.[key]) v.lex[key].st = st;
    const item = VB.data?.items.find((x) => x.k === key);
    if (item) item.st = st;
    if (!pop.hidden) renderCard();
    if (WD.key === key && dlgWord.open) renderWordStatus();
    if (VB.open) refreshVocabAfterMark(key);
  }

  function primaryKey(c) {
    if (c.scan) return c.res.cands[0].key;
    return c.ph ? c.ph[0] : c.key;
  }

  function onKey(e, key) {
    if (!active()) return false;
    if (key === 'Shift' && !e.repeat && LOOK.el && LOOK.el.isConnected && S.settings.lookupTrigger !== 'click'
        && (pop.hidden || LOOK.anchor !== LOOK.el)) {
      clearTimeout(LOOK.timer);
      showCard(LOOK.el, LOOK.pinned, true);
      return false;
    }
    if (pop.hidden || !LOOK.ctx) return false;
    const k = primaryKey(LOOK.ctx);
    if (key === 'Escape') { e.preventDefault(); hideCard(); return true; }
    if (KEY_ST[key]) {
      e.preventDefault();
      if (!e.repeat) toggleStatus(k, KEY_ST[key]);    // 按住不放不要一直來回切換
      return true;
    }
    if (key === ' ') hideCard(); // 接著照原本的播放／暫停
    return false;
  }

  /* ---------- 跳到某部影片的某個時間 ---------- */

  // 跳過去的時間點：比句首早一點點，但不早於前一句的結尾。句子之間常常只隔幾十毫秒，
  // 早到前一句裡面的話，影片停著時畫面上會一直是前一句的字幕
  function leadTime(t) {
    let at = Math.max(0, t - 0.05);
    for (const slot of [1, 2]) {
      for (const c of S.cues[slot] || []) {
        if (c.start < t && c.end > at && c.end <= t + 0.001) at = Math.min(t, c.end + 0.002);
      }
    }
    return at;
  }

  // 跳到別部影片時字幕是之後才讀進來的：讀好時如果還停在剛才跳到的位置，照字幕再對一次
  const JUMP = { mid: '', t: 0, at: -1 };

  function jumpToMedia(mid, t) {
    closeMini();                                       // 回到完整播放器，小視窗（可能是別句）一起收掉
    showView('player');
    const go = () => {
      JUMP.mid = mid;
      JUMP.t = t;
      JUMP.at = leadTime(t);
      video.currentTime = JUMP.at;
      S.userScrollAt = 0;
      updateSubs(true);
    };
    if (S.current === mid && video.readyState >= 1) { go(); return; }
    if (S.current !== mid) openMedia(mid);
    video.addEventListener('loadedmetadata', () => setTimeout(go, 0), { once: true });
  }

  /* ---------- 單字頁 ---------- */

  // 分頁在前端做：詞表一次拿整份（日文約 1.2MB，後端彙整要 1.5 秒上下），之後換頁、搜尋、排序、
  // 狀態篩選都不用再問伺服器，篩選按鈕上的數量也要整份才算得出來；標記狀態只改本地資料。
  // 真正慢的是畫出幾千列，所以一次只畫一頁（最多 200 列）。
  const VB = { open: false, data: null, page: 1, nav: [], req: 0, pollTimer: 0, langs: null, fadeTimer: 0, guard: null };
  const LANG_NAME = { ja: '日文', en: '英文', zh: '中文' };
  // 正在排隊或建立的字典任務（第一次自動安裝，或設定頁、單字頁按的「建立」）
  const dictJob = (lang) => S.jobs.find((j) => j.type === 'dict' && (j.status === 'queued' || j.status === 'running')
    && (j.params || {}).lang === lang);
  const ckipJob = () => S.jobs.find((j) => j.type === 'model' && (j.params || {}).model === 'ckip'
    && ['queued', 'running', 'paused'].includes(j.status));
  // 「建立」「重建」：排進佇列（POST /api/dicts/{lang}/build）。舊版伺服器沒有這個 API（SU.api 不是 true）時不顯示按鈕
  async function buildDict(lang, force = false) {
    try {
      await api('POST', `/api/dicts/${lang}/build`, { force_download: force });
      toast(`已排進佇列，${force ? '重建' : '建立'}${LANG_NAME[lang]}字典`);
      await refreshSoon();
      return true;
    } catch (err) {
      toast(err.message, true);
      return false;
    }
  }
  const PAGE_SIZES = [50, 100, 200];
  const pageSize = () => (PAGE_SIZES.includes(+S.settings.vocabPageSize) ? +S.settings.vocabPageSize : 100);
  const statusFilter = () => S.settings.vocabStatus || 'learn';

  async function openVocabPage() {
    VB.open = true;
    hideCard();
    if (!video.paused) video.pause();
    try { VB.langs = await api('GET', '/api/vocab/langs'); } catch { VB.langs = VB.langs || {}; }
    if (!S.settings.vocabLang) {
      S.settings.vocabLang = Object.entries(VB.langs || {}).sort((a, b) => b[1] - a[1])[0]?.[0] || 'ja';
      saveSettings();
    }
    renderVocabHead();
    loadVocabPage();
  }

  function mediaOfLang(lang) {
    const tl = { ja: 'ja', en: 'en', zh: 'zh-TW' }[lang];
    return S.media.filter((m) => m.tracks.some((t) => t.kind === 'asr' && t.lang === tl));
  }

  function renderVocabHead() {
    const lang = S.settings.vocabLang;
    for (const b of $$('#vb-lang button')) {
      b.classList.toggle('on', b.dataset.v === lang);
      $('span', b).textContent = VB.langs?.[b.dataset.v] ?? '';
    }
    const list = mediaOfLang(lang);
    if (S.settings.vocabScope && !list.some((m) => m.id === S.settings.vocabScope)) S.settings.vocabScope = '';
    const html = `<option value="">全部${LANG_NAME[lang]}影片（${list.length} 部）</option>`
      + list.map((m) => `<option value="${m.id}">${esc(titleParts(m).main)}</option>`).join('');
    const sel = $('#vb-scope');
    if (sel.dataset.html !== html) { sel.innerHTML = html; sel.dataset.html = html; }
    sel.value = S.settings.vocabScope || '';
    $('#vb-other').checked = !!S.settings.vocabOther;
    $('#vb-other-wrap').lastChild.textContent = lang === 'en' ? ' 顯示人名、the、of 這類功能詞' : ' 顯示人名、地名、感嘆詞這類';
    const sort = $('#vb-sort');
    sort.querySelector('[value="first"]').hidden = !S.settings.vocabScope;
    sort.querySelector('[value="alpha"]').textContent = lang === 'ja' ? '五十音' : lang === 'en' ? '字母' : '筆畫';
    if (!S.settings.vocabScope && S.settings.vocabSort === 'first') S.settings.vocabSort = 'count';
    sort.value = S.settings.vocabSort || 'count';
    $('#vb-table').classList.toggle('one-video', !!S.settings.vocabScope);
  }

  async function loadVocabPage() {
    clearTimeout(VB.pollTimer);
    const lang = S.settings.vocabLang, media = S.settings.vocabScope || '';
    const req = ++VB.req;
    if (!VB.data || VB.data.lang !== lang || (VB.data.media || '') !== media) {
      VB.data = null;
      $('#vb-list').innerHTML = '<div class="vb-empty">讀取中…</div>';
      $('#vb-summary').textContent = '';
    }
    let data;
    try {
      data = await api('GET', `/api/vocab?lang=${lang}&media=${encodeURIComponent(media)}`);
    } catch (err) {
      if (req === VB.req) $('#vb-list').innerHTML = `<div class="vb-empty">${esc(err.message)}</div>`;
      return;
    }
    if (req !== VB.req || !VB.open) return;
    const sameList = VB.data && VB.data.lang === data.lang && (VB.data.media || '') === (data.media || '');
    VB.data = data;
    if (!sameList) VB.page = 1;
    for (const it of data.items) V.status.set(it.k, it.st || 0);
    renderVocabNote();
    renderVocabList();
    if (data.pending > 0 || data.busy || data.zh?.running || (!data.dict_ready && dictJob(lang))) {
      VB.pollTimer = setTimeout(() => { if (VB.open) loadVocabPage(); }, 3000);
    }
  }

  function renderVocabNote() {
    const d = VB.data;
    const box = $('#vb-note');
    const parts = [];
    if (d.lang === 'zh' && !d.dict_ready) {
      parts.push('<span>中文斷詞還沒安裝，這個分頁暫時不支援。</span>');
    } else if (!d.dict_ready && dictJob(d.lang)) {
      parts.push(`<span>${LANG_NAME[d.lang]}字典建立中，好了會自動更新。</span>`);
    } else if (!d.dict_ready && SU.api) {
      parts.push(`<span>${LANG_NAME[d.lang]}字典還沒建立。</span><button type="button" class="btn small" data-dict-build="${d.lang}">建立</button>`);
    } else if (!d.dict_ready) {
      parts.push(`<span>${LANG_NAME[d.lang]}字典還沒建立，請先執行 tools\\build_dict_${d.lang}.py。</span>`);
    }
    if (d.lang === 'zh' && d.zh?.running) parts.push('<span>正在用 CKIP 斷詞（只用 CPU），第一次要一兩分鐘…</span>');
    if (d.lang === 'zh' && d.zh?.error) parts.push(`<span class="bad">${esc(d.zh.error)}</span>`);
    if (d.pending > 0) parts.push(`<span>還有 ${d.pending} 部影片的字幕正在分析，好了會自動更新。</span>`);
    if (d.lang === 'ja' && d.missing_zh > 0) {
      const queued = S.jobs.some((j) => j.type === 'glosses' && (j.status === 'queued' || j.status === 'running'));
      parts.push(`<span>有 ${d.missing_zh.toLocaleString()} 個詞還沒有中文解釋，暫時顯示英文（標「英」）。</span>`
        + (queued ? '<span class="muted">翻譯任務已經在佇列裡</span>'
          : '<button type="button" class="btn small" id="vb-gloss">排進佇列翻成中文</button>'));
    }
    box.innerHTML = parts.join('');
    box.hidden = !parts.length;
  }

  // 語言、範圍（後端已經處理）之外的條件：詞類與搜尋。狀態篩選的數字就是照這一份算的
  function baseItems() {
    const d = VB.data;
    if (!d) return [];
    const q = $('#vb-search').value.trim().toLowerCase();
    return d.items.filter((it) => {
      if (it.grp !== 'word' && !S.settings.vocabOther) return false;
      if (q && !(it.d.toLowerCase().includes(q) || (it.r || '').includes(q) || (it.g || '').toLowerCase().includes(q))) return false;
      return true;
    });
  }

  const matchStatus = (st, f) => (f === 'all' ? true : f === 'learn' ? st === 0 || st === 1 : st === +f);

  function sortItems(items) {
    const sort = S.settings.vocabSort;
    if (sort === 'videos') items.sort((a, b) => b.v - a.v || b.n - a.n);
    else if (sort === 'first') items.sort((a, b) => a.t - b.t || b.n - a.n);
    else if (sort === 'alpha') {
      const lang = VB.data?.lang;
      const coll = new Intl.Collator(lang === 'ja' ? 'ja' : lang === 'zh' ? 'zh-Hant-TW-u-co-stroke' : 'en');
      items.sort((a, b) => coll.compare(a.r || a.d, b.r || b.d));
    } else items.sort((a, b) => b.n - a.n || b.v - a.v);
    return items;
  }

  function filteredItems() {
    return sortItems(baseItems().filter((it) => matchStatus(stOf(it.k), statusFilter())));
  }

  function renderStatusBar(base) {
    const counts = { all: base.length, 0: 0, 1: 0, 2: 0, 3: 0, learn: 0 };
    for (const it of base) {
      const st = stOf(it.k);
      counts[st] += 1;
      if (st === 0 || st === 1) counts.learn += 1;
    }
    for (const b of $$('#vb-status button')) {
      b.classList.toggle('on', b.dataset.v === statusFilter());
      $('span', b).textContent = (counts[b.dataset.v] || 0).toLocaleString();
    }
  }

  function rowHtml(it) {
    const st = stOf(it.k);
    const src = srcLabel(it.gs);
    const tags = (it.ph ? '<span class="wp-tag">片語</span>' : '') + (it.grp !== 'word' && GRP_ZH[it.grp] ? `<span class="wp-tag">${GRP_ZH[it.grp]}</span>` : '');
    const acts = [1, 2, 3].map((n) =>
      `<button type="button" class="vb-st${st === n ? ` on st${n}` : ''}" data-st="${n}"`
      + ` title="${st === n ? '再按一次取消' : `標成${ST_LABEL[n]}`}">${ST_LABEL[n]}</button>`).join('');
    return `<div class="vb-row${st ? ' st' + st : ''}" data-k="${esc(it.k)}">
      <span class="vb-word"${VB.data.lang === 'ja' ? ' lang="ja"' : ''}><b>${esc(it.d)}</b>${it.r ? `<small>${esc(it.r)}</small>` : ''}${tags}</span>
      <span class="vb-gloss" title="${esc(it.g || '')}">${esc(it.g || '')}${src ? `<span class="vb-src">${src === '機器翻譯' ? '機翻' : src}</span>` : ''}</span>
      <span class="vb-lv">${esc(it.lv || '')}</span>
      <span class="num">${it.n.toLocaleString()}</span>
      <span class="num vb-col-videos">${it.v}</span>
      <span class="vb-acts">${acts}</span>
    </div>`;
  }

  // 頁碼：頭尾一定看得到，中間留目前這一頁前後各一個，斷掉的地方放省略號
  function pageNums(cur, total) {
    const want = new Set([1, total, cur - 1, cur, cur + 1]);
    if (cur <= 3) [2, 3, 4].forEach((n) => want.add(n));
    if (cur >= total - 2) [total - 1, total - 2, total - 3].forEach((n) => want.add(n));
    const out = [];
    let prev = 0;
    for (const n of [...want].filter((n) => n >= 1 && n <= total).sort((a, b) => a - b)) {
      if (prev && n - prev === 2) out.push(prev + 1);   // 只差一頁時直接寫頁碼，省略號不會比較短
      else if (prev && n - prev > 2) out.push('…');
      out.push(n);
      prev = n;
    }
    return out;
  }

  function renderPager(pages, count) {
    const box = $('#vb-pager');
    box.hidden = !count;
    if (!count) return;
    const many = pages > 1;
    $('#vb-prev').hidden = $('#vb-next').hidden = $('#vb-goto').parentElement.hidden = !many;
    $('#vb-prev').disabled = VB.page <= 1;
    $('#vb-next').disabled = VB.page >= pages;
    $('#vb-pages').innerHTML = many ? pageNums(VB.page, pages).map((p) => (p === '…'
      ? '<span class="vb-gap">…</span>'
      : `<button type="button" class="vb-page${p === VB.page ? ' on' : ''}" data-p="${p}">${p}</button>`)).join('') : '';
    const goto = $('#vb-goto');
    goto.max = String(pages);
    if (document.activeElement !== goto) goto.value = String(VB.page);
    $('#vb-size').value = String(pageSize());
  }

  function renderVocabList() {
    clearTimeout(VB.fadeTimer);
    VB.fadeTimer = 0;
    const d = VB.data;
    if (!d) return;
    const base = baseItems();
    renderStatusBar(base);
    const items = sortItems(base.filter((it) => matchStatus(stOf(it.k), statusFilter())));
    const size = pageSize();
    const pages = Math.max(1, Math.ceil(items.length / size));
    VB.page = Math.min(Math.max(1, VB.page), pages);     // 這一頁的字都標掉了就退一頁
    VB.nav = items.map((it) => it.k);
    const shown = items.slice((VB.page - 1) * size, VB.page * size);
    const total = items.reduce((s, it) => s + it.n, 0);
    $('#vb-summary').textContent = items.length
      ? `第 ${VB.page} / ${pages} 頁，共 ${items.length.toLocaleString()} 個詞、出現 ${total.toLocaleString()} 次`
      : '';
    $('#vb-list').innerHTML = shown.length
      ? shown.map(rowHtml).join('')
      : `<div class="vb-empty">${d.items.length ? '沒有符合條件的單字' : d.tracks ? '還沒有統計資料' : `還沒有${LANG_NAME[d.lang]}影片的字幕`}</div>`;
    renderPager(pages, items.length);
  }

  // 標記之後更新列表。被篩掉的那一列先淡出一下再拿掉，看得出是哪一個字不見了；
  // 拿掉時底下的列會補上來，所以 VB.guard 會讓同一個位置接下來的點擊暫時不算（見 #vb-list 的 click）
  const FADE_MS = 300;
  function refreshVocabAfterMark(key) {
    const item = VB.data?.items.find((x) => x.k === key);
    const row = $$('#vb-list .vb-row').find((r) => r.dataset.k === key);
    const leaving = !!(row && item && !matchStatus(stOf(key), statusFilter()));
    if (!leaving && !VB.fadeTimer) { renderVocabList(); return; }
    if (row && item) {
      const tmp = document.createElement('div');
      tmp.innerHTML = rowHtml(item);
      const fresh = tmp.firstElementChild;
      fresh.classList.toggle('leaving', leaving);
      row.replaceWith(fresh);
    }
    renderStatusBar(baseItems());
    if (leaving && VB.guard) VB.guard.until = performance.now() + FADE_MS + 400;
    clearTimeout(VB.fadeTimer);
    VB.fadeTimer = setTimeout(renderVocabList, FADE_MS);
  }

  function goToPage(p) {
    VB.page = Math.max(1, p || 1);
    renderVocabList();
    // 捲回列表頂端；已經看得到列表開頭（例如在上面點篩選）就不動，免得篩選列被捲走
    const page = $('#vocab-page');
    const top = Math.max(0, $('#vb-table').offsetTop - 16);
    if (page.scrollTop > top) page.scrollTop = top;
  }

  $('#vb-lang').addEventListener('click', (e) => {
    const b = e.target.closest('button[data-v]');
    if (!b || b.dataset.v === S.settings.vocabLang) return;
    S.settings.vocabLang = b.dataset.v;
    S.settings.vocabScope = '';
    saveSettings();
    renderVocabHead();
    loadVocabPage();
  });
  $('#vb-scope').addEventListener('change', (e) => {
    S.settings.vocabScope = e.target.value;
    saveSettings();
    renderVocabHead();
    loadVocabPage();
  });
  $('#vb-status').addEventListener('click', (e) => {
    const b = e.target.closest('button[data-v]');
    if (!b || b.dataset.v === statusFilter()) return;
    S.settings.vocabStatus = b.dataset.v;
    saveSettings();
    goToPage(1);
  });
  $('#vb-other').addEventListener('change', (e) => {
    S.settings.vocabOther = e.target.checked;
    saveSettings();
    goToPage(1);
  });
  $('#vb-sort').addEventListener('change', (e) => { S.settings.vocabSort = e.target.value; saveSettings(); goToPage(1); });
  let searchTimer = 0;
  $('#vb-search').addEventListener('input', () => {
    clearTimeout(searchTimer);
    searchTimer = setTimeout(() => goToPage(1), 150);
  });
  $('#vb-prev').addEventListener('click', () => goToPage(VB.page - 1));
  $('#vb-next').addEventListener('click', () => goToPage(VB.page + 1));
  $('#vb-pages').addEventListener('click', (e) => {
    const b = e.target.closest('button[data-p]');
    if (b) goToPage(+b.dataset.p);
  });
  // 跳頁：輸入太大會停在最後一頁，框裡換成實際到的頁碼
  function commitGoto() {
    const goto = $('#vb-goto');
    const p = parseInt(goto.value, 10);
    if (p > 0 && p !== VB.page) goToPage(p);
    goto.value = String(VB.page);
  }
  $('#vb-goto').addEventListener('keydown', (e) => {
    // Enter 或 Tab 確認時焦點留在框裡：接著打的字不會跑去別的地方，頁面也不會因為焦點移到下一個欄位又被捲下去
    if (e.key === 'Enter' || (e.key === 'Tab' && e.target.value !== String(VB.page))) {
      e.preventDefault();
      commitGoto();
      e.target.select();
    }
  });
  $('#vb-goto').addEventListener('change', commitGoto);
  $('#vb-size').addEventListener('change', (e) => {
    const first = (VB.page - 1) * pageSize();          // 換每頁筆數時，盡量停在原來看到的那一批
    S.settings.vocabPageSize = +e.target.value;
    saveSettings();
    goToPage(Math.floor(first / pageSize()) + 1);
  });
  $('#vb-note').addEventListener('click', async (e) => {
    const build = e.target.closest('[data-dict-build]');
    if (build) {
      build.disabled = true;
      if (await buildDict(build.dataset.dictBuild)) loadVocabPage();
      else build.disabled = false;
      return;
    }
    if (!e.target.closest('#vb-gloss')) return;
    try {
      await api('POST', '/api/glosses', { scope: 'library' });
      toast('已排進佇列，翻好後會自動換成中文');
      refreshSoon();
      setTimeout(renderVocabNote, 1600);
    } catch (err) { toast(err.message, true); }
  });
  $('#vb-list').addEventListener('click', (e) => {
    const row = e.target.closest('.vb-row[data-k]');
    if (!row) return;
    // 剛標掉一列、底下的列補上來的那一下：滑鼠沒移動的連點不算，免得標到下一個字
    const g = VB.guard;
    if (g && performance.now() < g.until && Math.abs(e.clientX - g.x) < 8 && Math.abs(e.clientY - g.y) < 8) return;
    const stBtn = e.target.closest('[data-st]');
    if (stBtn) {
      if (e.detail > 1) return;                        // 連點兩下只算第一下
      VB.guard = { x: e.clientX, y: e.clientY, until: 0 };
      toggleStatus(row.dataset.k, +stBtn.dataset.st);
    } else openWord(row.dataset.k);
  });

  /* ---------- 單字詳細視窗 ---------- */

  const dlgWord = $('#dlg-word');
  const WD = { key: '', item: null, detail: null, groups: null, ex: [], nav: [], req: 0, allSenses: false };
  const SENSES_SHOWN = 6;

  function openWord(key) {
    const d = VB.data;
    if (!d || !key) return;
    const first = !dlgWord.open;
    WD.key = key;
    WD.item = d.items.find((x) => x.k === key) || null;
    WD.detail = null;
    WD.groups = null;
    WD.allSenses = false;
    // 方向鍵走的順序在開視窗時定下來，中途標記單字不會讓它從清單裡消失
    if (first || !WD.nav.includes(key)) WD.nav = VB.nav.includes(key) ? VB.nav.slice() : [key];
    $('#wd-title').textContent = WD.item ? WD.item.d : key.split(':').pop();
    $('#wd-detail').innerHTML = '<div class="vb-hint">讀取中…</div>';
    $('#wd-ex').innerHTML = '';
    renderWordStatus();
    const body = dlgWord.querySelector('.dlg-body');
    if (first) {
      dlgWord.showModal();
      body.focus();                                      // 不要把游標停在上一個／下一個上面
      // 已經開著的小視窗、提示要搬進對話框，不然會被對話框擋住點不到
      if (!mini.hidden) moveMini(dlgWord);
      if (!undoBox.hidden) toastHost().prepend(undoBox);
    }
    body.scrollTop = 0;
    fetchWord(key, d.lang);
  }

  async function fetchWord(key, lang) {
    const req = ++WD.req;
    const q = new URLSearchParams({ key, lang, media: S.settings.vocabScope || '' });
    const [detail, ex] = await Promise.all([
      api('GET', `/api/lexeme?key=${encodeURIComponent(key)}`).catch(() => null),
      api('GET', `/api/vocab/examples?${q}`).catch(() => null),
    ]);
    if (req !== WD.req) return;
    WD.detail = detail;
    WD.groups = ex ? ex.groups || [] : [];
    renderWord();
  }

  function renderWord() {
    $('#wd-detail').innerHTML = wordDetailHtml();
    $('#wd-ex').innerHTML = wordExamplesHtml();
    renderWordStatus();
  }

  function wordDetailHtml() {
    const d = WD.detail, it = WD.item, lang = VB.data?.lang;
    const word = d?.d || it?.d || '';
    const reading = (d ? d.r : it?.r) || '';
    const extra = [];
    if (d?.accent) extra.push(`<span class="wp-acc" title="重音（辭書形）">[${esc(d.accent)}]</span>`);
    if (d?.pn) extra.push(`<span class="wp-read">/${esc(d.pn)}/</span>`);
    if (d?.bopomofo) extra.push(`<span class="wp-read">${esc(d.bopomofo)}</span>`);
    const tags = tagsHtml({ lv: d?.lv || it?.lv, common: d?.common ?? it?.common, misc: d?.misc,
                            grp: it?.grp || d?.grp, upos: d?.upos, phrase: it?.ph || d?.ph });
    let html = `<div class="wp-head wd-word"${lang === 'ja' ? ' lang="ja"' : ''}><span class="wp-word">${esc(word)}</span>`
      + `${reading ? `<span class="wp-read">${esc(reading)}</span>` : ''}${extra.join('')}${tags}</div>`;
    if (!d) return `${html}<div class="vb-hint">讀取中…</div>`;
    if (lang === 'ja' && d.senses && d.senses.length) {
      // する、ある這種常用詞有二三十個義項，全部攤開會把例句擠到很下面：先顯示前面幾個，其餘點一下展開
      const cut = !WD.allSenses && d.senses.length > SENSES_SHOWN + 2 ? SENSES_SHOWN : d.senses.length;
      html += '<ol class="wp-senses">' + d.senses.slice(0, cut).map((s) => {
        const labels = [...new Set((s.pos || []).map(posZh))].concat((s.misc || []).map((m) => MISC_ZH[m]).filter(Boolean));
        const main = s.zh ? `${esc(s.zh)}<div class="wp-en">${esc(s.en)}</div>` : `${esc(s.en)}<span class="wp-src">英</span>`;
        return `<li>${labels.length ? `<span class="wp-pos">${esc(labels.join('・'))}</span>` : ''}${main}</li>`;
      }).join('') + '</ol>';
      if (cut < d.senses.length) {
        html += `<button type="button" class="btn small wd-more" data-more>還有 ${d.senses.length - cut} 個解釋</button>`;
      }
      const forms = [...new Set([...(d.forms || []), ...(d.readings || [])])].filter((f) => f !== word && f !== reading);
      if (forms.length) html += `<div class="wp-meta">其他寫法：${esc(forms.slice(0, 8).join('、'))}</div>`;
    } else if (lang === 'en') {
      const lines = sortLines(d.lines, d.upos);
      html += lines.length
        ? `<ul class="wp-lines">${lines.map((l) => `<li>${esc(l)}</li>`).join('')}</ul>`
        : glossHtml(it?.g || '', it?.gs || '');
      const bits = [];
      if (d.z) bits.push(`詞頻 ${d.z.toFixed(1)}`);
      if (d.collins) bits.push(`柯林斯 ${'★'.repeat(d.collins)}`);
      if (d.oxford) bits.push('牛津 3000');
      if (d.toefl) bits.push('托福');
      if (d.ielts) bits.push('雅思');
      if (d.sp) bits.push(`口語：${d.sp}`);
      if (bits.length) html += `<div class="wp-meta">${esc(bits.join(' · '))}</div>`;
    } else if (lang === 'zh' && d.defs && d.defs.length) {
      html += `<ol class="wp-senses">${d.defs.map((x) => `<li>${esc(x)}</li>`).join('')}</ol>`;
    } else {
      html += glossHtml(it?.g || '', it?.gs || '');
    }
    // 次數跟列表用同一份（每部影片只算最新那條字幕），不然視窗和列表的數字會對不起來
    if (it && VB.data?.media) {
      html += `<div class="wp-meta">這部影片出現 ${it.n.toLocaleString()} 次</div>`;
    } else {
      const n = it?.n || d.total_count || 0;
      const v = it?.v || d.total_videos || 0;
      if (n) html += `<div class="wp-meta">片庫裡出現 ${n.toLocaleString()} 次${v ? `、${v} 部影片` : ''}</div>`;
    }
    return html;
  }

  function wordExamplesHtml() {
    if (!WD.groups) return '<div class="wd-sec"><div class="wp-label">例句</div><div class="vb-hint">讀取例句…</div></div>';
    WD.ex = [];
    if (!WD.groups.length) return '<div class="wd-sec"><div class="wp-label">例句</div><div class="vb-hint">字幕裡找不到例句</div></div>';
    const lang = VB.data?.lang;
    const body = WD.groups.map((g) => {
      const m = mediaById(g.media_id);
      const title = m ? titleParts(m).main : g.title;
      const rows = g.items.map((x) => {
        const n = WD.ex.push({ media: g.media_id, start: x.start, end: x.end, text: x.text, hl: x.hl, ruby: x.ruby,
                               zh: x.zh, title, lang }) - 1;
        return `<div class="vb-exi"><time>${fmtTime(x.start)}</time>`
          + `<span class="vb-sent"${lang === 'ja' ? ' lang="ja"' : ''}>${sentHtml(x.text, x.hl, x.ruby)}`
          + `${x.zh ? `<span class="vb-zh">${esc(x.zh)}</span>` : ''}</span>`
          + `<span class="vb-exacts"><button type="button" class="btn small" data-ex-jump="${n}">跳到影片</button>`
          + `<button type="button" class="btn small" data-ex-mini="${n}">小視窗播放</button></span></div>`;
      }).join('');
      return `<div class="vb-exg"><div class="vb-ext"><b>${esc(title)}</b><span>${g.count} 次</span></div>${rows}</div>`;
    }).join('');
    return `<div class="wd-sec"><div class="wp-label">例句</div>${body}</div>`;
  }

  function renderWordStatus() {
    $('#wd-status').innerHTML = statusButtons(WD.key);
    const i = WD.nav.indexOf(WD.key);
    $('#wd-prev').disabled = i <= 0;
    $('#wd-next').disabled = i < 0 || i >= WD.nav.length - 1;
    $('#wd-hint').textContent = `N 不會 · X 已會 · I 忽略 · ↑↓ 換單字`
      + (i >= 0 && WD.nav.length > 1 ? `（第 ${(i + 1).toLocaleString()} / ${WD.nav.length.toLocaleString()} 個）` : '');
  }

  function stepWord(dir) {
    const i = WD.nav.indexOf(WD.key);
    const next = i < 0 ? null : WD.nav[i + dir];
    if (!next) return;
    const j = VB.nav.indexOf(next);                  // 目前列表裡的位置（標記過的字可能已經被篩掉）
    if (j >= 0) {
      const page = Math.floor(j / pageSize()) + 1;
      if (page !== VB.page) goToPage(page);
    }
    openWord(next);
  }

  $('#wd-status').addEventListener('click', (e) => {
    const b = e.target.closest('[data-st]');
    if (b && e.detail < 2) toggleStatus(b.dataset.key, +b.dataset.st);
  });
  $('#wd-detail').addEventListener('click', (e) => {
    if (!e.target.closest('[data-more]')) return;
    WD.allSenses = true;
    $('#wd-detail').innerHTML = wordDetailHtml();
  });
  $('#wd-ex').addEventListener('click', (e) => {
    const jump = e.target.closest('[data-ex-jump]');
    const play = e.target.closest('[data-ex-mini]');
    if (jump) {
      const x = WD.ex[+jump.dataset.exJump];
      dlgWord.close();
      if (x) jumpToMedia(x.media, x.start);
    } else if (play) {
      openMini(WD.ex[+play.dataset.exMini]);
    }
  });
  $('#wd-prev').addEventListener('click', () => stepWord(-1));
  $('#wd-next').addEventListener('click', () => stepWord(1));
  dlgWord.addEventListener('close', () => {
    WD.req += 1;
    if (dlgWord.contains(undoBox)) toastHost().prepend(undoBox);   // 「復原」提示留著，關掉視窗後還按得到
    if (mini.parentElement === dlgWord) moveMini(topHost());   // 小視窗留著繼續播
  });
  // 視窗開著時的鍵盤：app.js 的快捷鍵看到有對話框開著就不會動作，不會打架
  document.addEventListener('keydown', (e) => {
    if (!dlgWord.open || e.ctrlKey || e.metaKey || e.altKey || e.isComposing) return;
    if (e.target && e.target.closest && e.target.closest('input, select, textarea')) return;
    const k = e.key.length === 1 ? e.key.toLowerCase() : e.key;
    if (KEY_ST[k]) { e.preventDefault(); if (!e.repeat) toggleStatus(WD.key, KEY_ST[k]); }   // 方向鍵可以按住連續換，狀態鍵不行
    else if (k === 'ArrowUp' || k === 'ArrowLeft') { e.preventDefault(); stepWord(-1); }
    else if (k === 'ArrowDown' || k === 'ArrowRight') { e.preventDefault(); stepWord(1); }
  });

  /* ---------- 小視窗播放（只播一句） ---------- */

  const mini = $('#mini');
  const mv = $('#mini-video');
  const miniLoad = $('#mini-load');
  const MINI = { ex: null, raf: 0, loadTimer: 0, onMeta: null };
  const MINI_MIN = { w: 300, h: 280 };                 // 再小的話影片和字幕都看不清楚
  // 對話框開著的時候，浮動元素要放進對話框裡才點得到（modal 會擋住外面的東西）；全螢幕時要放進全螢幕的元素
  const topHost = () => document.querySelector('dialog[open]') || document.fullscreenElement || document.body;
  const clamp = (v, lo, hi) => Math.min(Math.max(v, lo), Math.max(lo, hi));

  // 只負責擺到畫面上（擠進目前的視窗大小），不存設定：視窗變小時只是暫時擠進來，不要蓋掉記住的位置
  function placeMini(box) {
    const w = clamp(Math.round(box.w), MINI_MIN.w, innerWidth - 8);
    const h = clamp(Math.round(box.h), MINI_MIN.h, innerHeight - 8);
    const x = clamp(Math.round(box.x), 4, innerWidth - w - 4);
    const y = clamp(Math.round(box.y), 4, innerHeight - h - 4);
    mini.style.left = `${x}px`;
    mini.style.top = `${y}px`;
    mini.style.width = `${w}px`;
    mini.style.height = `${h}px`;
    return { x, y, w, h };
  }

  // 使用者拖過才記住。放在右半邊的記離右邊的距離，瀏覽器視窗變寬之後還是貼在右側
  function saveMiniBox({ x, y, w, h }) {
    S.settings.miniBox = x + w / 2 > innerWidth / 2 ? { r: innerWidth - x - w, y, w, h } : { x, y, w, h };
    saveSettings();
  }

  // 「載入中」：換影片時馬上出現；循環跳回句首這種很快就好的跳轉，等一下還沒好才出現，不會一閃一閃
  function miniLoading(on, text = '載入中…') {
    clearTimeout(MINI.loadTimer);
    if (!on) { miniLoad.hidden = true; return; }
    miniLoad.textContent = text;
    if (on === 'now') miniLoad.hidden = false;
    else MINI.loadTimer = setTimeout(() => { miniLoad.hidden = false; }, 250);
  }

  function miniBox() {
    const b = S.settings.miniBox;
    if (b && +b.w && +b.h && (b.x != null || b.r != null)) {
      const w = +b.w, h = +b.h;
      return { w, h, x: b.r != null ? innerWidth - w - +b.r : +b.x, y: +b.y || 0 };
    }
    // 沒拖過：單字視窗開著時放在視窗右邊的空白，不要蓋到視窗裡的按鈕。
    // 空間不夠就縮到最小、貼右邊，上緣對齊標題列下面：蓋到的只有可以捲動的內容，標題列的關閉和底下的狀態按鈕都不會被蓋
    const gap = 12, dlg = document.querySelector('dialog[open]');
    const w = 400, h = 330;
    if (dlg) {
      const r = dlg.getBoundingClientRect();
      const room = innerWidth - r.right - gap * 2;
      if (room >= MINI_MIN.w) return { x: r.right + gap, y: Math.max(gap, r.top), w: Math.min(w, room), h };
      const head = dlg.querySelector('.dlg-head')?.getBoundingClientRect();
      return { x: innerWidth - MINI_MIN.w - gap, y: (head ? head.bottom : r.top) + gap, w: MINI_MIN.w, h: MINI_MIN.h };
    }
    return { x: innerWidth - w - 24, y: 72, w, h };
  }

  // 搬家（進出對話框、全螢幕）。Chrome 把正在播的 video 從 DOM 拿下來會自己暫停，搬完再接著播
  function moveMini(host) {
    if (mini.parentElement === host) return;
    const playing = !mv.paused;
    try {
      if (host.moveBefore) host.moveBefore(mini, null); else host.appendChild(mini);
    } catch { host.appendChild(mini); }
    if (playing && mv.paused) mv.play().catch(() => {});
  }

  function openMini(ex) {
    if (!ex || !ex.media) return;
    MINI.ex = ex;
    if (!video.paused) video.pause();                 // 不要兩個影片一起出聲
    moveMini(topHost());
    placeMini(miniBox());
    mini.hidden = false;
    const m = mediaById(ex.media);
    const title = ex.title || (m ? titleParts(m).main : '');
    $('#mini-title').textContent = `${fmtTime(ex.start)}　${title}`;
    $('#mini-title').title = title;
    const sub = $('#mini-sub');
    sub.innerHTML = sentHtml(ex.text, ex.hl, ex.ruby) + (ex.zh ? `<span class="vb-zh">${esc(ex.zh)}</span>` : '');
    if (ex.lang === 'ja') sub.setAttribute('lang', 'ja'); else sub.removeAttribute('lang');
    sub.scrollTop = 0;
    $('#mini-loop').classList.toggle('on', !!S.settings.miniLoop);
    // 音量、速度跟主播放器目前的設定一樣
    mv.volume = S.settings.volume;
    mv.muted = S.settings.muted;
    const src = `/media/${ex.media}/video?v=${m && m.has_proxy ? 'p' : 'o'}`;
    if (mv.dataset.src !== src) {
      mv.dataset.src = src;
      miniLoading('now');
      mv.src = src;        // preload="metadata"：只拿檔頭，設 currentTime 之後瀏覽器才用 Range 去要那一段
    }
    mv.defaultPlaybackRate = mv.playbackRate = +S.settings.rate || 1;
    playSentence();
  }

  function playSentence() {
    const ex = MINI.ex;
    if (!ex) return;
    // 上次讀檔失敗（檔案被移走、伺服器剛好重啟）：重新讀，不然會一直等不到檔頭
    if (mv.error && mv.dataset.src) {
      miniLoading('now');
      mv.src = mv.dataset.src;
    }
    const go = () => {
      MINI.onMeta = null;
      if (MINI.ex !== ex) return;                      // 讀檔頭的時候又換了一句
      try { mv.currentTime = Math.max(0, ex.start - 0.05); } catch { /* 還沒準備好 */ }
      mv.play().catch(() => {});
    };
    if (MINI.onMeta) mv.removeEventListener('loadedmetadata', MINI.onMeta);   // 重播按很多次也只等一次
    MINI.onMeta = null;
    if (mv.readyState >= 1) go();
    else {
      miniLoading('now');
      MINI.onMeta = go;
      mv.addEventListener('loadedmetadata', go, { once: true });
    }
  }

  // 關掉時放掉影片：連線和解碼器不要一直留著，下次打開重新讀檔頭
  function closeMini() {
    mini.hidden = true;
    miniLoading(false);
    cancelAnimationFrame(MINI.raf);
    MINI.raf = 0;
    MINI.ex = null;
    if (MINI.onMeta) mv.removeEventListener('loadedmetadata', MINI.onMeta);
    MINI.onMeta = null;
    if (!mv.getAttribute('src')) return;
    mv.pause();
    mv.removeAttribute('src');
    delete mv.dataset.src;
    mv.load();
  }

  // 播到這一句的結尾就停（開了循環就跳回句首）
  function miniCheckEnd() {
    const ex = MINI.ex;
    if (!ex || mini.hidden || mv.seeking) return;
    // 影片本身播完（這句剛好是最後一句）時瀏覽器已經先暫停了，循環的話還是要接回去
    if (mv.paused && !mv.ended) return;
    if (!mv.ended && mv.currentTime < ex.end - 0.02) return;
    if (S.settings.miniLoop) {
      mv.currentTime = Math.max(0, ex.start - 0.05);
      if (mv.paused) mv.play().catch(() => {});
    } else if (!mv.paused) mv.pause();
  }
  // timeupdate 一秒只有四次，等它會多播出下一句的開頭，所以播放中每一個畫面都對一次時間
  function miniTick() {
    MINI.raf = 0;
    miniCheckEnd();
    if (!mv.paused && !mini.hidden) MINI.raf = requestAnimationFrame(miniTick);
  }
  mv.addEventListener('play', () => { if (!MINI.raf) MINI.raf = requestAnimationFrame(miniTick); });
  mv.addEventListener('timeupdate', miniCheckEnd);    // 分頁在背景時畫面不更新，靠這個
  mv.addEventListener('ended', miniCheckEnd);
  for (const ev of ['waiting', 'seeking']) mv.addEventListener(ev, () => miniLoading(true));
  for (const ev of ['canplay', 'playing', 'seeked']) mv.addEventListener(ev, () => miniLoading(false));
  mv.addEventListener('error', () => { if (mv.getAttribute('src')) miniLoading('now', '讀不到影片檔'); });
  // 點畫面：播放中就暫停；停著就接著播，已經播完這句就從頭再播
  function toggleMini() {
    const ex = MINI.ex;
    if (!ex) return;
    if (!mv.paused) mv.pause();
    else if (mv.error || mv.currentTime >= ex.end - 0.1 || mv.currentTime < ex.start - 0.3) playSentence();
    else mv.play().catch(() => {});
  }
  mv.addEventListener('click', toggleMini);
  // 主播放器開始播就把小視窗停下來，不要兩邊一起出聲
  video.addEventListener('play', () => { if (!mv.paused) mv.pause(); });
  document.addEventListener('fullscreenchange', () => {
    if (!mini.hidden) moveMini(topHost());
    if (!undoBox.hidden) toastHost().prepend(undoBox);
  });
  // 單字頁（或單字視窗）上小視窗開著時，空白鍵控制小視窗；播放器頁的空白鍵照舊控制主影片
  document.addEventListener('keydown', (e) => {
    if (e.key !== ' ' || mini.hidden || !MINI.ex || e.repeat || e.ctrlKey || e.metaKey || e.altKey) return;
    if (!VB.open && !dlgWord.open) return;
    if (e.target.closest && e.target.closest('input, select, textarea, button, a[href]')) return;
    e.preventDefault();
    toggleMini();
  });

  $('#mini-close').addEventListener('click', closeMini);
  $('#mini-replay').addEventListener('click', playSentence);
  $('#mini-loop').addEventListener('click', () => {
    S.settings.miniLoop = !S.settings.miniLoop;
    saveSettings();
    $('#mini-loop').classList.toggle('on', S.settings.miniLoop);
    if (S.settings.miniLoop && mv.paused) playSentence();
  });
  $('#mini-jump').addEventListener('click', () => {
    const ex = MINI.ex;
    closeMini();
    if (dlgWord.open) dlgWord.close();
    if (ex) jumpToMedia(ex.media, ex.start);
  });

  // 拖標題列搬家、拖四個角或四邊改大小
  function grab(e, dir) {
    // 手機上小視窗固定在底部（vocab.css 設 --mini-sheet），不能拖
    if (e.button !== 0 || getComputedStyle(mini).getPropertyValue('--mini-sheet').trim() === '1') return;
    e.preventDefault();
    const r = mini.getBoundingClientRect();
    const box = { x: r.left, y: r.top, w: r.width, h: r.height };
    const sx = e.clientX, sy = e.clientY;
    const el = e.currentTarget;
    el.setPointerCapture(e.pointerId);
    let placed = null;
    const move = (ev) => {
      const dx = ev.clientX - sx, dy = ev.clientY - sy;
      if (!dir) { placed = placeMini({ ...box, x: box.x + dx, y: box.y + dy }); return; }
      // 改大小：對面那一邊固定不動，拉的那一邊停在最小尺寸或畫面邊緣
      const right = box.x + box.w, bottom = box.y + box.h;
      const next = { ...box };
      if (dir.includes('e')) next.w = clamp(right + dx, box.x + MINI_MIN.w, innerWidth - 4) - box.x;
      if (dir.includes('s')) next.h = clamp(bottom + dy, box.y + MINI_MIN.h, innerHeight - 4) - box.y;
      if (dir.includes('w')) { next.x = clamp(box.x + dx, 4, right - MINI_MIN.w); next.w = right - next.x; }
      if (dir.includes('n')) { next.y = clamp(box.y + dy, 4, bottom - MINI_MIN.h); next.h = bottom - next.y; }
      placed = placeMini(next);
    };
    const up = () => {
      el.removeEventListener('pointermove', move);
      el.removeEventListener('pointerup', up);
      el.removeEventListener('pointercancel', up);
      if (placed) saveMiniBox(placed);
    };
    el.addEventListener('pointermove', move);
    el.addEventListener('pointerup', up);
    el.addEventListener('pointercancel', up);
  }
  $('#mini-head').addEventListener('pointerdown', (e) => { if (!e.target.closest('button')) grab(e, ''); });
  for (const g of $$('.mini-grip')) g.addEventListener('pointerdown', (e) => grab(e, g.dataset.dir));
  addEventListener('resize', () => { if (!mini.hidden) placeMini(miniBox()); });

  /* ---------- 設定頁 ---------- */

  function renderLookupSettings() {
    $('#set-lookup-on').checked = active();
    $('#set-lookup-trigger').value = S.settings.lookupTrigger;
    $('#set-hover-delay').value = String(S.settings.hoverDelay);
    $('#set-lookup-pause').value = S.settings.lookupPause;
    $('#set-lookup-pause-asmr').value = S.settings.lookupPauseAsmr;
    for (const id of ['#set-lookup-trigger', '#set-hover-delay', '#set-lookup-pause', '#set-lookup-pause-asmr']) {
      $(id).disabled = !active();
    }
    $('#set-hover-delay').disabled = !active() || S.settings.lookupTrigger !== 'hover';
  }

  $('#set-lookup-on').addEventListener('change', (e) => {
    S.settings.lookupOn = e.target.checked;
    saveSettings();
    hideCard();
    renderLookupSettings();
    renderTranscript();
    updateSubs(true);
    if (active()) for (const slot of [1, 2]) onCuesLoaded(slot, S.trackSel[slot]);
  });
  for (const [id, key, num] of [['#set-lookup-trigger', 'lookupTrigger'], ['#set-hover-delay', 'hoverDelay', true],
    ['#set-lookup-pause', 'lookupPause'], ['#set-lookup-pause-asmr', 'lookupPauseAsmr']]) {
    $(id).addEventListener('change', (e) => {
      S.settings[key] = num ? +e.target.value : e.target.value;
      saveSettings();
      renderLookupSettings();
    });
  }
  $('#set-auto-glosses').addEventListener('change', (e) => saveValue('auto_glosses', e.target.checked));
  $('#btn-glosses').addEventListener('click', async () => {
    try {
      await api('POST', '/api/glosses', { scope: 'library' });
      toast('已排進佇列，翻好後查字和單字頁會自動換成中文');
      refreshSoon();
      refreshDictCard();
    } catch (err) { toast(err.message, true); }
  });

  // 設定頁字典卡片：data 是上一次的 /api/dicts；sig 是字典任務（和 CKIP 下載）的狀態，變了才重新讀
  // status：只看任務和狀態（開始、結束時重新讀 /api/dicts）；sig 另外帶進度（只重畫）
  const DICTS = { data: null, sig: '', status: '', loading: false };
  const dictJobsSig = (withProgress = true) => JSON.stringify(S.jobs
    .filter((j) => j.type === 'dict' || (j.type === 'model' && (j.params || {}).model === 'ckip'))
    .map((j) => [j.id, j.status, withProgress ? Math.round((j.progress || 0) * 100) : 0]));

  $('#dict-status').addEventListener('click', async (e) => {
    const b = e.target.closest('[data-dict-build], [data-zh-add]');
    if (!b) return;
    if (b.dataset.dictBuild) {
      const lang = b.dataset.dictBuild;
      const force = !!b.dataset.force;
      if (force && !(await askConfirm({ title: `重建${LANG_NAME[lang]}字典`, text: '重新下載來源再建一次，要幾分鐘。', ok: '重建' }))) return;
      b.disabled = true;
      await buildDict(lang, force);
    } else {
      // 加入中文：先排中文辭典，再下載 CKIP 斷詞模型（已經有的、已經在排的跳過）
      b.disabled = true;
      const d = DICTS.data || { zh: {} };
      try {
        if (!d.zh.dict && !dictJob('zh')) {
          try {
            await api('POST', '/api/dicts/zh/build', { force_download: false });
          } catch (err) {
            if (err.code !== 'duplicate') throw err;
          }
        }
        if (!d.zh.models && !ckipJob()) await api('POST', '/api/models/ckip/download');
        toast('已排進佇列，裝好後單字頁的中文分頁就能用');
        await refreshSoon();
      } catch (err) { toast(err.message, true); }
    }
    refreshDictCard();
  });

  // 字典三列：狀態文字，最右邊放按鈕。進度直接看佇列（S.jobs），不用每次都重新讀 /api/dicts
  function renderDictRows(d) {
    const row = (name, ok, text, btn = '') => `<div class="dict-row"><b>${name}</b><span class="${ok ? '' : 'off'}">${text}</span>${btn}</div>`;
    // 伺服器有「建立字典」的 API 才放按鈕（新版的 /api/dicts 每本字典都帶 job 欄位）；舊版照舊叫人跑 tools 底下的腳本
    const canBuild = SU.api || 'job' in d.ja;
    const buildBtn = (lang, label, force) => `<button type="button" class="btn small${force ? '' : ' primary'}" data-dict-build="${lang}"${force ? ' data-force="1"' : ''}>${label}</button>`;
    const busy = (lang) => {
      const j = dictJob(lang) || (d[lang] && d[lang].job);
      if (!canBuild || !j || !['queued', 'running'].includes(j.status)) return '';
      return j.status === 'queued' ? '等待建立' : `建立中 ${Math.round((j.progress || 0) * 100)}%`;
    };
    const est = (lang, s) => fmtRemain(Math.max(60, (d[lang] && d[lang].est_s) || s));
    // 要下載多少：後端給 download_bytes（原始資料的大小），舊的回應沒有就用今天上游的大小
    const dl = (lang, mb) => fmtSize((d[lang] && d[lang].download_bytes) || mb * 1024 ** 2);
    // 最近一次建立失敗（字典還沒好時才說）：滑鼠移上去看原因，按鈕改成「重試」
    const failed = (lang) => {
      const last = S.jobs.filter((j) => j.type === 'dict' && (j.params || {}).lang === lang && j.status !== 'queued' && j.status !== 'running')
        .sort((a, b) => (b.finished_at || 0) - (a.finished_at || 0))[0];
      return canBuild && last && last.status === 'failed' ? last : null;
    };
    const failText = (j) => `<span title="${esc(j.error || '')}">上次建立失敗</span>`;
    // 第一次自動安裝會自己再試的（等網路、稍後重試）照進度頁的說法，不放按鈕
    const autoRetry = (lang) => {
      const it = ((SU.data && SU.data.items) || []).find((i) => i.key === `dict:${lang}`);
      if (!it || !SU.summary || SU.summary.status !== 'installing') return '';
      return it.state === 'net_wait' ? '等網路恢復' : it.state === 'retry_wait' ? '連線中斷，等一下重試' : '';
    };
    const jaBusy = busy('ja') || autoRetry('ja'), enBusy = busy('en') || autoRetry('en'), zhBusy = busy('zh');
    const jaFail = !d.ja.ready && failed('ja'), enFail = !d.en.ready && failed('en');
    const ckip = canBuild && ckipJob();
    const zhOk = d.zh.models && d.zh.dict;
    const zhNeed = (d.zh.models ? 0 : 776 * 1024 ** 2) + (d.zh.dict ? 0 : d.zh.download_bytes || 14 * 1024 ** 2);
    let zhText = d.zh.models
      ? `CKIP 斷詞已安裝${d.zh.dict ? `，教育部辭典 ${d.zh.size_mb} MB` : '，還沒有中文辭典（只統計不解釋）'}`
      : 'CKIP 斷詞模型還沒安裝，到「模型管理」下載後單字頁的中文分頁才能用';
    if (canBuild && !d.zh.models) zhText = '還沒安裝';
    if (zhBusy) zhText = zhBusy;
    else if (ckip) zhText = ckip.status === 'paused' ? '下載已暫停' : `下載中 ${Math.round((ckip.progress || 0) * 100)}%`;
    $('#dict-status').innerHTML = [
      row('日文', d.ja.ready, jaBusy || (d.ja.ready
        ? `JMdict ${esc(d.ja.source_date)}，${(+d.ja.entries).toLocaleString()} 個條目（${d.ja.size_mb} MB）`
        : jaFail ? failText(jaFail)
          : canBuild ? `還沒建立（下載 ${dl('ja', 14)}，${est('ja', 20)}）` : '還沒建立，請執行 tools\\build_dict_ja.py'),
      !canBuild || jaBusy ? '' : d.ja.ready ? buildBtn('ja', '重建', true) : buildBtn('ja', jaFail ? '重試' : '建立')),
      row('英文', d.en.ready && d.en.nlp, enBusy || (d.en.ready
        ? `ECDICT ${(+d.en.entries).toLocaleString()} 個詞條（${d.en.size_mb} MB）${d.en.nlp ? '' : '，但 spaCy 沒有安裝'}`
        : enFail ? failText(enFail)
          : canBuild ? `還沒建立（下載 ${dl('en', 49)}，${est('en', 300)}）` : '還沒建立，請執行 tools\\build_dict_en.py'),
      !canBuild || enBusy ? '' : d.en.ready ? buildBtn('en', '重建', true) : buildBtn('en', enFail ? '重試' : '建立')),
      row('中文', d.zh.models, zhText,
        !canBuild || zhOk || zhBusy || ckip ? ''
          : `<button type="button" class="btn small primary" data-zh-add>加入中文（下載 ${zhNeed >= 512 * 1024 ** 2 ? gbText(zhNeed) : fmtSize(zhNeed)}）</button>`),
    ].join('');
    DICTS.sig = dictJobsSig();
  }

  // full=false：字典任務開始或結束時只更新字典三列，不重算單字釋義
  async function refreshDictCard(full = true) {
    let d;
    DICTS.loading = true;
    try {
      // 自動安裝進行中：一起讀安裝狀態，字典列才知道哪一本會自己再試
      if (SU.summary && SU.summary.status === 'installing') await refreshSetup();
      d = await api('GET', '/api/dicts');
    } catch { return; } finally { DICTS.loading = false; }
    DICTS.data = d;
    DICTS.status = dictJobsSig(false);
    $('#dict-dir').textContent = d.dir;
    renderDictRows(d);
    if (!full) return;
    $('#set-auto-glosses').checked = !!d.glosses.auto;
    const btn = $('#btn-glosses');
    const hint = $('#glosses-hint');
    if (!d.glosses.translator) { btn.disabled = true; hint.textContent = '需要先下載 Hy-MT2-7B'; return; }
    if (d.glosses.queued) { btn.disabled = true; hint.textContent = '已經在佇列裡'; return; }
    hint.textContent = '計算中…';
    try {
      const p = await api('GET', '/api/glosses/pending');
      btn.disabled = !p.count || p.queued;
      hint.textContent = p.queued ? '已經在佇列裡'
        : p.count ? `還有 ${p.count.toLocaleString()} 個義項沒有中文，要${fmtRemain(p.count * 0.15)}（會載入 Hy-MT2-7B）`
          : '片庫裡的日文單字都有中文解釋了';
    } catch { hint.textContent = ''; }
  }

  /* ---------- 給 app.js 呼叫 ---------- */

  function onView(name) {
    if (name !== 'player') hideCard();
    VB.open = name === 'vocab';
    if (!VB.open) {
      clearTimeout(VB.pollTimer);
      if (dlgWord.open) dlgWord.close();
    }
    if (VB.open) openVocabPage();
    if (name === 'settings') { renderLookupSettings(); refreshDictCard(); }
  }

  function onData() {
    const done = S.jobs.filter((j) => j.type === 'glosses' && j.status === 'done').map((j) => j.finished_at || 0);
    const latest = done.length ? Math.max(...done) : 0;
    if (V.glossDone === null) { V.glossDone = latest; return; }
    if (latest > V.glossDone) {
      // 單字釋義任務剛做完：重新讀詞表，查字和單字頁就會換成中文
      V.glossDone = latest;
      V.detail.clear();
      for (const id of Object.keys(V.vocab)) loadVocab(id, true);
      if (VB.open) loadVocabPage();
      if (SET.open) refreshDictCard();
    }
    // 字典正在建立（或 CKIP 在下載）：設定頁的進度跟著更新，任務開始或結束時重新讀字典狀態
    if (SET.open && DICTS.data && !DICTS.loading) {
      if (dictJobsSig(false) !== DICTS.status) refreshDictCard(false);
      else if (dictJobsSig() !== DICTS.sig) renderDictRows(DICTS.data);
    }
    if (VB.open) renderVocabHead();
  }

  window.WL = {
    active, onCuesLoaded, onCueChange, pauseCheck, onKey, onView, onData,
    cardOpen: () => !pop.hidden,
    // 給 tests/ui_check.py 檢查用
    _look: LOOK, _v: V, _vb: VB, _wd: WD, _mini: MINI, _undo: UNDO, _jump: JUMP, hideCard, openWord, openMini, closeMini,
  };
})();
