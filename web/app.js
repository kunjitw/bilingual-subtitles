'use strict';

/* ================= 小工具 ================= */

const $ = (sel, el = document) => el.querySelector(sel);
const $$ = (sel, el = document) => [...el.querySelectorAll(sel)];
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

function fmtTime(sec) {
  if (!isFinite(sec) || sec < 0) sec = 0;
  const s = Math.floor(sec % 60), m = Math.floor(sec / 60) % 60, h = Math.floor(sec / 3600);
  return h ? `${h}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}` : `${m}:${String(s).padStart(2, '0')}`;
}
function fmtSize(b) {
  if (b >= 1024 ** 3) return (b / 1024 ** 3).toFixed(1) + ' GB';
  if (b >= 1024 ** 2) return (b / 1024 ** 2).toFixed(0) + ' MB';
  return Math.max(1, Math.round(b / 1024)) + ' KB';
}
function fmtRemain(sec) {
  if (sec < 60) return '不到 1 分鐘';
  if (sec < 3600) return `約 ${Math.round(sec / 60)} 分鐘`;
  return `約 ${(sec / 3600).toFixed(1)} 小時`;
}

const store = {
  get(key, fallback) {
    try { const v = localStorage.getItem('vs:' + key); return v == null ? fallback : JSON.parse(v); } catch { return fallback; }
  },
  set(key, value) { try { localStorage.setItem('vs:' + key, JSON.stringify(value)); } catch { /* 無痕模式等情況 */ } },
};

async function api(method, url, body) {
  const res = await fetch(url, {
    method,
    headers: body ? { 'Content-Type': 'application/json' } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) {
    let msg = res.statusText, data = null;
    try { data = await res.json(); msg = typeof data.detail === 'string' ? data.detail : '輸入的資料有誤'; } catch { /* 非 JSON */ }
    // code：後端的錯誤代碼（例如 plan_changed）；plan：安裝計畫變了時附上新的計畫
    const err = new Error(msg);
    err.status = res.status;
    err.code = (data && typeof data.code === 'string') ? data.code : '';
    if (data && data.plan) err.plan = data.plan;
    throw err;
  }
  return res.json();
}

// 回傳 true（按確定）、false（取消或關掉）；有 alt 時多一個按鈕，按了回傳 'alt'。cancel 給空字串就只剩確定鈕（純說明用）
function askConfirm({ title, text, items = [], ok = '確定', warn = false, alt = '', cancel = '取消' }) {
  return new Promise((resolve) => {
    const dlg = $('#dlg-confirm');
    $('#confirm-title').textContent = title;
    $('#confirm-text').textContent = text;
    $('#confirm-list').innerHTML = items.map((i) => `<li>${esc(i)}</li>`).join('');
    const btn = $('#confirm-ok');
    btn.textContent = ok;
    btn.classList.toggle('warn', warn);
    const altBtn = $('#confirm-alt');
    altBtn.textContent = alt;
    altBtn.hidden = !alt;
    $('#confirm-cancel').textContent = cancel;
    $('#confirm-cancel').hidden = !cancel;
    let done = false;
    const finish = (v) => { if (done) return; done = true; resolve(v); dlg.close(); };
    btn.onclick = () => finish(true);
    altBtn.onclick = () => finish('alt');
    dlg.addEventListener('close', () => finish(false), { once: true });
    dlg.showModal();
  });
}

let toastTimer = 0;
function toast(msg, bad = false) {
  const el = $('#toast');
  el.textContent = msg;
  el.classList.toggle('bad', bad);
  el.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { el.hidden = true; }, bad ? 6000 : 3000);
}

/* ================= 狀態 ================= */

const S = {
  meta: null,
  media: [],
  jobs: [],
  gpu: null,
  sig: '',
  current: null,
  srcKey: '',
  trackSel: { 1: null, 2: null }, // 1 = 原文、2 = 從這條原文翻出來的翻譯
  showTr: true, // 要不要顯示翻譯（翻譯選單選「不顯示」才關掉），每部影片跟選擇一起記住
  cues: { 1: [], 2: [] },
  idx: { 1: -1, 2: -1 },
  hi: -1,
  pairs: [],
  primary: 1,
  loop: null,
  userScrollAt: 0,
  seekAt: -1e9,
  transcriptClickAt: -1e9,
  cueVer: { 1: '', 2: '' }, // 目前載入的字幕是哪個版本（見 cueVersion）
  settings: {},
};

// 預設外觀：原文和翻譯都放在影片外下方，字小一點，描邊加底色，影片外只保留一行高。
// 右側欄預設展開：新使用者第一件事是加影片，要看得到佇列進度（已存過設定的人照自己的）
const DEFAULT_SETTINGS = {
  place1: 'outBottom', place2: 'outBottom',
  size1: 3.7, size2: 2.5,
  pos: 1, gap: 0, style: 'both', bg: 70, width: 100,
  reserveLines: 1, useLetterbox: true,
  blur2: false, hide1: false, hide2: false, swapLines: false,
  rate: 1, volume: 1, muted: false, follow: true,
  autoplay: true, autoNext: true, repeat: 'off',
  furi: 'all', titleMode: 'both',
  hideLeft: false, hideRight: false, sideTab: 'transcript',
  addTranslate: true, resume: true,
};
// 這個程式需要的驅動和 CUDA 版本（torch cu130、llama.cpp CUDA 13），跟 app/syscheck.py 一致
const DRIVER_MIN = 580;
const CUDA_MIN = '13.0';
const cudaText = (g) => `CUDA ${(g && g.cuda) || '讀不到'}（需要 ${CUDA_MIN} 以上）`;

// 「字幕外觀」選單的恢復預設只重設這些，播放速度、音量、側欄、播放方式都不動
const LOOK_KEYS = ['place1', 'place2', 'size1', 'size2', 'pos', 'gap', 'style', 'bg', 'width', 'reserveLines', 'useLetterbox',
  'blur2', 'hide1', 'hide2', 'swapLines', 'furi'];

// 標題顯示：both 原文＋中文、original 只顯示原文、translated 有翻譯就只顯示中文
function titleParts(m) {
  const zh = m.title_zh && m.title_zh !== m.title ? m.title_zh : '';
  const mode = S.settings.titleMode;
  if (mode === 'original' || !zh) return { main: m.title, sub: '' };
  if (mode === 'translated') return { main: zh, sub: '' };
  return { main: m.title, sub: zh };
}

// 日文假名標註：cue.ruby 是 [起, 迄, 假名, 旗標] 的字元位置陣列
// 旗標 h：有多種讀法的詞；u：使用者改過；rt 空字串代表使用者指定不要標
// 查字（web/vocab.js）：cue.w 是 [起, 迄, key, …] 的詞單位，包成 span.w，假名放在詞裡面；英文片語成員帶 data-ph
function cueHtml(cue) {
  const text = cue.text || '';
  const ruby = S.settings.furi !== 'off' && cue.ruby ? cue.ruby : [];
  const seg = (a, b) => { // [a, b) 這段文字，含整個落在裡面的假名
    let html = '', p = a;
    for (const [s, e, rt, flag] of ruby) {
      if (s < p || e > b || e > text.length) continue;
      html += esc(text.slice(p, s));
      html += rt
        ? `<ruby data-s="${s}" data-e="${e}" class="${flag ? 'f-' + flag : ''}">${esc(text.slice(s, e))}<rt>${esc(rt)}</rt></ruby>`
        : `<span class="no-ruby" data-s="${s}" data-e="${e}">${esc(text.slice(s, e))}</span>`;
      p = e;
    }
    return html + esc(text.slice(p, b));
  };
  const words = window.WL?.active() && cue.w ? cue.w : [];
  if (!words.length) return seg(0, text.length);
  const phAt = new Map();
  (cue.ph || []).forEach(([, spans], k) => spans.forEach(([s]) => phAt.set(s, k)));
  let html = '', prev = 0;
  words.forEach(([s0, e0], k) => {
    // 假名區塊跨過詞的邊界（800人 整組標一個讀音）時，把詞撐大到包住它；撐大後跟前一個詞重疊就不包
    let s = s0, e = e0, grown = true;
    while (grown) {
      grown = false;
      for (const [rs, re] of ruby) {
        if (rs < e && re > s && (rs < s || re > e)) { s = Math.min(s, rs); e = Math.max(e, re); grown = true; }
      }
    }
    if (s < prev || e > text.length) return;
    const ph = phAt.has(s0) ? ` data-ph="${phAt.get(s0)}"` : '';
    html += seg(prev, s) + `<span class="w" data-k="${k}"${ph}>${seg(s, e)}</span>`;
    prev = e;
  });
  return html + seg(prev, text.length);
}

const hasRuby = (cues) => S.settings.furi !== 'off' && cues.some((c) => c.ruby && c.ruby.length);
const FURI_MODES = { all: '全部顯示', hover: '滑鼠移上去才顯示', off: '關閉' };
const PLACES = {
  inBottom: '影片內下方',
  inTop: '影片內上方',
  outBottom: '影片外下方',
  outTop: '影片外上方',
};

(function loadSettings() {
  const saved = store.get('settings', {});
  // 舊版只有一個字級設定，轉成兩行各自的字級
  if (saved.size && !saved.size1) {
    saved.size1 = saved.size;
    saved.size2 = Math.round(saved.size * 0.86 * 10) / 10;
  }
  if (saved.ruby === false && !saved.furi) saved.furi = 'off';
  S.settings = { ...DEFAULT_SETTINGS, ...saved };
})();
const saveSettings = () => store.set('settings', S.settings);

const JOB_LABEL = { transcribe: '轉字幕', translate: '翻譯', download: '下載', proxy: '轉相容格式', model: '下載模型', titles: '翻譯標題', glosses: '單字釋義', health: '檢查時間軸', dict: '建立字典' };
const DICT_LANG = { ja: '日文', en: '英文', zh: '中文' };
// 任務進度文字：階段名稱跟任務名稱一樣時（檢查時間軸）不重複顯示
const jobStage = (j) => (j.stage && j.stage !== JOB_LABEL[j.type] ? j.stage : '');
const LANG_SHORT = { 'zh-TW': '中', ja: '日', en: '英' };

const mediaById = (id) => S.media.find((m) => m.id === id);
const jobsOf = (mid) => S.jobs.filter((j) => j.media_id === mid);
const activeJobOf = (mid) => {
  const js = jobsOf(mid);
  return js.find((j) => j.status === 'running') || js.filter((j) => j.status === 'queued').sort((a, b) => a.position - b.position)[0];
};

function trackLabel(t, all) {
  const lang = S.meta.track_langs[t.lang] || t.lang;
  let label = t.kind === 'translation' ? `${lang}（${t.model} 翻譯）` : `${lang}（${t.model}）`;
  const same = all.filter((x) => x.lang === t.lang && x.kind === t.kind && x.model === t.model);
  if (same.length > 1) {
    const d = new Date(t.created_at * 1000);
    label += ` ${d.getMonth() + 1}/${d.getDate()} ${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}`;
  }
  return label;
}

/* ================= 輪詢 ================= */

async function poll() {
  try {
    const st = await api('GET', '/api/state');
    $('#offline').hidden = true;
    S.gpu = st.gpu;
    renderGpu();
    onSetupSummary(st);
    // 進度頁開著、或還沒判斷過要不要自動打開時，另外讀完整的安裝狀態
    if (SU.open || (!SU.decided && SU.api)) await refreshSetup();
    // 佇列暫停、繼續時任務本身沒變，也要重畫佇列（「暫停中」的標示）
    const sig = JSON.stringify([st.media, st.jobs, !!(st.gpu && st.gpu.queue_paused)]);
    if (sig !== S.sig) {
      S.sig = sig;
      S.media = st.media;
      S.series = st.series || [];
      S.jobs = st.jobs;
      onData();
    } else {
      renderQueueTimes();
    }
  } catch {
    $('#offline').hidden = false;
  }
  setTimeout(poll, 1500);
}

function onData() {
  renderLibrary();
  renderQueue();
  if (SET.open) refreshSettings();
  // 檢查時間軸的任務數變了（排進去或做完）才重算設定頁的「尚未檢查」數量
  if (SET.open && healthHintJobs !== S.jobs.filter((j) => j.type === 'health' && (j.status === 'queued' || j.status === 'running')).length) {
    refreshHealthHint();
  }
  window.WL?.onData();
  if (!S.current && S.media.length) {
    const fromHash = (location.hash.match(/^#m=([0-9a-f]+)/) || [])[1];
    const last = fromHash && mediaById(fromHash) ? fromHash : store.get('last', null);
    if (last && mediaById(last)) openMedia(last);
  }
  if (S.current) {
    const m = mediaById(S.current);
    if (!m) { closeMedia(); return; }
    renderDeck(m);
    syncVideoSource(m);
    syncTracks(m);
    renderTranscriptStatus(m);
    if ($('#dlg-tracks').open) renderTrackManager();
    if ($('#dlg-translate').open) updateTranslateNotice();
  }
  if (dlgAdd.open) updateUrlDup();
}

// 顯卡上載入著的模型（常駐的語音模型、翻譯模型），簡短名稱用頓號連起來
function gpuLoadedText(g, full = false) {
  const names = ((g && g.loaded) || []).map((m) => (full ? m.label : m.short) || m.label);
  return names.join('、');
}

function renderGpu() {
  const g = S.gpu || {};
  const el = $('#gpu');
  const busy = !!g.job_id;
  el.classList.toggle('busy', busy);
  el.classList.toggle('paused', !busy && !!g.queue_paused);
  const loaded = gpuLoadedText(g);
  let text = '顯卡閒置';
  if (busy) text = [g.stage, g.model].filter(Boolean).join(' · ') || '處理中';
  else if (g.queue_paused) text = '佇列已暫停';
  else if (loaded) text = `顯卡閒置 · 已載入 ${loaded}`;
  $('#gpu-text').textContent = text;
  el.title = loaded ? `已載入：${gpuLoadedText(g, true)}\n查看佇列` : '查看佇列';
  const info = g.info;
  $('#gpu-vram').textContent = info ? `${(info.used_mb / 1024).toFixed(1)} / ${Math.round(info.total_mb / 1024)} GB` : '';
  renderGpuNote();
  if (!$('#settings-page').hidden) renderGpuLoaded();
}

const freeGbText = (g) => (g && g.info ? `可用 ${(g.info.free_mb / 1024).toFixed(1)} GB` : '');

// 自動釋放的時間，給說明文字用
function idleReleaseText(g) {
  const secs = (g && g.idle_release_s) || 120;
  return secs >= 60 && secs % 60 === 0 ? `${secs / 60} 分鐘` : `${secs} 秒`;
}

// 佇列上方：顯卡載入了什麼、還剩多少顯存；按「釋放顯卡」暫停了佇列時說明怎麼繼續
function renderGpuNote() {
  const g = S.gpu || {};
  $('#btn-queue-resume').hidden = !g.queue_paused;
  const loaded = gpuLoadedText(g, true);
  const gpuLine = [loaded ? `顯卡已載入 ${loaded}` : '顯卡沒有載入模型', freeGbText(g)].filter(Boolean).join(' · ');
  const note = $('#queue-gpu-note');
  note.textContent = g.queue_paused ? `佇列已暫停，顯卡任務不會開始，按「繼續佇列」從中斷的地方接著做。${gpuLine}` : gpuLine;
  note.classList.toggle('paused', !!g.queue_paused);
  note.hidden = !g.info && !g.queue_paused;
}

// 設定頁顯示卡區：載入著哪些模型、釋放顯卡按鈕
function renderGpuLoaded() {
  const g = S.gpu || {};
  const text = gpuLoadedText(g, true);
  const free = freeGbText(g);
  const loaded = text
    ? `目前載入：${text}（顯卡沒事做滿 ${idleReleaseText(g)}自動釋放）`
    : `目前沒有載入模型${free ? `，${free}` : ''}`;
  // 在設定頁按「釋放顯卡」暫停了佇列：這裡也說清楚，並且可以直接繼續（手機上頂列看不到文字）
  $('#gpu-loaded').textContent = g.queue_paused ? `佇列已暫停，顯卡任務不會開始。${loaded}` : loaded;
  $('#gpu-loaded').classList.toggle('paused', !!g.queue_paused);
  $('#btn-set-queue-resume').hidden = !g.queue_paused;
}

// 「釋放顯卡」：佇列沒有任務就直接釋放；還有任務時問一次要不要暫停佇列（不暫停的話任務馬上又會把模型載回來）
async function releaseGpu(btn) {
  if (btn) btn.disabled = true;
  try {
    let r = await api('POST', '/api/gpu/release', {});
    if (r.needs_confirm) {
      const go = await askConfirm({ title: '釋放顯卡', text: r.message, ok: '暫停並釋放' });
      if (!go) return;
      r = await api('POST', '/api/gpu/release', { pause_queue: true });
      if (r.needs_confirm) { toast(r.message, true); return; }
    }
    const free = r.free_mb != null ? `，可用 ${(r.free_mb / 1024).toFixed(1)} GB` : '';
    toast(`已釋放顯卡，沒有載入模型${free}${r.paused ? '。佇列已暫停，要接著做請按「繼續佇列」' : ''}`);
    refreshSoon();
  } catch (err) {
    toast(err.message, true);
  } finally {
    if (btn) btn.disabled = false;
  }
}

$('#btn-gpu-free').addEventListener('click', (e) => releaseGpu(e.currentTarget));
async function resumeQueue() {
  try {
    await api('POST', '/api/queue/resume');
    toast('佇列繼續，模型需要時會再載入');
    refreshSoon();
  } catch (err) { toast(err.message, true); }
}
$('#btn-queue-resume').addEventListener('click', resumeQueue);
$('#btn-set-queue-resume').addEventListener('click', resumeQueue);

/* ================= 播放列表 ================= */

function visibleMedia() {
  const q = $('#search').value.trim().toLowerCase();
  const lang = S.settings.libLang || 'all';
  return S.media.filter((m) => (lang === 'all' || m.language === lang)
    && (!q || m.title.toLowerCase().includes(q) || (m.title_zh || '').toLowerCase().includes(q)
      || (m.series?.series || '').toLowerCase().includes(q)));
}

// 播放列表的樣子：list（全部影片）或 series（依作品，作品 → 季 → 集，後端 app/series.py 排好）
const seriesView = () => S.settings.libView === 'series' && (S.series || []).length > 0;

// 依作品的樹，只留目前搜尋、語言篩選看得到的集數
function visibleSeries() {
  const ok = new Set(visibleMedia().map((m) => m.id));
  return (S.series || []).map((w) => ({
    ...w,
    seasons: w.seasons.map((s) => ({ ...s, items: s.items.filter((id) => ok.has(id)) })).filter((s) => s.items.length),
  })).filter((w) => w.seasons.length);
}

// 播完自動播下一部的順序：依作品的樣子照集數，否則照一般清單
function playOrder() {
  if (seriesView()) {
    const ids = visibleSeries().flatMap((w) => w.seasons.flatMap((s) => s.items));
    if (ids.includes(S.current)) return ids.map(mediaById).filter(Boolean);
  }
  return visibleMedia();
}

function renderLibView() {
  const has = (S.series || []).length > 0;
  $('#lib-view').hidden = !has;
  const v = seriesView() ? 'series' : 'list';
  for (const b of $$('#lib-view button')) b.classList.toggle('on', b.dataset.v === v);
}
$('#lib-view').addEventListener('click', (e) => {
  const b = e.target.closest('button[data-v]');
  if (!b) return;
  S.settings.libView = b.dataset.v;
  saveSettings();
  renderLibrary();
  $('#media-list').scrollTop = 0;
});

// 展開、收起的狀態記在這台裝置（key 是作品名，或作品名加季）
const seriesOpen = store.get('seriesOpen', {});
function isOpen(key, fallback) { return key in seriesOpen ? !!seriesOpen[key] : fallback; }

const hasSubs = (m) => m.tracks.some((t) => t.kind === 'asr');
function countText(ids) {
  const done = ids.filter((id) => { const m = mediaById(id); return m && hasSubs(m); }).length;
  return `${ids.length} 集，字幕 ${done}`;
}

function renderSeriesList(ul) {
  const works = visibleSeries();
  if (!works.length) {
    ul.innerHTML = `<li class="lib-empty"><span>找不到符合的影片</span></li>`;
    return;
  }
  const searching = !!$('#search').value.trim();
  ul.innerHTML = works.map((w) => {
    const all = w.seasons.flatMap((s) => s.items);
    const wOpen = searching || isOpen(`w:${w.key}`, works.length === 1 || all.includes(S.current));
    const seasons = w.seasons.map((s) => {
      const sOpen = searching || isOpen(`s:${s.key}`, w.seasons.length === 1 || s.items.includes(S.current));
      const eps = s.items.map((id) => {
        const m = mediaById(id);
        if (!m) return '';
        const no = m.series?.episode_label || '';
        const dur = m.duration ? `<span class="sr-dur">${fmtTime(m.duration)}</span>` : '';
        return `<li class="sr-ep${id === S.current ? ' on' : ''}" data-id="${id}" title="${esc(m.title)}">
          <span class="sr-no">${esc(no)}</span><span class="sr-meta item-meta">${mediaMetaHtml(m)}</span>${dur}</li>`;
      }).join('');
      return `<li class="sr-season${sOpen ? ' open' : ''}">
        <button type="button" class="sr-head" data-toggle="s:${esc(s.key)}" aria-expanded="${sOpen}">
          <span class="sr-caret"></span><span class="sr-title">${esc(s.title)}</span><span class="sr-count">${countText(s.items)}</span></button>
        ${sOpen ? `<ul class="sr-eps">${eps}</ul>` : ''}</li>`;
    }).join('');
    return `<li class="sr-work${wOpen ? ' open' : ''}">
      <button type="button" class="sr-head" data-toggle="w:${esc(w.key)}" aria-expanded="${wOpen}">
        <span class="sr-caret"></span><span class="sr-title">${esc(w.title)}</span><span class="sr-count">${countText(all)}</span></button>
      ${wOpen ? `<ul class="sr-seasons">${seasons}</ul>` : ''}</li>`;
  }).join('');
}

// 一部影片在播放列表上的狀態：處理中、排隊中、失敗、有哪些字幕
function mediaMetaHtml(m) {
  const job = activeJobOf(m.id);
  if (job && job.status === 'running') {
    const pct = Math.round((job.progress || 0) * 100);
    return `<span class="st run">${esc([JOB_LABEL[job.type], jobStage(job)].filter(Boolean).join(' · '))} ${pct}%</span><div class="minibar"><i style="width:${pct}%"></i></div>`;
  }
  if (job) return `<span class="st">${job.waiting ? '等模型' : '排隊中'}（${esc(JOB_LABEL[job.type])}）</span>`;
  const failed = jobsOf(m.id).filter((j) => j.status === 'failed').sort((a, b) => b.finished_at - a.finished_at)[0];
  const langs = [...new Set(m.tracks.map((t) => t.lang))];
  if (failed && !m.tracks.length) return `<span class="st bad">${esc(JOB_LABEL[failed.type])}失敗</span>`;
  if (langs.length) return langs.map((l) => `<span class="chip">${esc(LANG_SHORT[l] || l)}</span>`).join('');
  return `<span>還沒有字幕</span>`;
}

function renderLangFilter() {
  const lang = S.settings.libLang || 'all';
  for (const b of $$('#lang-filter button')) {
    const v = b.dataset.v;
    b.classList.toggle('on', v === lang);
    const n = v === 'all' ? S.media.length : S.media.filter((m) => m.language === v).length;
    $('span', b).textContent = n || '';
  }
}
$('#lang-filter').addEventListener('click', (e) => {
  const b = e.target.closest('button');
  if (!b) return;
  S.settings.libLang = b.dataset.v;
  saveSettings();
  renderLibrary();
});

function renderLibrary() {
  renderLangFilter();
  renderLibView();
  const ul = $('#media-list');
  ul.classList.toggle('series', seriesView());
  if (!S.media.length) {
    ul.innerHTML = `<li class="lib-empty"><span>播放列表是空的</span><button class="btn" data-act="add">新增影片</button></li>`;
    return;
  }
  if (seriesView()) { renderSeriesList(ul); return; }
  const list = visibleMedia();
  if (!list.length) {
    ul.innerHTML = `<li class="lib-empty"><span>找不到符合的影片</span></li>`;
    return;
  }
  ul.innerHTML = list.map((m) => {
    const meta = mediaMetaHtml(m);
    const thumb = m.has_thumb ? `<img src="/media/${m.id}/thumb.jpg" alt="" loading="lazy">` : '';
    const dur = m.duration ? `<span class="dur">${fmtTime(m.duration)}</span>` : '';
    const t = titleParts(m);
    const tip = m.title_zh ? `${m.title}\n${m.title_zh}` : m.title;
    return `<li class="item${m.id === S.current ? ' on' : ''}" data-id="${m.id}">
      <div class="thumb">${thumb}${dur}</div>
      <div class="item-body"><div class="item-title" title="${esc(tip)}">${esc(t.main)}</div>
        ${t.sub ? `<div class="item-sub">${esc(t.sub)}</div>` : ''}<div class="item-meta">${meta}</div></div>
    </li>`;
  }).join('');
}

$('#media-list').addEventListener('click', (e) => {
  if (e.target.closest('[data-act="add"]')) { openAddDialog(); return; }
  const head = e.target.closest('[data-toggle]');
  if (head) {
    seriesOpen[head.dataset.toggle] = head.getAttribute('aria-expanded') !== 'true';
    store.set('seriesOpen', seriesOpen);
    renderLibrary();
    return;
  }
  const li = e.target.closest('.item, .sr-ep');
  if (li) openMedia(li.dataset.id);
});
$('#search').addEventListener('input', renderLibrary);

/* ================= 佇列 ================= */

function jobDetail(j) {
  const p = j.params || {};
  if (j.type === 'transcribe') {
    const engine = S.meta.engines[p.engine]?.label || p.engine;
    const sens = p.sensitive || p.profile === 'asmr' ? ' · 高靈敏度' : '';
    return `${S.meta.languages[p.language] || ''} · ${engine}${sens}`;
  }
  if (j.type === 'translate') return S.meta.translators[p.translator]?.label || p.translator;
  if (j.type === 'download') return '網址';
  if (j.type === 'model') return p.label || p.model || '';
  if (j.type === 'titles' || j.type === 'glosses') return S.meta.translators[p.translator]?.label || '';
  if (j.type === 'health') {
    const t = mediaById(j.media_id)?.tracks.find((x) => x.id === p.track_id);
    return t ? `${S.meta.track_langs[t.lang] || t.lang}（${t.model}）` : '';
  }
  if (j.type === 'dict') return DICT_LANG[p.lang] || p.lang || '';
  return '';
}

// 任務用哪個模型（佇列依模型分組顯示）
function jobModel(j) {
  const p = j.params || {};
  if (j.type === 'transcribe') return S.meta.engines[p.engine]?.label || p.engine;
  if (j.type === 'translate' || j.type === 'titles' || j.type === 'glosses') return S.meta.translators[p.translator]?.label || p.translator;
  if (j.type === 'download') return '網址下載';
  if (j.type === 'model') return '模型下載';
  if (j.type === 'dict') return '字典';
  if (j.type === 'proxy') return 'NVENC 轉檔';
  // 檢查時間軸用 Qwen3-ASR 加對齊模型，歸在 Qwen3-ASR 那組
  if (j.type === 'health') return S.meta.engines.qwen?.label || 'Qwen3-ASR-1.7B';
  return j.type;
}

function renderQueue() {
  const active = S.jobs.filter((j) => j.status === 'running' || j.status === 'queued' || j.status === 'paused');
  const problems = S.jobs.filter((j) => j.status === 'failed');
  const ended = S.jobs.filter((j) => j.status === 'done' || j.status === 'canceled');
  $('#queue-count').textContent = active.length || '';
  const pausedNote = S.gpu && S.gpu.queue_paused ? '（佇列已暫停）' : '';
  $('#queue-summary').textContent = active.length ? `${active.length} 個進行中或等待中${pausedNote}`
    : (S.jobs.length ? `沒有進行中的任務${pausedNote}` : pausedNote);
  const ul = $('#queue');
  if (!S.jobs.length) {
    ul.innerHTML = `<li class="pane-empty">佇列是空的</li>`;
    return;
  }
  const queued = active.filter((j) => j.status === 'queued');

  // 依模型分組：正在跑的模型排最前面，其他依最早排入的任務排序
  const groups = new Map();
  for (const j of active) {
    const key = jobModel(j);
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(j);
  }
  const ordered = [...groups.entries()].sort((a, b) => {
    const ra = a[1].some((j) => j.status === 'running') ? 0 : 1;
    const rb = b[1].some((j) => j.status === 'running') ? 0 : 1;
    if (ra !== rb) return ra - rb;
    return Math.min(...a[1].map((j) => j.position)) - Math.min(...b[1].map((j) => j.position));
  });

  let html = ordered.map(([label, list]) => {
    const running = list.some((j) => j.status === 'running');
    list.sort((a, b) => (a.status === 'running' ? -1 : b.status === 'running' ? 1 : a.position - b.position));
    return `<li class="qgroup${running ? ' running' : ''}">
      <div class="qgroup-head"><span class="qdot"></span><b>${esc(label)}</b>
        <span class="qgroup-state">${running ? '執行中' : '等待中'} · ${list.length} 個任務</span></div>
      <ul class="qgroup-list">${list.map((j) => jobHtml(j, queued)).join('')}</ul>
    </li>`;
  }).join('');
  if (problems.length) {
    html += `<li class="qgroup failed"><div class="qgroup-head"><span class="qdot"></span><b>失敗</b>
      <span class="qgroup-state">${problems.length} 個任務</span></div>
      <ul class="qgroup-list">${problems.map((j) => jobHtml(j, queued)).join('')}</ul></li>`;
  }
  if (ended.length) {
    const open = $('#queue details.qended')?.open ? ' open' : '';
    html += `<li class="qgroup ended"><details class="qended"${open}><summary class="qgroup-head"><span class="qdot"></span><b>已結束</b>
      <span class="qgroup-state">${ended.length} 個任務</span></summary>
      <ul class="qgroup-list">${ended.map((j) => jobHtml(j, queued)).join('')}</ul></details></li>`;
  }
  ul.innerHTML = html;
  renderQueueTimes();
}

function jobHtml(j, queued) {
  {
    const m = mediaById(j.media_id);
    let title;
    if (j.type === 'model') title = j.params?.label || '模型';
    else if (j.type === 'dict') title = j.params?.label || `${DICT_LANG[j.params?.lang] || ''}字典`;
    else if (j.type === 'titles') title = j.params?.label || '影片標題';
    else if (j.type === 'glosses' && !j.media_id) title = j.params?.label || '日文單字釋義';
    else title = m ? titleParts(m).main : '（已刪除的影片）';
    let state = '', body = '', actions = '';
    if (j.status === 'running') {
      const pct = Math.round((j.progress || 0) * 100);
      state = `${pct}%`;
      body = `<div class="minibar"><i style="width:${pct}%"></i></div>
        <div class="job-stage"><span>${esc(jobStage(j))}</span><span class="eta" data-eta="${j.id}"></span></div>`;
      actions = (j.type === 'model' ? `<button class="btn small" data-act="pause" data-id="${j.id}">暫停</button>` : '')
        + `<button class="btn small" data-act="cancel" data-id="${j.id}">取消</button>`;
    } else if (j.status === 'paused') {
      const pct = Math.round((j.progress || 0) * 100);
      state = '已暫停';
      body = `<div class="minibar"><i style="width:${pct}%"></i></div><div class="job-stage"><span>下載的部分會保留，按繼續接著下載</span></div>`;
      actions = `<button class="btn small" data-act="resume" data-id="${j.id}">繼續</button>
        <button class="btn small" data-act="cancel" data-id="${j.id}">取消</button>`;
    } else if (j.status === 'queued') {
      const dep = j.depends_on && S.jobs.find((x) => x.id === j.depends_on);
      state = dep && dep.status !== 'done' ? '等前一步' : `第 ${queued.indexOf(j) + 1} 位`;
      // 要用的模型還在下載（第一次自動安裝，或自己按的下載）：裝好才會開始
      if (j.waiting && !(dep && dep.status !== 'done')) {
        state = '等模型';
        body = `<div class="job-stage"><span>${waitingHtml(j.waiting)}</span></div>`;
      }
      // 「釋放顯卡」暫停了佇列：顯卡任務不會開始（網址、模型下載照常）
      if (S.gpu && S.gpu.queue_paused && GPU_JOB_TYPES.includes(j.type)) {
        state = '暫停中';
        if ((j.progress || 0) > 0) body = `<div class="job-stage"><span>停在 ${Math.round(j.progress * 100)}%，繼續佇列後接著做</span></div>`;
      }
      actions = `<button class="icon-btn small" data-act="up" data-id="${j.id}" title="往前"><svg><use href="#i-up"/></svg></button>
        <button class="icon-btn small" data-act="down" data-id="${j.id}" title="往後"><svg><use href="#i-down"/></svg></button>
        ${j.type === 'model' ? `<button class="btn small" data-act="pause" data-id="${j.id}">暫停</button>` : ''}
        <button class="btn small" data-act="cancel" data-id="${j.id}">取消</button>`;
    } else if (j.status === 'failed') {
      state = '失敗';
      body = `<div class="job-err">${esc(j.error || '未知錯誤')}</div>`;
      actions = `<button class="btn small" data-act="retry" data-id="${j.id}">重試</button><button class="btn small" data-act="remove" data-id="${j.id}">移除</button>`;
    } else if (j.status === 'canceled') {
      state = '已取消';
      if (j.error) body = `<div class="job-stage">${esc(j.error)}</div>`;
      actions = `<button class="btn small" data-act="retry" data-id="${j.id}">重新排入</button><button class="btn small" data-act="remove" data-id="${j.id}">移除</button>`;
    } else {
      state = '完成';
      if (j.started_at && j.finished_at) body = `<div class="job-stage">花了 ${fmtTime(j.finished_at - j.started_at)}</div>`;
    }
    return `<li class="job ${j.status}">
      <div class="job-top"><span class="job-kind">${esc(JOB_LABEL[j.type] || j.type)} · ${esc(jobDetail(j))}</span><span class="job-state${state === '等模型' ? ' wait' : ''}">${state}</span></div>
      <div class="job-title" data-open="${j.media_id}" title="${esc(title)}">${esc(title)}</div>
      ${body}
      ${actions ? `<div class="job-actions">${actions}</div>` : ''}
    </li>`;
  }
}

function renderQueueTimes() {
  const now = Date.now() / 1000;
  for (const el of $$('[data-eta]')) {
    const j = S.jobs.find((x) => x.id === el.dataset.eta);
    // 暫停、中斷後接著做的任務：started_at 是這次開始的時間，只算這次前進了多少（resumed_from 是這次開始時的進度）
    const moved = (j && j.progress || 0) - (j && j.resumed_from || 0);
    if (!j || !j.started_at || !(moved > 0.03)) { el.textContent = ''; continue; }
    const elapsed = now - j.started_at;
    el.textContent = '剩 ' + fmtRemain(elapsed * (1 - j.progress) / moved).replace('約 ', '約');
  }
}

$('#queue').addEventListener('click', async (e) => {
  const open = e.target.closest('[data-open]');
  if (open && mediaById(open.dataset.open)) { openMedia(open.dataset.open); return; }
  const btn = e.target.closest('[data-act]');
  if (!btn) return;
  if (btn.dataset.act === 'setup') { openSetup(); return; }
  const id = btn.dataset.id;
  try {
    if (btn.dataset.act === 'cancel') await api('POST', `/api/jobs/${id}/cancel`);
    if (btn.dataset.act === 'pause') await api('POST', `/api/jobs/${id}/pause`);
    if (btn.dataset.act === 'resume') await api('POST', `/api/jobs/${id}/resume`);
    if (btn.dataset.act === 'retry') {
      const job = S.jobs.find((j) => j.id === id);
      const go = job ? await confirmRetry(job) : { force: false };
      if (!go) return;
      await api('POST', `/api/jobs/${id}/retry`, { force: go.force });
    }
    if (btn.dataset.act === 'remove') await api('DELETE', `/api/jobs/${id}`);
    if (btn.dataset.act === 'up' || btn.dataset.act === 'down') await api('POST', `/api/jobs/${id}/move`, { direction: btn.dataset.act });
    refreshSoon();
  } catch (err) { toast(err.message, true); }
});
$('#btn-clear').addEventListener('click', async () => { await api('POST', '/api/jobs/clear'); refreshSoon(); });

// 會用顯卡的任務（跟後端 jobs.GPU_TYPES 一樣）：佇列暫停時這些不會開始
const GPU_JOB_TYPES = ['transcribe', 'translate', 'proxy', 'titles', 'glosses', 'health'];

// 暫停中的模型下載也算還在佇列裡（跟後端 jobs._active_jobs 一樣）
const isActive = (j) => j.status === 'running' || j.status === 'queued' || j.status === 'paused';

// 佇列裡跟這個任務做同一件事的其他任務（跟後端 jobs.same_active_job 的判斷一樣）。
// 同一部影片用別的模型轉字幕不算重複，跟「用其他模型轉字幕」一樣確認過就能排
function sameActiveJob(j) {
  const p = j.params || {};
  const src = j.type === 'translate' ? translateJobSource(j) : '';
  return S.jobs.find((x) => {
    if (x.id === j.id || x.type !== j.type || !isActive(x)) return false;
    const xp = x.params || {};
    if (j.type === 'transcribe') return x.media_id === j.media_id && xp.engine === p.engine;
    if (['proxy', 'download'].includes(j.type)) return x.media_id === j.media_id;
    if (j.type === 'translate') return !!src && translateJobSource(x) === src;
    if (j.type === 'health') return xp.track_id === p.track_id;
    if (j.type === 'model') return xp.model === p.model && (xp.variant || '') === (p.variant || '');
    if (j.type === 'dict') return xp.lang === p.lang;
    if (j.type === 'glosses') return xp.scope === 'library' || (p.scope !== 'library' && x.media_id === j.media_id);
    return true;
  });
}

const DUP_TEXT = {
  transcribe: ['這部影片正在用 {model} 轉字幕', '這部影片已經在排隊用 {model} 轉字幕'],
  health: ['這條字幕正在檢查時間軸', '這條字幕已經在排隊檢查時間軸'],
  proxy: ['這部影片正在轉檔', '這部影片已經在排隊轉檔'],
  download: ['這部影片正在下載', '這部影片已經在排隊下載'],
  model: ['這個模型正在下載', '這個模型已經在排隊下載'],
  titles: ['影片標題正在翻譯', '影片標題已經在排隊翻譯'],
  glosses: ['單字釋義正在產生', '單字釋義已經在排隊'],
  dict: ['這本字典正在建立', '這本字典已經在排隊建立'],
};

function tellAlreadyQueued(j, dup) {
  const m = mediaById(j.media_id);
  if (j.type === 'translate') {
    const src = m?.tracks.find((t) => t.id === translateJobSource(j));
    if (m && src) return tellTranslating(m, src, dup);
  }
  const [running, queued] = DUP_TEXT[j.type] || ['同樣的任務正在執行', '同樣的任務已經在排隊'];
  const items = [];
  if (m) items.push(titleParts(m).main);
  if (j.type === 'health') {
    const t = m?.tracks.find((x) => x.id === (j.params || {}).track_id);
    if (t) items.push(`字幕：${trackLabel(t, m.tracks)}`);
  }
  if (j.type === 'model' || j.type === 'dict') items.push(jobDetail(j));
  return askConfirm({
    title: (dup.status === 'running' ? running : queued).replace('{model}', jobModel(dup)),
    text: '完成後就會出現，不用再排一次。',
    items, ok: '知道了', cancel: '',
  });
}

// 重試要用的模型還沒下載：說明並提供「去下載」（後端也會擋，這裡先講清楚）
// 還在下載的模型（pending）不算沒下載：任務會排隊等它裝好
function retryMissingModel(j) {
  const p = j.params || {};
  const missing = (v) => v && v.installed === false && !v.pending;
  if (j.type === 'transcribe' && missing(S.meta.engines[p.engine])) return S.meta.engines[p.engine].label;
  if (j.type === 'health' && missing(S.meta.engines.qwen)) return S.meta.engines.qwen.label;
  if ((j.type === 'translate' || j.type === 'titles') && missing(S.meta.translators[p.translator])) {
    return S.meta.translators[p.translator].label;
  }
  return '';
}

// 失敗或取消的任務重試、重新排入前：
// 同一件事已經在排隊或執行，只說明不再排；會重做已經有的結果（字幕、翻譯、時間軸檢查、相容播放檔）先確認。
// 回傳 null 表示不送出；否則回傳 { force }，確認過才帶 force（後端沒帶 force 會擋）
async function confirmRetry(j) {
  const dup = sameActiveJob(j);
  if (dup) { await tellAlreadyQueued(j, dup); return null; }
  // 前一步沒做完的話，只排這一步會一直停在「等前一步」（後端也會擋）
  const dep = j.depends_on && S.jobs.find((x) => x.id === j.depends_on);
  if (dep && (dep.status === 'failed' || dep.status === 'canceled')) {
    await askConfirm({
      title: '前一個步驟沒有完成',
      text: `請先重試前一個任務（${JOB_LABEL[dep.type] || dep.type}），這個任務會跟著重新排入。`,
      ok: '知道了', cancel: '',
    });
    return null;
  }
  await refreshMeta();
  const missing = retryMissingModel(j);
  if (missing) {
    const go = await askConfirm({
      title: `${missing} 還沒下載`,
      text: '這個任務要用的模型還沒下載，重試也會失敗。下載好之後再按重試。',
      ok: '去下載',
    });
    if (go) openModelManager(j.type === 'translate' || j.type === 'titles' ? 'translator' : 'asr');
    return null;
  }
  const again = j.status === 'failed' ? '重試' : '重新排入';
  const m = mediaById(j.media_id);
  const p = j.params || {};
  if (j.media_id && !m) {
    await askConfirm({ title: '影片已經刪除了', text: '這個任務不用重試，可以按「移除」。', ok: '知道了', cancel: '' });
    return null;
  }
  if (!m) return { force: false };
  let ok = true;
  if (j.type === 'translate') {
    const srcId = translateJobSource(j);
    const src = m.tracks.find((t) => t.id === srcId);
    if (srcId && !src) {
      await askConfirm({ title: '原文字幕已經刪除了', text: '這個翻譯任務不用重試，可以按「移除」。', ok: '知道了', cancel: '' });
      return null;
    }
    const already = src ? translationsOf(m, srcId) : [];
    if (already.length) {
      const replacing = already.some((t) => t.id === p.replace_track_id);
      ok = await askConfirm({
        title: '這條字幕已經翻譯過了',
        text: replacing
          ? `${again}會再翻一次「${trackLabel(src, m.tracks)}」，佔用顯卡，翻完會取代舊的翻譯。確定嗎？`
          : `${again}會再翻一次「${trackLabel(src, m.tracks)}」，佔用顯卡，並且多出一條新的翻譯（舊的不會被覆蓋）。確定嗎？`,
        items: already.map((t) => `${t.id === p.replace_track_id ? '會被取代' : '現有'}：${trackLabel(t, m.tracks)}，${t.cue_count} 句`),
        ok: '還是再翻一次',
      });
      return ok ? { force: true } : null;
    }
  }
  if (j.type === 'transcribe') {
    // 跟「用其他模型轉字幕」一樣：已經有字幕、或正在用別的模型轉，都先確認
    const existing = m.tracks.filter((t) => t.kind === 'asr');
    const pending = S.jobs.filter((x) => x.id !== j.id && x.type === 'transcribe' && isActive(x) && x.media_id === m.id);
    if (!existing.length && !pending.length) return { force: false };
    ok = await askConfirm({
      title: existing.length ? '這部影片已經有字幕了' : '這部影片正在用其他模型轉字幕',
      text: existing.length
        ? `${again}會用 ${jobModel(j)} 再轉一次，佔用顯卡，並且多出一條新的字幕軌（舊的不會被覆蓋）。確定嗎？`
        : `完成後字幕就會出現。確定要同時用 ${jobModel(j)} 再轉一次嗎？`,
      items: [
        ...existing.map((t) => `現有：${trackLabel(t, m.tracks)}，${t.cue_count} 句`),
        ...pending.map((x) => `${x.status === 'running' ? '正在轉' : '排隊中'}：${jobModel(x)}`),
      ],
      ok: '還是再轉一次',
    });
    return ok ? { force: true } : null;
  }
  if (j.type === 'health') {
    const t = m.tracks.find((x) => x.id === p.track_id);
    if (!t) {
      await askConfirm({ title: '字幕已經刪除了', text: '這個檢查時間軸的任務不用重試，可以按「移除」。', ok: '知道了', cancel: '' });
      return null;
    }
    if (!healthOf(t)) return { force: false };
    ok = await askConfirm({
      title: '這條字幕已經檢查過時間軸',
      text: `${again}會再檢查一次，佔用顯卡，預估${fmtRemain(t.cue_count * HEALTH_S_PER_LINE)}。只有更對得上的行才會調整時間，調整前的字幕檔會留一份備份。確定嗎？`,
      items: [healthText(t)],
      ok: '再檢查一次',
    });
    return ok ? { force: true } : null;
  }
  if (j.type === 'proxy' && m.has_proxy) {
    ok = await askConfirm({
      title: '這部影片已經有相容播放檔',
      text: `${again}會再轉一次檔，佔用顯卡，轉好後取代現在的檔案。確定嗎？`,
      ok: '還是再轉一次',
    });
    return ok ? { force: true } : null;
  }
  return { force: false };
}

async function refreshSoon() {
  try {
    const st = await api('GET', '/api/state');
    S.gpu = st.gpu; renderGpu();
    onSetupSummary(st);
    S.sig = JSON.stringify([st.media, st.jobs, !!(st.gpu && st.gpu.queue_paused)]);
    S.media = st.media; S.series = st.series || []; S.jobs = st.jobs;
    onData();
  } catch { /* 下一次輪詢會再試 */ }
}

/* ================= 分頁 ================= */

// 版面由 style.css 依視窗大小決定（--layout）：desktop 桌面三欄、stack 直向、side 平板橫放、short 手機橫放
const layoutMode = () => getComputedStyle(document.documentElement).getPropertyValue('--layout').trim() || 'desktop';

// 目前要顯示哪個分頁。桌面只有逐字稿、佇列；手機、平板多了播放列表和字幕設定（影片資訊），另外記一份
function currentTab() {
  const mode = layoutMode();
  if (mode === 'desktop') return S.settings.sideTab === 'queue' ? 'queue' : 'transcript';
  let name = S.settings.compactTab || 'transcript';
  if (name === 'info' && mode === 'side') name = 'transcript'; // 平板橫放時影片資訊一直在影片下面
  if ((name === 'transcript' || name === 'info') && !S.current) name = 'library'; // 還沒打開影片，先給播放列表
  return name;
}
function applyTabs() {
  const name = currentTab();
  for (const t of $$('.tab')) t.classList.toggle('on', t.dataset.tab === name);
  $('#pane-transcript').hidden = name !== 'transcript';
  $('#pane-queue').hidden = name !== 'queue';
  document.body.dataset.ctab = name;
  applyImmersive();
}

// 手機橫放收起右欄＝專心看影片：頂列也收起來（style.css 的 body.immersive），要叫回來按控制列上的側欄按鈕。
// 還沒打開影片（沒有控制列）、在單字頁或設定頁時不收，不然會找不到頂列
function applyImmersive() {
  const on = layoutMode() === 'short' && !!S.settings.hideRight && !!S.current
    && $('#settings-page').hidden && $('#vocab-page').hidden;
  document.body.classList.toggle('immersive', on);
}

function showTab(name, reveal = false) {
  const key = layoutMode() === 'desktop' ? 'sideTab' : 'compactTab';
  if ((key === 'compactTab' || name === 'transcript' || name === 'queue') && S.settings[key] !== name) {
    S.settings[key] = name;
    saveSettings();
  }
  applyTabs();
  // 只有使用者主動要看（例如點右上角顯卡狀態）才自動展開右側欄
  if (reveal && S.settings.hideRight && layoutMode() !== 'stack') togglePanel('right', false);
}

// 左右側欄收合，狀態會記住。手機、平板的播放列表在分頁裡不能收；直向時也沒有右側欄可以收
function applyPanels() {
  const mode = layoutMode();
  document.body.classList.toggle('hide-left', !!S.settings.hideLeft && mode === 'desktop');
  document.body.classList.toggle('hide-right', !!S.settings.hideRight && mode !== 'stack');
  $('#toggle-left').classList.toggle('off', !!S.settings.hideLeft);
  $('#toggle-right').classList.toggle('off', !!S.settings.hideRight);
  applyImmersive();
  requestAnimationFrame(layoutSubs);
}
function togglePanel(side, hide) {
  const key = side === 'left' ? 'hideLeft' : 'hideRight';
  S.settings[key] = hide === undefined ? !S.settings[key] : hide;
  saveSettings();
  applyPanels();
}
$('#toggle-left').addEventListener('click', () => togglePanel('left'));
$('#toggle-right').addEventListener('click', () => togglePanel('right'));
$('#c-panel').addEventListener('click', () => togglePanel('right', false));
for (const t of $$('.tab')) t.addEventListener('click', () => showTab(t.dataset.tab));
$('#gpu').addEventListener('click', () => showTab('queue', true));
// 轉向、視窗拉寬拉窄跨過版面的分界時，重新套用側欄和分頁（字幕大小由播放區的 ResizeObserver 重算）
let lastLayout = '';
addEventListener('resize', () => {
  const mode = layoutMode();
  if (mode === lastLayout) return;
  lastLayout = mode;
  applyPanels();
  applyTabs();
});

/* ================= 播放器 ================= */

const video = $('#video');
const player = $('#player');
const subsEl = $('#subs');
const lineEl = { 1: $('#line1 span'), 2: $('#line2 span') };

function openMedia(id) {
  if (S.current === id) return;
  const m = mediaById(id);
  if (!m) return;
  S.current = id;
  store.set('last', id);
  history.replaceState(null, '', `#m=${id}`);
  S.srcKey = '';
  S.trackSel = { 1: null, 2: null };
  S.cues = { 1: [], 2: [] };
  S.idx = { 1: -1, 2: -1 };
  S.hi = -1;
  S.userScrollAt = 0;
  setLoop(null);
  $('#transcript').innerHTML = '';
  $('#transcript').scrollTop = 0;
  lineEl[1].textContent = lineEl[2].textContent = '';
  $('#player-empty').hidden = true;
  player.classList.add('has-media');
  $('#deck').hidden = false;
  $('#tr-search').value = '';
  renderLibrary();
  renderDeck(m);
  syncVideoSource(m);
  syncTracks(m, true);
  renderTranscriptStatus(m);
  showControls();
  applyTabs();
}

function closeMedia() {
  S.current = null;
  video.removeAttribute('src');
  video.load();
  player.classList.remove('has-media');
  $('#player-empty').hidden = false;
  $('#player-msg').hidden = true;
  $('#deck').hidden = true;
  $('#transcript').innerHTML = '';
  $('#tr-status').hidden = true;
  renderLibrary();
  applyTabs();
}

function syncVideoSource(m) {
  const key = m.path ? `${m.id}:${m.has_proxy ? 'p' : 'o'}` : `${m.id}:none`;
  if (key === S.srcKey) return;
  const keepTime = S.srcKey.startsWith(m.id + ':') ? video.currentTime : 0;
  S.srcKey = key;
  $('#player-msg').hidden = true;
  if (!m.path) {
    video.removeAttribute('src');
    video.load();
    showPlayerMsg('影片下載中，下載完成後就能播放');
    return;
  }
  video.src = `/media/${m.id}/video?v=${m.has_proxy ? 'p' : 'o'}`;
  video.playbackRate = S.settings.rate;
  video.loop = S.settings.repeat === 'one';
  updatePlayButton();
  video.addEventListener('loadedmetadata', () => {
    if (keepTime) {
      video.currentTime = keepTime;
    } else if (S.settings.resume) {
      // 從上次看到的地方繼續（快看完的就從頭開始）
      const saved = store.get('pos:' + m.id, 0);
      if (saved > 5 && saved < (video.duration || 0) - 10) video.currentTime = saved;
    }
    // 安裝進度頁蓋著的時候不要在後面自己播起來
    if (S.settings.autoplay && !keepTime && !SU.open) video.play().catch(() => {});
    updatePlayButton();
  }, { once: true });
}

let lastPosSave = 0;
function savePosition(force = false) {
  if (!S.current || !video.duration) return;
  const now = Date.now();
  if (!force && now - lastPosSave < 3000) return;
  lastPosSave = now;
  store.set('pos:' + S.current, Math.floor(video.currentTime));
}
video.addEventListener('timeupdate', () => savePosition());
video.addEventListener('pause', () => savePosition(true));
window.addEventListener('beforeunload', () => savePosition(true));

// 播完自動換下一部
video.addEventListener('ended', () => {
  updatePlayButton();
  if (!S.settings.autoNext && S.settings.repeat !== 'all') return;
  const list = playOrder();
  const i = list.findIndex((m) => m.id === S.current);
  const next = list[i + 1] || (S.settings.repeat === 'all' ? list[0] : null);
  if (next && next.id !== S.current) {
    const wantPlay = S.settings.autoplay;
    openMedia(next.id);
    if (wantPlay) video.addEventListener('loadedmetadata', () => video.play().catch(() => {}), { once: true });
  }
});

function showPlayerMsg(text, action) {
  const el = $('#player-msg');
  el.innerHTML = `<p>${esc(text)}</p>` + (action ? `<button class="btn primary" data-act="${action.act}">${esc(action.label)}</button>` : '');
  el.hidden = false;
}

video.addEventListener('error', () => {
  const m = mediaById(S.current);
  if (!m || !m.path) return;
  if (m.has_proxy) { showPlayerMsg('影片無法播放，檔案可能已經被移動或刪除'); return; }
  const pending = jobsOf(m.id).some((j) => j.type === 'proxy' && (j.status === 'queued' || j.status === 'running'));
  showPlayerMsg(
    pending ? '正在轉成相容格式，完成後會自動載入' : `瀏覽器沒辦法直接播放這個格式（${(m.vcodec || '未知').toUpperCase()}）`,
    pending ? null : { act: 'proxy', label: '轉成相容格式' },
  );
});
video.addEventListener('loadeddata', () => { $('#player-msg').hidden = true; layoutSubs(); });
$('#player-msg').addEventListener('click', async (e) => {
  if (e.target.closest('[data-act="proxy"]')) {
    try { await api('POST', `/api/media/${S.current}/proxy`); toast('已加入佇列'); refreshSoon(); } catch (err) { toast(err.message, true); }
  }
});

/* ----- 字幕軌 ----- */

// 字幕選擇固定是 1 = 原文、2 = 從這條原文翻出來的翻譯（翻譯每行的時間跟來源一致，兩行才對得上）
function defaultSelection(m) {
  const first = m.tracks.filter((t) => t.kind === 'asr').pop();
  return { 1: first?.id || '', 2: (first && newestTranslation(m, first.id)?.id) || '' };
}

// 把一組選擇整理成上面的規則。tr 記住要不要看翻譯，只有在翻譯選單選「不顯示」才是 false，
// 所以原文沒有翻譯時換到有翻譯的原文、或翻譯剛做好，翻譯會自己出來。
// 舊版存的選擇沒有 tr，而且兩行可以隨便選（1 放翻譯、2 放原文，或兩行都是原文）：原文取選到的原文，翻譯取跟它同一組的。
// 空字串是使用者選的「不顯示」要保留；存的字幕軌已經刪掉就用預設
function pairSelection(m, sel) {
  const byId = Object.fromEntries(m.tracks.map((t) => [t.id, t]));
  const a = byId[sel[1]], b = byId[sel[2]];
  let src = a?.kind === 'asr' ? a : b?.kind === 'asr' ? b : null;
  const other = src && src === b ? sel[1] : sel[2]; // 翻譯那一行原本存的值
  // 選到的翻譯：有原文時要是它翻出來的；沒選到原文時，從翻譯找回它的原文（來源已刪除的翻譯不算）
  const picked = [b, a].find((t) => t?.kind === 'translation' && (src ? t.source_track_id === src.id : byId[t.source_track_id]));
  if (!src && picked) src = byId[picked.source_track_id];
  const orig = src ? src.id : sel[1] === '' ? '' : defaultSelection(m)[1];
  let tr = sel.tr;
  // 舊格式：翻譯那一行存的是「不顯示」而且當時有翻譯可選，才算使用者自己關掉翻譯
  if (tr === undefined) tr = !!picked || !(other === '' && (!orig || newestTranslation(m, orig)));
  let trans = '';
  if (picked && picked.source_track_id === orig) trans = picked.id;
  else if (orig && tr) trans = newestTranslation(m, orig)?.id || '';
  return { 1: orig, 2: trans, tr: !!picked || !!tr };
}

function syncTracks(m, initial = false) {
  const known = store.get('known:' + m.id, null);
  let next;
  if (initial) {
    const saved = store.get('tracks:' + m.id, null);
    next = pairSelection(m, saved && typeof saved === 'object' ? saved : {});
    if (saved && saved.tr === undefined) store.set('tracks:' + m.id, next); // 舊版的選擇轉成新規則後存回去
  } else {
    next = pairSelection(m, { ...S.trackSel, tr: S.showTr });
  }
  // 新產生的字幕軌（包括頁面沒開著的時候做好的）：原文沒選才自動選上新的原文；
  // 目前原文有新翻譯就換成新的，翻譯選了「不顯示」就不動。第一次開這部影片沒有紀錄，照預設選
  if (Array.isArray(known)) {
    const before = `${next[1]}|${next[2]}`;
    for (const t of m.tracks.filter((x) => !known.includes(x.id))) {
      if (t.kind === 'asr' && !next[1]) {
        next[1] = t.id;
        next[2] = (next.tr && newestTranslation(m, t.id)?.id) || '';
      } else if (t.kind === 'translation' && next[1] && t.source_track_id === next[1] && next.tr) {
        next[2] = t.id;
      }
    }
    // 自動換上的也要存，重新整理後才不會回到舊的那條
    if (`${next[1]}|${next[2]}` !== before) store.set('tracks:' + m.id, next);
  }
  store.set('known:' + m.id, m.tracks.map((t) => t.id));
  applyTrackSel(m, next, initial);
}

// 套用整理好的選擇：更新兩個選單，換了字幕軌就重新讀取
function applyTrackSel(m, next, initial = false) {
  S.showTr = next.tr;
  fillTrackSelects(m, next);
  for (const slot of [1, 2]) {
    if (next[slot] !== S.trackSel[slot] || initial) {
      S.trackSel[slot] = next[slot];
      loadCues(slot, next[slot]);
    } else if (next[slot] && S.cueVer[slot] !== cueVersion(m, next[slot])) {
      loadCues(slot, next[slot]); // 伺服器改過這條字幕（例如檢查時間軸修正了時間），重新讀取
    }
  }
}

// 字幕檔的版本：檢查時間軸會改原文和逐行翻譯的時間但不改行數，所以帶上原文字幕的檢查時間
function cueVersion(m, trackId) {
  const t = m.tracks.find((x) => x.id === trackId);
  if (!t) return '';
  const src = t.kind === 'translation' ? m.tracks.find((x) => x.id === t.source_track_id) : t;
  return `${t.cue_count}:${healthOf(src)?.at || ''}`;
}

// 時間軸檢查摘要（新版伺服器給物件，舊版伺服器直接給資料庫裡的 JSON 字串）
function healthOf(t) {
  const h = t?.health;
  if (!h) return null;
  if (typeof h === 'object') return h;
  try { return JSON.parse(h); } catch { return null; }
}

// 原文選單只列辨識出來的字幕，翻譯選單只列從目前原文翻出來的翻譯（來源已刪除的翻譯不列）
function fillTrackSelects(m, sel) {
  const asr = m.tracks.filter((t) => t.kind === 'asr');
  const trs = sel[1] ? m.tracks.filter((t) => t.kind === 'translation' && t.source_track_id === sel[1]) : [];
  fillTrackSelect($('#sel-1'), asr, '不顯示', sel[1]);
  fillTrackSelect($('#sel-2'), trs, !sel[1] ? '先選原文字幕' : trs.length ? '不顯示' : '這條字幕還沒有翻譯', sel[2]);
  // 看不到的那個選單放每條原文的翻譯名稱和所有提示文字，翻譯選單的寬度就固定是這部影片裡最寬的那個
  const names = asr.flatMap((a) => {
    const list = m.tracks.filter((t) => t.kind === 'translation' && t.source_track_id === a.id);
    return list.map((t) => trackLabel(t, list));
  });
  const ghost = ['先選原文字幕', '不顯示', '這條字幕還沒有翻譯', ...names].map((n) => `<option>${esc(n)}</option>`).join('');
  const g = $('#sel-2-ghost');
  if (g.dataset.html !== ghost) { g.innerHTML = ghost; g.dataset.html = ghost; }
}

function fillTrackSelect(sel, list, empty, value) {
  const html = `<option value="">${empty}</option>` + list.map((t) => `<option value="${t.id}">${esc(trackLabel(t, list))}</option>`).join('');
  if (sel.dataset.html !== html) { sel.innerHTML = html; sel.dataset.html = html; }
  sel.value = value || '';
  sel.disabled = !list.length;
}

async function loadCues(slot, trackId) {
  const mediaId = S.current;
  const track = (mediaById(mediaId)?.tracks || []).find((t) => t.id === trackId);
  $(`#line${slot}`).dataset.lang = track ? track.lang : '';
  S.cueVer[slot] = trackId && mediaById(mediaId) ? cueVersion(mediaById(mediaId), trackId) : '';
  if (!trackId) {
    S.cues[slot] = [];
  } else {
    try {
      const cues = await api('GET', `/api/tracks/${trackId}/cues`);
      if (S.current !== mediaId || S.trackSel[slot] !== trackId) return;
      S.cues[slot] = cues;
    } catch (err) { toast('讀取字幕失敗：' + err.message, true); S.cues[slot] = []; }
  }
  S.idx[slot] = -1;
  renderTranscript();
  applySubSettings();
  updateSubs(true);
  window.WL?.onCuesLoaded(slot, trackId);
}

function newestTranslation(m, sourceId) {
  return m.tracks.filter((t) => t.kind === 'translation' && t.source_track_id === sourceId).pop() || null;
}

// slot 1 選原文：翻譯原本有顯示就換成新原文最新的翻譯，原本選「不顯示」就維持不顯示。
// slot 2 選翻譯：原文切到它的來源（字幕管理可以直接選別條原文的翻譯）
function selectTrack(slot, value) {
  const m = mediaById(S.current);
  if (!m) return;
  const sel = { 1: S.trackSel[1], 2: S.trackSel[2], tr: S.showTr };
  if (slot === 1 && value !== S.trackSel[1]) {
    sel[1] = value;
    sel[2] = (value && S.showTr && newestTranslation(m, value)?.id) || '';
  } else if (slot === 2) {
    sel[2] = value;
    sel.tr = !!value;
    const t = m.tracks.find((x) => x.id === value);
    if (t?.source_track_id) sel[1] = t.source_track_id;
  }
  const next = pairSelection(m, sel);
  applyTrackSel(m, next);
  store.set('tracks:' + m.id, next);
}
$('#sel-1').addEventListener('change', (e) => selectTrack(1, e.target.value));
$('#sel-2').addEventListener('change', (e) => selectTrack(2, e.target.value));
// 原文和翻譯放在同一個位置時，切換誰在上面
$('#btn-swap').addEventListener('click', () => setSetting('swapLines', !S.settings.swapLines));

/* ----- 字幕顯示 ----- */

// 跳到某句時會停在開頭前 0.05 秒，找句子時放寬一點，暫停狀態下字幕才會顯示
const CUE_LEAD = 0.06;
function findCue(cues, t) {
  let lo = 0, hi = cues.length - 1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    const c = cues[mid];
    if (t < c.start - CUE_LEAD) hi = mid - 1;
    else if (t >= c.end) lo = mid + 1;
    else return mid;
  }
  return -1;
}
function lastStarted(cues, t) {
  let lo = 0, hi = cues.length - 1, ans = -1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (cues[mid].start - CUE_LEAD <= t) { ans = mid; lo = mid + 1; } else hi = mid - 1;
  }
  return ans;
}

function updateSubs(force = false) {
  const t = video.currentTime;
  let changed = false;
  const nav = navCue();
  for (const slot of [1, 2]) {
    const i = cueAt(S.cues[slot], t, nav);
    if (force || i !== S.idx[slot]) {
      S.idx[slot] = i;
      lineEl[slot].innerHTML = i >= 0 ? cueHtml(S.cues[slot][i]) : '';
      lineEl[slot].dataset.cue = i >= 0 ? `${slot}:${i}` : '';
      lineEl[slot].parentElement.classList.toggle('empty', i < 0);
      lineEl[slot].parentElement.classList.remove('peek'); // 觸控點開的遮住翻譯，換句後再遮回去
      window.WL?.onCueChange(slot);
      changed = true;
    }
  }
  if (changed) fitOutside();
  const P = S.cues[S.primary];
  // 逐字稿高亮跟畫面字幕同一句；沒有剛跳過去的那一句時，照舊用「最後一句開始的」（句子之間的空檔也會留在上一句）
  const picked = nav ? cueAt(P, t, nav) : -1;
  const hi = picked >= 0 ? picked : lastStarted(P, t);
  if (force || hi !== S.hi) highlightTranscript(hi);
  if (S.loop && (t >= S.loop.end || t < S.loop.start - 1)) {
    NAV.seekAt = performance.now();   // 循環跳回句首不算使用者自己拉進度條
    video.currentTime = S.loop.start;
  }
}

let raf = 0;
function frame() {
  window.WL?.pauseCheck(); // 查字時「這句播完才暫停」要在換句之前判斷
  updateSubs();
  updateSeekUI();
  raf = !video.paused && !video.ended ? requestAnimationFrame(frame) : 0;
}
video.addEventListener('play', () => { if (!raf) raf = requestAnimationFrame(frame); updatePlayButton(); pokeControls(); });
video.addEventListener('pause', () => {
  updatePlayButton();
  // 手指點字查詢時影片會暫停：控制列不要跟著跳出來，不然會蓋住字幕，下一個字要先把控制列點掉才點得到
  if (lastPointer !== 'mouse' && window.WL?.cardOpen()) return;
  showControls();
});
video.addEventListener('seeking', () => {
  S.seekAt = performance.now();
  // 不是按上下句、點逐字稿、循環跳回句首造成的（拉進度條、方向鍵快轉、從查字卡跳過去）：忘掉剛才跳到的那一句。
  // NAV.seekAt 是一次性的：對到就馬上用掉，不然後面幾百毫秒內使用者自己的跳轉會被當成自己人
  const own = S.seekAt - NAV.seekAt < 300;
  NAV.seekAt = -1e9;
  if (!own) navReset();
  // 用進度條或快捷鍵跳轉時，逐字稿要立刻跟上，不管剛才有沒有手動捲過；點逐字稿本身跳轉則保留原位
  if (S.seekAt - S.transcriptClickAt > 500) S.userScrollAt = 0;
});
video.addEventListener('seeked', () => { S.seekAt = performance.now(); updateSubs(true); updateSeekUI(); });
video.addEventListener('timeupdate', () => { if (!raf) { updateSubs(); updateSeekUI(); } });
video.addEventListener('durationchange', updateSeekUI);
for (const ev of ['emptied', 'loadstart', 'canplay', 'playing', 'waiting']) {
  video.addEventListener(ev, updatePlayButton);
}
video.addEventListener('progress', updateSeekUI);
video.addEventListener('loadedmetadata', () => {
  // 手機直向的影片框照影片本身的比例（style.css 的 --video-ar）
  if (video.videoWidth && video.videoHeight) $('#screen').style.setProperty('--video-ar', `${video.videoWidth} / ${video.videoHeight}`);
  layoutSubs();
  updateSeekUI();
});

function applySubSettings() {
  const st = S.settings;
  for (const slot of [1, 2]) {
    const el = $(`#line${slot}`);
    el.classList.toggle('has-ruby', hasRuby(S.cues[slot]));
    const place = PLACES[st['place' + slot]] ? st['place' + slot] : 'inBottom';
    const zone = $(`#zone-${place}`);
    if (el.parentElement !== zone) zone.appendChild(el);
    const inside = place.startsWith('in');
    el.classList.toggle('outline', inside && (st.style === 'outline' || st.style === 'both'));
    el.classList.toggle('box', inside && (st.style === 'box' || st.style === 'both'));
    el.classList.toggle('blur', slot === 2 && !!st.blur2);
    el.hidden = !!st['hide' + slot];
    $(`#place-${slot}`).value = place;
  }
  // 原文和翻譯放在同一個位置時預設原文在上，交換上下位置後翻譯在上
  for (const zone of ['inTop', 'inBottom', 'outTop', 'outBottom']) {
    const box = $(`#zone-${zone}`);
    const lines = [...box.children].sort((a, b) => a.id.localeCompare(b.id) * (st.swapLines ? -1 : 1));
    lines.forEach((l) => box.appendChild(l));
  }
  const samePlace = $('#line1').parentElement === $('#line2').parentElement;
  // 畫面上只有一行（沒有翻譯、選了不顯示或按 1、2 隱藏）時交換看不出效果，先停用，免得默默改掉設定
  const bothShown = !!S.trackSel[1] && !!S.trackSel[2] && !st.hide1 && !st.hide2;
  $('#btn-swap').disabled = !samePlace || !bothShown;
  $('#btn-swap').title = !bothShown ? '原文和翻譯都顯示時才能交換上下位置'
    : !samePlace ? '原文和翻譯放在不同位置，放在同一個位置時才能交換上下'
    : st.swapLines ? '交換上下位置：目前翻譯在上、原文在下' : '交換上下位置：目前原文在上、翻譯在下';
  $('#transcript').classList.toggle('blur2', !!st.blur2);
  $('#set-size1').value = st.size1;
  $('#set-size2').value = st.size2;
  $('#set-pos').value = st.pos;
  $('#set-gap').value = st.gap;
  $('#set-style').value = st.style;
  $('#set-bg').value = st.bg;
  $('#set-width').value = st.width;
  $('#v-width').textContent = `${st.width}%`;
  $('#set-reserve').value = String(st.reserveLines);
  $('#set-letterbox').checked = st.useLetterbox !== false;
  $('#set-furi').value = st.furi || 'all';
  $('#screen').classList.toggle('furi-hover', st.furi === 'hover');
  $('#transcript').classList.toggle('furi-hover', st.furi === 'hover');
  $('#v-size1').textContent = st.size1.toFixed(1);
  $('#v-size2').textContent = st.size2.toFixed(1);
  $('#v-pos').textContent = `${st.pos}%`;
  $('#v-gap').textContent = `${st.gap}%`;
  $('#v-bg').textContent = `${st.bg}%`;
  $('#set-blur2').checked = !!st.blur2;
  $('#tr-follow').checked = !!st.follow;
  layoutSubs();
}

function layoutSubs() {
  const r = player.getBoundingClientRect();
  let w = r.width, h = r.height, x = 0, y = 0;
  if (video.videoWidth && video.videoHeight && r.width && r.height) {
    const scale = Math.min(r.width / video.videoWidth, r.height / video.videoHeight);
    w = video.videoWidth * scale; h = video.videoHeight * scale;
    x = (r.width - w) / 2; y = (r.height - h) / 2;
  }
  subsEl.style.left = `${x}px`;
  subsEl.style.top = `${y}px`;
  subsEl.style.width = `${w}px`;
  subsEl.style.height = `${h}px`;
  const st = S.settings;
  const mode = layoutMode();
  // 字級是影片高度的百分比。手機、平板的影片小，照算字會小到看不清楚、拉桿也沒作用，
  // 所以另外加 230px 當底數：預設值在手機上約 16px、平板直放約 24px，跟桌面看起來差不多大
  const subPx = mode === 'desktop'
    ? (base, size, min) => Math.max(min, (base * size) / 100)
    : (base, size, min) => Math.max(min, ((base + 230) * size) / 100);
  const fs1 = subPx(h, st.size1, 16);
  const screenEl = $('#screen');
  const screen = screenEl.style;
  screen.setProperty('--fs1', `${fs1}px`);
  screen.setProperty('--fs2', `${subPx(h, st.size2, 14)}px`);
  screen.setProperty('--pad', `${(h * st.pos) / 100}px`);
  screen.setProperty('--gap', `${(fs1 * st.gap) / 100}px`);
  screen.setProperty('--bg-alpha', String(st.bg / 100));
  screen.setProperty('--linew', `${st.width}%`);
  // 影片外的字級改用整個播放區高度，避免「字幕列變高→影片變矮→字級變小→字幕列再變」的來回跳動。
  // 手機直向的播放區高度包含字幕列，不能拿來算：改用播放區寬度換算的 16:9 高度（寬度不會跟著字幕變）；
  // 橫向和全螢幕時播放區大小固定，取 16:9 高度和播放區高度比較小的那個，字才不會比畫面還擠
  const sr = screenEl.getBoundingClientRect();
  const full = document.body.classList.contains('in-fs');
  let sh = sr.height || r.height;
  if (mode === 'stack' && !full) sh = (sr.width * 9) / 16;
  else if (mode !== 'desktop') {
    // 手機橫放收起右欄時，寬度照右欄還在的時候算（style.css 的 --cside）：
    // 收起右欄是要讓影片變大，字幕跟著變大的話字幕列變高，影片反而被擠小
    let colW = sr.width;
    if (mode === 'short' && !full && document.body.classList.contains('hide-right')) colW -= Math.min(380, Math.max(280, innerWidth * 0.34));
    sh = Math.min((colW * 9) / 16, sh);
  }
  const fs1o = subPx(sh, st.size1, 16);
  screen.setProperty('--fs1o', `${fs1o}px`);
  screen.setProperty('--fs2o', `${subPx(sh, st.size2, 14)}px`);
  // 影片外兩行的間距也跟著影片外的字級走，不然間距會隨影片大小變，字幕列高度又跟著變
  screen.setProperty('--gapo', `${(fs1o * st.gap) / 100}px`);
  reserveOutside();
  // 手機直向：查字卡、字幕外觀這些從底部滑出來的東西最多蓋到字幕列下緣（vocab.css 的 --below-screen）
  if (mode === 'stack') {
    const below = Math.round(innerHeight - screenEl.getBoundingClientRect().bottom);
    document.documentElement.style.setProperty('--below-screen', `${below}px`);
  }
}

// 影片外的字幕列固定高度，字幕出現、消失、換句時影片都不動。
// 高度不用估算：拿一個看不見的複本，套上跟字幕一樣的類別、字型、假名和粗體，
// 每行放「保留行數」行的樣本字，連同上下留白和邊線一起量。字級、字型、視窗大小、假名設定變了都會重新量
const outsideReserve = { outTop: 0, outBottom: 0 };
let outsideMeasure = null;
function measureOutside(zone, lines, rows) {
  if (!outsideMeasure) {
    outsideMeasure = document.createElement('div');
    outsideMeasure.setAttribute('aria-hidden', 'true');
    $('#screen').appendChild(outsideMeasure);
  }
  outsideMeasure.className = `outside active sub-measure ${zone === 'outTop' ? 'top' : 'bottom'}`;
  outsideMeasure.innerHTML = lines.map((el) => {
    const cls = [...el.classList].filter((c) => c !== 'empty').join(' ');
    const lang = el.dataset.lang ? ` data-lang="${esc(el.dataset.lang)}"` : '';
    // 有假名的行，樣本也帶假名，量到的就是有假名時的行高
    const row = (el.classList.contains('has-ruby') ? '<ruby>字<rt>じ</rt></ruby>' : '') + '字あAgy';
    return `<div class="${cls}"${lang}><span>${Array(rows).fill(row).join('\n')}</span></div>`;
  }).join('');
  const h = outsideMeasure.getBoundingClientRect().height;
  outsideMeasure.innerHTML = '';
  return h;
}

// 手機直向的影片框照影片比例固定（style.css 的 stack 版面），大小不跟著字幕列變
const playerFixedSize = () => getComputedStyle(player).flexGrow === '0';

function reserveOutside() {
  const rows = S.settings.reserveLines || 2;
  for (const zone of ['outTop', 'outBottom']) {
    const box = $(`#zone-${zone}`);
    const lines = [...box.children].filter((el) => !el.hidden);
    box.classList.toggle('active', lines.length > 0);
    // 播放器沒顯示時量不到（高度 0），先不固定，下次版面變化會再量
    const h = lines.length ? measureOutside(zone, lines, rows) : 0;
    outsideReserve[zone] = h ? Math.ceil(h) : 0;
    // 手機直向也跟桌面一樣固定高度：字幕列長高的話，下面的分頁列和逐字稿每換一句就會上下跳
    box.style.minHeight = '';
    box.style.height = outsideReserve[zone] ? `${outsideReserve[zone]}px` : '';
  }
  updateOutsideMode();
  fitOutside();
}

// 一句字幕超過保留的行數時，字幕列高度還是不變，多出來的行疊到影片上（加底色），影片不會被推動
function fitOutside() {
  for (const zone of ['outTop', 'outBottom']) {
    const box = $(`#zone-${zone}`);
    let over = false;
    if (outsideReserve[zone] && box.classList.contains('active')) {
      const cs = getComputedStyle(box);
      const shown = [...box.children].filter((el) => !el.hidden && !el.classList.contains('empty'));
      const used = shown.reduce((sum, el) => sum + el.getBoundingClientRect().height, 0)
        + (parseFloat(cs.rowGap) || 0) * Math.max(0, shown.length - 1)
        + parseFloat(cs.paddingTop) + parseFloat(cs.paddingBottom) + parseFloat(cs.borderTopWidth) + parseFloat(cs.borderBottomWidth);
      over = used > outsideReserve[zone] + 0.5;
    }
    box.classList.toggle('over', over);
  }
}

// 影片本來就有黑邊（letterbox）時，字幕列直接浮在黑邊上，影片不用縮小；黑邊不夠才把影片擠小。
// 用保留的高度判斷，不看當下字幕有幾行，字幕出現、消失或折行時才不會在兩種模式之間來回切換
function updateOutsideMode() {
  const scr = $('#screen').getBoundingClientRect();
  const aspect = video.videoWidth && video.videoHeight ? video.videoHeight / video.videoWidth : null;
  const zones = ['outTop', 'outBottom'].map((id) => ({ box: $(`#zone-${id}`), h: outsideReserve[id] }));
  const active = zones.filter((z) => z.box.classList.contains('active'));
  const floating = new Set();
  if (active.length && S.settings.useLetterbox !== false && aspect) {
    // 窄螢幕版面的影片高度固定，跟字幕列無關；一般版面是整個播放區扣掉留在版面裡的字幕列
    const fixedH = playerFixedSize() ? player.getBoundingClientRect().height : 0;
    const letterbox = (playerH) => (playerH - Math.min(playerH, scr.width * aspect)) / 2;
    if (fixedH) {
      for (const z of active) if (letterbox(fixedH) >= z.h + 4) floating.add(z.box);
    } else if (active.every((z) => letterbox(scr.height) >= z.h + 4)) {
      active.forEach((z) => floating.add(z.box));
    } else if (active.length === 2) {
      // 上下兩列的黑邊不夠一起浮：另一列留在版面裡把影片擠小之後，看這一列放不放得進黑邊
      for (const z of active) {
        const other = active.find((o) => o !== z);
        if (letterbox(scr.height - other.h) >= z.h + 4) floating.add(z.box);
      }
    }
  }
  for (const z of zones) z.box.classList.toggle('floating', floating.has(z.box));
  // 手機、平板：浮在黑邊上的字幕列貼著影片放。黑邊很高時（平板橫放、直拿全螢幕）放在畫面邊緣會離影片很遠，
  // 還會跟控制列疊在一起。位置只跟保留高度和播放區大小有關，換句時一樣不會動
  let edge = -1;
  if (floating.size && layoutMode() !== 'desktop') {
    const playerH = scr.height - active.filter((z) => !floating.has(z.box)).reduce((sum, z) => sum + z.h, 0);
    edge = (playerH - Math.min(playerH, scr.width * aspect)) / 2;
  }
  for (const z of zones) {
    const off = edge >= 0 && floating.has(z.box) ? `${Math.max(0, Math.floor(edge - z.h))}px` : '';
    const side = z.box.classList.contains('top') ? 'top' : 'bottom';
    if (z.box.style[side] !== off) z.box.style[side] = off;
  }
  const screenStyle = $('#screen').style;
  // 觸控：影片外下方的字幕列在影片正下方時，超過保留行數的那一行會疊在影片最底下。
  // 控制列往上讓出一行的高度（style.css 的 --ctrl-lift），控制列出現時那一行的字照樣點得到。
  // 用行高算、不看當下這句有沒有超過，按鈕才不會隨換句上下移動
  const ob = $('#zone-outBottom');
  let ctrlLift = 0;
  if (ob.classList.contains('active') && !floating.has(ob) && outsideReserve.outBottom) {
    for (const el of ob.children) if (!el.hidden) ctrlLift = Math.max(ctrlLift, parseFloat(getComputedStyle(el).lineHeight) || 0);
  }
  screenStyle.setProperty('--ctrl-lift', `${Math.ceil(ctrlLift)}px`);
  // 觸控時控制列出現，影片內下方和浮在黑邊上的字幕要往上讓開多少（style.css 的 --lift-in、--lift-float）。
  // 只看版面、控制列高度和「離畫面邊緣」設定，不看當下字幕有幾行，換句時不會變。
  // 控制列上緣那幾 px 是透明的漸層，字壓到一點沒關係
  const controls = $('#controls');
  const ctrlH = controls.offsetHeight;
  const pr = player.getBoundingClientRect();
  const ctrlTop = pr.bottom - (parseFloat(getComputedStyle(controls).bottom) || 0) - Math.max(0, ctrlH - 6);
  const subsBottom = pr.top + (parseFloat(subsEl.style.top) || 0) + (parseFloat(subsEl.style.height) || pr.height);
  const liftIn = ctrlH ? Math.max(0, subsBottom - (parseFloat(screenStyle.getPropertyValue('--pad')) || 0) - ctrlTop) : 0;
  const obBottom = scr.bottom - (parseFloat(ob.style.bottom) || 0);
  const liftFloat = ctrlH && floating.has(ob) ? Math.max(0, obBottom - ctrlTop) : 0;
  screenStyle.setProperty('--lift-in', `${Math.round(liftIn)}px`);
  screenStyle.setProperty('--lift-float', `${Math.round(liftFloat)}px`);
}
new ResizeObserver(layoutSubs).observe(player);
document.fonts?.ready.then(() => layoutSubs());

function setSetting(key, value) {
  S.settings[key] = value;
  saveSettings();
  applySubSettings();
}
for (const [id, key] of [['#set-size1', 'size1'], ['#set-size2', 'size2'], ['#set-pos', 'pos'], ['#set-gap', 'gap'],
  ['#set-bg', 'bg'], ['#set-width', 'width']]) {
  $(id).addEventListener('input', (e) => setSetting(key, +e.target.value));
}
$('#set-reserve').addEventListener('change', (e) => setSetting('reserveLines', +e.target.value));
$('#set-letterbox').addEventListener('change', (e) => setSetting('useLetterbox', e.target.checked));
function setFuri(mode) {
  setSetting('furi', mode);
  renderTranscript();
  applySubSettings();
  updateSubs(true);
}
$('#set-furi').addEventListener('change', (e) => setFuri(e.target.value));
$('#set-style').addEventListener('change', (e) => setSetting('style', e.target.value));
$('#set-blur2').addEventListener('change', (e) => setSetting('blur2', e.target.checked));
$('#set-reset').addEventListener('click', () => {
  for (const k of LOOK_KEYS) S.settings[k] = DEFAULT_SETTINGS[k];
  saveSettings();
  renderTranscript();
  applySubSettings();
  updateSubs(true);
  toast('字幕外觀已恢復預設');
});
$('#tr-follow').addEventListener('change', (e) => { S.settings.follow = e.target.checked; saveSettings(); });

for (const slot of [1, 2]) {
  const sel = $(`#place-${slot}`);
  sel.innerHTML = Object.entries(PLACES).map(([k, v]) => `<option value="${k}">${v}</option>`).join('');
  sel.addEventListener('change', (e) => setSetting('place' + slot, e.target.value));
}

/* ----- 控制列 ----- */

function updatePlayButton() {
  $('#c-play use').setAttribute('href', video.paused ? '#i-play' : '#i-pause');
}
function togglePlay() {
  if (!S.current || !video.src) return;
  if (video.paused) video.play().catch(() => {}); else video.pause();
}
$('#c-play').addEventListener('click', togglePlay);

// 最近一次是用手指還是滑鼠：手指點完瀏覽器會補送滑鼠事件（mousemove、mouseleave），不能當成滑鼠移動
let lastPointer = 'mouse';
let touchAt = -1e9;
const recentTouch = () => performance.now() - touchAt < 800;
document.addEventListener('pointerdown', (e) => {
  lastPointer = e.pointerType;
  if (e.pointerType === 'touch') touchAt = performance.now();
}, true);
document.addEventListener('pointermove', (e) => { if (e.pointerType === 'mouse') lastPointer = 'mouse'; }, { capture: true, passive: true });

let clickTimer = 0;
video.addEventListener('click', () => {
  if (recentTouch()) return; // 手指點影片是叫出、收起控制列（下面 player 的 click）
  clearTimeout(clickTimer);
  clickTimer = setTimeout(togglePlay, 220);
});
video.addEventListener('dblclick', () => { if (recentTouch()) return; clearTimeout(clickTimer); toggleFullscreen(); });
// 觸控：點影片畫面叫出控制列，再點一次收起來。點到按鈕、進度條、字幕上的字就照它們自己的動作
player.addEventListener('click', (e) => {
  if (!recentTouch() || !player.classList.contains('has-media')) return;
  if (e.target.closest('button, select, input, a, .seek, .sub-line > span, #wpop, .player-msg')) return;
  if (player.classList.contains('controls-on')) hideControls(); else pokeControls();
});
// 遮住翻譯：觸控時點一下翻譯顯示、再點一下遮回去（滑鼠照舊移上去顯示）
$('#screen').addEventListener('click', (e) => {
  const span = e.target.closest('.sub-line.blur > span');
  if (span && lastPointer !== 'mouse') span.parentElement.classList.toggle('peek');
});
// 假名「點字才顯示」（滑鼠是移上去才顯示）：手指點一個詞查字時，順便把這個詞的假名亮出來，換句後照舊隱藏。
// 查字關著時點假名本來就會打開修改讀音的彈窗並顯示假名，不用另外處理
document.addEventListener('click', (e) => {
  if (lastPointer === 'mouse' || S.settings.furi !== 'hover' || !window.WL?.active()) return;
  const w = e.target.closest?.('.w');
  if (!w || !w.closest('#screen, #transcript')) return;
  const rubies = [...w.querySelectorAll('ruby')];
  const on = !rubies.every((r) => r.classList.contains('reveal'));
  rubies.forEach((r) => r.classList.toggle('reveal', on));
}, true);
// 觸控裝置沒有「滑鼠移上去」：假名選項的文字跟著改
const touchMQ = matchMedia('(hover: none) and (pointer: coarse)');
function applyTouchLabels() {
  $('#set-furi option[value="hover"]').textContent = touchMQ.matches ? '點字才顯示' : FURI_MODES.hover;
}
touchMQ.addEventListener?.('change', applyTouchLabels);
applyTouchLabels();

function updateSeekUI() {
  const d = video.duration || 0, t = video.currentTime || 0;
  $('#seek-fill').style.width = d ? `${(t / d) * 100}%` : '0';
  let buf = 0;
  // video.buffered 每次讀都是新的快照，邊跑迴圈邊讀的話中途多一段或少一段會超出範圍，先存起來
  const ranges = video.buffered;
  for (let i = 0; i < ranges.length; i++) {
    if (ranges.start(i) <= t + 1) buf = Math.max(buf, ranges.end(i));
  }
  $('#seek-buf').style.width = d ? `${(buf / d) * 100}%` : '0';
  $('#c-time').textContent = `${fmtTime(t)} / ${fmtTime(d)}`;
}

const seekEl = $('#seek');
function seekRatio(e) {
  const r = seekEl.getBoundingClientRect();
  return Math.min(1, Math.max(0, (e.clientX - r.left) / r.width));
}
seekEl.addEventListener('pointerdown', (e) => {
  if (!video.duration) return;
  if (e.pointerType !== 'mouse') e.preventDefault(); // 手指拖進度條時不要選到文字
  seekEl.setPointerCapture(e.pointerId);
  seekEl.classList.add('drag');
  video.currentTime = seekRatio(e) * video.duration;
  updateSeekUI();
});
seekEl.addEventListener('pointermove', (e) => {
  if (!video.duration) return;
  const ratio = seekRatio(e);
  const tip = $('#seek-tip');
  tip.hidden = false;
  tip.style.left = `${ratio * 100}%`;
  tip.textContent = fmtTime(ratio * video.duration);
  if (seekEl.classList.contains('drag')) { video.currentTime = ratio * video.duration; updateSeekUI(); }
});
seekEl.addEventListener('pointerup', () => seekEl.classList.remove('drag'));
seekEl.addEventListener('pointerleave', () => { $('#seek-tip').hidden = true; });

function seekBy(sec) {
  if (!video.duration) return;
  video.currentTime = Math.min(video.duration, Math.max(0, video.currentTime + sec));
  pokeControls();
}

function navCues() {
  return S.cues[S.primary].length ? S.cues[S.primary] : S.cues[S.primary === 1 ? 2 : 1];
}
// 跳到某句時停在句首前一點點（比 CUE_LEAD 小，停著時字幕才顯示得出來，播放時開頭的音也不會被切掉）
const JUMP_LEAD = 0.05;

// 跳到某句要停在哪個時間：句首前一點點，但不能早到前一句還在的時間裡。句子之間常常只隔幾十毫秒，
// 早到前一句裡面的話，畫面上的字幕會還是前一句，看起來像沒換句（vocab.js 的 leadTime 同一個道理）
function jumpAt(start) {
  let at = Math.max(0, start - JUMP_LEAD);
  for (const slot of [1, 2]) {
    const cues = S.cues[slot];
    const i = lastStarted(cues, at);
    for (const c of [cues[i - 1], cues[i]]) {
      if (c && c.start < start - 0.001 && c.end > at && c.end <= start + 0.001) at = Math.min(start, c.end + 0.002);
    }
  }
  return at;
}
// 剛跳到的是第幾句。影片還停在那個位置時就以這一句為準，不從 currentTime 反推：
// 瀏覽器跳轉的時間有幾十毫秒誤差，連續快按時也可能還沒跳完，反推會算成同一句，按了像沒反應
const NAV = { cues: null, i: -1, t: -1e9, at: -1e9, seekAt: -1e9 };
// 連續按上一句、下一句時，從上次跳到的那一句往前後算，不看播放到哪裡。
// 不然句子比按鍵間隔短時，影片已經播回下一句，按上一句又跳回同一句，看起來像按了沒反應
const NAV_CHAIN_MS = 2000;

// 現在算第幾句：跟逐字稿高亮一樣，用「最後一句開始的」；剛跳過去、而且還在那一句裡面，就用剛才跳到的那一句
// （只看時間差的話，0.05 秒那種很短的句子會在 0.3 秒內播過好幾句，按下一句反而往回跳）
function navIdx(cues) {
  const t = video.currentTime;
  const cue = NAV.cues === cues && NAV.i >= 0 ? cues[NAV.i] : null;
  if (cue && t >= NAV.t - 0.05 && t < cue.end && t - NAV.t < 0.3) return NAV.i;
  return lastStarted(cues, t);
}
// 畫面上要顯示第幾句：前後句時間重疊、或句子比 CUE_LEAD 還短時，同一個時間點會對到好幾句。
// 剛跳過去而且還在那一句裡面時，就以它為準；另一條字幕軌挑開始時間跟它最接近的那一句，兩行才會是同一句
function navCue() {
  const cue = NAV.i >= 0 && NAV.cues ? NAV.cues[NAV.i] : null;
  const t = video.currentTime;
  return cue && t >= cue.start - CUE_LEAD && t < cue.end ? cue : null;
}
function cueAt(cues, t, nav = navCue()) {
  if (nav && cues === NAV.cues) return NAV.i;
  if (nav) {
    // 翻譯軌通常跟原文一句對一句（逐行對照翻譯），對得起來就直接用同一個行號，
    // 不然時間完全一樣的連續短句分不出誰是誰
    const same = cues[NAV.i];
    if (same && Math.abs(same.start - nav.start) < 0.01) return NAV.i;
    let best = -1, diff = Infinity;
    const near = lastStarted(cues, t);
    for (let k = Math.max(0, near - 3); k <= Math.min(cues.length - 1, near + 3); k++) {
      const c = cues[k];
      if (!c || t < c.start - CUE_LEAD || t >= c.end) continue;
      const d = Math.abs(c.start - nav.start);
      if (d < diff) { diff = d; best = k; }
    }
    if (best >= 0) return best;
  }
  return findCue(cues, t);
}
function jumpTo(cues, i) {
  const cue = cues && cues[i];
  if (!cue) return;
  const at = jumpAt(cue.start);
  NAV.seekAt = performance.now();
  video.currentTime = at;
  NAV.cues = cues; NAV.i = i; NAV.t = at; NAV.at = NAV.seekAt;
  if (S.loop) setLoop(cue);
  updateSubs(true);
}
// 使用者自己拉進度條、用方向鍵快轉、從查字卡跳過去：不算連續按上下句，下一次要從新位置算
function navReset() {
  NAV.cues = null; NAV.i = -1; NAV.t = -1e9; NAV.at = -1e9;
}
// 連續按上一句、下一句、重播、循環時都以剛跳到的那一句為準
function chainIdx(cues) {
  if (NAV.cues === cues && NAV.i >= 0 && performance.now() - NAV.at < NAV_CHAIN_MS) return NAV.i;
  return navIdx(cues);
}
// 上一句、下一句：一定從現在這一句往前或往後一句，按一次就換一句，不會停在原地
function stepCue(dir) {
  const cues = navCues();
  if (!cues.length) return;
  const to = chainIdx(cues) + dir;
  if (to >= cues.length || to < -1) return;   // 已經在最後一句，或還在第一句之前
  jumpTo(cues, Math.max(0, to));              // 在第一句裡面按上一句：回到這一句的開頭
}
function prevCue() { stepCue(-1); }
function nextCue() { stepCue(1); }
function replayCue() {
  const cues = navCues();
  const i = chainIdx(cues);
  if (i >= 0) { jumpTo(cues, i); if (video.paused) video.play().catch(() => {}); }
}
function setLoop(cue) {
  S.loop = cue ? { start: jumpAt(cue.start), end: cue.end } : null;
  $('#c-loop').classList.toggle('on', !!S.loop);
}
function toggleLoop() {
  if (S.loop) { setLoop(null); return; }
  const cues = navCues();
  const i = chainIdx(cues);
  if (i >= 0) setLoop(cues[i]); else toast('這個位置沒有字幕可以循環');
}
$('#c-prev').addEventListener('click', prevCue);
$('#c-next').addEventListener('click', nextCue);
$('#c-replay').addEventListener('click', replayCue);
$('#c-loop').addEventListener('click', toggleLoop);

function setRate(rate) {
  S.settings.rate = rate; saveSettings();
  video.playbackRate = rate;
  $('#c-rate').value = String(rate);
}
$('#c-rate').addEventListener('change', (e) => setRate(+e.target.value));
function stepRate(dir) {
  const opts = $$('#c-rate option').map((o) => +o.value);
  const i = opts.indexOf(S.settings.rate);
  const next = opts[Math.min(opts.length - 1, Math.max(0, (i < 0 ? opts.indexOf(1) : i) + dir))];
  setRate(next);
  toast(`播放速度 ${next}x`);
}

function applyVolume() {
  video.volume = S.settings.volume;
  video.muted = S.settings.muted;
  $('#c-vol').value = S.settings.muted ? 0 : S.settings.volume;
  $('#c-mute use').setAttribute('href', S.settings.muted || S.settings.volume === 0 ? '#i-mute' : '#i-vol');
}
$('#c-vol').addEventListener('input', (e) => {
  S.settings.volume = +e.target.value; S.settings.muted = S.settings.volume === 0; saveSettings(); applyVolume();
});
$('#c-mute').addEventListener('click', () => {
  S.settings.muted = !S.settings.muted;
  if (!S.settings.muted && S.settings.volume === 0) S.settings.volume = 0.6;
  saveSettings(); applyVolume();
});

const fsElement = () => document.fullscreenElement || document.webkitFullscreenElement || null;
function toggleFullscreen() {
  // 全螢幕包含影片外的字幕列，不然那兩行字幕會不見
  if (document.body.classList.contains('pseudo-fs')) { setPseudoFullscreen(false); return; }
  if (fsElement()) { (document.exitFullscreen || document.webkitExitFullscreen).call(document); return; }
  const el = $('#screen');
  const request = el.requestFullscreen || el.webkitRequestFullscreen;
  // iPhone 的 Safari 只能讓 video 本身全螢幕（字幕列、查字都會不見），改成把播放區蓋滿畫面
  if (!request) { setPseudoFullscreen(true); return; }
  Promise.resolve(request.call(el)).then(() => {
    // 手機拿直的按全螢幕：橫式影片直接轉成橫向（Android 可以，不支援的瀏覽器會拒絕，不用管）
    if (lastPointer === 'touch' && video.videoWidth > video.videoHeight) window.screen.orientation?.lock?.('landscape').catch(() => {});
  }).catch(() => { if (lastPointer !== 'mouse') setPseudoFullscreen(true); });
}
function setPseudoFullscreen(on) {
  document.body.classList.toggle('pseudo-fs', on);
  onFullscreenChange();
}
function onFullscreenChange() {
  const on = !!fsElement() || document.body.classList.contains('pseudo-fs');
  document.body.classList.toggle('in-fs', on);
  $('#c-fs use').setAttribute('href', on ? '#i-fs-exit' : '#i-fs');
  if (!fsElement() && lastPointer !== 'mouse') { try { window.screen.orientation?.unlock?.(); } catch { /* 不支援 */ } }
  setTimeout(layoutSubs, 50);
}
$('#c-fs').addEventListener('click', toggleFullscreen);
document.addEventListener('onfullscreenchange' in document ? 'fullscreenchange' : 'webkitfullscreenchange', onFullscreenChange);

let idleTimer = 0;
function showControls() {
  player.classList.add('controls-on');
  player.classList.remove('idle');
  clearTimeout(idleTimer);
}
function hideControls() {
  clearTimeout(idleTimer);
  player.classList.remove('controls-on');
  player.classList.add('idle');
}
function pokeControls() {
  showControls();
  // 手指點過按鈕之後 :hover 會一直留著，觸控時不看它；手指操作多給一秒
  const touch = lastPointer !== 'mouse';
  idleTimer = setTimeout(() => {
    if (!video.paused && (touch || !$('#controls').matches(':hover')) && !window.WL?.cardOpen()) {
      player.classList.remove('controls-on');
      player.classList.add('idle');
    }
  }, touch ? 3000 : 2000);
}
// 只有游標在影片區域才顯示控制列；移出去立刻收起來，影片外的字幕列不會觸發
// （手指點完瀏覽器補送的滑鼠事件不算）
player.addEventListener('mousemove', () => { if (!recentTouch()) pokeControls(); });
player.addEventListener('mouseenter', () => { if (!recentTouch()) pokeControls(); });
player.addEventListener('mouseleave', () => {
  if (recentTouch()) return;
  clearTimeout(idleTimer);
  player.classList.remove('controls-on', 'idle');
});
// 手指按控制列上的東西（播放、換句、拖進度條）時，控制列重新計時，不會按到一半收起來
$('#controls').addEventListener('pointerdown', (e) => { if (e.pointerType !== 'mouse') pokeControls(); });
// 收起的控制列只是變透明。手機、平板由 style.css 讓它點不到；有觸控螢幕的筆電這類裝置 CSS 分不出來，
// 這裡再擋一次：手指點到看不見的控制列，只叫出控制列，不按到底下的按鈕、不跳轉
let ghostTapAt = -1e9;
$('#controls').addEventListener('pointerdown', (e) => {
  ghostTapAt = -1e9;
  if (e.pointerType === 'mouse' || player.classList.contains('controls-on')) return;
  ghostTapAt = performance.now();
  e.preventDefault();
  e.stopPropagation();
  pokeControls();
}, true);
$('#controls').addEventListener('click', (e) => {
  if (performance.now() - ghostTapAt > 1000) return;
  ghostTapAt = -1e9;
  e.preventDefault();
  e.stopPropagation();
}, true);

/* ================= 逐字稿 ================= */

function renderTranscript() {
  S.primary = S.cues[1].length ? 1 : 2;
  const P = S.cues[S.primary];
  const Q = S.cues[S.primary === 1 ? 2 : 1];
  const ol = $('#transcript');
  S.hi = -1;
  if (!P.length) {
    ol.innerHTML = '';
    renderTranscriptStatus(mediaById(S.current));
    return;
  }
  const aligned = P.length === Q.length && P.every((c, i) => Math.abs(c.start - Q[i].start) < 0.01);
  let lastJ = -1;
  S.pairs = P.map((c, i) => {
    if (!Q.length) return '';
    if (aligned) return Q[i].text;
    // 一句譯文橫跨好幾行原文時，只在第一行下面顯示
    const j = findCue(Q, (c.start + c.end) / 2);
    if (j < 0 || j === lastJ) return '';
    lastJ = j;
    return Q[j].text;
  });
  const keepScroll = ol.dataset.media === S.current ? ol.scrollTop : 0;
  ol.classList.toggle('with-ruby', hasRuby(P));
  // chk === 0：檢查時間軸後這行仍然對不上，時間旁邊加個小圓點
  const chk = '<i class="chk" title="這行的時間可能對不上"></i>';
  ol.innerHTML = P.map((c, i) => `<li data-i="${i}"><time${c.chk === 0 ? ' title="這行的時間可能對不上"' : ''}>${fmtTime(c.start)}${c.chk === 0 ? chk : ''}</time><div><p class="a" data-cue="${S.primary}:${i}">${cueHtml(c)}</p>${S.pairs[i] ? `<p class="b">${esc(S.pairs[i])}</p>` : ''}</div></li>`).join('');
  ol.dataset.media = S.current || '';
  ol.scrollTop = keepScroll;
  applyTranscriptFilter();
  renderTranscriptStatus(mediaById(S.current));
  updateSubs(true);
}

function highlightTranscript(i) {
  const ol = $('#transcript');
  const prev = ol.querySelector('li.on');
  if (prev) prev.classList.remove('on');
  S.hi = i;
  if (i < 0) return;
  const li = ol.children[i];
  if (!li) return;
  li.classList.add('on');
  if (S.settings.follow && Date.now() - S.userScrollAt > 4000 && !$('#pane-transcript').hidden) {
    const box = ol.getBoundingClientRect();
    const r = li.getBoundingClientRect();
    if (r.top < box.top + 40 || r.bottom > box.bottom - box.height * 0.4) {
      const top = ol.scrollTop + (r.top - box.top) - box.height * 0.3;
      // 正常播放往下一句才用平滑捲動；快轉、跳轉或距離很遠時直接到位，不讓使用者等動畫
      const jumped = performance.now() - S.seekAt < 800 || Math.abs(top - ol.scrollTop) > box.height;
      ol.scrollTo({ top, behavior: jumped ? 'instant' : 'smooth' });
    }
  }
}

$('#transcript').addEventListener('click', (e) => {
  if (e.target.closest('.w') && window.WL?.active()) return; // 點字是查字，不跳句
  const li = e.target.closest('li[data-i]');
  if (!li) return;
  // 遮住翻譯時手指點翻譯：只把這句翻譯亮出來，再點一次遮回去，不跳句（滑鼠照舊移上去顯示）
  if (S.settings.blur2 && lastPointer !== 'mouse' && e.target.closest('.b')) { li.classList.toggle('peek'); return; }
  const cues = S.cues[S.primary];
  const i = +li.dataset.i;
  if (!cues[i]) return;
  S.transcriptClickAt = performance.now();
  jumpTo(cues, i);
  // 點句子是「從這裡播」，不是查字：這一句播完不要被「查字時暫停」停下來
  window.WL?.onPlayFrom();
});
for (const ev of ['wheel', 'touchmove', 'keydown']) {
  $('#transcript').addEventListener(ev, () => { S.userScrollAt = Date.now(); }, { passive: true });
}

function applyTranscriptFilter() {
  const q = $('#tr-search').value.trim().toLowerCase();
  const P = S.cues[S.primary];
  const items = $$('#transcript li[data-i]');
  for (const li of items) {
    const i = +li.dataset.i;
    const hit = !q || (P[i]?.text || '').toLowerCase().includes(q) || (S.pairs[i] || '').toLowerCase().includes(q);
    li.hidden = !hit;
  }
}
$('#tr-search').addEventListener('input', applyTranscriptFilter);

function renderTranscriptStatus(m) {
  const el = $('#tr-status');
  if (!m) { el.hidden = true; return; }
  const job = jobsOf(m.id).find((j) => j.status === 'running' && (j.type === 'transcribe' || j.type === 'translate' || j.type === 'download'))
    || activeJobOf(m.id);
  const hasCues = S.cues[1].length || S.cues[2].length;
  if (job && job.type !== 'proxy') {
    const pct = Math.round((job.progress || 0) * 100);
    el.innerHTML = job.status === 'running'
      ? `<span>${esc(JOB_LABEL[job.type])}${jobStage(job) ? '：' + esc(jobStage(job)) : ''} ${pct}%</span><div class="minibar"><i style="width:${pct}%"></i></div>`
      : job.waiting ? `<span>${esc(JOB_LABEL[job.type])}：${waitingHtml(job.waiting, false)}</span>`
        : `<span>${esc(JOB_LABEL[job.type])}排隊中</span>`;
    el.hidden = false;
  } else {
    el.hidden = true;
  }
  if (!hasCues && !$('#transcript li[data-i]')) {
    const failed = jobsOf(m.id).filter((j) => j.status === 'failed').sort((a, b) => b.finished_at - a.finished_at)[0];
    let html;
    if (job) html = `<li class="pane-empty">字幕完成後會出現在這裡</li>`;
    else if (m.tracks.length) html = `<li class="pane-empty">目前沒有選擇字幕</li>`;
    else if (failed) html = `<li class="pane-empty"><span>上次${esc(JOB_LABEL[failed.type])}失敗</span><button class="btn" data-act="queue">查看佇列</button></li>`;
    else html = `<li class="pane-empty"><span>這部影片還沒有字幕</span><button class="btn" data-act="transcribe">轉字幕</button></li>`;
    $('#transcript').innerHTML = html;
  }
}
$('#transcript').addEventListener('click', (e) => {
  const act = e.target.closest('[data-act]')?.dataset.act;
  if (act === 'transcribe') openAddDialog('transcribe');
  if (act === 'queue') showTab('queue');
});

/* ================= 修改假名讀音 ================= */

const RP = {};
const rubyPop = $('#ruby-pop');

function rubyContext(el) {
  const line = el.closest('.sub-line');
  if (line) {
    const slot = line.id === 'line1' ? 1 : 2;
    return { slot, trackId: S.trackSel[slot], index: S.idx[slot] };
  }
  const li = el.closest('#transcript li[data-i]');
  if (li) return { slot: S.primary, trackId: S.trackSel[S.primary], index: +li.dataset.i };
  return null;
}

function closeRubyPop() {
  rubyPop.hidden = true;
  $$('ruby.reveal').forEach((r) => r.classList.remove('reveal'));
}

async function openRubyPop(el, ctx) {
  const cue = S.cues[ctx.slot][ctx.index];
  if (!cue) return;
  const s = +el.dataset.s, e = +el.dataset.e;
  const base = cue.text.slice(s, e);
  const current = (cue.ruby || []).find((r) => r[0] === s && r[1] === e)?.[2] || '';
  Object.assign(RP, ctx, { s, e, base, current, surface: base });
  el.classList.add('reveal');
  if (!video.paused) video.pause();

  $('#rp-word').innerHTML = `${esc(base)}<small>${esc(current || '（目前不標）')}</small>`;
  $('#rp-cands').innerHTML = '<span class="hint">讀取候選讀音…</span>';
  $('#rp-input').value = current;
  // 全螢幕時彈窗要放進全螢幕的元素裡才看得到
  (document.fullscreenElement || document.body).appendChild(rubyPop);
  rubyPop.hidden = false;
  const r = el.getBoundingClientRect();
  const w = rubyPop.offsetWidth, h = rubyPop.offsetHeight;
  const top = r.bottom + 8 + h > window.innerHeight ? Math.max(8, r.top - 8 - h) : r.bottom + 8;
  rubyPop.style.top = `${top}px`;
  rubyPop.style.left = `${Math.max(8, Math.min(window.innerWidth - w - 8, r.left + r.width / 2 - w / 2))}px`;

  try {
    const info = await api('GET', `/api/tracks/${ctx.trackId}/cues/${ctx.index}/lookup?s=${s}&e=${e}`);
    RP.surface = info.surface || base;
    const cands = [...new Set([current, ...(info.alternatives || [])].filter(Boolean))];
    $('#rp-cands').innerHTML = cands.length > 1
      ? cands.map((c) => `<button type="button" data-rt="${esc(c)}" class="${c === current ? 'on' : ''}">${esc(c)}</button>`).join('')
      : '<span class="hint">這個詞沒有其他常見讀法，讀音不對可以自己輸入</span>';
  } catch {
    $('#rp-cands').innerHTML = '<span class="hint">讀不到候選讀音，可以自己輸入</span>';
  }
}

async function applyRuby(rt) {
  try {
    const res = await api('PUT', `/api/tracks/${RP.trackId}/cues/${RP.index}/ruby`, { s: RP.s, e: RP.e, rt });
    for (const slot of [1, 2]) {
      if (S.trackSel[slot] === RP.trackId && S.cues[slot][RP.index]) S.cues[slot][RP.index] = res.cue;
    }
    closeRubyPop();
    renderTranscript();
    updateSubs(true);
    toast(rt ? `已改成「${rt}」（只改這一句）` : '這裡不再標假名');
  } catch (err) { toast(err.message, true); }
}

// capture 階段攔下點假名的動作，逐字稿就不會同時跳到那一句
document.addEventListener('click', (e) => {
  const el = e.target.closest('ruby[data-s], .no-ruby[data-s]');
  if (!el) {
    if (!rubyPop.hidden && !rubyPop.contains(e.target)) closeRubyPop();
    return;
  }
  // 假名在可查的詞裡：點下去是查字，改讀音改用解釋卡上的「修改讀音」
  if (el.closest('.w') && window.WL?.active()) return;
  const ctx = rubyContext(el);
  if (!ctx || !ctx.trackId || ctx.index < 0) return;
  e.preventDefault();
  e.stopPropagation();
  openRubyPop(el, ctx);
}, true);

$('#rp-cands').addEventListener('click', (e) => {
  const b = e.target.closest('[data-rt]');
  if (b) applyRuby(b.dataset.rt);
});
$('#rp-apply').addEventListener('click', () => {
  const rt = $('#rp-input').value.trim();
  if (!/^[ぁ-ゖー]+$/.test(rt)) { toast('讀音請輸入平假名', true); return; }
  applyRuby(rt);
});
$('#rp-input').addEventListener('keydown', (e) => { if (e.key === 'Enter') $('#rp-apply').click(); });
$('#rp-none').addEventListener('click', () => applyRuby(''));
$('#rp-close').addEventListener('click', closeRubyPop);
$('#rp-dict').addEventListener('click', async () => {
  const typed = $('#rp-input').value.trim();
  const guess = RP.surface === RP.base ? typed : '';
  const reading = prompt(`「${RP.surface}」整個詞的讀音（平假名）。只適合名詞，加入後所有日文字幕都會照這個讀。`, guess);
  if (!reading) return;
  try {
    await api('POST', '/api/furigana/dict', { surface: RP.surface, reading: reading.trim() });
    closeRubyPop();
    for (const slot of [1, 2]) if (S.trackSel[slot]) loadCues(slot, S.trackSel[slot]);
    toast(`已加入辭典：${RP.surface}（${reading.trim()}）`);
  } catch (err) { toast(err.message, true); }
});
document.addEventListener('keydown', (e) => { if (e.key === 'Escape' && !rubyPop.hidden) closeRubyPop(); });

/* ================= 影片資訊列 ================= */

function renderDeck(m) {
  const t = titleParts(m);
  $('#media-title').textContent = t.main;
  $('#media-title').title = m.title_zh ? `${m.title}\n${m.title_zh}` : m.title;
  $('#media-title-sub').textContent = t.sub;
  $('#media-title-sub').hidden = !t.sub;
  const source = { local: '本機檔案', upload: '上傳', url: '網址下載' }[m.source] || '';
  const parts = [
    m.duration ? fmtTime(m.duration) : null,
    m.width ? `${m.width}×${m.height}` : null,
    m.vcodec ? m.vcodec.toUpperCase() : null,
    m.language ? S.meta.languages[m.language] : null,
    source,
  ].filter(Boolean);
  // 來源可追溯：網址可以點回去，本機檔案顯示完整路徑
  const src = m.url
    ? `<a class="src" href="${esc(m.url)}" target="_blank" rel="noreferrer" title="${esc(m.url)}">${esc(m.url)}</a>`
    : m.path ? `<span class="src" title="${esc(m.path)}">${esc(m.path)}</span>` : '';
  $('#media-meta').innerHTML = parts.map((p) => `<span>${esc(p)}</span>`).join('') + src;
  const translatable = m.tracks.some((t) => t.lang === 'ja' || t.lang === 'en');
  const transcribing = jobsOf(m.id).some((j) => j.type === 'transcribe' && (j.status === 'queued' || j.status === 'running'));
  $('#btn-transcribe').disabled = !m.path || transcribing;
  $('#btn-transcribe').title = transcribing ? '這部影片已經在轉字幕了' : '';
  $('#btn-translate').disabled = !translatable;
  $('#btn-translate').title = translatable ? '' : '需要先有日文或英文字幕';
  $('#btn-files').disabled = false;
}

async function startTranscribe(tryOtherModel = false) {
  const m = mediaById(S.current);
  if (!m) return;
  const existing = m.tracks.filter((t) => t.kind === 'asr');
  // 字幕管理的「用其他模型轉字幕」在轉字幕途中也按得到，正在轉的也要先說
  const pending = jobsOf(m.id).filter((j) => j.type === 'transcribe' && (j.status === 'queued' || j.status === 'running'));
  if (existing.length || pending.length) {
    const ok = await askConfirm({
      title: existing.length ? '這部影片已經有字幕了' : '這部影片正在轉字幕',
      text: existing.length
        ? '再轉一次會重新佔用顯卡，並且多出一條新的字幕軌（舊的不會被覆蓋，可以在字幕管理裡比較）。確定要再轉一次嗎？'
        : '完成後字幕就會出現。確定要同時用其他模型再轉一次嗎？',
      items: [
        ...existing.map((t) => `現有：${trackLabel(t, m.tracks)}，${t.cue_count} 句`),
        ...pending.map((j) => `${j.status === 'running' ? '正在轉' : '排隊中'}：${jobModel(j)}`),
      ],
      ok: '還是再轉一次',
    });
    if (!ok || S.current !== m.id) return;
  }
  if ($('#dlg-tracks').open) $('#dlg-tracks').close();
  openAddDialog('transcribe', existing.length > 0 || pending.length > 0);
  if (tryOtherModel || existing.length || pending.length) {
    // 已經有字幕時預選一個還沒用過的辨識模型，方便比較
    const used = new Set([...existing.map((t) => t.model), ...pending.map(jobModel)]);
    const other = [...$('#adv-engine').options].find((o) => o.value && !used.has(S.meta.engines[o.value]?.label));
    if (other) $('#adv-engine').value = other.value;
  }
}
$('#btn-transcribe').addEventListener('click', () => startTranscribe(false));
$('#btn-translate').addEventListener('click', () => openTranslateDialog());

/* ----- 選單 ----- */

let openMenuEl = null;
function closeMenu() { if (openMenuEl) { openMenuEl.hidden = true; openMenuEl = null; } }
function openMenu(btn, menu) {
  closeMenu();
  menu.hidden = false;
  const r = btn.getBoundingClientRect();
  const w = menu.offsetWidth, h = menu.offsetHeight;
  // 下方放不下就往上展開
  const top = r.bottom + 4 + h > window.innerHeight - 8 ? Math.max(8, r.top - 4 - h) : r.bottom + 4;
  menu.style.top = `${top}px`;
  menu.style.left = `${Math.max(8, Math.min(window.innerWidth - w - 8, r.right - w))}px`;
  openMenuEl = menu;
}
document.addEventListener('pointerdown', (e) => {
  if (openMenuEl && !openMenuEl.contains(e.target) && !e.target.closest('[data-menu]')) closeMenu();
});

/* ----- 字幕管理 ----- */

function renderTrackManager() {
  const m = mediaById(S.current);
  if (!m) return;
  const box = $('#track-table');
  const byId = Object.fromEntries(m.tracks.map((t) => [t.id, t]));
  if (!m.tracks.length) {
    box.innerHTML = '<div class="track-empty">這部影片還沒有字幕</div>';
  } else {
    // 原文字幕由新到舊，每條底下接著從它翻出來的翻譯；來源已刪除的翻譯放最後
    const ordered = [];
    for (const t of [...m.tracks].reverse()) {
      if (t.kind === 'translation') continue;
      ordered.push(t);
      ordered.push(...m.tracks.filter((x) => x.kind === 'translation' && x.source_track_id === t.id).reverse());
    }
    ordered.push(...m.tracks.filter((x) => x.kind === 'translation' && !byId[x.source_track_id]).reverse());
    box.innerHTML = ordered.map((t) => {
      const d = new Date(t.created_at * 1000);
      const when = `${d.getMonth() + 1}/${d.getDate()} ${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}`;
      const child = t.kind === 'translation' && byId[t.source_track_id];
      const origin = t.kind === 'translation'
        ? (child ? '翻譯自上面這條原文' : '翻譯（原文字幕已刪除，時間可能對不上）')
        : '語音辨識';
      const shown = S.trackSel[1] === t.id || S.trackSel[2] === t.id;
      const orphan = t.kind === 'translation' && !child;
      const canTranslate = t.kind === 'asr' && (t.lang === 'ja' || t.lang === 'en');
      const trJob = canTranslate && translateJobOf(t.id);
      const trBtn = !canTranslate ? ''
        : trJob ? `<button type="button" data-tr="${t.id}" disabled title="${esc(translateBusyText(trJob))}">${trJob.status === 'running' ? '翻譯中' : '等待翻譯'}</button>`
          : `<button type="button" data-tr="${t.id}">翻譯</button>`;
      const checking = t.kind === 'asr' && healthJob(t);
      // 原文列的「顯示」選為原文，翻譯跟著帶；翻譯列的「顯示」連原文一起切到它的來源。來源已刪除的翻譯不能顯示
      return `<div class="track-item${shown ? ' sel' : ''}${child ? ' child' : ''}">
        <div><div class="name">${esc(trackLabel(t, m.tracks))}</div>
          <div class="sub">${esc(origin)} · ${t.cue_count} 句 · ${when}</div>
          ${t.kind === 'asr' ? `<div class="sub">${esc(healthText(t))}</div>` : ''}</div>
        <div class="slots">
          ${orphan ? '' : `<button type="button" data-show="${t.id}" class="${shown ? 'on' : ''}">${shown ? '顯示中' : '顯示'}</button>`}
        </div>
        <div class="acts">
          ${trBtn}
          ${t.kind === 'asr' ? `<button type="button" data-health="${t.id}"${checking ? ' disabled' : ''}>檢查時間軸</button>` : ''}
          <a href="/api/tracks/${t.id}/download?fmt=srt">SRT</a>
          <a href="/api/tracks/${t.id}/download?fmt=vtt">VTT</a>
          <button type="button" class="del" data-del="${t.id}">刪除</button>
        </div>
      </div>`;
    }).join('');
  }
  const both = S.trackSel[1] && S.trackSel[2];
  const link = $('#tracks-bilingual');
  link.classList.toggle('disabled', !both);
  // 上下順序跟畫面一致：兩行在同一個位置而且交換過，翻譯放上面
  const flip = S.settings.swapLines && $('#line1').parentElement === $('#line2').parentElement;
  const [top, bottom] = flip ? [S.trackSel[2], S.trackSel[1]] : [S.trackSel[1], S.trackSel[2]];
  link.href = both ? `/api/media/${m.id}/bilingual?top=${top}&bottom=${bottom}&fmt=srt` : '#';
  $('#tracks-translate').disabled = !m.tracks.some((t) => t.kind === 'asr' && (t.lang === 'ja' || t.lang === 'en'));
  $('#tracks-transcribe').disabled = !m.path;
}

// 估計值：檢查時間軸每 100 行約 20 秒（跟伺服器 health.SECONDS_PER_LINE 一致）
const HEALTH_S_PER_LINE = 0.2;
const healthJob = (t) => S.jobs.find((j) => j.type === 'health' && (j.status === 'running' || j.status === 'queued')
  && (j.params || {}).track_id === t.id);

function healthText(t) {
  const job = healthJob(t);
  if (job) return job.status === 'running' ? `正在檢查時間軸 ${Math.round((job.progress || 0) * 100)}%` : '時間軸排隊等待檢查';
  const h = healthOf(t);
  if (!h) return '時間軸尚未檢查';
  const parts = [];
  if (h.fixed) parts.push(`修正 ${h.fixed} 行`);
  if (h.after_bad) parts.push(`仍有 ${h.after_bad} 行可能對不上`);
  let text = `時間軸已檢查：${parts.length ? parts.join('，') : '沒有發現對不上的行'}`;
  if (h.skipped_translations?.length) text += `，${h.skipped_translations.length} 條翻譯沒有跟著調整`;
  if (S.meta.health_version && h.v !== S.meta.health_version) text += '（用舊版方法檢查，可以重新檢查）';
  return text;
}

$('#btn-files').addEventListener('click', () => {
  if (!mediaById(S.current)) return;
  renderTrackManager();
  $('#dlg-tracks').showModal();
});

$('#track-table').addEventListener('click', async (e) => {
  const m = mediaById(S.current);
  if (!m) return;
  const showBtn = e.target.closest('[data-show]');
  const trBtn = e.target.closest('[data-tr]');
  const delBtn = e.target.closest('[data-del]');
  const healthBtn = e.target.closest('[data-health]');
  if (healthBtn) {
    const t = m.tracks.find((x) => x.id === healthBtn.dataset.health);
    if (!t) return;
    const checked = !!healthOf(t);
    if (checked) {
      const ok = await askConfirm({
        title: '這條字幕已經檢查過時間軸',
        text: `再檢查一次會重新佔用顯卡，預估${fmtRemain(t.cue_count * HEALTH_S_PER_LINE)}。只有更對得上的行才會調整時間，調整前的字幕檔會留一份備份。確定要再檢查嗎？`,
        items: [healthText(t)],
        ok: '再檢查一次',
      });
      if (!ok) return;
    }
    try {
      await api('POST', `/api/tracks/${t.id}/health`, { force: checked });
      toast('已加入佇列，檢查完字幕會自動更新');
      await refreshSoon();
      renderTrackManager();
      if (!$('#dlg-tracks').open) $('#dlg-tracks').showModal();
    } catch (err) { toast(err.message, true); }
  } else if (showBtn) {
    const t = m.tracks.find((x) => x.id === showBtn.dataset.show);
    if (!t) return;
    selectTrack(t.kind === 'translation' ? 2 : 1, t.id);
    renderTrackManager();
  } else if (trBtn) {
    openTranslateDialog(trBtn.dataset.tr); // 確認完才關掉字幕管理，取消的話停在這裡
  } else if (delBtn) {
    const t = m.tracks.find((x) => x.id === delBtn.dataset.del);
    if (!t) return;
    const kids = m.tracks.filter((x) => x.source_track_id === t.id);
    const ok = await askConfirm({
      title: '刪除字幕軌',
      text: `要刪除「${trackLabel(t, m.tracks)}」這條字幕嗎？刪掉之後要重新轉一次才會有。`,
      items: kids.length ? [`從它翻譯出來的 ${kids.length} 條翻譯也會一起刪除`] : [],
      ok: '刪除', warn: true,
    });
    if (!ok) return;
    try {
      await api('DELETE', `/api/tracks/${t.id}`);
      await refreshSoon();
      renderTrackManager();
      if (!$('#dlg-tracks').open) $('#dlg-tracks').showModal();
    } catch (err) { toast(err.message, true); }
  }
});
// 這兩個按鈕要先確認時，確認框疊在字幕管理上面，確定了才關掉字幕管理
$('#tracks-transcribe').addEventListener('click', () => startTranscribe(true));
$('#tracks-translate').addEventListener('click', () => openTranslateDialog());

$('#btn-look').addEventListener('click', (e) => openMenu(e.currentTarget, $('#menu-look')));

$('#btn-more').addEventListener('click', (e) => {
  const m = mediaById(S.current);
  if (!m) return;
  const menu = $('#menu-more');
  menu.innerHTML = [
    `<button class="menu-item" data-act="rename">重新命名</button>`,
    `<button class="menu-item" data-act="copy-source">複製來源${m.url ? '網址' : '路徑'}</button>`,
    m.url ? `<a class="menu-item" href="${esc(m.url)}" target="_blank" rel="noreferrer">開啟來源網址</a>` : '',
    m.path ? `<button class="menu-item" data-act="reveal">在檔案總管中顯示</button>` : '',
    m.path && !m.has_proxy ? `<button class="menu-item" data-act="proxy">轉成相容播放格式</button>` : '',
    `<div class="menu-sep"></div>`,
    `<button class="menu-item danger" data-act="delete">從播放列表移除</button>`,
  ].join('');
  openMenu(e.currentTarget, menu);
});
$('#menu-more').addEventListener('click', async (e) => {
  const act = e.target.closest('[data-act]')?.dataset.act;
  if (!act) return;
  closeMenu();
  const m = mediaById(S.current);
  if (!m) return;
  try {
    if (act === 'rename') {
      const title = prompt('新的名稱', m.title);
      if (title && title.trim() && title !== m.title) await api('PATCH', `/api/media/${m.id}`, { title });
    } else if (act === 'copy-source') {
      const text = m.url || m.path || '';
      try { await navigator.clipboard.writeText(text); toast('已複製：' + text); } catch { toast(text); }
    } else if (act === 'reveal') {
      await api('POST', `/api/media/${m.id}/reveal`);
    } else if (act === 'proxy') {
      await api('POST', `/api/media/${m.id}/proxy`);
      toast('已加入佇列，轉好後會自動換成新檔案播放');
    } else if (act === 'delete') {
      const ok = await askConfirm({
        title: '從播放列表移除',
        text: `要移除「${m.title}」和它的 ${m.tracks.length} 條字幕嗎？`,
        items: [m.source === 'local' ? '原始影片檔不會被刪除' : '下載或上傳的影片檔會一起刪除'],
        ok: '移除', warn: true,
      });
      if (!ok) return;
      await api('DELETE', `/api/media/${m.id}`);
      closeMedia();
    }
    refreshSoon();
  } catch (err) { toast(err.message, true); }
});

/* ================= 新增影片 / 轉字幕 對話框 ================= */

const dlgAdd = $('#dlg-add');
// 來源：url（YouTube 和其他 yt-dlp 支援的網站，預設）、upload（上傳檔案）
const A = { mode: 'add', src: 'url', lang: null, file: null, busy: false, force: false };

function segSet(seg, value) {
  for (const b of $$('button', seg)) b.classList.toggle('on', b.dataset.v === value);
}

function openAddDialog(mode = 'add', force = false) {
  const m = mode === 'transcribe' ? mediaById(S.current) : null;
  A.mode = mode;
  A.force = force;
  A.lang = m?.language || null;
  A.file = null;
  $('#add-title').textContent = m ? `轉字幕：${m.title}` : '新增影片';
  $('#source-field').hidden = !!m;
  $('#add-error').textContent = '';
  $('#upload-progress').hidden = true;
  $('#lang-seg').classList.remove('need');
  $('#drop-text').textContent = '把檔案拖到這裡，或點一下選擇';
  $('#upload-file').value = '';
  $('#url').value = '';
  A.plugin = null;
  $('#url-plugin').hidden = true;
  resetEpList();
  $('#do-translate').checked = S.settings.addTranslate !== false;
  $('#do-sensitive').checked = false;
  $('#adv-engine').dataset.lang = '';
  $('#adv-translator').dataset.lang = '';
  segSet($('#lang-seg'), A.lang);
  if (!m) setSource('url');   // 每次都從 YouTube 開始，最常用
  updateAddForm();
  dlgAdd.showModal();
  // 模型可能剛裝好或剛開始下載（第一次自動安裝），重新讀一次模型清單；使用者選過的不會被換掉
  refreshMeta().then(() => { if (dlgAdd.open) updateAddForm(); });
  // 打開就可以直接貼網址（手指點開的不要，手機的鍵盤一跳出來會把對話框蓋掉一半）
  if (!m && lastPointer === 'mouse') $('#url').focus();
}

function setSource(src) {
  A.src = src === 'upload' ? 'upload' : 'url';
  segSet($('#src-seg'), A.src);
  for (const p of $$('.src-pane')) p.hidden = p.dataset.pane !== A.src;
  updateAddForm();
}
$('#src-seg').addEventListener('click', (e) => {
  const b = e.target.closest('button'); if (!b) return;
  setSource(b.dataset.v);
  if (A.src === 'url' && lastPointer === 'mouse') $('#url').focus();
});
$('#lang-seg').addEventListener('click', (e) => {
  const b = e.target.closest('button'); if (!b) return;
  A.lang = b.dataset.v; segSet($('#lang-seg'), A.lang); $('#lang-seg').classList.remove('need'); updateAddForm();
});
$('#do-translate').addEventListener('change', () => {
  S.settings.addTranslate = $('#do-translate').checked;
  saveSettings();
  updateAddForm();
});
$('#url').addEventListener('input', updateAddForm);
$('#url').addEventListener('input', checkUrlPlugin);

// 本機外掛（app/plugins.py）處理的網址：自動選好外掛固定的語言、要不要翻譯。送出後後端也會照外掛的設定
let urlPluginSeq = 0;
let urlPluginTimer = null;
function checkUrlPlugin() {
  clearTimeout(urlPluginTimer);
  const raw = $('#url').value.trim();
  const seq = ++urlPluginSeq;
  if (A.plugin) {
    // 換了網址：外掛改過的翻譯勾選還原成平常的預設
    if (typeof A.plugin.options?.translate === 'boolean') $('#do-translate').checked = S.settings.addTranslate !== false;
    A.plugin = null;
    $('#url-plugin').hidden = true;
    resetEpList();
    updateAddForm();
  }
  if (A.mode !== 'add' || A.src !== 'url' || !/^https?:\/\/\S+/.test(raw)) return;
  urlPluginTimer = setTimeout(async () => {
    let res;
    try { res = await api('GET', `/api/url-plugin?url=${encodeURIComponent(raw)}`); } catch { return; }
    if (seq !== urlPluginSeq || !res?.plugin || !dlgAdd.open || $('#url').value.trim() !== raw) return;
    applyUrlPlugin(res);
  }, 250);
}
function applyUrlPlugin(res) {
  A.plugin = res;
  const o = res.options || {};
  const parts = [];
  if (o.language && S.meta.languages[o.language]) {
    A.lang = o.language;
    segSet($('#lang-seg'), A.lang);
    $('#lang-seg').classList.remove('need');
    parts.push(`語言${S.meta.languages[o.language]}`);
  }
  if (typeof o.translate === 'boolean') {
    $('#do-translate').checked = o.translate;
    parts.push(o.translate ? '翻譯' : '不翻譯');
  }
  $('#url-plugin').textContent = `這個網址由外掛「${res.plugin}」下載${parts.length ? `，固定${parts.join('、')}` : ''}。`;
  $('#url-plugin').hidden = false;
  $('#add-error').textContent = '';
  if (res.expand && !A.ep) loadEpList($('#url').value.trim());
  updateAddForm();
}

/* ----- 外掛的集數清單：貼一集的網址，列出整部作品讓使用者勾（後端 /api/url-plugin/list、/api/media/batch） ----- */

// A.ep：null（沒有清單）或 { url, loading, error, data }；data.groups[].items[].checked 是目前的勾選
let epSeq = 0;
function resetEpList() {
  epSeq++;
  A.ep = null;
  $('#ep-picker').hidden = true;
  $('#ep-groups').innerHTML = '';
  $('#url-hint').hidden = false;
}
async function loadEpList(url) {
  const seq = ++epSeq;
  A.ep = { url, loading: true, error: '', data: null };
  renderEpList();
  updateAddForm();
  try {
    const data = await api('GET', `/api/url-plugin/list?url=${encodeURIComponent(url)}`);
    if (seq !== epSeq) return;
    for (const g of data.groups) {
      for (const it of g.items) it.checked = g.checked && !it.in_library;
    }
    A.ep = { url, loading: false, error: '', data };
  } catch (err) {
    if (seq !== epSeq) return;
    A.ep = { url, loading: false, error: err.message, data: null };
  }
  renderEpList();
  updateAddForm();
}
const epChecked = () => (A.ep?.data ? A.ep.data.groups.flatMap((g) => g.items.filter((it) => it.checked)) : []);
function epCountText(n) { return n ? `已選 ${n} 集` : '還沒選'; }

function renderEpList() {
  const ep = A.ep;
  $('#ep-picker').hidden = !ep;
  $('#url-hint').hidden = !!ep;
  if (!ep) return;
  const data = ep.data;
  $('#ep-work').textContent = data ? (data.title || '這部作品') : '這部作品';
  $('#ep-legend').hidden = !data;
  $('#ep-status').textContent = ep.loading ? '正在讀取這部作品有哪些集數，第一次要幾秒…'
    : ep.error ? `讀不到集數清單（${ep.error}）。按「加入佇列」只會加入貼上的這一集。`
      : '勾要加入的集數，點組名可以整組勾或取消。';
  $('#ep-status').classList.toggle('error-text', !!ep.error);
  if (!data) { $('#ep-groups').innerHTML = ''; updateEpTotals(); return; }
  $('#ep-groups').innerHTML = data.groups.map((g) => {
    const lang = g.language && S.meta.languages[g.language] ? `，轉${S.meta.languages[g.language]}字幕` : '';
    return `<section class="ep-group${g.checked ? '' : ' off-default'}" data-g="${esc(g.key)}">
      <label class="ep-ghead"><input type="checkbox" data-gall="${esc(g.key)}">
        <span class="ep-gtitle">${esc(g.title)}</span><span class="ep-gcount" data-gcount="${esc(g.key)}"></span></label>
      ${g.note || !g.checked ? `<p class="ep-note">${esc(g.note || '預設不勾')}${esc(lang)}</p>` : ''}
      <div class="ep-grid">${g.items.map((it, i) => `<label class="ep${it.in_library ? ' have' : ''}${it.current ? ' cur' : ''}"
          title="${esc(it.title || it.label)}${it.in_library ? '（已在播放列表）' : ''}${it.current ? '（你貼的這集）' : ''}">
          <input type="checkbox" data-g="${esc(g.key)}" data-i="${i}"${it.checked ? ' checked' : ''}><span>${esc(it.label)}</span></label>`).join('')}</div>
    </section>`;
  }).join('');
  updateEpTotals();
}

function updateEpTotals() {
  const data = A.ep?.data;
  if (!data) { $('#ep-total').textContent = ''; return; }
  for (const g of data.groups) {
    const n = g.items.filter((it) => it.checked).length;
    const all = $(`[data-gall="${CSS.escape(g.key)}"]`, $('#ep-groups'));
    if (all) { all.checked = n > 0 && n === g.items.length; all.indeterminate = n > 0 && n < g.items.length; }
    const count = $(`[data-gcount="${CSS.escape(g.key)}"]`, $('#ep-groups'));
    if (count) count.textContent = `${n} / ${g.items.length} 集`;
  }
  $('#ep-total').textContent = epCountText(epChecked().length);
}

$('#ep-groups').addEventListener('change', (e) => {
  const data = A.ep?.data;
  const box = e.target;
  if (!data || box.type !== 'checkbox') return;
  if (box.dataset.gall) {
    const g = data.groups.find((x) => x.key === box.dataset.gall);
    if (!g) return;
    for (const it of g.items) it.checked = box.checked;
    for (const el of $$(`input[data-g="${CSS.escape(g.key)}"]`, $('#ep-groups'))) el.checked = box.checked;
  } else if (box.dataset.g) {
    const it = data.groups.find((x) => x.key === box.dataset.g)?.items[Number(box.dataset.i)];
    if (it) it.checked = box.checked;
  }
  updateEpTotals();
  updateAddForm();
});

// 同一部影片的網址有很多寫法（youtu.be、shorts、後面帶時間或播放清單），YouTube 用影片 ID 比對，其他網站比對網址本身
function urlKey(raw) {
  let u;
  try { u = new URL(String(raw || '').trim()); } catch { return ''; }
  const host = u.hostname.toLowerCase().replace(/^(www|m|music)\./, '');
  let id = '';
  if (host === 'youtu.be') id = u.pathname.split('/')[1] || '';
  else if (host === 'youtube.com' || host === 'youtube-nocookie.com') {
    id = u.searchParams.get('v') || (u.pathname.match(/^\/(?:shorts|live|embed|v)\/([^/]+)/) || [])[1] || '';
  }
  return id ? `yt:${id}` : `${host}${u.pathname.replace(/\/+$/, '')}${u.search}`;
}
function sameUrlMedia(raw) {
  const key = urlKey(raw);
  return key ? S.media.filter((m) => m.url && urlKey(m.url) === key) : [];
}
function updateUrlDup() {
  const el = $('#url-dup');
  // 有集數清單時，已經在播放列表的集數標在清單上
  const dup = A.mode === 'add' && A.src === 'url' && !A.ep?.data ? sameUrlMedia($('#url').value)[0] : null;
  el.hidden = !dup;
  if (!dup) { el.innerHTML = ''; return; }
  const title = titleParts(dup).main;
  el.innerHTML = `<span title="${esc(title)}">播放列表裡已經有這部影片：${esc(title)}</span>
    <button type="button" class="btn small" data-open-dup="${dup.id}">打開</button>`;
}
function openExisting(id) {
  dlgAdd.close();
  showView('player');
  openMedia(id);
}
$('#url-dup').addEventListener('click', (e) => {
  const b = e.target.closest('[data-open-dup]');
  if (b && mediaById(b.dataset.openDup)) openExisting(b.dataset.openDup);
});

// 模型選單：只列已下載或正在下載（pending）、支援這個語言的模型，預設值來自設定頁；
// 換語言才重設，使用者選過的保留（模型下載完、選項文字變了也保留）
function fillModelSelect(sel, entries, lang, def) {
  const options = Object.entries(entries).filter(([, v]) => v.langs.includes(lang) && (v.installed !== false || v.pending));
  const tags = (k, v) => [k === def ? '預設' : '', v.installed === false ? '下載中' : ''].filter(Boolean).join('，');
  const html = options.map(([k, v]) => `<option value="${k}">${esc(v.label)}${tags(k, v) ? `（${tags(k, v)}）` : ''}</option>`).join('');
  if (sel.dataset.lang === lang && sel.dataset.html === html) return;
  const keep = sel.dataset.lang === lang ? sel.value : '';
  sel.innerHTML = html;
  sel.dataset.html = html;
  sel.value = options.some(([k]) => k === keep) ? keep : options.some(([k]) => k === def) ? def : (options[0]?.[0] || '');
  sel.dataset.lang = lang;
}

// 選到還在下載的模型：影片會先排隊，裝好自動開始
function updateAddPending() {
  const lang = A.lang;
  const pend = (entries, sel) => !sel.closest('[hidden]') && entries[sel.value]?.installed === false && !!entries[sel.value]?.pending;
  $('#add-pending').hidden = !lang || !(pend(S.meta.engines, $('#adv-engine'))
    || (lang !== 'zh' && $('#do-translate').checked && pend(S.meta.translators, $('#adv-translator'))));
}
$('#adv-engine').addEventListener('change', updateAddPending);
$('#adv-translator').addEventListener('change', updateAddPending);

function updateAddForm() {
  const lang = A.lang;
  $('#model-field').hidden = !lang;
  $('#translate-field').hidden = !lang || lang === 'zh';
  if (lang) {
    const def = S.meta.defaults[lang] || {};
    fillModelSelect($('#adv-engine'), S.meta.engines, lang, def.engine);
    const showT = lang !== 'zh' && $('#do-translate').checked;
    $('#adv-translator-wrap').hidden = !showT;
    if (showT) fillModelSelect($('#adv-translator'), S.meta.translators, lang, def.translator);
  }

  let sourceOk = true;
  const list = A.mode === 'add' && A.src === 'url' && A.ep ? A.ep : null;
  if (A.mode === 'add') {
    if (A.src === 'upload') sourceOk = !!A.file;
    if (A.src === 'url') sourceOk = /^https?:\/\/\S+/.test($('#url').value.trim());
  }
  // 有集數清單：讀取中不能送，讀到了要至少勾一集（讀不到就照一般網址只加這一集）
  const picked = list?.data ? epChecked().length : 0;
  if (list && (list.loading || (list.data && !picked))) sourceOk = false;
  $('#add-submit').disabled = A.busy || !sourceOk;
  $('#add-submit').textContent = A.mode === 'transcribe' ? '開始轉字幕'
    : list?.loading ? '讀取集數中…' : list?.data ? `加入佇列（${picked} 集）` : '加入佇列';
  updateUrlDup();
  updateAddPending();
}

/* ----- 上傳 ----- */

function pickFile(file) {
  if (!file) return;
  A.file = file;
  $('#drop-text').textContent = `${file.name}（${fmtSize(file.size)}）`;
  updateAddForm();
}
$('#upload-file').addEventListener('change', (e) => pickFile(e.target.files[0]));
const drop = $('#drop');
drop.addEventListener('dragover', (e) => { e.preventDefault(); drop.classList.add('over'); });
drop.addEventListener('dragleave', () => drop.classList.remove('over'));
drop.addEventListener('drop', (e) => { e.preventDefault(); drop.classList.remove('over'); pickFile(e.dataTransfer.files[0]); });

function uploadFile(file) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('PUT', `/api/upload?name=${encodeURIComponent(file.name)}`);
    const bar = $('#upload-progress');
    bar.hidden = false;
    xhr.upload.onprogress = (e) => { if (e.lengthComputable) $('i', bar).style.width = `${(e.loaded / e.total) * 100}%`; };
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) resolve(JSON.parse(xhr.responseText).path);
      else { try { reject(new Error(JSON.parse(xhr.responseText).detail)); } catch { reject(new Error('上傳失敗')); } }
    };
    xhr.onerror = () => reject(new Error('上傳失敗'));
    xhr.send(file);
  });
}

/* ----- 送出 ----- */

$('#form-add').addEventListener('submit', async (e) => {
  e.preventDefault();
  if (!A.lang && A.mode === 'add' && A.src === 'url' && !A.plugin) {
    // 貼上網址馬上按送出，外掛的檢查（checkUrlPlugin）還沒回來：先問一次，外掛有固定語言就不用選
    try {
      const res = await api('GET', `/api/url-plugin?url=${encodeURIComponent($('#url').value.trim())}`);
      if (res?.plugin) applyUrlPlugin(res);
    } catch { /* 照一般網址處理 */ }
  }
  if (A.mode === 'add' && A.src === 'url' && A.ep?.loading) return;   // 集數清單讀好、勾完再送
  if (!A.lang) {
    $('#lang-seg').classList.add('need');
    $('#add-error').textContent = '請選擇影片語言';
    return;
  }
  const batch = A.mode === 'add' && A.src === 'url' && A.ep?.data ? epChecked().map((it) => it.url) : null;
  if (batch && !batch.length) return;
  const dup = A.mode === 'add' && A.src === 'url' && !batch ? sameUrlMedia($('#url').value) : [];
  if (dup.length) {
    const choice = await askConfirm({
      title: '播放列表裡已經有這部影片',
      text: '可以直接打開現有的，不用重新下載和轉字幕。還是要再加一次嗎？',
      items: dup.map((m) => `${titleParts(m).main}（${m.tracks.length ? `${m.tracks.length} 條字幕` : '還沒有字幕'}）`),
      ok: '還是再加一次', alt: '打開現有的',
    });
    if (choice === 'alt') { openExisting(dup[0].id); return; }
    if (!choice) return;
  }
  const opts = {
    language: A.lang,
    translate: A.lang !== 'zh' && $('#do-translate').checked,
    engine: $('#adv-engine').value || null,
    translator: A.lang !== 'zh' && $('#do-translate').checked ? ($('#adv-translator').value || null) : null,
    sensitive: $('#do-sensitive').checked,
    force: A.force,
  };
  A.busy = true; updateAddForm();
  $('#add-error').textContent = '';
  try {
    if (batch) {
      const res = await api('POST', '/api/media/batch', { ...opts, url: A.ep.url, items: batch });
      dlgAdd.close();
      toast(`已加入 ${res.count} 集，開始排隊下載和轉字幕`);
      refreshSoon();
      return;
    }
    if (A.mode === 'transcribe') {
      await api('POST', `/api/media/${S.current}/transcribe`, opts);
    } else {
      let body;
      if (A.src === 'url') body = { ...opts, source: 'url', url: $('#url').value.trim() };
      if (A.src === 'upload') {
        // 先確認需要的模型都下載了，不要等大檔案傳完才說缺模型
        await api('POST', '/api/media/precheck', opts);
        $('#add-submit').textContent = '上傳中…';
        const path = await uploadFile(A.file);
        body = { ...opts, source: 'upload', path };
      }
      const res = await api('POST', '/api/media', body);
      await refreshSoon();
      openMedia(res.id);
    }
    dlgAdd.close();
    toast(A.mode === 'transcribe' ? '已加入佇列' : '已加入播放列表，開始排隊轉字幕');
    refreshSoon();
  } catch (err) {
    $('#add-error').textContent = err.message;
  } finally {
    A.busy = false; updateAddForm();
    $('#upload-progress').hidden = true;
  }
});

for (const btn of $$('[data-close]')) btn.addEventListener('click', () => btn.closest('dialog').close());
for (const dlg of $$('dialog')) {
  dlg.addEventListener('click', (e) => { if (e.target === dlg) dlg.close(); });
}
$('#btn-add').addEventListener('click', () => openAddDialog('add'));

/* ================= 翻譯對話框 ================= */

// 翻譯任務翻的是哪一條原文。新增影片時排的翻譯要等轉字幕做完，從前一個任務的結果才知道（跟 jobs.translation_source 一樣）
function translateJobSource(j) {
  return (j.params || {}).source_track_id || S.jobs.find((x) => x.id === j.depends_on)?.result?.track_id || '';
}
const translateJobOf = (trackId) => S.jobs.find((j) => j.type === 'translate' && (j.status === 'running' || j.status === 'queued')
  && translateJobSource(j) === trackId);
const translationsOf = (m, trackId) => m.tracks.filter((t) => t.kind === 'translation' && t.source_track_id === trackId);
const translateBusyText = (j) => (j.status === 'running'
  ? `這條字幕正在翻譯（${Math.round((j.progress || 0) * 100)}%），完成後就會出現`
  : '這條字幕已經排在翻譯佇列裡，完成後就會出現');

function confirmRetranslate(m, src, already) {
  return askConfirm({
    title: '這條字幕已經翻譯過了',
    text: `「${trackLabel(src, m.tracks)}」已經有翻譯。再翻一次會重新佔用顯卡，並且多出一條新的翻譯（舊的不會被覆蓋）。確定要再翻一次嗎？`,
    items: already.map((t) => `現有：${trackLabel(t, m.tracks)}，${t.cue_count} 句`),
    ok: '還是再翻一次',
  });
}

function tellTranslating(m, src, job) {
  return askConfirm({
    title: job.status === 'running' ? '這條字幕正在翻譯' : '這條字幕已經在排隊翻譯',
    text: '完成後翻譯就會出現，不用再排一次。',
    items: [`原文：${trackLabel(src, m.tracks)}`, `翻譯模型：${jobModel(job)}`],
    ok: '知道了', cancel: '',
  });
}

function tellAllTranslating(m, sources) {
  return askConfirm({
    title: '這部影片的字幕都在翻譯',
    text: '完成後翻譯就會出現，不用再排一次。',
    items: sources.map((t) => {
      const job = translateJobOf(t.id);
      return `${trackLabel(t, m.tracks)}：${job.status === 'running' ? '翻譯中' : '等待翻譯'}（${jobModel(job)}）`;
    }),
    ok: '知道了', cancel: '',
  });
}

// 開對話框前已經確認過要重翻的原文，送出時不再問第二次；switchedFrom：預選的原文正在翻譯，改選了別條
const TR = { confirmed: '', switchedFrom: '' };

// 模型清單可能在設定頁下載或刪除過，用之前重新讀一次（讀不到就用手上的）
async function refreshMeta() {
  try { S.meta = await api('GET', '/api/meta'); } catch { /* 下次再讀 */ }
}

// 跟轉字幕一樣：預選的原文已經有翻譯就先確認。正在翻譯的原文在清單裡標示「翻譯中」而且不能選。
// 從播放器或字幕管理上方打開（沒指定原文）時，預選的那條正在翻譯就改選別條可以翻的，全部都在翻才只說明。
// 從字幕管理某一列點進來的（指定原文），那條正在翻譯就只說明
async function openTranslateDialog(sourceId) {
  const m0 = mediaById(S.current);
  if (!m0) return;
  await refreshMeta();
  const m = mediaById(S.current);
  if (!m || m.id !== m0.id) return;
  const sources = m.tracks.filter((t) => t.lang === 'ja' || t.lang === 'en');
  if (!sources.length) return;
  const named = sources.find((t) => t.id === sourceId);
  let preferred = named || sources.find((t) => t.id === S.trackSel[1])
    || sources.find((t) => t.kind === 'asr') || sources[0];
  let switchedFrom = '';
  const busy = translateJobOf(preferred.id);
  if (busy) {
    const free = named ? [] : sources.filter((t) => !translateJobOf(t.id));
    if (!free.length) {
      if (named || sources.length === 1) await tellTranslating(m, preferred, busy);
      else await tellAllTranslating(m, sources);
      return;
    }
    switchedFrom = preferred.id;
    // 先挑還沒翻過的；都翻過就挑第一條，送出前會照規則先確認
    preferred = free.find((t) => !translationsOf(m, t.id).length) || free[0];
  }
  const already = translationsOf(m, preferred.id);
  if (!switchedFrom && already.length && !(await confirmRetranslate(m, preferred, already))) return;
  if (S.current !== m.id) return;
  TR.confirmed = !switchedFrom && already.length ? preferred.id : '';
  TR.switchedFrom = switchedFrom;
  if ($('#dlg-tracks').open) $('#dlg-tracks').close();
  const sel = $('#tr-source');
  sel.innerHTML = sources.map((t) => `<option value="${t.id}" data-lang="${t.lang}"></option>`).join('');
  syncTranslateSources();
  sel.value = preferred.id;
  fillTranslatorSelect();
  $('#translate-error').textContent = '';
  updateTranslateNotice();
  $('#dlg-translate').showModal();
}

// 原文清單的名稱和能不能選跟著佇列更新：正在翻譯或排隊的標示出來，不能選。
// 只改有變的地方，不重畫整個選單（使用者正打開下拉選單時才不會被關掉）
function syncTranslateSources() {
  const m = mediaById(S.current);
  if (!m) return;
  for (const o of $('#tr-source').options) {
    const t = m.tracks.find((x) => x.id === o.value);
    const job = t && translateJobOf(t.id);
    const text = t ? trackLabel(t, m.tracks) + (job ? `（${job.status === 'running' ? '翻譯中' : '等待翻譯'}）` : '') : '（已刪除）';
    if (o.textContent !== text) o.textContent = text;
    const disabled = !t || !!job;
    if (o.disabled !== disabled) o.disabled = disabled;
  }
}

// 對話框裡的說明：預選的原文正在翻譯所以改選了這條、選到的這條已經有翻譯或正在翻譯。
// 正在翻譯或沒有可用的翻譯模型時不能送出
function updateTranslateNotice() {
  syncTranslateSources();
  const m = mediaById(S.current);
  const id = $('#tr-source').value;
  const busy = m && id ? translateJobOf(id) : null;
  const already = m && id ? translationsOf(m, id) : [];
  const models = [...new Set(already.map((t) => t.model))].join('、');
  const lines = [];
  const from = m && TR.switchedFrom && TR.switchedFrom !== id && translateJobOf(TR.switchedFrom)
    ? m.tracks.find((t) => t.id === TR.switchedFrom) : null;
  if (from) lines.push(`「${trackLabel(from, m.tracks)}」正在翻譯，完成後就會出現，所以先幫你選了另一條。`);
  if (busy) lines.push(translateBusyText(busy));
  else if (already.length) lines.push(`這條字幕已經有 ${models} 的翻譯，再翻會多一條新的`);
  const el = $('#tr-notice');
  el.textContent = lines.join('\n');
  el.hidden = !lines.length;
  $('#tr-submit').disabled = !!busy || !$('#tr-model').value;
}

// 翻譯模型只列已經下載的；還沒下載的寫在下面，按「去下載」到設定頁的模型管理
function fillTranslatorSelect() {
  const lang = $('#tr-source').selectedOptions[0]?.dataset.lang;
  const def = S.meta.defaults[lang]?.translator;
  const all = Object.entries(S.meta.translators).filter(([, v]) => v.langs.includes(lang));
  // 還在下載的（pending）也能選：翻譯任務會排隊等它裝好
  const ready = all.filter(([, v]) => v.installed !== false || v.pending);
  const missing = all.filter(([, v]) => v.installed === false && !v.pending);
  const sel = $('#tr-model');
  sel.innerHTML = ready.length
    ? ready.map(([k, v]) => {
      const tags = [k === def ? '建議' : '', v.installed === false ? '下載中' : ''].filter(Boolean).join('，');
      return `<option value="${k}">${esc(v.label)}${tags ? `（${tags}）` : ''}</option>`;
    }).join('')
    : '<option value="">還沒有可以用的翻譯模型</option>';
  sel.value = ready.some(([k]) => k === def) ? def : (ready[0]?.[0] || '');
  sel.disabled = !ready.length;
  $('#tr-missing').hidden = !missing.length;
  $('#tr-missing-text').textContent = ready.length
    ? `還沒下載：${missing.map(([, v]) => v.label).join('、')}`
    : `還沒下載可以翻${S.meta.languages[lang] || ''}的翻譯模型，下載好就能翻譯。`;
  $('#tr-submit').disabled = !ready.length || !!translateJobOf($('#tr-source').value);
}

// 到設定頁的模型管理，捲到那一組（role：translator、asr…）
function openModelManager(role) {
  SET.focusRole = role || 'translator';
  showView('settings');
}
$('#tr-go-models').addEventListener('click', () => {
  $('#dlg-translate').close();
  openModelManager('translator');
});

$('#tr-source').addEventListener('change', () => { fillTranslatorSelect(); updateTranslateNotice(); });
$('#form-translate').addEventListener('submit', async (e) => {
  e.preventDefault();
  const m = mediaById(S.current);
  const sourceId = $('#tr-source').value;
  if (!m || !sourceId || !$('#tr-model').value) return;
  // 對話框開著的時候狀態可能變了（別的地方排了翻譯、翻譯剛做好），送出前用最新的資料再看一次
  if (translateJobOf(sourceId)) { updateTranslateNotice(); return; }
  const already = translationsOf(m, sourceId);
  if (already.length && sourceId !== TR.confirmed) {
    const src = m.tracks.find((t) => t.id === sourceId);
    if (!src || !(await confirmRetranslate(m, src, already))) return;
    TR.confirmed = sourceId;
  }
  try {
    await api('POST', `/api/tracks/${sourceId}/translate`, { translator: $('#tr-model').value, force: already.length > 0 });
    $('#dlg-translate').close();
    toast('已加入佇列');
    refreshSoon();
  } catch (err) { $('#translate-error').textContent = err.message; }
});

$('#btn-help').addEventListener('click', () => $('#dlg-help').showModal());

/* ================= 快捷鍵 ================= */

document.addEventListener('keydown', (e) => {
  if (e.ctrlKey || e.metaKey || e.altKey) return;
  if (e.target.closest('input, select, textarea') || document.querySelector('dialog[open]')) return;
  const key = e.key.length === 1 ? e.key.toLowerCase() : e.key;
  if (window.WL?.onKey(e, key)) return; // 查字卡開著時的 Esc、X、I、Shift
  if (key === '?') { $('#dlg-help').showModal(); e.preventDefault(); return; }
  if (key === 'Escape') {
    closeMenu();
    if (document.body.classList.contains('pseudo-fs')) setPseudoFullscreen(false);
    return;
  }
  // 單字頁、設定頁蓋在播放器上：播放器的快捷鍵不作用，免得在看不到的地方播放或改掉字幕設定
  if (SET.open || !$('#vocab-page').hidden) return;
  if (key === '[' || key === ']') { e.preventDefault(); togglePanel(key === '[' ? 'left' : 'right'); return; }
  if (!S.current) return;
  const actions = {
    ' ': togglePlay, k: togglePlay,
    ArrowLeft: () => seekBy(-5), ArrowRight: () => seekBy(5),
    ArrowUp: () => { S.settings.volume = Math.min(1, S.settings.volume + 0.05); S.settings.muted = false; saveSettings(); applyVolume(); },
    ArrowDown: () => { S.settings.volume = Math.max(0, S.settings.volume - 0.05); saveSettings(); applyVolume(); },
    a: prevCue, d: nextCue, r: replayCue, l: toggleLoop,
    f: toggleFullscreen,
    m: () => $('#c-mute').click(),
    ',': () => stepRate(-1), '.': () => stepRate(1),
    b: () => { S.settings.blur2 = !S.settings.blur2; saveSettings(); applySubSettings(); toast(S.settings.blur2 ? '翻譯已遮住，滑鼠移上去才顯示' : '翻譯正常顯示'); },
    y: () => {
      const order = ['all', 'hover', 'off'];
      const next = order[(order.indexOf(S.settings.furi || 'all') + 1) % order.length];
      setFuri(next);
      toast(`假名：${FURI_MODES[next]}`);
    },
    1: () => { S.settings.hide1 = !S.settings.hide1; saveSettings(); applySubSettings(); toast(S.settings.hide1 ? '已隱藏原文' : '已顯示原文'); },
    2: () => { S.settings.hide2 = !S.settings.hide2; saveSettings(); applySubSettings(); toast(S.settings.hide2 ? '已隱藏翻譯' : '已顯示翻譯'); },
  };
  const fn = actions[key];
  if (fn) { e.preventDefault(); fn(); pokeControls(); }
});

/* ================= 設定頁 ================= */

const SET = { data: null, open: false, focusRole: '' };
const LANG_CHIP = { zh: '中', ja: '日', en: '英' };
// 模型依用途分組，每組在流程裡的哪一步用到寫在說明裡。辨識、對齊顯存放得下就同時載入，翻譯模型放不下時先釋放它們
const MODEL_GROUPS = [
  { roles: ['asr'], title: '語音辨識', hint: '把聲音轉成文字，轉字幕時第一步載入。新增影片時可以選要用哪一個。' },
  { roles: ['aligner'], title: '時間軸對齊', hint: '辨識完用它算出每個字在影片裡的時間，顯存夠就跟辨識模型一起留在顯卡上。所有辨識模型共用這一個。' },
  { roles: ['translator'], title: '翻譯', hint: '字幕翻成繁體中文、翻譯標題時載入。' },
  { roles: ['separator'], title: '人聲分離', hint: '歌曲或配樂很滿的影片，辨識前先用它把人聲抽出來。' },
  { roles: ['furigana', 'segmenter'], title: '輔助（CPU）', hint: '在 CPU 上跑的小模型，不佔顯卡：判斷日文漢字讀音、中文字幕斷詞。' },
];

// 檔案大小：已安裝寫磁碟上實際的大小，沒安裝寫要下載的大小。
// 顯存佔用：載入前實際檢查的門檻（後端算好的字串）。辨識、對齊是最低需求，顯存越大一次處理越多段、越快
function modelSpecs(m) {
  const disk = `檔案大小 ${m.installed ? m.disk : m.size}`;
  let vram = '不用顯卡';
  if (m.vram) vram = m.vram_min ? `顯存佔用 至少約 ${m.vram}，顯存越大越快` : `顯存佔用 約 ${m.vram}`;
  return `${disk} · ${vram}`;
}

// 授權：接在規格後面的小字，非商用、授權不明的換醒目的顏色（後端 models.license_limit 判斷），點了開官方頁面
function modelLicenseHtml(m) {
  if (!m.license) return '';
  const limit = m.license_limit;
  const cls = limit ? `lic lic-${limit}` : 'lic';
  const tip = limit === 'noncommercial' ? '只能非商業使用，條件看官方頁面'
    : limit === 'unknown' ? '作者沒有公開授權，使用條件不確定' : '模型的授權，條件看官方頁面';
  const text = m.license.startsWith('授權') ? m.license : `授權 ${m.license}`;   // 「授權不明」不要變成「授權 授權不明」
  return /^https:\/\//.test(m.license_url || '')
    ? `<a class="${cls}" href="${esc(m.license_url)}" target="_blank" rel="noreferrer" title="${tip}">${esc(text)}</a>`
    : `<span class="${cls}" title="${tip}">${esc(text)}</span>`;
}

// 下載任務的按鈕：下載中、排隊中可以暫停或取消；暫停的可以繼續（從中斷的地方接著下載）
// withStage=false：狀態文字另外放（版本列把它放在左邊規格底下，長的重試倒數才不會把按鈕那一欄撐開）
function modelJobActs(job, withStage = true) {
  const pct = Math.round((job.progress || 0) * 100);
  if (job.status === 'paused') {
    return `<span class="size">已暫停 ${pct}%</span>
      <button class="btn small primary" data-job-resume="${job.id}">繼續</button>
      <button class="btn small" data-model-cancel="${job.id}">取消</button>`;
  }
  const stage = withStage ? `<span class="size">${esc(job.stage || '')}</span>` : '';
  return `<div class="minibar"><i style="width:${pct}%"></i></div>${stage}
    <button class="btn small" data-job-pause="${job.id}">暫停</button>
    <button class="btn small" data-model-cancel="${job.id}">取消</button>`;
}

function noFitBadge(vram) {
  const usable = SET.data && SET.data.gpu_usable_mb;
  return `<span class="badge-warn" title="載入前要有 ${esc(vram)} 可用顯存，這張顯卡最多只有 ${(usable / 1024).toFixed(1)} GB">這張顯卡放不下</span>`;
}

// 有版本（量化）的翻譯模型：每個版本一行，各自下載、刪除；裝了好幾個版本時可以改用別的
function variantRowHtml(m, v) {
  const attrs = `="${m.id}" data-variant="${esc(v.name)}"`;
  let acts;
  if (v.job) acts = modelJobActs(v.job, false);
  else if (v.installed) {
    acts = (v.active ? '<span class="state-on">使用中</span>'
      : `<span class="state-on">已安裝</span><button class="btn small" data-variant-use${attrs}>改用</button>`)
      + `<button class="btn small" data-model-del${attrs}>刪除</button>`;
  } else if (v.leftover) {
    acts = `<button class="btn small" data-model-clear${attrs}>清掉</button><button class="btn small primary" data-model-dl${attrs}>下載</button>`;
  } else {
    acts = `<button class="btn small primary" data-model-dl${attrs}>下載</button>`;
  }
  const size = `檔案大小 ${v.installed ? v.disk : v.size}`;
  const vram = v.vram ? ` · 顯存佔用 約 ${v.vram}${v.vram_estimate ? '（估計）' : ''}` : '';
  const leftover = v.leftover && !v.job ? `<span class="leftover">沒下載完，佔用 ${esc(v.disk)}</span>` : '';
  const stage = v.job && v.job.status !== 'paused' && v.job.stage ? `<span class="vstage">${esc(v.job.stage)}</span>` : '';
  return `<div class="variant-row${v.installed ? ' on' : ''}">
    <div class="vinfo"><span class="vname" title="${esc(`${v.repo}/${v.file}`)}">${esc(v.name)}</span>
      <span class="specs">${esc(size + vram)}${v.fits === false ? noFitBadge(v.vram) : ''}${leftover}</span>${stage}</div>
    <div class="acts">${acts}</div>
  </div>`;
}

function modelRowHtml(m) {
  const variants = m.variants || [];
  let acts = '';
  if (variants.length) {
    // 版本各自有按鈕；只有不屬於任何版本的殘檔時，才在模型這一列顯示「清掉」
    if (m.leftover && !variants.some((v) => v.leftover || v.job)) acts = `<button class="btn small" data-model-clear="${m.id}">清掉</button>`;
  } else if (m.job) {
    acts = modelJobActs(m.job);
  } else if (m.installed) {
    acts = `<span class="state-on">已安裝</span><button class="btn small" data-model-del="${m.id}">刪除</button>`;
  } else if (m.leftover) {
    acts = `<button class="btn small" data-model-clear="${m.id}">清掉</button>
      <button class="btn small primary" data-model-dl="${m.id}">下載</button>`;
  } else {
    acts = `<button class="btn small primary" data-model-dl="${m.id}">下載</button>`;
  }
  const defaults = (m.used_by || []).length
    ? `<span class="badge-default" title="目前是這些情況的預設模型">預設：${esc(m.used_by.join('、'))}</span>`
    : '';
  const noFit = m.fits === false && !variants.length ? noFitBadge(m.vram) : '';
  const specs = variants.length ? '' : esc(modelSpecs(m));
  const showLeftover = m.leftover && !m.job && !variants.some((v) => v.leftover || v.job);
  const leftover = showLeftover ? `<div class="leftover">沒有下載完，留下的檔案佔用 ${esc(m.disk)}</div>` : '';
  return `<div class="model-row${m.installed ? ' on' : ''}${defaults ? ' is-default' : ''}">
    <div>
      <div class="title"><b>${esc(m.label)}</b>
        <span class="size">${m.langs.map((l) => LANG_CHIP[l] || l).join('')}</span>${defaults}</div>
      <div class="note">${esc(m.note)}</div>
      <div class="specs">${specs}${noFit}${modelLicenseHtml(m)}</div>
      ${leftover}
      <div class="size">${esc(m.installed || m.leftover ? m.path : m.repo)}</div>
    </div>
    <div class="acts">${acts}</div>
    ${variants.length ? `<div class="variants">${variants.map((v) => variantRowHtml(m, v)).join('')}</div>` : ''}
  </div>`;
}

async function refreshSettings() {
  try {
    SET.data = await api('GET', '/api/settings');
  } catch (err) { toast(err.message, true); return; }
  renderSettings();
}

function renderSettings() {
  const d = SET.data;
  if (!d) return;
  $('#combo-table').innerHTML = '<div class="combo-row combo-head"><span></span><span>辨識模型</span><span>翻譯模型</span></div>'
    + d.combos.map((c) => `<div class="combo-row">
        <span class="name">${esc(c.label)}</span>
        <select data-combo="${c.key}" data-kind="engine">${c.engines.map((e) => `<option value="${e.key}"${e.key === c.engine ? ' selected' : ''}>${esc(e.label)}</option>`).join('')}</select>
        ${c.translators.length
          ? `<select data-combo="${c.key}" data-kind="translator">${c.translators.map((t) => `<option value="${t.key}"${t.key === c.translator ? ' selected' : ''}>${esc(t.label)}</option>`).join('')}</select>`
          : '<span class="name">不需要翻譯</span>'}
      </div>`).join('');

  $('#model-list').innerHTML = MODEL_GROUPS.map((g) => {
    const rows = d.catalog.filter((m) => g.roles.includes(m.role));
    if (!rows.length) return '';
    return `<div class="model-group" data-roles="${g.roles.join(' ')}"><h3>${g.title}</h3><p class="hint">${g.hint}</p>${rows.map(modelRowHtml).join('')}</div>`;
  }).join('');
  if (SET.focusRole) {
    // 從翻譯對話框、佇列按「去下載」過來的：捲到那一組模型
    const group = $(`#model-list [data-roles~="${SET.focusRole}"]`) || $('#model-list');
    SET.focusRole = '';
    group.scrollIntoView({ block: 'start' });
  }

  const st = d.storage;
  $('#models-dir').textContent = st ? st.dir : '';
  $('#models-storage').textContent = st ? `模型共佔用 ${st.used}，${st.drive} 還剩 ${st.free}。` : '';
  $('#set-cookies-browser').value = d.values.cookies_browser || '';
  $('#set-cookies-file').value = d.values.cookies_file || '';
  $('#set-group').checked = d.values.group_by_model !== false;
  $('#set-title-mode').value = S.settings.titleMode || 'both';
  $('#set-translation-mode').value = d.values.translation_mode || 'line';
  $('#set-health-check').checked = d.values.health_check !== false;
  $('#set-autoplay').checked = !!S.settings.autoplay;
  $('#set-autonext').checked = !!S.settings.autoNext;
  $('#set-resume').checked = S.settings.resume !== false;
  $('#set-repeat').value = S.settings.repeat || 'off';
  renderLan(d);
  const g = S.gpu && S.gpu.info;
  $('#gpu-info').innerHTML = g
    ? `<span>${esc(g.name)}</span><span>已用 <b>${(g.used_mb / 1024).toFixed(1)} GB</b></span>
       <span>可用 <b>${(g.free_mb / 1024).toFixed(1)} GB</b></span><span>總共 ${Math.round(g.total_mb / 1024)} GB</span>
       ${g.driver ? `<span>驅動 ${esc(g.driver)}（需要 ${DRIVER_MIN} 以上）</span><span>${esc(cudaText(g))}</span>` : ''}`
    : '讀不到顯卡資訊';
  renderGpuLoaded();
  renderSetupCard();
}

$('#btn-gpu-release').addEventListener('click', (e) => releaseGpu(e.currentTarget));

// 頂列分頁：播放器 / 單字 / 設定
function showView(name) {
  for (const b of $$('#topnav button')) b.classList.toggle('on', b.dataset.view === name);
  $('#settings-page').hidden = name !== 'settings';
  $('#vocab-page').hidden = name !== 'vocab';
  applyImmersive();
  SET.open = name === 'settings';
  if (SET.open) {
    if (!video.paused) video.pause();
    refreshSettings();
    refreshPluginSettings();
    refreshRedoHint();
    refreshHealthHint();
  }
  window.WL?.onView(name);
}
const openSettings = () => showView('settings');
const closeSettings = () => showView('player');
$('#topnav').addEventListener('click', (e) => {
  const b = e.target.closest('button[data-view]');
  if (b) showView(b.dataset.view);
});
$('#brand').addEventListener('click', () => showView('player'));
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && SET.open && !document.querySelector('dialog[open]')) showView('player');
});

$('#combo-table').addEventListener('change', async () => {
  const engine = {}, translator = {};
  for (const sel of $$('#combo-table select')) {
    (sel.dataset.kind === 'engine' ? engine : translator)[sel.dataset.combo] = sel.value;
  }
  try {
    await api('PUT', '/api/settings', { default_engine: engine, default_translator: translator });
    S.meta = await api('GET', '/api/meta');
    await refreshSettings();
    toast('預設模型已更新');
  } catch (err) { toast(err.message, true); }
});

// 刪除模型、清掉殘檔：影響說明由後端照目前的預設模型和佇列算好（GET /api/models/{id}/impact）
// variant：只刪這個版本
async function confirmModelDelete(id, variant) {
  const m = SET.data.catalog.find((x) => x.id === id);
  if (!m) return false;
  const q = variant ? `?variant=${encodeURIComponent(variant)}` : '';
  const info = await api('GET', `/api/models/${id}/impact${q}`);
  if (info.busy) { toast(info.busy, true); return false; }
  const label = info.label || m.label;
  const files = (info.paths || []).map((p) => `會刪除 ${p}`);
  if (!info.installed) {
    return askConfirm({
      title: '清掉沒下載完的檔案',
      text: `「${label}」沒有下載完，留下的檔案佔用 ${info.disk}。清掉後要用時再重新下載。`,
      items: files, ok: '清掉', warn: true,
    });
  }
  return askConfirm({
    title: '刪除模型',
    text: `要刪除「${label}」嗎？會空出 ${info.disk}，之後需要可以再下載。`,
    items: [...info.lines, ...files], ok: '刪除', warn: true,
  });
}

$('#model-list').addEventListener('click', async (e) => {
  const dl = e.target.closest('[data-model-dl]');
  const del = e.target.closest('[data-model-del]');
  const clear = e.target.closest('[data-model-clear]');
  const cancel = e.target.closest('[data-model-cancel]');
  const pause = e.target.closest('[data-job-pause]');
  const resume = e.target.closest('[data-job-resume]');
  const use = e.target.closest('[data-variant-use]');
  const variant = (dl || del || clear || use)?.dataset.variant || '';
  const q = variant ? `?variant=${encodeURIComponent(variant)}` : '';
  let deleting = false;
  try {
    if (dl) {
      const res = await api('POST', `/api/models/${dl.dataset.modelDl}/download${q}`);
      toast(res.resumed ? '接著下載' : '已加入下載佇列');
    } else if (cancel) {
      await api('POST', `/api/jobs/${cancel.dataset.modelCancel}/cancel`);
    } else if (pause) {
      await api('POST', `/api/jobs/${pause.dataset.jobPause}/pause`);
      toast('已暫停，下載的部分會保留');
    } else if (resume) {
      await api('POST', `/api/jobs/${resume.dataset.jobResume}/resume`);
      toast('接著下載');
    } else if (use) {
      const chosen = { ...(SET.data.values.model_variants || {}), [use.dataset.variantUse]: variant };
      await api('PUT', '/api/settings', { model_variants: chosen });
      S.meta = await api('GET', '/api/meta');
      toast(`已改用 ${variant}`);
    } else if (del || clear) {
      const id = del ? del.dataset.modelDel : clear.dataset.modelClear;
      if (!(await confirmModelDelete(id, variant))) return;
      deleting = true;
      const res = await api('DELETE', `/api/models/${id}${q}`);
      toast(`${clear ? '已清掉' : '已刪除'}，空出 ${res.freed}`);
    } else return;
    await refreshSettings();
    refreshSoon();
  } catch (err) {
    toast(err.message, true);
    // 刪到一半失敗時，畫面要顯示實際剩下的狀態（可能已經變成沒安裝、留著殘檔）
    if (deleting) { await refreshSettings(); refreshSoon(); }
  }
});

async function saveValue(key, value) {
  try { await api('PUT', '/api/settings', { [key]: value }); } catch (err) { toast(err.message, true); }
}
$('#set-cookies-browser').addEventListener('change', (e) => saveValue('cookies_browser', e.target.value));
$('#set-cookies-file').addEventListener('change', (e) => saveValue('cookies_file', e.target.value.trim()));
$('#set-group').addEventListener('change', (e) => saveValue('group_by_model', e.target.checked));

/* ----- 外掛設定（後端 /api/plugins/...）：只寫不讀，畫面上只有有沒有設定、什麼時候、哪個瀏覽器 ----- */

function fmtDateTime(sec) {
  const d = new Date(sec * 1000);
  const p = (n) => String(n).padStart(2, '0');
  return `${d.getFullYear()}/${d.getMonth() + 1}/${d.getDate()} ${p(d.getHours())}:${p(d.getMinutes())}`;
}

// UA 的短雜湊，跟後端 plugins.ua_fingerprint 一樣（FNV-1a 32 位元，UTF-8）
function uaFingerprint(ua) {
  let h = 0x811c9dc5;
  for (const b of new TextEncoder().encode(ua)) h = Math.imul(h ^ b, 0x01000193) >>> 0;
  return h.toString(16).padStart(8, '0');
}

// 外掛要瀏覽器 UA 的（cookie 跟 UA 綁在一起）：網頁打開時，存的 UA 跟這個瀏覽器的不一樣才送，不顯示任何東西。
// 後端只接受同一種瀏覽器、同一種系統的新版本（在手機上打開不會蓋掉電腦的）
async function syncPluginUserAgent() {
  let res;
  try { res = await api('GET', '/api/plugins/settings'); } catch { return; }
  const fp = uaFingerprint(navigator.userAgent);
  for (const p of res.plugins || []) {
    if (!p.auto_user_agent || p.ua_fp === fp) continue;
    try { await api('PUT', `/api/plugins/${encodeURIComponent(p.id)}/user-agent`, { user_agent: navigator.userAgent }); } catch { /* 下次打開再試 */ }
  }
}

async function refreshPluginSettings() {
  let res;
  try { res = await api('GET', '/api/plugins/settings'); } catch { return; }
  renderPluginSettings(res.plugins || []);
}

function pluginBlockHtml(p) {
  const anySet = p.fields.some((f) => f.set);
  const when = p.fields.map((f) => f.saved_at).filter(Boolean).sort().pop();
  const status = anySet
    ? `已設定${when ? `，${fmtDateTime(when)}` : ''}${p.browser ? `，${esc(p.browser)}` : ''}`
    : '還沒設定';
  const fields = p.fields.map((f) => {
    const id = `pf-${p.id}-${f.key}`;
    const input = f.type === 'textarea'
      ? `<textarea id="${id}" data-field="${esc(f.key)}" rows="3" autocomplete="off" spellcheck="false" placeholder="${esc(f.set ? '要換新的再貼一次' : f.placeholder)}"></textarea>`
      : `<input type="text" id="${id}" data-field="${esc(f.key)}" autocomplete="off" spellcheck="false" placeholder="${esc(f.set ? '要換新的再貼一次' : f.placeholder)}">`;
    return `<div class="plugin-field">
      ${f.steps.length ? `<div class="plugin-steps-title">怎麼取得 ${esc(f.label)}</div><ol class="plugin-steps">${f.steps.map((s) => `<li>${esc(s)}</li>`).join('')}</ol>` : ''}
      <label for="${id}" class="plugin-label">${esc(f.label)}</label>
      ${input}
      ${f.help ? `<p class="hint">${esc(f.help)}</p>` : ''}
    </div>`;
  }).join('');
  return `<div class="plugin-block" data-plugin="${esc(p.id)}">
    <h3>${esc(p.title)}</h3>
    ${p.intro ? `<p class="hint">${esc(p.intro)}</p>` : ''}
    <p class="plugin-status${anySet ? ' ok' : ''}"><i></i>${status}</p>
    ${p.problem ? `<p class="plugin-problem">${esc(p.problem)}</p>` : ''}
    ${fields}
    <div class="row-actions">
      <button type="button" class="btn primary" data-plugin-save>儲存</button>
      ${anySet ? '<button type="button" class="btn" data-plugin-clear>清除</button>' : ''}
    </div>
  </div>`;
}

function renderPluginSettings(list) {
  $('#plugins-card').hidden = !list.length;
  // 正在輸入的內容不要被重畫洗掉
  const typing = $$('#plugin-list [data-field]').some((el) => el.value);
  if (typing && $('#plugin-list').children.length === list.length) return;
  $('#plugin-list').innerHTML = list.map(pluginBlockHtml).join('');
}

$('#plugin-list').addEventListener('click', async (e) => {
  const block = e.target.closest('[data-plugin]');
  if (!block) return;
  const pid = block.dataset.plugin;
  const save = e.target.closest('[data-plugin-save]');
  const clear = e.target.closest('[data-plugin-clear]');
  if (!save && !clear) return;
  const btn = save || clear;
  try {
    let res;
    if (save) {
      const values = {};
      for (const el of $$('[data-field]', block)) if (el.value.trim()) values[el.dataset.field] = el.value;
      if (!Object.keys(values).length) { toast('請先貼上內容', true); return; }
      btn.disabled = true;
      res = await api('PUT', `/api/plugins/${encodeURIComponent(pid)}/settings`, { values, user_agent: navigator.userAgent });
      for (const el of $$('[data-field]', block)) el.value = '';
      toast('已儲存');
    } else {
      const ok = await askConfirm({ title: '清除設定', text: '要清除存著的內容嗎？之後要用時再貼一次。', ok: '清除', warn: true });
      if (!ok) return;
      btn.disabled = true;
      res = await api('DELETE', `/api/plugins/${encodeURIComponent(pid)}/settings`);
      toast('已清除');
    }
    block.outerHTML = pluginBlockHtml(res.plugin);
  } catch (err) {
    toast(err.message, true);
    btn.disabled = false;
  }
});

/* ----- 手機、平板（區網開關，後端見 app/server.py 的 reject_lan_when_off、_network_view） ----- */

function renderLan(d) {
  const n = d.network || {};
  $('#set-lan-access').checked = !!d.values.lan_access;
  const port = n.port || location.port || '80';
  const urls = (n.addresses || []).map((ip) => `http://${ip}:${port}/`).join('、');
  let text = '';
  if (n.lan_access && n.host_env_local) {
    text = `環境變數 VS_HOST=${n.host_env} 讓程式只開本機，這個開關沒有作用。要讓手機、平板連線，請拿掉 VS_HOST 後重新啟動程式`;
  } else if (n.lan_access && n.restart_needed) {
    text = `已經打開，重新啟動程式後才會開始接受手機、平板連線。${urls ? `重新啟動後可以用 ${urls}` : ''}`;
  } else if (n.lan_access) {
    text = urls ? `手機、平板現在可以用 ${urls} 連線` : '已經打開，但找不到這台電腦在區網裡的位址';
  } else if (n.listening_lan) {
    text = '已經關掉，手機、平板連進來會被拒絕';
  } else if (n.host_env) {
    text = `程式照環境變數 VS_HOST=${n.host_env} 開啟`;
  }
  $('#lan-status').textContent = text;
}
$('#set-lan-access').addEventListener('change', async (e) => {
  const on = e.target.checked;
  // 有沒有在聽區網、VS_HOST 是程式啟動時決定的，切換開關不會變：伺服器已經在聽區網時打開馬上生效
  const n = (SET.data && SET.data.network) || {};
  let msg = '已關掉，手機、平板現在起不能連線';
  if (on && n.host_env_local) msg = '已打開，但 VS_HOST 讓程式只開本機，手機、平板還是連不進來';
  else if (on && n.listening_lan) msg = '已打開，手機、平板現在就能連線';
  else if (on) msg = '已打開，重新啟動程式後手機、平板就能連線';
  try {
    await api('PUT', '/api/settings', { lan_access: on });
    toast(msg);
  } catch (err) {
    e.target.checked = !on;
    toast(err.message, true);
  }
  refreshSettings();
});

$('#set-translation-mode').addEventListener('change', (e) => {
  saveValue('translation_mode', e.target.value);
  toast(e.target.value === 'line' ? '之後的翻譯會逐行對照' : '之後的翻譯會整句通順');
});

/* ----- 字幕時間軸 ----- */

$('#set-health-check').addEventListener('change', (e) => {
  saveValue('health_check', e.target.checked);
  toast(e.target.checked ? '之後轉完字幕會自動檢查時間軸' : '之後轉完字幕不檢查時間軸');
});
let healthHintJobs = -1;
async function refreshHealthHint() {
  healthHintJobs = S.jobs.filter((j) => j.type === 'health' && (j.status === 'queued' || j.status === 'running')).length;
  try {
    const info = await api('GET', '/api/health/pending');
    const queued = info.queued ? `，另有 ${info.queued} 條在佇列裡` : '';
    $('#health-hint').textContent = info.count
      ? `有 ${info.count} 條字幕還沒檢查（共 ${info.cues} 行），預估${fmtRemain(info.estimate_s)}${queued}`
      : (info.queued ? `${info.queued} 條字幕在佇列裡等待檢查` : '所有字幕都檢查過了');
    $('#btn-health-all').disabled = !info.count;
    return info;
  } catch {
    $('#health-hint').textContent = '';
    return null;
  }
}
$('#btn-health-all').addEventListener('click', async () => {
  const info = await refreshHealthHint();
  if (!info || !info.count) return;
  const ok = await askConfirm({
    title: '檢查所有尚未檢查的字幕',
    text: `會把 ${info.count} 條字幕（共 ${info.cues} 行）排進佇列，佔用顯卡預估${fmtRemain(info.estimate_s)}（實際看影片長度和顯卡）。只有更對得上的行才會調整時間，逐行翻譯會跟著調整，調整前的字幕檔會留一份備份。`,
    ok: '開始檢查',
  });
  if (!ok) return;
  try {
    const res = await api('POST', '/api/health/all');
    toast(`已加入佇列，共 ${res.count} 條字幕`);
    await refreshSoon();
    refreshHealthHint();
  } catch (err) { toast(err.message, true); }
});
$('#set-title-mode').addEventListener('change', (e) => {
  S.settings.titleMode = e.target.value;
  saveSettings();
  renderLibrary();
  renderQueue();
  const m = mediaById(S.current);
  if (m) renderDeck(m);
});
async function refreshRedoHint() {
  try {
    const info = await api('GET', '/api/translations/outdated');
    $('#redo-hint').textContent = info.count
      ? `有 ${info.count} 條舊翻譯（共 ${info.cues} 句），重翻約 ${fmtRemain(info.estimate_s)}`
      : '沒有需要重翻的舊翻譯';
    $('#btn-redo-translations').disabled = !info.count || info.pending;
    if (info.pending) $('#redo-hint').textContent = '重翻任務已經在佇列裡';
    return info;
  } catch { return null; }
}
$('#btn-redo-translations').addEventListener('click', async () => {
  const info = await refreshRedoHint();
  if (!info || !info.count) return;
  const ok = await askConfirm({
    title: '重新翻譯舊的字幕',
    text: `會把 ${info.count} 條整句模式的翻譯改用逐行對照重翻，佔用顯卡${fmtRemain(info.estimate_s)}。新的翻譯完成後才會刪掉舊的，中間不會沒有字幕。`,
    ok: '開始重翻',
  });
  if (!ok) return;
  try {
    const res = await api('POST', '/api/translations/redo');
    toast(`已加入佇列，共 ${res.count} 條`);
    refreshSoon();
    refreshRedoHint();
  } catch (err) { toast(err.message, true); }
});

$('#btn-translate-titles').addEventListener('click', async () => {
  try {
    const res = await api('POST', '/api/titles/translate');
    toast(res.count ? `已加入佇列，共 ${res.count} 個標題` : '所有標題都翻譯過了');
    refreshSoon();
  } catch (err) { toast(err.message, true); }
});
$('#set-autoplay').addEventListener('change', (e) => { S.settings.autoplay = e.target.checked; saveSettings(); });
$('#set-autonext').addEventListener('change', (e) => { S.settings.autoNext = e.target.checked; saveSettings(); });
$('#set-resume').addEventListener('change', (e) => { S.settings.resume = e.target.checked; saveSettings(); });
$('#set-repeat').addEventListener('change', (e) => {
  S.settings.repeat = e.target.value;
  saveSettings();
  video.loop = S.settings.repeat === 'one';
});

/* ================= 首次自動安裝 ================= */

// 後端（app/setup.py）在伺服器啟動時判斷是不是新使用者，自動排模型下載和字典建置，這裡只負責顯示：
// 全畫面進度頁（#setup）、右上角狀態（#setup-badge）、設定頁「自動安裝」卡片。
// api：伺服器有沒有這組 API，看 /api/state 有沒有 setup 這個欄位（沒有自動安裝時是 null）。
// 舊版伺服器沒有這個欄位，就不去讀 /api/setup（免得每次都在主控台留下 404）。
// data：GET /api/setup 的完整狀態；summary：/api/state 附的摘要；decided：判斷過要不要自動打開進度頁（打開網頁時判斷一次）
const SU = { api: false, data: null, summary: null, open: false, decided: false, busy: false, readySig: '' };

const SETUP_ROLE = { asr: '語音辨識', aligner: '時間軸', furigana: '日文假名', translator: '翻譯', dict_ja: '日文字典', dict_en: '英文字典' };
const SETUP_NAME = { dict_ja: 'JMdict', dict_en: 'ECDICT' };
// 失敗的錯誤代碼：[顯示的文字, 有沒有重試按鈕]；其他代碼顯示「失敗了」加重試
const SETUP_FAIL = {
  disk_full: ['磁碟空間不夠', true], no_permission: ['沒辦法寫入資料夾', true], ssl: ['連線被攔截，換個網路再試', true],
  gated: ['要登入才能下載', false], not_found: ['下載來源變了，請更新程式', false], changed: ['下載來源變了，請更新程式', false],
  file_locked: ['檔案被鎖住', true], corrupt: ['檔案壞了', true], corrupt_source: ['檔案壞了', true],
  no_7z: ['少了解壓元件，請重新執行 start.bat', false], dict_locked: ['重新打開程式後換上', false],
};
const GIB = 1024 ** 3;
const gbText = (b) => `${((b || 0) / GIB).toFixed(1)} GB`;
const setupOff = (d) => !d || !!d.off || d.status === 'off';

// 已下載 / 總共：總共不到 1 GB 時用 MB
function pairText(done, total) {
  if ((total || 0) >= GIB) return `${((done || 0) / GIB).toFixed(1)} / ${(total / GIB).toFixed(1)} GB`;
  const mb = (b) => Math.round((b || 0) / 1024 ** 2);
  return `${mb(done)} / ${Math.max(1, mb(total))} MB`;
}

function gpuLineText(g, withDriver = true) {
  if (!g || !g.name) return '';
  const gb = g.gb ?? (g.total_mb ? Math.round(g.total_mb / 1024) : null);
  return [g.name, gb ? `${gb} GB` : '', withDriver && g.driver ? `驅動 ${g.driver}` : '', withDriver ? cudaText(g) : '']
    .filter(Boolean).join(' · ');
}

// 不能裝的原因和怎麼辦
function setupBlockText(block, gpu) {
  const v = (block && block.values) || {};
  const g = gpu || {};
  const card = String(v.name || g.name || '').replace(/^NVIDIA\s+(GeForce\s+)?/i, '');
  const gbOf = (mb) => { const x = Math.floor((mb / 1024) * 10) / 10; return Number.isInteger(x) ? String(x) : x.toFixed(1); };
  switch (block && block.code) {
    case 'no_gpu':
      return ['沒有找到 NVIDIA 顯示卡。轉字幕和翻譯要用 NVIDIA 顯示卡，顯存 8 GB 以上。', '有 NVIDIA 顯示卡的話，先裝好驅動再按「重新檢查」。'];
    case 'driver_old': {
      const drv = v.driver || g.driver;
      const cuda = v.cuda || g.cuda;
      return [`顯示卡驅動太舊（目前 ${drv || '未知'}，支援 CUDA ${cuda || '讀不到'}）。這個程式需要驅動 ${DRIVER_MIN} 版以上（支援 CUDA ${CUDA_MIN}）。`,
        '到 NVIDIA 官網或 NVIDIA App 更新驅動，再按「重新檢查」。'];
    }
    case 'arch_old':
      return [`這張顯示卡${card ? `（${card}）` : ''}太舊，程式用不了。`, '需要 RTX 20、GTX 16 系列或更新的 NVIDIA 顯示卡。'];
    case 'vram_small': {
      const mb = v.total_mb || g.total_mb;
      return [mb ? `顯存只有 ${gbOf(mb)} GB，模型放不下。` : '顯存不夠，模型放不下。', '需要顯存 8 GB 以上的 NVIDIA 顯示卡。'];
    }
    case 'gpu_hidden':
      return ['環境變數 CUDA_VISIBLE_DEVICES 把顯示卡藏起來了。', '到 Windows 的環境變數拿掉它，再重新打開程式。'];
    case 'disk': {
      const need = v.need_bytes || 0, free = v.free_bytes || 0;
      const short = v.short_bytes ?? Math.max(0, need - free);
      return [`${v.drive || '磁碟'} 空間不夠：要 ${gbText(need)}，只剩 ${gbText(free)}。`,
        `清出 ${(Math.ceil((short / GIB) * 10) / 10).toFixed(1)} GB 以上再按「重新檢查」。`];
    }
    case 'detect_failed':
      return ['檢查電腦時出錯了。', '按「重新檢查」再試一次。一直這樣的話，把 data\\app.log 傳給作者。'];
    default:   // 不認得的代碼：用後端給的說明
      return [(block && (block.reason || block.message)) || '沒辦法自動安裝。', (block && block.fix) || '按「重新檢查」再試一次。'];
  }
}

const setupActBtn = (act, key, label) => `<button type="button" class="btn small" data-su-act="${act}" data-key="${esc(key)}">${label}</button>`;
// 名稱：字典顯示來源（JMdict、ECDICT），模型顯示名稱和版本
const setupItemName = (it) => it.source || SETUP_NAME[it.role] || it.label || it.id || it.key;

// 重試倒數的秒數：後端給數字，或 {in_s / wait_s / seconds} 這類物件
function retrySeconds(r) {
  const s = typeof r === 'number' ? r : r && (r.in_s ?? r.wait_s ?? r.seconds ?? r.next_s);
  return s > 0 ? Math.ceil(s) : 0;
}

function setupItemHtml(it) {
  const state = it.state || 'waiting';
  const size = it.size_bytes || 0;
  const byteFrac = size ? Math.min(1, (it.done_bytes || 0) / size) : 0;
  const prog = it.progress != null ? it.progress : byteFrac;
  let frac = 0, text = '', btn = '', tip = '';
  if (state === 'done') { frac = 1; text = '好了'; }
  else if (state === 'waiting') text = '等待中';
  else if (state === 'downloading') {
    frac = prog;
    text = size ? pairText(it.done_bytes, size) + (it.eta_s > 0 ? ` · ${fmtRemain(it.eta_s)}` : '') : `下載中 ${Math.round(prog * 100)}%`;
  } else if (state === 'checking') { frac = prog; text = '檢查已下載的部分'; }
  else if (state === 'retry_wait') {
    frac = byteFrac;
    const s = retrySeconds(it.retry);
    text = s ? `連線中斷，${s} 秒後重試` : '連線中斷，等一下重試';
  } else if (state === 'net_wait') { frac = byteFrac; text = '等網路恢復'; }
  else if (state === 'paused') { frac = prog; text = '已暫停'; btn = setupActBtn('resume', it.key, '繼續'); }
  else if (state === 'building') {
    frac = it.progress || 0;
    const stage = it.stage || 'build';
    text = stage === 'extract' ? '解壓縮' : stage === 'finish' ? '換上新字典' : `建立中 ${Math.round(frac * 100)}%`;
  } else if (state === 'canceled') { text = '已取消'; btn = setupActBtn('retry', it.key, '重新下載'); }
  else if (state === 'failed') {
    frac = byteFrac;
    const [t, retry] = SETUP_FAIL[it.error && it.error.code] || ['失敗了', true];
    text = t;
    if (retry) btn = setupActBtn('retry', it.key, '重試');
    if (it.error && it.error.message) tip = ` title="${esc(it.error.message)}"`;
  } else text = state;
  const name = setupItemName(it);
  return `<li class="s-${esc(state)}">
    <span class="su-role">${esc(it.use || SETUP_ROLE[it.role] || '')}</span>
    <span class="su-name" title="${esc(name)}">${esc(name)}</span>
    <span class="su-right"><span${tip}>${esc(text)}</span>${btn}</span>
    <div class="su-bar"><i style="width:${Math.round(Math.min(1, Math.max(0, frac)) * 100)}%"></i></div>
  </li>`;
}

// 總進度那一行：已下載 / 總共 · 速度 · 剩多久
function setupTotalText(d) {
  const t = d.totals || {};
  const items = d.items || [];
  const parts = [pairText(t.done_bytes, t.total_bytes)];
  const modelMoving = items.some((i) => i.kind !== 'dict' && (i.state === 'downloading' || i.state === 'checking'));
  const moving = items.some((i) => ['downloading', 'checking', 'building', 'retry_wait', 'waiting'].includes(i.state));
  // 有項目在等網路，而且模型沒有真的在下載（沒有速度）
  if (items.some((i) => i.state === 'net_wait') && (!modelMoving || !(t.speed_bps > 0))) parts.push('等網路恢復');
  else if (!moving && items.some((i) => i.state === 'paused')) parts.push('已暫停');
  else if (moving) {   // 沒有在動的（只剩失敗、取消的項目）就只寫大小
    // 後端在下載都完成時速度給 0；剛開始下載的幾秒沒有速度和剩餘時間，寫「計算中」
    if (t.speed_bps > 0) parts.push(`${(t.speed_bps / 1024 ** 2).toFixed(1)} MB/s`);
    else if ((t.done_bytes || 0) >= (t.total_bytes || 0) && items.some((i) => i.state === 'building')) parts.push('建立字典');
    if (t.eta_s > 0) parts.push(fmtRemain(t.eta_s));
    if (parts.length === 1) parts.push('計算中');
  }
  return parts.join(' · ');
}

function setupNotes(d) {
  const notes = [];
  const g = d.gpu || {}, c = d.choice || {};
  if (g.driver_warn) notes.push('驅動版本比較舊，翻譯開不起來時先更新驅動。');
  if (c.id === 'hymt2-1.8b') notes.push('顯存比較小，翻譯改用小模型 Hy-MT2-1.8B。');
  else if (c.variant && c.default_variant && c.variant !== c.default_variant) {
    const gb = g.gb ?? (g.total_mb ? Math.round(g.total_mb / 1024) : '');
    notes.push(`${gb ? `顯存 ${gb} GB，` : ''}翻譯模型用小一點的 ${c.variant} 版。`);
  }
  if (d.disk && d.disk.warn) notes.push(`裝完 ${d.disk.drive || '磁碟'} 只剩 ${Math.max(0, Math.round((d.disk.after_bytes || 0) / GIB))} GB，處理影片可能不夠。`);
  return notes;
}

function setupReadyText(r, items = []) {
  if (!r) return '';
  if (!r.transcribe) return '辨識和時間軸裝好就能先轉字幕。';
  if (!r.translate) {
    const tr = items.find((i) => i.role === 'translator');
    return tr && (tr.state === 'failed' || tr.state === 'canceled') ? '可以先轉字幕了。' : '可以先轉字幕了，翻譯還在下載。';
  }
  if (r.dict_ja === false || r.dict_en === false) return '可以轉字幕和翻譯了，字典還在建立。';
  return '';
}

const setHtml = (el, html) => { if (el.dataset.html !== html) { el.innerHTML = html; el.dataset.html = html; } };
const setText = (el, text) => { el.textContent = text; el.hidden = !text; };

function renderSetup() {
  const d = SU.data;
  if (!SU.open || !d) return;
  const st = d.status;
  const blocked = st === 'blocked', done = st === 'done', pending = st === 'pending';
  const installing = !blocked && !done && !pending;
  let title = '正在安裝';
  let sub = d.reason === 'redetect' ? '下載建議的模型和字典。' : '第一次使用，要下載模型和字典。';
  if (blocked) { title = '沒辦法自動安裝'; sub = ''; }
  else if (done) {
    title = '裝好了';
    // 手機上頂列的按鈕只寫「新增」
    const add = matchMedia('(max-width: 600px)').matches ? '新增' : '新增影片';
    sub = S.media.length ? '可以開始用了。' : `可以開始用了。按上方「${add}」加第一部影片。`;
  }
  else if (pending) { title = '正在準備'; sub = '檢查顯示卡和磁碟空間…'; }
  $('#setup-title').textContent = title;
  setText($('#setup-sub'), sub);
  setText($('#setup-gpu'), gpuLineText(d.gpu));

  const notes = installing ? setupNotes(d) : [];
  setHtml($('#setup-notes'), notes.map((n) => `<p>${esc(n)}</p>`).join(''));
  $('#setup-notes').hidden = !notes.length;

  // 還在檢查顯示卡時進度條來回跑（不知道要多久）
  $('#setup-total').hidden = !(installing || pending);
  $('#setup-total-bar').parentElement.classList.toggle('busy', pending);
  $('#setup-items').hidden = !installing;
  if (installing) {
    $('#setup-total-bar').style.width = `${Math.max(0, Math.min(100, (d.totals && d.totals.percent) || 0))}%`;
    setHtml($('#setup-items'), (d.items || []).map(setupItemHtml).join(''));
  }
  setText($('#setup-total-text'), installing ? setupTotalText(d) : '');
  setText($('#setup-ready'), installing ? setupReadyText(d.ready, d.items || []) : '');

  // 裝好了，但有使用者取消（或失敗）的項目
  const missing = done ? (d.items || []).filter((i) => i.state === 'canceled' || i.state === 'failed') : [];
  setHtml($('#setup-missing'), missing.map((i) => `<div><span>沒裝：${esc(setupItemName(i))}（${i.state === 'canceled' ? '已取消' : '失敗了'}）</span>${setupActBtn('retry', i.key, '重新下載')}</div>`).join(''));
  $('#setup-missing').hidden = !missing.length;

  $('#setup-block').hidden = !blocked;
  if (blocked) {
    const [why, how] = setupBlockText(d.block, d.gpu);
    $('#setup-why').textContent = why;
    $('#setup-how').textContent = how;
  }
  $('#setup-change').hidden = !installing;
  $('#setup-close').hidden = !blocked;
  $('#setup-main').textContent = blocked ? '重新檢查' : done ? '開始使用' : '先去用';
  $('#setup-foot').hidden = !(installing || pending);
}

// 右上角的安裝狀態：安裝中一直顯示；不能裝時顯示到按「關閉」為止
function renderSetupBadge() {
  let s = SU.summary;
  const d = SU.data;
  if (!s && d && !setupOff(d)) {
    const items = d.items || [];
    s = { status: d.status, percent: d.totals && d.totals.percent, dismissed: d.dismissed,
      problem: items.some((i) => i.state === 'failed'), net_wait: items.some((i) => i.state === 'net_wait') };
  }
  let long = '', short = '', cls = '';
  if (SU.api && s && s.status === 'installing') {
    if (s.problem) { long = '安裝有問題'; short = '!'; cls = 'bad'; }
    else if (s.net_wait) { long = '安裝等網路'; short = '等網路'; cls = 'wait'; }
    else { const p = Math.round(s.percent || 0); long = `安裝 ${p}%`; short = `${p}%`; }
  } else if (SU.api && s && s.status === 'blocked' && !s.dismissed && !(d && d.dismissed)) {
    long = '沒辦法安裝'; short = '!'; cls = 'bad';
  }
  const el = $('#setup-badge');
  el.hidden = !long;
  el.classList.toggle('bad', cls === 'bad');
  el.classList.toggle('wait', cls === 'wait');
  $('#setup-badge-long').textContent = long;
  $('#setup-badge-short').textContent = short;
}

// 設定頁「自動安裝」卡片：舊版伺服器沒有這組 API 時整張不顯示
function renderSetupCard() {
  const card = $('#setup-card');
  card.hidden = SU.api !== true;
  if (card.hidden) return;
  const s = SU.summary;
  const installing = !!s && s.status === 'installing';
  $('#btn-setup-install').disabled = SU.busy || installing;
  setHtml($('#setup-card-status'), installing
    ? `安裝中 ${Math.round(s.percent || 0)}%，<button type="button" class="link-btn" data-setup-open>看進度</button>` : '');
}

const shouldAutoOpenSetup = (d) => !setupOff(d) && !d.dismissed && d.reason !== 'redetect' && d.reason !== 'existing'
  && (['pending', 'installing', 'blocked'].includes(d.status) || (d.status === 'done' && d.reason === 'new'));

function applySetupData(d) {
  if (!d || typeof d !== 'object') return;
  SU.api = true;
  SU.data = d;
  if (!SU.decided) {
    SU.decided = true;
    if (shouldAutoOpenSetup(d)) openSetup(false);
  }
  if (SU.open) renderSetup();
  renderSetupBadge();
  if (SET.open) renderSetupCard();
}

async function refreshSetup() {
  try {
    const d = await api('GET', '/api/setup');
    applySetupData(d);
    return d;
  } catch (err) {
    if (err.status === 404 || err.status === 405) {   // 舊版伺服器：沒有自動安裝
      SU.api = false; SU.data = null; SU.decided = true;
      if (SU.open) closeSetup(false);
      renderSetupBadge();
      if (SET.open) renderSetupCard();
    }
    return null;
  }
}

// 進度頁開著時，後面的頂列、播放器、設定頁都不能按到（Tab 也跳不過去）
function setBehindInert(on) {
  for (const el of $$('.topbar, .layout, #settings-page, #vocab-page')) el.inert = on;
}

function openSetup(fetch = true) {
  if (!SU.api) return;
  SU.open = true;
  $('#setup').hidden = false;
  setBehindInert(true);
  if (!video.paused) video.pause();
  closeMenu();
  window.WL?.hideCard?.();
  if (!$('#mini').hidden) window.WL?.closeMini?.();
  renderSetup();
  $('#setup-box').focus({ preventScroll: true });
  if (fetch) refreshSetup();
}

// 關掉進度頁（先去用、開始使用、關閉、Esc）：記住按過了，重新整理後不會再自動蓋上
function closeSetup(dismiss = true) {
  SU.open = false;
  $('#setup').hidden = true;
  setBehindInert(false);
  if (dismiss && SU.data && !SU.data.dismissed) {
    SU.data.dismissed = true;
    if (SU.summary) SU.summary.dismissed = true;
    api('POST', '/api/setup/dismiss').catch(() => { /* 下次打開網頁會再出現，不影響安裝 */ });
  }
  renderSetupBadge();
}

// /api/state 帶的摘要：更新右上角狀態；辨識、翻譯模型裝好時重新讀模型清單，新增影片的選單才會更新
function onSetupSummary(st) {
  if (st && 'setup' in st) SU.api = true;
  const sum = st && st.setup;
  const prev = SU.summary;
  SU.summary = sum && !setupOff(sum) ? sum : null;
  const s = SU.summary;
  const sig = s ? JSON.stringify([s.status, s.ready || null]) : '';
  if (SU.readySig && sig !== SU.readySig) refreshMeta().then(() => { if (dlgAdd.open) updateAddForm(); });
  SU.readySig = sig;
  if (prev && prev.status === 'installing' && s && s.status === 'done' && !SU.open) toast('模型和字典都裝好了');
  renderSetupBadge();
  if (SET.open) renderSetupCard();
}

// 佇列裡等模型的任務：等什麼、還要多久
function waitingHtml(w, link = true) {
  const names = esc((w.labels && w.labels.length ? w.labels : w.models || []).join('、'));
  const see = link ? '<button type="button" class="link-btn" data-act="setup">看安裝進度</button>' : '看安裝進度';
  switch (w.state) {
    case 'paused': return `等 ${names}（下載已暫停）`;
    case 'net_wait': return `等 ${names}（等網路恢復）`;
    case 'failed': return `等 ${names}（下載失敗，${see}）`;
    case 'downloading': return `等 ${names} 下載完${w.eta_s > 0 ? `，${fmtRemain(w.eta_s)}` : ''}`;
    default: return `等 ${names} 下載完`;
  }
}

// 設定頁「重新偵測並安裝建議模型」：先算計畫給使用者看，確認了才裝（不自動打開全畫面進度頁）
async function startSetupInstall(btn) {
  SU.busy = true;
  btn.disabled = true;
  try {
    let plan = await api('GET', '/api/setup/plan');
    for (;;) {
      if (plan.block) {
        const [why, how] = setupBlockText(plan.block, plan.gpu);
        await askConfirm({ title: '沒辦法自動安裝', text: why + how, ok: '知道了', cancel: '' });
        return;
      }
      const items = plan.items || [];
      const changes = plan.changes || [];
      if (plan.nothing || (items.every((i) => i.installed) && !changes.length)) { toast('建議的模型和字典都裝好了'); return; }
      const lines = [
        ...items.filter((i) => !i.installed).map((i) => `下載 ${i.label}${i.need_bytes ? `（${fmtSize(i.need_bytes)}）` : ''}`),
        ...changes.map((c) => c.text),
      ];
      const have = items.filter((i) => i.installed).map((i) => i.label);
      if (have.length) lines.push(`已經有：${have.join('、')}`);
      const ok = await askConfirm({ title: '安裝建議模型', text: gpuLineText(plan.gpu, false), items: lines, ok: '開始安裝' });
      if (!ok) return;
      try {
        const res = await api('POST', '/api/setup/install', { signature: plan.signature });
        if (res && res.nothing) toast('建議的模型和字典都裝好了');
        else {
          if (res && res.status) applySetupData(res);
          toast('開始安裝，右上角可以看進度');
        }
        refreshSoon();
        return;
      } catch (err) {
        if (err.code === 'plan_changed' && err.plan) {
          toast('顯示卡或檔案有變，請再看一次。', true);
          plan = err.plan;
          // 等上一個確認框的 close 事件跑完，不然它會把馬上重開的確認框當成被關掉
          await new Promise((r) => setTimeout(r, 0));
          continue;
        }
        throw err;
      }
    }
  } catch (err) {
    toast(err.message, true);
  } finally {
    SU.busy = false;
    renderSetupCard();
  }
}

$('#setup-main').addEventListener('click', async (e) => {
  if (!SU.data || SU.data.status !== 'blocked') { closeSetup(); return; }
  const btn = e.currentTarget;
  btn.disabled = true;
  try {
    const res = await api('POST', '/api/setup/recheck');
    applySetupData(res);
    if (res && res.status === 'blocked') toast('還是沒辦法安裝');
    refreshSoon();
  } catch (err) {
    // 還是不能裝時後端回 400 blocked 或 disk（畫面上的原因可能變了，例如磁碟剩多少）：原因看畫面，不另外用 toast 講一次數字
    if (err.code === 'blocked' || err.code === 'disk') { await refreshSetup(); toast('還是沒辦法安裝'); }
    else if (err.code === 'not_blocked') await refreshSetup();
    else toast(err.message, true);
  } finally {
    btn.disabled = false;
  }
});
$('#setup-close').addEventListener('click', () => closeSetup());
$('#setup-change').addEventListener('click', () => { closeSetup(); openModelManager('translator'); });
$('#setup').addEventListener('click', async (e) => {
  const b = e.target.closest('[data-su-act]');
  if (!b) return;
  b.disabled = true;
  try {
    const res = await api('POST', `/api/setup/items/${encodeURIComponent(b.dataset.key)}/${b.dataset.suAct}`);
    if (res && res.status) applySetupData(res); else refreshSetup();
    refreshSoon();
  } catch (err) {
    toast(err.message, true);
    refreshSetup();
  } finally {
    b.disabled = false;
  }
});
$('#setup-badge').addEventListener('click', () => openSetup());
$('#setup-card-status').addEventListener('click', (e) => { if (e.target.closest('[data-setup-open]')) openSetup(); });
$('#btn-setup-install').addEventListener('click', (e) => startSetupInstall(e.currentTarget));
// 進度頁開著時播放器、單字頁的快捷鍵都不作用（在 window 的捕獲階段就攔下來）；Esc 等於「先去用」「關閉」
window.addEventListener('keydown', (e) => {
  if (!SU.open || document.querySelector('dialog[open]')) return;
  e.stopPropagation();
  if (e.key === 'Escape') { e.preventDefault(); closeSetup(); }
}, true);

/* ================= 啟動 ================= */

(async function init() {
  lastLayout = layoutMode();
  applyPanels();
  applyTabs();
  applySubSettings();
  applyVolume();
  setRate(S.settings.rate);
  updatePlayButton();
  try {
    S.meta = await api('GET', '/api/meta');
  } catch {
    $('#offline').hidden = false;
    setTimeout(init, 2000);
    return;
  }
  // 先決定要不要蓋上安裝進度頁，再畫片庫，播放列表才不會先閃出來
  try {
    const st = await api('GET', '/api/state');
    onSetupSummary(st);
    if (SU.api) await refreshSetup();
  } catch { /* 輪詢會再試 */ }
  poll();
  syncPluginUserAgent();
})();
