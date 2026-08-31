/* agent console - replay recorded runs, edit the prompt, launch a new one. */
'use strict';

const $ = (s, r) => (r || document).querySelector(s);
const $$ = (s, r) => Array.from((r || document).querySelectorAll(s));
const SVGNS = 'http://www.w3.org/2000/svg';

// Reproduced official FM baselines, per dataset. 27k has never been measured,
// so it gets no reference line rather than a wrong one.
const BASELINE = { pure: 0.6015, '1k': 0.6451 };
const VERDICT_COLOR = {
  KEPT: 'var(--kept)', noise: 'var(--noise)', worse: 'var(--worse)',
  failed: 'var(--failed)', 'no-op': 'var(--noop)', screen: 'var(--screen)',
  duplicate: 'var(--screen)'
};

const state = {
  runs: [], run: null, runId: null, cursor: 0, timer: null,
  live: { records: [], events: [], runId: null, dataset: 'pure', done: false }
};

/* ------------------------------------------------------------------ tabs */
$$('.tab').forEach(t => t.addEventListener('click', () => {
  $$('.tab').forEach(x => x.classList.toggle('active', x === t));
  $$('.screen').forEach(s => s.classList.toggle(
    'active', s.id === 'screen-' + t.dataset.screen));
  if (t.dataset.screen === 'agent') loadPrompt();
}));

function fmt(v, d) { return v == null ? '--' : Number(v).toFixed(d == null ? 6 : d); }
function esc(s) {
  return String(s == null ? '' : s).replace(/[&<>]/g,
    c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));
}
function clock(ts) { return ts ? String(ts).slice(11, 19) : ''; }

/* ------------------------------------------------------------- run list */
async function loadRuns() {
  const r = await fetch('/api/runs').then(x => x.json());
  state.runs = r.runs || [];
  renderRuns();
}

function substantive(r) {
  return r.name !== 'iterations' && r.iterations >= 3;
}

function renderRuns() {
  const only = $('#filter-real').checked;
  const list = state.runs.filter(r => !only || substantive(r));
  $('#runs').innerHTML = list.map(r => {
    // A run that scored above 0.9 did not out-model anyone: it recovered the
    // label. Flagging it in the list is the honest way to keep it visible.
    const leak = r.best != null && r.best > 0.9;
    return `<div class="runitem ${r.id === state.runId ? 'sel' : ''}" data-id="${r.id}">
      <div class="rn"><span>${esc(r.name)}</span>
        <span class="best">${r.best == null ? '--' : fmt(r.best, 4)}</span></div>
      <div class="rmeta">
        <span class="badge ${r.dataset === '1k' ? 'ds1k' : ''}">${r.dataset}</span>
        <span>${r.iterations} iter</span>
        ${r.failed ? `<span>${r.failed} failed</span>` : ''}
        ${r.noop ? `<span>${r.noop} no-op</span>` : ''}
        <span>$${r.cost_usd.toFixed(2)}</span>
        ${leak ? '<span class="badge warn">label leak</span>' : ''}
      </div></div>`;
  }).join('') || '<div class="rmeta" style="padding:14px">No runs.</div>';
  $$('#runs .runitem').forEach(el =>
    el.addEventListener('click', () => selectRun(el.dataset.id)));
}
$('#filter-real').addEventListener('change', renderRuns);

/* --------------------------------------------------------------- replay */
async function selectRun(id) {
  stopPlay();
  state.runId = id;
  renderRuns();
  const data = await fetch('/api/run?id=' + encodeURIComponent(id)).then(x => x.json());
  if (data.error) return;
  state.run = data;
  state.run.summary = state.runs.find(r => r.id === id) || {};
  state.cursor = 0;
  const n = data.iterations.length;
  $('#run-title').textContent = state.run.summary.name || id;
  const s = state.run.summary;
  $('#run-meta').textContent =
    `${id}  |  ${n} iterations  |  model ${s.model || '?'}  |  $${(s.cost_usd || 0).toFixed(2)}`
    + (s.started ? `  |  ${String(s.started).slice(0, 16).replace('T', ' ')}` : '');
  const sc = $('#scrubber');
  sc.max = n; sc.value = 0; sc.disabled = false;
  ['#btn-play', '#btn-step', '#btn-reset'].forEach(b => $(b).disabled = false);
  draw();
}

function nodesUpTo(recs, cursor) {
  return recs.filter(r => r.iteration <= cursor);
}

/* Lay the run out as the tree it actually is: every record carries a parent,
 * so a run is a search tree and not a list. Depth left to right, children
 * stacked vertically, parents centred on their children. */
function layout(recs) {
  const byIter = new Map(recs.map(r => [r.iteration, r]));
  const kids = new Map();
  const roots = [];
  recs.forEach(r => {
    const p = parseInt(r.parent, 10);
    if (Number.isFinite(p) && byIter.has(p) && p !== r.iteration) {
      if (!kids.has(p)) kids.set(p, []);
      kids.get(p).push(r.iteration);
    } else roots.push(r.iteration);
  });
  const pos = new Map();
  let slot = 0;
  const seen = new Set();
  function walk(it, depth) {
    if (seen.has(it)) return 0;
    seen.add(it);
    const ch = (kids.get(it) || []).sort((a, b) => a - b);
    let y;
    if (!ch.length) { y = slot++; }
    else {
      const ys = ch.map(c => walk(c, depth + 1)).filter(v => v != null);
      y = ys.length ? (Math.min(...ys) + Math.max(...ys)) / 2 : slot++;
    }
    pos.set(it, { depth, y });
    return y;
  }
  roots.sort((a, b) => a - b).forEach(r => walk(r, 0));
  recs.forEach(r => { if (!pos.has(r.iteration)) walk(r.iteration, 0); });
  return { pos, byIter, kids };
}

function el(tag, attrs, parent) {
  const n = document.createElementNS(SVGNS, tag);
  for (const k in attrs) n.setAttribute(k, attrs[k]);
  if (parent) parent.appendChild(n);
  return n;
}

function drawTree(svgSel, recs, cursor, opts) {
  const svg = $(svgSel);
  svg.innerHTML = '';
  if (!recs.length) return;
  const { pos, byIter } = layout(recs);
  const COLW = 58, ROWH = 25, PADX = 26, PADY = 18;
  let maxD = 0, maxY = 0;
  pos.forEach(p => { maxD = Math.max(maxD, p.depth); maxY = Math.max(maxY, p.y); });
  const W = PADX * 2 + maxD * COLW + 30;
  const H = PADY * 2 + maxY * ROWH + 20;
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  svg.setAttribute('height', H);
  svg.style.minWidth = W + 'px';
  const X = it => PADX + pos.get(it).depth * COLW;
  const Y = it => PADY + pos.get(it).y * ROWH;

  const visible = recs.filter(r => r.iteration <= cursor);
  const best = bestSoFar(recs, cursor);

  // edges first so nodes sit on top
  visible.forEach(r => {
    const p = parseInt(r.parent, 10);
    if (!Number.isFinite(p) || !pos.has(p) || p > cursor) return;
    const x1 = X(p), y1 = Y(p), x2 = X(r.iteration), y2 = Y(r.iteration);
    const mx = (x1 + x2) / 2;
    const par = byIter.get(p);
    const recovered = par && (par.status === 'error' || par.status === 'no-op');
    el('path', {
      d: `M${x1},${y1} C${mx},${y1} ${mx},${y2} ${x2},${y2}`,
      class: 'edge' + (r.iteration === cursor ? ' hot' : '')
        + (recovered ? ' recover' : '')
    }, svg);
  });

  visible.forEach(r => {
    const g = el('g', {
      class: 'node' + (r.iteration === cursor ? ' cur' : '')
        + (best && r.iteration === best.iteration ? ' best' : '')
    }, svg);
    const failed = r.status === 'error';
    el('circle', {
      cx: X(r.iteration), cy: Y(r.iteration),
      r: r.iteration === cursor ? 7 : (failed ? 5.5 : 5),
      fill: VERDICT_COLOR[r.verdict] || 'var(--noise)'
    }, g);
    el('text', {
      x: X(r.iteration), y: Y(r.iteration) - 9, 'text-anchor': 'middle'
    }, g).textContent = r.iteration;
    g.addEventListener('click', () => {
      if (opts && opts.onclick) opts.onclick(r.iteration);
    });
    const t = el('title', {}, g);
    t.textContent = `#${r.iteration}  ${r.verdict || ''}  ${fmt(r.valid_primary, 5)}`;
  });
}

function bestSoFar(recs, cursor) {
  let best = null;
  recs.forEach(r => {
    if (r.iteration > cursor) return;
    if (r.valid_primary == null) return;
    if (r.verdict === 'screen' || r.verdict === 'duplicate') return;
    if (!best || r.valid_primary > best.valid_primary) best = r;
  });
  return best;
}

function drawChart(svgSel, recs, cursor, dataset) {
  const svg = $(svgSel);
  svg.innerHTML = '';
  const box = svg.parentElement.getBoundingClientRect();
  const W = Math.max(320, box.width || 480), H = 118;
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  svg.setAttribute('height', H);
  const scored = recs.filter(r => r.valid_primary != null
    && r.verdict !== 'screen' && r.verdict !== 'duplicate');
  if (!scored.length) return;
  const base = BASELINE[dataset];
  const vals = scored.map(r => r.valid_primary).concat(base ? [base] : []);
  let lo = Math.min(...vals), hi = Math.max(...vals);
  const pad = Math.max((hi - lo) * 0.25, 0.0008);
  lo -= pad; hi += pad;
  const L = 46, R = 10, T = 10, B = 16;
  const n = Math.max(...recs.map(r => r.iteration));
  const px = i => L + (n <= 1 ? 0 : (i - 1) / (n - 1)) * (W - L - R);
  const py = v => T + (1 - (v - lo) / (hi - lo)) * (H - T - B);

  [lo, (lo + hi) / 2, hi].forEach(v => {
    el('line', { x1: L, x2: W - R, y1: py(v), y2: py(v), class: 'gridline' }, svg);
    el('text', { x: 4, y: py(v) + 3, class: 'axlabel' }, svg).textContent = v.toFixed(4);
  });
  if (base) {
    el('line', { x1: L, x2: W - R, y1: py(base), y2: py(base), class: 'baseline' }, svg);
    el('text', { x: W - R - 2, y: py(base) - 4, class: 'axlabel', 'text-anchor': 'end' },
      svg).textContent = 'FM baseline';
  }

  const shown = scored.filter(r => r.iteration <= cursor);
  if (shown.length) {
    const d = shown.map((r, i) => `${i ? 'L' : 'M'}${px(r.iteration)},${py(r.valid_primary)}`).join(' ');
    el('path', { d, class: 'spark', stroke: 'var(--accent)' }, svg);
    // best-so-far as a step line: the number the convergence rule watches
    let b = -Infinity; const pts = [];
    shown.forEach(r => { b = Math.max(b, r.valid_primary); pts.push([r.iteration, b]); });
    let dd = '';
    pts.forEach((p, i) => {
      dd += i ? `L${px(p[0])},${py(pts[i - 1][1])}L${px(p[0])},${py(p[1])}`
        : `M${px(p[0])},${py(p[1])}`;
    });
    el('path', { d: dd, class: 'spark', stroke: 'var(--kept)', 'stroke-dasharray': '3 2', 'stroke-width': 1.2 }, svg);
    shown.forEach(r => el('circle', {
      cx: px(r.iteration), cy: py(r.valid_primary),
      r: r.iteration === cursor ? 3.6 : 2,
      fill: VERDICT_COLOR[r.verdict] || 'var(--noise)'
    }, svg));
  }
  recs.filter(r => r.status === 'error' && r.iteration <= cursor).forEach(r => {
    el('line', {
      x1: px(r.iteration), x2: px(r.iteration), y1: T, y2: H - B,
      stroke: 'var(--failed)', 'stroke-width': 1, 'stroke-dasharray': '2 2',
      opacity: .55
    }, svg);
  });
}

function deltaSpan(v, prev, digits) {
  if (v == null) return '--';
  if (prev == null) return fmt(v, digits);
  const d = v - prev;
  const cls = d > 0 ? 'up' : (d < 0 ? 'down' : '');
  const sign = d > 0 ? '+' : '';
  return `${fmt(v, digits)} <span class="${cls}">${sign}${d.toFixed(digits || 6)}</span>`;
}

function renderDetail(sel, verdictSel, rec, run) {
  const box = $(sel);
  if (!rec) { box.innerHTML = '<p class="empty">No iteration selected.</p>'; return; }
  const vEl = $(verdictSel);
  if (vEl) {
    vEl.textContent = rec.verdict || rec.status || '';
    vEl.style.color = VERDICT_COLOR[rec.verdict] || 'var(--fg-dim)';
  }
  const titleEl = $('#detail-title');
  if (titleEl && sel === '#detail') {
    titleEl.textContent = `Iteration ${rec.iteration}`
      + (rec.parent && rec.parent !== '-' ? ` (from ${rec.parent})` : '');
  }

  const recs = (run && run.iterations) || [];
  const parent = recs.find(r => String(r.iteration) === String(rec.parent));
  let h = '';
  h += `<p class="hyp">${esc(rec.hypothesis || '(no hypothesis recorded)')}</p>`;

  if (rec.error) h += `<div class="errbox">${esc(rec.error)}</div>`;

  // Recovery is a property of the run log, not of this record, so read it
  // from events rather than inferring it.
  const evs = (run && run.events) || state.live.events;
  const rec_ev = (evs || []).find(e => e.kind === 'solution_recovered'
    && String(e.iteration) === String(rec.iteration));
  if (rec_ev) h += `<div class="recbox"><strong>recovered</strong> &mdash; ${esc(rec_ev.detail)}</div>`;

  h += '<dl class="kv">';
  h += `<dt>primary</dt><dd>${deltaSpan(rec.valid_primary, parent && parent.valid_primary, 6)}</dd>`;
  h += `<dt>GAUC</dt><dd>${deltaSpan(rec.GAUC, parent && parent.GAUC, 6)}</dd>`;
  h += `<dt>nDCG@5</dt><dd>${deltaSpan(rec['nDCG@5'], parent && parent['nDCG@5'], 6)}</dd>`;
  if (rec.primary_std != null) h += `<dt>seed &plusmn;</dt><dd>${fmt(rec.primary_std, 6)}</dd>`;
  if (rec.delta != null) h += `<dt>vs baseline</dt><dd>${rec.delta > 0 ? '+' : ''}${fmt(rec.delta, 5)}</dd>`;
  h += `<dt>status</dt><dd>${esc(rec.status || '')}</dd>`;
  if (rec.seconds != null) h += `<dt>run time</dt><dd>${Math.round(rec.seconds)}s</dd>`;
  if (rec.cost_usd != null) h += `<dt>cost</dt><dd>$${Number(rec.cost_usd).toFixed(3)}</dd>`;
  if (rec.tokens_in != null) h += `<dt>tokens</dt><dd>${rec.tokens_in} in / ${rec.tokens_out} out</dd>`;
  if (rec.solution) h += `<dt>solution</dt><dd>${esc(rec.solution.split('/').pop())}</dd>`;
  h += '</dl>';

  if (rec.per_seed && rec.per_seed.length) {
    h += '<div class="subhead">per seed</div><div class="seedrow">'
      + rec.per_seed.map(s => `<span>seed ${s.seed}: ${fmt(s.primary, 5)}</span>`).join('')
      + '</div>';
  }

  // GAUC and nDCG@5 routinely move in opposite directions and the mean hides
  // it - call that out where it happens, since it is where the real gain came
  // from in this project.
  if (parent && rec.GAUC != null && parent.GAUC != null
    && rec['nDCG@5'] != null && parent['nDCG@5'] != null) {
    const dg = rec.GAUC - parent.GAUC, dn = rec['nDCG@5'] - parent['nDCG@5'];
    if (dg * dn < 0 && Math.abs(dg) > 1e-5 && Math.abs(dn) > 1e-5) {
      h += `<div class="recbox" style="background:rgba(88,176,255,.07);
        border-color:rgba(88,176,255,.3);border-left-color:var(--accent);color:#bcdcff">
        The two metrics moved in <strong>opposite directions</strong>
        (GAUC ${dg > 0 ? '+' : ''}${dg.toFixed(6)},
         nDCG@5 ${dn > 0 ? '+' : ''}${dn.toFixed(6)}). The mean hides this.</div>`;
    }
  }

  const diff = run && run.diffs && run.diffs[String(rec.iteration)];
  if (diff) {
    h += '<div class="subhead">code diff from parent</div>';
    h += '<pre class="diff">' + diff.split('\n').map(l => {
      const c = l.startsWith('+') ? 'dl-add' : l.startsWith('-') ? 'dl-del'
        : l.startsWith('@@') ? 'dl-hunk' : l.startsWith('#') ? 'dl-meta' : '';
      return `<span class="${c}">${esc(l)}</span>`;
    }).join('\n') + '</pre>';
  }
  if (rec.stdout_tail) {
    h += '<div class="subhead">stdout tail</div><pre class="src">'
      + esc(rec.stdout_tail) + '</pre>';
  }
  box.innerHTML = h;
}

function renderEvents(sel, events, upto, cutoffTs) {
  const box = $(sel);
  // Events carrying an iteration filter on it. run_start / converged / run_end
  // carry none, so fall back to their timestamp - otherwise a run shows as
  // converged before the replay has played a single experiment.
  const shown = events.filter(e => {
    if (e.iteration != null) return e.iteration <= upto;
    if (!cutoffTs) return false;
    return !e.ts || e.ts <= cutoffTs;
  });
  box.innerHTML = shown.map(e =>
    `<div class="ev ${esc(e.kind)}"><span class="t">${clock(e.ts)}</span>
      <span class="k">${esc(e.kind)}</span>
      <span class="d" title="${esc(e.detail)}">${esc(e.detail)}</span></div>`
  ).join('');
  box.scrollTop = box.scrollHeight;
}

function draw() {
  const run = state.run;
  if (!run) return;
  const recs = run.iterations;
  const ds = (state.run.summary && state.run.summary.dataset) || 'pure';
  drawTree('#tree', recs, state.cursor, { onclick: i => { stopPlay(); setCursor(i); } });
  drawChart('#chart', recs, state.cursor, ds);
  const cur = recs.find(r => r.iteration === state.cursor);
  renderDetail('#detail', '#detail-verdict', cur, run);
  const curTs = cur && cur.timestamp;
  renderEvents('#events', run.events, state.cursor, curTs);
  const best = bestSoFar(recs, state.cursor);
  $('#scrub-label').textContent = state.cursor
    ? `iteration ${state.cursor} / ${recs.length}   best ${best ? fmt(best.valid_primary, 5) : '--'}`
    : `${recs.length} iterations - press Play`;
  $('#scrubber').value = state.cursor;
}

function setCursor(i) {
  state.cursor = Math.max(0, Math.min(i, state.run.iterations.length));
  draw();
}
$('#scrubber').addEventListener('input', e => { stopPlay(); setCursor(+e.target.value); });
$('#btn-step').addEventListener('click', () => { stopPlay(); setCursor(state.cursor + 1); });
$('#btn-reset').addEventListener('click', () => { stopPlay(); setCursor(0); });
$('#btn-play').addEventListener('click', () => {
  if (state.timer) return stopPlay();
  if (state.cursor >= state.run.iterations.length) setCursor(0);
  $('#btn-play').textContent = 'Pause';
  const tick = () => {
    if (state.cursor >= state.run.iterations.length) return stopPlay();
    setCursor(state.cursor + 1);
    state.timer = setTimeout(tick, +$('#speed').value);
  };
  state.timer = setTimeout(tick, 120);
});
function stopPlay() {
  if (state.timer) clearTimeout(state.timer);
  state.timer = null;
  $('#btn-play').textContent = 'Play';
}

/* --------------------------------------------------------- prompt editor */
let promptLoaded = false;
async function loadPrompt(force) {
  if (promptLoaded && !force) return;
  const p = await fetch('/api/prompt').then(x => x.json());
  $('#prompt').value = p.text;
  $('#promptstat').innerHTML =
    `<div>source: <span class="${p.source === 'override' ? 'tagged' : ''}">${p.source}</span></div>
     <div>sha256: ${p.hash}</div>
     <div>${p.chars.toLocaleString()} chars &middot; ${p.lines} lines</div>`;
  promptLoaded = true;
}
$('#btn-save-prompt').addEventListener('click', async () => {
  const msg = $('#prompt-msg');
  msg.className = 'msg'; msg.textContent = 'saving...';
  const r = await fetch('/api/prompt', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text: $('#prompt').value })
  }).then(x => x.json());
  if (r.ok) {
    msg.className = 'msg ok';
    msg.textContent = r.reverted
      ? 'identical to the shipped prompt - override removed'
      : 'saved to agent/prompt_override.txt';
    loadPrompt(true);
  } else { msg.className = 'msg err'; msg.textContent = r.error || 'save failed'; }
});
$('#btn-reset-prompt').addEventListener('click', async () => {
  await fetch('/api/prompt', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ reset: true })
  });
  const msg = $('#prompt-msg');
  msg.className = 'msg ok'; msg.textContent = 'reverted to the shipped prompt';
  loadPrompt(true);
});

/* ---------------------------------------------------------------- launch */
const PRESETS = {
  demo: { max_iter: 3, max_experiments: 50, wall: 1, cost: 0 },
  short: { max_iter: 10, max_experiments: 50, wall: 2, cost: 0 },
  full: { max_iter: 100, max_experiments: 50, wall: 6, cost: 0 }
};
$$('.preset').forEach(b => b.addEventListener('click', () => {
  $$('.preset').forEach(x => x.classList.toggle('sel', x === b));
  const p = PRESETS[b.dataset.preset];
  $('#opt-max-iter').value = p.max_iter;
  $('#opt-max-experiments').value = p.max_experiments;
  $('#opt-max-wall').value = p.wall;
  $('#opt-max-cost').value = p.cost;
}));

function readOpts() {
  return {
    dataset: $('#opt-dataset').value,
    model: $('#opt-model').value.trim(),
    max_iter: +$('#opt-max-iter').value,
    max_experiments: +$('#opt-max-experiments').value,
    max_wall_seconds: Math.round(+$('#opt-max-wall').value * 3600),
    max_cost_usd: +$('#opt-max-cost').value,
    run_name: $('#opt-run-name').value.trim()
  };
}
function showStep(id) {
  $$('#screen-launch .step').forEach(s => s.classList.toggle('active', s.id === id));
  window.scrollTo(0, 0);
}
$('#btn-back').addEventListener('click', () => showStep('step-settings'));

$('#btn-review').addEventListener('click', async () => {
  const o = readOpts();
  showStep('step-review');
  $('#rev-system').textContent = 'loading...';
  const pf = await fetch('/api/preflight?dataset=' + o.dataset).then(x => x.json());
  if (pf.error) { $('#rev-system').textContent = 'error: ' + pf.error; return; }

  const facts = [
    ['dataset', o.dataset], ['model', o.model || 'from .env'],
    ['max iterations', o.max_iter], ['max experiments', o.max_experiments],
    ['wall ceiling', (o.max_wall_seconds / 3600).toFixed(1) + ' h'],
    ['cost ceiling', o.max_cost_usd > 0 ? '$' + o.max_cost_usd.toFixed(2) : 'off'],
    ['baseline to beat', pf.baseline], ['epsilon', pf.epsilon],
    ['prompt', pf.identity.prompt_source], ['prompt sha256', pf.identity.prompt_hash],
    ['est. prompt tokens', pf.est_tokens.toLocaleString()]
  ];
  $('#review-facts').innerHTML = facts.map(
    f => `<div class="fact"><div class="l">${esc(f[0])}</div><div class="v">${esc(f[1])}</div></div>`
  ).join('');

  const warn = $('#review-warn');
  const notes = [];
  if (pf.identity.prompt_source === 'override') {
    notes.push('This run will use an <strong>edited prompt</strong> ('
      + pf.identity.prompt_hash + '), not the shipped one. The hash is written '
      + 'into <code>run_start</code> in events.jsonl, so the run record says so '
      + 'permanently.');
  }
  if (o.dataset !== 'pure') {
    notes.push('KuaiRand-1k is a <strong>bonus</strong> benchmark, scored '
      + 'against a baseline measured here rather than an organiser-published one.');
  }
  if (o.max_iter >= 30) {
    notes.push('This is a full-length run: roughly '
      + Math.round(o.max_iter * 2) + ' minutes and several dollars of API spend.');
  }
  warn.hidden = !notes.length;
  warn.innerHTML = notes.map(n => '<div>' + n + '</div>').join('');

  $('#rev-system').textContent = pf.system;
  $('#rev-user').textContent = pf.user;
  $('#rev-sys-meta').textContent = pf.system.length.toLocaleString() + ' chars';
  $('#rev-user-meta').textContent = pf.user.length.toLocaleString() + ' chars';
  $('#rev-tools-meta').textContent = pf.tools.length + ' tools';
  $('#rev-tools').innerHTML = pf.tools.map(
    t => `<span class="toolchip">${esc(t)}</span>`).join('');
  $('#confirm-hint').textContent =
    `python -m agent --dataset ${o.dataset} --max-iter ${o.max_iter} --run-id ${o.run_name}-N`;
  $('#ack').checked = false;
  $('#btn-confirm').disabled = true;
});
$('#ack').addEventListener('change', e => $('#btn-confirm').disabled = !e.target.checked);

$('#btn-confirm').addEventListener('click', async () => {
  const o = readOpts();
  $('#btn-confirm').disabled = true;
  const r = await fetch('/api/run', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(o)
  }).then(x => x.json());
  if (!r.ok) {
    $('#confirm-hint').textContent = 'could not start: ' + (r.error || '?');
    $('#btn-confirm').disabled = false;
    return;
  }
  state.live = { records: [], events: [], runId: r.run_id, dataset: o.dataset, done: false };
  $('#live-title').textContent = 'Running ' + r.run_id;
  $('#live-meta').textContent = r.cmd;
  $('#console').textContent = '';
  $('#btn-stop').hidden = false;
  $('#btn-newrun').hidden = true;
  $('#livepill').hidden = false;
  $('#livepill-text').textContent = 'running';
  showStep('step-live');
  drawLive();
});

$('#btn-stop').addEventListener('click', async () => {
  $('#btn-stop').disabled = true;
  await fetch('/api/stop', { method: 'POST' });
});
$('#btn-newrun').addEventListener('click', () => { showStep('step-settings'); loadRuns(); });

function drawLive() {
  const recs = state.live.records;
  drawTree('#live-tree', recs, Math.max(...recs.map(r => r.iteration), 0), {});
  drawChart('#live-chart', recs, Math.max(...recs.map(r => r.iteration), 0),
    state.live.dataset);
  const last = recs[recs.length - 1];
  renderDetail('#live-detail', '#live-verdict', last,
    { iterations: recs, events: state.live.events, diffs: {} });
}

function connectStream() {
  const es = new EventSource('/api/stream');
  es.onmessage = ev => {
    let m;
    try { m = JSON.parse(ev.data); } catch (e) { return; }
    if (m.type === 'stdout') {
      const c = $('#console');
      c.textContent += m.line + '\n';
      if ($('#follow').checked) c.scrollTop = c.scrollHeight;
    } else if (m.type === 'iteration') {
      const recs = state.live.records;
      const i = recs.findIndex(r => r.iteration === m.record.iteration);
      if (i >= 0) recs[i] = m.record; else recs.push(m.record);
      recs.sort((a, b) => a.iteration - b.iteration);
      drawLive();
    } else if (m.type === 'event') {
      state.live.events.push(m.event);
      const c = $('#console');
      c.textContent += `  << ${m.event.kind}: ${String(m.event.detail).slice(0, 160)}\n`;
      if ($('#follow').checked) c.scrollTop = c.scrollHeight;
      drawLive();
    } else if (m.type === 'exit') {
      state.live.done = true;
      $('#live-title').textContent = 'Finished ' + (state.live.runId || '');
      $('#livepill-text').textContent = 'exit ' + m.code;
      $('#btn-stop').hidden = true;
      $('#btn-stop').disabled = false;
      $('#btn-newrun').hidden = false;
      setTimeout(() => { $('#livepill').hidden = true; }, 4000);
    }
  };
  es.onerror = () => { /* stdlib server drops idle streams; EventSource retries */ };
}

/* ------------------------------------------------------------------ boot */
(async function () {
  await loadRuns();
  const pure = state.runs.filter(s => substantive(s) && s.dataset === 'pure');
  const first = pure.sort((a, b) => b.iterations - a.iterations)[0]
    || state.runs.filter(substantive)[0];
  if (first) selectRun(first.id);
  connectStream();
  const st = await fetch('/api/status').then(x => x.json());
  if (st.running) {
    state.live.runId = st.run_id;
    state.live.dataset = st.dataset || 'pure';
    $('#livepill').hidden = false;
    $('#console').textContent = (st.lines || []).join('\n') + '\n';
  }
  window.addEventListener('resize', () => { if (state.run) draw(); });
})();
