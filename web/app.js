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

/* Display names only. The verdict strings themselves are data - they sit in 595
 * iteration records and ledger.py branches on the exact text - so the jargon is
 * translated at the edge rather than renamed at the source. */
const VERDICT_LABEL = {
  'no-op': 'same model',
  noise: 'within noise',
  screen: 'dev screen',
  failed: 'crashed'
};
const verdictLabel = v => VERDICT_LABEL[v] || v || '';

/* Plain-English versions of the harness's own diagnostics. The harness text is
 * written for the AGENT - "check that your change reaches the model that gets
 * saved" is steering, not description - so it stays as it is and this explains
 * the same thing to a person reading the log. */
function plainStatus(rec) {
  if (rec.status === 'no-op') {
    const twin = rec.no_op_twin;
    return 'Different code, same model. This ran, but produced exactly the '
      + 'same predictions as iteration ' + (twin != null ? twin : 'an earlier one')
      + ' - so the change never reached the model that got saved and nothing '
      + 'new was tested. It is not evidence that the idea does not work.';
  }
  if (rec.verdict === 'screen') {
    return 'Scored on the train-only holdout, not on validation. Useful for '
      + 'screening an idea cheaply, but not comparable with the other rows.';
  }
  if (rec.verdict === 'duplicate') {
    return 'The same solution as an earlier iteration, so it was not re-run.';
  }
  if (rec.status === 'cheating') {
    return 'Scored above the oracle ceiling, which means the solution read the '
      + 'label instead of predicting it. Not a result.';
  }
  return null;
}



/* Only events worth interrupting a scroll for. web_search is deliberately out:
   one run logs 171 of them and the rail would be a solid purple line. */
const TICK_KIND = {
  solution_error: 'var(--failed)',
  crash: 'var(--failed)',
  solution_recovered: 'var(--kept)',
  converged: 'var(--accent)',
  budget_stop: 'var(--worse)',
  interrupted: 'var(--worse)'
};

const state = {
  runs: [], run: null, runId: null, cursor: 0, timer: null, animate: false,
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

/* ------------------------------------------------- web_search event sources */
/* The search tool was an API call, not a page, so there is no "result page" to
 * reopen. What the record does hold is the query the agent typed and whatever
 * URLs came back inside the result it read. 37% of the 171 searches across our
 * runs carry at least one URL; the rest carry prose only, and for those the
 * honest offer is to re-run the query rather than to pretend there was a link.
 */
const TRACKING_PARAMS = ['utm_source', 'utm_medium', 'utm_campaign',
                         'utm_term', 'utm_content', 'ref', 'ref_src'];

function cleanUrl(raw) {
  // Log text is agent-written, so treat it as untrusted: anything but http(s)
  // is dropped rather than rendered as a link.
  let t = String(raw || '').replace(/[.,;:!?)\]]+$/, '');
  try {
    const u = new URL(t);
    if (u.protocol !== 'http:' && u.protocol !== 'https:') return null;
    TRACKING_PARAMS.forEach(k => u.searchParams.delete(k));
    return u.toString();
  } catch (e) { return null; }
}

function searchSources(ev) {
  const found = String(ev && ev.result || '')
    .match(/https?:\/\/[^\s<>()\[\]"']+/g) || [];
  const out = [];
  const seen = new Set();
  found.forEach(r => {
    const u = cleanUrl(r);
    if (!u || seen.has(u)) return;
    seen.add(u);
    out.push(u);
  });
  return out;
}

function hostOf(u) {
  try { return new URL(u).host.replace(/^www\./, ''); } catch (e) { return u; }
}

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
  const ds = ($$('.dspill.active')[0] || {}).dataset || {};
  const dsFilter = ds.ds || 'all';
  const list = state.runs.filter(r =>
    (!only || substantive(r)) && (dsFilter === 'all' || r.dataset === dsFilter)
    && !r.name.startsWith('shakedown'));
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
        ${r.noop ? `<span title="different code, same model - nothing tested"
          >${r.noop} same-model</span>` : ''}
        <span>$${r.cost_usd.toFixed(2)}</span>
        ${leak ? '<span class="badge warn">label leak</span>' : ''}
      </div></div>`;
  }).join('') || '<div class="rmeta" style="padding:14px">No runs.</div>';
  $$('#runs .runitem').forEach(el =>
    el.addEventListener('click', () => selectRun(el.dataset.id)));
}
$('#filter-real').addEventListener('change', renderRuns);
$$('.dspill').forEach(b => b.addEventListener('click', () => {
  $$('.dspill').forEach(x => x.classList.toggle('active', x === b));
  renderRuns();
}));

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
  renderVerdicts(data.iterations);
  draw();
}

/* Lay the run out as a search trajectory.
 *
 * X is the iteration, so time runs left to right. Y is the node's RANK among
 * every scored node in the run, best at the top - a branch that climbs found
 * something and a branch that drops did not.
 *
 * Rank rather than the score itself, because the scores do not spread: in
 * record-run-13, 26 of 32 nodes sit between 0.603 and 0.605 while one collapse
 * sits at 0.5735. On a linear axis that outlier stretches the scale and jams
 * the rest into a hairline band. Rank spends the canvas evenly and keeps the
 * ordering, which is what the tree is for; the chart underneath still shows
 * true magnitudes, so nothing is lost between the two.
 *
 * Ranks are computed over the WHOLE run, never over what is currently visible,
 * so a node's position never changes as the replay plays.
 */
function layout(recs, dataset) {
  const byIter = new Map(recs.map(r => [r.iteration, r]));
  const kids = new Map();
  recs.forEach(r => {
    const p = parseInt(r.parent, 10);
    if (Number.isFinite(p) && byIter.has(p) && p !== r.iteration) {
      if (!kids.has(p)) kids.set(p, []);
      kids.get(p).push(r.iteration);
    }
  });

  // A dev screen or a duplicate carries a number that is not comparable with a
  // valid-split score, so it is not ranked - same rule bestSoFar() applies.
  const rankable = r => r.valid_primary != null
    && r.verdict !== 'screen' && r.verdict !== 'duplicate';
  const scored = recs.filter(rankable).sort((a, b) =>
    a.valid_primary - b.valid_primary || a.iteration - b.iteration);
  const rank = new Map();
  scored.forEach((r, i) => rank.set(r.iteration, i));   // 0 = worst
  const nRanked = scored.length;

  const iters = recs.map(r => r.iteration).sort((a, b) => a - b);
  const ix = new Map(iters.map((it, i) => [it, i]));

  const pos = new Map();
  recs.forEach(r => {
    let y;
    if (rank.has(r.iteration)) {
      y = nRanked > 1 ? rank.get(r.iteration) / (nRanked - 1) : 0.5;
    } else {
      // Failed, no-op, screened: nothing moved, so sit level with the parent.
      // That is the honest position - the experiment produced no ranking
      // information - and it makes a dead end read as a flat stub.
      const p = parseInt(r.parent, 10);
      const pr = rank.get(p);
      y = pr != null && nRanked > 1 ? pr / (nRanked - 1) : 0.5;
      pos.set(r.iteration, { ix: ix.get(r.iteration), y, unranked: true });
      return;
    }
    pos.set(r.iteration, { ix: ix.get(r.iteration), y, unranked: false });
  });

  // Where the official FM baseline would sit on this same rank axis, so the
  // vertical position means something absolute and not just "better than the
  // other things this run happened to try".
  let baselineY = null;
  const base = BASELINE[dataset];
  if (base != null && nRanked > 1) {
    let below = 0;
    scored.forEach(r => { if (r.valid_primary < base) below++; });
    baselineY = (below - 0.5) / (nRanked - 1);
    if (baselineY > 1 || baselineY < 0) baselineY = null;
  }

  const best = scored.length ? scored[scored.length - 1] : null;
  const worst = scored.length ? scored[0] : null;
  return { pos, byIter, kids, nRanked, baselineY, best, worst,
           nIters: iters.length };
}

function el(tag, attrs, parent) {
  const n = document.createElementNS(SVGNS, tag);
  // Skip null/undefined: setAttribute would write the string "null" and quietly
  // override the stylesheet with an invalid value.
  for (const k in attrs) {
    if (attrs[k] != null) n.setAttribute(k, attrs[k]);
  }
  if (parent) parent.appendChild(n);
  return n;
}

/* Node glyphs, drawn around the origin so the parent <g> can place them with a
 * translate and the arrival animation can be a transform scale.
 *
 * Shape carries the verdict class as well as colour does, which is what makes
 * the tree readable in greyscale and to a red/green-colourblind reader - the
 * palette leans on exactly that pair.
 */
function nodeGlyph(rec, unranked) {
  // Order matters: a crash is also unranked, so testing `unranked` first meant
  // every failure drew as a hollow ring and the x was unreachable.
  if (rec.status === 'error') return { kind: 'cross', r: 5.6 };
  // Hollow reads as "nothing was measured here", which is exactly what a
  // same-model, a dev screen and a duplicate all are.
  if (rec.verdict === 'no-op' || unranked) return { kind: 'ring', r: 5 };
  if (rec.verdict === 'worse') return { kind: 'down', r: 5.4 };
  if (rec.verdict === 'KEPT') return { kind: 'kept', r: 5 };
  return { kind: 'dot', r: 4.6 };
}

function drawGlyph(parent, glyph, colour) {
  const r = glyph.r;
  if (glyph.kind === 'down') {
    // a wedge pointing down: this branch lost ground
    return el('path', {
      d: `M${-r},${-r * 0.72} L${r},${-r * 0.72} L0,${r * 1.04} Z`,
      fill: colour, class: 'glyph filled'
    }, parent);
  }
  if (glyph.kind === 'cross') {
    // an x: it never produced a number at all
    const a = r * 0.78;
    return el('path', {
      d: `M${-a},${-a} L${a},${a} M${-a},${a} L${a},${-a}`,
      stroke: colour, 'stroke-width': 2.4, 'stroke-linecap': 'round',
      fill: 'none', class: 'glyph stroked'
    }, parent);
  }
  if (glyph.kind === 'ring') {
    // hollow: nothing was measured, so there is nothing to fill in
    return el('circle', {
      r: r, fill: 'var(--bg)', stroke: colour, 'stroke-width': 1.6,
      class: 'glyph stroked ring'
    }, parent);
  }
  if (glyph.kind === 'kept') {
    // a filled core inside its own ring: this one counted
    const g = el('g', { class: 'glyph' }, parent);
    el('circle', { r: r + 2.6, fill: 'none', stroke: colour,
                   'stroke-width': 1.2, opacity: .55, class: 'keptring' }, g);
    el('circle', { r: r, fill: colour, class: 'filled' }, g);
    return g;
  }
  return el('circle', { r: r, fill: colour, class: 'glyph filled' }, parent);
}

/* Incremental, animated tree rendering.
 *
 * The first version cleared the SVG and redrew everything on every tick, which
 * made animation impossible: an element that is destroyed 400ms later cannot
 * draw itself in. So the SVG keeps a registry of the elements it has already
 * built, and a step forward only ADDS the new edge and node. Scrubbing
 * backwards removes what is now in the future; switching runs rebuilds.
 *
 * Layout is still computed over the whole run, so a node's position never
 * depends on how much has been revealed and nothing shifts as it plays.
 */
// COLW is per ITERATION now, not per depth level, so the canvas is as wide as
// the run is long and scrolls. PLOTH is the rank axis's height in px.
const TREE_GEOM = { COLW: 34, PADX: 46, PADY: 22, PLOTH: 300 };

function treeState(svg, key) {
  if (!svg._reg || svg._key !== key) {
    svg.innerHTML = '';
    svg._edgeLayer = el('g', { class: 'edges' }, svg);
    svg._nodeLayer = el('g', { class: 'nodes' }, svg);
    svg._reg = { nodes: new Map(), edges: new Map() };
    svg._key = key;
    svg._fresh = true;
  } else {
    svg._fresh = false;
  }
  return svg._reg;
}

/* Draw the branch from parent to child, then let the node arrive at its end. */
function animateEdge(path, dur) {
  let len;
  try { len = path.getTotalLength(); } catch (e) { return; }
  if (!len) return;
  const prevDash = path.style.strokeDasharray;
  path.style.strokeDasharray = len + ' ' + len;
  const a = path.animate(
    [{ strokeDashoffset: len }, { strokeDashoffset: 0 }],
    { duration: dur, easing: 'cubic-bezier(.45,.05,.25,1)', fill: 'forwards' });
  a.onfinish = () => {
    // Hand the dash pattern back to CSS - a recovery edge is dashed there.
    path.style.strokeDasharray = prevDash || '';
    path.style.strokeDashoffset = '';
  };
}

/* Scale the shape group rather than animating a circle's r. The parent <g>
 * carries the translate, so the shape's own origin is the node centre and a
 * plain transform scales about it - and unlike `r`, transform is animatable
 * everywhere without depending on SVG geometry properties being exposed to CSS. */
function animateNode(g, shape, dur, delay) {
  g.animate([{ opacity: 0 }, { opacity: 0 }, { opacity: 1 }],
    { duration: delay + dur, easing: 'linear', fill: 'backwards' });
  shape.animate(
    [{ transform: 'scale(0)' },
     { transform: 'scale(1.5)' },
     { transform: 'scale(1)' }],
    { duration: dur, delay: delay, easing: 'cubic-bezier(.34,1.56,.64,1)',
      fill: 'backwards' });
}

/* A ring that expands and fades where a node lands - reads as arrival without
 * moving anything that has to stay put. */
function ping(svg, x, y, colour, dur) {
  const c = el('circle', {
    cx: x, cy: y, r: 5, fill: 'none', stroke: colour,
    'stroke-width': 2, class: 'ping'
  }, svg);
  const a = c.animate(
    [{ r: 4, opacity: .85, strokeWidth: 2 },
     { r: 16, opacity: 0, strokeWidth: .5 }],
    { duration: dur, easing: 'ease-out' });
  a.onfinish = () => c.remove();
}

function drawTree(svgSel, recs, cursor, opts) {
  opts = opts || {};
  const svg = $(svgSel);
  if (!recs.length) { svg.innerHTML = ''; svg._key = null; return; }
  const L = layout(recs, opts.dataset || 'pure');
  const { pos, byIter } = L;
  const { COLW: COLW_MIN, PADX, PADY, PLOTH } = TREE_GEOM;
  // Enough vertical room that unique ranks do not collide at r=5, but capped so
  // a long run does not turn the pane into a scroll shaft.
  const plotH = Math.max(150, Math.min(PLOTH, L.nRanked * 14));
  // Fill the container for short runs; scroll for long ones.
  const containerW = svg.parentElement.clientWidth || 480;
  const nSpans = Math.max(1, L.nIters - 1);
  const naturalW = PADX * 2 + nSpans * COLW_MIN + 20;
  const COLW = naturalW < containerW
    ? (containerW - PADX * 2 - 20) / nSpans
    : COLW_MIN;
  const W = PADX * 2 + nSpans * COLW + 20;
  const H = PADY * 2 + plotH;
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  svg.setAttribute('height', H);
  svg.style.minWidth = W + 'px';
  const X = it => PADX + pos.get(it).ix * COLW;
  const Y = it => PADY + (1 - pos.get(it).y) * plotH;

  // Tooltip for the tree pane — anchored to .treepane (non-scrolling) so it
  // stays visible when .treewrap scrolls.
  const treePane = svg.closest('.treepane') || svg.parentElement;
  if (!treePane._tip) {
    const t = document.createElement('div');
    t.className = 'chart-tip';
    treePane.style.position = 'relative';
    treePane.appendChild(t);
    treePane._tip = t;
    treePane._showTip = function(evt, text) {
      t.textContent = text;
      t.classList.add('show');
      const wr = treePane.getBoundingClientRect();
      let tx = evt.clientX - wr.left + 10;
      let ty = evt.clientY - wr.top - 28;
      if (tx + t.offsetWidth > wr.width - 4) tx = evt.clientX - wr.left - t.offsetWidth - 10;
      if (ty < 0) ty = evt.clientY - wr.top + 12;
      t.style.left = tx + 'px';
      t.style.top = ty + 'px';
    };
    treePane._hideTip = function() { t.classList.remove('show'); };
  }
  treePane._tip.classList.remove('show');
  const treeShowTip = treePane._showTip;
  const treeHideTip = treePane._hideTip;

  const key = (opts.key || '') + '|' + recs.length + '|' + Math.round(COLW);
  const reg = treeState(svg, key);
  if (svg._fresh) {
    const g = el('g', { class: 'axis' }, svg);
    svg.insertBefore(g, svg._edgeLayer);
    // Rank is ordinal, so label the two ends with real scores - without them
    // "higher is better" has no scale a reader can calibrate against.
    if (L.best) {
      el('text', { x: 4, y: PADY + 4, class: 'axlabel' }, g)
        .textContent = 'best ' + fmt(L.best.valid_primary, 4);
    }
    if (L.worst && L.worst !== L.best) {
      el('text', { x: 4, y: PADY + plotH + 4, class: 'axlabel' }, g)
        .textContent = fmt(L.worst.valid_primary, 4);
    }
    if (L.baselineY != null) {
      const by = PADY + (1 - L.baselineY) * plotH;
      const baseVal = BASELINE[opts.dataset || 'pure'];
      el('line', { x1: PADX - 14, x2: W - 8, y1: by, y2: by, class: 'baseline' }, g);
      const bHit = el('line', { x1: PADX - 14, x2: W - 8, y1: by, y2: by, class: 'baseline-hit' }, g);
      const bLabel = 'FM baseline: ' + (baseVal != null ? baseVal.toFixed(4) : '?');
      bHit.addEventListener('mouseenter', e => treeShowTip(e, bLabel));
      bHit.addEventListener('mousemove', e => treeShowTip(e, bLabel));
      bHit.addEventListener('mouseleave', treeHideTip);
      el('text', { x: W - 8, y: by - 4, class: 'axlabel', 'text-anchor': 'end' }, g)
        .textContent = 'FM baseline' + (baseVal != null ? ' ' + baseVal.toFixed(4) : '');
    }
  }
  const animate = !!opts.animate && !svg._fresh;
  const dur = opts.dur || 380;
  const edgeMs = Math.round(dur * 0.62);
  const nodeMs = Math.round(dur * 0.45);

  // 1. scrubbed backwards: drop everything now in the future
  reg.nodes.forEach((g, it) => {
    if (it > cursor) { g.remove(); reg.nodes.delete(it); }
  });
  reg.edges.forEach((pth, it) => {
    if (it > cursor) { pth.remove(); reg.edges.delete(it); }
  });

  const best = bestSoFar(recs, cursor);
  const arriving = [];

  // 2. add what is newly visible
  recs.forEach(r => {
    const it = r.iteration;
    if (it > cursor || reg.nodes.has(it)) return;
    const isNew = animate;
    if (isNew) arriving.push(r);

    const p = parseInt(r.parent, 10);
    if (Number.isFinite(p) && pos.has(p) && p <= cursor) {
      const x1 = X(p), y1 = Y(p), x2 = X(it), y2 = Y(it);
      const mx = (x1 + x2) / 2;
      const par = byIter.get(p);
      const recovered = par && (par.status === 'error' || par.status === 'no-op');
      const path = el('path', {
        d: `M${x1},${y1} C${mx},${y1} ${mx},${y2} ${x2},${y2}`,
        class: 'edge' + (recovered ? ' recover' : '')
      }, svg._edgeLayer);
      reg.edges.set(it, path);
      if (isNew) animateEdge(path, edgeMs);
    }

    const unranked = pos.get(it).unranked;
    const colour = VERDICT_COLOR[r.verdict] || 'var(--noise)';
    const glyph = nodeGlyph(r, unranked);

    // translate on the parent, geometry at the origin: the shape can then be
    // scaled, swapped or restyled without recomputing any coordinates
    const g = el('g', {
      class: 'node' + (unranked ? ' unranked' : ''),
      transform: `translate(${X(it)},${Y(it)})`
    }, svg._nodeLayer);
    // A soft disc behind the glyph reads as a glow without an SVG filter, so it
    // costs one more circle instead of a per-node render pass.
    el('circle', { r: glyph.r * 2.5, fill: colour, class: 'halo' }, g);
    const shape = el('g', { class: 'nodeshape' }, g);
    drawGlyph(shape, glyph, colour);

    el('text', { y: -(glyph.r + 5), 'text-anchor': 'middle' }, g).textContent = it;
    el('title', {}, g).textContent =
      `#${it}  ${verdictLabel(r.verdict)}  ${fmt(r.valid_primary, 5)}`;
    g.addEventListener('click', () => { if (opts.onclick) opts.onclick(it); });
    // Hovering a node lights the whole line of descent behind it, which is the
    // question the tree exists to answer: where did this come from?
    const nodeLabel = `#${it}  ${verdictLabel(r.verdict)}  ${fmt(r.valid_primary, 5)}`;
    g.addEventListener('mouseenter', e => { markLineage(svg, reg, byIter, it); treeShowTip(e, nodeLabel); });
    g.addEventListener('mousemove', e => treeShowTip(e, nodeLabel));
    g.addEventListener('mouseleave', () => { clearLineage(svg); treeHideTip(); });
    g._shape = shape;
    g._colour = colour;
    reg.nodes.set(it, g);

    if (isNew) {
      const delay = reg.edges.has(it) ? edgeMs : 0;
      animateNode(g, shape, nodeMs, delay);
      setTimeout(() => {
        if (g.isConnected) ping(svg._nodeLayer, X(it), Y(it), colour, dur);
      }, delay);
    }
  });

  // 3. classes that depend on where the cursor is, not on what exists
  reg.nodes.forEach((g, it) => {
    g.classList.toggle('cur', it === cursor);
    g.classList.toggle('best', !!best && it === best.iteration);
  });
  reg.edges.forEach((pth, it) => pth.classList.toggle('hot', it === cursor));

  // 4. keep the newest node in view as the tree grows past the fold
  if (animate && arriving.length && opts.follow !== false) {
    const wrap = svg.parentElement;
    const last = arriving[arriving.length - 1].iteration;
    if (wrap && wrap.scrollWidth > wrap.clientWidth) {
      const target = (X(last) / W) * wrap.scrollWidth - wrap.clientWidth * 0.6;
      wrap.scrollTo({ left: Math.max(0, target), behavior: 'smooth' });
    }
  }
}

/* Walk parents from a node back to the root and flag that chain. Bounded by the
 * node count, so a malformed parent pointer cannot spin here. */
function markLineage(svg, reg, byIter, from) {
  clearLineage(svg);
  let it = from, guard = 0;
  const seen = new Set();
  while (it != null && !seen.has(it) && guard++ < 500) {
    seen.add(it);
    const node = reg.nodes.get(it);
    if (node) node.classList.add('lineage');
    const edge = reg.edges.get(it);
    if (edge) edge.classList.add('lineage');
    const rec = byIter.get(it);
    const p = rec ? parseInt(rec.parent, 10) : NaN;
    it = Number.isFinite(p) && reg.nodes.has(p) ? p : null;
  }
  svg.classList.add('tracing');
}

function clearLineage(svg) {
  svg.classList.remove('tracing');
  $$('.lineage', svg).forEach(e => e.classList.remove('lineage'));
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

function drawChart(svgSel, recs, cursor, dataset, animate) {
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
  // Tooltip element: one per chartwrap, reused across redraws
  const wrap = svg.parentElement;
  let tip = wrap.querySelector('.chart-tip');
  if (!tip) {
    tip = document.createElement('div');
    tip.className = 'chart-tip';
    wrap.appendChild(tip);
  }
  tip.classList.remove('show');

  function showTip(evt, text) {
    tip.textContent = text;
    tip.classList.add('show');
    const wr = wrap.getBoundingClientRect();
    let tx = evt.clientX - wr.left + 10;
    let ty = evt.clientY - wr.top - 28;
    if (tx + tip.offsetWidth > wr.width - 4) tx = evt.clientX - wr.left - tip.offsetWidth - 10;
    if (ty < 0) ty = evt.clientY - wr.top + 12;
    tip.style.left = tx + 'px';
    tip.style.top = ty + 'px';
  }
  function hideTip() { tip.classList.remove('show'); }

  if (base) {
    el('line', { x1: L, x2: W - R, y1: py(base), y2: py(base), class: 'baseline' }, svg);
    // invisible wide hit area for hover
    const hit = el('line', { x1: L, x2: W - R, y1: py(base), y2: py(base), class: 'baseline-hit' }, svg);
    hit.addEventListener('mouseenter', e => showTip(e, 'FM baseline: ' + base.toFixed(4)));
    hit.addEventListener('mousemove', e => showTip(e, 'FM baseline: ' + base.toFixed(4)));
    hit.addEventListener('mouseleave', hideTip);
    el('text', { x: W - R - 2, y: py(base) - 4, class: 'axlabel', 'text-anchor': 'end' },
      svg).textContent = 'FM baseline';
  }

  const shown = scored.filter(r => r.iteration <= cursor);
  if (shown.length) {
    // On a forward step the last segment is drawn rather than snapped in, so
    // the line advances instead of the whole series flickering to a new shape.
    const growing = animate && shown.length >= 2;
    const solid = growing ? shown.slice(0, -1) : shown;
    const d = solid.map((r, i) =>
      `${i ? 'L' : 'M'}${px(r.iteration)},${py(r.valid_primary)}`).join(' ');
    if (solid.length) el('path', { d, class: 'spark', stroke: 'var(--accent)' }, svg);
    if (growing) {
      const a = shown[shown.length - 2], b2 = shown[shown.length - 1];
      const seg = el('path', {
        d: `M${px(a.iteration)},${py(a.valid_primary)}` +
           `L${px(b2.iteration)},${py(b2.valid_primary)}`,
        class: 'spark', stroke: 'var(--accent)'
      }, svg);
      animateEdge(seg, Math.round((+$('#speed').value || 450) * 0.5));
    }
    // best-so-far as a step line: the number the convergence rule watches
    let b = -Infinity; const pts = [];
    shown.forEach(r => { b = Math.max(b, r.valid_primary); pts.push([r.iteration, b]); });
    let dd = '';
    pts.forEach((p, i) => {
      dd += i ? `L${px(p[0])},${py(pts[i - 1][1])}L${px(p[0])},${py(p[1])}`
        : `M${px(p[0])},${py(p[1])}`;
    });
    el('path', { d: dd, class: 'spark', stroke: 'var(--kept)', 'stroke-dasharray': '3 2', 'stroke-width': 1.2 }, svg);
    shown.forEach(r => {
      const rad = r.iteration === cursor ? 3.6 : 2;
      const colour = VERDICT_COLOR[r.verdict] || 'var(--noise)';
      const dot = el('circle', {
        cx: px(r.iteration), cy: py(r.valid_primary), r: rad,
        fill: colour, class: 'chart-dot'
      }, svg);
      // invisible wider hit area so small dots are easy to hover
      const hitDot = el('circle', {
        cx: px(r.iteration), cy: py(r.valid_primary), r: 8,
        fill: 'transparent', class: 'chart-dot'
      }, svg);
      const label = `#${r.iteration}  ${verdictLabel(r.verdict)}  ${fmt(r.valid_primary, 6)}`;
      hitDot.addEventListener('mouseenter', e => showTip(e, label));
      hitDot.addEventListener('mousemove', e => showTip(e, label));
      hitDot.addEventListener('mouseleave', hideTip);
      if (animate && r.iteration === cursor) {
        dot.animate([{ r: 0 }, { r: rad * 1.8 }, { r: rad }], {
          duration: 420, delay: 120,
          easing: 'cubic-bezier(.34,1.56,.64,1)', fill: 'backwards'
        });
      }
    });
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
    vEl.textContent = verdictLabel(rec.verdict || rec.status);
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

  const plain = plainStatus(rec);
  if (plain) {
    h += `<div class="whybox">${esc(plain)}</div>`;
  }
  if (rec.error) {
    // The harness's own wording is kept verbatim - it is what the agent was
    // told, and that is part of the record.
    h += plain
      ? `<details class="rawerr"><summary>what the harness told the agent</summary>
         <div class="errbox">${esc(rec.error)}</div></details>`
      : `<div class="errbox">${esc(rec.error)}</div>`;
  }

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
  h += `<dt>status</dt><dd>${esc(verdictLabel(rec.status))}</dd>`;
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
  const changed = box._shown !== rec.iteration;
  box._shown = rec.iteration;
  box.innerHTML = h;
  if (changed) {
    box.classList.remove('swap');
    void box.offsetWidth;          // restart the animation
    box.classList.add('swap');
  }
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
  const before = box._count || 0;
  box.innerHTML = shown.map((e, i) => {
    const isSearch = e.kind === 'web_search';
    const srcs = isSearch ? searchSources(e) : [];
    const mark = srcs.length
      ? `<span class="srccount">${srcs.length} source${srcs.length > 1 ? 's' : ''} &#8599;</span>`
      : (isSearch ? '<span class="srccount none">no link</span>' : '');
    const row = `<div class="ev ${esc(e.kind)}${i >= before ? ' new' : ''}` +
      `${isSearch ? ' expandable' : ''}" data-ix="${i}">
      <span class="t">${clock(e.ts)}</span>
      <span class="k">${esc(e.kind)}</span>
      <span class="d" title="${esc(e.detail)}"><span class="dtext">${esc(e.detail)}</span>${mark}</span></div>`;
    if (!isSearch) return row;
    return row + `<div class="evbody" data-body="${i}" hidden>${searchBody(e, srcs)}</div>`;
  }).join('');
  wireSearchRows(box);
  box._count = shown.length;
  box.scrollTop = box.scrollHeight;
  box._colours = shown.map(e => TICK_KIND[e.kind] || null);
  refreshEventTicks();
}

/* The expanded panel: the sources first, then the query, then the text the
 * agent actually read - truncated at 2000 chars by the tool, as recorded. */
function searchBody(ev, srcs) {
  const q = String(ev.detail || '');
  let h = '';
  if (srcs.length) {
    h += '<div class="srcs">' + srcs.map(u =>
      `<a class="src" href="${esc(u)}" target="_blank" rel="noopener noreferrer"
          title="${esc(u)}">${esc(hostOf(u))} &#8599;</a>`).join('') + '</div>';
  }
  h += '<div class="srcs">' +
    `<a class="src rerun" target="_blank" rel="noopener noreferrer"
        href="https://duckduckgo.com/?q=${encodeURIComponent(q)}"
        title="Run this query again - the agent's own search was an API call, ` +
    `not a page">re-run this query &#8599;</a></div>`;
  const res = String(ev.result || '').trim();
  h += res
    ? `<div class="evtext">${esc(res)}</div>`
    : '<div class="evtext none">(no result text recorded)</div>';
  return h;
}

/* Re-measure after a layout change: an expand or collapse moves every row
 * below it, so the rail has to be rebuilt or its marks point at the wrong
 * lines. requestAnimationFrame so the [hidden] toggle has actually applied. */
function refreshEventTicks() {
  const box = $('#events');
  if (!box || !box._colours) return;
  requestAnimationFrame(() => {
    const rows = $$('.ev', box);
    renderTicks('#event-ticks', box._colours, rows);
  });
}

function wireSearchRows(box) {
  $$('.ev.expandable', box).forEach(row => {
    row.addEventListener('click', e => {
      // let a link inside the panel do its own thing
      if (e.target.closest('a')) return;
      const body = box.querySelector(`[data-body="${row.dataset.ix}"]`);
      if (!body) return;
      const open = !body.hidden;
      body.hidden = open;
      row.classList.toggle('open', !open);
      refreshEventTicks();
    });
  });
}

/* Place a mark per notable row at its real position down the scroller.
 *
 * This used to be index/count, which was only right while every row was the
 * same height. An expanded web_search panel breaks that badly, so the position
 * now comes from the row's own offsetTop against the scroll height. Falls back
 * to index/count when the rows have not been laid out yet (a hidden pane
 * measures zero). */
function renderTicks(sel, colours, rows) {
  const box = $(sel);
  if (!box) return;
  const n = colours.length;
  if (!n) { box.innerHTML = ''; box._key = null; return; }
  const total = rows && rows.length === n ? rows[0].parentElement.scrollHeight : 0;
  const want = colours.map((c, i) => {
    if (!c) return null;
    let top;
    if (total > 0 && rows[i]) {
      top = ((rows[i].offsetTop + rows[i].offsetHeight / 2) / total) * 100;
    } else {
      top = ((i + 0.5) / n) * 100;
    }
    return { c, top: Math.max(0, Math.min(100, top)) };
  }).filter(Boolean);
  // Rebuild only when the set changed, so the tick-in animation does not
  // replay on every unrelated redraw.
  const key = want.map(t => t.c + '@' + t.top.toFixed(2)).join('|');
  if (box._key === key) return;
  box._key = key;
  box.innerHTML = want.map(t =>
    `<div class="tick" style="top:${t.top.toFixed(2)}%; background:${t.c}"></div>`
  ).join('');
}

/* One mark per iteration on the scrubber track, coloured by verdict. */
function renderVerdicts(recs) {
  const box = $('#verdicts');
  if (!box) return;
  box.innerHTML = recs.map(r =>
    `<i style="background:${VERDICT_COLOR[r.verdict] || 'var(--noise)'}"
        title="#${r.iteration} ${esc(verdictLabel(r.verdict))}"></i>`
  ).join('');
}

/* ---------------------------------------------------------- resizable panels */
/* Three splitters on the replay screen. Each one only ever writes a CSS custom
 * property on :root, so the layout stays in the stylesheet and no element gets
 * an inline width - which means a reset is one property removal, not a rebuild.
 *
 * Pointer events, not mouse: setPointerCapture keeps the drag alive when the
 * pointer leaves the 7px handle, which it does immediately, and the same code
 * then works for touch and pen without a second path.
 */
const LAYOUT_KEY = 'agent-console-layout';
const LAYOUT_DEFAULTS = {
  '--runlist-w': '268px',
  '--tree-w': '56%',
  '--events-h': '168px'
};

function loadLayout() {
  try { return JSON.parse(localStorage.getItem(LAYOUT_KEY)) || {}; }
  catch (e) { return {}; }          // private window, or site data blocked
}

function saveLayout(patch) {
  try {
    localStorage.setItem(LAYOUT_KEY,
      JSON.stringify(Object.assign(loadLayout(), patch)));
  } catch (e) { /* sizes just do not persist; the session still works */ }
}

function applyLayout() {
  const L = loadLayout();
  Object.keys(LAYOUT_DEFAULTS).forEach(v => {
    if (typeof L[v] === 'string' && L[v]) {
      document.documentElement.style.setProperty(v, L[v]);
    }
  });
}

function makeSplitter(sel, opts) {
  const el = $(sel);
  if (!el) return;
  SPLITTERS.push(opts);
  const root = document.documentElement;
  const axis = opts.axis;
  let origin = 0, startSize = 0, raf = null, active = false;

  const measure = () => {
    const t = opts.target();
    if (!t) return 0;
    const r = t.getBoundingClientRect();
    return axis === 'x' ? r.width : r.height;
  };

  // The chart reads its own container width, so a resize needs a redraw. One
  // per frame at most - a pointermove can fire far more often than that.
  const redraw = () => {
    if (raf) return;
    raf = requestAnimationFrame(() => {
      raf = null;
      if (state.run) draw();
      if (state.live && state.live.records.length) drawLive();
    });
  };

  el.addEventListener('pointerdown', e => {
    e.preventDefault();
    active = true;
    origin = axis === 'x' ? e.clientX : e.clientY;
    startSize = measure();
    el.classList.add('dragging');
    document.body.classList.add(axis === 'x' ? 'resizing-x' : 'resizing-y');
    try { el.setPointerCapture(e.pointerId); } catch (err) { /* no capture */ }
  });

  el.addEventListener('pointermove', e => {
    if (!active) return;
    const now = axis === 'x' ? e.clientX : e.clientY;
    // The run log grows upwards, so its handle moves opposite to its size.
    const delta = (now - origin) * (opts.invert ? -1 : 1);
    const max = opts.max ? opts.max() : Infinity;
    const next = Math.round(
      Math.max(opts.min, Math.min(startSize + delta, Math.max(opts.min, max))));
    root.style.setProperty(opts.varName, next + 'px');
    redraw();
  });

  const end = e => {
    if (!active) return;
    active = false;
    el.classList.remove('dragging');
    document.body.classList.remove('resizing-x', 'resizing-y');
    if (e && e.pointerId != null) {
      try { el.releasePointerCapture(e.pointerId); } catch (err) { /* gone */ }
    }
    saveLayout({ [opts.varName]: root.style.getPropertyValue(opts.varName) });
    if (state.run) draw();
  };
  el.addEventListener('pointerup', end);
  el.addEventListener('pointercancel', end);
  el.addEventListener('lostpointercapture', end);

  // Dragging back to a sensible default is fiddly; double-click just does it.
  el.addEventListener('dblclick', () => {
    root.style.setProperty(opts.varName, LAYOUT_DEFAULTS[opts.varName]);
    saveLayout({ [opts.varName]: LAYOUT_DEFAULTS[opts.varName] });
    if (state.run) draw();
  });
}

const SPLITTERS = [];

/* A max is only true for the window size it was measured in. Shrink the window
 * after a drag and a stored px width can exceed what now fits, so re-apply the
 * bounds on resize rather than letting a pane overflow its grid. */
function reclampLayout() {
  const root = document.documentElement;
  SPLITTERS.forEach(o => {
    const cur = parseFloat(root.style.getPropertyValue(o.varName));
    if (!Number.isFinite(cur)) return;          // still on the % default
    const max = o.max ? o.max() : Infinity;
    const next = Math.max(o.min, Math.min(cur, Math.max(o.min, max)));
    if (Math.round(next) !== Math.round(cur)) {
      root.style.setProperty(o.varName, Math.round(next) + 'px');
    }
  });
}

function initSplitters() {
  applyLayout();
  makeSplitter('#split-runlist', {
    axis: 'x', varName: '--runlist-w', min: 170,
    target: () => $('.runlist'),
    max: () => Math.min(560, window.innerWidth - 480)
  });
  makeSplitter('#split-panes', {
    axis: 'x', varName: '--tree-w', min: 240,
    target: () => $('.treepane'),
    // leave the iteration panel readable: it needs room for a diff
    max: () => ($('.panes') ? $('.panes').clientWidth - 300 : Infinity)
  });
  makeSplitter('#split-events', {
    axis: 'y', varName: '--events-h', min: 56, invert: true,
    target: () => $('.eventpane'),
    // the tree and iteration panes above must keep a usable height
    max: () => ($('.stage') ? $('.stage').clientHeight - 300 : Infinity)
  });
}

function draw() {
  const run = state.run;
  if (!run) return;
  const recs = run.iterations;
  const ds = (state.run.summary && state.run.summary.dataset) || 'pure';
  // Animate only a forward step. Scrubbing, resetting and switching runs
  // should land instantly - an animation you can outrun is just lag.
  const tick = +$('#speed').value;
  drawTree('#tree', recs, state.cursor, {
    key: state.runId,
    dataset: ds,
    animate: state.animate,
    dur: Math.max(160, Math.min(430, tick * 0.62)),
    onclick: i => { stopPlay(); setCursor(i, false); }
  });
  drawChart('#chart', recs, state.cursor, ds, state.animate);
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

function setCursor(i, animate) {
  const next = Math.max(0, Math.min(i, state.run.iterations.length));
  // A single step forward is the only move worth animating.
  state.animate = animate === undefined ? (next === state.cursor + 1) : animate;
  state.cursor = next;
  draw();
  state.animate = false;
}
$('#scrubber').addEventListener('input', e => { stopPlay(); setCursor(+e.target.value, false); });
$('#btn-step').addEventListener('click', () => { stopPlay(); setCursor(state.cursor + 1, true); });
$('#btn-reset').addEventListener('click', () => { stopPlay(); setCursor(0, false); });
$('#btn-play').addEventListener('click', () => {
  if (state.timer) return stopPlay();
  if (state.cursor >= state.run.iterations.length) setCursor(0, false);
  $('#btn-play').classList.add('playing');
  const tick = () => {
    if (state.cursor >= state.run.iterations.length) return stopPlay();
    setCursor(state.cursor + 1, true);
    state.timer = setTimeout(tick, +$('#speed').value);
  };
  state.timer = setTimeout(tick, 120);
});
function stopPlay() {
  if (state.timer) clearTimeout(state.timer);
  state.timer = null;
  $('#btn-play').classList.remove('playing');
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
  state.live = { records: [], events: [], runId: r.run_id, dataset: o.dataset,
                 done: false, lines: 0, ticks: [] };
  $('#console-ticks').innerHTML = '';
  $('#console-ticks')._key = null;
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

function drawConsoleTicks() {
  const box = $('#console-ticks');
  const n = state.live.lines || 0;
  const ticks = state.live.ticks || [];
  if (!box || !n || !ticks.length) return;
  const key = ticks.map(t => t.colour + '@' + t.line).join('|') + '/' + n;
  if (box._key === key) return;
  box._key = key;
  box.innerHTML = ticks.map(t =>
    `<div class="tick" style="top:${((t.line - 0.5) / n * 100).toFixed(2)}%;
      background:${t.colour}"></div>`).join('');
}

function drawLive() {
  const recs = state.live.records;
  const head = Math.max(...recs.map(r => r.iteration), 0);
  drawTree('#live-tree', recs, head, {
    key: state.live.runId, dataset: state.live.dataset,
    animate: true, dur: 420
  });
  drawChart('#live-chart', recs, head, state.live.dataset, true);
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
      state.live.lines = (state.live.lines || 0) + 1;
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
      state.live.lines = (state.live.lines || 0) + 1;
      // The event's own line index is known exactly here, so the console rail
      // needs no pattern-matching over stdout to find the failures.
      const colour = TICK_KIND[m.event.kind];
      if (colour) {
        state.live.ticks = state.live.ticks || [];
        state.live.ticks.push({ line: state.live.lines, colour: colour });
      }
      drawConsoleTicks();
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
  initSplitters();
  connectStream();
  const st = await fetch('/api/status').then(x => x.json());
  if (st.running) {
    state.live.runId = st.run_id;
    state.live.dataset = st.dataset || 'pure';
    $('#livepill').hidden = false;
    $('#console').textContent = (st.lines || []).join('\n') + '\n';
  }
  window.addEventListener('resize', () => {
    reclampLayout();
    if (state.run) draw();
  });
})();
