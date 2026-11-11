// Grader audit UI. All numbers are computed in the browser from data/ by src/audit.js;
// data/results.json holds the Monte Carlo evaluation produced by scripts/build.mjs with the same code.
import * as A from './src/audit.js';

const $ = (s, el = document) => el.querySelector(s);
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const pct = (x, d = 1) => (Number.isFinite(x) ? (100 * x).toFixed(d) + '%' : '–');
const f2 = (x) => (Number.isFinite(x) ? x.toFixed(2) : '–');
const STORE_KEY = 'trajectory-grader-audit-v1';
const CATS = { writing: 'Writing', roleplay: 'Roleplay', reasoning: 'Reasoning', math: 'Math', coding: 'Coding', extraction: 'Extraction', stem: 'STEM', humanities: 'Humanities' };
const MODEL = { 'gpt-4': 'GPT-4', 'gpt-3.5-turbo': 'GPT-3.5', 'claude-v1': 'Claude v1', 'alpaca-13b': 'Alpaca 13B', 'llama-13b': 'LLaMA 13B', 'vicuna-13b-v1.2': 'Vicuna 13B' };
const STATUS = {
  healthy: { label: 'Healthy', cls: 'ok' },
  inconclusive: { label: 'Inconclusive', cls: 'warn' },
  below: { label: 'Below threshold', cls: 'bad' },
  insufficient: { label: 'Needs labels', cls: 'info' },
};

let D = null; // data
let store = loadStore();
const cache = new Map();

function loadStore() {
  try { return Object.assign(defaultStore(), JSON.parse(localStorage.getItem(STORE_KEY) || '{}')); } catch { return defaultStore(); }
}
function defaultStore() { return { settings: null, labels: {}, prefilled: {}, topups: {}, reopt: {}, truth: true }; }
function save() { try { localStorage.setItem(STORE_KEY, JSON.stringify(store)); } catch { /* private mode */ } }
function settings() { return store.settings || { budget: D.results.config.budget, thetaMin: D.results.config.thetaMin, alpha: D.results.config.alpha }; }
function skey(gid) { const s = settings(); return `${gid}|${s.budget}|${s.thetaMin}|${s.alpha}`; }

// ---------------- data + computation ----------------
async function boot() {
  try {
    const [pairs, results] = await Promise.all([fetchJSON('data/mtbench_pairs.json'), fetchJSON('data/results.json')]);
    D = { raw: pairs.items, results, stream: results.stream, views: {}, texts: null };
    for (const g of Object.keys(A.GRADERS)) D.views[g] = A.viewFor(D.raw, g);
    render();
  } catch (e) {
    console.warn(e);
    $('#main').innerHTML = `<div class="error-box"><i data-lucide="cloud-off"></i><h3 style="margin:8px 0 4px">Couldn't load audit data</h3><p class="dim">${esc(e.message)}</p><button class="btn" id="retry"><i data-lucide="rotate-cw"></i>Retry</button></div>`;
    icons();
    $('#retry').onclick = () => { $('#main').innerHTML = ''; boot(); };
  }
}
async function fetchJSON(u) { const r = await fetch(u); if (!r.ok) throw new Error(`${u}: HTTP ${r.status}`); return r.json(); }
async function texts() { if (!D.texts) D.texts = await fetchJSON('data/texts.json'); return D.texts; }

function audit(gid) {
  const key = skey(gid);
  if (cache.has(key)) return cache.get(key);
  const s = settings(), cfg = D.results.config;
  const items = D.views[gid];
  const run = A.runMonitor(items, D.stream, { budget: s.budget, alpha: s.alpha, thetaMin: s.thetaMin, window: cfg.window, tau: cfg.tau, grader: gid, design: 'active', seed: cfg.runSeed });
  const res = { gid, items, run, key };
  cache.set(key, res);
  return res;
}

// Week 8 is the live week: sampled pairs are labelled through the queue (persisted per policy).
function liveWeek(gid) {
  const a = audit(gid);
  const s = settings();
  const W = a.run.weeks.length;
  const wk = a.run.weeks[W - 1];
  const items = wk.ids.map((i) => a.items[i]);
  let pi = wk.pi.slice(), xi = wk.xi.slice();
  const order = shuffled(xi.map((x, k) => (x ? k : -1)).filter((k) => k >= 0), 7700 + gid.length);
  const tops = store.topups[a.key] || [];
  tops.forEach((t, j) => {
    const r = A.topUp(items, pi, xi, wk.pred, wk.thetaPlan, t.extra, A.mulberry32(9100 + j), D.results.config.tau);
    const added = r.xi.map((x, k) => (x && !xi[k] ? k : -1)).filter((k) => k >= 0);
    order.push(...shuffled(added, 9300 + j));
    pi = r.pi; xi = r.xi;
  });
  const labels = (store.labels[a.key] ||= {});
  if (!store.prefilled[a.key]) {
    // demo starting point: the first 60% of the queue already carry the recorded LMSYS expert vote
    const base = order.slice(0, Math.ceil(0.6 * wk.xi.reduce((t, x) => t + x, 0)));
    for (const k of base) labels[items[k].idx] = items[k].expert;
    store.prefilled[a.key] = true; save();
  }
  const S = order.length;
  const lab = order.filter((k) => labels[items[k].idx]);
  const L = lab.length;
  const xiL = items.map(() => 0); lab.forEach((k) => (xiL[k] = 1));
  const labArr = items.map((it, k) => (xiL[k] ? labels[it.idx] : null));
  const piL = pi.map((p) => (S ? (p * L) / S : p));
  const est = A.estimate(items, piL, xiL, labArr, wk.preds, s.alpha);
  // pooled window with the previous (complete) week
  const prev = a.run.weeks[W - 2];
  const prevItems = prev.ids.map((i) => a.items[i]);
  const pooled = A.estimate([...prevItems, ...items], [...prev.pi, ...piL], [...prev.xi, ...xiL], [...prev.labels, ...labArr], [...prev.preds, ...wk.preds], s.alpha);
  const status = A.status(pooled, s.thetaMin, s.alpha / 2);
  const toDecide = Math.min(A.labelsToDecide(pooled, s.thetaMin, s.alpha / 2), items.length - S);
  return { a, wk, items, pi, xi, order, labels, S, L, est, pooled, status, toDecide, truth: A.trueAccuracy(items), week: W };
}

function shuffled(arr, seed) {
  const r = A.mulberry32(seed), a = arr.slice();
  for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; }
  return a;
}

function weeksFor(gid) {
  const a = audit(gid), lw = liveWeek(gid);
  const ws = a.run.weeks.slice(0, -1).map((w) => ({ week: w.week, n: w.n, theta: w.theta, lo: w.lo, hi: w.hi, nLab: w.nLab, truth: w.truth, status: w.status, win: w.win, provisional: false }));
  ws.push({ week: lw.week, n: lw.items.length, theta: lw.est.theta, lo: lw.est.lo, hi: lw.est.hi, nLab: lw.L, truth: lw.truth, status: lw.status, win: lw.pooled, provisional: lw.L < lw.S });
  return ws;
}

// ---------------- routing ----------------
function parseHash() {
  const h = location.hash.replace(/^#/, '') || '/graders/scalar';
  const [path, qs] = h.split('?');
  const parts = path.split('/').filter(Boolean);
  return { parts, q: new URLSearchParams(qs || '') };
}
function go(path, q = {}) {
  const qs = new URLSearchParams(Object.entries(q).filter(([, v]) => v !== null && v !== undefined && v !== '')).toString();
  const h = '#' + path + (qs ? '?' + qs : '');
  if (location.hash !== h) location.hash = h; else render();
}
window.addEventListener('hashchange', () => render());

function render() {
  if (!D) return;
  const { parts, q } = parseHash();
  document.querySelectorAll('.nav-item').forEach((n) => n.classList.toggle('active', n.getAttribute('href') === '#/' + (parts[0] || 'graders')));
  const main = $('#main');
  if (parts[0] !== 'graders') {
    const name = { overview: 'Overview', signals: 'Signals', datasets: 'Datasets', training: 'Training runs', evaluations: 'Evaluations', deployments: 'Deployments', settings: 'Settings' }[parts[0]] || 'Not found';
    main.innerHTML = `<div class="page"><div class="crumbs">General assistant / ${esc(name)}</div><h1 class="title">${esc(name)}</h1>
      <div class="card" style="margin-top:18px"><div class="empty"><i data-lucide="scale"></i><h3>Not part of this concept demo</h3><div>The demo adds one feature to the Graders page.</div><a class="btn" style="margin-top:12px" href="#/graders">Open Graders</a></div></div></div>`;
    closeDrawer(); icons(); return;
  }
  const gid = parts[1];
  if (!gid) { renderList(main); closeDrawer(); icons(); return; }
  if (!A.GRADERS[gid]) { main.innerHTML = `<div class="page"><div class="card" style="margin-top:18px"><div class="empty"><i data-lucide="search-x"></i><h3>Grader not found</h3><a class="btn" href="#/graders">Back to Graders</a></div></div></div>`; icons(); return; }
  const tab = parts[2] || 'audit';
  renderGrader(main, gid, tab, q);
  const pair = q.get('pair');
  if (tab === 'queue' && pair !== null) openDrawer(gid, Number(pair), q); else closeDrawer();
  icons();
}

// ---------------- graders list ----------------
function renderList(main) {
  const s = settings();
  const rows = Object.values(A.GRADERS).map((g) => {
    const lw = liveWeek(g.id);
    const st = store.reopt[g.id] ? { label: 'Re-optimizing', cls: 'info' } : STATUS[lw.status];
    const ev = D.results.evaluation[g.id];
    return `<tr class="click" data-go="#/graders/${g.id}">
      <td><div style="font-weight:600">${esc(g.name)}</div><div class="dim" style="font-size:12px">${esc(g.judge)}</div></td>
      <td><span class="chip">${esc(g.kind)}</span></td>
      <td class="num">${pct(ev.seedTruth)}</td>
      <td>${ciCell(lw.pooled, s.thetaMin)}</td>
      <td class="num">${lw.L} / ${lw.S}</td>
      <td><span class="chip ${st.cls}"><span class="d"></span>${st.label}</span></td>
      <td class="r"><i data-lucide="chevron-right" class="dim"></i></td></tr>`;
  }).join('');
  main.innerHTML = `<div class="page">
    <div class="page-head"><div><div class="crumbs">General assistant / Graders</div><h1 class="title">Graders</h1>
    <div class="subtitle">Graders score production trajectories so the model can train on them. Each grader is audited weekly against a small, actively chosen set of expert labels.</div></div>
    <div class="head-actions"><a class="btn" href="https://www.trajectory.ai/field-notes/continual-grading" target="_blank" rel="noopener"><i data-lucide="book-open"></i>Continual Grading</a><button class="btn dark" disabled title="Not part of this demo"><i data-lucide="plus"></i>New grader</button></div></div>
    <div class="stats">${(() => { const lws = Object.keys(A.GRADERS).map((g) => liveWeek(g)); const L = lws.reduce((t, l) => t + l.L, 0), S = lws.reduce((t, l) => t + l.S, 0); const N = lws[0].items.length; return `<div class="stat"><span class="l">Production pairs graded, week ${lws[0].week}</span><span class="v">${N.toLocaleString()}</span></div><div class="stat"><span class="l">Expert labels this week</span><span class="v">${L} <span style="color:#a39d94;font-weight:400">/ ${S}</span></span></div><div class="stat"><span class="l">Graders needing attention</span><span class="v">${lws.filter((l) => l.status !== 'healthy').length} <span style="color:#a39d94;font-weight:400">/ ${lws.length}</span></span></div>`; })()}</div>
    <div class="card" style="margin-top:12px"><div class="card-head"><div class="card-title">All graders</div><span class="card-sub">Audited weekly · threshold ${pct(s.thetaMin, 0)}</span></div><div class="table-wrap"><table class="t">
    <thead><tr><th>Grader</th><th>Type</th><th>Selection accuracy</th><th>Ranking accuracy, weeks ${D.stream.weeks.length - 1}–${D.stream.weeks.length} (90% CI)</th><th>Expert labels, week ${D.stream.weeks.length}</th><th>Status</th><th></th></tr></thead>
    <tbody>${rows}</tbody></table></div></div>
    <p class="note" style="margin-top:12px">Ranking accuracy is the share of expert-ranked response pairs the grader orders the same way (expert ties excluded), estimated on this agent's production pairs. Threshold ${pct(s.thetaMin, 0)}.</p></div>`;
  main.querySelectorAll('[data-go]').forEach((tr) => (tr.onclick = () => (location.hash = tr.dataset.go)));
}

function ciCell(e, th) {
  const lo = 0.4, hi = 1, x = (v) => `${(100 * (v - lo)) / (hi - lo)}%`;
  return `<span class="ci"><span class="num" style="width:44px">${pct(e.theta)}</span><span class="ci-bar" aria-label="90% interval ${pct(e.lo)} to ${pct(e.hi)}"><i style="left:${x(e.lo)};width:calc(${x(e.hi)} - ${x(e.lo)})"></i><b style="left:calc(${x(e.theta)} - 1px)"></b><em style="left:${x(th)}"></em></span><span class="dim num">${pct(e.lo, 0)}–${pct(e.hi, 0)}</span></span>`;
}

// ---------------- grader page ----------------
function renderGrader(main, gid, tab, q) {
  const g = A.GRADERS[gid];
  const lw = liveWeek(gid);
  const pending = lw.S - lw.L;
  const re = store.reopt[gid];
  const st = re ? { label: 'Re-optimizing', cls: 'info' } : STATUS[lw.status];
  main.innerHTML = `<div class="page">
    <div class="page-head"><div>
      <div class="crumbs"><a href="#/graders">General assistant</a><span>/</span><a href="#/graders">Graders</a><span>/</span><span>${esc(g.name)}</span></div>
      <h1 class="title">${esc(g.name)} <span class="chip ${st.cls}"><span class="d"></span>${st.label}</span> <span class="chip preview">Research preview</span></h1>
      <div class="subtitle">${esc(g.judge)} · selected by best-of-N on ${D.results.config.seedSize} expert-labeled pairs · grades the General assistant's production traffic</div>
    </div>
    <div class="head-actions">
      <div class="export-wrap" style="position:relative"><button class="btn" id="export-btn" aria-haspopup="menu"><i data-lucide="download"></i>Export</button></div>
      ${re ? `<button class="btn" id="cancel-reopt"><i data-lucide="x"></i>Cancel re-optimization</button>` : `<button class="btn dark" id="reopt-btn" ${lw.status === 'below' ? '' : 'disabled title="Enabled when accuracy is confidently below the threshold"'}><i data-lucide="refresh-cw"></i>Re-optimize grader</button>`}
    </div></div>
    <div class="tabs" role="tablist">
      <a class="tab ${tab === 'audit' ? 'active' : ''}" href="#/graders/${gid}" role="tab"><i data-lucide="gauge"></i>Audit <span class="chip new" style="padding:0 6px;font-size:10.5px">New</span></a>
      <a class="tab ${tab === 'queue' ? 'active' : ''}" href="#/graders/${gid}/queue" role="tab"><i data-lucide="list-todo"></i>Label queue <span class="count">${pending}</span></a>
      <a class="tab ${tab === 'program' ? 'active' : ''}" href="#/graders/${gid}/program" role="tab"><i data-lucide="file-code-2"></i>Program</a>
    </div>
    <div id="tab-body"></div></div>`;
  const body = $('#tab-body');
  if (tab === 'audit') renderAudit(body, gid, lw, q);
  else if (tab === 'queue') renderQueue(body, gid, lw, q);
  else if (tab === 'program') renderProgram(body, gid);
  else body.innerHTML = `<div class="empty"><h3>Unknown tab</h3></div>`;
  $('#export-btn').onclick = (e) => exportMenu(e, gid);
  const rb = $('#reopt-btn'); if (rb) rb.onclick = () => reoptDialog(gid, lw);
  const cb = $('#cancel-reopt'); if (cb) cb.onclick = () => { delete store.reopt[gid]; save(); toast('Re-optimization cancelled'); render(); };
}

function renderAudit(body, gid, lw, q) {
  const s = settings(), ev = D.results.evaluation[gid];
  const ws = weeksFor(gid);
  const showTruth = q.get('truth') === '1' || (q.get('truth') !== '0' && store.truth);
  const sel = Math.min(ws.length, Math.max(1, Number(q.get('week')) || ws.length));
  const w = ws[sel - 1];
  const live = sel === ws.length;
  const status = STATUS[w.status];
  const defaults = s.budget === D.results.config.budget && s.thetaMin === D.results.config.thetaMin && s.alpha === D.results.config.alpha;
  const savings = defaults ? ev.labelSavings : null;
  body.innerHTML = `
  <div class="strip">
    <div class="cell peri"><div class="cell-label">Ranking accuracy · week ${sel}${w.provisional ? ' (provisional)' : ''}</div>
      <div class="cell-value">${(100 * w.theta).toFixed(1)}<small>%</small></div><div class="cell-foot num">90% CI ${pct(w.lo)} – ${pct(w.hi)}</div></div>
    <div class="cell peach"><div class="cell-label">Selection accuracy</div><div class="cell-value">${(100 * ev.seedTruth).toFixed(1)}<small>%</small></div>
      <div class="cell-foot">on ${D.results.config.seedSize} pairs at selection time</div></div>
    <div class="cell"><div class="cell-label">Expert labels</div><div class="cell-value num">${w.nLab}<small style="color:#9a948b"> / ${live ? lw.S : w.nLab}</small></div>
      <div class="cell-foot dim">of ${w.n} production pairs ${live ? 'this week' : `in week ${sel}`}</div></div>
    <div class="cell"><div class="cell-label">Status vs ${pct(s.thetaMin, 0)} threshold</div>
      <div style="margin-top:10px"><span class="chip ${status.cls}" style="font-size:13px;padding:3px 10px"><span class="d"></span>${status.label}</span></div>
      <div class="cell-foot dim" style="margin-top:8px">${w.week > 1 ? `weeks ${w.week - 1}–${w.week} pooled: ${pct(w.win.theta)} (${pct(w.win.lo, 0)}–${pct(w.win.hi, 0)})` : `week 1: ${pct(w.win.theta)}`}</div></div>
  </div>
  <div class="grid-2">
    <div class="stack">
      <div class="card">
        <div class="card-head"><div><div class="card-title">Ranking accuracy on production traffic</div><div class="card-sub">Weekly estimate from expert labels, with 90% confidence band. Click a week to inspect it.</div></div>
          <label class="toggle"><input type="checkbox" id="truth-t" ${showTruth ? 'checked' : ''}>Show hindsight truth</label></div>
        <div class="card-body" style="padding-top:8px"><div class="chart-wrap"><div class="chart" id="chart"></div></div>
          <div class="legend" style="margin-top:6px"><span><i class="sw" style="background:var(--periwinkle)"></i>Estimate + 90% CI</span><span><i class="sw" style="background:#fff;border:1.5px dashed #c5372f;height:0;width:14px;border-radius:0"></i>Threshold ${pct(s.thetaMin, 0)}</span>${showTruth ? '<span><i class="sw" style="background:var(--peach)"></i>Hindsight truth (all recorded expert votes; demo only)</span>' : ''}<span><i class="sw" style="background:#e9e6f9"></i>Traffic mix shift</span></div>
        </div>
      </div>
      ${live ? decisionCard(gid, lw) : weekNote(w)}
      <div class="card">
        <div class="card-head"><div><div class="card-title">Where this week's expert labels go</div><div class="card-sub">Inclusion probability by grader confidence. Labels concentrate where the grader's agreement with experts is least predictable.</div></div><a class="btn" href="#/graders/${gid}/queue"><i data-lucide="list-todo"></i>Open queue</a></div>
        <div class="card-body">${allocationBars(gid, lw)}</div>
      </div>
    </div>
    <div class="stack">
      <div class="card"><div class="card-head"><div class="card-title"><i data-lucide="flask-conical"></i>Why fewer labels</div></div>
        <div class="card-body">
          ${savings != null ? `<div style="font-size:30px;letter-spacing:-.04em;line-height:1">${(100 * savings).toFixed(0)}<span style="font-size:17px">% fewer</span></div><div class="note" style="margin-top:4px">expert labels than random sampling for the same interval width: <b>${s.budget}/week</b> here vs <b>${ev.equivalentUniformLabels.toFixed(0)}/week</b> chosen at random.</div>` : `<div class="note">Monte Carlo numbers below are recomputed for your policy.</div>`}
          <dl class="kv" style="margin-top:12px" id="mc-kv">${mcRows(ev)}</dl>
          <p class="note" style="margin-top:10px">${D.results.evaluation[gid].active ? `${defaults ? D.results.config.reps : 200} simulated labeling runs on the same production weeks. The estimator stays unbiased no matter how good the predictor is; a better predictor only narrows the interval.` : ''}</p>
        </div></div>
      ${policyCard(gid)}
      <div class="card"><div class="card-head"><div class="card-title"><i data-lucide="info"></i>How the audit works</div></div><div class="card-body note">
        Each week the audit predicts, from the grader's own output (here: ${gid === 'scalar' ? 'the score margin' : 'whether its verdict was order-consistent'}), how likely it is to agree with an expert. It samples pairs for expert review with probability proportional to the residual uncertainty, then combines the prediction with the labeled residuals (a difference estimator, weighted by 1/inclusion probability). The result is an unbiased estimate of this week's ranking accuracy with a valid confidence interval, from ${s.budget} labels instead of all ${w.n}.
      </div></div>
    </div>
  </div>`;
  drawChart($('#chart'), ws, sel, showTruth, s.thetaMin, gid);
  $('#truth-t').onchange = (e) => { store.truth = e.target.checked; save(); go(`/graders/${gid}`, { week: q.get('week'), truth: e.target.checked ? '' : '0' }); };
  wirePolicy(gid);
  const tb = $('#topup-btn'); if (tb) tb.onclick = () => doTopUp(gid, Number(tb.dataset.n));
  const rb2 = $('#reopt-inline'); if (rb2) rb2.onclick = () => reoptDialog(gid, lw);
}

function mcRows(ev) {
  return `<dt>Avg 90% CI width, active</dt><dd>${pct(ev.active.width)}</dd>
    <dt>Avg 90% CI width, random labels</dt><dd>${pct(ev.uniformAtBudget.width)}</dd>
    <dt>Random labels + difference estimator</dt><dd>${ev.uniformModelAssisted ? pct(ev.uniformModelAssisted.width) : '–'}</dd>
    <dt>Coverage of 90% CI, active</dt><dd>${pct(ev.active.coverage)}</dd>
    <dt>Coverage of 90% CI, random labels</dt><dd>${pct(ev.uniformAtBudget.coverage)}</dd>
    <dt>RMSE, active vs random</dt><dd>${pct(ev.active.rmse)} vs ${pct(ev.uniformAtBudget.rmse)}</dd>`;
}

function weekNote(w) {
  return `<div class="callout"><i data-lucide="history"></i><div class="note">Week ${w.week} is closed: ${w.nLab} expert labels, estimate ${pct(w.theta)} (${pct(w.lo)}–${pct(w.hi)}). <a href="#/graders/${parseHash().parts[1]}" style="color:var(--link)">Back to current week</a></div></div>`;
}

function decisionCard(gid, lw) {
  const s = settings();
  const pending = lw.S - lw.L;
  if (store.reopt[gid]) {
    const r = store.reopt[gid];
    return `<div class="callout ok"><i data-lucide="refresh-cw"></i><div class="note"><b>Re-optimization queued</b> ${new Date(r.at).toLocaleString()} with ${r.labels} expert labels (${r.selection} held out for selection). The new metagrader will be audited from its first production week.</div></div>`;
  }
  if (lw.status === 'below') {
    return `<div class="callout bad"><i data-lucide="triangle-alert"></i><div class="note" style="flex:1"><b>Ranking accuracy is below ${pct(s.thetaMin, 0)} with 95% confidence</b> (weeks ${lw.week - 1}–${lw.week} pooled: ${pct(lw.pooled.theta)}, upper bound ${pct(lw.pooled.theta + A.tQuantile(1 - s.alpha / 2, lw.pooled.nLab - 1) * lw.pooled.se)}). Rewards from this grader are likely steering training on the wrong pairs. Re-run the metagrader search on the labels collected since selection.</div><button class="btn dark" id="reopt-inline"><i data-lucide="refresh-cw"></i>Re-optimize</button></div>`;
  }
  if (lw.status === 'healthy') return `<div class="callout ok"><i data-lucide="circle-check"></i><div class="note"><b>Above threshold with 95% confidence.</b> No action needed this week.</div></div>`;
  const n = lw.toDecide;
  const can = Number.isFinite(n) && n > 0;
  return `<div class="callout"><i data-lucide="scale"></i><div class="note" style="flex:1"><b>Not yet decided.</b> Pooled estimate ${pct(lw.pooled.theta)} (${pct(lw.pooled.lo)}–${pct(lw.pooled.hi)}) still straddles ${pct(s.thetaMin, 0)}.
    ${pending ? `${pending} sampled pair${pending === 1 ? '' : 's'} still need an expert label.` : ''} ${Number.isFinite(n) ? `About <b>${n} more label${n === 1 ? '' : 's'}</b> at this design would settle it either way.` : 'The estimate sits on the threshold; more labels alone may not settle it this week.'}</div>
    ${pending ? `<a class="btn" href="#/graders/${gid}/queue?filter=pending"><i data-lucide="pen-line"></i>Label ${pending}</a>` : ''}
    ${can ? `<button class="btn dark" id="topup-btn" data-n="${n}"><i data-lucide="plus"></i>Top up ${n}</button>` : ''}</div>`;
}

function allocationBars(gid, lw) {
  const g = A.GRADERS[gid];
  const rows = g.buckets.map((name, k) => {
    const idx = lw.items.map((it, j) => (it.k === k ? j : -1)).filter((j) => j >= 0);
    if (!idx.length) return '';
    const share = idx.length / lw.items.length;
    const meanPi = idx.reduce((t, j) => t + lw.pi[j], 0) / idx.length;
    const p = A.predict(lw.wk.pred, lw.items[idx[0]]);
    return { name, share, meanPi, pG: p.pG, pD: p.pD, n: idx.length };
  }).filter(Boolean);
  const max = Math.max(...rows.map((r) => r.meanPi));
  const colors = ['var(--peach)', 'var(--periwinkle)', 'var(--sage)', 'var(--periwinkle)', 'var(--sage)'];
  return `<div class="bars">${rows.map((r, i) => `<div><div class="bar-row-head"><span class="bar-name">${esc(r.name)}</span><span class="bar-meta">${r.n} pairs (${pct(r.share, 0)}) · predicted agreement ${pct(r.pG, 0)} · expert tie ${pct(1 - r.pD, 0)}</span></div>
    <div class="bar-track"><div class="bar-fill" style="width:${Math.max(8, (92 * r.meanPi) / max)}%;background:${colors[i % colors.length]}">${(100 * r.meanPi).toFixed(1)}%</div></div></div>`).join('')}
    <div class="bar-axis"><span>Chance a pair is sent to an expert</span><span>uniform would be ${pct(settings().budget / lw.items.length)} everywhere</span></div></div>`;
}

function policyCard(gid) {
  const s = settings();
  return `<div class="card"><div class="card-head"><div class="card-title"><i data-lucide="sliders-horizontal"></i>Audit policy</div><span class="card-sub">applies to all graders</span></div><div class="card-body">
    <div class="field"><span class="lab">Expert labels per week</span><div class="seg" role="group" aria-label="Expert labels per week">${[20, 40, 60, 80].map((b) => `<button type="button" data-b="${b}" class="${b === s.budget ? 'on' : ''}">${b}</button>`).join('')}</div></div>
    <div class="field"><label for="th-in">Re-optimize below (ranking accuracy)</label><input class="input" id="th-in" type="number" min="0.5" max="0.95" step="0.01" value="${s.thetaMin}"><span class="hint">Selection-time accuracy was ${pct(D.results.evaluation[gid].seedTruth)}.</span></div>
    <div class="field"><span class="lab">Confidence level</span><div class="seg" role="group" aria-label="Confidence level">${[0.2, 0.1, 0.05].map((a) => `<button type="button" data-a="${a}" class="${a === s.alpha ? 'on' : ''}">${100 - 100 * a}%</button>`).join('')}</div></div>
    <div style="display:flex;gap:8px;align-items:center"><button class="btn" id="pol-apply" disabled><i data-lucide="check"></i>Apply policy</button><button class="btn ghost" id="pol-reset">Reset</button><span class="dim" id="pol-msg" style="font-size:12px"></span></div>
  </div></div>`;
}

let draft = null;
function wirePolicy(gid) {
  const s = settings();
  draft = { ...s };
  const upd = () => {
    const changed = draft.budget !== s.budget || draft.thetaMin !== s.thetaMin || draft.alpha !== s.alpha;
    $('#pol-apply').disabled = !changed || !(draft.thetaMin >= 0.5 && draft.thetaMin <= 0.95);
    $('#pol-msg').textContent = !(draft.thetaMin >= 0.5 && draft.thetaMin <= 0.95) ? 'Threshold must be between 0.50 and 0.95' : changed ? 'Re-runs the audit for weeks 1–8' : '';
  };
  document.querySelectorAll('[data-b]').forEach((b) => (b.onclick = () => { draft.budget = Number(b.dataset.b); document.querySelectorAll('[data-b]').forEach((x) => x.classList.toggle('on', x === b)); upd(); }));
  document.querySelectorAll('[data-a]').forEach((b) => (b.onclick = () => { draft.alpha = Number(b.dataset.a); document.querySelectorAll('[data-a]').forEach((x) => x.classList.toggle('on', x === b)); upd(); }));
  $('#th-in').oninput = (e) => { draft.thetaMin = Math.round(Number(e.target.value) * 100) / 100; upd(); };
  $('#pol-apply').onclick = () => applyPolicy(gid, draft);
  $('#pol-reset').onclick = () => { store.settings = null; save(); cache.clear(); toast('Policy reset to defaults'); render(); };
  const isDefault = s.budget === D.results.config.budget && s.thetaMin === D.results.config.thetaMin && s.alpha === D.results.config.alpha;
  if (!isDefault) recomputeMC(gid);
}

function applyPolicy(gid, d) {
  $('#pol-apply').disabled = true; $('#pol-msg').textContent = 'Re-running audit…';
  setTimeout(() => { store.settings = { ...d }; save(); cache.clear(); toast(`Policy applied: ${d.budget} labels/week, threshold ${pct(d.thetaMin, 0)}`); render(); }, 30);
}

function recomputeMC(gid) {
  const kv = $('#mc-kv'); if (!kv) return;
  kv.innerHTML = `<dt>Simulating 200 labeling runs…</dt><dd><span class="sk" style="display:inline-block;width:60px;height:12px"></span></dd>`;
  setTimeout(() => {
    const s = settings(), cfg = D.results.config, items = D.views[gid];
    const base = { budget: s.budget, alpha: s.alpha, thetaMin: s.thetaMin, window: cfg.window, tau: cfg.tau, grader: gid };
    const mean = (a) => a.reduce((t, x) => t + x, 0) / a.length;
    const act = A.monteCarlo(items, D.stream, { ...base, design: 'active' }, 200);
    const uni = A.monteCarlo(items, D.stream, { ...base, design: 'uniform' }, 200);
    const ma = A.monteCarlo(items, D.stream, { ...base, design: 'uniform-ma' }, 200);
    const ev = { uniformModelAssisted: { width: mean(ma.width) }, active: { width: mean(act.width), coverage: mean(act.coverage), rmse: mean(act.rmse) }, uniformAtBudget: { width: mean(uni.width), coverage: mean(uni.coverage), rmse: mean(uni.rmse) } };
    if ($('#mc-kv')) $('#mc-kv').innerHTML = mcRows(ev);
  }, 20);
}

function doTopUp(gid, n) {
  const a = audit(gid);
  (store.topups[a.key] ||= []).push({ extra: n, at: Date.now() });
  save();
  const lw = liveWeek(gid);
  toast(`Added ${lw.S - lw.L} pairs to the label queue`);
  go(`/graders/${gid}/queue`, { filter: 'pending' });
}

function reoptDialog(gid, lw) {
  const a = audit(gid);
  const prior = a.run.weeks.slice(0, -1).reduce((t, w) => t + w.nLab, 0);
  const total = D.results.config.seedSize + prior + lw.L;
  const sel = Math.round(total * 0.3);
  modal(`<div class="card-head"><h2>Re-optimize ${esc(A.GRADERS[gid].name)}</h2></div><div class="card-body">
    <p class="note" style="margin:0">Runs the metagrader search again on every expert label collected since selection, then promotes the candidate with the highest held-out ranking accuracy. Training runs keep using the current grader until the new one is promoted.</p>
    <dl class="kv"><dt>Expert-labeled pairs</dt><dd>${total}</dd><dt>Construction split</dt><dd>${total - sel}</dd><dt>Selection split (held out)</dt><dd>${sel}</dd><dt>Weeks 7–8 traffic share, open-ended tasks</dt><dd>${pct(openShare(lw), 0)}</dd></dl>
  </div><div class="modal-foot"><button class="btn" data-close>Cancel</button><button class="btn dark" id="reopt-go"><i data-lucide="refresh-cw"></i>Start search</button></div>`);
  $('#reopt-go').onclick = () => { store.reopt[gid] = { at: Date.now(), labels: total, selection: sel }; save(); closeModal(); toast('Metagrader search queued'); render(); };
}
function openShare(lw) { const o = new Set(['writing', 'roleplay', 'reasoning', 'humanities']); return lw.items.filter((i) => o.has(i.cat)).length / lw.items.length; }

// ---------------- chart ----------------
function drawChart(el, ws, sel, showTruth, th, gid) {
  const W = Math.max(520, el.clientWidth || 760), H = 300, m = { l: 40, r: 120, t: 14, b: 30 };
  const iw = W - m.l - m.r, ih = H - m.t - m.b;
  const ymin = 0.5, ymax = 0.95;
  const x = (wk) => m.l + ((wk - 1) / (ws.length - 1)) * iw;
  const y = (v) => m.t + (1 - (Math.min(ymax, Math.max(ymin, v)) - ymin) / (ymax - ymin)) * ih;
  const cfg = D.results.config;
  // Catmull-Rom to cubic Bezier (tension 0.5), like the training-reward curves in the app
  const smooth = (pts, cont = false) => pts.map((p, i) => {
    if (i === 0) return `${cont ? 'L' : 'M'}${p[0].toFixed(1)},${p[1].toFixed(1)}`;
    const p0 = pts[i - 2] || pts[i - 1], p1 = pts[i - 1], p3 = pts[i + 1] || p;
    const c1 = [p1[0] + (p[0] - p0[0]) / 6, p1[1] + (p[1] - p0[1]) / 6], c2 = [p[0] - (p3[0] - p1[0]) / 6, p[1] - (p3[1] - p1[1]) / 6];
    return `C${c1[0].toFixed(1)},${c1[1].toFixed(1)} ${c2[0].toFixed(1)},${c2[1].toFixed(1)} ${p[0].toFixed(1)},${p[1].toFixed(1)}`;
  }).join('');
  const band = smooth(ws.map((w) => [x(w.week), y(w.hi)])) + smooth(ws.slice().reverse().map((w) => [x(w.week), y(w.lo)]), true) + 'Z';
  const line = smooth(ws.map((w) => [x(w.week), y(w.theta)]));
  const tline = smooth(ws.map((w) => [x(w.week), y(w.truth)]));
  const grid = [0.5, 0.6, 0.7, 0.8, 0.9].map((v) => `<line x1="${m.l}" x2="${m.l + iw}" y1="${y(v)}" y2="${y(v)}" stroke="#ece8e1"/><text x="${m.l - 8}" y="${y(v) + 4}" text-anchor="end" fill="#8a847b" font-size="11">${v.toFixed(1)}</text>`).join('');
  const xt = ws.map((w) => `<text x="${x(w.week)}" y="${H - 8}" text-anchor="middle" fill="${w.week === sel ? '#34312d' : '#8a847b'}" font-size="11" ${w.week === sel ? 'font-weight="600"' : ''}>${w.week === ws.length ? 'Week ' + w.week : w.week}</text>`).join('');
  const shift = `<rect x="${x(cfg.shiftFrom)}" y="${m.t}" width="${x(cfg.shiftTo) - x(cfg.shiftFrom)}" height="${ih}" fill="#e9e6f9" opacity=".55"/><text x="${(x(cfg.shiftFrom) + x(cfg.shiftTo)) / 2}" y="${m.t + 12}" text-anchor="middle" fill="#7c79a8" font-size="10.5">traffic shifts to open-ended tasks</text>`;
  const sw = ws[sel - 1];
  const dots = ws.map((w) => `<circle cx="${x(w.week)}" cy="${y(w.theta)}" r="${w.week === sel ? 5.5 : 3.6}" fill="${w.week === sel ? '#9aa1ea' : '#fff'}" stroke="#9aa1ea" stroke-width="1.6" ${w.provisional ? 'stroke-dasharray="2 1.6"' : ''}/>`).join('');
  const tdots = showTruth ? `<path d="${tline}" fill="none" stroke="#efb08c" stroke-width="2"/>` + ws.map((w) => `<circle cx="${x(w.week)}" cy="${y(w.truth)}" r="${w.week === sel ? 5 : 3.2}" fill="${w.week === sel ? '#efb08c' : '#fff'}" stroke="#efb08c" stroke-width="1.6"/>`).join('') : '';
  const lab = (yy, l, v) => `<text x="${x(sel) + 14}" y="${yy}" fill="#746f68" font-size="12">${l}</text><text x="${x(sel) + 14}" y="${yy + 24}" fill="#34312d" font-size="23" font-weight="500" letter-spacing="-0.9">${v}</text>`;
  let ly1 = y(sw.theta) - 16, ly2 = y(sw.truth) - 16;
  if (showTruth && Math.abs(ly1 - ly2) < 48) { if (ly1 <= ly2) ly2 = ly1 + 48; else ly1 = ly2 + 48; }
  const callouts = (sel === ws.length || true) ? lab(Math.min(H - 70, Math.max(18, ly1)), 'Estimate', pct(sw.theta)) + (showTruth ? lab(Math.min(H - 40, Math.max(18, ly2)), 'Truth', pct(sw.truth)) : '') : '';
  el.innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Weekly ranking accuracy estimates with 90% confidence intervals">
    ${shift}${grid}
    <line x1="${m.l}" x2="${m.l + iw}" y1="${y(th)}" y2="${y(th)}" stroke="#c5372f" stroke-dasharray="4 4" stroke-width="1.2"/>
    <text x="${m.l + iw - 4}" y="${y(th) + 14}" text-anchor="end" fill="#c5372f" font-size="10.5">re-optimize below ${pct(th, 0)}</text>
    <defs><linearGradient id="bandg" x1="0" x2="0" y1="0" y2="1"><stop offset="0" stop-color="#abb1ec" stop-opacity=".42"/><stop offset="1" stop-color="#abb1ec" stop-opacity=".2"/></linearGradient></defs><path d="${band}" fill="url(#bandg)"/>
    <line x1="${x(sel)}" x2="${x(sel)}" y1="${m.t}" y2="${m.t + ih}" stroke="#9aa1ea" stroke-width="1.4"/>
    <path d="${line}" fill="none" stroke="#9aa1ea" stroke-width="2.2"/>
    ${tdots}${dots}${callouts}${xt}
    ${ws.map((w) => `<rect data-wk="${w.week}" x="${x(w.week) - iw / (ws.length - 1) / 2}" y="${m.t}" width="${iw / (ws.length - 1)}" height="${ih}" fill="transparent" style="cursor:pointer"/>`).join('')}
  </svg>`;
  const tip = document.createElement('div'); tip.className = 'tip'; tip.hidden = true; el.parentElement.appendChild(tip);
  el.querySelectorAll('rect[data-wk]').forEach((r) => {
    const w = ws[Number(r.dataset.wk) - 1];
    r.onmousemove = (e) => {
      const b = el.getBoundingClientRect();
      tip.hidden = false;
      tip.innerHTML = `<b>Week ${w.week}${w.provisional ? ' · provisional' : ''}</b><div class="row"><span class="dim">Estimate</span><span class="num">${pct(w.theta)}</span></div><div class="row"><span class="dim">90% CI</span><span class="num">${pct(w.lo)}–${pct(w.hi)}</span></div><div class="row"><span class="dim">Expert labels</span><span class="num">${w.nLab} of ${w.n}</span></div>${showTruth ? `<div class="row"><span class="dim">Hindsight truth</span><span class="num">${pct(w.truth)}</span></div>` : ''}<div class="row"><span class="dim">Status</span><span>${STATUS[w.status].label}</span></div>`;
      const left = Math.min(e.clientX - b.left + 14, b.width - 190);
      tip.style.left = left + 'px'; tip.style.top = Math.max(0, e.clientY - b.top - 40) + 'px';
    };
    r.onmouseleave = () => (tip.hidden = true);
    r.onclick = () => go(`/graders/${gid}`, { week: w.week === ws.length ? '' : w.week, truth: showTruth ? '' : '0' });
  });
}

// ---------------- label queue ----------------
function renderQueue(body, gid, lw, q) {
  const filter = q.get('filter') || 'all';
  const g = A.GRADERS[gid];
  const rows = lw.order.map((k, pos) => ({ k, pos, it: lw.items[k], pi: lw.pi[k], lab: lw.labels[lw.items[k].idx] }));
  const shown = rows.filter((r) => (filter === 'pending' ? !r.lab : filter === 'labeled' ? !!r.lab : true));
  const pend = rows.filter((r) => !r.lab).length;
  const tr = shown.map((r) => {
    const p = A.predict(lw.wk.pred, r.it);
    return `<tr class="click ${String(r.it.idx) === q.get('pair') ? 'sel' : ''}" data-pair="${r.it.idx}">
      <td class="dim num">${r.pos + 1}</td>
      <td><div class="trunc" data-q="${r.it.qid}">${esc(qPreview(r.it))}</div><div class="dim" style="font-size:11.5px">MT-Bench #${r.it.qid} · turn ${r.it.turn}</div></td>
      <td><span class="chip">${CATS[r.it.cat]}</span></td>
      <td><span class="models">${MODEL[r.it.a] || r.it.a}<span class="vs">vs</span>${MODEL[r.it.b] || r.it.b}</span></td>
      <td class="score">${gid === 'scalar' ? `${r.it.sa} · ${r.it.sb}` : { a: 'A', b: 'B', tie: 'Tie', inc: 'Inconsistent' }[r.it.pw]}</td>
      <td class="num">${pct(p.pG, 0)}</td>
      <td class="num">${pct(r.pi, 1)}</td>
      <td>${r.lab ? `<span class="chip ok"><i data-lucide="check" style="width:12px;height:12px"></i>${{ a: 'A better', b: 'B better', tie: 'Tie' }[r.lab]}</span>` : '<span class="chip warn">Pending</span>'}</td></tr>`;
  }).join('');
  body.innerHTML = `<div class="card" style="margin-top:16px">
    <div class="toolbar"><div style="display:flex;gap:10px;align-items:center">
      <div class="seg" role="group" aria-label="Filter">${['all', 'pending', 'labeled'].map((f) => `<button data-f="${f}" class="${f === filter ? 'on' : ''}">${f[0].toUpperCase() + f.slice(1)} <span class="dim">${f === 'all' ? rows.length : f === 'pending' ? pend : rows.length - pend}</span></button>`).join('')}</div>
      <div class="prog" title="${lw.L} of ${lw.S} labeled"><i style="width:${(100 * lw.L) / Math.max(1, lw.S)}%"></i></div><span class="dim num" style="font-size:12px">${lw.L} / ${lw.S} labeled · week ${lw.week}</span></div>
      <div style="display:flex;gap:8px"><button class="btn" id="fill-btn" ${pend ? '' : 'disabled'} title="Fill pending pairs with the recorded LMSYS expert votes"><i data-lucide="users"></i>Use recorded expert votes</button><button class="btn" id="csv-q"><i data-lucide="download"></i>CSV</button><button class="btn ghost" id="reset-q" title="Reset labels and top-ups for this policy"><i data-lucide="rotate-ccw"></i>Reset</button></div></div>
    ${shown.length ? `<div class="table-wrap"><table class="t"><thead><tr><th>#</th><th>Task</th><th>Category</th><th>Responses</th><th>${gid === 'scalar' ? 'Scores A · B' : 'Verdict'}</th><th>Pred. agreement</th><th>Inclusion prob.</th><th>Expert label</th></tr></thead><tbody>${tr}</tbody></table></div>`
      : `<div class="empty"><i data-lucide="${filter === 'pending' ? 'party-popper' : 'inbox'}"></i><h3>${filter === 'pending' ? 'All caught up' : 'Nothing here yet'}</h3><div>${filter === 'pending' ? `Every sampled pair this week has an expert label. <a href="#/graders/${gid}" style="color:var(--link)">See the audit</a>` : 'No labeled pairs yet.'}</div></div>`}
  </div>
  <p class="note" style="margin-top:10px">${esc(g.name)} · queue is in random order so a partly finished queue is still a random subsample (the provisional estimate assumes this). Shortcuts in a pair: <kbd>1</kbd> A better, <kbd>2</kbd> B better, <kbd>3</kbd> tie, <kbd>J</kbd>/<kbd>K</kbd> next/previous, <kbd>Esc</kbd> close.</p>`;
  body.querySelectorAll('[data-f]').forEach((b) => (b.onclick = () => go(`/graders/${gid}/queue`, { filter: b.dataset.f === 'all' ? '' : b.dataset.f })));
  body.querySelectorAll('tr[data-pair]').forEach((t) => (t.onclick = () => go(`/graders/${gid}/queue`, { filter: filter === 'all' ? '' : filter, pair: t.dataset.pair })));
  $('#fill-btn').onclick = () => { rows.filter((r) => !r.lab).forEach((r) => (lw.labels[r.it.idx] = r.it.expert)); save(); toast(`Filled ${pend} pairs with recorded expert votes`); render(); };
  $('#csv-q').onclick = () => download(`${gid}-week${lw.week}-label-queue.csv`, queueCSV(gid, lw), 'text/csv');
  $('#reset-q').onclick = () => { const a = audit(gid); delete store.labels[a.key]; delete store.prefilled[a.key]; delete store.topups[a.key]; save(); toast('Queue reset'); render(); };
  if (D.texts) fillPreviews(body); else texts().then(() => fillPreviews(body)).catch(() => {});
}
function qPreview(it) { return D.texts ? D.texts.questions[it.qid][it.turn - 1] : `MT-Bench question ${it.qid}`; }
function fillPreviews(body) { body.querySelectorAll('[data-q]').forEach((d) => { const tr = d.closest('tr'); const id = Number(tr.dataset.pair); const it = D.raw[id]; d.textContent = D.texts.questions[it.qid][it.turn - 1]; }); }

// ---------------- drawer ----------------
async function openDrawer(gid, idx, q) {
  const lw = liveWeek(gid);
  const k = lw.items.findIndex((it) => it.idx === idx);
  const dr = $('#drawer');
  if (k < 0 || !lw.xi[k]) { dr.innerHTML = `<div class="drawer-head"><b>Pair not in this week's queue</b><button class="btn ghost" style="margin-left:auto" data-close-drawer aria-label="Close"><i data-lucide="x"></i></button></div><div class="empty"><h3>This pair was not sampled for expert review.</h3></div>`; showDrawer(); return; }
  const it = lw.items[k];
  dr.innerHTML = `<div class="drawer-head"><span class="chip">${CATS[it.cat]}</span><b>MT-Bench #${it.qid} · turn ${it.turn}</b><button class="btn ghost" style="margin-left:auto" data-close-drawer aria-label="Close"><i data-lucide="x"></i></button></div><div class="drawer-body"><div class="sk" style="height:80px"></div><div class="sk" style="height:260px;margin-top:12px"></div></div>`;
  showDrawer(); icons();
  let T;
  try { T = await texts(); } catch (e) { $('.drawer-body', dr).innerHTML = `<div class="empty"><h3>Couldn't load pair text</h3><div>${esc(e.message)}</div></div>`; return; }
  const cur = parseHash();
  if (cur.q.get('pair') !== String(idx)) return;
  const p = A.predict(lw.wk.pred, it);
  const pos = lw.order.indexOf(k);
  const lab = lw.labels[it.idx];
  const qs = T.questions[it.qid];
  const ans = (m) => (T.answers[`${it.qid}|${m}`] || [])[it.turn - 1] || '(missing)';
  const jt = (m) => T.judgments[`${it.qid}|${m}|${it.turn}`] || '';
  const v = it.votes;
  dr.innerHTML = `<div class="drawer-head"><span class="chip">${CATS[it.cat]}</span><b>MT-Bench #${it.qid} · turn ${it.turn}</b><span class="dim">${pos + 1} of ${lw.order.length}</span>
      <div style="margin-left:auto;display:flex;gap:6px"><button class="btn ghost" id="prev-p" aria-label="Previous pair (K)"><i data-lucide="chevron-up"></i></button><button class="btn ghost" id="next-p" aria-label="Next pair (J)"><i data-lucide="chevron-down"></i></button><button class="btn ghost" data-close-drawer aria-label="Close (Esc)"><i data-lucide="x"></i></button></div></div>
    <div class="drawer-body">
      ${it.turn === 2 ? `<div class="dim" style="font-size:12px;margin-bottom:4px">Turn 1</div><div class="q" style="opacity:.75">${esc(qs[0])}</div><div class="dim" style="font-size:12px;margin:10px 0 4px">Turn 2 (graded)</div>` : '<div class="dim" style="font-size:12px;margin-bottom:4px">User</div>'}
      <div class="q">${esc(qs[it.turn - 1])}</div>
      <div class="resp">
        ${['a', 'b'].map((s) => { const m = it[s]; const sc = s === 'a' ? it.sa : it.sb; return `<div class="card"><div class="card-head"><div class="card-title">Response ${s.toUpperCase()} <span class="dim" style="font-weight:400">${MODEL[m] || m}</span></div>${gid === 'scalar' ? `<span class="chip ${it.pref === s ? 'info' : ''}">score ${sc}/10</span>` : ''}</div><div class="card-body">${esc(ans(m))}</div>${gid === 'scalar' ? `<div class="verdict"><b>Grader rationale</b>\n${esc(jt(m))}</div>` : ''}</div>`; }).join('')}
      </div>
      <div class="card" style="margin-top:12px"><div class="card-body"><dl class="kv">
        <dt>Grader prefers</dt><dd>${{ a: 'Response A', b: 'Response B', tie: 'Neither (tie)' }[it.pref]}${gid === 'pairwise' ? ` · verdict ${{ a: 'A', b: 'B', tie: 'tie', inc: 'order-inconsistent' }[it.pw]}` : ''}</dd>
        <dt>Predicted agreement with expert</dt><dd>${pct(p.pG, 0)} · expert tie ${pct(1 - p.pD, 0)}</dd>
        <dt>Inclusion probability</dt><dd>${pct(lw.pi[k], 1)} · weight 1/${(1 / lw.pi[k]).toFixed(1)}</dd></dl>
        <details class="reveal" style="margin-top:10px"><summary>Reveal recorded LMSYS expert votes</summary><div class="note" style="margin-top:6px">${v[0]} for A · ${v[1]} for B · ${v[2]} tie → majority: <b>${{ a: 'A better', b: 'B better', tie: 'Tie' }[it.expert]}</b></div></details>
      </div></div>
    </div>
    <div class="drawer-foot"><span style="font-weight:500">Expert label</span>
      <div class="label-btns">${[['a', 'A is better', '1'], ['b', 'B is better', '2'], ['tie', 'Tie', '3']].map(([l, t, kb]) => `<button class="btn ${lab === l ? 'on' : ''}" data-label="${l}">${t}<kbd>${kb}</kbd></button>`).join('')}</div>
      ${lab ? '<button class="btn ghost" id="clear-l">Clear</button>' : ''}<span class="dim" style="margin-left:auto;font-size:12px">Saved in this browser</span></div>`;
  icons();
  dr.querySelectorAll('[data-label]').forEach((b) => (b.onclick = () => setLabel(gid, it.idx, b.dataset.label, true)));
  const cl = $('#clear-l'); if (cl) cl.onclick = () => { delete lw.labels[it.idx]; save(); render(); };
  $('#prev-p').onclick = () => step(gid, -1);
  $('#next-p').onclick = () => step(gid, 1);
}
function showDrawer() { const dr = $('#drawer'); dr.classList.add('open'); dr.setAttribute('aria-hidden', 'false'); $('#scrim').hidden = false; dr.querySelectorAll('[data-close-drawer]').forEach((b) => (b.onclick = closeDrawerNav)); }
function closeDrawer() { const dr = $('#drawer'); dr.classList.remove('open'); dr.setAttribute('aria-hidden', 'true'); $('#scrim').hidden = true; }
function closeDrawerNav() { const { parts, q } = parseHash(); q.delete('pair'); go('/' + parts.join('/'), Object.fromEntries(q)); }
$('#scrim').onclick = closeDrawerNav;

function setLabel(gid, idx, l, advance) {
  const a = audit(gid);
  (store.labels[a.key] ||= {})[idx] = l; save();
  const before = liveWeek(gid);
  if (advance) {
    const k = before.items.findIndex((it) => it.idx === idx);
    const pos = before.order.indexOf(k);
    const nextPending = before.order.slice(pos + 1).concat(before.order.slice(0, pos)).find((kk) => !before.labels[before.items[kk].idx]);
    const { parts, q } = parseHash();
    if (nextPending !== undefined) { toast(`Labeled · ${before.S - before.L} left`); go('/' + parts.join('/'), { ...Object.fromEntries(q), pair: before.items[nextPending].idx }); return; }
    toast(`All ${before.S} sampled pairs labeled`);
    go(`/graders/${gid}`);
    return;
  }
  render();
}
function step(gid, d) {
  const lw = liveWeek(gid); const { parts, q } = parseHash();
  const idx = Number(q.get('pair')); const k = lw.items.findIndex((it) => it.idx === idx); const pos = lw.order.indexOf(k);
  const np = lw.order[(pos + d + lw.order.length) % lw.order.length];
  go('/' + parts.join('/'), { ...Object.fromEntries(q), pair: lw.items[np].idx });
}

// ---------------- program tab ----------------
function renderProgram(body, gid) {
  const g = A.GRADERS[gid], ev = D.results.evaluation[gid];
  const prompt = D.texts ? D.texts.prompt_single : 'Loading…';
  body.innerHTML = `<div class="grid-2"><div class="card"><div class="card-head"><div class="card-title">Grader program</div><span class="chip">${esc(g.kind)}</span></div><div class="card-body">
    ${gid === 'scalar' ? `<p class="note" style="margin-top:0">One judge call per response returns a 1–10 rating; the higher-rated response wins and equal ratings are a tie (counted as a miss for ranking accuracy).</p><pre class="q mono" style="font-size:11.5px" id="prompt">${esc(prompt)}</pre>`
      : `<p class="note" style="margin-top:0">The judge sees both responses and picks a winner, then is called again with the order swapped. If the two verdicts disagree the pair is recorded as order-inconsistent (a tie).</p>`}
    </div></div><div class="card"><div class="card-head"><div class="card-title">Selection</div></div><div class="card-body"><dl class="kv"><dt>Judge</dt><dd>${esc(g.judge)}</dd><dt>Selection pairs</dt><dd>${D.results.config.seedSize}</dd><dt>Selection accuracy</dt><dd>${pct(ev.seedTruth)}</dd><dt>Feature used by the audit</dt><dd>${gid === 'scalar' ? 'score margin' : 'verdict consistency'}</dd></dl></div></div></div>`;
  if (!D.texts && gid === 'scalar') texts().then((T) => { const el = $('#prompt'); if (el) el.textContent = T.prompt_single; }).catch(() => {});
}

// ---------------- export ----------------
function exportMenu(e, gid) {
  e.stopPropagation();
  const old = $('.menu'); if (old) { old.remove(); return; }
  const m = document.createElement('div');
  m.className = 'menu card'; m.setAttribute('role', 'menu');
  m.style.cssText = 'position:absolute;right:0;top:34px;z-index:20;padding:4px;min-width:220px;box-shadow:var(--elevation-md)';
  m.innerHTML = [['weekly', 'Weekly estimates (CSV)'], ['queue', 'Week 8 label queue (CSV)'], ['json', 'Full audit (JSON)']].map(([k, t]) => `<div class="palette-item" role="menuitem" tabindex="0" data-x="${k}"><i data-lucide="file-down"></i>${t}</div>`).join('');
  e.currentTarget.parentElement.appendChild(m); icons();
  m.querySelectorAll('[data-x]').forEach((it) => (it.onclick = () => {
    const lw = liveWeek(gid);
    if (it.dataset.x === 'weekly') download(`${gid}-weekly-ranking-accuracy.csv`, weeklyCSV(gid), 'text/csv');
    if (it.dataset.x === 'queue') download(`${gid}-week${lw.week}-label-queue.csv`, queueCSV(gid, lw), 'text/csv');
    if (it.dataset.x === 'json') download(`${gid}-audit.json`, JSON.stringify({ grader: gid, policy: settings(), weeks: weeksFor(gid), evaluation: D.results.evaluation[gid], source: 'LMSYS MT-Bench human judgments (CC BY 4.0)' }, null, 2), 'application/json');
    m.remove();
  }));
  setTimeout(() => document.addEventListener('click', () => m.remove(), { once: true }), 0);
}
function weeklyCSV(gid) {
  const rows = [['week', 'pairs', 'expert_labels', 'estimate', 'ci90_lo', 'ci90_hi', 'pooled_estimate', 'status', 'provisional', 'hindsight_truth']];
  for (const w of weeksFor(gid)) rows.push([w.week, w.n, w.nLab, w.theta.toFixed(4), w.lo.toFixed(4), w.hi.toFixed(4), w.win.theta.toFixed(4), w.status, w.provisional, w.truth.toFixed(4)]);
  return rows.map((r) => r.join(',')).join('\n');
}
function queueCSV(gid, lw) {
  const rows = [['queue_pos', 'mtbench_question_id', 'turn', 'category', 'model_a', 'model_b', 'grader_pref', 'inclusion_prob', 'expert_label']];
  lw.order.forEach((k, pos) => { const it = lw.items[k]; rows.push([pos + 1, it.qid, it.turn, it.cat, it.a, it.b, it.pref, lw.pi[k].toFixed(4), lw.labels[it.idx] || '']); });
  return rows.map((r) => r.join(',')).join('\n');
}
function download(name, text, type) {
  const a = document.createElement('a'); a.href = URL.createObjectURL(new Blob([text], { type })); a.download = name; document.body.appendChild(a); a.click();
  setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 100); toast(`Exported ${name}`);
}

// ---------------- palette, modal, toast, keys ----------------
function openPalette() {
  const p = $('#palette'); p.hidden = false; const inp = $('#palette-q'); inp.value = ''; inp.focus(); paletteList(''); texts().then(() => paletteList(inp.value)).catch(() => {});
}
function closePalette() { $('#palette').hidden = true; }
let palSel = 0, palItems = [];
function paletteList(qs) {
  const ql = qs.toLowerCase();
  const pages = [['Graders', '#/graders'], ['Scalar metagrader · Audit', '#/graders/scalar'], ['Scalar metagrader · Label queue', '#/graders/scalar/queue'], ['Pairwise grader · Audit', '#/graders/pairwise'], ['Pairwise grader · Label queue', '#/graders/pairwise/queue']].filter(([t]) => t.toLowerCase().includes(ql)).map(([t, h]) => ({ t, h, k: 'Page' }));
  let pairs = [];
  if (D.texts && ql.length > 1) {
    for (const gid of Object.keys(A.GRADERS)) {
      const lw = liveWeek(gid);
      for (const k of lw.order) { const it = lw.items[k]; const qt = D.texts.questions[it.qid][it.turn - 1]; if (qt.toLowerCase().includes(ql) || CATS[it.cat].toLowerCase().includes(ql)) pairs.push({ t: qt.slice(0, 70), h: `#/graders/${gid}/queue?pair=${it.idx}`, k: A.GRADERS[gid].kind + ' queue' }); if (pairs.length > 30) break; }
    }
  }
  palItems = [...pages, ...pairs].slice(0, 40); palSel = 0;
  $('#palette-list').innerHTML = palItems.length ? palItems.map((x, i) => `<div class="palette-item ${i === palSel ? 'on' : ''}" role="option" data-i="${i}"><i data-lucide="${x.k === 'Page' ? 'file' : 'message-square'}"></i><span class="trunc" style="max-width:420px">${esc(x.t)}</span><span class="k">${x.k}</span></div>`).join('') : '<div class="empty" style="padding:20px">No matches</div>';
  icons();
  $('#palette-list').querySelectorAll('[data-i]').forEach((el) => (el.onclick = () => { closePalette(); location.hash = palItems[Number(el.dataset.i)].h; }));
}
$('#palette-q').oninput = (e) => paletteList(e.target.value);
$('#palette').onclick = (e) => { if (e.target.id === 'palette') closePalette(); };
$('#open-palette').onclick = openPalette;

function modal(html) { const m = $('#modal'); m.innerHTML = `<div class="modal" role="dialog" aria-modal="true">${html}</div>`; m.hidden = false; icons(); m.querySelectorAll('[data-close]').forEach((b) => (b.onclick = closeModal)); m.onclick = (e) => { if (e.target === m) closeModal(); }; }
function closeModal() { $('#modal').hidden = true; $('#modal').innerHTML = ''; }
let toastT;
function toast(t) { const el = $('#toast'); el.innerHTML = `<i data-lucide="check"></i>${esc(t)}`; el.hidden = false; icons(); clearTimeout(toastT); toastT = setTimeout(() => (el.hidden = true), 2200); }

document.addEventListener('keydown', (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') { e.preventDefault(); $('#palette').hidden ? openPalette() : closePalette(); return; }
  if (!$('#palette').hidden) {
    if (e.key === 'Escape') closePalette();
    if (e.key === 'ArrowDown' || e.key === 'ArrowUp') { e.preventDefault(); palSel = (palSel + (e.key === 'ArrowDown' ? 1 : -1) + palItems.length) % Math.max(1, palItems.length); $('#palette-list').querySelectorAll('.palette-item').forEach((el, i) => el.classList.toggle('on', i === palSel)); }
    if (e.key === 'Enter' && palItems[palSel]) { closePalette(); location.hash = palItems[palSel].h; }
    return;
  }
  if (!$('#modal').hidden) { if (e.key === 'Escape') closeModal(); return; }
  if (e.target.closest('input, textarea')) return;
  const { parts, q } = parseHash();
  if (parts[2] === 'queue' && q.get('pair')) {
    const gid = parts[1];
    if (e.key === 'Escape') closeDrawerNav();
    else if (e.key === '1') setLabel(gid, Number(q.get('pair')), 'a', true);
    else if (e.key === '2') setLabel(gid, Number(q.get('pair')), 'b', true);
    else if (e.key === '3') setLabel(gid, Number(q.get('pair')), 'tie', true);
    else if (e.key.toLowerCase() === 'j') step(gid, 1);
    else if (e.key.toLowerCase() === 'k') step(gid, -1);
  }
});
let rz;
window.addEventListener('resize', () => { clearTimeout(rz); rz = setTimeout(() => { const { parts } = parseHash(); if (parts[0] === 'graders' && parts[1] && !parts[2]) render(); }, 150); });

function icons() { if (window.lucide) window.lucide.createIcons({ attrs: { 'stroke-width': 1.75 } }); }
icons();
boot();
