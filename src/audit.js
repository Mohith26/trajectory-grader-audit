// Grader audit: active expert labeling + design-based inference for a grader's ranking accuracy.
//
// Target (Trajectory's own metric, Continual Grading post): ranking accuracy
//   theta = P( r_G(x, y+) > r_G(x, y-) | expert prefers one response ),
// over the finite population of response pairs a grader scored in one week of production.
// Expert ties are excluded, exactly as in the post, which makes theta a ratio A / D with
//   A_i = 1[expert not tie and grader strictly agrees],  D_i = 1[expert not tie].
//
// Design: Poisson sampling with inclusion probabilities pi_i chosen to minimise the variance of the
// difference (model-assisted / AIPW) estimator, pi_i ~ sqrt(Var(A_i - theta D_i | x_i)), mixed with
// a uniform floor (Zrnic & Candes 2024, "Active Statistical Inference"; Sarndal et al. GREG).
// Estimator: theta_hat = sum(At_i) / sum(Dt_i), At_i = pA_i + xi_i/pi_i (A_i - pA_i), same for D.
// Unbiased for both sums whatever the predictor is; variance from the Horvitz-Thompson formula for
// Poisson sampling of the linearised residual. Monitoring across weeks spends alpha as
// alpha_t = alpha * 6 / (pi^2 t^2) so every weekly alarm holds simultaneously (union bound).

export function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// ---------- graders ----------
// Each grader maps a pair to a preference ('a' | 'b' | 'tie') and a feature bucket k used by the
// predictor. Ranking accuracy counts a grader tie as a miss (strict inequality, as in the post).
const scalarPref = (it) => (it.sa > it.sb ? 'a' : it.sb > it.sa ? 'b' : 'tie');
const pairPref = (it) => (it.pw === 'a' || it.pw === 'b' ? it.pw : 'tie');
export const GRADERS = {
  scalar: {
    id: 'scalar',
    name: 'Scalar metagrader',
    judge: 'gpt-4 · single-v1',
    kind: 'Scalar',
    pref: scalarPref,
    bucket: (it) => Math.min(4, Math.floor(Math.abs(it.sa - it.sb))),
    buckets: ['Score tie', '1 pt margin', '2 pt margin', '3 pt margin', '4+ pt margin'],
    zeroAgree: [0],
  },
  pairwise: {
    id: 'pairwise',
    name: 'Pairwise grader',
    judge: 'gpt-4 · pair-v2, both orders',
    kind: 'Pairwise',
    pref: pairPref,
    bucket: (it) => (it.pw === 'inc' ? 0 : it.pw === 'tie' ? 1 : 2),
    buckets: ['Order-inconsistent', 'Consistent tie', 'Consistent winner'],
    zeroAgree: [0, 1],
  },
};
for (const g of Object.values(GRADERS)) g.nBuckets = g.buckets.length;

// Attach grader preference and bucket to every item (views share the underlying data).
export function viewFor(items, graderId) {
  const g = GRADERS[graderId];
  return items.map((it, i) => ({ ...it, idx: i, pref: g.pref(it), k: g.bucket(it) }));
}

// Outcomes for one item given the expert label.
export function outcome(item, expert = item.expert) {
  const D = expert === 'tie' ? 0 : 1;
  const A = D && item.pref === expert ? 1 : 0;
  return { A, D };
}

// ---------- predictor ----------
// Beta-smoothed rates per margin bucket for P(D=1|k) and P(agree | D=1, k), fitted on labelled items,
// then partially pooled per (bucket, category) toward the bucket rate with strength m (empirical Bayes).
export function fitPredictor(labelled, { nBuckets = 5, zeroAgree = [0], prior = { d: [3, 1], g: [1, 1] }, m = 8, byCategory = false } = {}) {
  const cD = Array.from({ length: nBuckets }, () => [0, 0]);
  const cG = Array.from({ length: nBuckets }, () => [0, 0]);
  const cell = {};
  for (const { item, expert } of labelled) {
    const k = item.k;
    const { A, D } = outcome(item, expert);
    cD[k][0] += D; cD[k][1] += 1;
    if (D) { cG[k][0] += A; cG[k][1] += 1; }
    const c = (cell[k + '|' + item.cat] ||= { d: [0, 0], g: [0, 0] });
    c.d[0] += D; c.d[1] += 1;
    if (D) { c.g[0] += A; c.g[1] += 1; }
  }
  const pD = cD.map(([s, n]) => (s + prior.d[0]) / (n + prior.d[0] + prior.d[1]));
  // A grader tie can never "strictly agree", so P(agree | tie buckets) is exactly 0.
  const pG = cG.map(([s, n], k) => (zeroAgree.includes(k) ? 0 : (s + prior.g[0]) / (n + prior.g[0] + prior.g[1])));
  const cells = {};
  if (byCategory) {
    for (const [key, c] of Object.entries(cell)) {
      const k = Number(key.split('|')[0]);
      cells[key] = { pD: (c.d[0] + m * pD[k]) / (c.d[1] + m), pG: zeroAgree.includes(k) ? 0 : (c.g[0] + m * pG[k]) / (c.g[1] + m) };
    }
  }
  return { pD, pG, cells, n: labelled.length };
}

export function predict(pred, item) {
  const k = item.k;
  const c = pred.cells && pred.cells[k + '|' + item.cat];
  const pD = c ? c.pD : pred.pD[k];
  const pG = c ? c.pG : pred.pG[k];
  return { pD, pA: pD * pG, pG, k };
}

// Conditional variance of the linearised ratio residual A - theta D given the features.
export function residualVar(pD, pG, theta) {
  const m = pD * (pG - theta);
  const s2 = pD * (pG * (1 - theta) ** 2 + (1 - pG) * theta ** 2);
  return Math.max(0, s2 - m * m);
}

// ---------- design ----------
// Scale non-negative scores to inclusion probabilities summing to `budget`, capped at 1.
export function capToBudget(scores, budget) {
  // Water-filling: pi_i = min(1, lambda * s_i) with sum = budget. If the positive-score items cannot
  // absorb the budget (all capped at 1), the remainder is spread evenly over the zero-score items.
  const n = scores.length;
  const B = Math.min(budget, n);
  const pi = new Array(n).fill(0);
  const fixed = new Array(n).fill(false);
  for (let iter = 0; iter <= n; iter++) {
    let rem = B, tot = 0;
    for (let i = 0; i < n; i++) { if (fixed[i]) rem -= 1; else tot += scores[i]; }
    let changed = false;
    for (let i = 0; i < n; i++) {
      if (fixed[i]) { pi[i] = 1; continue; }
      pi[i] = tot > 0 ? (rem * scores[i]) / tot : 0;
      if (pi[i] >= 1) { fixed[i] = true; pi[i] = 1; changed = true; }
    }
    if (!changed) break;
  }
  const used = pi.reduce((t, p) => t + p, 0);
  if (B - used > 1e-9) {
    const free = pi.map((p, i) => (p < 1 && scores[i] <= 0 ? i : -1)).filter((i) => i >= 0);
    for (const i of free) pi[i] = Math.min(1, (B - used) / free.length);
  }
  return pi;
}

export function activeDesign(items, pred, theta, budget, tau = 0.2) {
  const sd = items.map((it) => {
    const p = predict(pred, it);
    return Math.sqrt(residualVar(p.pD, p.pG, theta));
  });
  const opt = capToBudget(sd, budget);
  const uni = Math.min(1, budget / items.length);
  return capToBudget(opt.map((p) => (1 - tau) * p + tau * uni), budget);
}

export function uniformDesign(items, budget) {
  return items.map(() => Math.min(1, budget / items.length));
}

export function poissonSample(pi, rng) {
  return pi.map((p) => (rng() < p ? 1 : 0));
}

// ---------- inference ----------
export function normalQuantile(p) {
  // Acklam's rational approximation, |error| < 1.2e-9
  const a = [-3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2, 1.38357751867269e2, -3.066479806614716e1, 2.506628277459239];
  const b = [-5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2, 6.680131188771972e1, -1.328068155288572e1];
  const c = [-7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783];
  const d = [7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996, 3.754408661907416];
  const pl = 0.02425;
  if (p < pl) { const q = Math.sqrt(-2 * Math.log(p)); return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1); }
  if (p > 1 - pl) return -normalQuantile(1 - p);
  const q = p - 0.5, r = q * q;
  return ((((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q) / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
}

// Student-t quantile via the Cornish-Fisher expansion around the normal quantile (df >= 3 is accurate
// to ~1e-3, which is far below the Monte Carlo error of any coverage check here).
export function tQuantile(p, df) {
  const z = normalQuantile(p);
  if (!Number.isFinite(df) || df > 1e6) return z;
  const d = Math.max(df, 2);
  const z3 = z ** 3, z5 = z ** 5, z7 = z ** 7;
  return z + (z3 + z) / (4 * d) + (5 * z5 + 16 * z3 + 3 * z) / (96 * d * d) + (3 * z7 + 19 * z5 + 17 * z3 - 15 * z) / (384 * d ** 3);
}

// labels: array (same length as items) of expert labels or null when not sampled.
// preds: array of {pA, pD} (all zeros gives the plain Horvitz-Thompson / sample-ratio estimator).
export function estimate(items, pi, xi, labels, preds, alpha = 0.1, sided = 'two') {
  let sA = 0, sD = 0, nLab = 0;
  const parts = [];
  for (let i = 0; i < items.length; i++) {
    const p = preds[i];
    let At = p.pA, Dt = p.pD;
    if (xi[i] && labels[i] != null) {
      const { A, D } = outcome(items[i], labels[i]);
      At += (A - p.pA) / pi[i];
      Dt += (D - p.pD) / pi[i];
      parts.push([i, A, D]);
      nLab++;
    }
    sA += At; sD += Dt;
  }
  const theta = sD > 0 ? Math.min(1, Math.max(0, sA / sD)) : NaN;
  // HT variance of sum_i (At_i - theta Dt_i) under Poisson sampling
  let v = 0;
  for (const [i, A, D] of parts) {
    const p = preds[i];
    const e = (A - theta * D) - (p.pA - theta * p.pD);
    v += ((1 - pi[i]) / (pi[i] * pi[i])) * e * e;
  }
  const se = sD > 0 ? Math.sqrt(v) / sD : NaN;
  const z = tQuantile(1 - (sided === 'two' ? alpha / 2 : alpha), Math.max(2, nLab - 1));
  return { theta, se, z, lo: Math.max(0, theta - z * se), hi: Math.min(1, theta + z * se), nLab, effD: sD };
}

// Decision against a threshold from a one-sided interval at level alpha on each side.
export function status(est, thetaMin, alpha = 0.05) {
  if (!Number.isFinite(est.theta) || !Number.isFinite(est.se)) return 'insufficient';
  const z = tQuantile(1 - alpha, Math.max(2, est.nLab - 1));
  if (est.theta + z * est.se < thetaMin) return 'below';
  if (est.theta - z * est.se > thetaMin) return 'healthy';
  return 'inconclusive';
}

// Extra labels needed (at the current design efficiency, variance ~ 1/labels) before the one-sided
// interval excludes the threshold. Returns 0 if already decided, Infinity if the estimate sits on it.
export function labelsToDecide(est, thetaMin, alpha = 0.05) {
  if (!(est.nLab > 0) || !Number.isFinite(est.se)) return Infinity;
  const gap = Math.abs(est.theta - thetaMin);
  const z = tQuantile(1 - alpha, Math.max(2, est.nLab - 1));
  if (z * est.se < gap) return 0;
  if (gap < 1e-9) return Infinity;
  return Math.max(1, Math.ceil(est.nLab * ((z * est.se) / gap) ** 2 - est.nLab));
}

// Top-up sample: raise inclusion probabilities from pi to pi' >= pi and draw only among unsampled
// items with probability (pi' - pi) / (1 - pi), so each item's overall inclusion probability is pi'.
export function topUp(items, pi, xi, pred, theta, extra, rng, tau = 0.2) {
  const target = activeDesign(items, pred, theta, pi.reduce((s, p) => s + p, 0) + extra, tau);
  const pi2 = pi.map((p, i) => Math.max(p, target[i]));
  const xi2 = xi.map((x, i) => (x ? 1 : pi[i] >= 1 ? 0 : rng() < (pi2[i] - pi[i]) / (1 - pi[i]) ? 1 : 0));
  return { pi: pi2, xi: xi2, added: xi2.reduce((s, x, i) => s + (x && !xi[i] ? 1 : 0), 0) };
}

export function trueAccuracy(items) {
  let a = 0, d = 0;
  for (const it of items) { const o = outcome(it); a += o.A; d += o.D; }
  return d ? a / d : NaN;
}

export function alphaSpend(alpha, t) {
  return (alpha * 6) / (Math.PI * Math.PI * t * t);
}

// ---------- production stream ----------
// Deterministically assigns the real pairs to a seed (selection) set and W weeks whose category mix
// shifts from `mixStart` to `mixEnd` between weeks `shiftFrom` and `shiftTo`.
export function buildStream(items, { seed = 20261102, weeks = 8, seedSize = 60, mixStart, mixEnd, shiftFrom = 4, shiftTo = 6 }) {
  const rng = mulberry32(seed);
  const byCat = {};
  const idx = items.map((_, i) => i);
  for (let i = idx.length - 1; i > 0; i--) { const j = Math.floor(rng() * (i + 1)); [idx[i], idx[j]] = [idx[j], idx[i]]; }
  for (const i of idx) (byCat[items[i].cat] ||= []).push(i);
  const cats = Object.keys(mixStart);
  const take = (w) => {
    const live = cats.filter((c) => byCat[c] && byCat[c].length && w[c] > 0);
    const tot = live.reduce((s, c) => s + w[c], 0);
    let r = rng() * tot;
    for (const c of live) { r -= w[c]; if (r <= 0) return byCat[c].pop(); }
    return byCat[live[live.length - 1]].pop();
  };
  const seedSet = [];
  for (let k = 0; k < seedSize; k++) seedSet.push(take(mixStart));
  const remaining = items.length - seedSize;
  const per = Math.floor(remaining / weeks);
  const out = [];
  for (let w = 1; w <= weeks; w++) {
    const f = Math.min(1, Math.max(0, (w - shiftFrom) / (shiftTo - shiftFrom)));
    const mix = {};
    for (const c of cats) mix[c] = (1 - f) * mixStart[c] + f * mixEnd[c];
    const n = w === weeks ? remaining - per * (weeks - 1) : per;
    const wk = [];
    for (let k = 0; k < n; k++) {
      let i = take(mix);
      if (i === undefined) i = take(Object.fromEntries(cats.map((c) => [c, 1])));
      if (i !== undefined) wk.push(i);
    }
    out.push(wk);
  }
  return { seedSet, weeks: out };
}

// ---------- full monitoring run ----------
// Runs W weeks of labelling with either the active or the uniform design and returns per-week results.
// `labelFor(i)` returns the expert label for item index i (real recorded label by default).
export function runMonitor(items, stream, { budget = 40, alpha = 0.1, thetaMin = 0.72, design = 'active', tau = 0.2, seed = 1, window = 2, grader = 'scalar', labelFor } = {}) {
  const rng = mulberry32(seed);
  const lab = labelFor || ((i) => items[i].expert);
  const g = GRADERS[grader];
  const popts = { nBuckets: g.nBuckets, zeroAgree: g.zeroAgree };
  const history = stream.seedSet.map((i) => ({ item: items[i], expert: lab(i) }));
  const seedPred = fitPredictor(history, popts);
  let thetaPrev = trueAccuracySample(history);
  const weeks = [];
  for (let t = 0; t < stream.weeks.length; t++) {
    const ids = stream.weeks[t];
    const wkItems = ids.map((i) => items[i]);
    const pred = design === 'uniform' ? null : fitPredictor(history, popts);
    const pi = design === 'active' ? activeDesign(wkItems, pred, thetaPrev, budget, tau) : uniformDesign(wkItems, budget);
    const xi = poissonSample(pi, rng);
    const labels = ids.map((i, k) => (xi[k] ? lab(i) : null));
    const preds = wkItems.map((it) => (pred ? predict(pred, it) : { pA: 0, pD: 0 }));
    const est = estimate(wkItems, pi, xi, labels, preds, alpha);
    const at = alphaSpend(alpha, t + 1);
    const cur = { wkItems, pi, xi, labels, preds };
    const win = [...weeks.slice(Math.max(0, weeks.length - window + 1)).map((w) => w._cur), cur];
    const cat = (k) => win.flatMap((w) => w[k]);
    const pooled = estimate(cat('wkItems'), cat('pi'), cat('xi'), cat('labels'), cat('preds'), alpha);
    const anytime = estimate(cat('wkItems'), cat('pi'), cat('xi'), cat('labels'), cat('preds'), at, 'one');
    const alarm = anytime.hi < thetaMin;
    const st = status(pooled, thetaMin, alpha / 2);
    weeks.push({ week: t + 1, n: ids.length, ...est, window: win.length, win: { theta: pooled.theta, lo: pooled.lo, hi: pooled.hi, se: pooled.se, nLab: pooled.nLab },
      status: st, toDecide: labelsToDecide(pooled, thetaMin, alpha / 2), ucbAnytime: anytime.hi, alphaT: at, alarm,
      truth: trueAccuracy(wkItems), windowTruth: trueAccuracy(cat('wkItems')), thetaPlan: thetaPrev, pred, pi, xi, ids, labels, preds, _cur: cur });
    ids.forEach((i, k) => { if (xi[k]) history.push({ item: items[i], expert: labels[k] }); });
    if (Number.isFinite(est.theta)) thetaPrev = est.theta;
  }
  return { weeks, seedPred, seedAccuracy: trueAccuracySample(stream.seedSet.map((i) => ({ item: items[i], expert: lab(i) }))) };
}

function trueAccuracySample(labelled) {
  let a = 0, d = 0;
  for (const { item, expert } of labelled) { const o = outcome(item, expert); a += o.A; d += o.D; }
  return d ? a / d : 0.75;
}

// Monte Carlo over sampling seeds (the data and stream stay fixed): coverage, width, alarms.
export function monteCarlo(items, stream, opts, reps = 200, seed0 = 1000) {
  const W = stream.weeks.length;
  const acc = { below: Array(W).fill(0), cover: Array(W).fill(0), width: Array(W).fill(0), labels: Array(W).fill(0), alarmBy: Array(W).fill(0), falseAlarm: 0, sqErr: Array(W).fill(0) };
  for (let r = 0; r < reps; r++) {
    const run = runMonitor(items, stream, { ...opts, seed: seed0 + r });
    let first = -1;
    run.weeks.forEach((w, t) => {
      acc.cover[t] += w.lo <= w.truth && w.truth <= w.hi ? 1 : 0;
      acc.width[t] += w.hi - w.lo;
      acc.labels[t] += w.nLab;
      acc.sqErr[t] += (w.theta - w.truth) ** 2;
      if (w.alarm && first < 0) first = t;
      if (w.alarm && w.windowTruth >= opts.thetaMin) acc.falseAlarm += 1;
      if (w.status === 'below') acc.below[t] += 1;
    });
    if (first >= 0) for (let t = first; t < W; t++) acc.alarmBy[t] += 1;
  }
  const f = (a) => a.map((x) => x / reps);
  return { reps, coverage: f(acc.cover), width: f(acc.width), labels: f(acc.labels), alarmBy: f(acc.alarmBy), rmse: acc.sqErr.map((x) => Math.sqrt(x / reps)), belowRate: f(acc.below), falseAlarmsPerRun: acc.falseAlarm / reps };
}
