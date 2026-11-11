// Builds data/results.json: the production stream, the seeded audit runs shown in the app, and the
// Monte Carlo evaluation behind every headline number. Deterministic: node scripts/build.mjs
import fs from 'node:fs';
import * as A from '../src/audit.js';

export const CONFIG = {
  streamSeed: 20261102,
  runSeed: 24, // typical run: MAE within 0.006 of the 400-run median for both graders, first with full weekly coverage
  weeks: 8,
  seedSize: 60,
  budget: 40,
  alpha: 0.1,
  thetaMin: 0.74,
  window: 2,
  tau: 0.2,
  // Traffic mix: technical questions at launch, drifting to open-ended writing and roleplay from week 4 to 6.
  mixStart: { stem: 1, coding: 1, extraction: 1, math: 1, humanities: 0.12, reasoning: 0.12, roleplay: 0.12, writing: 0.12 },
  mixEnd: { stem: 0.12, coding: 0.12, extraction: 0.12, math: 0.12, humanities: 1, reasoning: 1, roleplay: 1, writing: 1 },
  shiftFrom: 4,
  shiftTo: 6,
  reps: 400,
  uniformBudgets: [40, 45, 50, 55, 60, 65, 70, 75, 80],
};

export function makeStream(raw, cfg = CONFIG) {
  return A.buildStream(raw, { seed: cfg.streamSeed, weeks: cfg.weeks, seedSize: cfg.seedSize, mixStart: cfg.mixStart, mixEnd: cfg.mixEnd, shiftFrom: cfg.shiftFrom, shiftTo: cfg.shiftTo });
}

export function equivalentUniform(activeWidth, curve) {
  // curve: [[budget, meanWidth]] with width decreasing in budget; linear interpolation, no extrapolation.
  for (let i = 0; i + 1 < curve.length; i++) {
    const [b0, w0] = curve[i], [b1, w1] = curve[i + 1];
    if (w0 >= activeWidth && activeWidth >= w1) return b0 + ((w0 - activeWidth) / (w0 - w1)) * (b1 - b0);
  }
  return null;
}

const mean = (a) => a.reduce((s, x) => s + x, 0) / a.length;

export function evaluate(raw, stream, cfg = CONFIG, reps = cfg.reps) {
  const out = {};
  for (const gid of Object.keys(A.GRADERS)) {
    const items = A.viewFor(raw, gid);
    const base = { budget: cfg.budget, alpha: cfg.alpha, thetaMin: cfg.thetaMin, window: cfg.window, tau: cfg.tau, grader: gid };
    const act = A.monteCarlo(items, stream, { ...base, design: 'active' }, reps);
    const ma = A.monteCarlo(items, stream, { ...base, design: 'uniform-ma' }, reps);
    const curve = cfg.uniformBudgets.map((b) => {
      const mc = A.monteCarlo(items, stream, { ...base, budget: b, design: 'uniform' }, reps);
      return { budget: b, width: mean(mc.width), coverage: mean(mc.coverage), rmse: mean(mc.rmse), labels: mean(mc.labels), belowLast: mc.belowRate[cfg.weeks - 1] };
    });
    const eq = equivalentUniform(mean(act.width), curve.map((c) => [c.budget, c.width]));
    out[gid] = {
      truth: stream.weeks.map((w) => A.trueAccuracy(w.map((i) => items[i]))),
      seedTruth: A.trueAccuracy(stream.seedSet.map((i) => items[i])),
      active: { width: mean(act.width), coverage: mean(act.coverage), rmse: mean(act.rmse), labels: mean(act.labels), perWeekWidth: act.width, perWeekCoverage: act.coverage, belowRate: act.belowRate, falseAlarmsPerRun: act.falseAlarmsPerRun },
      uniformModelAssisted: { width: mean(ma.width), coverage: mean(ma.coverage), rmse: mean(ma.rmse) },
      uniform: curve,
      uniformAtBudget: curve[0],
      equivalentUniformLabels: eq,
      labelSavings: eq ? 1 - cfg.budget / eq : null,
      widthReductionSameBudget: 1 - mean(act.width) / curve[0].width,
    };
  }
  return out;
}

export function productionRun(raw, stream, gid, cfg = CONFIG) {
  const items = A.viewFor(raw, gid);
  const run = A.runMonitor(items, stream, { budget: cfg.budget, alpha: cfg.alpha, thetaMin: cfg.thetaMin, window: cfg.window, tau: cfg.tau, grader: gid, design: 'active', seed: cfg.runSeed });
  return run.weeks.map((w) => ({ week: w.week, n: w.n, theta: w.theta, lo: w.lo, hi: w.hi, se: w.se, nLab: w.nLab, win: w.win, status: w.status, toDecide: w.toDecide, truth: w.truth, windowTruth: w.windowTruth }));
}

if (process.argv[1] && process.argv[1].endsWith('build.mjs')) {
  const t0 = Date.now();
  const raw = JSON.parse(fs.readFileSync(new URL('../data/mtbench_pairs.json', import.meta.url))).items;
  const stream = makeStream(raw);
  const evals = evaluate(raw, stream);
  const runs = Object.fromEntries(Object.keys(A.GRADERS).map((g) => [g, productionRun(raw, stream, g)]));
  const r6 = (x) => (typeof x === 'number' ? Math.round(x * 1e6) / 1e6 : x);
  const clean = (o) => JSON.parse(JSON.stringify(o, (k, v) => r6(v)));
  const results = clean({ config: CONFIG, stream, evaluation: evals, runs });
  fs.writeFileSync(new URL('../data/results.json', import.meta.url), JSON.stringify(results));
  for (const [g, e] of Object.entries(evals)) {
    console.log(`${g}: active@${CONFIG.budget} width ${e.active.width.toFixed(4)} cov ${e.active.coverage.toFixed(3)} | uniform@${CONFIG.budget} width ${e.uniformAtBudget.width.toFixed(4)} cov ${e.uniformAtBudget.coverage.toFixed(3)} | uniform+MA width ${e.uniformModelAssisted.width.toFixed(4)} | equivalent uniform labels ${e.equivalentUniformLabels?.toFixed(1)} -> savings ${(100 * e.labelSavings).toFixed(1)}% | width -${(100 * e.widthReductionSameBudget).toFixed(1)}%`);
    console.log('   truth', e.truth.map((x) => x.toFixed(3)).join(' '), '| P(below) by week', e.active.belowRate.map((x) => x.toFixed(2)).join(' '), '| uniform P(below wk8)', e.uniformAtBudget.belowLast.toFixed(2));
    console.log('   run', runs[g].map((w) => `${w.week}:${w.theta.toFixed(3)}[${w.lo.toFixed(2)},${w.hi.toFixed(2)}]${w.status}`).join(' '));
  }
  console.log('built in', Date.now() - t0, 'ms');
}
