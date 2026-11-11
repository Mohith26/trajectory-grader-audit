import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import * as A from '../src/audit.js';
import { CONFIG, makeStream, evaluate, productionRun } from '../scripts/build.mjs';

const raw = JSON.parse(fs.readFileSync(new URL('../data/mtbench_pairs.json', import.meta.url))).items;
const results = JSON.parse(fs.readFileSync(new URL('../data/results.json', import.meta.url)));
const stream = makeStream(raw);
const scalar = A.viewFor(raw, 'scalar');

test('data: 1,616 real MT-Bench pairs, expert label is the majority of recorded votes', () => {
  assert.equal(raw.length, 1616);
  assert.equal(raw.filter((i) => i.expert !== 'tie').length, 1274);
  for (const it of raw) {
    const [a, b] = it.votes;
    assert.equal(it.expert, a > b ? 'a' : b > a ? 'b' : 'tie');
    assert.ok(it.sa >= 1 && it.sa <= 10 && it.sb >= 1 && it.sb <= 10);
  }
});

test('stream: deterministic, disjoint, covers every pair once', () => {
  const s2 = makeStream(raw);
  assert.deepEqual(s2, stream);
  assert.deepEqual(results.stream, stream);
  const all = [...stream.seedSet, ...stream.weeks.flat()];
  assert.equal(all.length, raw.length);
  assert.equal(new Set(all).size, raw.length);
  assert.equal(stream.weeks.length, CONFIG.weeks);
});

test('the scalar grader really drifts on this stream (hindsight truth from all votes)', () => {
  const truth = stream.weeks.map((w) => A.trueAccuracy(w.map((i) => scalar[i])));
  assert.ok(truth.slice(0, 3).every((t) => t > 0.76), truth.join());
  assert.ok(truth.slice(6).every((t) => t < 0.70), truth.join());
});

test('capToBudget: sums to the budget, never exceeds 1 (randomized)', () => {
  const r = A.mulberry32(5);
  for (let rep = 0; rep < 200; rep++) {
    const n = 5 + Math.floor(r() * 200);
    const scores = Array.from({ length: n }, () => (r() < 0.1 ? 0 : r() ** 3 * 10));
    const B = 1 + Math.floor(r() * n);
    const pi = A.capToBudget(scores, B);
    assert.ok(pi.every((p) => p >= 0 && p <= 1 + 1e-12));
    assert.ok(Math.abs(pi.reduce((s, p) => s + p, 0) - B) < 1e-6);
  }
});

test('census: with every pair labelled the estimate equals the true ratio, whatever the predictor', () => {
  const items = stream.weeks[0].map((i) => scalar[i]);
  const pi = items.map(() => 1), xi = items.map(() => 1);
  const r = A.mulberry32(9);
  const preds = items.map(() => { const pD = r(); return { pD, pA: pD * r() }; });
  const est = A.estimate(items, pi, xi, items.map((i) => i.expert), preds);
  assert.ok(Math.abs(est.theta - A.trueAccuracy(items)) < 1e-12);
  assert.equal(est.se, 0);
});

test('difference estimator is design-unbiased for both sums (randomized property test)', () => {
  const r = A.mulberry32(11);
  for (let rep = 0; rep < 6; rep++) {
    const items = Array.from({ length: 60 }, (_, i) => scalar[Math.floor(r() * scalar.length)]);
    const preds = items.map(() => { const pD = r(); return { pD, pA: pD * r() }; }); // deliberately bad predictor
    const pi = A.capToBudget(items.map(() => 0.2 + r()), 15);
    let sA = 0, sD = 0; const R = 6000;
    for (let k = 0; k < R; k++) {
      const xi = A.poissonSample(pi, r);
      for (let i = 0; i < items.length; i++) {
        const { A: a, D: d } = A.outcome(items[i]);
        sA += preds[i].pA + (xi[i] ? (a - preds[i].pA) / pi[i] : 0);
        sD += preds[i].pD + (xi[i] ? (d - preds[i].pD) / pi[i] : 0);
      }
    }
    const tA = items.reduce((s, it) => s + A.outcome(it).A, 0), tD = items.reduce((s, it) => s + A.outcome(it).D, 0);
    assert.ok(Math.abs(sA / R - tA) < 0.6, `A ${sA / R} vs ${tA}`);
    assert.ok(Math.abs(sD / R - tD) < 0.6, `D ${sD / R} vs ${tD}`);
  }
});

test('predictor: grader ties can never strictly agree; probabilities stay in [0, 1]', () => {
  const g = A.GRADERS.scalar;
  const pred = A.fitPredictor(stream.seedSet.map((i) => ({ item: scalar[i], expert: scalar[i].expert })), { nBuckets: g.nBuckets, zeroAgree: g.zeroAgree });
  assert.equal(pred.pG[0], 0);
  for (const it of scalar) { const p = A.predict(pred, it); assert.ok(p.pD >= 0 && p.pD <= 1 && p.pA >= 0 && p.pA <= p.pD); }
});

test('top-up keeps each pair at its new inclusion probability (randomized)', () => {
  const items = stream.weeks[7].map((i) => scalar[i]);
  const g = A.GRADERS.scalar;
  const pred = A.fitPredictor(stream.seedSet.map((i) => ({ item: scalar[i], expert: scalar[i].expert })), { nBuckets: g.nBuckets, zeroAgree: g.zeroAgree });
  const pi = A.activeDesign(items, pred, 0.75, 40);
  const r = A.mulberry32(3);
  const counts = items.map(() => 0); const R = 4000; let pi2;
  for (let k = 0; k < R; k++) {
    const xi = A.poissonSample(pi, r);
    const t = A.topUp(items, pi, xi, pred, 0.75, 30, r);
    pi2 = t.pi;
    t.xi.forEach((x, i) => (counts[i] += x));
  }
  assert.ok(pi2.every((p, i) => p >= pi[i] - 1e-12));
  assert.ok(Math.abs(pi2.reduce((s, p) => s + p, 0) - 70) < 1.5);
  const worst = Math.max(...counts.map((c, i) => Math.abs(c / R - pi2[i])));
  assert.ok(worst < 0.035, `max deviation ${worst}`);
});

test('t quantile matches tables', () => {
  assert.ok(Math.abs(A.tQuantile(0.95, 10) - 1.812) < 0.003);
  assert.ok(Math.abs(A.tQuantile(0.975, 30) - 2.042) < 0.003);
  assert.ok(Math.abs(A.normalQuantile(0.975) - 1.959964) < 1e-5);
});

test('status and labels-to-decide are consistent', () => {
  const est = { theta: 0.62, se: 0.03, nLab: 40 };
  assert.equal(A.status(est, 0.74, 0.05), 'below');
  assert.equal(A.labelsToDecide(est, 0.74, 0.05), 0);
  const unsure = { theta: 0.71, se: 0.04, nLab: 40 };
  assert.equal(A.status(unsure, 0.74, 0.05), 'inconclusive');
  const n = A.labelsToDecide(unsure, 0.74, 0.05);
  const z = A.tQuantile(0.95, 39);
  const seAfter = unsure.se * Math.sqrt(40 / (40 + n));
  assert.ok(z * seAfter <= 0.03 + 1e-9 && n > 0);
});

test('90% intervals cover the weekly truth about 90% of the time (300 runs)', () => {
  const mc = A.monteCarlo(scalar, stream, { budget: 40, alpha: 0.1, thetaMin: 0.74, window: 2, tau: 0.2, grader: 'scalar', design: 'active' }, 300, 5000);
  const cov = mc.coverage.reduce((s, x) => s + x, 0) / mc.coverage.length;
  assert.ok(cov > 0.86 && cov < 0.94, `coverage ${cov}`);
});

test('active audit is tighter than random labels at the same budget, for both graders', () => {
  for (const g of ['scalar', 'pairwise']) {
    const items = A.viewFor(raw, g);
    const base = { budget: 40, alpha: 0.1, thetaMin: 0.74, window: 2, tau: 0.2, grader: g };
    const act = A.monteCarlo(items, stream, { ...base, design: 'active' }, 150, 7000);
    const uni = A.monteCarlo(items, stream, { ...base, design: 'uniform' }, 150, 7000);
    const m = (a) => a.reduce((s, x) => s + x, 0) / a.length;
    assert.ok(m(act.width) < 0.9 * m(uni.width), `${g}: ${m(act.width)} vs ${m(uni.width)}`);
  }
});

test('headline: committed results reproduce exactly, and the label saving is 34%', () => {
  const ev = evaluate(raw, stream, CONFIG, CONFIG.reps);
  const committed = results.evaluation.scalar;
  assert.ok(Math.abs(ev.scalar.labelSavings - committed.labelSavings) < 1e-6);
  assert.ok(Math.abs(ev.scalar.equivalentUniformLabels - committed.equivalentUniformLabels) < 1e-4);
  assert.equal(Math.round(100 * committed.labelSavings), 34);
  assert.ok(Math.abs(ev.pairwise.labelSavings - results.evaluation.pairwise.labelSavings) < 1e-6);
});

test('the illustrated production run is the committed one', () => {
  const run = productionRun(raw, stream, 'scalar');
  run.forEach((w, i) => {
    assert.ok(Math.abs(w.theta - results.runs.scalar[i].theta) < 1e-6);
    assert.equal(w.status, results.runs.scalar[i].status);
  });
});
