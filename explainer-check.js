
'use strict';
/* ================================================================
   RiMatch interactive explainer — live engine re-implementation.
   Mirrors rapidmatch modules: 2 profile · 5 bin/stratify · 7 score
   · 8 greedy · 9 tolerance · 11 balance · 12 drift (trim-only).
   Parity conventions copied from the Python source:
   - bin edges: quantile_cont (linear interp) on TARGET rows only
   - bucket: value <= edge_i -> bin i, else last bin
   - moments: population std (ddof=0) on target group; std 0 -> 1
   - a_k = w_k * z_k; d = sqrt(sum (a_k-b_k)^2); strength = exp(-d)
   - candidate cap: K strongest controls per target
   - greedy: stable strength-desc sort, no control reuse, n = 1
   - tolerance: cutoff = quantile(accepted strengths, tol);
     each target's strongest assignment is preserved (low_quality)
   - KS: max |F_t - F_c| over pooled support
   - JS: divergence form (not sqrt), natural log
   - drift: target-quantile bins, excess = floor((cs-ts)*n_c),
     trim weakest match_strength first, never swap in
   ================================================================ */

// reduced-motion is resolved live, not snapshotted at load — flipping the OS
// setting takes effect without a reload. RM_REDUCED is what the rest of the file reads.
const RM_QUERY = window.matchMedia('(prefers-reduced-motion: reduce)');
let RM_REDUCED = RM_QUERY.matches;
document.documentElement.classList.toggle('rm-reduced', RM_REDUCED);
RM_QUERY.addEventListener('change', e => {
  RM_REDUCED = e.matches;
  document.documentElement.classList.toggle('rm-reduced', RM_REDUCED);
  resetGlowParallax();
  clearTimersForAllScenes();
  // re-render + replay so every scene lands on its final frame instantly
  for(const n of [0, 1, 2, 3, 4, 5, 6, 7, 8]) SCENES[n].render();
  renderFooter();
  for(const n of visibleScenes){ if(n > 0 && SCENES[n]) SCENES[n].play(); }
  if(footerInView()) animateFooter();
});


function mulberry32(a){
  return function(){
    a |= 0; a = a + 0x6D2B79F5 | 0;
    let t = Math.imul(a ^ a >>> 15, 1 | a);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}

/* ---------------- population (deterministic, seeded) ---------------- */
function genPopulation(){
  const rng = mulberry32(20261001);
  const gauss = (mu, sd) => {
    let u = 0; while(u === 0) u = rng();
    const v = rng();
    return mu + sd * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  };
  const clamp = (x, a, b) => Math.min(b, Math.max(a, x));
  const rows = []; let rm = 1;
  for(let i = 1; i <= 30; i++){
    rows.push({ rm: rm++, id: 'T' + i, t: 1,
      region: rng() < 0.60 ? 'North' : 'South',
      income: Math.round(clamp(gauss(88, 9), 55, 120)),
      tenure: Math.round(clamp(gauss(7.5, 1.6), 2, 12) * 10) / 10,
      visits: Math.round(clamp(gauss(15, 3), 4, 25)) });
  }
  for(let i = 1; i <= 50; i++){
    rows.push({ rm: rm++, id: 'C' + i, t: 0,
      region: rng() < 0.45 ? 'North' : 'South',
      income: Math.round(clamp(gauss(64, 13), 30, 115)),
      tenure: Math.round(clamp(gauss(4.8, 2.3), 0.5, 12) * 10) / 10,
      visits: Math.round(clamp(gauss(9.5, 4), 1, 25)) });
  }
  return rows;
}
const POP = genPopulation();

/* ---------------- engine primitives ---------------- */
function quantileCont(sortedVals, q){
  const n = sortedVals.length;
  if(n === 0) return 0;
  if(n === 1) return sortedVals[0];
  const pos = q * (n - 1);
  const lo = Math.floor(pos), hi = Math.ceil(pos);
  const f = pos - lo;
  return sortedVals[lo] * (1 - f) + sortedVals[hi] * f;
}
function edgesOf(targetVals, nBins){
  const s = [...targetVals].sort((a, b) => a - b);
  const out = [];
  for(let i = 1; i < nBins; i++) out.push(quantileCont(s, i / nBins));
  return out;
}
function bucketOf(v, edges){
  let i = 0;
  while(i < edges.length && v > edges[i]) i++;
  return i;
}
function mean(a){ return a.reduce((x, y) => x + y, 0) / a.length; }
function pstd(a){
  const m = mean(a);
  return Math.sqrt(a.reduce((x, y) => x + (y - m) * (y - m), 0) / a.length);
}
function ksStat(a, b){
  if(!a.length || !b.length) return 0;
  const x = [...a].sort((p, q) => p - q), y = [...b].sort((p, q) => p - q);
  const vals = x.concat(y);
  let max = 0;
  const rank = (arr, v) => { // searchsorted side='right'
    let lo = 0, hi = arr.length;
    while(lo < hi){ const mid = (lo + hi) >> 1; if(arr[mid] <= v) lo = mid + 1; else hi = mid; }
    return lo;
  };
  for(const v of vals){
    const d = Math.abs(rank(x, v) / x.length - rank(y, v) / y.length);
    if(d > max) max = d;
  }
  return max;
}
function jsDistance(a, b){
  if(!a.length || !b.length) return 0;
  const labels = [...new Set(a.concat(b))].sort();
  const p = labels.map(l => a.filter(v => v === l).length / a.length);
  const q = labels.map(l => b.filter(v => v === l).length / b.length);
  const kl = (u, w) => {
    let s = 0;
    for(let i = 0; i < u.length; i++) if(u[i] > 0) s += u[i] * Math.log(u[i] / Math.max(w[i], 1e-12));
    return s;
  };
  const m = p.map((v, i) => 0.5 * (v + q[i]));
  return 0.5 * kl(p, m) + 0.5 * kl(q, m);
}

/* ---------------- full pipeline (one config in, everything out) ---------------- */
const MIN_POOL = 3; // min_control_pool_size for the demo
function runPipeline(cfg){
  const targets = POP.filter(r => r.t === 1);
  const controls = POP.filter(r => r.t === 0);

  // Module 5 — edges from target only, then stratify everyone
  const eInc = edgesOf(targets.map(r => r.income), cfg.nBins);
  const eTen = edgesOf(targets.map(r => r.tenure), cfg.nBins);
  for(const r of POP){
    r.bi = bucketOf(r.income, eInc);
    r.bt = bucketOf(r.tenure, eTen);
    r.stratum = r.bi + '-' + r.bt + '-' + r.region[0];
  }
  const strata = {};
  for(const r of POP){
    (strata[r.stratum] = strata[r.stratum] || { key: r.stratum, bi: r.bi, bt: r.bt, region: r.region, targets: [], controls: [] })
      [r.t ? 'targets' : 'controls'].push(r);
  }
  const noControl = new Set(), thin = new Set(), eligible = new Set();
  for(const k in strata){
    const s = strata[k];
    if(!s.targets.length) continue;
    if(s.controls.length === 0) noControl.add(k);
    else { eligible.add(k); if(s.controls.length < MIN_POOL) thin.add(k); }
  }

  // Module 7 — global target moments, weighted z, Euclidean, exp(-d)
  const mi = mean(targets.map(r => r.income)), si = pstd(targets.map(r => r.income)) || 1;
  const mt = mean(targets.map(r => r.tenure)), st = pstd(targets.map(r => r.tenure)) || 1;
  for(const r of POP){
    r.zi = (r.income - mi) / si;
    r.zt = (r.tenure - mt) / st;
    r.ai = cfg.wInc * r.zi;  // a_k = w_k * z_k
    r.at = cfg.wTen * r.zt;
  }
  let pairs = [];
  for(const k of eligible){
    const s = strata[k];
    for(const t of s.targets) for(const c of s.controls){
      const d = Math.sqrt((t.ai - c.ai) ** 2 + (t.at - c.at) ** 2);
      pairs.push({ t, c, d, s: Math.exp(-d), stratum: k });
    }
  }
  const totalPairs = pairs.length;

  // max_candidates_per_target — keep K strongest controls per target
  if(cfg.cap !== null){
    const byT = new Map();
    for(const p of pairs){
      if(!byT.has(p.t.rm)) byT.set(p.t.rm, []);
      byT.get(p.t.rm).push(p);
    }
    pairs = [];
    for(const arr of byT.values()){
      arr.sort((a, b) => b.s - a.s);
      pairs.push(...arr.slice(0, Math.min(cfg.cap, arr.length)));
    }
  }

  // Module 8 — global greedy, stable desc sort, no reuse, n = 1
  const sorted = [...pairs].sort((a, b) => b.s - a.s);
  const usedC = new Set(), usedT = new Set(), assignments = [];
  const walk = []; // playback log: every pair in order with its outcome
  for(const p of sorted){
    let out = 'assigned', why = '';
    if(usedT.has(p.t.rm)){ out = 'skipped'; why = p.t.id + ' already matched'; }
    else if(usedC.has(p.c.rm)){ out = 'skipped'; why = p.c.id + ' already used'; }
    if(out === 'assigned'){
      usedT.add(p.t.rm); usedC.add(p.c.rm);
      assignments.push({ t: p.t, c: p.c, s: p.s, d: p.d, rank: 1 });
    }
    walk.push({ t: p.t, c: p.c, s: p.s, d: p.d, out, why });
  }

  // Module 9 — tolerance: global quantile cutoff, primary preserved
  const strengths = assignments.map(a => a.s).sort((a, b) => a - b);
  const cutoff = strengths.length ? quantileCont(strengths, cfg.tolerance) : 0;
  let kept = assignments.filter(a => a.s >= cutoff);
  const keptT = new Set(kept.map(a => a.t.rm));
  const bestByT = new Map();
  for(const a of assignments){
    const cur = bestByT.get(a.t.rm);
    if(!cur || a.s > cur.s) bestByT.set(a.t.rm, a);
  }
  for(const [trm, a] of bestByT){
    if(!keptT.has(trm)){ kept.push(a); keptT.add(trm); }
  }
  for(const a of kept) a.lowQuality = a.s < cutoff;

  // target statuses
  const statuses = {};
  for(const t of targets){
    if(noControl.has(t.stratum)) statuses[t.rm] = 'no_control_available';
    else if(keptT.has(t.rm)) statuses[t.rm] = 'matched';
    else statuses[t.rm] = 'unmatched';
  }

  // Module 11 — balance: full target vs matched control (pre-drift)
  const matchedC = kept.map(a => a.c);
  const tInc = targets.map(r => r.income), tTen = targets.map(r => r.tenure),
        tVis = targets.map(r => r.visits), tReg = targets.map(r => r.region);
  const cInc = matchedC.map(r => r.income), cTen = matchedC.map(r => r.tenure),
        cVis = matchedC.map(r => r.visits), cReg = matchedC.map(r => r.region);
  const balBefore = {
    income: ksStat(tInc, cInc), tenure: ksStat(tTen, cTen),
    region: jsDistance(tReg, cReg), visits: ksStat(tVis, cVis),
  };

  // naive reference: seeded random pick of controls, same size as matched set
  const rng = mulberry32(7);
  const pool = [...controls];
  for(let i = pool.length - 1; i > 0; i--){ const j = Math.floor(rng() * (i + 1)); [pool[i], pool[j]] = [pool[j], pool[i]]; }
  const naive = pool.slice(0, matchedC.length);
  const balNaive = {
    income: ksStat(tInc, naive.map(r => r.income)),
    tenure: ksStat(tTen, naive.map(r => r.tenure)),
    region: jsDistance(tReg, naive.map(r => r.region)),
    visits: ksStat(tVis, naive.map(r => r.visits)),
  };

  // Module 12 — drift: monitor var (visits), target-quantile bins, weakest first
  const drift = { flagged: balBefore.visits > 0.05, events: [], keptControls: new Set(matchedC.map(c => c.rm)), bins: [] };
  const vEdges = [...new Set(edgesOf(tVis, cfg.nBins))];
  const nBinsV = vEdges.length + 1;
  const tShares = [], cShares = [];
  for(let b = 0; b < nBinsV; b++){
    tShares.push(tVis.filter(v => bucketOf(v, vEdges) === b).length / tVis.length);
    cShares.push(cVis.length ? cVis.filter(v => bucketOf(v, vEdges) === b).length / cVis.length : 0);
  }
  if(drift.flagged && matchedC.length){
    const strengthOf = new Map(kept.map(a => [a.c.rm, a.s]));
    const nC = matchedC.length;
    for(let b = 0; b < nBinsV; b++){
      const excess = Math.floor(Math.max(0, (cShares[b] - tShares[b]) * nC));
      if(excess <= 0) continue;
      const members = matchedC.filter(c => drift.keptControls.has(c.rm) && bucketOf(c.visits, vEdges) === b)
        .sort((x, y) => strengthOf.get(x.rm) - strengthOf.get(y.rm));
      const drop = members.slice(0, excess);
      drop.forEach(c => drift.keptControls.delete(c.rm));
      if(drop.length){
        drift.events.push({ bin: b, n: drop.length, ids: drop.map(c => c.id),
          targetShare: tShares[b], controlShare: cShares[b] });
        // engine re-filters the live control set per group; mirror that
        const liveVis = matchedC.filter(c => drift.keptControls.has(c.rm)).map(r => r.visits);
        for(let bb = 0; bb < nBinsV; bb++)
          cShares[bb] = liveVis.length ? liveVis.filter(v => bucketOf(v, vEdges) === bb).length / liveVis.length : 0;
      }
    }
  }
  drift.tShares = tShares;
  drift.cSharesBefore = matchedC.length
    ? Array.from({ length: nBinsV }, (_, b) => cVis.filter(v => bucketOf(v, vEdges) === b).length / cVis.length)
    : tShares.map(() => 0);
  drift.edges = vEdges;

  const finalC = matchedC.filter(c => drift.keptControls.has(c.rm));
  const balAfter = {
    visits: finalC.length ? ksStat(tVis, finalC.map(r => r.visits)) : 0,
    income: finalC.length ? ksStat(tInc, finalC.map(r => r.income)) : 0,
  };

  return {
    cfg, targets, controls, eInc, eTen, mi, si, mt, st,
    strata, eligible, noControl, thin, pairs, totalPairs, walk, assignments,
    cutoff, kept, statuses, matchedC, naive, balBefore, balNaive,
    drift, finalC, balAfter,
    nMatched: Object.values(statuses).filter(s => s === 'matched').length,
  };
}

/* ---------------- shared state & helpers ---------------- */
const DEFAULT_CFG = { wInc: 1, wTen: 1, nBins: 3, cap: null, tolerance: 0.8 };
let cfg = { ...DEFAULT_CFG };
let R = runPipeline(cfg);
let focusRm = null; // focused target for scenes 3-4

const fmt = (x, d = 2) => Number(x).toFixed(d);
const $ = id => document.getElementById(id);
const STORY_FOR_SCENE = [0, 1, 2, 3, 3, 4, 5, 6, 7];
const STORY_LABELS = ['Profile', 'Stratify', 'Standardize', 'Score', 'Allocate', 'Filter', 'Verify', 'Correct'];
function announce(message){
  const live = $('liveAnnouncer');
  if(live) live.textContent = message;
}
function updateStory(sceneNo){
  const storyNo = STORY_FOR_SCENE[sceneNo] ?? 0;
  const bar = $('storybar');
  if(bar) bar.classList.toggle('show', sceneNo > 0 || window.scrollY > window.innerHeight * 0.35);
  document.querySelectorAll('#storybar .step').forEach(step => {
    const n = +step.dataset.step;
    step.classList.toggle('active', n === storyNo);
    step.classList.toggle('done', n < storyNo);
  });
}

function ecdfPath(values, x, y, w, h, lo, hi){
  const vals = values.filter(Number.isFinite).sort((a, b) => a - b);
  if(!vals.length) return '';
  const px = v => x + (hi === lo ? 0.5 : (v - lo) / (hi - lo)) * w;
  let path = `M ${x} ${y + h}`;
  vals.forEach((v, i) => {
    const xx = Math.max(x, Math.min(x + w, px(v)));
    const yy = y + h - ((i + 1) / vals.length) * h;
    path += ` L ${xx.toFixed(2)} ${y + h - (i / vals.length) * h} L ${xx.toFixed(2)} ${yy.toFixed(2)}`;
  });
  return path + ` L ${x + w} ${y + h - h}`;
}
function ksGapPoint(a, b, x, y, w, h, lo, hi){
  const aa = a.filter(Number.isFinite).sort((p, q) => p - q);
  const bb = b.filter(Number.isFinite).sort((p, q) => p - q);
  const pooled = [...aa, ...bb].sort((p, q) => p - q);
  if(!aa.length || !bb.length || !pooled.length) return null;
  const rank = (arr, v) => { let l = 0, r = arr.length; while(l < r){ const m = (l + r) >> 1; if(arr[m] <= v) l = m + 1; else r = m; } return l; };
  let best = { d: -1, v: pooled[0], fa: 0, fb: 0 };
  pooled.forEach(v => { const fa = rank(aa, v) / aa.length, fb = rank(bb, v) / bb.length; if(Math.abs(fa - fb) > best.d) best = { d: Math.abs(fa - fb), v, fa, fb }; });
  const px = x + (hi === lo ? 0.5 : (best.v - lo) / (hi - lo)) * w;
  return { x: px, y1: y + h - best.fa * h, y2: y + h - best.fb * h, d: best.d };
}
function balanceViz(kind, targetValues, controlValues, statistic, threshold){
  if(kind === 'KS'){
    const a = targetValues.map(Number).filter(Number.isFinite), b = controlValues.map(Number).filter(Number.isFinite);
    const all = [...a, ...b];
    if(!all.length) return '<div class="hint">No values available for this plot.</div>';
    const lo = Math.min(...all), hi = Math.max(...all), x = 12, y = 8, w = 360, h = 82;
    const gap = ksGapPoint(a, b, x, y, w, h, lo, hi);
    return `<svg viewBox="0 0 390 112" role="img" aria-label="Empirical distribution comparison; KS ${fmt(statistic, 4)} against threshold ${fmt(threshold, 2)}">
      <line class="viz-axis" x1="${x}" y1="${y + h}" x2="${x + w}" y2="${y + h}"/><line class="viz-axis" x1="${x}" y1="${y}" x2="${x}" y2="${y + h}"/>
      <path class="ecdf-t" d="${ecdfPath(a, x, y, w, h, lo, hi)}"/><path class="ecdf-c" d="${ecdfPath(b, x, y, w, h, lo, hi)}"/>
      ${gap ? `<line class="ks-gap" x1="${gap.x.toFixed(2)}" y1="${gap.y1.toFixed(2)}" x2="${gap.x.toFixed(2)}" y2="${gap.y2.toFixed(2)}"/><text class="viz-label" x="${Math.min(gap.x + 5, 340)}" y="${Math.max(12, Math.min(gap.y1, gap.y2) - 5)}">KS ${fmt(gap.d, 3)}</text>` : ''}
      <text class="viz-label" x="${x}" y="106">${fmt(lo, 1)}</text><text class="viz-label" x="${x + w}" y="106" text-anchor="end">${fmt(hi, 1)}</text>
    </svg><div class="viz-legend"><span><span class="sw" style="background:var(--red-strong);"></span>full target</span><span><span class="sw" style="background:rgba(255,255,255,.74);"></span>unique control</span><span>gap = largest cumulative difference</span></div>`;
  }
  const labels = [...new Set([...targetValues, ...controlValues])].sort().slice(0, 8);
  const share = (values, label) => values.length ? values.filter(v => v === label).length / values.length : 0;
  return `<div class="cat-viz">${labels.map((label, i) => `<div class="cat-viz-row"><span class="cat-label">${label}</span><div class="cat-track"><div class="cat-fill t" style="width:${(share(targetValues, label) * 100).toFixed(1)}%;animation-delay:${i * 60}ms"></div></div><div class="cat-track"><div class="cat-fill c" style="width:${(share(controlValues, label) * 100).toFixed(1)}%;animation-delay:${i * 60 + 90}ms"></div></div></div>`).join('')}</div><div class="viz-legend"><span><span class="sw" style="background:var(--red);"></span>full target</span><span><span class="sw" style="background:rgba(255,255,255,.58);"></span>unique control</span><span>JS = ${fmt(statistic, 3)} / threshold ${fmt(threshold, 2)}</span></div>`;
}

// one live rAF per element — without this, dragging a slider fast leaves several
// loops fighting over the same textContent and the digits jitter
const numRafs = new WeakMap();
function animateNumber(el, to, dur, fmtFn){
  if(!el) return;
  if(RM_REDUCED){ el.textContent = fmtFn(to); return; }
  const prev = numRafs.get(el);
  if(prev !== undefined) cancelAnimationFrame(prev);
  const start = performance.now(), from = 0;
  function tick(now){
    const p = Math.min(1, (now - start) / dur);
    const e = 1 - Math.pow(1 - p, 3);
    el.textContent = fmtFn(from + (to - from) * e);
    if(p < 1){ numRafs.set(el, requestAnimationFrame(tick)); }
    else numRafs.delete(el);
  }
  numRafs.set(el, requestAnimationFrame(tick));
}
function after(ms, fn, bag){
  if(RM_REDUCED){ fn(); return 0; }
  const id = setTimeout(fn, ms);
  if(bag) bag.push(id);
  return id;
}
function clearTimers(bag){ if(bag){ bag.forEach(clearTimeout); bag.length = 0; } }
function clearTimersForAllScenes(){
  for(const k in sceneState){
    clearTimers(sceneState[k].timers);
    if(sceneState[k].raf) cancelAnimationFrame(sceneState[k].raf);
  }
}

// per-scene playback state
const sceneState = {};
function st(n){ return sceneState[n] = sceneState[n] || { played: false, timers: [] }; }

function pickFocusTarget(){
  // target with the most same-stratum candidates — richest scene 3 story
  let best = null, bestN = -1;
  for(const t of R.targets){
    const n = (R.strata[t.stratum] || { controls: [] }).controls.length;
    if(n > bestN){ bestN = n; best = t; }
  }
  return best;
}

/* ================================================================
   SCENES — each scene: render() builds HTML, play() animates it.
   ================================================================ */

/* ---------------- Scene 0: scan & profile ---------------- */
const SCAN_ORDER = (() => { // interleave T/C for display only
  const rng = mulberry32(99);
  const a = [...POP];
  for(let i = a.length - 1; i > 0; i--){ const j = Math.floor(rng() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; }
  return a;
})();

const SCENES = {};

SCENES[0] = {
  render(){
    $('sc0-inner').innerHTML = `
      <div class="scene-tag fade-up">Modules 1–2 · ingest &amp; profile</div>
      <h2 class="fade-up d1">First, look at the data</h2>
      <p class="lead fade-up d2">RiMatch streams the file into DuckDB and profiles it in <b>one batched pass</b> — row counts, null rates, column types, and the target/control split. Nothing is matched yet; we just need to know what we're holding.</p>
      <div class="scan-stage glass fade-up d3">
        <div class="scan-grid" id="scanGrid">${SCAN_ORDER.map(r =>
          `<div class="scan-cell" data-t="${r.t}"></div>`).join('')}</div>
        <div class="scanline" id="scanline"></div>
      </div>
      <div class="scan-stats">
        <div class="stat glass fade-up d3"><div class="v" id="stRows">0</div><div class="k">rows scanned</div></div>
        <div class="stat glass fade-up d3"><div class="v red" id="stT">0</div><div class="k">targeted (treatment = 1)</div></div>
        <div class="stat glass fade-up d4"><div class="v" id="stC">0</div><div class="k">not targeted (control pool)</div></div>
        <div class="stat glass fade-up d4"><div class="v" id="stNull">0</div><div class="k">nulls · 4 columns, 1 binary flag</div></div>
      </div>
      <p class="capacity-note fade-up d4">Control pool <b>50 ≥ 30 targets</b> — full unique 1:1 coverage is possible without reusing anyone. (If the pool were smaller, RiMatch would warn here and report the shortfall instead of silently reusing controls.)</p>`;
  },
  play(){
    const S = st(0); clearTimers(S.timers);
    const cells = [...document.querySelectorAll('#scanGrid .scan-cell')];
    cells.forEach(c => c.classList.remove('scanned', 't', 'c'));
    const line = $('scanline');
    line.classList.remove('run');
    void line.offsetWidth;
    line.classList.add('run');
    const cols = 20, rows = Math.ceil(cells.length / cols);
    cells.forEach((c, i) => {
      const row = Math.floor(i / cols);
      after(150 + row * (2100 / rows) + (i % cols) * 14, () => {
        c.classList.add('scanned', c.dataset.t === '1' ? 't' : 'c');
      }, S.timers);
    });
    after(RM_REDUCED ? 0 : 2300, () => {
      animateNumber($('stRows'), 80, 700, v => Math.round(v));
      animateNumber($('stT'), 30, 700, v => Math.round(v));
      animateNumber($('stC'), 50, 700, v => Math.round(v));
      $('stNull').textContent = '0';
    }, S.timers);
  }
};

/* ---------------- Scene 1: bin & stratify ---------------- */
function numlineSVG(id, values, edges, unit, color){
  const W = 600, H = 86, pad = 26;
  const lo = Math.min(...values) - 3, hi = Math.max(...values) + 3;
  const x = v => pad + (v - lo) / (hi - lo) * (W - 2 * pad);
  let dots = '';
  POP.forEach((r, i) => {
    const v = r[color];
    const lane = r.t ? 26 : 54;
    const jit = ((i * 37) % 13) - 6;
    dots += `<circle class="nl-dot" cx="${x(v).toFixed(1)}" cy="${lane + jit * 0.9}" r="3.6"
      fill="${r.t ? 'var(--red)' : 'rgba(255,255,255,0.30)'}"/>`;
  });
  let els = '';
  edges.forEach((e, i) => {
    els += `<line class="nl-edge" x1="${x(e).toFixed(1)}" y1="10" x2="${x(e).toFixed(1)}" y2="72"
      style="transition-delay:${0.25 + i * 0.18}s"/>
      <text class="nl-edge-label" x="${x(e).toFixed(1)}" y="82" text-anchor="middle"
      style="transition-delay:${0.5 + i * 0.18}s">${e.toFixed(unit === '$k' ? 0 : 1)}</text>`;
  });
  return `<svg class="nl" id="${id}" viewBox="0 0 ${W} ${H}">${dots}${els}</svg>`;
}

SCENES[1] = {
  render(){
    const cards = Object.values(R.strata)
      .filter(s => s.targets.length)
      .sort((a, b) => a.key < b.key ? -1 : 1);
    let ci = 0;
    const cardsHtml = cards.map(s => {
      const noCtl = s.controls.length === 0;
      const thin = !noCtl && s.controls.length < MIN_POOL;
      const dots = s.targets.map(() => '<span class="sdot t"></span>').join('') +
                   s.controls.map(() => '<span class="sdot c"></span>').join('');
      return `<div class="stratum-card ${noCtl ? 'no-ctl' : ''} ${thin ? 'thin' : ''}" style="transition-delay:${Math.min(ci++ * 0.05, 0.9)}s">
        <div class="sk">bin ${s.bi} · bin ${s.bt} · ${s.region}</div>
        <div class="cnt"><span class="rt">${s.targets.length} target${s.targets.length > 1 ? 's' : ''}</span> · ${s.controls.length} control${s.controls.length !== 1 ? 's' : ''}</div>
        <div class="sdots">${dots}</div>
        ${noCtl ? '<div class="stamp">no control available</div>' : thin ? `<div class="stamp">thin stratum (&lt; ${MIN_POOL}) — still matched</div>` : ''}
      </div>`;
    }).join('');
    $('sc1-inner').innerHTML = `
      <div class="scene-tag fade-up">Module 5 · binning &amp; stratification</div>
      <h2 class="fade-up d1">Bin on the target, group everyone</h2>
      <p class="lead fade-up d2">Numeric variables are cut into <b>${R.cfg.nBins} quantile bins</b> — and the edges are computed <b>from the target group only</b>, so the grid is defined by who was treated, never by the control pool. Those same edges are then applied to everyone. Rows that land in the same bins <b>and</b> share a region form a <b>stratum</b>; comparisons only ever happen inside one.</p>
      <div class="numline-box glass fade-up d3">
        <div class="nl-title"><span>income — ${R.eInc.length} target-derived edges</span><span class="mono">quantile_cont(income) WHERE treatment = 1</span></div>
        ${numlineSVG('nlInc', POP.map(r => r.income), R.eInc, '$k', 'income')}
      </div>
      <div class="numline-box glass fade-up d3">
        <div class="nl-title"><span>tenure — ${R.eTen.length} target-derived edges</span><span class="mono">quantile_cont(tenure) WHERE treatment = 1</span></div>
        ${numlineSVG('nlTen', POP.map(r => r.tenure), R.eTen, 'y', 'tenure')}
      </div>
      <div class="strata-block fade-up d4">
        <h3>${Object.keys(R.strata).length} strata formed · ${R.eligible.size} eligible · ${R.noControl.size} without controls · ${R.thin.size} thin</h3>
        <div class="strata-grid">${cardsHtml}</div>
      </div>`;
  },
  play(){ /* edges + cards animate via .scene.in CSS */ }
};

/* ---------------- Scene 2: z-score ---------------- */
SCENES[2] = {
  render(){
    const picks = [R.targets[0], R.targets[7], R.targets[14], R.targets[21]].filter(Boolean);
    const cards = picks.map((t, i) => `
      <div class="zcard" style="transition-delay:${0.15 + i * 0.14}s">
        <div class="who">${t.id} <span style="color:var(--ink-faint);font-weight:400;">· income $${t.income}k</span></div>
        <div><span class="mono">(${t.income} − ${fmt(R.mi, 1)}) / ${fmt(R.si, 2)}</span><span class="arrow">→</span><span class="mono">z = ${fmt(t.zi, 2)}</span></div>
      </div>`).join('');
    $('sc2-inner').innerHTML = `
      <div class="scene-tag fade-up">Module 7a · standardization</div>
      <h2 class="fade-up d1">Put dollars and years on the same ruler</h2>
      <p class="lead fade-up d2">Income spans ~$90k; tenure spans ~10 years. Raw, a $1,000 gap would silently crush a 1-year gap. So every numeric variable is z-scored — <span class="mono">z = (x − mean) / std</span> — using the mean and std of the <b>whole target group, computed once, globally</b>. Same moments for targets and controls alike: that's what makes a strength of 0.91 in one stratum comparable to 0.91 in another.</p>
      <div class="formula-box fade-up d3">
        <div class="line">income — target mean <span class="mono">${fmt(R.mi, 2)}</span> · target std <span class="mono">${fmt(R.si, 2)}</span> (population std, ddof=0)</div>
        <div class="line">tenure — target mean <span class="mono">${fmt(R.mt, 2)}</span> · target std <span class="mono">${fmt(R.st, 2)}</span></div>
      </div>
      <div class="zmorph">${cards}</div>
      <details class="why fade-up d4">
        <summary>Why not just use raw distance?</summary>
        <div class="why-body">
          <div class="why-toggle">
            <button id="btnRaw" class="active">Raw distance</button>
            <button id="btnStd">Standardized distance</button>
          </div>
          <table class="why-tbl" id="whyTable"></table>
          <p class="why-note" id="whyNote"></p>
        </div>
      </details>`;
    const whyData = {
      raw: { rows: [
        ['Target', '—', '$80,000', '2.0 yrs', ''],
        ['Candidate A', 'vs target', '$82,000 (+2,000)', '8.0 yrs (+6.0)', 'distance ~ 2000.01'],
        ['Candidate B', 'vs target', '$60,000 (+20,000)', '2.5 yrs (+0.5)', 'distance ~ 20000.00'],
      ], winner: 1, note: 'Raw distance picks Candidate A — income\'s dollar scale completely swamps tenure, even though B is nearly identical on tenure.' },
      std: { rows: [
        ['Target', '—', 'z = 0.00', 'z = 0.00', ''],
        ['Candidate A', 'vs target', 'dz ~ 0.13', 'dz ~ 2.00', 'distance ~ 2.00'],
        ['Candidate B', 'vs target', 'dz ~ 1.33', 'dz ~ 0.17', 'distance ~ 1.34'],
      ], winner: 2, note: 'Once both attributes are standardized, Candidate B wins instead — the decision flips once income can\'t dominate by raw scale alone.' }
    };
    const renderWhy = mode => {
      const d = whyData[mode];
      let html = '<thead><tr><th></th><th></th><th>Income</th><th>Tenure</th><th>Distance</th></tr></thead><tbody>';
      d.rows.forEach((r, i) => { html += `<tr class="${i === d.winner ? 'winner' : ''}"><td>${r[0]}</td><td>${r[1]}</td><td>${r[2]}</td><td>${r[3]}</td><td>${r[4]}</td></tr>`; });
      $('whyTable').innerHTML = html + '</tbody>';
      $('whyNote').textContent = d.note;
    };
    $('btnRaw').addEventListener('click', () => { $('btnRaw').classList.add('active'); $('btnStd').classList.remove('active'); renderWhy('raw'); });
    $('btnStd').addEventListener('click', () => { $('btnStd').classList.add('active'); $('btnRaw').classList.remove('active'); renderWhy('std'); });
    renderWhy('raw');
  },
  play(){}
};

/* ---------------- Scene 3: distance & strength ---------------- */
function focusTarget(){ return POP.find(r => r.rm === focusRm) || pickFocusTarget(); }
function candidatesFor(t){
  const s = R.strata[t.stratum];
  if(!s) return [];
  return s.controls.map(c => {
    const d = Math.sqrt((t.ai - c.ai) ** 2 + (t.at - c.at) ** 2);
    return { c, d, s: Math.exp(-d) };
  }).sort((a, b) => b.s - a.s);
}

SCENES[3] = {
  render(){
    const t = focusTarget();
    const cands = candidatesFor(t);
    // weighted-z scatter (a_income × a_tenure): straight-line distance here IS the score
    const W = 520, H = 380, pad = 38;
    const xs = POP.map(r => r.ai), ys = POP.map(r => r.at);
    const x0 = Math.min(...xs) - 0.3, x1 = Math.max(...xs) + 0.3;
    const y0 = Math.min(...ys) - 0.3, y1 = Math.max(...ys) + 0.3;
    const X = v => pad + (v - x0) / (x1 - x0) * (W - 2 * pad);
    const Y = v => H - pad - (v - y0) / (y1 - y0) * (H - 2 * pad);
    const candSet = new Set(cands.map(k => k.c.rm));
    let pts = '', lines = '';
    for(const r of POP){
      if(r.t === 0){
        pts += `<circle class="pt c ${candSet.has(r.rm) ? 'cand' : 'dim'}" data-c="${r.rm}" cx="${X(r.ai).toFixed(1)}" cy="${Y(r.at).toFixed(1)}" r="${candSet.has(r.rm) ? 5.5 : 4}"><title>${r.id} · ${r.region} · $${r.income}k · ${r.tenure}y</title></circle>`;
      }
    }
    for(const r of POP){
      if(r.t === 1){
        pts += `<circle class="pt t ${r.rm === t.rm ? 'focus' : ''}" data-tfocus="${r.rm}" cx="${X(r.ai).toFixed(1)}" cy="${Y(r.at).toFixed(1)}" r="${r.rm === t.rm ? 7.5 : 5}"><title>${r.id} · ${r.region} · $${r.income}k · ${r.tenure}y</title></circle>`;
      }
    }
    cands.forEach((k, i) => {
      lines += `<line class="dline" id="dl${i}" x1="${X(t.ai).toFixed(1)}" y1="${Y(t.at).toFixed(1)}" x2="${X(k.c.ai).toFixed(1)}" y2="${Y(k.c.at).toFixed(1)}" pathLength="100" stroke-dasharray="100" stroke-dashoffset="100"/>`;
    });
    const list = cands.slice(0, 8).map((k, i) => `
      <div class="cand-row ${i === 0 ? 'best' : ''}" data-i="${i}">
        <span class="cid">${k.c.id}</span>
        <div class="bar-track"><div class="bar-fill" data-w="${(k.s * 100).toFixed(0)}"></div></div>
        <span class="sv">d ${fmt(k.d)} · s ${fmt(k.s)}</span>
      </div>`).join('');
    $('sc3-inner').innerHTML = `
      <div class="scene-tag fade-up">Module 7b · scoring</div>
      <h2 class="fade-up d1">Every eligible pair gets a distance</h2>
      <p class="lead fade-up d2">Within <b>${t.id}</b>'s stratum, every control candidate is scored. Each axis below is a z-score <b>stretched by its weight</b> (<span class="mono">a = w · z</span>), so ordinary straight-line distance here <i>is</i> the weighted Euclidean distance. Then <span class="mono">strength = e<sup>−d</sup></span> relabels that gap on a 0–1 ruler: identical rows score 1, farther slides toward 0. Ranking never changes — only the scale becomes comparable across strata. <b>Click any red dot</b> to re-run this for another target.</p>
      <div class="scatter-layout">
        <div class="scatter-box glass fade-up d3">
          <svg class="scatter" viewBox="0 0 ${W} ${H}">
            <line class="axis" x1="${pad}" y1="${H - pad}" x2="${W - pad}" y2="${H - pad}"/>
            <line class="axis" x1="${pad}" y1="${pad}" x2="${pad}" y2="${H - pad}"/>
            <text x="${W / 2}" y="${H - 8}" text-anchor="middle">weighted z · income (w=${fmt(R.cfg.wInc, 1)})</text>
            <text x="12" y="${H / 2}" text-anchor="middle" transform="rotate(-90 12 ${H / 2})">weighted z · tenure (w=${fmt(R.cfg.wTen, 1)})</text>
            ${lines}${pts}
          </svg>
        </div>
        <div class="fade-up d4">
          <div class="formula-box" style="margin-top:0;">
            <div class="line"><b>${t.id}</b> · ${t.region} · $${t.income}k · ${t.tenure}y — <span class="mono">${cands.length} candidate${cands.length !== 1 ? 's' : ''} in stratum</span></div>
            <div class="line mono" style="font-size:12.5px;">d = √( (wᵢ·Δzᵢ)² + (wₜ·Δzₜ)² ) → s = e^(−d)</div>
          </div>
          <div class="cand-list" id="candList">${list || '<p class="hint">No candidates in this stratum.</p>'}</div>
        </div>
      </div>`;
    document.querySelectorAll('[data-tfocus]').forEach(el => {
      el.addEventListener('click', () => {
        focusRm = +el.dataset.tfocus;
        SCENES[3].render(); SCENES[3].play();
        SCENES[4].render(); SCENES[4].play();
      });
    });
  },
  play(){
    const S = st(3); clearTimers(S.timers);
    const cands = candidatesFor(focusTarget());
    cands.slice(0, 8).forEach((k, i) => {
      after(200 + i * 160, () => {
        const line = $('dl' + i);
        if(line){ line.style.transition = 'stroke-dashoffset 0.5s ease'; line.style.strokeDashoffset = '0'; }
        const pt = document.querySelector(`.pt[data-c="${k.c.rm}"]`);
        if(pt){
          pt.classList.add('pop');
          after(420, () => pt.classList.remove('pop'), S.timers);
        }
        const row = document.querySelector(`.cand-row[data-i="${i}"]`);
        if(row){
          row.classList.add('show');
          const fill = row.querySelector('.bar-fill');
          if(fill) fill.style.width = fill.dataset.w + '%';
        }
      }, S.timers);
    });
  }
};

/* ---------------- Scene 4: candidate cap ---------------- */
SCENES[4] = {
  render(){
    const t = focusTarget();
    const cands = candidatesFor(t);
    const K = R.cfg.cap;
    const list = cands.map((k, i) => `
      <div class="cand-row ${K !== null && i >= K ? 'cut-pending' : 'show'}" data-i="${i}">
        <span class="cid">${k.c.id}</span>
        <div class="bar-track"><div class="bar-fill" style="width:${(k.s * 100).toFixed(0)}%"></div></div>
        <span class="sv">s ${fmt(k.s)}</span>
      </div>`).join('');
    $('sc4-inner').innerHTML = `
      <div class="scene-tag fade-up">Module 7c · candidate cap <span class="mono" style="text-transform:none;">max_candidates_per_target</span></div>
      <h2 class="fade-up d1">Keep only the K closest per target</h2>
      <p class="lead fade-up d2">${K === null
        ? `The engine default is <b>no cap</b> — every pair is kept, exactly. On multi-million-row data you can set a cap (try the slider, top right) and each target keeps only its K strongest controls. A target's nearest control is never dropped, so match quality is preserved; only <b>retained memory</b> shrinks. The distance math itself is unchanged.`
        : `Cap is <b>K = ${K}</b>: each target keeps only its ${K} strongest control${K > 1 ? 's' : ''}. Watch ${t.id}'s weaker candidates fade below — they are never stored, never enter the global pool. ${t.id}'s nearest control is always safe.`}</p>
      <div class="scatter-layout">
        <div class="cand-list fade-up d3" id="capList" style="max-height:430px;overflow-y:auto;">${list}</div>
        <div class="fade-up d4">
          <div class="verify-stats" style="grid-template-columns:1fr 1fr;display:grid;gap:14px;">
            <div class="stat glass"><div class="v" id="capTotal">0</div><div class="k">pairs scored (all strata)</div></div>
            <div class="stat glass"><div class="v red" id="capKept">0</div><div class="k">pairs retained after cap</div></div>
          </div>
          <p class="hint" style="margin-top:12px;">Strata smaller than K are untouched, so everyday datasets behave byte-identically to the uncapped run.</p>
        </div>
      </div>`;
  },
  play(){
    const S = st(4); clearTimers(S.timers);
    animateNumber($('capTotal'), R.totalPairs, 800, v => Math.round(v));
    animateNumber($('capKept'), R.pairs.length, 800, v => Math.round(v));
    document.querySelectorAll('#capList .cut-pending').forEach((row, i) => {
      after(300 + i * 130, () => { row.classList.remove('cut-pending'); row.classList.add('show', 'cut'); }, S.timers);
    });
  }
};

/* ---------------- Scene 5: global greedy matching ---------------- */
SCENES[5] = {
  render(){
    const SHOW = Math.min(60, R.walk.length);
    const rows = R.walk.slice(0, SHOW).map((w, i) => `
      <div class="pair-row" data-i="${i}"><span class="rk">${i + 1}</span><span class="who">${w.t.id}–${w.c.id}</span><span class="ss">${fmt(w.s)}</span><span class="st"></span></div>`).join('');
    // bipartite layout: targets left (rm 1..30), controls right (rm 31..80)
    const W = 460, nT = R.targets.length, nC = R.controls.length;
    const H = Math.max(nT, nC) * 13 + 44;
    const yT = new Map(), yC = new Map();
    R.targets.forEach((t, i) => yT.set(t.rm, 26 + i * (H - 52) / Math.max(1, nT - 1)));
    R.controls.forEach((c, i) => yC.set(c.rm, 26 + i * (H - 52) / Math.max(1, nC - 1)));
    this._lay = { W, H, yT, yC };
    let dots = '', lines = '';
    for(const a of R.assignments){
      const y1 = yT.get(a.t.rm), y2 = yC.get(a.c.rm);
      lines += `<path class="bm-line" id="bl${a.t.rm}" d="M46 ${y1} C ${W * 0.38} ${y1}, ${W * 0.62} ${y2}, ${W - 46} ${y2}" pathLength="100" stroke-dasharray="100" stroke-dashoffset="100"/>`;
    }
    for(const t of R.targets){
      dots += `<circle class="bm-dot-t" id="bt${t.rm}" cx="40" cy="${yT.get(t.rm)}" r="4"><title>${t.id}</title></circle>`;
    }
    for(const c of R.controls){
      dots += `<circle class="bm-dot-c" id="bc${c.rm}" cx="${W - 40}" cy="${yC.get(c.rm)}" r="4"><title>${c.id}</title></circle>`;
    }
    dots += `<text x="40" y="14" text-anchor="middle">targets</text><text x="${W - 40}" y="14" text-anchor="middle">controls</text>`;
    $('sc5-inner').innerHTML = `
      <div class="scene-tag fade-up">Module 8 · global greedy matching</div>
      <h2 class="fade-up d1">Strongest pair anywhere wins, first</h2>
      <p class="lead fade-up d2">All <b>${R.pairs.length} retained pairs</b> from every stratum are pooled and sorted by strength — the whole population competes, not per-silo. The engine walks the list top-down and assigns a pair only if <b>both sides are still free</b>. A control row is used <b>at most once, ever</b> (sampling without replacement). Watch the walk happen.</p>
      <div class="play-row fade-up d3">
        <button class="btn primary" id="greedyReplay" aria-describedby="greedyNarrative">Replay the walk</button>
        <span class="live-count" id="greedyCount" aria-live="polite">0 assigned · 0 skipped · 0 / ${SHOW} pairs shown</span>
      </div>
      <div class="allocation-callout fade-up d3" id="greedyNarrative" aria-live="polite"><span class="signal"></span><span><strong>Ready.</strong> Press replay to watch the strongest available pair claim both sides.</span></div>
      <div class="greedy-layout">
        <div class="pair-list fade-up d3" id="pairList">${rows}
          ${R.walk.length > SHOW ? `<div class="pair-row" style="justify-content:center;"><span class="st">… ${R.walk.length - SHOW} more pairs fast-forwarded</span></div>` : ''}
        </div>
        <div class="bimap-box glass fade-up d4"><svg class="bimap" viewBox="0 0 ${W} ${H}">${lines}${dots}</svg></div>
      </div>`;
    $('greedyReplay').addEventListener('click', () => SCENES[5].play());
  },
  play(){
    const S = st(5); clearTimers(S.timers);
    if(S.raf) cancelAnimationFrame(S.raf);
    const SHOW = Math.min(60, R.walk.length);
    const rows = document.querySelectorAll('#pairList .pair-row[data-i]');
    rows.forEach(r => { r.classList.remove('cursor', 'done-yes', 'done-no'); r.querySelector('.st').textContent = ''; });
    document.querySelectorAll('.bm-line').forEach(l => { l.style.transition = 'none'; l.style.strokeDashoffset = '100'; });
    document.querySelectorAll('.bm-dot-c').forEach(d => d.classList.remove('used'));
    document.querySelectorAll('.bm-dot-t').forEach(d => d.classList.remove('matched'));
    const count = $('greedyCount');
    const narrative = $('greedyNarrative');
    const { yT, yC, W } = this._lay;
    let nA = 0, nS = 0;
    const upd = (message, strong = 'Walking the global strength queue.') => {
      count.textContent = `${nA} assigned · ${nS} skipped`;
      if(narrative) narrative.innerHTML = `<span class="signal"></span><span><strong>${strong}</strong> ${message}</span>`;
    };

    const revealAssign = (t, c) => {
      const line = $('bl' + t.rm);
       if(line){ line.style.transition = 'stroke-dashoffset 0.45s ease'; line.style.strokeDashoffset = '0'; line.classList.add('live'); after(560, () => line.classList.remove('live'), S.timers); }
      const dt = $('bt' + t.rm), dc = $('bc' + c.rm);
      if(dt) dt.classList.add('matched');
      if(dc) dc.classList.add('used');
    };
    const finish = () => {
      for(const a of R.assignments) revealAssign(a.t, a.c);
      rows.forEach((r, i) => {
        const w = R.walk[i];
        if(!w) return;
        r.classList.add(w.out === 'assigned' ? 'done-yes' : 'done-no');
        r.querySelector('.st').textContent = w.out === 'assigned' ? 'Assigned' : w.why;
      });
      nA = R.assignments.length; nS = R.walk.length - nA;
       count.textContent = `${nA} assigned · ${nS} skipped — done`;
       if(narrative) narrative.innerHTML = `<span class="signal"></span><span><strong>Allocation complete.</strong> ${nA} unique controls are locked to ${nA} targets; ${nS} candidate rows were rejected because one side was already claimed.</span>`;
    };
    if(RM_REDUCED){ finish(); return; }

    let i = 0;
    const stepMs = 60;
    const step = () => {
      if(i >= SHOW){ // fast-forward whatever wasn't shown
        for(const a of R.assignments) revealAssign(a.t, a.c);
        nA = R.assignments.length; nS = R.walk.length - nA;
        count.textContent = `${nA} assigned · ${nS} skipped — done`;
        if(narrative) narrative.innerHTML = `<span class="signal"></span><span><strong>Allocation complete.</strong> The visible queue is capped at ${SHOW} rows; the remaining candidates were fast-forwarded without changing the result.</span>`;
        return;
      }
      const w = R.walk[i];
      const row = rows[i];
      if(!w) return; // defensive: R was swapped under a stale chain
      rows.forEach(r => r.classList.remove('cursor'));
      if(row){
        row.classList.add('cursor');
        row.scrollIntoView({ block: 'nearest' });
        row.classList.add(w.out === 'assigned' ? 'done-yes' : 'done-no');
        row.querySelector('.st').textContent = w.out === 'assigned' ? 'Assigned' : w.why;
      }
       if(w.out === 'assigned'){ nA++; revealAssign(w.t, w.c); upd(`${w.t.id} takes ${w.c.id} at strength ${fmt(w.s)}. Both rows are now unavailable to later pairs.`, 'Assigned.'); }
       else { nS++; upd(`${w.t.id}–${w.c.id} is skipped: ${w.why}.`, 'Skipped.'); }
      i++;
      S.timers.push(setTimeout(step, stepMs));
    };
    S.timers.push(setTimeout(step, 350));
  }
};

/* ---------------- Scene 6: tolerance ---------------- */
SCENES[6] = {
  render(){
    const W = 600, H = 150, pad = 30;
    const x = s => pad + s * (W - 2 * pad);
    const jittered = R.assignments.map((a, i) => ({ a, jy: 62 + ((i * 29) % 44) - 22 }));
    const dots = jittered.map(({ a, jy }) =>
       `<circle class="strip-dot ${a.s < R.cutoff ? 'low-quality' : ''}" data-s="${a.s}" data-lq="${a.s < R.cutoff ? 1 : 0}" cx="${x(a.s).toFixed(1)}" cy="${jy}" r="5.5"
         fill="${a.s < R.cutoff ? '#E8B93D' : 'var(--red)'}"><title>${a.t.id}–${a.c.id} · ${fmt(a.s)}</title></circle>`).join('');
    const nLow = R.kept.filter(a => a.lowQuality).length;
    $('sc6-inner').innerHTML = `
      <div class="scene-tag fade-up">Module 9 · tolerance filter</div>
      <h2 class="fade-up d1">One global quality cutoff</h2>
      <p class="lead fade-up d2">Because z-scores are global, strengths can be cut <b>across all strata at once</b>. The cutoff is the <b>${fmt(R.cfg.tolerance, 2)} quantile</b> of accepted strengths — <span class="mono">${fmt(R.cutoff, 3)}</span> right now (move the tolerance slider and watch it travel). Assignments below it are filtered — <b>except</b> each target's single strongest assignment, which is always preserved so no target silently leaves the analysis. Those survivors are flagged <span class="tag">low_quality</span>, reported, never hidden.</p>
      <div class="strip-box glass fade-up d3">
        <svg class="strip" viewBox="0 0 ${W} ${H}">
          <line class="axis" x1="${pad}" y1="${H - 24}" x2="${W - pad}" y2="${H - 24}"/>
          <text x="${pad}" y="${H - 8}">0</text>
          <text x="${W - pad}" y="${H - 8}" text-anchor="end">strength 1.0 — identical</text>
          ${dots}
          <line class="cutline" id="cutline" x1="${pad}" y1="14" x2="${pad}" y2="${H - 24}"/>
          <text id="cutlabel" x="${pad}" y="10" font-size="10" fill="var(--red-strong)" font-family="IBM Plex Mono">cutoff ${fmt(R.cutoff, 3)}</text>
        </svg>
      </div>
      <div class="tol-stats">
        <div class="ls fade-up d3"><div class="v" id="tolA">0</div><div class="k">assignments from greedy</div></div>
        <div class="ls fade-up d3"><div class="v red" id="tolK">0</div><div class="k">kept after cutoff</div></div>
         <div class="ls fade-up d4"><div class="v" id="tolF">0</div><div class="k">filtered out</div></div>
         <div class="ls fade-up d4"><div class="v" id="tolL">0</div><div class="k">low_quality survivors</div></div>
         <div class="ls fade-up d4"><div class="v" id="tolP">0</div><div class="k">primary assignments protected</div></div>
       </div>`;
  },
  play(){
    const S = st(6); clearTimers(S.timers);
    const W = 600, pad = 30;
    const xCut = pad + R.cutoff * (W - 2 * pad);
    const line = $('cutline'), label = $('cutlabel');
    line.setAttribute('x1', pad); line.setAttribute('x2', pad);
    label.setAttribute('x', pad);
    const dots = [...document.querySelectorAll('.strip-dot')];
    // old code set opacity 0 and 1 in the same tick, so the stagger never ran and
    // the strip flashed instead of cascading. Hide everything, force a reflow, then
    // reveal one dot at a time.
    dots.forEach(d => {
      d.style.transition = 'opacity .4s ease, transform .4s cubic-bezier(.3,1.4,.5,1)';
      d.style.opacity = '0';
      d.style.transform = 'translateY(7px)';
    });
    void document.body.offsetHeight;
    dots.forEach((d, i) => after(140 + i * 26, () => {
      d.style.opacity = '1';
      d.style.transform = 'none';
    }, S.timers));
    animateNumber($('tolA'), R.assignments.length, 700, v => Math.round(v));
    animateNumber($('tolK'), R.kept.length, 700, v => Math.round(v));
    animateNumber($('tolF'), R.assignments.length - R.kept.length, 700, v => Math.round(v));
    animateNumber($('tolL'), R.kept.filter(a => a.lowQuality).length, 700, v => Math.round(v));
    animateNumber($('tolP'), R.kept.filter(a => a.lowQuality).length, 700, v => Math.round(v));
    after(500, () => {
      line.style.transition = RM_REDUCED ? 'none' : 'x1 0.9s cubic-bezier(.3,1,.4,1), x2 0.9s cubic-bezier(.3,1,.4,1)';
      line.setAttribute('x1', xCut); line.setAttribute('x2', xCut);
      label.setAttribute('x', Math.min(xCut + 6, W - pad - 90));
    }, S.timers);
    after(1500, () => {
      dots.forEach(d => {
        const below = +d.dataset.s < R.cutoff;
        const lq = d.dataset.lq === '1';
        d.style.transition = 'opacity 0.5s, r 0.5s';
        if(below && !lq) d.style.opacity = '0.14';
        if(lq){ d.setAttribute('stroke', '#fff'); d.setAttribute('stroke-width', '2'); }
      });
    }, S.timers);
  }
};

/* ---------------- Scene 7: verify (balance) ---------------- */
SCENES[7] = {
  render(){
    const fullTarget = R.targets;
    const uniqueControl = R.matchedC;
    const rowsDef = [
      { v: 'income', kind: 'KS', m: R.balBefore.income, n: R.balNaive.income, role: 'match var', t: fullTarget.map(r => r.income), c: uniqueControl.map(r => r.income) },
      { v: 'tenure', kind: 'KS', m: R.balBefore.tenure, n: R.balNaive.tenure, role: 'match var', t: fullTarget.map(r => r.tenure), c: uniqueControl.map(r => r.tenure) },
      { v: 'region', kind: 'JS', m: R.balBefore.region, n: R.balNaive.region, role: 'match var', t: fullTarget.map(r => r.region), c: uniqueControl.map(r => r.region) },
      { v: 'visits', kind: 'KS', m: R.balBefore.visits, n: R.balNaive.visits, role: 'monitor var — never matched on', t: fullTarget.map(r => r.visits), c: uniqueControl.map(r => r.visits) },
    ];
    const rows = rowsDef.map(r => {
      const thr = r.kind === 'KS' ? 0.05 : 0.10;
      const flagged = r.m > thr;
      const maxV = Math.max(r.m, r.n, 0.01);
       return `<div class="bal-row ${flagged ? 'flagged' : ''}">
         <div class="bh"><span class="vn">${r.v} <span style="color:var(--ink-faint);font-weight:400;font-size:12px;">· ${r.role}</span></span>
         <span class="kind">${r.kind}${flagged ? ' · <span class="flag">flagged &gt; ' + thr + '</span>' : ''}</span></div>
         <div class="bal-bars">
           <div class="bal-bar"><span class="who">matched control</span><div class="track"><div class="fill m" data-w="${(r.m / maxV * 100).toFixed(0)}"></div></div><span class="val">${fmt(r.m)}</span></div>
           <div class="bal-bar"><span class="who">random pick</span><div class="track"><div class="fill n" data-w="${(r.n / maxV * 100).toFixed(0)}"></div></div><span class="val">${fmt(r.n)}</span></div>
         </div>
         <div class="balance-viz">${balanceViz(r.kind, r.t, r.c, r.m, thr)}</div>
       </div>`;
    }).join('');
    const tMeanI = mean(R.targets.map(r => r.income));
    const mMeanI = R.matchedC.length ? mean(R.matchedC.map(r => r.income)) : 0;
    const nMeanI = mean(R.naive.map(r => r.income));
    $('sc7-inner').innerHTML = `
      <div class="scene-tag fade-up">Module 11 · balance check</div>
      <h2 class="fade-up d1">Does the control group still look like the target?</h2>
       <p class="lead fade-up d2">One number per column, computed on the <b>full target group</b> vs the available unique pseudo-controls. The visuals show the evidence behind the number: <b>KS</b> draws both empirical distributions and marks their largest cumulative gap; <b>JS</b> compares category shares. Match variables should be tight by construction. The monitor variable — <b>visits</b>, which the engine never saw during matching — is the honest test. A random same-sized pick of controls remains the naive baseline.</p>
      <div class="scan-stats fade-up d3" style="grid-template-columns:repeat(3,1fr);">
        <div class="stat glass"><div class="v">$${fmt(tMeanI, 1)}k</div><div class="k">target · mean income</div></div>
        <div class="stat glass"><div class="v red">$${fmt(mMeanI, 1)}k</div><div class="k">matched control · mean income</div></div>
        <div class="stat glass"><div class="v" style="color:var(--ink-faint);">$${fmt(nMeanI, 1)}k</div><div class="k">random pick · mean income</div></div>
      </div>
      <div class="bal-grid fade-up d4" style="margin-top:16px;">${rows}</div>
       <p class="thresh-note fade-up d4">Thresholds: KS &gt; 0.05 or JS &gt; 0.10 → flagged. ${R.drift.flagged ? '<b style="color:var(--red-strong);">A metric is flagged — scroll on to watch the correction.</b>' : 'Nothing is flagged right now — try moving the sliders to disturb the balance.'}</p>`;
  },
  play(){
    const S = st(7); clearTimers(S.timers);
    document.querySelectorAll('#sc7 .bal-bar .fill').forEach((f, i) => {
      f.style.width = '0';
      after(250 + i * 110, () => { f.style.width = f.dataset.w + '%'; }, S.timers);
    });
  }
};

/* ---------------- Scene 8: drift correction ---------------- */
SCENES[8] = {
  render(){
    const d = R.drift;
    let body;
    if(!d.flagged){
      body = `<p class="lead fade-up d2">With the current settings, <b>visits</b> came back balanced (KS ≤ 0.05), so there is nothing to trim. The engine would stop here. Want to see the corrector fire? Raise <b>n_bins</b> or lower a weight in the panel — disturbing the match usually re-introduces monitor drift.</p>
        <div class="ks-big"><div class="ls"><div class="v green">${fmt(R.balBefore.visits)}</div><div class="k">KS on visits — under threshold</div></div></div>`;
    } else {
      const bins = d.tShares.map((ts, b) => {
        const cs = d.cSharesBefore[b];
        const over = cs > ts + 1e-9;
        return `<div class="share-bin ${over ? 'over' : ''}">
          <div class="pair">
            <div class="col t" data-h="${(ts * 100).toFixed(0)}"></div>
            <div class="col c" data-h="${(cs * 100).toFixed(0)}"></div>
          </div>
          <div class="bl">${b < d.edges.length ? '≤' + d.edges[b] : '&gt;' + (d.edges[d.edges.length - 1] || 0)}</div>
        </div>`;
      }).join('');
      const events = d.events.map((e, i) => `
        <tr class="trim-row" data-i="${i}"><td>bin ≤${e.bin < d.edges.length ? d.edges[e.bin] : '∞'} visits</td><td>${(e.targetShare * 100).toFixed(0)}% vs ${(e.controlShare * 100).toFixed(0)}%</td><td>${e.ids.join(', ')}</td></tr>`).join('');
      body = `
      <p class="lead fade-up d2"><b>visits was never matched on</b>, and it drifted: some visit bands are over-represented in the matched control vs the target's share. The corrector finds those bands and <b>trims the weakest-strength matched rows first</b> — never touching strong matches, <b>never swapping in replacements</b>. Coverage can shrink; the remaining pairs never change identity.</p>
      <div class="bal-row glass fade-up d3">
        <div class="bh"><span class="vn">visits — share per target-quantile bin</span><span class="kind">yellow = over-represented</span></div>
        <div class="share-bars" id="shareBars">${bins}</div>
        <div class="legend"><span><span class="sw" style="background:var(--red);"></span>target share</span><span><span class="sw" style="background:rgba(255,255,255,0.30);"></span>matched control share</span></div>
      </div>
      <div class="ks-big fade-up d3">
        <div class="ls"><div class="v red" id="ksB">0.00</div><div class="k">KS before correction</div></div>
        <div class="ls"><div class="v green" id="ksA">0.00</div><div class="k">KS after trimming</div></div>
        <div class="ls"><div class="v" id="nTrim">0</div><div class="k">control rows trimmed</div></div>
      </div>
      <table class="drift-table fade-up d4"><thead><tr><th>over-represented group</th><th>target vs control share</th><th>trimmed (weakest first)</th></tr></thead>
      <tbody>${events || '<tr><td colspan="3">no single group over the floor — shares already even</td></tr>'}</tbody></table>
      <p class="hint fade-up d4" style="margin-top:10px;">Trim-only is a deliberate, locked design choice: backfilling a replacement would trade match quality on the primary variables to fix a secondary one. Control group: ${R.matchedC.length} → <b style="color:var(--ink);">${R.finalC.length}</b> rows.</p>`;
    }
    $('sc8-inner').innerHTML = `
      <div class="scene-tag fade-up">Module 12 · drift diagnosis &amp; correction</div>
      <h2 class="fade-up d1">Catch the drift you didn't match on</h2>
      ${body}`;
  },
  play(){
    const S = st(8); clearTimers(S.timers);
    document.querySelectorAll('#shareBars .col').forEach((c, i) => {
      c.style.height = '0';
      after(250 + i * 80, () => { c.style.height = c.dataset.h + '%'; }, S.timers);
    });
    if(R.drift.flagged){
      animateNumber($('ksB'), R.balBefore.visits, 700, v => fmt(v));
      animateNumber($('ksA'), R.balAfter.visits, 700, v => fmt(v));
      animateNumber($('nTrim'), R.drift.events.reduce((s, e) => s + e.n, 0), 700, v => Math.round(v));
      document.querySelectorAll('.trim-row').forEach((r, i) => {
        after(700 + i * 450, () => r.classList.add('trimmed'), S.timers);
      });
    }
  }
};

/* ---------------- footer ---------------- */
function renderFooter(){
  const pct = R.targets.length ? R.nMatched / R.targets.length : 0;
  const nLow = R.kept.filter(a => a.lowQuality).length;
  const nNoControl = Object.values(R.statuses).filter(s => s === 'no_control_available').length;
  const balanceChecks = [
    ['income', R.balBefore.income, 0.05], ['tenure', R.balBefore.tenure, 0.05],
    ['region', R.balBefore.region, 0.10], ['visits', R.balBefore.visits, 0.05],
  ];
  const nBalancePass = balanceChecks.filter(([, value, threshold]) => value <= threshold).length;
  const auditClass = pct >= 0.9 && nBalancePass === balanceChecks.length ? 'good' : (pct < 0.7 || nBalancePass < 2 ? 'warn' : '');
  $('footerInner').innerHTML = `
    <div class="scene-tag">Final audit · what the engine guarantees</div>
    <h2>Readable matching, accountable results</h2>
    <p class="lead">The run is complete. This scorecard keeps the important trade-offs visible: coverage can shrink when capacity or quality is limited, controls are never reused, and balance is measured against the <b>full target population</b>.</p>
    <div class="audit-grid">
      <div class="audit-card ${pct >= .9 ? 'good' : pct < .7 ? 'warn' : ''}"><div class="audit-value"><span data-num="${R.nMatched}" data-dec="0">0</span> / ${R.targets.length}</div><div class="audit-label">targets matched</div><div class="audit-detail">${(pct * 100).toFixed(0)}% coverage · ${nNoControl} no-control strata</div></div>
      <div class="audit-card good"><div class="audit-value"><span data-num="${R.finalC.length}" data-dec="0">0</span></div><div class="audit-label">unique controls retained</div><div class="audit-detail">no row reused · ${R.matchedC.length - R.finalC.length} trimmed for drift</div></div>
      <div class="audit-card ${nLow ? 'warn' : 'good'}"><div class="audit-value"><span data-num="${nLow}" data-dec="0">0</span></div><div class="audit-label">low-quality primaries</div><div class="audit-detail">protected below cutoff ${fmt(R.cutoff, 3)}</div></div>
      <div class="audit-card ${nBalancePass === balanceChecks.length ? 'good' : 'warn'}"><div class="audit-value"><span data-num="${nBalancePass}" data-dec="0">0</span> / ${balanceChecks.length}</div><div class="audit-label">balance checks pass</div><div class="audit-detail">KS / JS thresholds applied</div></div>
    </div>
    <div class="audit-checklist">
      <div class="audit-check"><span class="mark">✓</span><span>Every target row is accounted for; unmatched rows are reported, never silently dropped.</span></div>
      <div class="audit-check"><span class="mark">✓</span><span>Greedy allocation assigns each control at most once, so the pseudo-control group is unique.</span></div>
      <div class="audit-check"><span class="mark">✓</span><span>Bin edges and z-score moments are learned from the target population only.</span></div>
      <div class="audit-check ${R.drift.flagged ? 'warn' : ''}"><span class="mark">${R.drift.flagged ? '!' : '✓'}</span><span>${R.drift.flagged ? 'Monitor drift was detected and corrected by trimming weakest controls without backfilling.' : 'Monitor drift stayed below threshold; no correction was needed.'}</span></div>
    </div>
    <div class="play-row" style="margin-top:20px;"><button class="btn primary" id="footerReplay">Replay the allocation moment</button><span class="hint">or use the progress rail to revisit any stage.</span></div>
    <p>Every number on this page was computed live in your browser by a faithful re-implementation of the <span class="mono">rapidmatch</span> pipeline — target-only bin edges, global z-scores, weighted Euclidean distance, exp(−d) strengths, global greedy matching without replacement, quantile tolerance, JS/KS balance, and trim-only drift correction.</p>
    <p>Statuses: <span class="mono">matched</span> · <span class="mono">no_control_available</span> · <span class="mono">unmatched</span> — every target row is accounted for, never silently dropped.</p>`;
  $('footerReplay').addEventListener('click', () => {
    document.getElementById('sc5')?.scrollIntoView({ behavior: RM_REDUCED ? 'auto' : 'smooth', block: 'start' });
  });
}
function animateFooter(){
  document.querySelectorAll('#footerInner [data-num]').forEach(el => {
    const dec = +el.dataset.dec || 0;
    animateNumber(el, +el.dataset.num, 900, v => Number(v).toFixed(dec));
  });
}
function footerInView(){
  const f = document.querySelector('footer');
  if(!f) return false;
  const r = f.getBoundingClientRect();
  return r.top < window.innerHeight && r.bottom > 0;
}

/* ---------------- hero strip ---------------- */
function renderHero(){
  $('heroStrip').innerHTML = `
    <div class="ls"><div class="v">80</div><div class="k">customers</div></div>
    <div class="ls"><div class="v" style="color:var(--red-strong);">30 / 50</div><div class="k">target / control</div></div>
    <div class="ls"><div class="v">2 + 1</div><div class="k">numeric + categorical match vars</div></div>
    <div class="ls"><div class="v">1</div><div class="k">monitor var (visits)</div></div>`;
}

/* ---------------- recompute on slider change ---------------- */
const visibleScenes = new Set();
let swapToken = 0;
function recompute(){
  const token = ++swapToken;
  const inners = [...document.querySelectorAll('.scene-inner')];
  const foot = $('footerInner');

  const apply = () => {
    if(token !== swapToken) return; // a newer slider move already won
    // Kill every scene's pending timers FIRST. Scene 5's play() runs a self-
    // perpetuating setTimeout chain; if it re-renders while off-screen its
    // play() is never re-called, so that chain would keep indexing into the new,
    // shorter R.walk and throw on undefined.out.
    clearTimersForAllScenes();
    R = runPipeline(cfg);
    const ft = POP.find(r => r.rm === focusRm);
    if(!ft || !(R.strata[ft.stratum] || { controls: [] }).controls.length) focusRm = null;
    for(const n of [1, 2, 3, 4, 5, 6, 7, 8]) SCENES[n].render();
    renderFooter();
    requestAnimationFrame(() => {
      if(token !== swapToken) return;
      inners.forEach(e => e.classList.remove('swapping'));
      foot.classList.remove('swapping');
      for(const n of visibleScenes){ if(n > 0 && SCENES[n]) SCENES[n].play(); }
      if(footerInView()) animateFooter();
    });
    const pulse = $('pulse');
    pulse.classList.add('on');
    setTimeout(() => pulse.classList.remove('on'), 1400);
  };

  if(RM_REDUCED){ inners.forEach(e => e.classList.remove('swapping')); foot.classList.remove('swapping'); apply(); return; }
  // fade out → swap the DOM → fade back in, so a slider drag reads as a dissolve
  inners.forEach(e => e.classList.add('swapping'));
  foot.classList.add('swapping');
  setTimeout(apply, 190);
}

/* ---------------- panel wiring ---------------- */
function wirePanel(){
  const panel = $('panel');
  $('panelToggle').addEventListener('click', () => {
    const open = panel.classList.toggle('open');
    $('panelToggle').setAttribute('aria-expanded', String(open));
  });
  let deb = 0;
  const onInput = () => {
    cfg.wInc = +$('sWInc').value;
    cfg.wTen = +$('sWTen').value;
    cfg.nBins = +$('sNBins').value;
    const capV = +$('sCap').value;
    cfg.cap = capV >= 9 ? null : capV;
    cfg.tolerance = +$('sTol').value;
    $('vWInc').textContent = fmt(cfg.wInc, 1);
    $('vWTen').textContent = fmt(cfg.wTen, 1);
    $('vNBins').textContent = cfg.nBins;
    $('vCap').textContent = cfg.cap === null ? 'All' : cfg.cap;
    $('vTol').textContent = fmt(cfg.tolerance, 2);
    clearTimeout(deb);
    deb = setTimeout(recompute, 180);
  };
  ['sWInc', 'sWTen', 'sNBins', 'sCap', 'sTol'].forEach(id => $(id).addEventListener('input', onInput));
  $('resetCfg').addEventListener('click', () => {
    $('sWInc').value = 1; $('sWTen').value = 1; $('sNBins').value = 3; $('sCap').value = 9; $('sTol').value = 0.8;
    onInput();
  });
}

/* ---------------- scroll progress + scene rail ---------------- */
const RAIL_LABELS = ['scan', 'bin', 'z-score', 'score', 'cap', 'greedy', 'tolerance', 'balance', 'drift'];
const rail = $('rail'), progFill = $('progFill');
const glowA = document.querySelector('.glow.g1'), glowB = document.querySelector('.glow.g2');
RAIL_LABELS.forEach((label, i) => {
  const b = document.createElement('button');
  b.className = 'rd';
  b.type = 'button';
  b.setAttribute('aria-label', 'Jump to scene ' + i + ': ' + label);
  b.innerHTML = `<span class="tip">${String(i).padStart(2, '0')} · ${label}</span>`;
  b.addEventListener('click', () => {
    const el = document.getElementById('sc' + i);
    if(el) el.scrollIntoView({ behavior: RM_REDUCED ? 'auto' : 'smooth', block: 'start' });
  });
  rail.appendChild(b);
});
const railDots = [...rail.querySelectorAll('.rd')];

function resetGlowParallax(){
  if(glowA) glowA.style.transform = '';
  if(glowB) glowB.style.transform = '';
}
let scrollTicking = false;
function onScroll(){
  if(scrollTicking) return;
  scrollTicking = true;
  requestAnimationFrame(() => {
    const y = window.scrollY;
    const max = document.documentElement.scrollHeight - window.innerHeight;
    progFill.style.width = (max > 0 ? Math.min(1, y / max) * 100 : 0).toFixed(2) + '%';
    rail.classList.toggle('show', y > window.innerHeight * 0.45);
    $('storybar').classList.toggle('show', y > window.innerHeight * 0.35);
    if(!RM_REDUCED){ // fixed glows drift slower than the page — cheap parallax
      glowA.style.transform = `translate3d(0,${(y * 0.12).toFixed(1)}px,0)`;
      glowB.style.transform = `translate3d(0,${(-y * 0.08).toFixed(1)}px,0)`;
    }
    scrollTicking = false;
  });
}
window.addEventListener('scroll', onScroll, { passive: true });
window.addEventListener('resize', onScroll, { passive: true });
onScroll();

/* ---------------- init ---------------- */
renderHero();
for(const n of [0, 1, 2, 3, 4, 5, 6, 7, 8]) SCENES[n].render();
renderFooter();
wirePanel();
updateStory(0);

// forced reflow first, otherwise the hero's from-state is never painted and the
// whole entrance collapses into a no-op
void document.body.offsetHeight;
requestAnimationFrame(() => document.body.classList.add('booted'));

const observer = new IntersectionObserver(entries => {
  for(const e of entries){
    const n = +e.target.id.slice(2);
    if(e.isIntersecting){
      e.target.classList.add('in');
      visibleScenes.add(n);
      updateStory(n);
      SCENES[n].play();
    } else {
      visibleScenes.delete(n);
    }
    if(railDots[n]) railDots[n].classList.toggle('active', e.isIntersecting);
  }
}, { threshold: 0.28 });
document.querySelectorAll('.scene').forEach(s => observer.observe(s));

const footerObserver = new IntersectionObserver(entries => {
  for(const e of entries){
    if(e.isIntersecting){ animateFooter(); footerObserver.unobserve(e.target); }
  }
}, { threshold: 0.3 });
footerObserver.observe(document.querySelector('footer'));

// exposed for parity testing against the real rapidmatch engine
window.__RM__ = { POP, runPipeline, get R(){ return R; }, get cfg(){ return cfg; } };
