// S1-3: Embeddings learn similarity from nothing but next-token.
// A toy grammar with categories (animals, fruits, verbs). Same-category tokens
// share next-token distributions, so a tiny embedding -> softmax next-token
// model clusters them, though similarity was never supplied.

import { MLP, makeRng, pca2 } from "./nn.js";
import { Plot, LossPlot, PALETTE } from "./viz.js";

const TOTAL = 300;

const TOKENS = ["cat", "dog", "cow", "apple", "mango", "eat", "chase", "see"];
// category id: 0 animals, 1 fruits, 2 verbs
const CAT = [0, 0, 0, 1, 1, 2, 2, 2];
const CAT_NAME = ["animal", "fruit", "verb"];
const CAT_COLOR = [PALETTE.blue, PALETTE.aqua, PALETTE.orange];
const VOCAB = TOKENS.length;
const DIM = 16;

function members(cat) {
  const out = [];
  for (let i = 0; i < VOCAB; i++) if (CAT[i] === cat) out.push(i);
  return out;
}
const ANIMALS = members(0), FRUITS = members(1), VERBS = members(2);

// Cyclic grammar: animal -> verb -> (fruit or animal), fruit -> animal.
// Same-category tokens share their next-token distribution by construction.
function makeCorpus(seed, n) {
  const rng = makeRng(seed);
  const pick = (arr) => arr[Math.floor(rng() * arr.length)];
  const X = [], Y = [];
  let cur = pick(ANIMALS);
  for (let i = 0; i < n; i++) {
    let next;
    const cat = CAT[cur];
    if (cat === 0) next = pick(VERBS);
    else if (cat === 2) next = rng() < 0.6 ? pick(FRUITS) : pick(ANIMALS);
    else next = pick(ANIMALS);
    const onehot = new Array(VOCAB).fill(0);
    onehot[cur] = 1;
    X.push(onehot);
    Y.push(next);
    cur = next;
  }
  return { X, Y };
}

function embeddings(model) {
  // First layer W is (DIM x VOCAB); token i embedding is column i.
  const W = model.layers[0].W;
  const emb = [];
  for (let i = 0; i < VOCAB; i++) {
    const v = new Array(DIM);
    for (let u = 0; u < DIM; u++) v[u] = W[u][i];
    emb.push(v);
  }
  return emb;
}

function cosine(a, b) {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return dot / (Math.sqrt(na) * Math.sqrt(nb) + 1e-9);
}

export function initEmbed(root) {
  const canvas = root.querySelector('[data-canvas="embed-scatter"]');
  const plot = new Plot(canvas, { xmin: -1, xmax: 1, ymin: -1, ymax: 1 });
  const nnOut = root.querySelector('[data-out="embed-nn"]');
  const epochOut = root.querySelector('[data-out="embed-epoch"]');
  const lossOut = root.querySelector('[data-out="embed-loss"]');
  const lossPlot = new LossPlot(root.querySelector('[data-canvas="embed-loss-curve"]'), {
    ymin: 0.1, ymax: 2.5, maxEpoch: TOTAL,
  });
  const btn = root.querySelector('[data-action="embed-retrain"]');
  const projRng = makeRng(99);

  let model, corpus, epoch, raf, seed = 5;
  let hist = [];

  function build() {
    corpus = makeCorpus(seed, 1600);
    model = new MLP(
      [
        { in: VOCAB, out: DIM, act: "linear" },
        { in: DIM, out: VOCAB, act: "linear" },
      ],
      "multiclass",
      { seed: seed + 50, lr: 0.03 }
    );
    epoch = 0;
    hist = [];
  }

  function draw() {
    const emb = embeddings(model);
    const proj = pca2(emb, projRng);
    // autoscale with padding
    let xmin = Infinity, xmax = -Infinity, ymin = Infinity, ymax = -Infinity;
    for (const p of proj) {
      xmin = Math.min(xmin, p[0]); xmax = Math.max(xmax, p[0]);
      ymin = Math.min(ymin, p[1]); ymax = Math.max(ymax, p[1]);
    }
    const padX = (xmax - xmin) * 0.25 + 0.05;
    const padY = (ymax - ymin) * 0.25 + 0.05;
    plot.domain = { xmin: xmin - padX, xmax: xmax + padX, ymin: ymin - padY, ymax: ymax + padY };

    plot.clear();
    plot.axes();
    plot.points(proj, CAT, CAT_COLOR, 6);
    for (let i = 0; i < VOCAB; i++) {
      plot.text(proj[i][0], proj[i][1], TOKENS[i], PALETTE.ink, {
        dx: 9, font: "12px system-ui, sans-serif", baseline: "middle",
      });
    }

    // nearest-neighbor agreement in embedding space
    let same = 0;
    for (let i = 0; i < VOCAB; i++) {
      let best = -1, bestSim = -Infinity;
      for (let j = 0; j < VOCAB; j++) {
        if (i === j) continue;
        const s = cosine(emb[i], emb[j]);
        if (s > bestSim) { bestSim = s; best = j; }
      }
      if (CAT[best] === CAT[i]) same++;
    }
    nnOut.textContent = `${same} / ${VOCAB}`;
    epochOut.textContent = epoch;
    if (lossOut) lossOut.textContent = (hist.length ? hist[hist.length - 1][1] : 0).toFixed(3);
    lossPlot.draw([{ color: PALETTE.blue, data: hist }]);
  }

  function loop() {
    let l = 0;
    for (let k = 0; k < 5; k++) {
      l = model.step(corpus.X, corpus.Y);
      epoch++;
    }
    hist.push([epoch, l]);
    draw();
    if (epoch < TOTAL) raf = requestAnimationFrame(loop);
    else btn.disabled = false;
  }

  function run() {
    cancelAnimationFrame(raf);
    btn.disabled = true;
    build();
    draw();
    raf = requestAnimationFrame(loop);
  }

  btn.addEventListener("click", () => { seed += 1; run(); });
  window.addEventListener("resize", () => { plot.resize(); lossPlot.resize(); draw(); });

  return { start: run };
}
