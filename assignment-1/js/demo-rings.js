// S1-1: Activations exist for a reason.
// Two concentric noisy rings, not linearly separable. Both nets are the same
// 2 -> 16 -> 1 architecture with the same optimizer. The only difference is the
// hidden activation: identity (stays linear, collapses to a line, ~chance) vs
// ReLU (bends into a closed ring, ~100%).

import { MLP, makeRng, randn } from "./nn.js";
import { Plot, LossPlot, PALETTE } from "./viz.js";

const DOMAIN = { xmin: -3.2, xmax: 3.2, ymin: -3.2, ymax: 3.2 };
const COLORS = { 0: PALETTE.red, 1: PALETTE.blue };
const TOTAL = 600;

function makeRings(n, seed) {
  const rng = makeRng(seed);
  const X = [], Y = [];
  for (let i = 0; i < n; i++) {
    const cls = i % 2;
    const base = cls === 0 ? 1.0 : 2.35;
    const r = base + randn(rng) * 0.19;
    const a = rng() * Math.PI * 2;
    X.push([r * Math.cos(a), r * Math.sin(a)]);
    Y.push(cls);
  }
  return { X, Y };
}

export function initRings(root) {
  let dataSeed = 7;
  let data = makeRings(300, dataSeed);
  const plotL = new Plot(root.querySelector('[data-canvas="rings-linear"]'), DOMAIN);
  const plotR = new Plot(root.querySelector('[data-canvas="rings-relu"]'), DOMAIN);
  const lossPlot = new LossPlot(root.querySelector('[data-canvas="rings-loss-curve"]'), {
    ymin: 0.003, ymax: 1.0, maxEpoch: TOTAL,
  });
  const out = (k) => root.querySelector(`[data-out="${k}"]`);
  const btn = root.querySelector('[data-action="rings-retrain"]');
  const btnData = root.querySelector('[data-action="rings-newdata"]');

  let linear, relu, epoch, raf, seed = 1;
  let histLin = [], histRelu = [];

  function build() {
    linear = new MLP(
      [{ in: 2, out: 16, act: "linear" }, { in: 16, out: 1, act: "linear" }],
      "binary", { seed, lr: 0.03 }
    );
    relu = new MLP(
      [{ in: 2, out: 16, act: "relu" }, { in: 16, out: 1, act: "linear" }],
      "binary", { seed: seed + 100, lr: 0.03 }
    );
    epoch = 0;
    histLin = [];
    histRelu = [];
  }

  function batchProb(model) {
    return (pts) => model.probs(pts).map((r) => r[0]);
  }

  function draw() {
    plotL.clear();
    plotL.boundary(batchProb(linear), 54);
    plotL.axes();
    plotL.points(data.X, data.Y, COLORS);

    plotR.clear();
    plotR.boundary(batchProb(relu), 54);
    plotR.axes();
    plotR.points(data.X, data.Y, COLORS);

    out("rings-acc-linear").textContent = (linear.accuracy(data.X, data.Y) * 100).toFixed(1) + "%";
    out("rings-acc-relu").textContent = (relu.accuracy(data.X, data.Y) * 100).toFixed(1) + "%";
    out("rings-loss-linear").textContent = (histLin.length ? histLin[histLin.length - 1][1] : linear.loss(data.X, data.Y)).toFixed(3);
    out("rings-loss-relu").textContent = (histRelu.length ? histRelu[histRelu.length - 1][1] : relu.loss(data.X, data.Y)).toFixed(3);
    out("rings-epoch-linear").textContent = epoch;
    out("rings-epoch-relu").textContent = epoch;

    lossPlot.draw([
      { color: PALETTE.red, data: histLin },
      { color: PALETTE.aqua, data: histRelu },
    ]);
  }

  function loop() {
    let ll = 0, lr = 0;
    for (let k = 0; k < 4; k++) {
      ll = linear.step(data.X, data.Y);
      lr = relu.step(data.X, data.Y);
      epoch++;
    }
    histLin.push([epoch, ll]);
    histRelu.push([epoch, lr]);
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
  if (btnData) btnData.addEventListener("click", () => {
    dataSeed += 1;
    data = makeRings(300, dataSeed);
    seed += 1;
    run();
  });
  window.addEventListener("resize", () => {
    plotL.resize(); plotR.resize(); lossPlot.resize(); draw();
  });

  return { start: run };
}
