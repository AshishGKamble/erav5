// S1-1: Activations exist for a reason.
// Two concentric noisy rings, not linearly separable. A single linear layer
// gets stuck near chance; one ReLU hidden layer wraps the ring to ~99%.

import { MLP, makeRng, randn } from "./nn.js";
import { Plot, PALETTE } from "./viz.js";

const DOMAIN = { xmin: -3.2, xmax: 3.2, ymin: -3.2, ymax: 3.2 };
const COLORS = { 0: PALETTE.red, 1: PALETTE.blue };

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
  const data = makeRings(300, 7);
  const canvasL = root.querySelector('[data-canvas="rings-linear"]');
  const canvasR = root.querySelector('[data-canvas="rings-relu"]');
  const plotL = new Plot(canvasL, DOMAIN);
  const plotR = new Plot(canvasR, DOMAIN);
  const accL = root.querySelector('[data-out="rings-acc-linear"]');
  const accR = root.querySelector('[data-out="rings-acc-relu"]');
  const epochOut = root.querySelector('[data-out="rings-epoch"]');
  const btn = root.querySelector('[data-action="rings-retrain"]');

  let linear, relu, epoch, raf, seed = 1;

  function build() {
    linear = new MLP([{ in: 2, out: 1, act: "linear" }], "binary", { seed, lr: 0.08 });
    relu = new MLP(
      [
        { in: 2, out: 24, act: "relu" },
        { in: 24, out: 1, act: "linear" },
      ],
      "binary",
      { seed: seed + 100, lr: 0.03 }
    );
    epoch = 0;
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

    accL.textContent = (linear.accuracy(data.X, data.Y) * 100).toFixed(0) + "%";
    accR.textContent = (relu.accuracy(data.X, data.Y) * 100).toFixed(0) + "%";
    epochOut.textContent = epoch;
  }

  function loop() {
    for (let k = 0; k < 4; k++) {
      linear.step(data.X, data.Y);
      relu.step(data.X, data.Y);
      epoch++;
    }
    draw();
    if (epoch < 600) raf = requestAnimationFrame(loop);
    else btn.disabled = false;
  }

  function run() {
    cancelAnimationFrame(raf);
    btn.disabled = true;
    build();
    draw();
    raf = requestAnimationFrame(loop);
  }

  btn.addEventListener("click", () => {
    seed += 1;
    run();
  });
  window.addEventListener("resize", () => {
    plotL.resize();
    plotR.resize();
    draw();
  });

  return { start: run };
}
