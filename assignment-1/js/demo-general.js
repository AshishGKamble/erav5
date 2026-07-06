// S1-4: Memorization vs generalization, and data closes the gap.
// A high-capacity net on tiny data drives train accuracy to ~100% while
// held-out accuracy stays low. Growing the dataset closes the gap.

import { MLP, makeRng } from "./nn.js";
import { Plot, PALETTE } from "./viz.js";

const DOMAIN = { xmin: -3.5, xmax: 3.5, ymin: -3.5, ymax: 3.5 };
const COLORS = { 0: PALETTE.red, 1: PALETTE.blue };
const SIZES = [20, 200, 2000];
const NOISE = 0.1;

// True boundary is a smooth sine wave; labels get 10% random flips.
function trueLabel(x, y) {
  return y > 1.2 * Math.sin(1.3 * x) ? 1 : 0;
}

function sample(n, seed) {
  const rng = makeRng(seed);
  const X = [], Y = [];
  for (let i = 0; i < n; i++) {
    const x = DOMAIN.xmin + rng() * (DOMAIN.xmax - DOMAIN.xmin);
    const y = DOMAIN.ymin + rng() * (DOMAIN.ymax - DOMAIN.ymin);
    let lab = trueLabel(x, y);
    if (rng() < NOISE) lab = 1 - lab;
    X.push([x, y]);
    Y.push(lab);
  }
  return { X, Y };
}

export function initGeneral(root) {
  const plots = SIZES.map((n) =>
    new Plot(root.querySelector(`[data-canvas="gen-${n}"]`), DOMAIN)
  );
  const accTrain = SIZES.map((n) => root.querySelector(`[data-out="gen-train-${n}"]`));
  const accTest = SIZES.map((n) => root.querySelector(`[data-out="gen-test-${n}"]`));
  const gapChart = new Plot(root.querySelector('[data-canvas="gen-gap"]'), {
    xmin: -0.35, xmax: 2.35, ymin: 0.45, ymax: 1.02,
  });
  const btn = root.querySelector('[data-action="gen-retrain"]');

  const test = sample(1500, 777);
  let models, trains, updates, raf, seed = 11;

  function build() {
    trains = SIZES.map((n, i) => sample(n, 100 + seed * 7 + i));
    models = SIZES.map((n, i) =>
      new MLP(
        [
          { in: 2, out: 48, act: "relu" },
          { in: 48, out: 48, act: "relu" },
          { in: 48, out: 1, act: "linear" },
        ],
        "binary",
        { seed: seed * 13 + i, lr: 0.01 }
      )
    );
    updates = 0;
  }

  function batchProb(model) {
    return (pts) => model.probs(pts).map((r) => r[0]);
  }

  function minibatch(data, size, rng) {
    if (data.X.length <= size) return data;
    const X = [], Y = [];
    for (let k = 0; k < size; k++) {
      const idx = Math.floor(rng() * data.X.length);
      X.push(data.X[idx]);
      Y.push(data.Y[idx]);
    }
    return { X, Y };
  }

  const mbRng = makeRng(42);

  function drawGap(trainAccs, testAccs) {
    const c = gapChart;
    c.clear();
    // custom axis with categorical x labels
    const ctx = c.ctx;
    ctx.strokeStyle = PALETTE.grid;
    ctx.lineWidth = 1;
    for (let g = 5; g <= 10; g++) {
      const yy = g / 10;
      const py = c.py(yy);
      ctx.beginPath();
      ctx.moveTo(c.pad.l, py);
      ctx.lineTo(c.w - c.pad.r, py);
      ctx.stroke();
      ctx.fillStyle = PALETTE.muted;
      ctx.font = "11px system-ui, sans-serif";
      ctx.textAlign = "right";
      ctx.textBaseline = "middle";
      ctx.fillText((yy * 100).toFixed(0) + "%", c.pad.l - 5, py);
    }
    ctx.strokeStyle = PALETTE.axis;
    ctx.strokeRect(c.pad.l, c.pad.t, c.w - c.pad.l - c.pad.r, c.h - c.pad.t - c.pad.b);
    ctx.fillStyle = PALETTE.muted;
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    SIZES.forEach((n, i) => ctx.fillText("n=" + n, c.px(i), c.h - c.pad.b + 4));

    // shaded gap band between train and test
    for (let i = 0; i < SIZES.length; i++) {
      const x = c.px(i);
      ctx.strokeStyle = PALETTE.muted;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(x, c.py(trainAccs[i]));
      ctx.lineTo(x, c.py(testAccs[i]));
      ctx.stroke();
      ctx.setLineDash([]);
    }
    const trainSeries = SIZES.map((n, i) => [i, trainAccs[i]]);
    const testSeries = SIZES.map((n, i) => [i, testAccs[i]]);
    c.line(trainSeries, PALETTE.blue, { width: 2 });
    c.line(testSeries, PALETTE.orange, { width: 2 });
    for (let i = 0; i < SIZES.length; i++) {
      dot(ctx, c.px(i), c.py(trainAccs[i]), PALETTE.blue);
      dot(ctx, c.px(i), c.py(testAccs[i]), PALETTE.orange);
    }
  }

  function dot(ctx, x, y, color) {
    ctx.beginPath();
    ctx.arc(x, y, 4, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();
    ctx.lineWidth = 1.4;
    ctx.strokeStyle = PALETTE.surface;
    ctx.stroke();
  }

  function draw() {
    const trAcc = [], teAcc = [];
    SIZES.forEach((n, i) => {
      const p = plots[i];
      p.clear();
      p.boundary(batchProb(models[i]), 50);
      p.axes();
      p.points(trains[i].X, trains[i].Y, COLORS, n > 500 ? 1.8 : 3.2);
      const ta = models[i].accuracy(trains[i].X, trains[i].Y);
      const va = models[i].accuracy(test.X, test.Y);
      trAcc.push(ta);
      teAcc.push(va);
      accTrain[i].textContent = (ta * 100).toFixed(0) + "%";
      accTest[i].textContent = (va * 100).toFixed(0) + "%";
    });
    drawGap(trAcc, teAcc);
  }

  function loop() {
    for (let s = 0; s < 6; s++) {
      SIZES.forEach((n, i) => {
        const mb = minibatch(trains[i], 32, mbRng);
        models[i].step(mb.X, mb.Y);
      });
      updates++;
    }
    draw();
    if (updates < 1400) raf = requestAnimationFrame(loop);
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
  window.addEventListener("resize", () => {
    plots.forEach((p) => p.resize());
    gapChart.resize();
    draw();
  });

  return { start: run };
}
