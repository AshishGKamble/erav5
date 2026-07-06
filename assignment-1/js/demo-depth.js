// S1-2: Depth without nonlinearity is a lie.
// Five stacked linear layers collapse to one linear map, so a 5-layer linear
// net is no stronger than 1 layer. Both fail the ring task identically.
// Inserting ReLUs between the same five layers suddenly solves it. We also
// multiply the five weight matrices numerically to show the product is one map.

import { MLP, makeRng, randn, collapseLinear } from "./nn.js";
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

const FIVE = [8, 8, 8, 8];

export function initDepth(root) {
  const data = makeRings(300, 7);
  const plots = {
    one: new Plot(root.querySelector('[data-canvas="depth-one"]'), DOMAIN),
    lin: new Plot(root.querySelector('[data-canvas="depth-linear5"]'), DOMAIN),
    relu: new Plot(root.querySelector('[data-canvas="depth-relu5"]'), DOMAIN),
  };
  const out = {
    one: root.querySelector('[data-out="depth-acc-one"]'),
    lin: root.querySelector('[data-out="depth-acc-linear5"]'),
    relu: root.querySelector('[data-out="depth-acc-relu5"]'),
    product: root.querySelector('[data-out="depth-product"]'),
  };
  const btn = root.querySelector('[data-action="depth-retrain"]');

  let one, lin, relu, epoch, raf, seed = 3;

  function fiveLinear(s) {
    const defs = [];
    let prev = 2;
    for (const u of FIVE) {
      defs.push({ in: prev, out: u, act: "linear" });
      prev = u;
    }
    defs.push({ in: prev, out: 1, act: "linear" });
    return new MLP(defs, "binary", { seed: s, lr: 0.04 });
  }

  function fiveRelu(s) {
    const defs = [];
    let prev = 2;
    for (const u of FIVE) {
      defs.push({ in: prev, out: u, act: "relu" });
      prev = u;
    }
    defs.push({ in: prev, out: 1, act: "linear" });
    return new MLP(defs, "binary", { seed: s, lr: 0.02 });
  }

  function build() {
    one = new MLP([{ in: 2, out: 1, act: "linear" }], "binary", { seed, lr: 0.08 });
    lin = fiveLinear(seed + 10);
    relu = fiveRelu(seed + 20);
    epoch = 0;
  }

  function batchProb(model) {
    return (pts) => model.probs(pts).map((r) => r[0]);
  }

  function drawProduct() {
    // Collapse the five linear layers into one effective 1x2 matrix + bias.
    const { W, b } = collapseLinear(lin.layers);
    const w0 = W[0][0].toFixed(3);
    const w1 = W[0][1].toFixed(3);
    const bb = b[0].toFixed(3);
    out.product.innerHTML =
      `W&#8325;·W&#8324;·W&#8323;·W&#8322;·W&#8321; = ` +
      `<span class="mono">[ ${w0}, ${w1} ]</span>` +
      `  &nbsp; b = <span class="mono">${bb}</span>`;
  }

  function draw() {
    for (const [k, p] of Object.entries(plots)) {
      const model = k === "one" ? one : k === "lin" ? lin : relu;
      p.clear();
      p.boundary(batchProb(model), 54);
      p.axes();
      p.points(data.X, data.Y, COLORS);
    }
    out.one.textContent = (one.accuracy(data.X, data.Y) * 100).toFixed(0) + "%";
    out.lin.textContent = (lin.accuracy(data.X, data.Y) * 100).toFixed(0) + "%";
    out.relu.textContent = (relu.accuracy(data.X, data.Y) * 100).toFixed(0) + "%";
    drawProduct();
  }

  function loop() {
    for (let k = 0; k < 4; k++) {
      one.step(data.X, data.Y);
      lin.step(data.X, data.Y);
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
    for (const p of Object.values(plots)) p.resize();
    draw();
  });

  return { start: run };
}
