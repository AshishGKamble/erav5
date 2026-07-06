// Minimal neural network library in plain JavaScript.
// Matrices are represented as arrays of arrays (row-major). No dependencies.

// --- Seeded RNG (mulberry32) for reproducible demos ---
export function makeRng(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Standard normal via Box-Muller, driven by a uniform rng.
export function randn(rng) {
  let u = 0, v = 0;
  while (u === 0) u = rng();
  while (v === 0) v = rng();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

// --- Matrix helpers ---
export function zeros(rows, cols) {
  const m = new Array(rows);
  for (let i = 0; i < rows; i++) m[i] = new Array(cols).fill(0);
  return m;
}

export function matmul(A, B) {
  const n = A.length, k = A[0].length, p = B[0].length;
  const C = zeros(n, p);
  for (let i = 0; i < n; i++) {
    const Ai = A[i], Ci = C[i];
    for (let t = 0; t < k; t++) {
      const a = Ai[t], Bt = B[t];
      if (a === 0) continue;
      for (let j = 0; j < p; j++) Ci[j] += a * Bt[j];
    }
  }
  return C;
}

export function transpose(A) {
  const n = A.length, m = A[0].length;
  const T = zeros(m, n);
  for (let i = 0; i < n; i++)
    for (let j = 0; j < m; j++) T[j][i] = A[i][j];
  return T;
}

// --- Activation functions and their derivatives (given pre-activation z) ---
const ACTS = {
  relu: {
    fn: (z) => z.map((row) => row.map((v) => (v > 0 ? v : 0))),
    dz: (z) => z.map((row) => row.map((v) => (v > 0 ? 1 : 0))),
  },
  tanh: {
    fn: (z) => z.map((row) => row.map((v) => Math.tanh(v))),
    dz: (z) => z.map((row) => row.map((v) => 1 - Math.tanh(v) ** 2)),
  },
  linear: {
    fn: (z) => z.map((row) => row.slice()),
    dz: (z) => z.map((row) => row.map(() => 1)),
  },
};

function sigmoid(x) {
  return x >= 0 ? 1 / (1 + Math.exp(-x)) : Math.exp(x) / (1 + Math.exp(x));
}

function softmaxRow(row) {
  let mx = -Infinity;
  for (const v of row) if (v > mx) mx = v;
  let sum = 0;
  const out = row.map((v) => {
    const e = Math.exp(v - mx);
    sum += e;
    return e;
  });
  return out.map((e) => e / sum);
}

// A dense layer: y = act(x W^T + b). W is (units x inputDim).
class Dense {
  constructor(inputDim, units, activation, rng) {
    this.activation = activation;
    // He/Xavier style init scaled by fan-in.
    const scale = Math.sqrt(2 / inputDim);
    this.W = zeros(units, inputDim);
    this.b = new Array(units).fill(0);
    for (let i = 0; i < units; i++)
      for (let j = 0; j < inputDim; j++) this.W[i][j] = randn(rng) * scale;
    // Adam state.
    this.mW = zeros(units, inputDim);
    this.vW = zeros(units, inputDim);
    this.mb = new Array(units).fill(0);
    this.vb = new Array(units).fill(0);
  }

  forward(X) {
    // X: (N x inputDim) -> Z: (N x units)
    const N = X.length, units = this.W.length, inDim = this.W[0].length;
    const Z = zeros(N, units);
    for (let n = 0; n < N; n++) {
      const xn = X[n], Zn = Z[n];
      for (let u = 0; u < units; u++) {
        const Wu = this.W[u];
        let s = this.b[u];
        for (let j = 0; j < inDim; j++) s += xn[j] * Wu[j];
        Zn[u] = s;
      }
    }
    this.lastX = X;
    this.lastZ = Z;
    const act = ACTS[this.activation];
    this.lastA = act ? act.fn(Z) : Z.map((r) => r.slice());
    return this.lastA;
  }
}

// A stack of dense layers with a task-specific output.
// task: "binary" (sigmoid + BCE) or "multiclass" (softmax + cross-entropy).
export class MLP {
  constructor(layerDefs, task, { seed = 1, lr = 0.05 } = {}) {
    this.rng = makeRng(seed);
    this.task = task;
    this.lr = lr;
    this.layers = [];
    for (const def of layerDefs) {
      this.layers.push(new Dense(def.in, def.out, def.act, this.rng));
    }
    this.t = 0;
  }

  // Forward through hidden stack, returns raw logits of the last layer.
  logits(X) {
    let h = X;
    for (const layer of this.layers) h = layer.forward(h);
    return h;
  }

  probs(X) {
    const logit = this.logits(X);
    if (this.task === "binary") return logit.map((row) => [sigmoid(row[0])]);
    return logit.map((row) => softmaxRow(row));
  }

  // One Adam gradient step over the full batch. Returns mean loss.
  step(X, Y) {
    this.t += 1;
    const N = X.length;
    const logit = this.logits(X);

    // dL/dLogit for the chosen task (loss already includes softmax/sigmoid).
    let dLogit, loss = 0;
    if (this.task === "binary") {
      dLogit = zeros(N, 1);
      for (let n = 0; n < N; n++) {
        const p = sigmoid(logit[n][0]);
        const y = Y[n];
        const pc = Math.min(Math.max(p, 1e-7), 1 - 1e-7);
        loss += -(y * Math.log(pc) + (1 - y) * Math.log(1 - pc));
        dLogit[n][0] = (p - y) / N;
      }
    } else {
      const K = logit[0].length;
      dLogit = zeros(N, K);
      for (let n = 0; n < N; n++) {
        const p = softmaxRow(logit[n]);
        const y = Y[n];
        loss += -Math.log(Math.max(p[y], 1e-7));
        for (let k = 0; k < K; k++) dLogit[n][k] = (p[k] - (k === y ? 1 : 0)) / N;
      }
    }
    loss /= N;

    // Backprop through the dense stack.
    let dA = dLogit;
    for (let li = this.layers.length - 1; li >= 0; li--) {
      const layer = this.layers[li];
      const act = ACTS[layer.activation];
      // For the output layer the loss gradient is already wrt logits, so we do
      // not multiply by the activation derivative there.
      let dZ;
      if (li === this.layers.length - 1) {
        dZ = dA;
      } else {
        const g = act.dz(layer.lastZ);
        dZ = dA.map((row, i) => row.map((v, j) => v * g[i][j]));
      }
      // Gradients wrt W and b.
      const X = layer.lastX;
      const units = layer.W.length, inDim = layer.W[0].length;
      const gW = zeros(units, inDim);
      const gb = new Array(units).fill(0);
      for (let n = 0; n < N; n++) {
        const dZn = dZ[n], Xn = X[n];
        for (let u = 0; u < units; u++) {
          const d = dZn[u];
          if (d === 0) continue;
          gb[u] += d;
          const gWu = gW[u];
          for (let j = 0; j < inDim; j++) gWu[j] += d * Xn[j];
        }
      }
      // dA for the previous layer = dZ * W.
      const dPrev = zeros(N, inDim);
      for (let n = 0; n < N; n++) {
        const dZn = dZ[n], dP = dPrev[n];
        for (let u = 0; u < units; u++) {
          const d = dZn[u];
          if (d === 0) continue;
          const Wu = layer.W[u];
          for (let j = 0; j < inDim; j++) dP[j] += d * Wu[j];
        }
      }
      this._adam(layer, gW, gb);
      dA = dPrev;
    }
    return loss;
  }

  _adam(layer, gW, gb) {
    const b1 = 0.9, b2 = 0.999, eps = 1e-8, lr = this.lr;
    const bc1 = 1 - Math.pow(b1, this.t);
    const bc2 = 1 - Math.pow(b2, this.t);
    const units = layer.W.length, inDim = layer.W[0].length;
    for (let u = 0; u < units; u++) {
      for (let j = 0; j < inDim; j++) {
        const g = gW[u][j];
        layer.mW[u][j] = b1 * layer.mW[u][j] + (1 - b1) * g;
        layer.vW[u][j] = b2 * layer.vW[u][j] + (1 - b2) * g * g;
        const mh = layer.mW[u][j] / bc1;
        const vh = layer.vW[u][j] / bc2;
        layer.W[u][j] -= (lr * mh) / (Math.sqrt(vh) + eps);
      }
      const g = gb[u];
      layer.mb[u] = b1 * layer.mb[u] + (1 - b1) * g;
      layer.vb[u] = b2 * layer.vb[u] + (1 - b2) * g * g;
      const mh = layer.mb[u] / bc1;
      const vh = layer.vb[u] / bc2;
      layer.b[u] -= (lr * mh) / (Math.sqrt(vh) + eps);
    }
  }

  // Classification accuracy against integer/binary labels.
  accuracy(X, Y) {
    const P = this.probs(X);
    let correct = 0;
    for (let n = 0; n < P.length; n++) {
      let pred;
      if (this.task === "binary") pred = P[n][0] >= 0.5 ? 1 : 0;
      else {
        let best = 0;
        for (let k = 1; k < P[n].length; k++) if (P[n][k] > P[n][best]) best = k;
        pred = best;
      }
      if (pred === Y[n]) correct++;
    }
    return correct / P.length;
  }

  // Mean loss without a gradient step (for held-out evaluation).
  loss(X, Y) {
    const logit = this.logits(X);
    let loss = 0;
    const N = X.length;
    if (this.task === "binary") {
      for (let n = 0; n < N; n++) {
        const p = Math.min(Math.max(sigmoid(logit[n][0]), 1e-7), 1 - 1e-7);
        const y = Y[n];
        loss += -(y * Math.log(p) + (1 - y) * Math.log(1 - p));
      }
    } else {
      for (let n = 0; n < N; n++) {
        const p = softmaxRow(logit[n]);
        loss += -Math.log(Math.max(p[Y[n]], 1e-7));
      }
    }
    return loss / N;
  }
}

// Collapse a stack of linear layers (list of Dense) into one effective matrix
// and bias, proving that depth without nonlinearity is a single linear map.
export function collapseLinear(layers) {
  // Effective transform: y = W_eff x + b_eff, applying layers in order.
  let Weff = null, beff = null;
  for (const layer of layers) {
    const W = layer.W; // (units x inDim)
    const b = layer.b.slice();
    if (Weff === null) {
      Weff = W.map((r) => r.slice());
      beff = b;
    } else {
      // new W = W * Weff ; new b = W * beff + b
      Weff = matmul(W, Weff);
      const nb = new Array(W.length).fill(0);
      for (let u = 0; u < W.length; u++) {
        let s = b[u];
        for (let j = 0; j < W[u].length; j++) s += W[u][j] * beff[j];
        nb[u] = s;
      }
      beff = nb;
    }
  }
  return { W: Weff, b: beff };
}

// --- PCA to 2D via power iteration on the covariance matrix ---
export function pca2(vectors, rng) {
  const N = vectors.length, D = vectors[0].length;
  const mean = new Array(D).fill(0);
  for (const v of vectors) for (let j = 0; j < D; j++) mean[j] += v[j] / N;
  const centered = vectors.map((v) => v.map((x, j) => x - mean[j]));
  // Covariance (D x D).
  const cov = zeros(D, D);
  for (const v of centered)
    for (let i = 0; i < D; i++)
      for (let j = 0; j < D; j++) cov[i][j] += (v[i] * v[j]) / N;

  function topEigen(mat, exclude) {
    let vec = new Array(D).fill(0).map(() => randn(rng));
    for (let iter = 0; iter < 200; iter++) {
      let nv = new Array(D).fill(0);
      for (let i = 0; i < D; i++)
        for (let j = 0; j < D; j++) nv[i] += mat[i][j] * vec[j];
      // Deflate against previously found eigenvectors.
      for (const e of exclude) {
        let dot = 0;
        for (let i = 0; i < D; i++) dot += nv[i] * e[i];
        for (let i = 0; i < D; i++) nv[i] -= dot * e[i];
      }
      let norm = Math.sqrt(nv.reduce((s, x) => s + x * x, 0)) || 1;
      vec = nv.map((x) => x / norm);
    }
    return vec;
  }

  const e1 = topEigen(cov, []);
  const e2 = topEigen(cov, [e1]);
  return centered.map((v) => [
    v.reduce((s, x, j) => s + x * e1[j], 0),
    v.reduce((s, x, j) => s + x * e2[j], 0),
  ]);
}
