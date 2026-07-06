// Shared canvas plotting helpers. Clean scientific light theme.
// Colors come from the validated data-viz reference palette.

export const PALETTE = {
  surface: "#fcfcfb",
  plane: "#f9f9f7",
  ink: "#0b0b0b",
  secondary: "#52514e",
  muted: "#898781",
  grid: "#e1e0d9",
  axis: "#c3c2b7",
  // categorical slots used across demos
  blue: "#2a78d6",
  red: "#e34948",
  orange: "#eb6834",
  violet: "#4a3aa7",
  aqua: "#1baf7a",
  green: "#008300",
};

// Light background tints for decision-boundary fills (class regions).
const TINT = {
  blue: [206, 226, 250],
  red: [250, 214, 214],
};

// A 2D plotting surface over a fixed data domain, with hi-dpi scaling.
export class Plot {
  constructor(canvas, domain) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.domain = domain; // {xmin, xmax, ymin, ymax}
    this.pad = { l: 34, r: 12, t: 12, b: 26 };
    this.resize();
  }

  resize() {
    const dpr = window.devicePixelRatio || 1;
    const rect = this.canvas.getBoundingClientRect();
    const w = Math.max(1, Math.round(rect.width));
    const h = Math.max(1, Math.round(rect.height));
    this.canvas.width = w * dpr;
    this.canvas.height = h * dpr;
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.w = w;
    this.h = h;
  }

  px(x) {
    const { xmin, xmax } = this.domain;
    const iw = this.w - this.pad.l - this.pad.r;
    return this.pad.l + ((x - xmin) / (xmax - xmin)) * iw;
  }

  py(y) {
    const { ymin, ymax } = this.domain;
    const ih = this.h - this.pad.t - this.pad.b;
    return this.pad.t + (1 - (y - ymin) / (ymax - ymin)) * ih;
  }

  clear() {
    this.ctx.clearRect(0, 0, this.w, this.h);
    this.ctx.fillStyle = PALETTE.surface;
    this.ctx.fillRect(0, 0, this.w, this.h);
  }

  axes(xlabel, ylabel) {
    const c = this.ctx;
    const { xmin, xmax, ymin, ymax } = this.domain;
    c.strokeStyle = PALETTE.grid;
    c.lineWidth = 1;
    c.fillStyle = PALETTE.muted;
    c.font = "11px system-ui, -apple-system, sans-serif";
    const xt = niceTicks(xmin, xmax, 5);
    const yt = niceTicks(ymin, ymax, 5);
    c.textAlign = "center";
    c.textBaseline = "top";
    for (const t of xt) {
      const x = this.px(t);
      c.beginPath();
      c.moveTo(x, this.pad.t);
      c.lineTo(x, this.h - this.pad.b);
      c.stroke();
      c.fillText(fmt(t), x, this.h - this.pad.b + 4);
    }
    c.textAlign = "right";
    c.textBaseline = "middle";
    for (const t of yt) {
      const y = this.py(t);
      c.beginPath();
      c.moveTo(this.pad.l, y);
      c.lineTo(this.w - this.pad.r, y);
      c.stroke();
      c.fillText(fmt(t), this.pad.l - 5, y);
    }
    // baseline frame
    c.strokeStyle = PALETTE.axis;
    c.strokeRect(this.pad.l, this.pad.t, this.w - this.pad.l - this.pad.r, this.h - this.pad.t - this.pad.b);
  }

  // Paint a class-region background from a batched probability function.
  // batchProbFn(points) takes an array of [x,y] and returns an array of
  // class-1 probabilities, evaluated once per frame for speed. res = columns.
  boundary(batchProbFn, res = 60) {
    const c = this.ctx;
    const iw = this.w - this.pad.l - this.pad.r;
    const ih = this.h - this.pad.t - this.pad.b;
    const { xmin, xmax, ymin, ymax } = this.domain;
    const cols = res;
    const rows = Math.round(res * (ih / iw));
    const img = c.createImageData(Math.round(iw), Math.round(ih));
    // Build all grid points, evaluate in one batched call, reshape.
    const pts = [];
    for (let r = 0; r < rows; r++) {
      const yv = ymax - (r / (rows - 1)) * (ymax - ymin);
      for (let cc = 0; cc < cols; cc++) {
        const xv = xmin + (cc / (cols - 1)) * (xmax - xmin);
        pts.push([xv, yv]);
      }
    }
    const flat = batchProbFn(pts);
    const grid = new Array(rows);
    for (let r = 0; r < rows; r++) {
      grid[r] = new Array(cols);
      for (let cc = 0; cc < cols; cc++) grid[r][cc] = flat[r * cols + cc];
    }
    const W = img.width, H = img.height;
    for (let py = 0; py < H; py++) {
      const gr = Math.min(rows - 1, Math.floor((py / H) * rows));
      for (let px = 0; px < W; px++) {
        const gc = Math.min(cols - 1, Math.floor((px / W) * cols));
        const p = grid[gr][gc];
        const t = TINT.blue, s = TINT.red;
        // interpolate red -> blue by p, softened toward white near 0.5
        const mix = (a, b) => Math.round(a + (b - a) * p);
        const idx = (py * W + px) * 4;
        img.data[idx] = mix(s[0], t[0]);
        img.data[idx + 1] = mix(s[1], t[1]);
        img.data[idx + 2] = mix(s[2], t[2]);
        img.data[idx + 3] = 235;
      }
    }
    c.putImageData(img, Math.round(this.pad.l), Math.round(this.pad.t));
  }

  points(pts, labels, colorMap, r = 3.4) {
    const c = this.ctx;
    for (let i = 0; i < pts.length; i++) {
      c.beginPath();
      c.arc(this.px(pts[i][0]), this.py(pts[i][1]), r, 0, Math.PI * 2);
      c.fillStyle = colorMap[labels[i]];
      c.fill();
      c.lineWidth = 1.2;
      c.strokeStyle = "rgba(252,252,251,0.9)";
      c.stroke();
    }
  }

  line(series, color, opts = {}) {
    const c = this.ctx;
    c.strokeStyle = color;
    c.lineWidth = opts.width || 2;
    if (opts.dash) c.setLineDash(opts.dash);
    c.beginPath();
    for (let i = 0; i < series.length; i++) {
      const x = this.px(series[i][0]);
      const y = this.py(series[i][1]);
      if (i === 0) c.moveTo(x, y);
      else c.lineTo(x, y);
    }
    c.stroke();
    c.setLineDash([]);
  }

  // draw text label at data coords
  text(x, y, str, color, opts = {}) {
    const c = this.ctx;
    c.fillStyle = color;
    c.font = opts.font || "12px system-ui, -apple-system, sans-serif";
    c.textAlign = opts.align || "left";
    c.textBaseline = opts.baseline || "middle";
    c.fillText(str, this.px(x) + (opts.dx || 0), this.py(y) + (opts.dy || 0));
  }
}

// A compact log-scale line chart for live loss curves.
export class LossPlot {
  constructor(canvas, opts = {}) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.ymin = opts.ymin || 0.003;
    this.ymax = opts.ymax || 1.0;
    this.maxEpoch = opts.maxEpoch || 600;
    this.pad = { l: 46, r: 14, t: 12, b: 24 };
    this.resize();
  }

  resize() {
    const dpr = window.devicePixelRatio || 1;
    const rect = this.canvas.getBoundingClientRect();
    const w = Math.max(1, Math.round(rect.width));
    const h = Math.max(1, Math.round(rect.height));
    this.canvas.width = w * dpr;
    this.canvas.height = h * dpr;
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.w = w;
    this.h = h;
  }

  xpos(e) {
    const iw = this.w - this.pad.l - this.pad.r;
    return this.pad.l + (e / this.maxEpoch) * iw;
  }

  ypos(v) {
    const ih = this.h - this.pad.t - this.pad.b;
    const clamped = Math.min(this.ymax, Math.max(this.ymin, v));
    const frac =
      (Math.log10(clamped) - Math.log10(this.ymin)) /
      (Math.log10(this.ymax) - Math.log10(this.ymin));
    return this.pad.t + (1 - frac) * ih;
  }

  // series: [{ color, data: [[epoch, loss], ...] }]
  draw(series) {
    const c = this.ctx;
    c.clearRect(0, 0, this.w, this.h);
    c.fillStyle = PALETTE.surface;
    c.fillRect(0, 0, this.w, this.h);
    // log gridlines
    c.font = "11px system-ui, -apple-system, sans-serif";
    c.fillStyle = PALETTE.muted;
    c.strokeStyle = PALETTE.grid;
    c.lineWidth = 1;
    const ticks = [1, 0.3, 0.1, 0.03, 0.01, 0.003];
    c.textAlign = "right";
    c.textBaseline = "middle";
    for (const t of ticks) {
      if (t > this.ymax || t < this.ymin) continue;
      const y = this.ypos(t);
      c.beginPath();
      c.moveTo(this.pad.l, y);
      c.lineTo(this.w - this.pad.r, y);
      c.stroke();
      c.fillText(String(t), this.pad.l - 6, y);
    }
    c.strokeStyle = PALETTE.axis;
    c.strokeRect(this.pad.l, this.pad.t, this.w - this.pad.l - this.pad.r, this.h - this.pad.t - this.pad.b);
    // x labels
    c.textAlign = "center";
    c.textBaseline = "top";
    c.fillStyle = PALETTE.muted;
    c.fillText("0", this.pad.l, this.h - this.pad.b + 5);
    c.fillText(String(this.maxEpoch), this.w - this.pad.r, this.h - this.pad.b + 5);
    // series
    for (const s of series) {
      if (!s.data.length) continue;
      c.strokeStyle = s.color;
      c.lineWidth = s.width || 2;
      if (s.dash) c.setLineDash(s.dash);
      c.beginPath();
      for (let i = 0; i < s.data.length; i++) {
        const x = this.xpos(s.data[i][0]);
        const y = this.ypos(s.data[i][1]);
        if (i === 0) c.moveTo(x, y);
        else c.lineTo(x, y);
      }
      c.stroke();
      c.setLineDash([]);
    }
  }
}

function niceTicks(min, max, count) {
  const range = max - min;
  const raw = range / count;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const norm = raw / mag;
  let step;
  if (norm < 1.5) step = 1;
  else if (norm < 3) step = 2;
  else if (norm < 7) step = 5;
  else step = 10;
  step *= mag;
  const start = Math.ceil(min / step) * step;
  const ticks = [];
  for (let t = start; t <= max + 1e-9; t += step) ticks.push(Math.round(t / step) * step);
  return ticks;
}

function fmt(v) {
  if (Math.abs(v) >= 1000) return (v / 1000).toFixed(v % 1000 === 0 ? 0 : 1) + "k";
  if (Number.isInteger(v)) return String(v);
  return v.toFixed(Math.abs(v) < 1 ? 2 : 1);
}
