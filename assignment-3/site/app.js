/* =========================================================================
   India-First 40B - chart engine
   Hand-built inline SVG (no libraries). Marks follow the dataviz spec:
   thin marks, 2px surface gaps, hairline recessive grid, direct labels,
   legend for >=2 series, hover tooltip + table-view twin per chart.
   Colours are read from CSS custom properties so charts follow the theme;
   everything re-renders on theme change.
   ========================================================================= */
(function () {
  "use strict";

  const SVGNS = "http://www.w3.org/2000/svg";
  const tip = document.getElementById("tooltip");

  /* ---- tiny element helpers ---- */
  function S(tag, attrs) {
    const e = document.createElementNS(SVGNS, tag);
    if (attrs) for (const k in attrs) e.setAttribute(k, attrs[k]);
    return e;
  }
  function txt(parent, x, y, s, cls, attrs) {
    const t = S("text", Object.assign({ x, y }, attrs || {}));
    if (cls) t.setAttribute("class", cls);
    t.textContent = s;
    parent.appendChild(t);
    return t;
  }
  function cvar(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }
  function host(id) { const h = document.getElementById(id); if (h) h.innerHTML = ""; return h; }
  function mkSvg(host, vb, minW) {
    const s = S("svg", { viewBox: vb, class: "chart", role: "img" });
    s.style.width = "100%";
    s.style.minWidth = (minW || 520) + "px";
    s.style.height = "auto";
    host.appendChild(s);
    return s;
  }

  /* ---- shared tooltip behaviour ---- */
  function hover(node, html) {
    node.style.cursor = "default";
    node.addEventListener("mousemove", (ev) => {
      tip.innerHTML = html;
      tip.style.left = ev.clientX + "px";
      tip.style.top = ev.clientY + "px";
      tip.style.opacity = "1";
    });
    node.addEventListener("mouseleave", () => { tip.style.opacity = "0"; });
  }

  /* ---- legend + table builders (HTML) ---- */
  function legend(id, items) {
    const box = document.getElementById(id);
    if (!box) return;
    box.innerHTML = "";
    items.forEach((it) => {
      const span = document.createElement("span");
      span.className = "item";
      const sw = document.createElement("span");
      sw.className = "swatch" + (it.line ? " line" : "");
      sw.style.background = it.color;
      span.appendChild(sw);
      span.appendChild(document.createTextNode(it.label));
      box.appendChild(span);
    });
  }
  function table(id, cols, rows) {
    const box = document.getElementById(id);
    if (!box) return;
    let h = "<table class='viz-table'><thead><tr>";
    cols.forEach((c, i) => (h += `<th class='${i ? "num" : ""}'>${c}</th>`));
    h += "</tr></thead><tbody>";
    rows.forEach((r) => {
      h += "<tr>";
      r.forEach((v, i) => (h += `<td class='${i ? "num" : ""}'>${v}</td>`));
      h += "</tr>";
    });
    h += "</tbody></table>";
    box.innerHTML = h;
  }

  /* number format */
  const nf = (n) => n.toLocaleString("en-US");

  /* ===================================================================
     FIGURE: THE WASTE BOMB  (500B tokens -> effective words)
     =================================================================== */
  function drawWaste() {
    const h = host("figWaste"); if (!h) return;
    const W = 720, H = 210, mL = 12, mR = 14, mT = 30, mB = 10;
    const svg = mkSvg(h, `0 0 ${W} ${H}`, 560);
    const plotW = W - mL - mR;
    const TOTAL = 500;
    const x = (v) => mL + (v / TOTAL) * plotW;

    const rows = [
      { label: "Bad multilingual tokenizer", fert: 13.5, words: 37, color: cvar("--critical"), icon: "▼" },
      { label: "Our target tokenizer", fert: 1.5, words: 333, color: cvar("--good"), icon: "▲" },
    ];
    const rowH = 46, gap = 40, y0 = mT + 18;

    txt(svg, mL, mT - 12, "Real words the model learns from, out of 500B collected tokens", "ax-title");

    rows.forEach((r, i) => {
      const y = y0 + i * (rowH + gap);
      // track (full 500B)
      svg.appendChild(S("rect", { x: mL, y, width: plotW, height: rowH, rx: 6, fill: cvar("--grid") }));
      // filled effective words
      const w = x(r.words) - mL;
      const bar = S("rect", { x: mL, y, width: Math.max(w, 3), height: rowH, rx: 6, fill: r.color });
      svg.appendChild(bar);
      hover(bar, `<b>${r.label}</b><br>fertility ${r.fert} &rarr; <b>${r.words}B words</b> of 500B (${Math.round(r.words/TOTAL*100)}%)`);
      // label above
      txt(svg, mL + 2, y - 6, `${r.icon} ${r.label}  ·  fertility ${r.fert}`, "val-label");
      // value at end of the effective portion
      const pct = Math.round((r.words / TOTAL) * 100);
      txt(svg, mL + Math.max(w, 3) + 8, y + rowH / 2 + 4, `${r.words}B words  (${pct}%)`, "val-label");
    });

    legend("figWaste-legend", [
      { label: "Effective words learned", color: cvar("--good") },
      { label: "Wasted as sub-word glue", color: cvar("--grid") },
    ]);
    table("figWaste-table",
      ["Scenario", "Fertility", "Effective words (of 500B)", "Share"],
      [["Bad tokenizer", "13.5", "37B", "7%"], ["Our target", "1.5", "333B", "67%"]]);
  }

  /* ===================================================================
     FIGURE 1: BENCHMARK TARGETS (grouped columns, 3 series)
     =================================================================== */
  const BENCH = [
    { name: "MMLU-Pro", g3: 67.6, g4: 85.2, ours: 85 },
    { name: "AIME 2026", g3: 20.8, g4: 89.2, ours: 89 },
    { name: "LiveCodeBench v6", g3: 29.1, g4: 80.0, ours: 80 },
    { name: "GPQA Diamond", g3: 42.4, g4: 84.3, ours: 84 },
    { name: "BBEH", g3: 19.3, g4: 74.4, ours: 74 },
    { name: "τ²-bench", g3: 16.2, g4: 76.9, ours: 77 },
    { name: "MMMLU", g3: 70.7, g4: 88.4, ours: 88 },
    { name: "IndicGenBench", g3: 63.4, g4: 0, ours: 75, indic: true },
  ];
  function drawBench() {
    const h = host("figBench"); if (!h) return;
    const W = 780, H = 348, mL = 34, mR = 12, mT = 16, mB = 80;
    const svg = mkSvg(h, `0 0 ${W} ${H}`, 700);
    const plotW = W - mL - mR, plotH = H - mT - mB;
    const y = (v) => mT + plotH - (v / 100) * plotH;
    const cOurs = cvar("--series-1"), cG4 = cvar("--series-3"), cG3 = cvar("--text-muted");

    [0, 25, 50, 75, 100].forEach((t) => {
      svg.appendChild(S("line", { x1: mL, y1: y(t), x2: W - mR, y2: y(t), stroke: cvar("--grid"), "stroke-width": 1 }));
      txt(svg, mL - 6, y(t) + 4, t, "ax-label", { "text-anchor": "end" });
    });

    const groupW = plotW / BENCH.length;
    const barW = Math.min(16, (groupW - 8) / 3);
    const gap = 2;
    const series = [
      { key: "g3", color: cG3, label: "Gemma 3 27B" },
      { key: "g4", color: cG4, label: "Gemma 4 31B" },
      { key: "ours", color: cOurs, label: "Our 40B target" },
    ];

    BENCH.forEach((b, gi) => {
      const gx = mL + gi * groupW + (groupW - (barW * 3 + gap * 2)) / 2;
      series.forEach((s, si) => {
        const v = b[s.key];
        const bx = gx + si * (barW + gap);
        if (v > 0) {
          const bh = (v / 100) * plotH;
          const r = S("rect", { x: bx, y: y(v), width: barW, height: bh, rx: 3, fill: s.color });
          svg.appendChild(r);
          hover(r, `<b>${b.name}</b><br>${s.label}: <b>${v}</b>`);
          if (s.key === "ours") txt(svg, bx + barW / 2, y(v) - 5, v, "val-label", { "text-anchor": "middle", "font-size": "10" });
        } else {
          txt(svg, bx + barW / 2, mT + plotH - 4, "n/p", "ax-label", { "text-anchor": "middle", "font-size": "9", fill: cvar("--serious") });
        }
      });
      const lx = mL + gi * groupW + groupW / 2, ly = mT + plotH + 14;
      const tnode = txt(svg, lx, ly, b.name, "ax-label", { "text-anchor": "end", transform: `rotate(-30 ${lx} ${ly})` });
      if (b.indic) tnode.setAttribute("fill", cvar("--series-1"));
    });

    // wedge note under the Indic group
    txt(svg, W - mR, mT + plotH + 66, "↑ Gemma 4 drops Indic - our lead", "ax-title",
      { "text-anchor": "end", fill: cvar("--series-1"), "font-size": "11" });

    legend("figBench-legend", [
      { label: "Our 40B target (parity + Indic lead)", color: cOurs },
      { label: "Gemma 4 31B", color: cG4 },
      { label: "Gemma 3 27B (context)", color: cG3 },
    ]);
    table("figBench-table",
      ["Benchmark", "Gemma 3 27B", "Gemma 4 31B", "Our target"],
      BENCH.map((b) => [b.name, b.g3, b.g4 || "n/p", b.ours]));
  }

  /* ===================================================================
     FIGURE 2: FERTILITY vs VOCAB  (scatter, allocation beats size)
     =================================================================== */
  const TOKZ = [
    { n: "Sarvam-1", v: 68, f: 1.53 },
    { n: "IndicSuperTokenizer", v: 200, f: 1.23, proof: true },
    { n: "GPT-OSS", v: 200, f: 1.72 },
    { n: "Llama-4", v: 201, f: 1.83 },
    { n: "Gemma 3 & 4 (same 262K)", v: 262, f: 1.47 },
  ];
  function drawScatter() {
    const h = host("figScatter"); if (!h) return;
    const W = 720, H = 320, mL = 44, mR = 16, mT = 20, mB = 46;
    const svg = mkSvg(h, `0 0 ${W} ${H}`, 560);
    const plotW = W - mL - mR, plotH = H - mT - mB;
    const xmin = 40, xmax = 280, ymin = 1.0, ymax = 2.0;
    const x = (v) => mL + ((v - xmin) / (xmax - xmin)) * plotW;
    const y = (f) => mT + ((f - ymin) / (ymax - ymin)) * plotH; // higher fertility = lower on chart? we want lower=better toward bottom
    // y grid
    [1.0, 1.25, 1.5, 1.75, 2.0].forEach((t) => {
      svg.appendChild(S("line", { x1: mL, y1: y(t), x2: W - mR, y2: y(t), stroke: cvar("--grid"), "stroke-width": 1 }));
      txt(svg, mL - 8, y(t) + 4, t.toFixed(2), "ax-label", { "text-anchor": "end" });
    });
    // x ticks
    [68, 128, 200, 262].forEach((t) => {
      txt(svg, x(t), mT + plotH + 18, t + "K", "ax-label", { "text-anchor": "middle" });
    });
    txt(svg, mL, mT + plotH + 38, "Vocabulary size →", "ax-title");
    txt(svg, mL - 34, mT - 6, "Hindi fertility (lower is better)", "ax-title");

    // target band 1.2-1.3
    const band = S("rect", { x: mL, y: y(1.3), width: plotW, height: y(1.2) - y(1.3), fill: cvar("--series-1"), opacity: 0.10 });
    svg.appendChild(band);
    txt(svg, W - mR - 4, y(1.25) + 3, "Our target zone 1.2-1.3", "val-label", { "text-anchor": "end", "font-size": "10", fill: cvar("--series-1") });

    // points
    TOKZ.forEach((p) => {
      const cx = x(p.v), cy = y(p.f);
      const col = p.proof ? cvar("--series-2") : cvar("--text-muted");
      const ring = S("circle", { cx, cy, r: p.proof ? 7 : 6, fill: col, stroke: cvar("--surface-1"), "stroke-width": 2 });
      svg.appendChild(ring);
      hover(ring, `<b>${p.n}</b><br>${p.v}K vocab &middot; Hindi fertility <b>${p.f}</b>`);
      const anchor = p.v > 240 ? "end" : "start";
      const dx = p.v > 240 ? -10 : 10;
      txt(svg, cx + dx, cy + (p.proof ? -11 : 4), p.n, p.proof ? "val-label" : "val-label sec",
        { "text-anchor": anchor, "font-size": "10.5" });
    });

    // Odia annotation
    txt(svg, x(201), y(1.9), "↑ Llama-4 on Odia = 10.5 (off-scale: no allocation)", "val-label sec",
      { "text-anchor": "middle", "font-size": "10", fill: cvar("--serious") });

    legend("figScatter-legend", [
      { label: "Our 200K target zone", color: cvar("--series-1") },
      { label: "IndicSuperTokenizer (published proof at 200K)", color: cvar("--series-2") },
      { label: "General multilingual", color: cvar("--text-muted") },
    ]);
    table("figScatter-table",
      ["Tokenizer", "Vocab", "Hindi fertility"],
      TOKZ.map((p) => [p.n, p.v + "K", p.f]));
  }

  /* ===================================================================
     FIGURE 3: VOCAB PARAM COST  (line)
     =================================================================== */
  const COST = [
    { v: 64, p: 1.3 }, { v: 128, p: 2.6 }, { v: 200, p: 4.1 },
    { v: 256, p: 5.2 }, { v: 320, p: 6.6 }, { v: 400, p: 8.2 },
  ];
  function drawCost() {
    const h = host("figCost"); if (!h) return;
    const W = 720, H = 260, mL = 40, mR = 16, mT = 18, mB = 44;
    const svg = mkSvg(h, `0 0 ${W} ${H}`, 520);
    const plotW = W - mL - mR, plotH = H - mT - mB;
    const xmin = 64, xmax = 400, ymax = 9;
    const x = (v) => mL + ((v - xmin) / (xmax - xmin)) * plotW;
    const y = (p) => mT + plotH - (p / ymax) * plotH;
    [0, 2, 4, 6, 8].forEach((t) => {
      svg.appendChild(S("line", { x1: mL, y1: y(t), x2: W - mR, y2: y(t), stroke: cvar("--grid"), "stroke-width": 1 }));
      txt(svg, mL - 6, y(t) + 4, t + "%", "ax-label", { "text-anchor": "end" });
    });
    COST.forEach((d) => txt(svg, x(d.v), mT + plotH + 18, d.v + "K", "ax-label", { "text-anchor": "middle" }));
    txt(svg, mL, mT - 4, "Embedding weights as share of a 40B model", "ax-title");

    // line
    let dpath = "";
    COST.forEach((d, i) => (dpath += (i ? "L" : "M") + x(d.v) + " " + y(d.p) + " "));
    svg.appendChild(S("path", { d: dpath, fill: "none", stroke: cvar("--series-1"), "stroke-width": 2, "stroke-linejoin": "round", "stroke-linecap": "round" }));
    // dots
    COST.forEach((d) => {
      const chosen = d.v === 200;
      const dot = S("circle", { cx: x(d.v), cy: y(d.p), r: chosen ? 6 : 4, fill: cvar("--series-1"), stroke: cvar("--surface-1"), "stroke-width": 2 });
      svg.appendChild(dot);
      hover(dot, `<b>${d.v}K vocab</b><br>${d.p}% of 40B params`);
    });
    // mark our choice
    const cx = x(200), cy = y(4.1);
    svg.appendChild(S("line", { x1: cx, y1: cy, x2: cx, y2: y(0), stroke: cvar("--series-1"), "stroke-width": 1, "stroke-dasharray": "0" , opacity: .25}));
    txt(svg, cx + 8, cy - 8, "Our choice: 200K = 4.1%", "val-label", { "font-size": "11", fill: cvar("--series-1") });

    table("figCost-table",
      ["Vocab", "Embedding share of 40B"],
      COST.map((d) => [d.v + "K", d.p + "%"]));
  }

  /* ===================================================================
     FIGURE 4: DATA RECIPE (100% stacked bar) + INDIC breakdown
     =================================================================== */
  function stackBar(hostId, legendId, tableId, segs, note) {
    const h = host(hostId); if (!h) return;
    const W = 720, H = 84, mL = 2, mR = 2, mT = 26, barH = 40;
    const svg = mkSvg(h, `0 0 ${W} ${H}`, 480);
    const plotW = W - mL - mR;
    const gap = 2;
    let cursor = mL;
    const total = segs.reduce((a, s) => a + s.pct, 0);
    txt(svg, mL, mT - 10, note, "ax-title");
    segs.forEach((s, i) => {
      const w = (s.pct / total) * plotW - (i < segs.length - 1 ? gap : 0);
      const r = S("rect", { x: cursor, y: mT, width: Math.max(w, 1), height: barH, rx: 4, fill: s.color });
      svg.appendChild(r);
      hover(r, `<b>${s.name}</b><br>${s.pct}%${s.tok ? " &middot; " + s.tok : ""}`);
      // inline label if wide enough
      if (w > 46) {
        const lc = s.dark ? "#fff" : (s.light ? "#0b0b0b" : "#fff");
        txt(svg, cursor + 8, mT + barH / 2 + 4, s.pct + "%", null, { fill: lc, "font-size": "12", "font-weight": "650" });
      }
      cursor += (s.pct / total) * plotW;
    });
    if (legendId) legend(legendId, segs.map((s) => ({ label: `${s.name} (${s.pct}%${s.tok ? ", " + s.tok : ""})`, color: s.color })));
    if (tableId) table(tableId, ["Segment", "Share", "Tokens"], segs.map((s) => [s.name, s.pct + "%", s.tok || "-"]));
  }
  function drawRecipe() {
    stackBar("figRecipe", "figRecipe-legend", "figRecipe-table", [
      { name: "English (Indian-weighted)", pct: 35, tok: "~4.6T", color: cvar("--series-1"), dark: true },
      { name: "Indic", pct: 22, tok: "~2.9T", color: cvar("--series-2"), dark: true },
      { name: "Code", pct: 20, tok: "~2.6T", color: cvar("--series-3"), dark: true },
      { name: "Math & science + CoT", pct: 15, tok: "~2.0T", color: cvar("--series-4"), dark: true },
      { name: "Cross-lingual bridges", pct: 8, tok: "~1.0T", color: cvar("--series-5"), dark: true },
    ], "Phase A - base pre-training mix (~13T)");

    stackBar("figIndic", null, null, [
      { name: "Organic (repeated ~4x)", pct: 10, tok: "~0.3T", color: cvar("--series-1"), dark: true },
      { name: "Translated / synthetic", pct: 90, tok: "~2.7T", color: cvar("--text-muted"), dark: true },
    ], "Inside the Indic slice - the honest breakdown");
  }

  /* ---- render everything ---- */
  function renderAll() {
    drawWaste(); drawBench(); drawScatter(); drawCost(); drawRecipe();
  }

  /* ---- table toggles ---- */
  document.querySelectorAll(".tbl-toggle").forEach((btn) => {
    btn.addEventListener("click", () => {
      const el = document.getElementById(btn.dataset.table);
      if (!el) return;
      const show = el.hasAttribute("hidden");
      if (show) el.removeAttribute("hidden"); else el.setAttribute("hidden", "");
      btn.textContent = show ? "Hide data table" : "Show data table";
    });
  });

  /* ---- theme toggle ---- */
  const root = document.documentElement;
  const iconEl = document.getElementById("themeIcon");
  const labelEl = document.getElementById("themeLabel");
  function effectiveDark() {
    const t = root.getAttribute("data-theme");
    if (t === "dark") return true;
    if (t === "light") return false;
    return window.matchMedia("(prefers-color-scheme: dark)").matches;
  }
  function syncToggle() {
    const dark = effectiveDark();
    iconEl.innerHTML = dark ? "&#9789;" : "&#9788;"; // moon / sun
    labelEl.textContent = dark ? "Dark" : "Light";
  }
  function applyStored() {
    const s = localStorage.getItem("a3-theme");
    if (s) root.setAttribute("data-theme", s);
  }
  document.getElementById("themeToggle").addEventListener("click", () => {
    const dark = effectiveDark();
    const next = dark ? "light" : "dark";
    root.setAttribute("data-theme", next);
    localStorage.setItem("a3-theme", next);
    syncToggle();
    renderAll(); // re-read theme colours
  });
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
    if (!root.getAttribute("data-theme")) { syncToggle(); renderAll(); }
  });

  /* ---- init ---- */
  applyStored();
  syncToggle();
  renderAll();
  window.addEventListener("resize", () => { /* SVG is viewBox-scaled; nothing needed */ });
})();
