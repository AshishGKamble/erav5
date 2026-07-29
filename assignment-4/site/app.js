/* ============================================================================
   India-First Cleanup - Assignment 4. Loads the REAL pipeline outputs
   (data/stats.json, data/examples.json, data/manifest.json) and renders the
   funnel, the walkable 8-stage pipeline, the language + fertility charts, two
   live in-browser demos, and the final-stats table. Hand-built inline SVG,
   dataviz-skill validated palette, theme-aware. No dependencies, no network.
   ============================================================================ */
const SVGNS = "http://www.w3.org/2000/svg";
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const cvar = n => getComputedStyle(document.body).getPropertyValue(n).trim();
const fmt = n => n == null ? "-" : n.toLocaleString("en-US");
const pct = (a, b) => b ? Math.round(1000 * a / b) / 10 : 0;

function S(tag, attrs = {}) {
  const e = document.createElementNS(SVGNS, tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  return e;
}
function txt(x, y, s, attrs = {}) {
  const t = S("text", { x, y, ...attrs }); t.textContent = s; return t;
}
function mkSvg(host, w, h) {
  const svg = S("svg", { viewBox: `0 0 ${w} ${h}`, width: "100%", role: "img" });
  host.innerHTML = ""; host.appendChild(svg); return svg;
}
const tip = $("#tooltip");
function hover(el, label) {
  el.style.cursor = "default";
  el.addEventListener("mousemove", e => {
    tip.textContent = label; tip.style.opacity = 1;
    tip.style.left = (e.clientX + 12) + "px"; tip.style.top = (e.clientY + 12) + "px";
  });
  el.addEventListener("mouseleave", () => { tip.style.opacity = 0; });
}

let STATS = {}, EX = {}, MAN = {};

/* --------------------------------------------------------------- stage copy */
const STAGES = [
  { key: "extract", n: 1, title: "Extract", bonus: false,
    what: "Pull the real text out of the interactions structure and drop empty rows and residual markup.",
    why: "The document, not the web page: empty cells and stray tags are not training data.",
    trade: "We keep extraction light here - indic-align is already text. The heavy extraction lesson (trafilatura vs a naive tag-strip) was Session 3.",
    v4: "Naive HTML stripping kept navigation and cookie banners as if they were content.",
    stat: s => ({ big: fmt(s.extract.docs_in), sub: `documents extracted (${fmt(s.extract.empties_dropped)} empty dropped)` }),
    exKey: null },
  { key: "normalize", n: 2, title: "Normalize", bonus: false,
    what: "NFC normalize, unescape HTML entities, strip invisible / bidi / control noise, collapse whitespace, and flag ghost special tokens - while preserving the Indic joiners ZWJ and ZWNJ.",
    why: "The content hash is computed AFTER cleaning, so two docs that differ only in invisible junk dedupe to the same hash. And stray control characters take permanent vocab slots.",
    trade: "NFC, never NFKC: NFKC is lossy on Indic (it collapses conjuncts and nukta). We accept keeping some compatibility variants rather than ever corrupt a script.",
    v4: "Skip this pass and invisible characters survive into the tokenizer - the session audit found 46 such garbage tokens holding real vocabulary slots.",
    stat: s => ({ big: fmt(s.normalize.zwnj_preserved + s.normalize.zwj_preserved), sub: `Indic joiners preserved &middot; ${fmt(s.normalize.ghost_hits)} ghost tags flagged` }),
    exKey: "ghost" },
  { key: "language", n: 3, title: "Language ID", bonus: false,
    what: "Detect script and language per document, compare it to the claimed language (the column name), and flag mismatches and code-switching.",
    why: "A label is a claim, not a fact. Romanized Hindi tagged 'hi' is Latin text a script detector reads as English - it must be quarantined for transliteration, not silently filed as Hindi or English.",
    trade: "Script detection is near-perfect across distinct Indic scripts but cannot separate romanized Hindi from English - so those are FLAGGED for review, not trusted either way.",
    v4: "Trusting the folder or column label instead of detecting: a Bengali file can sit in an 'Assamese' folder and be counted as Assamese.",
    stat: s => ({ big: fmt(s.language.claimed_vs_detected_mismatch), sub: `claimed-vs-detected mismatches (${fmt(s.language.romanized_flagged)} romanized)` }),
    exKey: "language" },
  { key: "quality", n: 4, title: "Quality filter", bonus: false,
    what: "Run Gopher/C4 heuristics - word length, symbol ratio, repetition, stop-words - with a dedicated Indic Always-ON channel.",
    why: "Every threshold in the classic chain was calibrated on English writing habits, which silently under-value low-resource Indic text.",
    trade: "Filtering is a trade, not a free lunch: stricter rules shrink the corpus while raising average quality. We deliberately exempt Indic docs from the English stop-word and word-length rules.",
    v4: "An English-tuned filter quietly deletes good low-resource Indic text, because it demands English stop-words and English word lengths.",
    stat: s => ({ big: fmt(s.quality.indic_saved_by_alwayson), sub: `Indic docs the Always-ON channel saved that an English chain would have cut` }),
    exKey: "quality" },
  { key: "dedup", n: 5, title: "Deduplicate", bonus: false,
    what: "Shingle each doc, build a MinHash signature, and use LSH banding to find near-duplicates GLOBALLY across the whole slice - not per file.",
    why: "Near-identical documents get memorized instead of learned from. AI4Bharat has noted that Sangraha, its large Indic web crawl, shipped with no deduplication at all.",
    trade: "A lower threshold catches more dupes but risks cutting legitimately similar docs; we set Jaccard ~0.70 (FineWeb-ish). Cross-lingual copies are intentionally not caught - different languages, different shingles.",
    v4: "Deduplicating each file on its own misses copies that live in different files; only a global pass catches them.",
    stat: s => ({ big: fmt(s.dedup.removed), sub: `near+exact duplicates removed (${fmt(s.dedup.exact_dupes)} exact, ${fmt(s.dedup.near_dupes)} near)` }),
    exKey: "dedup" },
  { key: "pii", n: 6, title: "PII scrub", bonus: false,
    what: "A regex layer masks email, phone, IP and Aadhaar-style numbers; an honorific-anchored name layer masks personal names.",
    why: "Structured identifiers have exact shapes a regex nails; names have no fixed shape, so they need a model/heuristic layer with a real precision-recall tension.",
    trade: "We anchor names on honorifics (Shri/Dr/Smt/...) for high precision and accept modest recall - aggressive NER starts masking Indic place names that are not people.",
    v4: "Skipping a PII pass ships real emails, phone numbers and names straight into the training data.",
    stat: s => ({ big: fmt(s.pii.total_masked), sub: `PII items masked in ${fmt(s.pii.docs_with_pii)} docs` }),
    exKey: "pii" },
  { key: "decontaminate", n: 7, title: "Decontaminate", bonus: true,
    what: "Scan every doc's n-grams against the held-out benchmark questions and drop any overlap. A safety pass additionally drops flagged-toxic docs.",
    why: "A benchmark's test split is the exam; if it leaks into training the reported score climbs while true ability does not move at all.",
    trade: "Low leakage here is the GOOD outcome, not a weak number - the point is that the firewall exists and runs on every shard. It is decided at sourcing, not as an afterthought.",
    v4: "Letting a benchmark's test questions into training inflates the reported score without improving the model - contamination.",
    stat: s => ({ big: fmt(s.decontaminate.contaminated_removed + s.decontaminate.toxic_removed), sub: `removed: ${fmt(s.decontaminate.contaminated_removed)} contaminated + ${fmt(s.decontaminate.toxic_removed)} toxic, vs ${fmt(s.decontaminate.holdout_questions)} hold-out Qs` }),
    exKey: "decontaminate" },
  { key: "manifest", n: 8, title: "Manifest", bonus: false,
    what: "Every shard ships a provenance manifest: source, license, cleaning-script hash, ingest time, SHA-256 of the cleaned text, real token count and language distribution.",
    why: "No shard enters the corpus without one - that is what lets the gating rule mean something and the corpus defend its own lineage.",
    trade: "We count tokens with the real tokenizer instead of estimating words x 1.3 (which is wrong for Indic by 2-10x) - slower, but the numbers are honest.",
    v4: "Without a manifest and a content-based id, a re-run can label the same document differently and the corpus cannot prove where it came from.",
    stat: s => ({ big: (s.manifest.total_tokens / 1e6).toFixed(1) + "M", sub: `real tokens &middot; ${fmt(s.manifest.admitted)} shards admitted &middot; deterministic: ${s.manifest.determinism_ok ? "yes" : "no"}` }),
    exKey: null }
];

/* --------------------------------------------------------------- KPI tiles */
function renderKpi() {
  const s = STATS, sm = s._summary, m = s.manifest;
  const tiles = [
    ["Strategies", "8", "pipeline stages (+2 bonus)"],
    ["Tokens (cleaned)", (m.total_tokens / 1e6).toFixed(1) + "<span class='u'>M</span>", "real MuRIL count"],
    ["Documents", fmt(sm.docs_out), `${sm.survival_pct}% of ${fmt(sm.docs_in)} survived`],
    ["Languages", String(s.language.languages_seen), "detected in the corpus"],
    ["Duplicates cut", fmt(s.dedup.removed), "global MinHash+LSH"],
    ["PII masked", fmt(s.pii.total_masked), "email / phone / IP / name"],
  ];
  $("#kpi").innerHTML = tiles.map(([l, v, sub]) =>
    `<div class="tile"><div class="label">${l}</div><div class="value">${v}</div><div class="sub">${sub}</div></div>`
  ).join("");
}

/* --------------------------------------------------------------- funnel */
function renderFunnel() {
  const data = STATS._summary.funnel;            // [ [name, count], ... ]
  const host = $("#figFunnel");
  const W = 720, rowH = 34, padL = 110, padR = 70, padT = 8;
  const H = padT * 2 + data.length * rowH;
  const svg = mkSvg(host, W, H);
  const max = Math.max(...data.map(d => d[1]));
  const ramp = ["--seq-250", "--seq-350", "--seq-350", "--seq-450", "--seq-450", "--seq-550", "--seq-550", "--seq-650"];
  data.forEach((d, i) => {
    const y = padT + i * rowH + 4;
    const bw = (W - padL - padR) * d[1] / max;
    svg.appendChild(txt(padL - 8, y + 13, d[0], { "text-anchor": "end", fill: cvar("--text-2"), "font-size": 12 }));
    const bar = S("rect", { x: padL, y, width: Math.max(bw, 2), height: rowH - 12, rx: 4, fill: cvar(ramp[i]) });
    hover(bar, `${d[0]}: ${fmt(d[1])} docs`);
    svg.appendChild(bar);
    svg.appendChild(txt(padL + bw + 6, y + 13, fmt(d[1]), { fill: cvar("--text-2"), "font-size": 11, "font-variant-numeric": "tabular-nums" }));
  });
  const first = data[0][1], last = data[data.length - 1][1];
  $("#funnelNote").innerHTML = `<b>${fmt(first)}</b> documents in &rarr; <b>${fmt(last)}</b> clean shards out ` +
    `(<b>${pct(last, first)}%</b> survive). The two biggest cuts are quality filtering and deduplication.`;
  buildTable("figFunnel-table", ["Stage", "Documents"], data.map(d => [d[0], fmt(d[1])]));
}

/* --------------------------------------------------------------- strategy cards */
function renderStratCards() {
  $("#stratCards").innerHTML = STAGES.map(st =>
    `<div class="card"><div class="k">${st.n}. ${st.title}${st.bonus ? " + safety" : ""}</div>
     <p>${st.what}</p></div>`
  ).join("");
}

/* --------------------------------------------------------------- walker */
function exampleHTML(key) {
  const list = EX[key]; if (!list || !list.length) return "";
  const e = list[0];
  if (key === "normalize" || key === "ghost")
    return baBox("raw", esc(e.before), "cleaned", esc(e.after));
  if (key === "language") {
    if (e.claimed) return `<div class="example"><div class="ba"><div class="box"><div class="h">claimed &rarr; detected</div>
      <b>${e.claimed}</b> &rarr; <b>${e.detected}</b>${e.romanized ? " (romanized)" : ""}</div>
      <div class="box after"><div class="h">text</div>${esc(e.text)}</div></div></div>`;
    return `<div class="example"><div class="ba"><div class="box after"><div class="h">detected ${e.detected} @ ${e.conf}</div>${esc(e.text)}</div></div></div>`;
  }
  if (key === "quality")
    return `<div class="example"><div class="ba"><div class="box"><div class="h">dropped &middot; ${e.lang}</div>${esc(e.text)}</div>
      <div class="box"><div class="h">failed rules</div>${(e.failed || []).map(f => `<code>${f}</code>`).join(" ")}</div></div></div>`;
  if (key === "dedup")
    return baBox("kept", esc(e.a), "duplicate (dropped)", esc(e.b));
  if (key === "pii")
    return `<div class="example"><div class="box after"><div class="h">masked (${Object.entries(e.masked).map(([k, v]) => k + "&times;" + v).join(", ")})</div>${maskHi(esc(e.after))}</div></div>`;
  if (key === "decontaminate")
    return `<div class="example"><div class="box"><div class="h">contaminated - shares hold-out n-gram</div>${esc(e.text)}<div class="h" style="margin-top:.3rem">overlap</div><code>${esc((e.overlap || [])[0] || "")}</code></div></div>`;
  return "";
}
function baBox(hA, a, hB, b) {
  return `<div class="example"><div class="ba">
    <div class="box"><div class="h">${hA}</div>${a}</div>
    <div class="box after"><div class="h">${hB}</div>${b}</div></div></div>`;
}
function maskHi(s) { return s.replace(/\[(EMAIL|PHONE|IP|NAME|AADHAAR)\]/g, '<span class="masked">[$1]</span>'); }

function renderStage(i) {
  const st = STAGES[i], s = STATS;
  $$(".pstage").forEach((b, j) => b.classList.toggle("on", j === i));
  const stat = st.stat(s);
  const manView = st.key === "manifest"
    ? `<div class="pb-lab" style="margin-top:.6rem">emitted manifest</div>
       <div class="out mono" style="font-size:.72rem">${esc(JSON.stringify(MAN, null, 1)).slice(0, 900)}</div>` : "";
  $("#pipeBody").innerHTML = `
    <div class="pb-main">
      <div class="pb-kicker">Stage ${st.n} of 8${st.bonus ? " &middot; + safety bonus" : ""}</div>
      <div class="pb-title">${st.title}</div>
      <p class="pb-what"><span class="pb-lab">What it does</span>${st.what}</p>
      <p class="pb-why"><span class="pb-lab">Why</span>${st.why}</p>
      ${exampleHTML(st.exKey) || manView}
    </div>
    <div class="pb-side">
      <div class="pb-lab">Real result on our slice</div>
      <div class="pb-stat">${stat.big}</div>
      <div class="pb-stat-sub">${stat.sub}</div>
      <div class="pb-trade"><b>Tradeoff.</b> ${st.trade}</div>
      <div class="pb-v4"><b>The mistake it prevents.</b> ${st.v4}</div>
    </div>`;
}
function renderWalker() {
  $("#pipeStages").innerHTML = STAGES.map((st, i) =>
    `<button class="pstage${st.bonus ? " bonus" : ""}" data-i="${i}">
       <div class="pn">STAGE ${st.n}</div><div class="pt">${st.title}</div><div class="pdot"></div></button>`
  ).join("");
  $$(".pstage").forEach(b => b.addEventListener("click", () => renderStage(+b.dataset.i)));
  renderStage(0);
}

/* --------------------------------------------------------------- source table */
function renderSourceTable() {
  const ps = STATS.extract.per_source;
  const rows = Object.entries(ps).sort((a, b) => b[1] - a[1])
    .map(([k, v]) => [k, fmt(v)]);
  const el = $("#sourceTable");
  el.innerHTML = `<thead><tr><th>Source (indic-align)</th><th class="num">Documents extracted</th></tr></thead>
    <tbody>${rows.map(r => `<tr><td>${r[0]}</td><td class="num">${r[1]}</td></tr>`).join("")}</tbody>
    <tfoot><tr><td>total slice</td><td class="num">${fmt(STATS.extract.docs_in)}</td></tr></tfoot>`;
}

/* --------------------------------------------------------------- language chart */
function renderLangChart() {
  const dist = STATS.language.distribution;             // {Name: count}
  const data = Object.entries(dist).sort((a, b) => b[1] - a[1]);
  const host = $("#figLang");
  const W = 720, rowH = 26, padL = 96, padR = 64, padT = 6;
  const H = padT * 2 + data.length * rowH;
  const svg = mkSvg(host, W, H);
  const max = Math.max(...data.map(d => d[1]));
  data.forEach((d, i) => {
    const y = padT + i * rowH + 3;
    const bw = (W - padL - padR) * d[1] / max;
    svg.appendChild(txt(padL - 8, y + 12, d[0], { "text-anchor": "end", fill: cvar("--text-2"), "font-size": 11.5 }));
    const bar = S("rect", { x: padL, y, width: Math.max(bw, 2), height: rowH - 10, rx: 4, fill: cvar("--s1") });
    hover(bar, `${d[0]}: ${fmt(d[1])} docs`); svg.appendChild(bar);
    svg.appendChild(txt(padL + bw + 6, y + 12, fmt(d[1]), { fill: cvar("--text-2"), "font-size": 10.5, "font-variant-numeric": "tabular-nums" }));
  });
  buildTable("figLang-table", ["Language", "Documents"], data.map(d => [d[0], fmt(d[1])]));
}

/* --------------------------------------------------------------- fertility chart (MuRIL, by family) */
const FAMILY = { Tamil: "drav", Telugu: "drav", Kannada: "drav", Malayalam: "drav", English: "en" };
const FAMCOL = { drav: "--s2", en: "--s3", ia: "--s1" };
const FAMLAB = { ia: "Indo-Aryan", drav: "Dravidian", en: "English" };
function renderFertChart() {
  const fm = STATS.manifest.fertility_by_language || {};        // MuRIL, all 13 (incl. English)
  const data = Object.entries(fm).sort((a, b) => a[1] - b[1]);
  const host = $("#figFert");
  const W = 720, rowH = 26, padL = 96, padR = 48, padT = 6;
  const H = padT * 2 + data.length * rowH;
  const svg = mkSvg(host, W, H);
  const max = Math.max(...data.map(d => d[1]), 2.4);
  data.forEach(([name, v], i) => {
    const y = padT + i * rowH + 3;
    const bw = Math.max((W - padL - padR) * v / max, 2);
    const fam = FAMILY[name] || "ia";
    svg.appendChild(txt(padL - 8, y + 12, name, { "text-anchor": "end", fill: cvar("--text-2"), "font-size": 11.5 }));
    const bar = S("rect", { x: padL, y, width: bw, height: rowH - 10, rx: 4, fill: cvar(FAMCOL[fam]) });
    hover(bar, `${name}: ${v} tokens/word`); svg.appendChild(bar);
    svg.appendChild(txt(padL + bw + 6, y + 12, v.toFixed(2), { fill: cvar("--text-2"), "font-size": 10.5, "font-variant-numeric": "tabular-nums" }));
  });
  const fams = [...new Set(data.map(([n]) => FAMILY[n] || "ia"))];
  $("#figFert-legend").innerHTML = fams.map(f =>
    `<span class="it"><span class="sw" style="background:${cvar(FAMCOL[f])}"></span>${FAMLAB[f]}</span>`).join("");
  $("#fertNote").innerHTML =
    `Every language lands between <b>${data[0][1].toFixed(2)}</b> (${data[0][0]}) and ` +
    `<b>${data[data.length - 1][1].toFixed(2)}</b> (${data[data.length - 1][0]}) tokens per word. ` +
    `Overall fertility is <b>${STATS.manifest.overall_fertility}</b>; Hindi's ${fm["Hindi"] || ""} matches the Assignment-3 target.`;
  buildTable("figFert-table", ["Language", "Fertility (tokens/word)"],
    data.map(([n, v]) => [n, v.toFixed(2)]));
}

/* --------------------------------------------------------------- final table */
function renderFinal() {
  const s = STATS, sm = s._summary, m = s.manifest;
  const rows = [
    ["Dataset", sm.dataset],
    ["License", sm.license + " (real token counts, not words &times; 1.3)"],
    ["Documents in &rarr; out", `${fmt(sm.docs_in)} &rarr; ${fmt(sm.docs_out)} (${sm.survival_pct}% survive)`],
    ["Tokenizer", "MuRIL (google/muril-base-cased, 197K vocab)"],
    ["Real tokens in &rarr; out (MuRIL)", `${fmt(sm.tokens_in)} &rarr; ${fmt(m.total_tokens)} &middot; overall fertility ${m.overall_fertility}`],
    ["Ghost tags flagged / joiners kept", `${fmt(s.normalize.ghost_hits)} / ${fmt(s.normalize.zwnj_preserved + s.normalize.zwj_preserved)}`],
    ["Language mismatches flagged", `${fmt(s.language.claimed_vs_detected_mismatch)} (${fmt(s.language.romanized_flagged)} romanized)`],
    ["Quality: dropped / Indic saved by Always-ON", `${fmt(s.quality.dropped)} / ${fmt(s.quality.indic_saved_by_alwayson)}`],
    ["Duplicates removed (exact + near)", `${fmt(s.dedup.removed)} (${fmt(s.dedup.exact_dupes)} + ${fmt(s.dedup.near_dupes)})`],
    ["PII masked", `${fmt(s.pii.total_masked)} in ${fmt(s.pii.docs_with_pii)} docs`],
    ["Decontaminate: contaminated / toxic removed", `${fmt(s.decontaminate.contaminated_removed)} / ${fmt(s.decontaminate.toxic_removed)} (vs ${fmt(s.decontaminate.holdout_questions)} hold-out Qs)`],
    ["Manifest shards admitted / blocked", `${fmt(m.admitted)} / ${fmt(m.blocked)} &middot; deterministic: ${m.determinism_ok ? "yes" : "no"}`],
    ["Pipeline runtime", `${sm.runtime_sec}s`],
  ];
  $("#finalTable").innerHTML = `<thead><tr><th>Metric</th><th>Value</th></tr></thead>
    <tbody>${rows.map(r => `<tr><td>${r[0]}</td><td>${r[1]}</td></tr>`).join("")}</tbody>`;
  $("#finalCallout").innerHTML = `<b>Bottom line.</b> From ${fmt(sm.docs_in)} raw documents we shipped ` +
    `<b>${fmt(sm.docs_out)}</b> clean, deduplicated, PII-scrubbed, decontaminated shards - ` +
    `<b>${(m.total_tokens / 1e6).toFixed(1)}M real tokens</b> across ${s.language.languages_seen} languages, ` +
    `each stamped with a reproducible manifest.`;
}

/* --------------------------------------------------------------- table helper */
function buildTable(id, headers, rows) {
  const host = $("#" + id);
  host.innerHTML = `<div class="data-table-wrap"><table class="data-table"><thead><tr>${
    headers.map((h, i) => `<th class="${i ? "num" : ""}">${h}</th>`).join("")}</tr></thead><tbody>${
    rows.map(r => `<tr>${r.map((c, i) => `<td class="${i ? "num" : ""}">${c}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`;
}
function wireTableToggles() {
  $$(".tbl-toggle").forEach(b => b.addEventListener("click", () => {
    const t = $("#" + b.dataset.table); const on = t.hasAttribute("hidden");
    if (on) t.removeAttribute("hidden"); else t.setAttribute("hidden", "");
    b.textContent = on ? "Hide data table" : "Show data table";
  }));
}
function esc(s) { return String(s).replace(/[&<>]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c])); }

/* --------------------------------------------------------------- live: clean_text */
const NOISE_CP = [0x200B, 0x200E, 0x200F, 0xFEFF, 0xFFFD, 0x202A, 0x202B, 0x202C, 0x202D, 0x202E, 0x2066, 0x2067, 0x2068, 0x2069];
const NOISE_RE = new RegExp("[" + NOISE_CP.map(c => "\\u" + c.toString(16).padStart(4, "0")).join("") + "]" + "|[\\x00-\\x08\\x0b\\x0c\\x0e-\\x1f]", "g");
const GHOST_RE = /\[\/?(?:USER|ASSISTANT|SYSTEM|INST)\]|<\|[a-z_]+\|>|<<SYS>>|\[\/?INST\]/gi;
function liveClean() {
  const raw = $("#ntIn").value;
  let noise = (raw.match(NOISE_RE) || []).length;
  const ghost = (raw.match(GHOST_RE) || []).length;
  const join = (raw.match(/[\u200c\u200d]/g) || []).length;
  let s = raw.normalize("NFC");
  const doc = document.createElement("textarea"); doc.innerHTML = s; s = doc.value; // unescape entities
  s = s.replace(/<[^>]{1,40}>/g, " ").replace(NOISE_RE, "").replace(GHOST_RE, "")
       .replace(/[^\S\n]+/g, " ").replace(/\n{3,}/g, "\n\n").trim();
  $("#ntOut").textContent = s;
  $("#ntNoise").textContent = noise; $("#ntGhost").textContent = ghost; $("#ntJoin").textContent = join;
}

/* --------------------------------------------------------------- live: minhash/jaccard */
function shingles(t, k = 5) {
  const w = t.toLowerCase().split(/\s+/).filter(Boolean);
  const s = new Set();
  if (w.length < k) { if (w.length) s.add(w.join(" ")); return s; }
  for (let i = 0; i <= w.length - k; i++) s.add(w.slice(i, i + k).join(" "));
  return s;
}
function liveDup() {
  const A = shingles($("#dupA").value), B = shingles($("#dupB").value);
  let inter = 0; A.forEach(x => { if (B.has(x)) inter++; });
  const uni = A.size + B.size - inter;
  const jac = uni ? inter / uni : 0;
  $("#dupJac").textContent = jac.toFixed(3);
  $("#dupShare").textContent = inter;
  const v = $("#dupVerdict");
  if (jac >= 0.7) { v.textContent = "DUPLICATE - dropped"; v.className = "pill bad"; }
  else { v.textContent = "unique - kept"; v.className = "pill ok"; }
}

/* --------------------------------------------------------------- theme + boot */
function initTheme() {
  const root = document.documentElement, btn = $("#themeToggle");
  const saved = localStorage.getItem("a4theme");
  if (saved) root.setAttribute("data-theme", saved);
  const sync = () => {
    const dark = root.getAttribute("data-theme") === "dark" ||
      (!root.getAttribute("data-theme") && matchMedia("(prefers-color-scheme: dark)").matches);
    $("#themeIcon").innerHTML = dark ? "&#9790;" : "&#9788;";
    $("#themeLabel").textContent = dark ? "Dark" : "Light";
  };
  btn.addEventListener("click", () => {
    const cur = root.getAttribute("data-theme") === "dark" ? "light" : "dark";
    root.setAttribute("data-theme", cur); localStorage.setItem("a4theme", cur);
    sync(); renderAllCharts();
  });
  sync();
}
function renderAllCharts() {
  renderFunnel(); renderLangChart(); renderFertChart();
}

/* --------------------------------------------------------------- before -> after summary */
function renderSummary() {
  const sm = STATS._summary, m = STATS.manifest;
  const tin = sm.tokens_in, tout = m.total_tokens;
  const box = (lab, docs, toks, cls) =>
    `<div class="io-box ${cls}"><div class="io-lab">${lab}</div>
       <div class="io-big">${fmt(docs)}<span class="io-u">docs</span></div>
       <div class="io-sub">${toks ? (toks / 1e6).toFixed(1) + "M tokens" : ""}</div></div>`;
  $("#summaryStrip").innerHTML =
    box("Raw slice in", sm.docs_in, tin, "in") +
    `<div class="io-arrow">&rarr;<span>${sm.survival_pct}% kept</span></div>` +
    box("Clean corpus out", sm.docs_out, tout, "out");
}
function renderPipeInOut() {
  const sm = STATS._summary, m = STATS.manifest;
  $("#pipeInOut").innerHTML =
    `<b>${fmt(sm.docs_in)}</b> documents enter &rarr; <b>${fmt(sm.docs_out)}</b> clean shards leave ` +
    `(<b>${sm.survival_pct}%</b> kept) &middot; ${sm.tokens_in ? (sm.tokens_in / 1e6).toFixed(1) + "M &rarr; " : ""}` +
    `${(m.total_tokens / 1e6).toFixed(1)}M tokens.`;
}

/* --------------------------------------------------------------- demo presets + stats */
function _chip(text, on) {
  const b = document.createElement("button"); b.className = "preset"; b.textContent = text;
  b.addEventListener("click", on); return b;
}
function _setActive(container, btn) {
  [...container.children].forEach(c => c.classList.toggle("on", c === btn));
}
function setupDemos() {
  const S = STATS;
  const NT = {
    "English scrape": "Cookie notice: We &amp; our 47 partners ​use cookies. It&#8217;s the crawler&#8217;s job to fetch every page.",
    "Hindi + ghost tag": "[USER] ﻿नमस्ते‍ दुनिया - यह ठीक है। <|endoftext|>",
    "Code snippet": "def clean(s):\n    return s.strip()   # drop &lt;tags&gt; &amp; junk",
    "Worst case": "﻿[SYSTEM] We &amp; 47 partners​ use cookies.\nएक‍ पंक्ति। <|endoftext|>"
  };
  const DUP = {
    "Near-duplicate": ["Our web crawler pulls millions of raw pages from the open internet every single day.",
                       "Our web crawler pulls millions of raw pages from the open internet each and every day."],
    "Exact copy": ["The quick brown fox jumps over the lazy dog by the river bank.",
                   "The quick brown fox jumps over the lazy dog by the river bank."],
    "Unrelated": ["Photosynthesis converts sunlight into chemical energy in green plants.",
                  "The Mumbai local train network carries over seven million passengers every day."]
  };
  const npr = $("#ntPresets");
  Object.entries(NT).forEach(([k, v]) => { const b = _chip(k, () => { $("#ntIn").value = v; liveClean(); _setActive(npr, b); }); npr.appendChild(b); });
  const dpr = $("#dupPresets");
  Object.entries(DUP).forEach(([k, ab]) => { const b = _chip(k, () => { $("#dupA").value = ab[0]; $("#dupB").value = ab[1]; liveDup(); _setActive(dpr, b); }); dpr.appendChild(b); });
  $("#ntIn").value = NT["English scrape"]; _setActive(npr, npr.firstChild);
  $("#dupA").value = DUP["Near-duplicate"][0]; $("#dupB").value = DUP["Near-duplicate"][1]; _setActive(dpr, dpr.firstChild);
  // typing into a box means the content no longer matches a preset - clear the active chip
  $("#ntIn").addEventListener("input", () => _setActive(npr, null));
  $("#dupA").addEventListener("input", () => _setActive(dpr, null));
  $("#dupB").addEventListener("input", () => _setActive(dpr, null));
  ["input", "keyup"].forEach(ev => { $("#ntIn").addEventListener(ev, liveClean); $("#dupA").addEventListener(ev, liveDup); $("#dupB").addEventListener(ev, liveDup); });
  liveClean(); liveDup();
  $("#ntStat").innerHTML = `<b>From the full run:</b> Normalize flagged ${fmt(S.normalize.ghost_hits)} ghost tags and preserved ${fmt(S.normalize.zwnj_preserved + S.normalize.zwj_preserved)} Indic joiners across ${fmt(S.normalize.docs)} documents - and computes each document's content hash AFTER this pass.`;
  $("#dupStat").innerHTML = `<b>From the full run:</b> global MinHash+LSH removed ${fmt(S.dedup.removed)} duplicates (${fmt(S.dedup.exact_dupes)} exact + ${fmt(S.dedup.near_dupes)} near) from ${fmt(S.dedup.docs_in)} documents.`;
}

async function boot() {
  try {
    [STATS, EX, MAN] = await Promise.all([
      fetch("data/stats.json").then(r => r.json()),
      fetch("data/examples.json").then(r => r.json()),
      fetch("data/manifest.json").then(r => r.json()),
    ]);
  } catch (e) {
    $("#kpi").innerHTML = "<div class='tile'><div class='label'>Data not loaded</div><div class='sub'>Run pipeline/clean.py to generate data/stats.json</div></div>";
    return;
  }
  renderSummary(); renderPipeInOut();
  renderKpi(); renderFunnel(); renderStratCards(); renderWalker();
  renderSourceTable(); renderLangChart(); renderFertChart(); renderFinal();
  wireTableToggles();
  setupDemos();
  initTheme();
}
boot();
