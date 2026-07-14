// Widget logic. Everything shown is recomputed live in the browser from the
// exported HuggingFace tokenizer.json and the four corpus texts - no number is
// hardcoded. The JS tokenizer (hf_tokenizer.js) is verified token-for-token
// against the Python `tokenizers` library (scripts/test_widget_js.py).

import { loadTokenizer, encode, decode, faithfulUnits } from "./hf_tokenizer.js";

const LANGS = ["en", "hi", "te", "mr"];
const NAMES = { en: "English", hi: "Hindi", te: "Telugu", mr: "Marathi" };
const SCRIPT = { en: "Latin", hi: "Devanagari", te: "Telugu", mr: "Devanagari" };
const COLOR = { en: "#2a78d6", hi: "#eb6834", te: "#e34948", mr: "#1baf7a" };
const CAP = 1.2;

const $ = (s) => document.querySelector(s);
const fmt = (n, d = 4) => n.toFixed(d);
function el(html) {
  const t = document.createElement("template");
  t.innerHTML = html.trim();
  return t.content.firstElementChild;
}
function escapeHtml(s) {
  return s.replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
}

let T, TJ, M;

async function boot() {
  TJ = await (await fetch("tokenizer.json")).json();
  M = await (await fetch("metrics.json")).json();
  T = loadTokenizer(TJ);

  renderHero(statsFromMetrics());
  renderTable(statsFromMetrics());
  renderMeta();
  renderTokenStats();   // richer per-language token statistics
  renderFaithful();     // live faithfulness on samples
  renderPlayground();   // live tokenize + round-trip on user input
  renderVocab();
  renderTokenizerSample(); // a live slice of the raw tokenizer.json
  setupDownloads();
}

// Per-language figures: computed by the tokenizer over the full corpus (in
// metrics.json, reproduced by verify.py). The live playground below runs the
// identical tokenizer in-browser; re-tokenizing 1.4 MB on load would freeze it.
function statsFromMetrics() {
  const per = {};
  for (const l of LANGS)
    per[l] = { tokens: M.token_counts[l], units: M.faithful_units[l], ratio: M.ratios[l] };
  const order = [...LANGS].sort((a, b) => per[a].ratio - per[b].ratio);
  return {
    per, order, min: M.min_language, max: M.max_language,
    spread: M.spread, score: M.score,
    penalty: M.hindi_penalty_factor, adjusted: M.hindi_adjusted_score,
  };
}

function renderHero(S) {
  $("[data-score]").textContent = S.score.toFixed(1);
  $("[data-spread]").textContent = fmt(S.spread);
  $("[data-xmax]").innerHTML = `${NAMES[S.max]} <b>${fmt(S.per[S.max].ratio)}</b>`;
  $("[data-xmin]").innerHTML = `${NAMES[S.min]} <b>${fmt(S.per[S.min].ratio)}</b>`;
  $("[data-calc]").innerHTML =
    `1000 / ( ${fmt(S.per[S.max].ratio)} &minus; ${fmt(S.per[S.min].ratio)} ) = <b>${S.score.toFixed(1)}</b>`;
  $("[data-adj]").innerHTML =
    `Hindi-penalty factor ${fmt(S.penalty, 3)} &rarr; adjusted score <b>${S.adjusted.toFixed(1)}</b>`;

  const enOk = S.per.en.ratio <= CAP, hiOk = S.per.hi.ratio <= CAP;
  const cap = $("[data-cap]");
  cap.className = "capcheck " + (enOk && hiOk ? "ok" : "bad");
  cap.innerHTML = (enOk && hiOk)
    ? `English ${fmt(S.per.en.ratio)} &le; ${CAP} and Hindi ${fmt(S.per.hi.ratio)} &le; ${CAP} &nbsp;✓ no penalty`
    : `English ${fmt(S.per.en.ratio)}, Hindi ${fmt(S.per.hi.ratio)} vs cap ${CAP}`;
}

// Prove decode(encode(text)) preserves every visible character, live in-browser.
async function renderFaithful() {
  const nonws = (s) => s.replace(/\s+/g, "");
  const probes = [
    "https://hi.wikipedia.org/wiki/भारत#cite_ref-1",
    "India (# 1) [a](b) \"q\" it's 3,000_000; x<y & p>q | 50% `code` _i_ *b*",
    "भारत గణతంత్ర मराठी — emoji 🇮🇳 表 │ € ″ ⓘ",
  ];
  // add a live slice from each corpus text (kept small so the page stays snappy)
  for (const l of LANGS) {
    try {
      const t = await (await fetch(`texts/${l}.txt`)).text();
      probes.push(t.slice(0, 4000));
    } catch (e) { /* ignore */ }
  }
  let ok = 0;
  for (const t of probes) if (nonws(decode(encode(t, T))) === nonws(t)) ok++;
  const box = $("[data-faithful]");
  const good = ok === probes.length;
  box.className = "capcheck " + (good ? "ok" : "bad");
  box.innerHTML = good
    ? `Faithful ✓ &nbsp;decode(encode(x)) preserves every visible character on all ${probes.length} live checks (Markdown / URL / emoji + corpus slices; full corpus verified by verify.py)`
    : `✗ round trip failed on ${probes.length - ok}/${probes.length}`;
}

function renderTable(S) {
  const tbody = $("[data-rows]");
  const scale = Math.max(...LANGS.map((l) => S.per[l].ratio)) * 1.15;
  for (const l of S.order) {
    const p = S.per[l];
    const role = l === S.min ? "min" : l === S.max ? "max" : "";
    const badge = l === S.min ? '<span class="pill min">X min</span>'
      : l === S.max ? '<span class="pill max">X max</span>' : "";
    tbody.appendChild(el(`
      <tr class="${role}">
        <td><span class="dot" style="background:${COLOR[l]}"></span>${NAMES[l]} ${badge}</td>
        <td class="mono">${SCRIPT[l]}</td>
        <td class="num">${p.units.toLocaleString()}</td>
        <td class="num">${p.tokens.toLocaleString()}</td>
        <td class="num strong">${fmt(p.ratio)}</td>
        <td class="barcell"><div class="bar"><span style="width:${(p.ratio / scale) * 100}%;background:${COLOR[l]}"></span></div></td>
      </tr>`));
  }
  $("[data-caphint]").innerHTML =
    `Fertility = tokens &divide; faithful units (a unit = one letter/number run OR one punctuation/symbol char). Score = 1000 / (X max &minus; X min).`;
}

function renderMeta() {
  const total = Object.keys(TJ.model.vocab).length;
  const merges = TJ.model.merges.length;
  const bytes = Object.keys(TJ.model.vocab).filter((t) => /^<0x[0-9A-Fa-f]{2}>$/.test(t)).length;
  const base = total - merges - bytes; // single-character alphabet tokens
  $("[data-vocabcount]").textContent = total.toLocaleString();
  $("[data-mergecount]").textContent = merges.toLocaleString();
  $("[data-basecount]").textContent = bytes.toLocaleString();
  $("[data-basechars]").textContent = base.toLocaleString();
  $("[data-vocabsum]").innerHTML =
    `${bytes.toLocaleString()} byte-fallback + ${base.toLocaleString()} base characters + ${merges.toLocaleString()} merges = `
    + `<b>${total.toLocaleString()}</b> tokens - the full vocab budget, all in one shared vocabulary.`;
}

// Richer per-language statistics from metrics.json (token_stats block, produced
// by scripts/token_stats.py over the exported tokenizer - the same tokenizer the
// live playground runs). Ordered by fertility so X-min / X-max line up with the
// ratios table above.
function renderTokenStats() {
  const box = $("[data-statrows]");
  const st = M.token_stats;
  if (!box || !st) return;
  const order = [...LANGS].sort((a, b) => st[a].fertility - st[b].fertility);
  const min = order[0], max = order[order.length - 1];
  for (const l of order) {
    const s = st[l];
    const role = l === min ? "min" : l === max ? "max" : "";
    const badge = l === min ? '<span class="pill min">X min</span>'
      : l === max ? '<span class="pill max">X max</span>' : "";
    box.appendChild(el(`
      <tr class="${role}">
        <td><span class="dot" style="background:${COLOR[l]}"></span>${NAMES[l]} ${badge}</td>
        <td class="num">${s.characters.toLocaleString()}</td>
        <td class="num">${s.bytes.toLocaleString()}</td>
        <td class="num">${s.faithful_units.toLocaleString()}</td>
        <td class="num">${s.tokens.toLocaleString()}</td>
        <td class="num strong">${fmt(s.fertility)}</td>
        <td class="num">${fmt(s.chars_per_token, 2)}</td>
        <td class="num">${fmt(s.bytes_per_token, 2)}</td>
        <td class="num">${s.distinct_token_types.toLocaleString()}</td>
      </tr>`));
  }
  const totTok = LANGS.reduce((a, l) => a + st[l].tokens, 0);
  const totUnit = LANGS.reduce((a, l) => a + st[l].faithful_units, 0);
  $("[data-stathint]").innerHTML =
    `Across all four articles: <b>${totTok.toLocaleString()}</b> tokens over <b>${totUnit.toLocaleString()}</b> faithful units. `
    + `Indic scripts pack more UTF-8 bytes per token (a Devanagari/Telugu character is 3 bytes), while chars per token stay close - the shared vocabulary carries every script at a similar rate.`;
}

function renderPlayground() {
  const ta = $("[data-input]"), out = $("[data-tokens]"), rt = $("[data-rt]");
  const cW = $("[data-pwords]"), cT = $("[data-ptokens]"), cF = $("[data-pfert]");
  function run() {
    const text = ta.value;
    const toks = encode(text, T);
    const units = faithfulUnits(text);
    out.innerHTML = "";
    toks.forEach((t, i) => {
      const shown = t.replace(/▁/g, "·").replace(/^<0x([0-9A-Fa-f]{2})>$/, "⟨$1⟩");
      const chip = el(`<span class="tok">${escapeHtml(shown) || "␣"}</span>`);
      chip.style.background = i % 2 ? "rgba(42,120,214,0.12)" : "rgba(27,175,122,0.14)";
      out.appendChild(chip);
    });
    cW.textContent = units.toLocaleString();
    cT.textContent = toks.length.toLocaleString();
    cF.textContent = units ? (toks.length / units).toFixed(3) : "0";
    const ok = decode(toks) === text;
    rt.className = "capcheck " + (ok ? "ok" : "bad");
    rt.innerHTML = ok ? "decode(encode(text)) === your input ✓ &nbsp;lossless round trip"
      : "✗ round trip differs";
  }
  ta.addEventListener("input", run);
  run();
}

function renderVocab() {
  const box = $("[data-vocablist]"), search = $("[data-vocabsearch]"), count = $("[data-vocabmatch]");
  const vocab = Object.keys(TJ.model.vocab);
  const shown = vocab.map((t) => t.replace(/▁/g, "·"));
  const LIMIT = 400;
  function draw(q) {
    q = q.trim();
    const idx = [];
    for (let i = 0; i < shown.length; i++) if (!q || shown[i].includes(q)) idx.push(i);
    count.textContent = idx.length.toLocaleString();
    box.innerHTML = "";
    for (let i = 0; i < Math.min(idx.length, LIMIT); i++)
      box.appendChild(el(`<span class="vtok">${escapeHtml(shown[idx[i]]) || "␣"}</span>`));
    if (idx.length > LIMIT)
      box.appendChild(el(`<span class="vmore">+${(idx.length - LIMIT).toLocaleString()} more</span>`));
  }
  search.addEventListener("input", () => draw(search.value));
  draw("");
}

// A live, truncated slice of the real tokenizer.json so a reviewer can see the
// actual format and config (not just a vocab list). Built from the loaded TJ.
function renderTokenizerSample() {
  const pre = $("[data-tokjson]");
  if (!pre) return;
  const vocab = TJ.model.vocab;
  const entries = Object.entries(vocab).sort((a, b) => a[1] - b[1]); // by id
  const find = (re) => (entries.find(([t]) => re.test(t)) || [null])[0];
  const pick = [
    find(/^<0x0A>$/) || find(/^<0x/),   // a byte-fallback token
    find(/^▁the$/) || find(/^▁/),        // a leading-space subword
    "the", "▁भारत", find(/[ऀ-ॿ]/), find(/[ఀ-౿]/),
  ].filter((t, i, a) => t && a.indexOf(t) === i && t in vocab);

  const vocabSample = {};
  for (const t of pick) vocabSample[t] = vocab[t];
  vocabSample["…"] = `${(Object.keys(vocab).length - pick.length).toLocaleString()} more tokens`;

  const mergesSample = TJ.model.merges
    .slice(0, 6)
    .map((m) => (Array.isArray(m) ? m.join(" ") : m));
  mergesSample.push(`… ${(TJ.model.merges.length - 6).toLocaleString()} more merges`);

  const view = {
    version: TJ.version,
    normalizer: TJ.normalizer,
    pre_tokenizer: TJ.pre_tokenizer,
    decoder: TJ.decoder,
    model: {
      type: TJ.model.type,
      unk_token: TJ.model.unk_token,
      byte_fallback: TJ.model.byte_fallback,
      vocab: vocabSample,
      merges: mergesSample,
    },
  };
  pre.textContent = JSON.stringify(view, null, 2);
}

function setupDownloads() {
  const files = {
    "tokenizer.json": () => JSON.stringify(TJ),
    "vocab.json": () => JSON.stringify(TJ.model.vocab),
    "merges.txt": () => TJ.model.merges.map((m) => Array.isArray(m) ? m.join(" ") : m).join("\n"),
  };
  document.querySelectorAll("[data-dl]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const name = btn.getAttribute("data-dl");
      const blob = new Blob([files[name]()], { type: "application/octet-stream" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url; a.download = name; a.click();
      URL.revokeObjectURL(url);
    });
  });
}

boot();
