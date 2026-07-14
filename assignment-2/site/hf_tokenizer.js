// Faithful JS reimplementation of the exported HuggingFace BPE tokenizer:
// BPE + byte_fallback + Metaspace (replacement "▁", prepend_scheme "never"),
// no normalizer. Verified token-for-token against the Python `tokenizers`
// library (see scripts/test_widget_js.py). Encoding/decoding depend only on the
// model vocab + merges shipped in tokenizer.json, so the widget tokenizes live.

// --- UTF-8 (manual, so it runs in any engine / the browser) ---
function utf8Bytes(ch) {
  const cp = ch.codePointAt(0);
  if (cp < 0x80) return [cp];
  if (cp < 0x800) return [0xc0 | (cp >> 6), 0x80 | (cp & 0x3f)];
  if (cp < 0x10000)
    return [0xe0 | (cp >> 12), 0x80 | ((cp >> 6) & 0x3f), 0x80 | (cp & 0x3f)];
  return [0xf0 | (cp >> 18), 0x80 | ((cp >> 12) & 0x3f),
          0x80 | ((cp >> 6) & 0x3f), 0x80 | (cp & 0x3f)];
}
function utf8Decode(bytes) {
  let s = "", i = 0;
  while (i < bytes.length) {
    const c = bytes[i++];
    let cp;
    if (c < 0x80) cp = c;
    else if ((c >> 5) === 0x6) cp = ((c & 0x1f) << 6) | (bytes[i++] & 0x3f);
    else if ((c >> 4) === 0xe)
      cp = ((c & 0xf) << 12) | ((bytes[i++] & 0x3f) << 6) | (bytes[i++] & 0x3f);
    else
      cp = ((c & 0x7) << 18) | ((bytes[i++] & 0x3f) << 12) |
           ((bytes[i++] & 0x3f) << 6) | (bytes[i++] & 0x3f);
    s += String.fromCodePoint(cp);
  }
  return s;
}

// Build a fast index from a parsed tokenizer.json.
export function loadTokenizer(tj) {
  const vocab = new Set(Object.keys(tj.model.vocab));
  const ranks = new Map();
  const merges = tj.model.merges;
  for (let i = 0; i < merges.length; i++) {
    const m = merges[i];
    const pair = Array.isArray(m) ? m[0] + " " + m[1] : m;
    ranks.set(pair, i);
  }
  return { vocab, ranks, byteFallback: !!tj.model.byte_fallback };
}

// Metaspace pre-tokenization: replace ' ' with ▁, split so each ▁ starts a piece.
export function preTokenize(text) {
  const s = text.replace(/ /g, "▁");
  const pieces = [];
  let cur = "";
  for (const ch of s) {
    if (ch === "▁") {
      if (cur) pieces.push(cur);
      cur = "▁";
    } else {
      cur += ch;
    }
  }
  if (cur) pieces.push(cur);
  return pieces;
}

// BPE over one piece, with byte fallback for out-of-vocab characters.
function encodePiece(piece, T) {
  let syms = [];
  for (const ch of piece) {
    if (T.vocab.has(ch)) {
      syms.push(ch);
    } else if (T.byteFallback) {
      for (const b of utf8Bytes(ch))
        syms.push("<0x" + b.toString(16).toUpperCase().padStart(2, "0") + ">");
    } else {
      syms.push(ch);
    }
  }
  while (syms.length > 1) {
    let bestRank = Infinity, bestI = -1;
    for (let i = 0; i < syms.length - 1; i++) {
      const r = T.ranks.get(syms[i] + " " + syms[i + 1]);
      if (r !== undefined && r < bestRank) { bestRank = r; bestI = i; }
    }
    if (bestI < 0) break;
    syms.splice(bestI, 2, syms[bestI] + syms[bestI + 1]);
  }
  return syms;
}

export function encode(text, T) {
  const out = [];
  for (const piece of preTokenize(text))
    for (const t of encodePiece(piece, T)) out.push(t);
  return out;
}

// Decoder: Sequence[ Replace(▁," "), ByteFallback, Fuse ].
export function decode(tokens) {
  const replaced = tokens.map((t) => t.replace(/▁/g, " "));
  let out = "", buf = [];
  const flush = () => { if (buf.length) { out += utf8Decode(buf); buf = []; } };
  for (const t of replaced) {
    const m = /^<0x([0-9A-Fa-f]{2})>$/.exec(t);
    if (m) buf.push(parseInt(m[1], 16));
    else { flush(); out += t; }
  }
  flush();
  return out;
}

// Faithful units for the playground display (the score denominator): one
// letter/mark/number run OR one visible punctuation/symbol char. Uses Unicode
// property escapes where supported, with a plain-regex fallback for old engines.
let UNIT_RE;
try {
  UNIT_RE = new RegExp("[\\p{L}\\p{M}\\p{N}]+|[^\\s\\p{L}\\p{M}\\p{N}]", "gu");
} catch (e) {
  UNIT_RE = /[0-9A-Za-zÀ-￿]+|[^\s0-9A-Za-zÀ-￿]/g;
}
export function faithfulUnits(text) {
  const m = text.match(UNIT_RE);
  return m ? m.length : 0;
}
