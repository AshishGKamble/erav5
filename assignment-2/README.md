# Assignment 2 - Faithful multilingual BPE tokenizer

A single **10,000-token Byte-Pair-Encoding (BPE) tokenizer** shared across the
**India** Wikipedia article in **English, Hindi, Telugu and Marathi**, plus a
self-contained browser **widget** that tokenizes live and shows the score.

The design goal is to make each language compress at a **similar** tokens-per-unit
rate (a small spread), **subject to a hard requirement**: the tokenizer must be
*faithful*, i.e. `decode(encode(text)) == text` for the visible characters of any
input. A tokenizer that silently drops characters produces low token counts but
is disqualified.

This README is a developer guide: read it top to bottom to understand the
packages, the architecture, the full code flow, and every design decision.

---

## Table of contents

1. [The assignment, precisely](#1-the-assignment-precisely)
2. [Technologies and packages](#2-technologies-and-packages)
3. [Architecture at a glance](#3-architecture-at-a-glance)
4. [Design decisions (the why)](#4-design-decisions-the-why)
5. [Code flow, file by file](#5-code-flow-file-by-file)
6. [The scoring metric, explained](#6-the-scoring-metric-explained)
7. [The widget: live in-browser tokenization](#7-the-widget-live-in-browser-tokenization)
8. [Reproduce and verify](#8-reproduce-and-verify)
9. [Results](#9-results)
10. [File reference](#10-file-reference)

---

## 1. The assignment, precisely

The design answers four constraints. Everything downstream exists to satisfy them.

| Constraint | What it means |
|---|---|
| **10,000-token budget** | One shared vocabulary of exactly 10,000 tokens covers all four languages (not 10k per language). |
| **Fertility metric** | For each language, `X = tokens / faithful_units`. A *faithful unit* is one Unicode letter/mark/number run **or** one visible punctuation/symbol character. |
| **Score** | Sort the four X values; `score = 1000 / (X_max - X_min)`. Smaller spread = higher score. |
| **Guard rails** | English `X <= 1.2`. A Hindi penalty `exp(max(0, X_hi/1.2 - 1))` divides the score if Hindi exceeds 1.2. |
| **Faithfulness (hard gate)** | `decode(encode(text))` must preserve every visible (non-whitespace) character, or the result is void. |

The whole tokenizer is engineered so that (a) faithfulness is guaranteed by
construction, and (b) the four fertilities are pushed close together.

---

## 2. Technologies and packages

**Language / runtime:** Python 3 for the training + verification pipeline;
vanilla JavaScript (ES modules, no framework, no build step) for the widget.

| Package | Role in this project | Why this one |
|---|---|---|
| `tokenizers` (HuggingFace) | Trains the BPE model, applies `byte_fallback` + Metaspace, and exports a portable `tokenizer.json`. Also used by `verify.py` to load and check. | Fast Rust-backed BPE with first-class `byte_fallback`, and a standard on-disk format anyone can load with `Tokenizer.from_file()`. |
| `regex` | Counts *faithful units* using Unicode property escapes (`\p{L}`, `\p{M}`, `\p{N}`). | Python's built-in `re` does not support `\p{...}`; `regex` does, and the metric is defined in those terms. |
| `requests` | Fetches each article's HTML from the Wikipedia REST API. | Simple, reliable HTTP. |
| `beautifulsoup4` + `lxml` | Parse the fetched HTML and prune inert machinery (script/style/meta/link, edit affordances). | `bs4` is the standard HTML DOM tool; `lxml` is its fast parser backend. |
| `html2text` | Converts the cleaned HTML into faithful Markdown-ish text (the training corpus). | Independent conversion path; configurable to never wrap lines and to keep Unicode verbatim (critical for Indic scripts). |
| `py_mini_racer` | Runs the widget's JavaScript tokenizer inside Python to compare it against the Python library. | A headless JS engine, so the browser tokenizer can be proven equal to Python token-for-token. |

The widget itself has **no dependencies** - it is plain HTML/CSS/JS served as
static files (deployed on Netlify).

---

## 3. Architecture at a glance

The pipeline is five deterministic stages. Each stage reads the previous stage's
files and writes the next. Arrows show data artifacts.

```
        Wikipedia REST HTML  (en / hi / te / mr "India" articles)
                 │
                 │  1. scripts/build_faithful_corpus.py
                 │     requests -> bs4/lxml prune -> html2text -> tidy()
                 ▼
     corpus/{lang}.faithful.txt   (+ .faithful.md, .meta.json, .raw.html)
                 │
                 │  2. scripts/hf_build.py
                 │     weight search -> train BPE(byte_fallback) ->
                 │     demote 256 byte tokens to non-special -> export
                 ▼
     tokenizer/tokenizer.json   +   tokenizer/metrics.json
                 │                         (also written into site/)
                 │  3. scripts/token_stats.py
                 │     augment metrics.json with per-language token stats
                 ▼
     metrics.json  (now includes token_stats)
                 │
                 ├─ 4. scripts/verify.py         independent faithfulness + score check
                 └─ 4. scripts/test_widget_js.py JS tokenizer == Python (via py_mini_racer)
                 │
                 ▼
     site/  (index.html + app.js + hf_tokenizer.js + styles.css
             + tokenizer.json + metrics.json + texts/)
                 │
                 │  5. deploy (drag site/ to Netlify)
                 ▼
            Live widget: tokenizes in the browser, shows the score
```

Two copies of `tokenizer.json` / `metrics.json` exist by design: the canonical
pair under `tokenizer/`, and a copy under `site/` that the static widget fetches.
`hf_build.py` writes both; `token_stats.py` updates both.

---

## 4. Design decisions (the why)

Each decision below is a direct answer to a constraint in section 1. The
rationale and the rejected alternatives are what matter.

### 4.1 Byte-fallback BPE (the core choice)

```
subword (BPE merges)  ->  character atoms  ->  byte fallback
```

The tokenizer resolves text in three tiers: known merges, then single known
characters, then raw UTF-8 bytes for anything unseen.

| Alternative | Why rejected |
|---|---|
| **Character-level** | The original attempt. Any character never seen in training is dropped, so `decode(encode(x)) != x` and the score is void. |
| **Pure byte-level** (GPT-2 style) | Faithful, but every Devanagari/Telugu character is 3 UTF-8 bytes = ~3 tokens, so Indic fertility explodes. |
| **BPE with `[UNK]`** | Unseen characters collapse to `[UNK]`; the round trip is lossy. |
| **Byte-fallback BPE (chosen)** | Seen characters (including every Indic letter) are single atoms, so Indic text stays compact; anything unseen falls back to its raw bytes, so the round trip is exact for *any* input - Markdown, URLs, emoji, digit separators. |

### 4.2 No normalizer

NFKC normalization rewrites characters (`"` -> `''`, `ﬁ` -> `fi`, full-width to
half-width). That would break the exact round trip. Byte fallback already
guarantees full coverage, so a normalizer buys nothing and only risks
faithfulness. `normalizer` is therefore `null`.

### 4.3 Metaspace pre-tokenizer and decoder

Spaces are replaced with `▁` (U+2581) so word boundaries survive tokenization and
are exactly recoverable: encode maps space -> `▁`, decode maps `▁` -> space
(`prepend_scheme = never`, so no phantom leading space is added). This is the
SentencePiece convention and keeps everything inside the character/byte tiers
above, rather than routing bytes through a ByteLevel layer.

### 4.4 The 256 byte tokens are demoted to non-special

`<0x00>`..`<0xFF>` are added as `special_tokens` during training so they are
guaranteed to be in the vocabulary. But special tokens are skipped by the default
`decode()`. So after training, `hf_build.py` rewrites `tokenizer.json` to set
`special = false` on all 256 byte tokens. Now the **default** `decode()`
reconstructs bytes back into characters, and faithfulness holds with no custom
decode logic on the grader's side.

### 4.5 Faithful full-article corpus (our own extraction)

The corpus is the **full article as faithful text** (headings, lists, links,
tables, references, punctuation), not clipped prose. Clipped prose inflates the
fertility ratios and is not reproducible. The extraction is our own pipeline
(`html2text` + our own DOM pruning and whitespace tidying) so the corpus is
independently generated, and `html2text` is configured with `unicode_snob` (no
ASCII folding) and `body_width = 0` (no line wrapping) to keep the text faithful.

### 4.6 Per-language weighting to minimize spread

To flatten the four fertilities, each language's text is repeated `w` times in the
training set, so BPE spends its merge budget on each language in proportion to its
weight. `hf_build.py` runs a small grid search over weights and keeps the
combination that **maximizes the score** (minimizes `X_max - X_min`) while
staying faithful and keeping English `<= 1.19`. The shipped weights are
`en:2, hi:2, te:4, mr:5`.

### 4.7 A JavaScript reimplementation for the widget (and a parity test)

The widget must tokenize live in the browser with no backend. `hf_tokenizer.js`
reimplements the exact encode/decode from the shipped `tokenizer.json`. Because a
reimplementation could drift from the real library, `test_widget_js.py` runs the
JS inside `py_mini_racer` and asserts it produces the **same token ids as the
Python `tokenizers` library**, token for token, on samples and on the full
corpus. Only then are the widget's numbers trustworthy.

---

## 5. Code flow, file by file

### `scripts/build_faithful_corpus.py` - build the corpus

- **Input:** none (fetches over the network). **Output:** `corpus/{lang}.faithful.txt`, `.faithful.md`, `.meta.json`, `.raw.html`.
- **Flow:** for each of en/hi/te/mr:
  1. `fetch_html()` - GET the article's REST HTML.
  2. `prune()` - parse with `bs4`/`lxml`, drop `script/style/meta/link/noscript` and MediaWiki edit affordances, return the `<body>` HTML.
  3. `to_markdown()` - an `html2text` converter configured `body_width=0`, `unicode_snob=True`, links/images kept.
  4. `tidy()` - collapse runs of blank lines, strip trailing spaces, keep every visible character.
- **Key idea:** the `.txt` the tokenizer trains on is the same faithful Markdown a human would read.

### `scripts/hf_build.py` - train, tune, export

- **Input:** `corpus/{lang}.faithful.txt`. **Output:** `tokenizer/tokenizer.json` + `metrics.json`, and copies in `site/`.
- **Key functions:**
  - `train_tokenizer(weights)` - writes each corpus `weights[lang]` times into a temp dir, trains `BPE(byte_fallback=True)` with a Metaspace pre-tokenizer and a `Sequence[Replace(▁->space), ByteFallback, Fuse]` decoder, vocab 10,000, then **demotes the 256 byte tokens to non-special** in the exported JSON.
  - `evaluate(tok)` - computes each language's fertility, the spread, the score, the Hindi penalty, and re-asserts faithfulness.
  - `search()` - grid search over weights; keeps the faithful, English `<= 1.19` combination with the best score.
  - `main()` - rebuilds the winner, prints the table, writes `tokenizer.json` + `metrics.json` to both `tokenizer/` and `site/`.
- **Guarantee:** it will not export unless `decode(encode(x)) == x` holds.

### `scripts/token_stats.py` - richer per-language statistics

- **Input:** the exported `tokenizer.json` + corpus. **Output:** adds a `token_stats` block to both `metrics.json` files. **Does not retrain**, so the score is untouched.
- Computes per language: characters, UTF-8 bytes, faithful units, tokens, fertility, chars-per-token, bytes-per-token, and the number of distinct token types used.

### `scripts/verify.py` - the independent check

- Loads `tokenizer.json` with `Tokenizer.from_file()`, then:
  1. asserts `decode(encode(x)) == x` on the four texts plus Markdown/URL/emoji samples (the faithfulness gate);
  2. recomputes fertilities, spread and score with the **same unit regex and `1000/spread` formula the assignment defines**, and checks them against `metrics.json`.
- This is the file to run to trust the numbers; it re-derives everything from scratch.

### `scripts/test_widget_js.py` - widget parity

- Runs `site/hf_tokenizer.js` inside `py_mini_racer` and compares its encode/decode against the Python `tokenizers` library on samples and on the full corpus. Prints a per-language token-count match. This is what makes the browser numbers as trustworthy as Python's.

### `site/` - the widget (see section 7)

- `hf_tokenizer.js` - the faithful JS tokenizer (encode/decode/faithfulUnits).
- `app.js` - fetches `tokenizer.json` + `metrics.json`, renders every section, runs the live playground and faithfulness checks.
- `index.html` / `styles.css` - structure and styling.
- `tokenizer.json` / `metrics.json` / `texts/` - the data the page reads.

---

## 6. The scoring metric, explained

```
faithful unit = one [\p{L}\p{M}\p{N}]+ run   OR   one visible punctuation/symbol char
             regex:  [\p{L}\p{M}\p{N}]+|[^\s\p{L}\p{M}\p{N}]

fertility(lang) = tokens(lang) / faithful_units(lang)
spread          = max(fertility) - min(fertility)
score           = 1000 / spread
hindi_penalty   = exp(max(0, fertility(hi)/1.2 - 1))          # 1.0 while hi <= 1.2
adjusted_score  = score / hindi_penalty
```

Fertility below 1.0 means the tokenizer emits fewer tokens than there are units
(strong compression), because BPE merges span multiple units and the faithful-unit
count includes every punctuation character. Lower fertility is better compression;
the **1.2 is a ceiling** (a guard against degenerate tokenizers), not a target.
The score depends only on the **spread**, so aligning the four languages matters
more than the absolute level.

---

## 7. The widget: live in-browser tokenization

`site/` is a static page with no backend. On load, `app.js` `boot()`:

1. `fetch("tokenizer.json")` and `fetch("metrics.json")`.
2. `loadTokenizer(tj)` builds a fast index: a `Set` of vocab tokens and a `Map`
   from merge-pair to rank.
3. Renders each section.

**How `hf_tokenizer.js` encodes** (mirrors HuggingFace exactly):

1. `preTokenize()` - replace spaces with `▁` and split so each `▁` starts a piece.
2. `encodePiece()` - split the piece into symbols: a known character stays as
   itself, an unknown character becomes its `<0xNN>` UTF-8 byte tokens. Then
   greedily merge the adjacent pair with the **lowest merge rank** until no merge
   applies (the standard BPE loop).
3. Concatenate the pieces' tokens.

**How it decodes:** map `▁` back to space, collect consecutive `<0xNN>` tokens and
UTF-8-decode them together (the `Fuse` step), emit everything else verbatim.

**What each section shows:**

- **Self score** and **per-language ratios** - read from `metrics.json`. (Re-tokenizing ~1.4 MB of corpus on every page load would freeze the browser, so the heavy figures are precomputed by the Python pipeline and reproduced exactly by `verify.py`.)
- **Faithfulness banner** - live: runs `decode(encode(x))` in the browser on Markdown/URL/emoji samples plus slices of each corpus text.
- **Token statistics by language** - from the `token_stats` block.
- **Tokenize anything** - live playground; tokenizes your input, shows units/tokens/fertility, and checks the round trip.
- **The vocabulary** - browse/search all 10,000 tokens.
- **Sample from tokenizer.json** - a live slice of the real file (config + sample vocab + merges).
- **Folder contents** - this repo's layout.
- **Downloads** - `tokenizer.json`, `vocab.json`, `merges.txt`, and the corpus texts.

---

## 8. Reproduce and verify

```bash
cd assignment-2
pip install tokenizers regex requests beautifulsoup4 lxml html2text py_mini_racer

# 1. fetch the faithful corpus (India page, en/hi/te/mr) -> corpus/
python3 scripts/build_faithful_corpus.py

# 2. train + tune weights + export tokenizer.json and metrics.json
python3 scripts/hf_build.py

# 3. add per-language token statistics to metrics.json (no retrain)
python3 scripts/token_stats.py

# 4. independently verify faithfulness AND the score
python3 scripts/verify.py

# 5. confirm the widget's JS tokenizer matches Python token-for-token
python3 scripts/test_widget_js.py

# preview the widget locally
cd site && python3 -m http.server 8000   # open http://localhost:8000
```

Deploy: `site/` is a plain static folder - drag it onto
<https://app.netlify.com/drop> (or any static host). The downloaded
`tokenizer.json` loads directly with `Tokenizer.from_file()`.

Note: the corpus is a live Wikipedia snapshot, so re-fetching may shift the exact
byte counts as the articles are edited. Faithfulness and English `<= 1.2` are
robust to that; the score magnitude depends on the snapshot.

---

## 9. Results

Fertility `X = tokens / faithful_units`. Score `= 1000 / (X_max - X_min)`.

| Language | Script | Faithful units | Tokens | X = tokens/unit |
|---|---|---:|---:|---:|
| Hindi | Devanagari | 72,170 | 55,834 | **0.7736** (X min) |
| English | Latin | 157,936 | 123,111 | 0.7795 |
| Marathi | Devanagari | 24,384 | 19,354 | 0.7937 |
| Telugu | Telugu | 29,315 | 23,537 | **0.8029** (X max) |

- English `0.7795 <= 1.2` and Hindi `0.7736 <= 1.2` (no Hindi penalty)
- spread `= 0.8029 - 0.7736 = 0.0293`
- **score `= 1000 / 0.0293 = 34,183`** (penalty factor 1.000, adjusted = raw)

Vocabulary composition: `256 byte-fallback + 300 base characters + 9,444 merges =
10,000` tokens - the full budget, in one shared vocabulary.

---

## 10. File reference

```
assignment-2/
  README.md                      this guide
  corpus/
    {lang}.faithful.txt          faithful text the tokenizer trains and is scored on
    {lang}.faithful.md           same content, .md extension
    {lang}.meta.json             source URL, converter, fetch time, counts
    {lang}.raw.html              the fetched HTML (intermediate; regenerable)
  scripts/
    build_faithful_corpus.py     fetch + convert the four India pages
    hf_build.py                  train the byte-fallback BPE, tune weights, export
    token_stats.py               add per-language token statistics to metrics.json
    verify.py                    standalone verifier: faithfulness + the score
    test_widget_js.py            prove the widget's JS == Python, token-for-token
  tokenizer/
    tokenizer.json               the trained tokenizer (HuggingFace format)
    metrics.json                 ratios, spread, score, token statistics
  site/                          the deployable widget
    index.html  app.js  hf_tokenizer.js  styles.css
    tokenizer.json  metrics.json  netlify.toml
    texts/{lang}.txt             corpus copies the widget fetches
```
