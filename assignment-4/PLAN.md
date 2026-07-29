# Assignment 4 - Build Plan (for approval)

> Status: **plan only, no code yet.** Approve this, then we build.
> Assignment: count the Session-4 data-cleaning strategies, pick a 10-100M-token dataset,
> actually run the cleanups on a slice, and publish a widget to Netlify.

---

## 0. The graded answer: how many strategies?

**8 strategies**, triangulated three ways (pipeline-map = 8 stages; closing commitment = 8;
TOC collapses the two dedup lessons to 8):

1. Extract  2. Normalize  3. Language ID  4. Quality filter
5. Deduplicate  6. PII scrub  7. Decontaminate  8. Manifest

Nuance we will state (proves we read the widgets):
- **Ghost-tag / format discipline is not a 9th stage** - it lives inside Normalize
  (`clean_text()` has a "flag ghost special tokens" toggle). We treat it as a named
  **bonus concern**, not a stage.
- **Deduplication is taught in two parts** (mechanism = MinHash+LSH; scale = global dedup).
- **Safety/toxic filtering** is a second bonus concern (indic-align ships a toxic subset).

---

## 1. Data

**Corpus to clean:** `CharuAgarwal/indic-align` (mirror of `ai4bharat/indic-align`, CC-BY-4.0).
- Stream a **12-language slice** = our Assignment-3 set: Hindi, Bengali, Marathi, Telugu, Tamil,
  Gujarati, Urdu, Kannada, Odia, Malayalam, Punjabi, Assamese.
- Cap **~3-4M tokens/language -> ~40M tokens total** (inside 10-100M). First-N per language,
  fixed order -> reproducible -> feeds the manifest.
- We never download the full 28 GB; streaming stops at the cap (~a few hundred MB).

**Decontamination hold-out (never trained on, only scanned against):**
`sarvamai/mmlu-indic` + `sarvamai/trivia-qa-indic-mcq` (both benchmark MCQ sets). This makes
Strategy 7 real, not illustrative.

**Verify at code time:** exact config/language availability in indic-align for all 12 (esp.
Assamese, Urdu); a tokenizer for real token counts + fertility; NER coverage per script.

---

## 2. The cleaning pipeline (Python, offline, one pass per stage)

Each stage logs `docs in -> docs out`, `tokens in -> tokens out`, and *what/why* removed, into a
machine-readable `stats.json` the widget consumes. It also captures 1-2 real **before/after**
examples per stage for the widget.

| # | Stage | What it does on indic-align |
|---|-------|------------------------------|
| 1 | **Extract** | Light (already text); strip residual markup/HTML; noted as upstream/given |
| 2 | **Normalize** (`clean_text`) | NFC; strip zero-width/BOM/bidi/C0-C1; HTML-unescape; collapse whitespace; **preserve ZWJ/ZWNJ**; hash **after** cleaning. **+ ghost-tag bonus:** detect literal `[USER]`/`[ASSISTANT]`/`<|endoftext|>` and unify to one canonical format |
| 3 | **Language ID** (the star) | Detect language + script per doc; compare to claimed language; flag mismatches and code-switch (don't trust the label/path) |
| 4 | **Quality filter** | Gopher/C4 heuristics (word len, symbol ratio, stop-words, dup-line frac, ...) + **Indic Always-ON** channel so low-resource langs aren't dropped by English-tuned thresholds |
| 5 | **Deduplicate** | Shingle -> MinHash -> LSH, **global** across the whole slice (not per-language/per-file); report near-dup pairs removed |
| 6 | **PII scrub** | Regex layer (email, phone incl. +91, IPv4) + name layer (IndicNER or light list); report masked counts + the Indic-name precision/recall tension |
| 7 | **Decontaminate** | n-gram overlap of corpus vs the mmlu-indic + trivia-qa hold-out; drop/flag contaminated docs; report real leakage caught. **+ safety bonus:** flag/drop toxic docs |
| 8 | **Manifest** | Per-shard provenance JSON + **SHA-256 on cleaned text**, real token counts, lang distribution, determinism re-run check; gating rule: no shard ships without a complete manifest |

Ordering matters: `clean_text()` runs **before** the content hash, so dedup + manifest trust it.

---

## 3. Outputs of the pipeline

- `cleaned/` - cleaned corpus shards (jsonl/parquet) - likely gitignored (big)
- `stats.json` - per-stage yields + before/after counts (widget data source)
- `manifest.json` - provenance + hashes + token/lang distribution
- `examples.json` - a few real before/after snippets per strategy (widget)
- **Fertility:** compute real per-language fertility on the cleaned slice (ties to A3's spine)

---

## 4. The widget (static site -> Netlify)

Same stack as assignment-2/3: plain HTML + CSS + one vanilla JS file, no deps, no network,
theme-aware, dataviz-skill palette, hand-built SVG. Sections:

1. **Hero funnel** - docs/tokens surviving each of the 8 stages (real numbers).
2. **How many strategies?** - 8, listed, with the reconciliation nuance.
3. **Dataset picked** - indic-align, why (12 langs, A3 tie), + the two hold-out benchmarks.
4. **What was cleaned & why** - one card per strategy: the real stat + a before/after example.
5. **Bonus concerns** - ghost-tag/format discipline + safety filtering.
6. **Final statistics** - headline table: docs in/out, tokens in/out, dedup %, PII masked,
   contamination caught, per-language fertility.
7. **A2/A3 tie-in** - one line: same pipeline feeds the India-first model; extends to code/math.

Optional (if time): a couple of live in-browser mini-demos (clean_text live, MinHash toy) like the
session's own widgets - but the primary content is the **real computed stats**.

---

## 5. File layout

```
assignment-4/
  screenshots/        input (Session 4 strategy captures)
  PLAN.md             this plan
  pipeline/           cleaning scripts (Python)
  data/               downloaded slice + cleaned outputs (gitignored)
  site/               index.html, styles.css, app.js, netlify.toml
  README.md           developer guide (per the README standard)
  FINDINGS.md         strategy analysis + sourced numbers
```

---

## 6. Build order once approved

1. Slice + download script (stream indic-align 12-lang cap; fetch hold-out sets).
2. Cleaning pipeline (8 stages + 2 bonuses), emit stats/manifest/examples.
3. Run it for real; capture the numbers.
4. Build the widget from the real stats; README + FINDINGS.
5. Deploy to Netlify; share link.

---

## 7. Open decisions to confirm

- **Runtime/scope:** full ~40M-token slice, or a smaller cap (e.g. ~20M) for faster iteration?
- **Tokenizer for counts/fertility:** a reference Indic tokenizer, or reuse our A2 byte-level stance?
- **Widget:** static-stats-only, or add the 1-2 live in-browser mini-demos?
```
