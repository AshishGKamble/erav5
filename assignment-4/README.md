# Assignment 4 - India-First Data Cleanup

Apply the **eight data-cleaning strategies** from ERA V5 Session 4 to a real **10-100M-token** Indic
dataset, produce **real statistics**, and present them in a **walkable, figure-driven widget** deployed
to Netlify.

> The point of the assignment (in the tutor's words): *"download a dataset, apply these strategies, so
> you can see how the deduplication works, how the names are removed, how the emails are removed."*
> Every number in the widget is **computed by the pipeline on real data**, never invented.

- **Live report:** `site/index.html` (deploy `site/` to Netlify).
- **Research corpus / working notes:** [`FINDINGS.md`](FINDINGS.md).
- **The plan we agreed before coding:** [`PLAN.md`](PLAN.md).

---

## 1. What this answers (the brief, point by point)

| Brief question | Where it is answered |
|----------------|----------------------|
| How many strategies, and what are they? | **Eight** - the count section + the walkable pipeline |
| What dataset was picked (not the example)? | `ai4bharat/indic-align`, 12-language ~22M-token slice |
| What was cleaned and why? | The walkable 8-stage pipeline: what / why / tradeoff / the-mistake-it-prevents / example |
| Any other strategy or concern? | Two bonus concerns: **ghost tags** (inside Normalize) and **safety/toxicity** |
| Final statistics? | The Final-statistics table + the hero funnel, from `stats.json` |

---

## 2. The eight strategies

`Extract -> Normalize -> Language ID -> Quality filter -> Deduplicate -> PII scrub -> Decontaminate -> Manifest`

Two are worth calling out because the count is a graded question: **ghost-tag / format discipline** is
not a ninth stage (it lives inside Normalize), and **deduplication** is taught in two lessons
(MinHash+LSH mechanism, then global dedup). See [`FINDINGS.md`](FINDINGS.md) §1 for the full
triangulation.

---

## 3. Technology choices (and why)

### Pipeline (`pipeline/`)

| Choice | Why |
|--------|-----|
| **Plain Python, one file per concern** | `download.py` (acquire), `langid.py` (a reusable detector), `clean.py` (the 8 stages). Readable top-to-bottom, no framework. |
| **`huggingface_hub` + `pyarrow`, streaming reads** | Pull only a bounded slice of a 28 GB repo; the big toxic file is read with a capped `iter_batches` so memory stays flat. |
| **`datasketch` MinHash + MinHashLSH** | Battle-tested near-dup detection; we set `num_perm=64`, 5-word shingles, Jaccard 0.70. |
| **Script-based language ID (our own, no model)** | The 12 Indic languages live in distinct Unicode blocks, so the script is a near-perfect signal - more reliable for Indic than fastText, and dependency-free. The two ambiguous families get a light heuristic. |
| **`google/muril-base-cased` tokenizer (197K)** | Covers all 17 scheduled Indian languages incl. Urdu + Assamese - the real-world instance of Assignment-3's ~200K focused vocab, so token counts and per-language fertility are honest (overall fertility 1.54, Hindi 1.22). |
| **Regex PII + honorific name layer** | Structured identifiers are exact-match; names are honorific-anchored for precision (the widget's precision/recall tradeoff, made concrete). |

### Widget (`site/`)

| Choice | Why |
|--------|-----|
| **Static HTML + CSS + one vanilla JS file** | Same stack as assignment-1/2/3; nothing to install, loads instantly on Netlify, zero network requests. |
| **Hand-built inline SVG charts (no chart library)** | Full control of the marks; renders offline; 0 KB of dependencies. |
| **dataviz-skill validated palette** | Colorblind-checked in light + dark; CSS custom properties so the theme swaps in one place and the SVG marks follow. |
| **`localStorage` theme toggle** | Remembers light/dark without a backend. |
| **Data-driven from `stats.json`** | The page renders whatever the pipeline actually produced - change the slice, rerun, redeploy, and every figure updates. |

---

## 4. Architecture and data flow

```
assignment-4/
├── PLAN.md                 the agreed plan (written before any code)
├── FINDINGS.md             working notes: strategy count, dataset rationale, per-stage results
├── README.md               this developer guide
├── screenshots/            the Session-4 strategy captures we analysed
├── pipeline/
│   ├── download.py         stream the indic-align slice + the mmlu-indic / trivia-qa hold-out
│   ├── langid.py           script-aware Indic language detector (12 scripts + code-switch)
│   └── clean.py            the 8-stage pipeline; emits site/data/*.json + data/cleaned/corpus.jsonl
├── data/                   downloaded slice + cleaned outputs        (gitignored)
└── site/                   the deployable widget (Netlify publish root)
    ├── index.html          structure: hero, count, walkable pipeline, dataset, charts, demos, stats
    ├── styles.css          design system: palette tokens, layout, pipeline walker, charts
    ├── app.js              loads the JSON, renders charts + the walker + two live demos
    ├── netlify.toml        static-site config (publish ".", security headers)
    └── data/               stats.json, examples.json, manifest.json  (written by clean.py)
```

### How the pieces connect

```
indic-align (HF) ──download.py──► data/raw/*.parquet ──┐
mmlu-indic + trivia-qa (HF) ──────► data/holdout/*.parquet ──┤
                                                             ▼
                                                   clean.py  (8 stages)
                                                             │
                        ┌────────────────────────────────────┼───────────────────────────┐
                        ▼                                     ▼                           ▼
             site/data/stats.json                 site/data/examples.json       site/data/manifest.json
             (per-stage counts, funnel,           (real before/after            (a sample provenance
              language dist, fertility)            snippets per stage)            manifest)
                        │                                     │                           │
                        └───────────────── app.js fetch() ───┴───────────────────────────┘
                                                             ▼
                                          index.html  (charts + walker + demos)
```

1. **`clean.py`** runs the eight stages in order. Ordering is deliberate: `clean_text()` runs **before**
   the content hash, so deduplication and the manifest trust the cleaned text, not the raw markup. Each
   stage appends to an in-memory `stats` dict and captures a couple of real before/after `examples`.
2. On finish it writes `stats.json` (all per-stage numbers + a funnel + language distribution +
   per-language fertility), `examples.json`, and a sample `manifest.json`.
3. **`app.js`** fetches those three files and renders everything: the KPI tiles, the hero funnel, the
   eight strategy cards, the **interactive pipeline walker** (click a stage -> what/why/stat/tradeoff/
   mistake-it-prevents/example), the language + fertility charts, the final-stats table, and two **live
   in-browser demos** (a JS port of `clean_text()`, and a MinHash/Jaccard dedup toy).
4. **`styles.css`** defines the palette once and overrides it for dark mode twice (via
   `prefers-color-scheme` and `[data-theme="dark"]`) so the toggle wins both ways; `app.js` reads those
   variables so SVG marks match the active theme.

---

## 5. The chart + walker engine (`app.js`)

- `S()` / `txt()` / `mkSvg()` - tiny SVG element helpers; `cvar()` reads a CSS custom property so marks
  follow the theme; `hover()` attaches the shared tooltip.
- `renderFunnel` / `renderLangChart` / `renderFertChart` - one function per figure, hand-built SVG bars.
- `STAGES[]` - the eight stages' copy (what/why/tradeoff/the-mistake-it-prevents) with a `stat(s)` closure that pulls
  the live number from `stats.json`; `renderWalker` / `renderStage` build the clickable pipeline.
- `liveClean()` - a faithful JS port of the Python `clean_text()` (same noise codepoints, same ghost-tag
  regex, keeps ZWJ/ZWNJ).
- `liveDup()` - shingles two documents, computes the true Jaccard, and shows the LSH verdict at 0.70.

---

## 6. Design decisions worth noting

- **Report the pipeline as a walk, not an essay.** The tutor asked for a natural, legible flow; the
  centrepiece is a clickable 8-stage map where each stage states *what it did, why, and the tradeoff*.
- **Honesty as a feature.** Low contamination, modest name-recall, and the romanized-detection limit are
  stated plainly - the brief rewards thinking-through over a pretend-perfect run.
- **Traceability.** Every figure traces to `stats.json`, which is produced by `clean.py`; nothing is
  hand-entered into the page.
- **Continuity.** The dataset, the 12 languages, and the MuRIL fertility all continue the Assignment-3
  India-first design; NFC-not-NFKC carries over from Assignment 2.

---

## 7. Run and deploy

```bash
cd assignment-4
python3 pipeline/download.py                 # stream the slice + hold-out
python3 pipeline/clean.py                     # run the 8 stages -> site/data/*.json
cd site && python3 -m http.server 8000        # preview at http://localhost:8000
```

To deploy: point Netlify at `assignment-4/site` (it publishes `.` as-is). `site/data/*.json` is
committed so the live report renders without rerunning the pipeline.
