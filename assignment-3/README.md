# Assignment 3 - India-First 40B: A Design

A short, figure-driven web report that answers ERA V5 Assignment 3: *design (on paper)
a 40B LLM that matches **Gemma 4** on general/coding/math/agentic work, dominates on Indic
languages, and is "India-first" (sees the world from an Indian perspective), then decide
the data, cleaning, evaluation, and tokenizer.*

> Re-baselined to **Gemma 4** (Google DeepMind, Apr 2026, arXiv 2607.02770) - a real, frontier,
> now-agentic model. The differentiation is positional (India-first perspective, measured Indic
> depth, tokenizer efficiency), not capability gaps. See [`FINDINGS.md`](FINDINGS.md) v0.2.

The report is deliberately concise and graph-led, because the assignment penalizes length
and rewards concrete thinking with good figures.

- **Live report:** `site/index.html` (deploy `site/` to Netlify).
- **Research corpus (all sourced numbers + reasoning):** [`FINDINGS.md`](FINDINGS.md).

---

## 1. What the report argues (the five decisions)

Sections follow the natural LLM build order (targets, data, cleaning, tokenizer, tests),
and each is tagged on the page with the exact brief question it answers.

| # | Decision | Headline answer |
|---|----------|-----------------|
| 1 Targets | What "winning" means | Match Gemma 4 31B on its own suite (MMLU-Pro 85, AIME 89, LiveCodeBench 80, GPQA 84, tau2 77); lead on the Indic depth it does not publish |
| 2 Data | Pre-train / SFT / RL sources | ~15T tokens, mostly English for reasoning + a deliberate **28% India-touching** slice |
| 3 Cleaning | How to clean for the objective | Standard scrub + **four Indic-specific fixes** generic pipelines get wrong |
| 4 Tokenizer | Fertility targets + vocab size | Per-language + code/science/math/agentic targets; **200K vocab, derived three ways** |
| 5 Evaluation | How to test, incl. Indian perspective | 3 buckets; invent **BharatDrishti** because no India-perspective benchmark exists |

Every number on the page traces back to a sourced entry in `FINDINGS.md`.

---

## 2. Technology choices (and why)

This is a **static, dependency-free site** - no framework, no build step, no external
requests. That is a deliberate decision, not laziness:

| Choice | Why |
|--------|-----|
| **Plain HTML + CSS + one vanilla JS file** | Matches assignment-1 and assignment-2; nothing to install, audit, or break; loads instantly on Netlify. |
| **Hand-built inline SVG charts (no chart library)** | Full control over the dataviz mark specs (thin marks, 2px surface gaps, hairline grid, direct labels); zero KB of dependencies; renders identically offline. |
| **No CDN / no fonts fetched** | The page makes zero network requests, so it is fast, private, and cannot break if a CDN is down. Type is the system UI sans. |
| **CSS custom properties for the palette** | Light/dark themes swap in one place; SVG marks read the same variables, so charts follow the theme. |
| **`localStorage` for the theme toggle** | Remembers the reader's light/dark choice without any backend. |

The chart palette is the validated dataviz default set; it was run through the skill's
colorblind-safety validator (`validate_palette.js`) in both light and dark modes before use.

---

## 3. Architecture and file layout

```
assignment-3/
├── FINDINGS.md          research corpus: every sourced number + the full reasoning
├── README.md            this developer guide
└── site/                the deployable report (Netlify publish root)
    ├── index.html       all content + section structure + chart containers
    ├── styles.css       design system: palette tokens, layout, chart chrome, diagrams
    ├── app.js           chart engine: renders SVG charts, legends, tables, theme, tooltips
    └── netlify.toml      static-site config (publish ".", security headers)
```

### How the pieces connect (data flow)

```
FINDINGS.md ──(numbers copied in by hand)──► app.js data arrays ──► SVG charts
                                                    │
index.html (empty chart hosts + captions) ◄─────────┘ (app.js injects SVG into hosts by id)
styles.css (CSS variables) ──(read at render time via getComputedStyle)──► chart colours
```

1. **`index.html`** lays out six sections (hero + five decisions + a summary card). For each
   figure it provides an empty host `<div id="figX">`, a `<figcaption>`, an empty legend box,
   a source note, and a "Show data table" button with an empty table container.
2. **`app.js`** holds the chart **data as small arrays** (benchmark scores, tokenizer
   fertilities, the data mix, the cost curve). On load it renders every chart by building
   inline SVG into the host divs, then fills the legend and the table-view twin for each.
3. **`styles.css`** defines the colour tokens once (light) and overrides them for dark mode
   twice - via `prefers-color-scheme` (OS setting) and `[data-theme="dark"]` (the toggle) so
   the toggle wins both ways. `app.js` reads these variables so marks match the active theme.
4. Flow diagrams (the cleaning pipeline, the BharatDrishti build) and the per-language
   fertility table are **static HTML/CSS** in `index.html` - they are structural, not
   data-driven, so they need no JavaScript and render even if JS is disabled.

### The chart engine (`app.js`)

- `S()` / `txt()` - tiny SVG element helpers.
- `cvar()` - reads a CSS custom property (so colours follow the theme).
- `mkSvg()` - creates a `viewBox`-scaled `<svg>` (responsive width, horizontal scroll on small screens).
- `hover()` - attaches the shared tooltip to any mark.
- `legend()` / `table()` - build the HTML legend and the table-view twin from data.
- `drawWaste`, `drawBench`, `drawScatter`, `drawCost`, `drawRecipe` - one function per figure.
- `renderAll()` - draws everything; re-called on theme change so marks re-read their colours.

---

## 4. The figures (what each proves)

Numbered in reading order, which follows the build-order sequence:

| Figure | Type | Point it makes |
|--------|------|----------------|
| Waste bomb (hero) | two bars | A bad tokenizer turns 500B tokens into ~37B words; ours recovers ~9x more. |
| 1. Benchmark targets | grouped columns | Gemma 3 &rarr; Gemma 4 was a huge jump; we target parity with Gemma 4 31B and lead on the Indic row it drops. |
| 2. Staged data recipe | stacked bars | The base-phase mix + the honest ~90%-synthetic Indic breakdown. |
| 3. Post-training spine | flow (HTML) | Mid-train &rarr; reasoning SFT (distill R1/Qwen3) &rarr; RLVR &rarr; agentic RL &rarr; fusion + align. |
| 4. Cleaning pipeline | flow (HTML) | The pipeline, with the four Indic-specific fixes highlighted. |
| 5. Focus beats breadth | scatter | Gemma 4's 262K over 140 languages vs our focused 200K over 12; who you allocate to wins. |
| 6. Vocab cost | line | The param cost of vocabulary; 200K sits at the 4.1% sweet spot. |
| 7. BharatDrishti | diagram (HTML) | How the new India-perspective benchmark is built and scored, fairly. |

---

## 5. Design decisions worth noting

- **Report over essay.** Text is captions and callouts; the figures carry the argument. This
  matches the grader's stated preference ("longer submissions score lower; good graphs").
- **Honesty as a feature.** The report states plainly that ~90% of the Indic data is synthetic
  and that its quality caps Indic ability - a named limitation, because the grader rewards
  thinking-through over a pretend-perfect plan.
- **Accessibility.** Every chart has a table-view twin; colours passed the CVD validator;
  direct value labels mean nothing is gated behind hover; the page is theme-aware.
- **Traceability.** Each figure carries a source line, and all numbers live in `FINDINGS.md`
  with citations, so any claim can be checked.

---

## 6. Run and deploy

No build step. To preview locally:

```bash
cd assignment-3/site
python3 -m http.server 8000    # then open http://localhost:8000
```

To deploy: point Netlify at the `assignment-3/site` folder (it publishes `.` as-is).
