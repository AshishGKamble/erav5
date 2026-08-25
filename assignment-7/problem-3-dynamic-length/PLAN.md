# Problem 3 - Dynamic length: the plan agreed before coding

**ERA V5 - Assignment 7 - Problem 3 - Ashish Kamble**

## The problem, as the assignment states it

> "Today Kronecker is limiting to presenting 32 position for every work (even "apple" or "a" as
> well). That's a waste of space. What can we do? How can it be dynamic and doesn't force us to crop
> a word (currently we cannot have a word of len more than 32)."

Two distinct complaints are packed into that paragraph: **waste**, where short tokens leave most of
the window empty, and **cropping**, where long tokens do not fit at all. The assignment treats them
as one problem. This work argues they are not, that they have opposite severities, and that the
second one has a consequence the brief does not mention.

## The claim this work will defend

**The fixed window is not one problem, it is two, and only one of them is about waste.**

1. **Waste** is what the assignment text describes. "a" occupies 32 position slots to carry 1 byte.
   Real but benign: the unused columns are zeros, nothing is lost, the cost is dimensionality.

2. **Truncation is the serious one, and it is not script neutral.** The window is counted in
   **bytes**, but languages are written in **characters**. In UTF-8, Latin is 1 byte per character
   while Devanagari, Telugu, Bengali, Tamil, Kannada, Malayalam, Gujarati, Oriya and Gurmukhi are
   **3 bytes per character**. A 32-byte window therefore holds 32 Latin characters but only about 10
   Indic ones. Past that, bytes are dropped, and dropped bytes are information the model never sees.

Waste costs dimensions. Truncation costs meaning, and it charges that cost almost entirely to the
languages an India-first model exists to serve. That is the finding.

The sharpest form of the harm is a **truncation collision**: two different words that are identical
in their first L bytes receive the *identical* embedding, so the model cannot distinguish them at
any depth. This should be near zero for English and non-zero for Indic scripts at the same window
size, and it is directly measurable.

## Why this repository can measure it and others cannot

Assignment 6 committed a licence-checked frozen corpus containing 8.1 MB of Indic text across ten
scripts (Devanagari, Telugu, Bengali, Malayalam, Tamil, Kannada, Gujarati, Oriya, Gurmukhi, and
Arabic-script Urdu), alongside web, code and math lanes for contrast. Assignment 2 built the
tokenizer. Assignment 6 already measured Indic fertility at 1.829 tokens per word against web's
0.374. The measurement below is a few hours of work here and a data-collection project anywhere else.

## The five experiments

Every number is produced by a script and written to `artifacts/`, never typed into prose.

### E1 - Occupancy: how much of the window is actually used

Per lane and per script, tokenize the frozen corpus and record bytes-per-token as a distribution.
Report mean occupancy at L = 16, 32 and 64, and the fraction of position columns that are zero.

**Expected:** heavy waste everywhere, worst for English. This confirms the assignment text and is
the least interesting result here, which is itself worth stating plainly.

### E2 - Characters per window by script (the reframing)

Same corpus, but measure **characters** rather than bytes: how many characters of each script fit in
L bytes. Report a per-script table of bytes per character and effective character capacity.

**Expected:** Latin about 1.0 bytes/char, the nine Indic scripts about 3.0, Urdu about 2.0. So the
same architectural constant delivers roughly a third of the context to Indic scripts.

### E3 - Truncation rate and truncation collisions (the headline)

For L in {16, 32, 64}, per script:

- fraction of tokens whose UTF-8 encoding exceeds L bytes and is therefore cropped
- number of **distinct** tokens that collapse onto an identical embedding after cropping, and the
  colliding groups themselves, listed in the artefacts so the claim can be checked by eye
- characters lost per cropped token

**Expected:** collision count near zero for English and materially non-zero for Indic scripts at
L = 16, shrinking but not vanishing at 32. Real colliding word pairs, printed, are the most
persuasive single artefact this problem can produce.

### E4 - Three candidate fixes, measured rather than advocated

| Fix | Mechanism | What it costs |
|---|---|---|
| **A. Length-aware normalisation** | normalise by √(actual length) instead of √L, and append an explicit length channel | almost nothing; does not address truncation |
| **B. Codepoint-position factorization** | factor on Unicode **codepoint** rather than byte, so L positions mean L characters for every script | char_dim must cover the codepoint space; solved with a two-level Kronecker (high byte ⊗ low byte) or hashed buckets, which is where the honest cost sits |
| **C. Dynamic allocation** | short tokens use fewer position slots, with the freed columns carrying a length tag | keeps the byte unit, so Indic still pays 3x |

Fix B is the one that actually targets the finding, and its cost has to be stated rather than
glossed: moving from 256 byte values to the Unicode codepoint space is not free, and a hashed
char_dim reintroduces collisions of a different kind, which E5 measures.

### E5 - Downstream proof

Train the assignment-6 NumPy transformer on the Indic lane twice with identical seeds, once with the
byte-position codec and once with the codepoint-position codec of fix B, and compare loss curves and
per-script token accuracy. Then repeat on the web lane to confirm the change does not harm Latin
text, because a fix that helps Indic by hurting English is not a fix.

**Expected:** measurable gain on the Indic lane, neutral on web. If the gain does not appear at this
scale, that is reported as the result, and the E3 collision counts stand on their own regardless,
since they are properties of the encoding rather than of any trained model.

## What is deliberately not claimed

- Small scale, CPU only, no GPU. Loss differences are quoted against a measured noise floor from
  repeated seeds, the same discipline assignment 5 used, and any effect inside that floor is
  reported as not established.
- Fix B is evaluated as an encoding change in isolation. Whether it interacts well with the
  tokenizer's own merges at 124M scale is not something this machine can answer.
- Grapheme clusters are not the same as codepoints. Devanagari conjuncts and combining marks mean
  a codepoint window is still not a *grapheme* window, and that residual is measured in E2 rather
  than being quietly ignored.

## Deliverables

- `run_demo.py` - one command, no network, regenerates everything in `artifacts/`
- `artifacts/evidence.json` and `evidence.md` - every number, recomputed from artefacts
- `site/` - static dashboard, no framework, no build step, hand-built inline SVG, light and dark
- `README.md` - the writeup, with the five experiments in order
- `tests/` - invariants: codec determinism, collision detection correctness, per-script tallies
