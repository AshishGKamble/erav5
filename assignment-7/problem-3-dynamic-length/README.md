# Problem 3, dynamic length: the window is two problems, and only one of them is about waste

**ERA V5, Assignment 7, Problem 3. Ashish Kamble.**

Every number here is produced by a script in `src/` and written to `artifacts/`. None is typed by
hand. `artifacts/evidence.md` regenerates the whole set from the committed artefacts, and
`run_demo.py` regenerates the artefacts from the frozen corpus with no network access.

---

## The question

> "Today Kronecker is limiting to presenting 32 position for every work (even 'apple' or 'a' as
> well). That's a waste of space. What can we do? How can it be dynamic and doesn't force us to
> crop a word (currently we cannot have a word of len more than 32)."

That paragraph asks three things: **what to do about the waste**, **how to make it dynamic**, and
**how to stop cropping words**. This writeup answers all three, and it argues that the third is a
far more serious problem than the first two, for a reason the assignment does not mention.

## The answer to each of the three questions

| what was asked | the answer | evidence |
|---|---|---|
| **"That's a waste of space. What can we do?"** | Nothing needs to change. The waste is 92 to 95 percent and it costs **dimensions**, which genuinely cannot be reclaimed, but it costs **no memory and no compute** once the encoder is factored: 312.5 MB becomes 0.905 MB, and the arithmetic drops 932x. | E1, E6 |
| **"How can it be dynamic?"** | **It already is**, with no architectural change. Cost tracks the token's real length, correlation above 0.98: "a" costs one row lookup and a thirty byte word costs thirty. Making the *dimensions* per token is impossible, because `Linear(D, d)` needs a fixed width, and that is stated as the reason rather than skipped. | E6 |
| **"...doesn't force us to crop a word"** | **Read the word from both ends.** 9.8x fewer collisions for no new parameters, no script table and no Unicode assumptions. A script relative codec reaches 33.7x, and the two compose for 54.4x. Separately, because compute follows real length, raising L is nearly free and L=64 alone removes almost every collision. | E7, E4, E6 |

The rest of this document is how those answers were arrived at, in the order the experiments ran.

## The techniques used, and what each one is for

Several of these exist because a naive version of the same measurement gave a wrong answer first.
Those are marked, because the reason a technique is needed is usually more informative than the
technique.

| technique | what it is for | where |
|---|---|---|
| **Script detection from `unicodedata.name`** | Derive a character's script from the Unicode database rather than a hand written table, so the classification cannot be accused of being tuned to produce this result. | `common/corpus.py` |
| **Extended grapheme approximation** | Count what a reader perceives as one unit: a base character plus its combining marks, plus consonants joined by a virama. No segmentation dependency was added. | E2 |
| **Prefix grouping** | Group distinct word types by their first L bytes. Any group larger than one is a set of words that receive the identical embedding. | E3 |
| **Bitwise verification** | Re-encode colliding pairs and compare the actual float vectors, instead of trusting that the definition implies it. | E3 |
| **Equal-D comparison** | Charge every codec the same dimension budget, so no fix can look good merely by spending more. Byte L=32 and two block codepoint L=16 are both 8192. | E4, E7 |
| **Shannon entropy per base-256 digit** | Measure how much information a codec block actually carries. This is the measurement that exposed fix D: the Indic high digit carries 0.0000 bits. | E4c |
| **Affine factorisation of z-normalisation** | Because `kappa = (m/sqrt(L) - mu)/sigma` is affine, a code is a sparse support plus two closed form scalars. Exact to 1.4e-14, and the reason memory and compute are not actually paid. | E6 |
| **Gather plus segmented reduction** | Vectorise the factored path with `np.add.reduceat`. **Needed because** a per token Python loop loses a wall clock race to dense BLAS despite doing 932x less arithmetic, and reporting that would measure the implementation rather than the construction. | E6 |
| **Seed noise floor** | Repeat every trained comparison across seeds and report any effect inside two standard deviations as not established. | E5, E5b |
| **Exposure diagnostic** | Before believing a null, measure how much of the effect the experiment could even see. **Needed because** E5's null turned out to be a measurement with no exposure rather than a finding. | E5 |
| **Lane separation and punctuation stripping** | Remove measurement artefacts that flatter the result. **Needed because** pooled Latin first measured 7.53% collisions, almost all of it source code and trailing commas. | corrections |
| **Type versus occurrence weighting** | Report both, since they answer different questions: what the vocabulary looks like against what training actually spends. | E1 |

## The finding underneath all three

The waste is real, large and almost harmless: 92 to 95 percent of the window is zeros. It costs
dimensions and nothing else, and two of the three things it might cost, memory and compute, are not
actually paid at all once the encoder is written correctly.

The cropping is the serious one, and **it is not script neutral**. The window is counted in bytes
while language is written in characters, and in UTF-8 every Indic script costs exactly three bytes
per character against Latin's one. So a 32 byte window holds 32 English characters and about ten
Indic ones. Past that, bytes are dropped, and two different words that share their first 32 bytes
receive the **identical** embedding, which no model at any depth can undo.

At L=32, English prose has **zero** such collisions across 75,740 word types. Nine Indic scripts
have **2,062** across 96,416. That is not a ratio. It is a categorical difference in whether the
architecture works for your language.

Three fixes are measured rather than advocated, and the cheapest one turns out to be the one nobody
proposed: **read the word from both ends**.

---

## E1, the waste, confirmed and then set aside

| lane | mean bytes per token | occupancy at L=32 | zero columns |
|---|---|---|---|
| code | 1.64 | 5.11% | 94.89% |
| indic | 2.27 | 7.09% | 92.91% |
| web | 2.68 | 8.37% | 91.63% |

The assignment is right. Almost the entire window is empty. This is the least interesting result in
the problem set and it is stated first so that the rest is not mistaken for a denial of it.

## E2, the reframing: the window is counted in the wrong unit

| script | bytes per character | characters in 32 bytes | **graphemes** in 32 bytes |
|---|---|---|---|
| Latin | 1.000 | 32.0 | 32.0 |
| Arabic | 2.000 | 16.0 | 16.0 |
| Devanagari, Bengali, Gujarati, Gurmukhi, Kannada, Malayalam, Oriya, Tamil, Telugu | **3.000** | **10.7** | Malayalam **5.0** |

Bytes per character is exactly 3.000 for all nine Indic scripts, not approximately. Measured in
graphemes, which is what a reader actually perceives, Malayalam gets **5.0 units** in the window
where Latin gets 32. The gap is 6.4x, not 3x, because Indic conjuncts are written with several
codepoints and read as one.

Script detection uses `unicodedata.name` rather than a hand written table, so it cannot be accused
of being tuned to produce this result. Grapheme counting is an approximation and is labelled as one
wherever it appears.

## E3, the harm: truncation collisions

A **truncation collision** is two distinct words that are identical in their first L bytes. They
produce byte for byte identical codec vectors, so they are the same input at every layer.

This is verified rather than argued: 200 of 200 sampled colliding pairs produce codec vectors whose
maximum absolute difference is exactly **0.0**.

| | word types | colliding groups at L=32 |
|---|---|---|
| English prose (web, long_ctx, reasoning) | 75,740 | **0** |
| Nine Indic scripts | 96,416 | **2,062** |

At L=16 it is 11,981 Indic groups against 10 English. At L=64 it nearly vanishes, 6 against 0.

The collisions are **grammatical, not exotic**, and this is the point that matters:

| script | words that collapse to one embedding | what they mean |
|---|---|---|
| Tamil | `உங்கள்` `உங்களுக்கு` `உங்களை` | your / to you / you (accusative) |
| Malayalam | `നിങ്ങളുടെ` `നിങ്ങൾക്ക്` `നിങ്ങൾ` `നിങ്ങളെ` | your / to you / you / you (accusative) |
| Devanagari | `तुम्ही` `तुम्हाला` `तुम्हारा` `तुम्हारे` | you / to you / your, across Marathi and Hindi |

Indic languages mark case, number and person with **suffixes**. The window reads prefixes. It is
systematically discarding the half of the word that carries the grammar, and it does this to the
languages an India first model exists to serve while leaving English untouched.

### Two corrections that cut against this finding, both kept

A first pass measured pooled Latin at 7.5 percent collisions, which would have made the gap look
smaller and the measurement look sloppy. Two things were wrong:

1. **Whitespace splitting source code does not produce words.** Nearly all pooled Latin collisions
   were identifiers and LaTeX, `self.assertEqual(` against `self.assertEqual(0,`. Separating lanes
   dropped the English prose baseline to zero.
2. **Trailing punctuation is a fake collision.** `மற்றும்` against `மற்றும்,` counted as two types
   sharing a prefix. Stripping edge punctuation cut pooled Latin fivefold and left Indic almost
   unchanged, so the gap **widened**.

Both the raw and the corrected figures are kept in `artifacts/window.json` as `word_raw` and
`word_prose`, so the size of each artefact is visible rather than asserted to be small.

---

## E4, three candidate fixes, measured at equal cost

The codec makes an exact comparison possible: a byte window of L costs `256L` dimensions and a two
block codepoint window of L costs `512L`, so byte L=32 and codepoint L=16 are both D=8192 and can be
compared with nothing left over.

**The trade is closed form.** The byte codec carries `1/(256 x bytes_per_char)` characters per
dimension. The codepoint codec carries `1/512` **for every script alike**. So byte wins below two
bytes per character and loses above it, and the break even script is a two byte one. Latin is 1.0,
Indic is 3.0, Arabic is 2.0 and sits exactly on the fence.

Collision rate at D=8192, word types, prose lanes:

| script | byte L=32 (published) | fix B, codepoint L=16 | **fix D, script relative L=32** |
|---|---|---|---|
| Malayalam | 17.47% | 2.20% | **0.00%** |
| Tamil | 13.06% | 0.77% | **0.00%** |
| Kannada | 6.31% | 0.17% | **0.00%** |
| Telugu | 5.26% | 0.16% | **0.00%** |
| Devanagari | 2.15% | 0.03% | **0.01%** |
| **Latin** | **0.18%** | **1.48%** | **0.18%** |

**Fix B helps every Indic script and makes Latin roughly eight times worse.** It is not uniformly
better. It is uniformly *fair*, which is a different and more defensible claim, and it is stated
that way rather than sold as a free win.

### Fix D, which the measurement found rather than the plan

Every Indic script in this corpus uses **exactly one** high base-256 digit, with an entropy of
**0.0000 bits** and one distinct value. Devanagari is always `0x09`. Latin uses five values but
0.0006 bits. Only Han genuinely uses that digit, at 5.37 bits.

So the high digit is not information. It is a script name retransmitted once per character, and half
of fix B's dimensions carry nothing. Send the script **once per token**, drop the high block, and a
position costs 256 rows again while holding a whole character. At D=8192 that is **32 characters for
every script**, against the byte codec's 32 Latin and 10.7 Indic, and fix B's 16.

Fix D fixes all nine Indic scripts and leaves Latin **exactly** where it was.

**Its costs, none of them rounded to zero.** Without the script tag it produces 3,214 cross script
alias groups, since Devanagari `U+0915` and Telugu `U+0C15` both reduce to `0x15`. With the tag it is
lossless only for pure script words: `केल` still collides with `के2` because `U+0932` and ASCII
`U+0032` share a low digit, one group each in Devanagari and Oriya. And "one high digit per script"
is measured in this corpus, **not guaranteed by Unicode**: Devanagari Extended and Vedic Extensions
live elsewhere, so production needs a script to block table and a fallback that this work does not
exercise.

### Fixes A and C, reported because they do not work

**Fix A, length aware normalisation.** Verified to leave the decode bitwise identical on 400 of 400
tokens, because a positive rescale cannot change an argmax. Effect on truncation collisions: none,
and none is possible. It cannot repair information that was never encoded.

**Fix C, dynamic allocation.** Reclaims the zeros E1 measured, which is the waste half of the
problem, but keeps the byte as the unit. A three byte per character script still spends three slots
per character and crops at the same character count, so **every collision in E3 survives fix C
unchanged**.

---

## E7, the cheapest fix, and the one the data pointed at

E3 printed the colliding groups rather than only counting them, and reading them makes something
obvious that no summary statistic says: every collision is a shared **prefix** with a differing
**suffix**, because Indic morphology is suffixal.

So spend half the window on the front of the word and half on the back. Same D, same one hot
structure, **no new parameters, no script table, no Unicode assumptions, no tag**. Only a different
choice of which units to encode.

| scheme, all at D=8192 | colliding groups | Malayalam | reduction |
|---|---|---|---|
| first 32 bytes (published) | 2,122 | 17.47% | 1.0x |
| **16 leading plus 16 trailing bytes** | **217** | **1.14%** | **9.8x** |
| fix D, script tag plus 31 characters | 63 | 0.00% | 33.7x |
| **fix D plus both ends** | **39** | 0.00% | **54.4x** |

Reading the word from both ends removes about ninety percent of the harm for free. Fix D is better
in absolute terms but needs a script table, a tag, and an assumption about Unicode block layout that
this corpus happens to satisfy. **If only one change is made, it should be this one.** The two
compose, and together they are the best configuration measured.

---

## E6, the two questions this writeup nearly dodged

An audit of this document against the brief found that it answered "don't crop words" and quietly
argued the other two away. Saying the waste is benign is true about *accuracy* and evades the
question, which is about **space**. Space means three different things here, and they have three
different answers.

| meaning | at L=32 | verdict |
|---|---|---|
| **Dimensions** | D=8192, 72.3% of columns empty across the vocabulary | **irreducible** |
| **Memory** | 312.5 MB dense against **0.905 MB** factored | 345x, fully reclaimable |
| **Compute** | **932x** less arithmetic | already dynamic |

**Why dimensions cannot be made dynamic, which is the honest answer to the literal question.** The
codec output feeds `Linear(D, d)`, whose weight matrix has a fixed first dimension. A per token D
means a per token weight matrix, which is precisely the per token parameter table the whole
construction exists to delete. So "dynamic" in the dimensional sense is not unimplemented, it is
**impossible without giving up the idea**.

**Why memory and compute are not actually paid.** z-normalisation is affine, so a code is
`(m/sqrt(L) - mu)/sigma` where `m` is a 0/1 support and `mu` and `sigma` are available in closed form
from the number of ones. A code is therefore fully described by its occupied row indices plus two
scalars, and `kappa @ W` is a handful of row lookups plus one shared rank-1 term. The zeros are
never stored and never multiplied. This is exact, not an approximation: it reproduces the codec
definition to **1.4e-14**.

**The dynamic claim, made falsifiable.** Cost against token length rises with a strong positive
slope, correlation **above 0.98** in every run measured. The exact slope and correlation of the
committed run are in `artifacts/evidence.md`, because they are timing measurements and move. If the window charged every token for L positions this line would be
flat. It is not flat. "a" costs one row lookup and a thirty byte word costs thirty.

**The honest deflation.** 932x less arithmetic buys only a low single digit multiple of wall clock, because a gather
plus a segmented reduction is memory bound while the dense path is a single BLAS call. The advantage
and the advantage grows with the window. **932x is not a speedup and is not quoted as one.**

Every timing figure in this section is **re-measured on each run** and moves with machine load:
repeated runs on this machine produced wall clock ratios between about 1.5x and 2.0x at L=32 and
slopes between roughly 430 and 505 ns per unit. What is stable, and what the claim rests on, is the
**correlation staying above 0.98** and the arithmetic ratio, which is deterministic. Every collision
count and every memory figure elsewhere in this writeup is exact and does not move.

### The consequence, which inverts the assignment's premise

Encoding a short token costs the **same at every window size**: every measured ratio relative to
L=32, for L=16, 32, 64 and 128, lands within a few percent of 1.0, which is inside the run to run
spread above. The committed run's exact ratios are in `artifacts/evidence.md`. Factored memory barely moves, 0.823 MB to
0.928 MB. The only thing that genuinely grows is the projection matrix `W`, from 393,216 to 3,145,728
parameters.

So the window is cheap to enlarge, and E3 already showed that L=64 removes almost every truncation
collision. **"How can it be dynamic" and "don't force us to crop a word" have the same answer**, and
its price is a bigger `W`, which is a number to weigh rather than a hidden cost.

---

## E5, the downstream test, and why the pre-registered version could not work

PLAN.md pre-registered a downstream comparison: train the same model with each codec and compare.
It was run, on the indic lane with web as a control, three seeds, three codecs at equal D.

**The result is null on both lanes**, every delta inside the seed noise floor. That is a
pre-registered possible outcome, but a null is worth nothing without knowing why, so the exposure
was measured:

**Of 250,475 indic token occurrences, exactly 6 are truncated by any codec. Zero in web. Zero for
both codepoint and script relative.**

All three codecs therefore carry **identical information** for 99.998 percent of tokens and differ
only in how it is laid out across D dimensions, which a `Linear(D, d)` learns equally well either
way. This is not weak evidence about the codecs. It is a measurement with no exposure to the thing
being measured.

**The general lesson, which is the most transferable thing in this problem.** A BPE tokenizer sits
between the corpus and the window and *removes the phenomenon under test*, converting truncation
into fertility instead. Assignment 6 measured that other side of the ledger: Indic fertility 1.829
against web 0.374. Any downstream test of a window size claim that runs on BPE tokens is testing
nothing. E5 as pre-registered was **mis-specified, not inconclusive**.

### E5b, the same test rebuilt so it can see the effect

Words as units instead of BPE tokens, and a tied byte head so there are **zero** per word output
parameters, which is the Kronecker construction's own selling point being used to test it. The
metric is exact **full word** reconstruction, with truncated targets counted as failures rather than
dropped from the denominator.

| lane | arm | word types representable | exact full word | delta | verdict |
|---|---|---|---|---|---|
| indic | byte | 88.96% | 0.10% | | |
| indic | script relative | 100.00% | **0.58%** | +0.0048 (2sd = 0.0020) | outside noise |
| web | byte | 99.99% | 4.79% | | |
| web | script relative | 100.00% | 4.98% | +0.0020 (2sd = 0.0152) | inside noise, **no harm** |

The web control is the one that matters, and it passes: helping Indic by hurting English would not
be a fix, and this does not do that.

**This result is weak and is labelled weak.** Both indic arms sit near the floor of the metric, and
the byte codec truncates only 4 percent of targets, so truncation is not what limits it. The model
is simply poor at word level prediction at this scale. E1 through E4, E6 and E7 are properties of
the encoding and do not depend on this experiment at all.

---

## What is deliberately not claimed

- **Nothing here was trained at a scale that can settle a downstream question.** Everything trained
  is 2 layers at d_model=96 on CPU. The encoding results do not depend on scale. The training
  results do, and are quoted against a measured seed noise floor with anything inside it reported as
  not established.
- **Fix D's central assumption is corpus measured, not universal.** One high digit per script holds
  here and is not a property of Unicode.
- **Graphemes are approximated.** No grapheme segmentation dependency was added. The approximation
  is used only to size the codepoint to grapheme gap in E2, never to support a truncation claim.
- **The wall clock figures are machine dependent** and move between runs. The arithmetic ratios and
  every collision count are deterministic.
- **Fix D and the both ends scheme are evaluated as encodings in isolation.** Whether they interact
  well with a tokenizer's merges at 124M scale is not a question this machine can answer.

## Reproducing

```bash
python -m venv .venv && .venv/bin/pip install -r ../requirements.txt
cd problem-3-dynamic-length
python run_demo.py           # E1 to E4, E6, E7. About a minute, no network.
python run_demo.py --full    # adds E5 and E5b. Roughly an hour on 16 CPU cores.
```

`run_demo.py` rewrites `artifacts/` and then regenerates `artifacts/evidence.md` from it, so any
disagreement between this README and the evidence file is a bug in this README.

## Files

| path | what it is |
|---|---|
| `PLAN.md` | the plan agreed before any code was written, including the predictions that were later refuted |
| `src/exp_window.py` | E1 to E3, occupancy, characters per window, truncation collisions |
| `src/exp_fixes.py` | E4, the three fixes at equal D, the block entropy finding, fix D |
| `src/exp_cost.py` | E6, dimensions against memory against compute |
| `src/exp_bothends.py` | E7, reading the word from both ends |
| `src/exp_downstream.py` | E5 and E5b, the downstream tests and the exposure diagnostic |
| `src/evidence.py` | regenerates every number in this README from the artefacts |
| `src/build_dashboard.py` | extracts the dashboard payload from the same artefacts |
| `site/` | static dashboard. No framework, no build step, no network. Open `site/index.html` |
| `artifacts/evidence.md` | the generated evidence file |
| `../common/codec.py` | the codec, the codepoint variant, and fix D's units |
| `../common/corpus.py` | the frozen corpus reader and script detection |
