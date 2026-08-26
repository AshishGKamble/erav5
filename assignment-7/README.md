# Assignment 7, Kronecker embeddings: two problems, worked separately

**ERA V5, Assignment 7. Ashish Kamble.**

The assignment offers five problems and says they are separate and should not be mixed. Two were
taken, and they are kept in separate folders with separate plans, experiments, artefacts and
writeups. This page is the hub, because the submission form accepts one link.

| | the problem | the writeup |
|---|---|---|
| **Problem 5** | Reversibility. Can the same embedding give back the same Kronecker code, and if so can the output head be deleted? | **[problem-5-reversibility/README.md](problem-5-reversibility/README.md)**, [dashboard](problem-5-reversibility/site/index.html) |
| **Problem 3** | Dynamic length. The window spends 32 positions on every word, wastes space, and crops anything longer. | **[problem-3-dynamic-length/README.md](problem-3-dynamic-length/README.md)**, [dashboard](problem-3-dynamic-length/site/index.html) |

---

## The answer to each question the brief asked

**Problem 3** asked three things. All three are answered, and only the third turned out to be serious.

| what was asked | the answer | where |
|---|---|---|
| "That's a waste of space. What can we do?" | Nothing needs to change. The zeros cost **dimensions**, which genuinely cannot be reclaimed, but they cost no memory and no compute once the encoder is factored: **345x** less memory, **932x** less arithmetic. | E1, E6 |
| "How can it be dynamic?" | **It already is**, if you implement it correctly. Cost tracks the token's real length, correlation **above 0.98**: "a" costs one row lookup, a thirty byte word costs thirty. Per token dimensions are impossible, because `Linear(D, d)` needs a fixed width. | E6 |
| "...doesn't force us to crop a word" | **15 front bytes, 16 back bytes, one checksum byte of the discarded middle: 707x fewer collisions** at the same D, with no script table and no Unicode assumption. It also beats the published construction at L=64 while using half the dimensions, so the right window is still 32. | E7, E4 |

**Problem 5** promised three payoffs. Two hold, and the third gets an honest split verdict.

| what was promised | the answer | where |
|---|---|---|
| "How do I make a reverse of this?" | **It already reverses.** Exact inversion, tolerates **60x** more error than the objection implies, and survives the 8192 to 768 projection because a code is only 8-sparse, which makes this compressed sensing. | E1, E2, E3 |
| "We can get rid of the final head!" | **Yes, at zero parameters** (0 against 100.7M at the paper's dimensions). But it costs accuracy: at this scale the vocabulary head **wins** on loss, and that is reported rather than buried. Its one real defect, about 12% invalid UTF-8, is **removed entirely** by constrained decoding. | E4, E8 |
| "A vocab of 1M without any issues!" | **Split verdict.** The capability is architecturally true and needs no experiment. The competence is **not demonstrated and not testable** on this machine. | E6, E7 |

---

## The construction both problems are about

```
kappa(b) = (1/sqrt(L)) * vec( sum_{p=1..L} c_{b_p} (x) p_p )
```

`c_v` is a one hot in R^256 over byte values and `p_p` is a one hot in R^L over byte positions, so
the code is a 256 by L matrix carrying exactly one 1 per occupied column, flattened and then
z-normalised. **There are zero learned parameters in the embedding.** The only trainable part of the
input side is `Linear(D, d_model)` with no bias.

## The two findings, in one line each

**Problem 5.** Reversibility was never blocked. The codec inverts exactly, it tolerates **60 times**
more error than the objection implies, and it survives an 8192 to 768 projection because a
code is only about 8-sparse, which makes recovery a compressed sensing problem rather than an
impossible one. The output head can indeed be deleted at zero parameter cost. Whether that is a good
idea is a separate question and the measured answer at this scale is **no**, which is reported as
measured.

**Problem 3.** The window is two problems with opposite severities. The waste is huge and harmless.
The cropping is smaller and serious, and **it is not script neutral**: at L=32, English prose has
**zero** truncation collisions across 75,740 word types while nine Indic scripts have **2,062**
across 96,416. The collisions are grammatical, collapsing case-marked forms of the same word onto one
embedding, because Indic morphology is suffixal and the window reads prefixes.

## What each problem contributes beyond answering the question

- **Problem 3 found a fix nobody proposed, and it is nearly free.** Every collision is a shared
  prefix with a differing suffix, so half the window goes on the front of the word and half on the
  back, and one position holds a checksum of whatever was discarded. Together that is **707 times**
  fewer collisions at the same cost, with no new parameters, no script table and no assumptions
  about Unicode.
- **Problem 3 also measured that the high byte of an Indic codepoint carries 0.0000 bits**, which
  makes a second fix possible: send the script once per token, drop the high block, and get 32
  characters for every script at the same cost as 10.7.
- **Problem 5 turned an objection into a number.** "The model predicts 0.31 not 0.30" is a relative
  error of about 0.03 against a measured decode margin of 37.83.
- **Problem 5's one real defect turned out to be a decoding rule, not an architecture.** Constrained
  decoding removes the invalid UTF-8 entirely, 11.62% to 0.00% on the same logits, and raises
  exact match by 3.27 points, with no retraining.
- **Both problems refuted their own pre-registered predictions** and kept the refutations in the
  writeups rather than quietly editing the plans.

## The shared machinery

Both problems are built on the same small set of pieces in `common/`, and two of them are worth
naming because they are what made the measurements affordable.

- **The codec is held factored, not dense.** z-normalisation is affine, so a code is
  `(m/sqrt(L) - mu)/sigma`: a sparse support plus two closed form scalars. A 10,000 token vocabulary
  at D=8192 is 312.5 MB dense and **0.905 MB** factored, and `kappa @ W` becomes a handful of row
  lookups instead of an 8192 dimensional dot product. It reproduces the float64 codec definition to
  **1.4e-14**, so nothing is approximated. This is also the answer to Problem 3's first two
  questions.
- **Assignment 6's transformer is imported, not copied.** The model under test is one that has
  already been used and checked, and the only new parts are the two ends: a Kronecker input with no
  per token table, and three interchangeable output heads.

Each problem's README carries a table of the techniques it used and what each one is for, including
the ones that exist only because a naive version of the same measurement gave a wrong answer first.

**Artefacts record the code that produced them.** Every JSON file carries the SHA-256 of each module
in `common/`, and `python common/provenance.py` reports any artefact whose code has since moved.
This exists because of a real failure: a matmul was vectorised after two training artefacts had been
written, the new path is mathematically identical and **not bit identical**, and nothing said so.
It was caught by comparing file timestamps by hand, which is not a method. `run_demo.py` now runs
the check at the end of every run.

## The methodological result, which applies to both

Three separate experiments here were run on units that **structurally could not exhibit the effect
being measured**, and each time the null looked like a finding:

| where | the flaw | how it was caught |
|---|---|---|
| Problem 3, E5 | BPE tokens truncate 0.0024% of occurrences, so every codec carried identical information | an exposure diagnostic |
| Problem 5, E6 | every scored target was already inside the vocabulary | separate input and target streams |
| Problem 5, E7 | rare in-vocabulary words score zero too, so a zero on unknown words proves nothing | a rarity matched control band |

**Measure the exposure before believing a null.** Two of these were only found by auditing the
writeups against the literal wording of the brief, which is the practice this assignment would keep.

## Honest summary of what is weak

- Everything trained here is 2 layers at d_model=96 on CPU with 3 seeds. The **encoding** results
  (Problem 3 E1 to E4, E6, E7; Problem 5 E1 to E3) need no model and do not depend on scale. The
  **training** results do, and every effect is quoted against a measured seed noise floor.
- Problem 5's recommended byte head **loses** on loss at this scale. Its case rests on a parameter
  count at dimensions this machine cannot reach.
- The brief's "vocab of 1M without any issues" is **not demonstrated**. The capability is
  architecturally true; the competence does not follow and this scale cannot decide.
- Problem 3's downstream result is real but sits near the floor of its metric.

## Layout

```
assignment-7/
  README.md                     this hub
  SUBMISSION.md                 what was submitted and where each claim lives
  requirements.txt              numpy and tokenizers. No torch, no GPU, no network.
  common/
    codec.py                    the codec, the codepoint variant, fix D's units
    vocabulary.py               reads the assignment-2 tokenizer
    corpus.py                   reads assignment-6's frozen corpus, detects script
    kron_model.py               Kronecker input, three output heads, factored codec, Adam, gradient check
    data.py                     packing, with assignment-6's segment rules kept
  problem-5-reversibility/      PLAN.md, README.md, run_demo.py, src/, artifacts/, site/
  problem-3-dynamic-length/     PLAN.md, README.md, run_demo.py, src/, artifacts/, site/
```

## Reproducing everything

```bash
cd assignment-7
python -m venv .venv && .venv/bin/pip install -r requirements.txt
cd problem-3-dynamic-length && ../.venv/bin/python run_demo.py          # about a minute
cd ../problem-5-reversibility && ../.venv/bin/python run_demo.py        # about half a minute
```

Add `--full` to either to include the training experiments, which take roughly an hour each on 16
CPU cores. No step needs a network connection or a GPU.

## What this builds on

The measurements are cheap here and would be a data collection project anywhere else, because
earlier assignments already committed the inputs:

- **Assignment 2** built the 10,000 token BPE tokenizer these experiments encode.
- **Assignment 6** froze a licence checked corpus of 17.7 million characters across seven lanes,
  including 8.1 MB of Indic text in ten scripts, and a NumPy transformer with hand written backward
  and Adam. Its transformer body is **imported** here rather than copied, so the model under test is
  one that has already been used and checked.

Licensing follows the repository: code MIT, prose CC BY 4.0, corpora referenced and never
redistributed.
