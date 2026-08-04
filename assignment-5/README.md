# V5 Mixture & Curriculum Plan - an India-first ~40B model

**ERA V5 · Assignment 5 · Ashish Kamble**

A fixed budget of 4 trillion tokens. This is how we spend it, in what order, and what we hold back
for the finish. Every share is defended against a benchmark it must win and against the tokens that
actually exist to fill it. Then we tried to break it: **21 proxy training runs across 9 rounds,
which killed two of our own proposed revisions.**

*This file is the whole submission. Everything needed to judge it is here; links go to computed
evidence for anyone who wants to check the arithmetic.*

---

## 1. The plan, in one page

| Lane | Share | Must win | Can we buy the tokens? |
|---|:---:|---|---|
| **General web** | **30%** | MMLU-Pro, common sense | Yes, 0.06 epochs of 20.3T |
| **Code** | **22%** | LiveCodeBench, SWE-bench | Yes, 0.49 epochs of 1.8T |
| **Math + STEM** | **10%** | AIME, GPQA | Repeat 2.8x over 143B |
| **Reasoning traces** | **6%** | AIME, GPQA, BBEH | 4 epochs **+ 88B generated** |
| **Agentic / tool-use** | **8%** | tau2-bench, BFCL | 4 epochs **+ 150B generated** |
| **Long-context** | **6%** | RULER, MRCR-v2 | 1.0 epoch, 60% already synthetic |
| **Indic** | **18%** | MILU, IndicGenBench | 0.93 epochs, but the tiers matter (§3.3) |

**Three admissions we would rather make ourselves than have a reviewer extract:**

1. **238B tokens (5.9% of budget) do not exist and must be manufactured**, almost all of it agentic
   and reasoning. §3.2 names every source and marks the imaginary ones.
2. **Native Indic is capped at 11.0% of budget no matter what share we pick.** Only 110B organic
   Indic tokens exist; at the 4-epoch reuse ceiling that is 440B, full stop. A bigger Indic lane
   buys machine-translated text, not Indic capability. This is arithmetic (§3.4).
3. **Our boldest promise rests on the thinnest data.** tau2-bench (≥77) is fed by a lane that is
   4.7% organic, contributes 2.8% of actual gradient, and our proxy cannot test it at all (§6).

**What all the testing changed: nothing. That is the finding.** We twice proposed revising this
mixture, ran the experiments, and both revisions died - one because it measured the wrong
distribution, one because its validation set was contaminated. The shares above are the ones we
first defended, still standing (§9).

---

## 2. What we are feeding, and what it owes

A **~40B India-first MoE**, small active fraction for cost. From Assignment 3 its non-negotiables
are coding and agentic work first, reasoning with controllable depth, long context, and a protected
Indic capability across 12 languages.

**Budget: 4T tokens**, the top of the 2.4-4T band. Pre-training ~97%, anneal reserve ~3%.

**The targets every lane is accountable to (A3):** MMLU-Pro ≥85 · AIME'26 ≥89 · LiveCodeBench-v6
≥80 · GPQA-Diamond ≥84 · BBEH ≥74 · **tau2-bench ≥77** · MMMLU ≥88 · MRCR-v2 (256K) ≥66 · and lead
the Indic benchmarks Gemma leaves blank.

A general assistant spends ~50% on web. We spend 30% and move the difference into code, agentic and
Indic, because we are building a sovereign coding and agentic model, not a chatbot. §3 tests whether
that ambition is affordable.

---

## 3. Can we actually buy these tokens?

Writing "agentic: 8%" takes a second. Finding 320B agentic tokens is close to impossible. This is
the check against wishful accounting, computed by [`supply/ledger.py`](supply/ledger.py).

### 3.1 Demand against real supply

| Lane | Demand | Real supply | Epochs | Verdict |
|---|---:|---:|---:|---|
| Web | 1200B | 20300B | 0.06 | Organic, abundant |
| Code | 880B | 1813B | 0.49 | Organic, comfortable |
| Math + STEM | 400B | 143B | **2.8** | Repetition, under the cap |
| Reasoning | 240B | 38B | **4.0** | At the cap **+ 88B generated** |
| Agentic | 320B | 43B | **4.0** | At the cap **+ 150B generated** |
| Long-context | 240B | 240B | 1.0 | Exactly enough, 60% already synthetic |
| Indic | 720B | 778B | 0.93 | Healthy in total, sick in its tiers (§3.3) |

**Why 4 epochs is the ceiling:** up to roughly four passes a repeated token is worth nearly as much
as a fresh one. Past that the model memorises instead of learning, at full price. Anything beyond
the cap we call generation, out loud.

### 3.2 Which datasets fill each slot

The three slots the brief singles out, pointed at named inventory sources. Full roster for all seven
lanes in [`supply/datasets.md`](supply/datasets.md). **`GEN` marks a source that does not exist yet:
a promise this plan is making, not a dataset we can point at today.**

| Slot | Named sources | Total | The uncomfortable part |
|---|---|---:|---|
| **Agentic** | ToolBench (0.08B) · xLAM/APIGen-60k (0.2B) · Glaive-function-calling-v2 (0.3B) · SWE-bench + terminal traces (2B) · **`GEN`** distilled Claude/Codex traces, cohort cloud-code sessions, simulated tool environments (40B) | 42.6B | **~2B is organic.** 95% of the named supply is itself to-be-generated, and the lane still needs +150B on top. |
| **Reasoning** | OpenThoughts / OpenR1 (8B) · AM-DeepSeek-Distilled-40M (30B) | 38B | **0% organic.** Both are already teacher-distilled and mostly English/Chinese. Long and ultra-depth traces do not exist in the wild. |
| **Long-context** | Books, Gutenberg / Books3-clean (30B) · arXiv full papers + repo-level code (60B) · **`GEN`** synthetic long via multi-doc packing (150B) | 240B | 90B organic. Naturally long *documents* are the constraint, not total tokens. |

The other lanes stand on firmer ground: web on DCLM-baseline, FineWeb and FineWeb-Edu (20.3T); code
on The Stack v2 and StarCoder2 (1.8T); math on OpenWebMath, Proof-Pile-2, peS2o/arXiv and NuminaMath
(143B).

### 3.3 Indic is four tiers, not one number

A single "18%" would hide everything that matters here.

| Tier | Share of lane | Demand | Supply | How it is met |
|---|:---:|---:|---:|---|
| **Verified** (Sangraha-Verified, IndicCorp v2, our A4 corpus) | 18% | 130B | ~86B | 1.5 epochs. Scarce and best. **Reserved for the anneal.** |
| **Unverified** (perplexity-filtered crawl) | 14% | 101B | ~24B | 4 epochs + ~5B generated |
| **Translated** (IndicTrans2, chrF++ gated) | 33% | 238B | ~305B | 0.8 epochs |
| **Synthetic** (Sangraha-Synthetic, BhashaKritika) | 35% | 252B | ~363B | 0.7 epochs |

Organic Indic is **~110B**. Verified alone is **3.2% of the whole budget**, and **~68% of the lane
is translated or synthetic**. At a 20T budget verified stays ~86B because it is supply-bound, so the
ratio shifts *further* toward synthetic. **Scaling the budget does not scale the scarce tier.**

### 3.4 The ceiling that settled the Indic share

110B organic Indic, 4-epoch ceiling, so 440B is everything that can ever be trained on. Against a 4T
budget that is **11.0%** - and it does not move:

| Indic share | Native % of budget | Synthetic % of budget |
|---:|---:|---:|
| 14% | 11.0% | 3.0% |
| **18% (ours)** | **11.0%** | **7.0%** |
| 24% | 11.0% | 13.0% |
| 32% | 11.0% | 21.0% |

**Raising the Indic lane cannot buy a single extra native Indic token.** It stacks synthetic text on
a fixed native core. §8.2 tested this empirically and it held. The lever that raises Indic
capability is **more verified data** - an acquisition problem, not a mixture problem.

---

## 4. The curriculum: order and length

Same tokens, different order, different model. Five stages: **Seed → General → Reasoning →
Long-context → Anneal.** Each cell is that lane's share *within* the stage.

| Lane \ Stage | **Seed** 8% | **General** 45% | **Reasoning** 25% | **Long-ctx** 19% | **Anneal** 3% | → integrated |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| General web | 58 | 39 | 21 | 12 | 8 | **29.96** |
| Code | 12 | 22 | 26 | 22 | 14 | **21.96** |
| Math + STEM | 6 | 10 | 12 | 9 | 11 | **10.02** |
| Reasoning traces | 2 | 4 | 9 | 7 | 16 | **6.02** |
| Agentic | 1 | 4 | 8 | 18 | 24 | **8.02** |
| Long-context | 3 | 3 | 6 | 14 | 9 | **6.02** |
| **Indic (protected floor)** | **18** | **18** | **18** | **18** | **18** | **18.00** |
| Difficulty band | B0-B1 | B1-B3 | B3-B4 | B4-B5 | **B5 only** | |
| Sequence length | 4K | 4-8K | 8-16K | **16-32K** | 32K | |

*Every column sums to 100, and the schedule was **solved** so the budget-weighted average reproduces
§1 within **0.04 points**: web = .08(58)+.45(39)+.25(21)+.19(12)+.03(8) = **29.96**. That precision
matters because §3 sizes supply against §1's shares - a curriculum quietly delivering something else
would make the entire supply analysis fiction.*

**The shape, and the reasoning behind it.** Web fades 58→8: breadth is foundational, then it hands
its budget over. Code, agentic and long-context climb, because they need a base to stand on. Indic
sits flat at exactly 18%, well above the 14% floor of §5.1, because a *protected* lane should not
fluctuate. Reasoning peaks in the anneal, where the hardest material meets the most capable model.

**One rule governs the ordering: a capability that depends on another comes later.** Long-context
always follows reasoning, because a long structured answer needs a model that can already think.

### 4.1 Difficulty bands

| Band | Level | Knowledge | Code |
|---|---|---|---|
| **B0** | Nursery | "The sun rises in the east." | `print("hello")` |
| **B1** | Grade school | "12 mangoes, sold 5, how many left?" | a `for` loop summing a list |
| **B2** | High school | "Balance: Fe + O2 → Fe2O3" | a function with input validation |
| **B3** | Undergraduate | "Prove the sum of the first n odd numbers is n²" | a REST handler with a unit test |
| **B4** | Graduate | "Derive the softmax-cross-entropy gradient" | a multi-file refactor, tests still green |
| **B5** | Research | a FrontierMath problem; a Supreme Court judgment analysis | a real SWE-bench patch |

Feed B5 too early and the model consumes without learning - and B5 is the most expensive data we
have.

### 4.2 Reasoning-length bands, on a separate axis

Difficulty is a property of the *problem*; length is a property of the *answer*. A hard question can
have a short answer and an easy one can be explored at length, so these are scheduled independently.

| Band | Thinking budget | Example |
|---|:---:|---|
| **short** | ≤64 tokens | "43 / 17 ≈ 2.5", no scratch work |
| **medium** | 64-512 | "1-1000 divisible by 3 or 5? 333+200−66 = **467**" |
| **long** | 512-4,000 | a competition geometry proof with explicit case analysis |
| **ultra** | 4,000-32,000 | a research problem, or a multi-step agentic debug with backtracking |

**Depth is trained, not prompted.** "Think harder" only works if the model has seen the same class
of problem answered at several tagged lengths. Without that data the instruction is decoration. Long
and ultra traces are distilled from a teacher run at matching effort.

### 4.3 Seams between stages

V4 saw gradients spike whenever the mixture shifted sharply. At every boundary we insert a **~0.5-1%
warm-up band blending old and new mixtures 60/40**, so the distribution diffuses instead of jumping.
Sequence length doubles per stage with **uniform-length batches** - a batch is all-4K or all-8K,
never mixed - and **nothing is padded below 4K**, because padding is compute bought and thrown away.

---

## 5. Guard-rails

### 5.1 The floor the selector may not cross

We use **OPUS-style online selection**: during training, score each candidate batch by how well its
gradient aligns with a difficulty-staged proxy of the target benchmarks, and keep the best ~50%.
Re-run every ~2B tokens, because that proxy drifts as the model improves - once math is solved, math
stops being informative. It is worth roughly **8x token efficiency**, which is why our collected
corpus (~1T) can exceed our trained budget.

The catch: **OPUS judges a sample from only its first ~512 tokens.** Two lanes lose badly under that
rule and are exempted:

- **Indic: floor of 14% of every batch, never trimmed.** The scoring proxy is English and code
  heavy, so Indic looks poorly aligned and gets discarded. Unprotected, OPUS would quietly delete
  our entire differentiator and we would find out at the end.
- **Agentic: 100% preserved, never trimmed.** An agent trace opens with plan and tool boilerplate.
  It reads like a log file, scores low, and gets thrown away - the data we have least of and paid
  most for.

**The rule worth carrying forward: an automatic filter's blind spot always lands on unusual data,
and unusual data is by definition what you are differentiating on.** Whenever you add a selector,
ask what it would delete that you cannot afford to lose, and hard-code protection.

### 5.2 The anneal reserve, decided now and spent last

**~3% (~120B tokens) is held back at composition time** and spent in a final low-learning-rate
cooldown, where the model barely moves and every step lands on the best material available.

| Reserved | Tokens | Why it waits |
|---|:---:|---|
| Tier-A verified Indic | 30B | Scarcest tier; biggest lift once the model can absorb it |
| Hardest verified agentic traces | 30B | Newest capability, wasted early |
| Ultra-length reasoning | 25B | Useless before the reasoning base exists |
| PhD / research STEM and math | 25B | B5 material, the "ready to absorb" window |
| Decontaminated benchmark-adjacent gold | 10B | Teaches the task *format*. Train splits only |

**The word that matters is *reserve*.** You cannot conjure Tier-A data in the last 3% if you already
spent it. The failure mode is never "forgot to anneal", it is "had nothing left to anneal with".

### 5.3 Keeping evaluation honest

Test splits never enter training. Every shard is decontaminated against the A3 eval suite with the
A4 n-gram scanner, and every lane carries source, licence and cleaning-script hash from A4's
provenance manifests. **We also audited our own proxy's validation sets and found a contaminated one
(§9.2).** A plan that demands decontamination should check itself first.

---

## 6. Which targets do these shares actually reach?

Computed by [`benchmarks.py`](benchmarks.py). **These are not predicted scores** - a tiny CPU proxy
cannot forecast MMLU-Pro, and inventing numbers is the exact failure §3 exists to prevent. Three
checkable facts per target instead.

| Benchmark | Target | Effective gradient | Organic backing | Verdict |
|---|:---:|:---:|:---:|---|
| MMLU-Pro | ≥85 | 22.0% | 100% | On track |
| LiveCodeBench-v6 | ≥80 | 22.0% | 100% | On track |
| MMMLU | ≥88 | 22.8% | 76% | Partly supply-limited |
| MILU / IndicGenBench | lead | 18.0% | 61% | Partly supply-limited |
| AIME '26 | ≥89 | 8.4% | 60% | Partly supply-limited |
| GPQA-Diamond | ≥84 | 8.4% | 60% | Partly supply-limited |
| BBEH | ≥74 | 13.2% | **30%** | **Supply-limited** |
| MRCR-v2 (256K) | ≥66 | 6.0% | 100% | **Untested** (proxy cannot reach) |
| tau2-bench | ≥77 | **2.8%** | **2%** | **Untested** (proxy cannot reach) |

- **Effective gradient** = share × the fraction of tokens that actually carry loss. Agentic traces
  mask their tool output, so **8% of tokens is 2.8% of learning**. Forget this and you will believe
  you trained agentic twice as hard as you did.
- **Organic backing** = how much of the demand real data could cover within 4 epochs. 2% means that
  lane is essentially all manufactured.
- **Verdict** follows from those two columns; it is derived, not chosen.

**Read the bottom row honestly:** our hardest promise has the least real data, the least gradient
and no test. BBEH is next. The fix is **not** a bigger share - raising agentic to 10% would push
generated tokens from 238B to ~398B - it is a better generation pipeline and a proxy that scores the
benchmark itself. A table where every row said "on track" would be a less trustworthy document.

---

## 7. The test this plan commits to

**No number in §1 is trusted at full scale until small proxy runs justify it:**

1. Train a **1B** proxy for **~20-40B tokens** on each candidate mixture, plus a 60/40 warm-up
   ablation at one stage seam.
2. Read **per-lane held-out loss** during the run and cheap benchmark deltas at the end
   (LiveCodeBench-lite, GSM8K, a MILU slice, a BFCL slice) - **with replicate seeds**, for reasons
   §9.1 makes painfully clear.
3. **Confirm** a mixture when up-weighting a lane lowers that lane's held-out loss *and* lifts its
   benchmark *without* regressing the protected floor. **Refute** it otherwise.
4. Promote only survivors to a **3B** confirmation, then commit the 4T budget. V4's 1B→2B and 3B→5B
   ballooning is precisely why cheap proxies exist.

**Metric: per-lane held-out loss at 1B, per-lane benchmark score at 3B.**

**What we could actually run.** This machine has 8 CPU cores and no GPU, so a 1B proxy is weeks of
compute and 3B is out of reach. We ran a **4.85M-parameter** GPT instead: 1500 steps, ~3M tokens,
5 lanes, 21 runs. That is ~200x below the specified scale and it is the study's biggest weakness
(§10). It was still sharp enough to kill two of our own recommendations.

**The discipline that made it readable.** Four rounds in, we realised we had never measured how much
a number moves for reasons unrelated to the mixture. Re-running the **identical** mixture at
different seeds:

| | web | code | math | reasoning | indic | **average** |
|---|---:|---:|---:|---:|---:|---:|
| **Seed-noise floor** | 0.144 | 0.329 | 0.318 | 0.240 | 0.346 | **0.051** |

Per-lane numbers are far noisier than the average. **Every figure below is quoted against its lane's
floor; anything smaller is reported as unreadable, not as a result.** Measuring this retired three
findings we had already written down.

---

## 8. Results

21 runs. Full log with predictions recorded *before* each round in
[`EXPERIMENTS.md`](EXPERIMENTS.md); raw curves in [`proxy/runs/`](proxy/runs).

### 8.1 What survived the noise floor

| Finding | Effect | Floor | What it establishes |
|---|---:|---:|---|
| Code share buys code | 0.488 | 0.329 | Allocation does buy capability |
| The Indic floor works (5%→20%) | 0.575 | 0.346 | The protected floor is measured value, not charity |
| Starving web costs common sense | 0.213 | 0.144 | "Great at code, no common sense", made numeric |
| **A lane rides on the whole diet** | 0.492 | 0.346 | Indic moves 0.492 at a **constant** 5% share; the variable is web |
| Code below ~18% hurts | 0.107 | 0.051 | Funding another lane out of code is the one move that measurably backfired |

The fourth row is why this document sizes a **mixture** and not seven independent lanes.

### 8.2 The experiment that settled Indic

For six rounds the proxy insisted "more Indic is better". We had one clean Indic bin, so more Indic
was free in a way it never is in reality. So we split the A4 corpus by **its own provenance labels**
(`anudesh` = native; `dolly`, `hhrlhf`, `toxicmatrix` = translated and synthetic), held native Indic
at the 11% §3.4 permits in **both** arms, and scored both on the **same held-out native set** -
because native Indic is what MILU measures.

| Scored on | Indic 18%<br>(11 native + 7 synth) | Indic 30%<br>(11 native + 19 synth) | Δ | Floor | Readable? |
|---|---:|---:|---:|---:|---|
| **Native Indic** | 6.062 | 5.971 | −0.091 | 0.243 | **no** |
| Translated / synthetic Indic | 4.738 | 4.216 | −0.522 | 0.087 | **yes** |

**Tripling the synthetic Indic mass buys fluency in machine-translated text and nothing measurable
on native Indic.** And here is why every earlier round disagreed: **the single Indic bin was 98.5%
translated or synthetic**, so "Indic loss" had been measuring the synthetic distribution all along.

A control arm at 30% *native* Indic - the lane the earlier proxy imagined, which cannot be supplied -
scored 5.886 against 6.062, still inside the floor. **Native Indic capability saturates once the
~11% the supply allows is spent.** Arithmetic and experiment reach 18% independently.

---

## 9. What we got wrong

The useful part of a study is the part that makes its authors look wrong.

| Claim we published | How it died |
|---|---|
| Every lane is monotone in its share | Generalised from 3 pairs never designed to test it |
| Indic is the one monotone lane | A later run broke it; our original evidence was itself sub-floor |
| Web scaffolds Indic, so cutting web costs Indic | 0.120 effect against a 0.346 floor |
| **v2 revision** (Indic 18→21, reasoning 6→9) improves the mixture | 0.011 against a 0.051 floor: no measurable difference |
| The clean test **refutes** raising Indic | That "refutation" was 0.073 against the floor: it showed nothing |
| More Indic buys Indic capability | §8.2: the bin was 98.5% synthetic and native Indic never moved |
| **Reasoning 6% → 9%** | §9.2: 100% train/validation leakage |

**Two were revisions already written into this plan.** One was a refutation we announced using the
same unmeasured-noise error that produced the claim it refuted.

### 9.1 Why "replicate seeds" is not boilerplate

We spent four rounds reading differences of 0.07 to 0.12 as signal. Then we re-ran the same mixture
at three seeds and found per-lane spreads of 0.14 to 0.35. **Most of what we had concluded was
noise.** A proxy study reporting one run per mixture is reporting its seed.

### 9.2 The audit that cost us our last recommendation

After §8.2 showed one metric had been measuring the wrong thing, we asked the question we should
have asked first: is each validation set genuinely unseen? We sampled 200 windows of 64 tokens per
lane and checked for verbatim matches in training data.

| Lane | web | code | math | reasoning | indic |
|---|---:|---:|---:|---:|---:|
| **Leakage** | 0% | 2% | 0% | **100%** | 6% |

`prepare_data.py` loops the GSM8K source **6 times** before splitting off the first 5% as
validation, so every reasoning validation window also sits in training. **Reasoning "held-out" loss
was never held out; it measured memorisation.** The 0.468 reasoning gain that survived every
noise-floor test - our last surviving proposal - is not evidence of capability.

Had we shipped it, we would have committed **+120B generated tokens** (5.9% → 8.9% of budget) on a
memorisation artefact. That is the price of not auditing your own metric.

### 9.3 The decision record

| Lane | Final | On what basis | Confidence |
|---|:---:|---|---|
| General web | **30%** | Benchmark floor; starving it measurably costs common sense (0.213 vs 0.144) | **High** |
| Code | **22%** | Abundant supply makes the share honest; below ~18% measurably hurts | **High** |
| Math + STEM | **10%** | Feeds AIME and GPQA; already 2.8 epochs, so raising it is expensive. No readable signal either way | Medium |
| Reasoning | **6%** | **Held only because its metric is invalid** (§9.2). The one share we cannot currently defend with evidence | **Low, open** |
| Agentic | **8%** | Generation-bound, not share-bound: +2 points costs +80B generated to move gradient 2.8%→3.5% | Medium, by reasoning not measurement |
| Long-context | **6%** | Already 1.0 epoch and 62.5% synthetic; won by training at length | Medium, by reasoning not measurement |
| Indic | **18%** | Settled twice over: experiment (§8.2) and arithmetic (§3.4) | **High** |

**Why "no change" is a result.** These shares came from the A3 benchmarks and the A4 inventory. We
then spent nine rounds and 21 runs attacking them, including two revisions we had already committed
to in writing. None survived. **A mixture that withstands its own authors trying that hard is better
defended than one tuned to a number.**

---

## 10. Limitations

1. **We ran 4.85M, not 1B and 3B.** No GPU on this machine. ~200x below the specified scale, so
   **the sign of a large effect is the claim, never the magnitude**, and nothing here is guaranteed
   to survive scaling. The harness, mixtures and metrics are already written for 1B; it is a
   rent-a-GPU job, not a rewrite.
2. **Held-out loss is not a benchmark.** Even a clean 1B run measures loss, and loss gains do not
   always become benchmark points.
3. **One lane's metric is invalid** (§9.2). All reasoning conclusions are void until it is fixed.
4. **Two lanes were never tested.** Agentic needs tool execution, long-context needs 32K sequences.
   Both are argued from the ledger and A3, not measured.
5. **The noise floor is itself approximate**, from 2-3 seeds per mixture. It is the most
   conservative number we measured; a serious study would use five or more.
6. **The native Indic bin is small** (1.03M tokens), making §8.2's headline metric the noisiest in
   the study. We trust it because it agrees with the independent arithmetic in §3.4; alone it would
   be weak.

---

## 11. Where the cleaning goes next

Cleaning continues toward the cumulative target, aimed at the slots this mixture shows to be
starved. Starvation is `demand − (organic supply × 4 epochs)`, straight from §3:

| Slot | Demand | Organic | Reachable | **Shortfall** | Priority |
|---|---:|---:|---:|---:|:---:|
| **Agentic** | 320B | ~2B | 8B | **312B** | **1** |
| **Reasoning** | 240B | 0B | 0B | **240B** | **2** |
| **Indic, verified tier** | 130B | 86B | 344B | covered, but **86B is a hard ceiling** | **3** |
| Long-context | 240B | 90B | 360B | covered by synthesis, quality-capped | 4 |

1. **Agentic first.** Biggest shortfall, and it feeds our hardest promise. The uniquely available
   source is **our own cohort's cloud-code sessions** - real multi-step tool use, already produced
   as a by-product of this course. It needs the A4 treatment plus a loss mask separating assistant
   spans from tool output, since only ~35% of those tokens carry gradient.
2. **Reasoning second.** 0% organic, so this is distillation rather than cleaning: generate at four
   tagged depths and **verify the final answer before keeping the trace**, because an unverified
   chain teaches a wrong method.
3. **Verified Indic third, and it is the highest-value token in the corpus** - not because the tier
   is short at 18%, but because §3.4 makes 86B a hard ceiling on native Indic and §8.2 shows no
   amount of synthetic substitutes for it. Every verified Indic token raises a ceiling no mixture
   decision can touch.

**A4 was not a prerequisite in name only.** §8.2 was possible only because A4 stamped provenance on
every document. Without `source` labels we could not have separated native from synthetic Indic, and
we would still believe more Indic was free. **Cleaning quality is what made this mixture testable.**

---

## 12. What would refute this plan

- **Indic held-out loss does not rise** when Indic is cut 18%→5% at 1B → the protected floor is
  over-insurance.
- **Web below 30% does not regress MMLU-Pro** at 1B → move budget to code and agentic.
- **The anneal reserve gives under ~1 point** at 3B → shrink it and return the tokens.
- **Agentic loss is flat regardless of share** → agentic belongs in SFT and RL, not pre-training,
  and 8% should fall.
- **Native Indic improves when synthetic Indic is added** at 1B → §8.2 does not scale and Indic
  should rise after all.

**Next three steps, in order:** fix the reasoning validation leak (split by document, not by token
offset) and re-test that share honestly; acquire more verified Indic, since capability there is
supply-bound; then re-run the ladder at 1B with replicate seeds, scored on MILU and AIME.

---

## 13. Reproduce, and the evidence behind every table

```bash
cd assignment-5
python3 supply/ledger.py          # -> supply/ledger.md + supply/datasets.md
python3 benchmarks.py             # -> benchmarks.md
python3 proxy/prepare_data.py     # tiny multi-lane corpus (reuses the A4 cleaned corpus)
python3 proxy/tokenize_lanes.py   # 16K BPE + per-lane train/val bins
bash    proxy/run_all.sh          # the mixture ladder -> proxy/runs/*.json
python3 proxy/tier_prepare.py     # split Indic by provenance (the §8.2 experiment)
python3 proxy/tier_train.py --name tier18   # also tier30, tier30_ideal
python3 experiments.py            # -> EXPERIMENTS.md
python3 build_dashboard.py && python3 -m http.server -d site 8777   # visual comparison tool
```

| If you want to check | Look at |
|---|---|
| Every run, prediction and withdrawal | [`EXPERIMENTS.md`](EXPERIMENTS.md) |
| Demand vs supply per lane and tier | [`supply/ledger.md`](supply/ledger.md) |
| Every named dataset per slot | [`supply/datasets.md`](supply/datasets.md) |
| Benchmark readiness maths | [`benchmarks.md`](benchmarks.md) |
| Per-lane loss, noise floors, audits | [`proxy/results.md`](proxy/results.md) |
| Raw training curves | [`proxy/runs/`](proxy/runs) |
| The concepts in plain language | [`CONCEPTS.md`](CONCEPTS.md) |
| An interactive comparison of every mixture | [`site/`](site) |

Deps: `datasets`, `tokenizers`, `torch` (CPU), `numpy`. Every computed artefact is committed, so all
tables reproduce from a clean clone without re-running anything.

---

**Continuity.** A2 built the tokenizer. A3 designed the 40B model and chose the benchmark targets
this plan composes backward from. A4 cleaned and provenance-stamped the corpus, and that provenance
is what made §8.2 possible. **A5 is the recipe that turns A4's tokens into A3's targets**, written
so every number is a claim someone can check - and several of them turned out to be wrong.

_Built by Ashish Kamble · defended against the inventory, the benchmarks, and 21 runs that tried to
break them._
