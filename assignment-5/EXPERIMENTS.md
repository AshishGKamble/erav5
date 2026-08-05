# Experiment log - what we tried, what refuted us, and why the final plan looks like it does

The plan claims a mixture is a hypothesis that a cheap run can test. This is the evidence that
the claim is operational and not decorative: **two of our own recommendations were killed by
our own runs**, and a third round exists only because we realised we had never measured whether
our differences were bigger than noise.

Narrative is authored in `experiments.py`; every number is read live from `proxy/runs/*.json`.

---

## All runs, in the order they were made

| # | Run | web/code/math/reas/Indic | web | code | math | reasoning | indic | avg |
|--:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | `naive_web` | 70/15/5/5/5 | 6.312 | 5.696 | 6.733 | 5.387 | 4.981 | 5.822 |
| 2 | `ours` | 35/25/12/8/20 | 6.541 | 5.475 | 6.649 | 5.273 | 4.407 | 5.669 |
| 3 | `code_heavy` | 8/55/25/7/5 | 6.754 | 5.208 | 6.321 | 5.482 | 5.473 | 5.848 |
| 4 | `indic_first` | 28/18/10/12/32 | 6.520 | 5.445 | 6.579 | 5.188 | 4.079 | 5.562 |
| 5 | `reasoning_fwd` | 35/15/10/20/20 | 6.268 | 5.847 | 6.917 | 4.804 | 4.402 | 5.648 |
| 6 | `web_lean` | 20/35/17/8/20 | 6.554 | 5.531 | 6.712 | 5.340 | 4.526 | 5.732 |
| 7 | `v2_proposed` | 30/23/12/10/25 | 6.430 | 5.619 | 6.510 | 5.256 | 4.431 | 5.649 |
| 8 | `indic_clean` | 35/17/10/8/30 | 6.467 | 5.539 | 6.659 | 5.455 | 4.479 | 5.720 |
| 9 | `ours_seed7` | 35/25/12/8/20 | 6.397 | 5.804 | 6.468 | 5.488 | 4.202 | 5.672 |
| 10 | `ours_seed99` | 35/25/12/8/20 | 6.522 | 5.505 | 6.729 | 5.248 | 4.190 | 5.639 |
| 11 | `indic30_web30` | 30/18/10/12/30 | 6.517 | 5.487 | 6.516 | 5.297 | 4.353 | 5.634 |
| 12 | `indic30_web30_seed7` | 30/18/10/12/30 | 6.436 | 5.761 | 6.630 | 5.094 | 4.007 | 5.585 |
| 13 | `indic_first_seed7` | 28/18/10/12/32 | 6.469 | 5.642 | 6.897 | 5.200 | 3.769 | 5.595 |
| 14 | `indic_first_seed99` | 28/18/10/12/32 | 6.538 | 5.756 | 6.697 | 5.048 | 3.987 | 5.605 |
| 15 | `web30_indic32` | 30/16/10/12/32 | 6.498 | 5.714 | 6.698 | 5.133 | 4.304 | 5.669 |
| 16 | `web30_indic32_seed7` | 30/16/10/12/32 | 6.478 | 5.727 | 6.852 | 5.325 | 4.217 | 5.720 |

_Best average: **indic_first** (5.562). Worst: code_heavy (5.848). 16 runs, 4.85M params, 1500 steps each._


---

## Round 1 - Baselines - does allocation buy capability at all?

**Added:** `naive_web`, `ours`, `code_heavy`

**Question.** Three philosophies: the lazy web-heavy default, our India-first pick, and the over-corrected coding model. Does a lane's share move that lane's held-out loss?

**Predicted before running:**
- Indic 5%->20% lowers Indic loss
- web 35%->8% raises web loss
- code 15%->55% lowers code loss

**Measured:**

| Comparison | lane | Δ |
|---|:---:|---:|
| `naive_web` → `ours` | indic | -0.575 (better) |
| `ours` → `code_heavy` | web | +0.213 (worse) |
| `naive_web` → `code_heavy` | code | -0.488 (better) |

**Verdict.** **3/3 confirmed.** Allocation buys capability, and `ours` took the best average.

**What we changed as a result.** Adopted `ours` as the plan's mixture. Wrote that 'every lane's loss is monotone in its budget share' - a claim that went beyond what these three runs could support.

---

## Round 2 - Attack the general claim, not the three pairs

**Added:** `indic_first`, `reasoning_fwd`, `web_lean`

**Question.** Round 1 tested three pairs and we generalised from them. Do the lanes really respond to their own share, and can Indic be the largest lane?

**Predicted before running:**
- Indic-first costs the lanes it defunds
- reasoning responds to its share once the base is intact
- web_lean: cutting web at constant Indic will show whether web props Indic up

**Measured:**

| Comparison | lane | Δ |
|---|:---:|---:|
| `ours` → `indic_first` | indic | -0.327 (better) |
| `ours` → `reasoning_fwd` | reasoning | -0.468 (better) |
| `ours` → `web_lean` | indic | +0.120 (worse) |
| `ours` → `indic_first` | avg | -0.107 (better) |

**Verdict.** **Two surprises.** `indic_first` beat `ours` on all five lanes at once, and `reasoning_fwd` produced the largest single-lane gain in the study. `web_lean` showed cutting web hurts Indic at a constant Indic share.

**What we changed as a result.** Withdrew the monotonicity claim (four of five lanes broke it). Proposed a revised 7-lane mixture v2: Indic 18->21, reasoning 6->9, web 30->26, code 22->20.

> **Later corrected.** Round 5 killed two of this round's conclusions. The `web_lean` finding (0.120) is below the Indic noise floor (0.216), so 'cutting web hurts Indic' is withdrawn, and most of the monotonicity breaks we cited were sub-floor too. What survives is `reasoning_fwd` (0.468) and `indic_first` (0.327 on Indic, 0.107 on the average) - both comfortably real.

---

## Round 3 - Test the revision we had just recommended

**Added:** `v2_proposed`

**Question.** v2 was argued from round 2 but never run. Does the mixture we recommended actually beat the one it replaces?

**Predicted before running:**
- v2 beats `ours` on Indic and reasoning without losing much code

**Measured:**

| Comparison | lane | Δ |
|---|:---:|---:|
| `ours` → `v2_proposed` | indic | +0.025 (worse) |
| `ours` → `v2_proposed` | reasoning | -0.017 (better) |
| `ours` → `v2_proposed` | code | +0.143 (worse) |
| `ours` → `v2_proposed` | avg | -0.020 (better) |

**Verdict.** **Refuted** (as we called it at the time). v2 came out level with `ours`, and Indic got *worse* despite a larger Indic share.

**What we changed as a result.** Diagnosed the cause: v2 funded Indic partly out of web, and round 2 had shown web props Indic up - so the two moves cancelled. Noticed that EVERY Indic test so far had also moved web, which confounds all of them.

> **Later corrected.** Round 5 shows 'refuted' was too strong in the other direction. v2 differs from `ours` by 0.020 on an average metric whose floor is 0.033, and by 0.025 on Indic against a 0.216 floor. The correct verdict is **no measurable difference** - v2 was neither confirmed nor refuted - and our diagnosis of *why* rested on a web effect we could not read either.

---

## Round 4 - Isolate Indic from web - the confound-free test

**Added:** `indic_clean`

**Question.** Pin web at 35% and reasoning at 8% (identical to `ours`) and fund Indic 20->30 entirely from code and math. Does the Indic lever pay above 20%?

**Predicted before running:**
- Indic loss falls; the lever is real once web is held constant

**Measured:**

| Comparison | lane | Δ |
|---|:---:|---:|
| `ours` → `indic_clean` | indic | +0.073 (worse) |
| `ours` → `indic_clean` | avg | +0.051 (worse) |

**Verdict.** **Refuted, and worse than refuted.** Indic got worse and the mixture got worse. The result also contradicts `indic_first`, which scored far better on Indic at a similar share - two runs at ~30% Indic that disagree by a wide margin.

**What we changed as a result.** Stopped recommending an Indic increase; the plan's original 18% stands. Started measuring the seed-to-seed noise floor, which we had never established - without it, a 0.07 delta cannot be called signal.

> **Later corrected.** Round 5 invalidates this round's headline. The 0.073 'refutation' is a third of the 0.216 Indic floor - it shows nothing either way. Worse for us, once the floor is known the proxy actually **favours more Indic**: `indic_first` at 32% is the best mixture tested on the stable average metric. The reason to hold at 18% is the ledger's supply-quality objection, not this run.

---

## Round 5 - How much of this study is even readable? (noise floor)

**Added:** `ours_seed7`, `ours_seed99`

**Question.** Re-run the SAME mixture at different seeds. The spread is the noise floor, and every delta above must be judged against it.

**Predicted before running:**
- seed spread is small (<0.05), so the mid-sized effects are real

**Measured:**

| Comparison | lane | Δ |
|---|:---:|---:|
| `ours` → `ours_seed7` | avg | +0.003 (worse) |
| `ours` → `ours_seed99` | avg | -0.030 (better) |
| `ours` → `ours_seed7` | indic | -0.205 (better) |
| `ours` → `ours_seed99` | indic | -0.216 (better) |

**Verdict.** Re-running the *identical* mixture at 2 other seeds moves per-lane loss by as much as **0.329**. Per-lane seed spread: **web 0.144**, **code 0.329**, **math 0.261**, **reasoning 0.240**, **indic 0.216** - but the **average is stable at 0.033**. Single-lane readings at this scale are far noisier than we assumed; the average is the trustworthy metric.

**Every effect in this study, judged against its own lane's floor:**

| Effect | lane | size | lane floor | readable? |
|---|:---:|---:|---:|---|
| Code share buys code (round 1) | code | 0.488 | 0.329 | **REAL** |
| Starving web costs common sense (round 1) | web | 0.213 | 0.144 | **REAL** |
| Indic floor works (round 1) | indic | 0.575 | 0.216 | **REAL** |
| Indic depends on the REST of the mix, at a constant 5% share | indic | 0.492 | 0.216 | **REAL** |
| Reasoning 8% -> 20% (round 2) | reasoning | 0.468 | 0.240 | **REAL** |
| Indic 20% -> 32%, indic-first (round 2) | indic | 0.327 | 0.216 | **REAL** |
| Cutting web 35% -> 20% at constant Indic (round 2) | indic | 0.120 | 0.216 | not readable |
| v2 revision, the mixture we recommended (round 3) | indic | 0.025 | 0.216 | not readable |
| Indic 20% -> 30% with web pinned, clean test (round 4) | indic | 0.073 | 0.216 | not readable |
| Best mixture found, on the stable average metric | avg | 0.107 | 0.033 | **REAL** |

**What we changed as a result.** Three findings we had published - including one of our own *refutations* - fell below the floor and were withdrawn. Every surviving claim in the plan now carries the floor beside it, and the 1B proxy specification gained a requirement it did not have before: **run replicate seeds**, because a single run cannot separate a 0.1 lane effect from chance.

---

## Round 6 - Tune the only winner - can Indic 32% survive a fuller web base?

**Added:** `indic30_web30`, `indic30_web30_seed7`

**Question.** `indic_first` (web 28 / Indic 32) is the one mixture that beat `ours` above the noise floor. Trade 2 points of Indic back into web (web 30 / Indic 30), holding its other lanes at 18/10/12. Does the win survive a less extreme Indic share?

**Predicted before running:**
- the win survives: Indic 30% is still far above the 20% that `ours` runs, and 30% web is a fuller base than indic_first's 28%
- if it does NOT survive, the indic_first result was partly about its low web share rather than its high Indic share

**Measured:**

| Comparison | lane | Δ |
|---|:---:|---:|
| `ours` → `indic30_web30` | avg | -0.035 (better) |
| `indic_first` → `indic30_web30` | avg | +0.072 (worse) |
| `ours` → `indic30_web30` | indic | -0.054 (better) |
| `indic_first` → `indic30_web30` | indic | +0.274 (worse) |
| `indic30_web30` → `indic30_web30_seed7` | avg | -0.048 (better) |

**Verdict.** **Prediction 1 half right, and the half that failed is the informative one.** Averaged over its two seeds the new mixture scores **5.610**, which beats `ours` (3-seed mean 5.660) by 0.050 against a 0.033 floor - a **real improvement, and the second-best mixture tested**. But it does *not* match `indic_first` (5.562): giving back 2 points of Indic for 2 points of web cost 0.047. On the Indic lane itself the move is not readable (0.086 against a 0.216 floor).

**What we changed as a result.** Kept high-Indic as a genuinely supported direction - it is now backed by TWO independent mixtures rather than one, and this one has replicate seeds. But we did not adopt either: Indic at 30% still drops organic backing to ~37%, so the ledger objection is untouched.

> **Later corrected.** Two honest limits on this round. (1) We moved Indic AND web together, so we cannot say which of the two costs the 0.047 - the same confounding that spoiled rounds 3 and 4. (2) `indic_first` has only ONE seed, and this mixture's own seed spread on the average was 0.048 - larger than the 0.033 floor taken from `ours`. Calling `indic_first` the better mixture therefore rests on an unreplicated run, and the honest statement is that 30% and 32% Indic are **not yet distinguishable**. Round 7 replicated `indic_first` and confirmed exactly that: at 3 seeds it means 5.588 against this mixture's 5.610 - a 0.022 gap against a 0.051 floor. **The two are tied.** The 0.047 'cost' this round reported was itself never readable.

---

## Round 7 - Replicate the winner, and isolate which change cost round 6

**Added:** `indic_first_seed7`, `indic_first_seed99`, `web30_indic32`, `web30_indic32_seed7`

**Question.** Two open questions. (a) `indic_first` led on ONE seed - does it survive replication? (b) Round 6 moved Indic and web together; hold Indic at 32% and raise web 28->30 alone (compensating from code 18->16) to see which change carried the cost.

**Predicted before running:**
- indic_first survives replication and stays the best mixture
- if `web30_indic32` matches indic_first, the web move was harmless and the Indic 32->30 drop was the cost; if it matches indic30_web30, the web move was the cost

**Measured:**

| Comparison | lane | Δ |
|---|:---:|---:|
| `ours` → `indic_first` | avg | -0.107 (better) |
| `indic_first` → `indic30_web30` | avg | +0.072 (worse) |
| `indic_first` → `web30_indic32` | avg | +0.107 (worse) |
| `indic30_web30` → `web30_indic32` | avg | +0.035 (worse) |

**Verdict.** **(a) Confirmed - and the floor rose.** `indic_first` replicates: 3-seed mean **5.588** vs `ours` 5.660, Δ 0.072 against a floor now measured at **0.051** (the worst seed spread across all replicated mixtures, up from the 0.033 we had been quoting). It remains the only mixture readably better than `ours`. **(b) Neither - the question was malformed.** `indic_first` (5.588) and `indic30_web30` (5.610) differ by 0.022, well inside the floor: **they are tied**, so round 6's 0.047 'cost' was never real. What IS readable is the third run: `web30_indic32` (5.695) is 0.107 worse than `indic_first` - and its only distinctive feature is **code cut to 16%**.

**What we changed as a result.** Replaced 'indic_first is the best mixture' with the claim the data actually supports: **high Indic (30-32%) with code held at >=18% beats the shipped 18-20% Indic**, and the exact optimum inside that band is not resolvable at this scale. Added a new constraint we did not have before: **do not fund Indic by cutting code below ~18%** - that is the one move in round 7 that measurably hurt.

> **Later corrected.** One caveat on the floor itself: 0.051 comes from 2-3 seeds per mixture, which is a crude estimate of a spread. It is almost certainly the right order of magnitude and it is the most conservative number we have measured, so we use it - but a serious study would run 5+ seeds before quoting a floor to three decimals.

---

## Round 8 - The tier experiment - was any of the Indic gain ever real?

**Added:** 

**Question.** Every earlier Indic result used ONE uniformly clean Indic bin, so 'more Indic' was free. The ledger says it is not - and its arithmetic is sharper than we realised: organic Indic is 110B, the ceiling is 4 epochs, so 440B is reachable = **11.0% of a 4T budget AT EVERY INDIC SHARE**. Raising the lane cannot buy more native Indic; it buys only translated and synthetic tokens. So: does 12 more points of SYNTHETIC Indic (7%->19%) improve capability on NATIVE Indic, which is what MILU measures? Split the A4 corpus by its own provenance (anudesh = native; dolly/hhrlhf/toxicmatrix = translated+synthetic), hold native at 11% in both arms, score both on the same held-out NATIVE set.

**Predicted before running:**
- if tier30 beats tier18 on native Indic, raise the lane despite the dilution
- if it does not, 18% is confirmed and the extra budget belongs elsewhere

**Measured** (both arms hold native Indic at 11%; scored on the same held-out sets):

| Scored on | tier18 (11 native + 7 synth) | tier30 (11 native + 19 synth) | Δ | floor | readable? |
|---|---:|---:|---:|---:|---|
| **NATIVE Indic** (what MILU measures) | 6.062 | 5.971 | -0.091 | 0.243 | **no** |
| translated/synthetic Indic | 4.738 | 4.216 | -0.522 | 0.087 | **yes** |

**Verdict. The Indic gain was never a native-Indic gain.** Tripling the synthetic Indic mass produces **no readable change in native Indic** (-0.091 against a 0.243 floor) while producing a large, unambiguous improvement on the **synthetic** distribution (-0.522 against a 0.087 floor). The model gets fluent in machine-translated Indic, which is not the capability we promised.

And the smoking gun: the single Indic bin used in rounds 1-7 is **98.5% translated/synthetic by document count**. So 'Indic held-out loss' in every earlier round was overwhelmingly a measurement of the synthetic distribution. `indic_first`'s win was real - it was just a win at the wrong thing.

The control settles the rest: `tier30_ideal` (30% *native* Indic - the lane the earlier proxy implicitly assumed, and which cannot be supplied) scores 5.886 against tier18's 6.062, a gap of -0.176 that is still inside the 0.243 floor. Even unbuyable clean Indic at 30% does not measurably beat 18%: **native Indic capability saturates once the ~11% the supply allows is spent.**

**What we changed as a result.** **Indic finalises at 18%** - confirmed for the right reason at last, by the experiment the plan named for itself. The `indic_first` direction is withdrawn entirely. The budget it would have consumed stays where §1 put it.

---

## Round 9 - Validity audit - is each lane's 'held-out' set actually held out?

**Added:** 

**Question.** Round 8 showed one lane's metric had been measuring the wrong distribution. That prompts the obvious question we should have asked first: for EVERY lane, is the validation set genuinely unseen? Sample 200 windows of 64 tokens from each lane's val bin and check whether they appear verbatim in that lane's train bin.

**Predicted before running:**
- all lanes are clean; val is a 5% head split of a single pass over the source

**Measured** - share of val windows found verbatim in the same lane's train bin:

| Lane | leakage | verdict |
|---|---:|---|
| web | 0% | clean |
| code | 2% | clean |
| math | 0% | clean |
| reasoning | 0% | clean |
| indic | 6% | clean |

**Verdict.** All lanes clean; every held-out number in this log is honest.

---

## Claims we made and then withdrew

The useful part of a log is the part that makes its author look wrong.

| Claim | Made after | Killed by | The mistake |
|---|---|---|---|
| Every lane's held-out loss is monotone in its budget share. | Round 1 (3 runs) | Round 2 - four of five lanes showed a larger share with a worse loss. | Generalised a rule from three hand-picked pairs that were never designed to test it. |
| Indic is the one lane that is genuinely monotone. | Round 2 (6 runs) | Round 5 - `indic_clean` at 30% Indic scores 4.479 against `ours_seed99` at 20% scoring 4.190, a gap of 0.289 that clears the 0.216 floor. (The round-3 evidence we first cited for this was itself only 0.029 - below the floor, and no evidence at all.) | Right conclusion, wrong evidence: we called the break using a gap we could not actually read, and only a later replicate produced one we could. |
| Raise Indic 18% -> 21% and reasoning 6% -> 9% (the v2 revision). | Round 2 (6 runs) | Round 3 - v2 scores 5.649 against 5.669 for `ours`, a 0.020 difference on a metric whose floor is 0.033. Not an improvement, and not a refutation either: simply no measurable effect. | Recommended a mixture before running it, from comparisons that confounded Indic with web - then described the null result as a refutation, which overstates it in the opposite direction. |
| Web scaffolds Indic, so cutting web always costs Indic. | Round 2 (6 runs) | Round 5 - the effect is 0.120 against an Indic noise floor of 0.216. Not readable. | Built a causal story on a difference smaller than seed noise, having never measured the noise. |
| The clean test refutes raising Indic - it made Indic worse. | Round 4 (8 runs) | Round 5 - that 'refutation' is 0.073 against a 0.216 floor. It shows nothing either way. | Announced a refutation with the same unmeasured-noise error that produced the claim it refuted. |
| Indic should stay at 18% because the proxy says more Indic does not pay. | Round 4 (8 runs) | Round 5 - on the stable average metric `indic_first` (Indic 32%) beats `ours` by 0.107 against a 0.033 floor, and its Indic gain (0.327) clears the 0.216 floor too. | The proxy actually supports MORE Indic. The real objection is supply quality, not the proxy - and we had reached the right answer through a wrong argument. |

---

## Why the final plan looks like it does

- **Indic finalises at 18%, settled by experiment (round 8).** For six rounds the proxy said 'more Indic'. Round 8 split the Indic lane by real provenance and showed why: **the single Indic bin was 98.5% translated/synthetic**, so every earlier 'Indic gain' was a gain on the *synthetic* distribution. Holding native Indic at the 11% the supply allows and tripling the synthetic mass on top moves native Indic by -0.091 against a 0.243 floor - **nothing** - while moving synthetic Indic by -0.522. The model was learning to be fluent in machine-translated text, which is not what MILU scores. The ledger was right and the proxy had been measuring the wrong quantity.
- **The arithmetic behind it, which needs no experiment at all.** Organic Indic is 110B and the repetition ceiling is 4 epochs, so 440B is reachable - **exactly 11.0% of a 4T budget, at every possible Indic share**. Raising the lane from 18% to 32% cannot buy a single extra native Indic token. It buys 14 more points of synthetic. Once stated that way the decision is not close.
- **Reasoning is the one share we would still move (6% → 9%).** Its gain was the largest in the study, and `reasoning_fwd` held web constant, so it is the only large effect that was never confounded. It is held as *proposed* because it costs +120B of generated tokens.
- **Web stays at 30%.** The readable evidence is the web lane itself: at 8% web, web loss is 0.213 worse than at 35% (floor 0.144). Our prettier claim - that cutting web drags *Indic* down with it - measured only 0.120 against a 0.216 Indic floor and has been withdrawn. So web is defended because gutting it demonstrably costs common sense, not because of a scaffolding story we could not actually measure.
- **Agentic stays at 8% and long-context at 6%.** Neither is share-bound: agentic is 4.7% organic and 0.35 trainable, long-context is already at 1.0 epoch with 62.5% synthetic supply. The proxy cannot test either, and we did not pretend otherwise.

### The experiment that settled the Indic share (now run - round 8)

We wrote this section as future work, then ran it. Splitting the Indic bin by the A4 corpus's own provenance labels (anudesh = native; dolly/hhrlhf/toxicmatrix = translated+synthetic) and scoring both arms on the same held-out **native** set showed the gain does not survive realistic dilution. **18% is confirmed for the right reason.** The remaining open question is narrower and more useful: native Indic capability appeared to saturate at the ~11% of budget the supply allows, even in an idealised arm we cannot supply - so the lever that would actually raise Indic capability is **more verified Indic data**, not a bigger Indic share. That is a data-acquisition problem, not a mixture problem, and it is where the next effort belongs.

**The honest summary for a reviewer:** the plan's original shares survived ten runs and five rounds of attack, two of which were recommendations we had written into the plan before testing them. We also published a refutation that our own noise floor later invalidated. The one change we would still make is **reasoning 6% → 9%** - the largest effect in the study, never confounded by web, and comfortably above its floor - and we have priced it at +120B generated tokens. Everything here is at 4.85M params with a per-lane noise floor of 0.14-0.33; the real decision belongs to the 1B proxy, scored on MILU and AIME, **with replicate seeds**.
