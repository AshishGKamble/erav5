# Proxy results - per-lane held-out loss

Tiny GPT trained on 16 candidate mixtures over identical lanes; lower loss = better at that lane. This is a demonstration-scale stand-in for the 1B/3B proxy - the *method* is what transfers.

| Mixture | code | indic | math | reasoning | web | avg |
|---|---:|---:|---:|---:|---:|---:|
| **naive_web** (Indic 5% / web 70%) | 5.696 | 4.981 | 6.733 | 5.387 | 6.312 | 5.822 |
| **ours** (Indic 20% / web 35%) | 5.475 | 4.407 | 6.649 | 5.273 | 6.541 | 5.669 |
| **code_heavy** (Indic 5% / web 8%) | 5.208 | 5.473 | 6.321 | 5.482 | 6.754 | 5.848 |
| **indic_first** (Indic 32% / web 28%) | 5.445 | 4.079 | 6.579 | 5.188 | 6.520 | 5.562 |
| **reasoning_fwd** (Indic 20% / web 35%) | 5.847 | 4.402 | 6.917 | 4.804 | 6.268 | 5.648 |
| **web_lean** (Indic 20% / web 20%) | 5.531 | 4.526 | 6.712 | 5.340 | 6.554 | 5.732 |
| **v2_proposed** (Indic 25% / web 30%) | 5.619 | 4.431 | 6.510 | 5.256 | 6.430 | 5.649 |
| **indic30_web30** (Indic 30% / web 30%) | 5.487 | 4.353 | 6.516 | 5.297 | 6.517 | 5.634 |
| **indic30_web30_seed7** (Indic 30% / web 30%) | 5.761 | 4.007 | 6.630 | 5.094 | 6.436 | 5.585 |
| **indic_clean** (Indic 30% / web 35%) | 5.539 | 4.479 | 6.659 | 5.455 | 6.467 | 5.720 |
| **indic_first_seed7** (Indic 32% / web 28%) | 5.642 | 3.769 | 6.897 | 5.200 | 6.469 | 5.595 |
| **indic_first_seed99** (Indic 32% / web 28%) | 5.756 | 3.987 | 6.697 | 5.048 | 6.538 | 5.605 |
| **ours_seed7** (Indic 20% / web 35%) | 5.804 | 4.202 | 6.468 | 5.488 | 6.397 | 5.672 |
| **ours_seed99** (Indic 20% / web 35%) | 5.505 | 4.190 | 6.729 | 5.248 | 6.522 | 5.639 |
| **web30_indic32** (Indic 32% / web 30%) | 5.714 | 4.304 | 6.698 | 5.133 | 6.498 | 5.669 |
| **web30_indic32_seed7** (Indic 32% / web 30%) | 5.727 | 4.217 | 6.852 | 5.325 | 6.478 | 5.720 |

_4.85M params · 1500 steps · ~3.07M tokens seen · block 128._

## Does the mixture behave like a testable hypothesis? (confirm / refute)

- ✅ CONFIRMED - **Protected Indic floor works.** Indic loss: ours 4.407 vs naive_web 4.981 (Δ +0.575). Raising Indic 5%→20% lowers Indic held-out loss - allocation buys capability.
- ✅ CONFIRMED - **Starving web costs common sense.** Web loss: ours 6.541 vs code_heavy 6.754 (Δ +0.213). The 'great at code, no common sense' failure, measured.
- ✅ CONFIRMED - **Code share buys code.** Code loss: code_heavy 5.208 vs naive_web 5.696 (Δ +0.488). The lane with the largest code share has the lowest code loss.

**3/3 predictions confirmed** on the pairs they name, so a mixture is a hypothesis a cheap run can test - exactly the claim the plan makes at 1B/3B scale.

## Audit 1 - is each lane monotone in its own share?

Sort the runs by a lane's share and check that loss falls as the share rises. This is the *naive* model of a mixture, and it does not hold everywhere.

| Lane | share → loss (ascending share) | monotone? |
|---|---|:---:|
| code | 15%→5.696, 15%→5.847, 16%→5.714, 16%→5.727, 17%→5.539, 18%→5.445, 18%→5.487, 18%→5.642, 18%→5.756, 18%→5.761, 23%→5.619, 25%→5.475, 25%→5.505, 25%→5.804, 35%→5.531, 55%→5.208 | **no** (floor 0.329) |
| indic | 5%→4.981, 5%→5.473, 20%→4.190, 20%→4.202, 20%→4.402, 20%→4.407, 20%→4.526, 25%→4.431, 30%→4.007, 30%→4.353, 30%→4.479, 32%→3.769, 32%→3.987, 32%→4.079, 32%→4.217, 32%→4.304 | yes (floor 0.346) |
| math | 5%→6.733, 10%→6.516, 10%→6.579, 10%→6.630, 10%→6.659, 10%→6.697, 10%→6.698, 10%→6.852, 10%→6.897, 10%→6.917, 12%→6.468, 12%→6.510, 12%→6.649, 12%→6.729, 17%→6.712, 25%→6.321 | yes (floor 0.318) |
| reasoning | 5%→5.387, 7%→5.482, 8%→5.248, 8%→5.273, 8%→5.340, 8%→5.455, 8%→5.488, 10%→5.256, 12%→5.048, 12%→5.094, 12%→5.133, 12%→5.188, 12%→5.200, 12%→5.297, 12%→5.325, 20%→4.804 | yes (floor 0.240) |
| web | 8%→6.754, 20%→6.554, 28%→6.469, 28%→6.520, 28%→6.538, 30%→6.430, 30%→6.436, 30%→6.478, 30%→6.498, 30%→6.517, 35%→6.268, 35%→6.397, 35%→6.467, 35%→6.522, 35%→6.541, 70%→6.312 | yes (floor 0.144) |

_Two runs at the same share scoring differently is not a violation - that is an interaction (Audit 2). Nor is a difference smaller than the lane's seed-noise floor._

**Where it genuinely breaks (bigger than the lane's noise floor):**

- **code**: `ours_seed7` gives code a larger share (25% vs 18% in `indic_first`) yet a *worse* loss (5.804 vs 5.445, gap 0.359 > floor 0.329).

**No readable violation in: indic, math, reasoning, web.** Every apparent break in these lanes is smaller than the spread we get by re-running the *same* mixture at another seed. We previously reported those breaks as a finding; with the floor measured, they are withdrawn. At this scale the honest statement is that the proxy **cannot resolve** whether these lanes are monotone - not that they are, and not that they are not.

## Audit 2 - how much does a lane depend on the REST of the mixture?

Pairs of runs holding one lane at the *same* share. Under a pure per-lane model these would score identically; the gap is the interaction effect.

| Lane | held at | run A | run B | gap | vs floor |
|---|:---:|---|---|---:|---|
| indic | 32% | indic_first_seed7 3.769 | web30_indic32 4.304 | **0.535** | 0.346 **real** |
| indic | 5% | naive_web 4.981 | code_heavy 5.473 | **0.492** | 0.346 **real** |
| indic | 30% | indic30_web30_seed7 4.007 | indic_clean 4.479 | **0.472** | 0.346 **real** |
| indic | 32% | indic_first_seed7 3.769 | web30_indic32_seed7 4.217 | **0.448** | 0.346 **real** |
| math | 10% | reasoning_fwd 6.917 | indic30_web30 6.516 | **0.402** | 0.318 **real** |
| math | 10% | indic30_web30 6.516 | indic_first_seed7 6.897 | **0.381** | 0.318 **real** |

**Largest interaction: indic at a constant 32% share still moves by 0.535** between `indic_first_seed7` and `web30_indic32`. A lane is not bought by its own share in isolation; it rides on what else is in the diet. This is the single most useful thing the proxy told us, and it is why the plan sizes a *mixture* rather than tuning lanes one at a time.
