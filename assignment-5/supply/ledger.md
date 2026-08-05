# Supply ledger (computed)

Budget = **4.0T** tokens (primary). Repetition ceiling = 4.0 epochs.
Numbers in **billions** of tokens. `gen` = must be generated/synthesised.

| Lane | Share | Demand | Real supply | Trainable frac | How it's met |
|------|------:|-------:|------------:|:--------------:|--------------|
| web | 29% | 1160 | 20300 | 1.00 | 0.06 epochs of 20300B |
| code | 20% | 800 | 1813 | 1.00 | 0.44 epochs of 1813B |
| math_stem | 10% | 400 | 143 | 1.00 | 2.8 epochs of 143B |
| reasoning_traces | 9% | 360 | 38 | 1.00 | 4.0 epochs of 38B; **208B generated** |
| agentic | 8% | 320 | 43 | 0.35 | 4.0 epochs of 43B; **150B generated**  · only 35% carries loss (rest = masked context) |
| long_context | 6% | 240 | 240 | 1.00 | 1.0 epochs of 240B |
| indic | 18% | 720 | 778 | 1.00 | 0.93 epochs of 778B |

## Indic lane - the four tiers

Indic demand = **720B** (18% of 4.0T).
| Tier | Share of Indic | Demand | Real supply | How it's met |
|------|---------------:|-------:|------------:|--------------|
| verified | 18% | 130 | 86 | 1.52 epochs of 86B |
| unverified | 14% | 101 | 24 | 4.0 epochs of 24B; **5B generated** |
| translated | 33% | 238 | 305 | 0.78 epochs of 305B |
| synthetic | 35% | 252 | 363 | 0.69 epochs of 363B |

**Honest headline:** organic Indic (verified+unverified) is ~110B; the 18% Indic lane (720B) is therefore **majority translated+synthetic** (~68% of the lane). Verified is only ~3.2% of the whole budget - it is the scarce, highest-value tier and we protect it in the anneal.

**Total tokens that must be generated across all lanes: ~358B (8.9% of budget)** - dominated by **agentic** and **reasoning**, where real supply barely exists (long-context is met at 1 epoch, but ~60% of that supply is itself synthetic).
