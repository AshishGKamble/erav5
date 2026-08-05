# Benchmark readiness (computed)

Which Assignment-3 targets the chosen mixture feeds, and how solid the ground is under each.
**These are not predicted scores** - a 4.85M CPU proxy cannot forecast MMLU-Pro. They are the
three checkable facts behind each target: how much gradient it actually gets, how much of that
could come from real data, and whether anything has tested it.

| Benchmark | A3 target | Effective gradient | Organic backing | Proxy evidence | Verdict |
|---|:---:|:---:|:---:|---|---|
| **MMLU-Pro** | ≥ 85 | 21.4% | 100% | web: responds (spread 0.49); math_stem: responds (spread 0.60) | On track |
| **AIME '26** | ≥ 89 | 9.6% | 60% | math_stem: responds (spread 0.60); reasoning: responds (spread 0.68) | Partly supply-limited |
| **LiveCodeBench-v6** | ≥ 80 | 20.0% | 100% | code: mixed (spread 0.64) | Fed, but lane is mixture-sensitive |
| **GPQA-Diamond** | ≥ 84 | 9.6% | 60% | math_stem: responds (spread 0.60); reasoning: responds (spread 0.68) | Partly supply-limited |
| **BBEH** | ≥ 74 | 15.0% | 30% | reasoning: responds (spread 0.68); web: responds (spread 0.49) | **Supply-limited** - rests on generated data |
| **tau2-bench** | ≥ 77 | 2.8% | 2% | agentic: untested | **Unevidenced** - proxy cannot reach this lane |
| **MMMLU** | ≥ 88 | 22.4% | 76% | indic: responds (spread 1.70); web: responds (spread 0.49) | Partly supply-limited |
| **MRCR-v2 (256K)** | ≥ 66 | 6.0% | 100% | long_ctx: untested | **Unevidenced** - proxy cannot reach this lane |
| **MILU / IndicGenBench** | lead | 18.0% | 61% | indic: responds (spread 1.70) | Partly supply-limited |

### How to read the columns

- **Effective gradient** = share x trainable fraction, summed over feeding lanes. Agentic is only 0.35 trainable (tool output is masked context), so its 8% token share is ~2.8% of real learning.
- **Organic backing** = `min(1, organic_supply x 4 epochs / demand)`. 100% means real data could cover the whole demand; 0% means every token behind that benchmark is generated or distilled.
- **Proxy evidence** = from 16 tiny-proxy runs. *responds* = loss falls as the lane's share rises by more than that lane's seed-noise floor; *mixed* = a real violation survives the floor; *untested* = the proxy cannot reach it at 128-token context with no tool execution.

### The honest bottom line

The targets standing on the thinnest ground are **tau2-bench** (2% organic backing), **BBEH** (30% organic backing), **AIME '26** (60% organic backing). Those are exactly the lanes the ledger flags as generation-heavy, so the risk is stated in two independent places rather than hidden. A reviewer should push hardest there - and the answer is not a bigger share, it is a better generation pipeline plus the 1B proxy that would actually measure these benchmarks instead of held-out loss.
