# Dataset roster - what actually fills each slot

Generated from `inventory.json`. `gen` marks a source that does not exist yet and must be
created; every such row is a promise the plan is making, not a dataset it can point at today.


## web  (MMLU, MMLU-Pro, GPQA(easy), common-sense)

| Dataset | Tokens (B) | Kind | Source |
|---|---:|:---:|---|
| DCLM-baseline | 4000 | organic | DCLM arXiv 2406.11794 (4T from a 240T pool) |
| FineWeb | 15000 | organic | FineWeb arXiv 2406.17557 (15T) |
| FineWeb-Edu (score>=3) | 1300 | organic | FineWeb-Edu (1.3T) |

## code  (LiveCodeBench, SWE-bench(+Live/Pro), Codeforces)

| Dataset | Tokens (B) | Kind | Source |
|---|---:|:---:|---|
| The Stack v2 (dedup, Software Heritage) | 900 | organic | Session inventory: ~600M files / ~900B tokens; StarCoder2 arXiv 2402.19173 |
| StarCoder2 train subset | 913 | organic | StarCoder2 (913B unique, overlaps Stack v2) |

## math_stem  (AIME, GPQA-Diamond, MMLU-Pro(STEM), FrontierMath)

| Dataset | Tokens (B) | Kind | Source |
|---|---:|:---:|---|
| OpenWebMath | 15 | organic | arXiv 2310.06786 (~14.7B) |
| Proof-Pile-2 | 55 | organic | Llemma arXiv 2310.10631 (ArXiv 29B + OWM 15B + AlgStack 11B) |
| peS2o + arXiv (papers) | 70 | organic | peS2o/arXiv full-text, S2ORC |
| NuminaMath / MetaMathQA (problems) | 3 | synthetic | NuminaMath ~1M problems; templated/augmented |

## reasoning_traces  (AIME, GPQA, MMLU-Pro, depth-control(low/med/high/ultra))

_Long chain-of-thought at graded depth. High/ultra depth is NOT freely available; must be distilled from a reasoning teacher._

| Dataset | Tokens (B) | Kind | Source |
|---|---:|:---:|---|
| OpenThoughts / OpenR1 (distilled CoT) | 8 | synthetic | OpenThoughts-114k -> 1M; R1-distill traces |
| AM-DeepSeek-Distilled-40M | 30 | synthetic | 3.34M prompts / 40M responses (mostly EN/ZH) |

## agentic  (tau2-bench, BFCL, Terminal-bench, WebArena, GAIA, BrowseComp)

_Tokens are huge but loss is only on assistant tokens; tool outputs/observations are masked context. Real supply is tiny -> this lane is generation-heavy (distill from Claude/Codex traces + our own cloud-code sessions + simulated tool environments)._

| Dataset | Tokens (B) | Kind | Source |
|---|---:|:---:|---|
| ToolBench | 0.08 | synthetic | Session inventory: 120k examples / 80M tokens; multi-tool over real REST APIs |
| xLAM / APIGen-60k | 0.2 | synthetic | arXiv 2409.03215 (60k verified function-calls) |
| Glaive-function-calling-v2 | 0.3 | synthetic | Glaive (~113k) |
| SWE-bench trajectories / terminal traces | 2 | organic | scarce; real GitHub patches + shell sessions |
| Distilled/self-collected agent traces (to generate) | 40 | GEN | GENERATED: Claude/Codex distill + cohort cloud-code sessions + simulated envs |

## long_context  (RULER, MRCR-v2, long-doc QA)

_Each SAMPLE must be long (4K->32K+), trained at length (not truncated). Organic long samples are scarce -> up-sample long docs + synthesize via multi-doc packing and long agent trajectories._

| Dataset | Tokens (B) | Kind | Source |
|---|---:|:---:|---|
| Books (Gutenberg/Books3-clean) | 30 | organic | long-form books |
| arXiv full papers + repo-level code | 60 | organic | naturally long documents |
| Synthetic long (multi-doc packing / long traces) | 150 | GEN | GENERATED: concatenate related docs; long agentic trajectories |

## indic  (MILU, IndicGenBench, MMMLU-Indic, BharatDrishti(ours, A3))

_12 Assignment-3 languages. Split across four tiers because most Indic mass is NOT organic. Full Indic conversational would need ~2T tokens (unavailable) -> repetition + translation + synthesis._

| Dataset | Tokens (B) | Kind | Source |
|---|---:|:---:|---|
| Sangraha-Verified | 64 | verified | IndicLLMSuite arXiv 2403.06350 (human sites + OCR + ASR) |
| IndicCorp v2 (cleaned crawl) | 21 | verified | ACL 2023.acl-long.693 |
| indic-align (our A4-cleaned) | 0.5 | verified | Assignment-4 pipeline output (MuRIL-counted; overlaps Sangraha family) |
| Sangraha-Unverified | 24 | unverified | perplexity-filtered crawl (CCNet-style) |
| IndicTrans2-translated (En->12 langs) | 300 | translated | arXiv 2305.16307; quality-capped, chrF++ gated |
| indic-align translated (Dolly/OASST) | 5 | translated | AI4Bharat indic-align |
| Sangraha-Synthetic | 163 | synthetic | WikiMedia MT to 14 langs + romanized transliteration |
| BhashaKritika + self-instruct (to generate) | 200 | GEN | GENERATED: newer synthetic pipelines |
