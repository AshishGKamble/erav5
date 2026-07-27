# ERA V5 Assignment 3 - India-First 40B Model: Design and Research Findings

> **Status: working research corpus (v0.2).** This file is our internal knowledge base and
> gets refined as research continues. It is NOT the final submission. The final Netlify
> report will be short and figure-driven (the assignment penalizes length); this document
> is deliberately long because it holds every sourced number and every design decision so
> the report can be assembled and cited from one place.
>
> **v0.2 re-baseline (2026-07-22).** The brief's "Gemma 4" is a REAL model: Google DeepMind
> released Gemma 4 on 2026-04-02 (arXiv 2607.02770, Apache 2.0), after our knowledge cutoff.
> v0.1 wrongly baselined against Gemma 3. This version re-baselines every capability decision
> against **Gemma 4 31B** (the flagship) and adds the reasoning + agentic training spine needed
> to match it. The India-first thesis and the tokenizer / fertility work survive and get stronger
> ("focus beats breadth"; Gemma 4 reuses Gemma 3's non-Indic-optimized 262K tokenizer).

---

## 0. The assignment (what is actually being asked)

Design (on paper, not train) a **40B-parameter LLM** that is:

1. as good as (or better than) **Gemma 4** on general, coding, math, and agentic work,
2. genuinely strong in **Indic languages**, and
3. **India-first**: views the world from an Indian (at least Asian) perspective.

We must decide and justify:

- **Data** for pre-training, post-training (SFT), and RL/alignment: what, from where, and why.
- **Cleaning**: how we clean the data for these objectives.
- **Evaluation**: how we test the model against the objectives (including "Indian perspective").
- **Fertility targets** per language and for coding / science / math / agentic tasks, and from
  those numbers the **tokenizer vocabulary size**.
- Package as a concise report on Netlify.

### Grading intent (from the session transcript)
- Reward **ingenuity**, punish repetition: "if you copy-paste what I've done you get zero."
- **Short, concrete, graph-driven.** "Longer submissions result in lower scores."
- The fertility argument is the spine: "you think you sent 500B tokens but the model saw
  only 37B words." A bad tokenizer with fertility 10 turns 100 words into 1000 tokens of noise.

### Steering decisions locked with the user
- **Positioning: India-first but fair** (prioritize Indian context/knowledge/framing and match
  Indian population values on contested topics, with an explicit fairness constraint - no
  anti-West bias; the transcript's "not a Genghis Khan" warning).
- **Realism: a realistic 2026 Indian-lab plan** (Sarvam / Krutrim / BharatGen scale), while also
  noting the future / idealized extension for each decision.

---

## 1. The thesis (one line the whole report defends)

> Match Gemma 4's frontier capability (reasoning + agentic) with a license-clean 2026 recipe,
> and win on the three things a global 140-language model structurally does not do: an
> **India-first perspective**, **measured Indic depth**, and **Indic tokenizer efficiency
> (focus beats breadth)**.

The fertility spine still holds underneath: a 40B model is compute-bound at ~800B tokens but
India-bound by real Indic words, so the tokenizer (the token/word exchange rate) is the
highest-leverage India decision. What changed at the Gemma 4 bar: frontier scores are
"thinking-model" numbers, so a reasoning + agentic training spine is now mandatory, not optional.

---

## 2. Research findings (sourced)

All numbers below are from the five research sweeps. Uncertainty is flagged inline. Full URLs
are consolidated in Section 7.

### 2.1 Indic languages and pre-training corpora

**Top Indian languages by speakers (2011 Census).** L1 = native, Total = L1+L2+L3.

| # | Language | L1 | Total | Family | Script |
|---|----------|----|-------|--------|--------|
| 1 | Hindi | 528M | 692M | Indo-Aryan | Devanagari |
| 2 | Bengali | 97M | 107M | Indo-Aryan | Eastern Nagari |
| 3 | Marathi | 83M | 99M | Indo-Aryan | Devanagari |
| 4 | Telugu | 81M | 95M | **Dravidian** | Telugu |
| 5 | Tamil | 69M | 77M | **Dravidian** | Tamil |
| 6 | Gujarati | 55M | 60M | Indo-Aryan | Gujarati |
| 7 | Urdu | 51M | 63M | Indo-Aryan | Perso-Arabic |
| 8 | Kannada | 44M | 59M | **Dravidian** | Kannada |
| 9 | Odia | 38M | 43M | Indo-Aryan | Odia |
| 10 | Malayalam | 35M | 36M | **Dravidian** | Malayalam |
| 11 | Punjabi | 33M | 36M | Indo-Aryan | Gurmukhi |
| 12 | Assamese | 15M | 24M | Indo-Aryan | Eastern Nagari |

8 of the top 12 are Indo-Aryan (Sanskrit-derived); the 4 major Dravidian languages are Telugu,
Tamil, Kannada, Malayalam. Maithili (~14M L1) is 13th. Hindi's 528M bundles dialects (Bhojpuri,
Awadhi, etc.).

**Open Indic pre-training corpora.**

- **Sangraha (AI4Bharat, 2024): 251B tokens / 22 languages.** Split: Verified 64B, Unverified
  24B, **Synthetic 163B** (English Wikimedia machine-translated to 14 languages + romanized
  transliterations). Per-language totals (millions of tokens): Hindi 34,544, Bengali 30,028,
  Tamil 17,360, Gujarati 17,179, Malayalam 16,409, Telugu 16,279, Urdu 14,402, Marathi 14,296,
  Kannada 14,255, Odia 12,539, Assamese 12,006, Punjabi 11,182.
- **IndicCorp v2 (2023): 20.9B tokens / 24 languages** (cleaned crawl, not synthetic).
- **Sarvam-1 training data (2024): ~2T Indic tokens** of a ~4T corpus (other ~2T English). 10
  languages, mostly synthetic / translation-augmented. (Flag: the "2T Indic" is largely
  synthetic, not organic.)
- **CulturaX (2023): 6.3T tokens / 167 languages** (cleaned mC4 + OSCAR). Indic per-language
  counts not confirmed.
- **Extremely low-resource (Sangraha totals, < 2B):** Kashmiri ~0.5B, Sindhi ~0.26B, Dogri
  ~0.06B, Bodo ~1.5B, Manipuri ~7.4B.

**Realistic organic ceiling.** Genuinely organic high-quality Indic text is only about
**50-100B unique tokens across all 22 languages combined** (Sangraha-Verified 64B + IndicCorp-v2
21B, overlapping). Only Hindi (~35B) and Bengali (~30B) are even mid-resource. Everything below
the top two is data-scarce; 7+ scheduled languages are near-zero. **Implication: any large Indic
budget must come from repetition + synthesis + English co-training, not organic text alone.**

**Synthetic / translation pipelines.** IndicTrans2 (open MT for all 22 scheduled languages, the
workhorse behind Sangraha Synthetic); romanization via IndicXlit-style transliteration;
BhashaKritika (2025, newer synthetic pipeline, figures unverified); back-translation.

**Code / math / science corpora.**

| Corpus | Size | Note |
|--------|------|------|
| The Stack v2 | ~900B tokens / 67.5 TB | StarCoder2 base, deduped |
| StarCoder2 train | 15B model on 913B unique tokens | curated subset of Stack v2 |
| Proof-Pile-2 (Llemma) | 55B tokens | ArXiv 29B + OpenWebMath 15B + AlgebraicStack 11B |
| OpenWebMath | ~14.7B tokens | math web text |
| FineWeb | 15T tokens | cleaned CommonCrawl English |
| FineWeb-Edu | 1.3T tokens (score >=3) | edu-filtered subset (variant sizes exist) |
| DCLM-baseline | 4T tokens | from a 240T-token CommonCrawl pool |

### 2.2 Tokenizer fertility and vocabulary size

**Fertility = tokens per word** (lower is better; English baseline ~1.1-1.4 for modern
tokenizers). Best cross-tokenizer table is IndicSuperTokenizer (IST, arXiv 2511.03237, Nov 2025):

| Tokenizer | Vocab | Hindi | Bengali | Tamil | Odia | English | Code |
|-----------|-------|-------|---------|-------|------|---------|------|
| IndicSuperTokenizer | 200K | **1.23** | **1.74** | **2.12** | - | 1.12 | 1.47 |
| Gemma-3 | 262K | 1.47 | - | 2.50 | - | 1.39 | ~1.5 |
| Sarvam-1 | 68K | 1.53 | - | 2.49 | - | 1.66 | - |
| GPT-OSS (o200k-family) | ~200K | 1.72 | - | - | 6.26 | 1.33 | ~1.5 |
| LLaMA-4 | 201K | 1.83 | - | - | **10.51** | 1.34 | 1.46 |

- **Sarvam-1 (68K vocab):** fertility 1.4-2.1 across 10 Indic languages, vs **4-8 tokens/word**
  for general multilingual models. About 80% fewer tokens than Llama-3.3 on Tamil.
- **22-official-language study (arXiv 2411.12240)** uses NSL (normalized sequence length), not
  raw fertility: SUTRA best in 14/22 languages, GPT-4o best in 5, Nanda (MBZUAI) best in 6,
  Project Indus competitive only in Devanagari.
- **Petrov et al. (NeurIPS 2023):** same text differs in token count by **up to 15x** across
  languages; the "tokenizer premium" (cost, latency, effective context) is a fairness issue.

**Key insight: allocation across scripts beats raw size.** IST at 200K beats Gemma-3 (262K) and
Llama-4 (201K); Llama-4 still explodes to 10.5 tokens/word on Odia despite a big vocab. "Careful
pre-tokenization outweighs naive vocabulary scaling."

**Code and math fertility.**
- Code fertility is fairly flat across modern tokenizers (~1.46-1.51). It needs **whitespace-run
  tokens** (indentation is structurally significant) and a **digit splitter**. StarCoder2
  tokenizer: BPE, vocab 49,152, single-token whitespace runs.
- Math has **no clean fertility number**; the dominant lever is **number tokenization**.
  Single-digit tokenization raises fertility on numbers but improves arithmetic. Otherwise math
  text tokenizes near English rates; LaTeX/symbols need coverage.

**Vocabulary parameter cost.** Embedding params = vocab_size x hidden_dim (per matrix). Tied
input/output embeddings pay once (Gemma-style); untied pay twice (Llama-style). At 40B,
hidden_dim 8192, tied:

| Vocab | Embed params | % of 40B (tied) | % of 40B (untied) |
|-------|-------------|-----------------|-------------------|
| 64K | 0.52B | 1.3% | 2.6% |
| 128K | 1.05B | 2.6% | 5.2% |
| 200K | 1.64B | 4.1% | 8.2% |
| 256K | 2.10B | 5.2% | 10.5% |

Other downsides of huge vocab: softmax/compute latency (worst at inference), rare-token
undertraining ("glitch"/Magikarp tokens, arXiv 2405.05417), memory.

**Vocab scaling law (Tao et al., NeurIPS 2024, arXiv 2407.13623).** Optimal vocab params scale
with non-vocab params as a power law, **exponent ~0.83** (vocab grows slower than the model but
still grows). Predicted compute-optimal vocab: 7B -> ~60-67K, 13B -> ~81-91K, 70B -> ~212-231K.
Interpolating, a **40B model -> ~160K** compute-optimal. Most current models are under-vocabed
(Llama-2-70B's 32K is ~7x too small). Gemma chose 256K/262K explicitly for multilingual breadth.

### 2.3 Gemma 4 - the real 2026 bar (re-baseline)

**Gemma 4** (Google DeepMind, released 2026-04-02, arXiv 2607.02770, Apache 2.0). Verified against
the arXiv report, the ai.google.dev model card, and HF `google/gemma-4-31B-it`. Five variants;
**flagship = 31B dense** (30.7B), 256K context, Arena-Text Elo 1451.

| Variant | Params | Type | Context |
|---------|--------|------|---------|
| E2B | 2.3B eff | dense (edge) | 128K |
| E4B | 4.5B eff | dense (edge) | 128K |
| 12B | 11.95B | dense, encoder-free multimodal | 256K |
| 26B-A4B | 26B / 3.8B active | MoE | 256K |
| **31B** | **30.7B** | **dense (flagship)** | 256K |

**Benchmark table (arXiv Table 5, thinking-mode; verbatim).** Google DROPPED the saturated classics
(MATH, GSM8K, HumanEval, MBPP, plain MMLU/BBH, SWE-bench, BFCL are ABSENT) and moved to harder
successors:

| Benchmark | 31B | 12B | (Gemma 3 27B) |
|-----------|-----|-----|---------------|
| MMLU-Pro | **85.2** | 77.2 | 67.6 |
| AIME 2026 (no tools) | **89.2** | 77.5 | 20.8 |
| LiveCodeBench v6 | **80.0** | 72.0 | 29.1 |
| Codeforces Elo | 2150 | 1659 | 110 |
| GPQA Diamond | **84.3** | 78.8 | 42.4 |
| BBEH (micro avg) | **74.4** | 53.0 | 19.3 |
| SciCode | 43.0 | 38.0 | 21.0 |
| IFEval | 98.9 | 97.2 | 90.4 |
| MMMLU (multilingual, blended) | **88.4** | 83.4 | 70.7 |
| MRCR v2 (128k, long-context) | 66.4 | 43.4 | 13.5 |
| Terminal Bench Hard | 36.0 | 18.0 | 4.0 |
| tau2 airline / retail / telecom | 75.0 / 86.4 / 69.3 | 75.0 / 77.6 / 54.4 | 39 / 6.6 / 3.1 |

**tau2-bench resolved:** no single aggregate; the authoritative agentic headline = **3-domain
average 76.9** (75.0+86.4+69.3). The "86" seen online is retail-only (cherry-picked).

**Tokenizer (the decisive fact for us): Gemma 4 reuses the SAME 262,144 SentencePiece tokenizer as
Gemma 3** - split digits, preserved whitespace, byte-level; NOT Indic-optimized, spread across
140+ languages. Google publishes **no fertility numbers** for any language.

**Multilingual / Indic = breadth-claimed, NOT depth-measured.** The only multilingual score is a
single **blended MMMLU 88.4** (no per-language breakdown). **Absent (confirmed): Global-MMLU,
MMLU-ProX, IndicGenBench, MILU, and any per-language Hindi / Tamil / Bengali / Telugu score.** The
"140+ languages / 35+ robust" claim has zero Indic-specific validation to cite against.

**Agentic is now native** (function-calling, structured JSON, `system` role) and strong (tau2 76.9
avg, up from Gemma 3's ~6.6). So the old "Gemma reports nothing on agentic" wedge is DEAD - agentic
is table-stakes now.

**Thinking mode:** per-request toggle (`<|think|>` control token), not a separate checkpoint; most
reported scores are thinking-on. **Training tokens: not disclosed.**

**Implications for our re-baseline:**
- Capability targets jump to Gemma 4 31B levels (MMLU-Pro ~85, AIME ~89, LiveCodeBench ~80,
  GPQA ~84, tau2 ~77). These are **thinking-model** numbers (see 2.6).
- The tokenizer wedge UPGRADES to "focus beats breadth": Gemma 4's 262K spread over 140 languages
  gives thin per-Indic-script allocation; our focused 200K for 12 languages targets lower Indic
  fertility on the same corpus.
- The Indic-depth wedge is now EVIDENCE-BASED: Gemma 4 proves nothing on Indic beyond a blended
  aggregate; we commit to and report IndicGenBench / MILU / per-language numbers it doesn't.

**Scaling / budget unchanged:** Chinchilla ~800B compute-optimal; over-train to ~15-20T for
inference efficiency.

### 2.4 Indic and India-perspective benchmarks (for evaluation)

**Indic capability benchmarks exist and are solid.**

| Benchmark | Tests | Languages | Size |
|-----------|-------|-----------|------|
| MILU (arXiv 2411.02538) | MCQ knowledge from Indian state/regional exams | 11 | ~85K MCQs |
| IndicGenBench (2404.16816) | summarization, MT, RC, QA | 29 | multi-way |
| IndicXTREME | 9 NLU tasks | 20 | 105 eval sets |
| IndicMMLU-Pro (2501.15747) | MMLU-Pro translated | 9 | mirrors MMLU-Pro |
| mmlu-indic (Sarvam) | MMLU translated | 10 | ~14K/lang |
| Global-MMLU (2412.03304) | MMLU + Culturally-Sensitive subset | 42 | CS subset 33,264 |
| BharatBench (Krutrim) | multimodal + Indian Cultural Context | 8 | 300 x 8 x 5 |
| Airavata eval / IndicQA / IndicSentiment / IndicXNLI | instruction, QA, sentiment, NLI | 11 | - |

**Cultural / values benchmarks include India only thinly.** INDICA (regional cultural
commonsense, 5 Indian regions, 1,630 QA; SOTA models only 13-21% fully correct), BLEnD (16+
cultures, ~52K QA), NormAd (75 countries, 2.6K situations), CulturalBench, GeoMLAMA.

**Perspective measurement methodology.** GlobalOpinionQA (Anthropic, arXiv 2306.16388):
aggregates World Values Survey + Pew (2,556 questions); metric = similarity of the model's
answer distribution to a country's human distribution. Findings: defaults skew US/Europe;
prompting a country shifts responses; translating the question does not reliably align to that
language's speakers.

**The gap (justifies building our own).** No benchmark measures **contested Indian-vs-Western
framing** (history, geopolitics, borders) or **UPSC / NCERT / Constitution-grounded** reasoning.
Geopolitical-bias methods exist but are country-generic. Government eval infrastructure exists
(AIKosh hosts MILU + mmlu-indic; IndiaAI Mission; BharatGen; Pariksha arena) but no India-
perspective test set.

### 2.5 Data cleaning and post-training / alignment sourcing

**Pre-training cleaning (reference recipes).**
- **FineWeb (arXiv 2406.17557):** trafilatura on WARC (not WET); fastText language ID keep
  English >=0.65; Gopher/MassiveText + selected C4 filters + custom line filters; **MinHash
  dedup per-dump, NOT global** (global dedup HURT: it stripped ~90% of old dumps and upsampled
  junk); PII regex for emails + IPs; FineWeb-Edu classifier (Llama-3-70B labels 0-5, keep >=3).
- **DCLM (2406.11794):** RefinedWeb heuristics -> Bloom-filter dedup -> fastText "good-vs-web"
  classifier (positive class = OpenHermes-2.5 + ELI5). This cheap classifier is now dominant.
- **CulturaX (2309.09400):** langID -> URL filter -> metric thresholds -> MinHash doc dedup.
- **Sangraha / Setu (2403.06350):** 4 Spark stages; Verified (human sites + OCR + ASR) /
  Unverified (perplexity filtering, CCNet-style) / Synthetic; **perplexity threshold at the 80th
  percentile set per language** (a global threshold fails across scripts).

**Indic-specific failure modes (what generic pipelines get wrong).**
1. **Language ID breaks on romanized + code-mixed text.** A FineWeb-style English>=0.65 gate
   silently deletes romanized Hindi / Hinglish. Fix: script-aware + transliteration-aware LID
   (retrained fastText + IndicXlit); treat code-mixing as a first-class class.
2. **NFC, never NFKC.** NFKC is lossy on Indic (collapses conjuncts / nukta / compatibility
   forms). Use NFC + ftfy mojibake repair + ZWJ/ZWNJ hygiene + confusable handling.
   (Corroborated first-hand by our Assignment-2 finding that NFKC breaks round-trip.)
3. **Per-language quality thresholds** (Sangraha's fix); FineWeb-Edu classifier is English-only
   and must be retrained for Indic.
4. **Per-source dedup, not global** (FineWeb lesson); MinHash 0.7-0.8 catches PTI/ANI syndication.
5. OCR noise in scanned Indic PDFs needs OCR-confidence heuristics.

**Code cleaning (StarCoder2, arXiv 2402.19173):** ScanCode license scan (permissive only),
MinHash 0.7 near-dedup, go-enry autogenerated/minified removal, StarPII model-based PII
redaction, benchmark decontamination. **Math (OpenWebMath, 2310.06786):** 5-stage funnel with a
MathScore classifier + KenLM perplexity + SimHash + LaTeX-preserving extraction.

**Post-training SFT.**
- **IndicInstruct / Airavata (2401.15006):** open English instruction sets (FLAN-v2, Dolly,
  OASST, LMSYS) **translated via IndicTrans2**, gated at **chrF++ >= 50** back-translation to
  drop bad translations; plus **native** wikiHow + Anudesh (native speakers write prompts). The
  team deliberately avoided proprietary models for data generation (licensing).
- Synthetic: Self-Instruct (2212.10560), Evol-Instruct / WizardLM (2304.12244).
- Agentic: xLAM / APIGen 60k verified function-calling (2409.03215), ToolBench, Glaive.

**RL / preference / alignment.**
- Generic preference: HH-RLHF, UltraFeedback (2310.01377), Nectar (182,954 prompts x 7
  responses, GPT-4 ranking), HelpSteer3 (multilingual). Optimizers: RLHF vs RLAIF vs **DPO**
  (2305.18290, cheapest for a 40B).
- **Value injection: Constitutional AI / RLAIF (2212.08073)** with an explicit "Indian
  constitution" (rights/duties, linguistic pluralism, secular-but-religiously-literate,
  non-Western-default framing) to scale a small native-annotated seed cheaply.
- **India preference data:** native-annotator platform (Pariksha, 2406.15053, ran 90k human +
  50k LLM evals across 10 languages - proof it scales); DOSA (2403.14651, 615 social artifacts
  across 19 Indian subcultures via participatory collection).
- **Pitfalls to name in the report:** translationese (mitigate with native data + chrF++ gate);
  annotator sourcing (urban "Indian English" != 22 languages; DOSA shows sharp cross-subculture
  variance, so stratify regionally); LLM-judge Western bias (calibrate RLAIF with native humans);
  value pluralism (do not flatten caste/religion/language diversity; use participatory sourcing).

### 2.6 Frontier-capability recipe (what it takes to MATCH Gemma 4)

Gemma 4 31B's scores (AIME 89, GPQA 84, LiveCodeBench 80) are **"thinking-model" numbers, not
instruct-model numbers** - reasoning mode alone is worth ~+50 on AIME (QwQ-32B 70.7 vs Phi-4 20.0).
Matching them requires a reasoning + agentic training spine our v0.1 design lacked.

**The stage stack to add (2025-26 open recipe):**
1. **Mid-training / annealing** - LR decayed to ~0 over ~50-500B tokens of the highest-quality
   code / math / reasoning + synthetic CoT. This "unlocks" later RL (OctoThinker: mid-training on
   reasoning data determines how much RLVR can gain). Standard in OLMo 2, Nemotron, Phi-4.
2. **Reasoning cold-start SFT** - thousands to ~800K long chain-of-thought traces. Cheapest path:
   **distill from a reasoning teacher.**
3. **RLVR (GRPO)** - RL with verifiable rewards: math = exact-match, code = unit-test execution.
   The main driver of AIME / LiveCodeBench (Qwen3 moved AIME'24 70->85 in ~170 RL steps). Expensive
   (~$0.5-1M at frontier). Caveat: RLVR mostly *sharpens* existing capability (pass@1), so it needs
   a capable base + reasoning SFT under it.
4. **Agentic SFT + multi-turn agentic RL** - tool sandboxes, rubric / checklist rewards, simulated
   users (Kimi K2: 3K real + 20K synthetic tools). Drives tau2-bench / function-calling.
5. **Thinking-mode fusion + general alignment** - fold non-thinking ability back with a toggle
   (Qwen3), then preference / RLHF + Constitutional-AI values + safety.

**Distillation (the cheap shortcut, license-critical):** distilling long-CoT traces from a reasoning
teacher beats from-scratch RL at ~1/10 the GPU-hours and even improves pass@64. Use **DeepSeek-R1
(MIT)** or **Qwen3 (Apache 2.0)** - both explicitly permit distillation and are stronger *dedicated*
reasoners. Gemma 4 is Apache 2.0 too, but R1 / Qwen3 keep our model independent, so they are the
teachers of choice.

**Data-mix implication:** frontier reasoning wants **code+math >=40%** in the reasoning phases,
which squeezes multilingual - so a code / math-rich base + a dedicated **in-language continued-
pretrain stage (~30% Indic)** to build depth without catastrophic forgetting. Verifiable-reward
gains transfer cross-lingually; chat / agent behavior in-language does not (needs in-language SFT).

**Reality check:** tau2 ~77 is at/above the current open frontier (Kimi K2 ~66); treat it as
aggressive - budget heavy agentic RL or a strong agentic teacher.

Sources: DeepSeek-R1 (Nature s41586-025-09422-z), Qwen3 (arXiv 2505.09388), GRPO / DeepSeekMath,
Kimi K2 (arXiv 2507.20534), OctoThinker (arXiv 2506.20512), SwallowCode (2505.02881), MathCoder2
(2410.08196), reasoning-cost (2505.18237), DistilQwen (2511.01354).

---

## 3. Integrated design decisions

### 3.1 Data: a staged curriculum (not one flat mix)

Budget ~15-20T tokens. At the Gemma 4 bar, frontier reasoning wants code+math >=40% in the
reasoning phases, which fights the India tilt - so we STAGE it instead of one flat mix:

**Phase A - Base pretrain (~11-14T):** a reasoning-capable, broadly multilingual base.

| Bucket | Share | Why |
|--------|-------|-----|
| English (Indian-weighted) | 35% | reasoning backbone; Indian-context English (NCERT, PIB, law) |
| Code | 20% | The Stack v2, rewritten (SwallowCode-style); code-with-tests for later RLVR |
| Math + science + synthetic CoT | 15% | OpenWebMath, proof-pile, MathCoder2-style, textbook-quality |
| Indic monolingual | 22% | Sangraha + IndicCorp organic (<=4 epochs) + IndicTrans2 synthetic |
| Cross-lingual bridges | 8% | parallel + transliteration + code-mixed; transfers reasoning into Indic |

**Phase B - Mid-training / annealing (~50-500B):** LR->0 on the highest-quality code / math /
reasoning + synthetic CoT; gates how much later RLVR can gain (see 2.6).

**Phase C - Indic continued-pretrain (~0.5-1T):** ~30% in-language Indic to build depth without
catastrophic forgetting of the reasoning base.

Indic-touching = ~**30%** overall (22% + 8% bridges), ~100x its natural web share - the India-first
bet, protected by a dedicated in-language stage rather than diluted across one flat mix.

**Honest constraint (grading rewards candor).** Organic Indic is ~50-100B unique tokens. Repeated
~4x (the data-constrained-scaling limit) that is ~0.3T, so **~90% of the Indic slice is synthetic /
translated** - and its quality caps the model's Indic ceiling. Stated plainly. The future / idealized
plan shifts the ratio toward organic as native Indic web and synthesis quality grow.

**Post-training then adds the reasoning + agentic spine (see 3.5).**

### 3.2 Cleaning pipeline (tuned to the objective)

Ordered pipeline, standard stack plus the Indic-specific fixes:

1. **Extraction:** trafilatura on WARC.
2. **Language ID:** script-aware + romanized/code-mixed-aware (retrained fastText + IndicXlit).
   NOT a naive English>=0.65 gate.
3. **Normalization:** NFC (never NFKC), ftfy, ZWJ/ZWNJ hygiene, confusable handling.
4. **Quality:** per-language perplexity percentile thresholds + retrained Indic educational
   classifier + DCLM "good-vs-web" fastText.
5. **Dedup:** MinHash per-source/per-dump (never global), 0.7-0.8 for news-wire.
6. **Code:** ScanCode permissive-only, StarPII redaction, autogen/minified removal, MinHash 0.7,
   plus LLM **rewriting** for quality (SwallowCode-style: syntax-validate -> lint -> rewrite); keep
   code-with-tests for later RLVR.
7. **Math:** MathScore + LaTeX-preserving extraction; MathCoder2-style math->code for reasoning.
8. **India-relevance:** a classifier used as a **sampler weight** to upsample Indian-context
   English (not a filter).
9. **Throughout:** PII (emails/IPs + StarPII), toxicity, and **aggressive decontamination against
   the Gemma 4 eval suite** (AIME 2026, LiveCodeBench v6, GPQA, BBEH, tau2, MMLU-Pro) - contamination
   on these newer sets is the biggest scoring risk.

### 3.3 Fertility targets and tokenizer size (the heart)

**Fertility targets (tokens/word, lower better).**

| Domain | Target | Anchor / reason |
|--------|--------|-----------------|
| English (Indian) | 1.10-1.15 | IST 1.12 |
| Hindi, Marathi | 1.2-1.4 | IST Hindi 1.23 |
| Bengali, Gujarati, Punjabi, Odia, Assamese | 1.3-1.6 | IST Bengali 1.74, beatable via allocation |
| Tamil, Telugu, Kannada, Malayalam | 1.6-2.0 | IST Tamil 2.12, Sarvam 2.49; **agglutinative floor, cannot reach 1.0** |
| Urdu | 1.4-1.7 | Perso-Arabic ambiguity |
| Code | ~1.45 (~0.3 tok/char) | IST 1.47; whitespace-run + digit-split tokens |
| Science | ~1.2-1.3 | near-English + notation; reuses math / Greek / SI-unit tokens |
| Math (numbers) | deliberately ~1 token/digit | **fertility traded for arithmetic accuracy** |
| Agentic / tool | structural overhead, minimized | reserved tool-call/JSON special tokens |

Three ingenuity points: (a) refuse a flat 1.0 target and justify per-morphology; (b) reframe the
admin's trick question - "fertility for math/agentic" is not a language number, it is the
**digit-tokenization tradeoff** and **structural-token overhead**; (c) Dravidian honesty.

**Vocabulary size - derived three ways, converging on ~200K.**
- **Bottom-up:** ~50K (Latin + code + math + shared) + ~12K subwords x 12 Indic languages
  (~144K) + ~5K math/symbol + reserved agentic tokens + 256 byte-fallback ~ **196K**.
- **Scaling law (Tao 2024):** ~160K compute-optimal for 40B, plus a multilingual-breadth premium.
- **Empirical:** IST hits Hindi 1.23 / Tamil 2.12 at exactly 200K, beating Gemma-3 (262K) and
  Llama-4 (201K).

**Decision: 200K vocab for the realistic 12-language build; 256K for the idealized all-22-language
future.** The wedge sharpens against Gemma 4: **focus beats breadth.** Gemma 4 reuses a 262K
SentencePiece spread over 140+ languages (thin per-Indic-script allocation); our focused 200K for
12 languages targets materially lower Indic fertility on the same corpus. Cost at hidden-8192
**tied** embeddings ~ 1.64B params (~4%); recommend tying to halve cost. Inherit our Assignment-2
**byte-fallback** for guaranteed round-trip faithfulness. (Llama-4 also shows big != good: 201K
vocab, still 10.5 tokens/word on Odia.)

### 3.4 Evaluation

**Capability floor (match/beat Gemma 4 31B), using Gemma 4's own harder suite:**

| Axis | Gemma 4 31B | Our 40B target |
|------|-------------|----------------|
| MMLU-Pro | 85.2 | >=85 |
| AIME 2026 | 89.2 | >=89 |
| LiveCodeBench v6 | 80.0 | >=80 |
| GPQA Diamond | 84.3 | >=84 |
| BBEH | 74.4 | >=74 |
| tau2-bench (3-domain avg) | 76.9 | >=77 (aggressive) |
| MMMLU (blended multiling.) | 88.4 | >=88 |
| MRCR v2 (long-context 256K) | 66.4 | >=66 (match 256K context) |
| IndicGenBench / MILU / per-language | **not published** | **LEAD - report what Gemma doesn't** |

**Indic capability (our wedge - Gemma 4 publishes only a blended MMMLU):** MILU, IndicGenBench,
IndicXTREME, IndicMMLU-Pro, Global-MMLU, BharatBench - reported **per language**. Report
**per-language fertility as an eval metric** too (closes the loop with 3.3).

**India-perspective - build "BharatDrishti" (our own), because none exists.** Construct from UPSC
PYQs + NCERT (classes 6-12) + Constitution / Supreme Court judgments + timestamped Indian press.
Two scores: (a) India-knowledge accuracy; (b) framing alignment to the Indian population
distribution via the GlobalOpinionQA method with an **India WVS / CSDS-Lokniti reference**. Plus a
**fairness counter-metric** (no anti-West skew) - this makes "India-first but fair" measurable and
answers a documented benchmark gap ("may become a paper").

**Note on the benchmark shift:** Gemma 4 itself dropped the saturated classics (MATH, GSM8K,
HumanEval, MBPP, plain MMLU/BBH) for harder successors (AIME 2026, LiveCodeBench v6, GPQA, BBEH,
MMLU-Pro). We follow suit and decontaminate against all of them during cleaning (3.2). Most Gemma 4
scores are thinking-mode-on, so we report thinking-on/off separately.

### 3.5 Post-training: the reasoning + agentic spine (new at the Gemma 4 bar)

Matching Gemma 4 needs stages v0.1 under-weighted. Full sequence:
1. **Mid-training / annealing** (see 2.6) on top-quality code / math / reasoning + synthetic CoT.
2. **Reasoning cold-start SFT** - long chain-of-thought, **distilled from DeepSeek-R1 (MIT) or
   Qwen3 (Apache)** - license-clean, stronger dedicated reasoners than Gemma, keeps us independent.
3. **RLVR (GRPO)** - verifiable rewards: math exact-match, code unit-tests. Drives AIME /
   LiveCodeBench. Reserve heavy from-scratch RLVR only to exceed the teacher.
4. **Agentic SFT + multi-turn agentic RL** - tool sandboxes with rubric / checklist rewards and
   simulated users; re-skinned to **Indian tools (IRCTC / UPI / DigiLocker)** so agentic skill is
   grounded in Indian workflows. Drives tau2 (target ~77, flagged aggressive).
5. **Thinking-mode fusion** - a `<|think|>`-style per-request toggle (like Gemma 4).
6. **Alignment** - Constitutional AI with an Indian value set + native-annotator preferences
   (Pariksha-style) + safety. This is where "India-first but fair" is enforced (see 3.4).

In-language note: verifiable-reward (math / code) gains transfer cross-lingually, but chat and agent
behavior in-language do NOT - so we keep dedicated Indic SFT + Indic agentic data.

---

## 4. The ingenuity spine (foreground for the grade)

1. **Match Gemma 4, win on India** - concede Gemma 4 is frontier (incl. agentic) and pivot the
   differentiation to three durable, evidence-based wedges: India-first perspective, measured Indic
   depth (Gemma publishes none), and Indic tokenizer efficiency (focus beats breadth).
2. **"Focus beats breadth" tokenizer** - Gemma 4's 262K over 140 languages vs our focused 200K over
   12; a vocab size *derived* three ways, not guessed.
3. Fertility reframed beyond Indic -> digit-tokenization tradeoff + agentic structural overhead;
   science added.
4. A **license-clean reasoning recipe** - distill from R1 / Qwen3, not Gemma; a legally-aware call.
5. A **new India-perspective benchmark** (BharatDrishti) filling a real, documented gap, with
   "India-first but fair" as a **measured** fairness constraint.
6. Honest data-scarcity math (~90% synthetic Indic) + first-hand Assignment-2 lessons (NFKC,
   byte-fallback).

---

## 5. Open knobs (defaults in bold; change any)

- Languages: **12 (top-10 by population + Punjabi + Assamese)**. Note: Urdu (~#7) is already in the
  top 10; the two additions beyond it are Punjabi (#11) and Assamese (#12). Tutor's anchor was
  "top 10 by population" with the selection left to us.
- Token budget: **15T realistic** / 20T idealized.
- Vocab: **200K realistic** / 256K future.
- Embeddings: **tied** (halves the vocab param cost).
- Report tone: **decision-brief with figures** (not essay).

---

## 6. Proposed final deliverable (Netlify report)

Short and figure-driven (~8 pages), because length is penalized:

- Fig: the "500B tokens -> effective words" fertility illustration (hero).
- Fig: benchmark-target bars vs **Gemma 4 31B** (MMLU-Pro, AIME, LiveCodeBench, GPQA, BBEH, tau2,
  MMMLU) + the Indic row Gemma leaves blank = our lead.
- Fig: "focus beats breadth" - Gemma 4 262K over 140 languages vs our 200K over 12 (Indic fertility).
- Fig: staged data curriculum (base -> anneal -> Indic continued-pretrain) + Indic organic-vs-synthetic.
- Fig: the post-training spine (mid-train -> reasoning SFT/distill -> RLVR -> agentic RL -> fusion -> align).
- Fig: vocab parameter-cost curve (200K = 4.1%).
- Fig: cleaning-pipeline flow (Indic fixes highlighted).
- Fig: BharatDrishti construction diagram.
- Master table: per-language fertility targets + vocab allocation.

---

## 7. Sources (consolidated)

**Languages / corpora:** List of languages by native speakers in India (2011 Census, Wikipedia);
Sangraha / IndicLLMSuite arXiv 2403.06350, HF ai4bharat/sangraha; IndicCorp v2 ACL 2023.acl-long.693;
Sarvam-1 sarvam.ai/blogs/sarvam-1, HF sarvamai/sarvam-1; CulturaX arXiv 2309.09400; IndicTrans2
arXiv 2305.16307; BhashaKritika arXiv 2511.10338; The Stack v2 / StarCoder2 arXiv 2402.19173;
Proof-Pile-2 / Llemma arXiv 2310.10631; OpenWebMath arXiv 2310.06786 / 2310.10631; FineWeb arXiv
2406.17557; DCLM arXiv 2406.11794.

**Tokenizer / fertility:** IndicSuperTokenizer arXiv 2511.03237; 22-language NSL study arXiv
2411.12240; Sarvam-1 (as above); Krutrim arXiv 2502.09642 / 2407.12481; Petrov et al. arXiv
2305.15425; Scaling Laws with Vocabulary (Tao 2024) arXiv 2407.13623; StarCoder2 arXiv 2402.19173;
Magikarp undertrained tokens arXiv 2405.05417; Gemma-2 report (DeepMind), Gemma-3 arXiv 2503.19786.

**Benchmarks (capability):** **Gemma 4 arXiv 2607.02770, ai.google.dev/gemma/docs/core/model_card_4,
HF google/gemma-4-31B-it (Apache 2.0, released 2026-04-02);** Gemma-3 arXiv 2503.19786; Qwen2.5
arXiv 2412.15115; Llama-3 Herd arXiv 2407.21783; Chinchilla / Beyond-Chinchilla arXiv 2401.00448;
tau2-bench (Sierra).

**Frontier recipe (reasoning + agentic):** DeepSeek-R1 (Nature s41586-025-09422-z); Qwen3 arXiv
2505.09388; GRPO / DeepSeekMath; Kimi K2 arXiv 2507.20534; OctoThinker arXiv 2506.20512; SwallowCode
arXiv 2505.02881; MathCoder2 arXiv 2410.08196; reasoning-cost arXiv 2505.18237; DistilQwen arXiv
2511.01354.

**Benchmarks (Indic / perspective):** MILU arXiv 2411.02538; IndicGenBench arXiv 2404.16816;
IndicXTREME (AI4Bharat IndicBERT); IndicMMLU-Pro arXiv 2501.15747; Global-MMLU arXiv 2412.03304;
BharatBench (Krutrim); INDICA arXiv 2601.15550; BLEnD arXiv 2406.09948; NormAd arXiv 2404.12464;
CulturalBench arXiv 2410.02677; GlobalOpinionQA arXiv 2306.16388, HF Anthropic/llm_global_opinions;
AIKosh (aikosh.indiaai.gov.in).

**Cleaning / alignment:** FineWeb 2406.17557; DCLM 2406.11794; CulturaX 2309.09400; Sangraha/Setu
2403.06350 + github AI4Bharat/setu; StarCoder2 2402.19173; OpenWebMath 2310.06786; IndicInstruct /
Airavata 2401.15006; Self-Instruct 2212.10560; Evol-Instruct 2304.12244; xLAM / APIGen 2409.03215;
UltraFeedback 2310.01377; Nectar (HF berkeley-nest/Nectar); Constitutional AI 2212.08073; DPO
2305.18290; Pariksha 2406.15053; DOSA 2403.14651.

---

## 8. Uncertainty flags (verify before publishing)

- Some arXiv IDs were cited from memory by the research sweeps and should be re-verified:
  RefinedWeb 2306.01116, Dolma 2402.00159, Llemma 2310.10631, ToolLLM/ToolBench 2307.16789,
  Self-Instruct 2212.10560, DPO 2305.18290, HH-RLHF 2204.05862.
- Sarvam "2T Indic" is largely synthetic, not organic.
- CulturaX per-language Indic counts unconfirmed.
- FineWeb-Edu size varies by variant (1.3T score>=3 vs larger mixes).
- The 22-language study reports NSL, not tokens/word (not directly comparable to fertility).
- Krutrim / Tamil-Llama / OpenHathi per-language fertility not independently confirmed.
- Math-specific fertility numbers did not surface cleanly (digit-splitting is the documented lever).
- Tao exponent ~0.83 is the paper's fit; treat as approximate.
- Some agentic-coding frontier percentages are directional; cite live leaderboards at report time.
- INDICA (2601.15550) and Indi-RomCoM (2606.30790) arXiv IDs look future-dated in the source and
  should be double-checked before citing.
- **Gemma 4 correction:** v0.1 wrongly treated "Gemma 4" as unreleased (our first research sweep
  filtered it out as SEO noise). It is real (2026-04-02). Gemma 4 does NOT disclose training tokens;
  most scores are thinking-mode-on; tau2 has no single aggregate (we use the 76.9 3-domain avg).
- **tau2 ~77 target is aggressive** - at/above the current open frontier (Kimi K2 ~66); it depends
  on heavy agentic RL or a strong agentic teacher.
- Our 40B "match Gemma 4 31B" targets assume a successful reasoning + agentic spine (2.6 / 3.5);
  they are design goals, not guarantees. AIME'26-era numbers are new and lightly cross-checked.
- Gemma 4 is Apache 2.0 (distillable), but we still prefer R1 / Qwen3 as reasoning teachers.
