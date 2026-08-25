# Assignment 6 - Training Data Execution System: the plan

_Written before any code, in the Assignment-4 tradition. This is the design we agreed; the README
will be the defence of what we actually built._

The brief asks for a small but complete **Training Data Execution System**: documents → tokenized
shards → manifests → mixture schedule → packing → batches → training → consumption ledger →
learning ledger → checkpoint → crash → resume → replay → audit.

---

## 0. What this assignment actually rewards

It reads like a data assignment. It is a **systems-determinism assignment**. Nothing is graded on
whether the model learns; 750 of the 1000 points sit in five blocks that each reward *proving an
invariant held*, and grading Step 3 explicitly inspects the code to confirm behaviour "was not
simulated or hardcoded".

So the design principle for the whole build:

> **Every claim in `evidence.json` must be recomputed from an artefact the run itself produced,
> never from a variable the run was holding at the time.**

Throughput is derived from the ledger, not from a live counter. Mixture compliance is derived from
consumption records, not from the scheduler's intentions. If we cannot reconstruct it after the
fact, we do not claim it.

---

## 1. The architectural spine: a deterministic, reconstructible batch stream

Everything else follows from one decision.

```
batch(step) = f(run_seed, plan_hash, manifest_hash, ledger[0:step])
```

The batch at any step is a pure function of the frozen inputs and the consumption history. There is
**no stateful sampler whose internal RNG position must be restored**. Consequences:

- **Resume** = reconstruct scheduler state from the ledger prefix, continue at `step+1`. Skipping or
  repeating a batch is not a bug we test for; it is structurally unavailable.
- **Replay** = recompute any interval and compare hashes to the ledger.
- **Fork** = same function, different branch config after step K. Prefix identity is guaranteed by
  construction, not by copying.

### 1.1 The honest caveat, and how we handle it

The mixture scheduler is deficit-based (§5), so `batch(step)` depends on the *whole* history, not
just on `step`. It is therefore **O(step) to reconstruct, not O(1) seekable**. We are not going to
hide that behind a claim of pure indexability.

The resolution: the checkpoint stores the scheduler state **and its hash**, and on resume we
*recompute* the state from the ledger prefix and assert it matches. Integer arithmetic over a few
hundred records is microseconds, so we get O(1) trust with O(n) verification. That yields an extra
provable event: `[PASS] scheduler_state_reconstructed`.

### 1.2 What is hashed, and what is not

**Hashed:** token ids, loss mask, position ids, segment ids, provenance spans, shard bytes,
tokenizer file, plan file, manifest.

**Never hashed:** loss values, timings, throughput. Floating-point results vary with BLAS backend
and thread count; putting them in a hash would make a correct system fail its own replay check on
someone else's machine. Loss is *logged and compared with a tolerance*, never used as an identity.

---

## 2. The proof strategy for crash/resume - a reference run

The brief says resume must "prove that the next batch is exactly the expected batch". A checkpoint
that predicts its own next batch and then matches it is self-fulfilling, and a sharp reader will say
so.

Instead the demo runs the pipeline **twice**:

| Run | What it does | Produces |
|---|---|---|
| **A - reference** | N steps, uninterrupted | `ledgers/consumption.A.jsonl` |
| **B - crash** | same config, hard crash at step K, resume from checkpoint | `ledgers/consumption.B.jsonl` |

Then assert **A ≡ B**, record for record, on batch id, token spans and batch hash. The oracle is
external to the mechanism being tested. Plus two contiguity assertions on B: no batch id appears
twice, and none is missing.

We additionally run a **negative control**: resume from B's checkpoint with a tampered `plan.json`.
The run must refuse to start, because the checkpoint records the plan hash. That produces
`[PASS] resume_rejects_modified_plan` - evidence that the guard is real and not decorative.

---

## 3. Frozen inputs, and why they are vendored

`assignment-6/` is fully self-contained. It vendors its inputs rather than reaching into sibling
folders, for a reason that is not merely tidiness: **no corpus is committed anywhere in this repo**
(`assignment-4/data/` and `assignment-5/proxy/data/` are both gitignored). A demo that reads a
sibling's gitignored directory fails on a fresh clone, which is exactly what the grader does.

A vendored input carrying a recorded `sha256` and a provenance note *is* the immutability the
assignment asks for. A cross-folder reference into a mutable directory would undercut the claim.

```
frozen/
  tokenizer.json      <- assignment-2, 10,000 vocab, 9,444 merges, byte_fallback
  plan.json           <- assignment-5, corrected (see §11)
  corpus/*.jsonl      <- seven lanes, fetched once at build time
  SOURCES.json        <- per-lane source, licence, sha256, fetch date
```

`fetch_corpus.py` is a **build-time tool we run once and commit the output of**. `run_demo.py`
never touches the network; it verifies hashes against `SOURCES.json` and aborts on mismatch.

### 3.1 The tokenizer, and a weakness we will report rather than hide

We reuse Assignment 2's tokenizer. It is a genuinely frozen artefact - frozen months ago for another
purpose, not minted by us for this demo - which makes it better evidence than a fresh one.

It was trained on four Wikipedia articles and **has never seen code**, so the code lane will fall
back to bytes and show poor fertility. We will measure per-lane fertility and publish it in the
manifest as a finding. Bad fertility does not threaten anything this assignment grades (mixture
accounting is in tokens either way), and reporting it is cheaper than pretending.

---

## 4. Corpus sources and licences

We are moving from Assignment 5's posture ("referenced, never redistributed") to committing bounded
slices, so every source needed a licence check. `NOTICE.md` must be updated to record this.

| Lane | Source | Licence | Verdict |
|---|---|---|---|
| web | `Salesforce/wikitext` (103-raw-v1) | CC BY-SA 3.0/4.0 | **OK** - share-alike, attribute. A2's Wikipedia extracts already set this precedent |
| code | `codeparrot/codeparrot-clean-valid` | **per-file, mixed** | **OK only when filtered** - see §4.1 |
| math | `open-web-math/open-web-math` | ODC-By 1.0 | **OK** with attribution; CommonCrawl ToU noted |
| reasoning | `openai/gsm8k` | MIT | **Clean** |
| indic | `ai4bharat/indic-align` (Anudesh, WikiHow, Dolly_T) | CC BY-4.0 | **Clean**, attribute |
| agentic | `glaiveai/glaive-function-calling-v2` | Apache-2.0, ungated | **Clean** |
| long_ctx | Project Gutenberg | US public domain | **Cleanest** |

**Rejected candidates, and why** - both were named in A5's inventory, so the reasons matter:

- **`Salesforce/xlam-function-calling-60k`** - CC BY-4.0, but **gated**: it requires logging in and
  accepting conditions. Redistributing content from behind an access gate is poor form regardless of
  the licence text, and it would force an authenticated fetch. Replaced by Glaive.
- **ToolBench (OpenBMB)** - currently Apache-2.0, but 2023 archives of the same repo show
  CC BY-NC-4.0. That licence-change ambiguity is not worth inheriting for a lane where the graded
  property is structural. Replaced by Glaive.
- **`bigcode/the-stack-smol`** - mixed licences with no convenient per-row filter field.

### 4.1 The code lane was a genuine blocker

`codeparrot-clean-valid` carries GPL-2.0 and AGPL-3.0 files. Committing those into an MIT repository
is a real licence conflict, not a technicality.

It also carries a per-row `license` field (verified: `mit`, `apache-2.0`, `bsd-3-clause`,
`gpl-2.0`, ...). So we **filter at fetch time to permissive licences only**
(`mit`, `apache-2.0`, `bsd-2-clause`, `bsd-3-clause`, `isc`) and record each file's licence and
repo in the shard manifest.

This is the Assignment-4 pattern reused - stamp provenance, then filter on it - and it does double
duty, because the assignment already wants manifests carrying licence records.

### 4.2 What the fetch measured, and two numbers that came back wrong

The corpus is built (23 MB, 8,737 documents, seven lanes). Two planning assumptions did not survive
contact with the data, and both are recorded here rather than quietly absorbed.

**Two lanes' trainable fractions diverge from `plan.json`, and both divergences come from making
the mask actually correct instead of assuming it.** Measured at token level after sharding - tokens,
not characters, are what carry loss, so token level is the number that matters:

| Lane | `plan.json` | Measured | Why |
|---|---:|---:|---|
| agentic | 0.35 | **0.507** | Glaive is not the shape A5 sized |
| reasoning | 1.00 | **0.575** | the GSM8K prompt is masked; only the answer is a target |

*Agentic.* Across 922 Glaive traces, assistant spans are 59.0% of characters but 50.7% of tokens -
the system prompt is dense JSON schema that tokenizes heavily, so it costs more tokens than its
character share suggests. Either way the lane is roughly half trainable, not a third. This does not
refute A5; it says Glaive is not the shape A5 sized. A5's 0.35 assumed SWE-bench and terminal traces,
where file dumps and stack traces dominate the transcript. Glaive is function-calling *chat*, where
the assistant does most of the talking and tool output is only 2.5% of characters.

*Reasoning.* `plan.json` marks this lane fully trainable, which is right for a reasoning **trace**,
where the chain of thought is the target. GSM8K is a question and an answer, and training the model
to produce the question teaches it nothing, so the prompt is masked - leaving 0.575. A plan written
about traces met a corpus made of question-answer pairs.

Both are reported as measured numbers. The masking **mechanism** is what this assignment grades, and
it works exactly as specified; the effective-gradient arithmetic downstream uses the measured
fractions, not the planned ones, because using the planned figures would mean claiming a gradient the
corpus does not deliver.

**The Indic lane needed two guards that the plan did not anticipate.** Taken naively in stream order,
the lane filled entirely from Anudesh and never reached the translated subsets - discarding the
provenance tier A5 section 8.2 depended on. Worse, it came out **95.1% Latin characters**, with only
10.9% of documents majority Indic script, because Anudesh is overwhelmingly English. An Indic lane
made of English would silently defeat its own demonstration: the protected floor exists to stop an
English-centric OPUS scorer discarding Indic, and English is exactly what such a scorer keeps.

Both are fixed in `fetch_corpus.py`: a per-subset byte budget (30/35/35 across Anudesh, WikiHow,
Dolly_T) and a script gate at 35% Indic letters. After the fix the lane is **99.2% Indic script**,
99.9% of documents are majority Indic, both tiers are present (1,808 native / 2,467 translated), and
the twelve languages are balanced to within two documents of each other.

A third, smaller correction: difficulty banding first used a flat 2,500-character threshold, which
put **93% of the lane in a single band** and would have made curriculum gating a no-op. Thresholds
now come from the measured distribution (discussion median ~420 chars, instructional ~3,000), giving
B1 29.1% / B2 41.4% / B3 26.6% / B4 2.9%.

**Determinism is confirmed empirically.** An independent refetch reproduced all 4,275 Indic documents
in identical order with identical content-addressed ids.

---

## 5. Mixture schedule, floors and OPUS

Input is A5's `plan.json`: seven lanes (web 29 / code 20 / math_stem 10 / reasoning 9 / agentic 8 /
long_ctx 6 / indic 18), five stages weighted 8/45/25/19/3, `indic_batch_pct: 14`,
`agentic: never OPUS-trimmed`, `opus_keep_frac: 0.5`.

**Scheduling is deficit-based, not i.i.d.** Independent sampling will not hit those shares over a
few hundred batches; actual shares wobble and we would fail our own compliance check. Each step
computes `deficit = target_share × total_consumed − consumed_lane` and fills slots by largest
deficit, subject to the Indic floor as a hard constraint on **every** batch.

### 5.1 OPUS has four decisions, and all four are exercised

| Decision | Trigger |
|---|---|
| `accept` | score ≥ accept threshold |
| `reject` | score < reject threshold |
| `defer` | between the two - requeued and re-scored after the proxy refreshes |
| `protected_override` | lane ∈ {indic, agentic} and the decision would have been `reject` |

The scorer reads **only the first 512 tokens** of a candidate, faithful to A5 §5.1 - which is
precisely *why* Indic and agentic need protecting, so the demo reproduces the failure the floor
exists to prevent rather than asserting it.

**The scorer is a frozen proxy snapshot, not live model state.** This matters for replay: if OPUS
depended on the current weights, reproducing a decision would require bit-identical training. Instead
each decision records the `proxy_snapshot_hash` it was made under, and replay re-scores against that
recorded snapshot. The snapshot refreshes on a schedule mapped from `opus_rerun_every_B: 2`.

Every decision lands in `ledgers/opus.jsonl` with candidate id, lane, score, threshold, decision,
snapshot hash and reason. The audit then proves: no protected-lane document was ever dropped, the
unprotected keep fraction lands near 0.5, and every override is attributable to a lane.

### 5.1a What OPUS actually does to the protected lanes - measured

The floor is not a precaution taken on faith. Scoring 120 candidates per lane against a snapshot
built from web, code and math - A5's "English- and code-heavy proxy", with Indic and agentic
deliberately absent from it:

| Lane | Mean score | Accept | Defer | Reject | Overridden |
|---|---:|---:|---:|---:|---:|
| web | 0.7530 | 28 | 0 | 0 | - |
| agentic | 0.6635 | 98 | 0 | 0 | **22** |
| code | 0.6556 | 17 | 1 | 2 | - |
| long_ctx | 0.6158 | 8 | 1 | 0 | - |
| math | 0.5592 | 36 | 26 | 6 | - |
| reasoning | 0.4202 | 7 | 45 | 68 | - |
| **indic** | **0.0994** | 0 | 0 | 0 | **120** |

**Indic scores 0.0994 where web scores 0.7530, and without the override 100% of the lane would have
been dropped or deferred.** Agentic loses 18%. That is A5 section 5.1's argument reproduced as a
measurement instead of an assertion: an English-and-code proxy reading only the first 512 tokens
does not merely under-rate Indic, it annihilates it. The demo shows the failure happening and the
override catching it.

Snapshot drift is real, not decorative: successive generations produce different hashes and
different scores, so a decision made under one snapshot cannot be silently reproduced under another.

**Two consequences for the batch builder.** First, OPUS decides *which document fills a slot*, not
*whether the slot is filled* - a rejected candidate is replaced by another draw from the same lane,
so filtering never distorts the mixture. Getting this backwards would let OPUS quietly starve the
reasoning lane, which it rejects 57% of. Second, thresholds must be calibrated on a sample drawn in
the same lane proportions the run will actually use; calibrating on a flat sample gave a realised
keep fraction of 0.39 against the 0.50 `plan.json` specifies.

### 5.1b Three scheduler bugs the measurements caught

Compliance went from unusable to 1.79 points through three fixes, each found by measuring rather
than reasoning. They are recorded because each is a trap the next person will hit.

| Fix | Before | After |
|---|---:|---:|
| Deficits made stage-local, not cumulative | 9.45 pts | 0.75 pts |
| OPUS exhaustion fallback (slot always fills) | 29.40 pts | 4.68 pts |
| Indic floor measured in tokens, not slots | 4.68 pts | **1.79 pts** |

**Stage-local deficits.** Web is 57% of Seed and 12% of Long-context. Computed against cumulative
consumption, web looks far over target by the later stages and gets starved repaying a debt it never
owed. Each stage is its own budget; deliver each stage's shares and the integral reproduces the plan.

**The exhaustion fallback.** A slot whose candidates were all rejected produced nothing, so the lane
was charged no tokens. Protected lanes always filled and unprotected ones sometimes did not, and
OPUS silently rewrote the mixture - Indic reached 47.4% against a planned 18%. This is the failure
this document warned about two sections earlier and then walked into anyway.

**The floor in tokens.** `ceil(0.14 x 16)` is 3 slots, 18.75% of slots. Indic packs at 0.944
utilisation against a 0.775 batch average, so those 3 slots became ~23% of tokens.

**On "14% of every batch".** A literal per-batch token floor cannot be enforced honestly - yield is
unknown until after packing, and reserving whole slots for the worst case is what caused the +4.68
overshoot. The floor is enforced as two measured guarantees instead: **Indic appears in 100% of
batches**, and its **cumulative token share never falls below 16.64%** against the 14% floor. Final
share 18.09% against 18.0% planned. The deviation from the literal wording is stated rather than
quietly redefined.

**A finding for A5.** `plan.json` protects Indic and agentic. The exhaustion counts say the proxy
also systematically under-rates **reasoning** (878 fallbacks), **long-context** (142) and **math**
(159). Those lanes are not protected, and without the fallback an English-and-code proxy would
starve them too. The protection list is incomplete.

### 5.2 The instructor's open question, answered

He raised it and left it open: **one shard mixed 10/90 with an OPUS override, or Indic kept
separate?**

**We keep Indic in separate shards, flagged `protected`.** Three reasons:

1. A mixed shard rejected by OPUS loses the other 90% with it.
2. OPUS judges from the first ~512 tokens, so a mixed shard's score is decided by whichever document
   happens to land first - an arbitrary property of shard assembly.
3. An override must be attributable to a lane to be auditable, and a mixed shard cannot say which
   lane it was overridden *for*.

Separate shards also let the scheduler enforce the 14% floor exactly rather than approximately.

### 5.3 His second open question: Indic curriculum order

He asked whether to train on simple news first or on RBI and legal documents. A5 already defines
difficulty bands B0-B5, so we band Indic **by register**: news → B1, general discussion → B2-B3,
legal/RBI/scientific → B4-B5, and let the existing stage gating schedule it. Cheap to build, and it
closes a question he asked aloud.

---

## 6. Packing policies per lane

The brief calls for "packing policies for different data types" and "correct loss masks, attention
masks and position ids". Every packed sequence carries `input_ids`, `loss_mask`, `position_ids`
(restarting per document), `segment_ids` (block-diagonal attention), and a provenance list.

| Lane | Policy | Loss mask |
|---|---|---|
| web | concat + hard chunk | all on |
| code | file-aware; block-diagonal per file; manifest records licence + repo | all on |
| math | document-aware best-fit packing | all on |
| reasoning | **atomic** - a trace is never split; dropped if it will not fit | prompt off, answer on |
| agentic | **atomic**; assistant spans on, tool output and system off | **≈0.35 trainable** |
| long_ctx | multi-document packing with per-doc position restart | all on |
| indic | document-aware + difficulty band; protected, never trimmed | all on |

Reasoning and agentic are atomic for the reason A5 §11 gives: a truncated chain of thought teaches a
wrong method. Agentic's mask is the showcase - it must reproduce `trainable_frac: 0.35` from
`plan.json` as a *measured* number, not a configured one.

### 6.1 Measured, after one policy rewrite

The first cut dropped any document larger than the window. At 1024 tokens that emptied the
long-context lane completely - 9 documents in, **0 sequences out** - and took most of code and math
with it. A policy that silently empties a lane is worse than one that truncates, because nothing in
the numbers says it happened. Oversize handling is now explicit per policy.

| Lane | Policy | Sequences | Utilisation | Loss-bearing | Docs/seq | Dropped |
|---|---|---:|---:|---:|---:|---:|
| web | CONCAT | 197 | 0.999 | 0.999 | 1.14 | 0 |
| code | DOC_ALIGNED | 247 | 0.997 | 0.997 | 1.07 | 0 |
| math | DOC_ALIGNED | 200 | 0.979 | 0.979 | 1.27 | 0 |
| reasoning | ATOMIC | 50 | 0.888 | 0.504 | 4.00 | 0 |
| agentic | ATOMIC | 123 | 0.707 | 0.213 | 1.07 | 68 |
| indic | DOC_ALIGNED | 207 | 0.944 | 0.944 | 1.62 | 0 |
| long_ctx | LONG_DOC | 219 | 0.977 | 0.977 | 1.00 | 0 |

**Agentic still drops 68 of 201 documents, and that number is the policy working, not failing.**
ATOMIC refuses to truncate a trace, so at a short window long traces cannot be placed. Sweeping the
window shows the constraint is the window, not the rule:

| Sequence length | 1024 | 2048 | 4096 | 8192 |
|---|---:|---:|---:|---:|
| agentic dropped (of 201) | 68 | 13 | **2** | **0** |
| agentic utilisation | 0.70 | 0.75 | 0.83 | 0.91 |
| reasoning dropped (of 300) | 0 | 0 | 0 | 0 |

At the 4K the lecture named, the drop rate is 1%. The demo runs at 1024 for the compute reason in
section 8.1, and pays 34% of the agentic lane for it - stated here rather than buried, because a
reader comparing lane token counts against the mixture would otherwise find the gap themselves.

All five packing invariants hold across 1,243 sequences: padding carries no loss, position ids start
correctly and stay contiguous within a placed run, attention is block-diagonal, and padding neither
attends nor is attended to.

---

## 7. Evaluation and validation firewall

Every shard carries `split: train | eval | holdout` in its manifest, and the split tag is part of the
shard hash so it cannot be edited without detection. The batch builder asserts `split == train`, with
an n-gram check behind it.

**We demonstrate the firewall by attacking it.** The demo deliberately attempts to admit an eval
shard into a loss-bearing batch; the block is what gets logged. An assertion that never fires proves
nothing.

This is also where A5's most expensive mistake gets redeemed. A5 §9.2 found a **100% train/validation
leak** because `prepare_data.py` repeated its source before splitting by token offset. This system
splits by document, records the split in the hash, and blocks at admission - three independent
mechanisms for the failure that cost us a published finding.

---

## 8. Ledgers, checkpoints, and the model

| Artefact | Contents |
|---|---|
| `ledgers/consumption.jsonl` | per batch: id, step, branch, stage, per-lane tokens, loss-bearing tokens, padding, provenance spans, batch hash |
| `ledgers/learning.jsonl` | per batch: loss, **per-lane loss**, per-sample loss, grad norm, lr |
| `ledgers/opus.jsonl` | every decision with score, threshold, snapshot hash, reason |
| `ledgers/firewall.jsonl` | every blocked admission attempt |

Checkpoints record step, branch, parent, **both ledger offsets**, scheduler state + hash, weight
hash, and the plan/manifest/tokenizer hashes they were made under - resume aborts if any differs.

**The model is a small transformer written from scratch in NumPy.** Not torch. The deciding reason is
grading Step 2: the grader regenerates `submission_artifacts/` and compares against the
`evidence.json` we committed. NumPy with fixed dtype and seeds reproduces bit-for-bit across
machines; torch CPU usually does, but thread count and BLAS backend can shift the last digits, and
that is not a risk worth taking under a 150-point section. Secondary: numpy + tokenizers is ~30MB
against torch's ~2GB, for an instructor who asked us not to make him download the world.

We are not using the permitted stub loop. "Learning trace - loss linked to source data" is a named
row in the evidence bundle and the ledgers block is 150 points; a real loss that falls and is
attributed per lane costs little and defends itself under inspection.

### 8.1 Sequence length, stated plainly

The demo trains at **1024 tokens**, not the 4K the lecture mentions. NumPy attention is O(n²) and a
4K context at this batch size needs about a gigabyte per layer per direction.

The packing policies, masks and position ids are correct at any length, and the **test suite
exercises 4096 and 32768 explicitly** without training on them - which is where correctness is
actually verified. `SEQ_LEN` is a config value. We state the limit rather than quietly packing short
and calling it 4K.

---

## 9. What `run_demo.py` does, in order

Mapped to the event list the brief requires in `run.log`:

```
 0  verify frozen inputs        -> [PASS] tokenizer_hash_verified
 1  build shards + manifest     -> shards created / manifests validated
 2  attack the firewall         -> [PASS] eval_shard_blocked
 3  compile mixture schedule    -> mixture compiled
 4  RUN A (reference)           -> batches packed / OPUS decisions recorded
 5  RUN B (crash at step K)     -> [PASS] checkpoint_saved / crash simulated
 6  resume B                    -> [PASS] resume_next_batch_matched
 7  assert A == B               -> [PASS] ledger_equivalence
 8  replay interval [i,j]       -> [PASS] replay_hash_matched
 9  fork from checkpoint K      -> branch forked
10  audit from the ledgers      -> audit completed
11  performance from the ledger -> performance measured
12  emit evidence bundle        -> evidence.json + evidence.md
```

---

## 10. Tests

Named for the invariant, not the function:

1. `test_shard_determinism` - build twice, byte-identical
2. `test_tokenizer_frozen` - hash matches; tampering aborts the run
3. `test_masks` - tool spans zeroed; position ids restart per doc; no cross-document attention
4. `test_no_eval_in_train` - every training batch's provenance is train-split only
5. `test_mixture_compliance` - actual shares within tolerance; Indic ≥14% in **every** batch
6. `test_resume_equivalence` - ledger A ≡ ledger B
7. `test_replay_hashes` - replayed interval matches
8. `test_fork_lineage` - prefix identical, suffix divergent
9. `test_opus_protected` - no protected-lane document ever dropped; overrides recorded
10. `test_ledger_contiguity` - batch ids contiguous, no duplicates

---

## 11. Prerequisites before coding

- **Correct `assignment-5/plan.json`.** `reasoning.generated_B` reads 88 and
  `supply_totals.generated_B` reads 238 - both are pre-adoption 6% figures left behind when reasoning
  moved to 9%. `ledger.py` computes the right numbers (208B and 358B), which is why `ledger.md` and
  the A5 README are correct and only `plan.json` is stale. A6 consumes it as source of truth, so it
  must be fixed first.
- **Update `NOTICE.md`** to record the committed corpus slices and their licences, since it currently
  states these datasets are referenced and never redistributed.

---

## 12. Layout

```
assignment-6/
  README.md              architecture + design decisions (the graded document)
  PLAN.md                this file
  run_demo.py            the one command
  fetch_corpus.py        build-time only; never run by run_demo
  frozen/                tokenizer, plan, corpus, SOURCES.json
  src/                   shards manifest packing mixture opus firewall
                         stream ledger checkpoint model train audit evidence
  tests/                 the ten invariants
  submission_artifacts/  generated: run.log, evidence.{json,md}, manifests/,
                         ledgers/, checkpoints/, performance.json
```

Deps: `numpy`, `tokenizers`. `datasets` is needed only by `fetch_corpus.py`, which the grader never
runs.

---

## 12a. Re-verification round: six gaps found by auditing our own claims

After the system passed 10/10, we re-read the brief against the *code* rather than against our own
evidence bundle. Six things were wrong or weak. All are fixed; the worst was a claim in our own
documentation that the code did not implement.

| # | Gap | Was | Now |
|---|---|---|---|
| 1 | Difficulty bands computed, stored, read by nothing - while README §4.4 claimed they gated the curriculum | **false claim** | 2,597 draws refused for sitting outside their stage band |
| 2 | Optimiser state not checkpointed; resume built a fresh Adam | silent bug | 60/60 losses bit-identical after resume |
| 3 | Deferral recorded but never acted on - `requeue_after` written, never read | cosmetic | 129 slots filled from deferred candidates |
| 4 | Firewall guarded a train-only pool, so it could never fire | untestable | 704 non-train documents in the live pool, refused at every draw |
| 5 | Anneal reserve in `plan.json`, absent from the code | not implemented | 290 reserved draws refused before the anneal |
| 6 | Token-level loss absent (sample-level present) | satisfied the "or" | both recorded |

**Gap 1 is the one worth dwelling on.** The band field was computed in `fetch_corpus.py`, carried
through sharding, used by `packing.py` for *ordering*, and loaded into the stage dict by
`mixture.py` - so it looked implemented from every angle except the one that mattered. Nothing ever
compared a document's band against its stage's range. A document-level field that exists, is
plumbed, and changes nothing is the hardest kind of gap to see, and evidence.json could not have
caught it: there was no row asserting the curriculum gated on anything.

**Gap 2 is the one that improved the result most.** The data stream does not depend on model state,
so ledger equivalence passed with a fresh optimiser and nothing looked wrong. Checkpointing Adam's
moments turned crash recovery from "the same batches were consumed" into "the same run happened" -
all 60 losses reproduce with a worst difference of exactly `0.0`.

### The policy improvement that came out of it

Deriving bands from length exposed an interaction: the hardest band is by construction the longest
documents, and ATOMIC dropped anything longer than the window - so late curriculum stages starved
the ATOMIC lanes outright, taking agentic to **0.73%** against a planned 8%.

The fix sharpened the rule from "never split a document" to **"never split a loss-bearing span"**.
A masked context span - a wall of function schemas, a tool dump - can be divided without teaching
the model anything wrong, because it is not a target. Only a chain of thought or a tool call must
stay whole. Glaive's system prompts alone exceed a 512-token window, which is why the blunt rule was
so expensive.

| | drop rate at 512 | mixture max delta |
|---|---:|---:|
| never split a document | 39% | 7.27 pts |
| never split a **loss-bearing span** | **14%** | **1.45 pts** |

A test verifies all 777 loss-bearing spans survive intact. Compliance ended better than before the
re-verification round started (1.45 against 3.47), which is the argument for auditing a passing
system rather than shipping it.

---

## 13. Known limitations, declared up front

1. **1024-token sequences, not 4K** (§8.1). Packing correctness is tested at 4K and 32K; training is
   not.
2. **The OPUS scorer is a frozen lexical proxy**, not a gradient-alignment scorer. Real OPUS scores
   gradient alignment against a benchmark proxy; ours scores token-profile similarity. The
   *mechanism* - four decisions, protected override, snapshot-versioned replay - is faithful; the
   scoring function is a stand-in and is labelled as one.
3. **The model is ~2M parameters and trains for a few hundred steps.** Loss falls; that is all it is
   asked to demonstrate. No claim is made about capability.
4. **Agentic and long-context corpora are small, and agentic is the wrong shape.** Both lanes exist
   to exercise packing policy, not to teach the model anything. Glaive is function-calling chat, not
   the SWE-bench and terminal traces the plan sized, so its measured trainable fraction is 0.590
   rather than 0.35 (section 4.2). Representative agentic data is not available under a licence that
   permits redistribution, which is itself the finding A5 section 11 predicted when it ranked agentic
   as the number-one acquisition priority.
