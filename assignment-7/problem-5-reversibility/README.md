# Problem 5, reversibility: the codec was never the thing that was broken

**ERA V5, Assignment 7, Problem 5. Ashish Kamble.**

Every number here is produced by a script in `src/` and written to `artifacts/`. None is typed by
hand. `artifacts/evidence.md` regenerates the whole set from the committed artefacts, and
`run_demo.py` regenerates the artefacts with no network access.

---

## The question

> "Kronecker is forward deterministic (same word will always give same embedding). How do I make a
> reverse of this (same embedding gives the same Kronecker)? If we can do this, then we can get rid
> of the final head as well! Then we can have a vocab of 1M as well without any issues!"

Three promises: **reverse it**, **delete the output head**, **get a 1M vocabulary for free**. This
writeup delivers the first two and returns a split verdict on the third, because the third turned
out not to be testable on this machine and saying so is more useful than pretending otherwise.

## The answer to each of the three promises

| what was promised | the answer | evidence |
|---|---|---|
| **"How do I make a reverse of this?"** | **It already reverses**, in three independent senses. Exact inversion of every token that fits the window; tolerates **60x** more error than the objection implies; and survives `Linear(8192, 768)` because a code is only 7.93-sparse, which makes recovery compressed sensing rather than magic. | E1, E2, E3 |
| **"We can get rid of the final head!"** | **Yes, at zero new parameters**, 0 against 100.7M at the paper's dimensions. But it costs accuracy: at this scale the **vocabulary head wins** on loss, 2.6367 against 4.7164 nats per token. The byte head's case is parameter scaling, which is arithmetic here and not measurement. Its one genuine defect, about 12% invalid UTF-8, is **removed entirely** by constrained decoding at no cost. | E4, E8 |
| **"A vocab of 1M without any issues!"** | **Split verdict.** The **capability** is architecturally true and needs no experiment: a vocabulary softmax has no output row for an unknown word and scores exactly zero at any amount of training. The **competence** is not demonstrated, and two experiments that tried to demonstrate it both failed for reasons unrelated to the claim. | E6, E7 |

The rest of this document is how those answers were arrived at, in the order the experiments ran.

## The techniques used, and what each one is for

Several of these exist because a naive version of the same measurement gave a wrong answer first.
Those are marked, because the reason a technique is needed is usually more informative than the
technique.

| technique | what it is for | where |
|---|---|---|
| **Per column argmax decoding** | Invert the codec by *ranking* rather than by values. This is the whole reason noise tolerance exists, and why z-normalisation cannot break decoding. | E1, E2 |
| **Margin relative length inference** | Infer where a token ends from each vector's own peak margin. **Needed because** a fixed global threshold silently truncates the longest tokens: the margin scales with occupancy, so long tokens genuinely have smaller margins. | E1, E2 |
| **Noise sweep in z-normalised units** | Quote noise as a multiple of the signal's own standard deviation, which is exactly 1 by construction, so the numbers are scale free and comparable across window sizes. | E2 |
| **Minimum norm preimage** | The standard structure blind linear inverse, used to probe what a projection retains. | E3 |
| **Constructive collision generation** | Build exact collisions at machine precision, `x = k_b - P(k_b - k_a)`, rather than searching for them. | E3 |
| **Off manifold residual test** | Re-encode whatever a vector decodes to and measure the distance back to it. **Needed because** after z-normalisation no entry is zero and every code has the same norm, so counting zeros or comparing norms measures nothing at all. | E3 |
| **Nearest neighbour separation** | Test injectivity directly. If distinct tokens stay distinguishable then no information was destroyed, whatever any particular decoder manages. | caveat |
| **Learned linear inverse on a split** | Fit a decoder on one set of tokens and score it on tokens it never saw. **Needed because** minimum norm is the wrong tool for a trained projection, and using it alone would have reported a refutation. | caveat |
| **Per token loss normalisation** | Convert nats per byte position into nats per token. **Needed because** the two heads live in different output spaces and an unconverted table would make either look arbitrarily better. | E4 |
| **Copy diagnostic** | Score predictions against the *current* token as well as the next. **Needed because** an untrained tied head is an 83.43% autoencoder, so its step 0 accuracy is wiring and would otherwise read as learning. | E5 |
| **Separate input and target streams** | Let a word enter as an unknown marker while still being required as a target. **Needed because** it is the only way to ask a model for a word it has no id for. | E7 |
| **Rarity matched control band** | Hold frequency constant so a zero can be attributed to vocabulary membership rather than to rarity. **Needed because** without it, E7 reads as a refutation of the brief rather than as an untestable question. | E7 |
| **Central difference gradient check** | Verify every parameter's gradient numerically, including the tied head's dual path where `W` receives gradient as both input projection and unembedding. **Needed because** it caught a float32 truncation on the input path that was invisible in training. | all trained |
| **Seed noise floor** | Repeat across seeds and report any effect inside two standard deviations as not established. | E4 |

## The shape of the argument

Reversibility is not blocked. It holds in three independent senses and the margins say by how much.
The objection that makes it look impossible is an artefact of an assumed decoding rule, and the
decode tolerates **60 times** more error than the objection implies.

Deleting the head works: a head tied to `W` transposed adds **zero parameters**. It also costs
accuracy at the scale this machine can reach, and that cost is stated in the table rather than
buried under an extrapolation.

The 1M vocabulary claim splits cleanly. The **capability** is architecturally true and needs no
experiment. The **competence** is not demonstrated, and two experiments that tried to demonstrate it
both failed for measurable reasons that had nothing to do with the claim.

---

## The objection, and why it does not survive contact with the construction

The natural doubt is this: a trained network never emits the exact numbers the codec produced. If
the codec writes `0.30, 0.20, 0.10` the model emits `0.31, 0.18, 0.09`. Close, not equal. So exact
inversion looks unreachable, and the natural remedy looks like a variational autoencoder, predicting
a distribution instead of a point.

That reasoning assumes decoding needs the **values**. It does not. The codec output is a 256 by L
matrix with exactly one 1 per occupied column, so decoding is an **argmax down each column**, and an
argmax needs correct **ranking**, not correct values.

It follows immediately that z-normalisation cannot break decoding either, since it is a strictly
increasing affine map applied to every entry alike, so it preserves every ordering.

**E2 puts a number on it.** The gap between the correct row and the runner up, after
z-normalisation, is **37.83** against a signal standard deviation of 1.0. The "0.30 against 0.31"
worry is a relative error of about 0.03. Exact token accuracy stays at **100%** up to noise of two
standard deviations and is still 99.69% at three.

| noise sigma, relative to signal | exact token, oracle length | exact token, inferred length |
|---|---|---|
| 0.00 | 100.00% | 100.00% |
| 1.00 | 100.00% | 100.00% |
| 2.00 | 100.00% | 96.47% |
| 3.00 | 99.69% | 80.18% |

**The headroom, derived rather than asserted.** "0.31 instead of 0.30" is a relative error of
`0.01 / 0.30 = 0.0333`. Exact accuracy is still 100% at a noise sigma of **2.0**, and the codec is
z-normalised so sigma is measured in units of the signal's own standard deviation. The decode
therefore tolerates `2.0 / 0.0333 =` **60 times** the error the objection describes. The point cloud
is not needed.

## E1, the codec inverts exactly

| window | D | tokens fitting | recovered exactly | tokens overflowing |
|---|---|---|---|---|
| L=16 | 4,096 | 8,384 | 8,384 (100%) | 1,360 |
| L=32 | 8,192 | 9,510 | 9,510 (100%) | 234 |
| L=64 | 16,384 | 9,726 | 9,726 (100%) | 18 |

Overflowing tokens recover their retained prefix exactly and not the token, because the dropped
bytes were never encoded. That loss is Problem 3's subject, not a defect in the codec. The longest
token in this vocabulary is 116 bytes.

## E3, where invertibility actually dies, and the prediction that was wrong

`PLAN.md` predicted that decoding through `Linear(8192, 768)` would land near chance, because the
map has a 7,424 dimensional nullspace and no left inverse. **That prediction is refuted, and the
reason is worth more than the prediction was.**

A codec vector is not an arbitrary point in R^8192. It is k-sparse with mean k = **7.93** occupied
columns, which is 0.097 percent dense. Recovering a k-sparse vector from a random linear measurement
is the compressed sensing regime, and it succeeds whenever the measurement count comfortably exceeds
k. With d_model=768 and k around 8, that condition is not marginal, it is met by two orders of
magnitude.

| d_model | nullspace dimension | minimum-norm decode accuracy |
|---|---|---|
| 8 | 8,184 | 0.67% |
| 128 | 8,064 | 55.67% |
| 256 | 7,936 | 79.67% |
| 768 | 7,424 | **99.33%** |

Exact collisions do exist and are constructed at machine precision, but they are **off manifold**:
the relative distance from a collision vector to the re-encoding of whatever it decodes to is 0.4166,
against exactly 0.0000 for a real code. They are not reachable outputs of the encoder.

### The caveat this left open, and how it closed

E3 used a **random** `W`. A trained one might concentrate and lose the property that makes recovery
work, so nothing was claimed end to end until that was checked. It was checked, and at first it
looked like a refutation:

| | minimum-norm decode, before training | after training | condition number |
|---|---|---|---|
| d_model=96 | 42.67% | 9.33% | 1.23 to 35.24 |
| d_model=256 | 75.00% | 30.67% | 1.43 to 40.41 |

Training does concentrate `W`. But **minimum norm is a structure blind decoder**. It is the correct
tool for the random `W` of E3 and the wrong tool for a trained one, and asking it to invert a
structured matrix measures the decoder, not the encoding. So the question was split in two:

| | exact duplicates | minimum relative separation | learned inverse, held-out tokens |
|---|---|---|---|
| random init | 0 | 0.2594 | 59.17% |
| after training | 0 | 0.0492 | **56.83%** |

Distinct tokens stay distinguishable after training, with zero duplicates, so no information is
destroyed. A linear decoder fitted on 2,500 tokens and scored on 600 it never saw loses **2.3
points**. Training breaks the decoder, not the encoding.

**The claim, in its sharpened and final form:** recovery is a property of the encoding **plus an
appropriate decoder**, and never of any decoder. A pleasing detail that supports the reading: minimum
norm beats the fitted inverse at random initialisation, 75% against 59%, exactly as compressed
sensing predicts, and then loses to it after training.

---

## E4, delete the output head: it works, and it costs something

Three heads, identical body, identical data, identical seeds. The vocabulary head is the baseline.
The untied byte head predicts per position byte logits. The tied head does the same thing reusing
`W` transposed and therefore adds **no parameters at all**.

**A unit trap governs this table.** A vocabulary head's loss is nats per **token**. A byte head's is
nats per **byte position**, at about 3.2 positions per token here. Reporting them together without
converting would make either head look arbitrarily better. Everything below is nats per token.

| head | loss per token | native loss | exact token | parameters |
|---|---|---|---|---|
| vocabulary | **2.6367** (sd 0.0085) | 2.6334 per token | **44.13%** | 1,980,864 |
| byte, untied | 3.8622 (sd 0.0545) | 1.1952 per position | 42.68% | 1,807,296 |
| byte, tied | 4.7164 (sd 0.1524) | 1.4365 per position | 31.73% | **1,020,864** |

Seed noise floor is 0.1524 nats per token. Both byte deltas exceed it.

**The vocabulary head wins at this scale, and that is the measured result.** It is stated first
because the alternative is to hide it.

Two things complicate the simple reading, and both are real. Per token loss compounds over about 3.2
byte positions while argmax accuracy does not, which is why the untied byte head is within 1.5 points
of the vocabulary head on exact token accuracy while looking far worse on loss. And the byte head's
actual argument was never that it wins at 10,000 tokens. It is that its head does not grow with the
vocabulary:

| head | output head parameters at d_model=768, vocab=131,072 |
|---|---|
| vocabulary | **100,663,296** |
| byte, untied | 6,291,456 |
| byte, tied | **0** |

**That row is arithmetic, not measurement, and is labelled as such throughout.** The honest summary
is that the loss comparison and the parameter comparison point in opposite directions at this scale,
and only one of them was measured here.

## E5, is there any gradient at step 0

The second objection: at random initialisation the predicted vector is nowhere near a real token, so
there is nothing to decode and no signal. `PLAN.md` predicted that cross entropy would climb off the
floor immediately and an MSE regression head would not.

**That prediction is also refuted.** Both climb, and by almost the same amount.

| objective | byte accuracy at step 0 | at the end | loss per token |
|---|---|---|---|
| cross entropy | 17.27% | **51.87%** | 17.392 to 5.067 |
| MSE to the code | 17.27% | **51.67%** | 17.392 to 14.006 |

The diagnostic that explains it is more interesting than the prediction was. An **untrained tied
head is already an autoencoder**: it reproduces the **current** token's bytes at **83.43%** accuracy
and the next token's at 16.68%, against a chance rate of 0.39%. This is not learning, it is wiring:
`xf @ W.T` reuses the same `W` that produced the embedding, so it is close to an identity on the
residual stream.

So the objection's premise, that there is nothing to decode at random initialisation, is simply
false, and it is false for **both** objectives rather than being repaired by the choice of loss. The
refutation strengthens the conclusion and kills the mechanism this plan had proposed for it.

A corollary worth stating because it is easy to misuse: the tied head's step 0 accuracy must never be
quoted as evidence of learning.

## E6, what tying costs

Over 49,551 predictions, decoded with the target's true length so that this measures byte prediction
alone and not length inference.

| outcome | rate |
|---|---|
| invalid UTF-8 | **11.98%** |
| valid and in vocabulary | 83.38% |
| valid but out of vocabulary | 4.64% |
| exact match to the target token | 38.16% |

**The invalid UTF-8 rate is a genuine defect**, and it is the direct cost of predicting positions
independently. It also has a remedy, which E8 below implements and measures.

The out of vocabulary strings are `तत`, `ततत`, `ర఍` and similar: degenerate repeats and strings
carrying unassigned codepoints. **They are not plausible words**, which matters for the next section.

## E8, constrained decoding, which removes that defect entirely

E6's diagnosis was right and the conclusion drawn from it was too pessimistic. The head is not wrong
about the distribution; it is simply never asked for a coherent sequence. Each position takes its
argmax alone, so nothing prevents a lead byte where a continuation byte was required.

So ask for a coherent one. At each position, mask the bytes that cannot legally follow what has
already been emitted, then take the argmax. **Same trained model, same logits, no retraining and no
architectural change.** It is a decoding rule.

| metric | unconstrained | constrained |
|---|---|---|
| invalid UTF-8 | 11.62% | **0.00%** |
| empty output | 0.00% | **0.00%** |
| in vocabulary | 83.95% | **91.77%** |
| exact match to the target | 41.25% | **44.51%** |

The defect is gone, and exact match **improves by 3.27 points** rather than being traded away. That
is worth stating carefully: a constraint cannot make a model better at predicting, only better at
being well formed. It helps here because the head's second choice at a position is often right when
its first choice was structurally impossible.

**Three implementation details, each of which was wrong first and each of which mattered:**

1. **Length structure is not validity.** Enforcing only "a lead byte expects N continuations" still
   left about 8% of decodes invalid, because `0xE0` must be followed by `0xA0..0xBF` or the sequence
   is overlong, and `0xED` must be followed by `0x80..0x9F` or it encodes a UTF-16 surrogate.
2. **The trailing trim must remove the lead byte too.** Popping continuation bytes until the
   expected count reaches zero leaves the lead byte in place, which is still invalid.
3. **A character must fit the remaining budget.** Without that check the decoder starts a three byte
   character with one position left, the trim then removes it, and a short token decodes to the
   **empty string**, which is trivially valid UTF-8 and flatters the validity rate while saying
   nothing. Fixing it raised exact match by 3.2 points, since the budget forces a character that
   can actually be completed.
4. **When the wanted character does not fit, stop rather than substitute.** This one was found by a
   test rather than by inspection. A window that ends mid-character cannot be reproduced by a
   decoder that only emits valid UTF-8, which is correct behaviour; what it must not do is fill the
   gap. For a one-hot code every permitted byte scores identically, so a naive fallback returns
   index 0 and pads the output with **NUL bytes**. Stopping instead drops exactly the split
   character and leaves a clean prefix. Stopping is conditional on having emitted something
   already, since at the first position it would just produce the empty string again.

**This connects directly to Problem 3.** That writeup measured that the published window cuts a word
mid-character essentially always for Indic, because 32 is not a multiple of 3. Constrained decoding
therefore *cannot* reproduce the retained bytes of a cropped Indic token, and correctly returns the
largest valid prefix instead. The two findings are the same fact seen from opposite ends: a byte
window that does not respect character boundaries produces codes that are not decodable as text.

What this does **not** do is fix E7. Being well formed is not the same as being right, and the open
vocabulary question below is unaffected.

---

## E7, the 1M vocabulary claim, and an honest split verdict

E6 looked like a refutation of the brief's third promise. It is not, because **E6 could not test
it**: every target E6 scored was already inside the vocabulary, so "can this address words it was
never given" was never actually put to the model.

E7 puts it. The input vocabulary is deliberately small at 8,000 words. A further 29,389 words (web)
and 14,657 (indic) appear **only as targets**, entering the model as an unknown marker. A vocabulary
softmax scores exactly zero on those by construction, since no output row exists for them.

| lane | in-vocabulary targets | targets outside the vocabulary |
|---|---|---|
| web | 7.54% | **0.00%** |
| indic | 2.29% | 0.03% |

Zero. That looks decisive, and it is not, because outside-vocabulary words are by construction also
the **rarer** words. A zero could mean "cannot emit an unknown word" or merely "cannot predict rare
words at all", and those support opposite conclusions. So the targets were banded by frequency, with
`in_tail` as a rarity matched control: the least frequent words that **are** in the vocabulary.

| band (web) | exact reconstruction |
|---|---|
| in_head, most frequent quarter | 9.09% |
| in_mid | **0.00%** |
| **in_tail, in vocabulary but rare** | **0.00%** |
| outside_near | 0.00% |
| outside_far | 0.00% |

**The model reconstructs only the most frequent quarter of words. In-vocabulary words below that
score zero too.** So vocabulary membership explains nothing here. It is a rarity cliff, and this
experiment cannot separate the two effects at this scale.

### The verdict

- **Capability: true, and provable without measurement.** A vocabulary softmax has no output row for
  a word outside its inventory and scores exactly zero at any amount of training. A tied byte head
  can represent any byte string that fits the window. The asymmetry is architectural.
- **Competence: not demonstrated, and not testable on this machine.** Both E6 and E7 failed to test
  it, each for a different measurable reason.

This writeup does **not** say the 1M vocabulary claim is false. It says the capability holds, the
competence does not follow from it, and this scale cannot decide between them.

---

## The mistake this assignment made three times

Three separate experiments were run on units that structurally could not exhibit the effect being
measured, and each time the null looked like a finding:

| experiment | the flaw | how it was caught |
|---|---|---|
| Problem 3, E5 | BPE tokens truncate 0.0024% of occurrences, so all codecs carried identical information | an exposure diagnostic |
| Problem 5, E6 | every scored target was already in the vocabulary | separate input and target streams |
| Problem 5, E7 | rare in-vocabulary words also score zero | a rarity matched control band |

**Measure the exposure before believing a null.** That is the most transferable thing in this
assignment and it is worth more than any single result in it.

## What is deliberately not claimed

- **The byte head is not shown to be better.** At this scale it is worse on loss. Its case rests on a
  parameter count at dimensions this machine cannot train.
- **The 1M vocabulary payoff is untested, not vindicated and not refuted.**
- **Everything trained here is 2 layers at d_model=96 on CPU**, 300 steps, three seeds, with every
  effect quoted against a measured seed noise floor. E1 to E3 need no model and do not depend on
  scale. E4 to E7 do.
- **The learned inverse in the caveat check is linear.** A nonlinear decoder was not tried, so the
  recovery numbers after training are a lower bound rather than the best achievable.
- **Constrained decoding is greedy, not optimal.** It masks illegal bytes and takes the argmax; the
  highest scoring valid sequence would need a beam search, which was not run. So 44.52% exact match
  is a lower bound on what the same logits can support.

## Reproducing

```bash
python -m venv .venv && .venv/bin/pip install -r ../requirements.txt
cd problem-5-reversibility
python tests/test_invariants.py   # the invariants, a few seconds
python run_demo.py           # E1 to E3. About half a minute, no network, no model.
python run_demo.py --full    # adds E4 to E7. Roughly an hour on 16 CPU cores.
```

Verification that runs every time: the factored codec reproduces the float64 codec definition to
**1.4e-14**, and a central difference gradient check covers every parameter including the tied
head's dual path, worst relative error about **5e-7**.

## Files

| path | what it is |
|---|---|
| `PLAN.md` | the plan agreed before any code, including the two predictions the runs refuted |
| `src/exp_codec.py` | E1 to E3, round trip, noise tolerance, the projection sweep |
| `src/exp_train.py` | E4 to E6, three heads, gradient at init, the cost of tying |
| `src/exp_recheck.py` | the open caveat, injectivity and a learned inverse under a trained W |
| `src/exp_openvocab.py` | E7, the open vocabulary test and its rarity matched control |
| `src/exp_constrained.py` | E8, constrained decoding against the unconstrained argmax |
| `src/evidence.py` | regenerates every number in this README from the artefacts |
| `../common/provenance.py` | stamps each artefact with the SHA of the code that wrote it, and reports any that have gone stale |
| `src/build_dashboard.py` | extracts the dashboard payload from the same artefacts |
| `site/` | static dashboard. No framework, no build step, no network. Open `site/index.html` |
| `artifacts/evidence.md` | the generated evidence file, for a reader |
| `artifacts/evidence.json` | the same numbers as data, for anything that wants to assert on one |
| `tests/test_invariants.py` | the properties the claims rest on, run with `python tests/test_invariants.py` |
| `../common/kron_model.py` | the Kronecker input, three output heads, the factored codec, Adam, the gradient check |
| `../common/codec.py` | the codec itself, zero learned parameters |
