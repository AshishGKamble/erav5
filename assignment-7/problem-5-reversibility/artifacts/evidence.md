# Problem 5, evidence

Every number below is read from `artifacts/*.json`. Regenerate with `python src/evidence.py`.

Vocabulary measured: 9,744 non-byte-fallback tokens from the assignment-2 tokenizer. Seed 20260825.


## E1, does the codec invert

| window | D | tokens fitting | recovered exactly | tokens overflowing |
|---|---|---|---|---|
| L=16 | 4,096 | 8,384 | 8,384 (100.00%) | 1,360 |
| L=32 | 8,192 | 9,510 | 9,510 (100.00%) | 234 |
| L=64 | 16,384 | 9,726 | 9,726 (100.00%) | 18 |

Overflowing tokens recover their retained prefix exactly but not the token. Those bytes were never encoded, which is Problem 3's territory, not a codec defect.


## E2, does inversion survive approximate prediction

Column margin after z-normalisation: **37.83** against a signal standard deviation of 1.0. Measured over 1,953 tokens at L=32.

| noise sigma (relative to signal) | exact token, oracle length | exact token, inferred length |
|---|---|---|
| 0.00 | 100.00% | 100.00% |
| 0.05 | 100.00% | 100.00% |
| 0.10 | 100.00% | 100.00% |
| 0.25 | 100.00% | 100.00% |
| 0.50 | 100.00% | 100.00% |
| 0.75 | 100.00% | 100.00% |
| 1.00 | 100.00% | 100.00% |
| 1.50 | 100.00% | 99.90% |
| 2.00 | 100.00% | 96.47% |
| 3.00 | 99.69% | 80.18% |
| 5.00 | 83.21% | 50.33% |
| 8.00 | 54.28% | 25.65% |
| 12.00 | 24.88% | 9.88% |

## E3, where invertibility actually dies

A code is k-sparse with mean k = **7.93** occupied columns out of D = 8,192, which is 0.097% dense. Recovering it from a random projection is therefore compressed sensing, not magic.

| d_model | nullspace dimension | minimum-norm decode accuracy |
|---|---|---|
| 8 | 8,184 | 0.67% |
| 16 | 8,176 | 1.67% |
| 32 | 8,160 | 11.67% |
| 48 | 8,144 | 21.00% |
| 64 | 8,128 | 27.67% |
| 96 | 8,096 | 43.67% |
| 128 | 8,064 | 55.67% |
| 192 | 8,000 | 68.67% |
| 256 | 7,936 | 79.67% |
| 384 | 7,808 | 94.00% |
| 512 | 7,680 | 98.00% |
| 768 | 7,424 | 99.33% |

Fifty percent recovery at d_model = 128, ninety-nine percent at 768. PLAN.md predicted near-chance decoding here; that prediction is **refuted**.


## The caveat E3 left open: does training destroy this


- **d_model=256**, minimum-norm decode 75.00% before training, 30.67% after. Condition number 1.43 to 38.86.

- **d_model=96**, minimum-norm decode 42.67% before training, 9.67% after. Condition number 1.23 to 35.20.

That looks like a refutation. It is not, because minimum norm is a **structure-blind** decoder, correct for the random W of E3 and wrong for a trained one. Splitting the question (`artifacts/recheck.json`, d_model=256):

| | exact duplicates | relative separation (min) | learned inverse, held-out tokens |
|---|---|---|---|
| random init | 0 | 0.2594 | 59.17% |
| after training | 0 | 0.0486 | 56.83% |

The decoder is fitted on 2,500 tokens and scored on 600 it never saw. Training breaks the decoder, not the encoding.


## E4, three output heads

Indic lane, d_model=96, L=32, 300 steps, seeds [1, 2, 3]. Seed noise floor sd **0.1524** nats per token.

| head | loss per token | native loss | units/token | exact token | parameters |
|---|---|---|---|---|---|
| vocab | 2.6367 (sd 0.0085) | 2.6334 | 1.00 | 44.13% | 1,980,864 |
| byte_untied | 3.8622 (sd 0.0545) | 1.1952 | 3.20 | 42.68% | 1,807,296 |
| byte_tied | 4.7164 (sd 0.1524) | 1.4365 | 3.20 | 31.73% | 1,020,864 |

At the paper's scale the parameter picture inverts, and this row is arithmetic, not measurement:

| head | output head parameters at d_model=768, vocab=131072 |
|---|---|
| vocab | 100,663,296 |
| byte_untied | 6,291,456 |
| byte_tied | 0 |

## E5, is there gradient at step 0

An untrained tied head reproduces the **current** token's bytes at 83.43% and the **next** token's at 16.68%, against a chance rate of 0.391%. It is an autoencoder before it is trained, because `xf @ W.T` reuses the same W that produced the embedding.

| objective | byte accuracy at step 0 | byte accuracy at end | loss per token, start to end |
|---|---|---|---|
| CE | 17.27% | 51.87% | 17.392 to 5.067 |
| MSE | 17.27% | 51.67% | 17.392 to 14.006 |

PLAN.md predicted MSE would not move. It moves. The objection is wrong for both objectives, and the reason is the autoencoder above, not the loss function.


## E6, what tying costs

Over 49,551 predictions, decoded with the target's true length.

| outcome | rate |
|---|---|
| invalid UTF-8 | 11.98% |
| valid and in vocabulary | 83.38% |
| valid but out of vocabulary | 4.64% |
| exact match to the target token | 38.16% |

Out-of-vocabulary examples: ర఍, ం఍, तत, ర఍ఁ, तऍ, ర఍ర, రఁ, ం఍ఁ. These are degenerate, not plausible words.


## E7, can the head emit words it was never given an id for


**indic**: input vocabulary 8,000, 14,657 further words appear only as targets.

| target band (by frequency) | exact reconstruction |
|---|---|
| in_head | 3.1693% |
| in_mid | 0.0341% |
| in_tail | 0.4065% |
| outside_near | 0.0000% |
| outside_far | 0.0398% |

A vocabulary softmax scores exactly 0.0 on the outside bands, by construction rather than by measurement.

**web**: input vocabulary 8,000, 29,389 further words appear only as targets.

| target band (by frequency) | exact reconstruction |
|---|---|
| in_head | 9.0935% |
| in_mid | 0.0000% |
| in_tail | 0.0000% |
| outside_near | 0.0000% |
| outside_far | 0.0000% |

A vocabulary softmax scores exactly 0.0 on the outside bands, by construction rather than by measurement.

`in_tail` is the rarity-matched control: the least frequent words that ARE in the inventory. It scores at the floor too, so the zero on the outside bands is a rarity cliff and says nothing about vocabulary membership. This experiment cannot test the claim at this scale.


## E8, constrained decoding, which removes E6's defect

Same trained head, same logits, 40,573 predictions. The only change is that bytes which cannot legally follow what has been emitted are masked before the argmax, and any incomplete trailing character is dropped.

| metric | unconstrained | constrained |
|---|---|---|
| invalid UTF-8 | 11.62% | 0.00% |
| valid UTF-8 | 88.38% | 100.00% |
| empty output | 0.00% | 0.00% |
| in vocabulary | 83.95% | 91.77% |
| exact match to the target | 41.25% | 44.51% |

Invalid UTF-8 removed: 11.62%. Exact match change: +3.27 points. No retraining, no architectural change.


## Verification

- Factored codec against the float64 definition: max absolute difference **1.42e-14**.
- Factored codec against `codec.encode` (float32): 7.65e-06, which is that function's own rounding.
- Central-difference gradient check, worst relative error per parameter: `W` 2.8e-07, `lnf_g` 2.0e-10, `wq0` 2.1e-07.
