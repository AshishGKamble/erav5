# Assignment 7, submission

**Ashish Kamble. ERA V5, Assignment 7, Kronecker embeddings.**

## Which problems

**Problem 5 (reversibility) and Problem 3 (dynamic length).**

The assignment says the problems are separate and should not be mixed, so they are in separate
folders with their own plans, experiments, artefacts and writeups. [README.md](README.md) is the
hub, since the form takes one link.

Live dashboards, the same numbers as charts:

- Problem 3, dynamic length: https://thunderous-cactus-b47590.netlify.app
- Problem 5, reversibility: https://euphonious-centaur-3a5265.netlify.app

## Where each claim lives

| claim | evidence | file |
|---|---|---|
| The codec inverts exactly | 9,510/9,510 tokens at L=32 | P5 E1, `artifacts/codec.json` |
| The "0.31 not 0.30" objection fails | margin 37.83 against signal sd 1.0 | P5 E2 |
| The projection does not destroy invertibility | 99.33% recovery at d_model=768, codes are 8-sparse | P5 E3 |
| Training does not destroy it either | zero duplicates, fitted decoder loses 2.3 points | P5 recheck, `artifacts/recheck.json` |
| The output head can be deleted | tied head, zero new parameters, 0 against 100.7M at paper scale | P5 E4, `artifacts/train.json` |
| The byte head loses at this scale | 4.7197 against 2.6454 nats per token | P5 E4 |
| Gradient exists at step 0, for both objectives | untrained tied head is an 83.43% autoencoder | P5 E5 |
| Tying costs 12.20% invalid UTF-8 | 49,551 predictions | P5 E6 |
| The 1M vocabulary claim is untestable here | rarity matched control scores zero too | P5 E7, `artifacts/openvocab.json` |
| The window is 92 to 95% zeros | per lane occupancy | P3 E1, `artifacts/window.json` |
| The cost is not script neutral | Latin 1.0, all nine Indic scripts exactly 3.0 bytes per character | P3 E2 |
| Truncation collisions are categorical | 0 English against 2,062 Indic groups at L=32 | P3 E3 |
| Collisions are bitwise identical embeddings | 200/200 pairs, max difference 0.0 | P3 E3 |
| Fix B helps Indic and hurts Latin | Malayalam 17.47 to 2.20%, Latin 0.18 to 1.48% | P3 E4, `artifacts/fixes.json` |
| The Indic high byte carries no information | 0.0000 bits, one distinct value | P3 E4c |
| Fix D fixes Indic and leaves Latin alone | Indic to 0.00%, Latin unchanged at 0.18% | P3 E4f |
| Reading both ends is nearly free and works | 2,122 to 217 colliding groups, 9.8x | P3 E7, `artifacts/bothends.json` |
| Memory and compute are not actually paid | 312.5 MB to 0.905 MB, 932x less arithmetic | P3 E6, `artifacts/cost.json` |
| Cost is proportional to token length | slope 502 ns per unit, correlation 0.9818 | P3 E6 |
| The downstream test could not work as designed | 6 of 250,475 tokens truncated | P3 E5, `artifacts/downstream.json` |

## Predictions that the runs refuted, and were kept

1. **P5 E3** predicted near chance decoding through the projection. Refuted: codes are k-sparse with
   mean k = 7.93, so recovery is compressed sensing and reaches 99.33%.
2. **P5 E5** predicted cross entropy would train and MSE would not. Refuted: both train, because the
   untrained tied head is already an autoencoder of its input.
3. **P3** predicted the downstream comparison would separate the codecs. Refuted, and for a reason
   that invalidated the experiment rather than the hypothesis.

## What is not claimed

The byte head is not shown to be better than a vocabulary head. The 1M vocabulary payoff is neither
demonstrated nor refuted, only shown to be untestable at this scale. Everything trained is 2 layers
at d_model=96 on CPU. Fix D's central assumption about Unicode block layout is measured in this
corpus and is not a property of Unicode.

## Reproducing

`python run_demo.py` in either problem folder. No network, no GPU, numpy and tokenizers only.
Each writeup's numbers regenerate from the committed artefacts with `python src/evidence.py`.
