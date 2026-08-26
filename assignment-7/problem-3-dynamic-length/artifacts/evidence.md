# Problem 3, evidence

Every number below is read from `artifacts/*.json`. Regenerate with `python src/evidence.py`.

Corpus: 17,659,426 characters, 6,812 token types, 319,479 word types across 7 lanes.


## E1, occupancy: how much of the window is used

| lane | tokens | mean bytes/token | occupancy at L=32 | zero columns |
|---|---|---|---|---|
| agentic | 901,985 | 2.22 | 6.95% | 93.05% |
| code | 1,894,126 | 1.64 | 5.11% | 94.89% |
| indic | 5,488,811 | 2.27 | 7.09% | 92.91% |
| long_ctx | 1,223,189 | 2.53 | 7.90% | 92.10% |
| math | 904,084 | 2.32 | 7.25% | 92.75% |
| reasoning | 665,508 | 2.26 | 7.06% | 92.94% |
| web | 1,124,430 | 2.68 | 8.37% | 91.63% |

## E2, characters per window by script

| script | bytes/char | chars in 32 bytes | graphemes in 32 bytes | capacity vs Latin |
|---|---|---|---|---|
| BENGALI | 3.000 | 10.7 | 6.1 | 0.33x |
| DEVANAGARI | 3.000 | 10.7 | 5.8 | 0.33x |
| GUJARATI | 3.000 | 10.7 | 6.0 | 0.33x |
| GURMUKHI | 3.000 | 10.7 | 6.4 | 0.33x |
| HAN | 3.000 | 10.7 | 10.7 | 0.33x |
| KANNADA | 3.000 | 10.7 | 5.4 | 0.33x |
| MALAYALAM | 3.000 | 10.7 | 5.0 | 0.33x |
| ORIYA | 3.000 | 10.7 | 5.6 | 0.33x |
| TAMIL | 3.000 | 10.7 | 5.2 | 0.33x |
| TELUGU | 3.000 | 10.7 | 5.2 | 0.33x |
| ARABIC | 2.000 | 16.0 | 16.0 | 0.50x |
| COMMON | 1.019 | 31.4 | 31.4 | 0.98x |
| LATIN | 1.000 | 32.0 | 32.0 | 1.00x |

## E3, truncation collisions at L=32 (word types, prose lanes)

| script | word types | cropped | in collisions | colliding groups |
|---|---|---|---|---|
| MALAYALAM | 12,063 | 42.69% | 17.47% | 687 |
| TAMIL | 10,825 | 34.88% | 13.06% | 491 |
| KANNADA | 9,895 | 29.42% | 6.31% | 233 |
| TELUGU | 12,938 | 20.63% | 5.26% | 248 |
| ORIYA | 6,275 | 11.49% | 2.68% | 74 |
| BENGALI | 13,413 | 9.95% | 2.53% | 144 |
| DEVANAGARI | 17,757 | 11.21% | 2.15% | 137 |
| GUJARATI | 8,249 | 6.13% | 1.31% | 46 |
| LATIN | 84,044 | 1.26% | 0.18% | 60 |
| GURMUKHI | 5,001 | 2.14% | 0.08% | 2 |
| ARABIC | 4,445 | 0.00% | 0.00% | 0 |
| COMMON | 12,572 | 1.36% | 0.00% | 0 |

- **L=16**: nine Indic scripts 11,981 colliding groups across 96,416 word types; English prose (web + long_ctx + reasoning) **10** across 75,740.

- **L=32**: nine Indic scripts 2,062 colliding groups across 96,416 word types; English prose (web + long_ctx + reasoning) **0** across 75,740.

- **L=64**: nine Indic scripts 6 colliding groups across 96,416 word types; English prose (web + long_ctx + reasoning) **0** across 75,740.

Bitwise check: 200/200 sampled colliding pairs produce identical codec vectors, max absolute difference 0.0. Verdict: confirmed.


## E4, the fixes compared at equal D = 8192

| script | byte L=32 | fix B codepoint L=16 | fix D script-relative L=32 |
|---|---|---|---|
| MALAYALAM | 17.47% | 2.20% | 0.00% |
| TAMIL | 13.06% | 0.77% | 0.00% |
| KANNADA | 6.31% | 0.17% | 0.00% |
| TELUGU | 5.26% | 0.16% | 0.00% |
| ORIYA | 2.68% | 0.10% | 0.03% |
| BENGALI | 2.53% | 0.04% | 0.00% |
| DEVANAGARI | 2.15% | 0.03% | 0.01% |
| GUJARATI | 1.31% | 0.07% | 0.00% |
| LATIN | 0.18% | 1.48% | 0.18% |
| GURMUKHI | 0.08% | 0.00% | 0.00% |
| ARABIC | 0.00% | 0.00% | 0.00% |
| COMMON | 0.00% | 0.18% | 0.16% |

### E4c, why fix B wastes half its dimensions

| script | high-digit entropy (bits, max 8) | distinct high digits |
|---|---|---|
| ARABIC | -0.0000 | 1 |
| BENGALI | -0.0000 | 1 |
| DEVANAGARI | -0.0000 | 1 |
| GUJARATI | -0.0000 | 1 |
| GURMUKHI | -0.0000 | 1 |
| KANNADA | -0.0000 | 1 |
| MALAYALAM | -0.0000 | 1 |
| ORIYA | -0.0000 | 1 |
| TAMIL | -0.0000 | 1 |
| TELUGU | -0.0000 | 1 |
| LATIN | 0.0006 | 5 |
| COMMON | 0.0897 | 24 |
| HAN | 5.3710 | 79 |

Fix D costs, measured: 3,214 cross-script alias groups if the script tag is dropped; with the tag, residual collisions remain only where a word mixes scripts.


## E6, what the window costs, three ways

| window | dimensions D | zeros | dense memory | factored memory | ratio |
|---|---|---|---|---|---|
| L=16 | 4,096 | 51.32% | 156.2 MB | 0.823 MB | 190x |
| L=32 | 8,192 | 72.32% | 312.5 MB | 0.905 MB | 345x |
| L=64 | 16,384 | 85.73% | 625.0 MB | 0.926 MB | 675x |
| L=128 | 32,768 | 92.84% | 1250.0 MB | 0.928 MB | 1346x |

| window | arithmetic reduction | wall-clock speedup |
|---|---|---|
| L=16 | 527x | 1.17x |
| L=32 | 932x | 2.07x |
| L=64 | 1812x | 3.75x |
| L=128 | 3609x | 7.03x |

Cost against token length: slope **424.7 ns per unit**, correlation **0.9862**. A flat line would refute the dynamic claim.

| window | time for a short token, relative to L=32 | projection W parameters |
|---|---|---|
| L=16 | 1.002x | 393,216 |
| L=32 | 1.000x | 786,432 |
| L=64 | 0.997x | 1,572,864 |
| L=128 | 1.035x | 3,145,728 |

## E7, reading the word from both ends

| scheme | colliding groups | Malayalam | reduction |
|---|---|---|---|
| 16 leading bytes plus 16 trailing bytes, same D | 217 | 1.14% | 9.8x |
| 16 front + 16 back bytes, each cut moved to a character boundary | 419 | 2.61% | 5.1x |
| 15 front and 16 back bytes plus a checksum byte of the discarded middle | 3 | 0.02% | 707.3x |
| script tag plus the first 31 characters | 63 | 0.00% | 33.7x |
| script tag, 15 leading plus 16 trailing characters | 39 | 0.00% | 54.4x |
| 31 leading bytes plus a checksum byte of everything discarded | 48 | 0.36% | 44.2x |
| the published construction: first 32 bytes | 2,122 | 17.47% | 1.0x |

### E7 verified as a codec, not only as a key

- Round trip, published prefix: 1.0000. Both ends: 1.0000.
- Bitwise: 217/217 colliding pairs produce identical vectors, max difference 0.0. Verdict: confirmed.

| script | prefix cut lands mid-character | both ends | both cuts aligned |
|---|---|---|---|
| BENGALI | 99.40% | 99.55% | 0.00% |
| DEVANAGARI | 99.55% | 99.70% | 0.00% |
| GUJARATI | 99.60% | 99.41% | 0.00% |
| GURMUKHI | 95.33% | 98.13% | 0.00% |
| KANNADA | 99.86% | 99.79% | 0.00% |
| LATIN | 2.27% | 3.59% | 0.00% |
| MALAYALAM | 99.67% | 99.88% | 0.00% |
| ORIYA | 99.72% | 99.72% | 0.00% |
| TAMIL | 99.42% | 99.81% | 0.00% |
| TELUGU | 99.78% | 99.78% | 0.00% |

Aligning both cuts costs capacity: 32.00 units retained against 30.12 aligned.


## Which window to use

| scheme | L | D | projection parameters | colliding groups |
|---|---|---|---|---|
| both ends + hash | 16 | 4,096 | 393,216 | 152 |
| both ends + hash | 32 | 8,192 | 786,432 | 3 |
| both ends + hash | 64 | 16,384 | 1,572,864 | 0 |
| both ends + hash | 128 | 32,768 | 3,145,728 | 0 |
| published prefix | 16 | 4,096 | 393,216 | 12,436 |
| published prefix | 32 | 8,192 | 786,432 | 2,122 |
| published prefix | 64 | 16,384 | 1,572,864 | 10 |
| published prefix | 128 | 32,768 | 3,145,728 | 0 |

## E5, downstream, and why the token-level version is null


**indic** (noise floor sd 0.0416 nats/token)

| arm | loss per token | delta vs byte | verdict |
|---|---|---|---|
| byte | 2.6367 | +0.0000 | not established, inside the seed noise floor |
| codepoint | 2.6512 | +0.0144 | not established, inside the seed noise floor |
| script_relative | 2.6498 | +0.0130 | not established, inside the seed noise floor |

Exposure: of 250,475 token occurrences, byte truncates 6 (0.0024%), codepoint 0, script-relative 0.

**web** (noise floor sd 0.0077 nats/token)

| arm | loss per token | delta vs byte | verdict |
|---|---|---|---|
| byte | 4.7894 | +0.0000 | not established, inside the seed noise floor |
| codepoint | 4.7937 | +0.0043 | not established, inside the seed noise floor |
| script_relative | 4.7896 | +0.0002 | not established, inside the seed noise floor |

Exposure: of 1,124,430 token occurrences, byte truncates 0 (0.0000%), codepoint 0, script-relative 0.

### E5b, the same test at word level, where truncation is present

| lane | arm | word types representable | exact full-word | targets truncated |
|---|---|---|---|---|
| indic | byte | 88.96% | 0.102% | 4.01% |
| indic | script_relative | 100.00% | 0.611% | 0.00% |
| web | byte | 99.99% | 4.794% | 0.00% |
| web | script_relative | 100.00% | 4.904% | 0.00% |

- indic: script-relative minus byte = +0.0051, seed noise sd 0.0013, exceeds noise: True

- web: script-relative minus byte = +0.0011, seed noise sd 0.0078, exceeds noise: False
