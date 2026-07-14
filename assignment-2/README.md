# Assignment 2 - Faithful multilingual BPE tokenizer (India, 4 languages)

One shared **10,000-token BPE tokenizer** for the **India** Wikipedia article in
**English, Hindi, Telugu and Marathi**, trained on a **faithful-Markdown** corpus.
The goal is a small spread between each language's tokens-per-unit ratio (with
English at or below 1.2) **subject to the hard requirement that the tokenizer is
faithful**: `decode(encode(text))` preserves every visible character of the input.

## Faithfulness first (why the earlier attempt scored 0)

A character-level tokenizer silently drops any character it never saw in
training (a URL `#`, an underscore `_`, …), so `decode(encode(text))` loses
visible text and the score is void. This tokenizer is a **byte-fallback BPE**
(SentencePiece / Llama style):

```
subword (BPE merges)  ->  character atoms  ->  byte fallback
```

- Seen characters (incl. every Devanagari / Telugu letter) are single atoms, so
  Indic text stays compact - unlike a pure byte-level tokenizer, which spends 3
  tokens per Indic character.
- Any character the tokenizer never saw falls back to its raw UTF-8 bytes, so
  **`decode(encode(text)) == text` holds for arbitrary input** - Markdown, URLs,
  punctuation, apostrophes, digit separators, emoji, any Unicode.
- **No NFKC.** NFKC is lossy (it re-spells `"` -> `''`, `(i)` -> `i`); byte
  fallback already gives full coverage, so it is omitted and the round trip is
  exact.

The build refuses to export unless the round trip holds, and `verify.py` re-checks
it independently.

## Result

Fertility X = `tokens / faithful_units`, where a **faithful unit** is one Unicode
letter/mark/number run OR one visible punctuation/symbol character - the exact
denominator the assignment's score is defined on.

| Language | Script | Faithful units | Tokens | X = tokens/unit |
|---|---|---:|---:|---:|
| Hindi | Devanagari | 72,170 | 55,834 | **0.7736** (X min) |
| English | Latin | 157,936 | 123,111 | 0.7795 |
| Marathi | Devanagari | 24,384 | 19,354 | 0.7937 |
| Telugu | Telugu | 29,315 | 23,537 | **0.8029** (X max) |

- English 0.7795 ≤ 1.2 ✓ and Hindi 0.7736 ≤ 1.2 ✓ (no Hindi penalty)
- spread = X max − X min = 0.8029 − 0.7736 = **0.0293**
- **score = 1000 / spread = 34,183** (Hindi-penalty factor 1.000, so adjusted = raw)

Numbers are on the shipped corpus snapshot in `corpus/` and are reproduced
exactly by `verify.py`.

## Method

- **Model:** HuggingFace `BPE(byte_fallback=True)`, vocab 10,000, `min_frequency=1`.
- **Pre-tokenizer / decoder:** Metaspace (`▁`), + ByteFallback decode.
- **Normalizer:** none (byte-perfect faithful).
- **Balancing:** the four languages are up-weighted during training
  (`en:2, hi:2, te:4, mr:5`) so their ratios line up and the spread is minimized -
  tuned by a small weight search in `hf_build.py`.

## Reproduce / verify

```bash
cd assignment-2
pip install tokenizers regex requests beautifulsoup4 lxml html2text py_mini_racer

# 1. fetch the faithful-Markdown corpus (India page, en/hi/te/mr) -> corpus/
python3 scripts/build_faithful_corpus.py

# 2. train + tune + export tokenizer.json and metrics.json (deterministic)
python3 scripts/hf_build.py

# 3. add per-language token statistics to metrics.json (no retrain)
python3 scripts/token_stats.py

# 4. independently verify faithfulness AND the score (the assignment's metric)
python3 scripts/verify.py

# 5. confirm the widget's in-browser tokenizer matches Python token-for-token
python3 scripts/test_widget_js.py
```

`verify.py` loads the exported `tokenizer.json` with HuggingFace
`Tokenizer.from_file()`, asserts `decode(encode(x)) == x` on the four texts plus
Markdown/URL/emoji samples, then reproduces the faithful-unit ratios, spread and
score (the same unit regex and 1000/spread formula the assignment defines).

## Files

```
corpus/            faithful-Markdown snapshots (en/hi/te/mr .faithful.txt/.md)
data/              earlier plain-text extracts (fallback if corpus/ is absent)
scripts/
  build_faithful_corpus.py  fetch + HTML->Markdown the four India pages
  hf_build.py               train the byte-fallback BPE + tune weights + export
  token_stats.py            add per-language token statistics to metrics.json
  verify.py                 standalone verifier: faithfulness + the assignment score
  test_widget_js.py         prove the widget's JS tokenizer == Python, token-for-token
tokenizer/         tokenizer.json (HF format) + metrics.json
site/              the deployable widget (see below)
```

## The widget

`site/` is a self-contained static site. It loads the exported `tokenizer.json`
and **tokenizes live in the browser** with a faithful JavaScript reimplementation
of the byte-fallback BPE (verified token-for-token against the Python
`tokenizers` library by `test_widget_js.py`). It shows the per-language ratios,
the spread and score, **per-language token statistics** (chars/bytes per token,
distinct token types), a live tokenizer playground with a round-trip check, a
searchable vocabulary, a **live sample of the raw `tokenizer.json`** (so the
encoding algorithm and config are inspectable, not just a vocab list), and
one-click tokenizer downloads.

```bash
cd site
python3 -m http.server 8000   # then open http://localhost:8000
```

Deploy: it is a plain static folder - drag `site/` onto
<https://app.netlify.com/drop> (or any static host). The downloaded
`tokenizer.json` loads directly with `Tokenizer.from_file()`.
