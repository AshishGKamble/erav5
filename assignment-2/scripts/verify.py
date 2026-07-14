#!/usr/bin/env python3
"""Standalone verifier for the faithful BPE tokenizer.

Loads the exported HuggingFace tokenizer.json and, on the four India texts:
  1. asserts decode(encode(text)) preserves every visible (non-whitespace)
     character  -> the faithful-tokenizer requirement, incl. the exact
     Markdown/URL sample that failed before;
  2. reproduces the assignment's faithful-unit fertilities, spread and score
     (the same unit regex and 1000/spread formula the assignment defines), and
     checks them against metrics.json.

    pip install tokenizers regex
    python3 verify.py
"""
import json
import math
from pathlib import Path

import regex
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CORPUS = ROOT / "corpus"
LANGS = ["en", "hi", "te", "mr"]


def load_text(c):
    faithful = CORPUS / f"{c}.faithful.txt"
    return (faithful if faithful.exists() else DATA / f"{c}.txt").read_text(encoding="utf-8")
UNIT_RE = regex.compile(r"[\p{L}\p{M}\p{N}]+|[^\s\p{L}\p{M}\p{N}]")
NONWS = regex.compile(r"\s+")

SAMPLES = [
    "https://hi.wikipedia.org/wiki/भारत#cite_ref-1",   # the exact failing sample
    "India (# 1) - [a](b), \"q\", it's 3,000_000; x<y & p>q | 50% `code` _i_ *b*",
    "भारत గణతంత్ర मराठी — emoji 🇮🇳 表 │ € ″ ⓘ",
]


def nonws(s):
    return NONWS.sub("", s)


def main():
    tok = Tokenizer.from_file(str(ROOT / "tokenizer" / "tokenizer.json"))
    texts = {c: load_text(c) for c in LANGS}

    print(f"tokenizer: vocab={tok.get_vocab_size()}\n")

    # 1. Faithfulness ----------------------------------------------------
    faithful = True
    for c in LANGS:
        if nonws(tok.decode(tok.encode(texts[c]).ids)) != nonws(texts[c]):
            faithful = False
            print(f"  ROUND-TRIP FAILED on full {c}.txt")
    for s in SAMPLES:
        d = tok.decode(tok.encode(s).ids)
        if nonws(d) != nonws(s):
            faithful = False
            print(f"  ROUND-TRIP FAILED on: {s!r}\n     got: {d!r}")
    print(f"  faithful (decode(encode(x)) keeps visible chars, 4 texts + samples): {faithful}")

    # 2. Assignment's faithful-unit score --------------------------------
    ratios = {}
    for c in LANGS:
        units = len(UNIT_RE.findall(texts[c]))
        tokens = len(tok.encode(texts[c]).ids)
        ratios[c] = tokens / units
        print(f"  {c}: tokens={tokens:6d} faithful_units={units:6d} fertility={ratios[c]:.4f}")

    order = sorted(LANGS, key=lambda c: ratios[c])
    spread = ratios[order[-1]] - ratios[order[0]]
    score = 1000 / spread
    hindi_penalty = math.exp(max(0.0, ratios["hi"] / 1.2 - 1.0))
    print(f"\n  min = {order[0]} ({ratios[order[0]]:.4f}), max = {order[-1]} ({ratios[order[-1]]:.4f})")
    print(f"  English <= 1.2 : {ratios['en'] <= 1.2}  (English = {ratios['en']:.4f})")
    print(f"  spread = {spread:.4f}")
    print(f"  SCORE = 1000 / spread = {score:.1f}")
    print(f"  hindi_penalty_factor = {hindi_penalty:.4f}  adjusted = {score / hindi_penalty:.1f}")

    metrics_path = ROOT / "tokenizer" / "metrics.json"
    if metrics_path.exists():
        claimed = json.loads(metrics_path.read_text(encoding="utf-8"))
        ok = abs(claimed["score"] - score) < 0.5 and faithful
        print(f"\n  claimed score = {claimed['score']:.1f}  -> {'MATCH' if ok else 'MISMATCH'}")
    if not faithful:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
