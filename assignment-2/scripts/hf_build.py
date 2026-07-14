#!/usr/bin/env python3
"""Build the faithful BPE tokenizer (byte_fallback) and tune training weights.

Design:
  - Model:        HuggingFace BPE with byte_fallback=True (subword -> char -> byte)
  - Normalizer:   none (byte-perfect faithful; NFKC would re-spell chars = lossy)
  - Pre-tokenizer/Decoder: Metaspace ("_" marker), + ByteFallback decode
  - Vocab:        10000, min_frequency=1
  - The 256 byte tokens are demoted to NON-special so the default decode()
    reconstructs them -> decode(encode(text)) preserves every visible character.

Score metric (the assignment's): fertility = tokens / faithful_units, where a
faithful unit is one Unicode letter/mark/number run OR one visible punctuation/
symbol char; score = 1000 / (max_fertility - min_fertility), English <= 1.2.
"""
from __future__ import annotations

import itertools
import json
import math
import tempfile
from pathlib import Path

import regex
from tokenizers import Tokenizer, decoders
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Metaspace
from tokenizers.trainers import BpeTrainer

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CORPUS = ROOT / "corpus"
LANGS = ["en", "hi", "te", "mr"]
LANG_NAMES = {"en": "English", "hi": "Hindi", "te": "Telugu", "mr": "Marathi"}
VOCAB = 10000
BYTE_TOKS = [f"<0x{b:02X}>" for b in range(256)]
UNIT_RE = regex.compile(r"[\p{L}\p{M}\p{N}]+|[^\s\p{L}\p{M}\p{N}]")
NONWS = regex.compile(r"\s+")


def load_text(c):
    """Prefer the faithful-Markdown corpus; fall back to the plain-text extract."""
    faithful = CORPUS / f"{c}.faithful.txt"
    return (faithful if faithful.exists() else DATA / f"{c}.txt").read_text(encoding="utf-8")


texts = {c: load_text(c) for c in LANGS}
units = {c: len(UNIT_RE.findall(texts[c])) for c in LANGS}
print("corpus source:", "faithful markdown" if (CORPUS / "en.faithful.txt").exists()
      else "plain-text extract (clipped)")


def nonws(s: str) -> str:
    return NONWS.sub("", s)


def train_tokenizer(weights: dict) -> Tokenizer:
    with tempfile.TemporaryDirectory() as tmp:
        files = []
        for c in LANGS:
            p = Path(tmp) / f"{c}.txt"
            p.write_text(texts[c], encoding="utf-8")
            files += [str(p)] * weights[c]
        tok = Tokenizer(BPE(unk_token=None, byte_fallback=True))
        tok.pre_tokenizer = Metaspace(replacement="▁", prepend_scheme="never")
        tok.decoder = decoders.Sequence(
            [decoders.Replace("▁", " "), decoders.ByteFallback(), decoders.Fuse()]
        )
        tok.train(files, BpeTrainer(vocab_size=VOCAB, min_frequency=1,
                                    special_tokens=BYTE_TOKS))
    # Demote byte tokens to non-special so the DEFAULT decode() reconstructs them.
    j = json.loads(tok.to_str())
    for a in j["added_tokens"]:
        if a["content"].startswith("<0x"):
            a["special"] = False
    return Tokenizer.from_str(json.dumps(j))


def evaluate(tok: Tokenizer):
    ratios = {c: len(tok.encode(texts[c]).ids) / units[c] for c in LANGS}
    spread = max(ratios.values()) - min(ratios.values())
    score = 1000 / spread
    penalty = math.exp(max(0.0, ratios["hi"] / 1.2 - 1.0))
    faithful = all(nonws(tok.decode(tok.encode(texts[c]).ids)) == nonws(texts[c])
                   for c in LANGS)
    return ratios, spread, score, score / penalty, penalty, faithful


def search():
    """Maximize the Hindi-adjusted score (the assignment's headline number),
    keeping English <= 1.2 and faithful."""
    best = None
    grid = itertools.product([2, 3, 4], [1, 2, 3], [3, 4, 5, 6], [2, 3, 4, 5])
    for en, hi, te, mr in grid:
        w = {"en": en, "hi": hi, "te": te, "mr": mr}
        tok = train_tokenizer(w)
        ratios, spread, score, adj, penalty, faithful = evaluate(tok)
        if faithful and ratios["en"] <= 1.19 and (best is None or adj > best[1]):
            best = (w, adj, spread, ratios, score, penalty)
            print(f"  best ADJ={adj:9.1f} (raw={score:.1f} pen={penalty:.3f})  w={w}  " +
                  " ".join(f"{c}={ratios[c]:.4f}" for c in LANGS), flush=True)
    return best


def main():
    print("faithful_units:", units)
    best_w = search()[0]
    print("\n=== BEST ===")
    print("weights:", best_w)

    # Rebuild the winner and export
    tok = train_tokenizer(best_w)
    ratios, spread, score, adj, penalty, faithful = evaluate(tok)
    order = sorted(LANGS, key=lambda c: ratios[c])
    for c in LANGS:
        print(f"  {c}: fertility={ratios[c]:.4f}")
    print(f"spread={spread:.4f}  SCORE={score:.1f}  hindi_penalty={penalty:.4f}  "
          f"adjusted={adj:.1f}  english={ratios['en']:.4f}")
    tok.save(str(ROOT / "tokenizer" / "tokenizer.json"))
    tok.save(str(ROOT / "site" / "tokenizer.json"))
    metrics = {
        "method": "HuggingFace BPE, byte_fallback, Metaspace, no normalizer",
        "languages": LANG_NAMES,
        "weights": best_w,
        "vocab_size": tok.get_vocab_size(),
        "faithful_units": units,
        "unit_policy": "one Unicode letter/mark/number run OR one visible punctuation/symbol char",
        "token_counts": {c: len(tok.encode(texts[c]).ids) for c in LANGS},
        "ratios": ratios,
        "min_language": order[0],
        "max_language": order[-1],
        "spread": spread,
        "score": score,
        "hindi_penalty_factor": penalty,
        "hindi_adjusted_score": adj,
        "english_ok": ratios["en"] <= 1.2,
        "hindi_ok": ratios["hi"] <= 1.2,
        "faithful": faithful,
    }
    for d in ("tokenizer", "site"):
        (ROOT / d / "metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print("faithful:", faithful, "| wrote tokenizer.json + metrics.json")


if __name__ == "__main__":
    main()
