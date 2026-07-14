#!/usr/bin/env python3
"""Augment metrics.json with richer per-language token statistics.

Loads the already-exported tokenizer.json (does NOT retrain, so the tuned
score is untouched) and, for each language's corpus, computes:
  characters, UTF-8 bytes, faithful units, tokens, fertility (tokens/unit),
  chars-per-token, bytes-per-token, and the number of distinct token types used.

Writes a `token_stats` block into both tokenizer/metrics.json and
site/metrics.json so the widget can render a Token-Statistics-by-Language table.

    python3 scripts/token_stats.py
"""
from __future__ import annotations

import json
from pathlib import Path

import regex
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CORPUS = ROOT / "corpus"
LANGS = ["en", "hi", "te", "mr"]
UNIT_RE = regex.compile(r"[\p{L}\p{M}\p{N}]+|[^\s\p{L}\p{M}\p{N}]")
WS_RE = regex.compile(r"\s+")


def load_text(c: str) -> str:
    faithful = CORPUS / f"{c}.faithful.txt"
    return (faithful if faithful.exists() else DATA / f"{c}.txt").read_text(encoding="utf-8")


def main() -> None:
    tok = Tokenizer.from_file(str(ROOT / "tokenizer" / "tokenizer.json"))
    stats = {}
    for c in LANGS:
        text = load_text(c)
        ids = tok.encode(text).ids
        units = len(UNIT_RE.findall(text))
        chars = len(text)
        visible = len(WS_RE.sub("", text))
        nbytes = len(text.encode("utf-8"))
        tokens = len(ids)
        stats[c] = {
            "characters": chars,
            "visible_characters": visible,
            "bytes": nbytes,
            "faithful_units": units,
            "tokens": tokens,
            "fertility": tokens / units,
            "chars_per_token": chars / tokens,
            "bytes_per_token": nbytes / tokens,
            "distinct_token_types": len(set(ids)),
        }
        print(f"  {c}: chars={chars:7d} bytes={nbytes:7d} units={units:6d} "
              f"tokens={tokens:6d} fert={tokens/units:.4f} "
              f"chars/tok={chars/tokens:.2f} bytes/tok={nbytes/tokens:.2f} "
              f"types={len(set(ids))}")

    for d in ("tokenizer", "site"):
        p = ROOT / d / "metrics.json"
        m = json.loads(p.read_text(encoding="utf-8"))
        m["token_stats"] = stats
        p.write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote token_stats into tokenizer/metrics.json + site/metrics.json")


if __name__ == "__main__":
    main()
