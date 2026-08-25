"""
Reading the assignment-2 tokenizer as a list of concrete surface strings.

The codec encodes *the bytes a token stands for*, so the vocabulary has to be turned back into text
before anything can be measured. Two details in this tokenizer make that non-obvious:

  * It is a Metaspace tokenizer, so a leading space is written as U+2581 (LOWER ONE EIGHTH BLOCK).
    Left alone it would be measured as a 3-byte character in every word-initial token, which would
    inflate every byte count in Problem 3 by a constant that has nothing to do with script.

  * `byte_fallback` is on, so ids 0..255 are the literal tokens `<0x00>`..`<0xFF>`. Those are single
    bytes wearing a six-character costume. Measuring them as written would say a byte token is six
    bytes long.

Both are handled here, once, so no experiment has to remember them.
"""
import json

METASPACE = "▁"


def load(path):
    """Return `(tokens, meta)` where tokens is a list of `(id, surface_text, is_byte_fallback)`."""
    with open(path, encoding="utf-8") as fh:
        spec = json.load(fh)
    model = spec["model"]
    vocab = model["vocab"]
    out = []
    for text, tid in sorted(vocab.items(), key=lambda kv: kv[1]):
        if len(text) == 6 and text.startswith("<0x") and text.endswith(">"):
            try:
                out.append((tid, bytes([int(text[3:5], 16)]).decode("latin-1"), True))
                continue
            except ValueError:
                pass
        out.append((tid, text.replace(METASPACE, " "), False))
    meta = {
        "path": path,
        "size": len(out),
        "type": model.get("type"),
        "byte_fallback": bool(model.get("byte_fallback")),
        "merges": len(model.get("merges", [])),
    }
    return out, meta


def real_tokens(tokens):
    """Vocabulary entries that stand for actual text, dropping the 256 byte-fallback singletons.

    The byte tokens are excluded from every reported rate because they are guaranteed to be one byte
    long and would flatter any occupancy or round-trip statistic by 2.56% of the vocabulary.
    """
    return [(tid, text) for tid, text, is_bf in tokens if not is_bf]
