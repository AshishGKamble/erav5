"""
Tokenize the frozen corpus into immutable shards.

Immutable means: the same code run on a different machine on a different day produces
byte-identical shards. Three rules buy that, and all three are load-bearing.

1. **Documents are sorted by content-addressed id before assignment.** Not by corpus order.
   That makes a shard a pure function of the *set* of documents it contains, so even a reordered
   corpus file yields identical shards.
2. **Nothing observed at run time enters a shard.** No timestamps, no paths, no machine identity.
   The manifest records when a build happened; the shard itself cannot.
3. **Splits are derived from document content, never assigned by position.** See split_for().

Shard files are pure data. Every hash lives in the manifest, which is the single authority - so
validation means recomputing from the files on disk and comparing, and tampering with either the
data or its description is detectable from the other side.
"""
import io
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (CORPUS, FROZEN, SHARDS, canonical, ensure_dirs, read_json, read_jsonl,
                    sha256_bytes, sha256_file, write_json)

SHARD_TOKENS = 200_000          # target tokens per shard, before the last partial one
EVAL_PCT = 5                    # held out for evaluation - must never enter a loss-bearing batch
HOLDOUT_PCT = 3                 # a second reserve, never touched by either training or eval

# The Indic lane gets its own shards and is flagged protected. The lecture left this open -
# one shard mixed 10/90 with an OPUS override, or Indic kept separate? Separate, because:
# a mixed shard rejected by OPUS loses the other 90% with it; OPUS judges from the first ~512
# tokens, so a mixed shard's score is decided by whichever document happens to land first; and an
# override has to be attributable to a lane to be auditable. Agentic is protected for the reason
# plan.json gives - an agent trace opens with plan and tool boilerplate and reads like a log file.
PROTECTED_LANES = {"indic", "agentic"}


def split_for(doc_id):
    """Assign a split from the document's own content hash.

    Document-level and deterministic, so a document is always in the same split and can never sit
    in two. This is the structural fix for the failure A5 section 9.2 recorded: that proxy split a
    validation set off by *token offset* over a source it had already repeated six times, so every
    held-out window also sat in training and the reasoning metric was measuring memorisation. A
    position-based split can leak. A content-addressed one cannot.

    The salt keeps this bucket independent of the id prefix used elsewhere, so nothing else in the
    system accidentally correlates with the split.
    """
    bucket = int(sha256_bytes(f"split:{doc_id}".encode("utf-8"))[:8], 16) % 100
    if bucket < EVAL_PCT:
        return "eval"
    if bucket < EVAL_PCT + HOLDOUT_PCT:
        return "holdout"
    return "train"


def load_tokenizer():
    """Load the Assignment-2 tokenizer and return it with its hash.

    The hash is carried into every shard's identity, so a shard built with a different tokenizer
    is a different shard by construction - it cannot be silently mixed with these.
    """
    from tokenizers import Tokenizer
    path = os.path.join(FROZEN, "tokenizer.json")
    tok = Tokenizer.from_file(path)
    return tok, sha256_file(path)


def tokenize_doc(tok, doc):
    """Return (ids, spans) where spans are [start, end, loss] token ranges within the document.

    Span documents (reasoning, agentic) are tokenized **one span at a time**. Encoding the whole
    text in one call would be marginally cheaper and would let BPE merge across a role boundary,
    which would put a single token half in the assistant's turn and half in the tool's output.
    There is no correct loss mask for such a token. Per-span encoding keeps every boundary exact,
    and exact boundaries are the thing this lane exists to demonstrate.
    """
    if "spans" in doc:
        ids, spans = [], []
        for sp in doc["spans"]:
            enc = tok.encode(sp["text"]).ids
            if not enc:
                continue
            spans.append([len(ids), len(ids) + len(enc), int(sp["loss"])])
            ids.extend(enc)
        return ids, spans
    ids = tok.encode(doc["text"]).ids
    return ids, [[0, len(ids), 1]]


# Difficulty bands per lane. Indic already carries a register-derived band from fetch_corpus.py;
# every other lane gets one here from its token-length distribution within the lane.
#
# Length is a **proxy** for difficulty and is labelled as one. It is defensible - a longer source
# file has more to hold in context, a longer derivation has more steps - but it is not a measure of
# conceptual difficulty, and a serious system would score difficulty directly. What matters for
# this assignment is that the curriculum's band gating operates on a real per-document property
# rather than a field nothing reads.
BAND_QUANTILES = [0.15, 0.35, 0.60, 0.82, 0.95]     # -> B0 B1 B2 B3 B4 B5


def assign_bands(lane, lengths):
    """Return the length thresholds separating bands for this lane."""
    if not lengths:
        return []
    import numpy as _np
    return [float(_np.quantile(lengths, q)) for q in BAND_QUANTILES]


def band_for(n_tokens, thresholds):
    for i, t in enumerate(thresholds):
        if n_tokens <= t:
            return f"B{i}"
    return "B5"


def build_lane(tok, tok_sha, lane, docs):
    """Group one lane's documents into shards, one shard per (split, sequence) group."""
    out = []
    by_split = {"train": [], "eval": [], "holdout": []}
    for d in docs:
        by_split[split_for(d["id"])].append(d)

    # Band thresholds are computed over the WHOLE lane, before splitting, so a document's band
    # does not depend on which split it landed in.
    tokenized = {}
    for d in docs:
        ids, spans = tokenize_doc(tok, d)
        tokenized[d["id"]] = (ids, spans)
    thresholds = assign_bands(lane, [len(v[0]) for v in tokenized.values() if v[0]])

    for split, group in by_split.items():
        if not group:
            continue
        # Sorted by id: a shard is a function of its document set, not of corpus order.
        group.sort(key=lambda d: d["id"])
        buf, n_tok = [], 0
        idx = 0
        for d in group:
            ids, spans = tokenized[d["id"]]
            if not ids:
                continue
            meta = dict(d.get("meta", {}))
            # Indic keeps its register-derived band; every other lane gets the length proxy.
            meta.setdefault("band", band_for(len(ids), thresholds))
            d = dict(d, meta=meta)
            buf.append((d, ids, spans))
            n_tok += len(ids)
            if n_tok >= SHARD_TOKENS:
                out.append(_flush(tok_sha, lane, split, idx, buf))
                buf, n_tok, idx = [], 0, idx + 1
        if buf:
            out.append(_flush(tok_sha, lane, split, idx, buf))
    return out


def _flush(tok_sha, lane, split, idx, buf):
    """Write one shard's .bin and .json, and return its manifest record."""
    ensure_dirs(SHARDS)
    shard_id = f"{lane}-{split}-{idx:04d}"

    stream, docs, off = [], [], 0
    loss_tokens, chars, licences = 0, 0, set()
    for d, ids, spans in buf:
        docs.append({"id": d["id"], "off": off, "len": len(ids), "spans": spans,
                     "meta": d.get("meta", {})})
        stream.extend(ids)
        loss_tokens += sum(e - s for s, e, l in spans if l)
        chars += d.get("chars", 0)
        lic = d.get("meta", {}).get("licence")
        if lic:
            licences.add(lic)
        off += len(ids)

    # uint16 little-endian: the Assignment-2 vocabulary is 10,000, so ids fit with room to spare,
    # and fixing the width and byte order keeps the file identical on any architecture.
    blob = np.asarray(stream, dtype="<u2").tobytes()
    bin_path = os.path.join(SHARDS, f"{shard_id}.bin")
    with io.open(bin_path, "wb") as fh:
        fh.write(blob)

    index = {"shard_id": shard_id, "lane": lane, "split": split,
             "protected": lane in PROTECTED_LANES, "tokenizer_sha256": tok_sha,
             "n_docs": len(docs), "n_tokens": off, "loss_tokens": loss_tokens,
             "raw_chars": chars, "licences": sorted(licences), "docs": docs}
    idx_path = os.path.join(SHARDS, f"{shard_id}.json")
    write_json(idx_path, index)

    content_sha = sha256_bytes(blob)
    index_sha = sha256_bytes(canonical(index))
    # Identity binds the split and the tokenizer into the hash. Relabelling an eval shard as
    # train changes the shard's identity, so the firewall cannot be defeated by editing a tag.
    shard_sha = sha256_bytes(canonical({
        "shard_id": shard_id, "lane": lane, "split": split,
        "protected": lane in PROTECTED_LANES, "tokenizer_sha256": tok_sha,
        "content_sha256": content_sha, "index_sha256": index_sha}))

    return {"shard_id": shard_id, "lane": lane, "split": split,
            "protected": lane in PROTECTED_LANES, "n_docs": len(docs), "n_tokens": off,
            "loss_tokens": loss_tokens, "raw_chars": chars, "licences": sorted(licences),
            "fertility": round(off / chars, 4) if chars else None,
            "bin": f"{shard_id}.bin", "index": f"{shard_id}.json",
            "content_sha256": content_sha, "index_sha256": index_sha,
            "tokenizer_sha256": tok_sha, "shard_sha256": shard_sha}


def build_all(log=print):
    """Tokenize every lane in the frozen corpus into shards. Returns the shard records."""
    sources = read_json(os.path.join(CORPUS, "SOURCES.json"))
    tok, tok_sha = load_tokenizer()
    log(f"  tokenizer sha256 {tok_sha[:16]}...  vocab {tok.get_vocab_size()}")

    records = []
    for entry in sources["lanes"]:
        lane = entry["lane"]
        docs = list(read_jsonl(os.path.join(CORPUS, entry["file"])))
        made = build_lane(tok, tok_sha, lane, docs)
        records.extend(made)
        ntok = sum(r["n_tokens"] for r in made)
        splits = {}
        for r in made:
            splits[r["split"]] = splits.get(r["split"], 0) + r["n_tokens"]
        log(f"  {lane:10} {len(made):3d} shards  {ntok:9,d} tokens  "
            f"fertility {ntok / max(1, sum(r['raw_chars'] for r in made)):.3f}  "
            f"{ {k: f'{v:,}' for k, v in sorted(splits.items())} }")
    return records


def load_shard_tokens(shard_id):
    """Read a shard's token stream back as a numpy array."""
    return np.fromfile(os.path.join(SHARDS, f"{shard_id}.bin"), dtype="<u2")


if __name__ == "__main__":
    recs = build_all()
    print(f"\n{len(recs)} shards, {sum(r['n_tokens'] for r in recs):,} tokens")
