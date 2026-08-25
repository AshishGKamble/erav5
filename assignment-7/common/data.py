"""
Turning the frozen corpus into packed token batches.

Assignment 6 has a full packing and ledger system; this is a deliberately smaller thing. It exists
so the two problems here can train on real text from the same frozen corpus with the same segment
rules, without dragging in the ledger, the firewall or the shard manifest, none of which these
experiments make any claim about.

The one rule kept from assignment 6, because dropping it would quietly corrupt every loss reported:
**attention and targets are masked by segment**, so a token never attends across a document boundary
and the last token of a document is never a training target.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import corpus  # noqa: E402


def tokenize_lane(corpus_root, tokenizer_path, lane, limit=None, max_docs=None):
    """Token id sequences for one lane, in file order so runs are reproducible."""
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(tokenizer_path)
    docs = []
    path = os.path.join(corpus_root, lane + ".jsonl")
    for i, text in enumerate(corpus.read_lane(path, limit=limit)):
        docs.append(np.asarray(tok.encode(text).ids, dtype=np.int64))
        if max_docs and len(docs) >= max_docs:
            break
    return docs


def pack(docs, T, max_seqs=None):
    """Pack documents into fixed-length sequences, one segment id per document.

    Segment id 0 is reserved for padding, so documents are numbered from 1. A document longer than
    T is split across sequences and each piece gets its own segment id, because the pieces genuinely
    cannot attend to each other and pretending otherwise would be the cross-document bug in a
    different costume.
    """
    seqs_i, seqs_p, seqs_s = [], [], []
    cur_i = np.zeros(T, dtype=np.int64)
    cur_p = np.zeros(T, dtype=np.int64)
    cur_s = np.zeros(T, dtype=np.int64)
    fill, seg = 0, 0

    def flush():
        nonlocal cur_i, cur_p, cur_s, fill, seg
        if fill:
            seqs_i.append(cur_i.copy())
            seqs_p.append(cur_p.copy())
            seqs_s.append(cur_s.copy())
        cur_i = np.zeros(T, dtype=np.int64)
        cur_p = np.zeros(T, dtype=np.int64)
        cur_s = np.zeros(T, dtype=np.int64)
        fill, seg = 0, 0

    for doc in docs:
        off = 0
        while off < len(doc):
            if fill >= T:
                flush()
                if max_seqs and len(seqs_i) >= max_seqs:
                    return (np.stack(seqs_i), np.stack(seqs_p), np.stack(seqs_s))
            room = T - fill
            take = min(room, len(doc) - off)
            seg += 1
            cur_i[fill:fill + take] = doc[off:off + take]
            cur_p[fill:fill + take] = np.arange(take)
            cur_s[fill:fill + take] = seg
            fill += take
            off += take
    flush()
    if not seqs_i:
        raise ValueError("no sequences packed")
    out = (np.stack(seqs_i), np.stack(seqs_p), np.stack(seqs_s))
    if max_seqs:
        out = tuple(a[:max_seqs] for a in out)
    return out


def split(arrays, val_fraction=0.2):
    """Deterministic tail split. No shuffling: the tail of the file is held out, whole."""
    n = arrays[0].shape[0]
    n_val = max(1, int(round(n * val_fraction)))
    n_tr = n - n_val
    if n_tr < 1:
        raise ValueError("not enough sequences to split")
    return tuple(a[:n_tr] for a in arrays), tuple(a[n_tr:] for a in arrays)


def batches(arrays, batch_size, rng=None, epochs=1):
    """Yield batches. With `rng` the order is shuffled per epoch, otherwise it is file order."""
    n = arrays[0].shape[0]
    for _ in range(epochs):
        order = rng.permutation(n) if rng is not None else np.arange(n)
        for s in range(0, n, batch_size):
            sel = order[s:s + batch_size]
            if sel.size:
                yield tuple(a[sel] for a in arrays)
