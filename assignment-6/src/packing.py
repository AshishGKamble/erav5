"""
Per-lane packing policies, with correct loss masks, attention masks and position ids.

The brief asks for "packing policies for different data types", and the lecture was explicit that
code and agentic data want different treatment. Seven lanes, four policies, each chosen for a
reason that survives being asked "why not just concatenate everything?":

  CONCAT       web            split freely; a document continues into the next sequence. Web is
                              abundant and a severed sentence costs almost nothing, so take the
                              utilisation.
  DOC_ALIGNED  code, math,    keep a document whole when it fits, several per sequence. A document
               indic          larger than the window is read in contiguous windows rather than
                              dropped - a large source file is legitimately read in pieces.
  ATOMIC       reasoning,     never truncate a *turn*. A document that fits goes in whole; one that
               agentic        does not is split **only at span boundaries**, so a chain of thought or
                              a tool call is never cut mid-way (A5 section 11: a truncated chain of
                              thought teaches a wrong method, and half a tool call is not a tool
                              call). Only a single span larger than the window is undroppable, and
                              that is recorded.
  LONG_DOC     long_ctx       one document per sequence, read in contiguous windows with positions
                              continuing across them. Never co-packed, because the point of the lane
                              is long-range structure and co-packing would fragment it.

**Policy is the splitting rule; ordering is a separate axis.** Code is ordered so files from one
repository pack together - cross-file attention inside a repo is real context worth having, and
block-diagonal masking still prevents cross-repo leakage. Indic is ordered by difficulty band so a
sequence has one band, which is what lets the curriculum gate it.

An earlier cut dropped every oversize document. At a 1024-token window that silently deleted the
entire long-context lane - 9 documents in, 0 sequences out - and most of code and math with it. A
policy that quietly empties a lane is worse than one that truncates, because nothing in the numbers
says it happened. Oversize handling is now explicit per policy, and every drop is recorded.

Three masks travel with every sequence, and they are not interchangeable:

  loss_mask     which tokens are targets. Zero on padding, on prompts, and on tool output.
  segment_ids   which document each token belongs to. Attention is block-diagonal on this, so a
                token can never see across a document boundary. Zero means padding, which attends
                to nothing and is attended to by nothing.
  position_ids  restart at 0 for every document. Without this the second document in a sequence
                would be told it starts at position 700, which is a lie about where it is.

Forgetting position restart or segment isolation is the classic packing bug: the loss mask looks
right, the numbers look plausible, and the model is quietly trained on cross-document nonsense.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PAD_ID = 0                       # never carries loss, never attends, never attended to
SEQ_LEN = 1024

CONCAT, DOC_ALIGNED, ATOMIC, LONG_DOC = "concat", "doc_aligned", "atomic", "long_doc"

POLICY = {
    "web": CONCAT,
    "code": DOC_ALIGNED,
    "math": DOC_ALIGNED,
    "reasoning": ATOMIC,
    "agentic": ATOMIC,
    "indic": DOC_ALIGNED,
    "long_ctx": LONG_DOC,
}

# How each policy orders documents before packing. Affinity, not splitting.
ORDER_BY = {"code": "repo", "indic": "band"}

POLICY_REASON = {
    "web": "split freely; boundary loss is cheap and utilisation is worth more",
    "code": "whole files where they fit, repo-ordered so related files share a window",
    "math": "whole documents, best fit; oversize read in windows",
    "reasoning": "never truncate a trace; oversize is dropped and recorded",
    "agentic": "never truncate a trace; assistant spans carry loss, tool output does not",
    "indic": "whole documents, band-ordered so a sequence has one difficulty",
    "long_ctx": "one document per sequence, read in contiguous windows, positions continue",
}


class Packed:
    """One packed sequence, plus the provenance needed to explain every token in it."""

    __slots__ = ("input_ids", "loss_mask", "position_ids", "segment_ids", "provenance", "lane")

    def __init__(self, lane, seq_len):
        self.lane = lane
        self.input_ids = np.full(seq_len, PAD_ID, dtype=np.int32)
        self.loss_mask = np.zeros(seq_len, dtype=np.int8)
        self.position_ids = np.zeros(seq_len, dtype=np.int32)
        self.segment_ids = np.zeros(seq_len, dtype=np.int32)
        self.provenance = []

    @property
    def used(self):
        return int((self.segment_ids != 0).sum())

    @property
    def loss_tokens(self):
        return int(self.loss_mask.sum())

    @property
    def utilisation(self):
        return self.used / len(self.input_ids)

    def summary(self):
        return {"lane": self.lane, "used": self.used, "pad": len(self.input_ids) - self.used,
                "loss_tokens": self.loss_tokens, "docs": len(self.provenance),
                "utilisation": round(self.utilisation, 4),
                "provenance": self.provenance}


def _place(seq, cursor, seg, doc, ids, spans, tok_from):
    """Write one document (or the head of one) into a sequence at `cursor`.

    `tok_from` is the offset within the document, so a CONCAT document resumed in the next
    sequence keeps counting positions from where it left off rather than restarting mid-sentence.
    """
    n = len(ids)
    seq.input_ids[cursor:cursor + n] = ids
    seq.segment_ids[cursor:cursor + n] = seg
    seq.position_ids[cursor:cursor + n] = np.arange(tok_from, tok_from + n, dtype=np.int32)
    for s, e, loss in spans:
        if not loss:
            continue
        # Clip the span to the slice of the document actually placed here.
        a, b = max(s, tok_from), min(e, tok_from + n)
        if a < b:
            seq.loss_mask[cursor + (a - tok_from):cursor + (b - tok_from)] = 1
    seq.provenance.append({"shard_id": doc["shard_id"], "doc_id": doc["doc_id"],
                           "tok_start": tok_from, "tok_end": tok_from + n,
                           "seq_start": cursor, "seq_end": cursor + n, "segment": seg})
    return cursor + n


def _order(lane, docs):
    """Deterministic ordering per lane affinity. Ties always break on doc_id, so order is total."""
    key = ORDER_BY.get(lane)
    if key == "repo":
        return sorted(docs, key=lambda d: (d.get("meta", {}).get("repo") or "", d["doc_id"]))
    if key == "band":
        return sorted(docs, key=lambda d: (d.get("meta", {}).get("band") or "", d["doc_id"]))
    return sorted(docs, key=lambda d: d["doc_id"])


def pack(lane, docs, seq_len=SEQ_LEN):
    """Pack a lane's documents into sequences. Returns (sequences, dropped).

    `docs` are dicts with shard_id, doc_id, ids, spans, meta. `dropped` records documents no policy
    could place, reported rather than silently discarded - a silent drop is how a lane quietly
    stops being trained on.
    """
    pol = POLICY.get(lane, DOC_ALIGNED)
    seqs, dropped = [], []
    cur, cursor, seg = None, 0, 0

    def flush():
        nonlocal cur, cursor, seg
        if cur is not None and cursor > 0:
            seqs.append(cur)
        cur, cursor, seg = None, 0, 0

    def windows(doc, ids, spans, co_pack):
        """Emit a document larger than the window as contiguous windows.

        `tok_from` keeps counting, so position ids continue across the split instead of restarting
        mid-document and telling the model it is at position 0 when it is not.
        """
        nonlocal cur, cursor, seg
        off = 0
        while off < len(ids):
            if not co_pack or cur is None:
                flush()
                cur, cursor, seg = Packed(lane, seq_len), 0, 0
            room = seq_len - cursor
            take = min(room, len(ids) - off)
            seg += 1
            cursor = _place(cur, cursor, seg, doc, ids[off:off + take], spans, off)
            off += take
            if cursor >= seq_len:
                flush()

    for doc in _order(lane, docs):
        ids, spans = doc["ids"], doc["spans"]
        if len(ids) == 0:
            continue

        if pol == CONCAT:
            windows(doc, ids, spans, co_pack=True)
            continue

        if pol == LONG_DOC:
            # Never co-packed: the lane exists for long-range structure, so a window belongs to
            # exactly one document even when that wastes the tail.
            windows(doc, ids, spans, co_pack=False)
            flush()
            continue

        if len(ids) > seq_len:
            if pol == ATOMIC:
                # Split between complete spans. An earlier version refused to split at all and
                # dropped the document, which cost 34% of the agentic lane at a 512 window - and
                # once difficulty bands were derived from length, it starved the lane outright in
                # the late stages, because the hardest band is by construction the longest
                # documents. Splitting on span boundaries keeps every turn intact and drops nothing
                # except a single span too large to place.
                # The rule is not "never split a span", it is **never split a loss-bearing span**.
                # A masked context span - a system prompt full of function schemas, a tool dump -
                # can be divided without teaching the model anything wrong, because the model is
                # not trained to produce it. Only a target span must stay whole. Without this
                # distinction Glaive's system prompts, which alone exceed a 512-token window, took
                # 39% of the lane with them.
                pieces = []
                for a, b, l in spans:
                    if b - a <= seq_len or l:
                        pieces.append((a, b, l))
                    else:
                        for c in range(a, b, seq_len):
                            pieces.append((c, min(c + seq_len, b), l))
                groups, cur_g, cur_len = [], [], 0
                oversize_span = False
                for a, b, l in pieces:
                    n = b - a
                    if n > seq_len:
                        oversize_span = True      # a loss-bearing span too large to place
                        break
                    if cur_len + n > seq_len:
                        groups.append(cur_g)
                        cur_g, cur_len = [], 0
                    cur_g.append((a, b, l))
                    cur_len += n
                if cur_g:
                    groups.append(cur_g)
                if oversize_span or not groups:
                    dropped.append({"doc_id": doc["doc_id"], "len": len(ids), "seq_len": seq_len,
                                    "reason": "atomic policy: a single span exceeds the window"})
                    continue
                for grp in groups:
                    lo, hi = grp[0][0], grp[-1][1]
                    flush()
                    cur, cursor, seg = Packed(lane, seq_len), 0, 0
                    seg = 1
                    cursor = _place(cur, 0, 1, doc, ids[lo:hi],
                                    [[a, b, l] for a, b, l in grp], lo)
                    flush()
                continue
            windows(doc, ids, spans, co_pack=True)     # DOC_ALIGNED reads it in windows
            continue

        # Fits whole. Start a new sequence if it will not fit in what is left of this one.
        if cur is None:
            cur, cursor, seg = Packed(lane, seq_len), 0, 0
        if cursor + len(ids) > seq_len:
            flush()
            cur, cursor, seg = Packed(lane, seq_len), 0, 0
        seg += 1
        cursor = _place(cur, cursor, seg, doc, ids, spans, 0)

    flush()
    return seqs, dropped


def attention_mask(segment_ids):
    """Block-diagonal causal mask: token i sees token j iff same segment, j <= i, not padding.

    Returned as a boolean matrix so a test can assert directly that no token attends across a
    document boundary. The model applies the same rule.
    """
    n = len(segment_ids)
    same = segment_ids[:, None] == segment_ids[None, :]
    causal = np.tril(np.ones((n, n), dtype=bool))
    real = segment_ids != 0
    return same & causal & real[:, None] & real[None, :]


def pack_report(seqs, dropped=None):
    """Aggregate packing efficiency. Reconstructible from provenance, not from a live counter."""
    if not seqs:
        return {"sequences": 0, "tokens": 0, "used": 0, "loss_tokens": 0,
                "dropped": len(dropped or []), "utilisation": 0.0, "loss_bearing_frac": 0.0}
    total = sum(len(s.input_ids) for s in seqs)
    used = sum(s.used for s in seqs)
    loss = sum(s.loss_tokens for s in seqs)
    return {"sequences": len(seqs), "tokens": total, "used": used, "loss_tokens": loss,
            "padding": total - used, "dropped": len(dropped or []),
            "utilisation": round(used / total, 4),
            "loss_bearing_frac": round(loss / total, 4)}
