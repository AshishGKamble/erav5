"""
Append-only ledgers: consumption, learning, OPUS decisions, firewall events.

Four files, one record per line, never rewritten. Together they answer the four questions the
brief says the system must be able to answer: what it consumed, why it consumed it, what the model
learned from it, and how the run can be reconstructed.

  consumption.jsonl  what entered each batch: lane composition, document spans, token counts,
                     loss-bearing counts, padding, and the batch's content hash
  learning.jsonl     what came back out: total loss, per-lane loss, per-sample loss, grad norm
  opus.jsonl         every accept / reject / defer / override, with the snapshot it was made under
  firewall.jsonl     every refused admission

Consumption and learning are kept **separate files keyed by batch id** rather than one merged
record, because they have different reproducibility guarantees. A consumption record is exactly
reproducible and is hashed; a learning record contains floating-point loss, which shifts with the
BLAS backend, and is compared with a tolerance. Merging them would force the whole record into the
weaker guarantee.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import LEDGERS, append_jsonl, read_jsonl, sha256_obj

CONSUMPTION, LEARNING, OPUS, FIREWALL = "consumption", "learning", "opus", "firewall"


def path_for(kind, branch="main"):
    suffix = "" if branch == "main" else f".{branch}"
    return os.path.join(LEDGERS, f"{kind}{suffix}.jsonl")


def batch_hash(seqs):
    """Identity of a batch: token ids, masks, positions, segments, provenance. Never loss.

    Floating point is deliberately excluded - see common.py. Two machines must agree on this hash,
    and they can only do that about integers.
    """
    payload = []
    for s in seqs:
        payload.append({
            "ids": s.input_ids.tolist(),
            "loss": s.loss_mask.tolist(),
            "pos": s.position_ids.tolist(),
            "seg": s.segment_ids.tolist(),
            "prov": [[p["shard_id"], p["doc_id"], p["tok_start"], p["tok_end"],
                      p["seq_start"], p["seq_end"]] for p in s.provenance]})
    return sha256_obj(payload)


def write_consumption(branch, rec):
    append_jsonl(path_for(CONSUMPTION, branch), rec)


def write_learning(branch, rec):
    append_jsonl(path_for(LEARNING, branch), rec)


def write_opus(branch, rec):
    append_jsonl(path_for(OPUS, branch), rec)


def write_firewall(branch, rec):
    append_jsonl(path_for(FIREWALL, branch), rec)


def read(kind, branch="main"):
    p = path_for(kind, branch)
    return list(read_jsonl(p)) if os.path.exists(p) else []


def truncate_to(kind, branch, n_records):
    """Cut a ledger back to `n_records` lines - used on resume to discard a partial tail.

    A crash can leave a record written for a batch whose step never completed. Resume rewinds to
    the checkpoint's recorded offset so the resumed run neither skips nor repeats.
    """
    p = path_for(kind, branch)
    if not os.path.exists(p):
        return 0
    lines = [ln for ln in open(p, encoding="utf-8").read().splitlines() if ln.strip()]
    keep = lines[:n_records]
    with open(p, "w", encoding="utf-8", newline="\n") as fh:
        for ln in keep:
            fh.write(ln + "\n")
    return len(lines) - len(keep)


def contiguity(records):
    """Batch ids must be contiguous from 0 with no repeats - the resume invariant, checked.

    'No skipped or repeated batches' is the requirement; this is the check that proves it rather
    than the hope that the mechanism was right.
    """
    steps = [r["step"] for r in records]
    problems = []
    if steps != sorted(steps):
        problems.append("steps are not monotonically ordered")
    if len(set(steps)) != len(steps):
        dupes = sorted({s for s in steps if steps.count(s) > 1})
        problems.append(f"repeated steps: {dupes[:10]}")
    if steps and steps != list(range(steps[0], steps[0] + len(steps))):
        missing = sorted(set(range(steps[0], steps[-1] + 1)) - set(steps))
        problems.append(f"skipped steps: {missing[:10]}")
    return problems
