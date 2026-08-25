"""
The evaluation and validation firewall.

One rule: a document whose shard is not `split == "train"` may never reach a loss-bearing batch.

It is enforced in three independent places, which is deliberate. A5 section 9.2 lost a published
finding to a validation leak that no single check would have caught - `prepare_data.py` repeated
its source six times before splitting the validation set off by token offset, so every held-out
window also sat in training and the reasoning metric was measuring memorisation. That failure was
invisible because nothing was checking.

  1. **Structural.** Splits are assigned from document content, not position (`shards.split_for`),
     so a document cannot land in two splits however the corpus is reordered or repeated.
  2. **Identity.** The split is hashed into the shard's identity, so relabelling an eval shard as
     train changes its hash and fails manifest validation.
  3. **Admission.** This module. Every candidate is checked at the point of use, and every refusal
     is logged rather than silently skipped.

The demo does not merely assert this works - it attacks the firewall with a real eval shard and
records the block. An assertion that never fires proves nothing.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

TRAIN = "train"


class Firewall:
    def __init__(self, manifest):
        # shard_id -> split, taken from the validated manifest.
        self.split_of = {r["shard_id"]: r["split"] for r in manifest["shards"]}
        self.blocked = []
        self.admitted = 0

    def check(self, shard_id, reason="batch admission"):
        """True if this shard may enter a loss-bearing batch. Refusals are recorded."""
        split = self.split_of.get(shard_id)
        if split == TRAIN:
            self.admitted += 1
            return True
        self.blocked.append({"shard_id": shard_id, "split": split or "unknown",
                             "reason": reason})
        return False

    def assert_batch(self, provenance):
        """Every document in a packed batch must come from a train shard. Returns violations."""
        bad = []
        for p in provenance:
            if self.split_of.get(p["shard_id"]) != TRAIN:
                bad.append({"shard_id": p["shard_id"], "doc_id": p["doc_id"],
                            "split": self.split_of.get(p["shard_id"], "unknown")})
        return bad

    def report(self):
        return {"admitted": self.admitted, "blocked": len(self.blocked),
                "blocked_shards": sorted({b["shard_id"] for b in self.blocked}),
                "blocked_splits": sorted({b["split"] for b in self.blocked}),
                "events": self.blocked}
