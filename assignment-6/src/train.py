"""
The training loop, and the crash / resume / fork machinery around it.

Training itself is beside the point - the lecture explicitly allowed a stub loop. It is real anyway,
because "learning trace: loss linked to source data" is a named row in the evidence bundle and a real
per-lane loss costs little to produce and defends itself under inspection.

What matters here is the bookkeeping around each step:

  * the consumption record is written **before** the optimiser step, so a crash mid-step leaves a
    ledger tail that resume can detect and rewind rather than a silent gap;
  * the batch hash covers token ids, masks, positions, segments and provenance - never loss;
  * per-lane loss is computed by masking the loss to each lane's sequences, so the learning ledger
    can say which data produced which number.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import checkpoint as ckpt
import ledger
import mixture as mx_mod
from common import sha256_obj
from model import Tiny


class CrashInjected(Exception):
    """Raised to simulate a hard failure. Deliberate, and caught only by the demo driver."""


class Adam:
    def __init__(self, params, lr=3e-3, b1=0.9, b2=0.95, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        gn = float(np.sqrt(sum(float((g.astype(np.float64) ** 2).sum())
                               for g in grads.values())))
        clip = min(1.0, 1.0 / max(gn, 1e-8))
        for k in params:
            g = grads[k] * clip
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * (g * g)
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            params[k] -= (self.lr * mh / (np.sqrt(vh) + self.eps)).astype(params[k].dtype)
        return gn

    def state(self):
        return {"t": self.t, "m": self.m, "v": self.v}

    def load(self, st):
        self.t = st["t"]
        self.m = {k: np.asarray(v) for k, v in st["m"].items()}
        self.v = {k: np.asarray(v) for k, v in st["v"].items()}


def stack_batch(seqs, seq_len):
    """Assemble packed sequences into the arrays the model consumes."""
    B = len(seqs)
    ids = np.zeros((B, seq_len), dtype=np.int64)
    pos = np.zeros((B, seq_len), dtype=np.int64)
    seg = np.zeros((B, seq_len), dtype=np.int64)
    lm = np.zeros((B, seq_len), dtype=np.int8)
    for i, s in enumerate(seqs):
        ids[i] = s.input_ids
        pos[i] = s.position_ids
        seg[i] = s.segment_ids
        lm[i] = s.loss_mask
    return ids, pos, seg, lm


def run(stream, mixture, branch, start_step, end_step, model, opt, bindings,
        consumed=None, by_stage=None, crash_at=None, checkpoint_at=(), log=print,
        write_ledgers=True):
    """Train from `start_step` to `end_step`. Returns a summary dict.

    `crash_at` raises CrashInjected *after* the consumption record is written and before the
    checkpoint, which is the awkward case resume has to handle correctly.
    """
    consumed = dict(consumed or {})
    by_stage = {k: dict(v) for k, v in (by_stage or {}).items()}
    losses, checkpoints = [], []

    for step in range(start_step, end_step):
        batch = stream.build_batch(step, by_stage)
        if not batch["seqs"]:
            continue
        seqs = batch["seqs"]
        ids, pos, seg, lm = stack_batch(seqs, stream.seq_len)

        bh = ledger.batch_hash(seqs)
        crec = {
            "step": step, "branch": branch, "stage": batch["stage"],
            "batch_hash": bh,
            "sequences": len(seqs), "seq_len": stream.seq_len,
            "lane_tokens": batch["lane_tokens"],
            "loss_tokens": batch["loss_tokens"],
            "padding": len(seqs) * stream.seq_len - sum(s.used for s in seqs),
            "utilisation": round(sum(s.used for s in seqs) / (len(seqs) * stream.seq_len), 4),
            "snapshot_id": batch["snapshot_id"],
            "picks": batch["picks"],
            "stage_bands": batch.get("stage_bands", []),
            "gates": batch.get("gates", {}),
            "provenance": [{"lane": s.lane,
                            "docs": [[p["shard_id"], p["doc_id"], p["tok_start"], p["tok_end"]]
                                     for p in s.provenance]} for s in seqs],
        }
        if write_ledgers:
            ledger.write_consumption(branch, crec)
            for d in batch["opus_decisions"]:
                ledger.write_opus(branch, d)

        if crash_at is not None and step == crash_at:
            # Deliberate hard failure, after the consumption record and before the checkpoint.
            raise CrashInjected(f"simulated crash at step {step}")

        loss, grads, per_seq, tok = model.loss_and_grad(
            ids, pos, seg, lm, lanes=[q.lane for q in seqs])
        gn = opt.step(model.p, grads)

        # Per-lane loss: which data produced which number.
        by_lane = {}
        for i, s in enumerate(seqs):
            by_lane.setdefault(s.lane, []).append(float(per_seq[i]))
        lrec = {"step": step, "branch": branch, "batch_hash": bh,
                "loss": round(float(loss), 6),
                "per_lane_loss": {k: round(float(np.mean(v)), 6) for k, v in sorted(by_lane.items())},
                "per_sequence_loss": [round(float(x), 6) for x in per_seq],
                "grad_norm": round(float(gn), 6),
                "loss_tokens": batch["loss_tokens"],
                # Token-level tracking. The brief asks for token-level OR sample-level; both are
                # recorded, because the per-token distribution is what shows whether a batch's
                # loss came from a few catastrophic tokens or was spread evenly.
                "token_loss": {"count": int(tok["count"]),
                               "mean": round(float(tok["mean"]), 6),
                               "p50": round(float(tok["p50"]), 6),
                               "p90": round(float(tok["p90"]), 6),
                               "max": round(float(tok["max"]), 6)},
                "token_loss_by_lane": {k: round(float(v), 6)
                                       for k, v in sorted(tok["by_lane"].items())}}
        if write_ledgers:
            ledger.write_learning(branch, lrec)
        losses.append(float(loss))

        for lane, n in batch["lane_tokens"].items():
            consumed[lane] = consumed.get(lane, 0) + n
            by_stage.setdefault(batch["stage"], {})
            by_stage[batch["stage"]][lane] = by_stage[batch["stage"]].get(lane, 0) + n

        if step in checkpoint_at:
            meta = ckpt.save(branch, step, model, consumed, by_stage,
                             offsets={"consumption": len(ledger.read(ledger.CONSUMPTION, branch)),
                                      "learning": len(ledger.read(ledger.LEARNING, branch))},
                             bindings=bindings, opt=opt)
            checkpoints.append(meta)
            log(f"    [PASS] checkpoint_saved  {branch} step {step}  "
                f"weights {meta['weights_sha256'][:12]}")

    return {"branch": branch, "steps": (start_step, end_step), "losses": losses,
            "consumed": consumed, "by_stage": by_stage, "checkpoints": checkpoints,
            "final_loss": losses[-1] if losses else None}
