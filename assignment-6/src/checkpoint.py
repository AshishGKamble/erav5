"""
Checkpoints tied to ledger offsets, with branch lineage.

A checkpoint here is not just weights. It records **where in the consumption ledger the run was**,
so resume can rewind a partial tail and continue at exactly the next batch. It also records the
hashes of everything the run depends on - plan, manifest, tokenizer - and refuses to resume if any
of them has changed, because a run resumed against a different plan is a different run wearing the
old one's name.

The scheduler state is stored **and** hashed. On resume the state is recomputed from the ledger
prefix and compared against the stored hash. That is the whole trust model: the stored value makes
resume O(1), the recomputation makes it verified, and nothing is believed on the strength of having
been written down.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CHECKPOINTS, ensure_dirs, read_json, sha256_obj, write_json


def path_for(branch, step):
    return os.path.join(CHECKPOINTS, f"{branch}-step{step:05d}")


def save(branch, step, model, consumed, by_stage, offsets, bindings, parent=None, opt=None):
    """Write weights and optimiser state (.npz) plus metadata (.json). Returns the metadata.

    The optimiser is saved deliberately. An earlier version restored weights only and rebuilt a
    fresh Adam, which loses the first- and second-moment estimates. The *data* stream still
    matched, because it does not depend on model state - but the training did not truly resume,
    and the resumed run's losses diverged from the reference for no reason a reader could see.
    With the moments restored, the learning ledger matches too, which turns crash recovery from a
    claim about batches into a claim about the whole run.
    """
    ensure_dirs(CHECKPOINTS)
    base = path_for(branch, step)
    blob = dict(model.state())
    if opt is not None:
        blob["__opt_t"] = np.asarray([opt.t])
        for k, v in opt.m.items():
            blob["__opt_m__" + k] = v
        for k, v in opt.v.items():
            blob["__opt_v__" + k] = v
    np.savez(base + ".npz", **blob)
    meta = {
        "branch": branch, "step": step, "parent": parent,
        "next_step": step + 1,
        "ledger_offsets": offsets,
        "scheduler": {"consumed": {k: int(v) for k, v in sorted(consumed.items())},
                      "by_stage": {s: {k: int(v) for k, v in sorted(d.items())}
                                   for s, d in sorted(by_stage.items())}},
        "scheduler_hash": sha256_obj({
            "step": step,
            "consumed": {k: int(v) for k, v in sorted(consumed.items())},
            "by_stage": {s: {k: int(v) for k, v in sorted(d.items())}
                         for s, d in sorted(by_stage.items())}}),
        "weights_sha256": model.weight_hash(),
        "optimiser_saved": opt is not None,
        "optimiser_step": (opt.t if opt is not None else None),
        "weights_file": os.path.basename(base) + ".npz",
        # What this run is bound to. Resume against different bindings is refused.
        "bindings": bindings,
        "model_cfg": model.cfg,
    }
    write_json(base + ".json", meta)
    return meta


def load(branch, step):
    return read_json(path_for(branch, step) + ".json")


def restore_weights(model, branch, step, opt=None):
    """Restore weights, and the optimiser moments too when an optimiser is supplied."""
    z = np.load(path_for(branch, step) + ".npz")
    model.load({k: z[k] for k in z.files if not k.startswith("__opt")})
    if opt is not None and "__opt_t" in z.files:
        opt.t = int(z["__opt_t"][0])
        opt.m = {k[len("__opt_m__"):]: z[k] for k in z.files if k.startswith("__opt_m__")}
        opt.v = {k[len("__opt_v__"):]: z[k] for k in z.files if k.startswith("__opt_v__")}
    return model


def verify(meta, bindings, ledger_records):
    """Check a checkpoint may be resumed. Returns (ok, problems).

    Two independent checks:
      * bindings - plan, manifest and tokenizer must be the ones this checkpoint was made under;
      * scheduler - the stored state must equal the state recomputed from the ledger prefix.
    """
    problems = []
    for key, want in bindings.items():
        got = meta.get("bindings", {}).get(key)
        if got != want:
            problems.append(f"binding '{key}' differs: checkpoint {str(got)[:12]}... "
                            f"vs current {str(want)[:12]}...")

    import mixture as mx_mod
    prefix = ledger_records[:meta["ledger_offsets"]["consumption"]]
    consumed, by_stage, next_step = mx_mod.replay_state(prefix)
    recomputed = sha256_obj({
        "step": meta["step"],
        "consumed": {k: int(v) for k, v in sorted(consumed.items())},
        "by_stage": {s: {k: int(v) for k, v in sorted(d.items())}
                     for s, d in sorted(by_stage.items())}})
    if recomputed != meta["scheduler_hash"]:
        problems.append("scheduler state does not match the ledger prefix it claims")
    if next_step != meta["next_step"]:
        problems.append(f"ledger prefix ends at step {next_step}, "
                        f"checkpoint expects {meta['next_step']}")
    return (not problems), problems
