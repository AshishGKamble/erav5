"""
Audit: replay, resume equivalence, fork lineage, mixture compliance, throughput.

Everything here is computed **from the artefacts on disk** - the ledgers, the manifest - never from
a variable a run happened to be holding. That is the difference between a report and a claim. If a
number cannot be reconstructed after the fact, it does not belong in the evidence bundle.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ledger
import mixture as mx_mod
from train import stack_batch


def replay_interval(stream, records, start, end):
    """Re-derive batches for [start, end) and compare against the ledger.

    The stream is a function of the frozen inputs and the consumption prefix, so replay means
    rebuilding the scheduler state from the records before `start` and running forward. Matching
    batch hashes means the token ids, masks, positions, segments and provenance all agree.
    """
    prefix = [r for r in records if r["step"] < start]
    _consumed, by_stage, _ = mx_mod.replay_state(prefix)
    by_stage = {k: dict(v) for k, v in by_stage.items()}

    want = {r["step"]: r for r in records if start <= r["step"] < end}
    checked, mismatches = 0, []
    for step in range(start, end):
        if step not in want:
            continue
        b = stream.build_batch(step, by_stage)
        got = ledger.batch_hash(b["seqs"])
        exp = want[step]["batch_hash"]
        if got != exp:
            mismatches.append({"step": step, "expected": exp, "recomputed": got})
        # Token spans must match too, not just the aggregate hash.
        got_spans = [[p["shard_id"], p["doc_id"], p["tok_start"], p["tok_end"]]
                     for s in b["seqs"] for p in s.provenance]
        exp_spans = [d for seq in want[step]["provenance"] for d in seq["docs"]]
        if got_spans != exp_spans:
            mismatches.append({"step": step, "reason": "token spans differ"})
        for lane, n in b["lane_tokens"].items():
            by_stage.setdefault(b["stage"], {})
            by_stage[b["stage"]][lane] = by_stage[b["stage"]].get(lane, 0) + n
        checked += 1
    return {"start": start, "end": end, "checked": checked,
            "mismatches": mismatches, "ok": not mismatches}


def resume_equivalence(reference, resumed):
    """Compare a clean reference run against a crashed-and-resumed one, record for record.

    This is the external oracle. A checkpoint that predicts its own next batch and then matches it
    proves nothing; two independent runs agreeing on every batch id, token span and hash does.
    """
    problems = []
    if len(reference) != len(resumed):
        problems.append(f"length differs: reference {len(reference)}, resumed {len(resumed)}")
    for a, b in zip(reference, resumed):
        if a["step"] != b["step"]:
            problems.append(f"step mismatch: {a['step']} vs {b['step']}")
            break
        if a["batch_hash"] != b["batch_hash"]:
            problems.append(f"step {a['step']}: batch hash differs")
        if a["lane_tokens"] != b["lane_tokens"]:
            problems.append(f"step {a['step']}: lane composition differs")
        if a["provenance"] != b["provenance"]:
            problems.append(f"step {a['step']}: provenance differs")
    problems.extend(f"resumed ledger: {p}" for p in ledger.contiguity(resumed))
    first_after = None
    return {"ok": not problems, "problems": problems[:20],
            "compared": min(len(reference), len(resumed)),
            "first_divergence": first_after}


def fork_lineage(main_records, fork_records, fork_step):
    """A fork must share its parent's prefix exactly and diverge only after the fork point."""
    problems = []
    pre_main = [r for r in main_records if r["step"] <= fork_step]
    pre_fork = [r for r in fork_records if r["step"] <= fork_step]
    shared = min(len(pre_main), len(pre_fork))
    for a, b in zip(pre_main[:shared], pre_fork[:shared]):
        if a["batch_hash"] != b["batch_hash"]:
            problems.append(f"step {a['step']}: fork prefix differs from parent")
    post_main = {r["step"]: r for r in main_records if r["step"] > fork_step}
    post_fork = {r["step"]: r for r in fork_records if r["step"] > fork_step}
    common = sorted(set(post_main) & set(post_fork))
    diverged = [s for s in common if post_main[s]["batch_hash"] != post_fork[s]["batch_hash"]]
    if common and not diverged:
        problems.append("fork never diverged from parent after the fork point")
    return {"ok": not problems, "problems": problems,
            "shared_prefix_steps": shared,
            "compared_after_fork": len(common),
            "diverged_after_fork": len(diverged)}


def compliance_from_ledger(records, mixture):
    consumed, _by_stage, _ = mx_mod.replay_state(records)
    c = mixture.compliance(consumed)
    indic_present = sum(1 for r in records if r["lane_tokens"].get("indic", 0) > 0)
    running, cum_min = {}, 1.0
    for r in records:
        for lane, n in r["lane_tokens"].items():
            running[lane] = running.get(lane, 0) + n
        tot = sum(running.values()) or 1
        cum_min = min(cum_min, running.get("indic", 0) / tot)
    c["indic_present_batches"] = indic_present
    c["indic_total_batches"] = len(records)
    c["indic_cumulative_floor_min_pct"] = round(cum_min * 100, 2)
    c["indic_floor_pct"] = mixture.indic_floor_pct
    c["floor_held"] = cum_min * 100 >= mixture.indic_floor_pct
    c["present_in_every_batch"] = indic_present == len(records)
    return c


def performance(records, learning, elapsed_s):
    """Throughput and packing efficiency, recomputed from the ledgers.

    Deliberately not read from a live counter. The brief says reported packing and throughput
    numbers must be reconstructible, so they are derived here from the same records a reader can
    inspect.
    """
    tokens = sum(sum(r["lane_tokens"].values()) for r in records)
    loss_tokens = sum(r["loss_tokens"] for r in records)
    slots = sum(r["sequences"] for r in records)
    capacity = sum(r["sequences"] * r["seq_len"] for r in records)
    padding = sum(r["padding"] for r in records)
    return {
        "batches": len(records),
        "sequences": slots,
        "elapsed_s": round(elapsed_s, 3),
        "tokens_total": tokens,
        "loss_bearing_tokens": loss_tokens,
        "padding_tokens": padding,
        "packing_utilisation": round(tokens / capacity, 4) if capacity else 0.0,
        "loss_bearing_frac_of_capacity": round(loss_tokens / capacity, 4) if capacity else 0.0,
        "tokens_per_s": round(tokens / elapsed_s, 1) if elapsed_s else None,
        "loss_bearing_tokens_per_s": round(loss_tokens / elapsed_s, 1) if elapsed_s else None,
        "batches_per_s": round(len(records) / elapsed_s, 3) if elapsed_s else None,
        "final_loss": learning[-1]["loss"] if learning else None,
        "first_loss": learning[0]["loss"] if learning else None,
    }


def learning_trace(learning):
    """Loss linked back to the data that produced it."""
    if not learning:
        return {}
    per_lane = {}
    for r in learning:
        for lane, v in r["per_lane_loss"].items():
            per_lane.setdefault(lane, []).append(v)
    n = max(1, len(learning) // 5)
    return {
        "steps": len(learning),
        "first_loss": learning[0]["loss"],
        "final_loss": learning[-1]["loss"],
        "mean_first_fifth": round(sum(r["loss"] for r in learning[:n]) / n, 6),
        "mean_last_fifth": round(sum(r["loss"] for r in learning[-n:]) / n, 6),
        "per_lane_final": {k: round(sum(v[-n:]) / len(v[-n:]), 6)
                           for k, v in sorted(per_lane.items())},
        "lanes_traced": sorted(per_lane),
        "loss_fell": learning[-1]["loss"] < learning[0]["loss"],
    }
