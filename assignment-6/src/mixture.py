"""
Curriculum stages, lane weights, and the protected floor.

Reads `frozen/plan.json` - Assignment 5's shipped plan - and turns it into per-batch lane choices.
Nothing here invents a share; the numbers are A5's and this module is only responsible for
delivering them.

**Scheduling is deficit-based, not independent sampling.** Drawing each slot at random from the
lane distribution is the obvious implementation and it does not work: over a few hundred batches the
realised shares wobble by several points, and the run then fails its own mixture-compliance check.
Instead each slot goes to whichever lane is furthest below its target *so far*. Compliance stops
being a statistical hope and becomes an invariant of the selection rule.

**Deficits are measured within the current stage, not across the whole run.** This is subtle and the
first version got it wrong. Web is 57% of the Seed stage and 12% of Long-context; if the deficit is
computed against cumulative consumption, then by the later stages web is far "above target" on the
run as a whole and gets starved to repay a debt it never owed - the run came out 9.45 points under
on web and 7.11 over on agentic. Each stage is its own budget. Deliver each stage's shares within
that stage and the integral over stages reproduces the plan automatically, which is exactly how
A5 solved the schedule in the first place.

**The stream is a function of history, not of step alone.** Because a deficit depends on everything
consumed before it, `choose(step)` cannot be evaluated in isolation - it needs the consumption
prefix. That is why the checkpoint stores scheduler state *and its hash*, and why resume recomputes
the state from the ledger and asserts it matches. O(1) trust, O(n) verification, no stored state
taken on faith.
"""
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import FROZEN, read_json, sha256_obj

# Lane keys differ slightly between plan.json and the corpus; one place to translate.
BANDS = ["B0", "B1", "B2", "B3", "B4", "B5"]

PLAN_TO_LANE = {"web": "web", "code": "code", "math_stem": "math",
                "reasoning": "reasoning", "agentic": "agentic",
                "long_ctx": "long_ctx", "indic": "indic"}


class Mixture:
    """The plan, loaded and made executable."""

    def __init__(self, plan_path=None):
        self.plan = read_json(plan_path or os.path.join(FROZEN, "plan.json"))
        fp = self.plan["full_plan"]
        self.lanes = [PLAN_TO_LANE[l["key"]] for l in fp["lanes"]]
        self.integrated = {PLAN_TO_LANE[l["key"]]: l["share"] for l in fp["lanes"]}
        self.stages = [{"key": s["key"], "name": s["name"], "weight": s["weight"],
                        "seq": s["seq"], "band": s["band"],
                        "shares": {PLAN_TO_LANE[k]: v for k, v in s["shares"].items()}}
                       for s in fp["stages"]]
        floor = fp["floor"]
        self.indic_floor_pct = floor["indic_batch_pct"]
        self.opus_keep_frac = floor["opus_keep_frac"]
        self.opus_rerun_every_B = floor["opus_rerun_every_B"]
        # plan.json names agentic "never OPUS-trimmed"; both protected lanes are listed here so
        # the override path has one source of truth.
        self.protected = {"indic", "agentic"}

    # ---------------------------------------------------------------- curriculum

    def stage_at(self, step, total_steps):
        """Which curriculum stage a step falls in, by cumulative stage weight.

        Stages are Seed 8% -> General 45% -> Reasoning 25% -> Long-ctx 19% -> Anneal 3%, so the
        boundaries are fractions of the run rather than fixed step counts.
        """
        frac = step / max(1, total_steps)
        acc = 0.0
        for st in self.stages:
            acc += st["weight"] / 100.0
            if frac < acc:
                return st
        return self.stages[-1]

    def stage_boundaries(self, total_steps):
        out, acc = [], 0.0
        for st in self.stages:
            acc += st["weight"] / 100.0
            out.append((st["key"], int(round(acc * total_steps))))
        return out

    def bands_for(self, stage):
        """The difficulty bands a stage may draw from.

        plan.json writes them as ranges - "B0-B1", "B3-B4", "B5" - which expand to sets. This is
        the curriculum's ordering rule made executable: feed B5 too early and the model consumes
        without learning, and B5 is the most expensive data there is.
        """
        spec = stage.get("band", "")
        found = [int(x) for x in re.findall(r"B(\d)", spec)]
        if not found:
            return set(BANDS)
        lo, hi = min(found), max(found)
        return {f"B{i}" for i in range(lo, hi + 1)}

    # ---------------------------------------------------------------- selection

    def choose(self, stage, stage_consumed, n_slots, slot_tokens):
        """Assign lanes to the slots of one batch. Pure: same inputs, same output, always.

        `stage_consumed` maps lane -> tokens consumed **within the current stage**. See the module
        docstring for why it is stage-local rather than cumulative.

        Deficit selection runs first over every slot, and the Indic floor is applied afterwards as a
        **repair**, promoting slots only if the natural allocation came in under it. That ordering
        matters and an earlier version got it wrong by reserving floor slots up front: with 8 slots
        a 14% floor rounds up to 2 slots, which is 25% - above Indic's own 18% target - so the floor
        displaced every other lane. The floor is protection against a selector trimming Indic away
        (plan.json: OPUS scores from the first ~512 tokens against an English- and code-heavy proxy,
        so Indic aligns weakly and would be discarded). It was never a scheduling target, and should
        bind only when something else has pushed Indic below it.
        """
        targets = {l: stage["shares"].get(l, 0) / 100.0 for l in self.lanes}
        running = dict(stage_consumed)
        picks = []

        for _ in range(n_slots):
            total = sum(running.values()) + slot_tokens
            # Deficit: how far below its stage target this lane would be after taking the slot.
            # Ties break on lane name so the choice is total and reproducible.
            best = max(self.lanes,
                       key=lambda l: (targets[l] * total - running.get(l, 0), l))
            picks.append(best)
            running[best] = running.get(best, 0) + slot_tokens

        # TWO guarantees, because they protect different things and neither implies the other:
        #   presence  - every batch contains Indic
        #   floor     - Indic's cumulative token share never drops below plan.json's 14%
        #
        # plan.json words it as "14% of every batch". A literal per-batch token floor cannot be
        # enforced honestly: token yield is not known until after packing, and reserving whole
        # slots to cover the worst case pushed Indic 4.7 points over plan (see below). The floor
        # exists so a selector cannot delete Indic from what the model trains on, and that is a
        # property of the cumulative share. The per-batch half of the promise is kept as a
        # presence guarantee, and both are measured in the audit rather than assumed.
        #
        # Floor repair, measured in TOKENS not slots. An earlier version counted slots:
        # ceil(0.14 x 16) = 3 slots = 18.75% of slots, but Indic packs at 0.944 utilisation against
        # a 0.775 batch average, so those 3 slots became ~23% of tokens and pushed Indic 4.7 points
        # over plan. The floor protects a share of what the model is actually trained on, and that
        # is counted in tokens. Because the deficit rule already targets Indic at 18%, well above
        # the 14% floor, this repair should almost never fire - and when it does, something else
        # has starved Indic, which is exactly the case it exists for.
        # Presence guarantee: every batch carries Indic, so no batch is ever Indic-free even
        # while the cumulative floor is comfortably satisfied.
        if "indic" not in picks:
            total = sum(running.values()) or 1
            cand = [i for i in range(n_slots) if picks[i] not in self.protected]
            if cand:
                i = max(cand, key=lambda j: (running.get(picks[j], 0) - targets[picks[j]] * total,
                                             picks[j]))
                running[picks[i]] = running.get(picks[i], 0) - slot_tokens
                picks[i] = "indic"
                running["indic"] = running.get("indic", 0) + slot_tokens

        floor = self.indic_floor_pct / 100.0
        guard = 0
        while guard < n_slots:
            total = sum(running.values()) or 1
            if running.get("indic", 0) / total >= floor:
                break
            # Demote the lane furthest ABOVE its stage target, never another protected lane.
            cand = [i for i in range(n_slots) if picks[i] != "indic"
                    and picks[i] not in self.protected]
            if not cand:
                break
            i = max(cand, key=lambda j: (running.get(picks[j], 0) - targets[picks[j]] * total,
                                         picks[j]))
            running[picks[i]] = running.get(picks[i], 0) - slot_tokens
            picks[i] = "indic"
            running["indic"] = running.get("indic", 0) + slot_tokens
            guard += 1

        return picks

    # ---------------------------------------------------------------- compliance

    def compliance(self, consumed):
        """Realised shares against the integrated plan. Computed from consumption, not intent."""
        total = sum(consumed.values()) or 1
        rows = []
        for lane in self.lanes:
            actual = 100.0 * consumed.get(lane, 0) / total
            planned = self.integrated[lane]
            rows.append({"lane": lane, "planned_pct": planned,
                         "actual_pct": round(actual, 2),
                         "delta": round(actual - planned, 2)})
        return {"rows": rows,
                "max_abs_delta": round(max(abs(r["delta"]) for r in rows), 2),
                "total_tokens": sum(consumed.values())}

    def state_hash(self, consumed, step):
        """Identity of the scheduler's position. Checkpointed, then recomputed on resume."""
        return sha256_obj({"step": step,
                           "consumed": {k: int(v) for k, v in sorted(consumed.items())}})

    def state_hash_full(self, consumed, by_stage, step):
        """Full scheduler identity: global consumption, per-stage consumption, and position."""
        return sha256_obj({
            "step": step,
            "consumed": {k: int(v) for k, v in sorted(consumed.items())},
            "by_stage": {s: {k: int(v) for k, v in sorted(d.items())}
                         for s, d in sorted(by_stage.items())}})


def replay_state(records):
    """Rebuild scheduler state from consumption-ledger records alone.

    Returns (consumed, stage_consumed, step). This is the verification half of the resume contract:
    the checkpoint says what the state was, this recomputes it from the append-only ledger, and the
    two must agree. Nothing is taken on faith from a stored blob.
    """
    consumed, by_stage, step = {}, {}, 0
    for r in records:
        st = by_stage.setdefault(r["stage"], {})
        for lane, n in r["lane_tokens"].items():
            consumed[lane] = consumed.get(lane, 0) + n
            st[lane] = st.get(lane, 0) + n
        step = max(step, r["step"] + 1)
    return consumed, by_stage, step
