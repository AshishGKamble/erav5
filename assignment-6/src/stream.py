"""
The batch stream: the one place where mixture, OPUS, the firewall and packing meet.

**A batch is a function of the frozen inputs and the consumption history.**

    batch(step) = f(seed, plan_hash, manifest_hash, ledger[0:step])

There is no stateful sampler whose RNG position has to be saved and restored. Candidate draws come
from a counter-based hash of (seed, step, slot, attempt), so the same step always draws the same
candidates on any machine. Resume then means: rebuild the scheduler state from the ledger prefix and
carry on. Skipping or repeating a batch is not a bug this system tests for - it is structurally
unavailable.

The honest caveat, stated because it would be easy to overclaim: a deficit depends on everything
consumed before it, so `batch(step)` is **O(step) to reconstruct, not O(1) seekable**. The checkpoint
therefore stores the scheduler state *and its hash*, and resume recomputes the state from the ledger
and asserts a match. O(1) trust, O(n) verification, nothing taken on faith from a stored blob.

**A slot is a sequence.** Each of a batch's slots is assigned one lane by the mixture scheduler, and
then filled with documents from that lane under that lane's packing policy. Keeping a sequence
single-lane is what makes lane accounting exact: every token in the consumption ledger is
attributable to the lane that was charged for it.

**OPUS decides which document fills a slot, never whether the slot is filled.** A rejected candidate
is replaced by another draw from the same lane. Getting this backwards would let OPUS silently
starve the lanes it scores badly - it rejects 57% of reasoning candidates - and the mixture would
drift while every individual decision looked defensible.
"""
import hashlib
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import opus
import packing
from common import SHARDS, read_json
from firewall import Firewall

MAX_ATTEMPTS = 24          # per slot, before falling back to the best candidate seen

# The anneal reserve, from plan.json: ~3% of the budget held back for a final low-learning-rate
# pass - Tier-A verified Indic, the hardest agentic traces, ultra-length reasoning, B5 STEM.
# "The word that matters is *reserve*: you cannot conjure Tier-A data in the last 3% if you already
# spent it." Here a document is reserved if it is the hardest band, or if it is native-tier Indic
# at B3 or above - the scarce tier A5 protects. Reserved documents are refused outside the anneal
# stage, and the refusals are counted so the reserve can be shown to have actually been held back.
RESERVE_BANDS = {"B5"}


def is_reserved(entry):
    meta = entry.get("meta", {})
    band = meta.get("band", "B2")
    if band in RESERVE_BANDS:
        return True
    return (entry.get("lane") == "indic" and meta.get("tier") == "native"
            and band in ("B3", "B4"))


def draw_index(seed, step, slot, attempt, n):
    """Counter-based draw. Pure in its arguments - no RNG object, nothing to checkpoint."""
    key = f"{seed}:{step}:{slot}:{attempt}".encode("utf-8")
    return int(hashlib.sha256(key).hexdigest()[:16], 16) % max(1, n)


class Catalogue:
    """Every document, grouped by lane, with its tokens resident in memory.

    Deliberately built over **all splits**, not just train. An earlier version loaded train-only,
    which meant the firewall could never fire during training - it was correct but untested in the
    path that matters, and "the check never blocks anything" is not a demonstration. Now eval and
    holdout documents sit in the candidate pool and are refused at draw time, every time, and each
    refusal is recorded.
    """

    def __init__(self, manifest, splits=("train", "eval", "holdout")):
        self.by_lane = {}
        self.tokens = {}
        for r in sorted(manifest["shards"], key=lambda x: x["shard_id"]):
            if r["split"] not in splits:
                continue
            idx = read_json(os.path.join(SHARDS, r["index"]))
            arr = np.fromfile(os.path.join(SHARDS, r["bin"]), dtype="<u2").astype(np.int32)
            self.tokens[r["shard_id"]] = arr
            entries = self.by_lane.setdefault(r["lane"], [])
            for i, d in enumerate(idx["docs"]):
                entries.append({"shard_id": r["shard_id"], "doc_id": d["id"], "off": d["off"],
                                "len": d["len"], "spans": d["spans"], "meta": d["meta"],
                                "split": r["split"], "lane": r["lane"]})
        for lane in self.by_lane:
            self.by_lane[lane].sort(key=lambda e: (e["shard_id"], e["doc_id"]))

    def doc(self, entry):
        arr = self.tokens[entry["shard_id"]]
        return {"shard_id": entry["shard_id"], "doc_id": entry["doc_id"], "meta": entry["meta"],
                "ids": arr[entry["off"]:entry["off"] + entry["len"]], "spans": entry["spans"]}

    def lane_size(self, lane):
        return len(self.by_lane.get(lane, ()))


def calibration_set(catalogue, mixture, per_lane=40):
    """Candidates for threshold calibration, drawn in the mixture's own lane proportions.

    Calibrating on a flat sample across lanes gave a realised keep fraction of 0.39 against the
    0.50 plan.json asks for, because the run does not draw lanes uniformly. Weighting the
    calibration sample by each unprotected lane's integrated share fixes the mismatch.
    """
    out = []
    unprotected = [l for l in mixture.lanes if l not in mixture.protected]
    total = sum(mixture.integrated[l] for l in unprotected) or 1
    for lane in unprotected:
        n = max(4, int(round(per_lane * len(unprotected) * mixture.integrated[lane] / total)))
        entries = catalogue.by_lane.get(lane, [])
        for i in range(min(n, len(entries))):
            # Deterministic spread across the lane rather than the first n documents.
            e = entries[(i * 7919) % len(entries)]
            out.append(catalogue.doc(e)["ids"])
    return out


def reference_profiles(catalogue, lanes=("web", "code", "math"), per_lane=80):
    """Token counts for the lanes that make up the OPUS proxy.

    web + code + math only. Indic and agentic are deliberately absent - that absence is what makes
    them score badly, and reproducing it is the point.
    """
    ref = {}
    for lane in lanes:
        v = np.zeros(opus.VOCAB, dtype=np.float64)
        entries = catalogue.by_lane.get(lane, [])
        for i in range(min(per_lane, len(entries))):
            ids = catalogue.doc(entries[(i * 7919) % len(entries)])["ids"]
            v += np.bincount(np.asarray(ids, dtype=np.int64), minlength=opus.VOCAB)
        ref[lane] = v
    return ref


class Stream:
    def __init__(self, manifest, mixture, catalogue, seq_len, batch_slots, total_steps,
                 seed=20260824, opus_rerun_steps=50):
        self.manifest = manifest
        self.mixture = mixture
        self.cat = catalogue
        self.seq_len = seq_len
        self.batch_slots = batch_slots
        self.total_steps = total_steps
        self.seed = seed
        self.opus_rerun_steps = opus_rerun_steps
        self.firewall = Firewall(manifest)
        self._snapshots = {}
        self._ref = reference_profiles(catalogue)
        self._calib = calibration_set(catalogue, mixture)

    def snapshot(self, step):
        """The OPUS snapshot governing `step`. Cached, but derived - never stored and trusted."""
        gen = opus.generation_for(step, self.opus_rerun_steps)
        if gen not in self._snapshots:
            self._snapshots[gen] = opus.build_snapshot(
                gen, self._ref, self._calib, keep_frac=self.mixture.opus_keep_frac)
        return self._snapshots[gen]

    def _fill_slot(self, step, slot, lane, snap, stage):
        """Fill one sequence from one lane. Returns (Packed or None, decisions, deferred, gate).

        Four filters run before a candidate is even scored, in this order:

          firewall  a non-train document is refused outright and the refusal logged;
          reserve   a document held for the anneal is refused outside the anneal stage;
          band      a document outside the stage's difficulty range is skipped;
          OPUS      the accept / reject / defer / override decision.

        **Deferral is materially different from rejection**, which is the whole point of having
        four decisions rather than three. Candidates are collected into pools and the slot is
        filled in preference order: accepted first, then *deferred*, and only then the best-scoring
        rejected candidate. A deferred document is "not now", and when the snapshot generation
        turns over it is re-scored and may accept - which the audit measures.

        A cross-step deferral queue was considered and rejected: it would make the batch stream
        depend on state that cannot be rebuilt from the consumption ledger, which would break
        replay from a mid-run offset. Everything here is derivable from (seed, step, slot, attempt)
        and the frozen inputs.
        """
        entries = self.cat.by_lane.get(lane, [])
        if not entries:
            return None, [], [], {}
        protected = lane in self.mixture.protected
        allowed_bands = self.mixture.bands_for(stage)
        in_anneal = stage["key"] == "anneal"

        pools = {opus.ACCEPT: [], opus.DEFER: [], opus.REJECT: []}
        decisions, deferred = [], []
        best = None
        gate = {"band_filtered": 0, "reserve_filtered": 0, "band_relaxed": False}

        for attempt in range(MAX_ATTEMPTS):
            e = entries[draw_index(self.seed, step, slot, attempt, len(entries))]

            # 1. Firewall. The catalogue holds every split, so this fires for real.
            if not self.firewall.check(e["shard_id"], reason=f"step {step} slot {slot}"):
                continue
            # 2. Anneal reserve.
            if is_reserved(e) and not in_anneal:
                gate["reserve_filtered"] += 1
                continue
            # 3. Difficulty band for this curriculum stage.
            if e["meta"].get("band", "B2") not in allowed_bands:
                gate["band_filtered"] += 1
                continue

            doc = self.cat.doc(e)
            decision, score, would = snap.decide(doc["ids"], lane, protected)
            rec = {"step": step, "slot": slot, "lane": lane, "stage": stage["key"],
                   "doc_id": doc["doc_id"], "shard_id": e["shard_id"],
                   "band": e["meta"].get("band"), "protected": protected,
                   "decision": decision, "score": round(score, 6), "would_have_been": would,
                   "snapshot_id": snap.id, "snapshot_hash": snap.hash,
                   "threshold_accept": round(snap.accept_at, 6),
                   "threshold_reject": round(snap.reject_at, 6),
                   "filled_from": None}
            decisions.append(rec)
            if best is None or score > best[0]:
                best = (score, doc, rec)

            bucket = opus.ACCEPT if decision in (opus.ACCEPT, opus.OVERRIDE) else decision
            pools[bucket].append((doc, rec))
            if decision == opus.DEFER:
                deferred.append({"step": step, "lane": lane, "doc_id": doc["doc_id"],
                                 "score": round(score, 6), "snapshot_id": snap.id,
                                 "requeue_after": snap.generation + 1})
            if sum(len(d["ids"]) for d, _ in pools[opus.ACCEPT]) >= self.seq_len:
                break

        # If the band gate starved the slot, relax it rather than emit nothing - and say so.
        if not any(pools.values()) and (gate["band_filtered"] or gate["reserve_filtered"]):
            gate["band_relaxed"] = True
            for attempt in range(MAX_ATTEMPTS):
                e = entries[draw_index(self.seed, step, slot, attempt, len(entries))]
                if not self.firewall.check(e["shard_id"], reason=f"step {step} slot {slot} relax"):
                    continue
                if is_reserved(e) and not in_anneal:
                    continue
                doc = self.cat.doc(e)
                decision, score, would = snap.decide(doc["ids"], lane, protected)
                rec = {"step": step, "slot": slot, "lane": lane, "stage": stage["key"],
                       "doc_id": doc["doc_id"], "shard_id": e["shard_id"],
                       "band": e["meta"].get("band"), "protected": protected,
                       "decision": decision, "score": round(score, 6), "would_have_been": would,
                       "snapshot_id": snap.id, "snapshot_hash": snap.hash,
                       "threshold_accept": round(snap.accept_at, 6),
                       "threshold_reject": round(snap.reject_at, 6),
                       "filled_from": None, "band_relaxed": True}
                decisions.append(rec)
                bucket = opus.ACCEPT if decision in (opus.ACCEPT, opus.OVERRIDE) else decision
                pools[bucket].append((doc, rec))
                if best is None or score > best[0]:
                    best = (score, doc, rec)
                if pools[opus.ACCEPT]:
                    break

        # Preference order: accepted, then deferred, then the best rejected.
        chosen, budget = [], self.seq_len
        for source in (opus.ACCEPT, opus.DEFER):
            for doc, rec in pools[source]:
                if budget <= 0:
                    break
                rec["filled_from"] = source
                chosen.append(doc)
                budget -= len(doc["ids"])
        if not chosen and best is not None:
            _score, doc, rec = best
            rec["filled_from"] = "exhaustion"
            chosen.append(doc)

        if not chosen:
            return None, decisions, deferred, gate
        seqs, _dropped = packing.pack(lane, chosen, seq_len=self.seq_len)
        return (seqs[0] if seqs else None), decisions, deferred, gate

    def build_batch(self, step, by_stage):
        """Build the batch for `step`. Deterministic given the scheduler state passed in."""
        stage = self.mixture.stage_at(step, self.total_steps)
        sc = by_stage.setdefault(stage["key"], {})
        picks = self.mixture.choose(stage, sc, self.batch_slots, self.seq_len)
        snap = self.snapshot(step)

        seqs, all_dec, all_def, lane_tokens, loss_tokens = [], [], [], {}, 0
        gates = {"band_filtered": 0, "reserve_filtered": 0, "band_relaxed": 0}
        for slot, lane in enumerate(picks):
            packed, dec, dfr, gate = self._fill_slot(step, slot, lane, snap, stage)
            all_dec.extend(dec)
            all_def.extend(dfr)
            gates["band_filtered"] += gate.get("band_filtered", 0)
            gates["reserve_filtered"] += gate.get("reserve_filtered", 0)
            gates["band_relaxed"] += 1 if gate.get("band_relaxed") else 0
            if packed is None:
                continue
            seqs.append(packed)
            lane_tokens[lane] = lane_tokens.get(lane, 0) + packed.used
            loss_tokens += packed.loss_tokens

        violations = self.firewall.assert_batch(
            [p for s in seqs for p in s.provenance])

        return {
            "step": step, "stage": stage["key"], "seqs": seqs, "picks": picks,
            "lane_tokens": lane_tokens, "loss_tokens": loss_tokens,
            "opus_decisions": all_dec, "deferred": all_def,
            "firewall_violations": violations,
            "snapshot_id": snap.id, "snapshot_hash": snap.hash,
            "stage_bands": sorted(self.mixture.bands_for(stage)),
            "gates": gates,
        }
