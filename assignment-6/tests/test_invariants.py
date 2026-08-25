"""
Automated tests for the invariants the submission claims.

Each test is named for the property it protects, not the function it calls. They run against the
artefacts `run_demo.py` produced, so a passing suite means the *shipped* evidence holds - not that
a fresh in-memory object would have behaved.

    python -m pytest tests/ -q          (or: python tests/test_invariants.py)
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
A6 = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(A6, "src"))

import audit                                             # noqa: E402
import ledger                                            # noqa: E402
import manifest as mf                                    # noqa: E402
import opus                                              # noqa: E402
import packing                                           # noqa: E402
import shards                                            # noqa: E402
from common import SHARDS, read_json                     # noqa: E402
from mixture import Mixture                              # noqa: E402
from model import Tiny                                   # noqa: E402
from stream import Catalogue, Stream                     # noqa: E402

M = read_json(mf.MANIFEST_PATH)
MX = Mixture()
CONS = ledger.read(ledger.CONSUMPTION, "main")


# ---------------------------------------------------------------- shards and manifest

def test_shard_determinism():
    """Rebuilding every shard from the frozen corpus reproduces identical hashes."""
    before = {r["shard_id"]: r["shard_sha256"] for r in M["shards"]}
    recs = shards.build_all(log=lambda *_a, **_k: None)
    after = {r["shard_id"]: r["shard_sha256"] for r in recs}
    assert before == after, "shard hashes changed on rebuild"


def test_tokenizer_frozen_and_manifest_valid():
    ok, problems = mf.verify_frozen_inputs()
    assert ok, problems
    ok, problems, stats = mf.validate(M)
    assert ok, problems
    assert stats["problems"] == 0


def test_tamper_is_detected():
    """A single flipped byte in any shard must fail validation."""
    import io as _io
    target = M["shards"][0]
    path = os.path.join(SHARDS, target["bin"])
    original = _io.open(path, "rb").read()
    try:
        b = bytearray(original)
        b[8] ^= 0xFF
        _io.open(path, "wb").write(bytes(b))
        ok, problems, _ = mf.validate(M)
        assert not ok and any("token data changed" in p for p in problems)
    finally:
        _io.open(path, "wb").write(original)


def test_splits_are_document_disjoint():
    """No document may appear in two splits - the failure A5 section 9.2 recorded."""
    seen = {}
    for r in M["shards"]:
        for d in read_json(os.path.join(SHARDS, r["index"]))["docs"]:
            assert seen.get(d["id"], r["split"]) == r["split"], f"{d['id']} in two splits"
            seen[d["id"]] = r["split"]
    assert len(seen) == M["totals"]["documents"]


# ---------------------------------------------------------------- packing and masks

def _load(lane, limit=60):
    sid = next(r["shard_id"] for r in M["shards"] if r["lane"] == lane and r["split"] == "train")
    idx = read_json(os.path.join(SHARDS, f"{sid}.json"))
    toks = shards.load_shard_tokens(sid)
    return [{"shard_id": sid, "doc_id": d["id"], "meta": d["meta"],
             "ids": toks[d["off"]:d["off"] + d["len"]].astype(np.int32), "spans": d["spans"]}
            for d in idx["docs"][:limit]]


def test_masks_positions_and_attention():
    """Padding carries no loss; positions restart per document; attention is block-diagonal."""
    for lane in MX.lanes:
        seqs, _ = packing.pack(lane, _load(lane), seq_len=512)
        assert seqs, f"{lane} produced no sequences"
        for s in seqs:
            pad = s.segment_ids == 0
            assert not s.loss_mask[pad].any(), f"{lane}: padding carries loss"
            for p in s.provenance:
                assert s.position_ids[p["seq_start"]] == p["tok_start"]
                run = s.position_ids[p["seq_start"]:p["seq_end"]]
                assert len(run) < 2 or np.all(np.diff(run) == 1)
            am = packing.attention_mask(s.segment_ids)
            assert not (am & (s.segment_ids[:, None] != s.segment_ids[None, :])).any()
            assert not am[pad].any() and not am[:, pad].any()


def test_packing_correct_at_long_sequence_lengths():
    """The demo trains at 512 for compute reasons; correctness is exercised at 4K and 32K here.

    Also records the ATOMIC drop rate as a function of window size, which is what shows the 34%
    agentic drop at 512 is the window's doing and not the policy's.
    """
    for seq_len in (4096, 32768):
        for lane in MX.lanes:
            seqs, _dropped = packing.pack(lane, _load(lane, limit=30), seq_len=seq_len)
            assert seqs, f"{lane} produced no sequences at seq_len={seq_len}"
            for s in seqs:
                pad = s.segment_ids == 0
                assert not s.loss_mask[pad].any()
                for p in s.provenance:
                    assert s.position_ids[p["seq_start"]] == p["tok_start"]
                    run = s.position_ids[p["seq_start"]:p["seq_end"]]
                    assert len(run) < 2 or np.all(np.diff(run) == 1)
                # Block-diagonal isolation, checked on a slice to keep the matrix tractable.
                seg = s.segment_ids[:2048]
                am = packing.attention_mask(seg)
                assert not (am & (seg[:, None] != seg[None, :])).any()

    docs = _load("agentic", limit=120)
    drops = {}
    for seq_len in (512, 4096):
        _seqs, dropped = packing.pack("agentic", docs, seq_len=seq_len)
        drops[seq_len] = len(dropped)
    assert drops[4096] < drops[512], f"ATOMIC drop rate should fall with window size: {drops}"


def test_agentic_tool_output_is_masked():
    """Only assistant spans carry loss in the agentic lane."""
    docs = _load("agentic", limit=40)
    seqs, _ = packing.pack("agentic", docs, seq_len=512)
    by_id = {d["doc_id"]: d for d in docs}
    checked = 0
    for s in seqs:
        for p in s.provenance:
            for a, b, loss in by_id[p["doc_id"]]["spans"]:
                lo, hi = max(a, p["tok_start"]), min(b, p["tok_end"])
                if lo >= hi:
                    continue
                seg = s.loss_mask[p["seq_start"] + (lo - p["tok_start"]):
                                  p["seq_start"] + (hi - p["tok_start"])]
                assert seg.all() if loss else not seg.any()
                checked += 1
    assert checked > 0


def test_model_does_not_attend_across_documents():
    """Perturbing one document must not move another document's logits at all."""
    V, T = 60, 12
    mdl = Tiny(vocab=V, d=16, n_layer=2, n_head=2, max_pos=T, seed=7)
    ids = np.array([[5, 9, 13, 21, 33, 41, 7, 8, 19, 26, 0, 0]], dtype=np.int64)
    seg = np.array([[1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 0, 0]], dtype=np.int64)
    pos = np.array([[0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 0, 0]], dtype=np.int64)
    a, _ = mdl.forward(ids, pos, seg)
    ids2 = ids.copy()
    ids2[0, 8] = 44
    b, _ = mdl.forward(ids2, pos, seg)
    assert np.abs(a[0][seg[0] == 1] - b[0][seg[0] == 1]).max() == 0.0
    assert np.abs(a[0][seg[0] == 2] - b[0][seg[0] == 2]).max() > 0.0


def test_gradients_match_finite_differences():
    """Hand-written backprop against central differences, in float64."""
    import model as M_
    old = M_.DTYPE
    try:
        M_.DTYPE = np.float64
        rng = np.random.default_rng(0)
        V, B, T = 60, 2, 12
        mdl = M_.Tiny(vocab=V, d=16, n_layer=2, n_head=2, max_pos=T, seed=7)
        ids = rng.integers(1, V, (B, T)).astype(np.int64)
        seg = np.array([[1] * 6 + [2] * 4 + [0] * 2, [1] * 12])
        pos = np.stack([np.concatenate([np.arange(6), np.arange(4), np.zeros(2)]),
                        np.arange(T)]).astype(np.int64)
        lm = (seg != 0).astype(np.int8)
        _loss, grads, _, _ = mdl.loss_and_grad(ids, pos, seg, lm)
        eps = 1e-5
        for name in ["emb", "pos", "wq0", "wk0", "wv0", "wo0", "w10", "w20",
                     "ln1_g0", "ln2_b0", "lnf_g", "w21"]:
            P = mdl.p[name]
            idx = tuple(rng.integers(0, s) for s in P.shape)
            o = float(P[idx])
            P[idx] = o + eps
            a, _, _, _ = mdl.loss_and_grad(ids, pos, seg, lm)
            P[idx] = o - eps
            c, _, _, _ = mdl.loss_and_grad(ids, pos, seg, lm)
            P[idx] = o
            num, ana = (a - c) / (2 * eps), float(grads[name][idx])
            rel = abs(num - ana) / max(1e-12, abs(num) + abs(ana))
            assert rel < 1e-4, f"{name}: rel {rel:.2e}"
    finally:
        M_.DTYPE = old


# ---------------------------------------------------------------- mixture, OPUS, firewall

def test_mixture_compliance_and_protected_floor():
    c = audit.compliance_from_ledger(CONS, MX)
    assert c["max_abs_delta"] <= 5.0, c["rows"]
    assert c["floor_held"], c["indic_cumulative_floor_min_pct"]
    assert c["present_in_every_batch"]


def test_no_eval_data_in_any_training_batch():
    """Every document in every shipped batch came from a train-split shard."""
    split_of = {r["shard_id"]: r["split"] for r in M["shards"]}
    n = 0
    for rec in CONS:
        for seq in rec["provenance"]:
            for shard_id, _doc, _a, _b in seq["docs"]:
                assert split_of[shard_id] == "train", f"{shard_id} is {split_of[shard_id]}"
                n += 1
    assert n > 0


def test_firewall_recorded_blocks():
    events = ledger.read(ledger.FIREWALL, "main")
    assert events, "firewall never fired - the attack did not happen"
    assert all(e["split"] in ("eval", "holdout") for e in events)


def test_opus_never_rejects_a_protected_lane():
    a = opus.audit(ledger.read(ledger.OPUS, "main"))
    assert a["protected_rejected"] == 0
    assert a["overrides"] > 0
    assert a["deferrals"] > 0
    assert len(a["snapshots_used"]) > 1, "snapshot drift never exercised"


# ---------------------------------------------------------------- run reconstruction

def test_ledger_is_contiguous():
    """No skipped or repeated batch ids, in any branch."""
    for branch in ("main", "crash"):
        recs = ledger.read(ledger.CONSUMPTION, branch)
        assert recs, branch
        assert ledger.contiguity(recs) == [], (branch, ledger.contiguity(recs))


def test_resume_equals_reference_run():
    eq = audit.resume_equivalence(CONS, ledger.read(ledger.CONSUMPTION, "crash"))
    assert eq["ok"], eq["problems"]
    assert eq["compared"] == len(CONS)


def test_resume_reproduces_losses_exactly():
    """Optimiser moments are checkpointed, so the resumed run must reproduce the same losses.

    Data equivalence alone would pass even with a fresh optimiser, because the batch stream does
    not depend on model state. Loss equivalence is what proves the *training* resumed.
    """
    ref = {r["step"]: r["loss"] for r in ledger.read(ledger.LEARNING, "main")}
    res = {r["step"]: r["loss"] for r in ledger.read(ledger.LEARNING, "crash")}
    shared = sorted(set(ref) & set(res))
    assert shared
    bad = [k for k in shared if ref[k] != res[k]]
    assert not bad, f"losses differ at steps {bad[:5]}"


def test_curriculum_gates_on_difficulty_band():
    """Stage band ranges are enforced on real draws, not merely declared in plan.json."""
    filtered = sum(r.get("gates", {}).get("band_filtered", 0) for r in CONS)
    assert filtered > 0, "band gating never fired - the curriculum is not gating anything"
    for r in CONS:
        assert r.get("stage_bands"), f"step {r['step']} recorded no stage bands"


def test_anneal_reserve_is_actually_held_back():
    """Reserved documents must be refused outside the anneal stage."""
    held = sum(r.get("gates", {}).get("reserve_filtered", 0)
               for r in CONS if r["stage"] != "anneal")
    assert held > 0, "the anneal reserve never blocked a draw"


def test_deferral_is_acted_on_not_just_recorded():
    """A deferred candidate must be preferred over a rejected one when filling a slot."""
    dec = ledger.read(ledger.OPUS, "main")
    filled = {}
    for d in dec:
        if d.get("filled_from"):
            filled[d["filled_from"]] = filled.get(d["filled_from"], 0) + 1
    assert filled.get("defer", 0) > 0, f"deferral never used to fill a slot: {filled}"
    assert filled.get("accept", 0) > filled.get("defer", 0)


def test_firewall_guards_a_pool_that_contains_eval_data():
    """The candidate pool must actually contain non-train documents for the firewall to matter."""
    cat = Catalogue(M)
    splits = {e["split"] for lane in cat.by_lane.values() for e in lane}
    assert splits == {"train", "eval", "holdout"}, splits
    non_train = sum(1 for lane in cat.by_lane.values() for e in lane if e["split"] != "train")
    assert non_train > 0


def test_atomic_never_splits_a_loss_bearing_span():
    """ATOMIC lanes may split between turns, but never inside a span that carries loss."""
    for lane in ("agentic", "reasoning"):
        docs = _load(lane, limit=120)
        by = {d["doc_id"]: d for d in docs}
        seqs, _ = packing.pack(lane, docs, seq_len=512)
        checked = 0
        for s in seqs:
            for p in s.provenance:
                for a, b, loss in by[p["doc_id"]]["spans"]:
                    if not loss or max(a, p["tok_start"]) >= min(b, p["tok_end"]):
                        continue
                    assert a >= p["tok_start"] and b <= p["tok_end"], \
                        f"{lane}: loss-bearing span {a}-{b} split across sequences"
                    checked += 1
        assert checked > 0


def test_replay_reproduces_recorded_hashes():
    cat = Catalogue(M)
    st = Stream(M, MX, cat, seq_len=CONS[0]["seq_len"],
                batch_slots=max(r["sequences"] for r in CONS),
                total_steps=len(CONS), opus_rerun_steps=20)
    rep = audit.replay_interval(st, CONS, 12, 24)
    assert rep["ok"], rep["mismatches"]
    assert rep["checked"] == 12


def test_fork_shares_prefix_and_diverges():
    lin = audit.fork_lineage(CONS, ledger.read(ledger.CONSUMPTION, "fork"), 20)
    assert lin["ok"], lin["problems"]
    assert lin["diverged_after_fork"] == lin["compared_after_fork"] > 0


def test_evidence_bundle_matches_the_ledgers():
    """Every PASS in evidence.json must still hold when recomputed from the artefacts."""
    from common import ART
    bundle = read_json(os.path.join(ART, "evidence.json"))
    assert bundle["summary"]["all_passed"], bundle["summary"]
    by = {r["key"]: r for r in bundle["requirements"]}
    c = audit.compliance_from_ledger(CONS, MX)
    assert by["mixture_compliance"]["data"]["max_abs_delta"] == c["max_abs_delta"]
    perf = read_json(os.path.join(ART, "performance.json"))
    assert by["throughput"]["data"]["tokens_total"] == perf["tokens_total"]
    assert perf["tokens_total"] == sum(sum(r["lane_tokens"].values()) for r in CONS)


if __name__ == "__main__":
    fns = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for name, fn in fns:
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as exc:
            failed += 1
            print(f"  FAIL  {name}: {str(exc)[:160]}")
    print(f"\n{len(fns) - failed}/{len(fns)} tests passed")
    sys.exit(1 if failed else 0)
