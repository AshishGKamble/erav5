"""
Build and validate the shard manifest.

The manifest is the single authority on what exists and what it hashes to. Shard files carry no
hashes of their own, so validation is a genuine cross-check: recompute from the bytes on disk and
compare against an independent description. Tamper with the data and the manifest catches it;
tamper with the manifest and it stops matching the data.

Four things are verified, and each maps to a claim the submission makes:

  frozen inputs    the tokenizer and corpus are the ones recorded, unchanged
  shard integrity  every .bin and .json still hashes to what the manifest says
  self-integrity   the manifest itself has not been edited since it was written
  split disjoint   no document appears in two splits - the evaluation firewall's foundation
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (CORPUS, FROZEN, MANIFESTS, SHARDS, canonical, read_json, sha256_bytes,
                    sha256_file, write_json)

MANIFEST_PATH = os.path.join(MANIFESTS, "manifest.json")


def build(shard_records, built_at=None):
    """Assemble the manifest from shard records plus the frozen-input provenance.

    `built_at` is passed in rather than read from the clock, so a caller that wants a reproducible
    manifest can supply a fixed value. It is metadata only and never enters a shard's identity.
    """
    prov = read_json(os.path.join(FROZEN, "PROVENANCE.json"))
    sources = read_json(os.path.join(CORPUS, "SOURCES.json"))

    lanes = {}
    for r in shard_records:
        e = lanes.setdefault(r["lane"], {"shards": 0, "tokens": 0, "loss_tokens": 0,
                                         "raw_chars": 0, "protected": r["protected"],
                                         "splits": {}, "licences": set()})
        e["shards"] += 1
        e["tokens"] += r["n_tokens"]
        e["loss_tokens"] += r["loss_tokens"]
        e["raw_chars"] += r["raw_chars"]
        e["splits"][r["split"]] = e["splits"].get(r["split"], 0) + r["n_tokens"]
        e["licences"].update(r["licences"])
    for e in lanes.values():
        e["licences"] = sorted(e["licences"])
        e["fertility"] = round(e["tokens"] / e["raw_chars"], 4) if e["raw_chars"] else None
        # Measured, not configured. plan.json predicts 0.35 for agentic; this reports what the
        # corpus actually contains, and the README explains the gap rather than adopting the plan.
        e["trainable_frac_measured"] = (round(e["loss_tokens"] / e["tokens"], 4)
                                        if e["tokens"] else None)

    manifest = {
        "manifest_version": 1,
        "built_at": built_at,
        "tokenizer": prov["tokenizer"],
        "plan": prov["plan"],
        "corpus_sources": sources["lanes"],
        "lanes": lanes,
        "shards": sorted(shard_records, key=lambda r: r["shard_id"]),
        "totals": {
            "shards": len(shard_records),
            "tokens": sum(r["n_tokens"] for r in shard_records),
            "loss_tokens": sum(r["loss_tokens"] for r in shard_records),
            "documents": sum(r["n_docs"] for r in shard_records),
        },
    }
    manifest["manifest_sha256"] = sha256_bytes(canonical(manifest))
    return manifest


def save(manifest):
    write_json(MANIFEST_PATH, manifest)
    return MANIFEST_PATH


def validate(manifest=None):
    """Recompute everything the manifest claims. Returns (ok, problems, stats).

    Nothing here trusts a value it was handed - every hash is recomputed from bytes on disk.
    """
    manifest = manifest or read_json(MANIFEST_PATH)
    problems, checked = [], 0

    # -- self-integrity: strip the recorded hash and recompute over the rest.
    body = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    if sha256_bytes(canonical(body)) != manifest.get("manifest_sha256"):
        problems.append("manifest_sha256 does not match its own contents")

    # -- frozen inputs still the ones recorded.
    tok_path = os.path.join(FROZEN, "tokenizer.json")
    if sha256_file(tok_path) != manifest["tokenizer"]["sha256"]:
        problems.append("tokenizer.json hash differs from the manifest record")
    for src in manifest["corpus_sources"]:
        p = os.path.join(CORPUS, src["file"])
        if not os.path.exists(p):
            problems.append(f"corpus file missing: {src['file']}")
        elif sha256_file(p) != src["sha256"]:
            problems.append(f"corpus file changed since fetch: {src['file']}")

    # -- shard integrity, recomputed from the bytes.
    seen_docs = {}
    for r in manifest["shards"]:
        bin_path = os.path.join(SHARDS, r["bin"])
        idx_path = os.path.join(SHARDS, r["index"])
        if not (os.path.exists(bin_path) and os.path.exists(idx_path)):
            problems.append(f"{r['shard_id']}: files missing")
            continue
        content_sha = sha256_file(bin_path)
        index = read_json(idx_path)
        index_sha = sha256_bytes(canonical(index))
        shard_sha = sha256_bytes(canonical({
            "shard_id": r["shard_id"], "lane": r["lane"], "split": r["split"],
            "protected": r["protected"], "tokenizer_sha256": r["tokenizer_sha256"],
            "content_sha256": content_sha, "index_sha256": index_sha}))
        if content_sha != r["content_sha256"]:
            problems.append(f"{r['shard_id']}: token data changed")
        if index_sha != r["index_sha256"]:
            problems.append(f"{r['shard_id']}: index changed")
        if shard_sha != r["shard_sha256"]:
            problems.append(f"{r['shard_id']}: identity changed (lane/split/tokenizer relabelled?)")

        # -- split disjointness. A document in two splits is the exact failure A5 section 9.2
        #    recorded, so it is checked directly rather than assumed from the hashing scheme.
        for d in index["docs"]:
            prev = seen_docs.get(d["id"])
            if prev is not None and prev != r["split"]:
                problems.append(f"document {d['id']} appears in both {prev} and {r['split']}")
            seen_docs[d["id"]] = r["split"]
        checked += 1

    stats = {"shards_checked": checked, "documents_checked": len(seen_docs),
             "problems": len(problems)}
    return (not problems), problems, stats


def verify_frozen_inputs():
    """Step 0 of the demo: refuse to proceed if an input is not what was recorded."""
    prov = read_json(os.path.join(FROZEN, "PROVENANCE.json"))
    sources = read_json(os.path.join(CORPUS, "SOURCES.json"))
    problems = []
    for key, fname in (("tokenizer", "tokenizer.json"), ("plan", "plan.json")):
        path = os.path.join(FROZEN, fname)
        if not os.path.exists(path):
            problems.append(f"frozen input missing: {fname}")
        elif sha256_file(path) != prov[key]["sha256"]:
            problems.append(f"frozen input modified: {fname}")
    for src in sources["lanes"]:
        p = os.path.join(CORPUS, src["file"])
        if not os.path.exists(p):
            problems.append(f"corpus slice missing: {src['file']}")
        elif sha256_file(p) != src["sha256"]:
            problems.append(f"corpus slice modified: {src['file']}")
    return (not problems), problems


if __name__ == "__main__":
    import shards
    ok, probs = verify_frozen_inputs()
    print("frozen inputs:", "OK" if ok else probs)
    recs = shards.build_all()
    m = build(recs, built_at="(cli)")
    save(m)
    ok, probs, stats = validate(m)
    print(f"\nmanifest sha256 {m['manifest_sha256'][:16]}...")
    print("validate:", "OK" if ok else probs, stats)
