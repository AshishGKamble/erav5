"""
Stamp every artefact with the code that produced it, so staleness is loud rather than invisible.

This exists because of a real failure. `kron_model.py` gained a vectorised matmul after two training
artefacts had already been written. The new path is mathematically identical and **not bit
identical**, so the committed numbers were no longer what the code produced, and nothing said so.
It was found by comparing file modification times by hand, which is not a method.

So each artefact now carries the SHA-256 of every file in `common/` plus the script that wrote it.
`check_stale()` recomputes those hashes and reports any artefact whose code has moved since. It
cannot repair anything; it can only stop the mismatch being silent.
"""
import hashlib
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def code_fingerprint(script_path=None):
    """SHA of every module in `common/`, plus the calling experiment script."""
    out = {}
    for name in sorted(os.listdir(HERE)):
        if name.endswith(".py") and name != "provenance.py":
            out["common/" + name] = _sha(os.path.join(HERE, name))
    if script_path and os.path.exists(script_path):
        out["script"] = _sha(script_path)
        out["script_name"] = os.path.basename(script_path)
    return out


def stamp(result, script_path=None):
    """Attach the fingerprint to a result dict, in place, and return it."""
    result["_provenance"] = {
        "code": code_fingerprint(script_path),
        "note": ("SHA-256 prefixes of the code that produced this artefact. If these disagree with "
                 "the working tree, the artefact predates a code change and must be regenerated. "
                 "Check with `python -m provenance` from `common/`."),
    }
    return result


def check_stale(roots):
    """Report artefacts whose recorded code fingerprint no longer matches the working tree."""
    current = code_fingerprint()
    rows = []
    for root in roots:
        if not os.path.isdir(root):
            continue
        for name in sorted(os.listdir(root)):
            if not name.endswith(".json"):
                continue
            path = os.path.join(root, name)
            try:
                with open(path, encoding="utf-8") as fh:
                    data = json.load(fh)
            except (ValueError, OSError):
                continue
            prov = data.get("_provenance", {}).get("code") if isinstance(data, dict) else None
            if not prov:
                rows.append((path, "no fingerprint", []))
                continue
            moved = [k for k, v in prov.items()
                     if k.startswith("common/") and current.get(k) != v]
            rows.append((path, "STALE" if moved else "ok", moved))
    return rows


if __name__ == "__main__":
    import sys
    base = os.path.join(HERE, "..")
    roots = [os.path.join(base, p, "artifacts") for p in
             ("problem-3-dynamic-length", "problem-5-reversibility")]
    bad = 0
    for path, status, moved in check_stale(roots):
        rel = os.path.relpath(path, base)
        if status == "ok":
            print(f"ok    {rel}")
        else:
            bad += 1
            print(f"{status:5s} {rel}" + (f"  (changed: {', '.join(moved)})" if moved else ""))
    print(f"\n{bad} artefact(s) need regenerating" if bad else "\nall artefacts match the current code")
    sys.exit(1 if bad else 0)


def backfill(roots, reason):
    """Stamp existing artefacts with the current fingerprint. One time use, and honest about it.

    Legitimate only when the artefacts are known to have been produced by exactly this `common/`
    code, which here means immediately after a verified full regeneration. The stamp records
    `backfilled` and the reason so nobody later mistakes it for a fingerprint taken at write time.
    The recorded script hash is the script as it stands now, which is why `check_stale` compares
    only the `common/` entries.
    """
    current = code_fingerprint()
    done = []
    for root in roots:
        if not os.path.isdir(root):
            continue
        for name in sorted(os.listdir(root)):
            if not name.endswith(".json"):
                continue
            path = os.path.join(root, name)
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
            if not isinstance(data, dict) or "_provenance" in data:
                continue
            data["_provenance"] = {"code": dict(current), "backfilled": True, "reason": reason}
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2, sort_keys=True, ensure_ascii=False)
            done.append(path)
    return done
