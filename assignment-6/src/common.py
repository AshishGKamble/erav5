"""
Shared primitives: paths, canonical serialisation, hashing.

Every hash in this system goes through `canonical()` first. That is not decoration - two
machines must agree byte-for-byte on what a record *is* before they can agree on its hash, and
Python's default `json.dumps` does not give that (key order and whitespace both vary with how
the dict was built).

What is deliberately NOT hashed anywhere: loss values, timings, throughput. Floating-point
results shift with the BLAS backend and thread count, so hashing them would make a correct
system fail its own replay check on someone else's machine. Losses are logged and compared with
a tolerance; token ids and spans are what carry identity.
"""
import hashlib
import io
import json
import os

A6 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FROZEN = os.path.join(A6, "frozen")
CORPUS = os.path.join(FROZEN, "corpus")
ART = os.path.join(A6, "submission_artifacts")
SHARDS = os.path.join(ART, "shards")
MANIFESTS = os.path.join(ART, "manifests")
LEDGERS = os.path.join(ART, "ledgers")
CHECKPOINTS = os.path.join(ART, "checkpoints")


def canonical(obj):
    """The one serialisation used for hashing: sorted keys, no incidental whitespace, UTF-8."""
    return json.dumps(obj, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_obj(obj):
    return hashlib.sha256(canonical(obj)).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with io.open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path):
    with io.open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with io.open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def read_json(path):
    with io.open(path, encoding="utf-8") as fh:
        return json.load(fh)


def append_jsonl(path, obj):
    """Append one record. Ledgers are append-only; nothing in this system rewrites a ledger line."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with io.open(path, "a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(obj, ensure_ascii=False, sort_keys=True,
                            separators=(",", ":")) + "\n")


def ensure_dirs(*paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)
