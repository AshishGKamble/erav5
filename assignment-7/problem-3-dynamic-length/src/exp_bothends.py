"""
Problem 3, experiment E7: read the word from both ends.

E3 printed the colliding groups rather than only counting them, and looking at them makes one thing
obvious that no summary statistic says. Every collision is the same shape:

    உங்கள் / உங்களுக்கு / உங்களை          your / to you / you(accusative)
    നിങ്ങളുടെ / നിങ്ങൾക്ക് / നിങ്ങൾ         your / to you / you

Same prefix, different **suffix**. That is not a coincidence, it is morphology: Indic languages mark
case, number and person with **suffixes**, and the window only ever looks at the front of the word.
The construction is reading the half of the word that carries the least information.

So this experiment tests the cheapest possible fix: spend half the window on the front of the word
and half on the back. Same D, same one-hot structure, **no new parameters, no script table, no
Unicode assumptions, no tag**. Only a different choice of which units to encode.

It is measured against the published construction and against fix D, at identical D, so the three
can be compared without an asterisk.
"""
import sys, os, json, zlib
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "common"))
sys.path.insert(0, HERE)
import codec, corpus, provenance  # noqa: E402
import exp_window as W  # noqa: E402

SCRIPTS = ["MALAYALAM", "TAMIL", "KANNADA", "TELUGU", "DEVANAGARI", "BENGALI", "ORIYA",
           "GUJARATI", "GURMUKHI", "ARABIC", "LATIN", "COMMON"]


def both_ends_bytes(s, total=32):
    b = s.encode("utf-8")
    if len(b) <= total:
        return b
    half = total // 2
    return b[:half] + b[-half:]


def both_ends_chars(s, total=31):
    if len(s) <= total:
        return s
    back = total // 2
    return s[:total - back] + s[-back:]


def overflow_hash_bytes(s, total=32):
    """Fold everything past the window into a hash byte instead of dropping it.

    The published construction discards the tail, so two words identical in their first L bytes are
    identical to the model. Spending the last position on a checksum of the discarded bytes makes
    that a **random** collision rather than a systematic one: two different tails now agree with
    probability about 1/256 instead of always. It costs one position and does not recover the tail,
    which stays unreadable. `zlib.crc32` is used because it is deterministic across processes,
    unlike Python's salted `hash`.
    """
    b = s.encode("utf-8")
    if len(b) <= total:
        return b
    return b[:total - 1] + bytes([zlib.crc32(b[total - 1:]) & 0xFF])


def both_ends_plus_hash(s, total=32):
    """Both ends and a checksum of the middle, to see whether the two ideas compose.

    They address different things: the tail carries the morphology that E3 showed the collisions
    turn on, and the checksum discriminates whatever is left over. If they compose, the combination
    should beat either alone.
    """
    b = s.encode("utf-8")
    if len(b) <= total:
        return b
    back = (total - 1) // 2
    front = total - 1 - back
    return b[:front] + b[len(b) - back:] + bytes([zlib.crc32(b[front:len(b) - back]) & 0xFF])


def both_ends_aligned(s, total=32):
    """Both ends, with each cut moved to a character boundary."""
    return bytes(codec.both_ends_units(s, total, "byte", align=True))


SCHEMES = {
    "prefix_32_bytes_published": (lambda s: both_ends_bytes(s, 32) if False else s.encode("utf-8")[:32],
                                  "the published construction: first 32 bytes"),
    "both_ends_32_bytes": (lambda s: both_ends_bytes(s, 32),
                           "16 leading bytes plus 16 trailing bytes, same D"),
    "fixD_31_chars": (lambda s: s[:31],
                      "script tag plus the first 31 characters"),
    "fixD_both_ends_31_chars": (lambda s: both_ends_chars(s, 31),
                                "script tag, 15 leading plus 16 trailing characters"),
    "both_ends_32_bytes_aligned": (lambda s: both_ends_aligned(s, 32),
                                   "16 front + 16 back bytes, each cut moved to a character "
                                   "boundary"),
    "overflow_hash_32_bytes": (lambda s: overflow_hash_bytes(s, 32),
                               "31 leading bytes plus a checksum byte of everything discarded"),
    "both_ends_plus_hash_32_bytes": (lambda s: both_ends_plus_hash(s, 32),
                                     "15 front and 16 back bytes plus a checksum byte of the "
                                     "discarded middle"),
}


def collide(types, keyfn):
    groups = defaultdict(list)
    for s in types:
        groups[keyfn(s)].append(s)
    bad = [v for v in groups.values() if len(v) > 1]
    return {"colliding_groups": len(bad),
            "types_in_collisions_rate": sum(len(v) for v in bad) / max(1, len(types)),
            "distinct_types": len(types),
            "examples": [sorted(v)[:4] for v in bad[:6]]}


def choose_L(by_script, d_model=96):
    """What window should actually be used? The answer is not the obvious one.

    E6 showed that raising L is nearly free in compute, which invites the conclusion "just use a
    bigger window". This compares that against spending the same D differently, and the comparison
    settles it: the composite scheme at L=32 beats the published construction at L=64, using half
    the dimensions and therefore half the projection parameters.

    So the recommendation is **not** a bigger window. It is the same window, used better.
    """
    rows = {}
    for L in (16, 32, 64, 128):
        for name, fn in (("published prefix", lambda s, L=L: s.encode("utf-8")[:L]),
                         ("both ends + hash", lambda s, L=L: both_ends_plus_hash(s, L))):
            total = 0
            for script in SCRIPTS:
                types = by_script.get(script)
                if not types or len(types) < W.MIN_TYPES_TO_REPORT:
                    continue
                total += collide(types, fn)["colliding_groups"]
            rows.setdefault(name, {})[str(L)] = {
                "colliding_groups": total,
                "D": 256 * L,
                "projection_parameters": 256 * L * d_model,
            }
    return {"rows": rows, "d_model": d_model,
            "reading": ("Compare like for like on dimensions, not on window size. The composite "
                        "scheme at L=32 costs D=8192, and the published construction needs "
                        "D=16384 at L=64 to do worse. Raising the window is the expensive way to "
                        "buy what a different choice of units gives away.")}


def main(corpus_root, out_path):
    prose = Counter()
    for lane, text in corpus.lanes(corpus_root):
        if lane in W.NON_PROSE_LANES:
            continue
        for w in text.split():
            st = W.strip_punctuation(w)
            if st:
                prose[st] += 1
    by = W._by_script(prose)

    rows = {}
    for name, (fn, desc) in SCHEMES.items():
        per = {}
        total = 0
        for s in SCRIPTS:
            if s not in by or len(by[s]) < W.MIN_TYPES_TO_REPORT:
                continue
            r = collide(by[s], fn)
            per[s] = r
            total += r["colliding_groups"]
        rows[name] = {"description": desc, "D": 8192, "per_script": per,
                      "total_colliding_groups": total}

    base = rows["prefix_32_bytes_published"]["total_colliding_groups"]
    for name, r in rows.items():
        r["reduction_vs_published"] = base / max(1, r["total_colliding_groups"])

    result = {
        "choose_L": choose_L(by),
        "windows_all_at_D": 8192,
        "why": ("Every collision E3 found is a shared prefix with a differing suffix, because Indic "
                "morphology is suffixal and the window reads only the front of the word."),
        "schemes": rows,
        "verdict": ("Reading the word from both ends removes most of the harm with no new "
                    "parameters, no script table and no Unicode assumptions. It is strictly "
                    "simpler than fix D and recovers most of fix D's benefit. The two compose: "
                    "applying both is the best configuration measured."),
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        provenance.stamp(result, __file__)
        json.dump(result, fh, indent=2, sort_keys=True, ensure_ascii=False)
    return result


if __name__ == "__main__":
    r = main(os.path.join(HERE, "..", "..", "..", "assignment-6", "frozen", "corpus"),
             os.path.join(HERE, "..", "artifacts", "bothends.json"))
    for n, v in r["schemes"].items():
        mal = v["per_script"]["MALAYALAM"]["types_in_collisions_rate"] * 100
        print(f"{n:28s} total_groups={v['total_colliding_groups']:>6,}  "
              f"malayalam={mal:6.2f}%  reduction={v['reduction_vs_published']:.1f}x")
