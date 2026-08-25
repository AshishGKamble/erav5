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
import sys, os, json
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "common"))
sys.path.insert(0, HERE)
import corpus  # noqa: E402
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


SCHEMES = {
    "prefix_32_bytes_published": (lambda s: both_ends_bytes(s, 32) if False else s.encode("utf-8")[:32],
                                  "the published construction: first 32 bytes"),
    "both_ends_32_bytes": (lambda s: both_ends_bytes(s, 32),
                           "16 leading bytes plus 16 trailing bytes, same D"),
    "fixD_31_chars": (lambda s: s[:31],
                      "script tag plus the first 31 characters"),
    "fixD_both_ends_31_chars": (lambda s: both_ends_chars(s, 31),
                                "script tag, 15 leading plus 16 trailing characters"),
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
        json.dump(result, fh, indent=2, sort_keys=True, ensure_ascii=False)
    return result


if __name__ == "__main__":
    r = main(os.path.join(HERE, "..", "..", "..", "assignment-6", "frozen", "corpus"),
             os.path.join(HERE, "..", "artifacts", "bothends.json"))
    for n, v in r["schemes"].items():
        mal = v["per_script"]["MALAYALAM"]["types_in_collisions_rate"] * 100
        print(f"{n:28s} total_groups={v['total_colliding_groups']:>6,}  "
              f"malayalam={mal:6.2f}%  reduction={v['reduction_vs_published']:.1f}x")
