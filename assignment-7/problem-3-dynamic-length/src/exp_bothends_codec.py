"""
Problem 3, E7 continued: is the both-ends scheme actually a codec, or only a key?

E7 measured collisions by grouping words on a byte key. That is a correct statement about the key
and **not** a statement about the codec, and the difference matters because this writeup recommends
the scheme. So the recommendation is checked here the way E3 checked the published construction:

  * does `encode` then `decode` return the units it was given
  * do two words sharing a both-ends key really produce **bitwise identical** vectors
  * and what does the extra cut cost

That last one is the point of the file. The published window cuts a word once, at the end. This
scheme cuts twice, and the second cut opens the retained tail, so a multi-byte character can be
split there as well. For a three-byte-per-character script that is not a rare event, and it is
measured here rather than waved at, along with the aligned variant that removes it.
"""
import sys, os, json
from collections import Counter, defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "common"))
sys.path.insert(0, HERE)
import codec, corpus, provenance  # noqa: E402
import exp_window as W  # noqa: E402

L = 32
SCRIPTS = ["MALAYALAM", "TAMIL", "KANNADA", "TELUGU", "DEVANAGARI", "BENGALI", "ORIYA",
           "GUJARATI", "GURMUKHI", "LATIN"]


def prose_words(root):
    c = Counter()
    for lane, text in corpus.lanes(root):
        if lane in W.NON_PROSE_LANES:
            continue
        for w in text.split():
            st = W.strip_punctuation(w)
            if st:
                c[st] += 1
    return c


def roundtrip(words, scheme, align=False):
    """encode then decode must return the units it was given."""
    ok = n = 0
    for w in words:
        u = (codec.both_ends_units(w, L, "byte", align=align) if scheme != "prefix"
             else codec.text_units(w, "byte")[:L])
        v, _, _ = codec.encode(u, L, "byte")
        back, _ = codec.decode(v, L, "byte", length=len(u))
        ok += back == list(u)
        n += 1
    return {"tokens": n, "recovered": ok, "rate": ok / n if n else None}


def bitwise_collisions(words, align=False, max_pairs=300):
    """Do words sharing a both-ends key produce identical vectors? Measure, do not assume."""
    groups = defaultdict(list)
    for w in words:
        groups[bytes(codec.both_ends_units(w, L, "byte", align=align))].append(w)
    pairs = [(v[0], v[1]) for v in groups.values() if len(v) > 1][:max_pairs]
    identical = 0
    worst = 0.0
    for a, b in pairs:
        va, _, _ = codec.encode(codec.both_ends_units(a, L, "byte", align=align), L, "byte")
        vb, _, _ = codec.encode(codec.both_ends_units(b, L, "byte", align=align), L, "byte")
        d = float(np.abs(va - vb).max())
        worst = max(worst, d)
        identical += d == 0.0
    return {"pairs_checked": len(pairs), "bitwise_identical": identical,
            "max_absolute_difference": worst,
            "verdict": "confirmed" if pairs and identical == len(pairs) else
                       ("no colliding pairs" if not pairs else "FAILED")}


def cut_quality(by_script, align=False):
    """How often does a cut land mid-character, and how much text survives as valid UTF-8?"""
    out = {}
    for script in SCRIPTS:
        types = by_script.get(script)
        if not types or len(types) < W.MIN_TYPES_TO_REPORT:
            continue
        cropped = pre_bad = be_bad = fixd_bad = 0
        for w in types:
            b = w.encode("utf-8")
            if len(b) <= L:
                continue
            cropped += 1
            try:
                bytes(codec.text_units(w, "byte")[:L]).decode("utf-8")
            except UnicodeDecodeError:
                pre_bad += 1
            try:
                bytes(codec.both_ends_units(w, L, "byte", align=align)).decode("utf-8")
            except UnicodeDecodeError:
                be_bad += 1
            # Fix D encodes characters, so a cut can never land inside one. Counted rather than
            # asserted, so the comparison is like for like.
            try:
                "".join(w[:L - 1])
            except (ValueError, TypeError):
                fixd_bad += 1
        if cropped:
            out[script] = {"cropped_types": cropped,
                           "prefix_invalid_utf8_rate": pre_bad / cropped,
                           "both_ends_invalid_utf8_rate": be_bad / cropped,
                           "fix_d_character_units_invalid_rate": fixd_bad / cropped}
    return out


def capacity_cost(by_script, align=True):
    """Aligning the back cut costs capacity. How much, in units actually retained?"""
    kept_plain = kept_aligned = n = 0
    for script in SCRIPTS:
        for w in by_script.get(script, {}):
            if len(w.encode("utf-8")) <= L:
                continue
            kept_plain += len(codec.both_ends_units(w, L, "byte", align=False))
            kept_aligned += len(codec.both_ends_units(w, L, "byte", align=True))
            n += 1
    return {"cropped_types": n,
            "mean_units_kept_unaligned": kept_plain / n if n else None,
            "mean_units_kept_aligned": kept_aligned / n if n else None}


def main(corpus_root, out_path, sample=4000):
    counts = prose_words(corpus_root)
    by_script = W._by_script(counts)
    rng = np.random.default_rng(20260825)
    allw = list(counts)
    probe = [allw[i] for i in rng.choice(len(allw), size=min(sample, len(allw)), replace=False)]

    result = {
        "window": L,
        "question": ("E7 measured a key. This file checks whether that key is a working codec, "
                     "because the writeup recommends it."),
        "roundtrip_prefix": roundtrip(probe, "prefix"),
        "roundtrip_both_ends": roundtrip(probe, "both_ends"),
        "roundtrip_both_ends_aligned": roundtrip(probe, "both_ends", align=True),
        "bitwise_collisions": bitwise_collisions(list(counts)),
        "bitwise_collisions_aligned": bitwise_collisions(list(counts), align=True),
        "cut_quality": cut_quality(by_script),
        "cut_quality_aligned": cut_quality(by_script, align=True),
        "capacity_cost_of_aligning": capacity_cost(by_script),
    }
    with open(out_path, "w") as fh:
        provenance.stamp(result, __file__)
        json.dump(result, fh, indent=2, sort_keys=True, ensure_ascii=False)
    return result


if __name__ == "__main__":
    r = main(os.path.join(HERE, "..", "..", "..", "assignment-6", "frozen", "corpus"),
             os.path.join(HERE, "..", "artifacts", "bothends_codec.json"))
    print("round trip  prefix     ", r["roundtrip_prefix"]["rate"])
    print("round trip  both ends  ", r["roundtrip_both_ends"]["rate"])
    print("bitwise collisions     ", r["bitwise_collisions"]["verdict"],
          r["bitwise_collisions"]["bitwise_identical"], "/", r["bitwise_collisions"]["pairs_checked"])
    print("\nmid-character cuts among cropped word types:")
    print(f"{'script':12s} {'prefix':>9s} {'both ends':>10s} {'aligned':>9s}")
    for s in r["cut_quality"]:
        a = r["cut_quality_aligned"].get(s, {})
        print(f"{s:12s} {r['cut_quality'][s]['prefix_invalid_utf8_rate']:>8.2%} "
              f"{r['cut_quality'][s]['both_ends_invalid_utf8_rate']:>10.2%} "
              f"{a.get('both_ends_invalid_utf8_rate', float('nan')):>9.2%}")
    print("\ncapacity:", r["capacity_cost_of_aligning"])
