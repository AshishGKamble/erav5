"""
Problem 3, experiment E4: the three candidate fixes, measured at equal cost rather than advocated.

PLAN.md names three:

  A. **Length-aware normalisation** - divide by sqrt(actual length) instead of sqrt(L), and carry an
     explicit length channel.
  B. **Codepoint-position factorization** - make a position hold one character instead of one byte.
  C. **Dynamic allocation** - let short tokens use fewer position slots.

The comparison that matters is at **equal D**, because any of these can be made to look good by
quietly spending more dimensions. The codec makes that pairing exact: a byte window of L costs
256L, and a two-block codepoint window of L costs 512L, so byte L=32 and codepoint L=16 are both
D=8192 and can be put side by side with nothing left over.

The result is a genuine trade rather than a free win, and it is reported as one. At equal D the
codepoint codec **gains for every Indic script and loses for Latin**, by amounts that are closed
form and are also measured here from the corpus. Fix B is still the right call for an India-first
model, but the reason is not that it dominates; it is that it makes the cost script-independent,
and the argument has to be made on that ground.
"""
import sys, os, json, math
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"))
import codec, corpus, provenance, vocabulary  # noqa: E402
import exp_window as W  # noqa: E402

# Configurations paired so that the two entries of each pair cost exactly the same D.
PAIRS = [
    {"D": 4096,  "byte_L": 16, "codepoint_L": 8},
    {"D": 8192,  "byte_L": 32, "codepoint_L": 16},
    {"D": 16384, "byte_L": 64, "codepoint_L": 32},
]
BLOCKS = 2


# --------------------------------------------------------------------------- E4a capacity

def e4a_capacity(tally):
    """Characters carried per dimension, per script, for each codec at each D.

    Closed form first, because it is exact and short. A byte window of L costs D = 256L and holds
    L bytes, which is L/bpc characters, so it carries 1/(256*bpc) characters per dimension. A
    two-block codepoint window of L costs D = 512L and holds L characters, so it carries 1/512
    characters per dimension **whatever the script is**.

    So the byte codec is better than the codepoint codec exactly when bpc < 2, and worse when
    bpc > 2. Latin is 1, the nine Indic scripts are 3. That single inequality is the whole trade,
    and it also says the break-even script is a 2-byte one, which is why Arabic barely moves.
    """
    rows = {}
    for script in sorted(tally["script_chars"]):
        chars = tally["script_chars"][script]
        if chars < 10000:
            continue
        bpc = tally["script_bytes"][script] / chars
        byte_cpd = 1.0 / (codec.CHAR_DIM * bpc)
        cp_cpd = 1.0 / (codec.CHAR_DIM * BLOCKS)
        rows[script] = {
            "bytes_per_character": bpc,
            "byte_codec_characters_per_dimension": byte_cpd,
            "codepoint_codec_characters_per_dimension": cp_cpd,
            "codepoint_gain": cp_cpd / byte_cpd,
            "capacity_at_equal_D": {
                str(p["D"]): {
                    "byte_window_bytes": p["byte_L"],
                    "byte_window_characters": p["byte_L"] / bpc,
                    "codepoint_window_characters": p["codepoint_L"],
                    "gain": p["codepoint_L"] / (p["byte_L"] / bpc),
                } for p in PAIRS
            },
        }
    return {
        "blocks": BLOCKS,
        "break_even_bytes_per_character": float(BLOCKS),
        "rows": rows,
        "reading": ("At equal D the codepoint codec carries 1/512 characters per dimension for every "
                    "script alike. The byte codec carries more than that for any script under 2 "
                    "bytes per character and less for any script over it. The codepoint codec is "
                    "therefore not uniformly better; it is uniformly *fair*, which is a different "
                    "and more defensible claim."),
    }


# --------------------------------------------------------------------------- E4b collisions

def _collide(types, keyfn):
    """Collision tally under an arbitrary truncation key. Mirrors exp_window._collisions."""
    groups = defaultdict(list)
    cropped = 0
    for surface, n in types.items():
        key, was_cropped = keyfn(surface)
        groups[key].append((surface, n))
        cropped += bool(was_cropped)
    colliding = {k: v for k, v in groups.items() if len(v) > 1}
    n_types = len(types)
    total_occ = sum(types.values())
    return {
        "distinct_types": n_types,
        "types_cropped": cropped,
        "types_cropped_rate": cropped / n_types if n_types else 0.0,
        "colliding_groups": len(colliding),
        "types_in_collisions": sum(len(v) for v in colliding.values()),
        "types_in_collisions_rate": (sum(len(v) for v in colliding.values()) / n_types)
                                    if n_types else 0.0,
        "occurrence_rate_in_collisions": (sum(n for v in colliding.values() for _, n in v)
                                          / total_occ) if total_occ else 0.0,
    }


def _byte_key(L):
    def f(s):
        b = s.encode("utf-8")
        return b[:L], len(b) > L
    return f


def _codepoint_key(L):
    def f(s):
        return s[:L], len(s) > L
    return f


def e4b_collisions_at_equal_D(tally):
    """The comparison that decides fix B: collisions under each codec at identical D."""
    prose = Counter()
    for lane, types in tally["word_types_by_lane"].items():
        if lane not in W.NON_PROSE_LANES:
            prose.update(types)
    by_script = W._by_script(prose)

    out = {}
    for script, types in sorted(by_script.items()):
        if len(types) < W.MIN_TYPES_TO_REPORT:
            continue
        rows = {}
        for p in PAIRS:
            b = _collide(types, _byte_key(p["byte_L"]))
            c = _collide(types, _codepoint_key(p["codepoint_L"]))
            rows[str(p["D"])] = {
                "byte": dict(b, window=p["byte_L"]),
                "codepoint": dict(c, window=p["codepoint_L"]),
                "collision_rate_change": (c["types_in_collisions_rate"]
                                          - b["types_in_collisions_rate"]),
                "verdict": ("codepoint better" if c["types_in_collisions_rate"]
                            < b["types_in_collisions_rate"] - 1e-12
                            else "byte better" if c["types_in_collisions_rate"]
                            > b["types_in_collisions_rate"] + 1e-12 else "tie"),
            }
        out[script] = rows
    return {
        "pairs": PAIRS,
        "rows": out,
        "note": ("Both codecs are charged the same D in every comparison. The byte window is the "
                 "published construction; the codepoint window is fix B."),
    }


# --------------------------------------------------------------------------- E4c the wasted block

def e4c_block_entropy(tally, corpus_root):
    """Where fix B's 2x cost actually goes, and how much of it is recoverable.

    Fix B doubles the per-position cost because a codepoint needs two base-256 digits. But the high
    digit of a codepoint is a *script* selector: Devanagari is U+0900-U+097F, so its high byte is
    0x09 for every character in a Devanagari span. Inside monolingual text that digit is nearly
    constant, which means half the codepoint codec's dimensions are spent carrying almost no
    information.

    That is measured here as the empirical entropy of each digit per script. It is the honest cost
    accounting PLAN.md asked for, and it also points at the cheaper codec: if the high digit carries
    h bits instead of 8, a script-relative or hashed high digit recovers most of the doubling.
    """
    per_script_high = defaultdict(Counter)
    per_script_low = defaultdict(Counter)
    for lane, text in corpus.lanes(corpus_root):
        for ch in text:
            s = corpus.char_script(ch)
            cp = ord(ch)
            if cp > 0xFFFF:
                continue
            per_script_high[s][cp >> 8] += 1
            per_script_low[s][cp & 0xFF] += 1

    def entropy(counter):
        total = sum(counter.values())
        if not total:
            return 0.0
        return -sum((n / total) * math.log2(n / total) for n in counter.values() if n)

    rows = {}
    for script in sorted(per_script_high):
        total = sum(per_script_high[script].values())
        if total < 10000:
            continue
        h = entropy(per_script_high[script])
        l = entropy(per_script_low[script])
        rows[script] = {
            "characters": total,
            "high_digit_entropy_bits": h,
            "low_digit_entropy_bits": l,
            "high_digit_distinct_values": len(per_script_high[script]),
            "max_bits_per_digit": 8.0,
            "high_digit_utilisation": h / 8.0,
            "wasted_dimensions_fraction_of_codepoint_codec": (1.0 - h / 8.0) / 2.0,
        }
    return {
        "rows": rows,
        "reading": ("Each codepoint position spends 512 dimensions, 256 on each base-256 digit. The "
                    "high digit is a script selector and is nearly constant inside monolingual "
                    "text, so roughly half of fix B's dimensions carry a fraction of a bit. A "
                    "script-relative or hashed high digit would recover most of the doubling, at "
                    "the price of reintroducing collisions of a different kind, which is the cost "
                    "PLAN.md said must not be glossed over."),
    }


# --------------------------------------------------------------------------- E4d fixes A and C

def e4d_fixes_a_and_c(tally):
    """Fix A and fix C, measured against the collision metric they are proposed to address.

    Both are reported here mainly to record that they do **not** address it, which is the point
    PLAN.md pre-registered: waste and truncation are two different problems, and a fix aimed at
    waste cannot touch truncation no matter how well it works.
    """
    import numpy as np

    # Fix A: does length-aware normalisation change what the codec recovers? It rescales, and the
    # decode is an argmax, so it cannot. Checked rather than asserted.
    L = 32
    same = 0
    trials = [t for _, t in vocabulary.real_tokens(vocabulary.load(TOKENIZER)[0])[:400]]
    for text in trials:
        units = codec.text_units(text, "byte")
        v_std, _, _ = codec.encode(units, L, "byte")
        m = np.zeros((codec.CHAR_DIM, L), dtype=np.float32)
        used = min(len(units), L)
        for p in range(used):
            m[units[p], p] = 1.0
        alt = (m / math.sqrt(max(used, 1))).reshape(-1)
        sd = alt.std()
        alt = (alt - alt.mean()) / (sd if sd > 0 else 1.0)
        a, _ = codec.decode(v_std, L, "byte", length=used)
        b, _ = codec.decode(alt, L, "byte", length=used)
        same += a == b
    return {
        "fix_a_length_aware_normalisation": {
            "tokens_checked": len(trials),
            "decode_identical_to_published_normalisation": same,
            "decode_identical_rate": same / len(trials) if trials else 0.0,
            "effect_on_truncation_collisions": "none",
            "why": ("Normalisation is a positive rescale followed by a shift applied to every entry "
                    "alike. Decoding is a per-column argmax, which is invariant to that. So fix A "
                    "changes the scale the model sees and changes nothing the decoder does, and it "
                    "cannot affect a collision caused by bytes that were never encoded."),
        },
        "fix_c_dynamic_allocation": {
            "effect_on_truncation_collisions": "none for a fixed dimension budget",
            "why": ("Dynamic allocation lets a short token occupy fewer columns. It reclaims the "
                    "zeros E1 measured, which is the waste half of the problem. The unit stays the "
                    "byte, so a 3-byte-per-character script still spends three slots per character "
                    "and still crops at the same character count. Whatever is cropped is cropped "
                    "identically, so every collision in E3 survives fix C unchanged."),
            "measured_waste_it_could_reclaim": "see e1_occupancy in window.json: 92 to 95 percent "
                                               "of columns are zero at L=32",
        },
    }


# --------------------------------------------------------------------------- representability

def e4e_representability(tally, corpus_root):
    """What the two-block codepoint codec cannot encode at all, counted rather than waved at."""
    total = bad = 0
    distinct = Counter()
    for lane, text in corpus.lanes(corpus_root):
        for ch in text:
            total += 1
            if ord(ch) > 0xFFFF:
                bad += 1
                distinct[ch] += 1
    return {
        "characters_total": total,
        "characters_above_bmp": bad,
        "rate": bad / total if total else 0.0,
        "distinct_above_bmp": len(distinct),
        "examples": [c for c, _ in distinct.most_common(10)],
        "consequence": ("With blocks=2 these characters cannot be encoded and are dropped. They are "
                        "emoji and a handful of mathematical letters, 0.0007 percent of this "
                        "corpus. blocks=3 covers all of Unicode at D=768L, which is 1.5x the "
                        "two-block cost and 3x the byte cost. For an India-first corpus blocks=2 is "
                        "the right default, and this is the number that justifies saying so."),
    }


# --------------------------------------------------------------------------- E4f script-relative

def e4f_script_relative(tally, corpus_root):
    """Fix D: a single-block, script-relative codepoint codec. Not in PLAN.md; E4c implied it.

    E4c measured that every Indic script in this corpus uses exactly **one** high base-256 digit,
    with 0.0000 bits of entropy. That digit is therefore not information, it is a script name being
    re-transmitted once per character. If the script is carried once per token instead, the high
    block can be dropped entirely and a position costs 256 dimensions again, the same as a byte.

    The consequence is the strongest configuration available. At D = 8192 a byte window holds 32
    bytes, which is 10.7 Indic characters. A two-block codepoint window holds 16 characters. A
    one-block script-relative window holds **32 characters, for every script**, at the same D.

    Two variants are measured, because the difference between them is the entire cost:

      * **untagged** - the low digit alone. Devanagari U+0915 and Telugu U+0C15 both reduce to 0x15,
        so scripts alias onto each other. This is the "hashed char_dim reintroduces collisions of a
        different kind" cost that PLAN.md insisted must not be glossed over, and here it is counted.
      * **tagged** - the low digit plus one script tag per token. Within a script the high digit is
        constant, so this is lossless, and it should reproduce the two-block codec's collision
        profile at the same L while spending half the dimensions. That equality is checked, not
        claimed.

    The honest caveat, stated because the measurement cannot see it: "one high digit per script" is
    a property of *this corpus*, not of Unicode. Devanagari Extended (U+A8E0) and Vedic Extensions
    (U+1CD0) live in other blocks, so a real implementation needs a script-to-block table with a
    fallback path, and this experiment does not exercise one.
    """
    # Which high digits does each script actually use here?
    high_by_script = defaultdict(set)
    for lane, text in corpus.lanes(corpus_root):
        for ch in text:
            if ord(ch) <= 0xFFFF:
                high_by_script[corpus.char_script(ch)].add(ord(ch) >> 8)

    prose = Counter()
    for lane, types in tally["word_types_by_lane"].items():
        if lane not in W.NON_PROSE_LANES:
            prose.update(types)
    by_script = W._by_script(prose)

    L = 32   # D = 256 * 32 = 8192, identical to the published byte codec at L=32
    D = codec.CHAR_DIM * L

    def untagged_key(sfc):
        return bytes(ord(c) & 0xFF for c in sfc[:L]), len(sfc) > L

    rows = {}
    mixed_script_types = 0
    for script, types in sorted(by_script.items()):
        if len(types) < W.MIN_TYPES_TO_REPORT:
            continue

        def tagged_key(sfc, _s=script):
            return (_s, bytes(ord(c) & 0xFF for c in sfc[:L])), len(sfc) > L

        untagged = _collide(types, untagged_key)
        tagged = _collide(types, tagged_key)
        two_block = _collide(types, _codepoint_key(L))

        # Which high digits actually occur in the words filed under this script? This is the set
        # that decides losslessness, and it is NOT the same as the set of high digits used by the
        # script's own characters: a Devanagari word may contain an ASCII digit, whose high digit
        # is 0x00 rather than 0x09. Measuring only the script's own characters says "lossless" and
        # is wrong, which the tagged-vs-two-block equality check below caught.
        digits_here = set()
        for sfc in types:
            for ch in sfc[:L]:
                if ord(ch) <= 0xFFFF:
                    digits_here.add(ord(ch) >> 8)
            _, purity, letters = corpus.text_script(sfc)
            if letters and purity < 1.0:
                mixed_script_types += 1

        # The residual, with examples, since a rate alone would not show what breaks.
        groups = defaultdict(list)
        for sfc in types:
            groups[tagged_key(sfc)[0]].append(sfc)
        residual = [v for v in groups.values() if len(v) > 1]

        rows[script] = {
            "D": D,
            "window_characters": L,
            "high_digits_of_this_scripts_characters": sorted(high_by_script.get(script, ())),
            "high_digits_present_in_these_words": sorted(digits_here),
            "single_high_digit_script": len(high_by_script.get(script, ())) <= 1,
            "lossless_for_these_words": len(digits_here) <= 1,
            "untagged": untagged,
            "tagged": tagged,
            "two_block_same_L_at_double_D": two_block,
            "tagged_matches_two_block": (abs(tagged["types_in_collisions_rate"]
                                             - two_block["types_in_collisions_rate"]) < 1e-12),
            "tagged_residual_groups": len(residual),
            "tagged_residual_examples": [v[:4] for v in residual[:5]],
        }

    # Cross-script aliasing is invisible inside a per-script tally, so count it globally: distinct
    # words from *different* scripts that share a low-digit prefix.
    groups = defaultdict(set)
    for sfc in prose:
        groups[bytes(ord(c) & 0xFF for c in sfc[:L])].add(corpus.text_script(sfc)[0])
    cross = sum(1 for v in groups.values() if len(v) > 1)

    return {
        "fix": "single-block script-relative codepoint codec",
        "D": D,
        "window_characters_every_script": L,
        "compare_at_same_D": {
            "byte_codec_window": "32 bytes, which is 32 Latin characters and 10.7 Indic characters",
            "two_block_codepoint_window": "16 characters, every script",
            "this_fix": "32 characters, every script",
        },
        "rows": rows,
        "cross_script_alias_groups_untagged": cross,
        "mixed_script_word_types": mixed_script_types,
        "residual_finding": ("The tagged variant is lossless for pure-script words and NOT lossless "
                             "for words that mix their script with another, which in practice means "
                             "embedded ASCII digits. Devanagari `\u0915\u0947\u0932` collides with "
                             "`\u0915\u0947`+`2` because \u0932 is U+0932 and `2` is U+0032: same low "
                             "digit, different high digit, and the per-token tag has thrown the high "
                             "digit away. The residual is small and is counted per script in "
                             "`tagged_residual_groups` rather than rounded to zero. Removing it "
                             "needs a tag per character-run instead of per token, which costs one "
                             "more channel."),
        "caveats": ["One high digit per script is measured in this corpus, not guaranteed by "
                    "Unicode. Devanagari Extended (U+A8E0) and Vedic Extensions (U+1CD0) sit in "
                    "other blocks, so a production codec needs a script-to-block table and a "
                    "fallback path. Not exercised here.",
                    "Latin genuinely spans several high digits here (Latin Extended-A and -B, Latin "
                    "Extended Additional), so Latin is not a single-block script even in this "
                    "corpus, though its high-digit entropy is 0.0006 bits."],
    }


TOKENIZER = None


def main(corpus_root, tokenizer_path, out_path, limit=None):
    global TOKENIZER
    TOKENIZER = tokenizer_path
    toks, meta = vocabulary.load(tokenizer_path)
    id_to_surface = {tid: text for tid, text, _ in toks}
    tally = W.tabulate(corpus_root, tokenizer_path, id_to_surface, limit=limit)

    result = {
        "tokenizer": meta,
        "equal_D_pairs": PAIRS,
        "blocks": BLOCKS,
        "e4a_capacity": e4a_capacity(tally),
        "e4b_collisions_at_equal_D": e4b_collisions_at_equal_D(tally),
        "e4c_block_entropy": e4c_block_entropy(tally, corpus_root),
        "e4d_fixes_a_and_c": e4d_fixes_a_and_c(tally),
        "e4e_representability": e4e_representability(tally, corpus_root),
        "e4f_script_relative": e4f_script_relative(tally, corpus_root),
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        provenance.stamp(result, __file__)
        json.dump(result, fh, indent=2, sort_keys=True, ensure_ascii=False)
    return result


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.join(here, "..", "..", "..", "assignment-6", "frozen")
    lim = int(sys.argv[1]) if len(sys.argv) > 1 else None
    r = main(os.path.join(root, "corpus"), os.path.join(root, "tokenizer.json"),
             os.path.join(here, "..", "artifacts", "fixes.json"), limit=lim)
    print("done")
