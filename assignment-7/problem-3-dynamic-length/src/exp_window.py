"""
Problem 3, experiments E1 to E3: what the fixed window actually costs, and who pays.

E1  occupancy - how much of the window is used, per lane and per script
E2  characters per window - the reframing: the window is counted in bytes, text is written in
    characters, and the exchange rate between them is not 1 for most of the world
E3  truncation rate and truncation collisions - the headline, where the cost stops being
    dimensions and starts being meaning

Everything is counted over assignment-6's frozen corpus with assignment-2's tokenizer. Nothing is
estimated from a table of "typical" byte lengths.

One measurement decision governs all three, and it is the one most likely to be argued with, so it
is made explicit. The Kronecker window applies to a **token**, so tokens are the primary unit here.
But the assignment states the problem in terms of **words** ("we cannot have a word of len more than
32"), and a BPE tokenizer splits long words before the window ever sees them. Reporting only tokens
would understate the harm the assignment is asking about; reporting only words would measure
something the architecture never encounters. Both are computed, side by side, and the gap between
them is itself a result: it is precisely the tokenizer's fertility doing the work, which is the cost
assignment 6 already measured on the other side of the ledger.
"""
import sys, os, json, unicodedata
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "common"))
import corpus, vocabulary  # noqa: E402
import provenance  # noqa: E402

WINDOWS = (16, 32, 64)
MIN_TYPES_TO_REPORT = 200      # a script with fewer distinct types than this is too thin to quote
MAX_EXAMPLES = 12


# --------------------------------------------------------------------------- grapheme clusters

def grapheme_count(text):
    """Approximate extended-grapheme-cluster count.

    Python ships no grapheme segmentation, and this repository does not add a dependency for one.
    The approximation is: a cluster starts at a base character and absorbs any following combining
    mark (categories Mn, Mc, Me), any ZWJ, and any character that follows a virama, which is how
    Indic conjuncts are written.

    This is an approximation and is labelled as one wherever it is reported. It is used only to size
    the gap between codepoints and graphemes in E2, never to make a truncation claim.
    """
    n = 0
    prev_joins = False
    for ch in text:
        combining, joins = _cluster_class(ch)
        if not (combining or prev_joins):
            n += 1
        prev_joins = joins
    return max(n, 1 if text else 0)


_cluster_cache = {}


def _cluster_class(ch):
    """(is_combining, joins_to_next) for one character, memoised.

    `joins_to_next` covers the Indic virama and its per-script aliases, which are the sign that the
    following consonant is part of the same written cluster: VIRAMA in most scripts, SIGN PULLI in
    Tamil, and ZERO WIDTH JOINER anywhere.
    """
    hit = _cluster_cache.get(ch)
    if hit is not None:
        return hit
    cat = unicodedata.category(ch)
    combining = cat in ("Mn", "Mc", "Me") or ch == "‍"
    try:
        name = unicodedata.name(ch)
    except ValueError:
        name = ""
    joins = "VIRAMA" in name or "SIGN PULLI" in name or ch == "‍"
    _cluster_cache[ch] = (combining, joins)
    return combining, joins


# --------------------------------------------------------------------------- corpus tabulation

def tabulate(corpus_root, tokenizer_path, id_to_surface, limit=None):
    """One pass over the corpus, producing every tally the three experiments need.

    Returns token types and word types with their frequencies and their scripts, plus per-script
    character tallies. Types, not occurrences, are what the collision analysis needs; occurrences
    are what the occupancy averages need. Both are kept.
    """
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(tokenizer_path)

    token_types = Counter()          # surface -> occurrences
    word_types = Counter()                    # raw whitespace split, every lane
    word_types_by_lane = defaultdict(Counter)  # lane -> stripped surface -> occurrences
    lane_token_bytes = defaultdict(Counter)   # lane -> byte-length -> occurrences
    script_chars = Counter()         # script -> characters seen
    script_bytes = Counter()         # script -> utf-8 bytes for those characters
    script_graphemes = Counter()
    lane_docs = Counter()

    for lane, text in corpus.lanes(corpus_root, limit=limit):
        lane_docs[lane] += 1
        enc = tok.encode(text)
        for tid in enc.ids:
            surface = id_to_surface.get(tid)
            if surface is None or surface == "":
                continue
            token_types[surface] += 1
            lane_token_bytes[lane][len(surface.encode("utf-8"))] += 1
        for w in text.split():
            if w:
                word_types[w] += 1
                st = strip_punctuation(w)
                if st:
                    word_types_by_lane[lane][st] += 1
        for ch in text:
            s = corpus.char_script(ch)
            script_chars[s] += 1
            script_bytes[s] += len(ch.encode("utf-8"))
        # Graphemes are counted per script over a sample of the text rather than per character,
        # because the notion only exists for a run of characters.
        for s, run in _script_runs(text):
            script_graphemes[s] += grapheme_count(run)

    return {
        "token_types": token_types,
        "word_types": word_types,
        "word_types_by_lane": word_types_by_lane,
        "lane_token_bytes": lane_token_bytes,
        "script_chars": script_chars,
        "script_bytes": script_bytes,
        "script_graphemes": script_graphemes,
        "lane_docs": lane_docs,
    }


# Lanes whose "words" are not words. Splitting source code on whitespace yields identifiers like
# `self.assertEqual(response.status_code,`, which share long prefixes for reasons that have nothing
# to do with language and would flatter the Latin baseline this problem is measured against.
NON_PROSE_LANES = ("code",)


def strip_punctuation(word):
    """Drop leading and trailing punctuation and symbols from a whitespace-split word.

    Without this, `word` and `word,` are two distinct types sharing every byte of their prefix, and
    they register as a truncation collision at every window size. That is an artefact of the split,
    not a property of the encoding, and it inflates every script at once. Stripping is applied to
    the primary numbers; the unstripped numbers are reported alongside them so the size of the
    artefact is visible rather than merely asserted to be small.
    """
    i, j = 0, len(word)
    while i < j and _is_edge_punct(word[i]):
        i += 1
    while j > i and _is_edge_punct(word[j - 1]):
        j -= 1
    return word[i:j]


_punct_cache = {}


def _is_edge_punct(ch):
    hit = _punct_cache.get(ch)
    if hit is None:
        cat = unicodedata.category(ch)
        hit = cat.startswith("P") or cat.startswith("S")
        _punct_cache[ch] = hit
    return hit


def _script_runs(text):
    """Split text into maximal runs of one script, so graphemes can be counted within a script."""
    cur, buf = None, []
    for ch in text:
        s = corpus.char_script(ch)
        if s != cur:
            if buf and cur is not None:
                yield cur, "".join(buf)
            cur, buf = s, [ch]
        else:
            buf.append(ch)
    if buf and cur is not None:
        yield cur, "".join(buf)


def _by_script(types):
    """Group a Counter of surface -> count into script -> {surface: count}."""
    out = defaultdict(Counter)
    for surface, n in types.items():
        s, _, letters = corpus.text_script(surface)
        if letters == 0:
            s = "COMMON"
        out[s][surface] = n
    return out


# --------------------------------------------------------------------------- E1

def e1_occupancy(tally):
    """How much of the window is actually used.

    Reported two ways because they answer different questions. Occurrence-weighted is what the model
    actually spends its dimensions on during training. Type-weighted is what the vocabulary looks
    like. The assignment's complaint ("even 'apple' or 'a'") is about the first.
    """
    per_lane = {}
    for lane, hist in sorted(tally["lane_token_bytes"].items()):
        total = sum(hist.values())
        windows = {}
        for L in WINDOWS:
            used = sum(min(b, L) * n for b, n in hist.items())
            windows[str(L)] = {
                "mean_occupancy": used / (total * L),
                "mean_columns_used": used / total,
                "zero_column_fraction": 1.0 - used / (total * L),
            }
        mean_bytes = sum(b * n for b, n in hist.items()) / total
        per_lane[lane] = {
            "tokens": total,
            "mean_bytes_per_token": mean_bytes,
            "median_bytes_per_token": _median_from_hist(hist),
            "max_bytes_per_token": max(hist),
            "windows": windows,
        }

    per_script = {}
    for script, types in sorted(_by_script(tally["token_types"]).items()):
        total = sum(types.values())
        if len(types) < MIN_TYPES_TO_REPORT:
            continue
        hist = Counter()
        for surface, n in types.items():
            hist[len(surface.encode("utf-8"))] += n
        windows = {}
        for L in WINDOWS:
            used = sum(min(b, L) * n for b, n in hist.items())
            windows[str(L)] = {"mean_occupancy": used / (total * L)}
        per_script[script] = {
            "token_types": len(types),
            "token_occurrences": total,
            "mean_bytes_per_token": sum(b * n for b, n in hist.items()) / total,
            "windows": windows,
        }

    return {
        "unit": "token, as produced by the assignment-2 tokenizer",
        "per_lane": per_lane,
        "per_script": per_script,
        "reading": ("Occupancy is the fraction of the L position columns a token actually fills. "
                    "Everything below it is zeros. This is the waste the assignment describes, and "
                    "it is real, but zeros cost dimensions and nothing else."),
    }


def _median_from_hist(hist):
    total = sum(hist.values())
    seen = 0
    for b in sorted(hist):
        seen += hist[b]
        if seen * 2 >= total:
            return b
    return None


# --------------------------------------------------------------------------- E2

def e2_characters_per_window(tally):
    """The reframing: L is counted in bytes, but text is written in characters."""
    rows = {}
    for script in sorted(tally["script_chars"]):
        chars = tally["script_chars"][script]
        if chars < 10000:
            continue
        bytes_ = tally["script_bytes"][script]
        graphemes = tally["script_graphemes"].get(script, 0)
        bpc = bytes_ / chars
        rows[script] = {
            "characters_observed": chars,
            "bytes_per_character": bpc,
            "codepoints_per_grapheme_approx": (chars / graphemes) if graphemes else None,
            "bytes_per_grapheme_approx": (bytes_ / graphemes) if graphemes else None,
            "characters_in_window": {str(L): L / bpc for L in WINDOWS},
            "graphemes_in_window_approx": {
                str(L): (L / (bytes_ / graphemes)) if graphemes else None for L in WINDOWS},
        }
    latin = rows.get("LATIN", {}).get("bytes_per_character")
    if latin:
        for script, r in rows.items():
            r["capacity_relative_to_latin"] = latin / r["bytes_per_character"]
    return {
        "rows": rows,
        "reading": ("A window of L bytes is a window of L characters only for scripts that cost one "
                    "byte per character. For a 3-byte script the same architectural constant buys a "
                    "third of the context. Graphemes are approximated, and the grapheme figure is "
                    "the one that matters for a reader, since a Devanagari conjunct is written as "
                    "several codepoints but read as one unit."),
    }


# --------------------------------------------------------------------------- E3

def _collisions(types, L, unit_name):
    """Group distinct types by their first L bytes and find the ones that collapse together.

    A truncation collision is not a metaphor here: two types sharing a first-L-byte prefix produce
    byte-for-byte identical codec vectors, hence an identical embedding at every layer, so no model
    of any depth can tell them apart. That is checked, not assumed, by the caller.
    """
    groups = defaultdict(list)
    cropped = 0
    chars_lost = 0
    invalid_prefix = 0
    for surface, n in types.items():
        b = surface.encode("utf-8")
        key = b[:L]
        groups[key].append((surface, n))
        if len(b) > L:
            cropped += 1
            kept = key.decode("utf-8", errors="ignore")
            chars_lost += len(surface) - len(kept)
            try:
                key.decode("utf-8")
            except UnicodeDecodeError:
                invalid_prefix += 1
    colliding = {k: v for k, v in groups.items() if len(v) > 1}
    types_in_collisions = sum(len(v) for v in colliding.values())
    occurrences_in_collisions = sum(n for v in colliding.values() for _, n in v)
    total_occ = sum(types.values())

    examples = sorted(colliding.values(), key=lambda v: (-sum(n for _, n in v), -len(v)))
    ex_out = []
    for grp in examples[:MAX_EXAMPLES]:
        members = sorted(grp, key=lambda sn: -sn[1])
        ex_out.append({
            "shared_prefix_bytes": L,
            "members": [m for m, _ in members[:6]],
            "member_count": len(members),
            "occurrences": sum(n for _, n in members),
        })

    return {
        "unit": unit_name,
        "window": L,
        "distinct_types": len(types),
        "types_cropped": cropped,
        "types_cropped_rate": cropped / len(types) if types else 0.0,
        "mean_characters_lost_per_cropped_type": chars_lost / cropped if cropped else 0.0,
        "cropped_types_with_invalid_utf8_prefix": invalid_prefix,
        "cropped_types_with_invalid_utf8_prefix_rate": (invalid_prefix / cropped) if cropped else 0.0,
        "colliding_groups": len(colliding),
        "types_in_collisions": types_in_collisions,
        "types_in_collisions_rate": types_in_collisions / len(types) if types else 0.0,
        "occurrence_rate_in_collisions": occurrences_in_collisions / total_occ if total_occ else 0.0,
        "examples": ex_out,
    }


def e3_truncation(tally):
    """Truncation rate and truncation collisions, per script.

    Three unit definitions are reported, because the choice of unit is doing real work and hiding it
    would be the easiest way to overstate this result:

      * **token** - what the architecture actually windows. With a BPE tokenizer in front, long
        words are already split before the window sees them, so this is the most conservative view.
      * **word_prose** - whitespace-split, punctuation-stripped, excluding the source-code lane.
        This is the primary word-level number and the honest one.
      * **word_raw** - whitespace-split with nothing removed, every lane. Reported only so the
        difference against `word_prose` shows how much of a naive measurement is artefact.
    """
    prose = Counter()
    for lane, types in tally["word_types_by_lane"].items():
        if lane in NON_PROSE_LANES:
            continue
        prose.update(types)

    out, excluded = {}, {}
    for unit_name, types_counter in (("token", tally["token_types"]),
                                     ("word_prose", prose),
                                     ("word_raw", tally["word_types"])):
        by_script = _by_script(types_counter)
        dest, drop = {}, {}
        for script, types in sorted(by_script.items()):
            if len(types) < MIN_TYPES_TO_REPORT:
                drop[script] = len(types)
                continue
            dest[script] = {str(L): _collisions(types, L, unit_name) for L in WINDOWS}
        out[unit_name] = dest
        excluded[unit_name] = drop

    out["_excluded_scripts"] = {
        "reason": "fewer than %d distinct types, too thin to quote a rate from" % MIN_TYPES_TO_REPORT,
        "counts": excluded,
    }
    out["_non_prose_lanes_excluded_from_word_prose"] = list(NON_PROSE_LANES)
    return out


def e3_by_lane_and_script(tally):
    """Collision rate broken down by lane as well as script.

    Pooling every lane into one Latin figure turned out to be misleading in this corpus, and the
    correction matters enough to be recomputed here rather than described. Pooled Latin collides at
    a low but non-zero rate, and essentially all of it comes from source code, LaTeX and identifiers
    rather than from language: `generate_random_number` against `generate_random_password`,
    `\\begin{align*}` against itself. Splitting by lane separates the two populations and gives a
    prose baseline that can be compared against Indic prose without an asterisk.

    Examples are omitted here; the pooled view in `e3_truncation` carries them.
    """
    out = {}
    for lane in sorted(tally["word_types_by_lane"]):
        by_script = _by_script(tally["word_types_by_lane"][lane])
        rows = {}
        for script, types in sorted(by_script.items()):
            if len(types) < MIN_TYPES_TO_REPORT:
                continue
            rows[script] = {}
            for L in WINDOWS:
                r = _collisions(types, L, "word")
                r.pop("examples", None)
                rows[script][str(L)] = r
        if rows:
            out[lane] = rows
    return {
        "rows": out,
        "reading": ("The prose lanes (web, long_ctx, reasoning) are the fair English baseline. The "
                    "code and math lanes are Latin script but not language, and they carry almost "
                    "all of the pooled Latin collisions."),
    }


def verify_collisions_are_real(tally, L=32, unit="byte", max_checks=200):
    """Do colliding types actually produce identical codec vectors? Check, do not assert.

    The whole argument rests on this being literally true, so it is measured against the codec
    itself rather than argued from the definition of a prefix.
    """
    import numpy as np
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "common"))
    import codec

    prose = Counter()
    for lane, types in tally["word_types_by_lane"].items():
        if lane not in NON_PROSE_LANES:
            prose.update(types)
    groups = defaultdict(list)
    for surface in prose:
        b = surface.encode("utf-8")
        if len(b) > L:
            groups[b[:L]].append(surface)
    pairs = [(v[0], v[1]) for v in groups.values() if len(v) > 1][:max_checks]
    identical = 0
    max_diff = 0.0
    for a, b in pairs:
        va, _, _ = codec.encode(codec.text_units(a, unit), L, unit)
        vb, _, _ = codec.encode(codec.text_units(b, unit), L, unit)
        d = float(np.abs(va - vb).max())
        max_diff = max(max_diff, d)
        identical += d == 0.0
    return {
        "pairs_checked": len(pairs),
        "pairs_with_bitwise_identical_codec_vectors": identical,
        "max_absolute_difference": max_diff,
        "claim": ("distinct words sharing a first-L-byte prefix receive the identical embedding"),
        "verdict": "confirmed" if pairs and identical == len(pairs) else
                   ("no colliding pairs found" if not pairs else "FAILED"),
    }


# --------------------------------------------------------------------------- main

def main(corpus_root, tokenizer_path, out_path, limit=None):
    toks, meta = vocabulary.load(tokenizer_path)
    id_to_surface = {tid: text for tid, text, _ in toks}

    tally = tabulate(corpus_root, tokenizer_path, id_to_surface, limit=limit)
    result = {
        "tokenizer": meta,
        "corpus": {
            "root": os.path.relpath(corpus_root),
            "documents_per_lane": dict(sorted(tally["lane_docs"].items())),
            "distinct_token_types": len(tally["token_types"]),
            "distinct_word_types": len(tally["word_types"]),
            "characters": sum(tally["script_chars"].values()),
        },
        "windows": list(WINDOWS),
        "e1_occupancy": e1_occupancy(tally),
        "e2_characters_per_window": e2_characters_per_window(tally),
        "e3_truncation": e3_truncation(tally),
        "e3_by_lane_and_script": e3_by_lane_and_script(tally),
        "e3_collision_check": verify_collisions_are_real(tally),
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
             os.path.join(here, "..", "artifacts", "window.json"), limit=lim)
    print(json.dumps(r["e2_characters_per_window"]["rows"], indent=2, ensure_ascii=False))
