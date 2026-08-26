"""
Problem 3 invariants. Run with `python tests/test_invariants.py`; exits non-zero on failure.

Plain asserts and no test framework, so the dependency list stays at numpy and tokenizers. These
cover the properties the writeup's claims rest on, and several of them exist because the naive
version of the same thing was wrong first.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "common"))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
import codec, corpus  # noqa: E402
import exp_window as W  # noqa: E402

FAILURES = []


def check(name, fn):
    try:
        fn()
        print("  ok   %s" % name)
    except AssertionError as exc:
        FAILURES.append((name, str(exc)))
        print("  FAIL %s: %s" % (name, exc))


# ---------------------------------------------------------------- codec determinism

def test_encoding_is_deterministic():
    for text in ("apple", "नमस्ते", "அருமை"):
        a, _, _ = codec.encode(codec.text_units(text, "byte"), 32, "byte")
        b, _, _ = codec.encode(codec.text_units(text, "byte"), 32, "byte")
        assert a.tobytes() == b.tobytes(), "%r encoded differently twice" % text


def test_znorm_cannot_change_the_decode():
    """z-normalisation is a strictly increasing affine map, so it preserves every argmax."""
    for text in ("apple", "ಕನ್ನಡ", "hello world"):
        u = codec.text_units(text, "byte")[:32]
        raw, _, _ = codec.encode(u, 32, "byte", znorm=False)
        nrm, _, _ = codec.encode(u, 32, "byte", znorm=True)
        assert codec.decode(raw, 32, "byte", length=len(u))[0] == \
               codec.decode(nrm, 32, "byte", length=len(u))[0], text


def test_roundtrip_for_every_unit_scheme():
    text = "നിങ്ങളുടെ"
    for units, L, unit, blocks in (
            (codec.text_units(text, "byte")[:32], 32, "byte", 2),
            (codec.text_units(text, "codepoint")[:16], 16, "codepoint", 2),
            (codec.both_ends_units(text, 32, "byte"), 32, "byte", 2),
            (codec.script_relative_units(text, 7, limit=31), 32, "byte", 2)):
        v, _, _ = codec.encode(units, L, unit, blocks)
        back, _ = codec.decode(v, L, unit, blocks, length=len(units))
        assert back == list(units), "round trip failed for %s" % unit


def test_both_ends_keeps_the_front_and_the_back():
    long = "a" * 50 + "ZZZ"
    u = codec.both_ends_units(long, 32, "byte")
    assert len(u) == 32
    assert bytes(u[:16]) == b"a" * 16, "front half not retained"
    assert bytes(u[-3:]) == b"ZZZ", "back half not retained"


def test_alignment_never_splits_a_character():
    """The unaligned cut may land mid-character; the aligned one must not."""
    word = "ക" * 20                      # 20 Malayalam characters, 60 bytes
    aligned = codec.both_ends_units(word, 32, "byte", align=True)
    bytes(aligned).decode("utf-8")            # raises if a cut split a character
    assert len(aligned) <= 32


# ---------------------------------------------------------------- collision detection

def test_collision_grouping_finds_exactly_the_right_pairs():
    """Constructed positives and negatives, so the detector is tested rather than trusted."""
    colliding = {"a" * 40: 1, "a" * 32 + "different": 1}      # identical first 32 bytes
    distinct = {"a" * 31 + "X": 1, "a" * 31 + "Y": 1}         # differ inside the window
    r = W._collisions(colliding, 32, "word")
    assert r["colliding_groups"] == 1, "missed a real collision"
    r2 = W._collisions(distinct, 32, "word")
    assert r2["colliding_groups"] == 0, "reported a collision that does not exist"


def test_colliding_words_really_share_an_embedding():
    """The claim the writeup rests on, checked against the codec rather than the key."""
    a, b = "a" * 40, "a" * 32 + "zzz"
    va, _, _ = codec.encode(codec.text_units(a, "byte"), 32, "byte")
    vb, _, _ = codec.encode(codec.text_units(b, "byte"), 32, "byte")
    assert float(np.abs(va - vb).max()) == 0.0, "prefix-identical words differ in embedding"


def test_punctuation_stripping_only_touches_the_edges():
    assert W.strip_punctuation("word,") == "word"
    assert W.strip_punctuation('"quoted"') == "quoted"
    assert W.strip_punctuation("...") == ""
    assert W.strip_punctuation("self.assertEqual") == "self.assertEqual", "stripped an interior dot"


# ---------------------------------------------------------------- per-script tallies

def test_script_detection():
    cases = {"नमस्ते": "DEVANAGARI", "apple": "LATIN",
             "வணக்கம்": "TAMIL",
             "مرحبا": "ARABIC",
             "2024": "COMMON", "నమస్కారం": "TELUGU"}
    for text, want in cases.items():
        got = corpus.text_script(text)[0]
        assert got == want, "%r classified as %s, expected %s" % (text, got, want)


def test_digits_are_common_not_latin():
    """Otherwise every script's statistics are diluted by shared characters."""
    assert corpus.char_script("5") == "COMMON"
    assert corpus.text_script("हि123")[0] == "DEVANAGARI", "digits swung the majority"


def test_indic_scripts_are_three_bytes_per_character():
    for ch in ("क", "க", "క", "ക", "অ", "ક", "ਕ", "କ",
               "ಕ"):
        assert len(ch.encode("utf-8")) == 3, "%r is not 3 bytes" % ch


def test_record_text_handles_both_corpus_shapes():
    assert corpus.record_text({"text": "x"}) == "x"
    assert corpus.record_text({"spans": [{"text": "a"}, {"text": "b"}]}) == "a\nb"


if __name__ == "__main__":
    print("Problem 3 invariants")
    for nm, fn in sorted(globals().items()):
        if nm.startswith("test_") and callable(fn):
            check(nm[5:].replace("_", " "), fn)
    print("\n%d failure(s)" % len(FAILURES) if FAILURES else "\nall invariants hold")
    sys.exit(1 if FAILURES else 0)
