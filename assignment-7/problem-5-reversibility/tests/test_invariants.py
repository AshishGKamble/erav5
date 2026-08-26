"""
Problem 5 invariants. Run with `python tests/test_invariants.py`; exits non-zero on failure.

Plain asserts and no test framework, so the dependency list stays at numpy and tokenizers. Several
of these exist because the naive version was wrong first, and those say so.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "common"))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
import codec, kron_model as K  # noqa: E402

FAILURES = []
WORDS = ["apple", "a", "hello world", "നിങ്ങളുടെ", "நீங்கள்", "प्रौद्योगिकी"]


def check(name, fn):
    try:
        fn()
        print("  ok   %s" % name)
    except AssertionError as exc:
        FAILURES.append((name, str(exc)))
        print("  FAIL %s: %s" % (name, exc))


# ---------------------------------------------------------------- codec round trip

def test_roundtrip_is_exact_for_everything_that_fits():
    for text in WORDS:
        u = codec.text_units(text, "byte")
        if len(u) > 32:
            continue
        v, cropped, _ = codec.encode(u, 32, "byte")
        assert cropped == 0
        back, _ = codec.decode(v, 32, "byte", length=len(u))
        assert back == list(u), "round trip failed for %r" % text


def test_overflow_recovers_the_prefix_and_admits_the_loss():
    long = "x" * 100
    u = codec.text_units(long, "byte")
    v, cropped, _ = codec.encode(u, 32, "byte")
    assert cropped == len(u) - 32, "cropped count wrong"
    back, _ = codec.decode(v, 32, "byte", length=32)
    assert back == list(u[:32]), "retained prefix not recovered"


# ---------------------------------------------------------------- decoder equivalence

def test_decode_ignores_znorm():
    for text in WORDS:
        u = codec.text_units(text, "byte")[:32]
        raw, _, _ = codec.encode(u, 32, "byte", znorm=False)
        nrm, _, _ = codec.encode(u, 32, "byte", znorm=True)
        assert codec.decode(raw, 32, "byte", length=len(u))[0] == \
               codec.decode(nrm, 32, "byte", length=len(u))[0], text


def test_inferred_length_matches_oracle_length_on_clean_codes():
    """The margin based length rule must be right when there is no noise at all."""
    for text in WORDS:
        u = codec.text_units(text, "byte")[:32]
        if not u:
            continue
        v, _, _ = codec.encode(u, 32, "byte")
        inferred, _ = codec.decode(v, 32, "byte")
        assert inferred == list(u), "length inference wrong for %r" % text


def test_factored_codec_equals_the_definition():
    """The factored form is exact, not an approximation. This is the check that licenses using it."""
    codes = K.KronCodes(WORDS, 32, "byte")
    worst = 0.0
    for i, text in enumerate(WORDS):
        want = K.exact_code(text, 32, "byte", 2)
        worst = max(worst, float(np.abs(codes.dense(i) - want).max()))
    assert worst < 1e-12, "factored codec differs from the float64 definition by %.2e" % worst


def test_vectorised_matmul_matches_the_loop():
    """Both paths must agree. They are not bit identical, which is why the tolerance is float32."""
    codes = K.KronCodes(WORDS, 32, "byte")
    W = np.random.default_rng(0).standard_normal((codes.D, 16)).astype(np.float32) / 90.0
    ids = np.arange(len(WORDS))
    a = codes.matmul(W, ids, vectorised=False)
    b = codes.matmul(W, ids, vectorised=True)
    assert float(np.abs(a - b).max()) < 1e-5, "vectorised and loop paths disagree"


# ---------------------------------------------------------------- nullspace collision

def test_constructed_collision_really_collides():
    """Two different codes forced onto the same projection, checked at machine precision."""
    codes = K.KronCodes(WORDS, 32, "byte")
    rng = np.random.default_rng(3)
    A = rng.standard_normal((64, codes.D)) / np.sqrt(codes.D)
    ka, kb = codes.dense(0), codes.dense(3)
    gram_inv = np.linalg.inv(A @ A.T)
    x = kb - A.T @ (gram_inv @ (A @ (kb - ka)))
    assert float(np.abs(A @ x - A @ ka).max()) < 1e-9, "construction did not produce a collision"
    assert float(np.abs(x - ka).max()) > 1e-6, "the two vectors are not actually different"


def test_collisions_are_off_manifold():
    """A real code re-encodes to itself; a collision vector does not. Counting zeros would not
    show this, because z-normalisation leaves no zeros."""
    codes = K.KronCodes(WORDS, 32, "byte")
    rng = np.random.default_rng(3)
    A = rng.standard_normal((64, codes.D)) / np.sqrt(codes.D)
    ka, kb = codes.dense(0), codes.dense(3)
    x = kb - A.T @ (np.linalg.inv(A @ A.T) @ (A @ (kb - ka)))
    dec, _ = codec.decode(x, 32, "byte", length=len(codes.units[3]))
    re_x, _, _ = codec.encode(dec, 32, "byte")
    residual = float(np.linalg.norm(x - re_x) / np.linalg.norm(re_x))
    assert residual > 0.01, "collision vector looks like a legal code (residual %.4f)" % residual


# ---------------------------------------------------------------- constrained decoding

def test_constrained_decode_is_always_valid_utf8():
    rng = np.random.default_rng(11)
    for _ in range(200):
        lg = rng.standard_normal((256, 32))
        units = codec.decode_constrained(lg, 32)
        text, ok = codec.units_to_text(units, "byte")
        assert ok, "constrained decode produced invalid UTF-8"


def test_constrained_decode_is_a_noop_on_well_formed_codes():
    """It must not damage output that was already valid UTF-8."""
    for text in WORDS:
        u = codec.text_units(text, "byte")[:32]
        if not u:
            continue
        try:
            bytes(u).decode("utf-8")
        except UnicodeDecodeError:
            continue                       # covered by the next test
        v, _, _ = codec.encode(u, 32, "byte")
        m = np.asarray(v, dtype=np.float64).reshape(256, 32)
        assert codec.decode_constrained(m, 32)[:len(u)] == list(u), text


def test_constrained_decode_drops_a_split_character_cleanly():
    """A window that ends mid-character cannot be reproduced by a decoder that only emits valid
    UTF-8, and that is the right behaviour. What it must not do is pad the gap: for a one-hot code
    every permitted byte scores the same, so a naive fallback returns index 0 and appends NUL
    bytes. The output must be a clean prefix of the retained units instead."""
    text = "प्रौद्योगिकी"                       # 36 bytes, so 32 lands mid-character
    u = codec.text_units(text, "byte")[:32]
    try:
        bytes(u).decode("utf-8")
        raise AssertionError("test fixture no longer truncates mid-character")
    except UnicodeDecodeError:
        pass
    v, _, _ = codec.encode(u, 32, "byte")
    m = np.asarray(v, dtype=np.float64).reshape(256, 32)
    got = codec.decode_constrained(m, 32)
    assert got == list(u[:len(got)]), "output is not a prefix of the retained units"
    assert 0 not in got, "padded with NUL bytes instead of stopping"
    out, ok = codec.units_to_text(got, "byte")
    assert ok and out == text[:len(out)], "did not drop exactly the split character"


def test_constrained_decode_does_not_emit_empty_for_short_targets():
    """The budget rule. Without it a short token decodes to nothing, which is trivially valid
    UTF-8 and would flatter the validity rate while saying nothing."""
    rng = np.random.default_rng(5)
    empties = 0
    for _ in range(200):
        lg = rng.standard_normal((256, 1))
        if not codec.decode_constrained(lg, 1):
            empties += 1
    assert empties == 0, "%d of 200 single-position decodes were empty" % empties


# ---------------------------------------------------------------- determinism

def test_model_init_is_reproducible():
    codes = K.KronCodes(WORDS, 32, "byte")
    a = K.KronTiny(codes, d=16, n_layer=1, n_head=2, max_pos=8, head="byte_tied",
                   vocab=len(WORDS), seed=42)
    b = K.KronTiny(codes, d=16, n_layer=1, n_head=2, max_pos=8, head="byte_tied",
                   vocab=len(WORDS), seed=42)
    for k in a.p:
        assert a.p[k].tobytes() == b.p[k].tobytes(), "parameter %s differs at the same seed" % k


def test_gradients_are_correct_including_the_tied_path():
    """W receives gradient as the input projection AND as the unembedding. An implementation that
    adds only one of the two still trains, just wrongly."""
    codes = K.KronCodes(WORDS, 16, "byte")
    m = K.KronTiny(codes, d=16, n_layer=1, n_head=2, max_pos=8, head="byte_tied",
                   vocab=len(WORDS), seed=5)
    for k in m.p:
        m.p[k] = m.p[k].astype(np.float64)
    ids = np.array([[0, 1, 2, 3]])
    pos = np.array([[0, 1, 2, 3]])
    seg = np.ones((1, 4), dtype=np.int64)
    res = K.grad_check(m, ids, pos, seg, ["W", "wq0", "lnf_g"], n_probe=4)
    worst = max(v["max_rel_error"] for v in res.values())
    assert worst < 1e-4, "gradient check worst relative error %.2e" % worst


if __name__ == "__main__":
    print("Problem 5 invariants")
    for nm, fn in sorted(globals().items()):
        if nm.startswith("test_") and callable(fn):
            check(nm[5:].replace("_", " "), fn)
    print("\n%d failure(s)" % len(FAILURES) if FAILURES else "\nall invariants hold")
    sys.exit(1 if FAILURES else 0)
