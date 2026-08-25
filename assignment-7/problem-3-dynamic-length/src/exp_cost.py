"""
Problem 3, experiment E6: what the fixed window actually costs, measured three ways.

The assignment asks two questions this problem set had not yet answered, and they are the two it
asks first:

> "Today Kronecker is limiting to presenting 32 position for every work (even 'apple' or 'a' as
>  well). **That's a waste of space. What can we do? How can it be dynamic**...?"

E1 measured the waste and found it enormous, 92 to 95 percent of columns empty at L=32. It is
tempting to stop there and argue the waste is benign because zeros are harmless. That argument is
correct about *accuracy* and evades the question actually asked, which is about **space**.

So the cost is separated into the three things "space" can mean, because they have three different
answers and only one of them is bad news:

  1. **Dimensions.** D = 256L is fixed and cannot be reclaimed. `Linear(D, d)` needs a fixed input
     width, so a token cannot be given its own D. This is the one real, irreducible cost, and the
     honest answer to "can it be dynamic" in the dimensional sense is **no**, with a reason.

  2. **Memory.** Entirely reclaimable. A code is a list of occupied row indices plus two scalars,
     so nothing has to store the zeros.

  3. **Compute.** Already dynamic, with no architectural change at all. `kappa @ W` touches only
     the occupied rows, so the work is proportional to the token's **actual length**: "a" costs one
     row lookup and a thirty-byte word costs thirty. The window never charges a short token for
     positions it does not use.

The practical consequence is the useful part, and it inverts the assignment's premise. Because
compute scales with real length rather than with L, **L is cheap to increase**. Raising the window
from 32 to 64 costs a short token nothing at all, and E3 measured that L=64 removes nearly every
truncation collision. The answer to "how can it be dynamic" and the answer to "don't force us to
crop a word" turn out to be the same answer.
"""
import sys, os, json, time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "common"))
sys.path.insert(0, HERE)
import codec, kron_model as K, vocabulary  # noqa: E402

WINDOWS = (16, 32, 64, 128)
D_MODEL = 96
SEED = 20260825


def _time(fn, repeats, warmup=2):
    for _ in range(warmup):
        fn()
    best = np.inf
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def dimensions_cost(codes_by_L):
    """The irreducible cost, and why it is irreducible."""
    return {
        "rows": {str(L): {"D": c.D, "mean_occupied_columns": float(c._counts.mean()),
                          "occupancy": float(c._counts.mean()) / L,
                          "zero_column_fraction": 1.0 - float(c._counts.mean()) / L}
                 for L, c in codes_by_L.items()},
        "reclaimable": False,
        "why": ("The codec output is the input to `Linear(D, d)`, whose weight matrix has a fixed "
                "first dimension. A per-token D would mean a per-token weight matrix, which is the "
                "per-token parameter table the construction exists to remove. So the dimensional "
                "waste is real and cannot be given back without abandoning the idea."),
    }


def memory_cost(codes_by_L, vocab_size):
    rows = {}
    for L, c in codes_by_L.items():
        nb = c.nbytes()
        rows[str(L)] = dict(nb, D=c.D, vocab=vocab_size,
                            dense_mb=nb["dense_bytes"] / 2 ** 20,
                            factored_mb=nb["factored_bytes"] / 2 ** 20)
    return {"rows": rows, "reclaimable": True,
            "why": ("A code is fully described by its occupied row indices plus two scalars, so the "
                    "zeros are never stored. This is not an approximation: the factored form "
                    "reproduces the codec definition to about 1e-14.")}


def compute_cost(codes_by_L, ids, repeats=5):
    """Dense matmul against the factored path, plus the arithmetic each one actually performs."""
    rng = np.random.default_rng(SEED)
    rows = {}
    for L, c in codes_by_L.items():
        W = (rng.standard_normal((c.D, D_MODEL)) / np.sqrt(c.D)).astype(np.float32)
        t_fact = _time(lambda: c.matmul(W, ids, vectorised=True), repeats)
        entry = {
            "D": c.D,
            "tokens": int(ids.size),
            "factored_seconds": t_fact,
            "factored_multiply_accumulates": float(c._counts[ids.reshape(-1)].sum()) * D_MODEL,
            "dense_multiply_accumulates": float(ids.size) * c.D * D_MODEL,
        }
        entry["arithmetic_ratio"] = (entry["dense_multiply_accumulates"]
                                     / max(1.0, entry["factored_multiply_accumulates"]))
        # The dense comparison is only run where the matrix fits comfortably in memory.
        dense_bytes = len(c.rows) * c.D * 4
        if dense_bytes < 1.5 * 2 ** 30:
            Kd = np.zeros((len(c.rows), c.D), dtype=np.float32)
            for i in range(len(c.rows)):
                Kd[i, c.rows[i]] = c.scale[i]
                Kd[i] -= c.shift[i]
            flat = ids.reshape(-1)
            t_dense = _time(lambda: Kd[flat] @ W, repeats)
            entry["dense_seconds"] = t_dense
            entry["speedup"] = t_dense / t_fact
            del Kd
        else:
            entry["dense_seconds"] = None
            entry["speedup"] = None
            entry["dense_skipped_because"] = "dense matrix exceeds 1.5 GB at this window"
        rows[str(L)] = entry
    return {"rows": rows, "d_model": D_MODEL,
            "why": ("Only the occupied rows of W are touched, so the arithmetic is proportional to "
                    "the number of units the token really has. Wall-clock is reported alongside "
                    "the arithmetic because the two differ: the dense path is one large BLAS call "
                    "and the factored path is a gather plus a segmented reduction.")}


def scaling_with_length(codes, texts, repeats=5, per_bucket=4096):
    """The dynamic claim, made falsifiable: does cost track token length?

    Tokens are bucketed by their true length and each bucket is timed on its own. If the window
    charged every token for L positions these timings would be flat. If cost follows real length
    they rise with the bucket.
    """
    rng = np.random.default_rng(SEED)
    W = (rng.standard_normal((codes.D, D_MODEL)) / np.sqrt(codes.D)).astype(np.float32)
    by_len = {}
    for i in range(len(texts)):
        by_len.setdefault(int(codes._counts[i]), []).append(i)
    out = []
    for length in sorted(by_len):
        pool = by_len[length]
        if len(pool) < 8 or length == 0:
            continue
        ids = np.asarray(rng.choice(pool, size=per_bucket, replace=True))
        t = _time(lambda: codes.matmul(W, ids, vectorised=True), repeats)
        out.append({"occupied_units": length, "tokens_timed": int(ids.size),
                    "seconds": t, "nanoseconds_per_token": t / ids.size * 1e9,
                    "distinct_tokens_available": len(pool)})
    if len(out) > 2:
        x = np.array([r["occupied_units"] for r in out], dtype=np.float64)
        y = np.array([r["nanoseconds_per_token"] for r in out], dtype=np.float64)
        A = np.vstack([x, np.ones_like(x)]).T
        slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
        corr = float(np.corrcoef(x, y)[0, 1])
    else:
        slope = intercept = corr = None
    return {"buckets": out, "linear_fit_slope_ns_per_unit": slope,
            "linear_fit_intercept_ns": intercept, "correlation": corr,
            "reading": ("A positive slope with a high correlation is the dynamic claim in its "
                        "measurable form: the encoder charges per unit present, not per window "
                        "slot. A flat line would refute it.")}


def raising_L_is_cheap(codes_by_L, ids, repeats=5):
    """The synthesis: because cost follows length, a bigger window is nearly free for short tokens.

    E3 measured that L=64 removes almost every truncation collision. This asks what that costs,
    and the answer is what makes it a recommendation rather than a wish.
    """
    base = None
    rows = {}
    rng = np.random.default_rng(SEED)
    short = [i for i in range(len(codes_by_L[32].rows)) if 0 < codes_by_L[32]._counts[i] <= 4]
    short_ids = np.asarray(rng.choice(short, size=min(8192, len(short) * 8), replace=True))
    for L, c in sorted(codes_by_L.items()):
        W = (rng.standard_normal((c.D, D_MODEL)) / np.sqrt(c.D)).astype(np.float32)
        t = _time(lambda: c.matmul(W, short_ids, vectorised=True), repeats)
        nb = c.nbytes()
        rows[str(L)] = {"D": c.D, "short_token_seconds": t,
                        "factored_mb": nb["factored_bytes"] / 2 ** 20,
                        "projection_parameters": c.D * D_MODEL}
        if L == 32:
            base = t
    for L, r in rows.items():
        r["short_token_time_vs_L32"] = r["short_token_seconds"] / base if base else None
    return {"rows": rows, "tokens_timed": int(short_ids.size),
            "definition_of_short": "tokens occupying 4 or fewer units",
            "reading": ("Encoding a short token costs essentially the same at any window size, "
                        "because the work follows the units present. What genuinely grows with L "
                        "is the projection matrix `W`, at 256 x L x d_model parameters, and that "
                        "is the number to weigh against the truncation E3 counted.")}


def main(tokenizer_path, out_path, n_ids=8192):
    toks, meta = vocabulary.load(tokenizer_path)
    texts = [t for _, t, _ in toks]
    codes_by_L = {L: K.KronCodes(texts, L, "byte") for L in WINDOWS}
    rng = np.random.default_rng(SEED)
    ids = rng.integers(0, len(texts), size=n_ids)

    result = {
        "tokenizer": meta,
        "vocabulary": len(texts),
        "windows": list(WINDOWS),
        "d_model": D_MODEL,
        "question": ("The assignment asks 'that's a waste of space, what can we do, how can it be "
                     "dynamic'. Space means three different things here and they have three "
                     "different answers."),
        "dimensions": dimensions_cost(codes_by_L),
        "memory": memory_cost(codes_by_L, len(texts)),
        "compute": compute_cost(codes_by_L, ids),
        "scaling_with_length": scaling_with_length(codes_by_L[32], texts),
        "raising_L_is_cheap": raising_L_is_cheap(codes_by_L, ids),
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(result, fh, indent=2, sort_keys=True)
    return result


if __name__ == "__main__":
    root = os.path.join(HERE, "..", "..", "..", "assignment-6", "frozen")
    r = main(os.path.join(root, "tokenizer.json"),
             os.path.join(HERE, "..", "artifacts", "cost.json"))
    m = r["memory"]["rows"]["32"]
    c = r["compute"]["rows"]["32"]
    print(f"memory  L=32: dense {m['dense_mb']:.1f} MB -> factored {m['factored_mb']:.3f} MB "
          f"({m['ratio']:.0f}x)")
    print(f"compute L=32: arithmetic ratio {c['arithmetic_ratio']:.0f}x, "
          f"wall-clock speedup {c['speedup']}")
    s = r["scaling_with_length"]
    print(f"scaling: slope {s['linear_fit_slope_ns_per_unit']:.3f} ns/unit, corr {s['correlation']:.4f}")
