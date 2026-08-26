"""
Problem 5, experiments E1 to E3: the three claims that need no training at all.

E1  the codec inverts exactly
E2  inversion survives approximate prediction, which is the objection that makes P5 look impossible
E3  the irreversibility is in the learned projection, and can be localised constructively

Nothing here is a simulation. Every rate is counted from actual encode/decode round trips over the
actual assignment-2 vocabulary.
"""
import sys, os, json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "common"))
import codec, vocabulary  # noqa: E402
import provenance  # noqa: E402

SEED = 20260825


def e1_roundtrip(tokens, windows=(16, 32, 64), unit="byte"):
    """Encode every vocabulary token, decode it, count exact recoveries.

    Reported separately for tokens that fit the window and tokens that overflow it, because a
    failure in the second group is a truncation loss and belongs to Problem 3, not a codec defect.
    """
    out = {}
    for L in windows:
        fit_ok = fit_n = over_ok = over_n = 0
        for _, text in tokens:
            units = codec.text_units(text, unit)
            v, cropped, _ = codec.encode(units, L, unit)
            back, _ = codec.decode(v, L, unit, length=min(len(units), L))
            ok = back == units[:L]
            if cropped:
                over_n += 1
                over_ok += ok
            else:
                fit_n += 1
                fit_ok += ok
        out[str(L)] = {
            "window": L, "D": codec.dim(L, unit),
            "tokens_fitting": fit_n, "tokens_fitting_recovered": fit_ok,
            "tokens_fitting_rate": fit_ok / fit_n if fit_n else None,
            "tokens_overflowing": over_n,
            "tokens_overflowing_prefix_recovered": over_ok,
            "tokens_overflowing_prefix_rate": over_ok / over_n if over_n else None,
            "_overflow_note": ("For overflowing tokens this is the rate at which the RETAINED "
                               "prefix comes back exactly. The token itself is not recovered and "
                               "cannot be: the dropped bytes are gone. That loss is Problem 3."),
        }
    return out


def e2_noise(tokens, L=32, unit="byte", sample=2000, sigmas=None, seed=SEED):
    """Sweep Gaussian noise and measure exact-token decode accuracy.

    The codec output is z-normalised, so its own standard deviation is exactly 1. Noise is therefore
    quoted as a multiple of the signal's own scale, which makes the numbers comparable across window
    sizes and independent of any arbitrary units.

    Two decoders are measured, because they fail differently:
      * oracle length  - the true token length is supplied, isolating the argmax
      * inferred length - the length is recovered from the column margin, as it would have to be at
        inference time when nothing knows the answer in advance
    """
    rng = np.random.default_rng(seed)
    if sigmas is None:
        sigmas = [0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0]
    idx = rng.choice(len(tokens), size=min(sample, len(tokens)), replace=False)
    chosen = [tokens[i] for i in idx]
    enc = []
    for _, text in chosen:
        units = codec.text_units(text, unit)
        if len(units) > L:
            continue                      # truncation is Problem 3; keep this measurement clean
        v, _, _ = codec.encode(units, L, unit)
        enc.append((units, v))

    # Analytic margin: after z-normalisation the gap between the correct row and every other row in
    # an occupied column is (1/sqrt(L)) / sd(v). Measured here rather than assumed.
    gaps = []
    for units, v in enc[:200]:
        m = v.reshape(codec.CHAR_DIM, L)
        col = m[:, 0]
        srt = np.sort(col)
        gaps.append(float(srt[-1] - srt[-2]))
    margin = float(np.mean(gaps))

    rows = []
    for s in sigmas:
        ok_oracle = ok_inferred = 0
        for units, v in enc:
            noisy = v + rng.normal(0.0, s, size=v.shape).astype(v.dtype) if s > 0 else v
            back, _ = codec.decode(noisy, L, unit, length=len(units))
            ok_oracle += back == units
            back2, _ = codec.decode(noisy, L, unit)
            ok_inferred += back2 == units
        rows.append({
            "sigma": s,
            "sigma_relative_to_signal_sd": s,
            "exact_token_accuracy_oracle_length": ok_oracle / len(enc),
            "exact_token_accuracy_inferred_length": ok_inferred / len(enc),
        })
    return {
        "window": L, "D": codec.dim(L, unit), "tokens_measured": len(enc),
        "column_margin_after_znorm": margin,
        "signal_sd": 1.0,
        "sweep": rows,
    }


def e3_projection(tokens, L=32, unit="byte", d_models=(8, 16, 32, 48, 64, 96, 128, 192, 256, 384,
                                                          512, 768), sample=300, seed=SEED):
    """Where does invertibility actually die?

    This experiment was pre-registered in PLAN.md with the prediction that decoding through
    `Linear(D -> d_model)` would land near chance, because the map has a 7424-dimensional nullspace
    and no left inverse. **That prediction is wrong, and the numbers below are why.**

    A codec vector is not an arbitrary point in R^8192. It is k-sparse, with k equal to the number
    of occupied position columns, and k is small: single digits on average for this vocabulary.
    Recovering a k-sparse vector from a random linear measurement is the compressed-sensing regime,
    and it succeeds whenever the measurement count is comfortably above k. With d_model = 768 and
    k around 9, that condition is not marginal, it is met by two orders of magnitude.

    So the honest result is a **capacity threshold**, not a wall: sweep d_model and the minimum-norm
    preimage goes from useless to essentially perfect, and the transition is where the answer lives.

    Exact collisions still exist, and are still constructed here at machine precision. What the
    sweep shows is that they are off-manifold: reaching one costs a vector that is neither sparse
    nor a possible codec output, so they do not threaten a decoder that only ever sees real codes.
    """
    rng = np.random.default_rng(seed)
    D = codec.dim(L, unit)

    fitting = [u for u in (codec.text_units(t, unit) for _, t in tokens) if 0 < len(u) <= L]
    pick = rng.choice(len(fitting), size=min(sample, len(fitting)), replace=False)
    items = [(fitting[i], codec.encode(fitting[i], L, unit)[0].astype(np.float64)) for i in pick]
    occupancy = [len(u) for u, _ in items]

    sweep = []
    collision = None
    for dm in d_models:
        W = (rng.standard_normal((dm, D)) / np.sqrt(D))
        gram_inv = np.linalg.inv(W @ W.T)

        def p_row(y):
            return W.T @ (gram_inv @ (W @ y))

        ok = 0
        for units, k in items:
            rec = p_row(k)
            back, _ = codec.decode(rec, L, unit, length=len(units))
            ok += back == units
        sweep.append({"d_model": dm, "nullspace_dimension": D - dm,
                      "minimum_norm_decode_accuracy": ok / len(items)})

        if dm == max(d_models):
            hits = errs = 0
            offman_collision, offman_real = [], []
            for i in range(0, len(items) - 1, 2):
                (ua, ka), (ub, kb) = items[i], items[i + 1]
                x = kb - p_row(kb - ka)          # identical projection to ka, by construction
                errs = max(errs, float(np.abs(W @ x - W @ ka).max()))
                dec, _ = codec.decode(x, L, unit, length=len(ub))
                hits += dec == ub
                # Is the collision vector itself a legal codec output? Re-encode what it decodes to
                # and measure the residual. A real code has residual exactly 0 by construction; a
                # collision does not, and that gap is the sense in which it is off-manifold.
                # (Counting zeros or comparing norms says nothing here: z-normalisation leaves no
                # zeros, and gives every code the same norm.)
                re_x, _, _ = codec.encode(dec, L, unit)
                offman_collision.append(float(np.linalg.norm(x - re_x) / np.linalg.norm(re_x)))
                re_b, _, _ = codec.encode(ub, L, unit)
                offman_real.append(float(np.linalg.norm(kb - re_b) / np.linalg.norm(re_b)))
            n_pairs = len(range(0, len(items) - 1, 2))
            collision = {
                "d_model": dm, "pairs_tested": n_pairs,
                "success_rate": hits / n_pairs,
                "max_projection_error": errs,
                "offmanifold_residual_collision": float(np.mean(offman_collision)),
                "offmanifold_residual_real_code": float(np.mean(offman_real)),
                "metric": ("relative L2 distance from the vector to the re-encoding of whatever it "
                           "decodes to; 0 means the vector is a legal codec output"),
            }

    accs = [r["minimum_norm_decode_accuracy"] for r in sweep]
    def crossing(target):
        for r, a in zip(sweep, accs):
            if a >= target:
                return r["d_model"]
        return None

    return {
        "window": L, "D": D,
        "codes_measured": len(items),
        "occupied_columns_mean": float(np.mean(occupancy)),
        "occupied_columns_max": int(np.max(occupancy)),
        "sparsity_fraction_mean": float(np.mean(occupancy)) / D,
        "sweep": sweep,
        "d_model_for_50pc_recovery": crossing(0.5),
        "d_model_for_99pc_recovery": crossing(0.99),
        "collision": collision,
        "prediction_in_plan": "near-chance decode accuracy through the projection",
        "prediction_outcome": "refuted",
        "verdict": ("The projection is not where reversibility dies. A Kronecker code is k-sparse "
                    "with k in single digits, so a random d_model-dimensional projection retains "
                    "enough to recover it once d_model comfortably exceeds k. Exact collisions "
                    "exist but are dense vectors far off the codec manifold, so they are not "
                    "reachable outputs of the encoder."),
    }


def main(tokenizer_path, out_path):
    toks, meta = vocabulary.load(tokenizer_path)
    real = vocabulary.real_tokens(toks)
    result = {
        "seed": SEED,
        "tokenizer": meta,
        "vocabulary_measured": len(real),
        "e1_roundtrip": e1_roundtrip(real),
        "e2_noise": e2_noise(real),
        "e3_projection": e3_projection(real),
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        provenance.stamp(result, __file__)
        json.dump(result, fh, indent=2, sort_keys=True)
    return result


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    r = main(os.path.join(here, "..", "..", "..", "assignment-6", "frozen", "tokenizer.json"),
             os.path.join(here, "..", "artifacts", "codec.json"))
    print(json.dumps(r["e1_roundtrip"], indent=2))
