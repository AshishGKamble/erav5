"""
Problem 5, the follow-up the E3 recheck forced: is the information gone, or is the decoder wrong?

`exp_train.e3_recheck_with_trained_W` swept the minimum-norm preimage decoder over a trained `W` and
found recovery collapsing (0.75 -> 0.31 at d_model=256). Taken at face value that closes the earlier
E3 result in the negative. But minimum-norm recovery is a decoder that assumes **nothing** about the
measurement matrix: it is the right tool for a random `W`, which is what E3 used, and it is the
wrong tool for a trained one. A trained `W` is highly structured, and asking a structure-blind
decoder to invert it measures the decoder, not the encoding.

So the question is split in two, because they have different answers and conflating them is how this
gets overclaimed in either direction:

  1. **Is the map still injective on real codes?** If two distinct tokens land on distinguishable
     points, no information has been destroyed, whatever any particular decoder manages. Measured as
     the nearest-neighbour separation among projected codes, before and against after training.

  2. **Can any linear decoder recover the code?** Minimum norm is the best *structure-blind* linear
     inverse. A structure-aware one is fitted from the codes themselves, on a training split, and
     scored on tokens it never saw, so it cannot be accused of memorising the answer.
"""
import sys, os, json

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import codec, kron_model as K, vocabulary  # noqa: E402
import exp_train as T  # noqa: E402

SEED = 20260825


def projected_separation(codes, W, ids, rng):
    """Nearest-neighbour separation among projected codes, normalised by their own scale.

    If distinct tokens stay distinguishable after projection then the projection has not destroyed
    the information, regardless of whether any given decoder finds it.
    """
    E = np.stack([codes.dense(i) for i in ids]) @ np.asarray(W, dtype=np.float64)
    n = E.shape[0]
    # Pairwise distances, then the smallest off-diagonal per row.
    sq = (E ** 2).sum(1)
    d2 = np.maximum(sq[:, None] + sq[None, :] - 2 * (E @ E.T), 0.0)
    np.fill_diagonal(d2, np.inf)
    nn = np.sqrt(d2.min(1))
    scale = np.sqrt(sq).mean()
    return {
        "tokens": int(n),
        "mean_norm": float(scale),
        "nearest_neighbour_min": float(nn.min()),
        "nearest_neighbour_mean": float(nn.mean()),
        "relative_separation_min": float(nn.min() / max(scale, 1e-30)),
        "relative_separation_mean": float(nn.mean() / max(scale, 1e-30)),
        "exact_duplicates": int((nn == 0).sum()),
    }


def learned_inverse(codes, W, fit_ids, test_ids, L, ridge=1e-6):
    """Fit a linear decoder from projected codes back to codes, on a split.

    `A` is solved on `fit_ids` only and applied to `test_ids`, so a high score cannot come from
    having seen the answer. This is the structure-aware counterpart to the minimum-norm decoder.
    """
    Wm = np.asarray(W, dtype=np.float64)
    Kfit = np.stack([codes.dense(i) for i in fit_ids])
    Efit = Kfit @ Wm
    G = Efit.T @ Efit + ridge * np.eye(Efit.shape[1])
    A = np.linalg.solve(G, Efit.T @ Kfit)          # (d, D)
    ok = 0
    for i in test_ids:
        k = codes.dense(i)
        rec = (k @ Wm) @ A
        back, _ = codec.decode(rec, L, "byte", length=codes.used[i])
        ok += back == list(codes.units[i])
    return {"fit_tokens": len(fit_ids), "test_tokens": len(test_ids),
            "held_out_decode_accuracy": ok / max(1, len(test_ids))}


def main(corpus_root, tokenizer_path, out_path, d=256, L=32, steps=300, n_probe=600,
         n_fit=2500, seed=1):
    toks, meta = vocabulary.load(tokenizer_path)
    texts = [t for _, t, _ in toks]
    tr, va = T.load_data(corpus_root, tokenizer_path, "indic", T=128, max_docs=400)
    codes = K.KronCodes(texts, L, "byte")
    rng = np.random.default_rng(SEED)

    m = K.KronTiny(codes, d=d, n_layer=2, n_head=4, max_pos=tr[0].shape[1],
                   head="byte_tied", vocab=len(texts), seed=seed)
    W0 = m.p["W"].copy()
    T.train(m, tr, va, steps, seed=SEED, eval_every=None)
    W1 = m.p["W"].copy()
    np.save(os.path.join(os.path.dirname(out_path), "W_trained_d%d.npy" % d), W1)

    usable = [i for i in range(len(texts)) if 0 < codes.used[i] <= L]
    probe = list(rng.choice(usable, size=min(n_probe, len(usable)), replace=False))
    rest = [i for i in usable if i not in set(probe)]
    fit = list(rng.choice(rest, size=min(n_fit, len(rest)), replace=False))

    out = {"d_model": d, "window": L, "D": codes.D, "training_steps": steps,
           "tokenizer": meta, "probe_tokens": len(probe), "fit_tokens": len(fit)}
    for label, W in (("random_init", W0), ("after_training", W1)):
        out[label] = {
            "separation": projected_separation(codes, W, probe, rng),
            "learned_linear_inverse": learned_inverse(codes, W, fit, probe, L),
        }
    sep_a = out["after_training"]["separation"]["relative_separation_min"]
    inv_a = out["after_training"]["learned_linear_inverse"]["held_out_decode_accuracy"]
    inv_b = out["random_init"]["learned_linear_inverse"]["held_out_decode_accuracy"]
    out["verdict"] = {
        "injective_after_training": bool(sep_a > 1e-9),
        "learned_inverse_accuracy_change": inv_a - inv_b,
        "reading": ("Separation greater than zero means distinct tokens remain distinguishable "
                    "after the trained projection, so the information survives. The learned inverse "
                    "says whether a decoder that knows the code distribution can act on it. "
                    "Minimum-norm recovery collapsing while these hold means the earlier E3 result "
                    "was about a structure-blind decoder, not about the encoding."),
    }
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)
    return out


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.join(here, "..", "..", "..", "assignment-6", "frozen")
    r = main(os.path.join(root, "corpus"), os.path.join(root, "tokenizer.json"),
             os.path.join(here, "..", "artifacts", "recheck.json"))
    print(json.dumps({k: v for k, v in r.items()
                      if k in ("random_init", "after_training", "verdict")}, indent=2))
