"""
Problem 5, experiments E4 to E6: the trained half of the argument.

E4  three output heads, identical in every other respect, compared at equal training
E5  gradient exists from step 0, and cross-entropy is the reason, not the architecture
E6  what tying costs, counted rather than asserted

**The measurement trap that governs this whole file.** A vocabulary head's loss is nats per *token*.
A byte head's loss is nats per *byte position*, and a token occupies several positions. The two
numbers are not comparable and putting them in one table without conversion would be the easiest
way to make either head look better than it is. Every comparison below is reported in **nats per
token**, obtained by multiplying a byte head's per-position loss by the measured number of scored
positions per token, with the native units kept alongside so the conversion can be checked.

A second trap, stated because it moves the headline: at initialisation the byte head is *worse* per
token than the vocabulary head, and necessarily so. Uniform over 256 values at about 2.3 positions
per token is roughly 12.7 nats, against about 9.2 for uniform over a 10k vocabulary. The byte head
has to earn its advantage, and where it ends up relative to that starting point is the result.
"""
import sys, os, json, math, time
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"))
import codec, data, kron_model as K, vocabulary  # noqa: E402
import provenance  # noqa: E402

SEED = 20260825
PAPER_DMODEL = 768
PAPER_VOCAB = 131072


def make_codes(texts, L=32, unit="byte"):
    return K.KronCodes(texts, L, unit)


def scored_positions_per_token(model, ids, seg_ids):
    """How many byte positions one token contributes to the loss. The conversion factor."""
    tgt, valid = model.targets(ids, seg_ids)
    _, occ, _, _, _ = model.byte_targets(tgt)
    vmask = occ & valid[:, :, None, None]
    return float(vmask.sum()) / max(1, float(valid.sum()))


def evaluate(model, arrays, batch_size=8, objective="ce", max_batches=None):
    """Validation loss plus accuracy, in both native and per-token units."""
    ids_a, pos_a, seg_a = arrays
    tot_loss = tot_units = 0.0
    tot_tokens = 0
    byte_hits = byte_total = 0
    tok_hits = tok_total = 0
    for b, (ids, pos, seg) in enumerate(data.batches(arrays, batch_size)):
        if max_batches and b >= max_batches:
            break
        xf, _ = model.forward(ids, pos, seg)
        tgt, valid = model.targets(ids, seg)
        n_tok = int(valid.sum())
        if n_tok == 0:
            continue
        tot_tokens += n_tok
        if model.head == "vocab":
            lg = (xf @ model.p["U"].T).astype(np.float64)
            lg -= lg.max(-1, keepdims=True)
            zt = np.take_along_axis(lg, tgt[:, :, None], axis=2)[:, :, 0]
            nll = -(zt - np.log(np.exp(lg).sum(-1)))
            tot_loss += float((nll * valid).sum())
            tot_units += n_tok
            pred = lg.argmax(-1)
            tok_hits += int(((pred == tgt) & valid).sum())
            tok_total += n_tok
        else:
            y, occ, _, blocks, L = model.byte_targets(tgt)
            vmask = occ & valid[:, :, None, None]
            flat = model.head_logits(xf).astype(np.float64)
            B, T = ids.shape
            lg = flat.reshape(B, T, blocks, codec.CHAR_DIM, L)
            lg = lg - lg.max(axis=3, keepdims=True)
            yt = np.take_along_axis(lg, y[:, :, :, None, :], axis=3)[:, :, :, 0, :]
            nll = -(yt - np.log(np.exp(lg).sum(axis=3)))
            tot_loss += float((nll * vmask).sum())
            tot_units += int(vmask.sum())
            pred = lg.argmax(axis=3)
            byte_hits += int(((pred == y) & vmask).sum())
            byte_total += int(vmask.sum())
            # Exact token: every occupied position correct, with the true length supplied.
            ok = ((pred == y) | ~vmask).all(axis=(2, 3)) & valid
            tok_hits += int(ok.sum())
            tok_total += n_tok
    per_unit = tot_loss / max(1, tot_units)
    return {
        "loss_native": per_unit,
        "native_unit": ("nats per token" if model.head == "vocab"
                        else "nats per occupied byte position"),
        "loss_per_token": tot_loss / max(1, tot_tokens),
        "scored_units_per_token": tot_units / max(1, tot_tokens),
        "byte_accuracy": (byte_hits / byte_total) if byte_total else None,
        "exact_token_accuracy_oracle_length": (tok_hits / tok_total) if tok_total else None,
        "tokens_scored": tot_tokens,
    }


def train(model, train_arrays, val_arrays, steps, lr=3e-3, batch_size=8, objective="ce",
          seed=SEED, eval_every=None, eval_batches=6):
    rng = np.random.default_rng(seed)
    opt = K.Adam(model.p, lr=lr)
    hist = []
    it = data.batches(train_arrays, batch_size, rng=rng, epochs=1000)
    t0 = time.time()
    for step in range(steps + 1):
        if eval_every and (step % eval_every == 0 or step == steps):
            ev = evaluate(model, val_arrays, batch_size, objective, max_batches=eval_batches)
            ev["step"] = step
            ev["seconds"] = time.time() - t0
            hist.append(ev)
        if step == steps:
            break
        ids, pos, seg = next(it)
        _, g, _ = model.loss_and_grad(ids, pos, seg, objective=objective)
        opt.step(model.p, g)
    return hist


# --------------------------------------------------------------------------- E4

def e4_three_heads(texts, tr, va, L=32, d=96, steps=300, seeds=(1, 2, 3), lr=3e-3, batch_size=8):
    """Three heads, identical body, identical data, identical seeds. Noise floor from repeats."""
    codes = make_codes(texts, L)
    out = {}
    for head in ("vocab", "byte_untied", "byte_tied"):
        runs = []
        for s in seeds:
            m = K.KronTiny(codes, d=d, n_layer=2, n_head=4, max_pos=tr[0].shape[1],
                           head=head, vocab=len(texts), seed=s)
            h = train(m, tr, va, steps, lr=lr, batch_size=batch_size, seed=SEED + s,
                      eval_every=max(1, steps // 6))
            runs.append({"seed": s, "history": h, "final": h[-1],
                         "params": m.param_breakdown()})
        finals = [r["final"]["loss_per_token"] for r in runs]
        out[head] = {
            "runs": runs,
            "final_loss_per_token_mean": float(np.mean(finals)),
            "final_loss_per_token_sd": float(np.std(finals, ddof=1)) if len(finals) > 1 else 0.0,
            "final_loss_per_token_runs": finals,
            "params_measured": runs[0]["params"],
            "params_at_paper_scale": paper_scale_params(head, L),
        }
    base = out["vocab"]["final_loss_per_token_mean"]
    noise = max(r["final_loss_per_token_sd"] for r in out.values())
    for head, r in out.items():
        delta = r["final_loss_per_token_mean"] - base
        r["delta_vs_vocab_head"] = delta
        r["exceeds_seed_noise"] = bool(abs(delta) > 2 * noise) if noise > 0 else None
    out["_noise_floor_sd_across_seeds"] = noise
    out["_comparison_unit"] = "nats per token, converted from each head's native unit"
    return out


def paper_scale_params(head, L, d=PAPER_DMODEL, vocab=PAPER_VOCAB):
    """Arithmetic, not measurement, and labelled as such."""
    D = codec.CHAR_DIM * L
    return {
        "note": "arithmetic at d_model=%d and vocab=%d, not measured here" % (d, vocab),
        "input_projection_W": D * d,
        "input_per_token_table": 0,
        "output_head": (vocab * d if head == "vocab" else D * d if head == "byte_untied" else 0),
        "baseline_bpe_input_table_eliminated": vocab * d,
    }


def copy_diagnostic(model, arrays, batch_size=8, max_batches=8):
    """Is the tied head, at initialisation, just echoing the token it was given?

    `xf @ W.T` with the same `W` that produced the input embedding is close to an identity on the
    residual stream, so an untrained tied head is expected to reproduce the **current** token's
    bytes rather than predict the next one. That would explain a step-0 byte accuracy far above the
    1/256 chance rate, and it matters: without it, the head's starting point looks like evidence of
    learning when it is evidence of wiring. Measured against both targets on the same predictions.
    """
    hit_next = hit_cur = tot = 0
    for b, (ids, pos, seg) in enumerate(data.batches(arrays, batch_size)):
        if b >= max_batches:
            break
        xf, _ = model.forward(ids, pos, seg)
        tgt, valid = model.targets(ids, seg)
        y_next, occ, _, blocks, L = model.byte_targets(tgt)
        y_cur, occ_cur, _, _, _ = model.byte_targets(ids)
        lg = model.head_logits(xf).astype(np.float64).reshape(
            ids.shape[0], ids.shape[1], blocks, codec.CHAR_DIM, L)
        pred = lg.argmax(axis=3)
        vm = occ & valid[:, :, None, None]
        vm_cur = occ_cur & valid[:, :, None, None]
        hit_next += int(((pred == y_next) & vm).sum())
        hit_cur += int(((pred == y_cur) & vm_cur).sum())
        tot += int(vm.sum())
    return {
        "byte_accuracy_vs_next_token": hit_next / max(1, tot),
        "byte_accuracy_vs_current_token": hit_cur / max(1, tot),
        "chance": 1.0 / codec.CHAR_DIM,
        "positions": tot,
    }


# --------------------------------------------------------------------------- E5

def e5_gradient_from_zero(texts, tr, va, L=32, d=96, steps=120, seed=1, lr=3e-3, batch_size=8):
    """Does anything move at step 0? Cross-entropy against MSE-to-kappa, same head, same seed."""
    codes = make_codes(texts, L)
    out = {}
    for objective in ("ce", "mse"):
        m = K.KronTiny(codes, d=d, n_layer=2, n_head=4, max_pos=tr[0].shape[1],
                       head="byte_tied", vocab=len(texts), seed=seed)
        if "_copy_at_init" not in out:
            out["_copy_at_init"] = copy_diagnostic(m, va)
        # Both objectives are scored with the SAME cross-entropy evaluator, so the curves are
        # comparable. Training an MSE model and then reporting its MSE would prove nothing.
        h = train(m, tr, va, steps, lr=lr, batch_size=batch_size, objective=objective,
                  seed=SEED, eval_every=max(1, steps // 12))
        first = h[0]
        out[objective] = {
            "history": h,
            "byte_accuracy_at_step_0": first["byte_accuracy"],
            "byte_accuracy_final": h[-1]["byte_accuracy"],
            "loss_per_token_at_step_0": first["loss_per_token"],
            "loss_per_token_final": h[-1]["loss_per_token"],
            "moved": bool(h[-1]["byte_accuracy"] > first["byte_accuracy"] + 1e-6),
        }
    out["_chance_byte_accuracy"] = 1.0 / codec.CHAR_DIM
    out["_reading"] = ("Both runs are evaluated with the same cross-entropy metric so the two "
                       "curves live on one axis. The claim under test is not that MSE is a bad "
                       "loss in general, it is that the 'nothing to decode at random init' "
                       "objection is a property of the objective and not of the architecture.")
    return out


# --------------------------------------------------------------------------- E6

def e6_tying_cost(texts, tr, va, L=32, d=96, steps=300, seed=1, lr=3e-3, batch_size=8,
                  max_report=15):
    """What comes out of a tied byte head that a vocabulary head could not have emitted."""
    codes = make_codes(texts, L)
    vocab_set = set(texts)
    m = K.KronTiny(codes, d=d, n_layer=2, n_head=4, max_pos=tr[0].shape[1],
                   head="byte_tied", vocab=len(texts), seed=seed)
    train(m, tr, va, steps, lr=lr, batch_size=batch_size, seed=SEED, eval_every=None)

    n = invalid_utf8 = valid_oov = exact = in_vocab = 0
    examples = Counter()
    for ids, pos, seg in data.batches(va, 8):
        xf, _ = m.forward(ids, pos, seg)
        tgt, valid = m.targets(ids, seg)
        y, occ, _, blocks, L_ = m.byte_targets(tgt)
        vmask = occ & valid[:, :, None, None]
        lg = m.head_logits(xf).astype(np.float64).reshape(ids.shape[0], ids.shape[1], blocks,
                                                          codec.CHAR_DIM, L_)
        pred = lg.argmax(axis=3)
        B, T = ids.shape
        for b in range(B):
            for t in range(T):
                if not valid[b, t]:
                    continue
                length = int(vmask[b, t, 0].sum())
                if length == 0:
                    continue
                units = [int(pred[b, t, 0, p]) for p in range(length)]
                n += 1
                text, ok = codec.units_to_text(units, "byte")
                if not ok:
                    invalid_utf8 += 1
                    continue
                if text in vocab_set:
                    in_vocab += 1
                    if text == texts[int(tgt[b, t])]:
                        exact += 1
                else:
                    valid_oov += 1
                    examples[text] += 1
    return {
        "predictions_scored": n,
        "invalid_utf8": invalid_utf8,
        "invalid_utf8_rate": invalid_utf8 / n if n else 0.0,
        "valid_utf8_out_of_vocabulary": valid_oov,
        "valid_utf8_out_of_vocabulary_rate": valid_oov / n if n else 0.0,
        "in_vocabulary": in_vocab,
        "in_vocabulary_rate": in_vocab / n if n else 0.0,
        "exact_token_match": exact,
        "exact_token_match_rate": exact / n if n else 0.0,
        "out_of_vocabulary_examples": [w for w, _ in examples.most_common(max_report)],
        "decode_length": ("oracle: the target token's true length is supplied, so these rates "
                          "measure byte prediction alone and not length inference. E2 measured "
                          "length inference from the column margin separately."),
        "reading": ("Out-of-vocabulary output is not an error rate to be minimised. It is the "
                    "mechanism behind the paper's claim that a byte head addresses a vocabulary it "
                    "was never given: a tied head can emit any byte string, including strings the "
                    "tokenizer has no id for. It is reported here as a property. Invalid UTF-8 is "
                    "the genuine cost of per-position independence, and that one is a defect."),
    }


# --------------------------------------------------------------------------- the open caveat

def e3_recheck_with_trained_W(texts, tr, va, L=32, d=96, steps=300, seed=1, lr=3e-3,
                              batch_size=8, sample=300):
    """Close the caveat E3 left open: does sparse recovery survive a *trained* projection?

    E3 swept `d_model` with a **random** `W` and found the codec recoverable from its projection,
    because a Kronecker code is k-sparse and the random measurement is the compressed-sensing
    regime. The honest caveat recorded there was that training might concentrate `W` and destroy
    the restricted-isometry-like property that makes recovery work. Nothing about E3 justified
    assuming otherwise, so the check is done here rather than argued.

    The trained `W` has shape (D, d), so `W.T` is exactly the (d, D) measurement matrix E3 swept.
    The same minimum-norm preimage decoder is applied to both, on the same tokens, so the only
    difference between the two numbers is training.
    """
    codes = make_codes(texts, L)
    rng = np.random.default_rng(SEED)
    m = K.KronTiny(codes, d=d, n_layer=2, n_head=4, max_pos=tr[0].shape[1],
                   head="byte_tied", vocab=len(texts), seed=seed)
    W0 = m.p["W"].copy()
    train(m, tr, va, steps, lr=lr, batch_size=batch_size, seed=SEED, eval_every=None)
    W1 = m.p["W"]

    fitting = [i for i in range(len(texts)) if 0 < codes.used[i] <= L]
    pick = rng.choice(len(fitting), size=min(sample, len(fitting)), replace=False)
    items = [(codes.units[fitting[i]], codes.dense(fitting[i])) for i in pick]

    def sweep(Wmat, label):
        A = np.asarray(Wmat.T, dtype=np.float64)          # (d, D)
        gram_inv = np.linalg.inv(A @ A.T + 1e-12 * np.eye(A.shape[0]))
        ok = 0
        for units, k in items:
            rec = A.T @ (gram_inv @ (A @ k))
            back, _ = codec.decode(rec, L, "byte", length=len(units))
            ok += back == list(units)
        s = np.linalg.svd(A, compute_uv=False)
        return {
            "which": label,
            "d_model": int(A.shape[0]),
            "minimum_norm_decode_accuracy": ok / len(items),
            "condition_number": float(s[0] / max(s[-1], 1e-30)),
            "singular_value_ratio_top_to_median": float(s[0] / max(np.median(s), 1e-30)),
            "frobenius_norm": float(np.linalg.norm(A)),
        }

    before, after = sweep(W0, "random init"), sweep(W1, "after training")
    return {
        "codes_measured": len(items),
        "window": L, "D": codes.D, "d_model": d, "training_steps": steps,
        "before": before,
        "after": after,
        "accuracy_change": after["minimum_norm_decode_accuracy"]
                           - before["minimum_norm_decode_accuracy"],
        "caveat_status": ("closed: recovery survives training"
                          if after["minimum_norm_decode_accuracy"]
                          >= before["minimum_norm_decode_accuracy"] - 0.02
                          else "OPEN: training degrades recovery, do not claim this end to end"),
    }


# --------------------------------------------------------------------------- main

def load_data(corpus_root, tokenizer_path, lane, T=128, max_docs=400, val_fraction=0.2):
    docs = data.tokenize_lane(corpus_root, tokenizer_path, lane, max_docs=max_docs)
    arrays = data.pack(docs, T)
    return data.split(arrays, val_fraction)


def main(corpus_root, tokenizer_path, out_path, lane="indic", steps=300, d=96, L=32,
         max_docs=400, T=128, seeds=(1, 2, 3)):
    toks, meta = vocabulary.load(tokenizer_path)
    texts = [t for _, t, _ in toks]        # indexed by token id, so byte-fallback rows stay in place
    tr, va = load_data(corpus_root, tokenizer_path, lane, T=T, max_docs=max_docs)

    codes = make_codes(texts, L)
    equiv = K.test_matches_codec(codes, texts[:400])

    probe = K.KronTiny(codes, d=32, n_layer=2, n_head=4, max_pos=T, head="byte_tied",
                       vocab=len(texts), seed=5)
    for k in probe.p:
        probe.p[k] = probe.p[k].astype(np.float64)
    gc = K.grad_check(probe, tr[0][:2, :8], tr[1][:2, :8], tr[2][:2, :8],
                      ["W", "wq0", "lnf_g"])

    t0 = time.time()
    result = {
        "seed": SEED,
        "tokenizer": meta,
        "config": {"lane": lane, "steps": steps, "d_model": d, "window": L, "T": T,
                   "train_sequences": int(tr[0].shape[0]), "val_sequences": int(va[0].shape[0]),
                   "vocab": len(texts), "seeds": list(seeds)},
        "codec_equivalence_check": equiv,
        "gradient_check": {k: v["max_rel_error"] for k, v in gc.items()},
        "e4_three_heads": e4_three_heads(texts, tr, va, L=L, d=d, steps=steps, seeds=seeds),
        "e5_gradient_from_zero": e5_gradient_from_zero(texts, tr, va, L=L, d=d,
                                                       steps=max(60, steps // 2)),
        "e6_tying_cost": e6_tying_cost(texts, tr, va, L=L, d=d, steps=steps),
        "e3_recheck_with_trained_W": {
            # d=96 is the demo width; E3's sweep put 50 percent recovery near d_model=128, so the
            # demo width sits below that threshold and its absolute accuracy is low by
            # construction. The controlled quantity is the before/after change at a fixed width,
            # and a second, wider run is included so the conclusion does not rest on one point
            # sitting below threshold.
            str(d): e3_recheck_with_trained_W(texts, tr, va, L=L, d=d, steps=steps),
            "256": e3_recheck_with_trained_W(texts, tr, va, L=L, d=256, steps=steps),
        },
    }
    result["wall_seconds"] = time.time() - t0
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        provenance.stamp(result, __file__)
        json.dump(result, fh, indent=2, sort_keys=True, ensure_ascii=False)
    return result


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.join(here, "..", "..", "..", "assignment-6", "frozen")
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    r = main(os.path.join(root, "corpus"), os.path.join(root, "tokenizer.json"),
             os.path.join(here, "..", "artifacts", "train.json"), steps=steps)
    print(json.dumps({k: v for k, v in r.items()
                      if k in ("config", "gradient_check", "wall_seconds")}, indent=2))
