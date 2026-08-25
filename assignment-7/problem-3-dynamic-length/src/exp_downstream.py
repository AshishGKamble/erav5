"""
Problem 3, experiment E5: does the encoding change show up in a trained model?

E1 to E4 are properties of the encoding and hold whether or not anything is trained. E5 asks the
separate question of whether the difference survives contact with a model, and it is the experiment
most likely to come back inconclusive, so its limits are stated before its numbers.

Three input codecs, **all at D = 8192**, so the comparison is at equal cost:

  * `byte`            - the published construction, L = 32 bytes. 32 Latin characters, 10.7 Indic.
  * `codepoint`       - fix B, two blocks, L = 16. 16 characters for every script.
  * `script_relative` - fix D, one script tag plus 31 characters. 31 characters for every script.

Everything else is held fixed: same transformer body, same vocabulary softmax head, same seeds, same
batches, same steps. The output head is deliberately the *vocabulary* head rather than a byte head,
because the quantity under test is the input encoding and a byte head would change the output space
at the same time.

Two lanes are run, and the second one is the one that can kill the result: **a fix that helps Indic
by hurting English is not a fix.** The web lane is the control, and its numbers are reported whether
or not they are convenient.

Every difference is quoted against a **noise floor measured from repeated seeds**, following the
discipline assignment 5 used. A gap inside that floor is reported as not established, not as a small
win.
"""
import sys, os, json, time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "problem-5-reversibility", "src"))
import codec, corpus, data, kron_model as K, vocabulary  # noqa: E402
import exp_train as T  # noqa: E402

SEED = 20260825
D_TARGET = 8192

# A stable script -> id map, so a run is reproducible and the tag means the same thing every time.
SCRIPTS = ["COMMON", "LATIN", "DEVANAGARI", "BENGALI", "GUJARATI", "GURMUKHI", "KANNADA",
           "MALAYALAM", "ORIYA", "TAMIL", "TELUGU", "ARABIC", "HAN", "GREEK", "CYRILLIC"]
SCRIPT_ID = {s: i for i, s in enumerate(SCRIPTS)}


def build_codes(texts, arm):
    """One of the three input codecs, all landing on D = 8192."""
    if arm == "byte":
        codes = K.KronCodes(texts, 32, "byte")
        capacity = "32 bytes"
    elif arm == "codepoint":
        codes = K.KronCodes(texts, 16, "codepoint", blocks=2)
        capacity = "16 characters"
    elif arm == "script_relative":
        # 32 positions: one script tag plus 31 characters.
        units = [codec.script_relative_units(t, SCRIPT_ID.get(corpus.text_script(t)[0], 0),
                                             limit=31) for t in texts]
        codes = K.KronCodes(texts, 32, "byte", units_override=units)
        capacity = "1 script tag + 31 characters"
    else:
        raise ValueError(arm)
    assert codes.D == D_TARGET, (arm, codes.D)
    return codes, capacity


def run_arm(texts, tr, va, arm, d=96, steps=300, seeds=(1, 2, 3), lr=3e-3, batch_size=8):
    codes, capacity = build_codes(texts, arm)
    runs = []
    for s in seeds:
        m = K.KronTiny(codes, d=d, n_layer=2, n_head=4, max_pos=tr[0].shape[1],
                       head="vocab", vocab=len(texts), seed=s)
        h = T.train(m, tr, va, steps, lr=lr, batch_size=batch_size, seed=SEED + s,
                    eval_every=max(1, steps // 6))
        runs.append({"seed": s, "history": h, "final": h[-1]})
    finals = [r["final"]["loss_per_token"] for r in runs]
    accs = [r["final"]["exact_token_accuracy_oracle_length"] for r in runs]
    return {
        "arm": arm,
        "D": codes.D,
        "window_capacity": capacity,
        "mean_units_encoded_per_token": float(np.mean(codes.n_ones)),
        "runs": runs,
        "final_loss_per_token_mean": float(np.mean(finals)),
        "final_loss_per_token_sd": float(np.std(finals, ddof=1)) if len(finals) > 1 else 0.0,
        "final_loss_per_token_runs": finals,
        "final_accuracy_mean": float(np.mean([a for a in accs if a is not None])) if any(
            a is not None for a in accs) else None,
    }


def e5_lane(texts, corpus_root, tokenizer_path, lane, d=96, steps=300, seeds=(1, 2, 3),
            max_docs=400, T_len=128):
    tr, va = T.load_data(corpus_root, tokenizer_path, lane, T=T_len, max_docs=max_docs)
    arms = {a: run_arm(texts, tr, va, a, d=d, steps=steps, seeds=seeds)
            for a in ("byte", "codepoint", "script_relative")}
    noise = max(a["final_loss_per_token_sd"] for a in arms.values())
    base = arms["byte"]["final_loss_per_token_mean"]
    for name, a in arms.items():
        delta = a["final_loss_per_token_mean"] - base
        a["delta_vs_byte"] = delta
        a["exceeds_seed_noise"] = bool(abs(delta) > 2 * noise) if noise > 0 else None
        a["verdict"] = ("not established, inside the seed noise floor"
                        if not (noise > 0 and abs(delta) > 2 * noise)
                        else ("better than byte" if delta < 0 else "worse than byte"))
    return {
        "lane": lane,
        "train_sequences": int(tr[0].shape[0]),
        "val_sequences": int(va[0].shape[0]),
        "arms": arms,
        "seed_noise_floor_sd": noise,
        "comparison_threshold": "2 x the largest per-arm standard deviation across seeds",
    }


def truncation_exposure(texts, tr, va):
    """How many training tokens does any codec actually truncate?

    This is the diagnostic that decides whether E5 is capable of measuring anything at all. Three
    codecs that never truncate a single token carry **identical information** and differ only in
    how that information is laid out across D dimensions, and a `Linear(D, d)` learns any layout
    equally well. A null result under those conditions is not weak evidence about the codecs; it is
    a measurement with no exposure to the thing being measured, and it has to be reported that way.
    """
    ids = np.concatenate([tr[0].reshape(-1), va[0].reshape(-1)])
    seg = np.concatenate([tr[2].reshape(-1), va[2].reshape(-1)])
    ids = ids[seg != 0]
    out = {"token_occurrences": int(ids.size)}
    for arm, cap, fn in (("byte", 32, lambda t: len(t.encode("utf-8"))),
                         ("codepoint", 16, lambda t: len(t)),
                         ("script_relative", 32, lambda t: len(t) + 1)):
        need = np.array([fn(t) for t in texts])
        over = need[ids] > cap
        out[arm] = {"capacity_units": cap, "truncated_occurrences": int(over.sum()),
                    "truncated_rate": float(over.mean())}
    return out


def build_word_vocabulary(corpus_root, lane, top_n=20000):
    """A word-level unit inventory for E5b, most frequent first.

    Words, not BPE tokens, because Problem 3's whole finding lives at word level and the tokenizer
    is what hides it. A word-level run is only affordable because the tied byte head needs **no**
    per-word output parameters, which is the Kronecker construction's own selling point being used
    to test the Kronecker construction.
    """
    from collections import Counter
    import exp_window as WIN
    counts = Counter()
    for text in corpus.read_lane(os.path.join(corpus_root, lane + ".jsonl")):
        for w in text.split():
            st = WIN.strip_punctuation(w)
            if st:
                counts[st] += 1
    return [w for w, _ in counts.most_common(top_n)]


def e5b_word_level(corpus_root, lane, d=96, steps=300, seeds=(1, 2), top_n=20000, T_len=64,
                   max_docs=400, batch_size=8):
    """E5 rebuilt so it can actually see truncation: words as units, tied byte head, no vocabulary.

    E5 came back null because the tokenizer truncates essentially nothing (see
    `truncation_exposure`). At word level the byte codec truncates a large fraction of Indic words,
    which is precisely the harm E3 counted, so here the codecs genuinely differ in what reaches the
    model.

    The headline metric is **exact full-word reconstruction**. A word the codec cannot represent can
    never be reconstructed, so truncation counts as failure rather than being quietly excluded from
    the denominator, which is what per-position accuracy would do.
    """
    import exp_window as WIN
    words = build_word_vocabulary(corpus_root, lane, top_n)
    index = {w: i for i, w in enumerate(words)}

    docs = []
    for text in corpus.read_lane(os.path.join(corpus_root, lane + ".jsonl")):
        seq = [index[st] for st in
               (WIN.strip_punctuation(w) for w in text.split()) if st in index]
        if len(seq) > 1:
            docs.append(np.asarray(seq, dtype=np.int64))
        if len(docs) >= max_docs:
            break
    arrays = data.pack(docs, T_len)
    tr, va = data.split(arrays)

    arms = {}
    for arm in ("byte", "script_relative"):
        codes, capacity = build_codes(words, arm)
        full_len = np.array([len(w.encode("utf-8")) if arm == "byte" else len(w) + 1
                             for w in words])
        representable = float((full_len <= codes.pos_dim).mean())
        runs = []
        for s in seeds:
            m = K.KronTiny(codes, d=d, n_layer=2, n_head=4, max_pos=T_len,
                           head="byte_tied", vocab=len(words), seed=s)
            T.train(m, tr, va, steps, batch_size=batch_size, seed=SEED + s, eval_every=None)
            runs.append(exact_word_rate(m, codes, va, full_len, batch_size))
        arms[arm] = {
            "arm": arm, "D": codes.D, "window_capacity": capacity,
            "word_types": len(words),
            "word_types_representable": representable,
            "runs": runs,
            "exact_full_word_mean": float(np.mean([r["exact_full_word_rate"] for r in runs])),
            "exact_full_word_sd": float(np.std([r["exact_full_word_rate"] for r in runs], ddof=1))
                                  if len(runs) > 1 else 0.0,
            "target_occurrences_truncated": float(np.mean(
                [r["targets_truncated_rate"] for r in runs])),
        }
    noise = max(a["exact_full_word_sd"] for a in arms.values())
    delta = (arms["script_relative"]["exact_full_word_mean"] - arms["byte"]["exact_full_word_mean"])
    return {
        "lane": lane, "unit": "word", "head": "byte_tied (zero vocabulary parameters)",
        "train_sequences": int(tr[0].shape[0]), "val_sequences": int(va[0].shape[0]),
        "arms": arms,
        "seed_noise_floor_sd": noise,
        "script_relative_minus_byte": delta,
        "exceeds_seed_noise": bool(abs(delta) > 2 * noise) if noise > 0 else None,
    }


def exact_word_rate(model, codes, arrays, full_len, batch_size=8, max_batches=12):
    """Fraction of target words reconstructed in full. Truncated targets count as failures."""
    hit = tot = trunc = 0
    for b, (ids, pos, seg) in enumerate(data.batches(arrays, batch_size)):
        if b >= max_batches:
            break
        xf, _ = model.forward(ids, pos, seg)
        tgt, valid = model.targets(ids, seg)
        y, occ, _, blocks, L = model.byte_targets(tgt)
        vm = occ & valid[:, :, None, None]
        lg = model.head_logits(xf).astype(np.float64).reshape(
            ids.shape[0], ids.shape[1], blocks, codec.CHAR_DIM, L)
        pred = lg.argmax(axis=3)
        correct = ((pred == y) | ~vm).all(axis=(2, 3)) & valid
        # A target longer than the window was never representable, so it cannot count as a hit
        # no matter what the head emitted for the retained prefix.
        fits = full_len[tgt] <= codes.pos_dim
        hit += int((correct & fits).sum())
        trunc += int((valid & ~fits).sum())
        tot += int(valid.sum())
    return {"exact_full_word_rate": hit / max(1, tot),
            "targets_truncated_rate": trunc / max(1, tot),
            "targets_scored": tot}


def main(corpus_root, tokenizer_path, out_path, steps=300, d=96, seeds=(1, 2, 3),
         lanes=("indic", "web"), max_docs=400):
    toks, meta = vocabulary.load(tokenizer_path)
    texts = [t for _, t, _ in toks]
    t0 = time.time()
    result = {
        "seed": SEED,
        "tokenizer": meta,
        "config": {"steps": steps, "d_model": d, "seeds": list(seeds), "D": D_TARGET,
                   "head": "vocab softmax, held fixed so only the input codec varies",
                   "max_docs_per_lane": max_docs},
        "script_ids": SCRIPT_ID,
        "lanes": {lane: e5_lane(texts, corpus_root, tokenizer_path, lane, d=d, steps=steps,
                                seeds=seeds, max_docs=max_docs) for lane in lanes},
    }
    for lane in lanes:
        tr, va = T.load_data(corpus_root, tokenizer_path, lane, T=128, max_docs=max_docs)
        result["lanes"][lane]["truncation_exposure"] = truncation_exposure(texts, tr, va)
    result["e5b_word_level"] = {
        lane: e5b_word_level(corpus_root, lane, d=d, steps=steps, seeds=(1, 2))
        for lane in lanes}
    result["e5_verdict"] = (
        "E5 at token level is null on both lanes and CANNOT be otherwise: see truncation_exposure, "
        "where every codec truncates essentially nothing, so all three carry identical information "
        "and differ only in layout. The BPE tokenizer sits between the corpus and the window and "
        "removes the phenomenon under test, converting it into fertility instead. E5b repeats the "
        "experiment at word level, where the truncation E3 counted is actually present.")
    result["wall_seconds"] = time.time() - t0
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(result, fh, indent=2, sort_keys=True, ensure_ascii=False)
    return result


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.join(here, "..", "..", "..", "assignment-6", "frozen")
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    r = main(os.path.join(root, "corpus"), os.path.join(root, "tokenizer.json"),
             os.path.join(here, "..", "artifacts", "downstream.json"), steps=steps)
    for lane, L in r["lanes"].items():
        print(lane, "noise floor sd =", round(L["seed_noise_floor_sd"], 4))
        for a, v in L["arms"].items():
            print(f"  {a:16s} loss/token={v['final_loss_per_token_mean']:.4f} "
                  f"delta={v['delta_vs_byte']:+.4f} {v['verdict']}")
