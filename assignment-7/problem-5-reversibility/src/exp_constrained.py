"""
Problem 5, E8: constrained decoding, which fixes the one genuine defect E6 found.

E6 measured a tied byte head emitting **12.20% invalid UTF-8**, and called it the real cost of
predicting positions independently. That diagnosis is right and the conclusion drawn from it was
too pessimistic: the head is not wrong about the distribution, it is simply never asked for a
coherent sequence. Each position takes its argmax alone, so nothing prevents a lead byte where a
continuation byte was required.

`codec.decode_constrained` asks for a coherent sequence instead: at each position the bytes that
cannot legally follow what has already been emitted are masked out before the argmax, and any
incomplete trailing character is dropped. **No retraining, no architectural change, the same
logits.**

What it cannot do is invent competence. If the head's preferred bytes were wrong, forcing them to
be well formed makes them wrong and well formed, so exact match is measured alongside validity
rather than validity alone.
"""
import sys, os, json
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "common"))
sys.path.insert(0, HERE)
import codec, data, kron_model as K, vocabulary  # noqa: E402
import provenance  # noqa: E402
import exp_train as T  # noqa: E402

SEED = 20260825


def compare(model, codes, texts, arrays, batch_size=8, max_batches=40, max_examples=12):
    """Unconstrained against constrained decoding, on identical logits."""
    vocab = set(texts)
    stats = {m: Counter() for m in ("unconstrained", "constrained")}
    fixed_examples, n = [], 0
    for b, (ids, pos, seg) in enumerate(data.batches(arrays, batch_size)):
        if b >= max_batches:
            break
        xf, _ = model.forward(ids, pos, seg)
        tgt, valid = model.targets(ids, seg)
        y, occ, _, blocks, L = model.byte_targets(tgt)
        vm = occ & valid[:, :, None, None]
        flat = model.head_logits(xf).astype(np.float64)
        lg = flat.reshape(ids.shape[0], ids.shape[1], codec.CHAR_DIM, L)
        for bi in range(ids.shape[0]):
            for t in range(ids.shape[1]):
                if not valid[bi, t]:
                    continue
                length = int(vm[bi, t, 0].sum())
                if length == 0:
                    continue
                n += 1
                want = texts[int(tgt[bi, t])]
                col = lg[bi, t]
                plain = [int(col[:, p].argmax()) for p in range(length)]
                con = codec.decode_constrained(col[:, :length], length)
                for name, units in (("unconstrained", plain), ("constrained", con)):
                    text, ok = codec.units_to_text(units, "byte")
                    st = stats[name]
                    st["scored"] += 1
                    if not ok:
                        st["invalid_utf8"] += 1
                        continue
                    st["valid_utf8"] += 1
                    if text == "":
                        st["empty"] += 1
                    if text == want:
                        st["exact_match"] += 1
                    if text in vocab:
                        st["in_vocabulary"] += 1
                if not codec.units_to_text(plain, "byte")[1]:
                    t2, ok2 = codec.units_to_text(con, "byte")
                    if ok2 and len(fixed_examples) < max_examples:
                        fixed_examples.append({"repaired_to": t2, "target": want,
                                               "correct": t2 == want})
    out = {}
    for name, st in stats.items():
        s = max(1, st["scored"])
        out[name] = {"scored": st["scored"],
                     "invalid_utf8": st["invalid_utf8"],
                     "invalid_utf8_rate": st["invalid_utf8"] / s,
                     "valid_utf8_rate": st["valid_utf8"] / s,
                     "empty_output_rate": st["empty"] / s,
                     "in_vocabulary_rate": st["in_vocabulary"] / s,
                     "exact_match_rate": st["exact_match"] / s}
    out["repaired_examples"] = fixed_examples
    out["predictions"] = n
    return out


def main(corpus_root, tokenizer_path, out_path, steps=300, d=96, L=32, seed=1):
    toks, meta = vocabulary.load(tokenizer_path)
    texts = [t for _, t, _ in toks]
    tr, va = T.load_data(corpus_root, tokenizer_path, "indic", T=128, max_docs=400)
    codes = K.KronCodes(texts, L, "byte")
    m = K.KronTiny(codes, d=d, n_layer=2, n_head=4, max_pos=tr[0].shape[1],
                   head="byte_tied", vocab=len(texts), seed=seed)
    T.train(m, tr, va, steps, seed=SEED, eval_every=None)

    res = compare(m, codes, texts, va)
    u, c = res["unconstrained"], res["constrained"]
    res["improvement"] = {
        "invalid_utf8_removed": u["invalid_utf8_rate"] - c["invalid_utf8_rate"],
        "exact_match_change": c["exact_match_rate"] - u["exact_match_rate"],
        "in_vocabulary_change": c["in_vocabulary_rate"] - u["in_vocabulary_rate"],
    }
    res["config"] = {"steps": steps, "d_model": d, "window": L, "lane": "indic", "seed": seed}
    res["reading"] = ("Constrained decoding costs nothing to apply and cannot make the model "
                      "better at predicting, only better at being well formed. Whether the "
                      "repaired strings are also *correct* is the number that matters, and it is "
                      "reported next to the validity rate rather than instead of it.")
    with open(out_path, "w") as fh:
        provenance.stamp(res, __file__)
        json.dump(res, fh, indent=2, sort_keys=True, ensure_ascii=False)
    return res


if __name__ == "__main__":
    root = os.path.join(HERE, "..", "..", "..", "assignment-6", "frozen")
    r = main(os.path.join(root, "corpus"), os.path.join(root, "tokenizer.json"),
             os.path.join(HERE, "..", "artifacts", "constrained.json"))
    print(f"{'metric':22s} {'unconstrained':>14s} {'constrained':>12s}")
    for k in ("invalid_utf8_rate", "valid_utf8_rate", "empty_output_rate",
              "in_vocabulary_rate", "exact_match_rate"):
        print(f"{k:22s} {r['unconstrained'][k]:>13.2%} {r['constrained'][k]:>12.2%}")
    print("\nrepaired examples:", r["repaired_examples"][:6])
