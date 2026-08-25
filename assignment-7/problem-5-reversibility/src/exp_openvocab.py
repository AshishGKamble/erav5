"""
Problem 5, experiment E7: the brief's third promise, tested in the form that can settle it.

The brief claims three payoffs, and this file is about the third:

> "If we can do this, then we can get rid of the final head as well! **Then we can have a vocab of
>  1M as well without any issues!**"

E6 looked like a refutation: a trained tied head emitted 12.2 percent invalid UTF-8 and its
out-of-vocabulary strings were degenerate (`तत`, `ततत`, unassigned codepoints). But E6 could not
actually test the claim, for the same structural reason Problem 3's E5 could not test its own: every
target it scored was already **inside** the vocabulary, so "can this address words it was never
given" was never put to the model.

The claim splits into two questions that E6 ran together, and they have different answers:

  1. **Can the head represent a word it has no id for?** This is architectural. A vocabulary softmax
     cannot, at any amount of training, because no output row exists for that word. A tied byte head
     can, because it emits bytes. This is provable rather than measurable, and it is the part of the
     claim that is true.

  2. **Does it emit the right one?** This is empirical, and it is what E6 measured badly.

So the test here keeps a deliberately small inventory, lets words outside it appear as **targets**
while entering as an unknown marker, and asks the head to produce them. A vocabulary head scores
exactly zero on those targets by construction; the number for the byte head is what this experiment
is for.
"""
import sys, os, json
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "common"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "problem-3-dynamic-length", "src"))
sys.path.insert(0, HERE)
import codec, corpus, data, kron_model as K  # noqa: E402
import exp_train as T  # noqa: E402
import exp_window as WIN  # noqa: E402

SEED = 20260825
UNK = 0


def build_streams(corpus_root, lane, inventory_size=8000, max_docs=1200, T_len=96,
                  max_word_units=32):
    """Word streams where the target vocabulary is deliberately larger than the input vocabulary.

    `words[0]` is the unknown marker. `words[1:inventory_size+1]` are the words the model has ids
    for and can be fed. Everything after that appears only ever as a **target**, so the model is
    asked to emit words it was never given an input embedding for.
    """
    counts = Counter()
    docs_raw = []
    for text in corpus.read_lane(os.path.join(corpus_root, lane + ".jsonl")):
        toks = [w for w in (WIN.strip_punctuation(x) for x in text.split()) if w]
        if len(toks) > 1:
            docs_raw.append(toks)
            counts.update(toks)
        if len(docs_raw) >= max_docs:
            break

    ranked = [w for w, _ in counts.most_common()]
    # Only words the codec can actually hold are eligible as targets; a word longer than the window
    # is a Problem 3 truncation, not an open-vocabulary question, and mixing them would confound.
    ranked = [w for w in ranked if len(w.encode("utf-8")) <= max_word_units]
    inventory = ranked[:inventory_size]
    outside = ranked[inventory_size:]
    words = ["\x00UNK"] + inventory + outside
    tid = {w: i + 1 for i, w in enumerate(inventory + outside)}
    n_inv = len(inventory)

    docs_t, docs_i = [], []
    for toks in docs_raw:
        t = [tid[w] for w in toks if w in tid]
        if len(t) > 1:
            docs_t.append(np.asarray(t, dtype=np.int64))
            docs_i.append(np.asarray([x if x <= n_inv else UNK for x in t], dtype=np.int64))

    tgt_arrays = data.pack(docs_t, T_len)
    in_arrays = data.pack(docs_i, T_len)
    return words, n_inv, tgt_arrays, in_arrays, set(counts)


def train_open(model, tgt_tr, in_tr, steps, lr=3e-3, batch_size=8, seed=SEED):
    rng = np.random.default_rng(seed)
    opt = K.Adam(model.p, lr=lr)
    n = tgt_tr[0].shape[0]
    for step in range(steps):
        sel = rng.integers(0, n, size=batch_size)
        _, g, _ = model.loss_and_grad(in_tr[0][sel], in_tr[1][sel], in_tr[2][sel],
                                      target_ids=tgt_tr[0][sel])
        opt.step(model.p, g)
    return model


def evaluate_open(model, codes, words, n_inv, tgt_va, in_va, corpus_words, batch_size=8,
                  max_batches=25, max_examples=25):
    """Reconstruction split by whether the target was inside the input vocabulary."""
    # Frequency bands. Words are ranked by frequency, so a word id IS its rank, and the band
    # boundaries straddle the inventory cut. This is the control that decides what a zero means.
    #
    # Targets outside the inventory are, by construction, also the RARER targets. A zero on them
    # could therefore mean "cannot emit a word it has no id for" or merely "cannot predict rare
    # words at all", and those call for opposite conclusions. The `in_tail` band holds the least
    # frequent words that ARE in the inventory, so it is matched on rarity and differs only in
    # vocabulary membership. If `in_tail` also scores near zero, this experiment cannot separate
    # the two and must say so instead of claiming a refutation.
    def band_of(w_id):
        if w_id <= n_inv * 0.25:
            return "in_head"
        if w_id <= n_inv * 0.75:
            return "in_mid"
        if w_id <= n_inv:
            return "in_tail"
        if w_id <= n_inv + (n_inv // 2):
            return "outside_near"
        return "outside_far"

    bands = {k: [0, 0] for k in ("in_head", "in_mid", "in_tail", "outside_near", "outside_far")}
    stats = {"in_inventory": [0, 0], "outside_inventory": [0, 0]}
    emitted = Counter()
    invalid = valid = 0
    hits_outside = Counter()
    n = tgt_va[0].shape[0]
    for b in range(0, min(n, max_batches * batch_size), batch_size):
        sl = slice(b, b + batch_size)
        ids, pos, seg = in_va[0][sl], in_va[1][sl], in_va[2][sl]
        if ids.shape[0] == 0:
            break
        xf, _ = model.forward(ids, pos, seg)
        tgt, valid_mask = model.targets(ids, seg, target_ids=tgt_va[0][sl])
        y, occ, _, blocks, L = model.byte_targets(tgt)
        vm = occ & valid_mask[:, :, None, None]
        lg = model.head_logits(xf).astype(np.float64).reshape(
            ids.shape[0], ids.shape[1], blocks, codec.CHAR_DIM, L)
        pred = lg.argmax(axis=3)
        correct = ((pred == y) | ~vm).all(axis=(2, 3)) & valid_mask
        for bi in range(ids.shape[0]):
            for t in range(ids.shape[1]):
                if not valid_mask[bi, t]:
                    continue
                w_id = int(tgt[bi, t])
                key = "in_inventory" if w_id <= n_inv else "outside_inventory"
                stats[key][1] += 1
                stats[key][0] += bool(correct[bi, t])
                bnd = band_of(w_id)
                bands[bnd][1] += 1
                bands[bnd][0] += bool(correct[bi, t])
                length = int(vm[bi, t, 0].sum())
                if length:
                    units = [int(pred[bi, t, 0, p]) for p in range(length)]
                    text, ok = codec.units_to_text(units, "byte")
                    if ok:
                        valid += 1
                        emitted[text] += 1
                        if key == "outside_inventory" and correct[bi, t]:
                            hits_outside[words[w_id]] += 1
                    else:
                        invalid += 1
    out = {}
    for key, (hit, tot) in stats.items():
        out[key] = {"targets": tot, "exact_reconstructions": hit,
                    "exact_rate": hit / tot if tot else None}
    emitted_outside = sum(n for w, n in emitted.items() if w not in set(words[1:n_inv + 1]))
    real_words = sum(n for w, n in emitted.items()
                     if w not in set(words[1:n_inv + 1]) and w in corpus_words)
    out["emissions"] = {
        "valid_utf8": valid, "invalid_utf8": invalid,
        "invalid_utf8_rate": invalid / max(1, valid + invalid),
        "emitted_outside_inventory": emitted_outside,
        "emitted_outside_inventory_that_are_real_words": real_words,
        "real_word_share_of_outside_emissions": real_words / max(1, emitted_outside),
    }
    out["frequency_bands"] = {
        k: {"targets": tot, "exact": hit, "exact_rate": (hit / tot) if tot else None}
        for k, (hit, tot) in bands.items()}
    tail = out["frequency_bands"]["in_tail"]
    out["control_reading"] = (
        "in_tail is the rarity-matched control: the least frequent words that ARE in the "
        "inventory. If it scores near zero too, a zero on the outside bands says nothing about "
        "vocabulary membership and this experiment cannot test the claim."
        if not tail["exact_rate"] or tail["exact_rate"] < 0.005 else
        "in_tail scores materially above zero, so rarity alone does not explain a zero on the "
        "outside bands and the gap is attributable to vocabulary membership.")
    out["correctly_emitted_outside_inventory_examples"] = [
        w for w, _ in hits_outside.most_common(max_examples)]
    return out


def run_lane(corpus_root, lane, d=192, steps=800, inventory_size=8000, seeds=(1, 2),
             max_docs=1200, L=32):
    words, n_inv, tgt_arrays, in_arrays, corpus_words = build_streams(
        corpus_root, lane, inventory_size=inventory_size, max_docs=max_docs)
    tgt_tr, tgt_va = data.split(tgt_arrays)
    in_tr, in_va = data.split(in_arrays)
    codes = K.KronCodes(words, L, "byte")
    runs = []
    for s in seeds:
        m = K.KronTiny(codes, d=d, n_layer=2, n_head=4, max_pos=tgt_tr[0].shape[1],
                       head="byte_tied", vocab=len(words), seed=s)
        train_open(m, tgt_tr, in_tr, steps, seed=SEED + s)
        runs.append(evaluate_open(m, codes, words, n_inv, tgt_va, in_va, corpus_words))
    return {
        "lane": lane, "d_model": d, "steps": steps, "window": L,
        "input_vocabulary": n_inv,
        "target_vocabulary": len(words) - 1,
        "targets_outside_input_vocabulary": len(words) - 1 - n_inv,
        "train_sequences": int(tgt_tr[0].shape[0]),
        "runs": runs,
        "frequency_bands_mean": {
            k: float(np.mean([(r["frequency_bands"][k]["exact_rate"] or 0.0) for r in runs]))
            for k in ("in_head", "in_mid", "in_tail", "outside_near", "outside_far")},
        "outside_exact_rate_mean": float(np.mean(
            [r["outside_inventory"]["exact_rate"] or 0.0 for r in runs])),
        "in_inventory_exact_rate_mean": float(np.mean(
            [r["in_inventory"]["exact_rate"] or 0.0 for r in runs])),
        "vocab_head_outside_exact_rate": 0.0,
        "vocab_head_note": ("Exactly zero, and not by measurement: a vocabulary softmax has no "
                            "output row for a word outside its inventory, so it cannot emit one at "
                            "any amount of training. That asymmetry is the architectural half of "
                            "the brief's claim and it holds regardless of the numbers above."),
    }


def main(corpus_root, out_path, steps=800, d=192, lanes=("web", "indic"), seeds=(1,)):
    result = {"seed": SEED, "lanes": {}}
    for lane in lanes:
        result["lanes"][lane] = run_lane(corpus_root, lane, d=d, steps=steps, seeds=seeds)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(result, fh, indent=2, sort_keys=True, ensure_ascii=False)
    return result


if __name__ == "__main__":
    root = os.path.join(HERE, "..", "..", "..", "assignment-6", "frozen", "corpus")
    r = main(root, os.path.join(HERE, "..", "artifacts", "openvocab.json"))
    for lane, v in r["lanes"].items():
        print(f"--- {lane}: input vocab {v['input_vocabulary']:,}, "
              f"targets outside it {v['targets_outside_input_vocabulary']:,}")
        print(f"    in-inventory exact {v['in_inventory_exact_rate_mean']:.4f}   "
              f"OUTSIDE-inventory exact {v['outside_exact_rate_mean']:.4f}   "
              f"(vocab head: {v['vocab_head_outside_exact_rate']})")
        b = v["frequency_bands_mean"]
        print("    bands: " + "  ".join(f"{k}={b[k]:.4f}" for k in
                                        ("in_head", "in_mid", "in_tail",
                                         "outside_near", "outside_far")))
        print(f"    {v['runs'][0]['control_reading']}")
        print(f"    examples: {v['runs'][0]['correctly_emitted_outside_inventory_examples'][:12]}")
