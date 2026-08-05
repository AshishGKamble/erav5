"""
Fix the reasoning lane's validation leak (README section 9.2).

The bug: `prepare_data.py`'s `gsm8k_iter` looped the GSM8K source 6 times before
`tokenize_lanes.py` split off the first 5% of TOKENS as validation. Because the content
repeats, every validation window also appears in training - the "held-out" loss was
measuring memorisation, and a 0.468 reasoning gain we had proposed acting on was an
artefact of it.

The fix: split by DOCUMENT, not by token offset. Hold out 5% of the unique GSM8K
examples, and build the training bin from the other 95% only. The two sets are then
disjoint by construction, whatever repetition happens afterwards.

This works entirely offline: reasoning.txt contains the source repeated exactly 6x, so
the unique examples can be recovered by splitting on the "Question: " delimiter and
de-duplicating. Reuses the tokenizer from tokenize_lanes.py so the new bins stay
comparable with every other lane.

Run: python3 proxy/fix_reasoning_split.py
"""
import os, json, numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
VAL_FRAC = 0.05
SEED = 1337
DELIM = "Question: "


def main():
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(DATA, "tokenizer.json"))

    text = open(os.path.join(DATA, "reasoning.txt"), encoding="utf-8").read()
    docs = [DELIM + d for d in text.split(DELIM) if d.strip()]
    uniq = list(dict.fromkeys(docs))          # order-preserving de-duplication
    print(f"documents {len(docs):,} -> unique {len(uniq):,} "
          f"(source was repeated {len(docs)/max(1,len(uniq)):.1f}x)")

    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(uniq))
    n_val = int(len(uniq) * VAL_FRAC)
    val_docs = [uniq[i] for i in idx[:n_val]]
    train_docs = [uniq[i] for i in idx[n_val:]]
    assert not (set(val_docs) & set(train_docs)), "document split is not disjoint"
    print(f"split by document: {len(train_docs):,} train / {len(val_docs):,} val, disjoint")

    def encode(rows):
        ids = []
        for i in range(0, len(rows), 2000):
            for enc in tok.encode_batch(rows[i:i + 2000]):
                ids.extend(enc.ids)
        return np.array(ids, dtype=np.uint16)

    tr, va = encode(train_docs), encode(val_docs)
    tr.tofile(os.path.join(DATA, "reasoning_train.bin"))
    va.tofile(os.path.join(DATA, "reasoning_val.bin"))

    meta_path = os.path.join(DATA, "meta.json")
    meta = json.load(open(meta_path))
    meta["lanes"]["reasoning"] = {"train_tokens": int(len(tr)), "val_tokens": int(len(va)),
                                  "split": "by document (disjoint), 5% held out"}
    json.dump(meta, open(meta_path, "w"), indent=2)
    print(f"wrote reasoning_train.bin ({len(tr):,} tokens) and reasoning_val.bin ({len(va):,} tokens)")

    # verify: the whole point of this script is that this number is now zero
    trb = np.asarray(tr).tobytes()
    rng2 = np.random.default_rng(0); hits = 0; N, W = 200, 64
    for _ in range(N):
        i = rng2.integers(0, len(va) - W)
        if np.asarray(va[i:i + W]).tobytes() in trb:
            hits += 1
    print(f"\nLEAKAGE CHECK: {hits}/{N} validation windows found in training "
          f"({100*hits/N:.0f}%){'  <- FIXED' if hits == 0 else '  <- STILL LEAKING'}")


if __name__ == "__main__":
    main()
