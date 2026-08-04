"""
Proxy step 2 - train a small BPE tokenizer on the mixed corpus, then tokenize each
lane into train/val uint16 bins. Small vocab keeps the tiny model's embedding cheap.
"""
import os, glob, json, numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
VOCAB = 16000
VAL_FRAC = 0.05
TOKJSON = os.path.join(DATA, "tokenizer.json")


def train_tokenizer(files):
    from tokenizers import ByteLevelBPETokenizer
    tok = ByteLevelBPETokenizer()
    tok.train(files=files, vocab_size=VOCAB, min_frequency=2,
              special_tokens=["<|endoftext|>"])
    tok.save(TOKJSON)
    print(f"  trained tokenizer -> {TOKJSON}")
    return tok


def main():
    files = sorted(glob.glob(os.path.join(DATA, "*.txt")))
    lanes = [os.path.splitext(os.path.basename(f))[0] for f in files]
    print("lanes:", lanes)
    tok = train_tokenizer(files)

    meta = {"vocab_size": VOCAB, "lanes": {}}
    for f, lane in zip(files, lanes):
        text = open(f, encoding="utf-8").read()
        ids = tok.encode(text).ids
        arr = np.array(ids, dtype=np.uint16)
        n_val = int(len(arr) * VAL_FRAC)
        val, train = arr[:n_val], arr[n_val:]
        train.tofile(os.path.join(DATA, f"{lane}_train.bin"))
        val.tofile(os.path.join(DATA, f"{lane}_val.bin"))
        meta["lanes"][lane] = {"train_tokens": int(len(train)), "val_tokens": int(len(val))}
        print(f"  {lane:10s} tokens: train {len(train):>9,}  val {len(val):>7,}")
    json.dump(meta, open(os.path.join(DATA, "meta.json"), "w"), indent=2)
    total = sum(v["train_tokens"] for v in meta["lanes"].values())
    print(f"total train tokens: {total:,} ({total/1e6:.1f}M)")


if __name__ == "__main__":
    main()
