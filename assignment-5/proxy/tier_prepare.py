"""
Tier experiment, step 1 - split the Indic lane into REAL quality tiers.

The whole study so far has a hole in it: the proxy has one uniformly clean Indic bin, so
"more Indic" is free in a way it never is in reality. The ledger says organic Indic runs
out at ~110B, so raising Indic from 18% to 30% means the marginal tokens are translated or
synthetic. The proxy priced the benefit of more Indic and was blind to that dilution.

We do not have to simulate the dilution: the Assignment-4 corpus carries provenance, and
its sources map onto the ledger's own tiers.

  anudesh                -> NATIVE Indic prompts (the 'verified' tier)         -> indic_hi
  dolly, hhrlhf          -> translated from English                            -> indic_lo
  toxicmatrix            -> synthetic                                          -> indic_lo

Reuses the tokenizer already trained in tokenize_lanes.py, so the new bins share a vocab
with every existing run and remain directly comparable.

Run: python3 proxy/tier_prepare.py  ->  data/indic_hi_*.bin, data/indic_lo_*.bin
"""
import os, json, numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
A4 = os.path.join(HERE, "..", "..", "assignment-4", "data", "cleaned", "corpus.jsonl")
TOKJSON = os.path.join(DATA, "tokenizer.json")
VAL_FRAC = 0.05

HI_SOURCES = {"anudesh"}                              # native Indic - the scarce, real tier
LO_SOURCES = {"dolly", "hhrlhf", "toxicmatrix"}       # translated + synthetic


def main():
    if not os.path.exists(A4):
        print("!! A4 corpus not found:", A4); return
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(TOKJSON)

    buckets = {"indic_hi": [], "indic_lo": []}
    counts = {"indic_hi": 0, "indic_lo": 0, "skipped": 0}
    for line in open(A4, encoding="utf-8"):
        try:
            d = json.loads(line)
        except Exception:
            continue
        src, text = d.get("source"), d.get("text", "")
        if not text:
            continue
        if src in HI_SOURCES:
            buckets["indic_hi"].append(text); counts["indic_hi"] += 1
        elif src in LO_SOURCES:
            buckets["indic_lo"].append(text); counts["indic_lo"] += 1
        else:
            counts["skipped"] += 1

    meta = json.load(open(os.path.join(DATA, "meta.json")))
    for lane, rows in buckets.items():
        # Encode in batches. Calling tok.encode() on one 64MB string needs >10GB and thrashes;
        # encode_batch over chunks is both faster and flat in memory.
        ids = []
        CH = 2000
        for i in range(0, len(rows), CH):
            for enc in tok.encode_batch(rows[i:i + CH]):
                ids.extend(enc.ids)
            if (i // CH) % 10 == 0:
                print(f"    {lane}: {min(i+CH,len(rows)):,}/{len(rows):,} docs -> {len(ids):,} tokens", flush=True)
        nchars = sum(len(r) for r in rows)
        arr = np.array(ids, dtype=np.uint16)
        n_val = int(len(arr) * VAL_FRAC)
        val, train = arr[:n_val], arr[n_val:]
        train.tofile(os.path.join(DATA, f"{lane}_train.bin"))
        val.tofile(os.path.join(DATA, f"{lane}_val.bin"))
        meta["lanes"][lane] = {"train_tokens": int(len(train)), "val_tokens": int(len(val))}
        print(f"  {lane:9s} docs {len(rows):>7,}  chars {nchars:>12,}  "
              f"tokens train {len(train):>9,} val {len(val):>7,}")

    json.dump(meta, open(os.path.join(DATA, "meta.json"), "w"), indent=2)
    hi, lo = meta["lanes"]["indic_hi"]["train_tokens"], meta["lanes"]["indic_lo"]["train_tokens"]
    print(f"\nrows: {counts}")
    print(f"native share of the Indic corpus: {100*hi/(hi+lo):.1f}%  "
          f"(the real inventory's organic share at 18% Indic is ~61%, at 30% it is ~37%)")
    print("\nNOTE: indic_hi is small, so any mixture drawing heavily on it will repeat it. That "
          "mirrors the real scarcity of verified Indic - but it also means a low indic_hi loss "
          "may partly be memorisation. Reported as a limitation, not hidden.")


if __name__ == "__main__":
    main()
