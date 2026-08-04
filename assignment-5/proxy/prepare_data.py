"""
Proxy step 1 - assemble a tiny multi-lane corpus.

Five lanes that map to the plan: web (knowledge/common-sense), code, math, reasoning,
and indic. Indic reuses our Assignment-4 cleaned corpus; the rest are small streamed
slices from public datasets. Each lane is capped at ~CAP_MB of raw text so the whole
thing tokenizes and trains on a CPU. Agentic and long-context are intentionally NOT
proxied at this scale (their signal needs tool execution / 32K sequences) - stated in
the README.
"""
import os, json, itertools, sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data"); os.makedirs(OUT, exist_ok=True)
os.environ.setdefault("HF_HOME", os.path.join(HERE, "..", ".hf_cache"))
CAP_MB = float(os.environ.get("CAP_MB", "35"))        # raw chars per lane
CAP = int(CAP_MB * 1_000_000)
A4_CORPUS = os.path.join(HERE, "..", "..", "assignment-4", "data", "cleaned", "corpus.jsonl")


def write_capped(path, text_iter, cap=CAP):
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for t in text_iter:
            if not t:
                continue
            t = t.strip().replace("\r", "")
            if len(t) < 30:
                continue
            f.write(t + "\n\n")
            n += len(t) + 2
            if n >= cap:
                break
    print(f"  wrote {path}  ({n/1e6:.1f} MB)")
    return n


def stream(repo, split, field, cfg=None, cap=CAP):
    from datasets import load_dataset
    ds = load_dataset(repo, cfg, split=split, streaming=True)
    def gen():
        for ex in ds:
            v = ex.get(field) if isinstance(ex, dict) else None
            if v: yield v
    return gen()


def indic_iter():
    # reuse the Assignment-4 cleaned indic-align corpus
    if not os.path.exists(A4_CORPUS):
        print("  !! A4 corpus not found:", A4_CORPUS); return iter(())
    def gen():
        for line in open(A4_CORPUS, encoding="utf-8"):
            try: yield json.loads(line).get("text", "")
            except Exception: continue
    return gen()


def gsm8k_iter():
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="train")
    def gen():
        # loop the (small) set a few times, formatting Q + reasoning answer
        for _ in range(6):
            for ex in ds:
                yield f"Question: {ex['question']}\nAnswer: {ex['answer']}"
    return gen()


LANES = {
    "web":   lambda: stream("Salesforce/wikitext", "train", "text", cfg="wikitext-103-raw-v1"),
    "code":  lambda: stream("codeparrot/codeparrot-clean-valid", "train", "content"),
    "math":  lambda: stream("open-web-math/open-web-math", "train", "text"),
    "reasoning": gsm8k_iter,
    "indic": indic_iter,
}
FALLBACK = {
    "web":   lambda: stream("wikitext", "train", "text", cfg="wikitext-103-raw-v1"),
    "code":  lambda: stream("bigcode/the-stack-smol", "train", "content", cfg="data/python"),
    "math":  lambda: stream("EleutherAI/proof-pile-2", "train", "text", cfg="default"),
}


def main():
    only = sys.argv[1:] or list(LANES)
    for lane in only:
        path = os.path.join(OUT, f"{lane}.txt")
        if os.path.exists(path) and os.path.getsize(path) > CAP * 0.8:
            print(f"  skip {lane} (exists)"); continue
        print(f"[{lane}] gathering...")
        try:
            write_capped(path, LANES[lane]())
        except Exception as e:
            print(f"  primary failed for {lane}: {repr(e)[:120]}")
            if lane in FALLBACK:
                try: write_capped(path, FALLBACK[lane]())
                except Exception as e2: print(f"  fallback failed: {repr(e2)[:120]}")
    print("done. lanes:", [f for f in os.listdir(OUT) if f.endswith('.txt')])


if __name__ == "__main__":
    main()
