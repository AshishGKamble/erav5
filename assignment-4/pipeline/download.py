"""
Assignment 4 - Stage 0: acquire a bounded slice of the corpus + the decontamination hold-out.

We DO NOT download the full 28 GB indic-align. We pull a few diverse *source* files
(each is multilingual) and sample rows later to ~40M tokens. Sources give us format
diversity (needed for the ghost-tag demo) and all 12 languages (needed for language-ID).

Hold-out sets (mmlu-indic, trivia-qa-indic-mcq) are BENCHMARKS: never trained on, only
scanned against in the decontamination stage. That is the whole point of keeping them separate.
"""
import os, sys
from huggingface_hub import hf_hub_download

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw")
HOLD = os.path.join(ROOT, "data", "holdout")
os.makedirs(RAW, exist_ok=True)
os.makedirs(HOLD, exist_ok=True)
os.environ.setdefault("HF_HOME", os.path.join(ROOT, ".hf_cache"))

CORPUS_REPO = "CharuAgarwal/indic-align"       # mirror of ai4bharat/indic-align (CC-BY-4.0)

# A small, diverse set of source files. Different sources = different formats (dolly=Alpaca-style
# translated instructions across 14 langs; anudesh=native conversation; toxic=safety alignment).
CORPUS_FILES = [
    "indicalign-instruct/anudesh/anudesh1.parquet",              # native prompts (en/hi heavy)
    "indicalign-instruct/dolly/Dolly.parquet",                   # Dolly translated -> our 12 langs
    "indicalign-toxic/hhrlhf/hh-rlhf.parquet",                   # safety: harmful/harmless pairs
    "indicalign-toxic/toxicmatrix/toxic_prompts_sarvam.parquet", # safety: toxic prompt matrix
]

# Our 12 Assignment-3 languages (ISO-639-1). Not all exist in every benchmark - we take what's there.
LANGS_12 = ["hi", "bn", "mr", "te", "ta", "gu", "ur", "kn", "or", "ml", "pa", "as"]

HOLDOUT = {
    "sarvamai/mmlu-indic":           ("test",       LANGS_12),
    "sarvamai/trivia-qa-indic-mcq":  ("validation", LANGS_12),
}


def grab(repo, path, dest, repo_type="dataset"):
    try:
        p = hf_hub_download(repo, path, repo_type=repo_type, local_dir=dest)
        print(f"  ok   {path}  ({os.path.getsize(p)/1e6:.1f} MB)")
        return p
    except Exception as e:
        print(f"  MISS {path}: {repr(e)[:110]}")
        return None


def main():
    print(f"== corpus slice from {CORPUS_REPO} ==")
    for f in CORPUS_FILES:
        grab(CORPUS_REPO, f, RAW)

    print("\n== decontamination hold-out (benchmarks - never trained on) ==")
    for repo, (split, langs) in HOLDOUT.items():
        got = 0
        for lang in langs:
            # discover the exact shard name for this lang/split
            from huggingface_hub import HfApi
            files = HfApi().list_repo_files(repo, repo_type="dataset")
            match = [x for x in files if x.startswith(f"{lang}/") and split in x and x.endswith(".parquet")]
            for m in match:
                sub = os.path.join(HOLD, repo.split("/")[-1])
                if grab(repo, m, sub):
                    got += 1
        print(f"  {repo}: {got} shards")

    print("\nDone. Raw ->", RAW, "| Hold-out ->", HOLD)


if __name__ == "__main__":
    main()
