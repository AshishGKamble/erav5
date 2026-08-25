"""
E5b re-run at a scale that can actually resolve the effect, and merged back into the artifact.

The first E5b pass produced an indic delta of +0.0023 against a 2-sigma threshold of 0.0022, from
two seeds, with both arms reconstructing under one percent of words. That is not a result, it is a
coin landing on its edge, and quoting it would be exactly the kind of marginal claim this repository
is supposed to avoid. The cause was data volume: the indic lane yielded 210 training sequences
against the web lane's 5,789, because indic text has a far longer tail of rare words and a fixed
20k-word inventory keeps proportionally less of it.

So: more documents, a larger word inventory, longer sequences, and three seeds instead of two. If
the effect is real it should survive; if it does not survive, that is the reported answer.
"""
import sys, os, json

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "common"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "problem-5-reversibility", "src"))
sys.path.insert(0, HERE)
import exp_downstream as X  # noqa: E402


def main(corpus_root, artifact_path, steps=400, seeds=(1, 2, 3), top_n=40000,
         max_docs=2000, T_len=96):
    out = {}
    for lane in ("indic", "web"):
        out[lane] = X.e5b_word_level(corpus_root, lane, d=96, steps=steps, seeds=seeds,
                                     top_n=top_n, T_len=T_len, max_docs=max_docs)
        out[lane]["scale"] = {"steps": steps, "seeds": list(seeds), "word_inventory": top_n,
                              "max_docs": max_docs, "sequence_length": T_len}
    with open(artifact_path) as fh:
        art = json.load(fh)
    art["e5b_word_level_small"] = art.get("e5b_word_level")
    art["e5b_word_level"] = out
    art["e5b_note"] = ("The first, smaller pass is kept as `e5b_word_level_small`. It is superseded "
                       "rather than deleted, because its indic delta sat within a hair of its own "
                       "noise floor and the difference between the two passes is the point.")
    with open(artifact_path, "w") as fh:
        json.dump(art, fh, indent=2, sort_keys=True, ensure_ascii=False)
    return out


if __name__ == "__main__":
    root = os.path.join(HERE, "..", "..", "..", "assignment-6", "frozen", "corpus")
    r = main(root, os.path.join(HERE, "..", "artifacts", "downstream.json"))
    for lane, v in r.items():
        print(f"--- {lane}: {v['train_sequences']} train seqs")
        for a, arm in v["arms"].items():
            print(f"    {a:16s} repr={arm['word_types_representable']:.4f} "
                  f"exact={arm['exact_full_word_mean']:.4f} sd={arm['exact_full_word_sd']:.4f} "
                  f"trunc_targets={arm['target_occurrences_truncated']:.4f}")
        print(f"    delta={v['script_relative_minus_byte']:+.4f} noise_sd={v['seed_noise_floor_sd']:.4f} "
              f"exceeds={v['exceeds_seed_noise']}")
