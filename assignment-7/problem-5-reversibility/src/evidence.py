"""
Regenerate every number quoted in the Problem 5 writeup, straight from the artefacts.

Nothing in README.md is typed by hand. This reads `artifacts/*.json` and emits
`artifacts/evidence.md`.

One conversion is applied here and stated wherever it is used, because without it two of the tables
would be meaningless: a vocabulary head's loss is nats per **token**, a byte head's is nats per
**byte position**, and a token spans several positions. Everything comparable is quoted per token.
"""
import json, os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "common"))
import provenance  # noqa: E402
ART = os.path.join(HERE, "..", "artifacts")


def load(name):
    with open(os.path.join(ART, name), encoding="utf-8") as fh:
        return json.load(fh)


def pct(x, n=2):
    return "n/a" if x is None else f"{x * 100:.{n}f}%"


def collect():
    """Every quoted number, as data. Mirrors `evidence.md` so the two cannot disagree."""
    k, t, r = load("codec.json"), load("train.json"), load("recheck.json")
    out = {
        "e1_roundtrip": {L: {"fitting": v["tokens_fitting"],
                             "recovered": v["tokens_fitting_recovered"],
                             "rate": v["tokens_fitting_rate"]}
                         for L, v in k["e1_roundtrip"].items()},
        "e2_margin": k["e2_noise"]["column_margin_after_znorm"],
        "e2_sweep": {str(row["sigma"]): row["exact_token_accuracy_oracle_length"]
                     for row in k["e2_noise"]["sweep"]},
        "e3_sparsity_k": k["e3_projection"]["occupied_columns_mean"],
        "e3_sweep": {str(row["d_model"]): row["minimum_norm_decode_accuracy"]
                     for row in k["e3_projection"]["sweep"]},
        "e4_heads": {h: {"loss_per_token": t["e4_three_heads"][h]["final_loss_per_token_mean"],
                         "sd": t["e4_three_heads"][h]["final_loss_per_token_sd"],
                         "exact": t["e4_three_heads"][h]["runs"][0]["final"]
                                  ["exact_token_accuracy_oracle_length"],
                         "params": t["e4_three_heads"][h]["params_measured"]["total"],
                         "head_params_at_paper_scale":
                             t["e4_three_heads"][h]["params_at_paper_scale"]["output_head"]}
                     for h in ("vocab", "byte_untied", "byte_tied")},
        "e4_noise_floor": t["e4_three_heads"]["_noise_floor_sd_across_seeds"],
        "e5_copy_at_init": t["e5_gradient_from_zero"]["_copy_at_init"],
        "e5_final": {o: t["e5_gradient_from_zero"][o]["byte_accuracy_final"]
                     for o in ("ce", "mse")},
        "e6_tying": {kk: t["e6_tying_cost"][kk] for kk in
                     ("invalid_utf8_rate", "in_vocabulary_rate",
                      "valid_utf8_out_of_vocabulary_rate", "exact_token_match_rate")},
        "caveat": {lab: {"duplicates": r[lab]["separation"]["exact_duplicates"],
                         "learned_inverse": r[lab]["learned_linear_inverse"]
                                            ["held_out_decode_accuracy"]}
                   for lab in ("random_init", "after_training")},
        "verification": {"codec_equivalence": t["codec_equivalence_check"],
                         "gradient_check": t["gradient_check"]},
    }
    try:
        o = load("openvocab.json")
        out["e7_bands"] = {lane: v["frequency_bands_mean"] for lane, v in o["lanes"].items()}
    except FileNotFoundError:
        pass
    try:
        c = load("constrained.json")
        out["e8_constrained"] = {"unconstrained": c["unconstrained"],
                                 "constrained": c["constrained"]}
    except FileNotFoundError:
        pass
    return out


def main():
    k, t, r = load("codec.json"), load("train.json"), load("recheck.json")
    try:
        o = load("openvocab.json")
    except FileNotFoundError:
        o = None
    out = []
    A = out.append

    A("# Problem 5, evidence\n")
    A("Every number below is read from `artifacts/*.json`. Regenerate with `python src/evidence.py`.\n")
    A(f"Vocabulary measured: {k['vocabulary_measured']:,} non-byte-fallback tokens from the "
      f"assignment-2 tokenizer. Seed {k['seed']}.\n")

    A("\n## E1, does the codec invert\n")
    A("| window | D | tokens fitting | recovered exactly | tokens overflowing |")
    A("|---|---|---|---|---|")
    for L in sorted(k["e1_roundtrip"], key=int):
        e = k["e1_roundtrip"][L]
        A(f"| L={L} | {e['D']:,} | {e['tokens_fitting']:,} | "
          f"{e['tokens_fitting_recovered']:,} ({pct(e['tokens_fitting_rate'])}) | "
          f"{e['tokens_overflowing']:,} |")
    A("\nOverflowing tokens recover their retained prefix exactly but not the token. Those bytes "
      "were never encoded, which is Problem 3's territory, not a codec defect.\n")

    A("\n## E2, does inversion survive approximate prediction\n")
    e2 = k["e2_noise"]
    A(f"Column margin after z-normalisation: **{e2['column_margin_after_znorm']:.2f}** against a "
      f"signal standard deviation of {e2['signal_sd']:.1f}. Measured over "
      f"{e2['tokens_measured']:,} tokens at L={e2['window']}.\n")
    A("| noise sigma (relative to signal) | exact token, oracle length | exact token, inferred length |")
    A("|---|---|---|")
    for row in e2["sweep"]:
        A(f"| {row['sigma']:.2f} | {pct(row['exact_token_accuracy_oracle_length'], 2)} | "
          f"{pct(row['exact_token_accuracy_inferred_length'], 2)} |")

    A("\n## E3, where invertibility actually dies\n")
    e3 = k["e3_projection"]
    A(f"A code is k-sparse with mean k = **{e3['occupied_columns_mean']:.2f}** occupied columns out "
      f"of D = {e3['D']:,}, which is {e3['sparsity_fraction_mean'] * 100:.3f}% dense. Recovering it "
      f"from a random projection is therefore compressed sensing, not magic.\n")
    A("| d_model | nullspace dimension | minimum-norm decode accuracy |")
    A("|---|---|---|")
    for row in e3["sweep"]:
        A(f"| {row['d_model']} | {row['nullspace_dimension']:,} | "
          f"{pct(row['minimum_norm_decode_accuracy'])} |")
    A(f"\nFifty percent recovery at d_model = {e3['d_model_for_50pc_recovery']}, "
      f"ninety-nine percent at {e3['d_model_for_99pc_recovery']}. "
      f"PLAN.md predicted near-chance decoding here; that prediction is **{e3['prediction_outcome']}**.\n")

    A("\n## The caveat E3 left open: does training destroy this\n")
    for dm, v in sorted(t["e3_recheck_with_trained_W"].items()):
        if not isinstance(v, dict) or "before" not in v:
            continue
        b, a = v["before"], v["after"]
        A(f"\n- **d_model={dm}**, minimum-norm decode {pct(b['minimum_norm_decode_accuracy'])} "
          f"before training, {pct(a['minimum_norm_decode_accuracy'])} after. Condition number "
          f"{b['condition_number']:.2f} to {a['condition_number']:.2f}.")
    A(f"\nThat looks like a refutation. It is not, because minimum norm is a **structure-blind** "
      f"decoder, correct for the random W of E3 and wrong for a trained one. Splitting the question "
      f"(`artifacts/recheck.json`, d_model={r['d_model']}):\n")
    A("| | exact duplicates | relative separation (min) | learned inverse, held-out tokens |")
    A("|---|---|---|---|")
    for label in ("random_init", "after_training"):
        s, li = r[label]["separation"], r[label]["learned_linear_inverse"]
        A(f"| {label.replace('_', ' ')} | {s['exact_duplicates']} | "
          f"{s['relative_separation_min']:.4f} | {pct(li['held_out_decode_accuracy'])} |")
    A(f"\nThe decoder is fitted on {r['fit_tokens']:,} tokens and scored on {r['probe_tokens']:,} it "
      f"never saw. Training breaks the decoder, not the encoding.\n")

    A("\n## E4, three output heads\n")
    e4 = t["e4_three_heads"]
    A(f"Indic lane, d_model={t['config']['d_model']}, L={t['config']['window']}, "
      f"{t['config']['steps']} steps, seeds {t['config']['seeds']}. "
      f"Seed noise floor sd **{e4['_noise_floor_sd_across_seeds']:.4f}** nats per token.\n")
    A("| head | loss per token | native loss | units/token | exact token | parameters |")
    A("|---|---|---|---|---|---|")
    for h in ("vocab", "byte_untied", "byte_tied"):
        v = e4[h]
        fin = v["runs"][0]["final"]
        A(f"| {h} | {v['final_loss_per_token_mean']:.4f} (sd {v['final_loss_per_token_sd']:.4f}) | "
          f"{fin['loss_native']:.4f} | {fin['scored_units_per_token']:.2f} | "
          f"{pct(fin['exact_token_accuracy_oracle_length'])} | "
          f"{v['params_measured']['total']:,} |")
    A("\nAt the paper's scale the parameter picture inverts, and this row is arithmetic, not "
      "measurement:\n")
    A("| head | output head parameters at d_model=768, vocab=131072 |")
    A("|---|---|")
    for h in ("vocab", "byte_untied", "byte_tied"):
        A(f"| {h} | {e4[h]['params_at_paper_scale']['output_head']:,} |")

    A("\n## E5, is there gradient at step 0\n")
    e5 = t["e5_gradient_from_zero"]
    ci = e5["_copy_at_init"]
    A(f"An untrained tied head reproduces the **current** token's bytes at "
      f"{pct(ci['byte_accuracy_vs_current_token'])} and the **next** token's at "
      f"{pct(ci['byte_accuracy_vs_next_token'])}, against a chance rate of "
      f"{pct(ci['chance'], 3)}. It is an autoencoder before it is trained, because `xf @ W.T` "
      f"reuses the same W that produced the embedding.\n")
    A("| objective | byte accuracy at step 0 | byte accuracy at end | loss per token, start to end |")
    A("|---|---|---|---|")
    for obj in ("ce", "mse"):
        v = e5[obj]
        A(f"| {obj.upper()} | {pct(v['byte_accuracy_at_step_0'])} | "
          f"{pct(v['byte_accuracy_final'])} | {v['loss_per_token_at_step_0']:.3f} to "
          f"{v['loss_per_token_final']:.3f} |")
    A("\nPLAN.md predicted MSE would not move. It moves. The objection is wrong for both "
      "objectives, and the reason is the autoencoder above, not the loss function.\n")

    A("\n## E6, what tying costs\n")
    e6 = t["e6_tying_cost"]
    A(f"Over {e6['predictions_scored']:,} predictions, decoded with the target's true length.\n")
    A("| outcome | rate |")
    A("|---|---|")
    for key, lab in (("invalid_utf8_rate", "invalid UTF-8"),
                     ("in_vocabulary_rate", "valid and in vocabulary"),
                     ("valid_utf8_out_of_vocabulary_rate", "valid but out of vocabulary"),
                     ("exact_token_match_rate", "exact match to the target token")):
        A(f"| {lab} | {pct(e6[key])} |")
    A(f"\nOut-of-vocabulary examples: {', '.join(e6['out_of_vocabulary_examples'][:8])}. These are "
      f"degenerate, not plausible words.\n")

    if o:
        A("\n## E7, can the head emit words it was never given an id for\n")
        for lane, v in o["lanes"].items():
            A(f"\n**{lane}**: input vocabulary {v['input_vocabulary']:,}, "
              f"{v['targets_outside_input_vocabulary']:,} further words appear only as targets.\n")
            A("| target band (by frequency) | exact reconstruction |")
            A("|---|---|")
            for band in ("in_head", "in_mid", "in_tail", "outside_near", "outside_far"):
                A(f"| {band} | {pct(v['frequency_bands_mean'][band], 4)} |")
            A(f"\nA vocabulary softmax scores exactly {v['vocab_head_outside_exact_rate']} on the "
              f"outside bands, by construction rather than by measurement.")
        A("\n`in_tail` is the rarity-matched control: the least frequent words that ARE in the "
          "inventory. It scores at the floor too, so the zero on the outside bands is a rarity "
          "cliff and says nothing about vocabulary membership. This experiment cannot test the "
          "claim at this scale.\n")

    try:
        cn = load("constrained.json")
    except FileNotFoundError:
        cn = None
    if cn:
        A("\n## E8, constrained decoding, which removes E6's defect\n")
        A(f"Same trained head, same logits, {cn['predictions']:,} predictions. The only change is "
          f"that bytes which cannot legally follow what has been emitted are masked before the "
          f"argmax, and any incomplete trailing character is dropped.\n")
        A("| metric | unconstrained | constrained |")
        A("|---|---|---|")
        for key, lab in (("invalid_utf8_rate", "invalid UTF-8"),
                         ("valid_utf8_rate", "valid UTF-8"),
                         ("empty_output_rate", "empty output"),
                         ("in_vocabulary_rate", "in vocabulary"),
                         ("exact_match_rate", "exact match to the target")):
            A(f"| {lab} | {pct(cn['unconstrained'][key])} | {pct(cn['constrained'][key])} |")
        imp = cn["improvement"]
        A(f"\nInvalid UTF-8 removed: {pct(imp['invalid_utf8_removed'])}. Exact match change: "
          f"{imp['exact_match_change'] * 100:+.2f} points. No retraining, no architectural "
          f"change.\n")

    A("\n## Verification\n")
    ce = t["codec_equivalence_check"]
    A(f"- Factored codec against the float64 definition: max absolute difference "
      f"**{ce['max_abs_vs_float64_definition']:.2e}**.")
    A(f"- Factored codec against `codec.encode` (float32): "
      f"{ce['max_abs_vs_codec_encode_float32']:.2e}, which is that function's own rounding.")
    A("- Central-difference gradient check, worst relative error per parameter: "
      + ", ".join(f"`{k}` {v:.1e}" for k, v in sorted(t["gradient_check"].items())) + ".")

    text = "\n".join(out) + "\n"
    with open(os.path.join(ART, "evidence.md"), "w", encoding="utf-8") as fh:
        fh.write(text)
    # The machine readable form of the same thing. `evidence.md` is for a reader and
    # `evidence.json` is for anything that wants to assert on a number without parsing prose.
    with open(os.path.join(ART, "evidence.json"), "w", encoding="utf-8") as fh:
        json.dump(provenance.stamp(collect(), __file__), fh, indent=2, sort_keys=True,
                  ensure_ascii=False)
    return text


if __name__ == "__main__":
    main()
    print("wrote artifacts/evidence.md")
