"""
Extract a compact dashboard payload from the artefacts.

The site renders only what this emits, so it cannot drift from the experiments. Nothing is hardcoded
in the HTML or the JavaScript.
"""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(HERE, "..", "artifacts")
OUT = os.path.join(HERE, "..", "site", "data", "dashboard.json")


def load(n):
    with open(os.path.join(ART, n), encoding="utf-8") as fh:
        return json.load(fh)


def main():
    k = load("codec.json")
    payload = {"vocab": k["vocabulary_measured"],
               "roundtrip": {L: {"D": v["D"], "fitting": v["tokens_fitting"],
                                 "recovered": v["tokens_fitting_recovered"],
                                 "overflow": v["tokens_overflowing"]}
                             for L, v in k["e1_roundtrip"].items()},
               "margin": k["e2_noise"]["column_margin_after_znorm"],
               "noise": [{"sigma": r["sigma"],
                          "oracle": r["exact_token_accuracy_oracle_length"],
                          "inferred": r["exact_token_accuracy_inferred_length"]}
                         for r in k["e2_noise"]["sweep"]],
               "sparsity": {"k": k["e3_projection"]["occupied_columns_mean"],
                            "D": k["e3_projection"]["D"],
                            "density": k["e3_projection"]["sparsity_fraction_mean"]},
               "projection": [{"d": r["d_model"], "acc": r["minimum_norm_decode_accuracy"],
                               "null": r["nullspace_dimension"]}
                              for r in k["e3_projection"]["sweep"]]}
    # How much noise the decode tolerates, against how much the objection actually implies.
    # "0.31 instead of 0.30" is a relative error of 0.01/0.30. The tolerated sigma is the largest
    # one at which exact accuracy is still 1.0, and the ratio between them is the headroom.
    tolerated = max([r["sigma"] for r in k["e2_noise"]["sweep"]
                     if r["exact_token_accuracy_oracle_length"] >= 1.0] or [0.0])
    objection = 0.01 / 0.30
    payload["headroom"] = {"tolerated_sigma": tolerated, "objection_relative_error": objection,
                           "ratio": tolerated / objection}
    try:
        t = load("train.json")
    except FileNotFoundError:
        t = None
    if t:
        e4 = t["e4_three_heads"]
        payload["heads"] = {h: {"loss": e4[h]["final_loss_per_token_mean"],
                                "sd": e4[h]["final_loss_per_token_sd"],
                                "exact": e4[h]["runs"][0]["final"]["exact_token_accuracy_oracle_length"],
                                "params": e4[h]["params_measured"]["total"],
                                "head_params_at_scale": e4[h]["params_at_paper_scale"]["output_head"]}
                            for h in ("vocab", "byte_untied", "byte_tied")}
        payload["noise_floor"] = e4["_noise_floor_sd_across_seeds"]
        payload["config"] = t["config"]
        e5 = t["e5_gradient_from_zero"]
        payload["init"] = e5["_copy_at_init"]
        payload["objectives"] = {o: {"start": e5[o]["byte_accuracy_at_step_0"],
                                     "end": e5[o]["byte_accuracy_final"],
                                     "curve": [{"step": h["step"], "acc": h["byte_accuracy"]}
                                               for h in e5[o]["history"]]}
                                 for o in ("ce", "mse")}
        payload["tying"] = {kk: t["e6_tying_cost"][kk] for kk in
                            ("invalid_utf8_rate", "in_vocabulary_rate",
                             "valid_utf8_out_of_vocabulary_rate", "exact_token_match_rate",
                             "predictions_scored", "out_of_vocabulary_examples")}
        payload["recheck_minnorm"] = {d: {"before": v["before"]["minimum_norm_decode_accuracy"],
                                          "after": v["after"]["minimum_norm_decode_accuracy"],
                                          "cond_before": v["before"]["condition_number"],
                                          "cond_after": v["after"]["condition_number"]}
                                      for d, v in t["e3_recheck_with_trained_W"].items()
                                      if isinstance(v, dict) and "before" in v}
    try:
        r = load("recheck.json")
        payload["recheck"] = {lab: {"duplicates": r[lab]["separation"]["exact_duplicates"],
                                    "separation": r[lab]["separation"]["relative_separation_min"],
                                    "learned": r[lab]["learned_linear_inverse"]["held_out_decode_accuracy"]}
                              for lab in ("random_init", "after_training")}
        payload["recheck"]["meta"] = {"fit": r["fit_tokens"], "probe": r["probe_tokens"],
                                      "d": r["d_model"]}
    except FileNotFoundError:
        pass
    try:
        o = load("openvocab.json")
        payload["openvocab"] = {lane: {"bands": v["frequency_bands_mean"],
                                       "inside": v["in_inventory_exact_rate_mean"],
                                       "outside": v["outside_exact_rate_mean"],
                                       "inv": v["input_vocabulary"],
                                       "outside_n": v["targets_outside_input_vocabulary"]}
                                for lane, v in o["lanes"].items()}
    except FileNotFoundError:
        pass
    try:
        cn = load("constrained.json")
        payload["constrained"] = {"unconstrained": cn["unconstrained"],
                                  "constrained": cn["constrained"],
                                  "improvement": cn["improvement"],
                                  "predictions": cn["predictions"]}
    except FileNotFoundError:
        pass
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, separators=(",", ":"))
    return payload


if __name__ == "__main__":
    main()
    print("wrote site/data/dashboard.json,", os.path.getsize(OUT), "bytes")
