"""
Extract a compact dashboard payload from the artefacts.

The site renders only what this file emits, so the dashboard cannot drift from the experiments: if a
number changes in `artifacts/`, it changes on the page the next time this runs. Nothing is hardcoded
in the HTML or the JavaScript.
"""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(HERE, "..", "artifacts")
OUT = os.path.join(HERE, "..", "site", "data", "dashboard.json")
INDIC = ["MALAYALAM", "TAMIL", "KANNADA", "TELUGU", "ORIYA", "BENGALI", "DEVANAGARI",
         "GUJARATI", "GURMUKHI"]
PROSE = ["web", "long_ctx", "reasoning"]


def load(n):
    with open(os.path.join(ART, n), encoding="utf-8") as fh:
        return json.load(fh)


def main():
    w, f, c, b = load("window.json"), load("fixes.json"), load("cost.json"), load("bothends.json")
    try:
        d = load("downstream.json")
    except FileNotFoundError:
        d = None

    ls = w["e3_by_lane_and_script"]["rows"]
    headline = {}
    for L in ("16", "32", "64"):
        headline[L] = {
            "indic_groups": sum(ls["indic"][s][L]["colliding_groups"] for s in INDIC if s in ls["indic"]),
            "indic_types": sum(ls["indic"][s][L]["distinct_types"] for s in INDIC if s in ls["indic"]),
            "english_groups": sum(ls[l]["LATIN"][L]["colliding_groups"] for l in PROSE if l in ls),
            "english_types": sum(ls[l]["LATIN"][L]["distinct_types"] for l in PROSE if l in ls),
        }

    tp = w["e3_truncation"]["word_prose"]
    tr = w["e3_truncation"]["word_raw"]
    # The two measurement corrections, as numbers rather than as a story. Pooled Latin before and
    # after excluding the code lane and stripping edge punctuation.
    corrections = {"latin_raw_16": tr["LATIN"]["16"]["types_in_collisions_rate"],
                   "latin_prose_16": tp["LATIN"]["16"]["types_in_collisions_rate"],
                   "malayalam_raw_16": tr["MALAYALAM"]["16"]["types_in_collisions_rate"],
                   "malayalam_prose_16": tp["MALAYALAM"]["16"]["types_in_collisions_rate"]}
    # Three bytes per character is a UTF-8 fact, not a definition of "Indic": Han costs three bytes
    # too. So the two lists are kept apart, and the collision tables use the explicit Indic list.
    three_byte = sorted(s for s, r in w["e2_characters_per_window"]["rows"].items()
                        if abs(r["bytes_per_character"] - 3.0) < 1e-9)
    indic_present = [s for s in INDIC if s in w["e2_characters_per_window"]["rows"]]
    payload = {
        "corrections": corrections,
        "three_byte_scripts": three_byte,
        "indic_scripts": indic_present,
        "indic_script_count": len(indic_present),
        "corpus": w["corpus"],
        "headline": headline,
        "collision_check": w["e3_collision_check"],
        "occupancy": {lane: {"mean_bytes": r["mean_bytes_per_token"],
                             "occupancy32": r["windows"]["32"]["mean_occupancy"],
                             "tokens": r["tokens"]}
                      for lane, r in w["e1_occupancy"]["per_lane"].items()},
        "scripts": {s: {"bytes_per_char": r["bytes_per_character"],
                        "chars32": r["characters_in_window"]["32"],
                        "graphemes32": r["graphemes_in_window_approx"]["32"]}
                    for s, r in w["e2_characters_per_window"]["rows"].items()},
        "collisions_by_script": {s: {L: tp[s][L]["types_in_collisions_rate"] for L in ("16", "32", "64")}
                                 for s in tp},
        "examples": {s: [g["members"] for g in tp[s]["16"]["examples"][:4]]
                     for s in ("TAMIL", "MALAYALAM", "DEVANAGARI", "TELUGU") if s in tp},
        "fixes": {s: {"byte": v["8192"]["byte"]["types_in_collisions_rate"],
                      "codepoint": v["8192"]["codepoint"]["types_in_collisions_rate"],
                      "script_relative": (f["e4f_script_relative"]["rows"][s]["tagged"]
                                          ["types_in_collisions_rate"]
                                          if s in f["e4f_script_relative"]["rows"] else None)}
                  for s, v in f["e4b_collisions_at_equal_D"]["rows"].items()},
        "entropy": {s: r["high_digit_entropy_bits"]
                    for s, r in f["e4c_block_entropy"]["rows"].items()},
        "schemes": {n: {"groups": v["total_colliding_groups"],
                        "reduction": v["reduction_vs_published"],
                        "description": v["description"],
                        "malayalam": v["per_script"]["MALAYALAM"]["types_in_collisions_rate"]}
                    for n, v in b["schemes"].items()},
        "cost": {
            "memory": {L: {"dense_mb": r["dense_mb"], "factored_mb": r["factored_mb"],
                           "ratio": r["ratio"], "D": r["D"]}
                       for L, r in c["memory"]["rows"].items()},
            "compute": {L: {"arithmetic_ratio": r["arithmetic_ratio"], "speedup": r["speedup"]}
                        for L, r in c["compute"]["rows"].items()},
            "scaling": c["scaling_with_length"]["buckets"],
            "scaling_fit": {"slope": c["scaling_with_length"]["linear_fit_slope_ns_per_unit"],
                            "corr": c["scaling_with_length"]["correlation"]},
            "raising_L": c["raising_L_is_cheap"]["rows"],
            "zeros": {L: r["zero_column_fraction"] for L, r in c["dimensions"]["rows"].items()},
        },
        "choose_L": b.get("choose_L"),
    }
    try:
        bc = load("bothends_codec.json")
        payload["codec_check"] = {
            "roundtrip_prefix": bc["roundtrip_prefix"]["rate"],
            "roundtrip_both_ends": bc["roundtrip_both_ends"]["rate"],
            "bitwise": bc["bitwise_collisions"],
            "cut_quality": bc["cut_quality"],
            "cut_quality_aligned": bc["cut_quality_aligned"],
            "capacity": bc["capacity_cost_of_aligning"],
        }
    except FileNotFoundError:
        pass
    if d:
        payload["downstream"] = {
            lane: {"noise": L["seed_noise_floor_sd"],
                   "arms": {a: {"loss": v["final_loss_per_token_mean"],
                                "delta": v["delta_vs_byte"], "verdict": v["verdict"]}
                            for a, v in L["arms"].items()},
                   "exposure": L.get("truncation_exposure")}
            for lane, L in d["lanes"].items()}
        if "e5b_word_level" in d:
            payload["e5b"] = {lane: {a: {"exact": v["exact_full_word_mean"],
                                         "representable": v["word_types_representable"],
                                         "truncated": v["target_occurrences_truncated"]}
                                     for a, v in L["arms"].items()}
                              for lane, L in d["e5b_word_level"].items()}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, separators=(",", ":"))
    return payload


if __name__ == "__main__":
    p = main()
    print("wrote site/data/dashboard.json,", os.path.getsize(OUT), "bytes")
