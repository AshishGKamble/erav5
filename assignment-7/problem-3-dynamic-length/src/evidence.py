"""
Regenerate every number quoted in the Problem 3 writeup, straight from the artefacts.

No figure in README.md is typed by hand. This script reads `artifacts/*.json` and emits
`artifacts/evidence.md`, so a reader who distrusts a table can regenerate it, and a reviewer who
changes an experiment finds the prose disagreeing with the evidence file immediately.
"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(HERE, "..", "artifacts")
INDIC = ["MALAYALAM", "TAMIL", "KANNADA", "TELUGU", "ORIYA", "BENGALI", "DEVANAGARI",
         "GUJARATI", "GURMUKHI"]
PROSE_LANES = ["web", "long_ctx", "reasoning"]


def load(name):
    with open(os.path.join(ART, name), encoding="utf-8") as fh:
        return json.load(fh)


def pct(x, n=2):
    return "n/a" if x is None else f"{x * 100:.{n}f}%"


def main():
    w, f, c, b = load("window.json"), load("fixes.json"), load("cost.json"), load("bothends.json")
    try:
        d = load("downstream.json")
    except FileNotFoundError:
        d = None
    out = []
    A = out.append

    A("# Problem 3, evidence\n")
    A("Every number below is read from `artifacts/*.json`. Regenerate with `python src/evidence.py`.\n")
    A(f"Corpus: {w['corpus']['characters']:,} characters, "
      f"{w['corpus']['distinct_token_types']:,} token types, "
      f"{w['corpus']['distinct_word_types']:,} word types across "
      f"{len(w['corpus']['documents_per_lane'])} lanes.\n")

    A("\n## E1, occupancy: how much of the window is used\n")
    A("| lane | tokens | mean bytes/token | occupancy at L=32 | zero columns |")
    A("|---|---|---|---|---|")
    for lane, r in w["e1_occupancy"]["per_lane"].items():
        x = r["windows"]["32"]
        A(f"| {lane} | {r['tokens']:,} | {r['mean_bytes_per_token']:.2f} | "
          f"{pct(x['mean_occupancy'])} | {pct(x['zero_column_fraction'])} |")

    A("\n## E2, characters per window by script\n")
    A("| script | bytes/char | chars in 32 bytes | graphemes in 32 bytes | capacity vs Latin |")
    A("|---|---|---|---|---|")
    for s, r in sorted(w["e2_characters_per_window"]["rows"].items(),
                       key=lambda kv: -kv[1]["bytes_per_character"]):
        g = r["graphemes_in_window_approx"]["32"]
        A(f"| {s} | {r['bytes_per_character']:.3f} | {r['characters_in_window']['32']:.1f} | "
          f"{g:.1f} | {r.get('capacity_relative_to_latin', float('nan')):.2f}x |")

    A("\n## E3, truncation collisions at L=32 (word types, prose lanes)\n")
    t = w["e3_truncation"]["word_prose"]
    A("| script | word types | cropped | in collisions | colliding groups |")
    A("|---|---|---|---|---|")
    for s in sorted(t, key=lambda k: -t[k]["32"]["types_in_collisions_rate"]):
        r = t[s]["32"]
        A(f"| {s} | {r['distinct_types']:,} | {pct(r['types_cropped_rate'])} | "
          f"{pct(r['types_in_collisions_rate'])} | {r['colliding_groups']:,} |")

    ls = w["e3_by_lane_and_script"]["rows"]
    for L in ("16", "32", "64"):
        gi = sum(ls["indic"][s][L]["colliding_groups"] for s in INDIC if s in ls["indic"])
        ti = sum(ls["indic"][s][L]["distinct_types"] for s in INDIC if s in ls["indic"])
        ge = sum(ls[l]["LATIN"][L]["colliding_groups"] for l in PROSE_LANES if l in ls)
        te = sum(ls[l]["LATIN"][L]["distinct_types"] for l in PROSE_LANES if l in ls)
        A(f"\n- **L={L}**: nine Indic scripts {gi:,} colliding groups across {ti:,} word types; "
          f"English prose (web + long_ctx + reasoning) **{ge}** across {te:,}.")
    cc = w["e3_collision_check"]
    A(f"\nBitwise check: {cc['pairs_with_bitwise_identical_codec_vectors']}/{cc['pairs_checked']} "
      f"sampled colliding pairs produce identical codec vectors, max absolute difference "
      f"{cc['max_absolute_difference']}. Verdict: {cc['verdict']}.\n")

    A("\n## E4, the fixes compared at equal D = 8192\n")
    eb, fd = f["e4b_collisions_at_equal_D"]["rows"], f["e4f_script_relative"]["rows"]
    A("| script | byte L=32 | fix B codepoint L=16 | fix D script-relative L=32 |")
    A("|---|---|---|---|")
    for s in sorted(eb, key=lambda k: -eb[k]["8192"]["byte"]["types_in_collisions_rate"]):
        x = eb[s]["8192"]
        dd = fd[s]["tagged"]["types_in_collisions_rate"] if s in fd else None
        A(f"| {s} | {pct(x['byte']['types_in_collisions_rate'])} | "
          f"{pct(x['codepoint']['types_in_collisions_rate'])} | {pct(dd)} |")

    A("\n### E4c, why fix B wastes half its dimensions\n")
    A("| script | high-digit entropy (bits, max 8) | distinct high digits |")
    A("|---|---|---|")
    for s, r in sorted(f["e4c_block_entropy"]["rows"].items(),
                       key=lambda kv: kv[1]["high_digit_entropy_bits"]):
        A(f"| {s} | {r['high_digit_entropy_bits']:.4f} | {r['high_digit_distinct_values']} |")

    fr = f["e4f_script_relative"]
    A(f"\nFix D costs, measured: {fr['cross_script_alias_groups_untagged']:,} cross-script alias "
      f"groups if the script tag is dropped; with the tag, residual collisions remain only where a "
      f"word mixes scripts.\n")

    A("\n## E6, what the window costs, three ways\n")
    A("| window | dimensions D | zeros | dense memory | factored memory | ratio |")
    A("|---|---|---|---|---|---|")
    for L in sorted(c["memory"]["rows"], key=int):
        m, dm = c["memory"]["rows"][L], c["dimensions"]["rows"][L]
        A(f"| L={L} | {m['D']:,} | {pct(dm['zero_column_fraction'])} | {m['dense_mb']:.1f} MB | "
          f"{m['factored_mb']:.3f} MB | {m['ratio']:.0f}x |")
    A("")
    A("| window | arithmetic reduction | wall-clock speedup |")
    A("|---|---|---|")
    for L in sorted(c["compute"]["rows"], key=int):
        r = c["compute"]["rows"][L]
        sp = f"{r['speedup']:.2f}x" if r["speedup"] else "not run"
        A(f"| L={L} | {r['arithmetic_ratio']:.0f}x | {sp} |")
    s = c["scaling_with_length"]
    A(f"\nCost against token length: slope **{s['linear_fit_slope_ns_per_unit']:.1f} ns per unit**, "
      f"correlation **{s['correlation']:.4f}**. A flat line would refute the dynamic claim.\n")
    A("| window | time for a short token, relative to L=32 | projection W parameters |")
    A("|---|---|---|")
    for L in sorted(c["raising_L_is_cheap"]["rows"], key=int):
        r = c["raising_L_is_cheap"]["rows"][L]
        A(f"| L={L} | {r['short_token_time_vs_L32']:.3f}x | {r['projection_parameters']:,} |")

    A("\n## E7, reading the word from both ends\n")
    A("| scheme | colliding groups | Malayalam | reduction |")
    A("|---|---|---|---|")
    for n, v in b["schemes"].items():
        mal = v["per_script"]["MALAYALAM"]["types_in_collisions_rate"]
        A(f"| {v['description']} | {v['total_colliding_groups']:,} | {pct(mal)} | "
          f"{v['reduction_vs_published']:.1f}x |")

    try:
        bc = load("bothends_codec.json")
    except FileNotFoundError:
        bc = None
    if bc:
        A("\n### E7 verified as a codec, not only as a key\n")
        A(f"- Round trip, published prefix: {bc['roundtrip_prefix']['rate']:.4f}. "
          f"Both ends: {bc['roundtrip_both_ends']['rate']:.4f}.")
        cc = bc["bitwise_collisions"]
        A(f"- Bitwise: {cc['bitwise_identical']}/{cc['pairs_checked']} colliding pairs produce "
          f"identical vectors, max difference {cc['max_absolute_difference']}. "
          f"Verdict: {cc['verdict']}.\n")
        A("| script | prefix cut lands mid-character | both ends | both cuts aligned |")
        A("|---|---|---|---|")
        for sc, r in bc["cut_quality"].items():
            al = bc["cut_quality_aligned"].get(sc, {})
            A(f"| {sc} | {pct(r['prefix_invalid_utf8_rate'])} | "
              f"{pct(r['both_ends_invalid_utf8_rate'])} | "
              f"{pct(al.get('both_ends_invalid_utf8_rate'))} |")
        cap = bc["capacity_cost_of_aligning"]
        A(f"\nAligning both cuts costs capacity: {cap['mean_units_kept_unaligned']:.2f} units "
          f"retained against {cap['mean_units_kept_aligned']:.2f} aligned.\n")

    if "choose_L" in b:
        A("\n## Which window to use\n")
        A("| scheme | L | D | projection parameters | colliding groups |")
        A("|---|---|---|---|---|")
        for name, rows in b["choose_L"]["rows"].items():
            for L, v in sorted(rows.items(), key=lambda kv: int(kv[0])):
                A(f"| {name} | {L} | {v['D']:,} | {v['projection_parameters']:,} | "
                  f"{v['colliding_groups']:,} |")

    if d:
        A("\n## E5, downstream, and why the token-level version is null\n")
        for lane, L in d["lanes"].items():
            te = L.get("truncation_exposure", {})
            A(f"\n**{lane}** (noise floor sd {L['seed_noise_floor_sd']:.4f} nats/token)\n")
            A("| arm | loss per token | delta vs byte | verdict |")
            A("|---|---|---|---|")
            for a, v in L["arms"].items():
                A(f"| {a} | {v['final_loss_per_token_mean']:.4f} | "
                  f"{v['delta_vs_byte']:+.4f} | {v['verdict']} |")
            if te:
                A(f"\nExposure: of {te['token_occurrences']:,} token occurrences, byte truncates "
                  f"{te['byte']['truncated_occurrences']} "
                  f"({pct(te['byte']['truncated_rate'], 4)}), codepoint "
                  f"{te['codepoint']['truncated_occurrences']}, script-relative "
                  f"{te['script_relative']['truncated_occurrences']}.")
        if "e5b_word_level" in d:
            A("\n### E5b, the same test at word level, where truncation is present\n")
            A("| lane | arm | word types representable | exact full-word | targets truncated |")
            A("|---|---|---|---|---|")
            for lane, L in d["e5b_word_level"].items():
                for a, v in L["arms"].items():
                    A(f"| {lane} | {a} | {pct(v['word_types_representable'])} | "
                      f"{pct(v['exact_full_word_mean'], 3)} | "
                      f"{pct(v['target_occurrences_truncated'])} |")
            for lane, L in d["e5b_word_level"].items():
                A(f"\n- {lane}: script-relative minus byte = "
                  f"{L['script_relative_minus_byte']:+.4f}, seed noise sd "
                  f"{L['seed_noise_floor_sd']:.4f}, exceeds noise: {L['exceeds_seed_noise']}")

    text = "\n".join(out) + "\n"
    with open(os.path.join(ART, "evidence.md"), "w", encoding="utf-8") as fh:
        fh.write(text)
    return text


if __name__ == "__main__":
    main()
    print("wrote artifacts/evidence.md")
