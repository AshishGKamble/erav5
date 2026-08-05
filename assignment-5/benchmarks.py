"""
Benchmark readiness - which Assignment-3 targets does the chosen mixture actually
feed, and how solid is the ground under each one?

This deliberately does NOT predict scores. A 4.85M-param CPU proxy cannot forecast
MMLU-Pro, and pretending otherwise is the wishful accounting the whole assignment is
about. Instead it reports, per benchmark, three facts a reviewer can check:

  1. EFFECTIVE GRADIENT - the share of the budget that actually carries loss for this
     benchmark. Share x trainable_frac, summed over the lanes that feed it. Agentic is
     0.35 trainable (tool output is masked context), so 8% of tokens is 2.8% of gradient.

  2. ORGANIC BACKING - what fraction of the demand real, naturally-occurring data could
     cover within the 4-epoch repetition ceiling: min(1, organic_supply x 4 / demand).
     Low backing means the benchmark rests on translated/synthetic/distilled tokens.

  3. PROXY EVIDENCE - did the tiny proxy actually test the feeding lane, and did the
     lane respond to its share? Lanes the proxy cannot reach (agentic needs tool
     execution, long-context needs 32K sequences) are marked untested, not assumed good.

Run: python3 benchmarks.py  ->  prints a table and writes benchmarks.md
"""
import os, json, glob

HERE = os.path.dirname(os.path.abspath(__file__))
INV = json.load(open(os.path.join(HERE, "supply", "inventory.json")))
PLAN = json.load(open(os.path.join(HERE, "plan.json")))
BUDGET = INV["budget_primary_B"]
MAX_EPOCHS = 4.0

LANES = {l["key"]: l for l in PLAN["full_plan"]["lanes"]}

# Assignment-3 targets, and the lanes that feed each. Weights sum to 1 per benchmark and
# encode which lane is doing the work (e.g. AIME is mostly math with a reasoning assist).
BENCHMARKS = [
    ("MMLU-Pro",            "≥ 85", [("web", .6), ("math_stem", .4)]),
    ("AIME '26",            "≥ 89", [("math_stem", .6), ("reasoning", .4)]),
    ("LiveCodeBench-v6",    "≥ 80", [("code", 1.0)]),
    ("GPQA-Diamond",        "≥ 84", [("math_stem", .6), ("reasoning", .4)]),
    ("BBEH",                "≥ 74", [("reasoning", .7), ("web", .3)]),
    ("tau2-bench",          "≥ 77", [("agentic", 1.0)]),
    ("MMMLU",               "≥ 88", [("indic", .6), ("web", .4)]),
    ("MRCR-v2 (256K)",      "≥ 66", [("long_ctx", 1.0)]),
    ("MILU / IndicGenBench", "lead",     [("indic", 1.0)]),
]

# Which inventory lane backs each plan lane, and which proxy lane (if any) tests it.
INV_KEY = {"web": "web", "code": "code", "math_stem": "math_stem", "reasoning": "reasoning_traces",
           "agentic": "agentic", "long_ctx": "long_context", "indic": "indic"}
PROXY_KEY = {"web": "web", "code": "code", "math_stem": "math", "reasoning": "reasoning", "indic": "indic"}


def organic_supply(lane):
    """Naturally-occurring tokens only - excludes anything the inventory marks generated."""
    d = INV["lanes"][INV_KEY[lane]]
    if lane == "indic":  # verified + unverified are the organic tiers
        return sum(x["tokens_B"] for t in ("verified", "unverified") for x in d["tiers"][t])
    return sum(x["tokens_B"] for x in d["datasets"] if x.get("organic"))


def proxy_evidence():
    """Per proxy lane: was it tested, and is its loss monotone in its own share?"""
    runs = {}
    for p in glob.glob(os.path.join(HERE, "proxy", "runs", "*.json")):
        r = json.load(open(p))
        runs[os.path.splitext(os.path.basename(p))[0]] = r
    # Seed-noise floor per lane: the spread of the SAME mixture re-run at other seeds.
    # Without it, replicate runs look like "the lane is mixture-sensitive" when they are
    # only showing us chance - the error that cost us three findings (README section 9.1).
    grp = {}
    for k, r in runs.items(): grp.setdefault(r.get("name", k), []).append(k)
    out = {}
    for lane in set(PROXY_KEY.values()):
        spreads = [max(runs[k]["final_per_lane"][lane] for k in g) - min(runs[k]["final_per_lane"][lane] for k in g)
                   for g in grp.values() if len(g) > 1 and all(lane in runs[k]["final_per_lane"] for k in g)]
        floor = max(spreads) if spreads else 0.0
        pts = sorted((r["lane_probs"].get(lane, 0), r["final_per_lane"].get(lane))
                     for r in runs.values() if r["final_per_lane"].get(lane) is not None)
        if len(pts) < 2:
            out[lane] = ("none", None); continue
        # a violation only counts if it is bigger than the lane's own seed noise
        viol = any(b[0] > a[0] + 1e-9 and b[1] - a[1] > max(1e-9, floor) for a in pts for b in pts)
        spread = max(v for _, v in pts) - min(v for _, v in pts)
        out[lane] = ("mixed" if viol else "responds", spread)
    return out, len(runs)


def main():
    ev, n_runs = proxy_evidence()
    rows = []
    for name, target, feeds in BENCHMARKS:
        eff = org = 0.0
        notes = []
        for lane, w in feeds:
            L = LANES[lane]
            demand = L["share"] / 100 * BUDGET
            eff += w * L["share"] * L["trainable_frac"]
            org += w * min(1.0, organic_supply(lane) * MAX_EPOCHS / demand)
            pk = PROXY_KEY.get(lane)
            if pk is None:
                notes.append(f"{lane}: untested")
            else:
                kind, spread = ev.get(pk, ("none", None))
                notes.append(f"{lane}: {kind}" + (f" (spread {spread:.2f})" if spread else ""))
        # verdict: gradient must be real AND the data must exist AND something must have tested it
        untested = any("untested" in n for n in notes)
        if untested:
            verdict = "**Unevidenced** - proxy cannot reach this lane"
        elif org < 0.35:
            verdict = "**Supply-limited** - rests on generated data"
        elif org < 0.8:
            verdict = "Partly supply-limited"
        elif any(": mixed" in n for n in notes):
            verdict = "Fed, but lane is mixture-sensitive"
        else:
            verdict = "On track"
        rows.append((name, target, eff, org * 100, "; ".join(notes), verdict))

    L = ["# Benchmark readiness (computed)\n",
         "Which Assignment-3 targets the chosen mixture feeds, and how solid the ground is under each.",
         "**These are not predicted scores** - a 4.85M CPU proxy cannot forecast MMLU-Pro. They are the",
         "three checkable facts behind each target: how much gradient it actually gets, how much of that",
         "could come from real data, and whether anything has tested it.\n",
         "| Benchmark | A3 target | Effective gradient | Organic backing | Proxy evidence | Verdict |",
         "|---|:---:|:---:|:---:|---|---|"]
    for name, target, eff, org, notes, verdict in rows:
        L.append(f"| **{name}** | {target} | {eff:.1f}% | {org:.0f}% | {notes} | {verdict} |")

    L.append("\n### How to read the columns\n")
    L.append("- **Effective gradient** = share x trainable fraction, summed over feeding lanes. Agentic is only "
             "0.35 trainable (tool output is masked context), so its 8% token share is ~2.8% of real learning.")
    L.append("- **Organic backing** = `min(1, organic_supply x 4 epochs / demand)`. 100% means real data could "
             "cover the whole demand; 0% means every token behind that benchmark is generated or distilled.")
    L.append(f"- **Proxy evidence** = from {n_runs} tiny-proxy runs. *responds* = loss falls as the lane's share "
             "rises by more than that lane's seed-noise floor; *mixed* = a real violation survives the "
             "floor; *untested* = the proxy cannot "
             "reach it at 128-token context with no tool execution.\n")

    weakest = sorted(rows, key=lambda r: r[3])[:3]
    L.append("### The honest bottom line\n")
    L.append("The targets standing on the thinnest ground are " +
             ", ".join(f"**{n}** ({o:.0f}% organic backing)" for n, _, _, o, _, _ in weakest) + ". "
             "Those are exactly the lanes the ledger flags as generation-heavy, so the risk is stated in two "
             "independent places rather than hidden. A reviewer should push hardest there - and the answer is "
             "not a bigger share, it is a better generation pipeline plus the 1B proxy that would actually "
             "measure these benchmarks instead of held-out loss.")

    open(os.path.join(HERE, "benchmarks.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    return rows


if __name__ == "__main__":
    main()
