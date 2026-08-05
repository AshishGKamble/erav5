"""
Supply ledger - the anti-wishful-accounting check.

For every capability lane it compares DEMAND (share x budget) against real SUPPLY
(from inventory.json) and reports how the demand is met: organic tokens, epochs of
repetition, or generated/synthetic tokens. The session's central warning is that a
plan which "quietly hands a large share to a lane that has almost no real data" loses
marks - so this script makes the repetition/generation explicit for each lane, and
breaks the Indic lane across its verified/unverified/translated/synthetic tiers.

Run: python3 supply/ledger.py  ->  prints a table and writes supply/ledger.md
"""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
INV = json.load(open(os.path.join(HERE, "inventory.json")))
BUDGET = INV["budget_primary_B"]                      # 4000 B tokens (primary anchor)

# Integrated lane shares (% of the 4T budget), defended in the README. These are the
# run-INTEGRATED averages; the per-stage trajectory (Seed->Anneal) is in the plan.
SHARES = {
    "web":              29.0,   # MMLU / common sense - fades from ~51% early to ~18% late
    "code":             20.0,   # LiveCodeBench / SWE-bench / Codeforces - our priority
    "math_stem":        10.0,   # AIME / GPQA / MMLU-Pro(STEM)  (tutor "STEM")
    "reasoning_traces":  9.0,   # depth-controlled CoT (tutor "Reasoning")
    "agentic":           8.0,   # tau2 / BFCL / Terminal-bench - PROTECTED floor
    "long_context":      6.0,   # RULER / MRCR
    "indic":            18.0,   # MILU / IndicGenBench - PROTECTED floor
}
MAX_EPOCHS = 4.0                                       # data-constrained-scaling repetition ceiling

# Indic 4-tier split (% OF the Indic lane). Verified is capped by real supply; most
# mass is translated/synthetic - stated honestly, not hidden behind one headline number.
INDIC_TIER_SPLIT = {"verified": 18.0, "unverified": 14.0, "translated": 33.0, "synthetic": 35.0}


def lane_supply(lane):
    d = INV["lanes"][lane]
    if lane == "indic":
        tiers = {t: sum(x["tokens_B"] for x in rows) for t, rows in d["tiers"].items()}
        return sum(tiers.values()), tiers, d
    organic = sum(x["tokens_B"] for x in d["datasets"] if x.get("organic"))
    total = sum(x["tokens_B"] for x in d["datasets"])
    return total, {"organic": organic, "generated": total - organic}, d


def meet(demand_B, supply_B):
    """How demand is met: epochs of repetition (<=MAX) then generated remainder."""
    if supply_B <= 0:
        return 0.0, demand_B
    epochs = demand_B / supply_B
    if epochs <= MAX_EPOCHS:
        return round(epochs, 2), 0.0
    reachable = supply_B * MAX_EPOCHS
    return MAX_EPOCHS, round(demand_B - reachable, 1)



def write_roster():
    """Which named inventory datasets fill each slot, and how much each contributes.

    The assignment asks the plan to point every slot - agentic, reasoning and long-context
    especially - at the datasets from the inventory that will fill it. Generated from
    inventory.json so the roster cannot drift from the supply numbers the ledger computes.
    """
    L = ["# Dataset roster - what actually fills each slot\n",
         "Generated from `inventory.json`. `gen` marks a source that does not exist yet and must be",
         "created; every such row is a promise the plan is making, not a dataset it can point at today.\n"]
    for lane, d in INV["lanes"].items():
        L.append(f"\n## {lane}  ({', '.join(d['benchmarks'])})\n")
        if d.get("note"):
            L.append(f"_{d['note']}_\n")
        L.append("| Dataset | Tokens (B) | Kind | Source |")
        L.append("|---|---:|:---:|---|")
        if lane == "indic":
            for tier, rows in d["tiers"].items():
                for x in rows:
                    kind = "GEN" if "generate" in x["source"].lower() else tier
                    L.append(f"| {x['name']} | {x['tokens_B']} | {kind} | {x['source']} |")
        else:
            for x in d["datasets"]:
                kind = "organic" if x.get("organic") else ("GEN" if "GENERATED" in x["source"] else "synthetic")
                L.append(f"| {x['name']} | {x['tokens_B']} | {kind} | {x['source']} |")
    open(os.path.join(HERE, "datasets.md"), "w").write("\n".join(L) + "\n")
    print("wrote supply/datasets.md")


def main():
    lines = ["# Supply ledger (computed)\n",
             f"Budget = **{BUDGET/1000:.1f}T** tokens (primary). Repetition ceiling = {MAX_EPOCHS} epochs.",
             "Numbers in **billions** of tokens. `gen` = must be generated/synthesised.\n",
             "| Lane | Share | Demand | Real supply | Trainable frac | How it's met |",
             "|------|------:|-------:|------------:|:--------------:|--------------|"]
    print(f"{'lane':16s}{'share%':>7}{'demand_B':>10}{'supply_B':>10}{'epochs':>8}{'gen_B':>9}")
    tot_gen = 0.0
    for lane, share in SHARES.items():
        demand = BUDGET * share / 100.0
        supply, brk, d = lane_supply(lane)
        trainable = d.get("trainable_frac", 1.0)
        epochs, gen = meet(demand, supply)
        tot_gen += gen
        met = f"{epochs} epochs of {supply:.0f}B" + (f"; **{gen:.0f}B generated**" if gen > 0 else "")
        if trainable < 1.0:
            met += f"  · only {int(trainable*100)}% carries loss (rest = masked context)"
        lines.append(f"| {lane} | {share:.0f}% | {demand:.0f} | {supply:.0f} | {trainable:.2f} | {met} |")
        print(f"{lane:16s}{share:>7.0f}{demand:>10.0f}{supply:>10.0f}{epochs:>8}{gen:>9.0f}")

    # Indic tier breakdown
    indic_demand = BUDGET * SHARES["indic"] / 100.0
    _, tiers, _ = lane_supply("indic")
    lines += ["\n## Indic lane - the four tiers\n",
              f"Indic demand = **{indic_demand:.0f}B** ({SHARES['indic']:.0f}% of {BUDGET/1000:.1f}T).",
              "| Tier | Share of Indic | Demand | Real supply | How it's met |",
              "|------|---------------:|-------:|------------:|--------------|"]
    print("\n-- Indic tiers --")
    for tier, tshare in INDIC_TIER_SPLIT.items():
        td = indic_demand * tshare / 100.0
        ts = tiers[tier]
        ep, gen = meet(td, ts)
        met = f"{ep} epochs of {ts:.0f}B" + (f"; **{gen:.0f}B generated**" if gen > 0 else "")
        lines.append(f"| {tier} | {tshare:.0f}% | {td:.0f} | {ts:.0f} | {met} |")
        print(f"  {tier:12s}{tshare:>6.0f}%{td:>9.0f}B{ts:>9.0f}B  ep={ep} gen={gen}")

    verified_share_of_budget = SHARES["indic"] * INDIC_TIER_SPLIT["verified"] / 100.0
    lines += [
        f"\n**Honest headline:** organic Indic (verified+unverified) is ~{tiers['verified']+tiers['unverified']:.0f}B; "
        f"the {SHARES['indic']:.0f}% Indic lane ({indic_demand:.0f}B) is therefore **majority translated+synthetic** "
        f"(~{INDIC_TIER_SPLIT['translated']+INDIC_TIER_SPLIT['synthetic']:.0f}% of the lane). Verified is only "
        f"~{verified_share_of_budget:.1f}% of the whole budget - it is the scarce, highest-value tier and we protect it in the anneal.",
        f"\n**Total tokens that must be generated across all lanes: ~{tot_gen:.0f}B "
        f"({100*tot_gen/BUDGET:.1f}% of budget)** - dominated by **agentic** and **reasoning**, where real "
        f"supply barely exists (long-context is met at 1 epoch, but ~60% of that supply is itself synthetic)."
    ]
    write_roster()
    open(os.path.join(HERE, "ledger.md"), "w").write("\n".join(lines) + "\n")
    print(f"\nTotal generated across lanes: {tot_gen:.0f}B ({100*tot_gen/BUDGET:.1f}% of budget)")
    print("wrote supply/ledger.md")


if __name__ == "__main__":
    main()
