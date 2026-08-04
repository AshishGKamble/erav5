"""
Merge plan.json (candidate design SETS + the chosen full plan) with each set's proxy
outcome (proxy/runs/<proxy_key>.json) into site/data/dashboard.json for the dashboard.
Re-run after editing plan.json or after a new proxy run.
"""
import os, json

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "site", "data"); os.makedirs(OUT, exist_ok=True)
plan = json.load(open(os.path.join(HERE, "plan.json")))

def load_run(key):
    p = os.path.join(HERE, "proxy", "runs", f"{key}.json")
    return json.load(open(p)) if os.path.exists(p) else None

# attach proxy outcomes to each set
sets = []
for s in plan["sets"]:
    r = load_run(s.get("proxy_key", ""))
    out = dict(s)
    if r:
        out["outcome"] = {"final": r["final_per_lane"], "avg": r["final_avg"],
                          "history": r["history"], "params_M": r["params_M"], "steps": r["steps"]}
    sets.append(out)
proxy_lanes = plan["proxy_lanes"]

# confirm/refute verdicts, computed from the sets' outcomes (set-agnostic wording)
def loss(setkey, lane):
    for s in sets:
        if s["key"] == setkey and "outcome" in s: return s["outcome"]["final"].get(lane)
    return None
verdicts = []
if loss("balanced","indic") and loss("web_heavy","indic"):
    d = loss("web_heavy","indic") - loss("balanced","indic")
    verdicts.append({"ok": d > 0, "claim": "Protecting Indic works",
        "detail": f"Indic loss: Balanced {loss('balanced','indic'):.3f} vs Web-heavy {loss('web_heavy','indic'):.3f} (Δ {d:+.3f}) - 5%→20% Indic buys Indic."})
if loss("code_forward","web") and loss("balanced","web"):
    d = loss("code_forward","web") - loss("balanced","web")
    verdicts.append({"ok": d > 0, "claim": "Starving web costs common sense",
        "detail": f"Web loss: Code-forward {loss('code_forward','web'):.3f} vs Balanced {loss('balanced','web'):.3f} (Δ {d:+.3f})."})
if loss("code_forward","code") and loss("web_heavy","code"):
    d = loss("web_heavy","code") - loss("code_forward","code")
    verdicts.append({"ok": d > 0, "claim": "Code share buys code",
        "detail": f"Code loss: Code-forward {loss('code_forward','code'):.3f} vs Web-heavy {loss('web_heavy','code'):.3f} (Δ {d:+.3f})."})

# ---- findings: the noise floor, the tier experiment, the validity audit, the withdrawals.
# These are what the study actually concluded, and without them the set-comparison above is
# misleading: several of those apparent winners did not survive scrutiny.
import glob as _g, statistics as _st
def _runs(d):
    return {os.path.splitext(os.path.basename(p))[0]: json.load(open(p)) for p in _g.glob(os.path.join(HERE, d, "*.json"))}
_main, _tier = _runs("proxy/runs"), _runs("proxy/runs_tier")
_grp = {}
for k, r in _main.items(): _grp.setdefault(r.get("name", k), []).append(k)
_lanes = ["web", "code", "math", "reasoning", "indic"]
floor = {}
for l in _lanes:
    sp = [max(_main[k]["final_per_lane"][l] for k in g) - min(_main[k]["final_per_lane"][l] for k in g)
          for g in _grp.values() if len(g) > 1]
    floor[l] = round(max(sp), 3) if sp else None
sp_avg = [max(_main[k]["final_avg"] for k in g) - min(_main[k]["final_avg"] for k in g)
          for g in _grp.values() if len(g) > 1]
floor["avg"] = round(max(sp_avg), 3) if sp_avg else None

def _tstat(name, lane):
    v = [r["final_per_lane"][lane] for r in _tier.values() if r["name"] == name]
    return (round(_st.mean(v), 3), round(max(v) - min(v), 3) if len(v) > 1 else 0.0) if v else (None, 0.0)
tier = {}
if _tier:
    for lane in ("indic_hi", "indic_lo"):
        a, fa = _tstat("tier18", lane); b, fb = _tstat("tier30", lane)
        tier[lane] = {"t18": a, "t30": b, "delta": round(b - a, 3), "floor": round(max(fa, fb), 3),
                      "readable": abs(b - a) > max(fa, fb)}
    tier["ideal_hi"] = _tstat("tier30_ideal", "indic_hi")[0]

findings = {
  "floor": floor,
  "tier": tier,
  "native_pinned_pct": 11.0,
  "leakage": {"web": 0, "code": 2, "math": 0, "reasoning": 100, "indic": 6},
  "withdrawn": [
    ["Every lane is monotone in its share", "generalised from 3 pairs never designed to test it"],
    ["Web scaffolds Indic", "0.120 effect against a 0.216 noise floor"],
    ["The v2 revision improves the mixture", "0.011 against a 0.033 floor - no measurable difference"],
    ["More Indic buys Indic capability", "the Indic bin was 98.5% synthetic; native Indic did not move"],
    ["Reasoning 6% -> 9%", "the reasoning val set has 100% train leakage - it measured memorisation"],
  ],
  "verdict": "Nine rounds, sixteen runs, every attempted improvement withdrawn. The shares in "
             "section 1 stand exactly as first defended.",
}

dashboard = {"findings": findings, "model": plan["model"], "budget_B": plan["budget_B"], "budget_future_B": plan["budget_future_B"],
             "proxy_lanes": proxy_lanes, "proxy_note": plan["proxy_note"],
             "sets": sets, "full_plan": plan["full_plan"], "verdicts": verdicts}
json.dump(dashboard, open(os.path.join(OUT, "dashboard.json"), "w"), ensure_ascii=False, indent=2)
scored = [s["key"] for s in sets if "outcome" in s]
print(f"wrote site/data/dashboard.json · sets: {[s['key'] for s in sets]} · with outcomes: {scored} · verdicts {sum(v['ok'] for v in verdicts)}/{len(verdicts)}")
