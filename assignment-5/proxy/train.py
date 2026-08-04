"""
Proxy step 3 - train the tiny GPT under a chosen data MIXTURE and measure per-lane
held-out loss. This is the whole point: a mixture is a hypothesis, and per-lane
val loss is the metric that confirms or refutes it. Up-weighting a lane should lower
that lane's held-out loss; starving a lane (e.g. web) should raise it - the session's
"good at code, no common sense" tradeoff, made measurable.

Usage: python3 proxy/train.py --name ours --steps 2000
"""
import os, json, time, argparse, math
import numpy as np, torch
from model import GPT, GPTConfig

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data"); RUNS = os.path.join(HERE, "runs"); os.makedirs(RUNS, exist_ok=True)
# Seed is set in main() from --seed so the same mixture can be re-run to measure the
# eval noise floor. Without that number, a 0.07 delta between two mixtures is unreadable.
torch.set_num_threads(int(os.environ.get("THREADS", "8")))

# Candidate mixtures over the five proxy lanes. OUR reasoning, not copied numbers:
# 'ours' renormalises our integrated plan (web30/code22/math10/reasoning6/indic18) over
# the proxy's five lanes and keeps the Indic protected floor (~20%). The two baselines
# are the failure modes the session warns about.
MIXES = {
    "naive_web":  {"web": .70, "code": .15, "math": .05, "reasoning": .05, "indic": .05},
    "ours":       {"web": .35, "code": .25, "math": .12, "reasoning": .08, "indic": .20},
    "code_heavy": {"web": .08, "code": .55, "math": .25, "reasoning": .07, "indic": .05},
    # --- round 2: each set isolates one question the first three left open ---
    # Q: can Indic be the LARGEST lane without costing the lanes underneath it?
    "indic_first":   {"web": .28, "code": .18, "math": .10, "reasoning": .12, "indic": .32},
    # Q: round 1 showed reasoning is NOT monotone in its own share - but the set that
    #    raised it also starved web+Indic. Raise reasoning with the base held intact.
    "reasoning_fwd": {"web": .35, "code": .15, "math": .10, "reasoning": .20, "indic": .20},
    # Q: controlled test of the web->Indic scaffolding effect. Indic and reasoning are
    #    pinned to 'ours' (.20/.08); ONLY web moves (.35 -> .20). If Indic loss rises,
    #    web is holding Indic up and cutting web to fund Indic is self-defeating.
    "web_lean":      {"web": .20, "code": .35, "math": .17, "reasoning": .08, "indic": .20},
    # --- round 3: the revision round 2 argues for (README section 14), renormalised from the
    #     7-lane proposal web26/code20/math10/reasoning9/agentic8/longctx6/indic21 onto the five
    #     proxy lanes. Moves Indic and reasoning UP, web and code DOWN - each backed by a round-2
    #     result, each stopping short of the proxy's optimum because supply caps it.
    "v2_proposed":   {"web": .30, "code": .23, "math": .12, "reasoning": .10, "indic": .25},
    # --- round 4: v2_proposed came out level with 'ours' and Indic got slightly WORSE despite a
    #     bigger Indic share. Suspected cause: v2 funded Indic partly out of WEB, and round 2 showed
    #     web scaffolds Indic - so the two moves cancelled. Every Indic test so far has changed web
    #     at the same time, which confounds them. This run isolates it: web and reasoning are pinned
    #     to 'ours' (.35/.08) and Indic 20->30 is funded ENTIRELY from code and math.
    "indic_clean":   {"web": .35, "code": .17, "math": .10, "reasoning": .08, "indic": .30},
    # --- round 6: tune the only mixture that beat 'ours' above the noise floor. `indic_first`
    #     was web 28 / indic 32; this trades 2 points of Indic back into web to test whether the
    #     win survives a slightly less extreme Indic share and a fuller web base. Its other lanes
    #     (18/10/12) already sum to 40, so this is a clean two-parameter move. Run at 2 seeds:
    #     round 5 taught us a single run cannot be trusted at this scale.
    "indic30_web30": {"web": .30, "code": .18, "math": .10, "reasoning": .12, "indic": .30},
    # --- round 7: round 6 moved Indic AND web together, so we could not attribute its 0.047 cost.
    #     This holds Indic at indic_first's 32% and raises web 28->30 alone, compensating from code
    #     (18->16). On a simplex nothing can move in true isolation; naming the compensator is the
    #     honest version. If this matches indic_first, web was not the cost and the Indic 32->30 drop
    #     was; if it matches indic30_web30, the web change was.
    "web30_indic32": {"web": .30, "code": .16, "math": .10, "reasoning": .12, "indic": .32},
}


def load_bins():
    meta = json.load(open(os.path.join(DATA, "meta.json")))
    lanes = list(meta["lanes"])
    tr = {l: np.memmap(os.path.join(DATA, f"{l}_train.bin"), dtype=np.uint16, mode="r") for l in lanes}
    va = {l: np.memmap(os.path.join(DATA, f"{l}_val.bin"), dtype=np.uint16, mode="r") for l in lanes}
    return lanes, tr, va, meta


def get_batch(bins, lane_probs, lanes, block, bs):
    xs, ys = [], []
    picks = np.random.choice(len(lanes), size=bs, p=lane_probs)
    for li in picks:
        d = bins[lanes[li]]
        i = np.random.randint(0, len(d) - block - 1)
        chunk = np.asarray(d[i:i + block + 1], dtype=np.int64)
        xs.append(chunk[:-1]); ys.append(chunk[1:])
    return torch.tensor(np.stack(xs)), torch.tensor(np.stack(ys))


@torch.no_grad()
def eval_per_lane(model, va, lanes, block, bs, iters=25):
    model.eval(); out = {}
    for l in lanes:
        d = va[l]
        if len(d) < block + 2: out[l] = float("nan"); continue
        losses = []
        for _ in range(iters):
            i = np.random.randint(0, len(d) - block - 1)
            chunk = np.asarray(d[i:i + block + 1], dtype=np.int64)
            x = torch.tensor(chunk[:-1])[None]; y = torch.tensor(chunk[1:])[None]
            _, loss = model(x, y); losses.append(loss.item())
        out[l] = round(float(np.mean(losses)), 4)
    model.train()
    out["_avg"] = round(float(np.mean([v for k, v in out.items() if not k.startswith("_")])), 4)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, choices=list(MIXES))
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--block", type=int, default=256)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--n_embd", type=int, default=288)
    ap.add_argument("--n_layer", type=int, default=6)
    ap.add_argument("--eval_every", type=int, default=250)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--tag", default="", help="suffix for the output filename (e.g. seed sweeps)")
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)

    lanes, tr, va, meta = load_bins()
    mix = MIXES[a.name]
    lane_probs = np.array([mix.get(l, 0.0) for l in lanes], dtype=np.float64)
    lane_probs = lane_probs / lane_probs.sum()
    cfg = GPTConfig(vocab_size=meta["vocab_size"], block_size=a.block,
                    n_layer=a.n_layer, n_head=6, n_embd=a.n_embd)
    model = GPT(cfg)
    print(f"[{a.name}] params={model.num_params()/1e6:.1f}M  lanes={lanes}  probs={dict(zip(lanes, lane_probs.round(3)))}")
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1, betas=(0.9, 0.95))

    def lr_at(step):
        warm = 100
        if step < warm: return 3e-4 * step / warm
        r = (step - warm) / max(1, a.steps - warm)
        return 3e-5 + 0.5 * (3e-4 - 3e-5) * (1 + math.cos(math.pi * r))

    hist = []; t0 = time.time()
    for step in range(a.steps + 1):
        if step % a.eval_every == 0:
            ev = eval_per_lane(model, va, lanes, a.block, a.bs)
            ev["step"] = step; ev["sec"] = round(time.time() - t0, 1); hist.append(ev)
            print(f"  step {step:>4}  avg {ev['_avg']}  " + " ".join(f"{l}={ev[l]}" for l in lanes))
        for g in opt.param_groups: g["lr"] = lr_at(step)
        x, y = get_batch(tr, lane_probs, lanes, a.block, a.bs)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    final = hist[-1]
    result = {"name": a.name, "mixture": mix, "lane_probs": dict(zip(lanes, lane_probs.round(4).tolist())),
              "params_M": round(model.num_params() / 1e6, 2), "steps": a.steps,
              "tokens_seen_M": round(a.steps * a.bs * a.block / 1e6, 2),
              "config": {"block": a.block, "bs": a.bs, "n_embd": a.n_embd, "n_layer": a.n_layer},
              "final_per_lane": {l: final[l] for l in lanes}, "final_avg": final["_avg"], "history": hist}
    result["seed"] = a.seed
    json.dump(result, open(os.path.join(RUNS, f"{a.name}{a.tag}.json"), "w"), indent=2)
    print(f"[{a.name}] DONE in {time.time()-t0:.0f}s  final per-lane: " +
          " ".join(f"{l}={final[l]}" for l in lanes))


if __name__ == "__main__":
    main()
