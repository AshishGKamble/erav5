"""
Tier experiment, step 2 - the question the rest of the study could not ask.

Every earlier run treated Indic as one uniformly clean bin, so "more Indic" was free. The
ledger says it is not. And once you write the ledger's arithmetic out, the real constraint
is sharper than "the marginal tokens are worse":

    organic Indic supply = 110B, repetition ceiling = 4 epochs  ->  440B reachable
    440B of a 4T budget  =  11.0% of the budget, AT EVERY INDIC SHARE

So native Indic is *pinned at 11%* whether the lane is 18% or 32%. Raising the Indic share
does not buy more native Indic - it is arithmetically impossible - it buys only translated
and synthetic tokens on top of a fixed native core.

That turns the open question into a crisp one:

    Does adding 12 more points of SYNTHETIC Indic (7% -> 19%) improve the model's
    capability on NATIVE Indic, which is what MILU and IndicGenBench actually measure?

Design. All mixtures hold native Indic (indic_hi) at ~11%. They differ in how much
translated/synthetic Indic (indic_lo) sits on top, and what that mass is taken from.
Every mixture is scored on the SAME held-out NATIVE Indic set - the benchmark is native
Indic, so that is the metric that matters.

  tier18        hi 11 / lo  7  - the shipped 18% Indic lane, at the ledger's real tier ratio
  tier30        hi 11 / lo 19  - the 30% lane the proxy asked for, at its real tier ratio
  tier30_ideal  hi 30 / lo  0  - the 30% lane as the EARLIER PROXY IMAGINED IT (all clean).
                                 Impossible to supply; included to show what that result was
                                 actually measuring.

Run: python3 proxy/tier_train.py --name tier30 --seed 7
"""
import os, json, time, argparse, math
import numpy as np, torch
from model import GPT, GPTConfig

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
RUNS = os.path.join(HERE, "runs_tier"); os.makedirs(RUNS, exist_ok=True)
torch.set_num_threads(int(os.environ.get("THREADS", "8")))

LANES = ["web", "code", "math", "reasoning", "indic_hi", "indic_lo"]

MIXES = {
    "tier18":       {"web": .35, "code": .21, "math": .12, "reasoning": .14, "indic_hi": .11, "indic_lo": .07},
    "tier30":       {"web": .30, "code": .18, "math": .10, "reasoning": .12, "indic_hi": .11, "indic_lo": .19},
    "tier30_ideal": {"web": .30, "code": .18, "math": .10, "reasoning": .12, "indic_hi": .30, "indic_lo": .00},
}


def load_bins():
    tr = {l: np.memmap(os.path.join(DATA, f"{l}_train.bin"), dtype=np.uint16, mode="r") for l in LANES}
    va = {l: np.memmap(os.path.join(DATA, f"{l}_val.bin"), dtype=np.uint16, mode="r") for l in LANES}
    return tr, va


def get_batch(bins, probs, block, bs):
    xs, ys = [], []
    for li in np.random.choice(len(LANES), size=bs, p=probs):
        d = bins[LANES[li]]
        i = np.random.randint(0, len(d) - block - 1)
        c = np.asarray(d[i:i + block + 1], dtype=np.int64)
        xs.append(c[:-1]); ys.append(c[1:])
    return torch.tensor(np.stack(xs)), torch.tensor(np.stack(ys))


@torch.no_grad()
def eval_per_lane(model, va, block, iters=25):
    model.eval(); out = {}
    for l in LANES:
        d = va[l]
        if len(d) < block + 2: out[l] = float("nan"); continue
        losses = []
        for _ in range(iters):
            i = np.random.randint(0, len(d) - block - 1)
            c = np.asarray(d[i:i + block + 1], dtype=np.int64)
            _, loss = model(torch.tensor(c[:-1])[None], torch.tensor(c[1:])[None])
            losses.append(loss.item())
        out[l] = round(float(np.mean(losses)), 4)
    model.train()
    out["_avg"] = round(float(np.mean([v for k, v in out.items() if not k.startswith("_")])), 4)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, choices=list(MIXES))
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--block", type=int, default=128)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--n_embd", type=int, default=192)
    ap.add_argument("--n_layer", type=int, default=4)
    ap.add_argument("--eval_every", type=int, default=300)
    ap.add_argument("--seed", type=int, default=1337)
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)

    tr, va = load_bins()
    meta = json.load(open(os.path.join(DATA, "meta.json")))
    mix = MIXES[a.name]
    probs = np.array([mix[l] for l in LANES], dtype=np.float64); probs /= probs.sum()
    cfg = GPTConfig(vocab_size=meta["vocab_size"], block_size=a.block,
                    n_layer=a.n_layer, n_head=6, n_embd=a.n_embd)
    model = GPT(cfg)
    print(f"[{a.name} seed={a.seed}] params={model.num_params()/1e6:.2f}M  probs={dict(zip(LANES, probs.round(3)))}")
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1, betas=(0.9, 0.95))

    def lr_at(s):
        if s < 100: return 3e-4 * s / 100
        r = (s - 100) / max(1, a.steps - 100)
        return 3e-5 + 0.5 * (3e-4 - 3e-5) * (1 + math.cos(math.pi * r))

    hist = []; t0 = time.time()
    for step in range(a.steps + 1):
        if step % a.eval_every == 0:
            ev = eval_per_lane(model, va, a.block); ev["step"] = step
            hist.append(ev)
            print(f"  step {step:>4}  native(indic_hi)={ev['indic_hi']}  avg={ev['_avg']}")
        for g in opt.param_groups: g["lr"] = lr_at(step)
        x, y = get_batch(tr, probs, a.block, a.bs)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    f = hist[-1]
    res = {"name": a.name, "seed": a.seed, "mixture": mix,
           "lane_probs": dict(zip(LANES, probs.round(4).tolist())),
           "params_M": round(model.num_params() / 1e6, 2), "steps": a.steps,
           "final_per_lane": {l: f[l] for l in LANES}, "final_avg": f["_avg"], "history": hist}
    tag = "" if a.seed == 1337 else f"_seed{a.seed}"
    json.dump(res, open(os.path.join(RUNS, f"{a.name}{tag}.json"), "w"), indent=2)
    print(f"[{a.name} seed={a.seed}] DONE {time.time()-t0:.0f}s  "
          f"NATIVE Indic loss = {f['indic_hi']}  (avg {f['_avg']})")


if __name__ == "__main__":
    main()
