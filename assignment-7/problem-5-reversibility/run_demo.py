"""
One command, no network, regenerates everything in `artifacts/`.

    python run_demo.py            # the codec experiments, about half a minute
    python run_demo.py --full     # adds the training experiments, roughly an hour on 16 CPU cores

E1 to E3 need no model at all: they are properties of the codec over the assignment-2 vocabulary and
they reproduce exactly. E4 to E7 train small transformers and are separated behind a flag because
they are slow and because they are where this problem's evidence is weakest.
"""
import argparse, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable

FAST = [("E1 to E3, round trip, noise tolerance, and the projection", "exp_codec.py")]
SLOW = [("E4 to E6, three heads, gradient at init, the cost of tying", "exp_train.py"),
        ("the open caveat, injectivity and a learned inverse under a trained W", "exp_recheck.py"),
        ("E7, can the head emit words it was never given an id for", "exp_openvocab.py"),
        ("E8, constrained decoding against the unconstrained argmax", "exp_constrained.py")]


def run(label, script):
    print(f"\n=== {label}\n    src/{script}", flush=True)
    t0 = time.time()
    r = subprocess.run([PY, os.path.join(HERE, "src", script)], cwd=HERE)
    if r.returncode != 0:
        raise SystemExit(f"{script} failed with exit code {r.returncode}")
    print(f"    done in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="also run the training experiments")
    args = ap.parse_args()
    for label, script in FAST + (SLOW if args.full else []):
        run(label, script)
    run("evidence, every number regenerated from the artefacts", "evidence.py")
    run("dashboard data, extracted from the same artefacts", "build_dashboard.py")
    print("\n=== provenance: do the artefacts match the code that is here now?",
          flush=True)
    subprocess.run([PY, os.path.join(HERE, "..", "common", "provenance.py")],
                   cwd=os.path.join(HERE, "..", "common"))
    if not args.full:
        print("\nTraining experiments skipped. Run with --full to include E4 to E7.")
