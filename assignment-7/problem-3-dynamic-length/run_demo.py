"""
One command, no network, regenerates everything in `artifacts/`.

    python run_demo.py            # the encoding experiments, about a minute
    python run_demo.py --full     # adds the training experiments, roughly an hour on 16 CPU cores

The encoding experiments (E1 to E4, E6, E7) are the load-bearing ones: they are properties of the
codec and the corpus, need no model, and reproduce exactly. The training experiments (E5, E5b) are
separated behind a flag because they are slow, they are the weakest evidence in this problem, and
nothing in the argument collapses without them.
"""
import argparse, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable

FAST = [("E1 to E3, occupancy, characters per window, truncation collisions", "exp_window.py"),
        ("E4, the three fixes at equal D, plus fix D", "exp_fixes.py"),
        ("E6, what the window costs in dimensions, memory and compute", "exp_cost.py"),
        ("E7, reading the word from both ends, hashing the overflow, and the window choice",
         "exp_bothends.py"),
        ("E7 verified as a codec rather than only as a key", "exp_bothends_codec.py")]
SLOW = [("E5 and E5b, downstream, token level and word level", "exp_downstream.py"),
        ("E5b at larger scale", "exp_e5b.py")]


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
        print("\nTraining experiments skipped. Run with --full to include E5 and E5b.")
