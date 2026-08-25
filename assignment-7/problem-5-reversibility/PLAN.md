# Problem 5 - Reversibility: the plan agreed before coding

**ERA V5 - Assignment 7 - Problem 5 - Ashish Kamble**

## The problem, as the assignment states it

> "Kronecker is forward deterministic (same word will always give same embedding). How do I make a
> reverse of this (same embedding gives the same Kronecker)? If we can do this, then we can get rid
> of the final head as well! Then we can have a vocab of 1M as well without any issues!"

Two payoffs are named there, and a third follows from them: if the mapping inverts, a model can be
made to predict a whole multi-character span in one step instead of one token at a time.

## The objection that makes this look impossible

Reversibility of the *codec* is easy to state and easy to doubt, because a trained network never
emits the exact numbers the codec produced. If the codec encodes a token as `0.30, 0.20, 0.10, ...`
the network will emit something like `0.31, 0.18, 0.09, ...`. Close, but not equal. The natural
conclusion is that exact inversion is unreachable, and the natural remedy is to stop predicting a
point and start predicting a distribution, which is the variational autoencoder move: predict a mean
plus a variance and treat the answer as a point cloud rather than a point.

A second objection compounds it. At random initialisation the predicted vector is nowhere near any
real token, so there is nothing to decode and no gradient signal telling the model which direction
would help.

This work argues both objections are artefacts of an assumed decoding rule and an assumed loss
function, and that neither survives contact with the actual construction.

## The claim this work will defend

**The Kronecker codec is not the thing that is broken, and the point cloud is not needed.**

The construction is

```
κ(b) = (1/√L) · vec( Σ_{p=1..L} c_{b_p} ⊗ p_p )
```

with `c_v` a one-hot in ℝ²⁵⁶ (byte value) and `p_p` a one-hot in ℝ^L (byte position). So κ is a
256 × L binary matrix, one 1 per occupied column, flattened and scaled. Decoding is therefore
`reshape(256, L)` followed by an **argmax down each position column**.

Argmax does not need the predicted numbers to be right. It needs the *ranking within a column* to be
right. Predicting 0.31 where the codec says 0.30 changes nothing at all, as long as the correct byte
is still the column maximum. The exactness problem does not exist at the codec.

What is genuinely irreversible is the learned projection `Linear(D → d_model)`. It maps 8192
dimensions into 768 and destroys information by construction; no left inverse exists. **The
irreversibility everyone is stuck on lives in the projection, not in the Kronecker code.**

That reframing turns the problem from "model the error distribution of a real-valued prediction"
into "stop routing the prediction through the bottleneck": predict in byte space with a
`d_model → 256 × L` head under per-position byte cross-entropy.

## The six experiments

Every claim below is a number produced by a script and written to `artifacts/`, never a number
typed into prose. Same discipline as assignment 6.

### E1 - The codec inverts exactly

Round-trip every token in the assignment-2 tokenizer vocabulary: token → bytes → κ → reshape →
argmax → bytes → token. Report exact-match rate, and separately the rate for tokens that fit the
window versus tokens that overflow it.

**Expected:** 100% for tokens of at most L bytes. Any failure is a truncation collision, not a codec
failure, and belongs to Problem 3.

### E2 - Inversion survives approximate prediction (the headline)

Take the true κ for each token, add Gaussian noise of standard deviation σ, decode, measure
exact-token accuracy as σ sweeps from 0 upward. Also sweep the *relative* noise σ/(1/√L) so the
result is scale free.

Report the analytic decision margin alongside the measured curve: a column decodes correctly while
the noise gap between the true byte and the best competing byte stays below the one-hot gap, so the
tolerated noise is predictable rather than empirical luck.

**This is the figure that answers the first objection.** If exact-token accuracy is still high at
noise many times larger than the 0.30-versus-0.31 error described above, the premise that motivates
the variational direction is measurably false.

### E3 - Locate the irreversibility precisely

- Rank of `W: 8192 → 768` is at most 768; nullspace dimension 7424. Verify numerically.
- Reconstruct κ from `W κ` via the Moore-Penrose pseudo-inverse, decode, measure accuracy.
- Construct explicit collisions: distinct tokens whose projections coincide to machine precision by
  adding a nullspace vector, and confirm the decoder cannot tell them apart.

**Expected:** near-chance decode accuracy through the projection, against ~100% at the codec. This is
the experiment that says *which component to fix*, and it is the part of the analysis that is
missing from the current discussion.

### E4 - The fix: a byte-space output head

Train the assignment-6 NumPy transformer on a held-out slice with three output heads, identical in
every other respect, identical seeds:

| Head | Parameters at d_model=768 | Scales with vocab? |
|---|---|---|
| A. Vocab softmax (baseline) | `768 × 131072` ≈ 100.7M | yes |
| B. Untied byte head `d_model → 256×L` | `768 × 8192` ≈ 6.3M | **no** |
| C. Tied byte head, unembedding = `Wᵀ` | **0 new parameters** | **no** |

Measured: next-token exact accuracy, per-position byte accuracy, loss curves, and parameters. The
demo runs at the small scale the machine allows; the parameter table is reported at both the demo
scale and the paper's 768-dimensional scale, clearly labelled which is measured and which is
arithmetic.

### E5 - Gradient exists from step 0

The second objection above: at random initialisation there is nothing close enough to decode, so
training cannot get started.

That is true for cosine or MSE matching against a target vector, and false for per-position
cross-entropy, which has a dense informative gradient at random init. Measure decode accuracy and
loss **from step 0**, and compare against an MSE-to-κ regression head trained identically, which is
what the objection implicitly assumes.

**Expected:** the CE head climbs off the floor immediately; the MSE head does not. This turns "impossible
at init" into "impossible *with that loss function*".

### E6 - What tying costs, honestly

The risk in head C is that per-position independence lets the model emit byte strings that are not
valid tokens, and that a tied projection has too few effective output degrees of freedom.

Measure both rather than assert either: rate of decoded byte strings that are not valid UTF-8, rate
that are valid UTF-8 but not in vocabulary, and whether those out-of-vocabulary strings are
plausible words. Out-of-vocabulary output is the *mechanism* behind the 1M-vocab claim, so this
number is reported as a property, not hidden as an error rate.

## What is deliberately not claimed

- Nothing here is validated at 124M parameters. The machine has 8 CPU cores and no GPU. Every
  training number is small scale and labelled as such, and the parameter counts at 768 are
  arithmetic on the construction, not measurements.
- The paper's own limitation - byte-similar but semantically distant tokens such as compute/commute
  cluster together - applies to the byte head too, and arguably more, since the output is now also
  byte structured. This is measured in E6, not argued away.
- KL divergence and the variational direction are not implemented. The argument is that E2 removes
  the motivation for them. If E2 came out the other way, that direction would be the right one.

## Deliverables

- `run_demo.py` - one command, no network, regenerates everything in `artifacts/`
- `artifacts/evidence.json` and `evidence.md` - every number, recomputed from artefacts
- `site/` - static dashboard, no framework, no build step, hand-built inline SVG, light and dark
- `README.md` - the writeup, with the six experiments in order
- `tests/` - invariants: codec round trip, decoder equivalence, nullspace collision, determinism
