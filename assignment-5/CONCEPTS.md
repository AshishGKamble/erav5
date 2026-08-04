# How data mixtures and curricula actually work - a plain-language reference

My own notes for Assignment 5. Not the deliverable (that is [README.md](README.md)); this is the
"why" behind every number there, written so I can re-read it in six months and still make good
decisions. Every example uses the real numbers from [`plan.json`](plan.json).

---

## Part 0. The one idea everything hangs off

**You have a fixed number of tokens. What you feed the model IS the model.**

Think of it as raising a child with a fixed number of school days. You cannot add days. Every hour
you spend on chemistry is an hour not spent on Hindi. So the only real question is:

> Given exactly 4 trillion tokens, how do I divide them so the model is good at the things I said
> it must be good at?

That division is called the **mixture**. It is the highest-leverage decision in the whole project,
because architecture changes give you a few percent and mixture changes give you tens of percent.

Two follow-up questions turn it into a real plan:
1. **What proportions?** (the mixture) - Parts 1 to 4.
2. **In what order?** (the curriculum) - Parts 5 and 6.

Everything else in the plan is a guard-rail protecting those two answers.

---

## Part 1. The vocabulary, in one line each

| Word | Plain meaning |
|---|---|
| **Lane** | One kind of data, grouped by the capability it buys. "Code" is a lane. "Indic" is a lane. |
| **Share** | What % of the total budget that lane gets. Web = 30% means 30% of 4T = 1200B tokens. |
| **Demand** | The tokens a share requires. 30% of 4T = **1200B demanded**. |
| **Supply** | The tokens that actually **exist in the world** for that lane. |
| **Epochs** | How many times you must re-read your supply to meet demand. Demand 400B on supply 143B = **2.8 epochs**. |
| **Generation** | When even repeating is not enough, you must **create** the data (synthesise or distill from a teacher model). |
| **Trainable fraction** | Of the tokens you feed in, what % the model actually learns from. Not always 100% (see below). |
| **Band** | A difficulty rung, B0 (nursery) to B5 (PhD). |
| **Stage** | A phase of the run with its own mixture, sequence length, and difficulty band. |

### The one that trips people up: trainable fraction

For plain web text, every token teaches the model something, so **trainable fraction = 1.00**.

For an agentic trace it is different. A trace looks like this:

```
user:      "fix the failing test"
assistant: "let me read the file"      <- model learns from this
tool:      <4000 lines of file dump>   <- model must SEE this but not imitate it
assistant: "the bug is the off-by-one" <- model learns from this
```

You **mask the loss** on tool output: the model reads it as context but is not trained to produce
it. In our plan agentic is `trainable_frac: 0.35`, so **8% of tokens is only about 2.8% of actual
gradient**. If you forget this, you will think you gave agentic twice the training you really did.

**Rule of thumb:** pretraining text = 100% trainable. Agentic/SFT/chat = only the assistant's own
tokens are trainable.

---

## Part 2. How to choose a lane's share - the five questions

Do not start from "what feels balanced". Start from **the benchmark you promised to win**, and work
backwards. For each lane ask these five, in order:

**Q1. What breaks if this lane is zero?**
If nothing breaks, it is not a lane. Web at 0% means no common sense. Indic at 0% means the whole
product thesis dies. That is what makes them lanes.

**Q2. Which benchmark does it move?**
Every lane must name one. Code moves LiveCodeBench and SWE-bench. Indic moves MILU and
IndicGenBench. **A lane that cannot name its benchmark is a lane you cannot defend**, and a
reviewer will find it immediately.

**Q3. Is it my differentiator or my hygiene?**
- **Differentiator** = the reason someone picks my model. Ours: code, agentic, Indic. Push these
  **above** what a general model would give them.
- **Hygiene** = if it is missing the model looks stupid, but it wins me nothing. Ours: general web.
  Give these the **minimum that avoids embarrassment**, not more.

This single distinction produced most of our numbers. A general assistant spends ~50% on web. We
spend **30%** and move the difference into code + agentic + Indic. That is the entire thesis in one
trade.

**Q4. What is the floor below which it collapses?**
Capabilities do not decay smoothly, they fall off a cliff. Our judgment: web below ~30% starts
costing MMLU-Pro. So 30% is not "web's fair share", it is **the cheapest web I can get away with**.
Say it that way and the number becomes defensible instead of arbitrary.

**Q5. Does the data actually exist?** (Part 3, and it is where most plans quietly lie.)

### Worked example: why Indic is 18%

- Q1: at 0%, no product. It is the whole point.
- Q2: MILU, IndicGenBench, MMMLU-Indic.
- Q3: differentiator, so push it up hard.
- Q4: Indic's natural share of web crawl is about **1%**. Training at 1% gives a model that has
  heard Hindi, not one that thinks in it.
- Q5: only ~110B organic Indic tokens exist. 18% of 4T = 720B. So **most of it must be translated
  or synthetic**, and I have to say so out loud.

18% is roughly 18x its natural share. That is the size of the deliberate bet, stated as a number.

---

## Part 3. The supply reality check - where plans lie

This is the part that separates a real plan from a wish. Writing "agentic: 8%" costs one keystroke.
Finding 320B agentic tokens is close to impossible.

For every lane, compute `demand / supply`:

| Result | What it means | Verdict |
|---|---|---|
| **< 1.0** | More data exists than you need | Honest. Just use it. |
| **1.0 to 4.0** | You must re-read the same data 1-4 times | Acceptable, with a cap. |
| **> 4.0** | Repeating this much starts memorising, not learning | You must **generate** the shortfall. |

**The 4-epoch rule.** Roughly: up to about 4 passes over the same data, repeated tokens are worth
almost as much as fresh ones. Past that, returns collapse and you are burning budget teaching the
model to recite. So 4 epochs is the ceiling, and anything beyond it must be honestly labelled
as generated.

Our lanes, from [`supply/ledger.md`](supply/ledger.md):

| Lane | Demand | Supply | Epochs | Honest verdict |
|---|---:|---:|---:|---|
| Web | 1200B | 20300B | 0.06 | Swimming in it |
| Code | 880B | 1813B | 0.49 | Comfortable |
| Math + STEM | 400B | 143B | **2.8** | Repeat, under the cap |
| Reasoning | 240B | 38B | **4.0** | At the cap **+ 88B generated** |
| Agentic | 320B | 43B | **4.0** | At the cap **+ 150B generated** |
| Long-context | 240B | 240B | 1.0 | Exactly enough (60% already synthetic) |
| Indic | 720B | 778B | 0.93 | Fine in total, **but see the tiers** |

**Total that must be generated: 238B, or 5.9% of the entire budget.**

Two lessons I want to keep:

**1. Say the uncomfortable number.** A plan that writes "agentic: 8%" without admitting 150B of it
does not exist is exactly the wishful accounting this assignment is testing for. Declaring it costs
nothing and buys all the credibility.

**2. A healthy total can hide a sick tier.** Indic looks fine at 0.93 epochs. But split it apart:

| Tier | What it is | Demand | Supply | Reality |
|---|---|---:|---:|---|
| **Verified** | Human-checked. UPSC, NCERT, Constitution. | 130B | 86B | 1.52 epochs. **The scarce gold.** |
| **Unverified** | Crawl, filtered by perplexity only | 101B | 24B | 4 epochs, still short |
| **Translated** | Machine-translated English, quality-gated | 238B | 305B | Fine |
| **Synthetic** | Model-generated Indic | 252B | 363B | Fine |

So the honest headline is: **~68% of my Indic lane is translated or synthetic, and truly verified
Indic is only 3.2% of the whole budget.** One number ("Indic 18%") would have hidden all of that.

**And scaling the budget does not fix it.** At 20T instead of 4T, verified Indic is still ~86B,
because that is all that exists. So the ratio gets *worse*, not better. Scarce tiers are
supply-bound, not budget-bound. That is a genuinely counter-intuitive point and worth remembering.

---

## Part 4. Candidate sets - how to choose between mixtures

A **set** is one complete recipe. Not a tweak, a full alternative philosophy.

The mistake is to pick one mixture, defend it, and ship. You have no idea if it is good, because
you never looked at an alternative. Instead, write down 2-4 genuinely different sets and **let a
cheap experiment pick the winner**.

### How to design a good set of sets

Cover the **spectrum of the argument**, not three shades of the same opinion:

| Set | Recipe (web/code/math/reasoning/Indic) | The argument it represents |
|---|---|---|
| **Web-heavy** | 70 / 15 / 5 / 5 / 5 | "Just do what everyone else does." The lazy default. |
| **Balanced** (our pick) | 35 / 25 / 12 / 8 / 20 | "Protect Indic, push code, keep enough web." |
| **Code-forward** | 8 / 55 / 25 / 7 / 5 | "We are a coding model, go all in." The over-correction. |

Notice the shape: **the baseline, my proposal, and the over-correction of my proposal.** My pick
sits in the middle, so if it wins, it wins against a real argument on each side. If I had only
compared 35/25 against 33/27, I would have proved nothing.

**Rules for building a set:**
- Every set must sum to 100. There is no free budget.
- Sets must **differ enough to produce a visible difference** in the result. A 2% change is noise.
- Every set should be one somebody could seriously advocate. A deliberate straw man teaches nothing.
- Change several lanes at once. These are *philosophies*, not one-variable ablations.

### How to judge the winner

Train a small model on each set, then measure **held-out loss per lane**: take text the model has
never seen, in each lane, and measure how surprised it is. **Lower = better at that lane.**

Our actual results (4.85M params, 1500 steps):

| Mixture | code | indic | math | reasoning | web | avg |
|---|---:|---:|---:|---:|---:|---:|
| Web-heavy | 5.696 | 4.981 | 6.733 | 5.387 | **6.312** | 5.822 |
| **Balanced (ours)** | 5.475 | **4.407** | 6.649 | **5.273** | 6.541 | **5.669** |
| Code-forward | **5.208** | 5.473 | **6.321** | 5.482 | 6.754 | 5.848 |

**How to read this table - four habits, learned the hard way:**

**1. Read the signs, not the magnitudes.** A 4.85M model is a toy. Loss of 5.67 means nothing in
absolute terms. What transfers to real scale is the **direction**: does raising a lane's share
lower that lane's loss? That is the claim.

**2. Measure the noise floor BEFORE you interpret anything.** This is the habit that cost me the
most to learn. Re-run the *identical* mixture at two or three different seeds and see how much the
numbers move on their own. Ours moved by **0.14 to 0.35 per lane** while the *average* moved only
0.05. I had spent four rounds reading differences of 0.07 to 0.12 as signal - most of it was noise,
and three published findings had to be withdrawn. **A study that reports one run per mixture is
reporting its seed.** Quote every difference against its lane's floor, and prefer the average, which
is far steadier than any single lane.

**3. Check that your metric measures what you think it does.** Two separate failures taught this:

- Our "Indic" validation set was **98.5% machine-translated and synthetic**, so for six rounds
  "Indic loss improved" actually meant "got better at translated text" - not the native Indic that
  MILU scores. Splitting the bin by provenance reversed the conclusion entirely.
- Our reasoning validation set had **100% overlap with training data**, because the data prep looped
  the source six times before splitting off the first 5%. That "held-out" loss was measuring
  memorisation.

**Before trusting any number, ask two questions: what distribution is this validation set actually
drawn from, and is it genuinely unseen?** Sampling 200 windows and grepping for them in the training
data takes minutes and would have saved me a week.

**4. Prefer the effect you predicted in advance.** Write your prediction down before the run.
A number that confirms a trade you designed on purpose is worth far more than a number that is
merely best, because the second kind is what overfitting to noise looks like.

### Predicting before you look

Write down what you expect *before* you run it. Ours were:
- Raising Indic 5% to 20% will lower Indic loss. → confirmed, delta 0.575
- Starving web to 8% will raise web loss. → confirmed, delta 0.213
- Raising code will lower code loss. → confirmed, delta 0.488

3/3. If a prediction had failed, that is the **most valuable** possible outcome, because I would
have learned my model of the world was wrong for the price of a CPU run instead of a 4T run.

**This is the whole point: a mixture is a hypothesis, not a preference.** Cheap test first, then
1B, then 3B, then commit the real budget.

---

## Part 5. Curriculum - the order, and how to choose stage weights

Same tokens, different order, different model. You do not teach calculus in week one.

We use five stages. Each has its own mixture, sequence length, and difficulty band:

| Stage | Budget | Seq | Band | Its job |
|---|---:|---|---|---|
| **Seed** | 8% | 4K | B0-B1 | Learn what language even is. Get the basics of every script in early. |
| **General** | 45% | 4-8K | B1-B3 | Build the broad base. This is the bulk of the run. |
| **Reasoning** | 25% | 8-16K | B3-B4 | Now that it knows facts, teach it to think. Introduce agentic. |
| **Long-context** | 19% | 16-32K | B4-B5 | Stretch attention over long inputs. Full agent trajectories. |
| **Anneal** | 3% | 32K | B5 | Low learning rate, only the best data. The finishing polish. |

### How to choose the stage weights

- **Seed is small (8%)** because it is only a warm start. Too much nursery data and you waste
  budget on stuff the model learned in the first hour.
- **General is biggest (45%)** because breadth takes the most tokens. Knowledge is the thing that
  genuinely needs volume.
- **Reasoning and long-context are the back half (25% + 19%)** because they only pay off once the
  base exists. Long documents given to a model that cannot think are just expensive noise.
- **Anneal is tiny (3%)** because at a near-zero learning rate the model barely moves. You only get
  a small nudge, so you make it the highest-quality nudge available.

**The ordering principle, in one line:** *capabilities that depend on other capabilities come
later.* You cannot ask for a long, structured answer before the model can think. So reasoning comes
before long-context, always.

### Reading the stage table

Each stage column sums to 100 (it is that stage's own mixture). The **integrated** share is the
budget-weighted average, and it must reproduce Part 2's numbers:

```
web = .08(55) + .45(38) + .25(22) + .19(14) + .03(10) = 30.0  ✓
```

Worth doing this arithmetic by hand once. It is the check that the curriculum and the mixture are
the same plan and not two documents that drifted apart.

The **shape** matters more than any single cell:
- **Web fades 55 → 10.** Common sense is foundational, so front-load it, then hand the budget over.
- **Code, agentic, long-context climb.** They need a base to build on.
- **Indic never drops below 18%.** Flat by design (Part 7).
- **Reasoning peaks in the anneal at 16%.** The hardest material lands when the model is most able
  to absorb it.

### Warm-up seams - the practical detail that saves the run

If you switch mixture abruptly at a stage boundary, the loss spikes and gradients go unstable. So
at every seam, insert a short band (~0.5-1% of budget) that is a **60/40 blend of the old and new
mixtures**. The distribution diffuses instead of jumping.

Two related practical rules:
- **Uniform-length batches.** A batch is all-4K or all-8K, never mixed.
- **Never pad below 4K.** Padding is compute you paid for and threw away.

---

## Part 6. Two independent axes: difficulty and length

Easy to confuse, and they are genuinely separate knobs.

### Difficulty bands (B0-B5) - how hard the content is

| Band | Level | Knowledge example | Code example |
|---|---|---|---|
| B0 | Nursery | "The sun rises in the east." | `print("hello")` |
| B1 | Grade school | "12 mangoes, sold 5, how many left?" | a `for` loop summing a list |
| B2 | High school | "Balance: Fe + O2 → Fe2O3" | a function with input validation |
| B3 | Undergraduate | "Prove sum of first n odd numbers is n^2" | a REST handler with a unit test |
| B4 | Graduate | "Derive the softmax-cross-entropy gradient" | a multi-file refactor, tests green |
| B5 | Research | a FrontierMath problem | a real SWE-bench patch |

The curriculum walks up this ladder. Feed B5 too early and the model "consumes without learning" -
the data is wasted, and it was your most expensive data.

### Reasoning-length bands - how long the model thinks

| Band | Thinking budget | Example |
|---|---|---|
| short | under 64 tokens | "43 / 17 is about 2.5" - no scratch work |
| medium | 64-512 | "1 to 1000 divisible by 3 or 5? 333+200-66 = 467" |
| long | 512-4,000 | a geometry proof with case analysis |
| ultra | 4,000-32,000 | a research problem, or a long agentic debug with backtracking |

**Why this is a separate axis:** a hard problem can have a short answer, and an easy problem can be
explored at length. Difficulty is a property of the *problem*; length is a property of the
*response*.

**The key insight: depth is trained, not prompted.** "Think harder" only works if the model has
seen **the same kind of problem answered at several different lengths**, each tagged with its band.
Then the tag becomes a control it has actually learned to obey. Without that data, the instruction
is decoration.

High and ultra traces mostly do not exist in the wild, so they are **distilled from a teacher model
run at matching effort**. That is a real cost, and it is why reasoning shows 88B generated.

---

## Part 7. The protected floor - and why anything needs protecting

We use **OPUS-style online selection**: during training, look at each candidate batch, score how
well it aligns with a "golden proxy" of the target benchmarks, and keep roughly the best 50%. This
is very effective - about **8x token efficiency**, which is why the collected corpus (~1T) can
exceed the trained budget.

But there is a catch that changes the whole design:

> **OPUS judges a sample from only its first ~512 tokens.**

That is a cheap heuristic, and it fails badly on exactly two of our lanes:

- **Indic.** The golden proxy is English and code heavy. Indic text scores as poorly aligned and
  gets dropped. Left alone, **OPUS would quietly delete our differentiator** and we would discover
  it at the end. So: **floor of 14% of every batch, never trimmed.**
- **Agentic.** The first 512 tokens of an agent trace are plan boilerplate and tool setup. It reads
  like a log file, scores low, gets dropped. So OPUS throws away the data we have least of and
  paid the most for. So: **100% preserved, never trimmed.**

Everything else stays OPUS-eligible, because for those lanes dropping the easy redundant stuff is
exactly what I want.

**The transferable lesson:** any automatic filter has a blind spot, and the blind spot always
lands on the unusual data - which is usually the data you are differentiating on. So when you add a
selector, immediately ask **"what would this delete that I cannot afford to lose?"** and hard-code
protection for it. Efficiency tools optimise for the average; your differentiator is by definition
not the average.

Also: **re-run the selection every ~2B tokens**, because the proxy drifts. Once the model is good
at math, math stops being informative and something else becomes the frontier.

---

## Part 8. The anneal reserve - decide now, spend last

At the very end, drop the learning rate to near zero and train on a small amount of the best data
you have. This gives a benchmark lift out of proportion to its size, because the model is barely
moving and every remaining step goes into high-quality material.

**The critical part is the word "reserve".** You must hold this data back **at composition time**,
before the run starts. If you spend your best Indic in the General stage, you cannot conjure more
in the last 3%. It does not exist. The mistake is not "forgot to anneal", it is "had nothing left
to anneal with".

Our 120B (3%) reserve:

| Slice | Tokens | Why it waits |
|---|---:|---|
| Tier-A verified Indic | 30B | Scarcest tier; biggest lift when the model can absorb it |
| Hardest verified agentic traces | 30B | Newest capability, wasted early |
| Ultra-length reasoning | 25B | Only useful once the reasoning base exists |
| PhD / research STEM | 25B | B5 material, the "ready to absorb" window |
| Decontaminated benchmark-adjacent gold | 10B | Teaches the *format* of the task (train splits only) |

That last row needs a hard rule: **train splits only, test splits never**, and every shard scanned
against the eval suite. Learning the shape of a task is legitimate; memorising the answers is
cheating and will be found.

---

## Part 9. The parameter cheat-sheet

Everything I can turn, what it does, and how to choose it.

| Parameter | What it controls | How to choose | Danger sign |
|---|---|---|---|
| **Total budget** | Size of everything | Compute available, and the scaling law for your parameter count | Picking a round number with no reasoning behind it |
| **Lane share** | Capability strength | Backwards from the benchmark; differentiator up, hygiene to its floor | Cannot name the benchmark it moves |
| **Epochs** | Repetition of scarce data | Ceiling of 4; past that, generate instead | Above 4 and not admitted |
| **Generated tokens** | Filling a real gap | Only where supply genuinely fails; name the pipeline | Hidden inside a share |
| **Trainable fraction** | Gradient that actually lands | 1.0 for text, ~0.35 for agentic (tool output masked) | Counting masked tokens as training |
| **Stage weights** | Time on each phase | Bulk in General; small Seed; tiny Anneal | Anneal large enough to need a real LR |
| **Sequence length** | Context per sample | Double per stage, uniform batches, never pad under 4K | Mixed-length batches wasting compute |
| **Difficulty band** | Content hardness | Climb B0 to B5 in step with the stages | B5 in Seed = expensive data, wasted |
| **Reasoning length** | Thinking budget | Independent of difficulty; same problem at several lengths | Expecting "think harder" to work untrained |
| **Protected floor** | Selector guard-rail | Any lane the selector would misjudge | No floor, and the differentiator quietly vanishes |
| **OPUS keep fraction** | Aggressiveness of filtering | ~50%, re-run every ~2B tokens | Set once, never refreshed, proxy goes stale |
| **Anneal reserve** | Final polish | 2-3%, reserved *at composition time* | Discovered at the end, with nothing left |
| **Warm-up seam** | Stability at boundaries | ~0.5-1%, 60/40 old-new blend | Loss spike at every stage change |

---

## Part 10. How to actually change something

Say I want to test "what if Indic were 30% instead of 20%?"

1. **Check supply first.** 30% of 4T = 1200B. Organic Indic is 110B. So the extra is entirely
   translated and synthetic. Is a *more synthetic* Indic lane still worth it? That is a real
   question, and the ledger asks it before I spend anything.
2. **Add the set** to `sets` in [`plan.json`](plan.json) with a name and a thesis sentence.
3. **Add the same recipe** to `MIXES` in [`proxy/train.py`](proxy/train.py). The two must match or
   the dashboard compares a design to the wrong outcome.
4. **Write the prediction down** before running. "Indic loss drops below 4.407, web loss rises."
5. `bash proxy/run_all.sh` then `python3 build_dashboard.py`.
6. **Read the sign, not the magnitude.** Did it go the way I predicted? Did anything I care about
   regress?

Then, and only then, decide. That loop is the actual deliverable of this assignment; the specific
numbers are just this week's output of it.

---

## Part 11. The mistakes I want to avoid

1. **Wishful accounting.** Writing a share without checking that the tokens exist. This is the
   headline failure mode, and the ledger exists to make it impossible.
2. **One headline number hiding a sick tier.** "Indic 18%" sounded healthy until it was split four
   ways. Always ask what a comfortable average is concealing.
3. **Believing an unfalsifiable plan.** If no cheap experiment could ever prove me wrong, I have
   written an opinion, not a plan. Every claim should name the test that would refute it.
4. **Trusting a filter blindly.** OPUS would have deleted Indic and agentic while reporting
   excellent efficiency the whole time.
5. **Spending the gold early.** No reserve means no anneal, and no way to get one back.
6. **Confusing tokens with gradient.** 8% agentic at 0.35 trainable is 2.8% of real learning.
7. **Comparing near-identical mixtures.** If the sets do not differ meaningfully, the experiment
   answers nothing and you have paid for it anyway.
8. **Reading toy-scale magnitudes as real.** At 4.85M params, only the direction transfers.

---

## The 30-second version

Fixed budget, so every share is a trade. Choose shares backwards from benchmarks, pushing
differentiators up and hygiene lanes down to their floor. Check every share against real supply and
say out loud where you must repeat (cap 4 epochs) or generate. Compare 3 genuinely different
recipes on a cheap proxy and read the direction, not the magnitude. Order the run easy to hard,
short to long, with blended seams. Protect the lanes an automatic selector would misjudge. Reserve
your best data for a low-LR finish, and reserve it before you start.

_Reference notes for Assignment 5. The plan itself is in [README.md](README.md); the numbers are
computed in [`supply/ledger.py`](supply/ledger.py) and tested in [`proxy/`](proxy)._
