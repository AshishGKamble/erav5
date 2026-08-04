# ERA V5 - assignments

Coursework for the ERA V5 arc, building toward an **India-first ~40B model**. Each assignment feeds
the next: a tokenizer, a model design, a cleaned corpus, and the training recipe that turns those
tokens into the benchmark targets.

## → Assignment 5: [V5 Mixture & Curriculum Plan](assignment-5/README.md) (current submission)

The training recipe: how a fixed 4T-token budget is split across capability lanes, in what order,
what is held back for the anneal, and the proxy study we ran to attack our own numbers.

| Document | What it is |
|---|---|
| **[assignment-5/README.md](assignment-5/README.md)** | **The plan. Start here.** |
| [assignment-5/EXPERIMENTS.md](assignment-5/EXPERIMENTS.md) | The trial-and-error log: 9 rounds, 21 runs, every claim we withdrew |
| [assignment-5/supply/ledger.md](assignment-5/supply/ledger.md) | Demand vs real supply per lane (computed) |
| [assignment-5/supply/datasets.md](assignment-5/supply/datasets.md) | Which named datasets fill each slot (computed) |
| [assignment-5/benchmarks.md](assignment-5/benchmarks.md) | Which A3 targets the mixture reaches (computed) |
| [assignment-5/proxy/results.md](assignment-5/proxy/results.md) | Per-lane held-out loss with noise floors (computed) |
| [assignment-5/CONCEPTS.md](assignment-5/CONCEPTS.md) | The underlying ideas in plain language |

## Earlier assignments

| # | Title | What it produced |
|---|---|---|
| 1 | Foundations | Course setup and first experiments |
| 2 | [Faithful multilingual BPE tokenizer](assignment-2/) | A byte-level tokenizer with a live comparison widget |
| 3 | [India-first 40B model design](assignment-3/) | The architecture and the benchmark targets Assignment 5 composes backward from |
| 4 | [Corpus cleaning](assignment-4/) | The cleaned, provenance-stamped Indic corpus. Its `source` labels are what made Assignment 5's decisive experiment possible |

_Built by Ashish Kamble_
