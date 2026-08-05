# Licensing and attribution

This repository mixes three kinds of material, and they are not all under the same terms.

| What | Where | Licence |
|---|---|---|
| **Code** we wrote | `*.py`, `*.js`, `*.sh`, `*.css`, `*.html` we authored | [MIT](LICENSE) |
| **Written documents** we wrote | `README.md`, `PLAN.md`, `FINDINGS.md`, `EXPERIMENTS.md`, `CONCEPTS.md`, `benchmarks.md`, and other prose | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) - reuse freely with attribution |
| **Third-party content** | see below | Retains its original licence. Ours to use, not ours to relicense. |

## Third-party content included in this repository

- **`assignment-2/corpus/`** contains article extracts fetched from **Wikipedia**
  (`en`, `hi`, `mr`, `te` and other language editions; the exact source URL and fetch
  timestamp for each file is recorded in the accompanying `*.meta.json`). Wikipedia text is
  licensed **[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)**, and these
  extracts remain under that licence. They are included so the tokenizer results are
  reproducible, not as original work.

## Third-party datasets referenced but NOT redistributed

The supply inventory in `assignment-5/supply/inventory.json` names datasets such as
DCLM-baseline, FineWeb, The Stack v2, StarCoder2, OpenWebMath, Proof-Pile-2, peS2o,
OpenThoughts, ToolBench, xLAM, Sangraha, IndicCorp v2, IndicTrans2 and others. **None of
their content is copied into this repository** - the inventory records token counts and
citations only, so that supply figures can be checked against the sources. Anyone using
those datasets is bound by each dataset's own licence and terms.

Corpora, token bins and model caches produced by the pipelines here are `.gitignore`d and
are not distributed.

## Scope

This is coursework. The plans, numbers and conclusions are ours and are offered as-is, with
their limitations stated in the documents themselves. Nothing here is a trained model or a
redistribution of anyone's dataset.

_Ashish Kamble, 2026_
