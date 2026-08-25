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

- **`assignment-6/frozen/corpus/`** contains **bounded slices** of seven public datasets, one per
  training lane, committed so that `python run_demo.py` is reproducible offline from a clean clone.
  Each slice records its source, licence, fetch date and `sha256` in
  `assignment-6/frozen/corpus/SOURCES.json`. **Each slice remains under its own licence**, listed
  below; none of them is relicensed by inclusion here.

  | Lane | Source | Licence |
  |---|---|---|
  | web | `Salesforce/wikitext` (wikitext-103-raw-v1) | CC BY-SA 3.0 / 4.0 |
  | code | `codeparrot/codeparrot-clean-valid`, **filtered to permissive licences only** | per-file: MIT, Apache-2.0, BSD-2/3-Clause, ISC |
  | math | `open-web-math/open-web-math` | ODC-By 1.0 (Common Crawl ToU also applies) |
  | reasoning | `openai/gsm8k` | MIT |
  | indic | `ai4bharat/indic-align`, via the Assignment-4 pipeline | CC BY 4.0 |
  | agentic | `glaiveai/glaive-function-calling-v2` | Apache-2.0 |
  | long-context | Project Gutenberg | Public domain in the US |

  Two notes on that table. The **web** slice is share-alike, so anything derived from it carries
  CC BY-SA forward. The **code** slice is filtered at fetch time to permissive licences using the
  dataset's own per-file `license` field, because `codeparrot-clean-valid` also contains GPL-2.0 and
  AGPL-3.0 files, and **copyleft source code is deliberately excluded** rather than redistributed
  from an MIT repository. Every retained file's licence and origin repository is recorded in the
  shard manifest.

## Third-party datasets referenced but NOT redistributed

The supply inventory in `assignment-5/supply/inventory.json` names datasets such as
DCLM-baseline, FineWeb, The Stack v2, StarCoder2, OpenWebMath, Proof-Pile-2, peS2o,
OpenThoughts, ToolBench, xLAM, Sangraha, IndicCorp v2, IndicTrans2 and others. **The
inventory itself copies none of their content** - it records token counts and citations only,
so that supply figures can be checked against the sources. Anyone using those datasets is
bound by each dataset's own licence and terms.

**The one exception, added in Assignment 6.** That assignment needs a corpus that survives a
clean clone with no network, so it commits bounded, licence-checked slices of seven datasets
under `assignment-6/frozen/corpus/` - documented in the section above. This is a deliberate
narrowing of the "referenced, never redistributed" position, and it applies **only** to those
seven slices. Two datasets named in the Assignment-5 inventory were considered for the agentic
lane and **rejected on licensing grounds**: **xLAM / APIGen-60k**, which is CC BY 4.0 but sits
behind an access gate that must be accepted individually, and **ToolBench**, which is Apache-2.0
today but was distributed under CC BY-NC 4.0 in 2023. Neither is redistributed here.

Apart from the Assignment-6 slices named above, corpora, token bins and model caches produced
by the pipelines here are `.gitignore`d and are not distributed.

## Scope

This is coursework. The plans, numbers and conclusions are ours and are offered as-is, with
their limitations stated in the documents themselves. Nothing here is a trained model. The only
third-party data redistributed is the Wikipedia extracts in `assignment-2/corpus/` and the seven
licence-checked slices in `assignment-6/frozen/corpus/`, each kept under its own licence and
recorded with its source and hash.

_Ashish Kamble, 2026_
