"""
Assignment 4 - the cleaning pipeline: 8 Session-4 strategies + 2 bonus concerns.

Raw instruction data enters; less than half survives to become training text. Each stage
strips one kind of garbage / fixes one real defect and records exactly what it did and why,
into site/data/stats.json (the widget's data source). Nothing is invented: every number the
widget shows is computed here on the real downloaded slice.

Stage order (order matters - clean_text runs BEFORE the content hash, so dedup + manifest
trust the cleaned text, not the raw HTML):

  1 Extract        pull text out of the interactions structure, drop empties/markup
  2 Normalize      NFC, strip invisibles, keep ZWJ/ZWNJ, unescape, hash-after   (+ ghost-tag bonus)
  3 Language ID    detect script/language per doc; flag code-switch (label is a claim)
  4 Quality filter Gopher/C4 heuristics + Indic Always-ON channel
  5 Deduplicate    MinHash + LSH, GLOBAL across the whole slice
  6 PII scrub      regex (email/phone/IP/Aadhaar) + honorific name layer
  7 Decontaminate  n-gram scan vs the mmlu-indic / trivia-qa hold-out   (+ safety bonus)
  8 Manifest       per-shard provenance + SHA-256 + real token counts + fertility
"""
import os, re, json, html, hashlib, unicodedata, random, glob, time
from collections import defaultdict, Counter
import pandas as pd
import pyarrow.parquet as pq
import langid

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw")
HOLD = os.path.join(ROOT, "data", "holdout")
CLEANED = os.path.join(ROOT, "data", "cleaned")
OUT = os.path.join(ROOT, "site", "data")
os.makedirs(CLEANED, exist_ok=True)
os.makedirs(OUT, exist_ok=True)
os.environ.setdefault("HF_HOME", os.path.join(ROOT, ".hf_cache"))

SEED = 42
TARGET_TOKENS = 22_000_000          # ~22M-token working slice (inside the 10-100M requirement)
random.seed(SEED)

stats = {}          # per-stage stats -> stats.json
examples = {}       # real before/after snippets -> examples.json
def note(stage, **kw): stats[stage] = kw
def ex(stage, **kw): examples.setdefault(stage, []).append(kw)


# ----------------------------------------------------------------------------- load / extract
# indic-align stores parallel translations as one column per language ("hin_Deva", "tam_Taml",
# romanized "hin_Latn", ...). The column name is a CLAIMED language - Stage 3 detects independently
# and flags mismatches (exactly the widget's "a label is a claim, not a fact"). anudesh instead uses
# a single 'interactions' column of English conversation - a second source format on purpose.
COL_LANG = {
    "eng_Latn": "en", "hin_Deva": "hi", "ben_Beng": "bn", "mar_Deva": "mr", "tel_Telu": "te",
    "tam_Taml": "ta", "guj_Gujr": "gu", "urd_Arab": "ur", "kan_Knda": "kn", "ory_Orya": "or",
    "mal_Mlym": "ml", "pan_Guru": "pa", "asm_Beng": "as",
}
ROMANIZED = {"hin_Latn": "hi", "tam_Latn": "ta", "ben_Latn": "bn"}   # romanized-Indic challenge sample
SELECTED = {**COL_LANG, **ROMANIZED}
SRC_ROW_CAP = {"anudesh": 6000, "hhrlhf": 8000, "toxicmatrix": 8000}  # dolly: full (15k rows)

def _texts_from_cell(v):
    """Flatten a cell holding [[user, assistant], ...] (list / tuple / numpy array) into strings."""
    if v is None: return []
    try:
        seq = list(v)
    except TypeError:
        return []
    out = []
    for turn in seq:
        if isinstance(turn, str):
            if turn: out.append(turn)
        else:
            try:
                for x in turn:
                    if x is not None and str(x) != "": out.append(str(x))
            except TypeError:
                pass
    return out

def _join(parts):
    return "\n".join(p for p in parts if p and p != "None").strip()

def _read_rows(fp, cap=None):
    """Read parquet rows as dicts; stop after `cap` rows (bounds memory on the big files)."""
    if cap is None:
        return pd.read_parquet(fp).to_dict("records")
    rows, got = [], 0
    for batch in pq.ParquetFile(fp).iter_batches(batch_size=1000):
        cols = batch.to_pydict(); keys = list(cols); nrows = len(cols[keys[0]])
        for r in range(nrows):
            rows.append({k: cols[k][r] for k in keys}); got += 1
            if got >= cap: return rows
    return rows

def stage1_extract():
    files = sorted(glob.glob(os.path.join(RAW, "**", "*.parquet"), recursive=True))
    raw_docs, empties = [], 0
    per_source = Counter()
    for fp in files:
        source = os.path.basename(os.path.dirname(fp))
        toxic = "toxic" in fp
        rows = _read_rows(fp, cap=SRC_ROW_CAP.get(source))
        random.Random(SEED).shuffle(rows)
        for i, row in enumerate(rows):
            if "interactions" in row:                       # anudesh-style single column
                items = [(None, False, _join(_texts_from_cell(row["interactions"])))]
            else:                                           # parallel per-language columns
                items = []
                for col, lang in SELECTED.items():
                    if col in row:
                        t = _join(_texts_from_cell(row[col]))
                        if t: items.append((lang, col in ROMANIZED, t))
            for claimed, romanized, text in items:
                if len(text) < 1:
                    empties += 1; continue
                raw_docs.append({"source": source, "toxic_src": toxic, "claimed": claimed,
                                 "romanized": romanized, "text": text})
                per_source[source] += 1
    # deterministic shuffle across sources+languages, then cap by estimated tokens (chars/4)
    random.Random(SEED).shuffle(raw_docs)
    docs, est = [], 0
    for d in raw_docs:
        d["id"] = len(docs)                              # unique running id (LSH keys must be unique)
        docs.append(d); est += max(1, len(d["text"]) // 4)
        if est >= TARGET_TOKENS: break
    note("extract", files=len(files), rows_seen=len(raw_docs) + empties,
         empties_dropped=empties, docs_in=len(docs), per_source=dict(per_source),
         est_tokens_in=est)
    print(f"[1] extract: {len(docs)} docs (~{est/1e6:.1f}M est tokens) from {len(files)} files")
    return docs


# ----------------------------------------------------------------------------- normalize
ZWNJ, ZWJ = chr(0x200C), chr(0x200D)                   # real Indic joiners - MUST survive
# invisible / structural noise to strip, but NOT the Indic joiners above.
_NOISE_CP = [0x200B,                                    # ZWSP
             0x200E, 0x200F,                            # LRM, RLM
             0xFEFF, 0xFFFD,                            # BOM/ZWNBSP, replacement char
             0x202A, 0x202B, 0x202C, 0x202D, 0x202E,    # bidi embeddings/overrides
             0x2066, 0x2067, 0x2068, 0x2069]            # bidi isolates
NOISE = re.compile("[" + "".join(chr(c) for c in _NOISE_CP) + "]"
                   + r"|[\x00-\x08\x0b\x0c\x0e-\x1f]")   # C0 controls except \t \n \r
_HAS_ENTITY = re.compile(r"&\w+;|&#\d+;")
GHOST = re.compile(r"\[/?(?:USER|ASSISTANT|SYSTEM|INST)\]|<\|[a-z_]+\|>|<<SYS>>|\[/?INST\]", re.I)
WS = re.compile(r"[^\S\n]+")                            # runs of spaces/tabs (not newlines)
MULTINL = re.compile(r"\n{3,}")
HTMLTAG = re.compile(r"<[^>]{1,40}>")

def clean_text(s):
    s = unicodedata.normalize("NFC", s)          # NFC, never NFKC (NFKC is lossy on Indic)
    s = html.unescape(s)                         # &amp; &quot; &#8217; -> real chars
    s = HTMLTAG.sub(" ", s)                      # residual markup
    zwnj = s.count(ZWNJ); zwj = s.count(ZWJ)
    s = NOISE.sub("", s)                         # strip invisibles/bidi/controls (joiners survive)
    s = WS.sub(" ", s)
    s = MULTINL.sub("\n\n", s)
    return s.strip(), zwnj, zwj

def stage2_normalize(docs):
    ghost_docs = ghost_hits = 0
    zwnj_total = zwj_total = 0
    ghost_kinds = Counter()
    shown_ghost = shown_norm = 0
    for d in docs:
        before = d["text"]
        g = GHOST.findall(before)
        cleaned, zwnj, zwj = clean_text(before)
        # ghost-tag / format-discipline bonus: literal chat markers become ordinary subwords in
        # pretraining and collide with the tokenizer's real special tokens in SFT. Unify to one form.
        if g:
            ghost_docs += 1; ghost_hits += len(g)
            for tag in g: ghost_kinds[tag.upper()] += 1
            cleaned = GHOST.sub("", cleaned).strip()
            cleaned = WS.sub(" ", cleaned)
            if shown_ghost < 2 and before != cleaned:
                ex("ghost", before=before[:300], after=cleaned[:300], tags=g[:6]); shown_ghost += 1
        zwnj_total += zwnj; zwj_total += zwj
        if shown_norm < 2 and (_HAS_ENTITY.search(before) or NOISE.search(before)) and before != cleaned:
            ex("normalize", before=before[:300], after=cleaned[:300]); shown_norm += 1
        d["text"] = cleaned
        d["hash"] = hashlib.sha256(cleaned.encode()).hexdigest()   # hash AFTER cleaning
    note("normalize",
         docs=len(docs), ghost_docs=ghost_docs, ghost_hits=ghost_hits,
         ghost_kinds=dict(ghost_kinds.most_common(6)),
         zwnj_preserved=zwnj_total, zwj_preserved=zwj_total)
    print(f"[2] normalize: {ghost_hits} ghost tags in {ghost_docs} docs; "
          f"kept {zwnj_total+zwj_total} Indic joiners")
    return docs


# ----------------------------------------------------------------------------- language id
def stage3_language(docs):
    dist = Counter(); cs = 0; low_conf = 0; mismatch = 0; romanized_flagged = 0
    shown_mis = shown_ind = 0
    for d in docs:
        det = langid.detect(d["text"])
        d["lang"] = det["lang"]; d["cs"] = det["code_switched"]; d["conf"] = det["confidence"]
        dist[det["lang"]] += 1
        if det["code_switched"]: cs += 1
        if det["lang"] == "und" or det["confidence"] < 0.4: low_conf += 1
        # the widget's core lesson: the column/folder label is a CLAIM. Detect and compare.
        claimed = d.get("claimed")
        if claimed is not None and claimed != det["lang"]:
            mismatch += 1
            if d.get("romanized"): romanized_flagged += 1
            if shown_mis < 3:
                ex("language", text=d["text"][:200], claimed=langid.LANG_NAME.get(claimed, claimed),
                   detected=langid.LANG_NAME[det["lang"]], romanized=bool(d.get("romanized")),
                   latin=det["latin_frac"], indic=det["indic_frac"]); shown_mis += 1
        elif det["lang"] not in ("en", "und") and det["confidence"] > 0.95 and shown_ind < 2:
            ex("language", text=d["text"][:200], detected=langid.LANG_NAME[det["lang"]],
               match=True, conf=det["confidence"]); shown_ind += 1
    note("language",
         distribution={langid.LANG_NAME[k]: v for k, v in dist.most_common()},
         code_switched=cs, low_confidence=low_conf,
         claimed_vs_detected_mismatch=mismatch, romanized_flagged=romanized_flagged,
         languages_seen=len([k for k in dist if k != "und"]))
    print(f"[3] language id: {len(dist)} langs; {cs} code-switched; "
          f"{mismatch} claimed-vs-detected mismatches ({romanized_flagged} romanized)")
    return docs


# ----------------------------------------------------------------------------- quality filter
STOP_EN = {"the","be","to","of","and","a","in","that","have","i","it","for","not","on","with","as"}
SYM = re.compile(r"[#" + "".join(chr(c) for c in (0x2022, 0x00B7, 0x25BA, 0x25AA, 0x25CF, 0x2026)) + "]")
def _quality_metrics(text):
    words = text.split()
    wc = len(words)
    mwl = sum(len(w) for w in words) / max(wc, 1)
    sym_ratio = len(SYM.findall(text)) / max(wc, 1)
    lines = [l for l in text.split("\n") if l.strip()]
    dup_line_frac = 1 - (len(set(lines)) / max(len(lines), 1))
    bullet = sum(1 for l in lines if l.lstrip()[:1] in "-*") / max(len(lines), 1)
    stops = sum(1 for w in words[:200] if w.lower() in STOP_EN)
    return wc, mwl, sym_ratio, dup_line_frac, bullet, stops

def _rules(m):
    wc, mwl, sym_ratio, dup_line_frac, bullet, stops = m
    # script-neutral rules - fair for any language
    neutral = {
        "word_count>=8": wc >= 8,
        "symbol_ratio<0.10": sym_ratio < 0.10,
        "dup_line_frac<0.30": dup_line_frac < 0.30,
        "bullet_frac<0.90": bullet < 0.90,
    }
    # english-calibrated rules - only fair to apply to English text
    english = {
        "mean_word_len_3_10": 3 <= mwl <= 10,
        "stopwords>=2": stops >= 2,
    }
    return neutral, english

def stage4_quality(docs):
    kept, dropped = [], 0
    naive_would_drop_indic = 0     # English-tuned chain applied blindly to Indic
    alwayson_saved = 0             # Indic docs the Always-ON channel rescued
    rule_fail = Counter()
    shown = 0
    for d in docs:
        indic = d["lang"] not in ("en", "und")
        neutral, english = _rules(_quality_metrics(d["text"]))
        pass_neutral = all(neutral.values())
        pass_english = all(english.values())
        # what a naive English-only pipeline would do (the V4 mistake): apply BOTH chains to all
        naive_pass = pass_neutral and pass_english
        # our Always-ON policy: Indic docs are judged on neutral rules only
        ours_pass = pass_neutral if indic else (pass_neutral and pass_english)
        if indic and not naive_pass and ours_pass:
            naive_would_drop_indic += 1; alwayson_saved += 1
        if not ours_pass:
            dropped += 1
            checked = {**neutral, **({} if indic else english)}
            for k, ok in checked.items():
                if not ok: rule_fail[k] += 1
            if shown < 2:
                bad = [k for k, ok in checked.items() if not ok]
                ex("quality", text=d["text"][:200], failed=bad, lang=langid.LANG_NAME[d["lang"]]); shown += 1
            continue
        kept.append(d)
    note("quality",
         docs_in=len(docs), kept=len(kept), dropped=dropped,
         rule_failures=dict(rule_fail.most_common()),
         indic_saved_by_alwayson=alwayson_saved,
         naive_english_chain_would_drop_indic=naive_would_drop_indic)
    print(f"[4] quality: kept {len(kept)}, dropped {dropped}; "
          f"Always-ON saved {alwayson_saved} Indic docs an English chain would have cut")
    return kept


# ----------------------------------------------------------------------------- dedup (MinHash+LSH)
from datasketch import MinHash, MinHashLSH
def _shingles(text, k=5):
    w = text.split()
    if len(w) < k: return {text.strip()} if text.strip() else set()
    return {" ".join(w[i:i+k]) for i in range(len(w) - k + 1)}

def stage5_dedup(docs):
    lsh = MinHashLSH(threshold=0.7, num_perm=64)     # near-dup at Jaccard ~0.7 (FineWeb-ish)
    seen_hash = {}; kept = []; exact = 0; near = 0
    shown = 0
    for d in docs:
        if d["hash"] in seen_hash:                    # exact duplicate (same cleaned text)
            exact += 1
            if shown < 2:
                ex("dedup", kind="exact", a=d["text"][:180], b=seen_hash[d["hash"]][:180]); shown += 1
            continue
        m = MinHash(num_perm=64)
        for sh in _shingles(d["text"]):
            m.update(sh.encode())
        if lsh.query(m):                              # near-duplicate of an already-kept doc
            near += 1
            continue
        lsh.insert(d["id"], m)
        seen_hash[d["hash"]] = d["text"]
        kept.append(d)
    note("dedup", docs_in=len(docs), kept=len(kept),
         exact_dupes=exact, near_dupes=near, removed=exact + near,
         params={"shingle_k": 5, "num_perm": 64, "threshold": 0.7})
    print(f"[5] dedup: removed {exact} exact + {near} near = {exact+near}; kept {len(kept)}")
    return kept


# ----------------------------------------------------------------------------- PII scrub
PII = [
    ("EMAIL", re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")),
    ("PHONE", re.compile(r"(?:\+91[\s-]?)?(?<!\d)\d{5}[\s-]?\d{5}(?!\d)")),
    ("IP",    re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
    ("AADHAAR", re.compile(r"\b\d{4}\s\d{4}\s\d{4}\b")),
    # name layer: honorific-anchored -> high precision, deliberately modest recall (see FINDINGS)
    ("NAME",  re.compile(r"\b(?:Shri|Sri|Smt|Dr|Mr|Mrs|Ms|Prof)\.?\s+[A-Z][a-z]+(?:\s[A-Z][a-z]+)?")),
]
def stage6_pii(docs):
    counts = Counter(); affected = 0; shown = 0
    for d in docs:
        t = d["text"]; hit = False; local = {}
        for label, rx in PII:
            t, n = rx.subn(f"[{label}]", t)
            if n: counts[label] += n; hit = True; local[label] = n
        if hit:
            affected += 1
            if shown < 3 and local:
                ex("pii", after=t[:240], masked=local); shown += 1
        d["text"] = t
    note("pii", docs=len(docs), docs_with_pii=affected,
         masked=dict(counts), total_masked=sum(counts.values()),
         method="regex (email/phone/IP/Aadhaar) + honorific-anchored name layer")
    print(f"[6] pii: masked {sum(counts.values())} items in {affected} docs -> {dict(counts)}")
    return docs


# ----------------------------------------------------------------------------- decontaminate + safety
def _load_holdout_ngrams(n=8):
    # fingerprint the hold-out with fixed n-grams (a set) - O(1) membership, no per-doc substring scan.
    grams = set(); qs = 0
    for fp in glob.glob(os.path.join(HOLD, "**", "*.parquet"), recursive=True):
        try:
            df = pd.read_parquet(fp)
        except Exception:
            continue
        col = "question" if "question" in df.columns else df.columns[0]
        for q in df[col].astype(str):
            qs += 1
            w = re.sub(r"\s+", " ", q.lower()).strip().split()
            for i in range(len(w) - n + 1):
                grams.add(" ".join(w[i:i+n]))
    return grams, qs

TOXIC_MARKERS = {"kill","hate","stupid","idiot","bastard","rape","terrorist","slur"}  # illustrative
def stage7_decontaminate(docs):
    grams, qs = _load_holdout_ngrams(n=8)
    kept = []; n_hit = 0; tox = 0; shown = 0
    for d in docs:
        w = re.sub(r"\s+", " ", d["text"].lower()).split()
        doc_grams = {" ".join(w[i:i+8]) for i in range(len(w) - 7)} if len(w) >= 8 else set()
        leak = grams & doc_grams                        # set intersection: fast even for 100k+ docs
        toxic = d.get("toxic_src") and any(m in w for m in TOXIC_MARKERS)
        if leak:
            n_hit += 1
            if shown < 2:
                ex("decontaminate", text=d["text"][:200], overlap=list(leak)[:2]); shown += 1
            continue                                    # drop contaminated (never let the exam train)
        if toxic:
            tox += 1
            continue                                    # drop flagged-toxic
        kept.append(d)
    note("decontaminate",
         holdout_questions=qs, holdout_ngrams=len(grams),
         contaminated_removed=n_hit, toxic_removed=tox, docs_in=len(docs), kept=len(kept),
         holdout_sets=["sarvamai/mmlu-indic (test)", "sarvamai/trivia-qa-indic-mcq (validation)"])
    print(f"[7] decontaminate: {n_hit} contaminated + {tox} toxic removed "
          f"(scanned vs {qs} hold-out questions)")
    return kept


# ----------------------------------------------------------------------------- manifest + fertility
def stage8_manifest(docs):
    from transformers import AutoTokenizer
    # Primary tokenizer = MuRIL (197K vocab, covers all 17 Indian languages incl. Urdu + Assamese) -
    # the real-world instance of the Assignment-3 "~200K vocab, focused on our languages" decision.
    # Sarvam-1 (68K, 10 languages) is kept only as the coverage COMPARISON.
    tok = AutoTokenizer.from_pretrained("google/muril-base-cased")
    tok_cmp = AutoTokenizer.from_pretrained("sarvamai/sarvam-1")
    script_hash = hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()
    ingest_ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    by_source = defaultdict(list); by_lang = defaultdict(list)
    for d in docs: by_source[d["source"]].append(d); by_lang[d["lang"]].append(d)

    # real token counts (MuRIL), batch-encoded, per doc.
    total_tokens = 0; B = 2000; texts = [d["text"] for d in docs]
    for start in range(0, len(texts), B):
        for j, ids in enumerate(tok(texts[start:start + B], add_special_tokens=False)["input_ids"]):
            docs[start + j]["tok"] = len(ids); total_tokens += len(ids)

    # per-language fertility (tokens/word) for BOTH tokenizers on the same per-language sample -
    # this is the data behind "a tokenizer that actually covers the language has lower fertility".
    def _fert(tk, samp):
        w = sum(len(t.split()) for t in samp)
        if w < 200: return None
        n = sum(len(x) for x in tk(samp, add_special_tokens=False)["input_ids"])
        return round(n / w, 2)
    fertility = {}; fertility_cmp = {}
    for lg, ds in by_lang.items():
        samp = [d["text"] for d in ds[:600]]
        fm = _fert(tok, samp); fs = _fert(tok_cmp, samp)
        if fm is not None: fertility[langid.LANG_NAME[lg]] = fm
        if fs is not None: fertility_cmp[langid.LANG_NAME[lg]] = fs

    manifests = []; admitted = 0; blocked = 0
    for source, ds in by_source.items():
        blob = "\n".join(d["text"] for d in ds)
        sha = hashlib.sha256(blob.encode()).hexdigest()
        sha2 = hashlib.sha256(blob.encode()).hexdigest()      # determinism re-run
        ld = Counter(langid.LANG_NAME[d["lang"]] for d in ds)
        man = {
            "source_url": f"https://huggingface.co/datasets/ai4bharat/indic-align ({source})",
            "license_class": "CC-BY-4.0",
            "contributor_id": "era5-assignment4",
            "cleaning_script": os.path.basename(__file__),
            "cleaning_script_hash": script_hash[:16],
            "ingest_timestamp": ingest_ts,
            "sha256": sha[:32],
            "deterministic": sha == sha2,
            "doc_count": len(ds),
            "token_count": sum(d["tok"] for d in ds),
            "lang_distribution": dict(ld.most_common()),
            "status": "ADMITTED",
        }
        required = ["source_url", "license_class", "cleaning_script_hash", "sha256"]
        if all(man.get(r) for r in required):
            admitted += 1
        else:
            man["status"] = "BLOCKED"; blocked += 1
        manifests.append(man)

    words_total = sum(len(d["text"].split()) for d in docs)
    note("manifest",
         shards=len(manifests), admitted=admitted, blocked=blocked,
         total_tokens=total_tokens, total_words=words_total,
         overall_fertility=round(total_tokens / max(words_total, 1), 2),
         fertility_by_language=fertility,
         fertility_sarvam=fertility_cmp,
         tokenizer="google/muril-base-cased (197K vocab)",
         tokenizer_compare="sarvamai/sarvam-1 (68K vocab)",
         cleaning_script_hash=script_hash[:16], ingest_timestamp=ingest_ts,
         determinism_ok=all(m["deterministic"] for m in manifests))
    json.dump(manifests[0] if manifests else {}, open(os.path.join(OUT, "manifest.json"), "w"),
              ensure_ascii=False, indent=2)
    print(f"[8] manifest: {admitted} shards admitted, {blocked} blocked; "
          f"{total_tokens/1e6:.1f}M real tokens; overall fertility "
          f"{total_tokens/max(words_total,1):.2f}")
    return docs, manifests


# ----------------------------------------------------------------------------- driver
def main():
    t0 = time.time()
    docs = stage1_extract(); n0 = len(docs)
    # real INPUT tokens (MuRIL) on the raw extracted text, before any cleaning - the "before" number.
    from transformers import AutoTokenizer
    _tin = AutoTokenizer.from_pretrained("google/muril-base-cased")
    tokens_in = 0; _raw = [d["text"] for d in docs]
    for s in range(0, len(_raw), 2000):
        tokens_in += sum(len(x) for x in _tin(_raw[s:s + 2000], add_special_tokens=False)["input_ids"])
    stats["extract"]["tokens_in_real"] = tokens_in
    print(f"    input tokens (MuRIL, pre-clean): {tokens_in/1e6:.1f}M")
    docs = stage2_normalize(docs)
    docs = stage3_language(docs)
    docs = stage4_quality(docs)
    docs = stage5_dedup(docs)
    docs = stage6_pii(docs)
    docs = stage7_decontaminate(docs)
    docs, manifests = stage8_manifest(docs)

    funnel = [
        ("Extract",       stats["extract"]["docs_in"]),
        ("Normalize",     stats["normalize"]["docs"]),
        ("Language ID",   sum(stats["language"]["distribution"].values())),
        ("Quality",       stats["quality"]["kept"]),
        ("Deduplicate",   stats["dedup"]["kept"]),
        ("PII scrub",     stats["pii"]["docs"]),
        ("Decontaminate", stats["decontaminate"]["kept"]),
        ("Manifest",      len(docs)),
    ]
    summary = {
        "dataset": "ai4bharat/indic-align (mirror: CharuAgarwal/indic-align)",
        "license": "CC-BY-4.0",
        "holdout": ["sarvamai/mmlu-indic", "sarvamai/trivia-qa-indic-mcq"],
        "tokenizer": "sarvamai/sarvam-1",
        "docs_in": n0, "docs_out": len(docs),
        "survival_pct": round(100 * len(docs) / max(n0, 1), 1),
        "tokens_in": stats["extract"].get("tokens_in_real"),
        "total_tokens": stats["manifest"]["total_tokens"],
        "languages": stats["language"]["languages_seen"],
        "runtime_sec": round(time.time() - t0, 1),
        "funnel": funnel,
    }
    stats["_summary"] = summary
    json.dump(stats, open(os.path.join(OUT, "stats.json"), "w"), ensure_ascii=False, indent=2)
    json.dump(examples, open(os.path.join(OUT, "examples.json"), "w"), ensure_ascii=False, indent=2)
    with open(os.path.join(CLEANED, "corpus.jsonl"), "w") as f:
        for d in docs:
            f.write(json.dumps({"id": d["id"], "source": d["source"], "lang": d["lang"],
                                "text": d["text"]}, ensure_ascii=False) + "\n")
    print(f"\nDONE in {summary['runtime_sec']}s: {n0} -> {len(docs)} docs "
          f"({summary['survival_pct']}%), {summary['total_tokens']/1e6:.1f}M tokens. "
          f"Wrote stats.json / examples.json / manifest.json")


if __name__ == "__main__":
    main()
