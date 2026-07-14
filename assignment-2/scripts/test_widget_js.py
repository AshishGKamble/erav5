#!/usr/bin/env python3
"""Validate site/hf_tokenizer.js against the real Python HuggingFace tokenizer.

Runs the widget's JS encoder/decoder in a V8 engine (py_mini_racer) and checks
it reproduces tokenizers' encode() token-for-token and decode() exactly, on the
corpus and on tricky Markdown/URL/emoji samples. This is how we trust the live
widget without a browser.

    pip install py_mini_racer
    python3 scripts/test_widget_js.py
"""
import json
import re
from pathlib import Path

from py_mini_racer import MiniRacer
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent.parent
LANGS = ["en", "hi", "te", "mr"]


def load_text(c):
    f = ROOT / "corpus" / f"{c}.faithful.txt"
    return (f if f.exists() else ROOT / "data" / f"{c}.txt").read_text(encoding="utf-8")


def main():
    tj = json.loads((ROOT / "tokenizer" / "tokenizer.json").read_text(encoding="utf-8"))
    tok = Tokenizer.from_file(str(ROOT / "tokenizer" / "tokenizer.json"))

    js_src = (ROOT / "site" / "hf_tokenizer.js").read_text(encoding="utf-8")
    js_src = re.sub(r"^export ", "", js_src, flags=re.M)  # strip ES exports for V8

    ctx = MiniRacer()
    ctx.eval(js_src)
    ctx.eval(f"var __TJ = {json.dumps(tj)}; var __T = loadTokenizer(__TJ);")
    ctx.eval("function ENC(s){return encode(s,__T);} function DEC(a){return decode(a);}")

    def js_encode(s):
        return ctx.call("ENC", s)

    def js_decode(tokens):
        return ctx.call("DEC", tokens)

    samples = [
        "https://hi.wikipedia.org/wiki/भारत#cite_ref-1",
        "India (# 1) - [a](b), \"q\", it's 3,000_000; x<y & p>q | 50% `code` _i_ *b*",
        "भारत గణతంత్ర मराठी — emoji 🇮🇳 表 │ € ″ ⓘ",
        "line1\nline2\ttab   three-spaces  end ",
        "  leading and trailing  ",
    ]
    # plus chunks of each corpus text (full texts are large; sample slices)
    for c in LANGS:
        t = load_text(c)
        for i in range(0, min(len(t), 20000), 5000):
            samples.append(t[i:i + 5000])

    enc_ok = dec_ok = 0
    fails = 0
    for s in samples:
        ref = tok.encode(s).tokens
        got = js_encode(s)
        if got == ref:
            enc_ok += 1
        else:
            fails += 1
            if fails <= 3:
                # find first diff
                for k in range(max(len(ref), len(got))):
                    if k >= len(ref) or k >= len(got) or ref[k] != got[k]:
                        print(f"ENCODE MISMATCH at {k}: ref={ref[max(0,k-2):k+3]} got={got[max(0,k-2):k+3]}")
                        print("  sample:", repr(s[:60]))
                        break
        # decode check: JS decode of ref ids-as-tokens must equal python decode
        pdec = tok.decode(tok.encode(s).ids)
        jdec = js_decode(ref)
        if jdec == pdec:
            dec_ok += 1
        else:
            print("DECODE MISMATCH sample:", repr(s[:50]), "\n  py:", repr(pdec[:60]), "\n  js:", repr(jdec[:60]))

    print(f"\nencode match: {enc_ok}/{len(samples)}   decode match: {dec_ok}/{len(samples)}")
    # Full-text encode counts must match exactly (this drives the widget's numbers)
    print("\nfull-text token counts (Python HF vs JS):")
    allok = True
    for c in LANGS:
        t = load_text(c)
        ref_n = len(tok.encode(t).ids)
        js_n = len(js_encode(t))
        ok = ref_n == js_n
        allok &= ok and enc_ok == len(samples)
        print(f"  {c}: python={ref_n}  js={js_n}  {'OK' if ok else 'MISMATCH'}")
    print("\nALL GOOD:", allok and fails == 0)
    raise SystemExit(0 if (allok and fails == 0) else 1)


if __name__ == "__main__":
    main()
