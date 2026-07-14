#!/usr/bin/env python3
"""Build a faithful multilingual corpus from the India Wikipedia articles.

For each of English, Hindi, Telugu and Marathi this fetches the article's
canonical HTML from the MediaWiki REST endpoint and renders it to Markdown-ish
plain text that keeps the *visible* article content (headings, list markers,
links, tables, references, image captions, categories). The tokenizer is then
trained and scored on this faithful text, so its round-trip must preserve every
visible character.

Conversion is done with `html2text` configured to never wrap lines and to keep
Unicode verbatim (`unicode_snob`), which matters for the Indic scripts - any
ASCII-folding would make the round trip lossy. Only inert machinery
(script/style/meta/link/noscript and MediaWiki edit affordances) is dropped; no
article prose is clipped.

    pip install requests beautifulsoup4 lxml html2text regex
    python3 scripts/build_faithful_corpus.py
    python3 scripts/hf_build.py

The four articles are public Wikipedia pages; re-running re-fetches them, so the
exact byte counts move as the articles are edited.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import html2text
import regex
import requests
from bs4 import BeautifulSoup

CORPUS_DIR = Path(__file__).resolve().parents[1] / "corpus"
REST_HTML = "https://{lang}.wikipedia.org/api/rest_v1/page/html/{title}"
HEADERS = {"User-Agent": "era-v5-a2-tokenizer/1.0 (coursework; contact via widget)"}

# lang code -> (display name, article title on that wiki)
ARTICLES = {
    "en": ("English", "India"),
    "hi": ("Hindi", "भारत"),
    "te": ("Telugu", "భారతదేశం"),
    "mr": ("Marathi", "भारत"),
}

# a faithful unit = one letter/mark/number run, or one visible punctuation char
UNIT = regex.compile(r"[\p{L}\p{M}\p{N}]+|[^\s\p{L}\p{M}\p{N}]")

# elements that carry no visible article text
DROP_TAGS = ("script", "style", "meta", "link", "noscript")
# MediaWiki chrome selectors (edit pencils, jump-to-nav helpers)
DROP_SELECTORS = (".mw-editsection", ".mw-jump-link", "sup.mw-ref-charref")


def fetch_html(lang: str, title: str) -> str:
    url = REST_HTML.format(lang=lang, title=requests.utils.quote(title, safe=""))
    resp = requests.get(url, headers=HEADERS, timeout=(8, 40))
    resp.raise_for_status()
    return resp.text


def prune(html: str) -> str:
    """Strip inert machinery only, returning the article body's HTML."""
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(list(DROP_TAGS)):
        tag.decompose()
    for sel in DROP_SELECTORS:
        for tag in soup.select(sel):
            tag.decompose()
    body = soup.body or soup
    return str(body)


def to_markdown() -> html2text.HTML2Text:
    h = html2text.HTML2Text()
    h.body_width = 0            # never hard-wrap: preserve original line structure
    h.unicode_snob = True       # keep Devanagari/Telugu verbatim, no ASCII folding
    h.ignore_links = False      # links are visible faithful content
    h.ignore_images = False
    h.ignore_emphasis = False
    h.single_line_break = False
    h.wrap_links = False
    h.protect_links = True
    return h


def tidy(text: str) -> str:
    """Collapse runaway blank lines and trailing spaces; keep everything visible."""
    lines = [ln.rstrip() for ln in text.replace("\xa0", " ").split("\n")]
    out, blanks = [], 0
    for ln in lines:
        if ln:
            blanks = 0
            out.append(ln)
        else:
            blanks += 1
            if blanks <= 2:
                out.append(ln)
    return "\n".join(out).strip() + "\n"


def build(lang: str, title: str, converter: html2text.HTML2Text) -> dict:
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    raw = fetch_html(lang, title)
    (CORPUS_DIR / f"{lang}.raw.html").write_text(raw, encoding="utf-8")

    text = tidy(converter.handle(prune(raw)))
    (CORPUS_DIR / f"{lang}.faithful.md").write_text(text, encoding="utf-8")
    (CORPUS_DIR / f"{lang}.faithful.txt").write_text(text, encoding="utf-8")

    meta = {
        "lang": lang,
        "title": title,
        "source_url": REST_HTML.format(lang=lang, title=title),
        "converter": "html2text",
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "chars": len(text),
        "faithful_units": len(UNIT.findall(text)),
    }
    (CORPUS_DIR / f"{lang}.meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def main() -> None:
    converter = to_markdown()
    for lang, (name, title) in ARTICLES.items():
        m = build(lang, title, converter)
        print(f"{lang} ({name}): {m['chars']:,} chars, {m['faithful_units']:,} faithful units")


if __name__ == "__main__":
    main()
