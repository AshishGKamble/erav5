"""
Reading assignment-6's frozen corpus, and deciding what script a piece of text is written in.

Problem 3's claim is that the fixed window charges its cost unevenly *by script*, so every number it
reports has to be broken down by script rather than by lane. The lane labels in the frozen corpus do
not do that: `indic.jsonl` is a single lane holding ten different writing systems, and the whole
point is that they do not all behave alike. So script is derived here, from the characters.

Detection uses `unicodedata.name`, which spells the script out as the first word of a character's
name: "DEVANAGARI LETTER KA", "TAMIL VOWEL SIGN U", "LATIN SMALL LETTER A". That is a property of
the Unicode database rather than a table maintained here, so it cannot drift out of date and it
cannot be accused of being tuned to produce the result this problem wants.

Two decisions worth stating because they change the numbers:

  * **Digits, punctuation, whitespace and symbols are COMMON, not a script.** A word is classified
    by its letters. Otherwise "2024" would be Latin and every script's statistics would be diluted
    by the same shared characters.

  * **A word's script is the majority script of its letters**, and words whose letters are all
    COMMON are reported as COMMON rather than being forced into a bucket. Mixed-script words exist
    (Latin brand names inside Devanagari sentences) and are counted honestly under their majority
    with the mix rate reported, instead of being dropped to make the tables tidier.
"""
import json
import os
import unicodedata

# Characters that belong to no script in particular. Checked by Unicode general category, so this
# is a statement about the character database rather than a hand-listed set.
_COMMON_CATEGORIES = {"Nd", "Nl", "No", "Zs", "Zl", "Zp", "Cc", "Cf", "Cn", "Co", "Cs"}

_script_cache = {}


def char_script(ch):
    """The script a single character belongs to, or "COMMON" if it belongs to none.

    The Unicode name's leading token is the script for every cased and Indic script this corpus
    contains. Han, Hiragana and Katakana name themselves differently ("CJK UNIFIED IDEOGRAph"),
    which is handled explicitly rather than being allowed to produce a bucket called "CJK".
    """
    cached = _script_cache.get(ch)
    if cached is not None:
        return cached
    cat = unicodedata.category(ch)
    if cat in _COMMON_CATEGORIES or cat.startswith("P") or cat.startswith("S"):
        script = "COMMON"
    else:
        try:
            name = unicodedata.name(ch)
        except ValueError:
            script = "UNKNOWN"
        else:
            if name.startswith("CJK"):
                script = "HAN"
            else:
                script = name.split(" ")[0]
                # "GREEK SMALL LETTER ALPHA" -> GREEK, but also "MODIFIER LETTER ..." and
                # "COMBINING ..." which are script-neutral marks.
                if script in ("MODIFIER", "COMBINING", "ZERO", "NO", "RIGHT", "LEFT"):
                    script = "COMMON"
    _script_cache[ch] = script
    return script


def text_script(text):
    """Majority script over the letters of `text`. Returns (script, purity, n_letters).

    `purity` is the fraction of letters that belong to the winning script, so a caller can tell a
    clean Devanagari word from one with a Latin acronym embedded in it.
    """
    counts = {}
    total = 0
    for ch in text:
        s = char_script(ch)
        if s == "COMMON":
            continue
        counts[s] = counts.get(s, 0) + 1
        total += 1
    if not total:
        return "COMMON", 1.0, 0
    best = max(counts.items(), key=lambda kv: (kv[1], kv[0]))
    return best[0], best[1] / total, total


def record_text(rec):
    """The text of one frozen-corpus record, whichever of the two shapes it uses.

    Assignment 6 wrote plain lanes as a single `text` field, and the two conversational lanes
    (`agentic`, `reasoning`) as a list of `spans` carrying a role and a loss mask. The loss mask is
    a training-time concern and irrelevant here: the window encodes every token the model reads,
    masked or not, so all spans count.
    """
    if "text" in rec:
        return rec["text"]
    if "spans" in rec:
        return "\n".join(sp["text"] for sp in rec["spans"] if sp.get("text"))
    raise KeyError("record has neither 'text' nor 'spans': %s" % sorted(rec))


def read_lane(path, limit=None):
    """Yield the text of each record in one lane file."""
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if limit is not None and i >= limit:
                return
            line = line.strip()
            if line:
                yield record_text(json.loads(line))


def lanes(root, names=None, limit=None):
    """Yield `(lane_name, text)` across the frozen corpus."""
    files = sorted(f for f in os.listdir(root) if f.endswith(".jsonl"))
    for fn in files:
        lane = fn[:-len(".jsonl")]
        if names and lane not in names:
            continue
        for text in read_lane(os.path.join(root, fn), limit=limit):
            yield lane, text
