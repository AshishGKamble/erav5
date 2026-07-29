"""
Script-aware Indic language identification (Stage 3 helper).

Why script-based, not fastText: the 12 languages we care about live in *distinct Unicode
blocks*, so the script itself is a near-perfect signal - far more reliable for Indic than
fastText lid.176, which routinely confuses Indic languages. Two families overlap, and we
resolve each with a light heuristic, exactly the "detect at runtime, confidence illustrative"
posture the session widget describes:
  - Devanagari  -> Hindi vs Marathi   (stop-word tiebreak)
  - Eastern Nagari -> Bengali vs Assamese (Assamese-only letters ৰ ৱ)

Returns a small dict per document. The point of the stage is the same as the widget's:
a document's *source/label is a claim, not a fact* - so we detect and can flag mismatches
and code-switching before anything files into a language bucket.
"""
import re

# One contiguous Unicode block per Indic script (start, end, language-or-family).
SCRIPT_BLOCKS = [
    (0x0900, 0x097F, "deva"),   # Devanagari  -> Hindi / Marathi
    (0x0980, 0x09FF, "beng"),   # Bengali     -> Bengali / Assamese
    (0x0A00, 0x0A7F, "guru"),   # Gurmukhi    -> Punjabi
    (0x0A80, 0x0AFF, "gujr"),   # Gujarati
    (0x0B00, 0x0B7F, "orya"),   # Odia
    (0x0B80, 0x0BFF, "taml"),   # Tamil
    (0x0C00, 0x0C7F, "telu"),   # Telugu
    (0x0C80, 0x0CFF, "knda"),   # Kannada
    (0x0D00, 0x0D7F, "mlym"),   # Malayalam
    (0x0600, 0x06FF, "arab"),   # Perso-Arabic -> Urdu
    (0x0750, 0x077F, "arab"),
]
SCRIPT_TO_LANG = {
    "guru": "pa", "gujr": "gu", "orya": "or", "taml": "ta", "telu": "te",
    "knda": "kn", "mlym": "ml", "arab": "ur",
}
LANG_NAME = {
    "hi": "Hindi", "mr": "Marathi", "bn": "Bengali", "as": "Assamese", "pa": "Punjabi",
    "gu": "Gujarati", "or": "Odia", "ta": "Tamil", "te": "Telugu", "kn": "Kannada",
    "ml": "Malayalam", "ur": "Urdu", "en": "English", "und": "Undetermined",
}

# Marathi function words rare/absent in Hindi (used only to split the shared Devanagari script).
# Hindi and Marathi share Devanagari, so one coincidental word is not enough - we require two
# distinct Marathi cues (or the distinctively-Marathi letter ळ) before calling a doc Marathi.
_MARATHI_HINTS = ("आहे", "आणि", "मला", "तू", "नाही", "होते", "त्या", "मध्ये", "काय",
                  "यांनी", "आम्ही", "करणे", "आपण", "तसेच", "म्हणून", "होता")
# Assamese-only letters inside the Bengali block.
_ASSAMESE_LETTERS = ("ৰ", "ৱ")  # ৰ ৱ
_LATIN = re.compile(r"[A-Za-z]")


def _script_counts(text):
    counts = {}
    latin = 0
    for ch in text:
        cp = ord(ch)
        if 0x41 <= cp <= 0x7A and ch.isalpha():
            latin += 1
            continue
        for lo, hi, name in SCRIPT_BLOCKS:
            if lo <= cp <= hi:
                counts[name] = counts.get(name, 0) + 1
                break
    return counts, latin


def detect(text):
    """Return {lang, script, confidence, code_switched, indic_frac, latin_frac}."""
    text = text or ""
    counts, latin = _script_counts(text)
    indic_total = sum(counts.values())
    alpha_total = indic_total + latin

    if alpha_total == 0:
        return {"lang": "und", "script": "none", "confidence": 0.0,
                "code_switched": False, "indic_frac": 0.0, "latin_frac": 0.0}

    indic_frac = indic_total / alpha_total
    latin_frac = latin / alpha_total

    # No Indic script at all -> English (Latin).
    if indic_total == 0:
        return {"lang": "en", "script": "latin", "confidence": round(latin_frac, 3),
                "code_switched": False, "indic_frac": 0.0, "latin_frac": round(latin_frac, 3)}

    dom_script = max(counts, key=counts.get)
    conf = counts[dom_script] / indic_total

    if dom_script == "deva":
        cues = sum(1 for h in _MARATHI_HINTS if h in text)
        if "ळ" in text: cues += 1                     # ळ (LA) is distinctively Marathi, rare in Hindi
        lang = "mr" if cues >= 2 else "hi"             # need two cues, not one coincidental word
    elif dom_script == "beng":
        lang = "as" if any(c in text for c in _ASSAMESE_LETTERS) else "bn"
    else:
        lang = SCRIPT_TO_LANG.get(dom_script, "und")

    # Code-switched: a meaningful mix of Latin and Indic in the same document (e.g. Hinglish).
    code_switched = latin_frac >= 0.15 and indic_frac >= 0.15

    return {"lang": lang, "script": dom_script, "confidence": round(conf, 3),
            "code_switched": code_switched,
            "indic_frac": round(indic_frac, 3), "latin_frac": round(latin_frac, 3)}


if __name__ == "__main__":
    tests = [
        "Photosynthesis is the process by which plants make food.",
        "नमस्ते दुनिया, यह एक हिंदी वाक्य है।",
        "मी तुला मदत करू शकतो का? हे एक मराठी वाक्य आहे आणि मला आवडते.",
        "আমি বাংলা ভাষা ভালোবাসি।",
        "আমি অসমীয়া ৰাজ্যত থাকোঁ আৰু ৱাটাৰ খাওঁ।",
        "இது ஒரு தமிழ் வாக்கியம் ஆகும்.",
        "yeh dataset bahut large scale par train hota hai",  # romanized/code mix
        "यह dataset बहुत large scale पर train होता है।",       # code-switched Hinglish
    ]
    for t in tests:
        d = detect(t)
        print(f"{LANG_NAME[d['lang']]:11s} conf={d['confidence']:.2f} "
              f"cs={d['code_switched']!s:5s} :: {t[:45]}")
