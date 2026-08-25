"""
The Kronecker codec, and the codepoint variant that Problem 3 proposes.

The published construction is

    kappa(b) = (1/sqrt(L)) * vec( sum_{p=1..L} c_{b_p} (x) p_p )

where `c_v` is a one-hot in R^256 over byte values, `p_p` is a one-hot in R^L over byte positions,
and the result is z-normalised per token. Written out, that is a 256 x L matrix carrying exactly one
1 per occupied column, flattened. There are no learned parameters anywhere in this file.

Three properties matter downstream and are easy to get wrong, so they are stated here and tested in
`tests/`:

  * **Decoding is a per-column argmax.** Because each column is one-hot, recovering the unit at
    position p means taking the argmax down column p. It needs the correct *ranking* inside a
    column, not correct values.

  * **z-normalisation cannot change the decode.** It is `x -> (x - mu) / sigma` with sigma > 0,
    a strictly increasing affine map applied to every entry alike, so it preserves every ordering
    and therefore every argmax. Decoding a z-normalised vector and decoding a raw one give the same
    answer. This is why `encode(..., znorm=True)` is safe rather than a complication.

  * **Unoccupied columns are ties, not zeros, after z-normalisation.** A token shorter than the
    window leaves columns that are all-equal, and an argmax over an all-equal column returns
    position 0 arbitrarily. So the decoder has to decide where the token ended. It does that from
    the column margin (top value minus runner-up), which is large for a real column and ~0 for an
    empty one. `decode` therefore reports a margin per position and the caller can either trust it
    or pass the true length. Both paths are measured rather than one being assumed.

The codepoint variant exists for Problem 3. Instead of one 256-way block per position it uses
`blocks` stacked 256-way blocks holding the base-256 digits of a Unicode codepoint, so a position
holds one *character* rather than one *byte*. Two blocks cover the Basic Multilingual Plane, which
is every script this repository's corpus contains; three cover all of Unicode.
"""
import numpy as np

DTYPE = np.float32
CHAR_DIM = 256


def text_units(text, unit, blocks=2):
    """Split a string into the codec's atomic units.

    `unit="byte"` gives UTF-8 byte values, which is the published construction.
    `unit="codepoint"` gives Unicode codepoints, which is what Problem 3 argues the window should
    have been counted in all along.
    """
    if unit == "byte":
        return list(text.encode("utf-8"))
    if unit == "codepoint":
        limit = 256 ** blocks
        # A codepoint that does not fit in `blocks` base-256 digits cannot be represented. With
        # blocks=2 that is everything above the BMP: emoji, historic scripts, some CJK extensions.
        # Report it rather than silently folding it onto a wrong character.
        return [ord(ch) if ord(ch) < limit else None for ch in text]
    raise ValueError("unit must be 'byte' or 'codepoint'")


def script_relative_units(text, script_id, limit=None):
    """Units for the script-relative codec of Problem 3's fix D.

    The high base-256 digit of a codepoint is a script selector, and Problem 3's E4c measured it to
    carry 0.0000 bits inside monolingual Indic text. So it is sent once per token rather than once
    per character: unit 0 is the script id, and every unit after it is the low digit of one
    character. That makes a position cost 256 rows again, the same as the byte codec, while a
    position holds a whole character.

    The returned units are ordinary 0..255 values, so `encode(..., unit="byte")` handles them
    directly and no new encoder path is needed. Inversion needs the script id to recover the high
    digit, which is exactly the trade fix D makes and the reason it is not free.
    """
    units = [script_id & 0xFF]
    for ch in text if limit is None else text[:limit]:
        units.append(ord(ch) & 0xFF)
    return units


def dim(pos_dim, unit, blocks=2):
    """Codec output dimensionality D."""
    return CHAR_DIM * pos_dim if unit == "byte" else CHAR_DIM * blocks * pos_dim


def encode(units, pos_dim, unit="byte", blocks=2, znorm=True, dtype=DTYPE):
    """Encode one token's units into the D-dimensional codec vector.

    Units beyond `pos_dim` are dropped, which is the cropping behaviour the assignment names. The
    number dropped is returned so callers never have to re-derive it.
    """
    rows = CHAR_DIM if unit == "byte" else CHAR_DIM * blocks
    m = np.zeros((rows, pos_dim), dtype=dtype)
    used = min(len(units), pos_dim)
    unrepresentable = 0
    for p in range(used):
        u = units[p]
        if u is None:
            unrepresentable += 1
            continue
        if unit == "byte":
            m[u, p] = 1.0
        else:
            for k in range(blocks):
                m[k * CHAR_DIM + ((u >> (8 * k)) & 0xFF), p] = 1.0
    v = (m / np.sqrt(pos_dim, dtype=np.float64).astype(dtype)).reshape(-1)
    if znorm:
        sd = v.std()
        v = (v - v.mean()) / (sd if sd > 0 else 1.0)
    return v.astype(dtype), max(0, len(units) - pos_dim), unrepresentable


def encode_many(seq_of_units, pos_dim, unit="byte", blocks=2, znorm=True):
    """Encode a list of tokens into an (N, D) matrix. Returns (matrix, cropped_counts)."""
    n = len(seq_of_units)
    out = np.zeros((n, dim(pos_dim, unit, blocks)), dtype=DTYPE)
    cropped = np.zeros(n, dtype=np.int32)
    for i, units in enumerate(seq_of_units):
        out[i], cropped[i], _ = encode(units, pos_dim, unit, blocks, znorm)
    return out, cropped


def decode(v, pos_dim, unit="byte", blocks=2, length=None, margin_threshold=None):
    """Invert the codec: reshape to columns, argmax down each column.

    Returns `(units, margins)`. If `length` is given it is trusted. Otherwise the token is taken to
    end at the first column whose margin falls below `margin_threshold`, which is the empty-column
    signature described in the module docstring.

    `margin_threshold=None` sets that cut at half the largest column margin *in this vector*. A fixed
    global threshold looks reasonable and is wrong: the margin after z-normalisation scales with how
    many columns are occupied, so a long token has a genuinely smaller margin than a short one, and
    any constant cut silently truncates the longest tokens. Making the threshold relative to the
    vector's own scale removes that failure mode, and costs nothing.
    """
    rows = CHAR_DIM if unit == "byte" else CHAR_DIM * blocks
    m = np.asarray(v, dtype=np.float64).reshape(rows, pos_dim)
    units, margins = [], []
    for p in range(pos_dim):
        if unit == "byte":
            col = m[:, p]
            top = int(np.argmax(col))
            srt = np.partition(col, -2)[-2:] if col.size > 1 else col
            margin = float(srt[-1] - srt[-2]) if col.size > 1 else 0.0
            units.append(top)
        else:
            u, margin = 0, np.inf
            for k in range(blocks):
                col = m[k * CHAR_DIM:(k + 1) * CHAR_DIM, p]
                d = int(np.argmax(col))
                srt = np.partition(col, -2)[-2:]
                # A codepoint is only as well recovered as its worst digit.
                margin = min(margin, float(srt[-1] - srt[-2]))
                u |= d << (8 * k)
            units.append(u)
        margins.append(margin)
    if length is None:
        cut = margin_threshold
        if cut is None:
            peak = max(margins) if margins else 0.0
            cut = 0.5 * peak
        length = pos_dim
        for p, mg in enumerate(margins):
            if mg < cut:
                length = p
                break
    return units[:length], margins


def units_to_text(units, unit):
    """Best-effort reconstruction. Invalid byte sequences are reported, not hidden."""
    if unit == "byte":
        try:
            return bytes(units).decode("utf-8"), True
        except (UnicodeDecodeError, ValueError):
            return None, False
    try:
        return "".join(chr(u) for u in units), True
    except (ValueError, OverflowError):
        return None, False
