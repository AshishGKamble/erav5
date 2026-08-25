"""
A transformer whose input side is the Kronecker codec, with three interchangeable output heads.

The transformer body is assignment-6's, imported rather than copied, so the comparison here is
against a model that has already been used and checked. What is new is the two ends:

**Input.** Assignment 6 embeds with a learned table `emb[ids]`. Here the token vector is the fixed
codec output and the only learned input-side parameter is `W`, a `Linear(D, d, bias=False)`. That is
the whole claim of the paper: the input side has no per-token parameters at all.

**Output.** Three heads, selected by string, identical in every other respect:

  * `vocab`      - a learned unembedding `U` of shape `(vocab, d)`. The baseline, and the thing whose
                   parameter count grows with the vocabulary.
  * `byte_untied`- a learned `(d, D)` head predicting per-position byte logits.
  * `byte_tied`  - the same prediction with **no new parameters at all**, reusing `Wᵀ`.

**The factored codec, which is why this runs at all.** A codec matrix over a 10k vocabulary at
D=8192 is 327 MB dense, and every token embedding would be an 8192-dimensional dot product against
a vector with about eight non-zeros. Both are avoidable. z-normalisation is affine, so

    kappa_i = (m_i / sqrt(L) - mu_i) / sigma_i

with `m_i` the 0/1 support and `mu_i`, `sigma_i` scalars available in closed form from the number of
ones. Therefore

    kappa_i @ W = (sum of the occupied rows of W) / (sqrt(L) sigma_i) - (mu_i / sigma_i) * colsum(W)

which is a handful of row lookups plus one shared rank-1 term, and the backward is a sparse scatter
plus the same rank-1 term. Nothing is approximated; `test_matches_codec` checks this against
`codec.encode` to float32 round-off.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "assignment-6", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import DTYPE, gelu, dgelu, layernorm, dlayernorm  # noqa: E402
import codec  # noqa: E402


class KronCodes:
    """The fixed codec matrix for a vocabulary, held in factored form.

    Zero learned parameters live here. `matmul` and `accumulate_grad` are the only two operations
    the model needs, and both avoid ever materialising the matrix.
    """

    def __init__(self, texts, pos_dim, unit="byte", blocks=2, units_override=None):
        """`units_override` supplies the unit lists directly, one per text.

        Problem 3's fix D needs a script id prepended to a sequence of low digits, which are still
        plain 0..255 units and so still go through the byte encoder. Passing them in avoids adding
        a third encoder path for what is only a different choice of what a unit means.
        """
        self.pos_dim, self.unit, self.blocks = pos_dim, unit, blocks
        self.D = codec.dim(pos_dim, unit, blocks)
        self.rows, self.units, self.used = [], [], []
        n_ones = np.zeros(len(texts), dtype=np.float64)
        for i, text in enumerate(texts):
            u = units_override[i] if units_override is not None else \
                codec.text_units(text, unit, blocks)
            used = min(len(u), pos_dim)
            idx = []
            for p in range(used):
                val = u[p]
                if val is None:            # above the representable range for `blocks`
                    continue
                if unit == "byte":
                    idx.append(val * pos_dim + p)
                else:
                    for k in range(blocks):
                        row = k * codec.CHAR_DIM + ((val >> (8 * k)) & 0xFF)
                        idx.append(row * pos_dim + p)
            self.rows.append(np.asarray(idx, dtype=np.int64))
            self.units.append(u[:used])
            self.used.append(used)
            n_ones[i] = len(idx)

        self._counts = np.asarray([len(r) for r in self.rows], dtype=np.int64)

        # Closed-form z-normalisation constants. The raw vector holds `n_ones` entries equal to
        # 1/sqrt(pos_dim) and is zero elsewhere, so both moments are exact rather than measured.
        D = float(self.D)
        root = np.sqrt(float(pos_dim))
        self.mu = n_ones / (root * D)
        var = n_ones / (float(pos_dim) * D) - self.mu ** 2
        self.sigma = np.sqrt(np.maximum(var, 0.0))
        self.sigma[self.sigma == 0.0] = 1.0        # a token with no representable units
        self.scale = 1.0 / (root * self.sigma)     # weight on the occupied rows
        self.shift = self.mu / self.sigma          # weight on the all-ones direction
        self.n_ones = n_ones

    def __len__(self):
        return len(self.rows)

    def nbytes(self):
        """Bytes actually held for the whole vocabulary, against the dense alternative."""
        idx = int(sum(r.nbytes for r in self.rows))
        scal = int(self.scale.nbytes + self.shift.nbytes + self._counts.nbytes)
        dense = len(self.rows) * self.D * 4        # float32
        return {"factored_bytes": idx + scal, "dense_bytes": dense,
                "ratio": dense / max(1, idx + scal),
                "mean_occupied_rows_per_token": float(self._counts.mean())}

    def dense(self, i):
        """Materialise one code. Test and diagnostic use only."""
        v = np.zeros(self.D, dtype=np.float64)
        v[self.rows[i]] = self.scale[i]
        v -= self.shift[i]
        return v

    def matmul(self, W, ids, vectorised=True):
        """E = kappa[ids] @ W, shape (..., d), without building kappa.

        The vectorised path gathers every occupied row across the whole batch at once and reduces
        them with `np.add.reduceat`, so the Python-level work is O(batch) instead of O(batch) NumPy
        calls. This matters for the honesty of Problem 3's cost comparison: a per-token Python loop
        would lose a wall-clock race against a dense BLAS matmul even though it does three orders of
        magnitude less arithmetic, and reporting that as "factoring is slower" would be an artefact
        of the implementation rather than a fact about the construction.
        """
        shape = np.asarray(ids).shape
        flat = np.asarray(ids).reshape(-1)
        colsum = W.sum(axis=0)
        if not vectorised:
            out = np.empty((flat.size, W.shape[1]), dtype=W.dtype)
            for n, i in enumerate(flat):
                out[n] = W[self.rows[i]].sum(axis=0) * self.scale[i] - self.shift[i] * colsum
            return out.reshape(shape + (W.shape[1],))

        counts = self._counts[flat]
        total = int(counts.sum())
        if total == 0:
            out = -self.shift[flat][:, None] * colsum[None, :]
            return out.astype(W.dtype).reshape(shape + (W.shape[1],))
        idx = np.concatenate([self.rows[i] for i in flat]) if flat.size else np.empty(0, np.int64)
        offsets = np.zeros(flat.size, dtype=np.int64)
        np.cumsum(counts[:-1], out=offsets[1:])
        gathered = W[idx]
        # reduceat needs every segment to be non-empty; empty tokens are patched afterwards.
        safe = np.minimum(offsets, max(total - 1, 0))
        summed = np.add.reduceat(gathered, safe, axis=0)
        summed[counts == 0] = 0.0
        out = summed * self.scale[flat][:, None] - self.shift[flat][:, None] * colsum[None, :]
        return out.astype(W.dtype).reshape(shape + (W.shape[1],))

    def accumulate_grad(self, ids, dE, out):
        """out += kappa[ids]^T @ dE, in place, shape (D, d)."""
        flat = np.asarray(ids).reshape(-1)
        dflat = dE.reshape(-1, out.shape[1])
        shared = np.zeros(out.shape[1], dtype=np.float64)
        for n, i in enumerate(flat):
            row = dflat[n]
            out[self.rows[i]] += row * self.scale[i]
            shared -= row * self.shift[i]
        out += shared           # broadcasts along the all-ones direction
        return out


def exact_code(text, pos_dim, unit, blocks):
    """The codec definition evaluated in float64, used as the reference for the factored form."""
    u = codec.text_units(text, unit, blocks)
    rows = codec.CHAR_DIM if unit == "byte" else codec.CHAR_DIM * blocks
    m = np.zeros((rows, pos_dim), dtype=np.float64)
    for p in range(min(len(u), pos_dim)):
        val = u[p]
        if val is None:
            continue
        if unit == "byte":
            m[val, p] = 1.0
        else:
            for k in range(blocks):
                m[k * codec.CHAR_DIM + ((val >> (8 * k)) & 0xFF), p] = 1.0
    v = (m / np.sqrt(pos_dim)).reshape(-1)
    sd = v.std()
    return (v - v.mean()) / (sd if sd > 0 else 1.0)


def test_matches_codec(codes, texts, atol_exact=1e-12, atol_f32=2e-5):
    """The factored form must reproduce the codec. Checked against two references, not assumed.

    Against the float64 definition it agrees to about 1e-14, which is the real correctness check.
    Against `codec.encode` it agrees to about 8e-6, and that residual is *not* an error here: it is
    float32 round-off inside `codec.encode`, which computes its standard deviation in float32 while
    the factored form gets mu and sigma in closed form. Both bounds are reported so the difference
    between "wrong" and "differently rounded" stays visible.
    """
    worst_exact = worst_f32 = 0.0
    for i, text in enumerate(texts):
        got = codes.dense(i)
        want32, _, _ = codec.encode(codec.text_units(text, codes.unit, codes.blocks),
                                    codes.pos_dim, codes.unit, codes.blocks)
        want64 = exact_code(text, codes.pos_dim, codes.unit, codes.blocks)
        worst_exact = max(worst_exact, float(np.abs(got - want64).max()))
        worst_f32 = max(worst_f32, float(np.abs(got - want32).max()))
    return {"max_abs_vs_float64_definition": worst_exact,
            "max_abs_vs_codec_encode_float32": worst_f32,
            "ok": worst_exact <= atol_exact and worst_f32 <= atol_f32}


class KronTiny:
    """Assignment-6's transformer with a Kronecker input and a selectable output head."""

    HEADS = ("vocab", "byte_untied", "byte_tied")

    def __init__(self, codes, d=96, n_layer=2, n_head=4, max_pos=512, head="byte_tied",
                 vocab=None, seed=1234):
        if head not in self.HEADS:
            raise ValueError("head must be one of %s" % (self.HEADS,))
        rng = np.random.default_rng(seed)
        sc = 0.02
        self.codes, self.head = codes, head
        self.cfg = {"d": d, "n_layer": n_layer, "n_head": n_head, "max_pos": max_pos,
                    "head": head, "D": codes.D, "pos_dim": codes.pos_dim,
                    "unit": codes.unit, "blocks": codes.blocks, "vocab": vocab, "seed": seed}
        # The input projection is scaled by 1/sqrt(D) rather than by a flat 0.02, because the codec
        # output has unit variance by construction and a flat scale would put the residual stream
        # at a wildly different scale from assignment-6's learned embedding.
        self.p = {
            "W": (rng.standard_normal((codes.D, d)) / np.sqrt(codes.D)).astype(DTYPE),
            "pos": (rng.standard_normal((max_pos, d)) * sc).astype(DTYPE),
            "lnf_g": np.ones(d, DTYPE), "lnf_b": np.zeros(d, DTYPE),
        }
        for i in range(n_layer):
            self.p[f"ln1_g{i}"] = np.ones(d, DTYPE)
            self.p[f"ln1_b{i}"] = np.zeros(d, DTYPE)
            self.p[f"ln2_g{i}"] = np.ones(d, DTYPE)
            self.p[f"ln2_b{i}"] = np.zeros(d, DTYPE)
            for nm, shape in (("wq", (d, d)), ("wk", (d, d)), ("wv", (d, d)), ("wo", (d, d)),
                              ("w1", (d, 4 * d)), ("w2", (4 * d, d))):
                self.p[f"{nm}{i}"] = (rng.standard_normal(shape) * sc).astype(DTYPE)
        if head == "vocab":
            self.p["U"] = (rng.standard_normal((vocab, d)) * sc).astype(DTYPE)
        elif head == "byte_untied":
            self.p["Hb"] = (rng.standard_normal((d, codes.D)) * sc).astype(DTYPE)

    # ------------------------------------------------------------------ params

    def n_params(self):
        return int(sum(v.size for v in self.p.values()))

    def param_breakdown(self):
        d = self.cfg["d"]
        return {
            "input_projection_W": int(self.p["W"].size),
            "input_per_token_table": 0,
            "output_head": int(self.p["U"].size if self.head == "vocab"
                               else self.p["Hb"].size if self.head == "byte_untied" else 0),
            "transformer_body": int(sum(v.size for k, v in self.p.items()
                                        if k not in ("W", "U", "Hb"))),
            "total": self.n_params(),
        }

    # ------------------------------------------------------------------ forward

    def forward(self, ids, pos_ids, seg_ids):
        B, T = ids.shape
        d, H = self.cfg["d"], self.cfg["n_head"]
        dh = d // H
        p = self.p

        # Follow the parameter dtype rather than the module default. Hardcoding float32 here
        # silently truncates the input path even when the model is run in float64, which is
        # invisible in training and shows up as a ~1e-4 relative error on W in a gradient check.
        dt = p["W"].dtype
        emb = self.codes.matmul(p["W"], ids).astype(dt)
        x = emb + p["pos"][np.clip(pos_ids, 0, self.cfg["max_pos"] - 1)]
        same = seg_ids[:, :, None] == seg_ids[:, None, :]
        real = seg_ids != 0
        causal = np.tril(np.ones((T, T), dtype=bool))[None, :, :]
        mask = same & causal & real[:, :, None] & real[:, None, :]

        cache = {"ids": ids, "pos_ids": pos_ids, "mask": mask, "xs": []}
        for i in range(self.cfg["n_layer"]):
            h, c1 = layernorm(x, p[f"ln1_g{i}"], p[f"ln1_b{i}"])
            q = (h @ p[f"wq{i}"]).reshape(B, T, H, dh).transpose(0, 2, 1, 3)
            k = (h @ p[f"wk{i}"]).reshape(B, T, H, dh).transpose(0, 2, 1, 3)
            v = (h @ p[f"wv{i}"]).reshape(B, T, H, dh).transpose(0, 2, 1, 3)
            att = (q @ k.transpose(0, 1, 3, 2))
            att *= (1.0 / np.sqrt(dh))
            m4 = mask[:, None, :, :]
            att[~np.broadcast_to(m4, att.shape)] = -1e9
            att -= att.max(-1, keepdims=True)
            np.exp(att, out=att)
            att *= m4
            att /= np.maximum(att.sum(-1, keepdims=True), 1e-9)
            sm = att
            o = (sm @ v).transpose(0, 2, 1, 3).reshape(B, T, d)
            proj = o @ p[f"wo{i}"]
            x2 = x + proj
            h2, c2 = layernorm(x2, p[f"ln2_g{i}"], p[f"ln2_b{i}"])
            f1 = h2 @ p[f"w1{i}"]
            a1 = gelu(f1)
            f2 = a1 @ p[f"w2{i}"]
            x = x2 + f2
            cache["xs"].append({"c1": c1, "h": h, "q": q, "k": k, "v": v, "sm": sm, "o": o,
                                "x2": x2, "c2": c2, "h2": h2, "f1": f1, "a1": a1})
        xf, cf = layernorm(x, p["lnf_g"], p["lnf_b"])
        cache["xf"], cache["cf"] = xf, cf
        return xf, cache

    def head_logits(self, xf):
        if self.head == "vocab":
            return xf @ self.p["U"].T
        if self.head == "byte_untied":
            return xf @ self.p["Hb"]
        return xf @ self.p["W"].T          # tied: zero new parameters

    # ------------------------------------------------------------------ backward body

    def _backward_body(self, dxf, cache, g):
        p = self.p
        d, H = self.cfg["d"], self.cfg["n_head"]
        dh = d // H
        dx, g["lnf_g"], g["lnf_b"] = dlayernorm(dxf, cache["cf"])
        for i in reversed(range(self.cfg["n_layer"])):
            c = cache["xs"][i]
            df2 = dx
            g[f"w2{i}"] += c["a1"].reshape(-1, 4 * d).T @ df2.reshape(-1, d)
            da1 = df2 @ p[f"w2{i}"].T
            df1 = da1 * dgelu(c["f1"])
            g[f"w1{i}"] += c["h2"].reshape(-1, d).T @ df1.reshape(-1, 4 * d)
            dh2 = df1 @ p[f"w1{i}"].T
            dx2_a, g[f"ln2_g{i}"], g[f"ln2_b{i}"] = dlayernorm(dh2, c["c2"])
            dx2 = dx + dx2_a
            dproj = dx2
            g[f"wo{i}"] += c["o"].reshape(-1, d).T @ dproj.reshape(-1, d)
            do = dproj @ p[f"wo{i}"].T
            do = do.reshape(dx.shape[0], dx.shape[1], H, dh).transpose(0, 2, 1, 3)
            dsm = do @ c["v"].transpose(0, 1, 3, 2)
            dv = c["sm"].transpose(0, 1, 3, 2) @ do
            datt = c["sm"] * (dsm - (dsm * c["sm"]).sum(-1, keepdims=True))
            datt = datt * cache["mask"][:, None, :, :] / np.sqrt(dh)
            dq = datt @ c["k"]
            dk = datt.transpose(0, 1, 3, 2) @ c["q"]

            def merge(t):
                return t.transpose(0, 2, 1, 3).reshape(dx.shape[0], dx.shape[1], d)

            for nm, dt in (("wq", dq), ("wk", dk), ("wv", dv)):
                g[f"{nm}{i}"] += c["h"].reshape(-1, d).T @ merge(dt).reshape(-1, d)
            dh_ = (merge(dq) @ p[f"wq{i}"].T + merge(dk) @ p[f"wk{i}"].T
                   + merge(dv) @ p[f"wv{i}"].T)
            dx1, g[f"ln1_g{i}"], g[f"ln1_b{i}"] = dlayernorm(dh_, c["c1"])
            dx = dx2 + dx1

        flat_pos = np.clip(cache["pos_ids"], 0, self.cfg["max_pos"] - 1).reshape(-1)
        dx2 = dx.reshape(-1, d)
        for j in range(d):
            g["pos"][:, j] += np.bincount(flat_pos, weights=dx2[:, j],
                                          minlength=self.cfg["max_pos"])
        # Input side: the only per-token gradient path, and it lands on W rather than on a table.
        gW = np.zeros(self.p["W"].shape, dtype=np.float64)
        self.codes.accumulate_grad(cache["ids"], dx2.astype(np.float64), gW)
        g["W"] += gW.astype(self.p["W"].dtype)
        return g

    # ------------------------------------------------------------------ losses

    @staticmethod
    def targets(ids, seg_ids, target_ids=None):
        """Next-token targets, valid only inside a segment. Assignment-6's rule, unchanged.

        `target_ids` lets the target stream differ from the input stream. That is not a convenience:
        it is what makes an open-vocabulary test possible. A word the model has no id for cannot be
        fed in as an input, but a byte head can still be asked to **emit** it, because the head
        predicts bytes rather than selecting a row of an output table. Passing the true word as the
        target while the input carries an unknown marker is exactly the situation the brief's
        "vocab of 1M" claim is about.
        """
        src = ids if target_ids is None else target_ids
        tgt = np.roll(src, -1, axis=1)
        nxt_same = seg_ids == np.roll(seg_ids, -1, axis=1)
        valid = nxt_same & (seg_ids != 0)
        valid[:, -1] = False
        return tgt, valid

    def byte_targets(self, tgt):
        """Per-position unit targets for the next token, plus which positions are occupied.

        Positions past the end of the target token carry no byte and are excluded from the loss
        rather than being given a padding class. That choice is forced by head C: a tied head emits
        exactly `256 x L` logits because `W` has `D = 256L` rows, so there is no room for a 257th
        padding class without breaking the tying that makes the head free. Length is recovered at
        decode time from the column margin instead, which E2 measured directly.
        """
        B, T = tgt.shape
        L = self.cfg["pos_dim"]
        rows = codec.CHAR_DIM if self.cfg["unit"] == "byte" else codec.CHAR_DIM * self.cfg["blocks"]
        blocks = 1 if self.cfg["unit"] == "byte" else self.cfg["blocks"]
        y = np.zeros((B, T, blocks, L), dtype=np.int64)
        occ = np.zeros((B, T, blocks, L), dtype=bool)
        flat = tgt.reshape(-1)
        yv, ov = y.reshape(-1, blocks, L), occ.reshape(-1, blocks, L)
        for n, tid in enumerate(flat):
            u = self.codes.units[tid]
            for p, val in enumerate(u):
                if val is None:
                    continue
                for k in range(blocks):
                    yv[n, k, p] = val if blocks == 1 else ((val >> (8 * k)) & 0xFF)
                    ov[n, k, p] = True
        return y, occ, rows, blocks, L

    def loss_and_grad(self, ids, pos_ids, seg_ids, objective="ce", target_ids=None):
        """Loss and gradients. `objective` is 'ce' or 'mse', the comparison E5 needs."""
        xf, cache = self.forward(ids, pos_ids, seg_ids)
        tgt, valid = self.targets(ids, seg_ids, target_ids)
        g = {k: np.zeros_like(v) for k, v in self.p.items()}
        d = self.cfg["d"]

        if self.head == "vocab":
            logits = (xf @ self.p["U"].T).astype(np.float64)
            logits -= logits.max(-1, keepdims=True)
            ztgt = np.take_along_axis(logits, tgt[:, :, None], axis=2)[:, :, 0]
            ex = np.exp(logits)
            Z = ex.sum(-1, keepdims=True)
            nll = -(ztgt - np.log(Z[:, :, 0])) * valid
            n = max(1, int(valid.sum()))
            loss = float(nll.sum() / n)
            dlog = ex / Z
            np.put_along_axis(dlog, tgt[:, :, None],
                              np.take_along_axis(dlog, tgt[:, :, None], axis=2) - 1.0, axis=2)
            dlog *= (valid / n)[:, :, None]
            dl2 = dlog.reshape(-1, self.cfg["vocab"])
            g["U"] += (dl2.T @ xf.reshape(-1, d)).astype(self.p["U"].dtype)
            dxf = (dl2 @ self.p["U"]).reshape(xf.shape).astype(xf.dtype)
            extra = {"objective": "ce", "unit_of_loss": "nats per next token over the vocabulary"}
            return loss, self._backward_body(dxf, cache, g), extra

        # ---- byte-space heads, tied or untied
        y, occ, rows, blocks, L = self.byte_targets(tgt)
        vmask = occ & valid[:, :, None, None]
        n = max(1, int(vmask.sum()))
        flat = self.head_logits(xf).astype(np.float64)          # (B, T, D)
        B, T = ids.shape
        lg = flat.reshape(B, T, blocks, codec.CHAR_DIM, L) if blocks > 1 \
            else flat.reshape(B, T, 1, codec.CHAR_DIM, L)

        if objective == "ce":
            lg = lg - lg.max(axis=3, keepdims=True)
            ex = np.exp(lg)
            Z = ex.sum(axis=3, keepdims=True)
            ytgt = np.take_along_axis(lg, y[:, :, :, None, :], axis=3)[:, :, :, 0, :]
            nll = -(ytgt - np.log(Z[:, :, :, 0, :])) * vmask
            loss = float(nll.sum() / n)
            dlg = ex / Z
            np.put_along_axis(dlg, y[:, :, :, None, :],
                              np.take_along_axis(dlg, y[:, :, :, None, :], axis=3) - 1.0, axis=3)
            dlg *= (vmask / n)[:, :, :, None, :]
        elif objective == "mse":
            # The objection's implicit assumption: regress the codec vector directly. Target is the
            # exact code of the next token, so this is a like-for-like comparison against CE.
            target = np.zeros(flat.shape, dtype=np.float64)
            tflat = tgt.reshape(-1)
            tv = target.reshape(-1, self.codes.D)
            for i, tid in enumerate(tflat):
                tv[i] = self.codes.dense(tid)
            w = valid.reshape(-1)[:, None]
            diff = (flat.reshape(-1, self.codes.D) - tv) * w
            nvalid = max(1, int(valid.sum()))
            loss = float((diff ** 2).sum() / (nvalid * self.codes.D))
            dflat = (2.0 * diff / (nvalid * self.codes.D))
            dlg = dflat.reshape(lg.shape)
        else:
            raise ValueError("objective must be 'ce' or 'mse'")

        dflat = dlg.reshape(B, T, self.codes.D)
        d2 = dflat.reshape(-1, self.codes.D)
        if self.head == "byte_untied":
            g["Hb"] += (xf.reshape(-1, d).T @ d2).astype(self.p["Hb"].dtype)
            dxf = (d2 @ self.p["Hb"].T).reshape(xf.shape).astype(xf.dtype)
        else:
            # Tied: W is used as the input projection AND as the unembedding, so it collects
            # gradient from both ends. This term is the output end; the input end is added inside
            # _backward_body. Getting only one of the two is the obvious way to break tying, and it
            # is what the gradient check below is for.
            g["W"] += (d2.T @ xf.reshape(-1, d)).astype(self.p["W"].dtype)
            dxf = (d2 @ self.p["W"]).reshape(xf.shape).astype(xf.dtype)
        extra = {"objective": objective, "positions_scored": n,
                 "unit_of_loss": ("nats per occupied byte position" if objective == "ce"
                                  else "mean squared error per codec dimension")}
        return loss, self._backward_body(dxf, cache, g), extra


class Adam:
    """Plain Adam. Same defaults assignment-6 trained with."""

    def __init__(self, params, lr=3e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        for k in params:
            gk = grads[k]
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * gk
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * (gk * gk)
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            params[k] -= (self.lr * mh / (np.sqrt(vh) + self.eps)).astype(params[k].dtype)


def grad_check(model, ids, pos_ids, seg_ids, keys, objective="ce", eps=1e-3, n_probe=6, seed=7):
    """Central-difference check on a few entries of each named parameter.

    Run in float64 on the parameter being probed. The tied head is the reason this exists: `W`
    receives gradient through the input projection and through the unembedding, and a plausible
    implementation that adds only one of them still trains, just wrongly.
    """
    rng = np.random.default_rng(seed)
    base_loss, g, _ = model.loss_and_grad(ids, pos_ids, seg_ids, objective=objective)
    out = {}
    for k in keys:
        P = model.p[k]
        flat = P.reshape(-1)
        picks = rng.choice(flat.size, size=min(n_probe, flat.size), replace=False)
        num, ana = [], []
        for i in picks:
            orig = float(flat[i])
            flat[i] = orig + eps
            lp, _, _ = model.loss_and_grad(ids, pos_ids, seg_ids, objective=objective)
            flat[i] = orig - eps
            lm, _, _ = model.loss_and_grad(ids, pos_ids, seg_ids, objective=objective)
            flat[i] = orig
            num.append((lp - lm) / (2 * eps))
            ana.append(float(g[k].reshape(-1)[i]))
        num, ana = np.array(num), np.array(ana)
        denom = np.maximum(np.abs(num) + np.abs(ana), 1e-12)
        out[k] = {"max_rel_error": float(np.abs(num - ana).max() / denom.max()),
                  "numeric": num.tolist(), "analytic": ana.tolist()}
    return out
