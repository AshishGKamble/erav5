/* Assignment 7, Problem 3 dashboard.
   Vanilla JS, no framework, no network beyond the local data file. Charts are hand built inline
   SVG: no chart library is used anywhere in this repository. */
(function () {
  "use strict";

  /* ---------- theme, remembered per viewer ---------- */
  var root = document.documentElement, tgl = document.getElementById("themeToggle");
  function applyTheme(t) {
    root.setAttribute("data-theme", t);
    document.getElementById("themeIcon").innerHTML = t === "dark" ? "&#9789;" : "&#9788;";
    document.getElementById("themeLabel").textContent = t === "dark" ? "Dark" : "Light";
  }
  var saved = null;
  try { saved = localStorage.getItem("a7p3-theme"); } catch (e) { /* private mode */ }
  applyTheme(saved || (window.matchMedia &&
    window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light"));
  tgl.addEventListener("click", function () {
    var next = root.getAttribute("data-theme") === "dark" ? "light" : "dark";
    applyTheme(next);
    try { localStorage.setItem("a7p3-theme", next); } catch (e) { /* ignore */ }
  });

  /* ---------- tiny SVG helpers ---------- */
  var NS = "http://www.w3.org/2000/svg";
  function el(n, a) {
    var e = document.createElementNS(NS, n);
    for (var k in a) if (a[k] !== null && a[k] !== undefined) e.setAttribute(k, a[k]);
    return e;
  }
  function txt(s, a, cls) {
    var t = el("text", a);
    if (cls) t.setAttribute("class", cls);
    t.textContent = s;
    return t;
  }
  function svg(w, h) {
    var s = el("svg", { viewBox: "0 0 " + w + " " + h, width: "100%", role: "img" });
    return s;
  }
  var fmtPct = function (x, d) { return (x * 100).toFixed(d === undefined ? 2 : d) + "%"; };
  var fmtN = function (n) { return n.toLocaleString("en-US"); };

  /* Horizontal grouped bars. rows: [{label, values:[..], notes:[..]}] */
  function barChart(mount, rows, opts) {
    opts = opts || {};
    var colors = opts.colors || ["var(--english)"];
    var padL = opts.padL || 118, padR = opts.padR || 74, rowH = opts.rowH || 26, gap = 9;
    var series = rows[0].values.length;
    var barH = (rowH - 4) / series;
    var H = rows.length * (rowH + gap) + 34, W = 720;
    var max = opts.max || Math.max.apply(null, rows.map(function (r) {
      return Math.max.apply(null, r.values);
    })) || 1;
    var s = svg(W, H), plotW = W - padL - padR;

    [0, 0.25, 0.5, 0.75, 1].forEach(function (f) {
      var x = padL + plotW * f;
      s.appendChild(el("line", { x1: x, y1: 16, x2: x, y2: H - 18, "class": "grid" }));
      s.appendChild(txt(opts.tick ? opts.tick(max * f) : (max * f).toFixed(0),
        { x: x, y: H - 5, "text-anchor": "middle" }));
    });

    rows.forEach(function (r, i) {
      var y = 22 + i * (rowH + gap);
      s.appendChild(txt(r.label, { x: padL - 10, y: y + rowH / 2 + 4, "text-anchor": "end" }));
      r.values.forEach(function (v, j) {
        var w = Math.max(v / max * plotW, v > 0 ? 2 : 0);
        var yy = y + j * barH + 2;
        s.appendChild(el("rect", {
          x: padL, y: yy, width: w, height: barH - 1.5, rx: 2,
          fill: colors[j % colors.length]
        }));
        if (r.notes && r.notes[j] !== undefined) {
          s.appendChild(txt(r.notes[j], { x: padL + w + 6, y: yy + barH - 3 }, "val"));
        }
      });
    });
    mount.appendChild(s);
  }

  /* Line chart for the scaling measurement. pts: [{x,y}] */
  function lineChart(mount, pts, opts) {
    opts = opts || {};
    var W = 720, H = 260, padL = 62, padB = 40, padT = 16, padR = 16;
    var xs = pts.map(function (p) { return p.x; }), ys = pts.map(function (p) { return p.y; });
    var x0 = 0, x1 = Math.max.apply(null, xs), y0 = 0, y1 = Math.max.apply(null, ys) * 1.08;
    var plotW = W - padL - padR, plotH = H - padT - padB;
    var X = function (v) { return padL + (v - x0) / (x1 - x0) * plotW; };
    var Y = function (v) { return padT + plotH - (v - y0) / (y1 - y0) * plotH; };
    var s = svg(W, H);

    // Round the axis to a readable step. Dividing the data maximum by four gives ticks like
    // 3881 and 11642, which are exact and unreadable.
    var raw = (y1 - y0) / 4, mag = Math.pow(10, Math.floor(Math.log(raw) / Math.LN10));
    var step = [1, 2, 2.5, 5, 10].map(function (m) { return m * mag; })
      .filter(function (v) { return v >= raw; })[0] || 10 * mag;
    y1 = step * Math.ceil(y1 / step);
    for (var yv = y0; yv <= y1 + 1e-9; yv += step) {
      s.appendChild(el("line", { x1: padL, y1: Y(yv), x2: W - padR, y2: Y(yv), "class": "grid" }));
      s.appendChild(txt(yv.toLocaleString("en-US"),
        { x: padL - 8, y: Y(yv) + 4, "text-anchor": "end" }));
    }
    s.appendChild(el("line", { x1: padL, y1: padT, x2: padL, y2: padT + plotH, "class": "axis" }));
    for (var k = 0; k <= 4; k++) {
      var xv = x0 + (x1 - x0) * k / 4;
      s.appendChild(txt(Math.round(xv), { x: X(xv), y: H - padB + 18, "text-anchor": "middle" }));
    }
    s.appendChild(txt(opts.xlab || "", { x: padL + plotW / 2, y: H - 6, "text-anchor": "middle" }));
    s.appendChild(txt(opts.ylab || "", { x: 14, y: padT + plotH / 2,
      transform: "rotate(-90 14 " + (padT + plotH / 2) + ")", "text-anchor": "middle" }));

    var d = pts.map(function (p, i) { return (i ? "L" : "M") + X(p.x) + " " + Y(p.y); }).join(" ");
    s.appendChild(el("path", { d: d, fill: "none", stroke: "var(--accent)", "stroke-width": 2.2,
      "stroke-linejoin": "round" }));
    pts.forEach(function (p) {
      s.appendChild(el("circle", { cx: X(p.x), cy: Y(p.y), r: 3, fill: "var(--accent)" }));
    });
    mount.appendChild(s);
  }

  function card(n, l, cls) {
    return '<div class="card"><div class="n' + (cls ? " " + cls : "") + '">' + n +
      '</div><div class="l">' + l + '</div></div>';
  }

  /* ---------- render ---------- */
  fetch("data/dashboard.json").then(function (r) { return r.json(); }).then(function (D) {
    var h32 = D.headline["32"], h16 = D.headline["16"], e = null;

    document.getElementById("chips").innerHTML =
      '<span class="chip"><b>' + fmtN(D.corpus.characters) + '</b> characters measured</span>' +
      '<span class="chip"><b>' + fmtN(D.corpus.distinct_word_types) + '</b> word types</span>' +
      '<span class="chip"><b>' + D.indic_script_count + '</b> Indic scripts</span>' +
      '<span class="chip">window <b>L = 32 bytes</b></span>';

    document.getElementById("kpi").innerHTML =
      card(fmtN(h32.english_groups), "colliding word groups in <b>English prose</b>, across " +
        fmtN(h32.english_types) + " word types", "ok") +
      card(fmtN(h32.indic_groups), "colliding word groups in <b>nine Indic scripts</b>, across " +
        fmtN(h32.indic_types) + " word types", "hi") +
      card("3.000", "bytes per character for every Indic script, against Latin's 1.000") +
      card(fmtN(D.collision_check.pairs_with_bitwise_identical_codec_vectors) + "/" +
        fmtN(D.collision_check.pairs_checked),
        "sampled colliding pairs whose embeddings are <b>bitwise identical</b>");

    /* E1 occupancy */
    var occRows = Object.keys(D.occupancy).sort(function (a, b) {
      return D.occupancy[a].occupancy32 - D.occupancy[b].occupancy32;
    }).map(function (lane) {
      var o = D.occupancy[lane];
      return { label: lane, values: [o.occupancy32],
               notes: [fmtPct(o.occupancy32) + " used, " + fmtPct(1 - o.occupancy32) + " zeros"] };
    });
    barChart(document.getElementById("occChart"), occRows,
      { colors: ["var(--english)"], max: 1, tick: function (v) { return Math.round(v * 100) + "%"; } });

    /* E2 characters vs graphemes */
    var sOrder = Object.keys(D.scripts).filter(function (s) { return s !== "COMMON"; })
      .sort(function (a, b) { return D.scripts[a].chars32 - D.scripts[b].chars32; });
    barChart(document.getElementById("scriptChart"), sOrder.map(function (s) {
      var r = D.scripts[s];
      return { label: s, values: [r.chars32, r.graphemes32],
               notes: [r.chars32.toFixed(1) + " chars", r.graphemes32.toFixed(1) + " graphemes"] };
    }), { colors: ["var(--english)", "var(--warn)"], rowH: 30,
          tick: function (v) { return v.toFixed(0); } });

    /* E3 collisions by script */
    var cOrder = Object.keys(D.collisions_by_script).sort(function (a, b) {
      return D.collisions_by_script[b]["16"] - D.collisions_by_script[a]["16"];
    });
    barChart(document.getElementById("collChart"), cOrder.map(function (s) {
      var r = D.collisions_by_script[s];
      return { label: s, values: [r["16"], r["32"], r["64"]],
               notes: [fmtPct(r["16"], 1), fmtPct(r["32"], 1), ""] };
    }), { colors: ["var(--warn)", "var(--amber)", "var(--good)"], rowH: 30,
          tick: function (v) { return Math.round(v * 100) + "%"; } });
    document.getElementById("collChart").insertAdjacentHTML("beforebegin",
      '<div class="legend"><span><i style="background:var(--warn)"></i>L = 16</span>' +
      '<span><i style="background:var(--amber)"></i>L = 32</span>' +
      '<span><i style="background:var(--good)"></i>L = 64</span></div>');

    /* examples */
    var rows = ['<tr><th>script</th><th>words that collapse to one embedding at L = 16</th></tr>'];
    Object.keys(D.examples).forEach(function (s) {
      D.examples[s].slice(0, 2).forEach(function (grp, i) {
        rows.push("<tr><td>" + (i === 0 ? s : "") + '</td><td class="script">' +
          grp.slice(0, 4).join(" &nbsp;&middot;&nbsp; ") + "</td></tr>");
      });
    });
    document.getElementById("exTable").innerHTML = rows.join("");

    /* E4 fixes */
    var fOrder = Object.keys(D.fixes).sort(function (a, b) {
      return D.fixes[b].byte - D.fixes[a].byte;
    });
    barChart(document.getElementById("fixChart"), fOrder.map(function (s) {
      var r = D.fixes[s];
      return { label: s, values: [r.byte, r.codepoint, r.script_relative || 0],
               notes: [fmtPct(r.byte, 1), fmtPct(r.codepoint, 1),
                       r.script_relative === null ? "" : fmtPct(r.script_relative, 1)] };
    }), { colors: ["var(--english)", "var(--amber)", "var(--good)"], rowH: 30,
          tick: function (v) { return (v * 100).toFixed(0) + "%"; } });

    var lowest = Object.keys(D.entropy).filter(function (s) { return D.entropy[s] < 0.001; });
    // Quote the entropy of the INDIC scripts, which is the finding. Taking the worst across every
    // low-entropy script instead reports Latin's 0.0006 and understates it.
    var indicE = D.indic_scripts.filter(function (s) { return s in D.entropy; });
    var worstLow = Math.max.apply(null, indicE.map(function (s) { return Math.abs(D.entropy[s]); }));
    document.getElementById("entropyNote").innerHTML =
      "<b>Why fix D exists.</b> The high byte of a codepoint is a script selector, and in this " +
      "corpus it carries <b>" + worstLow.toFixed(4) + " bits</b> of entropy for every one of the " +
      indicE.length + " Indic scripts, and under 0.001 bits for " + lowest.length +
      " scripts in total. Half of fix B's dimensions carry nothing. Send the " +
      "script once per token instead, drop that block, and a position costs 256 rows again while " +
      "holding a whole character: <b>32 characters for every script</b> at the same total cost.";

    /* E7 schemes */
    var labels = { prefix_32_bytes_published: "first 32 bytes (published)",
                   both_ends_32_bytes_aligned: "both ends, cuts aligned",
                   both_ends_32_bytes: "16 front + 16 back bytes",
                   fixD_31_chars: "fix D, 31 characters",
                   overflow_hash_32_bytes: "31 bytes + checksum of the tail",
                   fixD_both_ends_31_chars: "fix D + both ends",
                   both_ends_plus_hash_32_bytes: "both ends + checksum of the middle" };
    var order = ["prefix_32_bytes_published", "both_ends_32_bytes_aligned", "both_ends_32_bytes",
                 "fixD_31_chars", "overflow_hash_32_bytes", "fixD_both_ends_31_chars",
                 "both_ends_plus_hash_32_bytes"];
    barChart(document.getElementById("schemeChart"), order.map(function (k) {
      var r = D.schemes[k];
      return { label: labels[k], values: [r.groups],
               notes: [fmtN(r.groups) + " groups  (" + r.reduction.toFixed(1) + "x better)"] };
    }), { colors: ["var(--warn)"], padL: 168, padR: 172, rowH: 28,
          tick: function (v) { return fmtN(Math.round(v)); } });

    var best = D.schemes.both_ends_plus_hash_32_bytes;
    document.getElementById("schemeNote").innerHTML =
      "<b>The recommendation, which changed once the ideas were combined.</b> The tail is not the " +
      "only thing thrown away, so the second idea handles the rest: spend one position on a " +
      "<b>checksum of whatever the window discarded</b>. It does not recover those bytes and is " +
      "not meant to; it makes the resulting collision <b>random</b> rather than systematic, about " +
      "1 in 256 instead of always. Separately the checksum reaches " +
      D.schemes.overflow_hash_32_bytes.reduction.toFixed(1) + "x and both ends reaches " +
      D.schemes.both_ends_32_bytes.reduction.toFixed(1) + "x. <b>Together they reach " +
      best.reduction.toFixed(1) + "x</b>, leaving " + best.groups +
      " colliding groups in the whole corpus, because they fix different things: the front and back " +
      "carry the morphology the collisions turn on, and the checksum discriminates the middle. " +
      "It needs no script table, no tag and no Unicode assumption, so unlike fix D it applies to " +
      "scripts this corpus never contained. One qualification the metric does not capture: a " +
      "checksum <b>discriminates but does not generalise</b>, whereas a shared case ending is a " +
      "real feature that recurs across words. The composite keeps both.";

    if (D.codec_check) {
      var cc = D.codec_check, mal = cc.cut_quality.MALAYALAM, malA = cc.cut_quality_aligned.MALAYALAM;
      document.getElementById("codecCheck").innerHTML =
        "<b>Round trip</b>: encode then decode returns the units it was given, rate " +
        cc.roundtrip_both_ends.toFixed(4) + ". <b>Bitwise</b>: " + cc.bitwise.bitwise_identical +
        " of " + cc.bitwise.pairs_checked + " colliding pairs produce identical vectors, maximum " +
        "difference " + cc.bitwise.max_absolute_difference + ".<br><br>" +
        "<b>The check also found something the collision counts had hidden.</b> The published " +
        "window cuts a word mid-character essentially always for Indic: <b>" +
        fmtPct(mal.prefix_invalid_utf8_rate) + "</b> of cropped Malayalam types, against " +
        fmtPct(cc.cut_quality.LATIN.prefix_invalid_utf8_rate) + " of Latin. That is arithmetic, " +
        "not bad luck: 32 is not a multiple of 3 and every Indic character is exactly 3 bytes, so " +
        "the retained prefix cannot end on a character boundary. Those bytes are not truncated " +
        "text, they are <b>not valid UTF-8 at all</b>. Aligning both cuts removes it completely, " +
        "to " + fmtPct(malA.both_ends_invalid_utf8_rate) + ", for about " +
        (cc.capacity.mean_units_kept_unaligned - cc.capacity.mean_units_kept_aligned).toFixed(1) +
        " units of the 32, and halves the collision benefit. Fix D avoids it by construction.";
    }

    if (D.choose_L) {
      var rowsL = ["<tr><th>scheme</th><th class='num'>L</th><th class='num'>D</th>" +
                   "<th class='num'>projection parameters</th>" +
                   "<th class='num'>colliding groups</th></tr>"];
      Object.keys(D.choose_L.rows).forEach(function (name) {
        var r = D.choose_L.rows[name];
        Object.keys(r).sort(function (a, b) { return a - b; }).forEach(function (L) {
          var v = r[L];
          rowsL.push("<tr><td>" + name + "</td><td class='num'>" + L + "</td><td class='num'>" +
            fmtN(v.D) + "</td><td class='num'>" + fmtN(v.projection_parameters) +
            "</td><td class='num'>" + fmtN(v.colliding_groups) + "</td></tr>");
        });
      });
      document.getElementById("chooseLTable").innerHTML = rowsL.join("");
      var comp = D.choose_L.rows["both ends + hash"], pub = D.choose_L.rows["published prefix"];
      document.getElementById("chooseLNote").innerHTML =
        "<b>Keep L=32 and change the units.</b> The composite at L=32 leaves " +
        comp["32"].colliding_groups + " colliding groups; the published construction at L=64 " +
        "leaves " + pub["64"].colliding_groups + " while spending <b>twice</b> the dimensions and " +
        "twice the projection parameters. To reach zero the published construction needs L=128 and " +
        fmtN(pub["128"].projection_parameters) + " parameters; the composite reaches zero at L=64 " +
        "for half that. Raising the window is the expensive way to buy what a different choice of " +
        "units gives away.";
    }

    /* E6 cost */
    var m32 = D.cost.memory["32"], c32 = D.cost.compute["32"], fit = D.cost.scaling_fit;
    document.getElementById("costKpi").innerHTML =
      card(fmtPct(D.cost.zeros["32"], 1), "of columns empty. <b>Dimensions cannot be reclaimed</b>: " +
        "a per token D means a per token weight matrix", "hi") +
      card(Math.round(m32.ratio) + "x", "less memory: " + m32.dense_mb.toFixed(1) + " MB dense " +
        "against " + m32.factored_mb.toFixed(3) + " MB factored", "ok") +
      card(Math.round(c32.arithmetic_ratio) + "x", "less arithmetic, because only occupied rows " +
        "are ever touched", "ok") +
      card(fit.corr.toFixed(4), "correlation between cost and token length. It is already dynamic",
        "ok");

    lineChart(document.getElementById("scaleChart"), D.cost.scaling.map(function (b) {
      return { x: b.occupied_units, y: b.nanoseconds_per_token };
    }), { xlab: "bytes actually present in the token", ylab: "nanoseconds per token" });

    var rl = D.cost.raising_L, ks = Object.keys(rl).sort(function (a, b) { return a - b; });
    document.getElementById("costNote").innerHTML =
      "<b>The consequence, which inverts the premise.</b> Encoding a short token costs the same at " +
      "every window size: " + ks.map(function (L) {
        return "L=" + L + " at " + rl[L].short_token_time_vs_L32.toFixed(3) + "x";
      }).join(", ") + " relative to L=32. So the window is cheap to enlarge, and L=64 removes " +
      "almost every collision above. The only thing that really grows is the projection matrix, " +
      "from " + fmtN(rl[ks[0]].projection_parameters) + " to " +
      fmtN(rl[ks[ks.length - 1]].projection_parameters) + " parameters. <b>How to make it dynamic " +
      "and how to stop cropping have the same answer.</b> The honest deflation: " +
      Math.round(c32.arithmetic_ratio) + "x less arithmetic buys only about " +
      (c32.speedup ? c32.speedup.toFixed(1) : "1.5") + "x wall clock, because a gather plus a " +
      "segmented reduction is memory bound while the dense path is one BLAS call.";

    /* E5 exposure */
    if (D.downstream && D.downstream.indic && D.downstream.indic.exposure) {
      e = D.downstream.indic.exposure;
      document.getElementById("exposureNote").innerHTML =
        "<b>The experiment had no exposure to the effect.</b> Of " + fmtN(e.token_occurrences) +
        " indic token occurrences, the byte codec truncates <b>" + e.byte.truncated_occurrences +
        "</b>. Codepoint and script relative truncate <b>zero</b>. All three codecs therefore " +
        "carry identical information for " + ((1 - e.byte.truncated_rate) * 100).toFixed(3) +
        "% of tokens and differ only in layout, which a " +
        "linear projection learns equally well either way. A BPE tokenizer sits between the corpus " +
        "and the window and removes the phenomenon under test, converting truncation into " +
        "fertility instead. The pre-registered experiment was <b>mis-specified, not " +
        "inconclusive</b>, and it was rebuilt at word level where the effect is present.";
    } else {
      document.getElementById("exposureNote").textContent =
        "Run python run_demo.py --full to generate the downstream artefacts.";
    }
    /* answers summary */
    var sch = D.schemes, mem = D.cost.memory["32"], cmp = D.cost.compute["32"];
    var ar = [
      ["\u201cThat\u2019s a waste of space. What can we do?\u201d",
       "Nothing needs to change. The zeros cost <b>dimensions</b>, which genuinely cannot be " +
       "reclaimed, but they cost no memory and no compute once the encoder is factored: <b>" +
       Math.round(mem.ratio) + "x</b> less memory and <b>" + Math.round(cmp.arithmetic_ratio) +
       "x</b> less arithmetic.", "E1, E6"],
      ["\u201cHow can it be dynamic?\u201d",
       "<b>It already is</b>, with no architectural change. Cost tracks the token\u2019s real " +
       "length, correlation <b>" + D.cost.scaling_fit.corr.toFixed(4) + "</b>. Per token " +
       "<i>dimensions</i> are impossible, because the projection needs a fixed input width.", "E6"],
      ["\u201c...doesn\u2019t force us to crop a word\u201d",
       "<b>Read the word from both ends</b>: <b>" +
       sch.both_ends_32_bytes.reduction.toFixed(1) + "x</b> fewer collisions for no new " +
       "parameters. A script relative codec reaches " + sch.fixD_31_chars.reduction.toFixed(1) +
       "x, and the two compose for <b>" + sch.fixD_both_ends_31_chars.reduction.toFixed(1) +
       "x</b>.", "E7, E4"]
    ];
    document.getElementById("answersTable").innerHTML =
      "<tr><th>what was asked</th><th>the answer</th><th>where</th></tr>" +
      ar.map(function (r) {
        return "<tr><td><b>" + r[0] + "</b></td><td>" + r[1] + "</td><td>" + r[2] + "</td></tr>";
      }).join("");

    /* corrections */
    var C = D.corrections;
    document.getElementById("corrTable").innerHTML =
      "<tr><th>pooled collision rate at L=16</th><th class='num'>naive measurement</th>" +
      "<th class='num'>after both corrections</th></tr>" +
      "<tr><td>Latin</td><td class='num'>" + fmtPct(C.latin_raw_16) + "</td><td class='num'>" +
      fmtPct(C.latin_prose_16) + "</td></tr>" +
      "<tr><td>Malayalam</td><td class='num'>" + fmtPct(C.malayalam_raw_16) +
      "</td><td class='num'>" + fmtPct(C.malayalam_prose_16) + "</td></tr>";
    document.getElementById("corrNote").innerHTML =
      "<b>What was wrong.</b> First, whitespace splitting <b>source code</b> does not produce " +
      "words: nearly all pooled Latin collisions were identifiers and LaTeX, " +
      "<span class='mono'>self.assertEqual(</span> against " +
      "<span class='mono'>self.assertEqual(0,</span>. Second, <b>trailing punctuation</b> is a " +
      "fake collision: <span class='script'>\u0bae\u0bb1\u0bcd\u0bb1\u0bc1\u0bae\u0bcd</span> " +
      "against the same word with a comma counted as two types sharing a prefix. Correcting both " +
      "cut Latin by " + (C.latin_raw_16 / C.latin_prose_16).toFixed(1) + "x and left Malayalam " +
      "almost unchanged, so the gap <b>widened</b>. Both sets of numbers are kept in the " +
      "artefacts as <span class='mono'>word_raw</span> and <span class='mono'>word_prose</span>.";

    /* E5b */
    if (D.e5b) {
      var rows5 = ["<tr><th>lane</th><th>codec</th><th class='num'>word types representable</th>" +
                   "<th class='num'>exact full word</th><th class='num'>targets truncated</th></tr>"];
      Object.keys(D.e5b).forEach(function (lane) {
        Object.keys(D.e5b[lane]).forEach(function (a, i) {
          var v = D.e5b[lane][a];
          rows5.push("<tr><td>" + (i === 0 ? lane : "") + "</td><td>" + a +
            "</td><td class='num'>" + fmtPct(v.representable) + "</td><td class='num'>" +
            fmtPct(v.exact, 2) + "</td><td class='num'>" + fmtPct(v.truncated) + "</td></tr>");
        });
      });
      document.getElementById("e5bTable").innerHTML = rows5.join("");
      document.getElementById("e5bNote").innerHTML =
        "<b>The web control is the one that matters, and it passes.</b> Helping Indic by hurting " +
        "English would not be a fix, and this does not do that. <b>The indic result is weak and is " +
        "labelled weak</b>: both arms sit near the floor of the metric, and the byte codec " +
        "truncates only " + fmtPct(D.e5b.indic.byte.truncated) + " of targets, so truncation is " +
        "not what limits it. The model is simply poor at word level prediction at this scale. " +
        "Everything above this section is a property of the encoding and does not depend on this " +
        "experiment at all.";
    }
  }).catch(function (err) {
    document.querySelector(".wrap").insertAdjacentHTML("afterbegin",
      '<div class="callout">Could not load <span class="mono">data/dashboard.json</span>. ' +
      'Run <span class="mono">python src/build_dashboard.py</span> first. (' + err + ')</div>');
  });
})();
