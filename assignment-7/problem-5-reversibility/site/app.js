/* Assignment 7, Problem 5 dashboard.
   Vanilla JS, no framework, no network beyond the local data file. Charts are hand built inline
   SVG: no chart library is used anywhere in this repository. */
(function () {
  "use strict";

  var root = document.documentElement, tgl = document.getElementById("themeToggle");
  function applyTheme(t) {
    root.setAttribute("data-theme", t);
    document.getElementById("themeIcon").innerHTML = t === "dark" ? "&#9789;" : "&#9788;";
    document.getElementById("themeLabel").textContent = t === "dark" ? "Dark" : "Light";
  }
  var saved = null;
  try { saved = localStorage.getItem("a7p5-theme"); } catch (e) { /* private mode */ }
  applyTheme(saved || (window.matchMedia &&
    window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light"));
  tgl.addEventListener("click", function () {
    var next = root.getAttribute("data-theme") === "dark" ? "light" : "dark";
    applyTheme(next);
    try { localStorage.setItem("a7p5-theme", next); } catch (e) { /* ignore */ }
  });

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
  function svg(w, h) { return el("svg", { viewBox: "0 0 " + w + " " + h, width: "100%", role: "img" }); }
  var fmtPct = function (x, d) { return (x * 100).toFixed(d === undefined ? 2 : d) + "%"; };
  var fmtN = function (n) { return n.toLocaleString("en-US"); };

  /* Multi-series line chart. series: [{pts:[{x,y}], color, label}] */
  function lineChart(mount, series, opts) {
    opts = opts || {};
    var W = 720, H = 280, padL = 64, padB = 44, padT = 16, padR = 18;
    var allX = [], allY = [];
    series.forEach(function (s) {
      s.pts.forEach(function (p) { allX.push(p.x); allY.push(p.y); });
    });
    var x0 = opts.x0 !== undefined ? opts.x0 : Math.min.apply(null, allX);
    var x1 = Math.max.apply(null, allX);
    var y0 = 0, y1 = opts.y1 !== undefined ? opts.y1 : Math.max.apply(null, allY) * 1.05;
    var plotW = W - padL - padR, plotH = H - padT - padB;
    var lg = !!opts.logx;
    var LX = function (v) { return lg ? Math.log(Math.max(v, 1e-9)) : v; };
    var X = function (v) { return padL + (LX(v) - LX(x0)) / (LX(x1) - LX(x0)) * plotW; };
    var Y = function (v) { return padT + plotH - (v - y0) / (y1 - y0) * plotH; };
    var s = svg(W, H);

    // Explicit tick values where given, so a chart with headroom above its data does not
    // label its gridlines 26%, 51%, 77%, 102%.
    var yticks = opts.yticks;
    if (!yticks) {
      yticks = [];
      for (var i = 0; i <= 4; i++) yticks.push(y0 + (y1 - y0) * i / 4);
    }
    yticks.forEach(function (yv) {
      s.appendChild(el("line", { x1: padL, y1: Y(yv), x2: W - padR, y2: Y(yv), "class": "grid" }));
      s.appendChild(txt(opts.ytick ? opts.ytick(yv) : yv.toFixed(1),
        { x: padL - 8, y: Y(yv) + 4, "text-anchor": "end" }));
    });
    s.appendChild(el("line", { x1: padL, y1: padT, x2: padL, y2: padT + plotH, "class": "axis" }));
    (opts.xticks || []).forEach(function (v) {
      s.appendChild(txt(opts.xtick ? opts.xtick(v) : v, { x: X(v), y: H - padB + 18,
        "text-anchor": "middle" }));
      s.appendChild(el("line", { x1: X(v), y1: padT, x2: X(v), y2: padT + plotH, "class": "grid" }));
    });
    s.appendChild(txt(opts.xlab || "", { x: padL + plotW / 2, y: H - 8, "text-anchor": "middle" }));
    s.appendChild(txt(opts.ylab || "", { x: 14, y: padT + plotH / 2,
      transform: "rotate(-90 14 " + (padT + plotH / 2) + ")", "text-anchor": "middle" }));

    series.forEach(function (ser) {
      var d = ser.pts.map(function (p, i) { return (i ? "L" : "M") + X(p.x) + " " + Y(p.y); }).join(" ");
      s.appendChild(el("path", { d: d, fill: "none", stroke: ser.color, "stroke-width": 2.2,
        "stroke-linejoin": "round", "stroke-dasharray": ser.dash || null }));
      ser.pts.forEach(function (p) {
        s.appendChild(el("circle", { cx: X(p.x), cy: Y(p.y), r: 2.8, fill: ser.color }));
      });
    });
    mount.appendChild(s);
  }

  function barChart(mount, rows, opts) {
    opts = opts || {};
    var colors = opts.colors || ["var(--english)"];
    var padL = opts.padL || 150, padR = 78, rowH = opts.rowH || 27, gap = 9;
    var series = rows[0].values.length, barH = (rowH - 4) / series;
    var H = rows.length * (rowH + gap) + 30, W = 720;
    var max = opts.max || Math.max.apply(null, rows.map(function (r) {
      return Math.max.apply(null, r.values);
    })) || 1;
    var s = svg(W, H), plotW = W - padL - padR;
    [0, 0.5, 1].forEach(function (f) {
      var x = padL + plotW * f;
      s.appendChild(el("line", { x1: x, y1: 12, x2: x, y2: H - 16, "class": "grid" }));
      s.appendChild(txt(opts.tick ? opts.tick(max * f) : max * f,
        { x: x, y: H - 3, "text-anchor": "middle" }));
    });
    rows.forEach(function (r, i) {
      var y = 18 + i * (rowH + gap);
      s.appendChild(txt(r.label, { x: padL - 10, y: y + rowH / 2 + 4, "text-anchor": "end" }));
      r.values.forEach(function (v, j) {
        var w = Math.max(v / max * plotW, v > 0 ? 2 : 0);
        var yy = y + j * barH + 2;
        s.appendChild(el("rect", { x: padL, y: yy, width: w, height: barH - 1.5, rx: 2,
          fill: colors[j % colors.length] }));
        if (r.notes && r.notes[j] !== undefined) {
          s.appendChild(txt(r.notes[j], { x: padL + w + 6, y: yy + barH - 3 }, "val"));
        }
      });
    });
    mount.appendChild(s);
  }

  function card(n, l, cls) {
    return '<div class="card"><div class="n' + (cls ? " " + cls : "") + '">' + n +
      '</div><div class="l">' + l + '</div></div>';
  }

  fetch("data/dashboard.json").then(function (r) { return r.json(); }).then(function (D) {
    var r32 = D.roundtrip["32"];

    document.getElementById("chips").innerHTML =
      '<span class="chip"><b>' + fmtN(D.vocab) + '</b> tokens measured</span>' +
      '<span class="chip">window <b>L = 32</b>, D = ' + fmtN(r32.D) + '</span>' +
      '<span class="chip">codes are <b>' + D.sparsity.k.toFixed(2) + '</b> sparse</span>' +
      (D.config ? '<span class="chip">d_model <b>' + D.config.d_model + '</b>, ' +
        D.config.steps + ' steps, ' + D.config.seeds.length + ' seeds</span>' : "");

    document.getElementById("kpi").innerHTML =
      card(fmtN(r32.recovered) + "/" + fmtN(r32.fitting),
        "tokens that fit the window, recovered <b>exactly</b>", "ok") +
      card(D.margin.toFixed(1) + "x", "decode margin against a signal standard deviation of 1.0. " +
        "The objection needs about 0.03", "ok") +
      card(fmtPct(D.projection[D.projection.length - 1].acc, 1),
        "recovery through <span class='mono'>Linear(8192, 768)</span>, which was predicted to be " +
        "near chance", "ok") +
      (D.heads ? card(fmtN(D.heads.byte_tied.head_params_at_scale),
        "output head parameters for the tied head at the paper's scale, against " +
        fmtN(D.heads.vocab.head_params_at_scale), "ok") : "");

    /* E2 noise */
    lineChart(document.getElementById("noiseChart"), [
      { pts: D.noise.map(function (n) { return { x: n.sigma, y: n.oracle }; }),
        color: "var(--good)" },
      { pts: D.noise.map(function (n) { return { x: n.sigma, y: n.inferred }; }),
        color: "var(--amber)", dash: "5 3" }
    ], { x0: 0, y1: 1.02, xticks: [0, 2, 4, 6, 8, 10, 12], yticks: [0, .25, .5, .75, 1],
         ytick: function (v) { return Math.round(v * 100) + "%"; },
         xlab: "noise sigma, as a multiple of the signal's own standard deviation",
         ylab: "exact token accuracy" });
    document.getElementById("marginNote").innerHTML =
      "<b>The objection, in numbers.</b> The gap between the correct row and the runner up is <b>" +
      D.margin.toFixed(2) + "</b> after z-normalisation, against a signal standard deviation of " +
      "1.0. Predicting <span class='mono'>0.31</span> instead of <span class='mono'>0.30</span> is " +
      "a relative error of " + D.headroom.objection_relative_error.toFixed(4) + ". Exact accuracy " +
      "is still 100% at a noise sigma of <b>" + D.headroom.tolerated_sigma.toFixed(1) + "</b>, and " +
      "sigma is measured in units of the signal's own standard deviation. So the decode tolerates " +
      "<b>" + D.headroom.ratio.toFixed(0) + " times</b> the error the objection describes. The " +
      "point cloud is not needed.";

    /* E3 projection */
    lineChart(document.getElementById("projChart"), [
      { pts: D.projection.map(function (p) { return { x: p.d, y: p.acc }; }), color: "var(--accent)" }
    ], { logx: true, y1: 1.02, xticks: [8, 32, 128, 512, 768], yticks: [0, .25, .5, .75, 1],
         ytick: function (v) { return Math.round(v * 100) + "%"; },
         xlab: "d_model (log scale)", ylab: "minimum norm decode accuracy" });
    document.getElementById("sparseNote").innerHTML =
      "<b>Why the prediction was wrong.</b> A codec vector is not an arbitrary point in " +
      "R<sup>" + fmtN(D.sparsity.D) + "</sup>. It is k-sparse with mean k = <b>" +
      D.sparsity.k.toFixed(2) + "</b> occupied columns, which is " +
      (D.sparsity.density * 100).toFixed(3) + "% dense. Recovering a sparse vector from a random " +
      "linear measurement is the <b>compressed sensing</b> regime, and it succeeds once the " +
      "measurement count comfortably exceeds k. At d_model=768 against k around 8, that condition " +
      "is not marginal, it is met by two orders of magnitude.";

    /* caveat */
    if (D.recheck_minnorm && D.recheck) {
      var rows = ['<tr><th>test</th><th class="num">random init</th><th class="num">after training</th>' +
        '<th>what it measures</th></tr>'];
      Object.keys(D.recheck_minnorm).sort(function (a, b) { return a - b; }).forEach(function (d) {
        var v = D.recheck_minnorm[d];
        rows.push('<tr><td>minimum norm decode, d=' + d + '</td><td class="num">' +
          fmtPct(v.before, 1) + '</td><td class="num">' + fmtPct(v.after, 1) +
          '</td><td>a <b>structure blind</b> decoder</td></tr>');
      });
      rows.push('<tr><td>exact duplicates</td><td class="num">' +
        D.recheck.random_init.duplicates + '</td><td class="num">' +
        D.recheck.after_training.duplicates + '</td><td>is any information destroyed</td></tr>');
      rows.push('<tr><td>learned inverse, held out tokens</td><td class="num">' +
        fmtPct(D.recheck.random_init.learned, 1) + '</td><td class="num">' +
        fmtPct(D.recheck.after_training.learned, 1) +
        '</td><td>a <b>structure aware</b> decoder</td></tr>');
      document.getElementById("caveatTable").innerHTML = rows.join("");
      document.getElementById("caveatNote").innerHTML =
        "<b>Training breaks the decoder, not the encoding.</b> Minimum norm is the correct tool for " +
        "the random projection E3 used and the wrong tool for a trained one, so asking it to invert " +
        "a structured matrix measures the decoder. Distinct tokens stay distinguishable with " +
        "<b>zero duplicates</b>, and a decoder fitted on " + fmtN(D.recheck.meta.fit) +
        " tokens and scored on " + fmtN(D.recheck.meta.probe) + " it never saw loses only <b>" +
        ((D.recheck.random_init.learned - D.recheck.after_training.learned) * 100).toFixed(1) +
        " points</b>. The claim in its final form: recovery is a property of the encoding <b>plus " +
        "an appropriate decoder</b>, never of any decoder.";
    }

    /* E4 heads */
    if (D.heads) {
      var names = { vocab: "vocabulary softmax", byte_untied: "byte head, untied",
                    byte_tied: "byte head, tied to W transposed" };
      var hr = ['<tr><th>head</th><th class="num">loss per token</th><th class="num">exact token</th>' +
        '<th class="num">parameters here</th><th class="num">head parameters at paper scale</th></tr>'];
      ["vocab", "byte_untied", "byte_tied"].forEach(function (h) {
        var v = D.heads[h];
        hr.push("<tr><td>" + names[h] + '</td><td class="num">' + v.loss.toFixed(4) +
          " <span class='muted'>&plusmn;" + v.sd.toFixed(4) + "</span></td><td class='num'>" +
          fmtPct(v.exact, 2) + '</td><td class="num">' + fmtN(v.params) + '</td><td class="num">' +
          fmtN(v.head_params_at_scale) + "</td></tr>");
      });
      document.getElementById("headTable").innerHTML = hr.join("");
      document.getElementById("headNote").innerHTML =
        "<b>The vocabulary head wins at this scale, and that is the measured result.</b> It is " +
        "stated first because the alternative is to hide it. Seed noise floor is " +
        D.noise_floor.toFixed(4) + " nats per token, and both byte deltas exceed it. Two things " +
        "complicate the simple reading, and both are real: per token loss compounds over several " +
        "byte positions while argmax accuracy does not, which is why the untied byte head sits " +
        "within <b>" + ((D.heads.vocab.exact - D.heads.byte_untied.exact) * 100).toFixed(2) +
        " points</b> of the vocabulary head on exact token accuracy while looking far worse " +
        "on loss. And the byte head's argument was never that it wins at " + fmtN(D.vocab) +
        " tokens: it is that its head does not grow with the vocabulary. The last column is " +
        "<b>arithmetic, not measurement</b>, and the two comparisons point in opposite directions.";
    }

    /* E5 objectives */
    if (D.objectives) {
      lineChart(document.getElementById("initChart"), [
        { pts: D.objectives.ce.curve.map(function (h) { return { x: h.step, y: h.acc }; }),
          color: "var(--good)" },
        { pts: D.objectives.mse.curve.map(function (h) { return { x: h.step, y: h.acc }; }),
          color: "var(--amber)", dash: "5 3" }
      ], { x0: 0, y1: 0.6, xticks: [0, 30, 60, 90, 120, 150], yticks: [0, .15, .3, .45, .6],
           ytick: function (v) { return Math.round(v * 100) + "%"; },
           xlab: "training step", ylab: "byte accuracy" });
      document.getElementById("initNote").innerHTML =
        "<b>The prediction was refuted, and the reason is better than the prediction.</b> An " +
        "untrained tied head is already an <b>autoencoder</b>: it reproduces the <b>current</b> " +
        "token's bytes at <b>" + fmtPct(D.init.byte_accuracy_vs_current_token, 2) + "</b> and the " +
        "next token's at " + fmtPct(D.init.byte_accuracy_vs_next_token, 2) + ", against a chance " +
        "rate of " + fmtPct(D.init.chance, 3) + ". That is wiring, not learning: " +
        "<span class='mono'>xf @ W&#7488;</span> reuses the same W that produced the embedding. So " +
        "the objection's premise, that there is nothing to decode at random initialisation, is " +
        "false for <b>both</b> objectives rather than repaired by the choice of loss.";
    }

    /* E6 tying */
    if (D.tying) {
      document.getElementById("tyingKpi").innerHTML =
        card(fmtPct(D.tying.invalid_utf8_rate, 2), "invalid UTF-8. The direct cost of predicting " +
          "positions independently", "hi") +
        card(fmtPct(D.tying.in_vocabulary_rate, 2), "valid and in vocabulary", "ok") +
        card(fmtPct(D.tying.valid_utf8_out_of_vocabulary_rate, 2), "valid but outside the vocabulary") +
        card(fmtPct(D.tying.exact_token_match_rate, 2), "exact match to the target token");
      document.getElementById("tyingNote").innerHTML =
        "<b>The invalid UTF-8 rate is a genuine defect</b>, and it has an obvious remedy this work " +
        "did not implement: constrain the decode to emit only valid UTF-8, which would drive it to " +
        "zero without retraining anything. The out of vocabulary strings are " +
        D.tying.out_of_vocabulary_examples.slice(0, 5).map(function (w) {
          return "<span class='mono script'>" + w + "</span>";
        }).join(", ") + " and similar. If some of those render as empty boxes, that is the "  +
        "finding rather than a missing font: the head emits codepoints Unicode has not assigned, " +
        "so no font anywhere has a glyph for them. They are degenerate repeats and near misses, " +
        "<b>not plausible words</b>.";
    }

    /* E7 bands */
    if (D.openvocab && D.openvocab.web) {
      var lbl = { in_head: "in vocabulary, most frequent quarter", in_mid: "in vocabulary, middle",
                  in_tail: "in vocabulary but RARE (the control)",
                  outside_near: "outside vocabulary, nearer", outside_far: "outside vocabulary, far" };
      var order = ["in_head", "in_mid", "in_tail", "outside_near", "outside_far"];
      var b = D.openvocab.web.bands;
      barChart(document.getElementById("bandChart"), order.map(function (k) {
        return { label: lbl[k], values: [b[k]], notes: [fmtPct(b[k], 2)] };
      }), { colors: ["var(--accent)"], padL: 230, max: Math.max(b.in_head, 0.01),
            tick: function (v) { return (v * 100).toFixed(0) + "%"; } });
      document.getElementById("bandNote").innerHTML =
        "<b>A zero here proves nothing, and the control is what shows that.</b> Words outside the " +
        "vocabulary are by construction also the <b>rarer</b> words, so a zero could mean \"cannot " +
        "emit an unknown word\" or merely \"cannot predict rare words at all\". The " +
        "<b>in_tail</b> band is rarity matched: the least frequent words that <b>are</b> in the " +
        "vocabulary. It scores at the floor too. So vocabulary membership explains nothing, it is " +
        "a rarity cliff, and this experiment cannot separate the two at this scale. " +
        "<b>The verdict splits:</b> the <b>capability</b> is architecturally true and needs no " +
        "measurement, since a vocabulary softmax has no output row for an unknown word and scores " +
        "exactly zero at any amount of training. The <b>competence</b> is not demonstrated, and " +
        "this page does not claim the payoff is false, only that this scale cannot decide.";
    }
    /* answers summary */
    var ar = [
      ["\u201cHow do I make a reverse of this?\u201d",
       "<b>It already reverses</b>, in three independent senses: exact inversion of every token " +
       "that fits the window, tolerance of <b>" + D.headroom.ratio.toFixed(0) + "x</b> more error " +
       "than the objection implies, and survival of the projection because a code is only " +
       D.sparsity.k.toFixed(2) + "-sparse.", "E1, E2, E3"],
      ["\u201cWe can get rid of the final head!\u201d",
       D.heads ? "<b>Yes, at zero new parameters</b>, " + fmtN(D.heads.byte_tied.head_params_at_scale) +
         " against " + fmtN(D.heads.vocab.head_params_at_scale) + " at the paper\u2019s " +
         "dimensions. But it costs accuracy: at this scale the <b>vocabulary head wins</b> on " +
         "loss, " + D.heads.vocab.loss.toFixed(4) + " against " + D.heads.byte_tied.loss.toFixed(4) +
         " nats per token." : "Run the training experiments to populate this row.", "E4"],
      ["\u201cA vocab of 1M without any issues!\u201d",
       "<b>Split verdict.</b> The <b>capability</b> is architecturally true and needs no " +
       "experiment: a vocabulary softmax has no output row for an unknown word and scores exactly " +
       "zero at any amount of training. The <b>competence</b> is not demonstrated, and is not " +
       "testable at this scale.", "E6, E7"]
    ];
    document.getElementById("answersTable").innerHTML =
      "<tr><th>what was promised</th><th>the answer</th><th>where</th></tr>" +
      ar.map(function (r) {
        return "<tr><td><b>" + r[0] + "</b></td><td>" + r[1] + "</td><td>" + r[2] + "</td></tr>";
      }).join("");
  }).catch(function (err) {
    document.querySelector(".wrap").insertAdjacentHTML("afterbegin",
      '<div class="callout">Could not load <span class="mono">data/dashboard.json</span>. ' +
      'Run <span class="mono">python src/build_dashboard.py</span> first. (' + err + ')</div>');
  });
})();
