/* V5 dashboard - compare candidate design SETS, then expand the chosen one.
   Loads site/data/dashboard.json. Hand-built SVG, validated palette, theme-aware, no deps. */
const SVGNS = "http://www.w3.org/2000/svg";
const $ = (s, r = document) => r.querySelector(s);
const cvar = n => getComputedStyle(document.body).getPropertyValue(n).trim();
const fmt = n => n == null ? "-" : n.toLocaleString("en-US");
function S(t, a = {}) { const e = document.createElementNS(SVGNS, t); for (const k in a) e.setAttribute(k, a[k]); return e; }
function txt(x, y, s, a = {}) { const e = S("text", { x, y, ...a }); e.textContent = s; return e; }
function mkSvg(h, w, ht) { const s = S("svg", { viewBox: `0 0 ${w} ${ht}`, width: "100%", role: "img" }); h.innerHTML = ""; h.appendChild(s); return s; }
const tip = $("#tooltip");
function hover(el, label) { el.addEventListener("mousemove", e => { tip.textContent = label; tip.style.opacity = 1; tip.style.left = (e.clientX + 12) + "px"; tip.style.top = (e.clientY + 12) + "px"; }); el.addEventListener("mouseleave", () => tip.style.opacity = 0); }

let D = {};
// proxy lanes (5) - colour by lane for recipe bars
const PLANE = { web: "Web", code: "Code", math: "Math", reasoning: "Reasoning", indic: "Indic" };
const PLANE_COL = { web: "--s1", code: "--s2", math: "--s3", reasoning: "--s4", indic: "--s7" };
// sets - colour per set (extensible)
const SET_SLOTS = ["--s5", "--s1", "--s2", "--s4", "--s6", "--s8", "--s3", "--s7"];
function setCol(i) { return SET_SLOTS[i % SET_SLOTS.length]; }
// full-plan 7 lanes
const LANE7 = ["web", "code", "math_stem", "reasoning", "agentic", "long_ctx", "indic"];
const LANE7_COL = { web: "--s1", code: "--s2", math_stem: "--s3", reasoning: "--s4", agentic: "--s5", long_ctx: "--s6", indic: "--s7" };
const LANE7_NAME = { web: "General web", code: "Code", math_stem: "Math+STEM", reasoning: "Reasoning", agentic: "Agentic", long_ctx: "Long-ctx", indic: "Indic" };

const setC = i => D.sets[i]; // convenience
function bestIdx(vals) { let bi = 0; vals.forEach((v, i) => { if (v != null && (vals[bi] == null || v < vals[bi])) bi = i; }); return bi; }
function laneWins(setkey) { // which proxy lanes this set has the lowest loss on
    return D.proxy_lanes.filter(l => {
        const vals = D.sets.map(s => s.outcome ? s.outcome.final[l] : null);
        return D.sets[bestIdx(vals)].key === setkey;
    });
}

/* ---------------------------------------------------------------- KPI */
function renderKpi() {
    const chosen = D.sets.find(s => s.chosen) || D.sets[0];
    const conf = D.verdicts.filter(v => v.ok).length, tot = D.verdicts.length;
    const gen = D.full_plan.supply_totals;
    const tiles = [
        ["Candidate sets", String(D.sets.length), "compared on the proxy"],
        ["Chosen set", chosen.name.split(" ")[0], `avg loss ${chosen.outcome ? chosen.outcome.avg.toFixed(2) : "-"} (best)`],
        ["Model / budget", (D.model.match(/~?\d+B/) || ["~40B"])[0], `${(D.budget_B / 1000).toFixed(0)}T tokens`],
        ["Predictions confirmed", `${conf}/${tot}`, "mixture behaves as a hypothesis"],
        ["Must be generated", `${gen.generated_pct}<span class='u'>%</span>`, `~${gen.generated_B}B (agentic+reasoning)`],
        ["Proxy scale", "~5M", "params · CPU · directional"],
    ];
    $("#kpi").innerHTML = tiles.map(([l, v, s]) => `<div class="tile"><div class="label">${l}</div><div class="value">${v}</div><div class="sub">${s}</div></div>`).join("");
}

/* ---------------------------------------------------------------- set cards */
function miniRecipe(recipe) { // 100% stacked horizontal bar of the 5 proxy lanes
    const W = 240, H = 16; const svg = S("svg", { viewBox: `0 0 ${W} ${H}`, width: "100%", height: H });
    let acc = 0, tot = D.proxy_lanes.reduce((a, l) => a + (recipe[l] || 0), 0);
    D.proxy_lanes.forEach(l => { const v = recipe[l] || 0; if (!v) return; const w = W * v / tot; svg.appendChild(S("rect", { x: W * acc / tot, y: 0, width: Math.max(w - 1, .5), height: H, fill: cvar(PLANE_COL[l]), rx: 2 })); acc += v; });
    return svg.outerHTML;
}
function renderSetCards() {
    $("#setCards").innerHTML = D.sets.map((s, i) => {
        const wins = s.outcome ? laneWins(s.key).map(l => PLANE[l]).join(", ") || "none" : "-";
        const accent = cvar(setCol(i));
        return `<div class="card" style="border-top:3px solid ${accent}">
      <div class="k" style="color:${accent}">${s.chosen ? "★ chosen · " : ""}set ${i + 1}</div>
      <h3>${s.name}</h3>
      <p style="margin-bottom:.5rem">${s.thesis}</p>
      ${miniRecipe(s.recipe)}
      <div style="display:flex;justify-content:space-between;margin-top:.55rem;font-size:.82rem">
        <span>avg loss <b style="font-size:1.05rem">${s.outcome ? s.outcome.avg.toFixed(3) : "-"}</b></span>
        <span style="color:var(--text-2)">wins: <b>${wins}</b></span></div></div>`;
    }).join("");
}

/* ---------------------------------------------------------------- comparison matrix */
function renderCompareTable() {
    const sets = D.sets, lanes = D.proxy_lanes;
    let h = `<thead><tr><th>Metric</th>${sets.map((s, i) => `<th class="num" style="color:${cvar(setCol(i))}">${s.name.split(" - ")[0].split(" (")[0]}${s.chosen ? " ★" : ""}</th>`).join("")}</tr></thead><tbody>`;
    h += `<tr><td colspan="${sets.length + 1}" style="background:var(--surface-2);font-weight:700;font-size:.74rem;text-transform:uppercase;letter-spacing:.05em;color:var(--muted)">Design · recipe (% of budget)</td></tr>`;
    lanes.forEach(l => { h += `<tr><td>${PLANE[l]}</td>${sets.map(s => `<td class="num">${s.recipe[l] ?? "-"}%</td>`).join("")}</tr>`; });
    h += `<tr><td colspan="${sets.length + 1}" style="background:var(--surface-2);font-weight:700;font-size:.74rem;text-transform:uppercase;letter-spacing:.05em;color:var(--muted)">Outcome · held-out loss (lower is better)</td></tr>`;
    [...lanes, "_avg"].forEach(l => {
        const vals = sets.map(s => s.outcome ? (l === "_avg" ? s.outcome.avg : s.outcome.final[l]) : null);
        const bi = bestIdx(vals);
        const label = l === "_avg" ? "<b>Average</b>" : PLANE[l] + " loss";
        h += `<tr><td>${label}</td>${vals.map((v, i) => `<td class="num" style="${i === bi ? "color:var(--good-ink);font-weight:700" : ""}">${v != null ? v.toFixed(3) : "-"}${i === bi ? " ●" : ""}</td>`).join("")}</tr>`;
    });
    $("#compareTable").innerHTML = h + "</tbody>";
}

/* ---------------------------------------------------------------- recipe stacked bars (per set) */
function renderRecipeBars() {
    const sets = D.sets, W = 900, rowH = 62, barH = 34, padL = 132, padR = 96, padT = 10;
    const H = padT * 2 + sets.length * rowH, svg = mkSvg($("#figRecipes"), W, H);
    sets.forEach((s, i) => {
        const y = padT + i * rowH + (rowH - barH) / 2; let acc = 0;
        const tot = D.proxy_lanes.reduce((a, l) => a + (s.recipe[l] || 0), 0);
        const nm = s.name.split(" - ")[0].split(" (")[0];
        svg.appendChild(txt(padL - 10, y + barH / 2 - 2, nm, { "text-anchor": "end", fill: cvar("--text"), "font-size": 13, "font-weight": 600 }));
        if (s.chosen) svg.appendChild(txt(padL - 10, y + barH / 2 + 13, "★ chosen", { "text-anchor": "end", fill: cvar("--s1"), "font-size": 10 }));
        D.proxy_lanes.forEach(l => {
            const v = s.recipe[l] || 0; if (!v) return; const w = (W - padL - padR) * v / tot, x = padL + (W - padL - padR) * acc / tot;
            const seg = S("rect", { x, y, width: Math.max(w - 2, .5), height: barH, fill: cvar(PLANE_COL[l]), rx: 3 });
            hover(seg, `${nm} · ${PLANE[l]}: ${v}%`); svg.appendChild(seg);
            if (v >= 6) svg.appendChild(txt(x + w / 2, y + barH / 2 + 4, v, { "text-anchor": "middle", fill: "#fff", "font-size": 12, "font-weight": 700 }));
            acc += v;
        });
        if (s.outcome) svg.appendChild(txt(W - padR + 8, y + barH / 2 + 4, `avg ${s.outcome.avg.toFixed(2)}`, { fill: cvar("--text-2"), "font-size": 11.5, "font-variant-numeric": "tabular-nums" }));
    });
    $("#figRecipes-legend").innerHTML = D.proxy_lanes.map(l => `<span class="it"><span class="sw" style="background:${cvar(PLANE_COL[l])}"></span>${PLANE[l]}</span>`).join("");
}

/* ---------------------------------------------------------------- loss dot plot (per set) */
function renderLossDots() {
    const sets = D.sets, lanes = [...D.proxy_lanes, "_avg"];
    const rows = lanes.map(l => ({ lane: l, vals: sets.map((s, i) => ({ i, key: s.key, v: s.outcome ? (l === "_avg" ? s.outcome.avg : s.outcome.final[l]) : null })) }));
    const all = rows.flatMap(r => r.vals.map(x => x.v)).filter(v => v != null);
    const lo = Math.min(...all) - 0.15, hi = Math.max(...all) + 0.15;
    const W = 560, rowH = 34, padL = 84, padR = 30, padT = 10, padB = 26, H = padT + padB + rows.length * rowH;
    const svg = mkSvg($("#figLoss"), W, H), x = v => padL + (W - padL - padR) * (v - lo) / (hi - lo);
    for (let t = Math.ceil(lo * 2) / 2; t <= hi; t += 0.5) { svg.appendChild(S("line", { x1: x(t), y1: padT, x2: x(t), y2: H - padB, stroke: cvar("--grid"), "stroke-width": 1 })); svg.appendChild(txt(x(t), H - padB + 14, t.toFixed(1), { "text-anchor": "middle", fill: cvar("--muted"), "font-size": 10 })); }
    rows.forEach((r, i) => {
        const y = padT + i * rowH + rowH / 2, isAvg = r.lane === "_avg";
        svg.appendChild(txt(padL - 8, y + 4, isAvg ? "AVERAGE" : PLANE[r.lane], { "text-anchor": "end", fill: isAvg ? cvar("--text") : cvar("--text-2"), "font-size": 11.5, "font-weight": isAvg ? 700 : 400 }));
        const xs = r.vals.filter(v => v.v != null).map(v => x(v.v));
        if (xs.length) svg.appendChild(S("line", { x1: Math.min(...xs), y1: y, x2: Math.max(...xs), y2: y, stroke: cvar("--axis"), "stroke-width": 2, opacity: .45 }));
        r.vals.forEach(v => { if (v.v == null) return; const c = S("circle", { cx: x(v.v), cy: y, r: sets[v.i].chosen ? 6 : 5, fill: cvar(setCol(v.i)), stroke: cvar("--surface-1"), "stroke-width": 1.5 }); hover(c, `${sets[v.i].name.split(" - ")[0]} · ${isAvg ? "avg" : PLANE[r.lane]}: ${v.v.toFixed(3)}`); svg.appendChild(c); });
    });
    svg.appendChild(txt(padL, H - 2, "held-out loss (lower is better) →", { fill: cvar("--muted"), "font-size": 10 }));
    $("#figLoss-legend").innerHTML = sets.map((s, i) => `<span class="it"><span class="sw" style="background:${cvar(setCol(i))}"></span>${s.name.split(" - ")[0].split(" (")[0]}</span>`).join("");
}

function renderVerdicts() {
    $("#verdicts").innerHTML = D.verdicts.map(v => `<div class="card verdict ${v.ok ? "" : "bad"}"><div class="k">${v.ok ? "✓ confirmed" : "✗ refuted"}</div><h3>${v.claim}</h3><p>${v.detail}</p></div>`).join("");
}
function renderAddSet() {
    $("#addSet").innerHTML = `<b>Add a tuning set:</b> add an entry to <span class="mono">sets</span> in
    <span class="mono">plan.json</span> (name + recipe), add the same recipe to <span class="mono">MIXES</span> in
    <span class="mono">proxy/train.py</span>, run <span class="mono">bash proxy/run_all.sh</span> then
    <span class="mono">python3 build_dashboard.py</span> - a new column appears here and the winner may change.`;
}

/* ---------------------------------------------------------------- CHOSEN SET EXPANDED (full 7-lane) */
function renderExpandedLead() {
    const c = D.sets.find(s => s.chosen) || D.sets[0];
    $("#expandedLead").innerHTML = `The <b>${c.name}</b> set, expanded from 5 tested lanes to the shipped 7 by adding
    <b>Agentic (8%)</b> and <b>Long-context (6%)</b> - lanes the proxy can't measure but the benchmarks demand.
    Every share is sized against real supply below.`;
}
function renderShares() {
    const lanes = D.full_plan.lanes, W = 720, rowH = 30, padL = 108, padR = 150, padT = 6;
    const H = padT * 2 + lanes.length * rowH, svg = mkSvg($("#figShares"), W, H), max = Math.max(...lanes.map(l => l.share));
    lanes.forEach((l, i) => {
        const y = padT + i * rowH + 4, bw = (W - padL - padR) * l.share / max;
        svg.appendChild(txt(padL - 8, y + 13, l.name, { "text-anchor": "end", fill: cvar("--text-2"), "font-size": 11.5 }));
        const bar = S("rect", { x: padL, y, width: Math.max(bw, 2), height: rowH - 12, rx: 4, fill: cvar(LANE7_COL[l.key]) });
        hover(bar, `${l.name}: ${l.share}% · ${l.benchmark}`); svg.appendChild(bar);
        svg.appendChild(txt(padL + bw + 6, y + 13, `${l.share}%`, { fill: cvar("--text"), "font-size": 11, "font-weight": 600 }));
        const flag = l.generated_B > 0 ? `${l.generated_B}B generated` : (l.epochs > 1.5 ? `${l.epochs}× repeat` : "organic");
        svg.appendChild(txt(padL + bw + 40, y + 13, flag, { fill: l.generated_B > 0 ? cvar("--critical") : cvar("--muted"), "font-size": 10 }));
    });
}
function renderSupplyTable() {
    const lanes = D.full_plan.lanes;
    $("#supplyTable").innerHTML = `<thead><tr><th>Lane</th><th>Share</th><th>Benchmark</th><th class="num">Real supply (B)</th><th class="num">Epochs</th><th>How met</th></tr></thead><tbody>` +
        lanes.map(l => { const met = l.generated_B > 0 ? `<span class="pill gen">${l.generated_B}B generated</span>` : (l.epochs > 1.5 ? `<span class="pill">${l.epochs}× repeat</span>` : `<span class="pill ok">organic</span>`); const tr = l.trainable_frac < 1 ? ` · <span class="pill">${Math.round(l.trainable_frac * 100)}% trainable</span>` : ""; return `<tr><td><b>${l.name}</b></td><td>${l.share}%</td><td>${l.benchmark}</td><td class="num">${fmt(l.supply_B)}</td><td class="num">${l.epochs}</td><td>${met}${tr}</td></tr>`; }).join("") + `</tbody>`;
    const t = D.full_plan.supply_totals;
    $("#supplyNote").innerHTML = `<b>No wishful accounting:</b> ~<b>${t.generated_B}B (${t.generated_pct}%)</b> must be <b>generated</b> (agentic + reasoning have almost no real supply); organic Indic is only ~${t.organic_indic_B}B, so verified Indic is just <b>${t.verified_pct_of_budget}%</b> of the whole budget.`;
}
function renderStages() {
    const stages = D.full_plan.stages, W = 760, padL = 40, padR = 12, padT = 12, padB = 40, plot = 300, H = padT + padB + plot;
    const svg = mkSvg($("#figStages"), W, H), n = stages.length, gap = 26, bw = (W - padL - padR - gap * (n - 1)) / n;
    for (let t = 0; t <= 100; t += 25) { const y = padT + plot * (1 - t / 100); svg.appendChild(S("line", { x1: padL, y1: y, x2: W - padR, y2: y, stroke: cvar("--grid"), "stroke-width": 1 })); svg.appendChild(txt(padL - 6, y + 3, t + "%", { "text-anchor": "end", fill: cvar("--muted"), "font-size": 10 })); }
    stages.forEach((st, i) => {
        const x = padL + i * (bw + gap); let acc = 0;
        LANE7.forEach(lk => { const v = st.shares[lk] || 0; if (!v) return; const hh = plot * v / 100, y = padT + plot * (1 - (acc + v) / 100); const seg = S("rect", { x, y, width: bw, height: Math.max(hh - 1.5, .5), fill: cvar(LANE7_COL[lk]), rx: 2 }); hover(seg, `${st.name} · ${LANE7_NAME[lk]}: ${v}%`); svg.appendChild(seg); if (v >= 12) svg.appendChild(txt(x + bw / 2, y + hh / 2 + 3, v, { "text-anchor": "middle", fill: "#fff", "font-size": 9.5, "font-weight": 600 })); acc += v; });
        svg.appendChild(txt(x + bw / 2, padT + plot + 15, st.name, { "text-anchor": "middle", fill: cvar("--text"), "font-size": 11, "font-weight": 700 }));
        svg.appendChild(txt(x + bw / 2, padT + plot + 30, `${st.weight}% · ${st.seq}`, { "text-anchor": "middle", fill: cvar("--muted"), "font-size": 9.5 }));
    });
    $("#figStages-legend").innerHTML = LANE7.map(lk => `<span class="it"><span class="sw" style="background:${cvar(LANE7_COL[lk])}"></span>${LANE7_NAME[lk]}</span>`).join("");
    $("#stageCards").innerHTML = stages.map(st => `<div class="stage"><div class="sn">${st.weight}% of budget</div><div class="st">${st.name}</div><div class="meta">band ${st.band}<br>seq ${st.seq}</div><div class="meta" style="margin-top:.3rem">${st.note}</div></div>`).join("");
    $("#stageNote").innerHTML = `<b>Indic never below ${Math.min(...stages.map(s => s.shares.indic))}%</b> (protected floor); <b>web fades ${stages[0].shares.web}→${stages[n - 1].shares.web}%</b>; agentic + long-context climb; reasoning peaks in the anneal. Warm-up seams (~0.5-1% 60/40 blends) sit between stages to keep gradients smooth.`;
}
function renderTiers() {
    const tiers = D.full_plan.indic_tiers, W = 500, H = 66, padL = 4, padR = 4, y = 10, bh = 30, svg = mkSvg($("#figTiers"), W, H);
    const total = tiers.reduce((a, t) => a + t.share_of_indic, 0), cols = ["--s6", "--s4", "--s1", "--s5"]; let acc = 0;
    tiers.forEach((t, i) => { const w = (W - padL - padR) * t.share_of_indic / total, x = padL + (W - padL - padR) * acc / total; const seg = S("rect", { x, y, width: Math.max(w - 2, 1), height: bh, fill: cvar(cols[i]), rx: 3 }); hover(seg, `${t.name}: ${t.share_of_indic}% · ${t.demand_B}B demand vs ${t.supply_B}B supply (${t.epochs}×)`); svg.appendChild(seg); svg.appendChild(txt(x + w / 2, y + bh / 2 + 4, `${t.share_of_indic}%`, { "text-anchor": "middle", fill: "#fff", "font-size": 11, "font-weight": 700 })); acc += t.share_of_indic; });
    $("#figTiers-legend").innerHTML = tiers.map((t, i) => `<span class="it"><span class="sw" style="background:${cvar(cols[i])}"></span>${t.name} (${t.epochs}×)</span>`).join("");
    $("#tiersNote").innerHTML = `Verified is the scarce, best tier (1.5× of 86B), <b>reserved for the anneal</b>; ~68% of the lane is translated+synthetic. Scaling the budget does not scale the verified tier - it is supply-bound.`;
}
function renderDemand() {
    const lanes = D.full_plan.lanes, W = 560, rowH = 30, padL = 108, padR = 60, padT = 6, padB = 20, H = padT + padB + lanes.length * rowH;
    const svg = mkSvg($("#figDemand"), W, H), demand = lanes.map(l => D.budget_B * l.share / 100);
    const maxlog = Math.log10(Math.max(...lanes.map(l => l.supply_B), ...demand)), minlog = 1, x = v => padL + (W - padL - padR) * (Math.log10(Math.max(v, 10)) - minlog) / (maxlog - minlog);
    lanes.forEach((l, i) => {
        const y = padT + i * rowH + 4, dem = D.budget_B * l.share / 100;
        svg.appendChild(txt(padL - 8, y + 13, l.name, { "text-anchor": "end", fill: cvar("--text-2"), "font-size": 11 }));
        const sBar = S("rect", { x: padL, y: y + 1, width: Math.max(x(l.supply_B) - padL, 2), height: 8, rx: 2, fill: cvar("--s3") }); hover(sBar, `${l.name} supply: ${fmt(l.supply_B)}B`); svg.appendChild(sBar);
        const dBar = S("rect", { x: padL, y: y + 10, width: Math.max(x(dem) - padL, 2), height: 8, rx: 2, fill: cvar("--s1") }); hover(dBar, `${l.name} demand: ${fmt(Math.round(dem))}B`); svg.appendChild(dBar);
        if (l.generated_B > 0) svg.appendChild(txt(x(dem) + 5, y + 17, `+${l.generated_B}B gen`, { fill: cvar("--critical"), "font-size": 9.5 }));
    });
    $("#figDemand-legend").innerHTML = `<span class="it"><span class="sw" style="background:${cvar("--s1")}"></span>demand</span><span class="it"><span class="sw" style="background:${cvar("--s3")}"></span>real supply</span>`;
}
function renderAnneal() {
    const a = D.full_plan.anneal_reserve;
    $("#anneal").innerHTML = `<b>Anneal reserve (${a.pct}% ≈ ${a.total_B}B), decided now, spent last:</b> ` + a.items.map(x => `${x.name} ${x.B}B`).join(" · ") + `. Held back at composition time for the low-LR cooldown - a small reserve, a large benchmark lift.`;
}


/* ---------------------------------------------------------------- 05 what the experiments settled */
function renderPinned() {
    const F = D.findings; if (!F) return;
    $("#pinned").innerHTML = `<b>The arithmetic that decided it, before any experiment:</b> organic Indic is
      ~110B and the repetition ceiling is 4 epochs, so 440B is reachable - <b>exactly ${F.native_pinned_pct.toFixed(1)}%
      of a 4T budget, at every possible Indic share</b>. Raising the Indic lane from 18% to 32% cannot buy a single
      extra <i>native</i> Indic token; it buys 14 more points of translated and synthetic text. The lever that
      raises Indic capability is <b>acquiring more verified data</b>, not a bigger share.`;
}
function renderFloor() {
    const F = D.findings; if (!F) return;
    const host = $("#figFloor"); host.innerHTML = "";
    const rows = Object.entries(F.floor).filter(([k]) => k !== "avg");
    rows.push(["avg", F.floor.avg]);
    const W = 640, rowH = 30, padL = 92, H = rows.length * rowH + 34;
    const svg = S("svg", { viewBox: `0 0 ${W} ${H}`, width: "100%", height: H });
    const max = Math.max(...rows.map(r => r[1] || 0)) * 1.15 || 1;
    const x = v => padL + (v / max) * (W - padL - 70);
    rows.forEach(([lane, v], i) => {
        const y = 14 + i * rowH;
        svg.appendChild(txt(padL - 10, y + 13, lane, { "text-anchor": "end", fill: cvar("--text-2"), "font-size": 11.5 }));
        const isAvg = lane === "avg";
        const bar = S("rect", { x: padL, y: y + 2, width: Math.max(x(v) - padL, 2), height: 15, rx: 3,
                                fill: cvar(isAvg ? "--s3" : "--critical"), opacity: isAvg ? 1 : .85 });
        hover(bar, `${lane}: re-running the same mixture moves loss by ${v.toFixed(3)}`);
        svg.appendChild(bar);
        svg.appendChild(txt(x(v) + 7, y + 14, v.toFixed(3) + (isAvg ? "  (the stable metric)" : ""),
                            { fill: cvar(isAvg ? "--s3" : "--text-2"), "font-size": 11, "font-weight": isAvg ? 700 : 400 }));
    });
    host.appendChild(svg);
}
function renderTier() {
    const F = D.findings; if (!F || !F.tier || !F.tier.indic_hi) return;
    const t = F.tier, r = (o, label, note) => `<tr><td><b>${label}</b><div class="sub">${note}</div></td>
        <td class="num">${o.t18.toFixed(3)}</td><td class="num">${o.t30.toFixed(3)}</td>
        <td class="num">${o.delta >= 0 ? "+" : ""}${o.delta.toFixed(3)}</td><td class="num">${o.floor.toFixed(3)}</td>
        <td>${o.readable ? "<b style='color:var(--s3)'>yes</b>" : "<b style='color:var(--critical)'>no</b>"}</td></tr>`;
    $("#tierTable").innerHTML =
        `<thead><tr><th>Scored on</th><th class="num">Indic 18%<br><span class="sub">11 native + 7 synth</span></th>
         <th class="num">Indic 30%<br><span class="sub">11 native + 19 synth</span></th>
         <th class="num">&Delta;</th><th class="num">floor</th><th>readable?</th></tr></thead><tbody>`
        + r(t.indic_hi, "NATIVE Indic", "what MILU and IndicGenBench actually measure")
        + r(t.indic_lo, "translated / synthetic Indic", "not the promised capability")
        + `</tbody>`;
    $("#tierNote").innerHTML = `Tripling the synthetic Indic mass buys <b>fluency in machine-translated text</b>
      and <b>nothing measurable on native Indic</b>. The single Indic bin used in the earlier rounds was
      <b>98.5% translated/synthetic</b>, so every earlier "Indic gain" was a gain on the wrong distribution.
      A control arm at 30% <i>native</i> Indic - impossible to supply - scores ${t.ideal_hi.toFixed(3)} vs
      ${t.indic_hi.t18.toFixed(3)}, still inside the floor: native Indic capability <b>saturates</b> at the
      ~11% the supply allows. <b>Indic finalises at 18%.</b>`;
}
function renderLeak() {
    const F = D.findings; if (!F) return;
    $("#leakTable").innerHTML = `<thead><tr><th>Lane</th><th class="num">leakage</th><th>verdict</th></tr></thead><tbody>`
        + Object.entries(F.leakage).map(([l, p]) => `<tr><td>${l}</td><td class="num">${p}%</td>
            <td>${p > 50 ? "<b style='color:var(--critical)'>CONTAMINATED</b>" : p < 10 ? "clean" : "minor"}</td></tr>`).join("")
        + `</tbody>`;
}
function renderWithdrawn() {
    const F = D.findings; if (!F) return;
    $("#withdrawnTable").innerHTML = `<thead><tr><th>Claim</th><th>Why it fell</th></tr></thead><tbody>`
        + F.withdrawn.map(([c, w]) => `<tr><td><b>${c}</b></td><td class="sub">${w}</td></tr>`).join("") + `</tbody>`;
    $("#finalVerdict").innerHTML = `<b>Final verdict.</b> ${F.verdict} Had we adopted the reasoning increase we
      would have committed <b>+120B of generated tokens</b> on the strength of a memorisation artefact - the
      concrete cost of not auditing your own metric.`;
}

/* ---------------------------------------------------------------- theme + boot */
function initTheme() {
    const root = document.documentElement, btn = $("#themeToggle"), saved = localStorage.getItem("a5theme");
    if (saved) root.setAttribute("data-theme", saved);
    const sync = () => { const dark = root.getAttribute("data-theme") === "dark" || (!root.getAttribute("data-theme") && matchMedia("(prefers-color-scheme: dark)").matches); $("#themeIcon").innerHTML = dark ? "&#9790;" : "&#9788;"; $("#themeLabel").textContent = dark ? "Dark" : "Light"; };
    btn.addEventListener("click", () => { const c = root.getAttribute("data-theme") === "dark" ? "light" : "dark"; root.setAttribute("data-theme", c); localStorage.setItem("a5theme", c); sync(); renderAll(); });
    sync();
}
function renderAll() {
    renderKpi(); renderSetCards(); renderCompareTable(); renderRecipeBars(); renderLossDots(); renderVerdicts(); renderAddSet();
    renderExpandedLead(); renderShares(); renderSupplyTable(); renderStages(); renderTiers(); renderDemand(); renderAnneal();
    renderPinned(); renderFloor(); renderTier(); renderLeak(); renderWithdrawn();
}
async function boot() {
    try { D = await fetch("data/dashboard.json").then(r => r.json()); }
    catch (e) { $("#kpi").innerHTML = "<div class='tile'><div class='label'>Data not loaded</div><div class='sub'>run build_dashboard.py</div></div>"; return; }
    renderAll(); initTheme();
}
boot();
