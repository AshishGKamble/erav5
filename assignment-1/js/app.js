// Wires the four demos into the page. Each demo starts training the first time
// its section scrolls into view, so the page stays light until you reach it.

import { initRings } from "./demo-rings.js";
import { initDepth } from "./demo-depth.js";
import { initEmbed } from "./demo-embed.js";
import { initGeneral } from "./demo-general.js";

const REGISTRY = [
  { id: "s1-1", init: initRings },
  { id: "s1-2", init: initDepth },
  { id: "s1-3", init: initEmbed },
  { id: "s1-4", init: initGeneral },
];

const started = new Set();

function boot() {
  const controllers = {};
  for (const entry of REGISTRY) {
    const root = document.getElementById(entry.id);
    if (!root) continue;
    controllers[entry.id] = entry.init(root);
  }

  const observer = new IntersectionObserver(
    (entries) => {
      for (const e of entries) {
        if (e.isIntersecting && !started.has(e.target.id)) {
          started.add(e.target.id);
          const ctrl = controllers[e.target.id];
          if (ctrl) ctrl.start();
        }
      }
    },
    { threshold: 0.25 }
  );

  for (const entry of REGISTRY) {
    const root = document.getElementById(entry.id);
    if (root) observer.observe(root);
  }

  // Progress rail: highlight the active section in the side nav.
  const navLinks = Array.from(document.querySelectorAll(".rail a"));
  const navObserver = new IntersectionObserver(
    (entries) => {
      for (const e of entries) {
        if (e.isIntersecting) {
          navLinks.forEach((a) =>
            a.classList.toggle("active", a.getAttribute("href") === "#" + e.target.id)
          );
        }
      }
    },
    { threshold: 0.5 }
  );
  document.querySelectorAll("section.claim").forEach((s) => navObserver.observe(s));
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", boot);
} else {
  boot();
}
