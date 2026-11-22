#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "matplotlib"]
# ///
"""Paper figures from the batch logs: change by 2100 under three comparisons, and fitted decay scales by class.

    decay/fig_paper.py decay/out
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

out = Path(sys.argv[1])
load = lambda name: {r["slug"]: r for r in map(json.loads, (out / name).read_text().splitlines()) if r["ok"]}
clip, raw, cal = load("batch.jsonl"), load("raw_ratio.jsonl"), load("calibration_all.jsonl")
pooled = json.loads((out / "decay_pooled.json").read_text())["species"]
slugs = sorted(set(clip) & set(raw) & set(pooled))
mismatch = np.array([raw[s]["raw_future_ratio"] for s in slugs])
neither = np.array([raw[s]["raw_future_ratio"] / raw[s]["raw_now_ratio"] for s in slugs])
same = np.array([pooled[s]["change"] for s in slugs])

fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
bins = np.linspace(-3, 3, 61)
for v, lab, c in ((mismatch, "unclipped 2100 / clipped now", "#c0504d"), (neither, "no clip in either period", "#999999"),
                  (same, "same calibrated clip in both", "#1f77b4")):
    ax[0].hist(np.log2(v), bins, histtype="step", lw=1.8, color=c, label=f"{lab}: {np.mean(v > 1):.0%} gain")
ax[0].axvline(0, color="k", lw=0.8)
ax[0].set(xlabel="log2 change in total suitability, 2071-2100 vs 1980-2010", ylabel="species")
ax[0].legend(frameon=False, fontsize=8, loc="upper left")
ax[0].set_title("a", loc="left", fontweight="bold")

classes = ["Amphibians", "Reptiles", "Mammals", "Birds"]
own = {c: [] for c in classes}
for r in cal.values():
    if "winner" in r and r["class"] in own:
        own[r["class"]].append(r["halfnormal"]["fit"])
ax[1].boxplot([np.log10(own[c]) for c in classes], showfliers=False)
for i, c in enumerate(classes, 1):
    ax[1].scatter(i + np.random.default_rng(i).uniform(-0.25, 0.25, len(own[c])), np.log10(own[c]), s=3, alpha=0.3, color="#1f77b4")
ax[1].set_xticks(range(1, 5), [f"{c}\n(n={len(own[c])})" for c in classes])
ax[1].set_yticks(range(4), ["1", "10", "100", "1,000"])
ax[1].set(ylabel="fitted half-normal scale sigma (km)")
ax[1].set_title("b", loc="left", fontweight="bold")
fig.tight_layout()
fig.savefig(out / "fig" / "change_and_scales.png", dpi=200)
print(f"gain: mismatch {np.mean(mismatch > 1):.3f}, neither {np.mean(neither > 1):.3f}, same {np.mean(same > 1):.3f}, n={len(slugs)}")

# gain/loss transitions between unclipped-both and clipped-both, with a species bootstrap
pub = np.array([clip[s]["future_ratio"] for s in slugs])
idx = np.random.default_rng(0).integers(0, len(slugs), (2000, len(slugs)))
for name, v in (("published decay both", pub), ("calibrated decay both", same)):
    down, up = (neither > 1) & (v <= 1), (neither <= 1) & (v > 1)
    ci = lambda x: np.percentile(x[idx].mean(1), [5, 95]).round(3).tolist()
    print(f"{name}: gain {v.mean() > 0 and np.mean(v > 1):.3f} {ci(v > 1)}; gain->loss {down.sum()} ({down.mean():.3f} {ci(down)}); loss->gain {up.sum()}; near 1 (+-0.02) {np.sum(abs(v - 1) <= 0.02)}")
print(f"unclipped both: gain {np.mean(neither > 1):.3f} {np.percentile((neither > 1)[idx].mean(1), [5, 95]).round(3).tolist()}")
