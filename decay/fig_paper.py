#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "matplotlib"]
# ///
"""Paper Fig. 1 from the batch logs: share of species gaining habitat by 2071-2100 under each weighting, and calibrated decay scales by class.

    decay/fig_paper.py decay/out paper-media/fig1    # writes fig1.png (300 dpi) and fig1.pdf, prints the statistics
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

MM = 1 / 25.4
INK, MUTED, GRID = "#1a1a19", "#6b6a66", "#d9d8d4"
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#8c8b87"  # validated: dataviz validate_palette.js, all pairs, light
STYLE = {"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 7,
         "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 7, "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
         "xtick.color": MUTED, "ytick.color": INK, "xtick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 0,
         "axes.labelcolor": INK, "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
         "pdf.fonttype": 42, "savefig.dpi": 300}
CLASSES = ["Amphibians", "Reptiles", "Mammals", "Birds"]
CAP_SIGMA = 2000  # km; the fit search bound, effectively no decay
D10 = np.sqrt(2 * np.log(10))  # half-normal weight falls to 0.1 at D10 * sigma


def panel_label(ax, s):
    ax.text(0, 1, s, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom", ha="right")


def save(fig, stem):
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(stem.with_suffix(f".{ext}"), facecolor="white")


def main(out, stem):
    load = lambda name: {r["slug"]: r for r in map(json.loads, (out / name).read_text().splitlines()) if r["ok"]}
    clip, raw, cal = load("batch.jsonl"), load("raw_ratio.jsonl"), load("calibration_all.jsonl")
    pool = json.loads((out / "decay_pooled.json").read_text())
    pooled, by_class = pool["species"], pool["summary"]["classes"]
    slugs = sorted(set(clip) & set(raw) & set(pooled) & set(cal))
    cls = np.array([cal[s]["class"] for s in slugs])
    mismatch = np.array([raw[s]["raw_future_ratio"] for s in slugs])
    neither = np.array([raw[s]["raw_future_ratio"] / raw[s]["raw_now_ratio"] for s in slugs])
    pub = np.array([clip[s]["future_ratio"] for s in slugs])
    same = np.array([pooled[s]["change"] for s in slugs])
    rng = np.random.default_rng(0)

    def share(v, m):  # % gaining and species-bootstrap 90% interval
        g = (v[m] > 1).astype(float)
        boot = g[rng.integers(0, len(g), (2000, len(g)))].mean(1)
        return 100 * g.mean(), 100 * np.percentile(boot, [5, 95])

    series = [("inconsistent", mismatch, "o", "white", INK), ("no weighting", neither, "o", GREY, GREY),
              ("published", pub, "D", ORANGE, ORANGE), ("calibrated", same, "o", BLUE, BLUE)]
    rows = [("All species", np.ones(len(slugs), bool))] + [(c, cls == c) for c in CLASSES]

    plt.rcParams.update(STYLE)
    fig, (a, b) = plt.subplots(1, 2, figsize=(180 * MM, 70 * MM), gridspec_kw={"width_ratios": [1.15, 1], "wspace": 0.08})
    fig.subplots_adjust(left=0.115, right=0.985, top=0.93, bottom=0.15)

    # a: dot plot, one row per group, one marker per weighting
    print("share of species gaining, % [90% CI]")
    for y, (name, m) in enumerate(rows):
        vals = {}
        for lab, v, mk, fc, ec in series:
            p, ci = share(v, m)
            vals[lab] = p
            a.plot(ci, [y, y], color=ec if fc != "white" else INK, lw=0.8, solid_capstyle="butt", zorder=2)
            a.scatter(p, y, s=22 if mk == "o" else 18, marker=mk, facecolor=fc, edgecolor=ec, lw=0.8, zorder=3)
            print(f"  {name:11s} {lab:12s} {p:5.1f} [{ci[0]:.1f}-{ci[1]:.1f}]  n={m.sum()}")
        lo, hi = min(vals.values()), max(vals.values())
        a.plot([lo, hi], [y, y], color=GRID, lw=3.2, zorder=1, solid_capstyle="round")
        a.text(101.5, y, f"{vals['no weighting']:.0f}", color=GREY, va="center", ha="left", fontsize=7)
        a.text(108.5, y, "→", color=MUTED, va="center", ha="center", fontsize=7)
        a.text(110.5, y, f"{vals['calibrated']:.0f}", color=BLUE, va="center", ha="left", fontsize=7, fontweight="bold")
    for x, (lab, _, mk, fc, ec) in zip((0, 24, 50, 76), (series[2], series[3], series[1], series[0])):  # key row
        a.scatter(x + 1.5, -1, s=22 if mk == "o" else 18, marker=mk, facecolor=fc, edgecolor=ec, lw=0.8, clip_on=False)
        a.text(x + 4, -1, lab, fontsize=6.5, va="center")
    a.text(109, -1, "none → cal.", color=MUTED, ha="center", va="center", fontsize=6)
    a.axvline(50, color=GRID, lw=0.6, ls=(0, (2, 2)), zorder=0)
    a.set_yticks(range(len(rows)), [f"{n}\n(n={m.sum():,})" for n, m in rows])
    a.set_ylim(len(rows) - 0.5, -1.4)
    a.set_xlim(0, 117)
    a.set_xticks(range(0, 101, 25))
    a.spines["left"].set_visible(False)
    a.spines["bottom"].set_bounds(0, 100)
    a.set_xlabel("Species gaining habitat by 2071-2100 (%)")
    panel_label(a, "a")

    # b: distance at which the weight falls to 10%, per scored species, by class
    print("distance to 10% weight, km: pooled calibrated vs median published")
    for y, c in enumerate(CLASSES, start=1):
        rs = [cal[s] for s in slugs if cal[s]["class"] == c and "winner" in cal[s] and cal[s]["outside_mask"] >= 3]
        own = D10 * np.array([r["halfnormal"]["fit"] for r in rs])
        published = np.median([clip[r["slug"]]["d10_km"] for r in rs if clip[r["slug"]]["d10_km"] is not None])
        pooled_km = D10 * by_class[c]["scale_km"]
        cap = own >= D10 * CAP_SIGMA  # fits at the search bound: no decay, counted in the right-hand cell
        jit = np.random.default_rng(y).uniform(-0.22, 0.22, (~cap).sum())
        b.scatter(own[~cap], y + jit, s=2.5, color=BLUE, alpha=0.28, lw=0, zorder=2, rasterized=True)
        q1, q3 = np.percentile(own, [25, 75])
        b.plot([q1, q3], [y, y], color=BLUE, lw=2.4, alpha=0.45, solid_capstyle="butt", zorder=3)
        b.plot([pooled_km] * 2, [y - 0.3, y + 0.3], color=BLUE, lw=1.6, zorder=4, solid_capstyle="butt")
        b.scatter(published, y, s=18, marker="D", facecolor=ORANGE, edgecolor="white", lw=0.6, zorder=5)
        b.text(pooled_km, y - 0.34, f"{pooled_km:.0f}",
               color=BLUE, fontsize=6.5, fontweight="bold", ha="center", va="bottom")
        b.text(published, y - 0.34, f"{published:.0f}", color=ORANGE, fontsize=6.5, ha="center", va="bottom")
        b.text(9e3, y, f"{cap.sum()}", color=INK, fontsize=6.5, va="center", ha="center")
        b.text(2.3e4, y, f"{len(own)}", color=MUTED, fontsize=6, va="center", ha="center")
        print(f"  {c:11s} sigma {by_class[c]['scale_km']:6.1f} -> {pooled_km:6.1f} km   published median {published:6.1f} km   n={len(own)}")
    b.scatter([1.1, 1.35, 1.6], [-1, -1.08, -0.92], s=2.5, color=BLUE, alpha=0.5, lw=0)  # key row
    b.text(2, -1, "per species", fontsize=6.5, va="center")
    b.plot([20, 20], [-1.3, -0.7], color=BLUE, lw=1.6, solid_capstyle="butt")
    b.text(26, -1, "pooled calibrated", fontsize=6.5, va="center")
    b.scatter(700, -1, s=18, marker="D", facecolor=ORANGE, edgecolor="white", lw=0.6)
    b.text(900, -1, "published (median)", fontsize=6.5, va="center")
    b.text(9e3, -0.05, "no\ndecay", color=MUTED, fontsize=6, ha="center", va="bottom", linespacing=1.1)
    b.text(2.3e4, -0.05, "n", color=MUTED, fontsize=6, ha="center", va="bottom")
    b.axvspan(6e3, 3.2e4, color="#f3f2ef", lw=0, zorder=0)
    b.set_xscale("log")
    b.set_xlim(0.7, 3.2e4)
    b.set_xticks([1, 10, 100, 1000], ["1", "10", "100", "1,000"])
    b.xaxis.set_minor_locator(plt.NullLocator())
    b.spines["bottom"].set_bounds(0.7, 5e3)
    for x in (10, 100, 1000):
        b.axvline(x, color=GRID, lw=0.4, zorder=0)
    b.set_ylim(len(rows) - 0.5, -1.4)
    b.set_yticks([])
    b.spines["left"].set_visible(False)
    b.set_xlabel("Distance at which evidence weight falls to 10% (km)")
    panel_label(b, "b")
    save(fig, stem)

    # gain/loss transitions between no weighting and weighting in both periods, with a species bootstrap
    idx = rng.integers(0, len(slugs), (2000, len(slugs)))
    ci = lambda x: np.percentile(x[idx].mean(1), [5, 95]).round(3).tolist()
    for name, v in (("published decay both", pub), ("calibrated decay both", same)):
        down, up = (neither > 1) & (v <= 1), (neither <= 1) & (v > 1)
        print(f"{name}: gain->loss {down.sum()} ({down.mean():.3f} {ci(down)}); loss->gain {up.sum()}; near 1 (+-0.02) {np.sum(abs(v - 1) <= 0.02)}")
    print(f"published vs calibrated disagree in direction: {np.sum((pub > 1) != (same > 1))}; n={len(slugs)}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(Path(sys.argv[1]), sys.argv[2])
