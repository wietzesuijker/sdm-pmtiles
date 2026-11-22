#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scipy", "matplotlib", "openpyxl"]
# ///
"""Paper Figure 3: new records against the classic dispersal bounds (a), and calibrated sigma against dispersal traits (b).

(a) per-species record log-likelihood of each dispersal scenario relative to the fitted half-normal, summed over the
120 species of dispersal_bounds.py, and the share of species gaining habitat by 2071-2100; species-bootstrap 90% CIs.
(b) partial-residual plot of log sigma against log trait with the effort proxies (log n_new, log outside_mask) held
at their means, i.e. the traits.py regression made visible: birds hand-wing index, mammals natal dispersal distance.

    decay/fig3.py decay/out/dispersal_bounds.jsonl decay/out/calibration_all.jsonl hwi.xlsx combine_reported.csv OUT_DIR
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from pool import MIN_OUTSIDE, MIN_SE
from traits import analyse, birds, table

N_BOOT = 2000
MM = 1 / 25.4
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
CAL, BOUND = "#2a78d6", "#eb6834"  # reference palette slots 1-2: calibrated vs classic bound
mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 7,
    "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "svg.fonttype": "none",
})


def wls(X, y, w) -> np.ndarray:
    sw = np.sqrt(w)
    return np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)[0]


def ci(boot) -> list[float]:
    return np.percentile(boot, [5, 95], axis=0)


def bounds(path: Path, rng) -> dict:
    rows = sorted((r for r in map(json.loads, path.read_text().splitlines()) if r["ok"]), key=lambda r: r["slug"])
    if len(rows) < 10:
        sys.exit(f"{path}: only {len(rows)} ok species")
    # rows by slug, draws none, full, pooled from one seeded generator, as dispersal_bounds.summary, so the CIs match its log
    ll = {k: np.array([r["ll"][k] - r["ll"]["halfnormal"] for r in rows]) for k in ("none", "full", "pooled")}
    ll = {k: (v.sum(), ci(v[rng.integers(0, len(v), (N_BOOT, len(v)))].sum(1))) for k, v in ll.items()}
    gain = {k: np.array([r["change"][k] > 1 for r in rows], float) for k in ("none", "halfnormal", "full")}
    idx = rng.integers(0, len(rows), (N_BOOT, len(rows)))
    return {"n": len(rows),
            "ll": ll, "beats": {k: float((np.array([r["ll"][k] for r in rows]) > [r["ll"]["halfnormal"] for r in rows]).mean())
                                for k in ll},
            "gain": {k: (v.mean(), ci(v[idx].mean(1))) for k, v in gain.items()}}


def trait_fit(cal: list[dict], traits: dict[str, dict[str, float]], trait: str, stats: dict) -> dict:
    """Effort-adjusted partial residuals for plotting; slope and CI are traits.analyse's, so they match the text."""
    m = [(r, traits[r["slug"]][trait]) for r in cal if np.isfinite(traits.get(r["slug"], {}).get(trait, np.nan))]
    y = np.log([r["halfnormal"].get("fit", r["halfnormal"]["best"]) for r, _ in m])
    lo, hi = np.log([r["halfnormal"]["ci90"] for r, _ in m]).T
    w = 1 / np.maximum((hi - lo) / 3.29, MIN_SE) ** 2
    x = np.log([t for _, t in m])
    z = np.log([[r["n_new"], r["outside_mask"]] for r, _ in m])
    b = wls(np.column_stack([np.ones_like(y), x, z]), y, w)
    assert abs(b[1] - stats["slope"]) < 1e-3, (b[1], stats["slope"])
    resid = y - (z - z.mean(0)) @ b[2:]  # partial residual: log sigma with effort held at its mean
    return {"n": len(y), "n_class": len(cal), "x": x, "y": resid, "w": w, "b": b, "slope_ci": stats["slope_ci90"],
            "zbar": z.mean(0)}


def panel_a(ax_ll, ax_g, s):
    rows = [("none", "No dispersal"), ("halfnormal", "Calibrated (own σ)"), ("pooled", "Calibrated (class σ)"),
            ("full", "Full dispersal")]
    ys = np.arange(len(rows))[::-1]
    for ax in (ax_ll, ax_g):
        ax.set_ylim(-0.6, len(rows) - 0.4)
        ax.set_yticks(ys, [lab for _, lab in rows] if ax is ax_ll else [])
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
        for y in ys:
            ax.axhline(y, color=GRID, lw=0.5, zorder=0)
    for y, (k, _) in zip(ys, rows):
        c = CAL if k in ("halfnormal", "pooled") else BOUND
        if k in s["ll"]:
            v, (lo, hi) = s["ll"][k]
            ax_ll.plot([-hi, -lo], [y, y], color=c, lw=1.6, solid_capstyle="round")
            ax_ll.plot(-v, y, "o", ms=5, color=c, mec="white", mew=0.8)
            ax_ll.text(-v, y + 0.26, f"{-v:,.0f}", ha="center", va="bottom", fontsize=6, color=INK2)
        else:
            ax_ll.text(70, y, "reference", ha="left", va="center", fontsize=6, color=MUTED, style="italic")
        if k in s["gain"]:
            v, (lo, hi) = s["gain"][k]
            ax_g.plot([lo * 100, hi * 100], [y, y], color=c, lw=1.6, solid_capstyle="round")
            ax_g.plot(v * 100, y, "o", ms=5, color=c, mec="white", mew=0.8)
            ax_g.text(v * 100, y + 0.26, f"{v:.0%}", ha="center", va="bottom", fontsize=6, color=INK2)
        else:
            ax_g.text(50, y, "not computed", ha="center", va="center", fontsize=6, color=MUTED, style="italic")
    ax_ll.set_xscale("log")
    ax_ll.set_xlim(60, 1.5e5)
    ax_ll.set_xticks([1e2, 1e3, 1e4, 1e5], ["100", "1,000", "10,000", "100,000"])
    ax_ll.set_xlabel("Log-likelihood deficit vs own σ\n(summed over species; lower fits better)")
    ax_g.set_xlim(0, 100)
    ax_g.set_xticks([0, 25, 50, 75, 100], ["0", "25", "50", "75", "100%"])
    ax_g.axvline(50, color=GRID, lw=0.6, zorder=0)
    ax_g.set_xlabel("Species gaining habitat\nby 2071-2100")
    ax_ll.set_title(f"n = {s['n']} species; 90% species-bootstrap CIs", loc="right", fontsize=6.5, color=INK2)


def panel_b(ax, f, xlabel, ticks, title, color=CAL):
    ax.scatter(np.exp(f["x"]), np.exp(f["y"]), s=np.clip(f["w"] / f["w"].max() * 14, 2.5, 14), color=color, alpha=0.45,
               lw=0, rasterized=True)
    xx = np.linspace(f["x"].min(), f["x"].max(), 50)
    yy = f["b"][0] + f["b"][1] * xx + f["zbar"] @ f["b"][2:]
    ax.plot(np.exp(xx), np.exp(yy), color=INK, lw=1.2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
    ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())
    ax.set_yticks([0.1, 1, 10, 100, 1e3, 1e4], ["0.1", "1", "10", "100", "1,000", "10,000"])
    ax.yaxis.set_minor_locator(mpl.ticker.NullLocator())
    ax.set_xlabel(xlabel)
    b, lo, hi = (round(float(v), 3) + 1e-9 for v in (f["b"][1], *f["slope_ci"]))  # 3 dp then half-up, matching the traits.py figures in the text
    ax.text(0.03, 0.97, f"{title}\nslope {b:.2f} (90% CI {lo:.2f} to {hi:.2f})\nn = {f['n']} of {f['n_class']}",
            transform=ax.transAxes, ha="left", va="top", fontsize=6.5, color=INK, linespacing=1.4)


def main(bounds_path: Path, cal_path: Path, hwi_path: Path, combine_path: Path, out: Path) -> dict:
    rng = np.random.default_rng(0)
    s = bounds(bounds_path, rng)
    cal = [r for r in map(json.loads, cal_path.read_text().splitlines())
           if r["ok"] and "winner" in r and r["outside_mask"] >= MIN_OUTSIDE]
    sources = {"Birds": birds(hwi_path), "Mammals": table(combine_path, ("iucn2020_binomial", "phylacine_binomial"),
                                                          {"dispersal_km": "dispersal_km", "mass_g": "adult_mass_g"})}
    trng, stats = np.random.default_rng(0), {}  # one generator over classes and traits in traits.main's order
    for c, t in sources.items():
        rs = [r for r in cal if r["class"] == c]
        for trait in next(iter(t.values())):
            stats[c, trait] = analyse(rs, t, trait, trng)
    bird = trait_fit([r for r in cal if r["class"] == "Birds"], sources["Birds"], "hwi", stats["Birds", "hwi"])
    mammals = trait_fit([r for r in cal if r["class"] == "Mammals"], sources["Mammals"], "dispersal_km",
                        stats["Mammals", "dispersal_km"])

    fig = plt.figure(figsize=(180 * MM, 118 * MM))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.25, 0.8, 0.95], height_ratios=[1, 1.25], hspace=0.62, wspace=0.18,
                          left=0.135, right=0.97, top=0.95, bottom=0.1)
    ax_ll, ax_g = fig.add_subplot(gs[0, :2]), fig.add_subplot(gs[0, 2])
    ax_b, ax_m = fig.add_subplot(gs[1, :2]), fig.add_subplot(gs[1, 2])
    panel_a(ax_ll, ax_g, s)
    panel_b(ax_b, bird, "Hand-wing index (Sheard et al. 2020)", [10, 20, 40, 70], "Birds")
    panel_b(ax_m, mammals, "Natal dispersal distance (km)", [0.1, 1, 10, 100], "Mammals")
    ax_b.set_ylabel("Calibrated σ (km),\nadjusted for record effort")
    ax_m.sharey(ax_b)
    ax_m.tick_params(labelleft=False)
    fig.text(0.01, 0.97, "a", fontsize=9, fontweight="bold", va="top")
    fig.text(0.01, 0.53, "b", fontsize=9, fontweight="bold", va="top")
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"fig3.{ext}", dpi=300)

    r = lambda v: [round(float(x), 3) for x in np.atleast_1d(v)]
    return {"n_bounds": s["n"], "ll_vs_own": {k: r([v, *c]) for k, (v, c) in s["ll"].items()},
            "share_beating_own": {k: round(v, 3) for k, v in s["beats"].items()},
            "gain_share": {k: r([v, *c]) for k, (v, c) in s["gain"].items()},
            **{c: {"n": f["n"], "of": f["n_class"], "slope": r([f["b"][1], *f["slope_ci"]])}
               for c, f in (("birds_hwi", bird), ("mammals_natal", mammals))}}


def test():
    g = np.random.default_rng(1)
    x, a = g.normal(size=(2, 200))
    X = np.column_stack([np.ones(200), x, a])
    assert abs(wls(X, 0.5 * x + 0.3 * a, np.ones(200))[1] - 0.5) < 1e-9


if __name__ == "__main__":
    test()
    if len(sys.argv) != 6:
        sys.exit(__doc__)
    print(json.dumps(main(*map(Path, sys.argv[1:6])), indent=1))
