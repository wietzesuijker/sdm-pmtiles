#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scipy", "openpyxl"]
# ///
"""Does the calibrated half-normal sigma track dispersal traits, or observer effort?

Joins each scored species' own sigma (pool.py's scored set) to public traits by binomial and, per class and trait,
reports the Spearman correlation of log sigma with log trait and a weighted regression
log sigma ~ log trait + log n_new + log outside_mask (weights 1 / se^2, se from the bootstrap 90% interval as in
pool.py), with a species-bootstrap 90% interval on the trait slope.

Traits: birds hand-wing index and body mass (Sheard et al. 2020), mammals natal dispersal distance and body mass
(COMBINE, Soria et al. 2021, reported values), amphibians body size and mass (AmphiBIO, Oliveira et al. 2017).
Reptiles are skipped: the Slavenko et al. 2016 supplement could not be downloaded.

    decay/traits.py decay/out/calibration_all.jsonl hwi.xlsx combine_trait_data_reported.csv AmphiBIO_v1.csv
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from openpyxl import load_workbook
from pool import MIN_OUTSIDE, MIN_SE
from scipy.stats import spearmanr

N_BOOT = 1000


def key(name: str | None) -> str:
    return "-".join(str(name or "").lower().replace("_", " ").split()[:2])


def num(v) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return np.nan
    return x if x > 0 else np.nan


def birds(path: Path) -> dict[str, dict[str, float]]:
    rows = load_workbook(path, read_only=True)["speciesdata"].iter_rows(values_only=True)
    head = next(rows)
    out = {}
    for row in rows:
        r = dict(zip(head, row))
        t = {"hwi": num(r["HWI"]), "mass_g": 10 ** num(r["Body mass (log)"])}  # log10 g, birds > 1 g
        for name in (r["Species name"], r["IUCN name"]):
            out.setdefault(key(name), t)
    return out


def table(path: Path, names: tuple[str, ...], cols: dict[str, str]) -> dict[str, dict[str, float]]:
    out = {}
    with path.open(newline="", encoding="utf-8", errors="replace") as f:  # AmphiBIO is not clean UTF-8; binomials are ASCII
        for r in csv.DictReader(f):
            for name in names:
                out.setdefault(key(r[name]), {k: num(r[c]) for k, c in cols.items()})
    return out


def fit(y, x, n_new, outside, w) -> float:
    """Trait slope of the weighted regression log sigma ~ log trait + log n_new + log outside_mask."""
    X = np.column_stack([np.ones_like(y), x, np.log(n_new), np.log(outside)])
    sw = np.sqrt(w)
    return float(np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)[0][1])


def analyse(scored: list[dict], traits: dict[str, dict[str, float]], trait: str, rng) -> dict:
    m = [(r, traits[r["slug"]][trait]) for r in scored if np.isfinite(traits.get(r["slug"], {}).get(trait, np.nan))]
    res = {"n_scored": len(scored), "n_matched": len(m), "match_rate": round(len(m) / max(len(scored), 1), 3)}
    if len(m) < 10:
        return res
    y = np.log([r["halfnormal"].get("fit", r["halfnormal"]["best"]) for r, _ in m])
    lo, hi = np.log([r["halfnormal"]["ci90"] for r, _ in m]).T
    w = 1 / np.maximum((hi - lo) / 3.29, MIN_SE) ** 2
    x = np.log([t for _, t in m])
    n_new = np.array([r["n_new"] for r, _ in m], float)
    outside = np.array([r["outside_mask"] for r, _ in m], float)
    rho = spearmanr(y, x)
    boot = [fit(*(a[i] for a in (y, x, n_new, outside, w))) for i in rng.integers(0, len(y), (N_BOOT, len(y)))]
    res |= {"spearman_rho": round(float(rho.statistic), 3), "spearman_p": float(f"{rho.pvalue:.2g}"),
            "slope": round(fit(y, x, n_new, outside, w), 3),
            "slope_ci90": [round(float(v), 3) for v in np.percentile(boot, [5, 95])]}
    return res


def main(log: Path, hwi: Path, combine: Path, amphibio: Path) -> dict:
    rows = [r for r in map(json.loads, log.read_text().splitlines()) if r["ok"]]
    scored = [r for r in rows if "winner" in r and r["outside_mask"] >= MIN_OUTSIDE]
    sources = {
        "Birds": birds(hwi),
        "Mammals": table(combine, ("iucn2020_binomial", "phylacine_binomial"),
                         {"dispersal_km": "dispersal_km", "mass_g": "adult_mass_g"}),
        "Amphibians": table(amphibio, ("Species",), {"body_size_mm": "Body_size_mm", "mass_g": "Body_mass_g"}),
    }
    rng = np.random.default_rng(0)
    out = {}
    for cls, traits in sources.items():
        rs = [r for r in scored if r["class"] == cls]
        for trait in next(iter(traits.values())):
            out[f"{cls}/{trait}"] = analyse(rs, traits, trait, rng)
    out["Reptiles"] = "skipped: no downloadable body-size source (Slavenko et al. 2016 supplement blocked)"
    return out


def test():
    assert key("Tinamus_tao") == key("tinamus tao subsp") == "tinamus-tao"
    assert np.isnan(num("NA")) and np.isnan(num(0)) and num("2.5") == 2.5
    rng = np.random.default_rng(1)
    x, a, b = rng.normal(size=(3, 200))
    y = 0.5 * x + 0.3 * a + rng.normal(0, 0.01, 200)
    assert abs(fit(y, x, np.exp(a), np.exp(b), np.ones(200)) - 0.5) < 0.01  # slope recovered with proxies held fixed


if __name__ == "__main__":
    test()
    print(json.dumps(main(*map(Path, sys.argv[1:5])), indent=1))
