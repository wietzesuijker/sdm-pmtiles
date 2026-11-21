#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scipy", "rasterio", "mapbox-vector-tile"]
# ///
"""Score decay shapes against 2025-26 GBIF records for every manifest species.

Same test as calibrate.py (new records away from training points, presence-only
point-process likelihood, target-group effort), run from the bucket files without
writing rasters, and over three decay families of d = km to the evidence mask:

    stretch     the published curve g(d / s)
    exp         exp(-d / tau)
    halfnormal  exp(-d^2 / (2 sigma^2))

Appends one JSON line per species to the log and skips species already in it.

    decay/calibrate_batch.py decay/out/calibration_all.jsonl [jobs]
"""
from __future__ import annotations

import json
import subprocess
import sys
import urllib.request
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from calibrate import (
    BREEDING,
    EFFORT_KM,
    EPS,
    STRETCHES,
    block_sum,
    class_effort,
    new_records,
)
from evidence_decay import BUCKET, MAX_KM, NODATA, distance_km, fit_curve

SCALES = [1, 2, 5, 10, 20, 40, 80, 160, 320, 640, 1280, 2560]  # km, for tau and sigma
KM = np.arange(MAX_KM + 1)
BOOT = 200
_effort: dict = {}  # per worker: (class, months) -> effort grid, the same for every species


def tables(curve: np.ndarray) -> dict[tuple[str, float], np.ndarray]:
    """Decay value per km for every candidate, keyed (family, parameter)."""
    t = {}
    for s in STRETCHES:
        if s == float("inf"):
            t["stretch", s] = np.ones(MAX_KM + 1)
        elif s == 0:
            t["stretch", s] = (KM == 0).astype(float)
        else:
            t["stretch", s] = curve[np.minimum(np.floor(KM / s + 0.5), MAX_KM).astype(int)]
    for v in SCALES:
        t["exp", v] = np.exp(-KM / v)
        t["halfnormal", v] = np.exp(-(KM / v) ** 2 / 2)
    return t


def one(slug: str, keys: list[str], cls: str) -> dict:
    import rasterio

    try:
        def read(product):
            with rasterio.open(f"{BUCKET}/{slug}_{product}_1km_na_1980-2010_sett1.tif") as src:
                return src.read(1), src.crs, src.transform

        ens, crs, tf = read("sdm-ensemble")
        clip, clipped = read("sdm-clip")[0], read("sdm-ensemble-clipped")[0]
        dist = distance_km(clip)
        curve = fit_curve(ens, clip, clipped, dist)
        ens = ens.astype(np.int16)
        months = BREEDING if cls == "Birds" else ""
        rows, cols = new_records(slug, keys, months, ens, crs, tf)
        n = len(rows)
        if n < 10:
            return {"slug": slug, "ok": True, "class": cls, "n_new": n}
        key = (cls, months, tuple(tf), ens.shape)
        if key not in _effort:
            _effort[key] = class_effort(cls, months, crs, tf, ens.shape)
        effort = _effort[key]
        k = EFFORT_KM
        at = effort[rows // k, cols // k]
        per = {}
        for c, table in tables(curve).items():
            lam = np.where(ens == NODATA, 0, np.maximum(ens * table[dist], EPS))
            per[c] = np.log(lam[rows, cols] * at) - np.log((block_sum(lam, k) * effort).sum())
        return {"slug": slug, "ok": True, "class": cls, "n_new": n,
                "outside_mask": int((dist[rows, cols] > 0).sum()), **summarise(per)}
    # network, raster, GBIF and ogr2ogr failures; one bad species must not stop the batch
    except (OSError, ValueError, KeyError, IndexError, subprocess.CalledProcessError) as e:
        return {"slug": slug, "ok": False, "error": f"{type(e).__name__}: {e}"[:300]}


def j(p: float) -> float | str:
    """JSON has no infinity."""
    return "inf" if p == float("inf") else p


def summarise(per: dict) -> dict:
    """Best parameter and mean log-likelihood per family, the overall winner, and bootstrap spread."""
    def best(idx=slice(None)):
        means = {c: v[idx].mean() for c, v in per.items()}
        fam = {}
        for (f, p), m in means.items():
            if f not in fam or m > means[f, fam[f]]:
                fam[f] = p
        return fam, max(means, key=means.get)

    fam, win = best()
    n = len(next(iter(per.values())))
    boots = [best(i) for i in np.random.default_rng(0).integers(0, n, (BOOT, n))]
    out = {"winner": [win[0], j(win[1])],
           "winner_share": {f: sum(b[1][0] == f for b in boots) / BOOT for f in ("stretch", "exp", "halfnormal")},
           "ll_published": round(float(per["stretch", 1].mean()), 4)}
    for f, p in fam.items():
        ps = sorted(b[0][f] for b in boots)
        out[f] = {"best": j(p), "ll": round(float(per[f, p].mean()), 4),
                  "ci90": [j(ps[int(0.05 * BOOT)]), j(ps[int(0.95 * BOOT) - 1])]}
    return out


if __name__ == "__main__":
    log, jobs = Path(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 4
    manifest = json.load(urllib.request.urlopen(BUCKET.removeprefix("/vsicurl/") + "/manifest.json"))
    done = {json.loads(line)["slug"] for line in log.read_text().splitlines()} if log.exists() else set()
    todo = [s for s in sorted(manifest) if s not in done]
    print(f"{len(todo)} of {len(manifest)} to do", flush=True)
    with ProcessPoolExecutor(jobs) as pool, log.open("a") as out:
        futs = [pool.submit(one, s, manifest[s]["GBIF.IDs"][0].split("~"), manifest[s]["Class"][0]) for s in todo]
        for fut in as_completed(futs):
            res = fut.result()
            out.write(json.dumps(res) + "\n")
            out.flush()
            print(res["slug"], "ok" if res["ok"] else res["error"], flush=True)
