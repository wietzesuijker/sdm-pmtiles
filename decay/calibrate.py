#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scipy", "rasterio"]
# ///
"""Test the published evidence decay against new GBIF records.

For each species with a decay/out/<slug>_decay.tif (from evidence_decay.py),
pulls GBIF records from YEARS, drops any within EXCLUDE_KM of a model training
point (the filtered occurrence FlatGeobuf), and scores decay stretches s by the
presence-only point-process log-likelihood of the new records:

    LL(s) = mean_i log(lambda_s(x_i) / sum_cells lambda_s),  lambda_s = ens * g(d / s)

s = 1 is the published clip, s = 0 a hard clip to the mask, inf the raw
ensemble. Bootstraps records for a 90% interval on the best s. Writes
decay/out/calibration.json.

    decay/calibrate.py ursus-arctos canis-rufus ...
"""
from __future__ import annotations

import csv
import io
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

import numpy as np
from scipy import ndimage

from evidence_decay import BUCKET, MAX_KM, NODATA

OUT = Path(__file__).parent / "out"
YEARS = "2025,2026"
EXCLUDE_KM = 2
MAX_RECORDS = 3000
EPS = 0.5  # floor on lambda in ensemble units, so a hard clip is finite outside the mask
STRETCHES = [0, 0.0625, 0.125, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 5, 8, float("inf")]
GBIF = "https://api.gbif.org/v1/occurrence/search"
BREEDING = "&month=6,7"  # birds: migration records land far outside the modelled range


def gbif_points(keys: list[str], months: str = "") -> np.ndarray:
    pts = []
    for key in keys:
        offset = 0
        while len(pts) < MAX_RECORDS:
            q = (f"{GBIF}?taxonKey={key}&year={YEARS}&hasCoordinate=true&hasGeospatialIssue=false"
                 f"&occurrenceStatus=PRESENT&decimalLongitude=-180,-50&decimalLatitude=10,85&limit=300&offset={offset}{months}")
            page = json.load(urllib.request.urlopen(q, timeout=60))
            pts += [(r["decimalLongitude"], r["decimalLatitude"]) for r in page["results"]
                    if (r.get("coordinateUncertaintyInMeters") or 0) <= 5000]
            offset += 300
            if page["endOfRecords"]:
                break
    return np.array(pts[:MAX_RECORDS], float).reshape(-1, 2)


def training_points(slug: str, crs: str) -> np.ndarray:
    url = f"{BUCKET}/{slug}_gbif-occurrence-points-filtered_NA_na_NA_NA.fgb"
    out = subprocess.run(["ogr2ogr", "-f", "CSV", "-t_srs", crs, "/vsistdout/", url, "-lco", "GEOMETRY=AS_XY"],
                         capture_output=True, text=True, check=True).stdout
    rows = list(csv.DictReader(io.StringIO(out)))
    return np.array([(float(r["X"]), float(r["Y"])) for r in rows]).reshape(-1, 2)


def loglik(ens, dist, curve, rows, cols, s) -> tuple[float, np.ndarray]:
    """Per-record log of normalised intensity for stretch s."""
    if s == float("inf"):
        g = np.ones_like(dist, float)
    elif s == 0:
        g = (dist == 0).astype(float)
    else:
        g = curve[np.minimum(np.floor(dist / s + 0.5), MAX_KM).astype(int)]
    lam = np.where(ens == NODATA, 0, np.maximum(ens * g, EPS))
    return float(lam.sum()), np.log(lam[rows, cols])


def run(slug: str, keys: list[str], curve: np.ndarray, months: str = "") -> dict:
    import rasterio
    from rasterio.warp import transform

    with rasterio.open(OUT / f"{slug}_decay.tif") as src:
        ens, dist = src.read()
        crs, tf = src.crs, src.transform
    ens = ens.astype(np.int16)
    lonlat = gbif_points(keys, months)
    if not len(lonlat):
        return {"n_new": 0}
    x, y = transform("EPSG:4326", crs, lonlat[:, 0], lonlat[:, 1])
    rows, cols = rasterio.transform.rowcol(tf, x, y)
    rows, cols = np.array(rows), np.array(cols)
    ok = (rows >= 0) & (rows < ens.shape[0]) & (cols >= 0) & (cols < ens.shape[1])
    rows, cols = rows[ok], cols[ok]
    ok = ens[rows, cols] != NODATA
    rows, cols = rows[ok], cols[ok]
    # drop new records near any training point, so the test set is not the fit set
    train = training_points(slug, crs.to_string())
    near = np.zeros(ens.shape, bool)
    tr, tc = rasterio.transform.rowcol(tf, train[:, 0], train[:, 1])
    tr, tc = np.array(tr), np.array(tc)
    keep = (tr >= 0) & (tr < ens.shape[0]) & (tc >= 0) & (tc < ens.shape[1])
    near[tr[keep], tc[keep]] = True
    near = ndimage.binary_dilation(near, iterations=EXCLUDE_KM)
    ok = ~near[rows, cols]
    rows, cols = rows[ok], cols[ok]
    n = len(rows)
    if n < 10:
        return {"n_new": n}
    per = {}
    for s in STRETCHES:
        total, logs = loglik(ens, dist, curve, rows, cols, s)
        per[s] = logs - np.log(total)
    ll = {str(s): round(float(v.mean()), 4) for s, v in per.items()}
    best = max(per, key=lambda s: per[s].mean())
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(200):
        i = rng.integers(0, n, n)
        boots.append(max(per, key=lambda s: per[s][i].mean()))
    lo, hi = np.percentile([min(b, 1e9) for b in boots], [5, 95])
    return {
        "n_new": n, "outside_mask": int((dist[rows, cols] > 0).sum()),
        "best_stretch": best if best != float("inf") else "inf",
        "ci90": [float(lo) if lo < 1e9 else "inf", float(hi) if hi < 1e9 else "inf"],
        "ll_gain_best_vs_published": round(float(per[best].mean() - per[1].mean()), 4),
        "ll_gain_published_vs_ensemble": round(float(per[1].mean() - per[float("inf")].mean()), 4),
        "ll": ll,
    }


if __name__ == "__main__":
    curves = json.loads((OUT / "curves.json").read_text())
    manifest = json.load(urllib.request.urlopen(BUCKET.removeprefix("/vsicurl/") + "/manifest.json"))
    path = OUT / "calibration.json"
    result = json.loads(path.read_text()) if path.exists() else {}
    for slug in sys.argv[1:]:
        keys = manifest[slug]["GBIF.IDs"][0].split("~")
        months = BREEDING if manifest[slug]["Class"][0] == "Birds" else ""
        result[slug] = r = run(slug, keys, np.array(curves[slug]["curve"]), months)
        print(slug, {k: v for k, v in r.items() if k != "ll"}, flush=True)
        path.write_text(json.dumps(result, indent=1))
