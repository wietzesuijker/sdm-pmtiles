#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scipy", "rasterio", "mapbox-vector-tile"]
# ///
"""Test the published evidence decay against new GBIF records.

For each species with a decay/out/<slug>_decay.tif (from evidence_decay.py),
pulls GBIF records from YEARS, drops any within EXCLUDE_KM of a model training
point (the filtered occurrence FlatGeobuf), and scores decay stretches s by the
presence-only point-process log-likelihood of the new records:

    LL(s) = mean_i log(lambda_s(x_i) / sum_cells lambda_s),  lambda_s = ens * g(d / s)

s = 1 is the published clip, s = 0 a hard clip to the mask, inf the raw
ensemble. Bootstraps records for a 90% interval on the best s. New records
share the observer bias of the training records, which rewards tight clips, so
each species is scored twice: as above, and with lambda_s times the density of
all GBIF records of its class in the same years (a target-group background,
Phillips et al. 2009). Writes decay/out/calibration.json.

    decay/calibrate.py ursus-arctos canis-rufus ...
"""
from __future__ import annotations

import csv
import io
import json
import subprocess
import sys
import urllib.error
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
GBIF_MAP = "https://api.gbif.org/v2/map/occurrence/adhoc"
CLASS_KEYS = {"Mammals": ["359"], "Birds": ["212"], "Amphibians": ["131"],
              "Reptiles": ["11592253", "11418114", "11493978"]}  # GBIF splits reptiles in 3
EFFORT_KM = 25  # target-group grid; map squares at zoom 4 are ~0.35 deg


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


def class_effort(cls: str, months: str, crs, tf, shape) -> np.ndarray:
    """GBIF records of the class in YEARS per EFFORT_KM cell, plus one, from zoom 4 map tiles."""
    import mapbox_vector_tile
    from rasterio.warp import transform

    taxa = "".join(f"&taxonKey={k}" for k in CLASS_KEYS[cls])
    lon, lat, n = [], [], []
    for tx in range(12):  # EPSG:4326 zoom 4 tiles are 11.25 deg, these cover lon -180..-45, lat 90..0
        for ty in range(8):
            q = (f"{GBIF_MAP}/4/{tx}/{ty}.mvt?srs=EPSG:4326&year={YEARS}&hasGeospatialIssue=false"
                 f"&bin=square&squareSize=16{taxa}{months}")
            try:
                layer = mapbox_vector_tile.decode(urllib.request.urlopen(q, timeout=60).read()).get("occurrence")
            except urllib.error.HTTPError as e:  # GBIF answers 400 for some empty tiles
                if b"missing the expected layer" not in e.read():
                    raise
                continue
            for f in (layer or {}).get("features", []):
                ring = np.array(f["geometry"]["coordinates"][0], float)
                cx, cy = ring.mean(0) / layer["extent"]  # y counts up from the tile bottom
                lon.append(-180 + (tx + cx) * 11.25)
                lat.append(90 - (ty + 1 - cy) * 11.25)
                n.append(f["properties"]["total"])
    x, y = transform("EPSG:4326", crs, lon, lat)
    rows, cols = rasterio_rowcol(tf, x, y)
    effort = np.ones((-(-shape[0] // EFFORT_KM), -(-shape[1] // EFFORT_KM)))
    ok = (rows >= 0) & (rows < shape[0]) & (cols >= 0) & (cols < shape[1])
    np.add.at(effort, (rows[ok] // EFFORT_KM, cols[ok] // EFFORT_KM), np.array(n)[ok])
    return effort


def rasterio_rowcol(tf, x, y) -> tuple[np.ndarray, np.ndarray]:
    import rasterio

    r, c = rasterio.transform.rowcol(tf, x, y)
    return np.array(r), np.array(c)


def block_sum(a: np.ndarray, k: int) -> np.ndarray:
    h, w = -(-a.shape[0] // k) * k, -(-a.shape[1] // k) * k
    return np.pad(a, ((0, h - a.shape[0]), (0, w - a.shape[1]))).reshape(h // k, k, w // k, k).sum((1, 3))


def loglik(ens, dist, curve, rows, cols, s, effort=None) -> tuple[float, np.ndarray]:
    """Per-record log intensity for stretch s and its normaliser, optionally times sampling effort."""
    if s == float("inf"):
        g = np.ones_like(dist, float)
    elif s == 0:
        g = (dist == 0).astype(float)
    else:
        g = curve[np.minimum(np.floor(dist / s + 0.5), MAX_KM).astype(int)]
    lam = np.where(ens == NODATA, 0, np.maximum(ens * g, EPS))
    if effort is None:
        return float(lam.sum()), np.log(lam[rows, cols])
    k = EFFORT_KM
    return float((block_sum(lam, k) * effort).sum()), np.log(lam[rows, cols] * effort[rows // k, cols // k])


def score(per: dict) -> dict:
    """Best stretch, bootstrap 90% interval and gains from per-record normalised log intensities."""
    best = max(per, key=lambda s: per[s].mean())
    rng = np.random.default_rng(0)
    n = len(per[best])
    boots = [max(per, key=lambda s: per[s][i].mean()) for i in rng.integers(0, n, (200, n))]
    lo, hi = np.percentile([min(b, 1e9) for b in boots], [5, 95])
    return {
        "best_stretch": best if best != float("inf") else "inf",
        "ci90": [float(lo) if lo < 1e9 else "inf", float(hi) if hi < 1e9 else "inf"],
        "ll_gain_best_vs_published": round(float(per[best].mean() - per[1].mean()), 4),
        "ll_gain_published_vs_ensemble": round(float(per[1].mean() - per[float("inf")].mean()), 4),
        "ll": {str(s): round(float(v.mean()), 4) for s, v in per.items()},
    }


def run(slug: str, keys: list[str], curve: np.ndarray, cls: str, months: str = "") -> dict:
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
    rows, cols = rasterio_rowcol(tf, x, y)
    ok = (rows >= 0) & (rows < ens.shape[0]) & (cols >= 0) & (cols < ens.shape[1])
    rows, cols = rows[ok], cols[ok]
    ok = ens[rows, cols] != NODATA
    rows, cols = rows[ok], cols[ok]
    # drop new records near any training point, so the test set is not the fit set
    train = training_points(slug, crs.to_string())
    near = np.zeros(ens.shape, bool)
    tr, tc = rasterio_rowcol(tf, train[:, 0], train[:, 1])
    keep = (tr >= 0) & (tr < ens.shape[0]) & (tc >= 0) & (tc < ens.shape[1])
    near[tr[keep], tc[keep]] = True
    near = ndimage.binary_dilation(near, iterations=EXCLUDE_KM)
    ok = ~near[rows, cols]
    rows, cols = rows[ok], cols[ok]
    n = len(rows)
    if n < 10:
        return {"n_new": n}
    effort = class_effort(cls, months, crs, tf, ens.shape)
    out = {"n_new": n, "outside_mask": int((dist[rows, cols] > 0).sum())}
    for name, e in (("raw", None), ("target_group", effort)):
        per = {}
        for s in STRETCHES:
            total, logs = loglik(ens, dist, curve, rows, cols, s, e)
            per[s] = logs - np.log(total)
        out[name] = score(per)
    return out


if __name__ == "__main__":
    curves = json.loads((OUT / "curves.json").read_text())
    manifest = json.load(urllib.request.urlopen(BUCKET.removeprefix("/vsicurl/") + "/manifest.json"))
    path = OUT / "calibration.json"
    result = json.loads(path.read_text()) if path.exists() else {}
    for slug in sys.argv[1:]:
        keys = manifest[slug]["GBIF.IDs"][0].split("~")
        cls = manifest[slug]["Class"][0]
        months = BREEDING if cls == "Birds" else ""
        result[slug] = r = run(slug, keys, np.array(curves[slug]["curve"]), cls, months)
        print(slug, {k: {i: j for i, j in v.items() if i != "ll"} if isinstance(v, dict) else v
                     for k, v in r.items()}, flush=True)
        path.write_text(json.dumps(result, indent=1))
