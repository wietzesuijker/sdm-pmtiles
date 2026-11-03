#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scipy", "rasterio"]
# ///
"""Fit the evidence decay curve for every species in the bucket manifest.

Reads the full 1 km ensemble, mask and clipped layers (the overviews average
the binary mask, which moved d50 by up to 2.5x in a check), fits the curve
with evidence_decay.fit_curve and appends one row per species to
decay/out/triage.csv. Rerunning skips species already in the file.

    decay/sweep.py [workers]   # default 4, each needs ~2 GB
"""
from __future__ import annotations

import csv
import json
import os
import sys
import urllib.request
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from evidence_decay import BUCKET, NODATA, apply_curve, distance_km, fit_curve, km_below

os.environ.update(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_MAX_RETRY="5",
                  GDAL_HTTP_RETRY_DELAY="2", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif")
FIELDS = ["slug", "class", "common_name", "iucn_status", "n_occ", "d50_km", "d10_km",
          "rebuild_within_1_outside", "share_beyond_500km", "error"]
OUT = Path(__file__).parent / "out" / "triage.csv"


def fit(slug: str) -> dict:
    import rasterio

    def read(product: str) -> np.ndarray:
        url = f"{BUCKET}/{slug}_{product}_1km_na_1980-2010_sett1.tif"
        with rasterio.open(url) as src:
            return src.read(1)

    ens, clip, clipped = read("sdm-ensemble"), read("sdm-clip"), read("sdm-ensemble-clipped")
    dist = distance_km(clip)
    curve = fit_curve(ens, clip, clipped, dist)
    outside = (ens != NODATA) & (clipped != NODATA) & (clip == 0) & (ens >= 5)
    err = np.abs(apply_curve(ens, clip, dist, curve).astype(int) - clipped.astype(int))
    shown = np.where(clipped == NODATA, 0, clipped).astype(float)
    return {
        "d50_km": km_below(curve, 0.5), "d10_km": km_below(curve, 0.1),
        "rebuild_within_1_outside": round(float(np.mean(err[outside] <= 1)), 4) if outside.any() else "",
        # share of the displayed suitability that sits > 500 km from any evidence
        "share_beyond_500km": round(float(shown[dist > 500].sum() / max(shown.sum(), 1)), 4),
    }


def row(item: tuple[str, dict]) -> dict:
    slug, meta = item
    r = {"slug": slug, "class": meta["Class"][0], "common_name": meta["IUCN.Common.Name"][0],
         "iucn_status": meta["IUCN.Status"][0], "n_occ": meta["n.occ"][0]}
    try:
        r.update(fit(slug))
    except Exception as e:  # keep sweeping; failed species are retried on the next run
        r["error"] = str(e)[:200]
    return r


if __name__ == "__main__":
    url = BUCKET.removeprefix("/vsicurl/") + "/manifest.json"
    manifest = json.load(urllib.request.urlopen(url))
    done = set()
    if OUT.exists():
        with OUT.open() as f:
            rows = [r for r in csv.DictReader(f) if not r["error"]]
        done = {r["slug"] for r in rows}
        with OUT.open("w", newline="") as f:  # drop failed rows so they rerun
            w = csv.DictWriter(f, FIELDS); w.writeheader(); w.writerows(rows)
    else:
        OUT.parent.mkdir(exist_ok=True)
        with OUT.open("w", newline="") as f:
            csv.DictWriter(f, FIELDS).writeheader()
    todo = [(s, m) for s, m in manifest.items() if s not in done]
    print(f"{len(todo)} to fit, {len(done)} done", flush=True)
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 4) as pool, OUT.open("a", newline="") as f:
        w = csv.DictWriter(f, FIELDS)
        for i, r in enumerate(pool.imap_unordered(row, todo), 1):
            w.writerow(r); f.flush()
            if i % 50 == 0 or r.get("error"):
                print(i, r["slug"], r.get("error") or r["d10_km"], flush=True)
