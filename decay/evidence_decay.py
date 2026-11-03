#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scipy", "rasterio"]
# ///
"""Recover the evidence clip's decay curve from the published SDM files.

The bucket ships, per species, the raw ensemble, a binary evidence mask
(`sdm-clip`) and the clipped ensemble. The clipped layer is
ensemble * g(km to the nearest mask cell), with one monotone curve g per
species. This fits g as a 1 km lookup table, checks it rebuilds the clipped
layer, and writes what a browser needs to recompute the clip with a different
decay: the curve (curves.json) and a two-band raster (<slug>_decay.tif:
ensemble, km to the mask), plus the future ensembles (<slug>_ens_<period>.tif).

    decay/evidence_decay.py ursus-arctos gulo-gulo   # -> decay/out/
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

BUCKET = ("/vsicurl/https://object-arbutus.alliancecan.ca/swift/v1/"
          "4903c6264e0c4191a74a54c93b6fe1d9/def-pollock-eckert/vertebrates_NA/ECCC_2026")
NODATA = 255
MAX_KM = 5000
MIN_FIT = 5  # ratios from ensemble values below this are too quantised to fit
FUTURE = ["2071-2100"]  # the bucket ships these unclipped; the page clips them with the same curve


def distance_km(clip: np.ndarray) -> np.ndarray:
    """Km (1 km cells) from each cell to the nearest evidence-mask cell."""
    return np.minimum(ndimage.distance_transform_edt(clip != 1), MAX_KM).astype(np.uint16)


def fit_curve(ens: np.ndarray, clip: np.ndarray, clipped: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """Median clipped/ensemble ratio per km outside the mask, made monotone."""
    m = (ens != NODATA) & (clipped != NODATA) & (clip == 0) & (ens >= MIN_FIT)
    d, ratio = dist[m], clipped[m] / ens[m]
    order = np.argsort(d, kind="stable")
    d, ratio = d[order], ratio[order]
    km, start = np.unique(d, return_index=True)
    curve = np.full(MAX_KM + 1, np.nan)
    curve[0] = 1.0
    for s, e, k in zip(start, np.append(start[1:], len(d)), km):
        curve[k] = np.median(ratio[s:e])
    # km with no samples take the value of the nearest fitted km below
    filled = np.where(np.isnan(curve), 0, np.arange(len(curve)))
    curve = curve[np.maximum.accumulate(filled)]
    return np.minimum.accumulate(curve)


def apply_curve(ens: np.ndarray, clip: np.ndarray, dist: np.ndarray, curve: np.ndarray,
                scale: float = 1.0) -> np.ndarray:
    """Clipped layer for a decay stretched by `scale` (1 = published, 0 = hard clip).

    Rounds half up, like Math.round in index.html, so both give the same pixels.
    """
    if scale <= 0:
        g = (clip == 1).astype(float)
    else:
        g = curve[np.minimum(np.floor(dist / scale + 0.5), MAX_KM).astype(int)]
    out = np.where(clip == 1, ens, np.floor(ens * g + 0.5)).astype(np.uint8)
    return np.where(ens == NODATA, NODATA, out).astype(np.uint8)


def km_below(curve: np.ndarray, level: float) -> int | None:
    """First km where the curve drops below `level`, None if it never does."""
    hit = np.flatnonzero(curve < level)
    return int(hit[0]) if hit.size else None


def run(slug: str, out: Path) -> dict:
    import rasterio

    def read(product: str, period: str = "1980-2010"):
        src = rasterio.open(f"{BUCKET}/{slug}_{product}_1km_na_{period}_sett1.tif")
        return src.read(1), src.profile

    ens, profile = read("sdm-ensemble")
    clip, _ = read("sdm-clip")
    clipped, _ = read("sdm-ensemble-clipped")
    dist = distance_km(clip)
    curve = fit_curve(ens, clip, clipped, dist)
    valid = (ens != NODATA) & (clipped != NODATA)
    outside = valid & (clip == 0) & (ens >= MIN_FIT)
    err = np.abs(apply_curve(ens, clip, dist, curve).astype(int) - clipped.astype(int))
    profile.update(driver="COG", dtype="uint16", count=2, nodata=None, compress="DEFLATE",
                   predictor=2, interleave="pixel", overview_resampling="nearest")
    with rasterio.open(out / f"{slug}_decay.tif", "w", **profile) as dst:
        dst.write(np.stack([ens.astype(np.uint16), dist]))
    for period in FUTURE:
        fut, _ = read("sdm-ensemble", period)
        with rasterio.open(out / f"{slug}_ens_{period}.tif", "w", **{**profile, "count": 1}) as dst:
            dst.write(fut.astype(np.uint16), 1)
    return {
        "d50_km": km_below(curve, 0.5),
        "d10_km": km_below(curve, 0.1),
        "rebuild_within_1": round(float(np.mean(err[valid] <= 1)), 5),
        "rebuild_within_1_outside": round(float(np.mean(err[outside] <= 1)), 5),
        "max_err": int(err[valid].max()),
        "curve": np.round(curve, 4).tolist(),
    }


if __name__ == "__main__":
    out = Path(__file__).parent / "out"
    out.mkdir(exist_ok=True)
    path = out / "curves.json"
    curves = json.loads(path.read_text()) if path.exists() else {}
    for slug in sys.argv[1:]:
        curves[slug] = run(slug, out)
        r = curves[slug]
        print(f"{slug}: d50={r['d50_km']} km d10={r['d10_km']} km "
              f"rebuild<=1 all={r['rebuild_within_1']} outside={r['rebuild_within_1_outside']} max={r['max_err']}")
        path.write_text(json.dumps(curves))
