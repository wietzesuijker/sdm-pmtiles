#!/usr/bin/env python3
"""Generate a small set of synthetic single-band SDMs for the picker MVP.

Real SDMs are range-clipped, so each layer here gets its OWN extent and pattern.
Writes float32 0..1 source rasters into out/ (gitignored); build_sdm_pmtiles.py
then bakes each into data/species/<id>.pmtiles + <id>_cog.tif. Also writes
data/manifest.json — the catalogue the viewer reads. Swapping in real SDMs is a
drop-in: replace the data files + manifest entries, no viewer change.

    python make_species.py            # -> out/<id>.tif + data/manifest.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_bounds

# id, label, group, (west, south, east, north), [(cx,cy,sx,sy,amp), ...]
LAYERS = [
    ("boreal_generalist", "Boreal generalist", "birds", (-100, 44, -55, 62),
     [(0.30, 0.45, 0.16, 0.18, 1.0), (0.62, 0.60, 0.18, 0.14, 0.9), (0.45, 0.30, 0.12, 0.12, 0.7)]),
    ("atlantic_coastal", "Atlantic coastal", "amphibians", (-70, 43, -52, 52),
     [(0.55, 0.45, 0.14, 0.20, 1.0), (0.72, 0.65, 0.10, 0.12, 0.8)]),
    ("prairie_specialist", "Prairie specialist", "mammals", (-114, 49, -96, 57),
     [(0.45, 0.50, 0.18, 0.16, 1.0), (0.65, 0.40, 0.10, 0.10, 0.6)]),
    ("great_lakes", "Great Lakes basin", "plants", (-92, 41, -74, 49),
     [(0.40, 0.45, 0.15, 0.18, 1.0), (0.60, 0.55, 0.12, 0.12, 0.75), (0.25, 0.60, 0.08, 0.09, 0.6)]),
]
SIZE = 2000


def write_layer(lid, bounds, cores, seed) -> None:
    rng = np.random.default_rng(seed)
    w, s, e, n = bounds
    yy, xx = np.mgrid[0:SIZE, 0:SIZE].astype(np.float32)
    xn, yn = xx / SIZE, yy / SIZE
    field = np.zeros((SIZE, SIZE), dtype=np.float32)
    for cx, cy, sx, sy, amp in cores:
        field += amp * np.exp(-(((xn - cx) / sx) ** 2 + ((yn - cy) / sy) ** 2))
    field += 0.08 * (0.5 + 0.5 * np.sin(3.0 * xn) * np.cos(2.3 * yn))
    field += 0.02 * rng.standard_normal((SIZE, SIZE)).astype(np.float32)
    sdm = np.clip(field, 0, None)
    sdm = sdm / sdm.max()
    sdm[sdm < 0.18] = 0.0
    prof = dict(driver="GTiff", dtype="float32", count=1, width=SIZE, height=SIZE,
                crs="EPSG:4326", transform=from_bounds(w, s, e, n, SIZE, SIZE),
                nodata=None, compress="deflate", tiled=True, blockxsize=512, blockysize=512)
    dst = Path("out") / f"{lid}.tif"
    dst.parent.mkdir(exist_ok=True)
    with rasterio.open(dst, "w", **prof) as f:
        f.write(sdm, 1)
    print(f"{lid}: {(sdm==0).mean():.0%} zero -> {dst}")


def main() -> None:
    manifest = []
    for i, (lid, label, group, bounds, cores) in enumerate(LAYERS):
        write_layer(lid, bounds, cores, 20260721 + i)
        manifest.append({
            "id": lid, "label": label, "group": group, "bounds": list(bounds),
            "pmtiles": f"species/{lid}.pmtiles", "cog": f"species/{lid}_cog.tif",
        })
    Path("data").mkdir(exist_ok=True)
    Path("data/manifest.json").write_text(json.dumps({"layers": manifest}, indent=2))
    print(f"manifest: {len(manifest)} layers -> data/manifest.json")


if __name__ == "__main__":
    main()
