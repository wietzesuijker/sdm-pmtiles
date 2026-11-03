"""Export what the full-resolution page needs beside the bucket COGs.

For each species in out/curves.json: a distance-to-evidence COG on the bucket grid
(the page multiplies the bucket ensemble by curve(distance) on the GPU), and the
exact 1 km total suitability for every slider stretch, per period.

    uv run --with rasterio --with numpy --with scipy python decay/publish.py decay/out decay/data
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import rasterio

from evidence_decay import MAX_KM, NODATA

STEP = 0.0125  # slider step in index.html
SCALES = np.round(np.arange(0, 3 + STEP / 2, STEP), 4)


def totals(ens: np.ndarray, dist: np.ndarray, curve: np.ndarray, scales=SCALES) -> list[int]:
    """Sum of apply_curve(ens, ..., scale) for each scale, from an (ensemble, distance) histogram."""
    ok = ens != NODATA
    e, d = ens[ok].astype(np.int64), np.minimum(dist[ok], MAX_KM).astype(np.int64)
    hist = np.bincount(e * (MAX_KM + 1) + d, minlength=256 * (MAX_KM + 1)).reshape(256, MAX_KM + 1)
    ev, dv = np.nonzero(hist)
    n = hist[ev, dv]
    out = []
    for s in scales:
        if s <= 0:
            g = (dv == 0).astype(float)
        else:
            g = curve[np.minimum(np.floor(dv / s + 0.5), len(curve) - 1).astype(int)]
            g[dv == 0] = 1.0  # inside the evidence mask the ensemble is kept as is
        out.append(int((n * np.floor(ev * g + 0.5)).sum()))
    return out


def extent(ens: np.ndarray, dist: np.ndarray, transform) -> list[float]:
    """Map bbox [minx, miny, maxx, maxy] of the evidence mask, so the page opens on the species."""
    keep = (ens != NODATA) & (dist == 0)
    rows, cols = np.nonzero(keep.any(1))[0], np.nonzero(keep.any(0))[0]
    x0, y0 = transform * (cols[0], rows[-1] + 1)
    x1, y1 = transform * (cols[-1] + 1, rows[0])
    return [x0, y0, x1, y1]


def main(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    curves = json.loads((src / "curves.json").read_text())
    table = {"scales": SCALES.tolist()}
    for slug, c in curves.items():
        curve = np.asarray(c["curve"])
        with rasterio.open(src / f"{slug}_decay.tif") as f:
            now, dist = f.read(1), f.read(2)
            profile = f.profile
        for k in ("blockxsize", "blockysize", "tiled", "interleave"):
            profile.pop(k, None)
        # same 256 px tiles and overview levels as the bucket COGs, so the page can stack them
        profile.update(driver="COG", count=1, nodata=None, compress="DEFLATE", predictor=2, blocksize=256,
                       overview_resampling="nearest")
        with rasterio.open(dst / f"{slug}_dist.tif", "w", **profile) as out:
            out.write(dist, 1)
        row = {"1980-2010": totals(now, dist, curve), "1980-2010 raw": int(now[now != NODATA].sum()),
               "extent": extent(now, dist, profile["transform"])}
        fut = src / f"{slug}_ens_2071-2100.tif"
        if fut.exists():
            with rasterio.open(fut) as f:
                ens = f.read(1)
            row["2071-2100"] = totals(ens, dist, curve)
            row["2071-2100 raw"] = int(ens[ens != NODATA].sum())
        table[slug] = row
        print(slug, flush=True)
    (dst / "totals.json").write_text(json.dumps(table))
    for name in ("curves.json", "calibration.json"):
        if (src / name).exists():
            shutil.copy(src / name, dst / name)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
