#!/usr/bin/env python3
"""Scaling benchmark: build a Canada-wide 300 m single-band raster and serve it,
to measure the per-file ceiling for the openbiodiversity SDM migration.

The per-species SDMs are range-clipped (mostly smaller), but the "10 useful
layers" (land cover, richness, ...) ARE national, so a Canada-wide 300 m raster
is the realistic upper bound for one served file. This answers "is the demo
representative for the 1 TB corpus?" with a measured number rather than an
extrapolation.

EPSG:3978 (Canada Atlas Lambert, metres), 300 m pixels. The synthetic surface is
generated in row strips so peak memory stays bounded; then the standard pipeline
(build_sdm_pmtiles.build) bakes display PMTiles + a uint16 COG.

    python canada_benchmark.py            # -> out/canada300.{pmtiles,_cog.tif}
"""
from __future__ import annotations

import numpy as np
import rasterio
from rasterio.transform import from_origin

import build_sdm_pmtiles as b

# Canada bbox in EPSG:3978 (metres), 300 m grid.
XMIN, YMAX, RES = -2_400_000.0, 4_300_000.0, 300.0
W, H = 18_334, 17_334            # ~5.5 Mm x 5.2 Mm  ->  ~317.8 MP
STRIP = 1_000                     # rows per generation block


def make_canada_300m(dst: str) -> None:
    rng = np.random.default_rng(20260721)
    # A handful of range cores + a smooth continental gradient, in normalised
    # coords, thresholded so most of the country is absence (typical SDM/base).
    cores = [(0.28, 0.42, 0.09, 0.12, 1.0), (0.55, 0.60, 0.14, 0.09, 0.9),
             (0.72, 0.38, 0.08, 0.16, 0.75), (0.40, 0.78, 0.06, 0.07, 0.7),
             (0.83, 0.66, 0.10, 0.11, 0.6)]
    transform = from_origin(XMIN, YMAX, RES, RES)
    profile = dict(driver="GTiff", dtype="float32", count=1, width=W, height=H,
                   crs="EPSG:3978", transform=transform, nodata=None,
                   compress="deflate", tiled=True, blockxsize=512, blockysize=512,
                   BIGTIFF="YES")
    xn = (np.arange(W, dtype=np.float32) + 0.5) / W
    zero_px = 0
    with rasterio.open(dst, "w", **profile) as f:
        for r0 in range(0, H, STRIP):
            r1 = min(r0 + STRIP, H)
            yn = ((np.arange(r0, r1, dtype=np.float32) + 0.5) / H)[:, None]
            field = np.zeros((r1 - r0, W), dtype=np.float32)
            for cx, cy, sx, sy, amp in cores:
                field += amp * np.exp(-(((xn[None, :] - cx) / sx) ** 2
                                        + ((yn - cy) / sy) ** 2))
            field += 0.10 * (0.5 + 0.5 * np.sin(3.0 * xn[None, :]) * np.cos(2.3 * yn))
            field += 0.02 * rng.standard_normal((r1 - r0, W)).astype(np.float32)
            strip = np.clip(field, 0, None)
            strip = np.minimum(strip / 3.0, 1.0)   # fixed divisor -> comparable scale
            strip[strip < 0.18] = 0.0
            zero_px += int((strip == 0).sum())
            f.write(strip, 1, window=rasterio.windows.Window(0, r0, W, r1 - r0))
    import os
    mb = os.path.getsize(dst) / 1e6
    print(f"canada300: {W}x{H} ({W*H/1e6:.0f} MP) EPSG:3978 300m -> {dst} "
          f"({mb:.0f} MB source, {zero_px/(W*H):.0%} zero)")


def main() -> None:
    src = "out/canada300.tif"
    make_canada_300m(src)
    from pathlib import Path
    b.build(Path(src), Path("out"))


if __name__ == "__main__":
    main()
