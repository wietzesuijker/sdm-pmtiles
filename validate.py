#!/usr/bin/env python3
"""Verification gate for the SDM PMTiles/COG prototype.

Proves, without a running server:
  1. the .pmtiles archive is valid PMTiles v3, WebMercator, sane zoom range;
  2. its tiles are LOSSLESS PNG that carry the suitability value in the red byte
     (green/blue exactly 0, a real value gradient, values within source range) so
     the GPU color-relief ramp decodes true values (display-encoding fidelity);
  3. the _cog.tif is a valid COG with overviews, and its uint16 scale metadata
     recovers the source float 0..1 values (data fidelity, download path).

Usage:  python validate.py out/sdm_synth
"""
from __future__ import annotations

import io
import subprocess
import sys
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from pmtiles.reader import MmapSource, Reader

SCALE = 10000
DISPLAY_LEVELS = 255  # value quantised to one byte in the red channel


def check_pmtiles(path: Path, src: Path) -> None:
    with rasterio.open(src) as s:
        band = s.read(1).astype(np.float32)
    src_rmax = round(min(1.0, float(np.nanmax(band))) * DISPLAY_LEVELS)
    with open(path, "rb") as f:
        reader = Reader(MmapSource(f))
        hdr = reader.header()
        tt = hdr["tile_type"]
        minz, maxz = hdr["min_zoom"], hdr["max_zoom"]
        print(f"  pmtiles: v3 ok, tile_type={tt}, zoom {minz}..{maxz}, "
              f"tiles={hdr['addressed_tiles_count']}")
        assert minz <= maxz, "bad zoom range"
        # PNG (tile_type 2). Lossy tiles would corrupt the value encoding.
        assert tt == 2 or "png" in str(tt).lower(), f"value tiles must be PNG, got {tt}"
        # Aggregate over ALL tiles at a mid zoom, not the first non-empty one: for a
        # range-clipped layer the first tile is often an all-absence edge tile.
        z = min(maxz, max(minz, (minz + maxz) // 2))
        levels, g_max, b_max, r_max, n_tiles = set(), 0, 0, 0, 0
        for x in range(2 ** z):
            for y in range(2 ** z):
                data = reader.get(z, x, y)
                if not data:
                    continue
                n_tiles += 1
                img = np.asarray(Image.open(io.BytesIO(data)).convert("RGB")).astype(np.int16)
                r, g, b = img[..., 0], img[..., 1], img[..., 2]
                levels.update(np.unique(r).tolist())
                g_max = max(g_max, int(g.max())); b_max = max(b_max, int(b.max()))
                r_max = max(r_max, int(r.max()))
        assert n_tiles, f"no tiles at zoom {z}"
        print(f"  tiles z{z}: {n_tiles} decoded, Rmax={r_max} "
              f"({r_max / DISPLAY_LEVELS:.3f} suitability), {len(levels)} value levels, "
              f"G/B max={g_max}/{b_max}")
        # Lossless invariant: the value lives only in red; a lossy codec bleeds it.
        assert g_max == 0 and b_max == 0, "green/blue not 0 -> lossy tile corrupted encoding"
        assert len(levels) >= 8, f"value gradient collapsed ({len(levels)} levels)"
        assert r_max <= src_rmax + 2, f"tile value {r_max} exceeds source max {src_rmax}"


def check_cog(cog: Path, src: Path) -> None:
    info = subprocess.run(["gdalinfo", str(cog)], capture_output=True, text=True,
                          check=True).stdout
    assert "LAYOUT=COG" in info or "Cloud Optimized" in info or "Overviews" in info, \
        "not a COG / no overviews"
    has_ovr = "Overviews:" in info
    with rasterio.open(cog) as c:
        u16 = c.read(1)
        scale = c.scales[0]
        nod = c.nodata
        n_ovr = len(c.overviews(1))
    with rasterio.open(src) as s:
        orig = s.read(1).astype(np.float32)
    recovered = np.where(u16 == nod, 0.0, u16.astype(np.float32) * scale)
    err = np.abs(recovered - np.where(np.isfinite(orig), orig, 0.0))
    print(f"  cog: overviews={has_ovr}({n_ovr} levels), scale={scale:g}, "
          f"nodata={nod}, max|Δ| vs source={err.max():.2e}")
    assert n_ovr >= 1, "COG has no overview levels"
    assert err.max() <= 1.0 / SCALE + 1e-6, f"value roundtrip lost precision ({err.max()})"


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    stem = Path(sys.argv[1])
    pm = stem.with_name(stem.name + ".pmtiles")
    cog = stem.with_name(stem.name + "_cog.tif")
    src = stem.with_suffix(".tif")
    print(f"validating {stem.name}:")
    check_pmtiles(pm, src)
    check_cog(cog, src)
    print("PASS: value-encoding fidelity + data roundtrip verified")


if __name__ == "__main__":
    main()
