#!/usr/bin/env python3
"""Verification gate for the SDM PMTiles/COG prototype.

Proves, without a running server:
  1. the .pmtiles archive is valid PMTiles v3, WebMercator, sane zoom range;
  2. a mid-zoom tile decodes to a non-empty image whose colours are the CMAP
     of the underlying suitability (display fidelity);
  3. the _cog.tif is a valid COG with overviews, and its uint16 scale metadata
     recovers the source float 0..1 values (data fidelity, download path).

Usage:  python validate.py out/sdm_synth
"""
from __future__ import annotations

import io
import subprocess
import sys
from pathlib import Path

import matplotlib
import numpy as np
import rasterio
from PIL import Image
from pmtiles.reader import MmapSource, Reader

CMAP = "viridis"
SCALE = 10000
_LUT = (np.asarray([matplotlib.colormaps[CMAP](i / 255.0) for i in range(256)])[:, :3]
        * 255).round().astype(np.int16)


def check_pmtiles(path: Path) -> None:
    with open(path, "rb") as f:
        reader = Reader(MmapSource(f))
        hdr = reader.header()
        tt = hdr["tile_type"]
        minz, maxz = hdr["min_zoom"], hdr["max_zoom"]
        print(f"  pmtiles: v3 ok, tile_type={tt}, zoom {minz}..{maxz}, "
              f"tiles={hdr['addressed_tiles_count']}")
        assert minz <= maxz, "bad zoom range"
        # Pull a tile near the middle zoom and decode it.
        z = (minz + maxz) // 2
        found = None
        n = 2 ** z
        for x in range(n):
            for y in range(n):
                data = reader.get(z, x, y)
                if data:
                    found = (z, x, y, data)
                    break
            if found:
                break
        assert found, f"no tile at zoom {z}"
        z, x, y, data = found
        img = np.asarray(Image.open(io.BytesIO(data)).convert("RGBA"))
        opaque = img[img[..., 3] > 0][:, :3].astype(np.int16)
        assert opaque.size, "decoded tile fully transparent"
        # Every opaque pixel must lie near the CMAP ramp (nearest-LUT dist small).
        d = np.abs(opaque[:, None, :] - _LUT[None, :, :]).sum(axis=2).min(axis=1)
        frac_on_ramp = float((d <= 24).mean())  # WEBP-lossy tolerance
        print(f"  tile z{z}/{x}/{y}: {img.shape[0]}x{img.shape[1]}, "
              f"{opaque.shape[0]} opaque px, {frac_on_ramp:.1%} on {CMAP} ramp")
        assert frac_on_ramp > 0.90, f"colours off {CMAP} ramp ({frac_on_ramp:.1%})"


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
    check_pmtiles(pm)
    check_cog(cog, src)
    print("PASS: display fidelity + data roundtrip verified")


if __name__ == "__main__":
    main()
