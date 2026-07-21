#!/usr/bin/env python3
"""Prototype the openbiodiversity SDM serving migration on a single-band raster.

The ask (Ryan/W call 2026-07-17): serve ~500 species-distribution rasters
(float 0..1 suitability, ~30 m) fast on openbiodiversity.ca straight from object
storage, avoiding a tiling server, and let a user download the raster. PMTiles is
the display candidate to validate "good enough"; single-band only for v1.

This builds, from one single-band SDM raster, the two artifacts that answer the
call's two needs and lets you measure them:

  1. DISPLAY  -> <name>.pmtiles : colormap baked to RGBA WEBP image tiles,
                reprojected to WebMercator. MapLibre reads it natively over HTTP
                range requests via the pmtiles:// protocol -> no tiling server.
                (rio pmtiles needs >=3 bands, so a raster tileset is always a
                baked colormap: it shows suitability, it does not carry values.)

  2. DATA     -> <name>_cog.tif : float 0..1 scaled to uint16 (0..10000, 4 dp
                lossless), Cloud-Optimized GeoTIFF with overviews + DEFLATE.
                This is the downloadable, value-carrying artifact; it also
                range-serves from object storage for desktop/GIS use.

Usage (project venv has rasterio, rio_pmtiles, numpy, matplotlib):
    python build_sdm_pmtiles.py synth out/sdm_synth.tif      # reproducible fixture
    python build_sdm_pmtiles.py build out/sdm_synth.tif out  # -> pmtiles + cog + stats

The `rio` launcher shebang in the shared venv is stale, so rio_pmtiles is driven
through the rasterio CLI group in-process (see run_pmtiles).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import matplotlib
import numpy as np
import rasterio
from rasterio.enums import ColorInterp
from rasterio.transform import from_bounds

SCALE = 10000  # float 0..1 -> uint16 0..10000 (4 decimal places, lossless)
NODATA_U16 = 65535
CMAP = "viridis"  # perceptually uniform; standard for continuous suitability
ZOOM = "0..9"
_LUT = (np.asarray([matplotlib.colormaps[CMAP](i / 255.0) for i in range(256)])[:, :3]
        * 255).round().astype(np.uint8)


def make_synthetic_sdm(dst: Path, width: int = 3000, height: int = 3000) -> None:
    """Write a representative single-band SDM: float32 0..1, EPSG:4326, mostly
    near-zero with a few concentrated high-suitability cores (typical SDM sparsity).
    Seeded -> byte-reproducible."""
    rng = np.random.default_rng(20260721)
    # Eastern-Canada-ish species range extent (lon, lat).
    west, south, east, north = -95.0, 42.0, -55.0, 60.0
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    xn, yn = xx / width, yy / height

    field = np.zeros((height, width), dtype=np.float32)
    for cx, cy, sx, sy, amp in [
        (0.30, 0.35, 0.10, 0.14, 1.0),
        (0.62, 0.55, 0.16, 0.10, 0.85),
        (0.45, 0.75, 0.08, 0.08, 0.7),
        (0.80, 0.25, 0.12, 0.18, 0.6),
    ]:
        field += amp * np.exp(-(((xn - cx) / sx) ** 2 + ((yn - cy) / sy) ** 2))
    # SDM outputs are smooth model surfaces: faint low-frequency habitat term +
    # a little texture, no high-frequency noise.
    field += 0.10 * (0.5 + 0.5 * np.sin(3.0 * xn) * np.cos(2.3 * yn))
    field += 0.02 * rng.standard_normal((height, width)).astype(np.float32)

    sdm = np.clip(field, 0, None)
    sdm = sdm / sdm.max()
    sdm[sdm < 0.18] = 0.0  # threshold to absence -> most of the range unsuitable

    dst.parent.mkdir(parents=True, exist_ok=True)
    profile = dict(
        driver="GTiff", dtype="float32", count=1, width=width, height=height,
        crs="EPSG:4326", transform=from_bounds(west, south, east, north, width, height),
        nodata=None, compress="deflate", tiled=True, blockxsize=512, blockysize=512,
    )
    with rasterio.open(dst, "w", **profile) as f:
        f.write(sdm, 1)
    zero = float((sdm == 0).mean())
    print(f"synth: {width}x{height} float32 SDM -> {dst} "
          f"({dst.stat().st_size / 1e6:.1f} MB, {zero:.0%} zero)")


def run_pmtiles(args: list[str]) -> None:
    """Invoke `rio pmtiles ...` in-process (stale venv launcher shebang)."""
    from rasterio.rio.main import main_group  # noqa: PLC0415
    try:
        main_group.main(["pmtiles", *args], standalone_mode=False)
    except SystemExit as exc:  # click may raise SystemExit(0)
        if exc.code not in (0, None):
            raise


def _read_single_band(path: Path):
    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32)
        nodata = src.nodata
        prof = src.profile
    valid = np.isfinite(data)
    if nodata is not None:
        valid &= data != nodata
    data = np.where(valid, data, 0.0)
    return data, valid, prof


def build_display_pmtiles(src: Path, rgba_tmp: Path, out: Path) -> dict:
    """Bake CMAP into RGBA (alpha=0 where suitability==0), then rio pmtiles."""
    data, valid, prof = _read_single_band(src)
    vmax = float(data.max())
    # SDMs are probabilities in [0,1]; rescale on a FIXED 0..1 so colours are
    # absolute (comparable across species) and match the viewer's 0..1 legend.
    # Per-layer max-stretch would make the legend lie for a raster whose max < 1.
    norm = np.clip(data, 0, 1)
    rgb = _LUT[(norm * 255).round().astype(np.uint8)]
    visible = valid & (data > 0)
    alpha = np.where(visible, 255, 0).astype(np.uint8)

    prof.update(count=4, dtype="uint8", nodata=None, compress="deflate",
                tiled=True, blockxsize=512, blockysize=512)
    prof.pop("photometric", None)
    with rasterio.open(rgba_tmp, "w", **prof) as dst:
        for i in range(3):
            dst.write(rgb[..., i], i + 1)
        dst.write(alpha, 4)
        dst.colorinterp = [ColorInterp.red, ColorInterp.green,
                           ColorInterp.blue, ColorInterp.alpha]

    run_pmtiles([str(rgba_tmp), str(out), "--rgba", "--zoom-levels", ZOOM,
                 "--format", "WEBP", "--resampling", "cubic"])
    rgba_tmp.unlink(missing_ok=True)
    return {"vmax": vmax, "visible_px": int(visible.sum()),
            "size_mb": out.stat().st_size / 1e6}


def build_data_cog(src: Path, u16_tmp: Path, out: Path) -> dict:
    """float 0..1 -> uint16 scaled COG with overviews (downloadable, value-carrying)."""
    data, valid, prof = _read_single_band(src)
    scaled = np.where(valid, np.clip(data, 0, 6.5534) * SCALE, NODATA_U16)
    scaled = scaled.round().astype(np.uint16)
    prof.update(dtype="uint16", count=1, nodata=NODATA_U16, compress="deflate")
    with rasterio.open(u16_tmp, "w", **prof) as dst:
        dst.write(scaled, 1)
        dst.update_tags(1, SCALE=1.0 / SCALE, OFFSET=0.0)
        dst.scales = (1.0 / SCALE,)

    subprocess.run(
        ["gdal_translate", "-q", "-of", "COG", str(u16_tmp), str(out),
         "-co", "COMPRESS=DEFLATE", "-co", "PREDICTOR=2",
         "-co", "OVERVIEWS=IGNORE_EXISTING", "-co", "BLOCKSIZE=512"],
        check=True,
    )
    u16_tmp.unlink(missing_ok=True)
    return {"size_mb": out.stat().st_size / 1e6}


def build(src: Path, outdir: Path) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    name = src.stem
    disp = build_display_pmtiles(src, outdir / f"_{name}_rgba.tif",
                                 outdir / f"{name}.pmtiles")
    cog = build_data_cog(src, outdir / f"_{name}_u16.tif", outdir / f"{name}_cog.tif")
    stats = {
        "source": str(src), "source_mb": src.stat().st_size / 1e6,
        "display_pmtiles": disp, "data_cog": cog,
    }
    (outdir / f"{name}_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    return stats


def main() -> None:
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    cmd = sys.argv[1]
    if cmd == "synth":
        make_synthetic_sdm(Path(sys.argv[2]))
    elif cmd == "build":
        build(Path(sys.argv[2]), Path(sys.argv[3] if len(sys.argv) > 3 else "out"))
    else:
        sys.exit(f"unknown command: {cmd}")


if __name__ == "__main__":
    main()
