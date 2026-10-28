#!/usr/bin/env bash
# Builds the Web Mercator display COG for each SDM the viewer shows (current + 2071-2100).
# Usage: ./make_display_cogs.sh OUT_DIR slug [slug ...]
# ZOOM_LEVEL=7 gives ~1.2 km pixels, the closest web zoom to the 1 km source. Upload OUT_DIR with `swift upload`.
set -euo pipefail
SRC=/vsicurl/https://object-arbutus.alliancecan.ca/swift/v1/4903c6264e0c4191a74a54c93b6fe1d9/def-pollock-eckert/vertebrates_NA/ECCC_2026
OUT=${1:?usage: $0 OUT_DIR slug [slug ...]}; shift
[ $# -gt 0 ] || { echo "usage: $0 OUT_DIR slug [slug ...]" >&2; exit 2; }
mkdir -p "$OUT"
for slug in "$@"; do
  for f in "${slug}_sdm-ensemble-clipped_1km_na_1980-2010_sett1" "${slug}_sdm-ensemble_1km_na_2071-2100_sett1"; do
    gdalwarp -q -overwrite -t_srs EPSG:3857 -r near -ot Byte -srcnodata 255 -dstnodata 255 \
      -of COG -co COMPRESS=DEFLATE -co TILING_SCHEME=GoogleMapsCompatible -co ZOOM_LEVEL=7 \
      -co OVERVIEW_RESAMPLING=NEAREST "$SRC/$f.tif" "$OUT/${f}_3857.tif"
    echo "$OUT/${f}_3857.tif"
  done
done
