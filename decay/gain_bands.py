#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scipy", "rasterio"]
# ///
"""Where projected change happens: 2071-2100 minus 1980-2010 suitability, summed by distance band to the evidence mask.

Tests the paper's inference that projected gains lie far from known occurrence and losses near it.
Random sample of species, resumable JSON lines.

    decay/gain_bands.py decay/out/gain_bands.jsonl [n] [jobs]
"""
from __future__ import annotations

import json
import random
import sys
import urllib.request
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from evidence_decay import BUCKET, NODATA, distance_km

BANDS = [0, 1, 10, 50, 200, 1000, 10_000]  # km; [0, 1) is inside the mask


def one(slug: str) -> dict:
    import rasterio

    def read(product, period):
        with rasterio.open(f"{BUCKET}/{slug}_{product}_1km_na_{period}_sett1.tif") as src:
            return src.read(1)

    try:
        now, fut = read("sdm-ensemble", "1980-2010"), read("sdm-ensemble", "2071-2100")
        dist = distance_km(read("sdm-clip", "1980-2010"))
        ok = (now != NODATA) & (fut != NODATA)
        diff, band = fut[ok].astype(np.int64) - now[ok], np.digitize(dist[ok], BANDS) - 1
        return {"slug": slug, "ok": True, "now": int(now[ok].sum(dtype=np.int64)),
                "gain": np.bincount(band, np.maximum(diff, 0), len(BANDS) - 1).astype(int).tolist(),
                "loss": np.bincount(band, np.maximum(-diff, 0), len(BANDS) - 1).astype(int).tolist()}
    except (OSError, ValueError, KeyError) as e:  # one bad species must not stop the batch
        return {"slug": slug, "ok": False, "error": f"{type(e).__name__}: {e}"[:300]}


if __name__ == "__main__":
    log, n, jobs = Path(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 300, int(sys.argv[3]) if len(sys.argv) > 3 else 4
    manifest = json.load(urllib.request.urlopen(BUCKET.removeprefix("/vsicurl/") + "/manifest.json"))
    done = {json.loads(line)["slug"] for line in log.read_text().splitlines()} if log.exists() else set()
    todo = [s for s in random.Random(0).sample(sorted(manifest), n) if s not in done]
    with ProcessPoolExecutor(jobs) as pool, log.open("a") as out:
        for fut in as_completed([pool.submit(one, s) for s in todo]):
            out.write(json.dumps(fut.result()) + "\n")
            out.flush()
