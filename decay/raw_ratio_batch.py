#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scipy", "rasterio"]
# ///
"""Total raw 2071-2100 and unclipped 1980-2010 suitability over the published clipped 1980-2010 total, per species.

Pairs with clip_future_batch.py's future_ratio (same clip on both periods) to count how many species
flip from gain to loss once the clip is applied to 2100 too. Resumable JSON lines, like the clip batch.

    decay/raw_ratio_batch.py decay/out/raw_ratio.jsonl [jobs]
"""
from __future__ import annotations

import json
import sys
import urllib.request
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from evidence_decay import BUCKET, NODATA


def total(slug: str, product: str, period: str) -> int:
    import rasterio
    with rasterio.open(f"{BUCKET}/{slug}_{product}_1km_na_{period}_sett1.tif") as src:
        a = src.read(1)
    return int(a[a != NODATA].sum(dtype=np.int64))


def one(slug: str) -> dict:
    try:
        c = total(slug, "sdm-ensemble-clipped", "1980-2010")
        return {"slug": slug, "ok": True, "clipped_now": c,
                "raw_future_ratio": round(total(slug, "sdm-ensemble", "2071-2100") / c, 4),
                "raw_now_ratio": round(total(slug, "sdm-ensemble", "1980-2010") / c, 4)}
    except Exception as e:  # one bad species must not stop the batch
        return {"slug": slug, "ok": False, "error": f"{type(e).__name__}: {e}"[:300]}


if __name__ == "__main__":
    log, jobs = Path(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 3
    slugs = sorted(json.load(urllib.request.urlopen(BUCKET.removeprefix("/vsicurl/") + "/manifest.json")))
    done = {json.loads(line)["slug"] for line in log.read_text().splitlines()} if log.exists() else set()
    todo = [s for s in slugs if s not in done]
    print(f"{len(todo)} of {len(slugs)} to do", flush=True)
    with ProcessPoolExecutor(jobs) as pool, log.open("a") as out:
        for fut in as_completed([pool.submit(one, s) for s in todo]):
            out.write(json.dumps(fut.result()) + "\n")
            out.flush()
