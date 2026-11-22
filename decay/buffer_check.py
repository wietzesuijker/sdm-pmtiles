#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scipy", "rasterio", "mapbox-vector-tile"]
# ///
"""Score a hard dispersal buffer, g = 1 for d <= D else 0 (Bateman et al. 2013 partial dispersal), next to exp and halfnormal.

Random sample of scored species from calibration_all.jsonl, same records, effort and likelihood as calibrate_batch.py.

    decay/buffer_check.py [n] [jobs]
"""
import json
import multiprocessing
import random
import sys
import urllib.request
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import calibrate_batch as cb

cb.SHAPES["buffer"] = lambda v: (cb.KM <= v).astype(float)
D = Path(cb.__file__).parent / "out"
n, jobs = (int(a) for a in (sys.argv[1:] + ["120", "4"])[:2])
rows = [json.loads(line) for line in (D / "calibration_all.jsonl").read_text().splitlines()]
scored = sorted((r["slug"], r["class"]) for r in rows if "winner" in r and r["outside_mask"] >= 3)
log = D / "buffer_check.jsonl"
done = {json.loads(line)["slug"] for line in log.read_text().splitlines()} if log.exists() else set()
m = json.load(urllib.request.urlopen(cb.BUCKET.removeprefix("/vsicurl/") + "/manifest.json"))
with ProcessPoolExecutor(jobs, mp_context=multiprocessing.get_context("fork")) as pool, log.open("a") as out:
    futs = [pool.submit(cb.one, s, m[s]["GBIF.IDs"][0].split("~"), c)
            for s, c in random.Random(2).sample(scored, n) if s not in done]
    for f in as_completed(futs):
        out.write(json.dumps(f.result()) + "\n")
        out.flush()
