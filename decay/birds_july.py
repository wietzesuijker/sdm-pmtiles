#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scipy", "rasterio", "mapbox-vector-tile"]
# ///
"""Rescore a random sample of scored birds with July-only records, to test whether June migrants widen the bird scale.

Result (98 scorable of 120): median July / June-July half-normal sigma 1.0, so they do not.

    decay/birds_july.py
"""
import json
import multiprocessing
import random
import urllib.request
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import calibrate_batch as cb

cb.BREEDING = "&month=7"
D = Path(cb.__file__).parent / "out"
rows = [json.loads(l) for l in (D / "calibration_all.jsonl").read_text().splitlines()]
birds = sorted(r["slug"] for r in rows if r.get("class") == "Birds" and "winner" in r)
sample = random.Random(1).sample(birds, 120)
log = D / "birds_july.jsonl"
done = {json.loads(l)["slug"] for l in log.read_text().splitlines()} if log.exists() else set()
m = json.load(urllib.request.urlopen(cb.BUCKET.removeprefix("/vsicurl/") + "/manifest.json"))
with ProcessPoolExecutor(2, mp_context=multiprocessing.get_context("fork")) as pool, log.open("a") as out:
    futs = [pool.submit(cb.one, s, m[s]["GBIF.IDs"][0].split("~"), "Birds") for s in sample if s not in done]
    for f in as_completed(futs):
        out.write(json.dumps(f.result()) + "\n")
        out.flush()
