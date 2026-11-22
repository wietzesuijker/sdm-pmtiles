#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scipy", "rasterio", "mapbox-vector-tile"]
# ///
"""Refit the half-normal sigma under other likelihood settings, on the same 120 species as buffer_check.py.

Settings: lambda floor EPS 0.5 (baseline), 0.1 and 2.0; no effort term; a 50 km effort grid.
Every setting reruns in one worker per species, so all see the same records. Failed species are
logged with ok false and retried on the next run. Prints the summary when all are done.

    decay/sensitivity.py [n] [jobs]
"""
import json
import multiprocessing
import random
import statistics as st
import sys
import urllib.request
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

import calibrate
import calibrate_batch as cb
from pool import change_at

SETTINGS = {"base": (0.5, 25, True), "eps0.1": (0.1, 25, True), "eps2": (2.0, 25, True),
            "no_effort": (0.5, 25, False), "effort50": (0.5, 50, True)}
real_effort = cb.class_effort
caches: dict = {s: {} for s in SETTINGS}


def run(slug: str, keys: list[str], cls: str) -> dict:
    out = {"slug": slug, "class": cls, "ok": True, "sigma": {}}
    for name, (eps, k, effort) in SETTINGS.items():
        cb.EPS, cb.EFFORT_KM, calibrate.EFFORT_KM, cb._effort = eps, k, k, caches[name]
        cb.class_effort = real_effort if effort else lambda c, m, crs, tf, shape, k=k: np.ones((-(-shape[0] // k), -(-shape[1] // k)))
        r = cb.one(slug, keys, cls)
        if not r["ok"] or "halfnormal" not in r:
            return {"slug": slug, "ok": False, "error": f"{name}: {r.get('error', 'not scored')}"}
        out["sigma"][name], out["change"] = r["halfnormal"]["fit"], r["change"]["halfnormal"]
    return out


def summary(rows: list[dict]) -> None:
    base = {r["slug"]: r["sigma"]["base"] for r in rows}
    for name in SETTINGS:
        ratio = st.median(r["sigma"][name] / base[r["slug"]] for r in rows)
        by = {c: round(st.median(r["sigma"][name] for r in rows if r["class"] == c), 1)
              for c in sorted({r["class"] for r in rows})}
        flips = sum((change_at(r["change"], np.log(r["sigma"][name])) > 1)
                    != (change_at(r["change"], np.log(base[r["slug"]])) > 1) for r in rows)
        print(f"{name}: n={len(rows)} median sigma ratio {ratio:.3f}, median sigma {by}, change flips {flips}")


if __name__ == "__main__":
    D = Path(cb.__file__).parent / "out"
    n, jobs = (int(a) for a in (sys.argv[1:] + ["120", "6"])[:2])
    rows = [json.loads(line) for line in (D / "calibration_all.jsonl").read_text().splitlines()]
    scored = sorted((r["slug"], r["class"]) for r in rows if "winner" in r and r["outside_mask"] >= 3)
    sample = {s for s, _ in random.Random(2).sample(scored, n)}
    log = D / "sensitivity.jsonl"
    done = [r for r in map(json.loads, log.read_text().splitlines()) if r["ok"]] if log.exists() else []
    seen = {r["slug"] for r in done}
    m = json.load(urllib.request.urlopen(cb.BUCKET.removeprefix("/vsicurl/") + "/manifest.json"))
    with ProcessPoolExecutor(jobs, mp_context=multiprocessing.get_context("fork")) as pool, log.open("a") as out:
        futs = [pool.submit(run, s, m[s]["GBIF.IDs"][0].split("~"), c)
                for s, c in random.Random(2).sample(scored, n) if s not in seen]
        for f in as_completed(futs):
            r = f.result()
            out.write(json.dumps(r) + "\n")
            out.flush()
            done += [r] if r["ok"] else []
            print(r["slug"], "ok" if r["ok"] else r["error"], flush=True)
    summary([r for r in done if r["slug"] in sample])
