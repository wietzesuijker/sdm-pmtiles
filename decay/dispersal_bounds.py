#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scipy", "rasterio", "mapbox-vector-tile"]
# ///
"""Place the calibrated decay between the classic dispersal scenarios (Thuiller 2005, Bateman et al. 2013).

On the same 120 species as buffer_check.py, with calibrate_batch.py's records, effort and floor, scores:
none (g = 0 outside the evidence mask, the floor applies), full (g = 1, no clip), the species' fitted
half-normal and its pooled half-normal scale from decay_pooled.json. Also stores the 2071-2100 / 1980-2010
change with no clip and with a hard clip. Failed species are logged with ok false and retried on the next run.

    decay/dispersal_bounds.py [n] [jobs]
"""
import json
import multiprocessing
import random
import sys
import urllib.request
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import calibrate_batch as cb
import numpy as np
from pool import change_at

D = Path(cb.__file__).parent / "out"
POOLED = json.loads((D / "decay_pooled.json").read_text())["species"]
SCEN = {"none": lambda: (cb.KM == 0).astype(float), "full": lambda: np.ones(cb.KM.size)}
real_tables, real_change = cb.tables, cb.change


def run(slug: str, keys: list[str], cls: str) -> dict:
    scale = POOLED.get(slug, {}).get("scale_km")
    extra = {**SCEN, **({"pooled": lambda: np.exp(-(cb.KM / scale) ** 2 / 2)} if scale else {})}
    cb.tables = lambda curve: {**real_tables(curve), **{(f, 0): g() for f, g in extra.items()}}
    cb.change = lambda ens, fut, dist: {**real_change(ens, fut, dist), **{
        f: _ratio(ens, fut, dist, SCEN[f]()) for f in SCEN}}
    r = cb.one(slug, keys, cls)
    if not r["ok"] or "halfnormal" not in r:
        return {"slug": slug, "ok": False, "error": r.get("error", "not scored")}
    n, hn = r["n_new"], r["halfnormal"]
    ll = {"halfnormal": n * hn["ll_fit"], **{f: n * r[f]["ll"] for f in extra}}
    ch = {"none": r["change"]["none"], "full": r["change"]["full"],
          "halfnormal": round(change_at(r["change"]["halfnormal"], np.log(hn["fit"])), 4)}
    return {"slug": slug, "class": cls, "ok": True, "n_new": n, "sigma": hn["fit"], "pooled_km": scale,
            "ll": {k: round(v, 3) for k, v in ll.items()}, "change": ch}


def _ratio(ens, fut, dist, g) -> float:
    ok = (ens != cb.NODATA) & (fut != cb.NODATA)
    w = g[dist[ok]]
    return round(float((fut[ok] * w).sum() / max((ens[ok].astype(float) * w).sum(), 1)), 4)


def summary(rows: list[dict]) -> None:
    rng = np.random.default_rng(0)
    hn = np.array([r["ll"]["halfnormal"] for r in rows])
    print(f"n={len(rows)} halfnormal sum {hn.sum():.1f}")
    for f in ("none", "full", "pooled"):
        sub = [i for i, r in enumerate(rows) if f in r["ll"]]
        d = np.array([rows[i]["ll"][f] for i in sub]) - hn[sub]
        b = d[rng.integers(0, len(d), (2000, len(d)))].sum(1)
        print(f"{f}: n={len(sub)} sum {sum(rows[i]['ll'][f] for i in sub):.1f}, diff vs hn {d.sum():.1f} "
              f"CI90 [{np.percentile(b, 5):.1f}, {np.percentile(b, 95):.1f}], beats hn {(d > 0).mean():.3f}")
    for f in ("none", "full", "halfnormal"):
        print(f"gaining {f}: {np.mean([r['change'][f] > 1 for r in rows]):.3f}")


assert _ratio(np.array([[1, 1]]), np.array([[2, 4]]), np.array([[0, 1]]), np.array([1.0, 0.0])) == 2.0

if __name__ == "__main__":
    n, jobs = (int(a) for a in (sys.argv[1:] + ["120", "4"])[:2])
    rows = [json.loads(line) for line in (D / "calibration_all.jsonl").read_text().splitlines()]
    scored = sorted((r["slug"], r["class"]) for r in rows if "winner" in r and r["outside_mask"] >= 3)
    sample = random.Random(2).sample(scored, n)
    log = D / "dispersal_bounds.jsonl"
    done = [r for r in map(json.loads, log.read_text().splitlines()) if r["ok"]] if log.exists() else []
    seen = {r["slug"] for r in done}
    m = json.load(urllib.request.urlopen(cb.BUCKET.removeprefix("/vsicurl/") + "/manifest.json"))
    with ProcessPoolExecutor(jobs, mp_context=multiprocessing.get_context("fork")) as pool, log.open("a") as out:
        futs = [pool.submit(run, s, m[s]["GBIF.IDs"][0].split("~"), c) for s, c in sample if s not in seen]
        for f in as_completed(futs):
            r = f.result()
            out.write(json.dumps(r) + "\n")
            out.flush()
            done += [r] if r["ok"] else []
            print(r["slug"], "ok" if r["ok"] else r["error"], flush=True)
    want = {s for s, _ in sample}
    summary(sorted((r for r in done if r["slug"] in want), key=lambda r: r["slug"]))
