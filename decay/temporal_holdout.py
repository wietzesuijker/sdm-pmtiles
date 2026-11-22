#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scipy", "rasterio", "mapbox-vector-tile"]
# ///
"""Out-of-time check of the decay comparison: fit on 2025 records, score on 2026 records.

On the same 120 species as dispersal_bounds.py, with calibrate_batch.py's records, effort (of the scored year)
and floor, fits the half-normal sigma and the fixed buffer D on 2025 records only, then scores 2026 records
under: fitted half-normal, fitted buffer, pooled class half-normal (decay_pooled.json), no dispersal (hard
clip, floor outside) and full dispersal (g = 1). Species with fewer than MIN_YEAR records in either year are
logged with skipped true. Failed species are logged with ok false and retried on the next run.

    decay/temporal_holdout.py [n] [jobs]
    decay/temporal_holdout.py summarize
"""
import collections
import json
import multiprocessing
import random
import sys
import urllib.request
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import calibrate
import calibrate_batch as cb
import numpy as np
from scipy.optimize import minimize_scalar

D = Path(cb.__file__).parent / "out"
MIN_YEAR = 5
SCEN = ("halfnormal", "buffer", "pooled", "none", "full")
hn_table = lambda v: np.exp(-(cb.KM / v) ** 2 / 2)
buf_table = lambda v: (cb.KM <= v).astype(float)
_effort: dict = {}


def loglik(ens, dist, table, rows, cols, effort) -> np.ndarray:
    """Per-record log intensity under a decay table (value per km), normalised, times effort."""
    k = cb.EFFORT_KM
    lam = np.where(ens == cb.NODATA, 0, np.maximum(ens * table[dist], cb.EPS))
    return np.log(lam[rows, cols] * effort[rows // k, cols // k]) - np.log((cb.block_sum(lam, k) * effort).sum())


def fit(make, ll):
    """Grid argmax of mean log-likelihood, then a bounded refinement in log scale (as calibrate_batch.py)."""
    v = max(cb.SCALES, key=lambda v: ll(make(v)).mean())
    r = minimize_scalar(lambda x: -ll(make(np.exp(x))).mean(), method="bounded", options={"xatol": 0.02},
                        bounds=(np.log(max(v / 2.5, 0.25)), np.log(min(v * 2.5, cb.MAX_KM))))
    return (float(np.exp(r.x)) if -r.fun > ll(make(v)).mean() else float(v)), v


def run(slug: str, keys: list[str], cls: str, pooled: float | None) -> dict:
    import rasterio

    try:
        def read(product):
            with rasterio.open(f"{cb.BUCKET}/{slug}_{product}_1km_na_1980-2010_sett1.tif") as src:
                return src.read(1), src.crs, src.transform

        ens, crs, tf = read("sdm-ensemble")
        dist = cb.distance_km(read("sdm-clip")[0])
        ens = ens.astype(np.int16)
        months = cb.BREEDING if cls == "Birds" else ""
        rec = {}
        for year in ("2025", "2026"):
            calibrate.YEARS = year
            rows, cols = cb.new_records(slug, keys, months, ens, crs, tf)
            key = (cls, months, year, tuple(tf), ens.shape)
            if key not in _effort:
                _effort[key] = cb.class_effort(cls, months, crs, tf, ens.shape)
            rec[year] = rows, cols, _effort[key]
        n25, n26 = len(rec["2025"][0]), len(rec["2026"][0])
        if min(n25, n26) < MIN_YEAR:
            return {"slug": slug, "class": cls, "ok": True, "skipped": True, "n_2025": n25, "n_2026": n26}

        def ll(year, table):
            return loglik(ens, dist, table, *rec[year][:2], rec[year][2])

        sigma, _ = fit(hn_table, lambda t: ll("2025", t))
        buf, _ = fit(buf_table, lambda t: ll("2025", t))
        tables = {"halfnormal": hn_table(sigma), "buffer": buf_table(buf), "none": (cb.KM == 0).astype(float),
                  "full": np.ones(cb.KM.size), **({"pooled": hn_table(pooled)} if pooled else {})}
        out = {f: round(n26 * float(ll("2026", t).mean()), 3) for f, t in tables.items()}
        return {"slug": slug, "class": cls, "ok": True, "skipped": False, "n_2025": n25, "n_2026": n26,
                "sigma_2025": round(sigma, 2), "buffer_2025": round(buf, 1), "pooled_km": pooled, "ll": out}
    # network, raster, GBIF and ogr2ogr failures; one bad species must not stop the batch
    except (OSError, ValueError, KeyError, IndexError, cb.subprocess.CalledProcessError) as e:
        return {"slug": slug, "ok": False, "error": f"{type(e).__name__}: {e}"[:300]}


def summary(rows: list[dict], sigma_all: dict[str, float], n_skipped: int = 0) -> dict:
    """Summed 2026 log-likelihood, differences vs the 2025-fitted half-normal with species-bootstrap CIs."""
    rng = np.random.default_rng(0)
    hn = np.array([r["ll"]["halfnormal"] for r in rows])
    res = {"n": len(rows), "skipped": n_skipped, "sum": {"halfnormal": round(float(hn.sum()), 1)}, "diff": {}}
    for f in SCEN[1:]:
        sub = [i for i, r in enumerate(rows) if f in r["ll"]]
        d = np.array([rows[i]["ll"][f] for i in sub]) - hn[sub]
        b = d[rng.integers(0, len(d), (2000, len(d)))].sum(1)
        res["sum"][f] = round(sum(rows[i]["ll"][f] for i in sub), 1)
        res["diff"][f] = {"n": len(sub), "diff": round(float(d.sum()), 1), "beats": round(float((d > 0).mean()), 3),
                          "ci90": [round(float(np.percentile(b, 5)), 1), round(float(np.percentile(b, 95)), 1)]}
    ratio = [r["sigma_2025"] / sigma_all[r["slug"]] for r in rows if r["slug"] in sigma_all]
    res["median_sigma_ratio"] = round(float(np.median(ratio)), 3)
    return res


def summarize() -> None:
    """Summary of the whole log (last line per slug wins), without fetching anything."""
    last = {r["slug"]: r for r in map(json.loads, (D / "temporal_holdout.jsonl").read_text().splitlines())}
    ok = [r for r in last.values() if r["ok"]]
    kept = sorted((r for r in ok if not r["skipped"] and "ll" in r), key=lambda r: r["slug"])
    sigma_all = {r["slug"]: r["halfnormal"]["fit"] for r in map(json.loads, (D / "calibration_all.jsonl").read_text().splitlines()) if "winner" in r}
    res = summary(kept, sigma_all, len(ok) - len(kept))
    res |= {"failed": len(last) - len(ok), "classes": dict(collections.Counter(r["class"] for r in kept))}
    print(json.dumps(res, indent=1))


def test_holdout():
    ens, dist, one = np.full((2, 2), 10, np.int16), np.array([[0, 5], [5, 5]]), np.ones((1, 1))
    r = np.array([0])
    assert abs(loglik(ens, dist, np.ones(6), r, r, one)[0] + np.log(4)) < 1e-9  # uniform: 1 of 4 cells
    hard = (np.arange(6) == 0).astype(float)
    assert abs(loglik(ens, dist, hard, r, r, one)[0] - np.log(10 / 11.5)) < 1e-9  # floor 0.5 outside the mask
    rows = [{"slug": s, "sigma_2025": 20.0, "ll": {"halfnormal": -10.0, "buffer": -9.0, "pooled": -11.0,
                                                    "none": -12.0, "full": -10.0}} for s in "ab"]
    s = summary(rows, {"a": 10.0, "b": 40.0})
    assert s["median_sigma_ratio"] == 1.25 and s["diff"]["buffer"]["diff"] == 2.0 and s["diff"]["full"]["beats"] == 0


if __name__ == "__main__":
    test_holdout()
    if sys.argv[1:] == ["summarize"]:
        sys.exit(summarize())
    n, jobs = (int(a) for a in (sys.argv[1:] + ["120", "4"])[:2])
    allrows = [json.loads(line) for line in (D / "calibration_all.jsonl").read_text().splitlines()]
    scored = sorted((r["slug"], r["class"]) for r in allrows if "winner" in r and r["outside_mask"] >= 3)
    sample = random.Random(2).sample(scored, n)
    sigma_all = {r["slug"]: r["halfnormal"]["fit"] for r in allrows if "winner" in r}
    pooled = json.loads((D / "decay_pooled.json").read_text())["species"]
    log = D / "temporal_holdout.jsonl"
    done = [r for r in map(json.loads, log.read_text().splitlines()) if r["ok"]] if log.exists() else []
    seen = {r["slug"] for r in done}
    m = json.load(urllib.request.urlopen(cb.BUCKET.removeprefix("/vsicurl/") + "/manifest.json"))
    with ProcessPoolExecutor(jobs, mp_context=multiprocessing.get_context("fork")) as pool, log.open("a") as out:
        futs = [pool.submit(run, s, m[s]["GBIF.IDs"][0].split("~"), c, pooled.get(s, {}).get("scale_km"))
                for s, c in sample if s not in seen]
        for f in as_completed(futs):
            r = f.result()
            out.write(json.dumps(r) + "\n")
            out.flush()
            done += [r] if r["ok"] else []
            print(r["slug"], "ok" if r["ok"] else r["error"], flush=True)
    want = {s for s, _ in sample}
    mine = [r for r in done if r["slug"] in want]
    kept = sorted((r for r in mine if not r["skipped"]), key=lambda r: r["slug"])
    print(f"done {len(mine)} of {n}")
    print(json.dumps(summary(kept, sigma_all, len(mine) - len(kept)), indent=1))
