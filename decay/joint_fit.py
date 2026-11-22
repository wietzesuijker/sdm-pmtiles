#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scipy", "rasterio", "mapbox-vector-tile"]
# ///
"""Joint suitability exponent and decay scale, an in-repo approximation of Merow et al. (2017).

lambda(x) = max(S(x)^beta * g_sigma(d(x)), floor) * effort(x), S = ensemble / 100, g the half-normal of the distance
to the evidence mask. beta in [0, 3] and log sigma are fitted together by maximum likelihood on 2025 records and
scored on 2026 records, on temporal_holdout.py's 120-species sample, records, effort and floor. The comparison is
temporal_holdout's half-normal, i.e. beta = 1 with sigma fitted alone. The SDM stays fixed: beta only reweights it.
beta < 1 means the SDM is over-trusted relative to distance. Species with too few records in a year are logged
skipped. Failed species are logged with ok false and retried on the next run.

    decay/joint_fit.py [n] [jobs]
    decay/joint_fit.py summarize
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
import temporal_holdout as th
from scipy.optimize import minimize

D = Path(cb.__file__).parent / "out"
FLOOR = cb.EPS / 100  # the floor in S units (ensemble is 0-100)
BETA_MAX = 3.0
LOG_SIGMA = (np.log(0.25), np.log(cb.MAX_KM))


def compress(ens, dist, effort):
    """Valid cells as unique (suitability, distance, effort block) triples with counts, so each evaluation is cheap."""
    k = cb.EFFORT_KM
    rr, cc = np.nonzero(ens != cb.NODATA)
    nb = -(-ens.shape[1] // k)
    blk = (rr // k) * nb + cc // k
    key, cnt = np.unique((blk * 256 + ens[rr, cc]) * (cb.MAX_KM + 1) + dist[rr, cc], return_counts=True)
    d, rest = key % (cb.MAX_KM + 1), key // (cb.MAX_KM + 1)
    b = rest // 256
    return rest % 256 / 100, d, effort[b // nb, b % nb], cnt


def lam(s, d, beta, sigma):
    return np.maximum(s**beta * np.exp(-((d / sigma) ** 2) / 2), FLOOR)


def loglik(comp, rows, cols, ens, dist, effort, beta, sigma) -> np.ndarray:
    """Per-record log intensity, normalised over the valid cells, times effort."""
    s, d, eff, cnt = comp
    k = cb.EFFORT_KM
    at = effort[rows // k, cols // k]
    return np.log(lam(ens[rows, cols] / 100, dist[rows, cols], beta, sigma) * at) - np.log((cnt * eff * lam(s, d, beta, sigma)).sum())


def fit_joint(ll, sigma0: float):
    """Bounded 2-D search from the fixed-beta fit; never worse in sample than that start."""
    f = lambda p: -ll(p[0], np.exp(p[1])).mean()
    x0 = np.array([1.0, np.log(sigma0)])
    r = minimize(f, x0, method="L-BFGS-B", bounds=[(0, BETA_MAX), LOG_SIGMA], options={"eps": 1e-3})
    b, ls = r.x if r.fun < f(x0) else x0
    return float(b), float(np.exp(ls))


def run(slug: str, keys: list[str], cls: str) -> dict:
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
            if key not in th._effort:
                th._effort[key] = cb.class_effort(cls, months, crs, tf, ens.shape)
            rec[year] = rows, cols, th._effort[key]
        n25, n26 = len(rec["2025"][0]), len(rec["2026"][0])
        if min(n25, n26) < th.MIN_YEAR:
            return {"slug": slug, "class": cls, "ok": True, "skipped": True, "n_2025": n25, "n_2026": n26}
        comp = {y: compress(ens, dist, rec[y][2]) for y in rec}

        def ll(year):
            return lambda beta, sigma: loglik(comp[year], *rec[year][:2], ens, dist, rec[year][2], beta, sigma)

        sigma_f, _ = th.fit(lambda v: v, lambda v: ll("2025")(1.0, v))
        beta, sigma_j = fit_joint(ll("2025"), sigma_f)
        return {"slug": slug, "class": cls, "ok": True, "skipped": False, "n_2025": n25, "n_2026": n26,
                "beta": round(beta, 3), "sigma_joint": round(sigma_j, 2), "sigma_fixed": round(sigma_f, 2),
                "ll_2026": {"joint": round(n26 * float(ll("2026")(beta, sigma_j).mean()), 3),
                            "fixed": round(n26 * float(ll("2026")(1.0, sigma_f).mean()), 3)}}
    # network, raster, GBIF and ogr2ogr failures; one bad species must not stop the batch
    except (OSError, ValueError, KeyError, IndexError, cb.subprocess.CalledProcessError) as e:
        return {"slug": slug, "ok": False, "error": f"{type(e).__name__}: {e}"[:300]}


def summary(rows: list[dict], n_skipped: int = 0) -> dict:
    """Joint minus fixed summed 2026 log-likelihood with a 90% species bootstrap, and beta and sigma by class."""
    rows = sorted(rows, key=lambda r: r["slug"])
    d = np.array([r["ll_2026"]["joint"] - r["ll_2026"]["fixed"] for r in rows])
    b = d[np.random.default_rng(0).integers(0, len(d), (2000, len(d)))].sum(1)
    by = collections.defaultdict(list)
    for r in rows:
        by[r["class"]].append(r)
    med = lambda c, f: round(float(np.median([f(r) for r in by[c]])), 3)
    return {"n": len(rows), "skipped": n_skipped, "sum_fixed": round(sum(r["ll_2026"]["fixed"] for r in rows), 1),
            "diff": round(float(d.sum()), 1), "ci90": [round(float(np.percentile(b, 5)), 1), round(float(np.percentile(b, 95)), 1)],
            "improved": round(float((d > 0).mean()), 3), "median_beta": {c: med(c, lambda r: r["beta"]) for c in by},
            "median_sigma_ratio": {c: med(c, lambda r: r["sigma_joint"] / r["sigma_fixed"]) for c in by},
            "classes": {c: len(v) for c, v in by.items()}}


def summarize() -> None:
    last = {r["slug"]: r for r in map(json.loads, (D / "joint_fit.jsonl").read_text().splitlines())}
    ok = [r for r in last.values() if r["ok"]]
    kept = [r for r in ok if not r["skipped"]]
    print(json.dumps(summary(kept, len(ok) - len(kept)) | {"failed": len(last) - len(ok)}, indent=1))


def test_joint():
    ens, dist, one = np.array([[10, 40], [100, 100]], np.int16), np.array([[0, 5], [0, 9]]), np.ones((1, 1))
    comp = compress(ens, dist, one)
    assert comp[3].sum() == 4
    r, c = np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1])
    for beta in (0.0, 1.0, 2.5):  # the intensity over every valid cell is a distribution
        assert np.isclose(np.exp(loglik(comp, r, c, ens, dist, one, beta, 5.0)).sum(), 1)
    # beta = 1 is temporal_holdout's half-normal likelihood
    assert np.allclose(loglik(comp, r, c, ens, dist, one, 1.0, 5.0), th.loglik(ens, dist, th.hn_table(5.0), r, c, one))
    s = summary([{"slug": x, "class": "Birds", "beta": 0.5, "sigma_joint": 20.0, "sigma_fixed": 10.0,
                  "ll_2026": {"joint": -9.0, "fixed": -10.0}} for x in "ab"])
    assert s["diff"] == 2.0 and s["improved"] == 1.0 and s["median_sigma_ratio"] == {"Birds": 2.0}


if __name__ == "__main__":
    test_joint()
    if sys.argv[1:] == ["summarize"]:
        sys.exit(summarize())
    n, jobs = (int(a) for a in (sys.argv[1:] + ["120", "1"])[:2])
    allrows = [json.loads(line) for line in (D / "calibration_all.jsonl").read_text().splitlines()]
    scored = sorted((r["slug"], r["class"]) for r in allrows if "winner" in r and r["outside_mask"] >= 3)
    sample = random.Random(2).sample(scored, n)
    log = D / "joint_fit.jsonl"
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
    mine = [r for r in done if r["slug"] in want]
    kept = [r for r in mine if not r["skipped"]]
    print(f"done {len(mine)} of {n}")
    print(json.dumps(summary(kept, len(mine) - len(kept)), indent=1))
