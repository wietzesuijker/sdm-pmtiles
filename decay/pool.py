#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy"]
# ///
"""Pick the decay family that best predicts new records, then shrink each species' scale to its class.

Reads calibrate_batch.py's log. The family is the parametric one (exp or halfnormal) with the higher
summed log-likelihood over scored species. Each species' log(scale) is shrunk towards its class mean
(normal-normal empirical Bayes, standard error from the bootstrap 90% interval). Unscored species get
the class value. Writes decay_pooled.json next to the log and prints a summary.

    decay/pool.py decay/out/calibration_all.jsonl
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

MIN_SE = 0.35  # about half a grid step in log scale; the grid cannot resolve less


def shrink(y: np.ndarray, se: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Posterior means, class mean and between-species sd for estimates y with standard errors se."""
    w = 1 / se**2
    mu = float((w * y).sum() / w.sum())
    tau2 = max(float(np.var(y) - np.mean(se**2)), 1e-6)
    b = tau2 / (tau2 + se**2)
    return b * y + (1 - b) * mu, mu, tau2**0.5


def main(log: Path) -> dict:
    rows = [r for r in map(json.loads, log.read_text().splitlines()) if r["ok"]]
    scored = [r for r in rows if "winner" in r]
    totals = {f: sum(r[f]["ll"] for r in scored) for f in ("stretch", "exp", "halfnormal")}
    fam = max(("exp", "halfnormal"), key=totals.get)
    by_class = defaultdict(list)
    for r in scored:
        by_class[r["class"]].append(r)
    out, classes = {}, {}
    for cls, rs in by_class.items():
        y = np.log([r[fam]["best"] for r in rs])
        lo, hi = np.log([[r[fam]["ci90"][0], r[fam]["ci90"][1]] for r in rs]).T
        post, mu, sd = shrink(y, np.maximum((hi - lo) / 3.29, MIN_SE))
        classes[cls] = {"n": len(rs), "scale_km": round(float(np.exp(mu)), 1), "between_sd_log": round(sd, 3)}
        for r, p in zip(rs, post):
            out[r["slug"]] = {"scale_km": round(float(np.exp(p)), 1), "own_km": r[fam]["best"], "n_new": r["n_new"]}
    for r in rows:
        if r["slug"] not in out and r.get("class") in classes:
            out[r["slug"]] = {"scale_km": classes[r["class"]]["scale_km"], "own_km": None, "n_new": r.get("n_new", 0)}
    beats = sum(max(r["exp"]["ll"], r["halfnormal"]["ll"]) > r["ll_published"] for r in scored)
    summary = {"family": fam, "ll_total": {k: round(v, 1) for k, v in totals.items()},
               "ll_total_published": round(sum(r["ll_published"] for r in scored), 1),
               "n_rows": len(rows), "n_scored": len(scored), "parametric_beats_published": beats,
               "winner_counts": {f: sum(r["winner"][0] == f for r in scored) for f in totals}, "classes": classes}
    (log.parent / "decay_pooled.json").write_text(json.dumps({"summary": summary, "species": out}, indent=1))
    return summary


def test_shrink():
    post, mu, _ = shrink(np.array([0.0, 4.0]), np.array([0.1, 10.0]))
    assert abs(post[0]) < 0.1 and abs(post[1] - mu) < 0.5  # precise estimate kept, noisy one pulled in


if __name__ == "__main__":
    test_shrink()
    print(json.dumps(main(Path(sys.argv[1])), indent=1))
