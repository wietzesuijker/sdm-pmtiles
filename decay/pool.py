#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy"]
# ///
"""Pick the decay family that best predicts new records, then shrink each species' scale to its class.

Reads calibrate_batch.py's log. The family is the parametric one (exp or halfnormal) with the higher
summed log-likelihood over scored species. Each species' log(scale) is shrunk towards its class mean
(normal-normal empirical Bayes, standard error from the block-bootstrap 90% interval). Species with fewer than
MIN_OUTSIDE test records outside the mask, or fewer than 10 in all, get the class value. Each species' 2071-2100 / 1980-2010 change is read at its pooled scale and at the ends of its
90% posterior interval, so the count of species that gain comes with a range. Writes decay_pooled.json next to
the log and prints a summary.

    decay/pool.py decay/out/calibration_all.jsonl
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

SCALES = [1, 2, 5, 10, 20, 40, 80, 160, 320, 640, 1280, 2560]  # km, as in calibrate_batch.py
Z90 = 1.645
MIN_SE = 0.7  # about one grid step in log scale: a bootstrap interval that collapses to one grid value means "finer than the grid", not "precise"
MIN_OUTSIDE = 3  # test records outside the mask; with none, every candidate scores the same on the records and the tightest decay wins by construction


def shrink(y: np.ndarray, se: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Posterior means and sds, class mean and between-species sd for estimates y with standard errors se."""
    w = 1 / se**2
    mu = float((w * y).sum() / w.sum())
    tau2 = max(float(np.var(y) - np.mean(se**2)), 1e-6)
    b = tau2 / (tau2 + se**2)
    return b * y + (1 - b) * mu, (b * se**2) ** 0.5, mu, tau2**0.5


def change_at(ratios: list[float], log_scale: float) -> float:
    """Change at a scale, interpolated in log scale and log ratio between the grid values."""
    return float(np.exp(np.interp(log_scale, np.log(SCALES), np.log(ratios))))


def main(log: Path) -> dict:
    rows = [r for r in map(json.loads, log.read_text().splitlines()) if r["ok"]]
    scored = [r for r in rows if "winner" in r and r["outside_mask"] >= MIN_OUTSIDE]
    totals = {f: sum(r[f]["ll"] for r in scored) for f in ("stretch", "exp", "halfnormal")}
    fam = max(("exp", "halfnormal"), key=totals.get)
    by_class = defaultdict(list)
    for r in scored:
        by_class[r["class"]].append(r)
    out, classes = {}, {}
    for cls, rs in by_class.items():
        y = np.log([r[fam].get("fit", r[fam]["best"]) for r in rs])
        lo, hi = np.log([[r[fam]["ci90"][0], r[fam]["ci90"][1]] for r in rs]).T
        post, psd, mu, sd = shrink(y, np.maximum((hi - lo) / 3.29, MIN_SE))
        classes[cls] = {"n": len(rs), "mu": mu, "sd": sd, "scale_km": round(float(np.exp(mu)), 1),
                        "median_own_km": round(float(np.exp(np.median(y))), 1), "between_sd_log": round(sd, 3)}
        for r, p, q in zip(rs, post, psd):
            out[r["slug"]] = {"log_scale": (p, q), "own_km": r[fam].get("fit", r[fam]["best"]), "n_new": r["n_new"]}
    for r in rows:
        if r["slug"] not in out and r.get("class") in classes:
            c = classes[r["class"]]
            out[r["slug"]] = {"log_scale": (c["mu"], c["sd"]), "own_km": None, "n_new": r.get("n_new", 0)}
    by_slug = {r["slug"]: r for r in rows}
    gains = np.zeros(3, int)  # at the low end, the pooled scale, the high end of each species' interval
    for slug, o in out.items():
        p, q = o.pop("log_scale")
        o["scale_km"] = round(float(np.exp(p)), 1)
        ch = [change_at(by_slug[slug]["change"][fam], x) for x in (p - Z90 * q, p, p + Z90 * q)]
        o["change"], o["change_ci90"] = round(ch[1], 4), [round(min(ch[0], ch[2]), 4), round(max(ch[0], ch[2]), 4)]
        gains += np.array(ch) > 1
    for c in classes.values():
        del c["mu"], c["sd"]
    weighted = {f: sum(r["n_new"] * r[f].get("ll_fit", r[f]["ll"]) for r in scored) for f in totals}
    weighted["published"] = sum(r["n_new"] * r["ll_published"] for r in scored)
    # published curve outside the stretch family's own 90% interval: a test that accounts for the search
    rejects = sum(not (lo_s <= 1 <= hi_s) for lo_s, hi_s in
                  ((float(r["stretch"]["ci90"][0]), float(r["stretch"]["ci90"][1])) for r in scored))
    beats = sum(max(r["exp"]["ll"], r["halfnormal"]["ll"]) > r["ll_published"] for r in scored)
    summary = {"family": fam, "ll_total": {k: round(v, 1) for k, v in totals.items()},
               "ll_records": {k: round(v, 1) for k, v in weighted.items()},
               "ll_total_published": round(sum(r["ll_published"] for r in scored), 1),
               "n_rows": len(rows), "n_scored": len(scored), "parametric_beats_published": beats,
               "published_rejected_ci90": rejects, "n_low_outside": sum("winner" in r for r in rows) - len(scored),
               "species_gaining": {"pooled": int(gains[1]), "tight_end": int(gains[0]), "wide_end": int(gains[2]),
                                   "of": len(out)},
               "winner_counts": {f: sum(r["winner"][0] == f for r in scored) for f in totals}, "classes": classes}
    (log.parent / "decay_pooled.json").write_text(json.dumps({"summary": summary, "species": out}, indent=1))
    return summary


def test_shrink():
    post, _, mu, _ = shrink(np.array([0.0, 4.0]), np.array([0.1, 10.0]))
    assert abs(post[0]) < 0.1 and abs(post[1] - mu) < 0.5  # precise estimate kept, noisy one pulled in
    assert abs(change_at([1, 2] + [2] * 10, np.log(np.sqrt(2))) - 2**0.5) < 1e-9  # halfway between 1 and 2 km


if __name__ == "__main__":
    test_shrink()
    print(json.dumps(main(Path(sys.argv[1])), indent=1))
