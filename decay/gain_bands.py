#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scipy", "rasterio", "matplotlib"]
# ///
"""Where projected change happens: 2071-2100 minus 1980-2010 suitability, summed by distance band to the evidence mask.

Tests the paper's inference that projected gains lie far from known occurrence and losses near it.
Random sample of species, resumable JSON lines.

    decay/gain_bands.py decay/out/gain_bands.jsonl [n] [jobs]
    decay/gain_bands.py plot decay/out/gain_bands.jsonl paper-media/fig2    # Fig. 2, png (300 dpi) and pdf
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


def plot(log: Path, stem: str) -> None:
    import matplotlib.pyplot as plt

    from fig_paper import INK, MM, MUTED, STYLE, save

    rows = [r for r in map(json.loads, log.read_text().splitlines()) if r["ok"]]
    share = lambda k: np.array([np.array(r[k]) / max(sum(r[k]), 1) for r in rows if sum(r[k]) > 0])
    gain, loss = share("gain"), share("loss")
    rng = np.random.default_rng(0)
    for name, v in (("gain", gain), ("loss", loss)):
        boot = lambda f, v=v: np.percentile([f(v[rng.integers(0, len(v), len(v))]) for _ in range(2000)], [5, 95]).round(3).tolist()
        print(f"{name}: n={len(v)} mean share by band {(100 * v.mean(0)).round(1).tolist()}; "
              f">200 km {v[:, 4:].sum(1).mean():.3f} {boot(lambda x: x[:, 4:].sum(1).mean())}; inside {v[:, 0].mean():.3f} {boot(lambda x: x[:, 0].mean())}")
    n = min(len(gain), len(loss))
    both = [r for r in rows if sum(r["gain"]) > 0 and sum(r["loss"]) > 0]
    far = np.mean([sum(r["gain"][4:]) / sum(r["gain"]) > sum(r["loss"][4:]) / sum(r["loss"]) for r in both])
    print(f"species with more of gains than losses beyond 200 km: {far:.3f} of {len(both)}; pooled gain >200 km "
          f"{sum(sum(r['gain'][4:]) for r in rows) / sum(sum(r['gain']) for r in rows):.3f}")

    labels = ["Inside\nmask", "1-10", "10-50", "50-200", "200-1,000", ">1,000 km"]
    ramp = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0b2a52"]  # validated ordinal ramp, light = near
    plt.rcParams.update(STYLE)
    fig, ax = plt.subplots(figsize=(88 * MM, 52 * MM))
    fig.subplots_adjust(left=0.2, right=0.97, top=0.72, bottom=0.18)
    for y, (name, v) in enumerate((("Gains", gain), ("Losses", loss))):
        left = 0
        for i, w in enumerate(100 * v.mean(0)):
            ax.barh(y, w - 0.4, left=left + 0.2, height=0.62, color=ramp[i], lw=0)
            if w >= 6:
                ax.text(left + w / 2, y, f"{w:.0f}", ha="center", va="center", fontsize=6.5, color="white" if i >= 2 else INK)
            left += w
        far_share = 100 * v[:, 4:].sum(1).mean()
        ax.annotate("", (100, y + 0.4), (100 - far_share, y + 0.4), arrowprops={"arrowstyle": "-", "color": INK, "lw": 0.6})
        ax.text(100 - far_share / 2, y + 0.43, f"{far_share:.0f}% beyond 200 km", ha="center", va="top", fontsize=6.5)
    for i, lab in enumerate(labels):  # band key above, two rows of three, ordered like the bars
        x, yk = 34 * (i % 3), -1.0 + 0.24 * (i // 3)
        ax.add_patch(plt.Rectangle((x, yk - 0.08), 3, 0.16, color=ramp[i], clip_on=False))
        ax.text(x + 4.5, yk, lab.replace("\n", " "), fontsize=6, va="center", color=MUTED)
    ax.text(0, -1.3, "Distance to the evidence mask", fontsize=6.5, va="center")
    ax.set_ylim(1.62, -0.5)
    ax.set_xlim(0, 100)
    ax.set_yticks([0, 1], ["Gained\nsuitability", "Lost\nsuitability"])
    ax.set_xticks(range(0, 101, 25))
    ax.spines["left"].set_visible(False)
    ax.set_xlabel(f"Mean share per species (%), n={n:,}")
    save(fig, stem)


if __name__ == "__main__" and sys.argv[1:2] == ["plot"]:
    plot(Path(sys.argv[2]), sys.argv[3])
elif __name__ == "__main__":
    log, n, jobs = Path(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 300, int(sys.argv[3]) if len(sys.argv) > 3 else 4
    manifest = json.load(urllib.request.urlopen(BUCKET.removeprefix("/vsicurl/") + "/manifest.json"))
    done = {json.loads(line)["slug"] for line in log.read_text().splitlines()} if log.exists() else set()
    todo = [s for s in random.Random(0).sample(sorted(manifest), n) if s not in done]
    with ProcessPoolExecutor(jobs) as pool, log.open("a") as out:
        for fut in as_completed([pool.submit(one, s) for s in todo]):
            out.write(json.dumps(fut.result()) + "\n")
            out.flush()
