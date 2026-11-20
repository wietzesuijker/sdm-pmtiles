#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scipy", "rasterio"]
# ///
"""Clip the 2071-2100 ensemble of every manifest species and upload it beside the bucket files.

Each species runs in its own temp dir that is deleted after upload, so disk stays flat. Needs the
Swift credential in the environment (OS_* from the openrc). Appends one JSON line per species to
the log, and skips species already in it, so a rerun resumes.

    decay/clip_future_batch.py decay/out/batch.jsonl [jobs]
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import urllib.request
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from evidence_decay import BUCKET, run

CONTAINER, PREFIX = "def-pollock-eckert", "vertebrates_NA/ECCC_2026"
NAME = "{}_sdm-ensemble-clipped_1km_na_2071-2100_sett1.tif"


def one(slug: str) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        try:
            r = run(slug, Path(tmp))
            f = Path(tmp) / NAME.format(slug)
            subprocess.run(["swift", "upload", CONTAINER, str(f), "--object-name", f"{PREFIX}/{f.name}"],
                           check=True, capture_output=True)
            return {"slug": slug, "ok": True, "bytes": f.stat().st_size,
                    **{k: r[k] for k in ("rebuild_within_1_outside", "d10_km", "future_ratio")}}
        except Exception as e:  # one bad species must not stop the batch
            return {"slug": slug, "ok": False, "error": f"{type(e).__name__}: {e}"[:300]}


if __name__ == "__main__":
    log, jobs = Path(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 6
    manifest = BUCKET.removeprefix("/vsicurl/") + "/manifest.json"
    slugs = sorted(json.load(urllib.request.urlopen(manifest)))
    done = {json.loads(line)["slug"] for line in log.read_text().splitlines()} if log.exists() else set()
    todo = [s for s in slugs if s not in done]
    print(f"{len(todo)} of {len(slugs)} to do", flush=True)
    with ProcessPoolExecutor(jobs) as pool, log.open("a") as out:
        for fut in as_completed([pool.submit(one, s) for s in todo]):
            res = fut.result()
            out.write(json.dumps(res) + "\n")
            out.flush()
            print(res["slug"], "ok" if res["ok"] else res["error"], flush=True)
