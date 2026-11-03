#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["pandas", "scipy"]
# ///
"""Rank species for review from decay/out/triage.csv and write triage.html.

Score: share of the displayed (clipped) suitability that sits more than
500 km from any evidence cell. Prints the gate checks, writes a local page.

    decay/report.py   # -> decay/out/triage.html
"""
from __future__ import annotations

import html
from pathlib import Path

import pandas as pd

OUT = Path(__file__).parent / "out"
TOP = 40


def load() -> pd.DataFrame:
    d = pd.read_csv(OUT / "triage.csv")
    failed = d.error.notna().sum()
    if failed:
        raise SystemExit(f"{failed} species failed, rerun decay/sweep.py first")
    d["rank"] = d.share_beyond_500km.rank(ascending=False, method="min").astype(int)
    return d.sort_values("rank")


def checks(d: pd.DataFrame) -> list[str]:
    u = d[d.slug == "ursus-arctos"].iloc[0]
    return [
        f"{len(d)} species, rebuild within 1 outside the mask: median {d.rebuild_within_1_outside.median():.4f}, "
        f"min {d.rebuild_within_1_outside.min():.4f}, {(d.rebuild_within_1_outside < 0.99).sum()} below 0.99",
        f"ursus-arctos: share {u.share_beyond_500km:.3f}, rank {u['rank']} of {len(d)}",
        f"species with > 5% shown beyond 500 km: {(d.share_beyond_500km > 0.05).sum()}, "
        f"> 1%: {(d.share_beyond_500km > 0.01).sum()}",
        f"Spearman share vs n_occ: {d.share_beyond_500km.corr(d.n_occ, method='spearman'):.3f}, "
        f"d10 vs n_occ: {d.d10_km.corr(d.n_occ, method='spearman'):.3f}",
    ]


def page(d: pd.DataFrame, lines: list[str]) -> str:
    by_class = d.groupby("class").agg(
        species=("slug", "size"), median_d50_km=("d50_km", "median"), median_d10_km=("d10_km", "median"),
        over_5pct=("share_beyond_500km", lambda s: int((s > 0.05).sum())))
    top = d.head(TOP)[["rank", "slug", "common_name", "class", "iucn_status", "n_occ",
                       "d50_km", "d10_km", "share_beyond_500km"]]
    items = "".join(f"<li>{html.escape(x)}</li>" for x in lines)
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Evidence decay triage</title>
<style>:root{{--bg:#fafaf7;--fg:#1d1d1b;--line:#d8d8d2}}
@media (prefers-color-scheme:dark){{:root{{--bg:#161615;--fg:#ecece6;--line:#34342f}}}}
body{{margin:0;padding:16px;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif}}
main{{max-width:1100px;margin:0 auto}} .wrap{{overflow-x:auto}}
table{{border-collapse:collapse;font-variant-numeric:tabular-nums}} td,th{{padding:4px 8px;border-bottom:1px solid var(--line);text-align:right}}
td:nth-child(2),td:nth-child(3),td:nth-child(4),th{{text-align:left}}</style></head>
<body><main><h1>Evidence decay triage</h1>
<p>Each clipped SDM is the ensemble times a per-species curve of distance to the evidence mask. Score: share of the displayed suitability more than 500 km from any evidence. High scores are species to check first.</p>
<ul>{items}</ul><h2>By class</h2><div class="wrap">{by_class.to_html()}</div>
<h2>Top {TOP}</h2><div class="wrap">{top.to_html(index=False, float_format=lambda x: f"{x:.3f}")}</div>
</main></body></html>"""


if __name__ == "__main__":
    d = load()
    lines = checks(d)
    print("\n".join(lines))
    (OUT / "triage.html").write_text(page(d, lines))
    print(f"wrote {OUT / 'triage.html'}")
