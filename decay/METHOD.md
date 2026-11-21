# Evidence decay: method

This folder recovers the evidence clip used on the published SDM maps, applies it to future periods, and tests which decay shape best predicts new records.

## Terms

- **Ensemble**: the model's suitability map, `sdm-ensemble`, 1 km, values 0-100, nodata 255.
- **Evidence mask**: `sdm-clip`, 1 where the species is known (IUCN range plus training records), else 0.
- **d**: km from a cell to the nearest evidence-mask cell (Euclidean distance transform on the 1 km grid, capped at 5,000 km).
- **Decay g(d)**: a weight from 1 at d = 0 down towards 0. The clipped map is `ensemble x g(d)`; inside the mask it is the ensemble.
- **Published curve**: the g in the bucket's `sdm-ensemble-clipped` maps, one per species.

## 1. Recover the published curve (`evidence_decay.py`)

For each km of d, take the median of clipped / ensemble over cells outside the mask with ensemble >= 5, then force the curve to be non-increasing. The fit uses the full 1 km files only: the overviews average the binary mask and move the curve.

Check: rebuilding the clipped map from the curve matches the published map to within 1 unit on 99.85% of out-of-mask cells for the median species, across all 1,968 species.

## 2. Clip future periods with current evidence (`evidence_decay.py`, `clip_future_batch.py`)

The bucket ships 2071-2100 unclipped. The clipped 2100 map is `apply_curve(2100 ensemble, 1980-2010 mask, d, curve)`: the same mask, distances and curve as the current map. So the current and future maps differ only in climate, not in the clip.

Output: `<slug>_sdm-ensemble-clipped_1km_na_2071-2100_sett1.tif` for all 1,968 species, uploaded next to the bucket files.

## 3. Gain or loss by 2100 (`raw_ratio_batch.py`)

Change = total suitability in 2071-2100 / total in 1980-2010. Over all 1,968 species:

| comparison | species with a gain |
|---|---|
| unclipped 2100 / clipped current | 1,862 (95%) |
| unclipped 2100 / unclipped current | 1,374 (70%) |
| clipped 2100 / clipped current, same evidence | 811 (41%) |

1,051 species (53%) go from gain to loss between the first and last rows. None go from loss to gain.

## 4. Test decay shapes on new records (`calibrate.py`, `calibrate_batch.py`)

**Test data.** GBIF records from 2025-2026, after all training data, with coordinates, no geospatial issue, uncertainty <= 5 km, and in North America. Birds use June-July only, so migrants do not count as range. Records within 2 km of any training point are dropped, so the test set is not the fit set. Species with fewer than 10 records are not scored.

**Score.** Each candidate g defines an intensity `lambda(x) = max(ensemble(x) g(d(x)), 0.5) x effort(x)`. The score is the mean log-likelihood of the new records under the normalised intensity, `mean_i log(lambda(x_i) / sum_x lambda(x))` (a presence-only Poisson point process). Higher is better. The floor 0.5 keeps a hard clip finite outside the mask.

**Effort.** New records come from where people look. `effort(x)` is the count of all GBIF records of the species' class (mammals, birds, amphibians, reptiles) in the same years and months, per 25 km cell, from GBIF's map tiles (a target-group background, Phillips et al. 2009). Without it the test rewards tight clips only because observers return to the same places.

**Candidates.** All use d to the evidence mask:

- `stretch`: the published curve with distances scaled by s, g(d / s), s in {0, 1/16, 1/8, 1/4, 1/2, 3/4, 1, 1.5, 2, 3, 5, 8, inf}. s = 1 is the published clip, s = 0 a hard clip, inf no clip.
- `exp`: exp(-d / tau), tau in {1, 2, 5, 10, 20, 40, 80, 160, 320, 640, 1280, 2560} km. This is the paper's shape, measured to the evidence mask, not to the IUCN range.
- `halfnormal`: exp(-d^2 / (2 sigma^2)), sigma on the same grid.

**Uncertainty.** 200 bootstrap resamples of the records give a 90% interval for the best parameter in each family and how often each family wins.

**Output.** One JSON line per species in `out/calibration_all.jsonl`: `n_new`, `outside_mask`, the overall winner, `winner_share`, `ll_published` (score at s = 1), and for each family the best parameter, its score and 90% interval.

## 5. Pool across species (`pool.py`)

Many species have few new records, so their own best parameter is noisy. For the chosen family, take log(best parameter) per species with its bootstrap spread, and shrink each towards the mean of its class (empirical Bayes, normal-normal). Species with many records keep their own value. Species with few records move towards the class value. Species with fewer than 10 records get the class value.

## Limits

- One 2-year window of new records.
- The class is a broad target group: a birder and a herper do not survey the same places.
- The effort grid is 25 km, coarser than the 1 km maps.
- The paper's tau is measured to the IUCN range. This test measures every shape to the evidence mask, so it compares shapes, not the paper's exact distance.
- Some best values sit at the end of a grid (1 km, or s = 0). That means "as tight as tested", not an exact value.

## Reproduce

```
uv run --with numpy --with scipy --with rasterio --with mapbox-vector-tile \
  python decay/calibrate_batch.py decay/out/calibration_all.jsonl 4
python decay/pool.py decay/out/calibration_all.jsonl
```

Run long batches under `caffeinate -i`: when the Mac sleeps, bucket and GBIF reads fail with DNS errors. Both batch scripts resume and skip species already in the log; remove the failed lines to retry them.
