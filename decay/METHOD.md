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

The first row cannot show fewer gains than the last, because g <= 1 only lowers the future total. The fair contrast is the second row against the last, which treat both periods alike: 574 species (29%, 90% CI 27.4-30.8% from a species bootstrap, `fig_paper.py`) go from gain to loss and 11 from loss to gain.

## 4. Test decay shapes on new records (`calibrate.py`, `calibrate_batch.py`)

**Test data.** GBIF records from 2025-2026, after all training data, with coordinates, no geospatial issue, uncertainty <= 5 km, and in North America. Birds use June-July only, so migrants do not count as range. Records within 2 km of any training point are dropped, so the test set is not the fit set. Species with fewer than 10 records, or fewer than 3 outside the evidence mask, are not scored: records inside the mask have g = 1 under every candidate, so with none outside the tightest decay wins by construction.

**Score.** Each candidate g defines an intensity `lambda(x) = max(ensemble(x) g(d(x)), 0.5) x effort(x)`. The score is the mean log-likelihood of the new records under the normalised intensity, `mean_i log(lambda(x_i) / sum_x lambda(x))` (a presence-only Poisson point process). Higher is better. The floor 0.5 keeps a hard clip finite outside the mask.

**Effort.** New records come from where people look. `effort(x)` is the count of all GBIF records of the species' class (mammals, birds, amphibians, reptiles) in the same years and months, per 25 km cell, from GBIF's map tiles (a target-group background, Phillips et al. 2009). Without it the test rewards tight clips only because observers return to the same places.

**Candidates.** All use d to the evidence mask:

- `stretch`: the published curve with distances scaled by s, g(d / s), s in {0, 1/16, 1/8, 1/4, 1/2, 3/4, 1, 1.5, 2, 3, 5, 8, inf}. s = 1 is the published clip, s = 0 a hard clip, inf no clip.
- `exp`: exp(-d / tau), tau in {1, 2, 5, 10, 20, 40, 80, 160, 320, 640, 1280, 2560} km. This is the paper's shape, measured to the evidence mask, not to the IUCN range.
- `halfnormal`: exp(-d^2 / (2 sigma^2)), sigma on the same grid.

For `exp` and `halfnormal`, the best grid value is then refined by maximum likelihood (bounded 1-D search in log scale, lower bound 0.4 km), stored as `fit`.

**Uncertainty.** New records cluster along roads, trails and towns, so records are not independent. The bootstrap resamples 25 km blocks of records (the effort grid cells), not single records, 200 times. This gives a 90% interval for the best grid parameter in each family and how often each family wins.

**Change by 2100.** For every species, scored or not, the run also stores the 2071-2100 / 1980-2010 total suitability ratio with both periods clipped by `exp` and by `halfnormal` at each grid scale (`change`).

**Output.** One JSON line per species in `out/calibration_all.jsonl`: `n_new`, `outside_mask`, the overall winner, `winner_share`, `ll_published` (score at s = 1), and for each family the best parameter, its score and 90% interval.

## 5. Pool across species (`pool.py`)

Many species have few new records, so their own best parameter is noisy. For the chosen family, take log(best parameter) per species with its bootstrap spread, and shrink each towards the mean of its class (empirical Bayes, normal-normal). Species with many records keep their own value. Species with few records move towards the class value. Species with fewer than 10 records get the class value.

Each species' change by 2100 is read at its pooled scale and at both ends of its 90% posterior interval (interpolated between grid scales). `species_gaining` counts the species with a gain at the pooled scale, at the tight end and at the wide end, so the gain count comes with a range.

## Limits

- The decay multiplies a fixed ensemble after fitting. Merow et al. (2017) instead fit the range term inside the SDM. Doing that here needs a refit of the ensembles, which this repo does not do.
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

## References

- Merow C, Wilson AM, Jetz W. 2017. Integrating occurrence data and expert maps for improved species range predictions. Global Ecology and Biogeography 26(2): 243-258. doi:10.1111/geb.12539
- Mendes P, Velazco SJE, de Andrade AFA, De Marco P. 2020. Dealing with overprediction in species distribution models: How adding distance constraints can improve model accuracy. Ecological Modelling 431: 109180. doi:10.1016/j.ecolmodel.2020.109180
- Phillips SJ, Dudík M, Elith J, Graham CH, Lehmann A, Leathwick J, Ferrier S. 2009. Sample selection bias and presence-only distribution models: implications for background and pseudo-absence data. Ecological Applications 19(1): 181-197. doi:10.1890/07-2153.1
- Birds use June-July records. A July-only rescore of 120 random scored birds (`birds_july.py`, 98 scorable) gives a median July / June-July sigma of 1.0, so late migrants do not explain the wide bird scale.
- A fixed buffer (g = 1 within D km of the mask, else the floor) scored on 240 random scored species (`buffer_check.py`) fits new records slightly better than the half-normal: median D 68 km, record-weighted log-likelihood higher by 973 (90% species-bootstrap CI 557-1,438), better for 75% of species. The calibrated distance matters, the shape little.
- Sensitivity on 120 random scored species (`sensitivity.py`), half-normal sigma refitted under each setting with the same records. Median per-species sigma ratio to the baseline (floor 0.5, 25 km effort): floor 0.1 0.958, floor 2.0 1.134, no effort term 0.832, 50 km effort grid 1.000. Median sigma amphibians/birds/mammals/reptiles in km: baseline 10.5/339/9.9/11.6, floor 0.1 10.3/321/8.8/11.4, floor 2.0 11.2/318/14.0/13.2, no effort 9.6/179/13.8/8.8, 50 km effort 9.6/340/8.7/11.2. Species whose 2071-2100 change flips side of 1 at their own sigma: 4, 4, 5, 1. The class ordering holds in every setting. Dropping the effort term roughly halves the bird scale, so the effort correction matters for birds.
- Classic dispersal scenarios (Thuiller 2005; Bateman et al. 2013) on the same 120 species (`dispersal_bounds.py`), record-weighted log-likelihood summed over species: fitted half-normal -586,255, pooled half-normal -586,788, full dispersal (no clip) -594,226, no dispersal (hard clip at the mask, floor outside) -630,074. Versus the fitted half-normal (90% species-bootstrap CI): pooled -533 (-1,075 to -117), full -7,970 (-11,694 to -5,248), none -43,819 (-57,740 to -31,555). Share of species beating the fitted half-normal: pooled 0.8%, full 11.7%, none 2.5% (the fit is in-sample, so this favours it). Species gaining suitability by 2071-2100: no dispersal 30.8%, fitted half-normal 55.0%, full dispersal 78.3%. New records sit between the two classic bounds, much closer to full than to no dispersal, and the calibrated decay beats both.
