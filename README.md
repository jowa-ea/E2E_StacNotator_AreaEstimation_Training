# E2E STAC Notator + Area Estimation -- Kenya (2025 season)

| Notebook | Open in Colab |
| --- | --- |
| `area_estimation_bungoma2025.ipynb` -- pilot + Neyman design (statistically optimal, two annotation rounds, Bungoma County) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jowa-ea/E2E_StacNotator_AreaEstimation_Training/blob/main/area_estimation_bungoma2025.ipynb) |
| `area_estimation_bungoma2025_large_groups.ipynb` -- large-group design (single annotation round, sized to the training group, Bungoma & Muranga Counties) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jowa-ea/E2E_StacNotator_AreaEstimation_Training/blob/main/area_estimation_bungoma2025_large_groups.ipynb) |

Operationalizes the stratified random sampling methodology from the
[Sample-Based Map Accuracy and Area Estimation workshop](https://github.com/jowa-ea/EstellaKenyaSamplingClass)
to estimate **cropland area for the 2025 season in Bungoma and Muranga
Counties, Kenya**, using real classified maps rather than the workshop's
synthetic true/predicted map pair.

## Scope

This repository takes a real classified cropland map -- with no
pre-existing reference data -- through to a defensible, sample-based
cropland area and accuracy estimate. Reference (ground-truth) labels are
collected by interpreting each sampled point against satellite imagery in
**STAC Notator**, outside this codebase. Two sample-design variants are
provided, sharing the same map import/reprojection, CSV export/import, and
area/accuracy estimation code:

- **Pilot + Neyman** (`area_estimation_bungoma2025.ipynb` / `main.py`) --
  the statistically rigorous design, across two annotation rounds:
  - **Round 1 (pilot):** a small sample used only to prime Neyman allocation.
  - **Round 2:** just the units the pilot didn't already cover (e.g. 500 of
    a 600-unit Neyman sample, if the pilot covered 100) -- the pilot's own
    annotated units are reused rather than re-annotated.
- **Large groups** (`area_estimation_bungoma2025_large_groups.ipynb` /
  `main_large_groups.py`) -- a single-round design for training sessions
  with many participants: no pilot, no Neyman step. Covers **both Bungoma
  and Muranga Counties**, estimated independently (own pixel counts, own
  allocation, own area/accuracy estimate) but sampled and annotated
  together: a stratified random sample is drawn separately per county,
  then merged, shuffled once, and given one shared id space, so
  participants only work through a single combined CSV. Total sample size
  is set directly from the training group
  (`n_participants` x `n_samples_per_participant`, split evenly across
  counties) and allocated proportionally across each county's own strata
  (with a per-county minimum). See that notebook's **Background** section
  for the full step-by-step comparison.

## Workflow -- pilot + Neyman (`area_estimation_bungoma2025.ipynb`, `main.py`)

1. **Reproject to an equal-area projection.** A Lambert Azimuthal
   Equal-Area (LAEA) projection is auto-derived and centered on the map's
   own extent, so pixel counts (hectares) and pixel-based sampling are not
   distorted by EPSG:4326's non-constant pixel area.
2. **Set the accuracy threshold.** The user sets a target coefficient of
   variation (`cv_target`) and confidence level for the cropland area
   estimate, plus the random `seed` reused for every draw below, and the
   column names / file paths expected from the annotation tool.
3. **Pilot sample.** A small, proportionally-allocated stratified random
   sample (user-defined size) is drawn and exported for annotation in STAC
   Notator, to obtain prior per-stratum variances (Neyman priors).
   *(Round-1 checkpoint.)*
4. **Sample size and allocation.** The annotated pilot's priors are used to
   compute the total sample size (N) and per-stratum allocation (n_h)
   needed to hit the target CV (Neyman allocation).
5. **Full sample, nested with the pilot.** The full sample is drawn with
   the *same random seed* as the pilot, using an independent random stream
   per stratum -- so the pilot's sample units, and once annotated their
   labels, are reused inside the full sample rather than re-drawn or
   re-annotated. Only the units NOT already covered by the pilot are
   exported for a second annotation round. *(Round-2 checkpoint.)*
6. **Concatenate both annotation rounds** into one fully annotated sample.
7. **Design-based area estimate**, with uncertainty (Olofsson et al., 2014).
8. **Map accuracy** (overall, user's, producer's), with uncertainty.

## Workflow -- large groups (`area_estimation_bungoma2025_large_groups.ipynb`, `main_large_groups.py`)

1. **Reproject each county's map to its own equal-area projection.** Same
   as above, run independently per county (Bungoma's and Muranga's
   auto-derived projections differ, since they're centered on different
   regions).
2. **Set strata and the shared sample-size parameters.** The user sets
   `n_participants` and `n_samples_per_participant` (instead of
   `cv_target`/`pilot_n`) for the training group as a whole, plus
   `confidence`, `seed`, and the annotation-tool column names / file path.
3. **Pixel counts, per county.**
4. **Sample size and allocation, per county.** The combined total sample
   size `Ni_total = n_participants * n_samples_per_participant` is split
   evenly across counties; the user then sets a per-county `min_allocation`
   -- after looking at that county's own pixel counts from step 3 -- and
   each county's share is allocated across its own strata proportionally
   to stratum area (`Wi`), with each stratum getting at least that
   county's `min_allocation`.
5. **Draw one sample per county, merge, shuffle once, export one combined
   file.** Each county's sample is drawn independently (own raster, own
   allocation), reprojected to EPSG:4326, tagged with its county, then
   merged with the other county's, shuffled together, and given one shared
   id space -- so the exported CSV is a single file mixing both counties'
   units, for one combined annotation round in STAC Notator. *(Single
   checkpoint.)* How the exported rows get distributed to individual
   participants is a logistics step outside this notebook/script.
6. **Load the annotated sample and split it by county.** True labels are
   relabeled onto the map's own coding; each row's county is then
   recovered from a small **local id-to-county lookup file** written in
   step 5 -- not a `county` column trusted to survive STAC Notator's own
   export -- and used to split the combined annotated data back into one
   set per county.
7. **Design-based area estimate, per county**, with uncertainty (Olofsson
   et al., 2014), each county compared only against its own pixel counts.
8. **Map accuracy (overall, user's, producer's), per county**, with
   uncertainty.

## Content

1. `area_estimation_bungoma2025.ipynb` -- the pilot + Neyman workflow above
   in one notebook, with the narrative/rationale for each step. The two
   checkpoints (pilot annotation, round-2 annotation) are cells the user
   re-runs once the corresponding annotated file is saved; the file paths
   for both are user-editable variables (`pilot_annotated_path`,
   `round2_annotated_path`) so they don't have to live in `outputs/`.
2. `main.py` -- script equivalent of (1) for local, non-interactive runs.
   Stops at each checkpoint (`outputs/pilot_sample_annotated.csv`, then
   `outputs/bungoma2025_sample_units_round2_annotated.csv`) until the file
   exists; re-run after saving it to continue.
3. `area_estimation_bungoma2025_large_groups.ipynb` -- the large-groups
   workflow above in one notebook, covering Bungoma and Muranga Counties
   independently, including a **Background** section walking through the
   pilot + Neyman process step by step and explaining why/how this variant
   simplifies it. The single checkpoint (annotation) is a cell the user
   re-runs once the annotated file is saved, at `annotated_path`.
4. `main_large_groups.py` -- script equivalent of (3) for local,
   non-interactive runs. Stops at its one checkpoint
   (`outputs/large_groups_sample_annotated.csv`) until the file exists;
   re-run after saving it to continue.
5. `utils_ea_reprojection.py` -- auto-derive and apply an equal-area
   projection centered on a raster's own extent.
6. `utils_stratified_random_sampling.py` -- pixel counting, proportional
   and Neyman allocation, the seed-nested pilot/full sample draw,
   CSV export (optionally carrying through extra columns, e.g. the
   large-groups workflow's `county` tag, via `extra_cols`) with
   pilot-annotation reuse, combining the two annotation rounds, and the
   final area/accuracy estimators. Shared by both notebooks/scripts above.
   The annotation-tool's id/true-class column names (`id_col`/`true_col`)
   are parameters throughout, since different tools may export them under
   different names.
7. `input_data/BungomaCropland2025.tif` -- 2025-season Bungoma County
   cropland classification (EPSG:4326).
8. `input_data/MurangaCropland2025.tif` -- 2025-season Muranga County
   cropland classification (EPSG:4326). Used only by the large-groups
   workflow.

Outputs (reprojected raster, pixel counts, sample exports, annotated
samples, final area/accuracy estimates) are written to `outputs/`, created
on first run.

## Running in Colab

Click either badge above to open that notebook directly in Google Colab --
no local setup required. Each notebook's first two setup cells install
GDAL and the geospatial Python stack, then clone this (public) repository,
including `git-lfs` for the input raster, into the Colab runtime.

## Running locally

Install dependencies (`gdal`, `numpy`, `pandas`, `geopandas`, `shapely`,
`pyproj`, `scikit-learn`) and open either notebook (skip the two
Colab-setup cells at the top -- they assume a Debian/Colab shell), or run
`python main.py` (pilot + Neyman) / `python main_large_groups.py`
(large groups) for the non-interactive equivalents.


