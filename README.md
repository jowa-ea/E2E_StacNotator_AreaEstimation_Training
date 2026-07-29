# E2E STAC Notator + Area Estimation -- Bungoma County, Kenya (2025 season)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jowa-ea/E2E_StacNotator_AreaEstimation_Training/blob/main/area_estimation_bungoma2025.ipynb)

Operationalizes the stratified random sampling methodology from the
[Sample-Based Map Accuracy and Area Estimation workshop](https://github.com/jowa-ea/EstellaKenyaSamplingClass)
to estimate **cropland area for the 2025 season in Bungoma County, Kenya**,
using a real classified map rather than the workshop's synthetic
true/predicted map pair.

## Scope

This repository takes a real classified cropland map -- with no
pre-existing reference data -- through to a defensible, sample-based
cropland area and accuracy estimate. Reference (ground-truth) labels are
collected by interpreting each sampled point against satellite imagery in
**STAC Notator**, outside this codebase, across two annotation rounds:

- **Round 1 (pilot):** a small sample used only to prime Neyman allocation.
- **Round 2:** just the units the pilot didn't already cover (e.g. 500 of a
  600-unit Neyman sample, if the pilot covered 100) -- the pilot's own
  annotated units are reused rather than re-annotated.

## Workflow

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

## Content

1. `area_estimation_bungoma2025.ipynb` -- the full workflow above in one
   notebook, with the narrative/rationale for each step. The two
   checkpoints (pilot annotation, round-2 annotation) are cells the user
   re-runs once the corresponding annotated file is saved; the file paths
   for both are user-editable variables (`pilot_annotated_path`,
   `round2_annotated_path`) so they don't have to live in `outputs/`.
2. `main.py` -- script equivalent for local, non-interactive runs. Stops
   at each checkpoint (`outputs/pilot_sample_annotated.csv`, then
   `outputs/bungoma2025_sample_units_round2_annotated.csv`) until the file
   exists; re-run after saving it to continue.
3. `utils_ea_reprojection.py` -- auto-derive and apply an equal-area
   projection centered on a raster's own extent.
4. `utils_stratified_random_sampling.py` -- pixel counting, proportional
   and Neyman allocation, the seed-nested pilot/full sample draw,
   CSV/GeoJSON export with pilot-annotation reuse, combining the two
   annotation rounds, and the final area/accuracy estimators. The
   annotation-tool's id/true-class column names (`id_col`/`true_col`) are
   parameters throughout, since different tools may export them under
   different names.
5. `input_data/BungomaCropland2025.tif` -- 2025-season Bungoma County
   cropland classification (EPSG:4326).

Outputs (reprojected raster, pixel counts, sample exports, annotated
samples, final area/accuracy estimates) are written to `outputs/`, created
on first run.

## Running in Colab

Click the badge above to open the notebook directly in Google Colab -- no
local setup required. The first two setup cells install GDAL and the
geospatial Python stack, then clone this (public) repository, including
`git-lfs` for the input raster, into the Colab runtime.

## Running locally

Install dependencies (`gdal`, `numpy`, `pandas`, `geopandas`, `shapely`,
`pyproj`, `scikit-learn`) and open the notebook (skip the two Colab-setup
cells at the top -- they assume a Debian/Colab shell), or run `python
main.py` for the non-interactive equivalent.


