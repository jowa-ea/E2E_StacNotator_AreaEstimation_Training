#!/usr/bin/env python
# coding: utf-8
"""
Script equivalent of area_estimation_bungoma2025_large_groups.ipynb: a
single-round, multi-county variant of main.py's stratified sampling and
area/accuracy estimation, for training sessions with many participants.

main.py implements the statistically rigorous pilot + Neyman design (a
small pilot sample, annotated and used to compute per-stratum variance
priors, which size and allocate a second, optimally-allocated sample) in
two annotation rounds. That is a poor fit for a large training group: it
means pausing every participant mid-session while Sh/n_tot/allocation are
recomputed, then sending out a second, differently-sized batch.

This script instead sizes the (single) combined sample directly from the
training group itself -- Ni_total = n_participants * n_samples_per_participant,
split evenly across counties -- and allocates each county's share
proportionally across its own strata (optionally with a per-county
minimum), with no pilot round and no Neyman step. Bungoma and Muranga are
estimated independently (own pixel counts, own allocation, own area/
accuracy estimate) but drawn and annotated together: one stratified random
sample per county, merged into a single shuffled, single-id sample for one
combined annotation round. See the "Background" markdown cell in
area_estimation_bungoma2025_large_groups.ipynb for the full step-by-step
comparison between the two designs.

How the exported sample's rows get distributed to individual participants
is a logistics step outside this script.

Annotation happens outside this script, in a single round: the script
pauses at the one checkpoint (outputs/large_groups_sample_annotated.csv)
until that file exists.
"""

import os

import geopandas as gpd
import pandas as pd
from osgeo import gdal

import utils_ea_reprojection as ea
import utils_stratified_random_sampling as srs

gdal.UseExceptions()


def main():
    ## Paths
    base_dir = os.path.dirname(__file__)
    input_data_path = os.path.join(base_dir, "input_data")
    outputs_path = os.path.join(base_dir, "outputs")
    os.makedirs(outputs_path, exist_ok=True)

    # Each county's raw classification (EPSG:4326) and a short id_prefix used to
    # keep draw_samples_nested's internal _sample_key unique across counties
    # (both share the same stratum codes 0/1, so the prefix disambiguates them).
    counties = {
        "bungoma": {
            "raster": os.path.join(input_data_path, "BungomaCropland2025.tif"),
            "id_prefix": "BGM25LG",
        },
        "muranga": {
            "raster": os.path.join(input_data_path, "MurangaCropland2025.tif"),
            "id_prefix": "MUR25LG",
        },
    }

    ## User-defined parameters
    strata = [0, 1]  # 0 = non-cropland, 1 = cropland
    stratum_labels = {0: "Non-cropland", 1: "Cropland"}
    target_stratum = 1  # cropland: the stratum the final area estimates highlight

    # How the annotation tool codes the SAME classes in its own true_col
    # output -- may use different keys than stratum_labels, numeric or text
    # (here it's set to STAC Notator's own label-name spelling), but must
    # describe the exact same set of classes, spelled identically.
    stratum_true_labels = {"Non-cropland": "Non-cropland", "Cropland": "Cropland"}

    n_participants = 20  # training group size (combined, not per county)
    n_samples_per_participant = 40  # sample units per participant, across the combined pool of both counties
    confidence = 0.95  # confidence level used to report each county's final area estimate CI
    seed = 2025  # random seed, reused for both counties' draws and the combined shuffle

    # Set per county, after inspecting that county's own pixel counts (printed below) --
    # this script can't pause interactively like the notebook does, so re-tune and re-run
    # if these defaults don't fit the actual per-stratum weights.
    min_allocation = {
        "bungoma": 100,
        "muranga": 100,
    }

    # Column names expected in the annotation-tool export. Different tools may
    # name the point-id / true-class columns differently on export -- these
    # are the only two names that need to change to match whatever tool
    # actually produced the annotated CSV. true_col is set to STAC Notator's
    # own 'stacnotator_label_name' export column, so its raw export can be
    # used directly as the annotated file with no renaming.
    id_col = "id"
    true_col = "stacnotator_label_name"

    # Where to find the single, combined annotated file once STAC Notator work
    # is done -- update this if it's saved somewhere other than outputs/.
    annotated_path = os.path.join(outputs_path, "large_groups_sample_annotated.csv")

    # Step 1. Reproject each county's map to its own equal-area projection,
    # auto-centered on that raster's own extent (pixel counting and
    # equal-probability sampling both require equal-area pixels; EPSG:4326
    # pixels are not constant-area). Bungoma and Muranga are different
    # regions, so a single shared projection centered on one would distort
    # the other -- each gets its own.
    ea_rasters = {}
    for county, cfg in counties.items():
        proj_str = ea.derive_ea_proj_string(
            cfg["raster"], out_proj_path=os.path.join(outputs_path, f"{county}2025_ea_proj.txt")
        )
        print(f"{county}: auto-derived equal-area projection: {proj_str}")

        ea_raster = os.path.join(outputs_path, os.path.basename(cfg["raster"]).replace(".tif", "_ea.tif"))
        ea.raster_to_ea(cfg["raster"], ea_raster, proj_str, resampling_method="nearest")
        ea_rasters[county] = ea_raster

    # Step 2. Pixel counts / stratum weights, per county, on each county's own
    # equal-area raster.
    pixel_counts = {}
    for county in counties:
        pixel_counts_csv = os.path.join(outputs_path, f"pixelcounts_{county}2025.csv")
        pixel_counts[county] = srs.compute_pixel_counts(ea_rasters[county], strata=strata, output_csv=pixel_counts_csv)
        print(f"Pixel counts ({county}):\n", pixel_counts[county])

    # Step 3. Total sample size (Ni_total), sized directly from the training
    # group rather than a target CV, split evenly across counties and
    # allocated proportionally across each county's own strata.
    Ni_total = n_participants * n_samples_per_participant
    Ni = {county: Ni_total // len(counties) for county in counties}
    print(
        f"Ni_total = {n_participants} participants x {n_samples_per_participant} samples/participant "
        f"= {Ni_total}, split evenly per county: {Ni}"
    )

    allocation = {}
    for county in counties:
        allocation_csv = os.path.join(outputs_path, f"large_groups_sample_allocation_{county}.csv")
        allocation[county] = srs.allocate_proportional(
            pixel_counts[county], n_total=Ni[county], min_allocation=min_allocation[county], output_csv=allocation_csv
        )
        print(f"{county} allocation:", allocation[county])

    # Step 4. Draw one stratified random sample per county, reproject each to
    # EPSG:4326 (each county's own auto-derived EA CRS differs, so they can't
    # be combined as-is), tag with its county, merge, shuffle once, and
    # assign one shared id space -- then export a single combined CSV.
    county_gdfs = []
    for county, cfg in counties.items():
        gdf = srs.draw_samples_nested(ea_rasters[county], allocation[county], seed=seed, id_prefix=cfg["id_prefix"], v=True)
        gdf = gdf.to_crs(4326)
        gdf["county"] = county
        county_gdfs.append(gdf)

    combined_gdf = pd.concat(county_gdfs, ignore_index=True)
    combined_gdf = gpd.GeoDataFrame(combined_gdf, geometry="geometry", crs="EPSG:4326")

    combined_gdf = srs.shuffle_samples(combined_gdf, seed=seed)
    combined_gdf = srs.assign_ids(combined_gdf, id_col=id_col, v=True)

    # `county` (like `stratum`) is included in the export for the interpreter's own
    # context, but neither is assumed to survive STAC Notator's annotation round-trip --
    # only id_col/true_col are. Step 5 below re-reads this same file as the local,
    # authoritative record of each id's stratum/county, and merges in just true_col from
    # whatever comes back.
    sample_csv, _ = srs.export_sample_units(
        combined_gdf,
        os.path.join(outputs_path, "large_groups_sample_for_annotation.csv"),
        stratum_labels=stratum_labels, id_col=id_col, extra_cols=["county"], v=True,
    )

    # Checkpoint: annotate sample_csv (one combined file covering both
    # counties -- how its rows get divided among participants is outside
    # this script) in STAC Notator, save the result to annotated_path, then
    # re-run this script to continue.
    if not os.path.exists(annotated_path):
        print(
            f"\nSample exported to:\n  {sample_csv}\n"
            f"Annotate it in STAC Notator, save the result to:\n  {annotated_path}\n"
            f"(with columns '{id_col}' and '{true_col}'), then re-run this script to continue."
        )
        return

    # Step 5. Load the annotated sample. Only id_col/true_col are trusted from the
    # returned file -- stratum/county come from this notebook's own local record (the
    # sample it exported in the first place), merged in by id_col, the same way the
    # pilot+Neyman workflow's load_pilot_annotations() re-attaches `stratum` from its own
    # locally-kept pilot_gdf rather than trusting it to survive the round-trip.
    annotated_raw = pd.read_csv(annotated_path)
    missing = {id_col, true_col} - set(annotated_raw.columns)
    if missing:
        raise ValueError(f"'{annotated_path}' is missing required column(s): {missing}")
    annotated_raw = annotated_raw[[id_col, true_col]]

    sample_export_df = pd.read_csv(sample_csv)[[id_col, "stratum", "county"]]
    annotated_df = sample_export_df.merge(annotated_raw, on=id_col, how="left")
    annotated_df = srs.relabel_true_stratum(annotated_df, true_col, stratum_labels, stratum_true_labels)

    annotated_resolved_path = os.path.join(outputs_path, "large_groups_sample_annotated_resolved.csv")
    annotated_df.to_csv(annotated_resolved_path, index=False)

    pred, true = {}, {}
    for county in counties:
        county_resolved_path = os.path.join(outputs_path, f"large_groups_sample_annotated_resolved_{county}.csv")
        annotated_df[annotated_df["county"] == county].to_csv(county_resolved_path, index=False)
        pred[county], true[county], _ = srs.load_full_annotations(
            county_resolved_path, id_col=id_col, stratum_col="stratum", true_col=true_col
        )

    # Step 6. Design-based area/accuracy estimate (Olofsson et al., 2014),
    # per county, each against that county's own pixel counts.
    metrics = {}
    overall_accuracy = {}
    for county in counties:
        metrics_csv = os.path.join(outputs_path, f"area_estimates_{county}2025_large_groups.csv")
        metrics[county] = srs.compute_stratified_random_sampling_metrics(
            pixel_counts[county], pred[county], true[county], output_csv=metrics_csv, v=True
        )

        accuracy_csv = os.path.join(outputs_path, f"accuracy_metrics_{county}2025_large_groups.csv")
        _, overall_accuracy[county] = srs.compute_accuracy_metrics(
            pixel_counts[county], pred[county], true[county], output_csv=accuracy_csv, v=True
        )

    print("\n=== 2025-season cropland area estimates, large-group sample design ===")
    for county in counties:
        cropland = metrics[county].loc[target_stratum]
        print(f"--- {county.capitalize()} County ---")
        print(
            f"  Cropland area: {cropland['Area_ha']:.0f} ha +/- {cropland['CI_Ha']:.0f} ha "
            f"({cropland['CI%'] * 100:.1f}% relative precision at {int(confidence * 100)}% confidence; "
            f"from Ni={Ni[county]} units, not a pre-set CV target)"
        )
        print(f"  Overall accuracy: {overall_accuracy[county]['O']:.3f} +/- {overall_accuracy[county]['CI']:.3f}")


if __name__ == "__main__":
    main()
