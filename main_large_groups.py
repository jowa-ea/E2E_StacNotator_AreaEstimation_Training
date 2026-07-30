#!/usr/bin/env python
# coding: utf-8
"""
Script equivalent of area_estimation_bungoma2025_large_groups.ipynb: a
single-round variant of main.py's stratified sampling and area/accuracy
estimation, for training sessions with many participants.

main.py implements the statistically rigorous pilot + Neyman design (a
small pilot sample, annotated and used to compute per-stratum variance
priors, which size and allocate a second, optimally-allocated sample) in
two annotation rounds. That is a poor fit for a large training group: it
means pausing every participant mid-session while Sh/n_tot/allocation are
recomputed, then sending out a second, differently-sized batch.

This script instead sizes the (single) sample directly from the training
group itself -- Ni = n_participants * n_samples_per_participant -- and
allocates it proportionally across strata (optionally with a per-stratum
minimum), with no pilot round and no Neyman step. See the "Background"
markdown cell in area_estimation_bungoma2025_large_groups.ipynb for the
full step-by-step comparison between the two designs.

Annotation happens outside this script, in a single round: the script
pauses at the one checkpoint (outputs/large_groups_sample_annotated.csv)
until that file exists.
"""

import os

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

    crop_map_2025 = os.path.join(input_data_path, "BungomaCropland2025.tif")  # EPSG:4326

    ## User-defined parameters
    strata = [0, 1]  # 0 = non-cropland, 1 = cropland
    stratum_labels = {0: "Non-cropland", 1: "Cropland"}
    target_stratum = 1  # cropland: the stratum the final area estimate highlights

    # How the annotation tool codes the SAME classes in its own true_col
    # output -- may use different keys than stratum_labels, numeric or text
    # (here it's set to STAC Notator's own label-name spelling), but must
    # describe the exact same set of classes, spelled identically.
    stratum_true_labels = {"Non-cropland": "Non-cropland", "Cropland": "Cropland"}

    n_participants = 30  # number of training participants
    n_samples_per_participant = 20  # sample units each participant annotates
    min_allocation = 5  # floor (sample units) applied to every stratum during proportional allocation
    confidence = 0.95  # confidence level used to report the final area estimate's CI
    seed = 2025  # random seed, reused for the sample draw and row shuffle

    # Column names expected in the annotation-tool export. Different tools may
    # name the point-id / true-class columns differently on export -- these
    # are the only two names that need to change to match whatever tool
    # actually produced the annotated CSV. true_col is set to STAC Notator's
    # own 'stacnotator_label_name' export column, so its raw export can be
    # used directly as the annotated file with no renaming.
    id_col = "id"
    true_col = "stacnotator_label_name"

    # Where to find the single annotated file once STAC Notator work is
    # done -- update this if it's saved somewhere other than outputs/.
    annotated_path = os.path.join(outputs_path, "large_groups_sample_annotated.csv")

    # Step 1. Reproject to an equal-area projection auto-centered on the
    # map's own extent (pixel counting and equal-probability sampling both
    # require equal-area pixels; EPSG:4326 pixels are not constant-area)
    proj_str = ea.derive_ea_proj_string(
        crop_map_2025, out_proj_path=os.path.join(outputs_path, "bungoma2025_ea_proj.txt")
    )
    print("Auto-derived equal-area projection:", proj_str)

    crop_map_2025_ea = os.path.join(outputs_path, "BungomaCropland2025_ea.tif")
    ea.raster_to_ea(crop_map_2025, crop_map_2025_ea, proj_str, resampling_method="nearest")

    # Step 2. Pixel counts / stratum weights on the equal-area raster
    pixel_counts_csv = os.path.join(outputs_path, "pixelcounts_bungoma2025.csv")
    pixel_counts = srs.compute_pixel_counts(crop_map_2025_ea, strata=strata, output_csv=pixel_counts_csv)
    print("Pixel counts:\n", pixel_counts)

    # Step 3. Total sample size (Ni), sized directly from the training group
    # rather than a target CV, allocated proportionally across strata.
    Ni = n_participants * n_samples_per_participant
    print(f"Total sample size Ni = {n_participants} participants x {n_samples_per_participant} samples/participant = {Ni}")

    allocation_csv = os.path.join(outputs_path, "large_groups_sample_allocation.csv")
    allocation = srs.allocate_proportional(
        pixel_counts, n_total=Ni, min_allocation=min_allocation, output_csv=allocation_csv
    )
    print("Allocation:", allocation)

    # Step 4. Draw and export the (single) sample for annotation.
    full_gdf = srs.draw_samples_nested(crop_map_2025_ea, allocation, seed=seed, id_prefix="BGM25LG", v=True)

    full_gdf = srs.shuffle_samples(full_gdf, seed=seed)
    full_gdf = srs.assign_ids(full_gdf, id_col=id_col, v=True)

    sample_csv, _ = srs.export_sample_units(
        full_gdf,
        os.path.join(outputs_path, "large_groups_sample_for_annotation.csv"),
        stratum_labels=stratum_labels, id_col=id_col, v=True,
    )

    # Checkpoint: split sample_csv across the n_participants participants,
    # annotate in STAC Notator, combine everyone's results into a single
    # CSV with columns [id_col, true_col], save to annotated_path, then
    # re-run this script to continue.
    if not os.path.exists(annotated_path):
        print(
            f"\nSample exported to:\n  {sample_csv}\n"
            f"Split it across {n_participants} participants, annotate in STAC Notator, combine the "
            f"results and save to:\n  {annotated_path}\n"
            f"(with columns '{id_col}' and '{true_col}'), then re-run this script to continue."
        )
        return

    # Step 5. Load the annotated sample, relabeling the annotation tool's
    # raw true_col values onto the map's own stratum coding.
    annotated_raw = pd.read_csv(annotated_path)
    annotated_df = srs.relabel_true_stratum(annotated_raw, true_col, stratum_labels, stratum_true_labels)

    annotated_resolved_path = os.path.join(outputs_path, "large_groups_sample_annotated_resolved.csv")
    annotated_df.to_csv(annotated_resolved_path, index=False)

    pred, true, _ = srs.load_full_annotations(
        annotated_resolved_path, id_col=id_col, stratum_col="stratum", true_col=true_col
    )

    # Step 6. Design-based area/accuracy estimate (Olofsson et al., 2014).
    metrics_csv = os.path.join(outputs_path, "area_estimates_bungoma2025_large_groups.csv")
    metrics = srs.compute_stratified_random_sampling_metrics(pixel_counts, pred, true, output_csv=metrics_csv, v=True)

    accuracy_csv = os.path.join(outputs_path, "accuracy_metrics_bungoma2025_large_groups.csv")
    accuracy_metrics, overall_accuracy = srs.compute_accuracy_metrics(
        pixel_counts, pred, true, output_csv=accuracy_csv, v=True
    )

    cropland = metrics.loc[target_stratum]
    print("\n=== 2025-season cropland area estimate, Bungoma County (large-group sample design) ===")
    print(
        f"Cropland area: {cropland['Area_ha']:.0f} ha +/- {cropland['CI_Ha']:.0f} ha "
        f"({cropland['CI%'] * 100:.1f}% relative precision at {int(confidence * 100)}% confidence; "
        f"from Ni={Ni} units, not a pre-set CV target)"
    )
    print(f"Overall accuracy: {overall_accuracy['O']:.3f} +/- {overall_accuracy['CI']:.3f}")


if __name__ == "__main__":
    main()
