#!/usr/bin/env python
# coding: utf-8
"""
Script equivalent of area_estimation_bungoma2025.ipynb: designs the
stratified random reference sample, then computes the final cropland
area/accuracy estimate for the 2025 season in Bungoma County, Kenya,
operationalizing the methodology from the Sample-Based Map Accuracy and
Area Estimation workshop (https://github.com/jowa-ea/EstellaKenyaSamplingClass)
on a real (not synthetic) classified map.

Annotation happens outside this script, in two rounds, each a checkpoint
the script pauses at until the corresponding file exists in outputs/:
  1. The small pilot sample (needed to compute Neyman priors).
  2. The round-2 sample -- only the units NOT already covered by the pilot
     (needed, combined with the pilot's own annotations, for the final
     area/accuracy estimate).
"""

import os
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
    target_stratum = 1  # cropland: the stratum the accuracy threshold applies to

    # How the annotation tool codes the SAME classes in its own true_stratum
    # output -- may use different numeric keys than stratum_labels (e.g. the
    # tool might call cropland "2" while the map calls it "1"), but must
    # describe the exact same set of classes, spelled identically. Used to
    # relabel true_stratum onto the map's own coding so pred/true always
    # mean the same thing at the same numeric value.
    stratum_true_labels = {1: "Non-cropland", 2: "Cropland"}
    cv_target = 0.10  # target relative precision (CV) on the cropland area estimate
    confidence = 0.95  # confidence level for the area estimate's CI
    pilot_n = 100  # pilot sample size
    seed = 2025  # single seed reused for EVERY random draw below (pilot, full sample, shuffle)
    #             -- this is what makes the pilot's units nest inside the full sample.

    # Column names expected in annotation-tool exports. Different tools may
    # name the point-id / true-class columns differently on export -- these
    # are the only two names that need to change to match whatever tool
    # actually produced the annotated CSV (STAC Notator or otherwise).
    id_col = "id"
    true_col = "true_stratum"

    # Where to find each round's annotated file once STAC Notator work is
    # done -- update these if they're saved somewhere other than outputs/.
    pilot_annotated_path = os.path.join(outputs_path, "pilot_sample_annotated.csv")
    round2_annotated_path = os.path.join(outputs_path, "bungoma2025_sample_units_round2_annotated.csv")

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

    # Step 3. Small proportionally-allocated pilot sample, to obtain prior
    # per-stratum variances (Sh) for Neyman sample-size planning
    pilot_allocation_csv = os.path.join(outputs_path, "pilot_allocation.csv")
    pilot_allocation = srs.allocate_proportional(
        pixel_counts, n_total=pilot_n, min_allocation=0, output_csv=pilot_allocation_csv
    )
    print("Pilot allocation:", pilot_allocation)

    pilot_gdf = srs.draw_samples_nested(
        crop_map_2025_ea, pilot_allocation, seed=seed, id_prefix="BGM25", v=True
    )

    # Shuffle row order first, then assign the exported `id` from that shuffled
    # order -- assigning id before shuffling would let its value still betray
    # each unit's original per-stratum draw order, even after the rows
    # themselves were reordered.
    pilot_gdf = srs.shuffle_samples(pilot_gdf, seed=seed)
    pilot_gdf = srs.assign_ids(pilot_gdf, id_col=id_col, id_prefix="BGM25", v=True)

    pilot_csv, pilot_geojson, _ = srs.export_sample_units(
        pilot_gdf,
        os.path.join(outputs_path, "pilot_sample_for_annotation.csv"),
        os.path.join(outputs_path, "pilot_sample_for_annotation.geojson"),
        stratum_labels=stratum_labels, id_col=id_col, v=True,
    )

    # Checkpoint (round 1): annotate pilot_sample_for_annotation.csv in
    # STAC Notator and save the result (with an added `true_col` column) to
    # pilot_annotated_path, then re-run the script to continue.
    if not os.path.exists(pilot_annotated_path):
        print(
            f"\nPilot sample exported to:\n  {pilot_csv}\n  {pilot_geojson}\n"
            f"Annotate it in STAC Notator, save the result to:\n  {pilot_annotated_path}\n"
            f"(with columns '{id_col}' and '{true_col}'), then re-run this script to continue."
        )
        return

    pilot_pred, pilot_true, pilot_annotated = srs.load_pilot_annotations(
        pilot_gdf, pilot_annotated_path, id_col=id_col, true_col=true_col,
        stratum_labels=stratum_labels, true_labels=stratum_true_labels,
    )

    # Step 4. Neyman priors, total sample size, and per-stratum allocation
    Sh_csv = os.path.join(outputs_path, "pilot_stratum_variances.csv")
    Sh = srs.compute_pilot_variances(strata, pilot_pred, pilot_true, output_csv=Sh_csv, v=True)

    n_tot = srs.compute_neyman_sample_size(
        pixel_counts, Sh, cv_target=cv_target, confidence=confidence, target_stratum=target_stratum, v=True
    )

    neyman_allocation_csv = os.path.join(outputs_path, "neyman_allocation.csv")
    neyman_allocation = srs.allocate_neyman(
        pixel_counts, Sh, n_tot, output_allocation_csv=neyman_allocation_csv, v=True
    )

    allocation = srs.reconcile_allocation_with_pilot(neyman_allocation, pilot_allocation, v=True)

    # Step 5. Draw the full sample with the SAME seed, so the pilot's units
    # (and their annotations) nest inside it and don't need re-annotating.
    # E.g. if Neyman calls for 600 units total and the pilot already covered
    # 100 of them, only the other 500 get exported for round-2 annotation.
    full_gdf = srs.draw_samples_nested(crop_map_2025_ea, allocation, seed=seed, id_prefix="BGM25", v=True)
    # Matched on `_sample_key` (the stable per-stratum draw-order key), not
    # `id_col` -- `id_col` hasn't been (re)assigned for full_gdf yet, and is
    # reassigned independently per round anyway (see below).
    assert set(pilot_gdf["_sample_key"]) <= set(full_gdf["_sample_key"]), \
        "pilot sample did not nest inside the full sample"
    print(f"{len(pilot_gdf)} pilot units confirmed nested inside the {len(full_gdf)}-unit full sample.")

    full_gdf = srs.shuffle_samples(full_gdf, seed=seed)
    full_gdf = srs.assign_ids(full_gdf, id_col=id_col, id_prefix="BGM25", v=True)

    # Master export: all n_tot units, with the pilot's own annotations
    # already filled in -- kept as the bookkeeping file combined with the
    # round-2 annotations in Step 6. `pilot_truth` is matched in on
    # `_sample_key`, not `id_col`, since `id_col` was just reassigned above
    # from this round's own shuffle and no longer matches the pilot's.
    master_csv, master_geojson, _ = srs.export_sample_units(
        full_gdf,
        os.path.join(outputs_path, "bungoma2025_sample_units_master.csv"),
        os.path.join(outputs_path, "bungoma2025_sample_units_master.geojson"),
        stratum_labels=stratum_labels, pilot_truth=pilot_annotated, id_col=id_col, v=True,
    )

    # Round-2 export: only the units NOT already annotated in the pilot.
    round2_gdf = full_gdf[~full_gdf["_sample_key"].isin(pilot_annotated["_sample_key"])].copy()
    round2_csv, round2_geojson, _ = srs.export_sample_units(
        round2_gdf,
        os.path.join(outputs_path, "bungoma2025_sample_units_round2_for_annotation.csv"),
        os.path.join(outputs_path, "bungoma2025_sample_units_round2_for_annotation.geojson"),
        stratum_labels=stratum_labels, id_col=id_col, v=True,
    )
    print(
        f"\n{len(pilot_annotated)} unit(s) already annotated from the pilot round; "
        f"{len(round2_gdf)} unit(s) exported for round-2 annotation:\n  {round2_csv}\n  {round2_geojson}"
    )

    # Checkpoint (round 2): annotate the round-2 file in STAC Notator and
    # save the result (with an added `true_col` column) to
    # round2_annotated_path, then re-run the script to continue to the
    # final estimate.
    if not os.path.exists(round2_annotated_path):
        print(
            f"Annotate the round-2 sample in STAC Notator, save the result to:\n  {round2_annotated_path}\n"
            f"(with columns '{id_col}' and '{true_col}'), then re-run this script to compute "
            "the final area/accuracy estimate."
        )
        return

    # Step 6. Concatenate pilot + round-2 annotations into one fully
    # annotated sample, and compute the final design-based area/accuracy
    # estimate (Olofsson et al., 2014).
    annotated_path = os.path.join(outputs_path, "bungoma2025_sample_units_annotated.csv")
    srs.combine_annotation_rounds(
        master_csv, round2_annotated_path, annotated_path, id_col=id_col, true_col=true_col,
        stratum_labels=stratum_labels, true_labels=stratum_true_labels,
    )
    pred, true, annotated_df = srs.load_full_annotations(
        annotated_path, id_col=id_col, stratum_col="stratum", true_col=true_col
    )

    metrics_csv = os.path.join(outputs_path, "area_estimates_bungoma2025.csv")
    metrics = srs.compute_stratified_random_sampling_metrics(pixel_counts, pred, true, output_csv=metrics_csv, v=True)

    accuracy_csv = os.path.join(outputs_path, "accuracy_metrics_bungoma2025.csv")
    accuracy_metrics, overall_accuracy = srs.compute_accuracy_metrics(
        pixel_counts, pred, true, output_csv=accuracy_csv, v=True
    )

    cropland = metrics.loc[target_stratum]
    print("\n=== 2025-season cropland area estimate, Bungoma County ===")
    print(
        f"Cropland area: {cropland['Area_ha']:.0f} ha +/- {cropland['CI_Ha']:.0f} ha "
        f"({cropland['CI%'] * 100:.1f}% relative precision at {int(confidence * 100)}% confidence; "
        f"target was {cv_target * 100:.1f}%)"
    )
    print(f"Overall accuracy: {overall_accuracy['O']:.3f} +/- {overall_accuracy['CI']:.3f}")


if __name__ == "__main__":
    main()
