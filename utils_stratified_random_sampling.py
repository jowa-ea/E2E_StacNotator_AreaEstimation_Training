#!/usr/bin/env python
# coding: utf-8
"""
Stratified random sampling design for cropland area estimation: pixel
counting, a small pilot sample to prime Neyman allocation, Neyman
sample-size/allocation, and a nested full-sample draw whose pilot subset is
guaranteed to reuse the pilot's own sampled units (and, once annotated,
their reference labels). Adapted from the workshop utilities in
https://github.com/jowa-ea/EstellaKenyaSamplingClass, with the sampling and
export steps reworked for an operational setting with no pre-existing
reference data: this map's true classes come from external photo
interpretation (STAC Notator), not a synthetic reference raster.
"""

import os

import numpy as np
import pandas as pd
from osgeo import gdal
from shapely.geometry import Point
import geopandas as gpd

gdal.UseExceptions()


def compute_pixel_counts(input_map, strata=None, output_csv=None):
    """
    Compute pixel counts (in hectares) for each stratum in a raster map,
    and include stratum weights (Wi). Requires an equal-area CRS (see
    `utils_ea_reprojection.raster_to_ea`) for the hectare conversion to be
    meaningful across the whole raster.

    Parameters
    ----------
    input_map : str
        Path to raster file.
    strata : list, optional
        List of strata (class values) to compute pixel counts for.
        If None, all unique values in the raster will be used.
    output_csv : str, optional
        Path to save results as CSV.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: ['Stratum', 'Area_ha', 'Wi']
    """
    ds = gdal.Open(input_map)
    if ds is None:
        raise FileNotFoundError(f"Could not open raster: {input_map}")

    arr = ds.GetRasterBand(1).ReadAsArray().astype("uint8")

    if strata is None:
        strata = np.unique(arr)

    gt = ds.GetGeoTransform()
    pixel_area_ha = (gt[1] * abs(gt[5])) / 10000  # m^2 -> ha

    areas = [np.count_nonzero(arr == s) * pixel_area_ha for s in strata]
    total_area = sum(areas)
    weights = [area / total_area for area in areas]

    df = pd.DataFrame({"Stratum": strata, "Area_ha": areas, "Wi": weights})

    if output_csv:
        df.to_csv(output_csv, index=False)

    return df


def allocate_proportional(pixel_counts, n_total, min_allocation=0, output_csv=None):
    """
    Allocate `n_total` sample units to strata proportionally to their area
    (Wi), ensuring at least `min_allocation` per stratum. Used for the small
    pilot sample -- proportional allocation is a neutral default before any
    per-stratum variance estimate (Sh) exists to inform Neyman allocation.

    Parameters
    ----------
    pixel_counts : pd.DataFrame or dict
        DataFrame with columns ['Stratum', 'Area_ha'], or dict
        {stratum: area}.
    n_total : int
        Total number of sample units to allocate.
    min_allocation : int
    output_csv : str, optional

    Returns
    -------
    dict {stratum: n_h}
    """
    if isinstance(pixel_counts, pd.DataFrame):
        pixel_counts = dict(zip(pixel_counts["Stratum"], pixel_counts["Area_ha"]))

    total = sum(pixel_counts.values())
    Wi = {k: v / total for k, v in pixel_counts.items()}

    allocation = {k: min_allocation for k in pixel_counts}
    remaining = n_total - sum(allocation.values())
    if remaining > 0:
        for k in Wi:
            allocation[k] += int(remaining * Wi[k])

        diff = n_total - sum(allocation.values())
        if diff > 0:
            for k, _ in sorted(Wi.items(), key=lambda x: -x[1]):
                if diff <= 0:
                    break
                allocation[k] += 1
                diff -= 1

    if output_csv:
        pd.DataFrame(list(allocation.items()), columns=["Stratum", "Sample units"]).to_csv(
            output_csv, index=False
        )

    return allocation


def draw_samples_nested(map_path, sample_allocation, seed=0, id_prefix="SU", out_gpkg=None, v=False):
    """
    Draw a stratified random sample of pixel-center points from a
    categorical raster, using an *independent random stream per stratum*,
    keyed on (seed, stratum).

    Why per-stratum streams: this is what makes two allocations that share
    the same `seed` nest inside one another. If allocation A requests n_h
    units in stratum h and allocation B requests n_h' >= n_h units in that
    *same* stratum, the first n_h units accepted for B are -- pixel for
    pixel -- identical to A's, because both start from the same per-stratum
    RNG state and follow the same accept/reject sequence. A single RNG
    stream shared across strata would NOT have this property: consuming a
    different number of draws in one stratum shifts the RNG state every
    later stratum sees, so only the first-processed stratum would nest.
    Seeding each stratum independently decouples it from what happens in
    every other stratum.

    In this workflow, call this function once for the small pilot
    allocation and again later for the full (Neyman) allocation, with the
    same `seed` and the same map -- the pilot's sample units, and once
    annotated their reference labels, are then reusable within the full
    sample (see `export_sample_units(..., pilot_truth=...)`).

    Parameters
    ----------
    map_path : str
        Path to categorical raster (e.g. an equal-area reprojected map).
    sample_allocation : dict {stratum_value: n_samples}
    seed : int
        Base random seed.
    id_prefix : str
        Each unit gets an internal `_sample_key`,
        f"{id_prefix}_{stratum}_{k:05d}" (k = 1-indexed acceptance order
        within the stratum) -- nested draws share keys for their shared
        units. This is NOT the id exported for annotation: it deliberately
        encodes stratum and draw order, which is fine for internal
        matching (e.g. reusing pilot annotations inside the full sample)
        but would defeat row shuffling if shown to an interpreter. The
        exported `id` is assigned later, after shuffling, by `assign_ids`.
    out_gpkg : str, optional
        If given, also save the result as a GeoPackage.
    v : bool

    Returns
    -------
    gpd.GeoDataFrame with columns ['_sample_key', 'stratum', 'row', 'col',
    'geometry'] (geometry in the raster's own CRS).
    """
    ds = gdal.Open(map_path)
    if ds is None:
        raise FileNotFoundError(f"Could not open raster: {map_path}")
    band = ds.GetRasterBand(1)
    width = ds.RasterXSize
    height = ds.RasterYSize
    gt = ds.GetGeoTransform()
    proj = ds.GetProjection()

    records = {"_sample_key": [], "stratum": [], "row": [], "col": [], "geometry": []}

    for stratum, n_samples in sample_allocation.items():
        rng = np.random.default_rng(np.random.SeedSequence([seed, int(stratum)]))
        if v:
            print(f"Stratum {stratum}: sampling {n_samples} points (seed={seed})...")
        sampled = 0
        k = 0
        while sampled < n_samples:
            col = int(rng.integers(0, width))
            row = int(rng.integers(0, height))
            val = band.ReadAsArray(col, row, 1, 1)[0, 0]

            if val == stratum:
                k += 1
                x = gt[0] + (col + 0.5) * gt[1] + (row + 0.5) * gt[2]
                y = gt[3] + (col + 0.5) * gt[4] + (row + 0.5) * gt[5]

                records["_sample_key"].append(f"{id_prefix}_{stratum}_{k:05d}")
                records["stratum"].append(int(val))
                records["row"].append(row)
                records["col"].append(col)
                records["geometry"].append(Point(x, y))
                sampled += 1

        if v:
            print(f"  sampled {sampled}/{n_samples}")

    gdf = gpd.GeoDataFrame(records, geometry="geometry", crs=proj)

    if out_gpkg:
        gdf.to_file(out_gpkg, driver="GPKG")
        if v:
            print(f"Saved {len(gdf)} samples to {out_gpkg}")

    return gdf


def shuffle_samples(gdf, seed=0, out_gpkg=None, v=False):
    """
    Randomly shuffle sample-unit row order (without changing which units
    were selected), so an interpreter working through the exported CSV
    top to bottom isn't shown long runs of the same stratum.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
    seed : int
    out_gpkg : str, optional
    v : bool

    Returns
    -------
    gpd.GeoDataFrame, shuffled copy with reset index.
    """
    shuffled = gdf.sample(frac=1, random_state=seed).reset_index(drop=True)

    if out_gpkg:
        shuffled.to_file(out_gpkg, driver="GPKG")
        if v:
            print(f"Saved shuffled samples to {out_gpkg}")

    return shuffled


def assign_ids(gdf, id_col="id", v=False):
    """
    Assign the exported/display id, as sequential integers (0 to n-1) in
    the GeoDataFrame's *current* row order.

    Call this right after `shuffle_samples` (never before draw order is
    shuffled) -- if ids were assigned first and only the rows shuffled
    afterwards, the id itself would still tell an interpreter each unit's
    original per-stratum draw order (e.g. via `draw_samples_nested`'s
    internal `_sample_key`), defeating the point of shuffling. Cross-round
    matching (e.g. reusing pilot annotations inside the full sample) is
    done on `_sample_key`, not this id, so reassigning it here doesn't
    disturb that.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
    id_col : str
    v : bool

    Returns
    -------
    gpd.GeoDataFrame, copy with `id_col` (re)assigned from row order.
    """
    out = gdf.copy()
    out[id_col] = range(len(out))

    if v:
        print(f"Assigned '{id_col}' for {len(out)} unit(s), in row order.")

    return out


def relabel_true_stratum(df, true_col, stratum_labels, true_labels):
    """
    Remap an annotation tool's raw `true_col` codes onto the map's own
    stratum coding, by matching each code's text label.

    An annotation tool may code the same classes with different numeric
    values than the map does (e.g. it might call cropland "2" while the
    map calls it "1"). `true_labels` (STRATUM_TRUE_LABELS) documents the
    annotation tool's own {code: label} scheme; `stratum_labels`
    (STRATUM_LABELS) is the map/pred coding. This translates each
    `true_col` value to whichever `stratum_labels` code shares its label,
    so numbers in `true` and `pred` mean the same thing before they're
    ever compared (e.g. in a confusion matrix).

    Parameters
    ----------
    df : pd.DataFrame
        Must have `true_col`, holding raw codes matching `true_labels`'s keys.
    true_col : str
    stratum_labels : dict
        {map_code: label} -- the map/pred coding.
    true_labels : dict
        {annotation_code: label} -- the annotation tool's own coding. Must
        have exactly the same set of label values as `stratum_labels`
        (the keys/codes may differ).

    Returns
    -------
    pd.DataFrame
        Copy of `df` with `true_col` remapped to `stratum_labels` codes.
    """
    label_to_map_code = {label: code for code, label in stratum_labels.items()}

    missing_labels = set(true_labels.values()) - set(label_to_map_code)
    if missing_labels:
        raise ValueError(
            f"STRATUM_TRUE_LABELS has label(s) {missing_labels} with no matching "
            f"label in STRATUM_LABELS {stratum_labels} -- their labels must match "
            "(codes may differ)."
        )

    code_translation = {true_code: label_to_map_code[label] for true_code, label in true_labels.items()}

    out = df.copy()
    # Rows with no value yet (not-yet-annotated units) are left as missing,
    # not treated as invalid -- they're a normal, expected state while
    # annotation is still in progress, and the callers of this function
    # (combine_annotation_rounds, load_pilot_annotations) already have their
    # own clear "still needs annotation" checks downstream. Flagging them
    # here as "unknown values" would preempt that clearer error with a
    # confusing one (an empty list of "bad values", since there's nothing
    # actually invalid about a blank cell).
    present = out[true_col].notna()
    unknown = present & ~out[true_col].isin(code_translation)
    if unknown.any():
        bad = sorted(out.loc[unknown, true_col].unique())
        raise ValueError(
            f"'{true_col}' contains value(s) not in STRATUM_TRUE_LABELS {true_labels}: {bad}"
        )
    out[true_col] = out[true_col].map(code_translation)

    return out


def compute_pilot_variances(strata, pred, true, output_csv=None, v=False):
    """
    Estimate prior per-stratum standard deviations (S_h) from the annotated
    pilot sample, for Neyman sample-size/allocation planning (Cochran,
    1977; Olofsson et al., 2014, Eq. 5.55):

        S_h = sqrt(U_h * (1 - U_h))

    where U_h is the stratum's user's accuracy observed in the pilot sample
    (proportion of pilot units in map-stratum h whose annotated/reference
    class matches h).

    Parameters
    ----------
    strata : list
        Stratum values, in the same order as the map classes / pixel counts.
    pred, true : array-like
        Map-predicted and annotated/reference stratum of each pilot unit.
    output_csv : str, optional
    v : bool

    Returns
    -------
    pd.DataFrame with columns ['Stratum', 'n_pilot', 'Ui_pilot', 'Sh']
    """
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(pred, true, labels=strata)

    rows = []
    for i, s in enumerate(strata):
        n_pilot = int(cm[i].sum())
        Ui = cm[i, i] / n_pilot if n_pilot > 0 else 0.0
        Sh = np.sqrt(Ui * (1 - Ui))
        rows.append({"Stratum": s, "n_pilot": n_pilot, "Ui_pilot": Ui, "Sh": Sh})

    df = pd.DataFrame(rows)

    if output_csv:
        df.to_csv(output_csv, index=False)
    if v:
        print(df)

    return df


def _stratum_dict(df_or_dict, value_col):
    """Accept either a dict {Stratum: value} or a DataFrame with a 'Stratum' column."""
    if isinstance(df_or_dict, pd.DataFrame):
        return dict(zip(df_or_dict["Stratum"], df_or_dict[value_col]))
    return dict(df_or_dict)


# z-values for common confidence levels (two-sided). Avoids adding scipy as
# a hard dependency just for norm.ppf; falls back to scipy if an uncommon
# confidence level is requested and scipy happens to be available.
_Z_LOOKUP = {0.80: 1.2816, 0.85: 1.4395, 0.90: 1.6449, 0.95: 1.9600, 0.98: 2.3263, 0.99: 2.5758}


def _z_from_confidence(confidence):
    if confidence in _Z_LOOKUP:
        return _Z_LOOKUP[confidence]
    try:
        from scipy.stats import norm
        return float(norm.ppf(1 - (1 - confidence) / 2))
    except ImportError:
        raise ValueError(
            f"confidence={confidence} is not one of the supported standard levels "
            f"{sorted(_Z_LOOKUP)} and scipy is not installed to compute an exact "
            "z-value. Use one of the standard confidence levels, or install scipy."
        )


def compute_neyman_sample_size(pixel_counts, Sh, cv_target, confidence, target_stratum, v=False):
    """
    Total stratified-random-sample size needed to achieve a target relative
    precision (the user-defined accuracy threshold) at a given confidence
    level, via the Neyman sample-size formula (Cochran, 1977; Song et al.,
    2017, Eq. 2):

        n_tot = z^2 * (sum_h Wh*Sh)^2 / E^2

    where E = cv_target * p_target is the desired CI half-width, expressed
    as a fraction (cv_target) of p_target (the mapped proportion of
    `target_stratum`). Because E already scales with z (E = z * SE), the
    *actual* CV of the resulting estimator, SE(p_hat)/p_target, works out to
    cv_target / z: tightening the confidence level for the same cv_target
    requires a larger sample.

    Parameters
    ----------
    pixel_counts : pd.DataFrame
        Has columns ['Stratum', 'Area_ha', 'Wi'].
    Sh : pd.DataFrame or dict
        Prior per-stratum standard deviations, from `compute_pilot_variances`.
    cv_target : float
        Target relative precision (e.g. 0.10 for 10%). User-set accuracy
        threshold on the target stratum's area estimate.
    confidence : float
        Confidence level (e.g. 0.95).
    target_stratum : hashable
        Stratum whose mapped proportion (Wi) is used as p_target (typically
        the class of interest, e.g. cropland).
    v : bool

    Returns
    -------
    int
        Total required sample size, n_tot (rounded up).
    """
    Wi = _stratum_dict(pixel_counts, "Wi")
    Sh_d = _stratum_dict(Sh, "Sh")
    strata = list(Wi.keys())

    sum_WhSh = sum(Wi[s] * Sh_d[s] for s in strata)
    z = _z_from_confidence(confidence)
    p_target = Wi[target_stratum]
    E = cv_target * p_target

    n_tot = int(np.ceil((z * sum_WhSh / E) ** 2))

    if v:
        print(f"z={z}, sum(Wh*Sh)={sum_WhSh:.4f}, p_target={p_target:.4f}, E={E:.4f}")
        print(f"Required total sample size: n_tot={n_tot}")

    return n_tot


def allocate_neyman(pixel_counts, Sh, n_tot, output_allocation_csv=None, v=False):
    """
    Allocate n_tot sample units to strata via Neyman allocation (Cochran,
    1977; Song et al., 2017, Eq. 3):

        n_h = n_tot * Wh*Sh / sum_h(Wh*Sh)

    Parameters
    ----------
    pixel_counts : pd.DataFrame
        Has columns ['Stratum', 'Wi'].
    Sh : pd.DataFrame or dict
        Prior per-stratum standard deviations, from `compute_pilot_variances`.
    n_tot : int
        Total sample size to allocate, from `compute_neyman_sample_size`.
    output_allocation_csv : str, optional
    v : bool

    Returns
    -------
    dict {stratum: n_h}
    """
    Wi = _stratum_dict(pixel_counts, "Wi")
    Sh_d = _stratum_dict(Sh, "Sh")
    strata = list(Wi.keys())

    sum_WhSh = sum(Wi[s] * Sh_d[s] for s in strata)
    weights = {s: Wi[s] * Sh_d[s] for s in strata}
    allocation = {s: int(round(n_tot * weights[s] / sum_WhSh)) for s in strata}

    diff = n_tot - sum(allocation.values())
    order = sorted(strata, key=lambda s: -weights[s])
    i = 0
    while diff != 0 and order:
        s = order[i % len(order)]
        step = 1 if diff > 0 else -1
        allocation[s] += step
        diff -= step
        i += 1

    if output_allocation_csv:
        pd.DataFrame(list(allocation.items()), columns=["Stratum", "Sample units"]).to_csv(
            output_allocation_csv, index=False
        )

    if v:
        print(allocation)

    return allocation


def reconcile_allocation_with_pilot(neyman_allocation, pilot_allocation, v=False):
    """
    Ensure the Neyman-sized allocation requests at least as many units per
    stratum as the pilot already drew, so `draw_samples_nested` is
    guaranteed to nest the pilot's units (and their annotations) inside the
    full sample.

    Neyman allocation is computed independently per stratum from the
    pilot-derived Sh, so for an unusual pilot -- or a loose CV target -- it
    can come out below the pilot's own per-stratum count in a given
    stratum. Taking the elementwise max rules that out, at the cost of a
    (usually small) increase in total sample size above n_tot.

    Parameters
    ----------
    neyman_allocation, pilot_allocation : dict {stratum: n_h}
    v : bool

    Returns
    -------
    dict {stratum: n_h}
    """
    strata = set(neyman_allocation) | set(pilot_allocation)
    reconciled = {
        s: max(neyman_allocation.get(s, 0), pilot_allocation.get(s, 0)) for s in strata
    }

    if v:
        raised = {
            s: reconciled[s] - neyman_allocation.get(s, 0)
            for s in strata
            if reconciled[s] > neyman_allocation.get(s, 0)
        }
        if raised:
            print(f"Raised allocation above the Neyman target to keep pilot units nested: {raised}")
        print(f"Final allocation: {reconciled} (total={sum(reconciled.values())})")

    return reconciled


def export_sample_units(gdf, out_csv, stratum_labels,
                         pilot_truth=None, id_col="id", stratum_col="stratum",
                         match_col="_sample_key", true_col="true_stratum", v=False):
    """
    Reproject a sample GeoDataFrame to EPSG:4326 and export it as CSV
    (lat/lon columns) for use in an external annotation tool (e.g. STAC
    Notator).

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Sample units, e.g. from `draw_samples_nested` (any CRS). Should
        already be shuffled (`shuffle_samples`) with `id_col` assigned
        from that shuffled order (`assign_ids`) before calling this.
    out_csv : str
        Output path.
    stratum_labels : dict
        Maps numeric stratum value -> text label, e.g.
        {0: "Non-cropland", 1: "Cropland"}.
    pilot_truth : pd.DataFrame, optional
        Annotated pilot sample (e.g. from `load_pilot_annotations`), with
        at least [match_col, true_col]. Where a unit's `match_col`
        matches a pilot unit's, its annotation is copied in and `in_pilot`
        set True -- those units are already labeled and don't need
        re-annotating, which is what makes the pilot sample "reusable"
        within the full sample. Matching is done on `match_col` (the
        stable internal key from `draw_samples_nested`), not `id_col`,
        since `id_col` is reassigned after each round's own shuffle and so
        differs between the pilot's export and this one for the same unit.

        When `pilot_truth` is given, `true_col`/`true_stratum_label`
        are included in the export to carry those known pilot values
        forward (this is the master bookkeeping file). When it's None
        (a fresh annotation-round export), those columns are omitted
        entirely -- the annotation tool (e.g. STAC Notator) adds its own
        true-label column when the file comes back annotated, so there's
        no need to ship an empty placeholder column out to it.
    id_col, stratum_col, match_col, true_col : str
    v : bool

    Returns
    -------
    (csv_path, exported_dataframe)
    """
    out = gdf.to_crs(4326).copy()
    out["lon"] = out.geometry.x
    out["lat"] = out.geometry.y
    out["stratum_label"] = out[stratum_col].map(stratum_labels)

    out["in_pilot"] = False

    cols = [id_col, "lat", "lon", stratum_col, "stratum_label", "in_pilot"]

    if pilot_truth is not None:
        out[true_col] = pd.NA
        out["true_stratum_label"] = pd.NA

        truth = pilot_truth.set_index(match_col)
        matched = out[match_col].isin(truth.index)
        out.loc[matched, "in_pilot"] = True
        out.loc[matched, true_col] = out.loc[matched, match_col].map(truth[true_col])
        if "true_stratum_label" in truth.columns:
            out.loc[matched, "true_stratum_label"] = out.loc[matched, match_col].map(truth["true_stratum_label"])
        else:
            out.loc[matched, "true_stratum_label"] = out.loc[matched, true_col].map(stratum_labels)

        cols += [true_col, "true_stratum_label"]

    out_csv_df = out[cols].copy()
    out_csv_df.to_csv(out_csv, index=False)

    if v:
        print(f"Saved {len(out)} sample units to:\n  {out_csv}")
        if pilot_truth is not None:
            print(f"  {int(out['in_pilot'].sum())} unit(s) already annotated from the pilot round.")

    return out_csv, out_csv_df


def combine_annotation_rounds(master_csv_path, round2_annotated_csv_path, out_csv_path,
                               id_col="id", true_col="true_stratum",
                               stratum_labels=None, true_labels=None):
    """
    Combine the pilot round's annotations (already filled into the master
    full-sample export by `export_sample_units`, via `pilot_truth=`) with
    the round-2 annotations (the units NOT in the pilot, exported
    separately so annotators don't redo the pilot's units) into one fully
    annotated CSV, ready for `load_full_annotations`.

    Parameters
    ----------
    master_csv_path : str
        The full-sample export from `export_sample_units` (has `true_col`
        filled for pilot units, blank for the rest). The pilot's `true_col`
        values were already relabeled onto the map's coding when they were
        first loaded (see `load_pilot_annotations`), so only round-2's raw
        values need relabeling here.
    round2_annotated_csv_path : str
        Annotated round-2 CSV; must have at least [id_col, true_col] for
        the units NOT in the pilot.
    out_csv_path : str
    id_col, true_col : str
    stratum_labels, true_labels : dict, optional
        STRATUM_LABELS (map coding) and STRATUM_TRUE_LABELS (the
        annotation tool's own coding for `true_col`). When both are given,
        round-2's `true_col` is relabeled via `relabel_true_stratum` onto
        the map's coding before being merged into `master`. If either is
        omitted, `true_col` is used as-is (assumed already in the map's
        coding).

    Returns
    -------
    pd.DataFrame, also written to out_csv_path.
    """
    if not os.path.exists(round2_annotated_csv_path):
        raise FileNotFoundError(
            f"Round-2 annotated file not found: {round2_annotated_csv_path}\n"
            "Annotate the round-2 sample in STAC Notator first, save the result with "
            f"columns ['{id_col}', '{true_col}'] to this path, then re-run this step."
        )

    master = pd.read_csv(master_csv_path)
    round2 = pd.read_csv(round2_annotated_csv_path)

    missing = {id_col, true_col} - set(round2.columns)
    if missing:
        raise ValueError(f"Round-2 annotated file is missing required column(s): {missing}")

    if stratum_labels is not None and true_labels is not None:
        round2 = relabel_true_stratum(round2, true_col, stratum_labels, true_labels)

    round2_truth = round2.set_index(id_col)[true_col]
    still_blank = master[true_col].isna()
    master.loc[still_blank, true_col] = master.loc[still_blank, id_col].map(round2_truth)

    unresolved = master[master[true_col].isna()]
    if len(unresolved) > 0:
        ids = list(unresolved[id_col])
        raise ValueError(
            f"{len(unresolved)} sample unit(s) still have no annotation after combining pilot + "
            f"round-2 rounds: {ids[:5]}{'...' if len(ids) > 5 else ''}"
        )

    master.to_csv(out_csv_path, index=False)
    return master


def load_full_annotations(annotated_csv_path, id_col="id", stratum_col="stratum", true_col="true_stratum"):
    """
    Load the fully annotated sample -- every unit interpreted in STAC
    Notator, including the pilot units `export_sample_units` already
    filled in -- for final area/accuracy estimation.

    Expects the same CSV schema produced by `export_sample_units`
    (`bungoma2025_sample_units.csv`), with `true_col` completed for every
    row rather than just the pilot subset.

    Parameters
    ----------
    annotated_csv_path : str
        Path to the fully annotated CSV; must have at least
        [id_col, stratum_col, true_col], with no missing true_col values.
    id_col, stratum_col, true_col : str

    Returns
    -------
    pred : list
        Map-predicted stratum per unit.
    true : list
        Annotated (reference) stratum per unit.
    df : pd.DataFrame
        The loaded annotated sample.
    """
    if not os.path.exists(annotated_csv_path):
        raise FileNotFoundError(
            f"Fully annotated sample not found: {annotated_csv_path}\n"
            "Complete annotation of the remaining (non-pilot) units in STAC Notator, "
            f"fill in '{true_col}' for every row, and save the result to this path."
        )

    df = pd.read_csv(annotated_csv_path)
    missing = {id_col, stratum_col, true_col} - set(df.columns)
    if missing:
        raise ValueError(f"Annotated sample is missing required column(s): {missing}")

    unlabeled = df[df[true_col].isna()]
    if len(unlabeled) > 0:
        ids = list(unlabeled[id_col])
        raise ValueError(
            f"{len(unlabeled)} sample unit(s) still need annotation (missing '{true_col}'): "
            f"{ids[:5]}{'...' if len(ids) > 5 else ''}"
        )

    pred = df[stratum_col].astype(int).tolist()
    true = df[true_col].astype(int).tolist()
    return pred, true, df


def _build_confusion_proportions(pixel_counts, pred, true):
    """
    Shared building block for the estimators below: cross-tabulates
    pred/true by stratum (map class) and converts sample counts into
    estimated area proportions p_hat_ij = Wi * nij/ni (Olofsson et al.,
    2014, Eq. 4), assuming the strata are the map classes.

    Parameters
    ----------
    pixel_counts : pd.DataFrame
        Has columns ['Stratum', 'Area_ha'] (and optionally 'Wi').
    pred, true : array-like
        Predicted (map) and reference stratum of each sample unit.

    Returns
    -------
    strata : list
        Stratum values, in the order used for all matrix rows/columns.
    cm : np.ndarray
        Sample count confusion matrix, cm[i, j] = # samples with map
        stratum i and reference class j.
    Wi : np.ndarray
        Mapped area proportion of each stratum, same order as `strata`.
    Pij : np.ndarray
        Estimated area-proportion error matrix, p_hat_ij.
    """
    from sklearn.metrics import confusion_matrix

    strata = list(pixel_counts["Stratum"])
    pixel_counts_dict = pixel_counts.set_index("Stratum")["Area_ha"].to_dict()
    total_pixels = sum(pixel_counts_dict.values())
    Wi = np.array([pixel_counts_dict[s] / total_pixels for s in strata])

    cm = confusion_matrix(pred, true, labels=strata)
    numberOfClasses = len(strata)

    Pij = np.zeros((numberOfClasses, numberOfClasses))
    for i in range(numberOfClasses):
        ni = cm[i].sum()
        for j in range(numberOfClasses):
            Pij[i, j] = Wi[i] * cm[i, j] / ni if ni > 0 else 0

    return strata, cm, Wi, Pij


def compute_stratified_random_sampling_metrics(pixel_counts, pred, true, output_csv=None, v=False):
    """
    Compute design-based stratified area estimates and uncertainty metrics
    (Olofsson et al., 2014, Eqs. 8-10).

    Parameters
    ----------
    pixel_counts : pd.DataFrame
        Has columns ['Stratum', 'Area_ha'].
    pred, true : array-like
        Map-predicted and annotated/reference stratum of each sample unit.
    output_csv : str, optional
    v : bool

    Returns
    -------
    pd.DataFrame
        Per-class area estimates, SE, CI, and relative (%) SE/CI. Indexed
        by stratum.
    """
    strata, cm, Wi, Pij = _build_confusion_proportions(pixel_counts, pred, true)
    numberOfClasses = len(strata)
    total_pixels = pixel_counts.set_index("Stratum")["Area_ha"].sum()

    Areas = [sum(Pij[:, k]) * total_pixels for k in range(numberOfClasses)]

    AreasSTDERR = []
    for k in range(numberOfClasses):
        AreasSTDERR.append(
            np.sqrt(sum(
                ((Wi[i] * Pij[i, k]) - (Pij[i, k] ** 2)) / (cm[i].sum() - 1) if cm[i].sum() > 1 else 0
                for i in range(numberOfClasses)
            ))
        )

    AreasSE_total = [total_pixels * AreasSTDERR[k] for k in range(numberOfClasses)]

    df = pd.DataFrame({
        "Area_ha": Areas,
        "SE_Ha": AreasSE_total,
        "CI_Ha": [se * 1.96 for se in AreasSE_total],
        "SE%": [se / area if area != 0 else np.nan for se, area in zip(AreasSE_total, Areas)],
        "CI%": [(se * 1.96) / area if area != 0 else np.nan for se, area in zip(AreasSE_total, Areas)],
    }, index=strata)
    df.index.name = "class"

    if output_csv:
        df.to_csv(output_csv)
    if v:
        print(df)

    return df


def compute_accuracy_metrics(pixel_counts, pred, true, output_csv=None, v=False):
    """
    Compute sample-based map accuracy (overall, user's, producer's) with
    standard errors and 95% confidence intervals (Olofsson et al., 2014,
    Eqs. 1-3, 5-7). Assumes the strata correspond to the map classes, as in
    `compute_stratified_random_sampling_metrics`.

    Parameters
    ----------
    pixel_counts : pd.DataFrame
        Has columns ['Stratum', 'Area_ha'].
    pred, true : array-like
        Map-predicted and annotated/reference stratum of each sample unit.
    output_csv : str, optional
    v : bool

    Returns
    -------
    class_df : pd.DataFrame
        Per-class user's accuracy (Ui) and producer's accuracy (Pi), each
        with SE and 95% CI. Indexed by stratum.
    overall : dict
        {'O': overall accuracy, 'SE': ..., 'CI': ...}
    """
    strata, cm, Wi, Pij = _build_confusion_proportions(pixel_counts, pred, true)
    q = len(strata)
    ni = np.array([cm[i].sum() for i in range(q)])

    Ui = np.array([Pij[i, i] / Wi[i] if Wi[i] > 0 else np.nan for i in range(q)])
    Ui_var = np.array([
        Ui[i] * (1 - Ui[i]) / (ni[i] - 1) if ni[i] > 1 else np.nan
        for i in range(q)
    ])

    p_dot_j = Pij.sum(axis=0)
    total_pixels = pixel_counts.set_index("Stratum")["Area_ha"].sum()
    N_i = Wi * total_pixels

    Pj = np.array([Pij[j, j] / p_dot_j[j] if p_dot_j[j] > 0 else np.nan for j in range(q)])

    Pj_var = np.full(q, np.nan)
    for j in range(q):
        if p_dot_j[j] <= 0:
            continue
        Nj_hat_j = sum(N_i[i] * cm[i, j] / ni[i] if ni[i] > 0 else 0 for i in range(q))
        if Nj_hat_j <= 0 or ni[j] <= 1:
            continue
        term1 = (N_i[j] ** 2) * ((1 - Pj[j]) ** 2) * Ui[j] * (1 - Ui[j]) / (ni[j] - 1)
        term2 = (Pj[j] ** 2) * sum(
            (N_i[i] ** 2) * (cm[i, j] / ni[i]) * (1 - cm[i, j] / ni[i]) / (ni[i] - 1)
            for i in range(q) if i != j and ni[i] > 1
        )
        Pj_var[j] = (term1 + term2) / (Nj_hat_j ** 2)

    O = Pij.trace()
    O_var = sum(
        (Wi[i] ** 2) * Ui[i] * (1 - Ui[i]) / (ni[i] - 1) if ni[i] > 1 else 0
        for i in range(q)
    )

    class_df = pd.DataFrame({
        "Ui": Ui,
        "Ui_SE": np.sqrt(Ui_var),
        "Ui_CI": 1.96 * np.sqrt(Ui_var),
        "Pi": Pj,
        "Pi_SE": np.sqrt(Pj_var),
        "Pi_CI": 1.96 * np.sqrt(Pj_var),
    }, index=strata)
    class_df.index.name = "class"

    overall = {"O": O, "SE": np.sqrt(O_var), "CI": 1.96 * np.sqrt(O_var)}

    if output_csv:
        class_df.to_csv(output_csv)
    if v:
        print(class_df)
        print(f"Overall accuracy: {overall['O']:.3f} +/- {overall['CI']:.3f}")

    return class_df, overall


def load_pilot_annotations(pilot_gdf, annotated_csv_path, id_col="id", true_col="true_stratum",
                            stratum_labels=None, true_labels=None):
    """
    Load a pilot sample back in after external annotation (e.g. via STAC
    Notator) and align it with the map-predicted stratum of each unit, for
    use with `compute_pilot_variances`.

    Parameters
    ----------
    pilot_gdf : gpd.GeoDataFrame
        The pilot sample as drawn/prepared for export (has 'id',
        '_sample_key', and 'stratum', the map-predicted class). The
        annotation round-trip only preserves `id_col` (whatever the
        annotation tool hands back), so this merges on `id_col` but keeps
        `_sample_key` along for the ride -- that's the stable key needed
        later to match these pilot annotations into the full sample, since
        `id_col` itself gets reassigned to a new value after the full
        sample's own shuffle.
    annotated_csv_path : str
        Path to the annotated CSV; must have at least [id_col, true_col].
    id_col, true_col : str
    stratum_labels, true_labels : dict, optional
        STRATUM_LABELS (map coding) and STRATUM_TRUE_LABELS (the
        annotation tool's own coding for `true_col`). When both are given,
        `true_col` is relabeled via `relabel_true_stratum` onto the map's
        coding before anything is compared against `pred` -- this is what
        lets the annotation tool use its own numeric codes for the same
        classes. If either is omitted, `true_col` is used as-is (assumed
        already in the map's coding).

    Returns
    -------
    pred : list
        Map-predicted stratum per pilot unit.
    true : list
        Annotated (reference) stratum per pilot unit, in the map's coding.
    merged : pd.DataFrame
        Columns [id_col, '_sample_key', 'stratum', true_col].
    """
    if not os.path.exists(annotated_csv_path):
        raise FileNotFoundError(
            f"Annotated pilot file not found: {annotated_csv_path}\n"
            "Annotate the exported pilot sample in STAC Notator first, save the "
            f"result with columns ['{id_col}', '{true_col}'] to this path, then re-run this step."
        )

    annotated = pd.read_csv(annotated_csv_path)
    missing = {id_col, true_col} - set(annotated.columns)
    if missing:
        raise ValueError(f"Annotated pilot file is missing required column(s): {missing}")

    if stratum_labels is not None and true_labels is not None:
        annotated = relabel_true_stratum(annotated, true_col, stratum_labels, true_labels)

    merged = pilot_gdf[[id_col, "_sample_key", "stratum"]].merge(
        annotated[[id_col, true_col]], on=id_col, how="inner"
    )
    if len(merged) < len(pilot_gdf):
        missing_ids = sorted(set(pilot_gdf[id_col]) - set(merged[id_col]))
        print(
            f"Warning: {len(missing_ids)} pilot unit(s) have no matching annotation and were "
            f"dropped: {missing_ids[:5]}{'...' if len(missing_ids) > 5 else ''}"
        )

    pred = list(merged["stratum"])
    true = list(merged[true_col])
    return pred, true, merged
