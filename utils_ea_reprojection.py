#!/usr/bin/env python
# coding: utf-8
"""
Auto-derive and apply a study-region-centered equal-area projection.

Pixel counting (hectares per stratum) and equal-probability stratified
sampling both assume every pixel covers the same amount of ground area. In
geographic coordinates (EPSG:4326) that's false -- a degree of longitude
covers less ground distance near the poles than at the equator, so pixel
area varies across (and even within) a raster. An equal-area projection
removes that distortion: every pixel has the same ground area everywhere in
the raster, so pixel counts convert directly and unbiasedly into hectares,
and a uniformly-drawn pixel sample is a uniform area sample.

Rather than hardcoding a CRS for one study region, `derive_ea_proj_string`
centers a Lambert Azimuthal Equal-Area (LAEA) projection on the raster's own
bounding-box centroid, so the same code works for any new AOI or season.
"""

from osgeo import gdal, osr
from pyproj import CRS, Transformer

gdal.UseExceptions()


def derive_ea_proj_string(raster_path, input_crs=None, out_proj_path=None):
    """
    Derive a best-fit equal-area (LAEA) PROJ string centered on a raster's
    own extent.

    Parameters
    ----------
    raster_path : str
        Path to raster file.
    input_crs : str or pyproj.CRS, optional
        CRS to assume if the raster has none defined.
    out_proj_path : str, optional
        Path to write the PROJ string to.

    Returns
    -------
    str
        PROJ string of the derived equal-area projection.
    """
    ds = gdal.Open(raster_path)
    if ds is None:
        raise IOError(f"Could not open raster: {raster_path}")

    gt = ds.GetGeoTransform()
    xsize = ds.RasterXSize
    ysize = ds.RasterYSize

    minx = gt[0]
    maxy = gt[3]
    maxx = minx + gt[1] * xsize
    miny = maxy + gt[5] * ysize

    wkt = ds.GetProjection()
    if wkt:
        src_crs = CRS.from_wkt(wkt)
    elif input_crs is not None:
        src_crs = CRS.from_user_input(input_crs)
    else:
        raise ValueError("Input CRS must be provided if raster has no CRS")

    if src_crs.to_epsg() != 4326:
        transformer = Transformer.from_crs(src_crs, 4326, always_xy=True)
        minx, miny = transformer.transform(minx, miny)
        maxx, maxy = transformer.transform(maxx, maxy)

    lon_0 = (minx + maxx) / 2
    lat_0 = (miny + maxy) / 2

    proj_str = (
        f"+proj=laea +lat_0={lat_0} +lon_0={lon_0} "
        "+datum=WGS84 +units=m +no_defs"
    )

    if out_proj_path is not None:
        with open(out_proj_path, "w") as f:
            f.write(proj_str)

    ds = None
    return proj_str


def raster_to_ea(input_raster_path, output_raster_path, proj_string,
                  resampling_method="nearest", output_dtype=gdal.GDT_Byte):
    """
    Reproject a raster to the given PROJ string, DEFLATE-compressed.

    Parameters
    ----------
    input_raster_path : str
    output_raster_path : str
    proj_string : str
        Target projection (PROJ string), e.g. from `derive_ea_proj_string`.
    resampling_method : str, default "nearest"
        "nearest" (required for categorical/class rasters, e.g. cropland
        maps -- other methods would blend class codes), "bilinear",
        "cubic", or "lanczos".
    output_dtype : gdal data type, default gdal.GDT_Byte
    """
    src_ds = gdal.Open(input_raster_path)
    if src_ds is None:
        raise IOError(f"Could not open raster: {input_raster_path}")

    src_srs = osr.SpatialReference()
    src_srs.ImportFromWkt(src_ds.GetProjection())

    dst_srs = osr.SpatialReference()
    dst_srs.ImportFromProj4(proj_string)

    resampling_dict = {
        "nearest": gdal.GRA_NearestNeighbour,
        "bilinear": gdal.GRA_Bilinear,
        "cubic": gdal.GRA_Cubic,
        "lanczos": gdal.GRA_Lanczos,
    }
    resample_alg = resampling_dict.get(resampling_method.lower(), gdal.GRA_NearestNeighbour)

    creation_options = ["COMPRESS=DEFLATE", "TILED=YES"]

    gdal.Warp(
        output_raster_path,
        src_ds,
        dstSRS=dst_srs,
        resampleAlg=resample_alg,
        format="GTiff",
        outputType=output_dtype,
        creationOptions=creation_options,
    )

    src_ds = None
    print(f"Reprojected raster saved to: {output_raster_path} (DEFLATE compressed)")
