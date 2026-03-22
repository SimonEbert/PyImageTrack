# PyImageTrack/Cache.py
"""
Caching module for PyImageTrack.

This module provides functions for saving and loading alignment, tracking, and
level of detection (LoD) results to/from disk. Caching improves performance by
avoiding redundant computations when processing the same image pairs multiple times.

All cache files include metadata with file hashes to detect changes in input data.
"""
import os
import json
import hashlib
import geopandas as gpd
import rasterio
from rasterio.crs import CRS as RioCRS
import numpy as np
from shapely import box
from rasterio.transform import array_bounds
from affine import Affine
from PyImageTrack.CreateGeometries.HandleGeometries import make_safe_bounds_from_buffer

from PyImageTrack.ConsoleOutput import get_console


def _sha256(path: str) -> str:
    """
    Calculate SHA-256 hash of a file.

    Parameters
    ----------
    path : str
        Path to the file to hash.

    Returns
    -------
    str
        Hexadecimal SHA-256 hash of the file contents.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def alignment_cache_paths(align_dir: str, year1: str, year2: str):
    """
    Generate file paths for alignment cache files.

    Parameters
    ----------
    align_dir : str
        Directory where alignment cache files are stored.
    year1 : str
        Identifier for the first (reference) image.
    year2 : str
        Identifier for the second (aligned) image.

    Returns
    -------
    tuple
        A tuple of (aligned_tif_path, control_points_path, metadata_json_path).
    """
    aligned_tif = os.path.join(align_dir, f"aligned_image_{year2}.tif")
    aligned_depth_tif = os.path.join(align_dir, f"aligned_image_{year2}_depth.tif")
    control_pts = os.path.join(align_dir, f"alignment_control_points_{year1}_{year2}.fgb")
    meta_json   = os.path.join(align_dir, f"alignment_meta_{year1}_{year2}.json")
    return aligned_tif, aligned_depth_tif, control_pts, meta_json

def save_alignment_cache(image_pair, align_dir: str, year1: str, year2: str,
                         align_params: dict, filenames: dict, dates: dict,
                         version: str = "v1", save_truecolor_aligned: bool = False):
    """
    Save alignment results to cache files.

    Saves the aligned image, control points (if available), and metadata to disk.
    Validates the aligned image before saving to ensure it contains valid data.

    Parameters
    ----------
    image_pair : ImagePair
        ImagePair object containing alignment results.
    align_dir : str
        Directory where alignment cache files will be saved.
    year1 : str
        Identifier for the first (reference) image.
    year2 : str
        Identifier for the second (aligned) image.
    align_params : dict
        Dictionary of alignment parameters used.
    filenames : dict
        Dictionary mapping year identifiers to file paths.
    dates : dict
        Dictionary mapping year identifiers to observation dates.
    version : str, optional
        Script version identifier for cache validation. Default is "v1".
    save_truecolor_aligned : bool, optional
        If True, also save a true-color version of the aligned image. Default is False.

    Raises
    ------
    ValueError
        If the aligned image contains all NaN values, all zeros, or has very low variance.
    """
    os.makedirs(align_dir, exist_ok=True)
    aligned_tif, aligned_depth_tif, control_pts, meta_json = alignment_cache_paths(align_dir, year1, year2)

    crs = image_pair.crs
    if crs is not None and not isinstance(crs, RioCRS):
        if isinstance(crs, str) and crs.strip().isdigit():
            crs = int(crs.strip())
        crs = RioCRS.from_user_input(crs)

    # Validate the aligned image before saving
    if np.all(np.isnan(image_pair.image2_matrix)):
        raise ValueError(
            f"Cannot save alignment cache for pair {year1}->{year2}: "
            "Aligned image contains all NaN values. Alignment failed."
        )
    
    if np.all(image_pair.image2_matrix == 0):
        raise ValueError(
            f"Cannot save alignment cache for pair {year1}->{year2}: "
            "Aligned image contains all zero values. Alignment failed."
        )
    
    if np.var(image_pair.image2_matrix) < 1e-10:
        raise ValueError(
            f"Cannot save alignment cache for pair {year1}->{year2}: "
            "Aligned image has nearly constant values (very low variance). Alignment failed."
        )

    # write main aligned image (working image used for tracking)
    profile = {
        "driver": "GTiff",
        "count": 1 if image_pair.image2_matrix.ndim == 2 else image_pair.image2_matrix.shape[0],
        "dtype": str(image_pair.image2_matrix.dtype),
        "crs": crs,
        "width": image_pair.image2_matrix.shape[-1],
        "height": image_pair.image2_matrix.shape[-2],
        "transform": image_pair.image2_transform,
    }
    with rasterio.open(aligned_tif, "w", **profile) as dst:
        if profile["count"] == 1:
            dst.write(image_pair.image2_matrix, 1)
        else:
            dst.write(image_pair.image2_matrix)

    if image_pair.depth_image2 is not None:
        profile.update({"count": 1})
        with rasterio.open(aligned_depth_tif, "w", **profile) as dst:
            dst.write(image_pair.depth_image2, 1)
    # optionally write an additional true-color aligned image
    # this expects image_pair.image2_matrix_truecolor to be set (multi-band or single-band)
    if save_truecolor_aligned and getattr(image_pair, "image2_matrix_truecolor", None) is not None:
        truecolor_tif = os.path.join(align_dir, f"aligned_image_truecolor_{year2}.tif")
        true = image_pair.image2_matrix_truecolor

        tc_profile = profile.copy()
        # update band count and dtype based on the true-color array
        if true.ndim == 2:
            tc_profile["count"] = 1
            tc_profile["dtype"] = str(true.dtype)
        else:
            tc_profile["count"] = true.shape[0]
            tc_profile["dtype"] = str(true.dtype)

        with rasterio.open(truecolor_tif, "w", **tc_profile) as dst_tc:
            if tc_profile["count"] == 1:
                dst_tc.write(true, 1)
            else:
                dst_tc.write(true)

    # write control points if available
    if getattr(image_pair, "tracked_control_points", None) is not None and len(image_pair.tracked_control_points) > 0:
        image_pair.tracked_control_points.to_file(control_pts, driver="GeoJSON")

    # write metadata json
    meta = {
        "pair": {"year1": year1, "year2": year2, "date1": dates[year1], "date2": dates[year2]},
        "files": {"file1": filenames[year1], "file2": filenames[year2]},
        "files_hash": {k: _sha256(v) for k, v in filenames.items()},
        "alignment_params": align_params,
        "crs": str(image_pair.crs),
        "script_version": version,
    }
    with open(meta_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def load_alignment_cache(image_pair, align_dir: str, year1: str, year2: str) -> bool:
    """
    Load alignment results from cache files.

    Parameters
    ----------
    image_pair : ImagePair
        ImagePair object to populate with cached alignment results.
    align_dir : str
        Directory where alignment cache files are stored.
    year1 : str
        Identifier for the first (reference) image.
    year2 : str
        Identifier for the second (aligned) image.

    Returns
    -------
    bool
        True if cache was loaded successfully, False if cache files don't exist.
    """
    aligned_tif, aligned_depth_tif, control_pts, _ = alignment_cache_paths(align_dir, year1, year2)
    if not os.path.exists(aligned_tif):
        return False
    src = rasterio.open(aligned_tif, "r")
    arr = src.read()
    image_pair.crs = src.crs
    image_pair.image2_matrix = arr[0] if arr.shape[0] == 1 else arr
    image_pair.image2_transform = image_pair.image1_transform
    image_pair.images_aligned = True
    image_pair.valid_alignment_possible = True

    # Calculate correct safe image bounds
    if image_pair.crs is not None:# correctly georeferenced path
        # Spatial intersection (true georef)
        bounds_poly = box(*src.bounds)
        px_size = max(-src.transform[4], src.transform[0])# pixel size (>0)


    else:# non-orthophoto path: No crs given --> 'fake georeferencing'

        # Top-left crop to common size
        h = min(image_pair.image1_matrix.shape[-2], image_pair.image2_matrix.shape[-2])
        w = min(image_pair.image1_matrix.shape[-1], image_pair.image2_matrix.shape[-1])
        px = float(image_pair.fake_pixel_size)
        tform = Affine(px, 0, 0, 0, -px, 0)  # x = px*col ; y = -px*row


        # Bounds in that CRS (pixel grid space)
        bounds_poly = box(*array_bounds(h, w, tform))
        px_size = float(image_pair.fake_pixel_size)

    image_pair.safe_image_bounds_tracking = make_safe_bounds_from_buffer(
        px_size=px_size,
        buffer=max(getattr(image_pair.tracking_parameters, "search_extent_px", None))
        + getattr(image_pair.tracking_parameters, "movement_cell_size", None)/2,
        base_polygon=bounds_poly,
        crs=image_pair.crs)

    image_pair.safe_image_bounds_alignment = make_safe_bounds_from_buffer(
        px_size=px_size,
        buffer=max(getattr(image_pair.alignment_parameters, "control_search_extent_px", None))
        + getattr(image_pair.alignment_parameters, "control_cell_size", None)/2,
        base_polygon=bounds_poly,
        crs=image_pair.crs)


    if image_pair.depth_image1 is not None:
        with rasterio.open(aligned_depth_tif, "r") as src:
            depth_arr = src.read()
        image_pair.depth_image2 = np.squeeze(depth_arr)

    console = get_console()
    if image_pair.image1_matrix.shape != image_pair.image2_matrix.shape:
        console.warning("The two matrices have not the same shape, signifying probably either a channel mismatch or "
                        "non-aligned images.\n"
                        "Shape of the first image: " + str(image_pair.image1_matrix.shape) + "\n"
                        "Shape of the second image: " + str(image_pair.image2_matrix.shape))
    if os.path.exists(control_pts):
        try:
            image_pair.tracked_control_points = gpd.read_file(control_pts)
        except FileNotFoundError:
            console.warning("Did not find control points in alignment cache. Control points for this alignment are not"
                         "available.")
    return True

def tracking_cache_paths(track_dir: str, year1: str, year2: str):
    """
    Generate file paths for tracking cache files.

    Parameters
    ----------
    track_dir : str
        Directory where tracking cache files are stored.
    year1 : str
        Identifier for the first image.
    year2 : str
        Identifier for the second image.

    Returns
    -------
    tuple
        A tuple of (tracking_geojson_path, metadata_json_path).
    """
    raw_geojson = os.path.join(track_dir, f"tracking_raw_{year1}_{year2}.fgb")
    meta_json   = os.path.join(track_dir, f"tracking_meta_{year1}_{year2}.json")
    return raw_geojson, meta_json


def lod_cache_paths(track_dir: str, year1: str, year2: str):
    """
    Generate file paths for level of detection (LoD) cache files.

    Parameters
    ----------
    track_dir : str
        Directory where LoD cache files are stored.
    year1 : str
        Identifier for the first image.
    year2 : str
        Identifier for the second image.

    Returns
    -------
    tuple
        A tuple of (lod_geojson_path, metadata_json_path).
    """
    lod_geojson = os.path.join(track_dir, f"lod_points_{year1}_{year2}.fgb")
    meta_json   = os.path.join(track_dir, f"lod_meta_{year1}_{year2}.json")
    return lod_geojson, meta_json

def save_tracking_cache(image_pair, track_dir: str, year1: str, year2: str,
                        track_params: dict, filenames: dict, dates: dict, version: str = "v1"):
    """
    Save tracking results to cache files.

    Parameters
    ----------
    image_pair : ImagePair
        ImagePair object containing tracking results.
    track_dir : str
        Directory where tracking cache files will be saved.
    year1 : str
        Identifier for the first image.
    year2 : str
        Identifier for the second image.
    track_params : dict
        Dictionary of tracking parameters used.
    filenames : dict
        Dictionary mapping year identifiers to file paths.
    dates : dict
        Dictionary mapping year identifiers to observation dates.
    version : str, optional
        Script version identifier for cache validation. Default is "v1".
    """
    os.makedirs(track_dir, exist_ok=True)
    raw_geojson, meta_json = tracking_cache_paths(track_dir, year1, year2)
    if image_pair.tracking_results is None or len(image_pair.tracking_results) == 0:
        return
    image_pair.tracking_results.to_file(raw_geojson, driver="FlatGeobuf")
    meta = {
        "pair": {"year1": year1, "year2": year2, "date1": dates[year1], "date2": dates[year2]},
        "files": {"file1": filenames[year1], "file2": filenames[year2]},
        "files_hash": {k: _sha256(v) for k, v in filenames.items()},
        "tracking_params": track_params,
        "crs": str(image_pair.crs),
        "script_version": version,
    }
    with open(meta_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

def load_tracking_cache(image_pair, track_dir: str, year1: str, year2: str) -> bool:
    """
    Load tracking results from cache files.

    Parameters
    ----------
    image_pair : ImagePair
        ImagePair object to populate with cached tracking results.
    track_dir : str
        Directory where tracking cache files are stored.
    year1 : str
        Identifier for the first image.
    year2 : str
        Identifier for the second image.

    Returns
    -------
    bool
        True if cache was loaded successfully, False if cache files don't exist or error occurred.
    """
    raw_geojson, _ = tracking_cache_paths(track_dir, year1, year2)
    if not os.path.exists(raw_geojson):
        return False
    try:
        image_pair.tracking_results = gpd.read_file(raw_geojson)
        image_pair.crs = image_pair.tracking_results.crs
        return True
    except Exception:
        return False


def save_lod_cache(image_pair, track_dir: str, year1: str, year2: str,
                   filenames: dict, dates: dict, version: str = "v1"):
    """
    Save level of detection (LoD) points to cache files.

    Parameters
    ----------
    image_pair : ImagePair
        ImagePair object containing LoD points.
    track_dir : str
        Directory where LoD cache files will be saved.
    year1 : str
        Identifier for the first image.
    year2 : str
        Identifier for the second image.
    filenames : dict
        Dictionary mapping year identifiers to file paths.
    dates : dict
        Dictionary mapping year identifiers to observation dates.
    version : str, optional
        Script version identifier for cache validation. Default is "v1".
    """
    os.makedirs(track_dir, exist_ok=True)
    lod_geojson, meta_json = lod_cache_paths(track_dir, year1, year2)
    if image_pair.level_of_detection_points is None or len(image_pair.level_of_detection_points) == 0:
        return
    image_pair.level_of_detection_points.to_file(lod_geojson, driver="FlatGeobuf")
    meta = {
        "pair": {"year1": year1, "year2": year2, "date1": dates[year1], "date2": dates[year2]},
        "files": {"file1": filenames[year1], "file2": filenames[year2]},
        "files_hash": {k: _sha256(v) for k, v in filenames.items()},
        "crs": str(image_pair.crs),
        "script_version": version,
        "level_of_detection": image_pair.level_of_detection,
    }
    with open(meta_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def load_lod_cache(image_pair, track_dir: str, year1: str, year2: str) -> bool:
    """
    Load level of detection (LoD) points from cache files.

    Parameters
    ----------
    image_pair : ImagePair
        ImagePair object to populate with cached LoD points.
    track_dir : str
        Directory where LoD cache files are stored.
    year1 : str
        Identifier for the first image.
    year2 : str
        Identifier for the second image.

    Returns
    -------
    bool
        True if cache was loaded successfully, False if cache files don't exist or error occurred.
    """
    lod_geojson, meta_json = lod_cache_paths(track_dir, year1, year2)
    if not os.path.exists(lod_geojson):
        return False
    try:
        image_pair.level_of_detection_points = gpd.read_file(lod_geojson)
        # Restore the level_of_detection value from metadata
        if os.path.exists(meta_json):
            with open(meta_json, "r", encoding="utf-8") as f:
                meta = json.load(f)
                image_pair.level_of_detection = meta.get("level_of_detection")
        return True
    except Exception:
        return False
