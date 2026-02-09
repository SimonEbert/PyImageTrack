# PyImageTrack/Cache.py
import os
import json
import hashlib
import geopandas as gpd
import rasterio
from rasterio.crs import CRS as RioCRS
import logging

def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def alignment_cache_paths(align_dir: str, year1: str, year2: str):
    aligned_tif = os.path.join(align_dir, f"aligned_image_{year2}.tif")
    control_pts = os.path.join(align_dir, f"alignment_control_points_{year1}_{year2}.fgb")
    meta_json   = os.path.join(align_dir, f"alignment_meta_{year1}_{year2}.json")
    return aligned_tif, control_pts, meta_json

def save_alignment_cache(image_pair, align_dir: str, year1: str, year2: str,
                         align_params: dict, filenames: dict, dates: dict,
                         version: str = "v1"):
    os.makedirs(align_dir, exist_ok=True)
    aligned_tif, control_pts, meta_json = alignment_cache_paths(align_dir, year1, year2)

    crs = image_pair.crs
    if crs is not None and not isinstance(crs, RioCRS):
        if isinstance(crs, str) and crs.strip().isdigit():
            crs = int(crs.strip())
        crs = RioCRS.from_user_input(crs)

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


    # write control points if available
    if getattr(image_pair, "tracked_control_points", None) is not None and len(image_pair.tracked_control_points) > 0:
        image_pair.tracked_control_points.to_file(control_pts, driver="FlatGeobuf")

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
    aligned_tif, control_pts, _ = alignment_cache_paths(align_dir, year1, year2)
    if not os.path.exists(aligned_tif):
        return False
    with rasterio.open(aligned_tif, "r") as src:
        arr = src.read()
        image_pair.crs = src.crs
    image_pair.image2_matrix = arr[0] if arr.shape[0] == 1 else arr
    image_pair.image2_transform = image_pair.image1_transform
    image_pair.images_aligned = True
    image_pair.valid_alignment_possible = True
    if image_pair.image1_matrix.shape != image_pair.image2_matrix.shape:
        logging.warning("The two matrices have not the same shape, signifying probably either a channel mismatch or "
                        "non-aligned images.\n"
                        "Shape of the first image: " + str(image_pair.image1_matrix.shape) + "\n"
                        "Shape of the second image: " + str(image_pair.image2_matrix.shape))
    if os.path.exists(control_pts):
        try:
            image_pair.tracked_control_points = gpd.read_file(control_pts)
        except FileNotFoundError:
            logging.warning("Did not find control points in alignment cache. Control points for this alignment are not"
                         "available.")
    return True

def tracking_cache_paths(track_dir: str, year1: str, year2: str):
    raw_geojson = os.path.join(track_dir, f"tracking_raw_{year1}_{year2}.fgb")
    meta_json   = os.path.join(track_dir, f"tracking_meta_{year1}_{year2}.json")
    return raw_geojson, meta_json


def lod_cache_paths(track_dir: str, year1: str, year2: str):
    lod_geojson = os.path.join(track_dir, f"lod_points_{year1}_{year2}.fgb")
    meta_json   = os.path.join(track_dir, f"lod_meta_{year1}_{year2}.json")
    return lod_geojson, meta_json

def save_tracking_cache(image_pair, track_dir: str, year1: str, year2: str,
                        track_params: dict, filenames: dict, dates: dict, version: str = "v1"):
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
    }
    with open(meta_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def load_lod_cache(image_pair, track_dir: str, year1: str, year2: str) -> bool:
    lod_geojson, _meta_json = lod_cache_paths(track_dir, year1, year2)
    if not os.path.exists(lod_geojson):
        return False
    try:
        image_pair.level_of_detection_points = gpd.read_file(lod_geojson)
        return True
    except Exception:
        return False
