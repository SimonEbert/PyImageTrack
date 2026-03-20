#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyImageTrack Pipeline Orchestrator

This module provides the main pipeline orchestrator for PyImageTrack, which:
- Loads configuration from TOML files
- Aligns image pairs using cross-correlation
- Tracks movement between images
- Filters outliers and calculates level of detection
- Generates plots and saves results
- Supports caching for improved performance

The main entry point is `run_from_config()` which processes all image pairs
specified in the configuration file.
"""

import argparse
import csv
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional


import tomllib

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from pyproj import CRS as PyprojCRS

from .ImageTracking.ImagePair import ImagePair
from .Parameters.FilterParameters import FilterParameters
from .Parameters.AlignmentParameters import AlignmentParameters
from .Parameters.TrackingParameters import TrackingParameters

from .Utils import (
    collect_pairs, ensure_dir, abbr_alignment,
    abbr_tracking, abbr_filter, abbr_enhancement, abbr_output_units, parse_date,
    make_effective_extents_from_deltas,
)

from .Cache import (
    load_alignment_cache, save_alignment_cache,
    load_tracking_cache, save_tracking_cache,
    load_lod_cache, save_lod_cache,
)
from .CreateGeometries.HandleGeometries import random_points_on_polygon_by_number
from .ConsoleOutput import ConsoleOutput, get_console, reset_console


def _resolve_config_path(path: str) -> Path:
    """
    Resolve a configuration file path to an absolute path.

    For relative paths, resolves from the repository root (where pyproject.toml
    is located) to keep CLI behavior stable regardless of current working directory.

    Parameters
    ----------
    path : str
        Path to the configuration file (can be relative or absolute).

    Returns
    -------
    Path
        Absolute path to the configuration file.
    """
    path_obj = Path(path)
    if not path_obj.is_absolute():
        # Resolve relative config paths from the repo root to keep CLI stable.
        repo_root = Path(__file__).resolve()
        while repo_root != repo_root.parent and not (repo_root / "pyproject.toml").exists():
            repo_root = repo_root.parent
        path_obj = repo_root / path_obj
    return path_obj


def _load_config(path: str | Path) -> dict:
    """
    Load a TOML configuration file.

    Parameters
    ----------
    path : str or Path
        Path to the TOML configuration file.

    Returns
    -------
    dict
        Dictionary containing the parsed configuration.

    Raises
    ------
    FileNotFoundError
        If the configuration file does not exist.
    """
    path_obj = Path(path).expanduser()
    if not path_obj.is_absolute():
        path_obj = path_obj.resolve()
    if not path_obj.exists():
        raise FileNotFoundError(f"Config file not found: {path_obj}")
    with path_obj.open("rb") as f:
        return tomllib.load(f)


def _get(cfg: dict, section: str, key: str, default=None):
    """
    Get a configuration value with optional default.

    Parameters
    ----------
    cfg : dict
        Configuration dictionary.
    section : str
        Configuration section name.
    key : str
        Configuration key within the section.
    default : any, optional
        Default value to return if the key is not found. Default is None.

    Returns
    -------
    any
        The configuration value or the default if not found.
    """
    if section not in cfg or key not in cfg[section]:
        return default
    return cfg[section][key]


def _require(cfg: dict, section: str, key: str):
    """
    Get a required configuration value.

    Parameters
    ----------
    cfg : dict
        Configuration dictionary.
    section : str
        Configuration section name.
    key : str
        Configuration key within the section.

    Returns
    -------
    any
        The configuration value.

    Raises
    ------
    KeyError
        If the configuration value is not found.
    """
    if section not in cfg or key not in cfg[section]:
        raise KeyError(f"Missing required config value: [{section}] {key}")
    return cfg[section][key]


def _as_optional_value(value):
    """
    Convert a value to None if it represents an empty or null value.

    Parameters
    ----------
    value : any
        Value to check.

    Returns
    -------
    any or None
        None if the value is None, empty string, "none", or "null" (case-insensitive).
        Otherwise returns the original value.
    """
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in ("", "none", "null"):
        return None
    return value


def _resolve_path(value, base_dir: Path):
    """
    Resolve a path relative to a base directory.

    Parameters
    ----------
    value : str or None
        Path to resolve. If None, returns None.
    base_dir : Path
        Base directory for resolving relative paths.

    Returns
    -------
    str or None
        Absolute path as string, or None if value is None.
    """
    if value is None:
        return None
    path_obj = Path(value)
    if not path_obj.is_absolute():
        path_obj = (base_dir / path_obj).resolve()
    return str(path_obj)


def _find_file_recursive(search_folder: str, filename: str) -> str | None:
    """
    Recursively search for a file by name in a folder and its subfolders.
    
    Parameters
    ----------
    search_folder : str
        The base folder to search in.
    filename : str
        The filename to search for.
    
    Returns
    -------
    str or None
        The full path to the file if found and unique, None if not found.
    
    Raises
    ------
    ValueError
        If the file is found in multiple locations.
    """
    from .ConsoleOutput import get_console
    console = get_console()
    
    matches = []
    search_path = Path(search_folder)
    
    # Recursively search for the file
    for root, dirs, files in os.walk(search_path):
        if filename in files:
            matches.append(str(Path(root) / filename))
    
    # If no matches found, return None
    if not matches:
        return None
    
    # If multiple matches found, raise error
    if len(matches) > 1:
        error_msg = f"Duplicate files found with name '{filename}':\n"
        for match in matches:
            error_msg += f"  - {match}\n"
        console.error(error_msg)
        raise ValueError(
            f"Multiple files with the name '{filename}' found in the input folder "
            "and its subdirectories. Please ensure each file has a unique name. "
            f"See console output for details."
        )
    
    # Return the single match
    return matches[0]


def _resolve_input_path(value: str | None, input_folder: str, config_dir: Path) -> str | None:
    """
    Resolve an input file path, searching recursively in input_folder and its subfolders.
    
    This function implements the following resolution strategy:
    1. If the path is absolute, use it as-is
    2. If the path is relative:
       a. Recursively search for the filename in input_folder and all subfolders
       b. If not found recursively, try to resolve it relative to config_dir (fallback)
    
    Parameters
    ----------
    value : str or None
        Path to resolve. If None, returns None.
    input_folder : str
        The input folder where files are searched first (searches recursively in subfolders).
    config_dir : Path
        The directory containing the config file (fallback for relative paths).

    Returns
    -------
    str or None
        Absolute path as string, or None if value is None.

    Raises
    ------
    ValueError
        If the file is found in multiple locations within input_folder subdirectories.
    """
    if value is None:
        return None
    
    path_obj = Path(value)
    
    # If absolute path, use it directly
    if path_obj.is_absolute():
        return str(path_obj)
    
    # Try recursive search in input_folder first
    recursive_match = _find_file_recursive(input_folder, value)
    if recursive_match is not None:
        return recursive_match
    
    # Fallback: resolve relative to config_dir (original behavior)
    config_path = config_dir / path_obj
    if config_path.exists():
        return str(config_path.resolve())
    
    # File not found anywhere
    return None


def _crs_label(crs) -> str:
    """
    Get a string label for a coordinate reference system.

    Parameters
    ----------
    crs : any
        Coordinate reference system object or None.

    Returns
    -------
    str
        String representation of the CRS, or "none" if CRS is None.
    """
    return "none" if crs is None else str(crs)


def _normalize_crs(crs):
    """
    Normalize a coordinate reference system to a pyproj CRS object.

    Parameters
    ----------
    crs : any
        Coordinate reference system in any format accepted by pyproj.

    Returns
    -------
    PyprojCRS or None
        Normalized CRS object, or None if input is None.
    """
    if crs is None:
        return None
    return PyprojCRS.from_user_input(crs)


def _resolve_common_crs(polygons_crs, image_path_1, image_path_2):
    """
    Resolve and validate the common coordinate reference system for images and polygons.

    Ensures that both images have the same CRS and that it matches the polygon CRS.

    Parameters
    ----------
    polygons_crs : any
        Coordinate reference system of the polygons.
    image_path_1 : str
        Path to the first image.
    image_path_2 : str
        Path to the second image.

    Returns
    -------
    any
        The common CRS, or None if both images and polygons have no CRS.

    Raises
    ------
    ValueError
        If CRS mismatch is detected between images or between images and polygons.
    """
    with rasterio.open(image_path_1, "r") as file1, rasterio.open(image_path_2, "r") as file2:
        image_crs_1 = file1.crs
        image_crs_2 = file2.crs

    if image_crs_1 is None and image_crs_2 is None:
        image_crs = None
    elif image_crs_1 is None or image_crs_2 is None:
        raise ValueError(
            f"CRS missing in one image: {image_path_1} crs={image_crs_1}, "
            f"{image_path_2} crs={image_crs_2}."
        )
    else:
        if _normalize_crs(image_crs_1) != _normalize_crs(image_crs_2):
            raise ValueError(
                f"Image CRS mismatch: {image_path_1} crs={image_crs_1}, {image_path_2} crs={image_crs_2}."
            )
        image_crs = image_crs_1

    if polygons_crs is None and image_crs is None:
        return None
    if polygons_crs is None or image_crs is None:
        raise ValueError(
            f"CRS missing for {'polygons' if polygons_crs is None else 'images'}; "
            "both polygons and images must have CRS or neither."
        )
    if _normalize_crs(polygons_crs) != _normalize_crs(image_crs):
        raise ValueError(
            f"CRS mismatch between polygons ({polygons_crs}) and images ({image_crs})."
        )
    return image_crs


def make_effective_extents_from_deltas(deltas, cell_size, years_between=1.0, cap_per_side=None):
    """
    Convert delta-per-year extents into effective absolute extents.

    Converts user-specified delta extents (posx,negx,posy,negy) into effective
    absolute extents by adding half the template size per side and scaling
    deltas by the time between observations.

    Parameters
    ----------
    deltas : tuple
        A 4-element tuple (dx+, dx-, dy+, dy-) representing extra pixels beyond
        half the template per year. Format: (positive_x, negative_x, positive_y, negative_y).
    cell_size : int or float
        The movement_cell_size or control_cell_size (template size).
    years_between : float, optional
        Time span in years between the two images. Default is 1.0.
    cap_per_side : int, optional
        Optional maximum value to clamp each side (to keep windows bounded).
        If None, no clamping is applied. Default is None.

    Returns
    -------
    tuple
        A 4-element tuple (posx, negx, posy, negy) of effective extents as integers,
        each >= half the cell size.
    """
    half = int(cell_size) // 2
    def one(v):
        eff = half + int(round(float(v) * float(years_between)))
        if cap_per_side is not None:
            eff = min(int(cap_per_side), eff)
        return max(half, eff)
    px, nx, py, ny = deltas
    return (one(px), one(nx), one(py), one(ny))


def run_from_config(config_path: str, verbose: bool = False, quiet: bool = False,
                    use_colors: bool = True, log_file: str = None,
                    log_level: str = 'INFO', log_max_bytes: int = 10 * 1024 * 1024,
                    log_backup_count: int = 5, identifier: Optional[str] = None):
    """
    Run the PyImageTrack pipeline from a configuration file.

    This is the main entry point for the PyImageTrack pipeline. It processes
    all image pairs specified in the configuration file, performing alignment,
    tracking, filtering, and output generation as configured.

    Parameters
    ----------
    config_path : str
        Path to the TOML configuration file.
    verbose : bool, optional
        Enable verbose output with more detailed information. Default is False.
    quiet : bool, optional
        Enable quiet mode with minimal output. Default is False.
    use_colors : bool, optional
        Use ANSI colors in terminal output. Default is True.
    log_file : str, optional
        Path to log file. If None or "auto", uses "pyimagetrack.log" in the
        output folder. Default is None.
    log_level : str, optional
        Logging level for file output. Must be one of: "DEBUG", "INFO",
        "WARNING", "ERROR", "CRITICAL". Default is "INFO".
    log_max_bytes : int, optional
        Maximum size of log file before rotation in bytes. Default is 10MB.
    log_backup_count : int, optional
        Number of backup log files to keep. Default is 5.
    identifier : Optional[str], optional
        If provided, only process files matching this identifier. Used in batch mode.
        Default is None (single mode, processes all files).

    Raises
    ------
    FileNotFoundError
        If the configuration file or input folder does not exist.
    PermissionError
        If input/output folders are not accessible.
    ValueError
        If configuration values are invalid.
    """
    # ==============================
    # CONFIG (TOML)
    # ==============================

    # Initialize console output
    config_path = _resolve_config_path(config_path)
    cfg = _load_config(config_path)
    config_dir = config_path.parent

    input_folder = _resolve_path(_require(cfg, "paths", "input_folder"), config_dir)
    output_folder = _resolve_path(_require(cfg, "paths", "output_folder"), config_dir)

    # Validate input and output folders
    if not os.path.isdir(input_folder):
        raise FileNotFoundError(f"Input folder does not exist or is not a directory: {input_folder}")
    if not os.access(input_folder, os.R_OK):
        raise PermissionError(f"Input folder is not readable: {input_folder}")
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    if not os.access(output_folder, os.W_OK):
        raise PermissionError(f"Output folder is not writable: {output_folder}")
    
    # If identifier is provided, create identifier subfolder
    if identifier is not None:
        output_folder = os.path.join(output_folder, identifier)
        os.makedirs(output_folder, exist_ok=True)

    # Setup log file path
    if log_file is None or log_file == "auto":
        # Use a single rotating log file instead of timestamped files
        log_file_path = Path(output_folder) / "pyimagetrack.log"
    else:
        log_file_path = Path(log_file)

    # Initialize global console instance with correct settings for use in other modules
    from .ConsoleOutput import get_console
    get_console(verbose=verbose, quiet=quiet, use_colors=use_colors, log_file=log_file_path,
                log_level=log_level, log_max_bytes=log_max_bytes, log_backup_count=log_backup_count)
    
    console = ConsoleOutput(
        verbose=verbose,
        quiet=quiet,
        use_colors=use_colors,
        log_file=log_file_path,
        log_level=log_level,
        log_max_bytes=log_max_bytes,
        log_backup_count=log_backup_count
    )

    # Show banner
    console.show_banner()
    console.config_loaded("Loaded configuration from", config_path)

    date_csv_path = _resolve_input_path(_as_optional_value(_get(cfg, "paths", "date_csv_path")), input_folder, config_dir)
    pairs_csv_path = _resolve_input_path(_as_optional_value(_get(cfg, "paths", "pairs_csv_path")), input_folder, config_dir)

    poly_outside_filename = _get(cfg, "polygons", "stable_area_filename", "none")
    poly_inside_filename = _require(cfg, "polygons", "moving_area_filename")
    moving_id_column = _get(cfg, "polygons", "moving_id_column", "moving_id")
    
    # Replace wildcard in polygon filenames with identifier if provided
    if identifier is not None:
        if '*' in poly_outside_filename:
            poly_outside_filename = poly_outside_filename.replace('*', identifier)
        if '*' in poly_inside_filename:
            poly_inside_filename = poly_inside_filename.replace('*', identifier)
    elif '*' in poly_outside_filename or '*' in poly_inside_filename:
        raise ValueError(
            "Wildcard pattern (*) in polygon filenames is only allowed when identifier is specified. "
            "Please provide an identifier or remove the wildcard from the polygon filename."
        )

    pairing_mode = _require(cfg, "pairing", "mode")

    use_no_georeferencing = bool(_get(cfg, "no_georef", "use_no_georeferencing", False))
    if use_no_georeferencing:
        fake_pixel_size = float(_get(cfg, "no_georef", "fake_pixel_size", 1.0))
        convert_to_3d_displacement = bool(_get(cfg, "no_georef", "convert_to_3d_displacement", False))
        undistort_image = bool(_get(cfg, "no_georef", "undistort_image", False))
        if undistort_image:
            camera_intrinsics_matrix = np.array(_require(cfg, "no_georef",
                                                         "camera_intrinsics_matrix"))
            camera_distortion_coefficients = np.array(_require(cfg, "no_georef",
                                                               "camera_distortion_coefficients"))
        else:
            camera_intrinsics_matrix = None
            camera_distortion_coefficients = None
        if convert_to_3d_displacement:
            camera_to_3d_coordinates_transform = np.array(_as_optional_value(_get(cfg,
                                                                                  "no_georef",
                                                                                  "camera_to_3d_coordinates_transform",
                                                                                  None)))
        else:
            camera_to_3d_coordinates_transform = None
    else:
        fake_pixel_size = None
        convert_to_3d_displacement = False
        undistort_image = False
        camera_intrinsics_matrix = None
        camera_distortion_coefficients = None
        camera_to_3d_coordinates_transform = None

    downsample_factor = _as_optional_value(_get(cfg, "downsampling", "downsample_factor", 1))
    downsample_factor = int(downsample_factor) if downsample_factor is not None else 1

    do_alignment = bool(_get(cfg, "flags", "do_alignment", True))
    do_tracking = bool(_get(cfg, "flags", "do_tracking", True))
    do_filtering = bool(_get(cfg, "flags", "do_filtering", True))
    do_image_enhancement = bool(_get(cfg, "flags", "do_image_enhancement", False))
    display_plots = bool(_get(cfg, "output", "display_plots", False))

    use_alignment_cache = bool(_get(cfg, "cache", "use_alignment_cache", True))
    use_tracking_cache = bool(_get(cfg, "cache", "use_tracking_cache", True))
    use_lod_cache = bool(_get(cfg, "cache", "use_lod_cache", use_tracking_cache))

    write_truecolor_aligned = bool(_get(cfg, "output", "write_truecolor_aligned", False))

    # output units mode (required)
    output_units_mode = _require(cfg, "output_units", "mode")
    if output_units_mode not in ("per_year", "total"):
        raise ValueError(
            f"Invalid output_units.mode: '{output_units_mode}'. "
            "Must be either 'per_year' or 'total'."
        )

    # adaptive tracking window options
    use_adaptive_tracking_window = bool(_get(cfg, "adaptive_tracking_window", "use_adaptive_tracking_window", False))

    # ==============================
    # PARAMETERS (alignment, tracking, filter)
    # ==============================
    # Get image_bands from tracking configuration (used by both alignment and tracking)
    image_bands = _require(cfg, "tracking", "image_bands")
    
    alignment_params = AlignmentParameters({
        "number_of_control_points": _require(cfg, "alignment", "number_of_control_points"),
        # search extent tuple: (right, left, down, up) in pixels around the control cell
        "control_search_extent_px": tuple(_require(cfg, "alignment", "control_search_extent_px")),
        "control_cell_size": _require(cfg, "alignment", "control_cell_size"),
        "cross_correlation_threshold_alignment": _require(cfg, "alignment", "cross_correlation_threshold_alignment"),
        "maximal_alignment_movement": _as_optional_value(_get(cfg, "alignment", "maximal_alignment_movement")),
        "image_bands": image_bands
    })

    tracking_params = TrackingParameters({
        "image_bands": image_bands,
        "distance_of_tracked_points_px": _require(cfg, "tracking", "distance_of_tracked_points_px"),
        "movement_cell_size": _require(cfg, "tracking", "movement_cell_size"),
        "cross_correlation_threshold_movement": _require(cfg, "tracking", "cross_correlation_threshold_movement"),
        # search extent tuple: (right, left, down, up) in pixels around the movement cell
        # usually this refers to the offset in px between the images,
        # but if the adaptive mode is used, this means the expected offset in px per year
        "search_extent_px": tuple(_require(cfg, "tracking", "search_extent_px")),
        "initial_shift_values": _as_optional_value(_get(cfg, "tracking", "initial_shift_values")),
        "initial_estimate_mode": _get(cfg, "tracking", "initial_estimate_mode", "count"),
        "nb_initial_estimate_peaks": _get(cfg, "tracking", "nb_initial_estimate_peaks", 1),
        "correlation_threshold_initial_estimates": _get(cfg, "tracking", "correlation_threshold_initial_estimates",None),
        "min_distance_initial_estimates": _get(cfg, "tracking", "min_distance_initial_estimates", 1),
    })

    filter_params = FilterParameters({
        "level_of_detection_quantile": _require(cfg, "filter", "level_of_detection_quantile"),
        "number_of_points_for_level_of_detection": _require(cfg, "filter", "number_of_points_for_level_of_detection"),
        "difference_movement_bearing_threshold": _require(cfg, "filter", "difference_movement_bearing_threshold"),
        "difference_movement_bearing_moving_window_size": _require(cfg, "filter", "difference_movement_bearing_moving_window_size"),
        "standard_deviation_movement_bearing_threshold": _require(cfg, "filter", "standard_deviation_movement_bearing_threshold"),
        "standard_deviation_movement_bearing_moving_window_size": _require(cfg, "filter", "standard_deviation_movement_bearing_moving_window_size"),
        "difference_movement_rate_threshold": _require(cfg, "filter", "difference_movement_rate_threshold"),
        "difference_movement_rate_moving_window_size": _require(cfg, "filter", "difference_movement_rate_moving_window_size"),
        "standard_deviation_movement_rate_threshold": _require(cfg, "filter", "standard_deviation_movement_rate_threshold"),
        "standard_deviation_movement_rate_moving_window_size": _require(cfg, "filter", "standard_deviation_movement_rate_moving_window_size"),
    })

    # ==============================
    # IMAGE ENHANCEMENT PARAMETERS
    # ==============================
    enhancement_params = {
        "type": _get(cfg, "image_enhancement", "type", "none"),
        "kernel_size": _get(cfg, "image_enhancement", "kernel_size"),
        "clip_limit": _get(cfg, "image_enhancement", "clip_limit"),
    }
    # If enhancement is disabled, force type to "none" for correct folder naming
    if not do_image_enhancement:
        enhancement_params["type"] = "none"
    enhancement_code = abbr_enhancement(enhancement_params)
    
    # Output units code (combined with enhancement for cache key)
    output_units_code = abbr_output_units(output_units_mode)
    enhancement_code = enhancement_code + "_" + output_units_code

    # ==============================
    # SAVE OPTIONS (final outputs)
    # ==============================
    save_files = list(_require(cfg, "save", "files"))
    if not save_files:
        raise ValueError("save_files list cannot be empty. At least one output file type must be specified.")

    # Allow JPG/JPEG only if explicitly opted into fake georeferencing
    extensions = (".tif", ".tiff") if not use_no_georeferencing else (".tif", ".tiff", ".jpg", ".jpeg")

    # Collect pairs, optionally filtering by identifier
    if identifier is not None:
        year_pairs, id_to_file, id_to_date, id_hastime_from_filename, id_to_identifier = collect_pairs(
            input_folder=input_folder,
            date_csv_path=date_csv_path,
            pairs_csv_path=pairs_csv_path,
            pairing_mode=pairing_mode,
            extensions=extensions,
            identifier=identifier
        )
    else:
        year_pairs, id_to_file, id_to_date, id_hastime_from_filename = collect_pairs(
            input_folder=input_folder,
            date_csv_path=date_csv_path,
            pairs_csv_path=pairs_csv_path,
            pairing_mode=pairing_mode,
            extensions=extensions
        )

    console.info(f"Image pairs to process ({pairing_mode}): {len(year_pairs)}")

    # Load polygons
    poly_outside = None
    poly_outside_filename_resolved = _as_optional_value(poly_outside_filename)
    if poly_outside_filename_resolved is not None:
        # Resolve polygon path: check input_folder first, then support absolute paths
        poly_outside_path = _resolve_input_path(poly_outside_filename_resolved, input_folder, config_dir)
        try:
            poly_outside = gpd.read_file(poly_outside_path)
            if use_no_georeferencing:
                poly_outside = poly_outside.set_crs(None, allow_override=True)
            # Allow multiple features: combine to one reference polygon
            if len(poly_outside) > 0:
                poly_outside = gpd.GeoDataFrame(
                    geometry=[poly_outside.unary_union],
                    crs=poly_outside.crs,
                )
        except Exception as e:
            console.warning(f"Could not load stable area file '{poly_outside_filename_resolved}': {e}")
            console.warning("Using fallback mode (image_bounds minus moving_area as stable area).")
            poly_outside = None
    
    # Resolve polygon path: check input_folder first, then support absolute paths
    polygon_inside_path = _resolve_input_path(poly_inside_filename, input_folder, config_dir)
    if not os.path.exists(polygon_inside_path):
        raise FileNotFoundError(f"Moving area polygon file does not exist: {polygon_inside_path}")
    polygon_inside = gpd.read_file(polygon_inside_path)
    if len(polygon_inside) == 0:
        raise ValueError(f"Moving area polygon file is empty: {polygon_inside_path}")
    if use_no_georeferencing:
        polygon_inside = polygon_inside.set_crs(None, allow_override=True)

    # Ensure moving ID column exists and is filled (single- or multi-feature shapefiles)
    _missing_col = moving_id_column not in polygon_inside.columns
    if _missing_col:
        polygon_inside[moving_id_column] = pd.Series([None] * len(polygon_inside))
    missing_id_mask = polygon_inside[moving_id_column].isna() | (polygon_inside[moving_id_column].astype(str).str.strip() == "")
    if missing_id_mask.any():
        # Only warn if there are multiple polygons (single polygon doesn't need differentiation)
        if len(polygon_inside) > 1:
            console.warning(
                f"Moving area polygons missing '{moving_id_column}' values. Filling with row indices as strings."
            )
        # Always fill missing values so the grouping doesn't fail
        polygon_inside.loc[missing_id_mask, moving_id_column] = (
            polygon_inside.loc[missing_id_mask].index.astype(str)
        )

    # Group moving polygons by ID and dissolve geometry per ID
    moving_groups = []
    for id_value, group_df in polygon_inside.groupby(moving_id_column):
        union_geom = group_df.unary_union
        moving_groups.append(
            (str(id_value), gpd.GeoDataFrame(geometry=[union_geom], crs=group_df.crs))
        )

    # CRS validation
    if poly_outside is not None:
        poly_outside_crs = poly_outside.crs
        polygon_inside_crs = polygon_inside.crs
        if (poly_outside_crs is None) != (polygon_inside_crs is None):
            raise ValueError(
                "Polygon CRS mismatch: outside has "
                + _crs_label(poly_outside_crs)
                + ", inside has "
                + _crs_label(polygon_inside_crs)
            )
        if poly_outside_crs is not None and _normalize_crs(poly_outside_crs) != _normalize_crs(polygon_inside_crs):
            raise ValueError(
                "Polygon CRS mismatch: outside has "
                + _crs_label(poly_outside_crs)
                + ", inside has "
                + _crs_label(polygon_inside_crs)
            )
        polygons_crs = poly_outside_crs
    else:
        # When poly_outside is None, use polygon_inside CRS
        polygons_crs = polygon_inside.crs

    align_code  = abbr_alignment(alignment_params)
    # track_code and filter_code may depend on pair-specific overrides, so they are computed per pair

    successes, skipped = [], []

    # Start overall timer
    start_time = time.time()
    for year1, year2 in year_pairs:
            if year1 not in id_to_date or year2 not in id_to_date:
                skipped.append((year1, year2, "Date missing in CSV"))
                continue
            if year1 not in id_to_file or year2 not in id_to_file:
                skipped.append((year1, year2, "Input image missing"))
                continue

            filename_1 = id_to_file[year1]
            filename_2 = id_to_file[year2]
            date_1 = id_to_date[year1]
            date_2 = id_to_date[year2]

            dt1 = parse_date(date_1)
            dt2 = parse_date(date_2)

            def _fmt_label(id_key, dt):
                return dt.strftime("%Y-%m-%d %H:00") if id_hastime_from_filename.get(id_key, False) \
                    else dt.strftime("%Y-%m-%d")

            label_1 = _fmt_label(year1, dt1)
            label_2 = _fmt_label(year2, dt2)
            
            # Extract date tokens (years) from IDs for display and paths
            # IDs are now in format "date_token_identifier" or "date_token"
            date_token_1 = year1.split('_')[0] if '_' in year1 else year1
            date_token_2 = year2.split('_')[0] if '_' in year2 else year2
            
            # Use the already-parsed dates for folder naming (guaranteed correct)
            # This avoids malformed tokens from extraction issues
            folder_date_1 = dt1.strftime("%Y-%m-%d") if dt1.hour == 0 and dt1.minute == 0 and dt1.second == 0 else dt1.strftime("%Y-%m-%d-%H-%M")
            folder_date_2 = dt2.strftime("%Y-%m-%d") if dt2.hour == 0 and dt2.minute == 0 and dt2.second == 0 else dt2.strftime("%Y-%m-%d-%H-%M")
            
            # Image pair header
            pair_id_short = f"{date_token_1} -> {date_token_2}"
            if identifier is not None:
                pair_id_short += f"; id: {identifier}"
            console.section_header("PREPROCESSING", "Loading and preparing images", f"({pair_id_short})", level=2)
            console.info_verbose(f"File 1: {filename_1}")
            console.info_verbose(f"File 2: {filename_2}")

            if True: #try:
                image_crs = None if use_no_georeferencing else _resolve_common_crs(polygons_crs, filename_1, filename_2)
                # compute years_between (hour-precise)
                delta_hours = (dt2 - dt1).total_seconds() / 3600.0
                years_between = delta_hours / (24.0 * 365.25)
                console.info_verbose(f"Time between observations: {ConsoleOutput.format_duration(delta_hours)}")


            # alignment: convert user-entered deltas -> effective extents
            base_align_deltas = alignment_params.control_search_extent_px
            effective_align_extents = make_effective_extents_from_deltas(
                deltas=base_align_deltas,
                cell_size=alignment_params.control_cell_size,
                years_between=1.0,
                cap_per_side=None
            )

            # use deltas for folder code (so names reflect what user typed)
            pair_alignment_config_for_code = {
                "number_of_control_points": alignment_params.number_of_control_points,
                "control_cell_size": alignment_params.control_cell_size,
                "cross_correlation_threshold_alignment": alignment_params.cross_correlation_threshold_alignment,
                "control_search_extent_px": base_align_deltas,
            }
            align_code = abbr_alignment(pair_alignment_config_for_code)

            # movement: convert user-entered deltas -> effective extents
            base_track_deltas = tracking_params.search_extent_px
            adaptive_extents = make_effective_extents_from_deltas(
                deltas=base_track_deltas,
                cell_size=tracking_params.movement_cell_size,
                years_between=years_between if use_adaptive_tracking_window else 1.0,
                cap_per_side=None
            )

            pair_tracking_config_for_code = {
                "image_bands": tracking_params.image_bands,
                "distance_of_tracked_points_px": tracking_params.distance_of_tracked_points_px,
                "movement_cell_size": tracking_params.movement_cell_size,
                "cross_correlation_threshold_movement": tracking_params.cross_correlation_threshold_movement,
                "search_extent_px": base_track_deltas,  # user-entered deltas for folder code
            }

            pair_tracking_config = {
                "image_bands": tracking_params.image_bands,
                "distance_of_tracked_points_px": tracking_params.distance_of_tracked_points_px,
                "movement_cell_size": tracking_params.movement_cell_size,
                "cross_correlation_threshold_movement": tracking_params.cross_correlation_threshold_movement,
                "search_extent_deltas": base_track_deltas,
                "search_extent_px_effective": adaptive_extents,
            }
            
            track_code  = abbr_tracking(pair_tracking_config_for_code)
            filter_code = abbr_filter(filter_params)

            # Determine if alignment will be performed or loaded from cache
            will_do_alignment = do_alignment
            will_load_alignment_cache = False
            if do_alignment and use_alignment_cache:
                # We'll try to load from cache, but we don't know yet if it exists
                # For directory structure, we assume alignment will be done
                will_load_alignment_cache = True
            
            # Determine if filtering will be performed
            will_do_filtering = do_filtering

            # Set alignment code to "A_none" if alignment is disabled and not loaded from cache
            if not will_do_alignment:
                align_code = "A_none"
            
            # Set filter code to "F_none" if filtering is disabled
            if not will_do_filtering:
                filter_code = "F_none"

            # Directories
            # Use formatted dates for directory names (guaranteed correct, handles all date formats)
            base_pair_dir = os.path.join(output_folder, f"{folder_date_1}_{folder_date_2}")
            enhancement_dir = os.path.join(base_pair_dir, enhancement_code)
            align_dir  = os.path.join(enhancement_dir, align_code)
            track_dir  = os.path.join(align_dir,     track_code)
            filter_dir = os.path.join(track_dir,     filter_code)
            for d in (enhancement_dir, align_dir, track_dir, filter_dir):
                ensure_dir(d)

            # Params for ImagePair:
            #   - effective extents in the fields the algorithms read
            #   - user-entered deltas in separate keys for logging
            param_dict = {}
            param_dict.update(alignment_params.to_dict())
            param_dict.update(tracking_params.to_dict())

            param_dict["control_search_extent_px"]          = effective_align_extents   # used by code
            param_dict["control_search_extent_deltas"]      = base_align_deltas         # user input (for logs)
            param_dict["search_extent_px"]                  = adaptive_extents          # used by code
            param_dict["search_extent_deltas"]              = base_track_deltas         # user input (for logs)
            param_dict["use_no_georeferencing"]             = bool(use_no_georeferencing)
            if fake_pixel_size is not None:
                param_dict["fake_pixel_size"] = float(fake_pixel_size)
            param_dict["downsample_factor"]                 = int(downsample_factor)
            param_dict["undistort_image"]                   = undistort_image
            param_dict["camera_intrinsics_matrix"]          = camera_intrinsics_matrix
            param_dict["camera_distortion_coefficients"]    = camera_distortion_coefficients
            param_dict["convert_to_3d_displacement"]        = convert_to_3d_displacement
            param_dict["camera_to_3d_coordinates_transform"]= camera_to_3d_coordinates_transform
            # Image enhancement parameters
            param_dict["enhancement_type"]                 = enhancement_params.get("type", "none")
            param_dict["enhancement_kernel_size"]           = enhancement_params.get("kernel_size", 50)
            param_dict["enhancement_clip_limit"]            = enhancement_params.get("clip_limit", 0.9)
            # Output units mode
            param_dict["output_units_mode"]                = output_units_mode
            # Adaptive tracking window
            param_dict["use_adaptive_tracking_window"]     = use_adaptive_tracking_window
            # Image bands (ensure both key names are available for compatibility)
            param_dict["image_bands"]                      = tracking_params.image_bands

            param_dict["crs"]                               = image_crs
            param_dict["moving_id_column"]                  = moving_id_column
 
            image_pair = ImagePair(parameter_dict=param_dict)
            image_pair.load_images_from_file(
                filename_1=filename_1,
                observation_date_1=date_1,
                filename_2=filename_2,
                observation_date_2=date_2,
                selected_channels=tracking_params.image_bands
            )

            # optional image enhancement (CLAHE) before alignment/tracking
            if do_image_enhancement and hasattr(image_pair, "equalize_adapthist_images"):
                image_pair.equalize_adapthist_images()

            # alignment with cache
            if do_alignment:
                console.section_header("ALIGNMENT", "Co-registering image pair", f"({pair_id_short})", level=2)
                used_cache_alignment = False
                if use_alignment_cache:
                    used_cache_alignment = load_alignment_cache(image_pair, align_dir, year1, year2)
                    if used_cache_alignment:
                        console.cache_info("loaded", align_dir, f"{date_token_1}->{date_token_2}", cache_type="alignment")

                if not used_cache_alignment:
                    console.parameter_summary({
                        "Number of control points": alignment_params.number_of_control_points,
                        "Control cell size": f"{alignment_params.control_cell_size} px",
                        "Search extent": f"{alignment_params.control_search_extent_px} px",
                        "Cross-correlation threshold": alignment_params.cross_correlation_threshold_alignment,
                        "Maximal alignment movement": alignment_params.maximal_alignment_movement
                    })
                    
                    # When poly_outside is None, align_images will use image_bounds minus polygon_inside
                    try:
                        with console.timer("Alignment", verbose=True):
                                image_pair.align_images(poly_outside, polygon_inside=polygon_inside)
                    except ValueError as e:
                        console.error(f"Alignment failed for pair {date_token_1} -> {date_token_2}: {e}")
                        console.error("Skipping this pair. Please check your alignment parameters or input data.")
                        skipped.append((year1, year2, f"Alignment failed: {str(e)}"))
                        continue
                    
                    if not image_pair.valid_alignment_possible:
                        skipped.append((year1, year2, "Alignment not possible"))
                        continue
                
                # Save alignment results only if not loaded from cache
                if not used_cache_alignment:
                    save_alignment_cache(
                        image_pair, align_dir, year1, year2,
                        align_params=alignment_params.__dict__,
                        filenames={year1: filename_1, year2: filename_2},
                        dates={year1: date_1, year2: date_2},
                        save_truecolor_aligned=write_truecolor_aligned,
                    )
                    console.cache_info("saved", align_dir, f"{date_token_1}->{date_token_2}", cache_type="alignment")

            else:
                console.section_header("ALIGNMENT", "Co-registering image pair", f"({pair_id_short})", level=2)
                console.info("Alignment is disabled (skipping this step).")
                image_pair.valid_alignment_possible = True
                image_pair.images_aligned = False
                used_cache_alignment = False


            # ==============================
            # TRACKING (optional)
            # ==============================
            used_cache_tracking = False
            if do_tracking:
                console.section_header("TRACKING", "Detecting movement between images", f"({pair_id_short})", level=2)
                if use_tracking_cache:
                    used_cache_tracking = load_tracking_cache(image_pair, track_dir, year1, year2)
                    if used_cache_tracking:
                        if use_no_georeferencing and getattr(image_pair.tracking_results, "crs", None) is not None:
                            used_cache_tracking = False
                            image_pair.tracking_results = None
                            console.warning("CRS not compatible with no-georef; recomputing.")
                        else:
                            console.cache_info("loaded", track_dir, f"{date_token_1}->{date_token_2}", cache_type="tracking")


                if not used_cache_tracking:
                    adaptive_info = f" (scaled by {years_between:.3f} years)" if use_adaptive_tracking_window else " (disabled)"
                    console.parameter_summary({
                        "Image bands": tracking_params.image_bands,
                        "Distance of tracked points": f"{tracking_params.distance_of_tracked_points_px} px",
                        "Movement cell size": f"{tracking_params.movement_cell_size} px",
                        "Search extent": f"{tracking_params.search_extent_px} px",
                        "Cross-correlation threshold": tracking_params.cross_correlation_threshold_movement,
                        "Adaptive tracking window": f"enabled{adaptive_info}"
                    })
                    
                    # Track points regardless of alignment status
                    # If images are not aligned, track_points() will issue a warning
                    with console.timer("Tracking", verbose=True):
                        tracked_points_list = []
                        for moving_id_value, moving_gdf in moving_groups:
                            tracked_sub = image_pair.track_points(tracking_area=moving_gdf)
                            tracked_sub[moving_id_column] = moving_id_value
                            tracked_points_list.append(tracked_sub)
                        if not tracked_points_list:
                            raise ValueError("No tracking points found. track_points() returned empty results for all moving areas.")
                        tracked_points = pd.concat(tracked_points_list, ignore_index=True)
                    image_pair.tracking_results = tracked_points
                
                # Save tracking results only if not loaded from cache
                if not used_cache_tracking:
                    save_tracking_cache(
                        image_pair,
                        track_dir,
                        year1,
                        year2,
                        track_params=pair_tracking_config,
                        filenames={year1: filename_1, year2: filename_2},
                        dates={year1: date_1, year2: date_2},
                    )
                    console.cache_info("saved", track_dir, f"{date_token_1}->{date_token_2}", cache_type="tracking")
            else:
                console.info("Tracking is disabled (alignment-only run).")


            # ==============================
            # FILTERING + PLOTS + SAVING (optional)
            # ==============================
            if do_tracking:
                if do_filtering:
                    console.section_header("FILTERING", "Removing outliers from tracking results", f"({pair_id_short})", level=2)
                    
                    # First: Load or calculate Level of Detection
                    used_cache_lod = False
                    if use_lod_cache:
                        used_cache_lod = load_lod_cache(image_pair, track_dir, year1, year2)
                        if used_cache_lod:
                            if use_no_georeferencing and getattr(image_pair.level_of_detection_points, "crs", None) is not None:
                                used_cache_lod = False
                                image_pair.level_of_detection_points = None
                                console.warning("CRS not compatible with no-georef; recomputing.")
                            else:
                                console.cache_info("loaded", track_dir, f"{date_token_1}->{date_token_2}", cache_type="LoD")

                    if not used_cache_lod:
                        if poly_outside is not None:
                            lod_points = random_points_on_polygon_by_number(
                                poly_outside,
                                filter_params.number_of_points_for_level_of_detection
                            )
                        else:
                            # Use image_bounds minus moving_area for LoD calculation
                            lod_polygon = gpd.GeoDataFrame(
                                geometry=image_pair.image_bounds.difference(polygon_inside),
                                crs=image_pair.crs
                            )
                            lod_polygon = lod_polygon.rename(columns={0: 'geometry'})
                            lod_polygon.set_geometry('geometry', inplace=True)
                            lod_points = random_points_on_polygon_by_number(
                                lod_polygon,
                                filter_params.number_of_points_for_level_of_detection
                            )
                        image_pair.calculate_lod(lod_points, filter_parameters=filter_params)
                    
                    # Save LoD results only if not loaded from cache
                    if not used_cache_lod:
                        save_lod_cache(
                            image_pair,
                            track_dir,
                            year1,
                            year2,
                            filenames={year1: filename_1, year2: filename_2},
                            dates={year1: date_1, year2: date_2},
                        )
                        console.cache_info("saved", track_dir, f"{date_token_1}->{date_token_2}", cache_type="LoD")

                    # Second: Apply LoD filtering (remove points below detection threshold)
                    image_pair.filter_lod_points()
                    
                    # Third: Apply outlier filtering (statistical filters on remaining points)
                    image_pair.filter_outliers(filter_parameters=filter_params)
                else:
                    console.section_header("FILTERING", "Removing outliers from tracking results", f"({pair_id_short})", level=2)
                    console.info("Filtering is disabled (skipping this step).")

                if display_plots:
                    image_pair.plot_tracking_results_with_valid_mask()

                # write a small CSV with valid fraction
                try:
                    df_vf = image_pair.tracking_results
                    if moving_id_column in df_vf.columns:
                        grouped = df_vf.groupby(moving_id_column)["valid"].mean()
                        valid_rows = [(f"{date_token_1}_{date_token_2}", str(k), float(v)) for k, v in grouped.items()]
                    else:
                        valid_rows = [(f"{date_token_1}_{date_token_2}", "all", float(df_vf["valid"].mean()))]
                except Exception:
                    valid_rows = [(f"{date_token_1}_{date_token_2}", "all", None)]
                valid_csv = os.path.join(filter_dir, "valid_results_fraction.csv")
                with open(valid_csv, "w", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow(["pair", moving_id_column, "valid_fraction"])
                    for pair_label, id_val, vf in valid_rows:
                        w.writerow([pair_label, id_val, vf if vf is not None else "NA"])

                # final results go to the filter level
                console.section_header("OUTPUT", "Saving results", f"({pair_id_short})", level=2)
                if save_files:
                    console.processing("Saving results.")
                    image_pair.save_full_results(filter_dir, save_files=save_files)
                    console.success("Saved results.")
                    console.file_list("", save_files)
                else:
                    console.info("No output files configured (skipping this step).")
                    image_pair.save_full_results(filter_dir, save_files=save_files)
            else:
                console.section_header("FILTERING", "Removing outliers from tracking results", f"({pair_id_short})", level=2)
                console.info("Filtering is disabled (skipping this step).")
                console.section_header("OUTPUT", "Saving results", f"({pair_id_short})", level=2)
                console.info("Skipping filtering, plotting and saving of movement products (alignment-only mode).")
                # Alignment-only outputs exist in align_dir:
                # - aligned_image_<year2>.tif
                # - alignment_control_points_<year1>_<year2>.geojson
                # - alignment_meta_<year1>_<year2>.json
                        # Mark this pair as successfully processed

            # Extract identifier from year IDs for summary display
            identifier_from_id = None
            if '_' in year1:
                identifier_from_id = year1.split('_')[1]
            successes.append((date_token_1, date_token_2, identifier_from_id))

    # Print summary with total elapsed time
    total_elapsed = time.time() - start_time
    console.print_summary(successes, skipped, total_elapsed=total_elapsed)


# CLI entry point
# ==============================
# MAIN
# ==============================
def main(argv=None):
    """
    Command-line interface entry point for PyImageTrack.

    Parses command-line arguments and runs the pipeline.

    Parameters
    ----------
    argv : list, optional
        Command-line arguments. If None, uses sys.argv. Default is None.
    """
    parser = argparse.ArgumentParser(
        description="PyImageTrack: Image alignment and movement tracking pipeline"
    )
    parser.add_argument("--config", required=True, help="Path to TOML config file")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--quiet", action="store_true", help="Enable quiet mode (minimal output)")
    parser.add_argument("--no-color", action="store_true", help="Disable colored output")
    parser.add_argument("--log-file", type=str, default=None,
                       help="Path to log file (default: pyimagetrack.log in output folder)")
    parser.add_argument("--log-level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                       help="Logging level for file output (default: INFO)")
    parser.add_argument("--log-max-bytes", type=int, default=10 * 1024 * 1024,
                       help="Maximum log file size before rotation (default: 10MB)")
    parser.add_argument("--log-backup-count", type=int, default=5,
                       help="Number of backup log files to keep (default: 5)")
    args = parser.parse_args()
    try:
        run_from_config(args.config, verbose=args.verbose, quiet=args.quiet,
                        use_colors=not args.no_color, log_file=args.log_file,
                        log_level=args.log_level, log_max_bytes=args.log_max_bytes,
                        log_backup_count=args.log_backup_count)
    except Exception as e:
        sys.stderr.write(f"\nERROR: {e}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
