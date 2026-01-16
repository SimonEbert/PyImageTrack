#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Orchestrator: align, track, filter, plot, save with caching.
"""

import argparse
import csv
import os
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "tomllib is required to read TOML configs. Use Python 3.11+ or install tomli."
    ) from exc

import geopandas as gpd
import numpy as np
import rasterio
from pyproj import CRS as PyprojCRS

from .ImageTracking.ImagePair import ImagePair
from .Parameters.FilterParameters import FilterParameters
from .Parameters.AlignmentParameters import AlignmentParameters
from .Parameters.TrackingParameters import TrackingParameters

from .Utils import (
    collect_pairs, ensure_dir, abbr_alignment, 
    abbr_tracking, abbr_filter, parse_date,
)

from .Cache import (
    load_alignment_cache, save_alignment_cache,
    load_tracking_cache, save_tracking_cache,
    load_lod_cache, save_lod_cache,
)
from .CreateGeometries.HandleGeometries import random_points_on_polygon_by_number


def _resolve_config_path(path: str) -> Path:
    path_obj = Path(path)
    if not path_obj.is_absolute():
        # Resolve relative config paths from the repo root to keep CLI stable.
        repo_root = Path(__file__).resolve()
        while repo_root != repo_root.parent and not (repo_root / "pyproject.toml").exists():
            repo_root = repo_root.parent
        path_obj = repo_root / path_obj
    return path_obj


def _load_config(path: str | Path) -> dict:
    path_obj = Path(path).expanduser()
    if not path_obj.is_absolute():
        path_obj = path_obj.resolve()
    if not path_obj.exists():
        raise FileNotFoundError(f"Config file not found: {path_obj}")
    with path_obj.open("rb") as f:
        return tomllib.load(f)


def _get(cfg: dict, section: str, key: str, default=None):
    if section not in cfg or key not in cfg[section]:
        return default
    return cfg[section][key]


def _require(cfg: dict, section: str, key: str):
    if section not in cfg or key not in cfg[section]:
        raise KeyError(f"Missing required config value: [{section}] {key}")
    return cfg[section][key]


def _as_optional_value(value):
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in ("", "none", "null"):
        return None
    return value


def _resolve_path(value, base_dir: Path):
    if value is None:
        return None
    path_obj = Path(value)
    if not path_obj.is_absolute():
        path_obj = (base_dir / path_obj).resolve()
    return str(path_obj)


def _crs_label(crs) -> str:
    return "none" if crs is None else str(crs)


def _normalize_crs(crs):
    if crs is None:
        return None
    return PyprojCRS.from_user_input(crs)


def _resolve_common_crs(polygons_crs, image_path_1, image_path_2):
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
    Convert delta-per-year extents (posx,negx,posy,negy) into effective absolute extents
    by adding half the template size per side and scaling deltas by years_between.

    deltas: (dx+, dx-, dy+, dy-) meaning *extra* pixels beyond half the template per year.
    cell_size: movement_cell_size or control_cell_size
    years_between: time span in years between the two images
    cap_per_side: optional int to clamp each side (to keep windows bounded)

    Returns (posx, negx, posy, negy) as ints >= half.
    """
    half = int(cell_size) // 2
    def one(v):
        eff = half + int(round(float(v) * float(years_between)))
        if cap_per_side is not None:
            eff = min(int(cap_per_side), eff)
        return max(half, eff)
    px, nx, py, ny = deltas
    return (one(px), one(nx), one(py), one(ny))


def _recompute_lod_from_points(image_pair, filter_params) -> bool:
    points = getattr(image_pair, "level_of_detection_points", None)
    if points is None or len(points) == 0:
        return False
    quantile = filter_params.level_of_detection_quantile
    if quantile is None or "movement_distance_per_year" not in points.columns:
        return False
    image_pair.level_of_detection = np.nanquantile(points["movement_distance_per_year"], quantile)
    unit_name = points.crs.axis_info[0].unit_name if points.crs is not None else "pixel"
    print("Found level of detection with quantile " + str(quantile) + " as "
          + str(np.round(image_pair.level_of_detection, decimals=5)) + " " + str(unit_name) + "/year")
    return True



def run_from_config(config_path: str):
    # ==============================
    # CONFIG (TOML)
    # ==============================


    config_path = _resolve_config_path(config_path)
    cfg = _load_config(config_path)
    config_dir = config_path.parent

    input_folder = _resolve_path(_require(cfg, "paths", "input_folder"), config_dir)
    output_folder = _resolve_path(_require(cfg, "paths", "output_folder"), config_dir)

    date_csv_path = _resolve_path(_as_optional_value(_get(cfg, "paths", "date_csv_path")), config_dir)
    pairs_csv_path = _resolve_path(_as_optional_value(_get(cfg, "paths", "pairs_csv_path")), config_dir)

    poly_outside_filename = _require(cfg, "polygons", "outside_filename")
    poly_inside_filename = _require(cfg, "polygons", "inside_filename")

    pairing_mode = _require(cfg, "pairing", "mode")

    use_no_georeferencing = bool(_get(cfg, "no_georef", "use_no_georeferencing", False))
    fake_pixel_size = float(_get(cfg, "no_georef", "fake_pixel_size", 1.0))
    undistort_image = bool(_get(cfg, "no_georef", "undistort_image", False))
    camera_intrinsics_matrix = None
    camera_distortion_coefficients = None
    if undistort_image:
        camera_intrinsics_matrix = np.array(
            _require(cfg, "no_georef", "camera_intrinsics_matrix")
        )
        camera_distortion_coefficients = np.array(
            _require(cfg, "no_georef", "camera_distortion_coefficients")
        )

    downsample_factor = _as_optional_value(_get(cfg, "downsampling", "downsample_factor", 1))
    downsample_factor = int(downsample_factor) if downsample_factor is not None else 1

    do_alignment = bool(_get(cfg, "flags", "do_alignment", True))
    do_tracking = bool(_get(cfg, "flags", "do_tracking", True))
    do_filtering = bool(_get(cfg, "flags", "do_filtering", True))
    do_plotting = bool(_get(cfg, "flags", "do_plotting", True))
    do_image_enhancement = bool(_get(cfg, "flags", "do_image_enhancement", False))

    use_alignment_cache = bool(_get(cfg, "cache", "use_alignment_cache", True))
    use_tracking_cache = bool(_get(cfg, "cache", "use_tracking_cache", True))
    force_recompute_alignment = bool(_get(cfg, "cache", "force_recompute_alignment", False))
    force_recompute_tracking = bool(_get(cfg, "cache", "force_recompute_tracking", False))
    use_lod_cache = bool(_get(cfg, "cache", "use_lod_cache", use_tracking_cache))
    force_recompute_lod = bool(_get(cfg, "cache", "force_recompute_lod", False))

    write_truecolor_aligned = bool(_get(cfg, "output", "write_truecolor_aligned", False))

    # adaptive tracking window options
    use_adaptive_tracking_window = bool(_get(cfg, "adaptive_tracking_window", "use_adaptive_tracking_window", False))

    # ==============================
    # PARAMETERS (alignment, tracking, filter)
    # ==============================
    alignment_params = AlignmentParameters({
        "number_of_control_points": _require(cfg, "alignment", "number_of_control_points"),
        # search extent tuple: (right, left, down, up) in pixels around the control cell
        "control_search_extent_px": tuple(_require(cfg, "alignment", "control_search_extent_px")),
        "control_cell_size": _require(cfg, "alignment", "control_cell_size"),
        "cross_correlation_threshold_alignment": _require(cfg, "alignment", "cross_correlation_threshold_alignment"),
        "maximal_alignment_movement": _as_optional_value(_get(cfg, "alignment", "maximal_alignment_movement")),
    })

    tracking_params = TrackingParameters({
        "image_bands": _require(cfg, "tracking", "image_bands"),
        "distance_of_tracked_points_px": _require(cfg, "tracking", "distance_of_tracked_points_px"),
        "movement_cell_size": _require(cfg, "tracking", "movement_cell_size"),
        "cross_correlation_threshold_movement": _require(cfg, "tracking", "cross_correlation_threshold_movement"),
        # search extent tuple: (right, left, down, up) in pixels around the movement cell
        # usually this refers to the offset in px between the images,
        # but if the adaptive mode is used, this means the expected offset in px per year
        "search_extent_px": tuple(_require(cfg, "tracking", "search_extent_px")),
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
    # SAVE OPTIONS (final outputs)
    # ==============================
    save_files = list(_require(cfg, "save", "files"))

    # Allow JPG/JPEG only if explicitly opted into fake georeferencing
    extensions = (".tif", ".tiff") if not use_no_georeferencing else (".tif", ".tiff", ".jpg", ".jpeg")

    year_pairs, id_to_file, id_to_date, id_hastime_from_filename = collect_pairs(
        input_folder=input_folder,
        date_csv_path=date_csv_path,
        pairs_csv_path=pairs_csv_path,
        pairing_mode=pairing_mode,
        extensions=extensions
    )

    print(f"Image pairs to process ({pairing_mode}): {len(year_pairs)}")

    polygon_outside = gpd.read_file(os.path.join(input_folder, poly_outside_filename))
    polygon_inside = gpd.read_file(os.path.join(input_folder, poly_inside_filename))
    polygon_outside_crs = polygon_outside.crs
    polygon_inside_crs = polygon_inside.crs
    if (polygon_outside_crs is None) != (polygon_inside_crs is None):
        raise ValueError(
            "Polygon CRS mismatch: outside has "
            + _crs_label(polygon_outside_crs)
            + ", inside has "
            + _crs_label(polygon_inside_crs)
        )
    if polygon_outside_crs is not None and _normalize_crs(polygon_outside_crs) != _normalize_crs(polygon_inside_crs):
        raise ValueError(
            "Polygon CRS mismatch: outside has "
            + _crs_label(polygon_outside_crs)
            + ", inside has "
            + _crs_label(polygon_inside_crs)
        )
    polygons_crs = polygon_outside_crs

    align_code  = abbr_alignment(alignment_params)
    # track_code and filter_code may depend on pair-specific overrides, so they are computed per pair

    successes, skipped = [], []

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
        print(f"\nProcessed image pair: {year1} ({label_1}) → {year2} ({label_2})")

        print(f"   File 1: {filename_1}")
        print(f"   File 2: {filename_2}")

        try:
            image_crs = _resolve_common_crs(polygons_crs, filename_1, filename_2)
            # compute years_between (hour-precise)
            delta_hours = (dt2 - dt1).total_seconds() / 3600.0
            years_between = delta_hours / (24.0 * 365.25)


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

            # Directories
            base_pair_dir = os.path.join(output_folder, f"{year1}_{year2}")
            align_dir  = os.path.join(base_pair_dir, align_code)
            track_dir  = os.path.join(align_dir,     track_code)
            filter_dir = os.path.join(track_dir,     filter_code)
            for d in (align_dir, track_dir, filter_dir):
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
            param_dict["fake_pixel_size"]                   = float(fake_pixel_size)
            param_dict["downsample_factor"]                 = int(downsample_factor)
            param_dict["undistort_image"]                   = undistort_image
            param_dict["camera_intrinsics_matrix"]          = camera_intrinsics_matrix
            param_dict["camera_distortion_coefficients"]    = camera_distortion_coefficients

            param_dict["crs"]                               = image_crs
 
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
                used_cache_alignment = False
                if use_alignment_cache and not force_recompute_alignment:
                    used_cache_alignment = load_alignment_cache(image_pair, align_dir, year1, year2)
                    if used_cache_alignment:
                        print(f"[CACHE] Alignment loaded from: {align_dir}  (pair {year1}->{year2})")

                if not used_cache_alignment:
                    print("Starting image alignment.")
                    image_pair.align_images(polygon_outside)
                    if not image_pair.valid_alignment_possible:
                        skipped.append((year1, year2, "Alignment not possible"))
                        continue
                    if use_alignment_cache:
                        save_alignment_cache(
                            image_pair, align_dir, year1, year2,
                            align_params=alignment_params.__dict__,
                            filenames={year1: filename_1, year2: filename_2},
                            dates={year1: date_1, year2: date_2},
                            save_truecolor_aligned=write_truecolor_aligned,
                        )
                        print(f"[CACHE] Alignment saved to:   {align_dir}  (pair {year1}->{year2})")

            else:
                image_pair.valid_alignment_possible = True
                image_pair.images_aligned = False


            # ==============================
            # TRACKING (optional)
            # ==============================
            used_cache_tracking = False
            if do_tracking:
                if use_tracking_cache and not force_recompute_tracking:
                    used_cache_tracking = load_tracking_cache(image_pair, track_dir, year1, year2)
                    if used_cache_tracking:
                        print(f"[CACHE] Tracking loaded from:  {track_dir}  (pair {year1}->{year2})")


                if not used_cache_tracking:
                    if used_cache_alignment or getattr(image_pair, "images_aligned", False):
                        tracked_points = image_pair.track_points(tracking_area=polygon_inside)
                        image_pair.tracking_results = tracked_points
                    else:
                        image_pair.perform_point_tracking(
                            reference_area=polygon_outside,
                            tracking_area=polygon_inside
                        )

                    if use_tracking_cache:
                        save_tracking_cache(
                            image_pair,
                            track_dir,
                            year1,
                            year2,
                            track_params=pair_tracking_config,
                            filenames={year1: filename_1, year2: filename_2},
                            dates={year1: date_1, year2: date_2},
                        )
                        print(f"[CACHE] Tracking saved to:   {track_dir}  (pair {year1}->{year2})")
            else:
                print("Tracking is disabled (alignment-only run).")


            # ==============================
            # FILTERING + PLOTS + SAVING (optional)
            # ==============================
            if do_tracking:
                if do_filtering:
                    image_pair.filter_outliers(filter_parameters=filter_params)

                    used_cache_lod = False
                    if use_lod_cache and not force_recompute_lod:
                        used_cache_lod = load_lod_cache(image_pair, track_dir, year1, year2)
                        if used_cache_lod:
                            print(f"[CACHE] LoD loaded from:       {track_dir}  (pair {year1}->{year2})")

                    if not used_cache_lod:
                        lod_points = random_points_on_polygon_by_number(
                            polygon_outside,
                            filter_params.number_of_points_for_level_of_detection
                        )
                        image_pair.calculate_lod(lod_points, filter_parameters=filter_params)
                        if use_lod_cache:
                            save_lod_cache(
                                image_pair,
                                track_dir,
                                year1,
                                year2,
                                filenames={year1: filename_1, year2: filename_2},
                                dates={year1: date_1, year2: date_2},
                            )
                            print(f"[CACHE] LoD saved to:         {track_dir}  (pair {year1}->{year2})")
                    else:
                        if not _recompute_lod_from_points(image_pair, filter_params):
                            lod_points = random_points_on_polygon_by_number(
                                polygon_outside,
                                filter_params.number_of_points_for_level_of_detection
                            )
                            image_pair.calculate_lod(lod_points, filter_parameters=filter_params)
                            if use_lod_cache:
                                save_lod_cache(
                                    image_pair,
                                    track_dir,
                                    year1,
                                    year2,
                                    filenames={year1: filename_1, year2: filename_2},
                                    dates={year1: date_1, year2: date_2},
                                )
                                print(f"[CACHE] LoD saved to:         {track_dir}  (pair {year1}->{year2})")

                    image_pair.filter_lod_points()

                if do_plotting:
                    image_pair.plot_tracking_results_with_valid_mask()

                # write a small CSV with valid fraction
                try:
                    valid_fraction = float(image_pair.tracking_results["valid"].mean())
                except Exception:
                    valid_fraction = None
                valid_csv = os.path.join(filter_dir, "valid_results_fraction.csv")
                with open(valid_csv, "w", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow(["pair", "valid_fraction"])
                    w.writerow([f"{year1}_{year2}", valid_fraction if valid_fraction is not None else "NA"])

                # final results go to the filter level
                image_pair.save_full_results(filter_dir, save_files=save_files)
            else:
                print("Skipping filtering, plotting and saving of movement products (alignment-only mode).")
                # Alignment-only outputs exist in align_dir:
                # - aligned_image_<year2>.tif
                # - alignment_control_points_<year1>_<year2>.geojson
                # - alignment_meta_<year1>_<year2>.json
                        # Mark this pair as successfully processed

            successes.append((year1, year2))

        except Exception as e:
            skipped.append((year1, year2, f"Error: {str(e)}"))

    print("\nSummary:")
    print(f"Successfully processed: {len(successes)} pairs")
    for s in successes:
        print(f"   - {s[0]} → {s[1]}")
    print(f"\nSkipped: {len(skipped)} pairs")
    for s in skipped:
        print(f"   - {s[0]} → {s[1]} | Reason: {s[2]}")


# CLI entry point
# ==============================
# MAIN
# ==============================
def main(argv=None):
    parser = argparse.ArgumentParser(description="PyImageTrack pipeline")
    parser.add_argument("--config", required=True, help="Path to TOML config file")
    args = parser.parse_args()
    run_from_config(args.config)


if __name__ == "__main__":
    main()
