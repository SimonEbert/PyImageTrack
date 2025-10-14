#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Orchestrator: align, track, filter, plot, save with caching.
"""

import os
import csv
from datetime import datetime
import geopandas as gpd
import sys
sys.path.append('..')

from PyImageTrack.ImageTracking.ImagePair import ImagePair
from PyImageTrack.Parameters.FilterParameters import FilterParameters
from PyImageTrack.Parameters.AlignmentParameters import AlignmentParameters
from PyImageTrack.Parameters.TrackingParameters import TrackingParameters

from PyImageTrack.Utils import (
    collect_pairs, ensure_dir, abbr_alignment, 
    abbr_tracking, abbr_filter, parse_date,
)

from PyImageTrack.Cache import (
    load_alignment_cache, save_alignment_cache,
    load_tracking_cache, save_tracking_cache,
    tracking_cache_paths,
)
import json


# ==============================
# USER CONFIGURATION (paths, data names, CRS)
# ==============================
input_folder = "../Lisa_Kaunertal/Testdaten_Alignment"
date_csv_path = os.path.join(input_folder, "image_dates.csv") # can be  set to = None if the day is reflected in the filename
#date_csv_path = None
pairs_csv_path = os.path.join(input_folder, "image_pairs.csv") # can be  set to = None if all or successive pairing mode is selected below
#pairs_csv_path = None

poly_outside_filename = "stable_area_drone.shp"
poly_inside_filename  = "moving_area_drone.shp"
poly_CRS = 32632

output_folder = "../Lisa_Kaunertal/test_results"
pairing_mode = "custom"            # options: "all", "successive", "custom" (=from image_pairs.csv)

use_fake_georeferencing = False        # set True only when processing non-ortho JPGs
fake_pixel_size = 1.0                  # 1 px = 1 unit
fake_crs_epsg = poly_CRS               # use your polygon CRS for fake georef

do_alignment = True
do_plotting = True
do_image_enhancement = False           # optional image enhancement via CLAHE

use_alignment_cache = False
use_tracking_cache  = False
force_recompute_alignment = False
force_recompute_tracking  = False

# adaptive tracking window options
use_adaptive_tracking_window = True    # If True, the "search_extent_px" in the tracking parameters relates to the expected movement PER YEAR
#pixels_per_metre = 1.0                 # px per metre (depends on raster resolution)
#maximal_assumed_movement_rate = 3.5    # m/year (upper bound used for search window scaling)

# ==============================
# PARAMETERS (alignment, tracking, filter)
# ==============================
alignment_params = AlignmentParameters({
    "number_of_control_points": 2000,       
    # search extent tuple: (right, left, down, up) in pixels around the control cell
    "control_search_extent_px": (5, 5, 5, 5), # px
    "control_cell_size": 5, # px
    "cross_correlation_threshold_alignment": 0.8,                   
    "maximal_alignment_movement": None, # px, can be set to = None
})

tracking_params = TrackingParameters({
    "image_bands": 0,
    "distance_of_tracked_points_px": 50, # px
    "movement_cell_size": 100, # px
    "cross_correlation_threshold_movement": 0.5,
    # search extent tuple: (right, left, down, up) in pixels around the movement cell
    # usually this refers to the offset in px between the images, 
    # but if the adaptive mode is used, this means the expected offset in px per year
    "search_extent_px": (20, 5, 5, 20), # px OR px / year
})

filter_params = FilterParameters({
    "level_of_detection_quantile": 0.75,
    "number_of_points_for_level_of_detection": 1000,                
    "difference_movement_bearing_threshold": 360,                    # degrees
    "difference_movement_bearing_moving_window_size": 50,           # CRS units
    "standard_deviation_movement_bearing_threshold": 360,            # degrees
    "standard_deviation_movement_bearing_moving_window_size": 50,   # CRS units
    "difference_movement_rate_threshold": 10,                      # CRS units / year
    "difference_movement_rate_moving_window_size": 10,              # CRS units
    "standard_deviation_movement_rate_threshold": 10,              # CRS units / year
    "standard_deviation_movement_rate_moving_window_size": 50,      # CRS units
})

# ==============================
# SAVE OPTIONS (final outputs)
# ==============================
save_files = [
    # "first_image_matrix", 
    # "second_image_matrix",
    "movement_bearing_valid_tif",
    "movement_rate_valid_tif",
    "movement_bearing_outlier_filtered_tif", 
    "movement_rate_outlier_filtered_tif",
    "movement_bearing_LoD_filtered_tif",
    "movement_rate_LoD_filtered_tif",
    "movement_bearing_all_tif", 
    "movement_rate_all_tif",
    "mask_invalid_tif",
    "mask_LoD_tif",
    "mask_outlier_md_tif",
    "mask_outlier_msd_tif",
    "mask_outlier_bd_tif",
    "mask_outlier_bsd_tif",
    # "LoD_points_geojson",
    # "control_points_geojson",
    "statistical_parameters_txt",

]

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


# ==============================
# MAIN
# ==============================
def main():
    # Allow JPG/JPEG only if explicitly opted into fake georeferencing
    extensions = (".tif", ".tiff") if not use_fake_georeferencing else (".tif", ".tiff", ".jpg", ".jpeg")

    year_pairs, id_to_file, id_to_date, id_hastime_from_filename = collect_pairs(
        input_folder=input_folder,
        date_csv_path=date_csv_path,
        pairs_csv_path=pairs_csv_path,
        pairing_mode=pairing_mode,
        extensions=extensions
    )

    print(f"Image pairs to process ({pairing_mode}): {len(year_pairs)}")

    polygon_outside = gpd.read_file(os.path.join(input_folder, poly_outside_filename)).to_crs(epsg=poly_CRS)
    polygon_inside  = gpd.read_file(os.path.join(input_folder, poly_inside_filename)).to_crs(epsg=poly_CRS)

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

        def _fmt_label(id_key, date_str):
            return parse_date(date_str).strftime("%Y-%m-%d %H:00") if id_hastime_from_filename.get(id_key, False) \
                else parse_date(date_str).strftime("%Y-%m-%d")

        label_1 = _fmt_label(year1, date_1)
        label_2 = _fmt_label(year2, date_2)
        print(f"\nProcessed image pair: {year1} ({label_1}) → {year2} ({label_2})")

        print(f"   File 1: {filename_1}")
        print(f"   File 2: {filename_2}")

        try:
            # compute years_between (hour-precise)
            dt1 = parse_date(date_1)
            dt2 = parse_date(date_2)
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
            ensure_dir(align_dir); ensure_dir(track_dir); ensure_dir(filter_dir)

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
            param_dict["use_fake_georeferencing"]           = bool(use_fake_georeferencing)
            param_dict["fake_crs_epsg"]                     = int(fake_crs_epsg) if fake_crs_epsg is not None else None
            param_dict["fake_pixel_size"]                   = float(fake_pixel_size)

 
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
                        )
                        print(f"[CACHE] Alignment saved to:   {align_dir}  (pair {year1}->{year2})")
            else:
                image_pair.valid_alignment_possible = True
                image_pair.images_aligned = False


            # tracking with cache
            used_cache_tracking = False
            if use_tracking_cache and not force_recompute_tracking:
                used_cache_tracking = load_tracking_cache(image_pair, track_dir, year1, year2)
                if used_cache_tracking:
                    print(f"[CACHE] Tracking loaded from:  {track_dir}  (pair {year1}->{year2})")

                    # check if effective search areas are same (adaptive T/F)
                    _, meta_json = tracking_cache_paths(track_dir, year1, year2)
                    try:
                        with open(meta_json, "r", encoding="utf-8") as f:
                            meta = json.load(f)
                        cached_eff = tuple(int(x) for x in meta.get("tracking_params", {}).get("search_extent_px_effective", []))
                        current_eff = tuple(int(x) for x in adaptive_extents)
                        if cached_eff != current_eff:
                            print(f"[CACHE] Effective search extents mismatch: cache {cached_eff} vs current {current_eff} -> recompute.")
                            used_cache_tracking = False
                    except Exception as e:
                        print(f"[CACHE] Could not read/parse tracking meta ({e}). Will recompute.")
                        used_cache_tracking = False
                    
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

                    print(f"[CACHE] Tracking saved to:  {track_dir}  (pair {year1}->{year2})")

            # filtering and optional plot
            image_pair.full_filter(reference_area=polygon_outside, filter_parameters=filter_params)
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

if __name__ == "__main__":
    main()
