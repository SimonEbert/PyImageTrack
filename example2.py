import geopandas as gpd
import rasterio
import os
import re
import itertools
import pandas as pd

from ImageTracking.ImagePair import ImagePair
from Parameters.FilterParameters import FilterParameters

# === USER CONFIGURATION ===
# input_folder = "/home/lisa/projects/pyimagetrack/Test_Data/Orthophotos_Kaiserberg_historic"
input_folder = "../Test_Data/Orthophotos_Kaiserberg_historic"
date_csv_path = os.path.join(input_folder, "image_dates.csv")
pairs_csv_path = os.path.join(input_folder, "image_pairs.csv")
#output_folder = "/home/lisa/projects/pyimagetrack/Test_results"
output_folder = "../Test_Results"
pairing_mode = "successive"  # options: "all", "successive", "custom" (i.e. from the input csv "image_pairs")

# === SAVE OPTIONS ===
# tracking_results.geojson will always be saved, since it contains the full information.
save_files = ["movement_bearing_valid_tif", "movement_bearing_outlier_filtered_tif",
              "movement_rate_valid_tif", "movement_rate_outlier_filtered_tif", "invalid_mask_tif", "lod_mask_tif",
              "statistical_parameters_txt"]

# === TRACKING PARAMETERS ===
number_of_control_points = 2000
image_bands = 0
control_tracking_area_size = 60
control_cell_size = 40
distance_of_tracked_points = 5
movement_tracking_area_size = 60 # tracking size = movement cell size plus surrounding area in all 4 directions
movement_cell_size = 20
cross_correlation_threshold_alignment = 0.85
cross_correlation_threshold_movement = 0.7


# === FILTER PARAMETERS ===
# FILTERING OPTIONS: For preventing the use of a specific filter, set the respective values to None
# Filters points whose movement bearing deviates more than the given threshold from the movement bearing of surrounding
# points
filter_parameters = FilterParameters({
    "level_of_detection_quantile": 0.9,
    "number_of_points_for_level_of_detection": 1000,
    "difference_movement_bearing_threshold": 90, # in degrees
    "difference_movement_bearing_moving_window_size": 25, # in units of the used crs
    # Filters points, where the standard deviation of movement bearing of neighbouring points exceeds the given threshold
    "standard_deviation_movement_bearing_threshold": 45, # in degrees,
    "standard_deviation_movement_bearing_moving_window_size": 15, # in units of the used crs
    # Filters points whose movement rates deviate more than the given threshold from the movement rate of surrounding points
    "difference_movement_rate_threshold": 1, # in units of the used crs / year
    "difference_movement_rate_moving_window_size": 25, # in units of the used crs
    # Filters points whose standard deviation of movement rates of neighbouring points exceeds the given threshold
    "standard_deviation_movement_rate_threshold": 1, # in units of the used crs / year
    "standard_deviation_movement_rate_moving_window_size": 15 # in units of the used crs
})




param_string = f"MTA{movement_tracking_area_size}_MC{movement_cell_size}_LoDq{filter_parameters.level_of_detection_quantile}_CC{cross_correlation_threshold_movement}" # sets name for output subfolder

# read csv for dates
date_df = pd.read_csv(date_csv_path)
date_df.columns = date_df.columns.str.strip()
if not {"year", "date"}.issubset(date_df.columns):
    raise ValueError("CSV has to have 'year' and 'date' columns.")
year_to_date = dict(zip(date_df["year"].astype(str), date_df["date"]))

# find tifs in input folder
pattern = re.compile(r"^(\d{4})_.*\.tif$")
tif_files = [f for f in os.listdir(input_folder) if pattern.match(f)]
year_to_file = {pattern.match(f).group(1): os.path.join(input_folder, f) for f in tif_files}
available_years = sorted(year_to_file.keys())

# set image pairs as per parameter
if pairing_mode == "all":
    year_pairs = [(y1, y2) for y1, y2 in itertools.combinations(available_years, 2)]
elif pairing_mode == "successive":
    year_pairs = [(available_years[i], available_years[i + 1]) for i in range(len(available_years) - 1)]
elif pairing_mode == "custom":
    custom_df = pd.read_csv(pairs_csv_path, sep=";")
    if not {"year_earlier", "year_later"}.issubset(custom_df.columns):
        raise ValueError("Custom pairs csv requires 'year_earlier' and 'year_later'.")
    year_pairs = [(str(row["year_earlier"]), str(row["year_later"])) for _, row in custom_df.iterrows()]
else:
    raise ValueError("Invalid pairing_mode. Use 'all', 'successive' or 'custom'.")

print(f"Image pairs to process ({pairing_mode}): {len(year_pairs)}")

# load polygons
polygon_outside_RG = gpd.read_file(os.path.join(input_folder, "Area_outside_rock_glacier.shp")).to_crs(crs=31254)
rock_glacier_polygon = gpd.read_file(os.path.join(input_folder, "Area_inside_rock_glacier.shp")).to_crs(crs=31254)

# === Processing ===
successes = []
skipped = []

for year1, year2 in year_pairs:
    if year1 not in year_to_date or year2 not in year_to_date:
        skipped.append((year1, year2, "Date missing in CSV"))
        continue
    if year1 not in year_to_file or year2 not in year_to_file:
        skipped.append((year1, year2, "Tif input image missing"))
        continue

    filename_1 = year_to_file[year1]
    filename_2 = year_to_file[year2]
    date_1 = year_to_date[year1]
    date_2 = year_to_date[year2]

    print(f"\nProcessed image pair: {year1} ({date_1}) → {year2} ({date_2})")
    print(f"   File 1: {filename_1}")
    print(f"   File 2: {filename_2}")

    try:
        image_pair = ImagePair(parameter_dict={
            "image_alignment_number_of_control_points": number_of_control_points,
            "used_image_bands": image_bands,
            "image_alignment_control_tracking_area_size": control_tracking_area_size,
            "image_alignment_control_cell_size": control_cell_size,
            "distance_of_tracked_points": distance_of_tracked_points,
            "movement_tracking_area_size": movement_tracking_area_size,
            "movement_cell_size": movement_cell_size,
            "cross_correlation_threshold_alignment": cross_correlation_threshold_alignment,
            "cross_correlation_threshold_movement": cross_correlation_threshold_movement})

        image_pair.load_images_from_file(
            filename_1=filename_1,
            observation_date_1=date_1,
            filename_2=filename_2,
            observation_date_2=date_2,
            selected_channels=image_bands
        )

        image_pair.perform_point_tracking(reference_area=polygon_outside_RG, tracking_area=rock_glacier_polygon)

        image_pair.full_filter(reference_area=polygon_outside_RG, filter_parameters=filter_parameters)
        image_pair.plot_tracking_results_with_valid_mask()

        result_dir = os.path.join(output_folder, f"{year1}_{year2}", param_string)
        os.makedirs(result_dir, exist_ok=True)
        image_pair.save_full_results(result_dir, save_files=save_files)

        successes.append((year1, year2))
    except Exception as e:
        skipped.append((year1, year2, f"Error: {str(e)}"))

# === ZUSAMMENFASSUNG ===
print("\nSummary:")
print(f"Successfully processed: {len(successes)} pairs")
for s in successes:
    print(f"   - {s[0]} → {s[1]}")

print(f"\nSkipped: {len(skipped)} pairs")
for s in skipped:
    print(f"   - {s[0]} → {s[1]} | Reason: {s[2]}")
