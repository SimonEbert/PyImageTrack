import geopandas as gpd

from src import FilterParameters
from src import ImagePair

# Set parameters
number_of_control_points = 200
image_bands = 0
control_tracking_area_size = 70 # script still has a bias if this number is 0 or 3 mod 4
control_cell_size = 50
distance_of_tracked_points_px = 30
search_extent_px = (75,75,75,75)
movement_cell_size = 100
cross_correlation_threshold_alignment = 0.75
cross_correlation_threshold_movement = 0


# === SAVE OPTIONS ===
# tracking_results.geojson will always be saved, since it contains the full information.
save_files = ["first_image_matrix", "second_image_matrix",
              "movement_rate_valid_tif", "statistical_parameters_txt"]


# === FILTER PARAMETERS ===
# FILTERING OPTIONS: For preventing the use of a specific filter, set the respective values to None
# Filters points whose movement bearing deviates more than the given threshold from the movement bearing of surrounding
# points
filter_parameters = FilterParameters({
    "level_of_detection_quantile": 0.9,
    "number_of_points_for_level_of_detection": 1000,
    # Filters points, whose movement bearings deviate more than the given threshold from the movement rate of surrounding points
    "difference_movement_bearing_threshold": 60, # in degrees
    "difference_movement_bearing_moving_window_size":  17.5, # in units of the used crs
    # Filters points, where the standard deviation of movement bearing of neighbouring points exceeds the given threshold
    "standard_deviation_movement_bearing_threshold": 90, # in degrees,
    "standard_deviation_movement_bearing_moving_window_size":  50, # in units of the used crs
    # Filters points whose movement rates deviate more than the given threshold from the movement rate of surrounding points
    "difference_movement_rate_threshold": 0.2, # in units of the used crs / year
    "difference_movement_rate_moving_window_size":  30, # in units of the used crs
    # Filters points whose standard deviation of movement rates of neighbouring points exceeds the given threshold
    "standard_deviation_movement_rate_threshold": 1.2, # in units of the used crs / year
    "standard_deviation_movement_rate_moving_window_size": 50 # in units of the used crs
})


Ritigraben_pair_September = ImagePair(
    parameter_dict={"image_alignment_number_of_control_points": number_of_control_points,
                    "used_image_bands": image_bands,
                    "image_alignment_control_tracking_area_size": control_tracking_area_size,
                    "image_alignment_control_cell_size": control_cell_size,
                    "distance_of_tracked_points_px": distance_of_tracked_points_px,
                    "search_extent_px": search_extent_px,
                    "movement_cell_size": movement_cell_size,
                    "cross_correlation_threshold_alignment": cross_correlation_threshold_alignment,
                    "cross_correlation_threshold_movement": cross_correlation_threshold_movement})

Ritigraben_pair_September.load_images_from_file(filename_1="/home/simon/Documents/Promotion/Ritigraben/Hillshades_September/Hillshade-2025-08-27_12-34-10_5cm.tif",
                                            observation_date_1="01-07-2022",
                                            filename_2="/home/simon/Documents/Promotion/Ritigraben/Hillshades_September/Hillshade-2025-08-31_15-13-20_5cm.tif",
                                            observation_date_2="01-07-2023",
                                            selected_channels=0, NA_value=-9999)

# polygon_outside_RG = gpd.read_file("../Lisa_Kaunertal/Testdaten_Alignment/Area_outside_rock_glacier.shp")
# polygon_outside_RG = polygon_outside_RG.to_crs(crs=32632)

# rock_glacier_polygon = gpd.read_file("../Lisa_Kaunertal/Testdaten_Alignment/Area_inside_rock_glacier.shp")
# rock_glacier_polygon = rock_glacier_polygon.to_crs(crs=32632)

polygon_to_be_tracked = gpd.read_file("/home/simon/Documents/Promotion/Ritigraben/Hillshades_September/Hillshade_extent.shp")



# Kaiserberg_pair_19_21.save_full_results("../Lisa_Kaunertal/full_results", save_files=["first_image_matrix", "second_image_matrix"])



# Kaiserberg_pair_19_21.align_images(polygon_outside_RG)

# Ritigraben_pair_September.tracking_results = Ritigraben_pair_September.track_points(polygon_to_be_tracked)



Ritigraben_pair_September.tracking_results = gpd.read_file("../Lisa_Kaunertal/full_results/tracking_results_2022_2023.geojson")

# Kaiserberg_pair_19_21.full_filter(reference_area=polygon_outside_RG, filter_parameters=filter_parameters)
#
# Kaiserberg_pair_19_21.filter_outliers(filter_parameters=filter_parameters)



#
# Kaiserberg_pair_19_21.plot_tracking_results_with_valid_mask()

Ritigraben_pair_September.save_full_results("../Lisa_Kaunertal/full_results", save_files=save_files)
