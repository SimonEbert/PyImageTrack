import geopandas as gpd
import rasterio

from ImageTracking.ImagePair import ImagePair
from Plots.MakePlots import plot_raster_and_geometry
from Parameters.FilterParameters import FilterParameters

# Set parameters
number_of_control_points = 200
image_bands = 0
control_tracking_area_size = 60
control_cell_size = 40
distance_of_tracked_points = 5
movement_tracking_area_size = 60
movement_cell_size = 20
cross_correlation_threshold = 0.75


# === SAVE OPTIONS ===
# tracking_results.geojson will always be saved, since it contains the full information.
save_files = ["movement_bearing_valid_tif", "movement_bearing_outlier_filtered_tif",
              "movement_rate_valid_tif", "movement_rate_outlier_filtered_tif", "invalid_mask_tif", "lod_mask_tif",
              "statistical_parameters_txt"]


# === FILTER PARAMETERS ===
# FILTERING OPTIONS: For preventing the use of a specific filter, set the respective values to None
# Filters points whose movement bearing deviates more than the given threshold from the movement bearing of surrounding
# points
filter_parameters = FilterParameters({
    "level_of_detection_quantile": 0.9,
    "number_of_points_for_level_of_detection": 100,
    # Filters points, whose movement bearings deviate more than the given threshold from the movementt rate of surrounding points
    "difference_movement_bearing_threshold": 45, # in degrees
    "difference_movement_bearing_moving_window_size": 50, # in units of the used crs
    # Filters points, where the standard deviation of movement bearing of neighbouring points exceeds the given threshold
    "standard_deviation_movement_bearing_threshold": 30, # in degrees,
    "standard_deviation_movement_bearing_moving_window_size": 30, # in units of the used crs
    # Filters points whose movement rates deviate more than the given threshold from the movement rate of surrounding points
    "difference_movement_rate_threshold": 1, # in units of the used crs / year
    "difference_movement_rate_moving_window_size": 50, # in units of the used crs
    # Filters points whose standard deviation of movement rates of neighbouring points exceeds the given threshold
    "standard_deviation_movement_rate_threshold": 1, # in units of the used crs / year
    "standard_deviation_movement_rate_moving_window_size": 30 # in units of the used crs
})



# Set filter parameters

Kaiserberg_pair_19_21 = ImagePair(
    parameter_dict={"image_alignment_number_of_control_points": number_of_control_points,
                    "used_image_bands": image_bands,
                    "image_alignment_control_tracking_area_size": control_tracking_area_size,
                    "image_alignment_control_cell_size": control_cell_size,
                    "distance_of_tracked_points": distance_of_tracked_points,
                    "movement_tracking_area_size": movement_tracking_area_size,
                    "movement_cell_size": movement_cell_size,
                    "cross_correlation_threshold": cross_correlation_threshold})

Kaiserberg_pair_19_21.load_images_from_file(filename_1="../Test_Data/Orthophotos_Kaiserberg_historic/1953_ortho_1m_RG_rend_bw.tif",
                                            observation_date_1="02-09-1953",
                                            filename_2="../Test_Data/Orthophotos_Kaiserberg_historic/1970_ortho_1m_RG_rend_bw.tif",
                                            observation_date_2="29-09-1970",
                                            selected_channels=0)




polygon_outside_RG = gpd.read_file("../Test_Data/Orthophotos_Kaiserberg_historic/Area_outside_rock_glacier_adjusted.shp")
polygon_outside_RG = polygon_outside_RG.to_crs(crs=31254)

rock_glacier_polygon = gpd.read_file("../Test_Data/Orthophotos_Kaiserberg_historic/Area_inside_rock_glacier.shp")
rock_glacier_polygon = rock_glacier_polygon.to_crs(crs=31254)

Kaiserberg_pair_19_21.perform_point_tracking(reference_area=polygon_outside_RG, tracking_area=rock_glacier_polygon)

Kaiserberg_pair_19_21.full_filter(reference_area=polygon_outside_RG, filter_parameters=filter_parameters)

Kaiserberg_pair_19_21.plot_tracking_results_with_valid_mask()

Kaiserberg_pair_19_21.save_full_results("../Test_results/full_results", save_files=save_files)
