import datetime
import os

import geopandas as gpd
import numpy as np
import pandas as pd

# from ImageTracking.ImagePair import ImagePair
from ImageTracking.ImageBatch import ImageBatch
from src import FilterParameters

# Set parameters for this project
image_enhancement = True
maximal_assumed_movement_rate = 3.5
pixels_per_metre = 2
epsg_code = 32719
logger_filename = "./results.log"



# Set parameters
alignment_via_lsm = True
number_of_control_points = 2000
image_bands = 0
control_tracking_area_size = 60
control_cell_size = 40
tracking_method = "lsm"
distance_of_tracked_points = 5
movement_cell_size = 50
level_of_detection_quantile = 0.5
cross_correlation_threshold_alignment = 0.75
cross_correlation_threshold_movement = 0.75
maximal_alignment_movement = None # In Pixels !!


# === SAVE OPTIONS ===
# tracking_results.geojson will always be saved, since it contains the full information.
save_files = ["movement_bearing_valid_tif",
              "movement_rate_valid_tif", "invalid_mask_tif", "lod_mask_tif",
              "statistical_parameters_txt", "LoD_points_geojson", "control_points_geojson",
              "first_image_matrix", "second_image_matrix"]

# === FILTER PARAMETERS ===
# FILTERING OPTIONS: For preventing the use of a specific filter, set the respective values to a very high value
filter_parameters = FilterParameters({
    "level_of_detection_quantile": level_of_detection_quantile,
    "number_of_points_for_level_of_detection": 1000,
    #  Filters points, whose movement bearings deviate more than the given threshold from the movementt rate of surrounding points
        "difference_movement_bearing_threshold": 360, # in degrees
        "difference_movement_bearing_moving_window_size":  17.5, # in units of the used crs
    #     # Filters points, where the standard deviation of movement bearing of neighbouring points exceeds the given threshold
        "standard_deviation_movement_bearing_threshold": 360, # in degrees,
        "standard_deviation_movement_bearing_moving_window_size":  50, # in units of the used crs
    #     # Filters points whose movement rates deviate more than the given threshold from the movement rate of surrounding points
        "difference_movement_rate_threshold": 75, # in units of the used crs / year
        "difference_movement_rate_moving_window_size":  30, # in units of the used crs
    #     # Filters points whose standard deviation of movement rates of neighbouring points exceeds the given threshold
        "standard_deviation_movement_rate_threshold": 200, # in units of the used crs / year
        "standard_deviation_movement_rate_moving_window_size": 50 # in units of the used crs
})



rock_glacier_inventory_shapefile = gpd.read_file("../Analysis_rock_glaciers_Argentina/Rock_glacier_inventory/IANIGLApolygons_OBJECTID.shp")

rock_glacier_inventory_shapefile = rock_glacier_inventory_shapefile.to_crs(epsg=epsg_code)

residual_correlation_dataframe = pd.DataFrame()
level_of_detection_dataframe = pd.DataFrame()

for polygon_id in np.arange(1,82):

    # polygon 60-78 are the glaciers, 28, 29, 31-34, 36, 53-54, 58, 59 have no data from 2019
    # if 60 <= polygon_id < 79 or 28 <= polygon_id < 30 or 31 <= polygon_id < 35 or polygon_id == 36 or 53 <= polygon_id < 55 or 58 <= polygon_id < 60:
    #     continue
    if 60 <= polygon_id <79:
        continue

    print("Starting to assess polygon ", polygon_id)

    # get available orthophoto names
    list_of_observations = os.listdir("../Analysis_rock_glaciers_Argentina/PanOrtho_All/poly" + str(polygon_id))
    # remove "extra" entry which contains unused orthophotos
    list_of_observations.remove("extra")
    list_of_observations.sort()
    # list_of_observations = [list_of_observations[0], list_of_observations[1]]

    # get observation dates from filenames
    observation_dates_filename = [observation[:6] for observation in list_of_observations]

    # add 20 at the beginning of each datestring for easy conversion to datetime format (in YYYYMMDD format)
    observation_dates = ["20" + observation_date for observation_date in observation_dates_filename]

    observation_dates = [datetime.datetime.strptime(observation_date, "%Y%m%d")
                         for observation_date in observation_dates]

    Polygon_Image_Batch = ImageBatch(parameter_dict={
            "image_alignment_number_of_control_points": number_of_control_points,
            "used_image_bands": image_bands,
            "image_alignment_control_tracking_area_size": control_tracking_area_size,
            "image_alignment_control_cell_size": control_cell_size,
            "distance_of_tracked_points": distance_of_tracked_points,
            "movement_cell_size": movement_cell_size,
            "cross_correlation_threshold_alignment": cross_correlation_threshold_alignment,
            "cross_correlation_threshold_movement": cross_correlation_threshold_movement,
            "maximal_alignment_movement": maximal_alignment_movement
    })


    filename_list = ["../Analysis_rock_glaciers_Argentina/PanOrtho_All/poly" + str(polygon_id) + "/" + obs for obs in list_of_observations]
    Polygon_Image_Batch.load_images_from_file_list(filename_list, list_of_observation_dates=observation_dates,
                                                   maximal_assumed_movement_rate=maximal_assumed_movement_rate,
                                                   pixels_per_metre=pixels_per_metre)
    extent_area = Polygon_Image_Batch.data_bounds
    extent_area = Polygon_Image_Batch.list_of_image_pairs[0].image_bounds
    reference_area = gpd.GeoDataFrame(
                 geometry=extent_area.difference(
                     rock_glacier_inventory_shapefile.union_all().buffer(200), align=False))

    single_rock_glacier = rock_glacier_inventory_shapefile.loc[
                rock_glacier_inventory_shapefile["OBJECTID"] == polygon_id,]
    # set index of single rock glacier file to 0
    single_rock_glacier.set_index(np.arange(1), inplace=True)

    # buffer around rock glacier since inventory quality is not perfect
    single_rock_glacier = gpd.GeoDataFrame(geometry=single_rock_glacier.geometry.buffer(50),
                                                   crs=rock_glacier_inventory_shapefile.crs)
    # reference_area = gpd.GeoDataFrame(
    #     geometry=reference_area.intersection(single_rock_glacier.buffer(100)))

    if image_enhancement:
        Polygon_Image_Batch.equalize_adapthist_images()

    Polygon_Image_Batch.perform_point_tracking(reference_area=reference_area, tracking_area=single_rock_glacier)
    Polygon_Image_Batch.calculate_and_filter_lod(filter_parameters=filter_parameters, reference_area=reference_area)
    Polygon_Image_Batch.filter_outliers(filter_parameters=filter_parameters)

    Polygon_Image_Batch.save_full_results("../Analysis_rock_glaciers_Argentina/Tracking_results_alignment_check/poly" + str(polygon_id), save_files=save_files)

    for image_pair in Polygon_Image_Batch.list_of_image_pairs:
        if not image_pair.valid_alignment_possible:
            continue
        residual_correlation = np.corrcoef(image_pair.tracked_control_points["residuals_row"],
                                            image_pair.tracked_control_points["residuals_column"])[0,1]
        residual_correlation_dataframe.loc[
            "poly" + str(polygon_id), str(image_pair.image2_observation_date.year) + "_" +
            str(image_pair.image2_observation_date.year)] = np.abs(residual_correlation)
        level_of_detection_dataframe.loc[
            "poly" + str(polygon_id), str(image_pair.image2_observation_date.year) + "_" +
            str(image_pair.image2_observation_date.year)] = image_pair.level_of_detection

    residual_correlation_dataframe.to_csv(
            "../Analysis_rock_glaciers_Argentina/Paper_residual_correlation_dataframe_new_PyImageTrack.csv", index=True)
    level_of_detection_dataframe.to_csv(
            "../Analysis_rock_glaciers_Argentina/Paper_level_of_detection_dataframe_new_PyImageTrack.csv", index=True)


    # for filename_1 in list_of_observations[:-1]:
    #
    #     print("Starting to track first image " + str(observation_dates[list_of_observations.index(filename_1)]))
    #     logging.info("Starting to track first image " + str(observation_dates[list_of_observations.index(filename_1)]))
    #
    #
    #     output_folder_path = "../Analysis_rock_glaciers_Argentina/Tracking_results_long_period_1/poly" + str(polygon_id)
    #
    #     # extract single rock glacier polygon
    #     single_rock_glacier = rock_glacier_inventory_shapefile.loc[
    #         rock_glacier_inventory_shapefile["OBJECTID"] == polygon_id,]
    #     # set index of single rock glacier file to 0
    #     single_rock_glacier.set_index(np.arange(1), inplace=True)
    #
    #     # buffer around rock glacier since inventory quality is not perfect
    #     single_rock_glacier = gpd.GeoDataFrame(geometry=single_rock_glacier.geometry.buffer(50),
    #                                            crs=rock_glacier_inventory_shapefile.crs)
    #
    #     # CALCULATE movement_tracking_area_size
    #     observation_time_difference = observation_dates[list_of_observations.index(filename_1) + 1] - \
    #                                   observation_dates[list_of_observations.index(filename_1)]
    #
    #     years_between_observations = observation_time_difference.days / 365.25
    #
    #     movement_tracking_area_size = np.ceil(
    #         2 * maximal_assumed_movement_rate * years_between_observations * pixels_per_metre
    #         + movement_cell_size)
    #
    #     filename_2 = list_of_observations[list_of_observations.index(filename_1) + 1]
    #
    #     # instantiate ImagePair
    #     Image_Tracking_Pair = ImagePair(
    #     parameter_dict={"image_alignment_via_lsm": alignment_via_lsm,
    #                     "image_alignment_number_of_control_points": number_of_control_points,
    #                     "used_image_bands": image_bands,
    #                     "image_alignment_control_tracking_area_size": control_tracking_area_size,
    #                     "image_alignment_control_cell_size": control_cell_size,
    #                     "distance_of_tracked_points": distance_of_tracked_points,
    #                     "movement_tracking_area_size": movement_tracking_area_size,
    #                     "movement_cell_size": movement_cell_size,
    #                     "level_of_detection_quantile": level_of_detection_quantile,
    #                     "cross_correlation_threshold_movement": cross_correlation_threshold_movement,
    #                     "cross_correlation_threshold_alignment": cross_correlation_threshold_alignment
    #                     })
    #
    #     Image_Tracking_Pair.load_images_from_file(
    #         filename_1="../Analysis_rock_glaciers_Argentina/PanOrtho_All/poly" + str(polygon_id) + "/" + filename_1,
    #         observation_date_1=observation_dates[list_of_observations.index(filename_1)].strftime("%d-%m-%Y"),
    #         filename_2="../Analysis_rock_glaciers_Argentina/PanOrtho_All/poly" + str(polygon_id) + "/" + filename_2,
    #         observation_date_2=observation_dates[list_of_observations.index(filename_1) + 1].strftime("%d-%m-%Y"),
    #         selected_channels=0)
    #
    #
    #     # CALCULATE reference area
    #     file1 = rasterio.open("../Analysis_rock_glaciers_Argentina/PanOrtho_All/poly" + str(polygon_id) + "/" + filename_1)
    #     file2 = rasterio.open("../Analysis_rock_glaciers_Argentina/PanOrtho_All/poly" + str(polygon_id) + "/" + filename_2)
    #     bbox1 = file1.bounds
    #     bbox2 = file2.bounds
    #     minbbox = BoundingBox(left=max(bbox1[0], bbox2[0]),
    #                           bottom=max(bbox1[1], bbox2[1]),
    #                           right=min(bbox1[2], bbox2[2]),
    #                           top=min(bbox1[3], bbox2[3])
    #                           )
    #
    #     minimal_bounding_box = shapely.Polygon((
    #         (minbbox[0], minbbox[1]),
    #         (minbbox[0], minbbox[3]),
    #         (minbbox[2], minbbox[3]),
    #         (minbbox[2], minbbox[1])
    #     ))
    #
    #     extent_area = gpd.GeoDataFrame(geometry=[minimal_bounding_box], crs=single_rock_glacier.crs)
    #
    #     # get the area in the extent that is not covered by anything in the rock glacier inventory
    #     reference_area = gpd.GeoDataFrame(
    #         geometry=extent_area.difference(  # single_rock_glacier.buffer(200).difference(
    #             rock_glacier_inventory_shapefile.union_all().buffer(200), align=False))
    #
    #     # clip image files to 0, ..., 255
    #     Image_Tracking_Pair.image1_matrix[Image_Tracking_Pair.image1_matrix > 255] = 0
    #     Image_Tracking_Pair.image2_matrix[Image_Tracking_Pair.image2_matrix > 255] = 0
    #
    #     if image_enhancement:
    #         Image_Tracking_Pair.equalize_adapthist_images()
    #
    #     Image_Tracking_Pair.perform_point_tracking(reference_area=reference_area, tracking_area=single_rock_glacier)
    #     points_for_lod_calculation = random_points_on_polygon_by_number(reference_area, 1000)
    #     Image_Tracking_Pair.calculate_lod(points_for_lod_calculation, filter_parameters)
    #     Image_Tracking_Pair.filter_lod_points()
    #
    #     Image_Tracking_Pair.plot_tracking_results()
    #
    #     Image_Tracking_Pair.save_full_results(output_folder_path, save_files=save_files)

#     tracking_results = gpd.read_file(output_folder_path + "/tracking_results_2019_2025.geojson")
#     if polygon_id == 1:
#         full_tracking_results = tracking_results.copy()
#     else:
#         full_tracking_results = pd.concat([full_tracking_results, tracking_results])
#
# full_tracking_results_intersected = gpd.overlay(full_tracking_results, rock_glacier_inventory_shapefile, how='intersection')
#
# full_tracking_results.to_file("../Analysis_rock_glaciers_Argentina/Tracking_results_long_period/full_tracking_results.geojson", driver="GeoJSON")
# full_tracking_results_intersected.to_file("../Analysis_rock_glaciers_Argentina/Tracking_results_long_period/full_tracking_results_intersected_with"
#                                           "_rock_glacier_inventory.geojson", driver="GeoJSON")