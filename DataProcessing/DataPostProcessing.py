import geopandas as gpd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PixelMatching import track_movement
from CreateGeometries.HandleGeometries import georeference_tracked_points
from CreateGeometries.HandleGeometries import random_points_on_polygon_by_number
from dataloader.TrackingParameters import TrackingParameters


def calculate_LoD_points(image1_matrix: np.ndarray, image2_matrix: np.ndarray, image_transform, reference_area: gpd.GeoDataFrame,
                  number_of_reference_points, tracking_parameters: TrackingParameters, crs, years_between_observations)\
        -> float:

    points = random_points_on_polygon_by_number(reference_area, number_of_points=number_of_reference_points)
    tracked_points = track_movement(image1_matrix=image1_matrix, image2_matrix=image2_matrix,
                                              image_transform=image_transform, tracking_area=reference_area,
                   tracking_parameters=tracking_parameters, points_to_be_tracked=points)
    tracked_points = georeference_tracked_points(tracked_points, image_transform, crs=crs,
                                                 years_between_observations=years_between_observations)
    return tracked_points





import os
import matplotlib.pyplot as plt
import matplotlib

# list_of_available_ids = os.listdir("../../Output_results/2025_03_20_large_rock_glaciers")
#
#
# # full_tracking_results = gpd.GeoDataFrame()
# #
# # for rock_glacier_id in list_of_available_ids:
# #     print(rock_glacier_id)
# #     tracking_results = gpd.read_file(
# #         "../../Output_results/2025_03_20_large_rock_glaciers/" + str(rock_glacier_id) + "/tracking_results.geojson")
# #     plt.hist(tracking_results["movement_distance_per_year"])
# #     plt.xlabel("Movement distance per year")
# #     plt.ylabel("Frequency")
# #     plt.savefig("../../Output_results/2025_03_20_large_rock_glaciers/" + str(
# #         rock_glacier_id) + "/histogram_movement_distance_per_year.png")
# #
# #     full_tracking_results = pd.concat([full_tracking_results, tracking_results], ignore_index=True)
# #
# # full_tracking_results = gpd.GeoDataFrame(full_tracking_results)
# # full_tracking_results.to_file("../../Output_results/2025_03_20_large_rock_glaciers/full_tracking_results.geojson", driver="GeoJSON")
#
# full_tracking_results = gpd.read_file("../../Output_results/2025_03_20_large_rock_glaciers/full_tracking_results.geojson")
#
# plt.hist(full_tracking_results["movement_distance_per_year"], bins=500)
# plt.yscale("log")
# plt.xlabel("Movement distance per year")
# plt.ylabel("Frequency")
# plt.show()

#
# list_of_available_times = os.listdir("../../Output_results/2025_03_21_Kaiserberg_long_term")
# print(list_of_available_times)
#
# full_time_period_tracking_results = gpd.GeoDataFrame()
#
# print(len(full_time_period_tracking_results))
#
# for time_layer in list_of_available_times:
#     print(time_layer)
#     single_time_tracking_results = gpd.read_file("../../Output_results/2025_03_21_Kaiserberg_long_term/" + time_layer +
#                                                  "/tracking_results.geojson")
#
#     if (len(full_time_period_tracking_results) == 0):
#         full_time_period_tracking_results = single_time_tracking_results
#         full_time_period_tracking_results = full_time_period_tracking_results[["movement_distance_per_year", "geometry"]]
#         full_time_period_tracking_results.columns = ["movement_distance_per_year_" + time_layer, "geometry"]
#     else:
#         full_time_period_tracking_results[("movement_distance_per_year_" + time_layer)] = (
#             single_time_tracking_results)["movement_distance_per_year"]
# print(full_time_period_tracking_results.columns)
#
# full_time_period_tracking_results["yearly_velocity_difference_1971_2023"] = (
#                                                                                 full_time_period_tracking_results)[
#                                                                                 "movement_distance_per_year_Image_Aktuell_RGB"] - \
#                                                                             full_time_period_tracking_results[
#                                                                                 "movement_distance_per_year_Image_1970_1982"]
#
# full_time_period_tracking_results.plot(column="yearly_velocity_difference_1971_2023", legend=True)
# plt.show()
#
# full_time_period_tracking_results.to_file("../../Output_results/2025_03_21_Kaiserberg_long_term/Image_Aktuell_RGB/full_time_period_tracking_results.geojson", driver="GeoJSON")
