import geopandas as gpd
import numpy as np
import pandas as pd

from PixelMatching import track_movement
from CreateGeometries.HandleGeometries import georeference_tracked_points


def remove_points_below_LoD(image1_raster: np.ndarray, image2_raster: np.ndarray, image_transform,
                            refererence_area: gpd.GeoDataFrame,
                            tracked_pixels: pd.DataFrame):
    reference_tracked_pixels = track_movement(image1_matrix=image1_raster, image2_matrix=image2_raster,
                                              image_transform=image_transform, tracking_area=refererence_area,
                                              number_of_tracked_points=100, tracking_area_size=60, cell_size=30,
                                              tracking_method="lsm", remove_outliers=False, retry_matching=False)

    reference_tracked_pixels = georeference_tracked_points(reference_tracked_pixels, image_transform, 31254, 27)

    print(reference_tracked_pixels)
    mean_movement_reference_pixels = reference_tracked_pixels["movement_distance_per_year"].mean()
    print(mean_movement_reference_pixels)
    print(tracked_pixels)
    tracked_pixels[tracked_pixels["movement_distance_per_year"] <= mean_movement_reference_pixels] = np.nan

    return [tracked_pixels, mean_movement_reference_pixels]


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
