import time

import rasterio
import shapely
import geopandas as gpd
import matplotlib.pyplot as plt
import rasterio.plot
import numpy as np
import pandas as pd

import PixelMatching
from CreateGeometries.HandleGeometries import georeference_tracked_points
from dataloader import HandleFiles
from Plots.MakePlots import plot_movement_of_points
from datetime import datetime
import scipy


plt.rcParams['figure.dpi'] = 300

# Set the path for saving results
output_folder_path = "../Output_results/Kaiserberg/" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

# Set parameters
alignment_via_lsm = True
number_of_control_points = 100
image_bands = 0
control_tracking_area_size = 60
control_cell_size = 40
tracking_method = "lsm"
number_of_tracked_points = 2000
movement_tracking_area_size = 80
movement_cell_size = 30
remove_outliers = True
retry_matching = True
years_between_observations = 2.083# 10 for Winnebach, 2.083 for Kaiserberg, 2.83 for Kaiserberg true color





path1 = "../Test_Data/KBT_hillshade_2019-07.tif"
path2 = "../Test_Data/KBT_hillshade_2021-08.tif"
# path1 = "../Test_Data/Orthophotos_Kaiserberg/new_try/Orthophoto_2020_modified.tif"
# path2 = "../Test_Data/Orthophotos_Kaiserberg/new_try/Orthophoto_2023_modified.tif"
# path1 = "../Test_Data/Winnebachgebiet/DEM_2006_2007_modified.tif"
# path2 = "../Test_Data/Winnebachgebiet/DEM_2017_2020_modified.tif"


file1, file2 = HandleFiles.read_two_image_files(path1, path2)


polygon_outside_RG = gpd.read_file("../Test_Data/Area_outside_rock_glacier.shp")# _true_color

polygon_outside_RG = polygon_outside_RG.to_crs(crs=32632)

polygon_inside_RG = gpd.read_file("../Test_Data/Area_inside_rock_glacier_buffered.shp")
polygon_inside_RG = polygon_inside_RG.to_crs(crs=32632)
polygon_inside_RG_unbuffered = gpd.read_file("../Test_Data/Area_inside_rock_glacier.shp")
polygon_inside_RG_unbuffered = polygon_inside_RG_unbuffered.to_crs(crs=32632)

polygon_inside_RG_detail = gpd.read_file("../Test_Data/Area_inside_rock_glacier_detail.shp")
polygon_inside_RG_detail = polygon_inside_RG_detail.to_crs(crs=32632)

high_movement_area_polygon = gpd.read_file("../Test_Data/High_movement_area_rock_glacier.shp")
high_movement_area_polygon = high_movement_area_polygon.to_crs(crs=32632)


Winnebachgebiet_moving_area = gpd.read_file("../Test_Data/Winnebachgebiet/moving_area.shp")
Winnebachgebiet_reference_area = gpd.read_file("../Test_Data/Winnebachgebiet/reference_area.shp")
Winnebachgebiet_moving_area = Winnebachgebiet_moving_area.to_crs(crs=32632)
Winnebachgebiet_reference_area = Winnebachgebiet_reference_area.to_crs(crs=32632)

start_time = datetime.now()

# good parameters:
# polygon_outside_RG, number_of_control_points=100
[image1_matrix, image2_matrix, image_transform] = PixelMatching.align_images(file1, file2, reference_area=polygon_outside_RG, image_alignment_via_lsm=alignment_via_lsm, number_of_control_points=number_of_control_points, select_bands=image_bands, tracking_area_size=control_tracking_area_size, cell_size=control_cell_size)#True, 100, 0, 60, 40

# good parameters:
# polygon_inside_RG_unbuffered, number_of_tracked_points = 2000, tracking_area_size=80, cell_size=30

tracked_pixels = PixelMatching.track_movement(image1_matrix, image2_matrix, image_transform, tracking_area=polygon_inside_RG_unbuffered, number_of_tracked_points=number_of_tracked_points, tracking_area_size=movement_tracking_area_size, cell_size=movement_cell_size, remove_outliers=remove_outliers, retry_matching=retry_matching, tracking_method=tracking_method)
print("Finished assembling movement data frame")
tracked_pixels = georeference_tracked_points(tracked_pixels, image_transform, crs=32632, years_between_observations=years_between_observations)

#tracked_pixels = HandleFiles.read_tracking_results("../Output_results/Kaiserberg/2025_01_19_22_29_31/tracking_results.geojson")

computation_time = datetime.now() - start_time

parameter_dict = {"alignment_via_lsm": alignment_via_lsm,
                  "number_of_control_points": number_of_control_points,
                  "image_bands": image_bands,
                  "control_tracking_area_size": control_tracking_area_size,
                  "control_cell_size": control_cell_size,
                  "tracking_method": tracking_method,
                  "number_of_tracked_points": number_of_tracked_points,
                  "movement_tracking_area_size": movement_tracking_area_size,
                  "movement_cell_size": movement_cell_size,
                  "remove_outliers": remove_outliers,
                  "retry_matching": retry_matching,
                  "years_between_observations": years_between_observations,
                  "computation_time": computation_time
}


HandleFiles.write_results(tracked_pixels, parameter_dict, folder_path=output_folder_path)

plot_movement_of_points(image1_matrix, image_transform, tracked_pixels, save_path=output_folder_path + "/point_movement_" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + ".png")


















#
# results_without_outliers = HandleFiles.read_tracking_results("../Output_results/Kaiserberg/2025_01_16_18_26_28/tracking_results.geojson")
# results_with_outliers = HandleFiles.read_tracking_results("../Output_results/Kaiserberg/2025_01_21_18_35_06/tracking_results.geojson")
#
# fig, ax = plt.subplots(1, 2)
#
#
# results_without_outliers.plot(ax=ax[0], column="movement_distance_per_year", legend=True, markersize=8, marker="s", alpha=1.0,
#                  vmin=0, vmax=2.1
#                  )
#
# rasterio.plot.show(image1_matrix, transform=image_transform, ax=ax[0], cmap="Greys")
#
# # Arrow plotting
# for row in sorted(list(set(results_without_outliers.loc[:, "row"])))[::4]:
#     for column in sorted(list(set(results_without_outliers.loc[:, "column"])))[::4]:
#
#         arrow_point = results_without_outliers.loc[(results_without_outliers['row'] == row) & (results_without_outliers['column'] == column)]
#         if not arrow_point.empty:
#             arrow_point = arrow_point.iloc[0]
#             plt.arrow(arrow_point["geometry"].x, arrow_point["geometry"].y,
#                           arrow_point["movement_column_direction"] * 3 / arrow_point["movement_distance"],
#                           -arrow_point["movement_row_direction"] * 3 / arrow_point["movement_distance"],
#                           head_width=10, head_length=10, color="black", alpha=1, ax=ax[0])
# plt.title("Movement Distance in " + results_without_outliers.crs.axis_info[0].unit_name + " per year")
#
#
# results_with_outliers.plot(ax=ax[1], column="movement_distance_per_year", markersize=8, marker="s", alpha=1.0,
#                  vmin=0, vmax=2.1,
#                  )
#
# rasterio.plot.show(image1_matrix, transform=image_transform, ax=ax[1], cmap="Greys")
#
# # Arrow plotting
# for row in sorted(list(set(results_with_outliers.loc[:, "row"])))[::4]:
#     for column in sorted(list(set(results_with_outliers.loc[:, "column"])))[::4]:
#
#         arrow_point = results_with_outliers.loc[(results_with_outliers['row'] == row) & (results_with_outliers['column'] == column)]
#         if not arrow_point.empty:
#             arrow_point = arrow_point.iloc[0]
#             plt.arrow(arrow_point["geometry"].x, arrow_point["geometry"].y,
#                           arrow_point["movement_column_direction"] * 3 / arrow_point["movement_distance"],
#                           -arrow_point["movement_row_direction"] * 3 / arrow_point["movement_distance"],
#                           head_width=10, head_length=10, color="black", alpha=1)
# leg = ax[0].get_legend()
# leg.set_bbox_to_anchor((0, 0, 0.2, 0.2))
#
# fig.show()
#

