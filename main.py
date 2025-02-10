import shapely
import geopandas as gpd
import matplotlib.pyplot as plt
import rasterio.plot
import numpy as np

import PixelMatching
from CreateGeometries.HandleGeometries import georeference_tracked_points
from dataloader import HandleFiles
from Plots.MakePlots import plot_movement_of_points
from datetime import datetime
import skimage.exposure

plt.rcParams['figure.dpi'] = 300

# Set the path for saving results
output_folder_path = "../Output_results/Kaiserberg/" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

# Set parameters
alignment_via_lsm = True
number_of_control_points = 100
image_bands = None
control_tracking_area_size = 60
control_cell_size = 40
tracking_method = "lsm"
number_of_tracked_points = 2000
movement_tracking_area_size = 80
movement_cell_size = 30
remove_outliers = True
retry_matching = True
years_between_observations = 2.083  # 10 for Winnebach, 2.083 for Kaiserberg, 2.83 for Kaiserberg true color

path1 = "../Test_Data/KBT_hillshade_2019-07.tif"
path2 = "../Test_Data/KBT_hillshade_2021-08.tif"
# path1 = "../Test_Data/Orthophotos_Kaiserberg/Long_time_series/1953_modified.tif"
# path2 = "../Test_Data/Orthophotos_Kaiserberg/Long_time_series/1971_modified.tif"
# path1 = "../Test_Data/Orthophotos_Kaiserberg/new_try/Orthophoto_2020_modified.tif"
# path2 = "../Test_Data/Orthophotos_Kaiserberg/new_try/Orthophoto_2023_modified.tif"
# path1 = "../Test_Data/Winnebachgebiet/DEM_2006_2007_modified.tif"
# path2 = "../Test_Data/Winnebachgebiet/DEM_2017_2020_modified.tif"


file1, file2 = HandleFiles.read_two_image_files(path1, path2)

polygon_outside_RG = gpd.read_file("../Test_Data/Area_outside_rock_glacier.shp")

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
[image1_matrix, image2_matrix, image_transform] = (
    PixelMatching.align_images(file1, file2,
                               reference_area=polygon_outside_RG,
                               image_alignment_via_lsm=alignment_via_lsm,
                               number_of_control_points=number_of_control_points, select_bands=image_bands,
                               tracking_area_size=control_tracking_area_size,
                               cell_size=control_cell_size))


# image1_matrix = skimage.exposure.equalize_adapthist(image=image1_matrix.astype(int), kernel_size=movement_tracking_area_size, clip_limit=0.9)
# image2_matrix = skimage.exposure.equalize_adapthist(image=image2_matrix.astype(int), kernel_size=movement_tracking_area_size, clip_limit=0.9)
# rasterio.plot.show(image1_matrix)
# rasterio.plot.show(image2_matrix)


tracked_pixels = PixelMatching.track_movement(image1_matrix, image2_matrix, image_transform,
                                              tracking_area=polygon_inside_RG_unbuffered,
                                              number_of_tracked_points=number_of_tracked_points,
                                              tracking_area_size=movement_tracking_area_size,
                                              cell_size=movement_cell_size, remove_outliers=remove_outliers,
                                              retry_matching=retry_matching, tracking_method=tracking_method)
print("Finished assembling movement data frame")
tracked_pixels = georeference_tracked_points(tracked_pixels, image_transform, crs=32632,
                                             years_between_observations=years_between_observations)

# tracked_pixels = HandleFiles.read_tracking_results(
# "../Output_results/Kaiserberg/2025_01_19_22_29_31/tracking_results.geojson")

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

plot_movement_of_points(image1_matrix, image_transform, tracked_pixels,
                        save_path=output_folder_path + "/point_movement_" + datetime.now().strftime(
                            "%Y_%m_%d_%H_%M_%S") + ".png")

