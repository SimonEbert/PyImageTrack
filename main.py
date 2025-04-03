# import packages
import geopandas as gpd
import matplotlib.pyplot as plt
from datetime import datetime

import numpy as np
import rasterio

#import functions
import PixelMatching
from CreateGeometries.HandleGeometries import georeference_tracked_points
from dataloader import HandleFiles
from Plots.MakePlots import plot_movement_of_points

# improve figure quality
plt.rcParams['figure.dpi'] = 300

# Set the path for saving results
output_folder_path = "../Output_results/Kaiserberg/" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

# Set parameters
alignment_via_lsm = False
number_of_control_points = 100
image_bands = 0
control_tracking_area_size = 60
control_cell_size = 40
tracking_method = "lsm"
number_of_tracked_points = 2000
movement_tracking_area_size = 80
movement_cell_size = 50
remove_outliers = True
retry_matching = True
years_between_observations = 2.083

# set paths for the two image files
path1 = "../Test_Data/KBT_hillshade_2019-07.tif"
path2 = "../Test_Data/KBT_hillshade_2021-08.tif"

file1, file2 = HandleFiles.read_two_image_files(path1, path2)

polygon_outside_RG = gpd.read_file("../Test_Data/Area_outside_rock_glacier.shp")

polygon_outside_RG = polygon_outside_RG.to_crs(crs=32632)


polygon_inside_RG_unbuffered = gpd.read_file("../Test_Data/Area_inside_rock_glacier.shp")
polygon_inside_RG_unbuffered = polygon_inside_RG_unbuffered.to_crs(crs=32632)

# save current time for computation time measurements
start_time = datetime.now()

# crop the images to the same extent
[image1_matrix, image1_transform], [image2_matrix, _] = PixelMatching.get_overlapping_area(file1, file2)



# align the two images
[image1_matrix, image2_matrix, image_transform] = (
    PixelMatching.align_images(image1_matrix, image2_matrix, image1_transform,
                               reference_area=polygon_outside_RG,
                               image_alignment_via_lsm=alignment_via_lsm,
                               number_of_control_points=number_of_control_points, select_bands=image_bands,
                               tracking_area_size=control_tracking_area_size,
                               cell_size=control_cell_size))




tracked_pixels = PixelMatching.track_movement(image1_matrix, image2_matrix, image_transform,
                                              tracking_area=polygon_inside_RG_unbuffered,
                                              number_of_tracked_points=number_of_tracked_points,
                                              tracking_area_size=movement_tracking_area_size,
                                              cell_size=movement_cell_size, remove_outliers=remove_outliers,
                                              retry_matching=retry_matching, tracking_method=tracking_method)
print("Finished assembling movement data frame")
tracked_pixels = georeference_tracked_points(tracked_pixels, image_transform, crs=32632,
                                             years_between_observations=years_between_observations)

# read tracking results instead of performing the tracking
# tracked_pixels = HandleFiles.read_tracking_results(
# "../Output_results/Kaiserberg/2025_01_19_22_29_31/tracking_results.geojson")

# save full computation time
computation_time = datetime.now() - start_time

# save parameter dictionary
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

# write results with the parameter dictionary to the specified directory
HandleFiles.write_results(tracked_pixels, parameter_dict, folder_path=output_folder_path)


# plot and save the movement visualization
plot_movement_of_points(image1_matrix, image_transform, tracked_pixels,
                        save_path=output_folder_path + "/point_movement_" + datetime.now().strftime(
                           "%Y_%m_%d_%H_%M_%S") + ".png")




