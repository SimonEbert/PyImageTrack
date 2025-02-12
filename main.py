# import packages
import geopandas as gpd
import matplotlib.pyplot as plt
from datetime import datetime


import skimage.exposure
import rasterio.plot
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

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
alignment_via_lsm = True
number_of_control_points = 100
image_bands = None
control_tracking_area_size = 60
control_cell_size = 40
tracking_method = "lsm"
number_of_tracked_points = 2000
movement_tracking_area_size = 120
movement_cell_size = 70
remove_outliers = True
retry_matching = True
years_between_observations = 2.83

# set paths for the two image files
path1 = "../Test_Data/KBT_hillshade_2019-07.tif"
path2 = "../Test_Data/KBT_hillshade_2021-08.tif"
path1 = "../Test_Data/Orthophotos_Kaiserberg/new_try/Orthophoto_2020_modified.tif"
path2 = "../Test_Data/Orthophotos_Kaiserberg/new_try/Orthophoto_2023_modified.tif"

file1, file2 = HandleFiles.read_two_image_files(path1, path2)

polygon_outside_RG = gpd.read_file("../Test_Data/Area_outside_rock_glacier.shp")

polygon_outside_RG = polygon_outside_RG.to_crs(crs=32632)


polygon_inside_RG_unbuffered = gpd.read_file("../Test_Data/Area_inside_rock_glacier.shp")
polygon_inside_RG_unbuffered = polygon_inside_RG_unbuffered.to_crs(crs=32632)

# save current time for computation time measurements
start_time = datetime.now()

# align the two images and crop to the same extent
[image1_matrix, image2_matrix, image_transform] = (
    PixelMatching.align_images(file1, file2,
                               reference_area=polygon_outside_RG,
                               image_alignment_via_lsm=alignment_via_lsm,
                               number_of_control_points=number_of_control_points, select_bands=image_bands,
                               tracking_area_size=control_tracking_area_size,
                               cell_size=control_cell_size))



image1_matrix = skimage.exposure.equalize_adapthist(image=image1_matrix.astype(int), kernel_size=movement_tracking_area_size, clip_limit=0.9)
image2_matrix = skimage.exposure.equalize_adapthist(image=image2_matrix.astype(int), kernel_size=movement_tracking_area_size, clip_limit=0.9)
rasterio.plot.show(image1_matrix)
rasterio.plot.show(image2_matrix)


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




#
# fig, ax = plt.subplots()
# background = rasterio.open("../Test_Data/Orthophotos_Kaiserberg/new_try/Orthophoto_2020_modified.tif")
# rasterio.plot.show(background, ax=ax)
#
# polygon_inside_RG_unbuffered.plot(ax=ax, alpha=0.2, color="blue")
# polygon_inside_RG_unbuffered.boundary.plot(ax=ax, color="black", alpha=0.2)
#
#
# tracked_pixels.plot(ax=ax, alpha=0.5, color="orange", marker="s", markersize=200)
# tracked_pixels.plot(ax=ax, color="yellow", marker="o", markersize=5)
#
#
# # Create custom legend handles
# polygon_handle = Patch(color='blue', label='Rock glacier area', alpha=0.2)
# point_handle = Line2D([0], [0], marker='o', markerfacecolor='yellow', markersize=5, label='Tracked points', linestyle='', markeredgewidth=0)
# image_section_handle = Patch(color='orange', label='Image section used for tracking', alpha=0.5)
#
#
# # Add the legend to the plot
# ax.legend(handles=[polygon_handle, point_handle, image_section_handle], loc='lower right', fontsize='small')
# ax.set_title("Area-based feature tracking")
# plt.tight_layout()
# plt.show()



