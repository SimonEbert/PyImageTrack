import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio.plot
import matplotlib
import matplotlib.pyplot as plt
from dataloader import HandleFiles
import rasterio.mask
import shapely
from Plots.MakePlots import plot_movement_of_points
from CreateGeometries.HandleGeometries import get_overlapping_area
plt.rcParams['figure.dpi'] = 600



# VISUALIZATION of Outlier reasons for a full tracking


# valid_points = pd.read_csv("../Output_results/Kaiserberg/2025_01_16_18_26_28/tracking_results.csv")
#
#
# matplotlib.rcParams.update({'font.size': 13})
# plt.rc('figure', titlesize=15)
#
# fig, ax = plt.subplots()
# fig.subplots_adjust(bottom=0.2)
#
# ax.hist(valid_points['movement_distance_per_year'], bins=50, color="Orange", alpha=0.8)
# ax.set_yscale('log')
#
# ax.set_ylim(None, 1000)
# ax.set_xlabel('Distance (m/yr)')
# ax.set_ylabel('Frequency (logarithmic)')
# ax.set_title('Movement rates of valid matched points in m/yr')
# plt.show()
#
#
#
#
#
#
# result_data = pd.read_csv("../Output_results/Kaiserberg/2025_02_09_13_53_25/tracking_results.csv")
# print(len(result_data))
#
# outlier_points = result_data[(result_data["movement_row_direction"] == -0.001) | (result_data["movement_row_direction"] == -0.002) | (result_data["movement_row_direction"] == -0.003)| (result_data["movement_row_direction"] == -0.004)| (result_data["movement_row_direction"] == -0.005)]
# valid_points = result_data[~((result_data["movement_row_direction"] == -0.001) | (result_data["movement_row_direction"] == -0.002) | (result_data["movement_row_direction"] == -0.003)| (result_data["movement_row_direction"] == -0.004)| (result_data["movement_row_direction"] == -0.005))]
#
# outlier_points = outlier_points.astype({'movement_row_direction': str})
# outlier_points.loc[outlier_points["movement_row_direction"] == "-0.001", "movement_row_direction"] = "Cross-correlation\nyielded no\nvalid result"
# outlier_points.loc[outlier_points["movement_row_direction"] == "-0.002", "movement_row_direction"] = "Optimization did\nnot converge"
# outlier_points.loc[outlier_points["movement_row_direction"] == "-0.003", "movement_row_direction"] = "Unrealistic\ntransformation\ndeterminant"
# outlier_points.loc[outlier_points["movement_row_direction"] == "-0.004", "movement_row_direction"] = "Rotation\noutlier"
# outlier_points.loc[outlier_points["movement_row_direction"] == "-0.005", "movement_row_direction"] = "Velocity\noutlier"
#
#
#
# fig, ax = plt.subplots()
# fig.subplots_adjust(bottom=0.2)
# reasons_for_removal = outlier_points['movement_row_direction'].value_counts().index
# counts_of_reasons = outlier_points['movement_row_direction'].value_counts().values
# ax.bar(x=[0,1,2,3], tick_label=reasons_for_removal, height=counts_of_reasons, width=0.1, color="Orange", alpha=0.8)
# # ax.tick_params(axis='x', labelrotation=90)
# ax.set_yscale('log')
# ax.set_title('Reasons for outlier removal')
# ax.set_ylabel('Frequency (logarithmic)')
# ax.set_ylim(None, 1000)
#
# plt.show()


# OUTLIER MAP
# fig, ax = plt.subplots()
#
# background_rock_glacier = rasterio.open("../Test_Data/KBT_hillshade_2019-07.tif")
# background_rock_glacier_1 = rasterio.open("../Test_Data/KBT_hillshade_2021-08.tif")
# [background_rock_glacier, background_rock_glacier_transform], [_, _] = get_overlapping_area(background_rock_glacier, background_rock_glacier_1)
# rasterio.plot.show(background_rock_glacier, transform=background_rock_glacier_transform, ax=ax, cmap="Greys")
#
# outlier_geographical_data = gpd.read_file("../Output_results/Kaiserberg/2025_02_09_13_53_25/tracking_results.geojson")
# outlier_geographical_data = outlier_geographical_data[(outlier_geographical_data["movement_row_direction"] == -0.001) | (outlier_geographical_data["movement_row_direction"] == -0.002) | (outlier_geographical_data["movement_row_direction"] == -0.003)| (outlier_geographical_data["movement_row_direction"] == -0.004)| (outlier_geographical_data["movement_row_direction"] == -0.005)]
#
#
#
# # outlier_geographical_data = outlier_geographical_data.astype({'movement_row_direction': str})
# outlier_geographical_data.loc[outlier_geographical_data["movement_row_direction"] == -0.001, "movement_row_direction"] = "Cross-correlation yielded no valid result"
# outlier_geographical_data.loc[outlier_geographical_data["movement_row_direction"] == -0.002, "movement_row_direction"] = "Optimization did not converge"
# outlier_geographical_data.loc[outlier_geographical_data["movement_row_direction"] == -0.003, "movement_row_direction"] = "Unrealistic Transformation determinant"
# outlier_geographical_data.loc[outlier_geographical_data["movement_row_direction"] == -0.004, "movement_row_direction"] = "Rotation outlier"
# outlier_geographical_data.loc[outlier_geographical_data["movement_row_direction"] == -0.005, "movement_row_direction"] = "Velocity outlier"
#
# outlier_geographical_data.plot(categorical=True, ax=ax, column="movement_row_direction", markersize=8, marker="o", alpha=1.0, legend=True, legend_kwds={"loc": "lower left", "fontsize": "small", "reverse": True}, cmap="plasma")
# plt.title("Reasons for invalid matching of points")

plt.show()

#CORRELATION coefficient map
# data_with_correlation_coefficients = HandleFiles.read_tracking_results("../Output_results/Kaiserberg/2025_01_15_18_38_46/tracking_results.geojson")
#
# fig, ax = plt.subplots()
# raster_image = rasterio.open("../Test_Data/KBT_hillshade_2019-07.tif")
#
#
#
# rasterio.plot.show(raster_image, ax=ax)
# data_with_correlation_coefficients.plot("correlation", markersize=6, marker="s", alpha=0.5, legend=True, ax=ax, cmap="RdYlGn")
# ax.set_title("Best correlation coefficients")
# plt.show()




# DIRECTION plot


# default matplotlib imports
import matplotlib as mpl


#
# results_direction = HandleFiles.read_tracking_results("../Output_results/Kaiserberg/2025_01_16_18_26_28/tracking_results.geojson")
#
#
# results_direction['movement_angle'] = np.arctan2(-results_direction['movement_row_direction'], results_direction['movement_column_direction'])
# results_direction.loc[results_direction['movement_angle']<0, 'movement_angle'] = results_direction['movement_angle'] + 2*np.pi
# results_direction['movement_angle'] = np.degrees(results_direction['movement_angle'])
#
#
# import matplotlib.pyplot as plt
#
# # colormap
# from matplotlib.colors import Normalize
# from matplotlib.cm import ScalarMappable
# cmap = plt.cm.twilight_shifted # colormap (plt.cm.hsv = circular colormap)
#
# # colormap bar
# from mpl_toolkits.axes_grid1 import make_axes_locatable
#
# #####
#
# # data
# fig = plt.figure(figsize=(8,6))
# gs = fig.add_gridspec(1, 2, width_ratios=[3,1])
# ax0 = fig.add_subplot(gs[0])
# ax1 = fig.add_subplot(gs[1])
#
# results_direction.plot(ax=ax0, column="movement_angle", markersize=21, marker="s", alpha=1.0,cmap=cmap,
#                  )
# raster_image = rasterio.open("../Test_Data/Orthophotos_Kaiserberg/Orthophoto_2023_modified.tif")
#
# raster_image, raster_image_transform = rasterio.mask.mask(raster_image, shapes=[shapely.Polygon((
# 	(627450,5196350),
# 	(628100, 5196350),
# 	(628100, 5196785),
# 	(627450,5196785)
# 	))], crop=True)
#
#
#
# rasterio.plot.show(raster_image, ax=ax0, transform=raster_image_transform)
# import numpy as np
# thetas = np.linspace(0, 2*np.pi, 500)
#
# #####
#
# # colormap
# norm = Normalize(           # definition of norm
# 	vmin=0,                   # minimum value
# 	vmax=2*np.pi)             # maximum value
# scalarMap = ScalarMappable( # define relation from numerical value to color
# 	norm=norm,                # norm defined above
# 	cmap=cmap)                # colormap
#
# # figure
# for theta in thetas:
# 	ax1.scatter(np.cos(theta), np.sin(theta), marker='o', s=90,
# 		color=scalarMap.to_rgba(theta), alpha=0.25) # use scalarMap to get color from numerical value
#
#
#
#
# ax1.text(0.9,0, '0°-', ha='right', va='center',fontsize=12, fontweight='bold', color='black')
# ax1.text(-0.85,0, '180°', ha='left', va='center',fontsize=12, fontweight='bold', color='black')
# ax1.text(0,0.85, '90°', ha='center', va='top',fontsize=12, fontweight='bold', color='black')
# ax1.text(0,-0.85, '270°', ha='center', va='bottom',fontsize=12, fontweight='bold', color='black')
#
#
# plt.axis('off')
# fig.subplots_adjust(hspace=0.1)
# ax1.set_aspect('equal', 'box')
# ax1.set_title("Movement direction\nin degrees")
# ax0.set_title("Directions of the tracked points")
# plt.tight_layout()
#
# # show
# plt.show()



# ROBUSTNESS plot for one-channel images
# pd.set_option('display.max_columns', 50)
# pd.set_option('display.max_rows', 500)
#
#
# # Constant search area size
# results_tracked_cell_size_10 = HandleFiles.read_tracking_results("../Output_results/Kaiserberg/2025_01_16_18_03_01/tracking_results.geojson")
# results_tracked_cell_size_10.columns = ['row', 'column', 'movement_row_direction_size_10', 'movement_column_direction_size_10', 'movement_distance_pixels_size_10', 'movement_distance_size_10', 'movement_distance_per_year_size_10', 'geometry']
# results_tracked_cell_size_20 = HandleFiles.read_tracking_results("../Output_results/Kaiserberg/2025_01_16_18_14_00/tracking_results.geojson")
# results_tracked_cell_size_20.columns = ['row', 'column', 'movement_row_direction_size_20', 'movement_column_direction_size_20', 'movement_distance_pixels_size_20', 'movement_distance_size_20', 'movement_distance_per_year_size_20', 'geometry']
# results_tracked_cell_size_30 = HandleFiles.read_tracking_results("../Output_results/Kaiserberg/2025_01_16_18_26_28/tracking_results.geojson")
# results_tracked_cell_size_30.columns = ['row', 'column', 'movement_row_direction_size_30', 'movement_column_direction_size_30', 'movement_distance_pixels_size_30', 'movement_distance_size_30', 'movement_distance_per_year_size_30', 'geometry']
# results_tracked_cell_size_40 = HandleFiles.read_tracking_results("../Output_results/Kaiserberg/2025_01_16_18_37_21/tracking_results.geojson")
# results_tracked_cell_size_40.columns = ['row', 'column', 'movement_row_direction_size_40', 'movement_column_direction_size_40', 'movement_distance_pixels_size_40', 'movement_distance_size_40', 'movement_distance_per_year_size_40', 'geometry']
# results_tracked_cell_size_50 = HandleFiles.read_tracking_results("../Output_results/Kaiserberg/2025_01_16_18_54_20/tracking_results.geojson")
# results_tracked_cell_size_50.columns = ['row', 'column', 'movement_row_direction_size_50', 'movement_column_direction_size_50', 'movement_distance_pixels_size_50', 'movement_distance_size_50', 'movement_distance_per_year_size_50', 'geometry']
#
#
# merged_results = results_tracked_cell_size_10.merge(results_tracked_cell_size_20, on=["row", "column", "geometry"], copy=False)
# merged_results = merged_results.merge(results_tracked_cell_size_30, on=["row", "column", "geometry"], copy=False)
# merged_results = merged_results.merge(results_tracked_cell_size_40, on=["row", "column", "geometry"], copy=False)
# merged_results = merged_results.merge(results_tracked_cell_size_50, on=["row", "column", "geometry"], copy=False)
#
#
# merged_results['minimum_movement_distance_per_year'] = merged_results[['movement_distance_per_year_size_10', 'movement_distance_per_year_size_20', 'movement_distance_per_year_size_30', 'movement_distance_per_year_size_40', 'movement_distance_per_year_size_50']].min(axis=1)
# merged_results['maximum_movement_distance_per_year'] = merged_results[['movement_distance_per_year_size_10', 'movement_distance_per_year_size_20', 'movement_distance_per_year_size_30', 'movement_distance_per_year_size_40', 'movement_distance_per_year_size_50']].max(axis=1)
# merged_results['mean_movement_distance_per_year'] = merged_results[['movement_distance_per_year_size_10', 'movement_distance_per_year_size_20', 'movement_distance_per_year_size_30', 'movement_distance_per_year_size_40', 'movement_distance_per_year_size_50']].mean(axis=1)
# merged_results['maximal_difference_movement_distance_per_year'] = merged_results['maximum_movement_distance_per_year'] - merged_results['minimum_movement_distance_per_year']
#
# print(merged_results.maximal_difference_movement_distance_per_year.quantile(.99))
#
# print(merged_results['maximal_difference_movement_distance_per_year'].sort_values(ascending=False).head(10))
#
# fig, ax = plt.subplots()
# merged_results.plot(ax=ax, column="maximal_difference_movement_distance_per_year", legend=True, markersize=15, marker="s", alpha=1.0,
#                     norm=matplotlib.colors.LogNorm(vmin=0.000022, vmax=merged_results['maximal_difference_movement_distance_per_year'].max())
#     # # #             # vmin=0, vmax=5,
#                  )
# matplotlib.rcParams.update({'font.size': 12})
#
# ax.set_title("Difference between yearly movement rates in m")
# #plt.tight_layout()
# plt.show()
#
# #
# fig, ax = plt.subplots()
# plt.hist(merged_results['maximal_difference_movement_distance_per_year'], bins=250)
# # ax.set_yscale('log')
# ax.set_ylabel("Number of occurrences")
# ax.set_xlabel("Maximal error between different parameters in m")
# ax.set_title("Frequency of deviations between different parameters in m")
# plt.show()
#
#
#
# invalid_matching_size_10 = len(merged_results[np.isnan(merged_results['movement_distance_per_year_size_10'])])
# invalid_matching_size_20 = len(merged_results[np.isnan(merged_results['movement_distance_per_year_size_20'])])
# invalid_matching_size_30 = len(merged_results[np.isnan(merged_results['movement_distance_per_year_size_30'])])
# invalid_matching_size_40 = len(merged_results[np.isnan(merged_results['movement_distance_per_year_size_40'])])
# invalid_matching_size_50 = len(merged_results[np.isnan(merged_results['movement_distance_per_year_size_50'])])
#
#
# def calculate_percentage(x):
#     return x/2050*100
# def reverse_percentage(x):
#     return x/100*2050
# fig, ax = plt.subplots(layout='constrained')
# ax.plot([10, 20, 30, 40, 50], [invalid_matching_size_10, invalid_matching_size_20, invalid_matching_size_30, invalid_matching_size_40, invalid_matching_size_50])
# ax.set_xticks([10, 20, 30, 40, 50], ["10", "20", "30", "40", "50"], fontsize=14)
# ax.set_xlabel("Size of the tracked cell", fontsize=14)
# ax.set_ylabel("Number of points with invalid matchings\ndetected by the Implementation", fontsize=14)
# secax_y2 = ax.secondary_yaxis('right', functions=(calculate_percentage, reverse_percentage))
# secax_y2.set_ylabel("Percentage of invalid matchings\ndetected by the Implementation", fontsize=14)
# secax_y2.set_yticks([2, 3, 4, 5, 6, 7,8], ["2%", "3%", "4%", "5%", "6%",  "7%", "8%"], fontsize=14)
# plt.title("Invalid matchings for different tracked cell sizes", fontsize=16)
# plt.show()


# RUNTIME plots (data assembled from differeent parts)
#
# fig, ax = plt.subplots(layout='constrained')
# ax.plot([70, 80, 100, 120, 140], [597, 541, 664, 852, 1072], label="Single-band hillshade image")
# ax.plot([70, 80, 100, 120, 140], [2661, 2464, 2612, 2869, 3210], label="Multi-band image")
# ax.legend(loc='upper left')
# ax.set_xlabel("search area size", fontsize=14)
# ax.set_ylabel("Computation time in seconds", fontsize=14)
# ax.set_title("Computational time for different search area sizes", fontsize=16)
# plt.show()










# results_true_color = HandleFiles.read_tracking_results("../Output_results/Kaiserberg/2025_01_23_17_24_14/tracking_results.geojson")

# results_hillshade = HandleFiles.read_tracking_results("../Output_results/Kaiserberg/2025_01_16_18_26_28/tracking_results.geojson")
#
#
#
# results_true_color['movement_angle'] = np.arctan2(-results_true_color['movement_row_direction'], results_true_color['movement_column_direction'])
#
# results_hillshade['movement_angle'] = np.arctan2(-results_hillshade['movement_row_direction'], results_hillshade['movement_column_direction'])
#
#
# results_true_color.loc[results_true_color['movement_angle']<0, 'movement_angle'] = results_true_color['movement_angle'] + 2*np.pi
# results_hillshade.loc[results_hillshade['movement_angle']<0, 'movement_angle'] = results_hillshade['movement_angle'] + 2*np.pi
#
# results_true_color['movement_angle'] = np.degrees(results_true_color['movement_angle'])
# results_hillshade['movement_angle'] = np.degrees(results_hillshade['movement_angle'])
#
#
# direction_difference = results_true_color['movement_angle'] - results_hillshade['movement_angle']
# direction_difference[direction_difference < -180] += 360
# direction_difference[direction_difference > 180] -= 360
# #
#
# velocity_difference = results_true_color['movement_distance_per_year'] - results_hillshade['movement_distance_per_year']
#
# plt.hist(velocity_difference, bins='auto')
# plt.title("Movement rate deviations between hillshade and orthophoto data")
# plt.xlabel("Movement rate deviations in meter/year")
# plt.ylabel("Frequency")
# plt.show()






# results_true_color['angle_difference'] = direction_difference
# results_true_color.plot('angle_difference', legend=True)
# plt.show()

# TRUE-COLOR result plot
# results_true_color = HandleFiles.read_tracking_results("../Output_results/Kaiserberg/2025_01_23_17_24_14/tracking_results.geojson")
#
#
#
#
# raster_image = rasterio.open("../Test_Data/Orthophotos_Kaiserberg/Orthophoto_2023_modified.tif")
#
# raster_image, raster_image_transform = rasterio.mask.mask(raster_image, shapes=[shapely.Polygon((
# 	(627450,5196350),
# 	(628150, 5196350),
# 	(628150, 5196785),
# 	(627450,5196785)
# 	))], crop=True)
# plot_movement_of_points(raster_image, raster_image_transform, results_true_color)



#
# import scipy.interpolate.ndgriddata
# points = results[['row', 'column']]
# xi = np.meshgrid(np.arange(0, points.max(axis=0)['row']),np.arange(0,points.max(axis=0)['column']))
# points = points.to_numpy()
# values = results['movement_distance_per_year'].to_numpy()
#
# interpolated_results = scipy.interpolate.griddata(points,values,xi,method='linear', rescale=True)
# fig, ax = plt.subplots()
#
# rasterio.plot.show(raster_image, ax=ax)
# rasterio.plot.show(interpolated_results.transpose(), ax=ax)
# plt.show()


# MEAN calculation for the perfect matching parameters
# results_hillshade = HandleFiles.read_tracking_results("../Output_results/Kaiserberg/2025_01_16_18_26_28/tracking_results.geojson")
#
# mean_velocity = np.nanmean(results_hillshade['movement_distance_per_year'])
#
# print(mean_velocity)


