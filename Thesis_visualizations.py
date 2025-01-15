
import pandas as pd
import rasterio.plot
import matplotlib
import matplotlib.pyplot as plt
from dataloader import HandleFiles

# VISUALIZATION of Outlier reasons for a full tracking
# result_data = pd.read_csv("../Output_results/Kaiserberg/2025_01_15_14_27_31/tracking_results.csv")
# print(len(result_data))
#
# outlier_points = result_data[(result_data["movement_row_direction"] == -0.001) | (result_data["movement_row_direction"] == -0.002) | (result_data["movement_row_direction"] == -0.003)| (result_data["movement_row_direction"] == -0.004)| (result_data["movement_row_direction"] == -0.005)]
# valid_points = result_data[~((result_data["movement_row_direction"] == -0.001) | (result_data["movement_row_direction"] == -0.002) | (result_data["movement_row_direction"] == -0.003)| (result_data["movement_row_direction"] == -0.004)| (result_data["movement_row_direction"] == -0.005))]
#
# outlier_points = outlier_points.astype({'movement_row_direction': str})
# outlier_points.loc[outlier_points["movement_row_direction"] == "-0.001", "movement_row_direction"] = "Cross-correlation\nyielded no result"
# outlier_points.loc[outlier_points["movement_row_direction"] == "-0.002", "movement_row_direction"] = "Optimization did\nnot converge"
# outlier_points.loc[outlier_points["movement_row_direction"] == "-0.003", "movement_row_direction"] = "Unrealistic\nTransformation\ndeterminant"
# outlier_points.loc[outlier_points["movement_row_direction"] == "-0.004", "movement_row_direction"] = "Rotation\noutlier"
# outlier_points.loc[outlier_points["movement_row_direction"] == "-0.005", "movement_row_direction"] = "Velocity\noutlier"
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
# ax.set_title('Movement rates of valid matched points in m/yr')
# plt.show()
#
# fig, ax = plt.subplots()
# fig.subplots_adjust(bottom=0.2)
# reasons_for_removal = outlier_points['movement_row_direction'].value_counts().index
# counts_of_reasons = outlier_points['movement_row_direction'].value_counts().values
# ax.bar(x=[0,1,2,3], tick_label=reasons_for_removal, height=counts_of_reasons, width=0.1, color="Orange", alpha=0.8)
# # ax.tick_params(axis='x', labelrotation=90)
# ax.set_yscale('log')
# ax.set_title('Reasons for outlier removal')
#
# ax.set_ylim(None, 1000)
#
# plt.show()



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


# MAKE plot with directions


# default matplotlib imports
import matplotlib as mpl
# import matplotlib.pyplot as plt
#
# # colormap
# from matplotlib.colors import Normalize
# from matplotlib.cm import ScalarMappable
# cmap = plt.cm.twilight # colormap (plt.cm.hsv = circular colormap)
#
# # colormap bar
# from mpl_toolkits.axes_grid1 import make_axes_locatable
#
# #####
#
# # data
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
# fig, ax = plt.subplots()
# for theta in thetas:
# 	ax.scatter(np.cos(theta), np.sin(theta), marker='o', s=500,
# 		color=scalarMap.to_rgba(theta), alpha=0.5) # use scalarMap to get color from numerical value
#
# plt.axis('off')
#
# # show
# plt.show()
