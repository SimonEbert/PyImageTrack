import time

import rasterio
import shapely
import geopandas as gpd
import matplotlib.pyplot as plt
import rasterio.plot
import numpy as np
import pandas as pd

import PixelMatching
from CreateGeometries.CreateGeometries import grid_points_on_polygon
from CreateGeometries.CreateGeometries import get_raster_indices_from_points
from dataloader import read_files
from Plots.MakePlots import plot_raster_and_geometry
from Plots.MakePlots import plot_movement_of_points
from Plots.MakePlots import plot_movement_of_points_interpolated
from PixelMatching import get_buffer_around_point
# from PixelMatchingOptimizer import get_pixel_movements_optimizer
from datetime import datetime
import scipy


path1 = "../Test_Data/KBT_hillshade_2019-07.tif"
path2 = "../Test_Data/KBT_hillshade_2021-08.tif"


file1, file2 = read_files.read_two_image_files(path1, path2)

polygon_outside_RG = gpd.read_file("../Test_Data/Area_outside_rock_glacier.shp")

polygon_outside_RG = polygon_outside_RG.to_crs(crs=32632)
#bbox = polygon_outside_RG.bounds

#bbox_polygon = gpd.GeoSeries(shapely.Polygon((
#    (bbox["minx"][0] - 1, bbox["miny"][0] - 1),
#    (bbox["minx"][0] - 1, bbox["maxy"][0] + 1),
#    (bbox["maxx"][0] + 1, bbox["maxy"][0] + 1),
#    (bbox["maxx"][0] + 1, bbox["miny"][0] - 1)
#)),
#    crs=polygon_outside_RG.crs)
#gpd.GeoDataFrame(geometry=bbox_polygon.difference(polygon_outside_RG), crs=polygon_outside_RG.crs)

polygon_inside_RG = gpd.read_file("../Test_Data/Area_inside_rock_glacier_buffered.shp")
polygon_inside_RG = polygon_inside_RG.to_crs(crs=32632)
polygon_inside_RG_unbuffered = gpd.read_file("../Test_Data/Area_inside_rock_glacier.shp")
polygon_inside_RG_unbuffered = polygon_inside_RG_unbuffered.to_crs(crs=32632)

start_time = datetime.now()

# [matrix_test_1, _], _ = PixelMatching.get_overlapping_area(file1, file1)
#
# central_row = 1500
# central_column = 1500
# test_cell = PixelMatching.get_submatrix_symmetric([central_row, central_column], shape=[499, 499], matrix=matrix_test_1)
# interpolator_search_cell = scipy.interpolate.RegularGridInterpolator(
#         (np.arange(0, matrix_test_1.shape[0]), np.arange(0, matrix_test_1.shape[1])),
#         matrix_test_1, fill_value=None, bounds_error=False)
#     # assign indices in respect to indexing in the search cell matrix
#     # tracked_cell_matrix_embedded = np.zeros(search_cell_matrix.shape)
#     # tracked_cell_matrix_embedded[
#     #     int(central_row-np.floor(tracked_cell_matrix.shape[0]/2)):int(np.ceil(central_row+tracked_cell_matrix.shape[0]/2)),
#     #     int(central_column-np.floor(tracked_cell_matrix.shape[1]/2)):int(central_column+np.ceil(tracked_cell_matrix.shape[1]/2))] = tracked_cell_matrix
#
# indices = np.array(np.meshgrid(np.arange(np.ceil(central_row - test_cell.shape[0] / 2),
#                                              np.ceil(central_row + test_cell.shape[0] / 2)),
#                                    np.arange(np.ceil(central_column - test_cell.shape[1] / 2),
#                                              np.ceil(central_column + test_cell.shape[1] / 2)))
#                        ).T.reshape(-1, 2).T
#
# moved_cell = PixelMatching.move_cell_rotation_approach([8/9,-1/9,1/9,8/9, 0, 0], test_cell, interpolator_search_cell, [central_row, central_column], indices)
# rasterio.plot.show(test_cell)
# rasterio.plot.show(moved_cell)

# test_cell_embedded = np.zeros(matrix_test_1.shape)
# test_cell_embedded[
# int(central_row - np.floor(test_cell.shape[0] / 2)):int(
#     np.ceil(central_row + test_cell.shape[0] / 2)),
# int(central_column - np.floor(test_cell.shape[1] / 2)):int(
#     central_column + np.ceil(test_cell.shape[1] / 2))] = test_cell
#
# #
# # test_solution = scipy.optimize.dual_annealing(PixelMatching.lsm_loss_function_rotation, x0=np.array([0,0,0]),
# #                             args=(test_cell, interpolator_search_cell, [central_row, central_column], indices),
# # #                             bounds=((-1/2*np.pi, 1/2*np.pi), (-10, 10), (-10, 10)))  #  for rotating approach, SLSQP works well
# # print(test_solution.x)
# # print(test_solution.success)
# # rotation, shift_rows, shift_columns = test_solution.x
# # print("Error for found solution: ", PixelMatching.lsm_loss_function_rotation(np.array([rotation, shift_rows, shift_columns]), tracked_cell_matrix=test_cell, interpolator_search_cell=interpolator_search_cell, central_indices=[central_row, central_column], indices=indices))
# # print("Error for no moving: ", PixelMatching.lsm_loss_function_rotation(np.array([0, 0, 0]), tracked_cell_matrix=test_cell, interpolator_search_cell=interpolator_search_cell, central_indices=[central_row, central_column], indices=indices))
#
# rasterio.plot.show(matrix_test_1)
# print(PixelMatching.lsm_loss_function_rotation_test([1,0, 0,1, 0, 0], test_cell_embedded, matrix_test_1))
# scipy.optimize.dual_annealing(PixelMatching.lsm_loss_function_rotation_test, np.array([1, 0, 0, 1, 0, 0]), args=(test_cell_embedded, matrix_test_1))


tracked_pixels = PixelMatching.track_movement(file1, file2, polygon_inside_RG, 2000, tracking_area_size=50, remove_outliers=True)

plot_movement_of_points(file1.read(1), file1.transform, tracked_pixels, polygon_inside_RG_unbuffered)
print(tracked_pixels.head(20))
# plot_movement_of_points_interpolated(file1.read(1), file1.transform, tracked_pixels)




print(datetime.now() - start_time)
