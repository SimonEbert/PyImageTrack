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
from PixelMatching import get_buffer_around_point
from PixelMatchingOptimizer import get_pixel_movements_optimizer
from datetime import datetime

start_time = datetime.now()

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

polygon_inside_RG = gpd.read_file("../Test_Data/Area_inside_rock_glacier.shp")
polygon_inside_RG = polygon_inside_RG.to_crs(crs=32632)

points = grid_points_on_polygon(polygon=polygon_inside_RG, number_of_points=500)

[trial_area1_matrix, trial_area1_transform], [trial_area2_matrix,
                                              trial_area2_transform] = PixelMatching.get_overlapping_area(file1, file2)
rows, cols = get_raster_indices_from_points(points, trial_area2_transform)
tracked_points_pixels = np.array([rows, cols]).transpose()

get_pixel_movements_optimizer(trial_area1_matrix, trial_area2_matrix)

tracked_pixels = pd.DataFrame()
# for central_index in tracked_points_pixels:
#     track_cell1 = PixelMatching.get_submatrix_symmetric(central_index=central_index, shape=(20, 20),
#                                                         matrix=trial_area1_matrix)
#     print("Calculate tracking for pixel ", central_index)
#     search_area2 = PixelMatching.get_submatrix_symmetric(central_index=central_index, shape=(50, 50),
#                                                          matrix=trial_area2_matrix)
#     match = PixelMatching.track_cell(track_cell1, search_area2)
#     print("Pixel movement: ", match)
#     tracked_pixels = pd.concat([tracked_pixels, pd.DataFrame({"row": central_index[0],
#                                                              "column": central_index[1],
#                                                              "movement_row_direction": match[0],
#                                                              "movement_column_direction": match[1]},
#                                                              index=[len(tracked_pixels)])])
#     # target_cell2 = PixelMatching.get_submatrix_symmetric((central_index[0] + match[0], central_index[1] + match[1]), (20, 20), trial_area2_matrix)
# plot_movement_of_points(trial_area1_matrix, trial_area1_transform, tracked_pixels)
#
# print(datetime.now() - start_time)
