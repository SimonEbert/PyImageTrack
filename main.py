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
from PixelMatching import find_matching_area
from PixelMatching import get_buffer_around_point
from datetime import datetime

path1 = "../Test_Data/KBT_hillshade_2019-07.tif.tiff"
path2 = "../Test_Data/KBT_hillshade_2021-08.tif.tiff"


file1, file2 = read_files.read_two_image_files(path1, path2)

polygon_outside_RG = gpd.read_file("../Test_Data/Area_outside_rock_glacier.shp")

polygon_outside_RG = polygon_outside_RG.to_crs(crs=32632)
bbox = polygon_outside_RG.bounds

bbox_polygon = gpd.GeoSeries(shapely.Polygon((
    (bbox["minx"][0] - 1, bbox["miny"][0] - 1),
    (bbox["minx"][0] - 1, bbox["maxy"][0] + 1),
    (bbox["maxx"][0] + 1, bbox["maxy"][0] + 1),
    (bbox["maxx"][0] + 1, bbox["miny"][0] - 1)
)),
    crs=polygon_outside_RG.crs)
polygon_inside_RG = gpd.GeoDataFrame(geometry=bbox_polygon.difference(polygon_outside_RG), crs=polygon_outside_RG.crs)

points = grid_points_on_polygon(polygon=polygon_inside_RG, number_of_points=500)

[trial_area1_matrix, trial_area1_transform], [trial_area2_matrix,
                                              trial_area2_transform] = PixelMatching.get_overlapping_area(file1, file2)

rows, cols = get_raster_indices_from_points(points, trial_area1_transform)
tracked_points_pixels = np.array([rows, cols]).transpose()

plot_raster_and_geometry(trial_area1_matrix, trial_area1_transform, points, alpha=0.3)

tracked_pixels = pd.DataFrame()
for central_index in tracked_points_pixels:
    track_cell1 = PixelMatching.get_submatrix_symmetric(central_index=central_index, shape=(20, 20),
                                                        matrix=trial_area1_matrix)
    if 0 in track_cell1 or len(track_cell1) < 19:
        print("Skipping pixel ", central_index)
        continue
    else:
        print("Calculate tracking for pixel ", central_index)
        search_area2 = PixelMatching.get_submatrix_symmetric(central_index=central_index, shape=(50, 50),
                                                             matrix=trial_area2_matrix)
        match = PixelMatching.track_cell(track_cell1, search_area2)
        print("Pixel movement: ", match)
        tracked_pixels = pd.concat([tracked_pixels, pd.DataFrame({"row": central_index[0],
                                                                  "column": central_index[1],
                                                                  "movement_row_direction": match[0],
                                                                  "movement_column_direction": match[1]},
                                                                 index=[len(tracked_pixels)])])
    # target_cell2 = PixelMatching.get_submatrix_symmetric((central_index[0] + match[0], central_index[1] + match[1]), (20, 20), trial_area2_matrix)
plot_movement_of_points(trial_area1_matrix, trial_area1_transform, tracked_pixels)

