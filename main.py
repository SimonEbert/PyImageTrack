import time

import rasterio
import shapely
import geopandas as gpd
import matplotlib.pyplot as plt
import rasterio.plot
import numpy as np
import pandas as pd

import PixelMatching
import PixelMatchingOptimizer
from dataloader import read_files
from Plots.MakePlots import plot_raster_and_geometry
from Plots.MakePlots import plot_movement_of_points
from Plots.MakePlots import plot_movement_of_points_interpolated
from datetime import datetime
import scipy


path1 = "../Test_Data/KBT_hillshade_2019-07.tif"
path2 = "../Test_Data/KBT_hillshade_2021-08.tif"
# path1 = "../Test_Data/2017-08-30-00_00_2017-08-30-23_59_Sentinel-2_L2A_True_color32632.tiff"
# path2 = "../Test_Data/2024-08-10-00_00_2024-08-10-23_59_Sentinel-2_L2A_True_color32632.tiff"



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



[image1_matrix, image2_matrix, image_transform] = PixelMatching.align_images(file1, file2, polygon_outside_RG, 100, select_bands=0, image_alignment_via_lsm=True)


# tracked_pixels = PixelMatchingOptimizer.move_pixels_global(image1_matrix, image2_matrix)


tracked_pixels = PixelMatching.track_movement(image1_matrix, image2_matrix, image_transform, polygon_inside_RG, 2000, tracking_area_size=70, cell_size=40, remove_outliers=True, retry_matching=True, tracking_method="lsm")
print("Finished assembling movement data frame")
plot_movement_of_points(file1.read(1), file1.transform, tracked_pixels, polygon_inside_RG_unbuffered)
# print(tracked_pixels.head(20))
# plot_movement_of_points_interpolated(file1.read(1), file1.transform, tracked_pixels)




print(datetime.now() - start_time)
