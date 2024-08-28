import rasterio
import shapely
import geopandas as gpd
import matplotlib.pyplot as plt

from CreateGeometries.CreateGeometries import grid_points_on_polygon
from dataloader import read_files
from PixelMatching import find_matching_area
from PixelMatching import get_buffer_around_point

path1 = "../Test_Data/KBT_hillshade_2019-07.tif.tiff"
path2 = "../Test_Data/KBT_hillshade_2021-08.tif.tiff"

file1, file2 = read_files.read_two_image_files(path1, path2)

polygon = gpd.read_file("../Test_Data/Area_outside_rock_glacier.shp")


polygon = polygon.to_crs(crs=3035)

points = grid_points_on_polygon(polygon=polygon, number_of_points=100)


#optimal_match = find_matching_area(file1, file2, points.iloc[[0]], matching_radius=2)

trial_area1 = get_buffer_around_point(file1, points.iloc[[50]], 200)[0][0]
trial_area2 = get_buffer_around_point(file2, points.iloc[[50]], 200)[0][0]
print(trial_area1[1000:1018, 1000:1018])
print(trial_area2[1000:1018, 1000:1018])
