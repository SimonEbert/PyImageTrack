import rasterio
import shapely
import geopandas as gpd
import matplotlib.pyplot as plt

from CreateGeometries.CreateGeometries import grid_points_on_polygon
from dataloader import read_files
from PixelMatching import find_matching_area

path1 = "../Test_Data/KBT_hillshade_2019-07.tif.tiff"
path2 = "../Test_Data/KBT_hillshade_2021-08.tif.tiff"

file1, file2 = read_files.read_two_image_files(path1, path2)

polygon = gpd.read_file("../Test_Data/Area_outside_rock_glacier.shp")


polygon = polygon.to_crs(crs=3035)

points = grid_points_on_polygon(polygon=polygon, number_of_points=100)


find_matching_area(file1, points.iloc[[0]], matching_radius=2)