import rasterio
import rasterio.plot
import geopandas as gpd
import os

def read_two_image_files(path1: str, path2: str):
    """Reads two .tiff files from the given paths and returns them as rasterio objects"""
    file1 = rasterio.open(path1)
    file2 = rasterio.open(path2)
    return file1, file2


def write_results(georeferenced_tracked_pixels: gpd.GeoDataFrame, parameter_dict, folder_path: str):
    os.makedirs(folder_path)
    georeferenced_tracked_pixels.to_file(folder_path + "/tracking_results.geojson", driver='GeoJSON')
    georeferenced_tracked_pixels.to_csv(folder_path + "/tracking_results.csv")
    with open(folder_path + "/tracking_parameters.txt", "w") as parameter_file:
        print(parameter_dict, file=parameter_file)


def read_tracking_results(file_path: str):
    georeferenced_tracked_pixels = gpd.read_file(file_path)
    return georeferenced_tracked_pixels

