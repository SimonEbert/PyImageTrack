import rasterio
import rasterio.plot
import geopandas as gpd
import os


def read_two_image_files(path1: str, path2: str):

    """
    Reads two raster images using the rasterio package from the given paths
    path1, path2: str
        The paths to the two raster image files.
    Returns
    ----------
    file1, file2: The two files opened in the rasterio package
    """

    file1 = rasterio.open(path1)
    file2 = rasterio.open(path2)
    return file1, file2


def write_results(georeferenced_tracked_pixels: gpd.GeoDataFrame, parameter_dict, folder_path: str):

    """
    Writes the results of a performed tracking to a specified folder along with a file containing the parameters of this
    tracking. The files are being saved with the name "tracking_results", thus it is recommended to use a different
    folder for each individual tracking.
    Parameters
    ----------
    georeferenced_tracked_pixels: gpd.GeoDataFrame
        A GeoDataFrame containing the results of the tracking. This will be saved to
        "folder_path/tracking_results.geojson" and "folder_path/tracking_results.csv" respectively.
    parameter_dict
        A dictionary containing the parameters used for this run. Can contain additional notes regarding the tracking
        and will be saved to "folder_path/tracking_parameters.txt".
    folder_path: str
        The location of the folder, where the results should be saved. Not existing directories will be created.
    Returns
    ----------
    None
    """

    os.makedirs(folder_path)
    georeferenced_tracked_pixels.to_file(folder_path + "/tracking_results.geojson", driver='GeoJSON')
    georeferenced_tracked_pixels.to_csv(folder_path + "/tracking_results.csv")
    with open(folder_path + "/tracking_parameters.txt", "w") as parameter_file:
        print(parameter_dict, file=parameter_file)


def read_tracking_results(file_path: str):
    
    """
    Reads results of a previous tracking to a GeoDataFrame.
    Parameters
    ----------
    file_path: str
        The path to read the file from. The file is expected to be readable for GeoPandas (e.g. a geojson file)
    Returns
    ----------
    georeferenced_tracked_pixels: A GeoDataFrame containing the results of a previous tracking.
    """
    
    georeferenced_tracked_pixels = gpd.read_file(file_path)
    return georeferenced_tracked_pixels

