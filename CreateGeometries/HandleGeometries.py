import geopandas as gpd
import shapely
import numpy as np
import rasterio.transform
import pandas as pd
import rasterio.mask
from rasterio.coords import BoundingBox



def grid_points_on_polygon(polygon: gpd.GeoDataFrame, number_of_points: int = 10):
    """Creates a grid of roughly a given number of evenly spaced points on the given polygon. The exact number of points
    can vary depending on the shape of the polygon."""
    minlongitude, minlatitude, maxlongitude, maxlatitude = polygon.bounds.iloc[0]

    length_latitude = maxlatitude-minlatitude
    length_longitude = maxlongitude-minlongitude
    enclosing_rectangle = shapely.Polygon((
        (minlongitude, minlatitude),
        (maxlongitude, minlatitude),
        (maxlongitude, maxlatitude),
        (minlongitude, maxlatitude),
        (minlongitude, minlatitude)
    ))

    enclosing_rectangle = gpd.GeoDataFrame(index=[0], crs=polygon.crs, geometry=[enclosing_rectangle])

    area_ratio=(enclosing_rectangle.area/polygon.area).iloc[0]
    number_of_latitude_points = np.sqrt(length_latitude/length_longitude*number_of_points)
    number_of_longitude_points = (length_longitude/length_latitude*number_of_latitude_points)
    number_of_latitude_points *= np.sqrt(area_ratio)
    number_of_longitude_points *= np.sqrt(area_ratio)
    number_of_latitude_points = np.ceil(number_of_latitude_points)
    number_of_longitude_points = np.ceil(number_of_longitude_points)

    points = []
    for lat in np.arange(minlatitude, maxlatitude, length_latitude/number_of_latitude_points):
        for lon in np.arange(minlongitude, maxlongitude, length_longitude/number_of_longitude_points):
            points.append(shapely.Point(lon, lat))

    points = gpd.GeoDataFrame(crs=polygon.crs, geometry=points)


    points = points[points.intersects(polygon.loc[0, "geometry"])]
    print("Created ", len(points), " points on the polygon.")
    return points


def get_raster_indices_from_points(points: gpd.GeoDataFrame, raster_matrix_transform):
    xs = points["geometry"].x.to_list()
    ys = points["geometry"].y.to_list()
    rows, cols = rasterio.transform.rowcol(raster_matrix_transform, xs, ys)
    return rows, cols


def get_overlapping_area(file1, file2):
    bbox1 = file1.bounds
    bbox2 = file2.bounds
    minbbox = BoundingBox(left=max(bbox1[0], bbox2[0]),
                          bottom=max(bbox1[1], bbox2[1]),
                          right=min(bbox1[2], bbox2[2]),
                          top=min(bbox1[3], bbox2[3])
                          )

    minbbox_polygon = [shapely.Polygon((
                (minbbox[0], minbbox[1]),
                (minbbox[0], minbbox[3]),
                (minbbox[2], minbbox[3]),
                (minbbox[2], minbbox[1])
              ))]

    array_file1, array_file1_transform = rasterio.mask.mask(file1, shapes=minbbox_polygon, crop=True)
    array_file2, array_file2_transform = rasterio.mask.mask(file2, shapes=minbbox_polygon, crop=True)

    # If the matrices have different number of pixels, pad the smaller one with 0 to match the size of the larger one
    # TODO: Only works for multiple bands at the moment
    if array_file1.shape[-2] < array_file2.shape[-2]:
        array_file1 = np.pad(array_file1, pad_width=((0,0),(0,array_file2.shape[-2]-array_file1.shape[-2]), (0,0)), constant_values=(0, 0))
    if array_file1.shape[-2] > array_file2.shape[-2]:
        array_file2 = np.pad(array_file2, pad_width=((0,0),(0, array_file1.shape[-2] - array_file2.shape[-2]), (0,0)), constant_values=(0, 0))

    if array_file1.shape[-1] < array_file2.shape[-1]:
        array_file1 = np.pad(array_file1, pad_width=((0,0),(0,0),(0, array_file2.shape[-1] - array_file1.shape[-1])), constant_values=(0, 0))
    if array_file1.shape[-1] > array_file2.shape[-1]:
        array_file2 = np.pad(array_file2, pad_width=((0,0),(0,0), (0, array_file1.shape[-1] - array_file2.shape[-1])), constant_values=(0, 0))



    return [array_file1, array_file1_transform], [array_file2, array_file2_transform]





def georeference_tracked_points(tracked_pixels: pd.DataFrame, raster_transform, crs, years_between_observations = 1):
    [x, y] = rasterio.transform.xy(raster_transform, tracked_pixels.loc[:, "row"], tracked_pixels.loc[:, "column"])
    georeferenced_tracked_pixels = gpd.GeoDataFrame(tracked_pixels.loc[:,
                              ["row", "column", "movement_row_direction", "movement_column_direction",
                               "movement_distance_pixels"]],
                              geometry=gpd.points_from_xy(x=x, y=y), crs=crs)

    georeferenced_tracked_pixels.insert(5, "movement_distance", np.linalg.norm([-raster_transform[4] * georeferenced_tracked_pixels.loc[:, "movement_row_direction"].values, raster_transform[0]*georeferenced_tracked_pixels.loc[:, "movement_column_direction"].values], axis=0))
    georeferenced_tracked_pixels.insert(6, "movement_distance_per_year", georeferenced_tracked_pixels["movement_distance"]/years_between_observations)

    return georeferenced_tracked_pixels
