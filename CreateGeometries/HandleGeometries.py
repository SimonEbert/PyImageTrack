import geopandas as gpd
import shapely
import numpy as np
import rasterio.transform
import pandas as pd
import rasterio.mask
from rasterio.coords import BoundingBox
import matplotlib.pyplot as plt


def grid_points_on_polygon_by_number_of_points(polygon: gpd.GeoDataFrame, number_of_points: int = 10):
    """
    Creates an evenly spaced grid of points inside the given polygon. An approximation of the number of created points
    can be given, the actual number of points may differ depending on the shape of the polygon. The resulting
    GeoDataFrame will have the same coordinate reference system as the polygon.
    Parameters
    ----------
    polygon: gpd.GeoDataFrame
        The polygon where the points will be created.
    number_of_points: int = 10
        The approximate number of points to be created. The function calculates an approximate spacing based on this
        number and the area ratio of the given polygon and its enclosing rectangle so that the resulting grid is exactly
        evenly spaced and contains roughly this number of points.
    Returns
    ----------
    points: A GeoDataFrame containing the created points.
    """
    minlongitude, minlatitude, maxlongitude, maxlatitude = polygon.bounds.iloc[0]

    length_latitude = np.abs(maxlatitude - minlatitude)
    length_longitude = np.abs(maxlongitude - minlongitude)
    enclosing_rectangle = shapely.Polygon((
        (minlongitude, minlatitude),
        (maxlongitude, minlatitude),
        (maxlongitude, maxlatitude),
        (minlongitude, maxlatitude),
        (minlongitude, minlatitude)
    ))

    enclosing_rectangle = gpd.GeoDataFrame(index=[0], crs=polygon.crs, geometry=[enclosing_rectangle])

    area_ratio = (enclosing_rectangle.area / polygon.area).iloc[0]
    number_of_latitude_points = np.sqrt(length_latitude / length_longitude * number_of_points)
    number_of_longitude_points = (length_longitude / length_latitude * number_of_latitude_points)
    number_of_latitude_points *= np.sqrt(area_ratio)
    number_of_longitude_points *= np.sqrt(area_ratio)
    number_of_latitude_points = np.ceil(number_of_latitude_points)
    number_of_longitude_points = np.ceil(number_of_longitude_points)

    points = []
    for lat in np.arange(minlatitude, maxlatitude, length_latitude / number_of_latitude_points):
        for lon in np.arange(minlongitude, maxlongitude, length_longitude / number_of_longitude_points):
            points.append(shapely.Point(lon, lat))

    points = gpd.GeoDataFrame(crs=polygon.crs, geometry=points)

    points = points[points.intersects(polygon.loc[0, "geometry"])]
    print("Created ", len(points), " points on the polygon.")
    return points


def grid_points_on_polygon_by_distance(polygon: gpd.GeoDataFrame, distance_of_points: float = 10):

    minx = polygon.bounds.loc[0, 'minx']
    miny = polygon.bounds.loc[0, 'miny']
    maxx = polygon.bounds.loc[0, 'maxx']
    maxy = polygon.bounds.loc[0, 'maxy']

    extent_corners = gpd.GeoDataFrame(["minx_miny", "maxx_miny", "minx_maxy", "maxx_maxy"],
                                      columns=["names"],
                                      geometry=[shapely.geometry.Point(minx, miny),
                                                shapely.geometry.Point(maxx, miny),
                                                shapely.geometry.Point(minx, maxy),
                                                shapely.geometry.Point(maxx, maxy)],
                                      crs=polygon.crs)

    width_image_crs_unit = extent_corners.iloc[0].geometry.distance(extent_corners.iloc[1].geometry)
    height_image_crs_unit = extent_corners.iloc[0].geometry.distance(extent_corners.iloc[2].geometry)

    number_of_points_width = width_image_crs_unit / distance_of_points
    number_of_points_height = height_image_crs_unit / distance_of_points
    points = []
    for x in np.arange(minx, maxx, width_image_crs_unit / number_of_points_width):
        for y in np.arange(miny, maxy, height_image_crs_unit / number_of_points_height):
            points.append(shapely.geometry.Point(x, y))

    points = gpd.GeoDataFrame(crs=polygon.crs, geometry=points)

    print(type(polygon))

    points = points[points.intersects(polygon.loc[0, "geometry"])]
    print("Created ", len(points), " points on the polygon with distance ", distance_of_points,
          points.crs.axis_info[0].unit_name)
    return points



def get_raster_indices_from_points(points: gpd.GeoDataFrame, raster_matrix_transform):
    """
    Transforms the coordinates of points in a given coordinate reference system to their respective matrix indices for a
    given transform
    Parameters
    ----------
    points: gpd.GeoDataFrame
        A GeoDataFrame containing points in a certain coordinate reference system.
    raster_matrix_transform
        An object of the class Affine as used by the rasterio package, representing the transform from the matrix
        indices to the coordinate reference system of the points.
    Returns
    ----------
    rows, cols: The row and column indices respectively for the points.
    """

    xs = points["geometry"].x.to_list()
    ys = points["geometry"].y.to_list()
    rows, cols = rasterio.transform.rowcol(raster_matrix_transform, xs, ys)
    return rows, cols


def get_overlapping_area(file1, file2):

    """
    Crops the two files to their intersection and pads multi-channel images with different pixel sizes with 0 so that
    the resulting matrices have the same size.
    Parameters
    ----------
    file1, file2: The two raster image files as opened rasterio objects.
    Returns
    ----------
    [array_file1, array_file2]: The raster matrix for the first file and its respective transform.
    [array_file2, array_file2_transform]: The raster matrix for the first file and its respective transform.
    """
    
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
    if array_file1.shape[-2] < array_file2.shape[-2]:
        array_file1 = np.pad(array_file1,
                             pad_width=((0, 0), (0, array_file2.shape[-2] - array_file1.shape[-2]), (0, 0)),
                             constant_values=(0, 0))
    if array_file1.shape[-2] > array_file2.shape[-2]:
        array_file2 = np.pad(array_file2,
                             pad_width=((0, 0), (0, array_file1.shape[-2] - array_file2.shape[-2]), (0, 0)),
                             constant_values=(0, 0))

    if array_file1.shape[-1] < array_file2.shape[-1]:
        array_file1 = np.pad(array_file1,
                             pad_width=((0, 0), (0, 0), (0, array_file2.shape[-1] - array_file1.shape[-1])),
                             constant_values=(0, 0))
    if array_file1.shape[-1] > array_file2.shape[-1]:
        array_file2 = np.pad(array_file2,
                             pad_width=((0, 0), (0, 0), (0, array_file1.shape[-1] - array_file2.shape[-1])),
                             constant_values=(0, 0))

    return [array_file1, array_file1_transform], [array_file2, array_file2_transform]


def georeference_tracked_points(tracked_pixels: pd.DataFrame, raster_transform, crs, years_between_observations=1):
    
    """
    Georeferences a DataFrame with tracked points and calculates their movement (absolute and per year) in the unit
    specified by the coordinate reference system.
    Parameters
    ----------
    tracked_pixels: pd.DataFrame
        A DataFrame containing tracked pixels with columns "row", "column" (specifying the position of the point on the
        raster image), and "movement_row_direction", "movement_column_direction", "movement_distance_pixels" (specifying
        its movement in terms of raster pixels).
    raster_transform:
        An object of the class Affine as used by the rasterio package, representing the transform from the matrix
        indices to the coordinate reference system of the points.
    crs:
        An identifier for a coordinate reference system to which the resulting GeoDataFrame will be projected.
    years_between_observations = 1
        A float representing the number of years between the two images for calculating average yearly movement rates.
    Returns
    ----------
    georeferenced_tracked_pixels:
        A GeoDataFrame containing the tracked pixels with the previously mentioned columns, as well as the columns
        "movement_distance" and "movement_distance_per_year", specifying the movement in the unit of the given
        coordinate reference system and one geometry column.
    """
    
    [x, y] = rasterio.transform.xy(raster_transform, tracked_pixels.loc[:, "row"], tracked_pixels.loc[:, "column"])
    georeferenced_tracked_pixels = gpd.GeoDataFrame(tracked_pixels.loc[:,
                                                    ["row", "column", "movement_row_direction",
                                                     "movement_column_direction",
                                                     "movement_distance_pixels"]],
                                                    geometry=gpd.points_from_xy(x=x, y=y), crs=crs)

    georeferenced_tracked_pixels.insert(5, "movement_distance", np.linalg.norm(
        [-raster_transform[4] * georeferenced_tracked_pixels.loc[:, "movement_row_direction"].values,
         raster_transform[0] * georeferenced_tracked_pixels.loc[:, "movement_column_direction"].values], axis=0))
    georeferenced_tracked_pixels.insert(6, "movement_distance_per_year",
                                        georeferenced_tracked_pixels["movement_distance"] / years_between_observations)

    return georeferenced_tracked_pixels
