import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.mask
import rasterio.transform
import shapely


def get_submatrix_symmetric(central_index, shape, matrix):
    """
    Extracts a symmetric section of a given matrix and shape, so that central_index is in the centre of the returned
    array. If shape specifies an even height or width, it is decreased by one to ensure that there exists a unique
    central index in the returned array
    Parameters
    ----------
    central_index :
        A two-element list, containing the row and column indices of the entry, which lies in the centre of the returned
        array.
    shape :
        A two-element list, containing the row and column number of the returned array. If one of these is an even
        number, it will be decreased by one to ensure that a unique central index exists.
    matrix :
        The matrix from which the section is extracted.
    Returns
    ----------
    submatrix: A numpy array of the specified shape.
    """
    submatrix = matrix[...,
            max(0,int(central_index[0] - np.ceil(shape[0] / 2)) + 1):max(0,int(central_index[0] + np.ceil(shape[0] / 2))),
            max(0,int(central_index[1] - np.ceil(shape[1] / 2)) + 1):max(0,int(central_index[1] + np.ceil(shape[1] / 2)))]
    return submatrix


def grid_points_on_polygon_by_distance(polygon: gpd.GeoDataFrame,
                                       distance_of_points: float = 10,
                                       distance_px: float = None,
                                       pixel_size: float = None):
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
    points = points[points.intersects(polygon.geometry[0])]

    if polygon.crs is not None:
        unit_name = points.crs.axis_info[0].unit_name
    else:
        unit_name = "pixel"
    if distance_px is None:
        print(
            f"Created {len(points)} points on the polygon "
            f"with distance {distance_of_points:.1f} {unit_name}."
        )
    else:
        print(
            f"Created {len(points)} points on the polygon "
            f"with distance {distance_of_points:.1f} {unit_name} "
            f"({distance_px:.1f} px)."
        )

    return points


def random_points_on_polygon_by_number(polygon: gpd.GeoDataFrame, number_of_points: int):
    points = gpd.GeoDataFrame()
    """
    Creates randomly distributed points on a polygon given as a one-element GeoDataFrame."""
    while len(points) < number_of_points:
        # generate random points in the bounds of the polygon
        minx, miny, maxx, maxy = polygon.bounds.iloc[0]
        x = np.random.uniform(minx, maxx, 2 * number_of_points).tolist()
        y = np.random.uniform(miny, maxy, 2 * number_of_points).tolist()
        # create DataFrame with the new points
        new_points = gpd.GeoDataFrame(crs=polygon.crs, geometry=gpd.points_from_xy(x, y))
        points = pd.concat([points, new_points[new_points.intersects(polygon.geometry[0])]])
    points = points.head(number_of_points)
    points.set_index(np.arange(number_of_points), inplace=True)
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

    xs = np.array(points["geometry"].x.to_list(), dtype=float)
    ys = np.array(points["geometry"].y.to_list(), dtype=float)
    valid_mask = np.isfinite(xs) & np.isfinite(ys)
    if not np.all(valid_mask):
        xs = xs[valid_mask]
        ys = ys[valid_mask]
    rows, cols = rasterio.transform.rowcol(raster_matrix_transform, xs, ys)
    return rows, cols


def crop_images_to_intersection(file1, file2):
    """
    Crops the two files to their intersection based on the spatial information provided with the two images
    Parameters
    ----------
    file1, file2: The two raster image files as opened rasterio objects.
    Returns
    ----------
    [array_file1, array_file1_transform]: The raster matrix for the first file and its respective transform.
    [array_file2, array_file2_transform]: The raster matrix for the second file and its respective transform.
    """

    bbox1 = file1.bounds
    bbox2 = file2.bounds
    minbbox = rasterio.coords.BoundingBox(left=max(bbox1[0], bbox2[0]),
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

    return [array_file1, array_file1_transform], [array_file2, array_file2_transform]


def georeference_tracked_points(tracked_pixels: pd.DataFrame, raster_transform, crs,
                                years_between_observations: float=1) -> gpd.GeoDataFrame:
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
    years_between_observations: float = 1
        A float representing the number of years between the two images for calculating average yearly movement rates.
    Returns
    ----------
    georeferenced_tracked_pixels:
        A GeoDataFrame containing the tracked pixels with the previously mentioned columns, as well as the columns
        "movement_distance" and "movement_distance_per_year", specifying the movement in the unit of the given
        coordinate reference system and one geometry column.
    """
    [x, y] = rasterio.transform.xy(raster_transform, tracked_pixels.loc[:, "row"], tracked_pixels.loc[:, "column"])
    # georeferenced_tracked_pixels = gpd.GeoDataFrame(tracked_pixels.loc[:,
    #                                                 ["row", "column", "movement_row_direction",
    #                                                  "movement_column_direction",
    #                                                  "movement_distance_pixels", "movement_bearing_pixels"]],
    #                                                 geometry=gpd.points_from_xy(x=x, y=y), crs=crs)
    georeferenced_tracked_pixels = gpd.GeoDataFrame(tracked_pixels, geometry=gpd.points_from_xy(x=x, y=y), crs=crs)
    georeferenced_tracked_pixels["movement_distance"] = np.linalg.norm(
        [-raster_transform[4] * georeferenced_tracked_pixels.loc[:, "movement_row_direction"].values,
         raster_transform[0] * georeferenced_tracked_pixels.loc[:, "movement_column_direction"].values], axis=0)
    georeferenced_tracked_pixels["movement_distance_per_year"] = (georeferenced_tracked_pixels["movement_distance"]
                                                                  / years_between_observations)

    georeferenced_tracked_pixels["valid"] = True
    georeferenced_tracked_pixels.loc[
        np.isnan(georeferenced_tracked_pixels["movement_distance_per_year"]),
        "valid"] = False

    return georeferenced_tracked_pixels


def circular_std_deg(angles_deg):
    # Convert to radians
    angles_rad = np.deg2rad(angles_deg)

    # Compute mean resultant vector
    sin_sum = np.sum(np.sin(angles_rad))
    cos_sum = np.sum(np.cos(angles_rad))
    R = np.sqrt(sin_sum ** 2 + cos_sum ** 2) / len(angles_rad)

    # Circular standard deviation in radians
    circ_std_rad = np.sqrt(-2 * np.log(R))

    # Convert back to degrees
    circ_std_deg = np.rad2deg(circ_std_rad)
    return circ_std_deg


def get_submatrix_rect_from_extents(central_index, extents, matrix):
    """
    Extract an asymmetric rectangular submatrix around `central_index` using the extents
    tuple (pos_x, neg_x, pos_y, neg_y), measured in pixels from the center.
    It safely clips to matrix bounds and supports both 2D and 3D (channels-first) arrays.

    Returns
    -------
    submatrix : np.ndarray
        The extracted search window.
    center_in_submatrix : tuple(int, int)
        The (row, col) coordinates of the original center pixel inside the returned submatrix.
    """
    pos_x, neg_x, pos_y, neg_y = map(int, extents)
    row_c = int(central_index[0])
    col_c = int(central_index[1])

    r0 = max(0, row_c - neg_y)
    r1 = min(matrix.shape[-2], row_c + pos_y + 1)
    c0 = max(0, col_c - neg_x)
    c1 = min(matrix.shape[-1], col_c + pos_x + 1)

    if len(matrix.shape) == 3:
        sub = matrix[:, r0:r1, c0:c1]
    else:
        sub = matrix[r0:r1, c0:c1]

    center_in_sub = (row_c - r0, col_c - c0)
    return sub, center_in_sub
