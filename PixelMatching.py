
import matplotlib.pyplot as plt
import geopandas as gpd
import fiona
import rasterio.mask
import rasterio.plot
import rasterio.coords
import numpy as np
import shapely
import numpy.linalg as la
from rasterio.coords import BoundingBox


def get_buffer_around_point(raster_image, center_point: gpd.GeoDataFrame, buffer_radius: int = 30):
    center_point = center_point.to_crs(raster_image.crs)

    if len(center_point) != 1:
        raise ValueError("There is not exactly one point to be tracked.")

    masking_shape_matching_area = center_point.buffer(distance=buffer_radius)
    matching_area, matching_area_transform = rasterio.mask.mask(dataset=raster_image,
                                                                shapes=masking_shape_matching_area,
                                                                crop=True,
                                                                all_touched=True)
    raster_matrix = matching_area[0]

    buffered_raster_image = rasterio.open(
        "./temporal.tif",
        mode="w",
        driver="GTiff",
        height=matching_area.shape[0],
        width=matching_area.shape[1],
        count=1,
        dtype=matching_area.dtype,
        crs=raster_image.crs,
        transform=matching_area_transform)
    buffered_raster_image.write(raster_matrix, 1)
    buffered_raster_image.close()
    buffered_raster_image = rasterio.open("./temporal.tif", mode="r")
    return buffered_raster_image



def get_overlapping_area(file1, file2):
    bbox1 = file1.bounds
    bbox2 = file2.bounds
    minbbox = BoundingBox(left=max(bbox1[0], bbox2[0]),
                          bottom=max(bbox1[1], bbox2[1]),
                          right=min(bbox1[2], bbox2[2]),
                          top=min(bbox1[3], bbox2[3])
                          )

    minbbox = [shapely.Polygon((
                (minbbox[0], minbbox[1]),
                (minbbox[0], minbbox[3]),
                (minbbox[2], minbbox[3]),
                (minbbox[2], minbbox[1])
              ))]
    array_file1, array_file1_transform = rasterio.mask.mask(file1, shapes=minbbox, crop=True)
    array_file2, array_file2_transform = rasterio.mask.mask(file2, shapes=minbbox, crop=True)
    array_file1, array_file2 = array_file1[0], array_file2[0]
    return [array_file1, array_file1_transform], [array_file2, array_file2_transform]


def get_submatrix(shape, upper_left_index, matrix):
    submatrix = matrix[upper_left_index[0]:upper_left_index[0]+shape[0], upper_left_index[1]:upper_left_index[1]+shape[1]]
    return submatrix


def track_cell(tracked_cell_matrix: np.ndarray, search_cell_matrix: np.ndarray):
    height_tracked_cell = tracked_cell_matrix.shape[0]
    width_tracked_cell = tracked_cell_matrix.shape[1]
    height_search_cell = search_cell_matrix.shape[0]
    width_search_cell = search_cell_matrix.shape[1]
    for i in np.arange(0, height_search_cell-height_tracked_cell):
        for j in np.arange(0, width_search_cell-width_tracked_cell):
            search_subcell_matrix = get_submatrix(tracked_cell_matrix.shape, [i,j], search_cell_matrix)
            tracked_vector = tracked_cell_matrix.reshape(1)
            search_subcell_vector = search_subcell_matrix.reshape(1)
            corr = np.corrcoef(tracked_vector, search_subcell_vector)
            print(corr)

def find_matching_area(tracked_image, search_image, tracked_point: gpd.GeoDataFrame, matching_radius: int = 5, search_radius: int = 10):

    """global optimal_match_point
    tracked_point = tracked_point.to_crs(tracked_image.crs)

    if len(tracked_point) != 1:
        raise ValueError("There is not exactly one point to be tracked.")

    if tracked_image.crs != search_image.crs:
        raise ValueError("The two raster images are not based on the same crs")

    #if ~tracked_point.touches(raster_image):
        #raise ValueError("Tracked point and raster image do not intersect.")

    matching_area, matching_area_transform = get_buffer_around_point(raster_image=tracked_image,
                                                                     center_point=tracked_point,
                                                                     buffer_radius=matching_radius)

    search_area, search_area_transform = get_buffer_around_point(raster_image=search_image,
                                                                 center_point=tracked_point,
                                                                 buffer_radius=search_radius)

    rasterio.plot.show(matching_area)
    height = search_area.shape[1]
    width = search_area.shape[2]
    cols, rows = np.meshgrid(range(width), range(height))
    x, y = rasterio.transform.xy(search_area_transform, rows, cols)
    longitudes = np.array(x[0])
    latitudes = np.array(y[0])
    best_loss = np.inf
    for lon in longitudes:
        for lat in latitudes:
            center_point = gpd.GeoDataFrame(index=[0], crs=tracked_image.crs, geometry=[shapely.Point(lon, lat)])
            compare_area, compare_area_transform = get_buffer_around_point(raster_image=search_image,
                                                                           center_point=center_point,
                                                                           buffer_radius=matching_radius)
            loss = la.norm(matching_area[0]-compare_area[0], ord=1)
            if loss < best_loss:
                best_loss = loss
                optimal_match_point = [lon, lat]
    return optimal_match_point
    """


