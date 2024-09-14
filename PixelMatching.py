
import matplotlib.pyplot as plt
import geopandas as gpd
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

    minbbox_polygon = [shapely.Polygon((
                (minbbox[0], minbbox[1]),
                (minbbox[0], minbbox[3]),
                (minbbox[2], minbbox[3]),
                (minbbox[2], minbbox[1])
              ))]
    array_file1, array_file1_transform = rasterio.mask.mask(file1, shapes=minbbox_polygon, crop=True)
    array_file2, array_file2_transform = rasterio.mask.mask(file2, shapes=minbbox_polygon, crop=True)
    array_file1, array_file2 = array_file1[0], array_file2[0]
    return [array_file1, array_file1_transform], [array_file2, array_file2_transform]


def get_submatrix_symmetric(central_index, shape, matrix):
    """Makes even given shapes one number smaller to ascertain there is one unique central index"""
    submatrix = matrix[int(central_index[0]-np.ceil(shape[0]/2))+1:int(central_index[0]+np.ceil(shape[0]/2)),
                       int(central_index[1]-np.ceil(shape[1]/2))+1:int(central_index[1]+np.ceil(shape[1]/2))]
    #submatrix = matrix[central_index[0]:central_index[0]+shape[0], central_index[1]:central_index[1]+shape[1]]
    return submatrix


def get_submatrix_asymmetric(central_index, negative_x: int, positive_x: int, positive_y: int, negative_y: int, matrix):
    submatrix = matrix[central_index[0]-positive_y:central_index[0]+negative_y+1,
                       central_index[1]-negative_x:central_index[1]+positive_x+1]
    return submatrix


def track_cell(tracked_cell_matrix: np.ndarray, search_cell_matrix: np.ndarray):
    method = "crosscorr"
    height_tracked_cell = tracked_cell_matrix.shape[0]
    width_tracked_cell = tracked_cell_matrix.shape[1]
    height_search_cell = search_cell_matrix.shape[0]
    width_search_cell = search_cell_matrix.shape[1]
    best_correlation = 0
    for i in np.arange(np.ceil(height_tracked_cell/2), height_search_cell-np.ceil(height_tracked_cell/2)):
        for j in np.arange(np.ceil(width_tracked_cell/2), width_search_cell-np.ceil(width_tracked_cell/2)):
            search_subcell_matrix = get_submatrix_symmetric([i, j], tracked_cell_matrix.shape, search_cell_matrix)
            tracked_vector = tracked_cell_matrix.flatten()
            search_subcell_vector = search_subcell_matrix.flatten()
            if method == "corrcoeff":
                corr = np.corrcoef(tracked_vector, search_subcell_vector)[0, 1]
            if method == "crosscorr":
                tracked_vector = tracked_vector/np.linalg.norm(tracked_vector)
                search_subcell_vector = search_subcell_vector/np.linalg.norm(search_subcell_vector)
                corr = np.correlate(tracked_vector, search_subcell_vector, mode='valid')
            if corr > best_correlation:
                best_correlation = corr
                best_correlation_coordinates = [i, j]
    print("Correlation:", best_correlation)
    best_correlation_coordinates = np.floor(np.subtract(best_correlation_coordinates, [search_cell_matrix.shape[0]/2,
                                                                                      search_cell_matrix.shape[1]/2]))
    return best_correlation_coordinates



