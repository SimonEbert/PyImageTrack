
import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd
import rasterio.mask
import rasterio.plot
import rasterio.coords
import numpy as np
import scipy.interpolate
import shapely
import numpy.linalg as la
import scipy.optimize as opt
from rasterio.coords import BoundingBox
from CreateGeometries.CreateGeometries import grid_points_on_polygon
from CreateGeometries.CreateGeometries import get_raster_indices_from_points
import scipy


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
    best_correlation_coordinates = [search_cell_matrix.shape[0]/2, search_cell_matrix.shape[1]/2]
    for i in np.arange(np.ceil(height_tracked_cell/2), height_search_cell-np.ceil(height_tracked_cell/2)):
        for j in np.arange(np.ceil(width_tracked_cell/2), width_search_cell-np.ceil(width_tracked_cell/2)):
            search_subcell_matrix = get_submatrix_symmetric([i, j], tracked_cell_matrix.shape, search_cell_matrix)
            tracked_vector = tracked_cell_matrix.flatten()
            search_subcell_vector = search_subcell_matrix.flatten()
            corr = 0
            if method == "corrcoeff":
                corr = np.corrcoef(tracked_vector, search_subcell_vector)[0, 1]
            # check if the tracked and search subcell vectors have any non-zero elements (to avoid dividing by zero)
            if (method == "crosscorr") and (np.any(tracked_vector)) and (np.any(search_subcell_vector)):
                tracked_vector = tracked_vector/np.linalg.norm(tracked_vector)
                search_subcell_vector = search_subcell_vector/np.linalg.norm(search_subcell_vector)
                corr = np.correlate(tracked_vector, search_subcell_vector, mode='valid')
            if corr > best_correlation:
                best_correlation = corr
                best_correlation_coordinates = [i, j]
    # print("Correlation:", best_correlation)
    best_correlation_coordinates = np.floor(np.subtract(best_correlation_coordinates, [search_cell_matrix.shape[0]/2,
                                                                                      search_cell_matrix.shape[1]/2]))
    return best_correlation_coordinates


def move_cell_rotation_approach(coefficients, tracked_cell_matrix, interpolator_search_cell, central_indices, indices):
    central_row, central_column = central_indices[0], central_indices[1]
    # rotation_angle, shift_rows, shift_columns = coefficients
    t1, t2, t3, t4, shift_rows, shift_columns = coefficients
    # rotation_matrix = np.array([[np.cos(rotation_angle), -np.sin(rotation_angle)],
    #                             [np.sin(rotation_angle), np.cos(rotation_angle)]])
    rotation_matrix = np.array([[t1, t2], [t3, t4]])
    shift_vector = np.repeat(np.array([[shift_rows], [shift_columns]]), tracked_cell_matrix.shape[0] ** 2, axis=1)
    # transformation_matrix = np.array([[t1, t2, shift_rows],
    #                                  [t3, t4, shift_columns],
    #                                  [0, 0, 1]])
    moved_indices = np.matmul(rotation_matrix, indices) + shift_vector # +
    #                  np.repeat(np.array([
    #                      [-central_row * np.cos(rotation_angle) + central_column * np.sin(
    #                          rotation_angle) + central_row],
    #                      [-central_row * np.sin(rotation_angle) - central_column * np.cos(
    #                          rotation_angle) + central_column]]),
    #                      tracked_cell_matrix.shape[0] ** 2, axis=1)
    #
    # moved_indices = np.matmul(transformation_matrix, np.vstack((indices, np.repeat(1, len(indices[0])))))

    # moved_indices = moved_indices[0:2, :]


    # try:
    moved_cell_matrix = interpolator_search_cell(moved_indices.T).reshape(tracked_cell_matrix.T.shape)
    # except:
    #     moved_cell_matrix = np.full(tracked_cell_matrix.shape, np.inf)
    return moved_cell_matrix


def lsm_loss_function_rotation(coefficients, tracked_cell_matrix: np.ndarray, interpolator_search_cell, central_indices, indices):
    moved_cell_matrix = move_cell_rotation_approach(coefficients, tracked_cell_matrix, interpolator_search_cell, central_indices, indices)
    # return (moved_cell_matrix - tracked_cell_matrix).flatten() # for least squares optimizing
    return np.sum((tracked_cell_matrix-moved_cell_matrix)**2)


def lsm_loss_function_rotation_test(coefficients, tracked_cell_matrix_embedded, search_cell_matrix):
    # print(coefficients)
    # t1, t2, t3, t4, shift_rows, shift_columns = coefficients
    # transformation_matrix = np.array([[t1, t2], [t3, t4]])
    # # moved_track_matrix = scipy.ndimage.shift(scipy.ndimage.rotate(tracked_cell_matrix_embedded, rotation_angle, reshape=False), shift=[shift_rows, shift_columns])
    # moved_track_matrix = scipy.ndimage.affine_transform(tracked_cell_matrix_embedded, transformation_matrix, offset=[shift_rows, shift_columns])
    # mask = (moved_track_matrix > 0.5)
    # masked_search_matrix = np.where(mask, search_cell_matrix, 0)
    # error = np.sum((masked_search_matrix-moved_track_matrix)**2)
    #
    # print(error)
    return





def track_cell_lsm(tracked_cell_matrix: np.ndarray, search_cell_matrix: np.ndarray):

    # Interpolator_tracked_cell = scipy.interpolate.RegularGridInterpolator(
    #     (np.arange(0, tracked_cell_matrix.shape[0]), np.arange(0, tracked_cell_matrix.shape[1])),
    #     tracked_cell_matrix)
    interpolator_search_cell = scipy.interpolate.RegularGridInterpolator(
        (np.arange(0, search_cell_matrix.shape[0]), np.arange(0, search_cell_matrix.shape[1])),
        search_cell_matrix, fill_value=None, bounds_error=False)
    if len(tracked_cell_matrix[0]) == 0:
        return [0,0]
    # assign indices in respect to indexing in the search cell matrix
    central_row = np.round(search_cell_matrix.shape[0] / 2)
    central_column = np.round(search_cell_matrix.shape[1] / 2)
    # tracked_cell_matrix_embedded = np.zeros(search_cell_matrix.shape)
    # tracked_cell_matrix_embedded[
    #     int(central_row-np.floor(tracked_cell_matrix.shape[0]/2)):int(np.ceil(central_row+tracked_cell_matrix.shape[0]/2)),
    #     int(central_column-np.floor(tracked_cell_matrix.shape[1]/2)):int(central_column+np.ceil(tracked_cell_matrix.shape[1]/2))] = tracked_cell_matrix

    indices = np.array(np.meshgrid(np.arange(np.ceil(central_row - tracked_cell_matrix.shape[0] / 2),
                                             np.ceil(central_row + tracked_cell_matrix.shape[0] / 2)),
                                   np.arange(np.ceil(central_column - tracked_cell_matrix.shape[1] / 2),
                                             np.ceil(central_column + tracked_cell_matrix.shape[1] / 2)))
                       ).T.reshape(-1, 2).T
    # solution_global = opt.dual_annealing(lsm_loss_function_rotation_test, x0=np.array([45, 1, 1]),
    #     args=(tracked_cell_matrix_embedded, search_cell_matrix, [central_row, central_column], indices),
    #     bounds=((-1/2*np.pi,1/2*np.pi), (-20, 40), (-20, 40)))  #  for rotating approach, SLSQP works well
    [initial_shift_rows, initial_shift_columns] = track_cell(tracked_cell_matrix, search_cell_matrix)

    solution_global = opt.minimize(lsm_loss_function_rotation, x0=np.array([1, 0, 0, 1, initial_shift_rows, initial_shift_columns]), method='Nelder-Mead',
                                             args=(tracked_cell_matrix, interpolator_search_cell, [central_row, central_column], indices), bounds=((None, None), (None, None), (None, None), (None, None), (None, None), (None, None)))
    # solution_global = opt.least_squares(lsm_loss_function_rotation, x0=np.array([1, 0, 0, 1, initial_shift_rows, initial_shift_columns]),
    #                                     args=(tracked_cell_matrix, interpolator_search_cell, [central_row, central_column], indices))
    t1, t2, t3, t4, shift_rows, shift_columns = solution_global.x
    # print(solution_global.success)
    if not solution_global.success:
        return[np.nan,np.nan]
    # print(shift_rows, shift_columns)
    # print("Error: ", lsm_loss_function_rotation([rotation, shift_rows, shift_columns], tracked_cell_matrix, interpolator_search_cell, [central_row, central_column],indices))
    # rasterio.plot.show(tracked_cell_matrix, title=[np.round(shift_rows), np.round(shift_columns)])
    # rasterio.plot.show(move_cell_rotation_approach([rotation, shift_rows, shift_columns], tracked_cell_matrix, interpolator_search_cell, [central_row, central_column], indices))

    return [shift_rows, shift_columns]

    # rotation_angle, shift_rows, shift_columns = solution.x
    # rotation_matrix = np.array([[np.cos(rotation_angle), -np.sin(rotation_angle)],
    #                             [np.sin(rotation_angle), np.cos(rotation_angle)]])
    # shift_vector = np.repeat(np.array([[shift_rows], [shift_columns]]), tracked_cell_matrix.shape[0] ** 2, axis=1)
    # moved_indices = (np.matmul(rotation_matrix, indices) + shift_vector +
    #                  np.repeat(np.array([
    #                      [-central_row * np.cos(rotation_angle) + central_column * np.sin(
    #                          rotation_angle) + central_row],
    #                      [-central_row * np.sin(rotation_angle) - central_column * np.cos(
    #                          rotation_angle) + central_column]]),
    #                      tracked_cell_matrix.shape[0] ** 2, axis=1)
    #                  )
    # moved_cell_matrix = interpolator_search_cell(moved_indices.T).reshape(tracked_cell_matrix.shape)
    # rasterio.plot.show(tracked_cell_matrix)
    # rasterio.plot.show(moved_cell_matrix)


def remove_outlying_tracked_pixels(tracked_pixels):
    for index in np.arange(len(tracked_pixels)):
        # row = tracked_pixels.iloc[index]["row"]
        # column = tracked_pixels.iloc[index]["column"]
        #
        # smaller_row_index = 0
        # bigger_row_index = 0
        # smaller_column_index = 0
        # bigger_column_index = 0
        #
        #
        # print(index)
        # if row != min(tracked_pixels["row"]):
        #     smaller_row_index = max(tracked_pixels.loc[tracked_pixels["row"] < row, "row"])
        # if row != max(tracked_pixels["row"]):
        #     bigger_row_index = min(tracked_pixels.loc[tracked_pixels["row"] > row, "row"])
        # if column != min(tracked_pixels["column"]):
        #     smaller_column_index = max(tracked_pixels.loc[tracked_pixels["column"] < column, "column"])
        # if column != max(tracked_pixels["column"]):
        #     bigger_column_index = min(tracked_pixels.loc[tracked_pixels["column"] > column, "column"])
        # outlier = True
        # for i in [smaller_row_index, row, bigger_row_index]:
        #     for j in [smaller_column_index, column, bigger_column_index]:
        #         print(i, j)
        #         if int(tracked_pixels.loc[tracked_pixels["row"] == i][tracked_pixels["column"] == j][["movement_row_direction"]].to_numpy()) <= int(2*tracked_pixels.loc[tracked_pixels["row"] == i][tracked_pixels["column"] == j][["movement_row_direction"]].to_numpy()):
        #             outlier = False
        #             break
        #         if int(tracked_pixels.loc[tracked_pixels["row"] == i][tracked_pixels["column"] == j][["movement_column_direction"]].to_numpy()) <= int(2*tracked_pixels.loc[tracked_pixels["row"] == i][tracked_pixels["column"] == j][["movement_column_direction"]].to_numpy()):
        #             outlier = False
        #             break
        # if outlier:
        #     tracked_pixels[index]["movement_row_direction"] = 0
        #     tracked_pixels[index]["movement_column_direction"] = 0
        tracked_pixels = tracked_pixels.assign(movement_distance=
                                               np.linalg.norm(tracked_pixels.loc[:, ["movement_row_direction",
                                                                                     "movement_column_direction"]],
                                                              axis=1))
        higher_index = min(len(tracked_pixels)-1, index + 1)
        lower_index = max(0, index - 1)
        if tracked_pixels.iloc[index]["movement_distance"] > 8*max(tracked_pixels.iloc[lower_index]["movement_distance"],
                                                                   tracked_pixels.iloc[higher_index]["movement_distance"]):
            tracked_pixels.at[index, "movement_distance"] = 0
            tracked_pixels.at[index, "movement_row_direction"] = 0
            tracked_pixels.at[index, "movement_column_direction"] = 0
    return tracked_pixels


def track_movement(image1, image2, tracking_area: gpd.geodataframe, number_of_tracked_points: int, cell_size: int = 10,
                   tracking_area_size: int = 50, remove_outliers: bool = True):
    """Creates roughly number_of_tracked_points points on the tracking area and  tracks these points between image1 and
    image2. The size of the cell, which is tracked for each point as well as the search area can be adjusted."""

    points = grid_points_on_polygon(polygon=tracking_area, number_of_points=number_of_tracked_points)

    [image1_matrix, image1_transform], [image2_matrix, image2_transform] = get_overlapping_area(image1, image2)

    rows, cols = get_raster_indices_from_points(points, image2_transform)
    tracked_points_pixels = np.array([rows, cols]).transpose()

    tracked_pixels = pd.DataFrame()
    for central_index in tracked_points_pixels:
        # if ((len(tracked_pixels)+1) % 50 == 0):
        print("Starting to track pixel ", len(tracked_pixels)+1, " of ", len(tracked_points_pixels))

        track_cell1 = get_submatrix_symmetric(central_index=central_index, shape=(cell_size, cell_size),
                                              matrix=image1_matrix)
        if track_cell1.any() == 0:
            print("Skipping this cell for unavailable data")
            tracked_pixels = pd.concat([tracked_pixels, pd.DataFrame({"row": central_index[0],
                                                                      "column": central_index[1],
                                                                      "movement_row_direction": np.nan,
                                                                      "movement_column_direction": np.nan},
                                                                     index=[len(tracked_pixels)])])
            continue
        search_area2 = get_submatrix_symmetric(central_index=central_index,
                                               shape=(tracking_area_size, tracking_area_size),
                                               matrix=image2_matrix)
        match = track_cell_lsm(track_cell1, search_area2)

        tracked_pixels = pd.concat([tracked_pixels, pd.DataFrame({"row": central_index[0],
                                                                  "column": central_index[1],
                                                                  "movement_row_direction": match[0],
                                                                  "movement_column_direction": match[1]},
                                                                 index=[len(tracked_pixels)])])
    if remove_outliers:
        tracked_pixels = remove_outlying_tracked_pixels(tracked_pixels)
    return tracked_pixels

