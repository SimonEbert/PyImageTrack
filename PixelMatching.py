
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
from rasterio import features

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
    array_file1[array_file2 == 0] = 0
    array_file2[array_file1 == 0] = 0
    return [array_file1, array_file1_transform], [array_file2, array_file2_transform]


def get_submatrix_symmetric(central_index, shape, matrix):
    """Makes even given shapes one number smaller to ascertain there is one unique central index"""
    # matrix is threedimensional if there are several channels
    if len(matrix.shape) == 3:
        submatrix = matrix[:,
                           int(central_index[0]-np.ceil(shape[0]/2))+1:int(central_index[0]+np.ceil(shape[0]/2)),
                           int(central_index[1]-np.ceil(shape[1]/2))+1:int(central_index[1]+np.ceil(shape[1]/2))]
    else:
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
    height_tracked_cell = tracked_cell_matrix.shape[-2]
    width_tracked_cell = tracked_cell_matrix.shape[-1]
    height_search_cell = search_cell_matrix.shape[-2]
    width_search_cell = search_cell_matrix.shape[-1]
    best_correlation = 0
    best_correlation_coordinates = [search_cell_matrix.shape[-2]/2, search_cell_matrix.shape[-1]/2]
    tracked_vector = tracked_cell_matrix.flatten()
    tracked_vector = tracked_vector/np.linalg.norm(tracked_vector)


    for i in np.arange(np.ceil(height_tracked_cell/2), height_search_cell-np.ceil(height_tracked_cell/2)):
        for j in np.arange(np.ceil(width_tracked_cell/2), width_search_cell-np.ceil(width_tracked_cell/2)):
            search_subcell_matrix = get_submatrix_symmetric([i, j], (tracked_cell_matrix.shape[-2], tracked_cell_matrix.shape[-1]), search_cell_matrix)
            search_subcell_vector = search_subcell_matrix.flatten()
            search_subcell_vector = search_subcell_vector/np.linalg.norm(search_subcell_vector)


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
    best_correlation_coordinates = np.floor(np.subtract(best_correlation_coordinates, [search_cell_matrix.shape[-2]/2,
                                                                                      search_cell_matrix.shape[-1]/2]))
    return best_correlation_coordinates

# WE DON'T NEED THE WHOLE TRACKED MATRIX; JUST ITS SHAPE
def move_cell_rotation_approach(coefficients, tracked_cell_matrix, interpolator_search_cell, central_indices, indices):
    """Don't need central_indices"""
    if len(tracked_cell_matrix.shape) == 2:
        central_row, central_column = central_indices[0], central_indices[1]
        t1, t2, t3, t4, shift_rows, shift_columns = coefficients

        rotation_matrix = np.array([[t1, t2], [t3, t4]])
        # shift_vector = np.repeat(np.array([[shift_rows], [shift_columns]]), tracked_cell_matrix.shape[0] ** 2, axis=1) # ALIGN CHANGE
        shift_vector = np.repeat(np.array([[shift_rows], [shift_columns]]), tracked_cell_matrix.shape[0]*tracked_cell_matrix.shape[1], axis=1)

        moved_indices = np.matmul(rotation_matrix, indices) + shift_vector # +

        try:
            moved_cell_matrix = interpolator_search_cell(moved_indices.T).reshape(tracked_cell_matrix.shape)
        except:
            moved_cell_matrix = np.full(tracked_cell_matrix.shape, np.inf)
        return moved_cell_matrix
    else:
        t1, t2, t3, t4, shift_rows, shift_columns = coefficients

        rotation_matrix = np.array([[t1, t2], [t3, t4]])
        # shift_vector = np.repeat(np.array([[shift_rows], [shift_columns]]), tracked_cell_matrix.shape[0] ** 2, axis=1) # ALIGN CHANGE
        shift_vector = np.repeat(np.array([[shift_rows], [shift_columns]]),
                                 tracked_cell_matrix.shape[-2] * tracked_cell_matrix.shape[-1], axis=1)

        moved_indices = np.matmul(rotation_matrix, indices) + shift_vector
        try:
            moved_cell_matrix = np.full(tracked_cell_matrix.shape, 0)
            for band in range(tracked_cell_matrix.shape[0]):
                moved_cell_matrix[band, :, :] = interpolator_search_cell[band](moved_indices.T).reshape((tracked_cell_matrix.shape[-2], tracked_cell_matrix.shape[-1]))
        except:
            moved_cell_matrix = np.full(tracked_cell_matrix.shape, np.nan)
        return moved_cell_matrix

def lsm_loss_function_rotation(coefficients, tracked_cell_matrix: np.ndarray, interpolator_search_cell, central_indices, indices):
    moved_cell_matrix = move_cell_rotation_approach(coefficients, tracked_cell_matrix, interpolator_search_cell, central_indices, indices)
    return np.sum((tracked_cell_matrix-moved_cell_matrix)**2)







def track_cell_lsm(tracked_cell_matrix: np.ndarray, search_cell_matrix: np.ndarray, initial_shift_values = None):
    """Assume matrices to be quadratic"""
    if len(tracked_cell_matrix.shape) == 2: # For one channel images
        interpolator_search_cell = scipy.interpolate.RegularGridInterpolator(
            (np.arange(0, search_cell_matrix.shape[0]), np.arange(0, search_cell_matrix.shape[1])),
            search_cell_matrix, fill_value=None, bounds_error=False)
        if len(tracked_cell_matrix[0]) == 0:
            return [0, 0]
        # assign indices in respect to indexing in the search cell matrix
        central_row = np.round(search_cell_matrix.shape[0] / 2)
        central_column = np.round(search_cell_matrix.shape[1] / 2)


        indices = np.array(np.meshgrid(np.arange(np.ceil(central_row - tracked_cell_matrix.shape[0] / 2),
                                                 np.ceil(central_row + tracked_cell_matrix.shape[0] / 2)),
                                       np.arange(np.ceil(central_column - tracked_cell_matrix.shape[1] / 2),
                                                 np.ceil(central_column + tracked_cell_matrix.shape[1] / 2)))
                           ).T.reshape(-1, 2).T
        if initial_shift_values is None:
            [initial_shift_rows, initial_shift_columns] = track_cell(tracked_cell_matrix, search_cell_matrix)
        else:
            [initial_shift_rows, initial_shift_columns] = initial_shift_values

        solution_global = opt.minimize(lsm_loss_function_rotation, x0=np.array([1, 0, 0, 1, initial_shift_rows, initial_shift_columns]),
                                                 args=(tracked_cell_matrix, interpolator_search_cell, [central_row, central_column], indices), bounds=((None, None), (None, None), (None, None), (None, None), (None, None), (None, None))) # (initial_shift_rows -5, initial_shift_rows +5), (initial_shift_columns -5, initial_shift_columns +5)
        # solution_global = opt.least_squares(lsm_loss_function_rotation, x0=np.array([1, 0, 0, 1, initial_shift_rows, initial_shift_columns]),
        #                                     args=(tracked_cell_matrix, interpolator_search_cell, [central_row, central_column], indices))
        t1, t2, t3, t4, shift_rows, shift_columns = solution_global.x
        #What to do if the optimization did not work
        if not solution_global.success:
            return[np.nan, np.nan]
        transformation_matrix = np.array([[t1, t2], [t3, t4]])

        if np.abs(np.linalg.det(transformation_matrix) - 1) >= 0.2:
            print("Warning: Transformation matrix has unrealistic determinant: ", np.linalg.det(transformation_matrix))
            return [np.nan, np.nan]

        return [shift_rows, shift_columns]
    else: # MULTIBAND
        interpolator_search_cell_list = list()
        for band in range(tracked_cell_matrix.shape[0]):
            interpolator_search_cell_list.append(scipy.interpolate.RegularGridInterpolator(
                (np.arange(0, search_cell_matrix.shape[-2]), np.arange(0, search_cell_matrix.shape[-1])),
                search_cell_matrix[band, :, :], fill_value=None, bounds_error=False
            ))
            # assign indices in respect to indexing in the search cell matrix
        central_row = np.round(search_cell_matrix.shape[0] / 2)
        central_column = np.round(search_cell_matrix.shape[1] / 2)

        indices = np.array(np.meshgrid(np.arange(np.ceil(central_row - tracked_cell_matrix.shape[-2] / 2),
                                                     np.ceil(central_row + tracked_cell_matrix.shape[-2] / 2)),
                                           np.arange(np.ceil(central_column - tracked_cell_matrix.shape[-1] / 2),
                                                     np.ceil(central_column + tracked_cell_matrix.shape[-1] / 2)))
                               ).T.reshape(-1, 2).T
        if initial_shift_values is None:
                [initial_shift_rows, initial_shift_columns] = track_cell(tracked_cell_matrix, search_cell_matrix)
        else:
            [initial_shift_rows, initial_shift_columns] = initial_shift_values
        solution_global = opt.minimize(lsm_loss_function_rotation,
                                               x0=np.array([1, 0, 0, 1, initial_shift_rows, initial_shift_columns]),
                                               args=(tracked_cell_matrix, interpolator_search_cell_list,
                                                     [central_row, central_column], indices), bounds=(
                    (None, None), (None, None), (None, None), (None, None), (None, None), (None, None)))
        t1, t2, t3, t4, shift_rows, shift_columns = solution_global.x
        # What to do if the optimization did not work
        if not solution_global.success:
            return [np.nan, np.nan]
        # Check if the transformation matrix is singular
        transformation_matrix = np.array([[t1, t2], [t3, t4]])
        if np.abs(np.linalg.det(transformation_matrix) - 1) >= 0.2:
            print("Warning: Transformation matrix has unrealistic determinant: ", np.linalg.det(transformation_matrix))
            return [np.nan, np.nan]

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


def get_tracked_pixels_square(tracked_pixels, central_pixel_coordinates):
    central_pixel_row = central_pixel_coordinates[0]
    central_pixel_col = central_pixel_coordinates[1]
    neighbouring_tracked_pixels = pd.DataFrame()
    smaller_row_index = np.max(tracked_pixels[tracked_pixels["row"] < central_pixel_row]["row"])
    bigger_row_index = np.min(tracked_pixels[tracked_pixels["row"] > central_pixel_row]["row"])
    smaller_col_index = np.max(tracked_pixels[tracked_pixels["column"] < central_pixel_col]["column"])
    bigger_col_index = np.min(tracked_pixels[tracked_pixels["column"] > central_pixel_col]["column"])
    for row in [smaller_row_index, central_pixel_row, bigger_row_index]:
        for column in [smaller_col_index, central_pixel_col, bigger_col_index]:
            neighbouring_tracked_pixels = pd.concat([neighbouring_tracked_pixels, tracked_pixels.loc[(tracked_pixels["row"] == row) & (tracked_pixels["column"] == column)]])
    neighbouring_tracked_pixels.dropna(inplace=True, how="any")
    return neighbouring_tracked_pixels


def remove_outlying_tracked_pixels(tracked_pixels, only_rotation = False):
    for [row, col] in zip(tracked_pixels["row"].tolist(), tracked_pixels["column"].tolist()):
        neighbouring_pixels = get_tracked_pixels_square(tracked_pixels, [row, col])
        central_pixel = tracked_pixels[(tracked_pixels["row"] == row) & (tracked_pixels["column"] == col)]
        neighbouring_pixels = neighbouring_pixels[(neighbouring_pixels["row"] != row) | (neighbouring_pixels["column"] != col)]
        # print(np.abs(neighbouring_pixels["movement_row_direction"]-central_pixel["movement_row_direction"]))
        # print((np.abs(neighbouring_pixels["movement_row_direction"]-central_pixel["movement_row_direction"]).any() >= 15))
        # Check if more than three neigbouring pixels have a more than 5 pixel difference movement
        # if np.sum(np.abs(neighbouring_pixels["movement_row_direction"].values-central_pixel["movement_row_direction"].values) >= 5) > 3 | (np.abs(neighbouring_pixels["movement_row_direction"].values-central_pixel["movement_row_direction"].values) >= 12).any():
        #     tracked_pixels.loc[(tracked_pixels["row"] == row) & (tracked_pixels["column"] == col), ["movement_row_direction", "movement_column_direction"]] = np.nan
        # if np.sum(np.abs(neighbouring_pixels["movement_column_direction"].values-central_pixel["movement_column_direction"].values) >= 5) > 3 | (np.abs(neighbouring_pixels["movement_column_direction"].values-central_pixel["movement_column_direction"].values) >= 12).any():
        #     tracked_pixels.loc[(tracked_pixels["row"] == row) & (tracked_pixels["column"] == col), ["movement_row_direction", "movement_column_direction"]] = np.nan

        average_movement_angle = 0

        for i in range(len(neighbouring_pixels)):
            movement_vector = np.array([neighbouring_pixels.iloc[i]["movement_row_direction"], neighbouring_pixels.iloc[i]["movement_column_direction"]])
            movement_vector_length = np.linalg.norm(movement_vector)
            average_movement_angle += np.arccos(np.dot(movement_vector, np.array([1,0]))/movement_vector_length)
        if len(neighbouring_pixels) > 0:
            average_movement_angle /= len(neighbouring_pixels)
        movement_vector_central = np.array([central_pixel["movement_row_direction"],
                                   central_pixel["movement_column_direction"]])

        movement_vector_central_length = np.linalg.norm(movement_vector_central)
        movement_angle_central = np.arccos(
            np.dot(movement_vector_central.T, np.array([1, 0])) / movement_vector_central_length)
        if np.abs(average_movement_angle - movement_angle_central) >= np.pi/6:
            print("removed pixel", row, col, "for rotation reasons")
            tracked_pixels.loc[(tracked_pixels["row"] == row) & (tracked_pixels["column"] == col), ["movement_row_direction", "movement_column_direction", "movement_distance"]] = np.nan

        # if (np.sum(np.abs(neighbouring_pixels["movement_distance"].values-central_pixel["movement_distance"].values) >= 15) > 3) | ((np.abs(neighbouring_pixels["movement_distance"].values-central_pixel["movement_distance"].values) >= 20).any()) | (all(1.2*x <= central_pixel["movement_distance"].values for x in neighbouring_pixels["movement_distance"].tolist())) | (np.sum(np.isnan(neighbouring_pixels["movement_distance"])) >= 7):
        #     print("removed pixel", row, col)
        #
        #     tracked_pixels.loc[(tracked_pixels["row"] == row) & (tracked_pixels["column"] == col), ["movement_row_direction", "movement_column_direction", "movement_distance"]] = np.nan
    return tracked_pixels



def retrack_wrong_matching_pixels(tracked_pixels, track_matrix, search_matrix, cell_size, tracking_area_size, fallback_on_cross_correlation = False):
    for [row, col] in zip(tracked_pixels["row"].tolist(), tracked_pixels["column"].tolist()):
        central_pixel = tracked_pixels[(tracked_pixels["row"] == row) & (tracked_pixels["column"] == col)]
        if np.isnan(central_pixel["movement_row_direction"].values) & np.isnan(central_pixel["movement_column_direction"].values):
            neighbouring_pixels = get_tracked_pixels_square(tracked_pixels, [row, col])
            neighbouring_pixels = neighbouring_pixels[(neighbouring_pixels["row"] != row) | (neighbouring_pixels["column"] != col)]
            movement_row_direction_neighbours_mean = np.nanmean(neighbouring_pixels["movement_row_direction"].values)
            movement_column_direction_neighbours_mean = np.nanmean(neighbouring_pixels["movement_column_direction"].values)
            track_cell1 = get_submatrix_symmetric(central_index=[row, col], shape=(cell_size, cell_size),
                                                  matrix=track_matrix)

            search_area2 = get_submatrix_symmetric(central_index=[row, col],
                                                   shape=(tracking_area_size, tracking_area_size),
                                                   matrix=search_matrix)
            match = track_cell_lsm(track_cell1, search_area2, initial_shift_values=[movement_row_direction_neighbours_mean, movement_column_direction_neighbours_mean])
            if np.isnan(match[0]) & fallback_on_cross_correlation:
                print("Falling back on cross correlation for pixel", [row, col])
                match = track_cell(track_cell1, search_area2)
            tracked_pixels[(tracked_pixels["row"] == row) & (tracked_pixels["column"] == col)] = [row, col, match[0], match[1], np.linalg.norm([match[0], match[1]])]
    return tracked_pixels

def track_movement(image1_matrix, image2_matrix, image_transform, tracking_area: gpd.geodataframe, number_of_tracked_points: int, cell_size: int = 40,
                   tracking_area_size: int = 50, tracking_method: str = "lsm", remove_outliers: bool = True, retry_matching: bool = True):
    # 20 PIXELS CELL SIZE LEADS TO NON-ROBUST RESULTS (IT'S JUST TO SMALL FOR RECOGNIZING PATTERNS)
    """Creates roughly number_of_tracked_points points on the tracking area and  tracks these points between image1 and
    image2. The size of the cell, which is tracked for each point as well as the search area can be adjusted. Takes two aligned matrices image1 and image 2 and the respective transform"""

    points = grid_points_on_polygon(polygon=tracking_area, number_of_points=number_of_tracked_points)


    rows, cols = get_raster_indices_from_points(points, image_transform)
    tracked_points_pixels = np.array([rows, cols]).transpose()

    tracked_pixels = pd.DataFrame()
    for central_index in tracked_points_pixels:
        # if ((len(tracked_pixels)+1) % 50 == 0):
        print("Starting to track pixel ", len(tracked_pixels)+1, " of ", len(tracked_points_pixels), ": ", central_index)

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

        if tracking_method == "lsm":#MULTIBANDCHANGE
            match = track_cell_lsm(track_cell1, search_area2)
        else:
            match = track_cell(track_cell1, search_area2)

        tracked_pixels = pd.concat([tracked_pixels, pd.DataFrame({"row": central_index[0],
                                                                  "column": central_index[1],
                                                                  "movement_row_direction": match[0],
                                                                  "movement_column_direction": match[1]},
                                                                 index=[len(tracked_pixels)])])
    tracked_pixels.insert(4, "movement_distance",
                  np.linalg.norm(tracked_pixels.loc[:, ["movement_row_direction", "movement_column_direction"]], axis=1))

    if remove_outliers:
        tracked_pixels = remove_outlying_tracked_pixels(tracked_pixels, only_rotation=True)

    if retry_matching & (tracking_method == "lsm"):
        tracked_pixels = retrack_wrong_matching_pixels(tracked_pixels, image1_matrix, image2_matrix, cell_size, tracking_area_size, fallback_on_cross_correlation=False)
    if remove_outliers:
        tracked_pixels = remove_outlying_tracked_pixels(tracked_pixels)

    return tracked_pixels


def align_images(image1, image2, reference_area: gpd.GeoDataFrame, number_of_control_points: int, cell_size: int = 40,
                 tracking_area_size: int = 60, select_bands=None, image_alignment_via_lsm=False):
    """Assume the transform of the first matrix to be correct ("adjust" the second transform)"""


    [image1_matrix, image_transform], [image2_matrix, _] = get_overlapping_area(image1, image2)


    if select_bands is None:
        select_bands = [0, 1, 2]
    if len(image1_matrix.shape) == 3:
        image1_matrix = image1_matrix[select_bands, :, :]
        image2_matrix = image2_matrix[select_bands, :, :]
    # control_points = grid_points_on_polygon(polygon=reference_area, number_of_points=number_of_control_points)
    # rows, cols = get_raster_indices_from_points(control_points, image1_transform)
    # control_points_pixels = np.array([rows, cols]).transpose()
    # row_movements = 0
    # column_movements = 0
    # for central_index in control_points:
    #     track_cell1 = get_submatrix_symmetric(central_index=central_index, shape=(cell_size, cell_size),
    #                                           matrix=image1_matrix)
    #     search_area2 = get_submatrix_symmetric(central_index=central_index,
    #                                            shape=(tracking_area_size, tracking_area_size),
    #                                            matrix=image2_matrix)
    #     match = track_cell_lsm(track_cell1, search_area2)
    #     row_movements += match[0]
    #     column_movements += match[1]
    # row_movements /= number_of_control_points
    # column_movements /= number_of_control_points

    tracked_control_pixels = track_movement(image1_matrix, image2_matrix, image_transform, tracking_area=reference_area, number_of_tracked_points=number_of_control_points, tracking_method="crosscorr", cell_size=cell_size, tracking_area_size=tracking_area_size, remove_outliers=False)
    row_movements = np.nanmean(tracked_control_pixels["movement_row_direction"])
    column_movements = np.nanmean(tracked_control_pixels["movement_column_direction"])

    # global_match = track_cell_lsm(image2_matrix, image1_matrix, initial_shift_values=[row_movements, column_movements])
    # print("Shifted second matrix by following number (rows/columns) to match first matrix: ", row_movements, column_movements)
    if len(image1_matrix.shape) == 2:
        interpolator_image2_matrix = scipy.interpolate.RegularGridInterpolator(
        (np.arange(0, image2_matrix.shape[0]), np.arange(0, image2_matrix.shape[1])),
        image2_matrix, fill_value=None, bounds_error=False)
    else:
        interpolator_image2_matrix = list()
        for band in range(image1_matrix.shape[0]):
            interpolator_image2_matrix.append(scipy.interpolate.RegularGridInterpolator(
                (np.arange(0, image2_matrix.shape[-2]), np.arange(0, image2_matrix.shape[-1])),
                image2_matrix[band, :, :], fill_value=None, bounds_error=False
            ))
    central_row = np.round(image2_matrix.shape[-2] / 2)
    central_column = np.round(image2_matrix.shape[-1] / 2)
    indices = np.array(np.meshgrid(np.arange(np.ceil(central_row - image1_matrix.shape[-2] / 2),
                                             np.ceil(central_row + image1_matrix.shape[-2] / 2)),
                                   np.arange(np.ceil(central_column - image1_matrix.shape[-1] / 2),
                                             np.ceil(central_column + image1_matrix.shape[-1] / 2)))
                       ).T.reshape(-1, 2).T


    if image_alignment_via_lsm:
        inverse_transform = (~image_transform).to_shapely()
        transformation_matrix = np.array(inverse_transform)
        transformed_polygon = reference_area
        transformed_polygon["geometry"] = shapely.affinity.affine_transform(reference_area.loc[0]["geometry"], transformation_matrix)

        mask_matrix = rasterio.features.rasterize([reference_area.loc[0]["geometry"]], out_shape=image1_matrix.shape[-2:])

        mask_matrix = -mask_matrix+1
        if len(image1_matrix.shape) == 3:
            # mask_matrix = np.expand_dims(mask_matrix, 0)
            # mask_matrix = np.repeat(mask_matrix, image1_matrix.shape[0], axis=0)
            # print(mask_matrix.shape)
            masked_matrix1 = np.zeros(image1_matrix.shape)
            masked_matrix2 = np.zeros(image1_matrix.shape)

            for i in range(image1_matrix.shape[0]):
                masked_matrix1[i, :, :] = np.ma.masked_array(image1_matrix[i, :, :], mask=mask_matrix)
                masked_matrix2[i, :, :] = np.ma.masked_array(image2_matrix[i, :, :], mask=mask_matrix)

        else:
            masked_matrix1 = np.ma.masked_array(image1_matrix, mask=mask_matrix)
            masked_matrix2 = np.ma.masked_array(image2_matrix, mask=mask_matrix)

        rasterio.plot.show(masked_matrix1)
        rasterio.plot.show(masked_matrix2)

        matching = track_cell_lsm(masked_matrix1, masked_matrix2, initial_shift_values=[row_movements, column_movements])
        row_movements, column_movements = matching
        print(matching)


    new_matrix2 = move_cell_rotation_approach([1,0,0,1, row_movements, column_movements], image1_matrix, interpolator_image2_matrix, [central_row, central_column], indices)

    return [image1_matrix, new_matrix2, image_transform]









