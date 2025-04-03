import concurrent.futures
import os

import geopandas as gpd
import pandas as pd
import rasterio.mask
import rasterio.plot
import numpy as np
import scipy.interpolate
import shapely
import scipy.optimize as opt
import multiprocessing
from functools import partial

from CreateGeometries.HandleGeometries import grid_points_on_polygon_by_number_of_points
from CreateGeometries.HandleGeometries import grid_points_on_polygon_by_distance
from CreateGeometries.HandleGeometries import get_raster_indices_from_points
from rasterio import features
from CreateGeometries.HandleGeometries import get_overlapping_area


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
    # central_index = central_index.astype(float)#  np.array([float(central_index[0]), float(central_index[1])]
    # matrix is three-dimensional if there are several channels
    if len(matrix.shape) == 3:
        submatrix = matrix[:,
                    int(central_index[0] - np.ceil(shape[0] / 2)) + 1:int(central_index[0] + np.ceil(shape[0] / 2)),
                    int(central_index[1] - np.ceil(shape[1] / 2)) + 1:int(central_index[1] + np.ceil(shape[1] / 2))]
    else:
        submatrix = matrix[
                    int(central_index[0] - np.ceil(shape[0] / 2)) + 1:int(central_index[0] + np.ceil(shape[0] / 2)),
                    int(central_index[1] - np.ceil(shape[1] / 2)) + 1:int(central_index[1] + np.ceil(shape[1] / 2))]
    return submatrix


def track_cell(tracked_cell_matrix: np.ndarray, search_cell_matrix: np.ndarray):
    """
    Calculates the movement of an image section using the cross-correlation approach. 
    Parameters
    ----------
    tracked_cell_matrix: np.ndarray
        An array (a section of the first image), which is compared to sections of the search_cell_matrix (a section of
        the second image).
    search_cell_matrix: np.ndarray
        An array, which delimits the area in which possible matching image sections are searched.
    Returns
    ----------
    movement_for_best_correlation: A two-element list, giving the movement rates in x- and y-direction respectively.
    """
    
    height_tracked_cell = tracked_cell_matrix.shape[-2]
    width_tracked_cell = tracked_cell_matrix.shape[-1]
    height_search_cell = search_cell_matrix.shape[-2]
    width_search_cell = search_cell_matrix.shape[-1]
    best_correlation = 0
    # for multichannel images, flattening ensures that always the same band is being compared
    tracked_vector = tracked_cell_matrix.flatten()
    # normalize the tracked vector
    tracked_vector = tracked_vector - np.mean(tracked_vector)
    tracked_vector = tracked_vector / np.linalg.norm(tracked_vector)
    for i in np.arange(np.ceil(height_tracked_cell / 2), height_search_cell - np.ceil(height_tracked_cell / 2)):
        for j in np.arange(np.ceil(width_tracked_cell / 2), width_search_cell - np.ceil(width_tracked_cell / 2)):
            search_subcell_matrix = get_submatrix_symmetric([i, j],
                                                            (tracked_cell_matrix.shape[-2],
                                                             tracked_cell_matrix.shape[-1]),
                                                            search_cell_matrix)
            # flatten the comparison cell matrix
            search_subcell_vector = search_subcell_matrix.flatten()

            # initialize correlation for the current central pixel (i,j)
            corr = 0
            # check if the search subcell vectors has any non-zero elements (to avoid dividing by zero)
            if (np.any(search_subcell_vector)):
                # normalize search_subcell vector
                search_subcell_vector = search_subcell_vector - np.mean(search_subcell_vector)
                search_subcell_vector = search_subcell_vector / np.linalg.norm(search_subcell_vector)
                corr = np.correlate(tracked_vector, search_subcell_vector, mode='valid')
            if corr > best_correlation:
                best_correlation = corr
                best_correlation_coordinates = [i, j]
    if best_correlation < 0.5:
        return [np.nan, np.nan]
    movement_for_best_correlation = np.floor(np.subtract(best_correlation_coordinates,
                                                         [search_cell_matrix.shape[-2] / 2,
                                                          search_cell_matrix.shape[-1] / 2]))

    return movement_for_best_correlation


def move_cell_rotation_approach(coefficients, tracked_cell_matrix_shape, interpolator_search_cell, indices):
    
    """
    For given transformation coefficients and indices, returns the section of the search_cell with size given by
    tracked_cell_matrix_shape, obtained by applying the transformation to the indices.
    Parameters
    ----------
    coefficients :
        A list of six integers [t1,t2,t3,t4,b1,b2] specifying the transformation parameters as in track_cell_lsm.
    tracked_cell_matrix_shape :
        The shape of the tracked image section (from the first image). This is used to get a section of equal size from
        the second image, using the interpolator_search_cell.
    interpolator_search_cell :
        The interpolator for the search cell from the second image, used to obtain values for non-integer results of the
        application of the affine transform to the indices. As in lsm_loss_function_rotation, this has to be an object
        f the class RegularGridInterpolator.
    indices :
        A 2-dimensional array, with two rows and number of columns equal to the number of entries in
        tracked_cell_matrix, containing the indices of the pixels of tracked_cell_matrix expressed in terms of the
        search cell matrix. The affine transformation given by the coefficients is applied to these indices and the
        interpolated search cell is evaluated at the transformed indices.
    Returns
    ----------
    moved_cell_matrix: A numpy array of the shape given by tracked_cell_matrix_shape, which represents the section of
    the second image, moved according to the affine transform coefficients with respect to the search cell.
    """
    
    # for single-channel images
    if len(tracked_cell_matrix_shape) == 2:
        # central_row, central_column = central_indices[0], central_indices[1]
        t1, t2, t3, t4, shift_rows, shift_columns = coefficients

        rotation_matrix = np.array([[t1, t2], [t3, t4]])

        # repeat the shift vector for adequate addition of index vectors
        shift_vector = np.repeat(np.array([[shift_rows], [shift_columns]]),
                                 tracked_cell_matrix_shape[0] * tracked_cell_matrix_shape[1], axis=1)

        moved_indices = np.matmul(rotation_matrix, indices) + shift_vector

        # try to access interpolated values from the search cell
        try:
            moved_cell_matrix = interpolator_search_cell(moved_indices.T).reshape(tracked_cell_matrix_shape)
        except:
            moved_cell_matrix = np.full(tracked_cell_matrix_shape, np.inf)
        return moved_cell_matrix
    # for multi-channel images
    else:
        t1, t2, t3, t4, shift_rows, shift_columns = coefficients
        rotation_matrix = np.array([[t1, t2], [t3, t4]])

        # repeat the shift vector for adequate addition of index vectors
        shift_vector = np.repeat(np.array([[shift_rows], [shift_columns]]),
                                 tracked_cell_matrix_shape[-2] * tracked_cell_matrix_shape[-1], axis=1)

        moved_indices = np.matmul(rotation_matrix, indices) + shift_vector

        # try to access interpolated values from the search cell
        try:
            moved_cell_matrix = np.full(tracked_cell_matrix_shape, 0)
            # fill the moved matrix channel-wise
            for band in range(tracked_cell_matrix_shape[0]):
                moved_cell_matrix[band, :, :] = interpolator_search_cell[band](moved_indices.T).reshape(
                    (tracked_cell_matrix_shape[-2], tracked_cell_matrix_shape[-1]))
        except:
            moved_cell_matrix = np.full(tracked_cell_matrix_shape, np.nan)
        return moved_cell_matrix


def lsm_loss_function_rotation(coefficients, tracked_cell_matrix: np.ndarray, interpolator_search_cell, indices):
    
    """
    Calculates the loss of the optimization problem in the least-squares approach for given coefficients.
    Parameters
    ----------
    coefficients
        A list of six integers [t1,t2,t3,t4,b1,b2] specifying the transformation parameters as in track_cell_lsm and
        the least-squares approach.
    tracked_cell_matrix : np.ndarray
        The given image section from the first image to be compared to an image section of the second image, which is
        given by interpolator_search_cell.
    interpolator_search_cell :
        The interpolator for the section from the second image. An object of the class RegularGridInterpolator from
        scipy.interpolate. For details see move_cell_rotation_approach.
    central_indices
        The index of the centre pixel of the tracked image cell, expressed in terms of matrix indices of the search cell
        matrix (from the second image). For details see move_cell_rotation_approach.
    indices
        A 2-dimensional array, with two rows and number of columns equal to the number of entries in
        tracked_cell_matrix, containing the indices of the pixels of tracked_cell_matrix expressed in terms of the
        search cell matrix. For details see move_cell_rotation_approach.
    Returns
    ----------
    loss: A float, giving the loss for the provided transformation parameters
    """
    
    moved_cell_matrix = move_cell_rotation_approach(coefficients=coefficients,
                                                    tracked_cell_matrix_shape=tracked_cell_matrix.shape,
                                                    interpolator_search_cell=interpolator_search_cell,
                                                    indices=indices)
    return np.sum((tracked_cell_matrix - moved_cell_matrix) ** 2)


def track_cell_lsm_powell(tracked_cell_matrix: np.ndarray, search_cell_matrix: np.ndarray, initial_shift_values=None,
                          return_full_coefficients: bool = False, transformation_matrix_determinant_threshold: int = 0.2):
    
    """
    Calculates the transformation parameters for a given image cell to match optimally the corresponding section in the
    second image using the least-squares approach.
    Parameters
    ----------
    tracked_cell_matrix : np.ndarray
        The image section of the first image that will be matched to a part of the search image section.
    search_cell_matrix : np.ndarray
        The image section of the second, which is used as a search area for the image section from the first image. This
        array needs to be larger than the tracked_cell_matrix.
    initial_shift_values = None
        If None, the initial values for the shift between the images are being calculated from the cross-correlation
        approach using the function track_cell. Otherwise, the provided values will be used as initial shift values.
        The transformation matrix is always initialized as the identity matrix.
    return_full_coefficients : bool = False
        If True, returns the six parameters for the affine transformation, instead of the movement rates for the central
        pixel of tracked_cell_matrix.
    Returns
    ----------
    [shift_rows, shift_columns]: The y- and x-movement (in pixels) for the central pixel of tracked_cell_matrix.
    If return_full_coefficients == True, returns instead [t1,t2,t3,t4,b1,b2], where t1, t2, t3, t4 are the entries of
    the transformation matrix and b1, b2 are the entries of the shift vector. If the matching was not successful,
    return NaNs instead.
    """
    
    # For one channel images
    if len(tracked_cell_matrix.shape) == 2:

        interpolator_search_cell = scipy.interpolate.RegularGridInterpolator(
            (np.arange(0, search_cell_matrix.shape[0]), np.arange(0, search_cell_matrix.shape[1])),
            search_cell_matrix, fill_value=0, bounds_error=False)

        if len(tracked_cell_matrix[0]) == 0:
            return [np.nan, np.nan]
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
            if np.isnan(initial_shift_rows):
                return [np.nan, np.nan]
        else:
            [initial_shift_rows, initial_shift_columns] = initial_shift_values
        solution_global = opt.minimize(lsm_loss_function_rotation,
                                       x0=np.array([1, 0, 0, 1, initial_shift_rows, initial_shift_columns]),
                                       args=(tracked_cell_matrix, interpolator_search_cell, indices),
                                       method="Powell"
                                       #, bounds=((None, None), (None, None), (None, None), (None, None),
                                       #                         (initial_shift_rows-1, initial_shift_rows+1),
                                       #                         (initial_shift_columns-1, initial_shift_columns+1))
                                       )
        # retrieve optimal transformation coefficients
        t1, t2, t3, t4, b1, b2 = solution_global.x

        # What to do if the optimization did not work
        if not solution_global.success:
            print("did not converge")
            return [np.nan, np.nan]
        transformation_matrix = np.array([[t1, t2], [t3, t4]])






        # pd.set_option('display.max_rows', 30)
        # pd.set_option('display.max_columns', 30)
        # print(moved_tracked_cell)





        if np.abs(np.linalg.det(transformation_matrix) - 1) >= transformation_matrix_determinant_threshold:




            # # rasterio.plot.show(search_cell_matrix, title="Search cell")
            # rasterio.plot.show(tracked_cell_matrix, title="Tracked cell")
            # moved_tracked_cell = move_cell_rotation_approach([t1, t2, t3, t4, b1, b2], tracked_cell_matrix.shape,
            #                                                  interpolator_search_cell, indices)
            # rasterio.plot.show(moved_tracked_cell, title="Moved tracked cell\n" + str(transformation_matrix) + str(
            #     np.array([b1, b2])) + "\n" + str(np.linalg.det(transformation_matrix)))
            # rasterio.plot.show(move_cell_rotation_approach([1, 0, 0, 1, 0, 0], tracked_cell_matrix.shape,
            #                                                interpolator_search_cell, indices),
            #                    title="Unmoved tracked cell" + str(np.linalg.det(transformation_matrix)))


            print("Warning: Transformation matrix has unrealistic determinant: ", np.linalg.det(transformation_matrix))
            return [np.nan, np.nan]
        if return_full_coefficients:
            return [t1, t2, t3, t4, b1, b2]

        # find the impact from the transformation on the central index coordinates
        [new_central_row, new_central_column] = (np.matmul(transformation_matrix,
                                                           np.array([central_row, central_column]))
                                                 + np.array([b1, b2]))

        # calculate the distance vector between the original and the moved central points
        [shift_rows, shift_columns] = [new_central_row - central_row, new_central_column - central_column]
        return [shift_rows, shift_columns]
    # Multichannel images (when the matrix is three-dimensional)
    else:
        interpolator_search_cell_list = list()
        for band in range(tracked_cell_matrix.shape[0]):
            interpolator_search_cell_list.append(scipy.interpolate.RegularGridInterpolator(
                (np.arange(0, search_cell_matrix.shape[-2]), np.arange(0, search_cell_matrix.shape[-1])),
                search_cell_matrix[band, :, :], fill_value=0, bounds_error=False
            ))
            # assign indices in respect to indexing in the search cell matrix
        central_row = np.round(search_cell_matrix.shape[-2] / 2)
        central_column = np.round(search_cell_matrix.shape[-1] / 2)

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
                                       args=(tracked_cell_matrix, interpolator_search_cell_list, indices),
                                       method="Powell")
        t1, t2, t3, t4, b1, b2 = solution_global.x

        # What to do if the optimization did not work
        if not solution_global.success:
            print("did not converge")
            return [np.nan, np.nan]
        # Check if the transformation matrix is singular
        transformation_matrix = np.array([[t1, t2], [t3, t4]])
        if np.abs(np.linalg.det(transformation_matrix) - 1) >= 0.2:
            print("Warning: Transformation matrix has unrealistic determinant: ", np.linalg.det(transformation_matrix))
            return [np.nan, np.nan]
        if return_full_coefficients:
            return [t1, t2, t3, t4, b1, b2]
        [new_central_row, new_central_column] = np.matmul(transformation_matrix,
                                                          np.array([central_row, central_column])) + np.array(
            [b1, b2])
        [shift_rows, shift_columns] = [new_central_row - central_row, new_central_column - central_column]

        return [shift_rows, shift_columns]


from scipy.interpolate import RectBivariateSpline
import sklearn.linear_model

def iterate_transformation_coefficients(tracked_cell_matrix: np.ndarray,search_comparison_cell_matrix: np.ndarray):
    differential_search_cell_matrix_x = np.zeros(search_comparison_cell_matrix.shape)
    differential_search_cell_matrix_y = np.zeros(search_comparison_cell_matrix.shape)
    # should expand the search_comparison_cell_matrix by one compared to the tracked cell matrix
    for i in np.arange(1, search_comparison_cell_matrix.shape[0] - 1):
        for j in np.arange(1, search_comparison_cell_matrix.shape[1] - 1):
            differential_search_cell_matrix_x[i, j] = (search_comparison_cell_matrix[i+1, j] - search_comparison_cell_matrix[i-1, j])/2
            differential_search_cell_matrix_y[i, j] = (search_comparison_cell_matrix[i, j+1] - search_comparison_cell_matrix[i, j-1])/2

    # differential_x_times_x = np.multiply(differential_search_cell_matrix_x, np.repeat(np.expand_dims(np.arange(0, search_comparison_cell_matrix.shape[0]), axis=1), search_comparison_cell_matrix.shape[1], axis=1)).flatten()
    # differential_y_times_x = np.multiply(differential_search_cell_matrix_y, np.repeat(np.expand_dims(np.arange(0, search_comparison_cell_matrix.shape[0]), axis=1), search_comparison_cell_matrix.shape[1], axis=1)).flatten()
    # differential_x_times_y = np.multiply(differential_search_cell_matrix_x, np.repeat(np.expand_dims(np.arange(0, search_comparison_cell_matrix.shape[1]), axis=1), search_comparison_cell_matrix.shape[0], axis=1).transpose()).flatten()
    # differential_y_times_y = np.multiply(differential_search_cell_matrix_y, np.repeat(np.expand_dims(np.arange(0, search_comparison_cell_matrix.shape[1]), axis=1), search_comparison_cell_matrix.shape[0], axis=1).transpose()).flatten()


    differential_search_cell_vector_x = differential_search_cell_matrix_x.flatten()
    differential_search_cell_vector_y = differential_search_cell_matrix_y.flatten()

    # coefficients_regression = np.column_stack([differential_search_cell_vector_x, differential_x_times_x, differential_x_times_y, differential_search_cell_vector_y, differential_y_times_x, differential_y_times_y])
    coefficients_regression = np.column_stack([differential_search_cell_vector_x, differential_search_cell_vector_y])

    values_regression = (tracked_cell_matrix - search_comparison_cell_matrix).flatten()
    model = sklearn.linear_model.LinearRegression().fit(coefficients_regression, values_regression)
    return model.coef_



import datetime
def track_cell_lsm(tracked_cell_matrix: np.ndarray, search_cell_matrix: np.ndarray, initial_shift_values=None,
                   return_full_coefficients: bool = False, transformation_matrix_determinant_threshold: float = 0.2,):


    # assign indices in respect to indexing in the search cell matrix
    central_row = np.round(search_cell_matrix.shape[0] / 2)
    central_column = np.round(search_cell_matrix.shape[1] / 2)

    indices = np.array(np.meshgrid(np.arange(np.ceil(central_row - tracked_cell_matrix.shape[0] / 2),
                                             np.ceil(central_row + tracked_cell_matrix.shape[0] / 2)),
                                   np.arange(np.ceil(central_column - tracked_cell_matrix.shape[1] / 2),
                                             np.ceil(central_column + tracked_cell_matrix.shape[1] / 2)))
                       ).T.reshape(-1, 2).T

    if initial_shift_values is None:
        initial_shift_values = track_cell(tracked_cell_matrix, search_cell_matrix)
        if np.isnan(initial_shift_values[0]):
            return [np.nan, np.nan]

    coefficients = [1, 0, initial_shift_values[0], 0, 1, initial_shift_values[1]]

    use_spline = True
    search_cell_spline = scipy.interpolate.RectBivariateSpline(np.arange(0, search_cell_matrix.shape[0]),
                                                              np.arange(0, search_cell_matrix.shape[1]),
                                                              search_cell_matrix)
    iteration = 0
    test_time = datetime.datetime.now()
    while iteration < 50:
        # print(iteration)

        if use_spline:
            rotation_matrix = np.array([[coefficients[0], coefficients[1]], [coefficients[3], coefficients[4]]])
            # repeat the shift vector for adequate addition of index vectors
            shift_vector = np.repeat(np.array([[coefficients[2]], [coefficients[5]]]),
                                     tracked_cell_matrix.shape[0] * tracked_cell_matrix.shape[1], axis=1)

            moved_indices = np.matmul(rotation_matrix, indices) + shift_vector

            # try to access interpolated values from the search cell
            # print("evaluating spline")
            moved_cell_matrix = search_cell_spline.ev(moved_indices[0, :], moved_indices[1, :]).reshape(tracked_cell_matrix.shape)
            # print("evaluating spline d/dx")
            moved_cell_matrix_dx = search_cell_spline.ev(moved_indices[0, :], moved_indices[1, :], dx=1).reshape(tracked_cell_matrix.shape)
            moved_cell_matrix_dx_times_x = np.multiply(moved_cell_matrix_dx, indices[0, :].reshape(tracked_cell_matrix.shape))
            moved_cell_matrix_dx_times_y = np.multiply(moved_cell_matrix_dx, indices[1, :].reshape(tracked_cell_matrix.shape))
            # print("evaluating spline d/dy")
            moved_cell_matrix_dy = search_cell_spline.ev(moved_indices[0, :], moved_indices[1, :], dy=1).reshape(tracked_cell_matrix.shape)
            moved_cell_matrix_dy_times_x = np.multiply(moved_cell_matrix_dy, indices[0, :].reshape(tracked_cell_matrix.shape))
            moved_cell_matrix_dy_times_y = np.multiply(moved_cell_matrix_dy, indices[1, :].reshape(tracked_cell_matrix.shape))

            model = sklearn.linear_model.LinearRegression().fit(
                    np.column_stack([moved_cell_matrix_dx_times_x.flatten(), moved_cell_matrix_dx_times_y.flatten(),
                                    moved_cell_matrix_dx.flatten(), moved_cell_matrix_dy_times_x.flatten(),
                                    moved_cell_matrix_dy_times_y.flatten(), moved_cell_matrix_dy.flatten()]),
                    (tracked_cell_matrix - moved_cell_matrix).flatten())

            coefficient_adjustment = model.coef_
        # TODO: Add step size dampener (see Optimization script)
        coefficients += coefficient_adjustment
        if (coefficient_adjustment < 0.05).all():
            break
        iteration += 1
    # print(datetime.datetime.now() - test_time)
    # print("Took " + str(iteration) + " iterations")
    # rasterio.plot.show(tracked_cell_matrix, title="Tracked cell")
    # rasterio.plot.show(moved_cell_matrix, title="Moved cell")
    # rasterio.plot.show(search_cell_spline.ev(indices[0, :], indices[1, :]).reshape(tracked_cell_matrix.shape))

    if iteration == 50:
        print("Reached maximum number of iterations, returning no value")
        return [np.nan, np.nan]

    if return_full_coefficients:
        return [coefficients[0], coefficients[1], coefficients[3], coefficients[4], coefficients[2], coefficients[5]]

    transformation_matrix = np.array([[coefficients[0], coefficients[1]], [coefficients[3], coefficients[4]]])
    if np.abs(np.linalg.det(transformation_matrix) - 1) >= transformation_matrix_determinant_threshold:
        print("Warning: Transformation matrix has unrealistic determinant: ", np.linalg.det(transformation_matrix))

        return [np.nan, np.nan]
    [new_central_row, new_central_column] = np.matmul(np.array([[coefficients[0], coefficients[1]], [coefficients[3], coefficients[4]]]), np.array([central_row, central_column])) + np.array([coefficients[2], coefficients[5]])
    [shift_rows, shift_columns] = [new_central_row - central_row, new_central_column - central_column]


    return [shift_rows, shift_columns]



def get_tracked_pixels_square(tracked_pixels, central_pixel_coordinates):
    
    """
    Returns a dataframe which consists of the eight (or less) surrounding pixels, for a given pixel with matrix indices
    central_pixel_coordinates. This is a helper function for retrack_wrong_matching_pixels and
    remove_outlying_tracked_pixels
    Parameters
    ----------
    tracked_pixels :
        A DataFrame containing the tracked pixels with their position and movement rates as returned by track_movement.
    central_pixel_coordinates
        The coordinates (as matrix index tuple) of the central pixel, for which the adjacent pixels will be returned.
    Returns
    ----------
    neighbouring_tracked_pixels: A DataFrame containing all available adjacent pixels with their position and movement
    rates. The central pixel is not included.
    """
    
    # get central indices
    central_pixel_row = central_pixel_coordinates[0]
    central_pixel_col = central_pixel_coordinates[1]
    neighbouring_tracked_pixels = pd.DataFrame()

    # get indices around given central index
    smaller_row_index = np.max(tracked_pixels[tracked_pixels["row"] < central_pixel_row]["row"])
    bigger_row_index = np.min(tracked_pixels[tracked_pixels["row"] > central_pixel_row]["row"])
    smaller_col_index = np.max(tracked_pixels[tracked_pixels["column"] < central_pixel_col]["column"])
    bigger_col_index = np.min(tracked_pixels[tracked_pixels["column"] > central_pixel_col]["column"])

    # find all adjacent points, but not the central point itself
    for row in [smaller_row_index, central_pixel_row, bigger_row_index]:
        for column in [smaller_col_index, central_pixel_col, bigger_col_index]:
            neighbouring_tracked_pixels = pd.concat([neighbouring_tracked_pixels,
                                                     tracked_pixels.loc[
                                                         (tracked_pixels["row"] == row) &
                                                         (tracked_pixels["column"] == column)]])

    neighbouring_tracked_pixels.dropna(inplace=True, how="any")
    return neighbouring_tracked_pixels


def remove_outlying_tracked_pixels(tracked_pixels, from_rotation: bool = True, from_velocity: bool = True):
    """
    Removes outliers given by deviations in velocity or movement direction from a given dataframe of tracked pixels.
    Parameters
    ----------
    tracked_pixels :
        A DataFrame containing the tracked pixels with their position and movement rates as returned by track_movement.
    from_rotation : bool = True
        If True, pixels with a movement direction which differs more than 90° from the average movement direction of its
        adjacent pixels, will be considered outliers and their movement rates will be set to NaN.
    from_velocity : bool = True
        If True, pixels with a movement rate more than twice the average movement rate of its adjacent pixels, will be
        considered outliers and their movement rates will be set to NaN.
    Returns
    ----------
    tracked_pixels: A DataFrame containing one row for every tracked pixel, specifying the position of the tracked pixel
    (in terms of matrix indices) and the movement in x- and y-direction in pixels. Invalid matchings are marked by
    NaN values for the movement.
    """

    for [row, col] in zip(tracked_pixels["row"].tolist(), tracked_pixels["column"].tolist()):
        neighbouring_pixels = get_tracked_pixels_square(tracked_pixels, [row, col])
        central_pixel = tracked_pixels[(tracked_pixels["row"] == row) & (tracked_pixels["column"] == col)]
        neighbouring_pixels = neighbouring_pixels[(neighbouring_pixels["row"] != row) |
                                                  (neighbouring_pixels["column"] != col)]
        neighbouring_pixels = neighbouring_pixels[~neighbouring_pixels["movement_row_direction"].isna()]

        # initialize average movement angle
        average_movement_angle = 0

        # calculate the average movement angle
        for i in range(len(neighbouring_pixels)):
            movement_vector = np.array([neighbouring_pixels.iloc[i]["movement_row_direction"],
                                        neighbouring_pixels.iloc[i]["movement_column_direction"]])
            movement_vector_length = np.linalg.norm(movement_vector)
            average_movement_angle += np.arccos(np.dot(movement_vector, np.array([1, 0])) / movement_vector_length)

        # check if there exist any neighbouring pixels to avoid division by 0
        if len(neighbouring_pixels) > 0:
            average_movement_angle /= len(neighbouring_pixels)
            movement_vector_central = np.array([central_pixel["movement_row_direction"],
                                                central_pixel["movement_column_direction"]])

            movement_vector_central_length = np.linalg.norm(movement_vector_central)
            movement_angle_central = np.arccos(
                np.dot(movement_vector_central.T, np.array([1, 0])) / movement_vector_central_length)

            # check if the central pixel is a rotation outlier
            if (
                    np.abs(
                        average_movement_angle - movement_angle_central) > np.pi / 2) & from_rotation:
                print("removed pixel", row, col, "for rotation reasons")
                tracked_pixels.loc[
                    (tracked_pixels["row"] == row) &
                    (tracked_pixels["column"] == col),
                    ["movement_row_direction",
                     "movement_column_direction",
                     "movement_distance_pixels"]] = np.nan

            # check if the central pixel is a velocity outlier
            if (central_pixel["movement_distance_pixels"].values >
                2 * np.nanmean(neighbouring_pixels[
                                   "movement_distance_pixels"].values)) & from_velocity:
                print("removed pixel", row, col, "for velocity reasons")
                tracked_pixels.loc[
                    (tracked_pixels["row"] == row) & (tracked_pixels["column"] == col),
                    ["movement_row_direction",
                     "movement_column_direction",
                     "movement_distance_pixels"]] = np.nan

    return tracked_pixels

# ToDo: Implement method chooser
def retrack_wrong_matching_pixels(tracked_pixels, track_matrix, search_matrix, cell_size: int, tracking_area_size: int,
                                  fallback_on_cross_correlation: bool = False,
                                  transformation_matrix_determinant_threshold: int = 0.2):
    """
    Tries to retrack all pixels with movement given by np.nan via the least-squares approach.
    Parameters
    ----------
    tracked_pixels :
        A DataFrame containing the tracked pixels with their position and movement rates as returned by track_movement.
    track_matrix :
        The raster image matrix of the first image.
    search_matrix :
        The raster image matrix of the second image. The two matrices are assumed to be aligned (for example using
        align_images).
    cell_size : int
        The cell size used for tracking. For details see track_movement.
    tracking_area_size : int
        The size of the area in which the corresponding image section will be searched. For details see track_movement.
    fallback_on_cross_correlation : bool = False
        If True, the function will return the matching result from the cross-correlation approach, if the least-squares
        approach did not find a valid matching in the second try.
    Returns
    ----------
    tracked_pixels: A DataFrame containing one row for every tracked pixel, specifying the position of the tracked pixel
    (in terms of matrix indices) and the movement in x- and y-direction in pixels. Invalid matchings are marked by
    NaN values for the movement.
    """
    # check for every tracked point if it has a valid movement result
    for [row, col] in zip(tracked_pixels["row"].tolist(), tracked_pixels["column"].tolist()):
        central_pixel = tracked_pixels[(tracked_pixels["row"] == row) & (tracked_pixels["column"] == col)]

        # if the movement is NaN, it is no valid result and a second matching attempt is made
        if np.isnan(central_pixel["movement_row_direction"].values) & np.isnan(
                central_pixel["movement_column_direction"].values):
            neighbouring_pixels = get_tracked_pixels_square(tracked_pixels, [row, col])

            neighbouring_pixels = neighbouring_pixels[
                (neighbouring_pixels["row"] != row) | (neighbouring_pixels["column"] != col)]
            neighbouring_pixels = neighbouring_pixels[~neighbouring_pixels["movement_row_direction"].isna()]

            # check if there are adjacent points, which can provide an initial guess
            if len(neighbouring_pixels) > 0:
                movement_row_direction_neighbours_mean = np.nanmean(
                    neighbouring_pixels["movement_row_direction"].values)
                movement_column_direction_neighbours_mean = np.nanmean(
                    neighbouring_pixels["movement_column_direction"].values)

                # get tracked and search cells
                track_cell1 = get_submatrix_symmetric(central_index=[row, col], shape=(cell_size, cell_size),
                                                      matrix=track_matrix)

                search_area2 = get_submatrix_symmetric(central_index=[row, col],
                                                       shape=(tracking_area_size, tracking_area_size),
                                                       matrix=search_matrix)
                # perform the new matching attempt with provided initial values
                match = track_cell_lsm(track_cell1, search_area2,
                                              initial_shift_values=[movement_row_direction_neighbours_mean,
                                                             movement_column_direction_neighbours_mean],
                                       transformation_matrix_determinant_threshold=transformation_matrix_determinant_threshold)

                # If the second attempt was not valid neither, fallback on the cross-correlation result if desired
                if np.isnan(match[0]) & fallback_on_cross_correlation:
                    print("Falling back on cross correlation for pixel", [row, col])
                    match = track_cell(track_cell1, search_area2)
                tracked_pixels[(tracked_pixels["row"] == row) & (tracked_pixels["column"] == col)] = [row, col,
                                                                                                      match[0],
                                                                                                      match[1],
                                                                                                      np.linalg.norm(
                                                                                                          [match[0],
                                                                                                           match[1]])]
    return tracked_pixels





def track_cell_parallelized(central_index: np.ndarray):

    tracked_cell_size = 50
    search_area_size = 90

    # print("tracking pixel: ", central_index[0], type(central_index[0]), "and", central_index[1], type(central_index[1]))
    # get the first image section as tracked cell
    track_cell1 = get_submatrix_symmetric(central_index=central_index, shape=(tracked_cell_size, tracked_cell_size),
                                          matrix=shared_image_matrix1)

    # get the second image section as search cell
    search_area2 = get_submatrix_symmetric(central_index=central_index,
                                           shape=(search_area_size, search_area_size),
                                           matrix=shared_image_matrix2)

    if len(search_area2) == 0:
        return [np.nan, np.nan]

    match = track_cell_lsm(track_cell1, search_area2)
    return match


def track_movement(image1_matrix, image2_matrix, image_transform, tracking_area: gpd.GeoDataFrame,
                   number_of_tracked_points: int = None, distance_of_tracked_points: float = None,
                   cell_size: int = 40,
                   tracking_area_size: int = 50, tracking_method: str = "lsm", remove_outliers: bool = True,
                   retry_matching: bool = True, transformation_matrix_determinant_threshold: int = 0.2):
    """
     Calculates the movement of points between two aligned raster image matrices (with the same transform) in a given
     area using the cross-correlation or the least-squares approach.
    Parameters
    ----------
    image1_matrix :
        A numpy array with 2 or three dimensions, where the last two dimensions give the image height and width
        respectively and for a threedimensional array, the first dimension gives the channels of the raster image.
        This array should represent the earlier observation.
    image2_matrix :
        A numpy array of the same format as image1_matrix representing the second observation.
    image_transform :
        An object of the class Affine as provided by the rasterio package. The two images are assumed to be aligned
        (for example as a result of align_images) and therefore have the same transform.
    tracking_area : gpd.GeoDataFrame
        A GeoDataFrame containing a single polygon, where pixels should be tracked. In the area specified by this
        polygon, a regularly spaced grid of points is created using grid_points_on_polygon. The polygon is assumed to
        have the same coordinate reference system as the raster images (i.e. it is compatible with the given
        image_transform).
    number_of_tracked_points : int
        The number of points to be created by grid_points_on_polygon (for details see there), which is the number of
        separate pixels that are being tracked.
    cell_size : int = 40
        The size of the cells in pixels, which will be created in order to compare the two images. The function
        get_submatrix_symmetric is used for extracting the image section based on this value. This parameter determines
        the size ob detectable object as well as the influence of boundary effects.
    tracking_area_size : int = 50
        The size of the area in pixels, where fitting image sections are being searched. This parameter determines the
        maximum detectable movement rate and influences computation speed. This value must be higher than the parameter
        cell_size.
    tracking_method : str = "lsm"
        One of "lsm" (for least-squares matching) or "cross-correlation", determining the method used for tracking each
        cell.
    remove_outliers : bool = True
        Determines if outliers determined via large deviations in velocity and movement direction will be removed
        (for details see remove_outlying_tracked_pixels). Low correlation matchings
        (cross-correlation coefficient < 0.5) and matchings with a transformation determinant < 0.8 or > 1.2 will always
        be removed.
    retry_matching : bool = True
        When the first matching using the least-squares approach was unsuccessful (e.g. due to a low correlation
        coefficient for the initial values, a non-converging optimization problem or because the pixel was removed as an
        outlier), it can be rerun using a different starting value. If this parameter is True, the already calculated
        average movement rate for its adjacent pixels will be used as initial value for the least-squares optimization
        problem.
    Returns
    ----------
    tracked_pixels: A DataFrame containing one row for every tracked pixel, specifying the position of the tracked pixel
    (in terms of matrix indices) and the movement in x- and y-direction in pixels. Invalid matchings are marked by
    NaN values for the movement.

    """

    # get grid of tracked points
    if (number_of_tracked_points is not None) & (distance_of_tracked_points is None):
        points = grid_points_on_polygon_by_number_of_points(polygon=tracking_area,
                                                            number_of_points=number_of_tracked_points)
    if (number_of_tracked_points is None) & (distance_of_tracked_points is not None):
        points = grid_points_on_polygon_by_distance(polygon=tracking_area,
                                                            distance_of_points=distance_of_tracked_points)
    if (number_of_tracked_points is None) & (distance_of_tracked_points is None):
        raise ValueError("Expect either number_of_tracked_points or distance_of_tracked_points to be not None")
    if (number_of_tracked_points is not None) & (distance_of_tracked_points is not None):
        raise ValueError("Got number_of_tracked_points and distance_of_tracked_points. Only one of them can be"
                         "specified")

    # get the matrix indices for every point
    rows, cols = get_raster_indices_from_points(points, image_transform)
    tracked_points_pixels = np.array([rows, cols]).transpose()

    # initialize dataframe for the resulting tracked points
    tracked_pixels = pd.DataFrame()

    # Loop over all points
    if tracking_method == "cross-correlation":
        for central_index in tracked_points_pixels:
            # if ((len(tracked_pixels)+1) % 50 == 0):
            print("Starting to track pixel ", len(tracked_pixels) + 1, " of ", len(tracked_points_pixels), ": ",
                  central_index)

            # get the first image section as tracked cell
            track_cell1 = get_submatrix_symmetric(central_index=central_index, shape=(cell_size, cell_size),
                                                  matrix=image1_matrix)

            # get the second image section as search cell
            search_area2 = get_submatrix_symmetric(central_index=central_index,
                                                   shape=(tracking_area_size, tracking_area_size),
                                                   matrix=image2_matrix)

            # checks if the search area reached the boundary of the underlying raster image and skips when this is the case
            if len(search_area2) == 0:
                continue

            # perform tracking based on specified method
            if tracking_method == "lsm":
                match = track_cell_lsm(track_cell1, search_area2,
                                       transformation_matrix_determinant_threshold=transformation_matrix_determinant_threshold)
            elif tracking_method == "lsm_Powell":
                match = track_cell_lsm_powell(track_cell1, search_area2,
                                              transformation_matrix_determinant_threshold=transformation_matrix_determinant_threshold)
            elif tracking_method == "cross-correlation":
                match = track_cell(track_cell1, search_area2)
            else:
                raise ValueError("Tracking method not recognized.")

            # add the tracked pixel to the results dataframe
            tracked_pixels = pd.concat([tracked_pixels, pd.DataFrame({"row": central_index[0],
                                                                      "column": central_index[1],
                                                                      "movement_row_direction": match[0],
                                                                      "movement_column_direction": match[1]},
                                                                     index=[len(tracked_pixels)])])

    # for process in processes:
    #     process.join()

    if tracking_method == "lsm":
        # with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        # # track_cell_parallelized_partial = partial(track_cell_parallelized,
        # #                                       image1_matrix=image1_matrix,
        # #                                       image2_matrix=image2_matrix,
        # #                                       tracked_cell_size=cell_size,
        # #                                       search_area_size=tracking_area_size,
        # #                                       transformation_matrix_determinant_threshold=transformation_matrix_determinant_threshold)
        #
        # # Get the number of available CPU cores
        # # num_cores = multiprocessing.cpu_count()
        #
        #     dynamic_parameters = list(tracked_points_pixels)
        #     # with multiprocessing.Pool(processes=num_cores) as pool:
        #     tracking_results = list(executor.map(lambda x: track_cell_parallelized(
        #             image1_matrix=image1_matrix,
        #             image2_matrix=image2_matrix,
        #             tracked_cell_size=cell_size,
        #             search_area_size=tracking_area_size,
        #             central_index=x,
        #             transformation_matrix_determinant_threshold=transformation_matrix_determinant_threshold),
        #             dynamic_parameters))


        global shared_image_matrix1, shared_image_matrix2
        shared_image_matrix1 = image1_matrix
        shared_image_matrix2 = image2_matrix

        list_of_central_indices = tracked_points_pixels.tolist()
        print("Starting multiprocessing at ", datetime.datetime.now(), " with ", len(list_of_central_indices),
              " points (~ 4000 points/minute)")

        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            tracking_results = pool.map(track_cell_parallelized, list_of_central_indices)



        # access the respective tracked point coordinates and its movement
        movement_row_direction = [arr[0] for arr in tracking_results]
        movement_column_direction = [arr[1] for arr in tracking_results]
        rows = [index[0] for index in list(tracked_points_pixels)]
        columns = [index[1] for index in list(tracked_points_pixels)]
        # create dataframe with all tracked points results
        tracked_pixels = pd.DataFrame({"row": rows, "column": columns,
                                       "movement_row_direction": movement_row_direction,
                                       "movement_column_direction": movement_column_direction})

    # calculate the movement distance in pixels from the movement along the axes for the whole results dataframe
    tracked_pixels.insert(4, "movement_distance_pixels",
                          np.linalg.norm(tracked_pixels.loc[:, ["movement_row_direction", "movement_column_direction"]],
                                         axis=1))

    # Perform postprocessing, such as outlier removal and second matching attempt
    if remove_outliers:
        tracked_pixels = remove_outlying_tracked_pixels(tracked_pixels, from_rotation=True, from_velocity=True)
    if retry_matching & (tracking_method == "lsm"):
        tracked_pixels = retrack_wrong_matching_pixels(tracked_pixels, image1_matrix, image2_matrix, cell_size,
                                                       tracking_area_size, fallback_on_cross_correlation=False,
                                                       transformation_matrix_determinant_threshold=
                                                       transformation_matrix_determinant_threshold)
        if remove_outliers:
            tracked_pixels = remove_outlying_tracked_pixels(tracked_pixels, from_rotation=True, from_velocity=True)

    return tracked_pixels


def align_images(image1_matrix, image2_matrix, image1_transform, reference_area: gpd.GeoDataFrame, number_of_control_points: int, cell_size: int = 40,
                 tracking_area_size: int = 60, select_bands=None, image_alignment_via_lsm: bool = False):
    """
    Aligns two georeferenced images opened in rasterio by matching them in the area given by the reference area. In
    areas, where only one of the two images contains data, the matrix values will be set to 0 in both images.
    Parameters
    ----------
    image1_matrix :
        A raster image matrix to be aligned with the second image.
    image2_matrix :
        A raster image matrix to be aligned with the first image. The alignment takes place via the creation of an image
        transform for the second matrix. The transform of the first image has to be supplied. Thus, the first image is
        assumed to be correctly georeferenced.
    image1_transform :
        A transform, which represents the georeference information for the first image.
    reference_area : gpd.GeoDataFrame
        A single-element GeoDataFrame, containing a polygon for specifying the reference area used for the alignment.
        This is the area, where no movement is suspected.
    number_of_control_points: int
        An approximation of how many points should be created on the reference_area polygon to track possible camera
        position differences. For details see grid_points_on_polygon.
    cell_size: int = 40
        The size of image sections in pixels which are compared during the tracking. See track_movement for details.
    tracking_area_size: int = 60
        The size of the image sections in pixels which are used as a search area during the tracking. Must be greater
        than the parameter cell_size. See track_movement for details.
    select_bands = None
        For multichannel images the channel, which should be used for tracking. If it is None (the default) it will
        select the first three channels. If it is an integer, only the respective single channel will be selected.
        For multi-channel selection a list can be provided (e.g. [0,2,3] for the first, third and fourth channel of the
        given raster image). Channels are assumed to be in the same order for the two provided image files.
    image_alignment_via_lsm: bool = False
        If False, each control point is being tracked using the cross-correlation approach and their average
        displacement is taken into account (see also track_cell). If True, after calculating the average displacement,
        the least-squares approach  is used on the whole reference area,
        to determine the exact transformation matrix necessary to align the two images optimally. For large images,
        this step is cost-intensive and can take some time.
    Returns
    ----------
    [image1_matrix, new_matrix2, image_transform]: The two matrices representing the raster image as numpy arrays and
    their transform for the coordinate reference system of the input images. As the two matrices are aligned,
    they possess the same transformation.
    """

    # rename the image transform
    image_transform = image1_transform

    if select_bands is None:
        select_bands = [0, 1, 2]
    if len(image1_matrix.shape) == 3:
        image1_matrix = image1_matrix[select_bands, :, :]
        image2_matrix = image2_matrix[select_bands, :, :]

    # track control area using the cross-correlation approach
    tracked_control_pixels = track_movement(image1_matrix, image2_matrix, image_transform, tracking_area=reference_area,
                                            number_of_tracked_points=number_of_control_points,
                                            tracking_method="cross-correlation", cell_size=cell_size,
                                            tracking_area_size=tracking_area_size, remove_outliers=False)
    # calculate mean movement in the control area
    row_movements = np.nanmean(tracked_control_pixels["movement_row_direction"])
    column_movements = np.nanmean(tracked_control_pixels["movement_column_direction"])

    # for single-channel images
    if len(image1_matrix.shape) == 2:
        interpolator_image2_matrix = scipy.interpolate.RegularGridInterpolator(
            (np.arange(0, image2_matrix.shape[0]), np.arange(0, image2_matrix.shape[1])),
            image2_matrix, fill_value=0, bounds_error=False)

    # for multichannel images a list of interpolators is needed
    else:
        interpolator_image2_matrix = list()
        for band in range(image1_matrix.shape[0]):
            interpolator_image2_matrix.append(scipy.interpolate.RegularGridInterpolator(
                (np.arange(0, image2_matrix.shape[-2]), np.arange(0, image2_matrix.shape[-1])),
                image2_matrix[band, :, :], fill_value=0, bounds_error=False
            ))

    # prepare transformation of the second image
    central_row = np.round(image2_matrix.shape[-2] / 2)
    central_column = np.round(image2_matrix.shape[-1] / 2)
    indices = np.array(np.meshgrid(np.arange(np.ceil(central_row - image1_matrix.shape[-2] / 2),
                                             np.ceil(central_row + image1_matrix.shape[-2] / 2)),
                                   np.arange(np.ceil(central_column - image1_matrix.shape[-1] / 2),
                                             np.ceil(central_column + image1_matrix.shape[-1] / 2)))
                       ).T.reshape(-1, 2).T

    if image_alignment_via_lsm:

        # transform given reference area polygon and raster images to the same coordinate reference system
        inverse_transform = (~image_transform).to_shapely()
        transformation_matrix = np.array(inverse_transform)
        transformed_polygon = reference_area
        transformed_polygon["geometry"] = shapely.affinity.affine_transform(reference_area.loc[0]["geometry"],
                                                                            transformation_matrix)

        # get the mask for the two raster images from the reference area polygon
        mask_matrix = rasterio.features.rasterize([reference_area.loc[0]["geometry"]],
                                                  out_shape=image1_matrix.shape[-2:])
        # invert the mask
        mask_matrix = -mask_matrix + 1
        # for multichannel images
        if len(image1_matrix.shape) == 3:

            # repeat the matrix for every channel
            mask_matrix = np.repeat(mask_matrix[np.newaxis, :, :], image1_matrix.shape[0], axis=0)
            masked_matrix1 = np.ma.masked_array(image1_matrix, mask=mask_matrix)
            masked_matrix2 = np.ma.masked_array(image2_matrix, mask=mask_matrix)

        # for single-channel images
        else:

            masked_matrix1 = np.ma.masked_array(image1_matrix / image1_matrix.max(), mask=mask_matrix)
            masked_matrix2 = np.ma.masked_array(image2_matrix / image2_matrix.max(), mask=mask_matrix)

            masked_matrix1 = masked_matrix1.filled(0)
            masked_matrix2 = masked_matrix2.filled(0)

        # perform least-squares matching on the whole masked matrices and get the full transformation coefficients
        matching = track_cell_lsm(masked_matrix1, masked_matrix2,
                                         initial_shift_values=[row_movements, column_movements], return_full_coefficients=True)
    else:  # if alignment is performed via cross-correlation, assume the transformation matrix to be the identity
        # translation is the mean translation from ground control points
        matching = [1, 0, 0, 1, row_movements, column_movements]

    print("Transformation matrix coefficients for image:", matching, "to match optimally in the reference area.")


    # move the second image according to the found coefficients to match the first one
    new_matrix2 = move_cell_rotation_approach(matching, image1_matrix.shape, interpolator_image2_matrix, indices)

    # set areas where only one of the images contains data to 0 in both images
    image1_matrix[new_matrix2 == 0] = 0
    new_matrix2[image1_matrix == 0] = 0


    return [image1_matrix, new_matrix2, image_transform]
