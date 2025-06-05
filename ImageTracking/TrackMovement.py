import geopandas as gpd
import multiprocessing
import pandas as pd
import numpy as np
import logging
import tqdm
import sklearn
import datetime
import scipy

from CreateGeometries.HandleGeometries import get_submatrix_symmetric
from ImageTracking.TrackingResults import TrackingResults
from CreateGeometries.HandleGeometries import get_raster_indices_from_points
from ImageTracking.ImageInterpolator import ImageInterpolator

def track_cell_cc(tracked_cell_matrix: np.ndarray, search_cell_matrix: np.ndarray):
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
        tracking_results: TrackingResults
            An instance of the class TrackingResults containing the movement in row and column direction and the
            corresponding cross-correlation coefficient.
        """
    height_tracked_cell = tracked_cell_matrix.shape[-2]
    width_tracked_cell = tracked_cell_matrix.shape[-1]
    height_search_cell = search_cell_matrix.shape[-2]
    width_search_cell = search_cell_matrix.shape[-1]
    best_correlation = 0
    # for multichannel images, flattening ensures that always the same band is being compared
    tracked_vector = tracked_cell_matrix.flatten()

    if np.linalg.norm(tracked_vector) == 0:
        return TrackingResults(movement_rows=np.nan, movement_cols=np.nan, tracking_method="cross-correlation",
                               cross_correlation_coefficient=np.nan,
                               tracking_success=False)
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
            if np.linalg.norm(search_subcell_vector) == 0:
                continue

            # initialize correlation for the current central pixel (i,j)
            corr = 0
            # check if the search subcell vectors has any non-zero elements (to avoid dividing by zero)
            if np.any(search_subcell_vector):
                # normalize search_subcell vector
                search_subcell_vector = search_subcell_vector - np.mean(search_subcell_vector)
                search_subcell_vector = search_subcell_vector / np.linalg.norm(search_subcell_vector)
                corr = np.correlate(tracked_vector, search_subcell_vector, mode='valid')
            if len(corr) != 1:
                logging.info("Correlation was " + str(corr) + ". Skipping")
                continue
            if float(corr) > best_correlation:
                best_correlation = float(corr)
                best_correlation_coordinates = [i, j]
    if best_correlation <= 0:
        logging.info("Found no matching with positive correlation. Skipping")
        return TrackingResults(movement_rows=np.nan, movement_cols=np.nan,
                               tracking_method="cross-correlation",
                               cross_correlation_coefficient=np.nan,
                               tracking_success=False)


    movement_for_best_correlation = np.floor(np.subtract(best_correlation_coordinates,
                                                         [search_cell_matrix.shape[-2] / 2,
                                                          search_cell_matrix.shape[-1] / 2]))
    tracking_results = TrackingResults(movement_rows=movement_for_best_correlation[0],
                                       movement_cols=movement_for_best_correlation[1],
                                       tracking_method="cross-correlation",
                                       cross_correlation_coefficient=best_correlation,
                                       tracking_success=True)
    return tracking_results


def move_indices_from_transformation_matrix(transformation_matrix: np.array, indices: np.array):
    """
    Given a list of n indices (as an np.array with shape (2,n)), calculates the position of these indices after applying
    the given extended transformation matrix, which is a (2,3)-shaped np.array.
    Parameters
    ----------
    transformation_matrix: np.array
        The affine transformation matrix to be applied to the indices, as a (2,3)-shaped np.array, where the entries at
        [0:1,2] are the shift values and the other entries are the linear transformation matrix.
    indices: np.array
        Indices to apply the transformation matrix to. Expected to have shape (2,n), where n is the number of points.

    Returns
    -------
    movement_indices: np.array
        The indices after applying the transformation matrix, as a (2,n)-shaped np.array.
    """

    linear_transformation_matrix = np.array(transformation_matrix[0:2, 0:2])
    shift_vector = np.array(np.repeat(np.expand_dims(np.array(transformation_matrix[0:2, 2]), axis=1),
                                      indices.shape[1], axis=1))
    moved_indices = np.matmul(linear_transformation_matrix, indices) + shift_vector
    return moved_indices


def track_cell_lsm(tracked_cell_matrix: np.ndarray, search_cell_matrix: np.ndarray,
                   initial_shift_values: np.array = None) -> TrackingResults:
    """
    Tracks the movement of a given image section ('tracked_cell_matrix') within a given search cell
    ('search_cell_matrix') using the least-squares approach. Initial shift values can be provided, otherwise the cross-
    correlation approach is used to determine the optimal initial shift value.
    Parameters
    ----------
    tracked_cell_matrix: np.ndarray
        The array representing a section of the first image, which is compared to sections of the search_cell_matrix.
    search_cell_matrix: np.ndarray
        An array, which delimits the area in which possible matching image sections are searched.
    initial_shift_values: np.array=None
        Initial shift values in the format [initial_movement_rows, initial_movement_cols] to be used in the first step
        of the least-squares optimization problem.
    Returns
    -------
    tracking_results: TrackingResults
        An instance of the class TrackingResults containing the results of the tracking, that is the shift of the rows
        and columns at the central pixel respectively, as well as the corresponding extended transformation matrix. If
        the tracking does not provide valid results (e.g. because no valid initial values were found or the optimization
        problem did not converge after 50 iterations), the shift values and the transformation matrix are set to np.nan
        and None, respectively.
    """

    # assign indices in respect to indexing in the search cell matrix
    central_row = np.round(search_cell_matrix.shape[-2] / 2)
    central_column = np.round(search_cell_matrix.shape[-1] / 2)

    indices = np.array(np.meshgrid(np.arange(np.ceil(central_row - tracked_cell_matrix.shape[-2] / 2),
                                             np.ceil(central_row + tracked_cell_matrix.shape[-2] / 2)),
                                   np.arange(np.ceil(central_column - tracked_cell_matrix.shape[-1] / 2),
                                             np.ceil(central_column + tracked_cell_matrix.shape[-1] / 2)))
                       ).T.reshape(-1, 2).T

    if initial_shift_values is None:
        cross_correlation_results = track_cell_cc(tracked_cell_matrix, search_cell_matrix)
        initial_shift_values = [cross_correlation_results.movement_rows, cross_correlation_results.movement_cols]
        if np.isnan(initial_shift_values[0]):
            logging.info("Cross-correlation did not provide a result. Skipping.")
            return TrackingResults(movement_rows=np.nan, movement_cols=np.nan, tracking_method="least-squares",
                                   tracking_success=False)
    if np.isnan(initial_shift_values[0]):
        logging.info("Going with default shift values [0,0] as initial values")
        initial_shift_values = [0, 0]

    # initialize the transformation with the given initial shift values and the identity matrix as linear transformation
    coefficients = [1, 0, initial_shift_values[0], 0, 1, initial_shift_values[1]]
    # calculate transformation matrix form of the coefficients
    transformation_matrix = np.array([[coefficients[0], coefficients[1], coefficients[2]],
                                      [coefficients[3], coefficients[4], coefficients[5]]])

    search_cell_spline = scipy.interpolate.RectBivariateSpline(np.arange(0, search_cell_matrix.shape[-2]),
                                                               np.arange(0, search_cell_matrix.shape[-1]),
                                                               search_cell_matrix)

    # search_cell_spline = ImageInterpolator(search_cell_matrix)

    iteration = 0
    optimization_start_time = datetime.datetime.now()
    # Point to check the stopping condition. If the distance between the previous and current central point is smaller
    # than 0.1 (pixels), the iteration halts. For the first comparison, this point is initialized which has
    # distance > 0.1 to the central point always
    previous_moved_central_point = np.array([np.nan, np.nan])

    while iteration < 20:
        moved_indices = move_indices_from_transformation_matrix(transformation_matrix=transformation_matrix,
                                                                indices=indices)
        moved_cell_matrix = search_cell_spline.ev(moved_indices[0, :], moved_indices[1, :]).reshape(
            tracked_cell_matrix.shape)
        moved_cell_matrix_dx = search_cell_spline.ev(moved_indices[0, :], moved_indices[1, :], dx=1).reshape(
            tracked_cell_matrix.shape)

        moved_cell_matrix_dx_times_x = np.multiply(moved_cell_matrix_dx,
                                                   indices[0, :].reshape(tracked_cell_matrix.shape))

        moved_cell_matrix_dx_times_y = np.multiply(moved_cell_matrix_dx,
                                                   indices[1, :].reshape(tracked_cell_matrix.shape))

        moved_cell_matrix_dy = search_cell_spline.ev(moved_indices[0, :], moved_indices[1, :], dy=1).reshape(
            tracked_cell_matrix.shape)
        moved_cell_matrix_dy_times_x = np.multiply(moved_cell_matrix_dy,
                                                   indices[0, :].reshape(tracked_cell_matrix.shape))
        moved_cell_matrix_dy_times_y = np.multiply(moved_cell_matrix_dy,
                                                   indices[1, :].reshape(tracked_cell_matrix.shape))

        model = sklearn.linear_model.LinearRegression().fit(
            np.column_stack([moved_cell_matrix_dx_times_x.flatten(), moved_cell_matrix_dx_times_y.flatten(),
                             moved_cell_matrix_dx.flatten(), moved_cell_matrix_dy_times_x.flatten(),
                             moved_cell_matrix_dy_times_y.flatten(), moved_cell_matrix_dy.flatten()]),
            (tracked_cell_matrix - moved_cell_matrix).flatten())

        coefficient_adjustment = model.coef_

        # adjust coefficients accordingly
        coefficients += coefficient_adjustment
        # calculate transformation matrix form of the coefficients
        transformation_matrix = np.array([[coefficients[0], coefficients[1], coefficients[2]],
                                          [coefficients[3], coefficients[4], coefficients[5]]])

        # Calculate impact of the coefficient adjustment on the resulting movement rate for a stopping condition
        [new_central_row, new_central_column] = (np.matmul(np.array([[coefficients[0], coefficients[1]],
                                                                     [coefficients[3], coefficients[4]]]),
                                                           np.array([central_row, central_column]))
                                                 + np.array([coefficients[2], coefficients[5]]))
        # define the position of the newly calculated central point
        new_moved_central_point = np.array([new_central_row, new_central_column])
        # if the adjustment results in less than 0.1 pixel adjustment between the considered points, stop the iteration
        if np.linalg.norm(previous_moved_central_point - np.array([new_central_row, new_central_column])) < 0.1:
            break

        # continue iteration and redefine the previous moved central point
        previous_moved_central_point = new_moved_central_point
        iteration += 1

    if iteration == 20:
        logging.info("Did not converge after 20 iterations.")
        return TrackingResults(movement_rows=np.nan, movement_cols=np.nan, tracking_method="least-squares",
                               tracking_success=False)

    moved_indices = move_indices_from_transformation_matrix(transformation_matrix=transformation_matrix,
                                                            indices=indices)
    moved_cell_matrix = search_cell_spline.ev(moved_indices[0, :], moved_indices[1, :]).reshape(
        tracked_cell_matrix.shape)

    # flatten the comparison cell matrix
    moved_cell_submatrix_vector = moved_cell_matrix.flatten()

    moved_cell_submatrix_vector = moved_cell_submatrix_vector - np.mean(moved_cell_submatrix_vector)
    moved_cell_submatrix_vector = moved_cell_submatrix_vector / np.linalg.norm(moved_cell_submatrix_vector)
    tracked_cell_vector = tracked_cell_matrix.flatten()
    tracked_cell_vector = tracked_cell_vector - np.mean(tracked_cell_vector)
    tracked_cell_vector = tracked_cell_vector / np.linalg.norm(tracked_cell_vector)
    corr = np.correlate(tracked_cell_vector, moved_cell_submatrix_vector, mode='valid')

    [shift_rows, shift_columns] = [new_central_row - central_row, new_central_column - central_column]

    tracking_results = TrackingResults(movement_rows=shift_rows, movement_cols=shift_columns,
                                       tracking_method="least-squares", tracking_success=True,
                                       cross_correlation_coefficient=float(corr))

    return tracking_results




def track_cell_lsm_parallelized(central_index: np.ndarray):
    """
    Helper function for letting the least-squares approach run parallelized. It takes only a np.ndarray that represents
    one central index that should be tracked. All the other tracking variables (tracked and search cell sizes and the
    image data have to be declared separately as global variables.
    Parameters
    ----------
    central_index: np.ndarray
        A np.ndarray that represents one central index to be tracked

    Returns
    -------
     tracking_results: TrackingResults
        An instance of the class TrackingResults containing the results of the tracking, that is the shift of the rows
        and columns at the central pixel respectively, as well as the corresponding extended transformation matrix. If the tracking does not
        provide valid results (e.g. because no valid initial values were found or the optimization problem did not
        converge after 50 iterations), the shift values and the transformation matrix are set to np.nan and None,
        respectively.
    """
    tracked_cell_size = shared_tracked_cell_size
    search_area_size = shared_search_area_size

    # get the first image section as tracked cell
    track_cell1 = get_submatrix_symmetric(central_index=central_index, shape=(tracked_cell_size, tracked_cell_size),
                                          matrix=shared_image_matrix1)

    # get the second image section as search cell
    search_area2 = get_submatrix_symmetric(central_index=central_index,
                                           shape=(search_area_size, search_area_size),
                                           matrix=shared_image_matrix2)

    if len(search_area2) == 0:
        return TrackingResults(movement_rows=np.nan, movement_cols=np.nan, tracking_method="least-squares",
                               transformation_matrix=None,
                               tracking_success=False)
    logging.info("Tracking point" + str(central_index))
    tracking_results = track_cell_lsm(track_cell1, search_area2)
    return tracking_results


def track_movement_lsm(image1_matrix, image2_matrix, image_transform, points_to_be_tracked: gpd.GeoDataFrame,
                       movement_cell_size: int = 50, movement_tracking_area_size: int = 60,
                       save_columns: list[str] = None) -> pd.DataFrame:
    """
    Calculates the movement of given points between two aligned raster image matrices (with the same transform)
    using the least-squares approach.
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
    points_to_be_tracked :
        A GeoPandas-GeoDataFrame giving the position of points that will be tracked. Points will be converted to matrix
        indices for referencing during tracking.
    movement_cell_size : int = 50
        The size of the cells in pixels, which will be created in order to compare the two images. The function
        get_submatrix_symmetric is used for extracting the image section based on this value. This parameter determines
        the size ob detectable object as well as the influence of boundary effects.
    movement_tracking_area_size : int = 60
        The size of the area in pixels, where fitting image sections are being searched. This parameter determines the
        maximum detectable movement rate and influences computation speed. This value must be higher than the parameter
        cell_size.
    save_columns: list[str] = None
        The columns to be saved to the results dataframe. Default is None, which will save "movement_row_direction",
        "movement_column_direction", "movement_distance_pixels", and "movement_bearing_pixels".
        Possible further values are: "transformation_matrix" and "correlation_coefficient".
    Returns
    ----------
    tracked_pixels: A DataFrame containing one row for every tracked pixel, specifying the position of the tracked pixel
    (in terms of matrix indices) and the movement in x- and y-direction in pixels. Invalid matchings are marked by
    NaN values for the movement.
    """

    if len(points_to_be_tracked) == 0:
        raise ValueError("No ppoints provided in the points to be tracked GeoDataFrame. Please provide a GeoDataFrame"
                         "with  at least one element.")

    # ToDo: Find a way to make these variables NOT global
    global shared_image_matrix1, shared_image_matrix2, shared_tracked_cell_size, shared_search_area_size
    shared_image_matrix1 = image1_matrix
    shared_image_matrix2 = image2_matrix
    shared_tracked_cell_size = movement_cell_size
    shared_search_area_size = movement_tracking_area_size

    # create list of central indices in terms of the image matrix
    rows, cols = get_raster_indices_from_points(points_to_be_tracked, image_transform)
    points_to_be_tracked_matrix_indices = np.array([rows, cols]).transpose()
    list_of_central_indices = points_to_be_tracked_matrix_indices.tolist()

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        tracking_results = list(tqdm.tqdm(pool.imap(track_cell_lsm_parallelized, list_of_central_indices),
                                total=len(list_of_central_indices),
                                          desc="Tracking points",
                                          unit="points",
                                          smoothing=0.1,
                                          bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} {unit}"
                                                     "[{remaining}, {rate_fmt}]"))

    # access the respective tracked point coordinates and its movement
    movement_row_direction = [results.movement_rows for results in tracking_results]
    movement_column_direction = [results.movement_cols for results in tracking_results]
    # create dataframe with all tracked points results
    tracked_pixels = pd.DataFrame({"row": rows, "column": cols})
    if save_columns is None:
        save_columns = ["movement_row_direction", "movement_column_direction",
                        "movement_distance_pixels", "movement_bearing_pixels"]
    if "movement_row_direction" in save_columns:
        tracked_pixels["movement_row_direction"] = movement_row_direction
    if "movement_column_direction" in save_columns:
        tracked_pixels["movement_column_direction"] = movement_column_direction
    if "movement_distance_pixels" in save_columns:
        # calculate the movement distance in pixels from the movement along the axes for the whole results dataframe
        tracked_pixels["movement_distance_pixels"] = np.linalg.norm(
            tracked_pixels.loc[:, ["movement_row_direction", "movement_column_direction"]], axis=1)
    if "movement_bearing_pixels" in save_columns:
        tracked_pixels["movement_bearing_pixels"] = np.arctan2(-tracked_pixels["movement_row_direction"],
                                                               tracked_pixels["movement_column_direction"])
        tracked_pixels.loc[tracked_pixels['movement_bearing_pixels'] < 0, 'movement_bearing_pixels'] \
            = tracked_pixels['movement_bearing_pixels'] + 2 * np.pi
        tracked_pixels['movement_bearing_pixels'] = np.degrees(tracked_pixels['movement_bearing_pixels'])
    if "transformation_matrix" in save_columns:
        tracked_pixels["transformation_matrix"] = [results.transformation_matrix for results in tracking_results]
    if "correlation_coefficient" in save_columns:
        tracked_pixels["correlation_coefficient"] = [results.cross_correlation_coefficient for results in tracking_results]
    return tracked_pixels

