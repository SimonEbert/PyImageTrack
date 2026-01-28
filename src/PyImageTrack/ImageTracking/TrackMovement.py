import logging
import multiprocessing
from functools import partial
from multiprocessing import shared_memory

import geopandas as gpd
import numpy as np
import pandas as pd
import scipy
import sklearn
import tqdm

from ..CreateGeometries.HandleGeometries import get_raster_indices_from_points
from ..CreateGeometries.HandleGeometries import get_submatrix_symmetric, get_submatrix_rect_from_extents
from .TrackingResults import TrackingResults
from ..Parameters.AlignmentParameters import AlignmentParameters
from ..Parameters.TrackingParameters import TrackingParameters


def track_cell_cc(tracked_cell_matrix: np.ndarray, search_cell_matrix: np.ndarray, search_center=None):
    """
        Calculates the movement of an image section using the cross-correlation approach.
        Parameters
        ----------
        search_center: list
            A 2-element array containing the row (first entry) and column (second entry) of the respective search window
            center
        tracked_cell_matrix: np.ndarray
            An array (a section of the first image), which is compared to sections of the search_cell_matrix (a section
            of the second image).
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

    if tracked_vector.size == 0:
        return TrackingResults(movement_rows=np.nan, movement_cols=np.nan, tracking_method="cross-correlation",
                               cross_correlation_coefficient=np.nan,
                               tracking_success=False)

    # normalize the tracked vector
    tracked_vector = tracked_vector - np.mean(tracked_vector)

    if np.linalg.norm(tracked_vector) == 0:
        return TrackingResults(movement_rows=np.nan, movement_cols=np.nan, tracking_method="cross-correlation",
                               cross_correlation_coefficient=np.nan,
                               tracking_success=False)
    tracked_vector = tracked_vector / np.linalg.norm(tracked_vector)
    for i in np.arange(np.ceil(height_tracked_cell / 2), height_search_cell - np.ceil(height_tracked_cell / 2)):
        for j in np.arange(np.ceil(width_tracked_cell / 2), width_search_cell - np.ceil(width_tracked_cell / 2)):
            search_subcell_matrix = get_submatrix_symmetric([i, j],
                                                            (tracked_cell_matrix.shape[-2],
                                                             tracked_cell_matrix.shape[-1]),
                                                            search_cell_matrix)
            # flatten the comparison cell matrix
            search_subcell_vector = search_subcell_matrix.flatten()
            if search_subcell_vector.size == 0:
                continue
            # if np.linalg.norm(search_subcell_vector) == 0:
            #     continue

            # Initialize correlation for the current central pixel (i, j)
            corr = None

            # Only compute correlation if the search vector has any non-zero elements
            if np.any(search_subcell_vector):
                # Normalize search_subcell vector
                search_subcell_vector = search_subcell_vector - np.mean(search_subcell_vector)
                if np.linalg.norm(search_subcell_vector) == 0:
                    continue
                search_subcell_vector = search_subcell_vector / np.linalg.norm(search_subcell_vector)

                # np.correlate returns a 1-element ndarray for equal-length vectors
                corr = np.correlate(tracked_vector, search_subcell_vector, mode="valid")

            # If corr was not computed (e.g., all-zero window), skip this candidate
            if corr is None:
                continue
            # ToDO: There is still a bug here, that we need to take corr[0] --> Normally corr should suffice
            # corr is an ndarray here; take the scalar value safely
            corr_val = float(corr[0])
            if corr_val > best_correlation:
                best_correlation = corr_val
                best_correlation_coordinates = [i, j]

    if best_correlation <= 0:
        logging.info("Found no matching with positive correlation. Skipping")
        return TrackingResults(movement_rows=np.nan, movement_cols=np.nan,
                               tracking_method="cross-correlation",
                               cross_correlation_coefficient=np.nan,
                               tracking_success=False)

    # Use the provided logical center inside the search window if given (asymmetric windows)
    if search_center is None:
        central_row = search_cell_matrix.shape[-2] / 2
        central_col = search_cell_matrix.shape[-1] / 2
    else:
        central_row, central_col = map(float, search_center)

    movement_for_best_correlation = np.floor(
        np.subtract(best_correlation_coordinates, [central_row, central_col])
    )

    tracking_results = TrackingResults(
        movement_rows=movement_for_best_correlation[0],
        movement_cols=movement_for_best_correlation[1],
        tracking_method="cross-correlation",
        cross_correlation_coefficient=best_correlation,
        tracking_success=True
    )
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
                   initial_shift_values: np.array = None, search_center=None) -> TrackingResults:
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
    if search_center is None:
        central_row = np.round(search_cell_matrix.shape[-2] / 2)
        central_column = np.round(search_cell_matrix.shape[-1] / 2)
    else:
        central_row = float(search_center[0])
        central_column = float(search_center[1])

    indices = np.array(np.meshgrid(np.arange(np.ceil(central_row - tracked_cell_matrix.shape[-2] / 2),
                                             np.ceil(central_row + tracked_cell_matrix.shape[-2] / 2)),
                                   np.arange(np.ceil(central_column - tracked_cell_matrix.shape[-1] / 2),
                                             np.ceil(central_column + tracked_cell_matrix.shape[-1] / 2)))
                       ).T.reshape(-1, 2).T

    if initial_shift_values is None:
        cross_correlation_results = track_cell_cc(
            tracked_cell_matrix, search_cell_matrix, search_center=search_center
        )
        initial_shift_values = [cross_correlation_results.movement_rows, cross_correlation_results.movement_cols]
        if np.isnan(initial_shift_values[0]):
            logging.info("Cross-correlation did not provide a result. Skipping.")
            return TrackingResults(movement_rows=np.nan, movement_cols=np.nan, tracking_method="least-squares",
                                   tracking_success=False)
    if np.isnan(initial_shift_values[0]):
        logging.info("Going with default shift values [0,0] as initial values")
        initial_shift_values = [0, 0]

    # initialize the transformation with the given initial shift values and the identity matrix as linear transformation
    coefficients = [1, 0, initial_shift_values[0], 0, 1, initial_shift_values[1], 0, 1]
    # calculate transformation matrix form of the coefficients
    transformation_matrix = np.array([[coefficients[0], coefficients[1], coefficients[2]],
                                      [coefficients[3], coefficients[4], coefficients[5]]])

    search_cell_spline = scipy.interpolate.RectBivariateSpline(np.arange(0, search_cell_matrix.shape[-2]),
                                                               np.arange(0, search_cell_matrix.shape[-1]),
                                                               search_cell_matrix)

    iteration = 0
    # Point to check the stopping condition. If the distance between the previous and current central point is smaller
    # than 0.1 (pixels), the iteration halts. For the first comparison, this point is initialized as NaN which has
    # distance > 0.1 to the central point always
    previous_moved_central_point = np.array([np.nan, np.nan])

    while iteration < 50:
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

        # Prepare input and output for linear regression
        X = np.column_stack([moved_cell_matrix_dx_times_x.flatten(), moved_cell_matrix_dx_times_y.flatten(),
                             moved_cell_matrix_dx.flatten(), moved_cell_matrix_dy_times_x.flatten(),
                             moved_cell_matrix_dy_times_y.flatten(), moved_cell_matrix_dy.flatten(),
                             np.ones(moved_cell_matrix.shape).flatten(), moved_cell_matrix.flatten()
                             ])
        y = (tracked_cell_matrix - moved_cell_matrix).flatten()
        
        # Check for NaN values before fitting
        if np.any(np.isnan(X)) or np.any(np.isnan(y)):
            logging.info("NaN values detected in LSM optimization. Skipping this point.")
            return TrackingResults(movement_rows=np.nan, movement_cols=np.nan, tracking_method="least-squares",
                                   tracking_success=False)
        
        model = sklearn.linear_model.LinearRegression().fit(X, y)

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
        if np.linalg.norm(previous_moved_central_point - np.array([new_central_row, new_central_column])) < 0.01:
            break

        # continue iteration and redefine the previous moved central point
        previous_moved_central_point = new_moved_central_point
        iteration += 1

    if iteration == 50:
        logging.info("Did not converge after 50 iterations.")
        return TrackingResults(movement_rows=np.nan, movement_cols=np.nan, tracking_method="least-squares",
                               tracking_success=False)

    moved_indices = move_indices_from_transformation_matrix(transformation_matrix=transformation_matrix,
                                                            indices=indices)
    moved_cell_matrix = search_cell_spline.ev(moved_indices[0, :], moved_indices[1, :]).reshape(
        tracked_cell_matrix.shape)

    # flatten the comparison cell matrix
    moved_cell_submatrix_vector = moved_cell_matrix.flatten()

    if moved_cell_submatrix_vector.size == 0:
        return TrackingResults(movement_rows=np.nan, movement_cols=np.nan, tracking_method="least-squares",
                               tracking_success=False)

    moved_cell_submatrix_vector = moved_cell_submatrix_vector - np.mean(moved_cell_submatrix_vector)
    moved_cell_norm = np.linalg.norm(moved_cell_submatrix_vector)
    if moved_cell_norm == 0:
        return TrackingResults(movement_rows=np.nan, movement_cols=np.nan, tracking_method="least-squares",
                               tracking_success=False)
    moved_cell_submatrix_vector = moved_cell_submatrix_vector / moved_cell_norm
    tracked_cell_vector = tracked_cell_matrix.flatten()
    if tracked_cell_vector.size == 0:
        return TrackingResults(movement_rows=np.nan, movement_cols=np.nan, tracking_method="least-squares",
                               tracking_success=False)
    tracked_cell_vector = tracked_cell_vector - np.mean(tracked_cell_vector)
    tracked_cell_norm = np.linalg.norm(tracked_cell_vector)
    if tracked_cell_norm == 0:
        return TrackingResults(movement_rows=np.nan, movement_cols=np.nan, tracking_method="least-squares",
                               tracking_success=False)
    tracked_cell_vector = tracked_cell_vector / tracked_cell_norm
    corr = np.correlate(tracked_cell_vector, moved_cell_submatrix_vector, mode='valid')
    # if corr > 0.85:
    #      rasterio.plot.show(search_cell_spline.ev(indices[0,:],indices[1,:]).reshape(tracked_cell_matrix.shape), title="Image 2 unmoved")
    #      rasterio.plot.show(tracked_cell_matrix, title="Image 1 unmoved")
    #      rasterio.plot.show(moved_cell_matrix, title="Image 2 moved")

    [shift_rows, shift_columns] = [new_central_row - central_row, new_central_column - central_column]

    tracking_results = TrackingResults(movement_rows=shift_rows, movement_cols=shift_columns,
                                       tracking_method="least-squares", tracking_success=True,
                                       cross_correlation_coefficient=float(corr))
    return tracking_results


def track_cell_lsm_parallelized(central_index: np.ndarray, shm1_name, shm2_name, shape1, shape2, dtype,
                                tracked_cell_size, control_search_extents=None, search_extents=None):
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
    # Get matrices from shared memory
    shm1 = multiprocessing.shared_memory.SharedMemory(name=shm1_name)
    shared_image_matrix1 = np.ndarray(shape1, dtype=dtype, buffer=shm1.buf)

    shm2 = multiprocessing.shared_memory.SharedMemory(name=shm2_name)
    shared_image_matrix2 = np.ndarray(shape2, dtype=dtype, buffer=shm2.buf)

    # Extract the tracked (template) cell from image1
    track_cell1 = get_submatrix_symmetric(
        central_index=central_index,
        shape=(tracked_cell_size, tracked_cell_size),
        matrix=shared_image_matrix1
    )

    # Build the search window from extents
    if control_search_extents is not None:
        # Alignment mode
        search_area2, center_in_search = get_submatrix_rect_from_extents(
            central_index=np.array(central_index),
            extents=control_search_extents,
            matrix=shared_image_matrix2
        )
        search_center = center_in_search
    elif search_extents is not None:
        # Movement mode
        search_area2, center_in_search = get_submatrix_rect_from_extents(
            central_index=np.array(central_index),
            extents=search_extents,
            matrix=shared_image_matrix2
        )
        search_center = center_in_search
    else:
        # No extents configured (should be prevented earlier)
        return TrackingResults(
            movement_rows=np.nan, movement_cols=np.nan,
            tracking_method="least-squares",
            transformation_matrix=None, tracking_success=False
        )

    # Guard against empty windows (e.g., near borders)
    if getattr(search_area2, "size", 0) == 0:
        return TrackingResults(
            movement_rows=np.nan, movement_cols=np.nan,
            tracking_method="least-squares",
            transformation_matrix=None, tracking_success=False
        )
    logging.info("Tracking point" + str(central_index))
    tracking_results = track_cell_lsm(track_cell1, search_area2, search_center=search_center)
    return tracking_results


def track_movement_lsm(image1_matrix, image2_matrix, image_transform, points_to_be_tracked: gpd.GeoDataFrame,
                       tracking_parameters: TrackingParameters = None, alignment_parameters: AlignmentParameters = None,
                       alignment_tracking: bool = False,
                       save_columns: list[str] = None, task_label: str = "Tracking points") -> pd.DataFrame:
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
    tracking_parameters : TrackingParameters
        The tracking parameters used for tracking
    alignment_tracking : bool = False
        If the tracking parameters for alignment from the tracking parameters class should be used. Defaults to False,
        i.e. it used the tracking parameters associated with movement (e.g. movement_cell_size instead of
        alignment_cell_size)
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
        raise ValueError("No points provided in the points to be tracked GeoDataFrame. Please provide a GeoDataFrame"
                         "with  at least one element.")

    # extract relevant parameters
    if alignment_tracking:
        if alignment_parameters is None:
            raise ValueError("alignment_tracking=True requires alignment_parameters.")
        movement_cell_size = alignment_parameters.control_cell_size
        cross_correlation_threshold = alignment_parameters.cross_correlation_threshold_alignment
    else:
        if tracking_parameters is None:
            raise ValueError("alignment_tracking=False requires tracking_parameters.")
        movement_cell_size = tracking_parameters.movement_cell_size
        cross_correlation_threshold = tracking_parameters.cross_correlation_threshold_movement

    # Check image sizes and create shared memory for image1_matrix
    if image1_matrix.nbytes == 0:
        raise ValueError("Image1 matrix has zero size. Cannot create shared memory.")
    
    shared_memory_image1 = shared_memory.SharedMemory(create=True, size=image1_matrix.nbytes)
    shared_image_matrix1 = np.ndarray(image1_matrix.shape, dtype=image1_matrix.dtype, buffer=shared_memory_image1.buf)
    shared_image_matrix1[:] = image1_matrix[:]
    shape_image1 = image1_matrix.shape

    # Check image sizes and create shared memory for image2_matrix
    if image2_matrix.nbytes == 0:
        raise ValueError("Image2 matrix has zero size. Cannot create shared memory.")
    
    shared_memory_image2 = shared_memory.SharedMemory(create=True, size=image2_matrix.nbytes)
    shared_image_matrix2 = np.ndarray(image2_matrix.shape, dtype=image1_matrix.dtype, buffer=shared_memory_image2.buf)
    shared_image_matrix2[:] = image2_matrix[:]
    shape_image2 = image2_matrix.shape

    if shared_image_matrix1.dtype != shared_image_matrix1.dtype:
        raise ValueError("The datatypes of image1 and image2 must be identical.")
    image1_dtype = image1_matrix.dtype

    # Configure which asymmetric extents to use depending on mode.
    # Movement mode reads TrackingParameters.search_extent_px,
    # Alignment mode reads AlignmentParameters.control_search_extent_px.
    shared_search_extents = None
    shared_control_search_extents = None

    if alignment_tracking:
        if getattr(alignment_parameters, "control_search_extent_px", None):
            shared_control_search_extents = tuple(int(v) for v in alignment_parameters.control_search_extent_px)
        else:
            raise ValueError("Alignment: control_search_extent_px must be set (tuple posx,negx,posy,negy).")
    else:
        if getattr(tracking_parameters, "search_extent_px", None):
            shared_search_extents = tuple(int(v) for v in tracking_parameters.search_extent_px)
        else:
            raise ValueError("Movement: search_extent_px must be set (tuple posx,negx,posy,negy).")

    # create list of central indices in terms of the image matrix
    rows, cols = get_raster_indices_from_points(points_to_be_tracked, image_transform)
    points_to_be_tracked_matrix_indices = np.array([rows, cols]).transpose()
    list_of_central_indices = points_to_be_tracked_matrix_indices.tolist()
    # build partial function
    partial_lsm_tracking_function = partial(
        track_cell_lsm_parallelized,
        shm1_name=shared_memory_image1.name,
        shm2_name=shared_memory_image2.name,
        shape1=shape_image1,
        shape2=shape_image2,
        dtype=image1_dtype,
        tracked_cell_size=movement_cell_size,
        control_search_extents=shared_control_search_extents,
        search_extents=shared_search_extents
    )

    tracking_results = []
    try:
        procs = max(1, multiprocessing.cpu_count() - 1)
        with multiprocessing.Pool(processes=procs) as pool:
            tracking_results = list(
                tqdm.tqdm(
                    pool.imap(partial_lsm_tracking_function, list_of_central_indices),
                    total=len(list_of_central_indices),
                    desc=task_label,
                    unit="points",
                    smoothing=0.1,
                    bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} {unit}[{remaining}, {rate_fmt}]"
                )
            )
            # tracking_results = list(tqdm.tqdm(pool.imap(track_cell_lsm_parallelized, list_of_central_indices),
            #                                   total=len(list_of_central_indices),
            #                                   desc=task_label,
            #                                   unit="points",
            #                                   smoothing=0.1,
            #                                   bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} {unit}"
            #                                              "[{remaining}, {rate_fmt}]"))
    finally:
        # Clean-up image matrices from shared memory - always execute, even on error
        try:
            shared_memory_image1.close()
            shared_memory_image1.unlink()
        except Exception:
            pass  # Ignore cleanup errors
        try:
            shared_memory_image2.close()
            shared_memory_image2.unlink()
        except Exception:
            pass  # Ignore cleanup errors

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

    # Add correlation coefficient column BEFORE filtering on it
    tracked_pixels["correlation_coefficient"] = [results.cross_correlation_coefficient for results in tracking_results]

    # Filter by correlation threshold - handle case where column might not exist
    if "correlation_coefficient" in tracked_pixels.columns:
        tracked_pixels_above_cc_threshold = tracked_pixels[
            tracked_pixels["correlation_coefficient"] > cross_correlation_threshold]
    else:
        tracked_pixels_above_cc_threshold = tracked_pixels.copy()

    if "correlation_coefficient" not in save_columns and "correlation_coefficient" in tracked_pixels_above_cc_threshold.columns:
        tracked_pixels_above_cc_threshold = tracked_pixels_above_cc_threshold.drop(columns="correlation_coefficient")
    return tracked_pixels_above_cc_threshold
