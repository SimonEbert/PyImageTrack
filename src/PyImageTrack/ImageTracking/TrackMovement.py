import multiprocessing
from functools import partial
from multiprocessing import shared_memory

import geopandas as gpd
import numpy as np
import pandas as pd
import tqdm
from skimage.feature import match_template, peak_local_max


from ..CreateGeometries.HandleGeometries import get_raster_indices_from_points
from ..CreateGeometries.HandleGeometries import get_submatrix_symmetric, get_submatrix_rect_from_extents
from .TrackingResults import TrackingResults
from ..Parameters.AlignmentParameters import AlignmentParameters
from ..Parameters.TrackingParameters import TrackingParameters
from .ImageInterpolator import ImageInterpolator
from ..ConsoleOutput import get_console



# def track_cell_cc(tracked_cell_matrix: np.ndarray, search_cell_matrix: np.ndarray, search_center=None):
#     """
#         Calculates the movement of an image section using the cross-correlation approach.
#         Parameters
#         ----------
#         search_center: list
#             A 2-element array containing the row (first entry) and column (second entry) of the respective search window
#             center
#         tracked_cell_matrix: np.ndarray
#             An array (a section of the first image), which is compared to sections of the search_cell_matrix (a section
#             of the second image).
#         search_cell_matrix: np.ndarray
#             An array, which delimits the area in which possible matching image sections are searched.
#         Returns
#         ----------
#         tracking_results: TrackingResults
#             An instance of the class TrackingResults containing the movement in row and column direction and the
#             corresponding cross-correlation coefficient.
#         """
#     height_tracked_cell = tracked_cell_matrix.shape[-2]
#     width_tracked_cell = tracked_cell_matrix.shape[-1]
#     height_search_cell = search_cell_matrix.shape[-2]
#     width_search_cell = search_cell_matrix.shape[-1]
#     tracked_cell_matrix = get_submatrix_symmetric([np.ceil(height_tracked_cell / 2),
#                                                    np.ceil(height_tracked_cell / 2)],
#                                                   (tracked_cell_matrix.shape[-2], tracked_cell_matrix.shape[-1]),
#                                                   tracked_cell_matrix)
#
#     best_correlation = 0
#     # for multichannel images, flattening ensures that always the same band is being compared
#     tracked_vector = tracked_cell_matrix.flatten()
#
#     if tracked_vector.size == 0:
#         return TrackingResults(movement_rows=np.nan, movement_cols=np.nan, tracking_method="cross-correlation",
#                                cross_correlation_coefficient=np.nan,
#                                tracking_success=False)
#
#     # normalize the tracked vector
#     tracked_vector = tracked_vector - np.mean(tracked_vector)
#
#     if np.linalg.norm(tracked_vector) == 0:
#         return TrackingResults(movement_rows=np.nan, movement_cols=np.nan, tracking_method="cross-correlation",
#                                cross_correlation_coefficient=np.nan,
#                                tracking_success=False)
#     tracked_vector = tracked_vector / np.linalg.norm(tracked_vector)
#     for i in np.arange(np.ceil(height_tracked_cell / 2), height_search_cell - np.ceil(height_tracked_cell / 2)):
#         for j in np.arange(np.ceil(width_tracked_cell / 2), width_search_cell - np.ceil(width_tracked_cell / 2)):
#             search_subcell_matrix = get_submatrix_symmetric([i, j],
#                                                             (tracked_cell_matrix.shape[-2],
#                                                              tracked_cell_matrix.shape[-1]),
#                                                             search_cell_matrix)
#             # flatten the comparison cell matrix
#             search_subcell_vector = search_subcell_matrix.flatten()
#             if search_subcell_vector.size == 0:
#                 continue
#             # if np.linalg.norm(search_subcell_vector) == 0:
#             #     continue
#
#             # Initialize correlation for the current central pixel (i, j)
#             corr = None
#
#             # Only compute correlation if the search vector has any non-zero elements
#             if np.any(search_subcell_vector):
#                 # Normalize search_subcell vector
#                 search_subcell_vector = search_subcell_vector - np.mean(search_subcell_vector)
#                 if np.linalg.norm(search_subcell_vector) == 0:
#                     continue
#                 search_subcell_vector = search_subcell_vector / np.linalg.norm(search_subcell_vector)
#                 # np.correlate returns a 1-element ndarray for equal-length vectors
#                 corr = np.correlate(tracked_vector, search_subcell_vector, mode="valid")
#
#             # If corr was not computed (e.g., all-zero window), skip this candidate
#             if corr is None:
#                 continue
#             # corr is an ndarray here; take the scalar value safely
#             corr_val = float(corr[0])
#             if corr_val > best_correlation:
#                 best_correlation = corr_val
#                 best_correlation_coordinates = [i, j]
#
#     if best_correlation <= 0:
#         logging.info("Found no matching with positive correlation. Skipping")
#         return TrackingResults(movement_rows=np.nan, movement_cols=np.nan,
#                                tracking_method="cross-correlation",
#                                cross_correlation_coefficient=np.nan,
#                                tracking_success=False)
#
#     # Use the provided logical center inside the search window if given (asymmetric windows)
#     if search_center is None:
#         central_row = search_cell_matrix.shape[-2] / 2
#         central_col = search_cell_matrix.shape[-1] / 2
#     else:
#         central_row, central_col = map(float, search_center)
#
#     movement_for_best_correlation = np.floor(
#         np.subtract(best_correlation_coordinates, [central_row, central_col])
#     )
#
#     tracking_results = TrackingResults(
#         movement_rows=movement_for_best_correlation[0],
#         movement_cols=movement_for_best_correlation[1],
#         tracking_method="cross-correlation",
#         cross_correlation_coefficient=best_correlation,
#         tracking_success=True
#     )
#     return tracking_results



def track_cell_cc(tracked_cell_matrix: np.ndarray,
                  search_cell_matrix: np.ndarray,
                  tracking_parameters: TrackingParameters = None,
                  search_center=None):
    # skimage expects float arrays
    tracked = tracked_cell_matrix.astype(np.float32)
    search = search_cell_matrix.astype(np.float32)

    # --- Normalized cross-correlation ---
    try:
        if len(search.shape) == 2:
            corr_map = match_template(
                search,
                tracked,
                pad_input=False
            )
        else:
            corrs = []
            for c in range(search.shape[0]):
                corr = match_template(
                    search[c],
                    tracked[c],
                    pad_input=False
                )
                corrs.append(corr)

            corr_map = np.mean(corrs, axis=0)
    except ValueError:
        return None


    min_distance_initial_estimates = getattr(tracking_parameters, "min_distance_initial_estimates", 1)
    nb_initial_estimates = getattr(tracking_parameters, "nb_initial_estimate_peaks", 1)
    initial_estimate_mode = getattr(tracking_parameters, "initial_estimate_mode", "count")
    correlation_threshold = getattr(tracking_parameters, "correlation_threshold_initial_estimates", None)
    
    if initial_estimate_mode == "count":
       peaks = peak_local_max(corr_map, num_peaks=nb_initial_estimates,
                              min_distance=min_distance_initial_estimates)
    elif initial_estimate_mode == "threshold":
        peaks = peak_local_max(corr_map,
                               threshold_abs=correlation_threshold,
                               min_distance=min_distance_initial_estimates)
    else:
        raise ValueError("Unknown initial estimates mode " + str(initial_estimate_mode))

    # --- Convert top-left index to center coordinates ---
    template_center_row = tracked.shape[-2] // 2
    template_center_col = tracked.shape[-1] // 2

    peaks_centered = peaks + np.array([template_center_row, template_center_col])

    # --- Define search center ---
    if search_center is None:
        central_row = search_cell_matrix.shape[-2] / 2
        central_col = search_cell_matrix.shape[-1] / 2
    else:
        central_row, central_col = map(float, search_center)

    # --- Movement ---
    movement = np.floor(
        peaks_centered -
        np.array([central_row, central_col])
    )
    # ToDo: Necessary?
    # Fallback for single initial values (1d) returns from peak_local_max
    if len(movement.shape) == 1:
        movement = np.expand_dims(movement, axis=0)

    return movement

def move_indices_from_transformation_matrix(transformation_matrix: np.ndarray, indices: np.ndarray):
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

    indices_h = np.vstack([indices, np.ones(indices.shape[1])])
    moved_indices = (transformation_matrix @ indices_h)[0:2]
    return moved_indices


def track_cell_lsm(tracked_cell_matrix: np.ndarray, search_cell_matrix: np.ndarray,
                   tracking_parameters: TrackingParameters = None,
                   initial_shift_values: list = None, search_center=None) -> TrackingResults:
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
        initial_shift_values = track_cell_cc(
            tracked_cell_matrix, search_cell_matrix, search_center=search_center,
            tracking_parameters=tracking_parameters
        )

    if initial_shift_values is None or len(initial_shift_values) == 0:
        initial_shift_values = [[0, 0]]

    # initialize the transformation with the given initial shift values and the identity matrix as linear transformation

    # Interpolator for the image. If search_cell_matrix.ndim == 2, this is a direct wrapper for
    # scipy.interpolate.RectBivariateSpline (and should behave exactly equivalently)
    # If search_cell_matrix.ndim == 3, it expands the RectBivariateSpline functionality to apply interpolation only in
    # the last two axes, while the first axis is treated as containing several (independent) image channels
    search_cell_spline = ImageInterpolator(search_cell_matrix)


    lsm_results = []
    for initial_shift_value_estimate in initial_shift_values:
        coefficients = [1, 0, initial_shift_value_estimate[0], 0, 1, initial_shift_value_estimate[1], 0, 1]
        # calculate transformation matrix form of the coefficients
        transformation_matrix = np.array([[coefficients[0], coefficients[1], coefficients[2]],
                                          [coefficients[3], coefficients[4], coefficients[5]]])
        lsm_tracking_result = perform_lsm_loop(tracked_cell_matrix, search_cell_spline,indices,
                                               coefficients, transformation_matrix, central_row,
                                               central_column,)
        if lsm_tracking_result.tracking_success:
            lsm_results.append(lsm_tracking_result)
    if len(lsm_results) == 0:
        return TrackingResults(movement_rows=np.nan, movement_cols=np.nan, tracking_method="least-squares",
                               tracking_success=False)
    best_match_tracking_result = min(lsm_results, key=lambda x: x.rmse)
    return best_match_tracking_result


def perform_lsm_loop(tracked_cell_matrix:np.ndarray, search_cell_spline: ImageInterpolator,
                     indices: np.ndarray,
                     coefficients: list[float | int],
                     transformation_matrix: np.ndarray,
                     central_row,central_column)-> TrackingResults:

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
        moved_cell_matrix_dy = search_cell_spline.ev(moved_indices[0, :], moved_indices[1, :], dy=1).reshape(
            tracked_cell_matrix.shape)

        x = indices[0, :].reshape(tracked_cell_matrix.shape[-2:])
        y = indices[1, :].reshape(tracked_cell_matrix.shape[-2:])

        moved_cell_matrix_dx_times_x = moved_cell_matrix_dx * x[..., :, :]
        moved_cell_matrix_dx_times_y = moved_cell_matrix_dx * y[..., :, :]
        moved_cell_matrix_dy_times_x = moved_cell_matrix_dy * x[..., :, :]
        moved_cell_matrix_dy_times_y = moved_cell_matrix_dy * y[..., :, :]

        # Residuals: (H*W*C,)
        residuals = (tracked_cell_matrix - moved_cell_matrix).reshape(-1)
        
        # Jacobian: (H*W*C, 8)
        J = np.column_stack([
            moved_cell_matrix_dx_times_x.reshape(-1),
            moved_cell_matrix_dx_times_y.reshape(-1),
            moved_cell_matrix_dx.reshape(-1),
            moved_cell_matrix_dy_times_x.reshape(-1),
            moved_cell_matrix_dy_times_y.reshape(-1),
            moved_cell_matrix_dy.reshape(-1),
            np.ones(tracked_cell_matrix.shape).reshape(-1),
            moved_cell_matrix.reshape(-1),
        ])

        # Check for NaN values before fitting
        if np.any(np.isnan(J)) or np.any(np.isnan(residuals)):
            return TrackingResults(movement_rows=np.nan, movement_cols=np.nan, tracking_method="least-squares",
                                   tracking_success=False)

        coefficient_adjustment, *_ = np.linalg.lstsq(J, residuals, rcond=None)

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

    if iteration == 50:
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
                                       cross_correlation_coefficient=float(corr),
                                       rmse=np.sqrt(np.mean(np.square(residuals))))
    return tracking_results

def track_cell_lsm_parallelized(central_index: np.ndarray, shm1_name, shm2_name, shape1, shape2, dtype,
                                 tracked_cell_size,
                                 tracking_parameters: TrackingParameters = None, control_search_extents=None,
                                 search_extents=None, initial_shift_values: list = None):
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
    shm2 = multiprocessing.shared_memory.SharedMemory(name=shm2_name)
    
    try:
        shared_image_matrix1 = np.ndarray(shape1, dtype=dtype, buffer=shm1.buf)
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
        tracking_results = track_cell_lsm(track_cell1, search_area2, search_center=search_center,
                                           initial_shift_values=initial_shift_values,
                                          tracking_parameters=tracking_parameters)
        return tracking_results
    finally:
        # Close shared memory handles (don't unlink - only creator does that)
        shm1.close()
        shm2.close()


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
        initial_shift_values = getattr(alignment_parameters, "initial_shift_values", None)
    else:
        if tracking_parameters is None:
            raise ValueError("alignment_tracking=False requires tracking_parameters.")
        movement_cell_size = tracking_parameters.movement_cell_size
        cross_correlation_threshold = tracking_parameters.cross_correlation_threshold_movement
        initial_shift_values = tracking_parameters.initial_shift_values

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
    full_cell_shared_search_extents = None  # Will hold the appropriate extents based on mode

    if alignment_tracking:
        if getattr(alignment_parameters, "control_search_extent_px", None):
            shared_control_search_extents = tuple(int(v) for v in alignment_parameters.control_search_extent_px)
            full_cell_shared_search_extents = shared_control_search_extents
        else:
            raise ValueError("Alignment: control_search_extent_px must be set (tuple posx,negx,posy,negy).")
    else:
        if getattr(tracking_parameters, "search_extent_px", None):
            shared_search_extents = tuple(int(v) for v in tracking_parameters.search_extent_px)
            full_cell_shared_search_extents = shared_search_extents
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
        control_search_extents=shared_control_search_extents if alignment_tracking else None,
        search_extents=shared_search_extents if not alignment_tracking else None,
        initial_shift_values=initial_shift_values,
        tracking_parameters=tracking_parameters
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
    except Exception as e:
        get_console().warning("Failed to assemble multiprocessing. Error: " + str(e))
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
        # Calculate bearing in radians (mathematical convention: 0° = East)
        tracked_pixels["movement_bearing_pixels"] = np.arctan2(-tracked_pixels["movement_row_direction"],
                                                               tracked_pixels["movement_column_direction"])
        tracked_pixels.loc[tracked_pixels['movement_bearing_pixels'] < 0, 'movement_bearing_pixels'] \
            = tracked_pixels['movement_bearing_pixels'] + 2 * np.pi
        # Convert to degrees
        tracked_pixels['movement_bearing_pixels'] = np.degrees(tracked_pixels['movement_bearing_pixels'])
        # Convert from mathematical convention (0° = East) to geographic convention (0° = North)
        tracked_pixels['movement_bearing_pixels'] = (90 - tracked_pixels['movement_bearing_pixels']) % 360
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
    
    # Return the results
    return tracked_pixels_above_cc_threshold
