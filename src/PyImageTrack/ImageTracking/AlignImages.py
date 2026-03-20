import geopandas as gpd
import numpy as np
import scipy
import scipy.ndimage
import sklearn
import rasterio

from ..CreateGeometries.HandleGeometries import grid_points_on_polygon_by_distance
from .TrackMovement import move_indices_from_transformation_matrix
from .TrackMovement import track_movement_lsm
from ..Parameters.AlignmentParameters import AlignmentParameters
from ..ConsoleOutput import get_console
from .ImageInterpolator import ImageInterpolator


def move_image_matrix_from_transformation(image_matrix: np.ndarray, transformation: np.ndarray, target_shape=None):
    """
    Apply a transformation matrix to an image using spline interpolation.
    
    Parameters
    ----------
    image_matrix : np.ndarray
        The image matrix to transform.
    transformation : np.ndarray
        A 2x3 affine transformation matrix.
    target_shape : tuple, optional
        The shape of the output image. If None, uses the input image shape.
    
    Returns
    -------
    np.ndarray
        The transformed image matrix.
    """

    if target_shape is None:
        target_shape = image_matrix.shape

    indices = np.array(np.meshgrid(np.arange(0, target_shape[-2]), np.arange(0, target_shape[-1]))
                       ).T.reshape(-1, 2).T
    moved_indices = move_indices_from_transformation_matrix(transformation, indices)

    # Check if there are NaNs available
    image_contains_nans = np.isnan(image_matrix).any()
    if image_contains_nans:
        # Interpolate NaNs as closest value in the image (for valid spline interpolation)
        nan_mask = np.isnan(image_matrix)
        indices_nearest_values = scipy.ndimage.distance_transform_edt(nan_mask, return_distances=False,
                                                                      return_indices=True)
        image_matrix = image_matrix[tuple(indices_nearest_values)]

    image_matrix_spline = ImageInterpolator(image_matrix)
    
    if len(image_matrix.shape) == 2:
        moved_image_matrix = image_matrix_spline.ev(moved_indices[0, :], moved_indices[1, :]).reshape(
            target_shape)
    else:
        moved_image_matrix = image_matrix_spline.ev(moved_indices[0, :], moved_indices[1, :], shape=target_shape)

    # Put NaN values back into the image at the positions, where they were before the transformation
    if image_contains_nans:
        moved_image_matrix[nan_mask] = np.nan
    return moved_image_matrix


def align_images_lsm_scarce(image1_matrix, image2_matrix, image_transform, reference_area: gpd.GeoDataFrame,
                            alignment_parameters: AlignmentParameters,
                            return_alignment_transformation_matrix: bool = False):
    """
    Aligns two georeferenced images opened in rasterio by matching them in the area given by the reference area.
    Takes only those image sections into account that have a cross-correlation higher than the specified threshold
    (default: 0.95). It moves the second image to match the first image, i.e. after applying this transform one can
    assume the second image to have the same transform as the first one.
    
    Parameters
    ----------
    image1_matrix : np.ndarray
        A raster image matrix to be aligned with the second image.
    image2_matrix : np.ndarray
        A raster image matrix to be aligned with the first image. The alignment takes place via the creation of an image
        transform for the second matrix. The transform of the first image has to be supplied. Thus, the first image is
        assumed to be correctly georeferenced.
    image_transform : Affine
        An object of the class Affine as provided by the rasterio package. The two images are assumed to be aligned
        (for example as a result of align_images) and therefore have the same transform.
    reference_area : gpd.GeoDataFrame
        A single-element GeoDataFrame, containing a polygon for specifying the reference area used for the alignment.
        This is the area, where no movement is suspected.
    alignment_parameters : AlignmentParameters
        The alignment parameters used for alignment, e.g. control_search_size_px
    return_alignment_transformation_matrix:
        If true, the 3d homogeneous transformation matrix aligning image2 to image1 will be returned as a np.array
    Returns
    -------
    list
        [image1_matrix, new_matrix2, tracked_control_pixels_valid]: The two matrices representing the raster image
        as numpy arrays. As the two matrices are aligned, they possess the same transformation. The third element
        is a GeoDataFrame containing the tracked control points used for alignment.
        (alignment_transformation_matrix):
            If return_alignment_transformation_matrix, the used alignment transformation will be returned as a 3x3
            np.array representing a homogeneous transformation matrix.
    Raises
    ------
    ValueError
        If no polygon is provided in the reference area, if no points pass the cross-correlation threshold,
        or if the alignment produces invalid results (all NaN, all zeros, or very low variance).
    """

    if len(reference_area) == 0:
        raise ValueError(
            "No polygon provided in the reference area GeoDataFrame. "
            "Please provide a GeoDataFrame with exactly one element, or set stable_area_filename to 'none' "
            "to use image_bounds minus moving_area as the stable area."
        )

    number_of_control_points = alignment_parameters.number_of_control_points
    maximal_alignment_movement = alignment_parameters.maximal_alignment_movement

    # Estimate grid spacing from polygon area:
    # Spacing ≈ sqrt(area / desired_number_of_points)
    poly_area = float(reference_area.geometry.iloc[0].area)
    if number_of_control_points <= 0 or not np.isfinite(poly_area) or poly_area <= 0:
        raise ValueError("Invalid reference area or number_of_control_points for alignment grid.")

    approx_spacing = np.sqrt(poly_area / float(number_of_control_points))

    reference_area_point_grid = grid_points_on_polygon_by_distance(
        polygon=reference_area,
        distance_of_points=approx_spacing,
        distance_px=None
    )

    tracked_control_pixels = track_movement_lsm(image1_matrix, image2_matrix, image_transform,
                                                points_to_be_tracked=reference_area_point_grid,
                                                alignment_parameters=alignment_parameters, alignment_tracking=True,
                                                save_columns=["movement_row_direction",
                                                              "movement_column_direction",
                                                              "movement_distance_pixels",
                                                              "movement_bearing_pixels",
                                                              "correlation_coefficient"],
                                                task_label="[~] Tracking points for alignment"
                                                )
    tracked_control_pixels_valid = tracked_control_pixels[tracked_control_pixels["movement_row_direction"].notna()]

    if maximal_alignment_movement is not None:
        tracked_control_pixels_valid = tracked_control_pixels_valid[
            tracked_control_pixels_valid["movement_distance_pixels"] <= maximal_alignment_movement]
    if len(tracked_control_pixels_valid) == 0:
        raise ValueError("Was not able to track any points with a cross-correlation higher than the cross-correlation "
                         "threshold. Cross-correlation values were " + str(
            list(tracked_control_pixels[
                     "correlation_coefficient"])) + "\n(None-values may signify problems during tracking).")

    console = get_console()
    total_points = len(tracked_control_pixels)
    valid_points = len(tracked_control_pixels_valid)
    percentage = (valid_points / total_points * 100) if total_points > 0 else 0
    console.success(f"Used {valid_points} pixels for alignment ({percentage:.1f}% of {total_points} points passed threshold).")
    tracked_control_pixels_valid["new_row"] = (tracked_control_pixels_valid["row"]
                                               + tracked_control_pixels_valid["movement_row_direction"])
    tracked_control_pixels_valid["new_column"] = (tracked_control_pixels_valid["column"]
                                                  + tracked_control_pixels_valid["movement_column_direction"])

    linear_model_input = np.column_stack([tracked_control_pixels_valid["row"], tracked_control_pixels_valid["column"]])
    linear_model_output = np.column_stack(
        [tracked_control_pixels_valid["new_row"], tracked_control_pixels_valid["new_column"]])
    
    # Check for NaN values in input/output before fitting
    if np.any(np.isnan(linear_model_input)) or np.any(np.isnan(linear_model_output)):
        # Filter out rows with NaN values
        valid_mask = ~(np.isnan(linear_model_input).any(axis=1) | np.isnan(linear_model_output).any(axis=1))
        linear_model_input = linear_model_input[valid_mask]
        linear_model_output = linear_model_output[valid_mask]
        
        if len(linear_model_input) == 0:
            raise ValueError("All alignment points contain NaN values. This may indicate tracking issues "
                             "with the image pair. Consider checking image quality or adjusting alignment parameters.")
        
        console.warning(f"Filtered out {np.sum(~valid_mask)} points with NaN values from alignment data.")
    
    transformation_linear_model = sklearn.linear_model.LinearRegression()
    transformation_linear_model.fit(linear_model_input, linear_model_output)

    residuals = transformation_linear_model.predict(linear_model_input) - linear_model_output
    tracked_control_pixels_valid["residuals_row"] = residuals[:, 0]
    tracked_control_pixels_valid["residuals_column"] = residuals[:, 1]

    sampling_transformation_matrix = np.array(
        [[transformation_linear_model.coef_[0, 0], transformation_linear_model.coef_[0, 1],
          transformation_linear_model.intercept_[0]],
         [transformation_linear_model.coef_[1, 0], transformation_linear_model.coef_[1, 1],
          transformation_linear_model.intercept_[1]]])

    # Show transformation matrix only in verbose mode (before resampling message)
    if console.verbose:
        console.info("Transformation matrix:")
        matrix_str = str(sampling_transformation_matrix)
        lines = matrix_str.split('\n')
        # Indent matrix lines to align properly
        for i in range(len(lines)):
            lines[i] = '    ' + lines[i]
        for line in lines:
            console.print(line, color='dim')
    
    console.processing("Resampling second image matrix with transformation matrix\n" + str(sampling_transformation_matrix) +
          "\nThis may take some time.")
    moved_image2_matrix = move_image_matrix_from_transformation(image2_matrix, sampling_transformation_matrix,
                                                                target_shape=image1_matrix.shape)
    console.success("Second image resampled.")

    # Validate the aligned image - check for empty or invalid results
    if np.all(np.isnan(moved_image2_matrix)):
        raise ValueError(
            "Alignment produced an image with all NaN values. "
            "This may indicate that the transformation matrix is invalid or the control points are insufficient. "
            "Try adjusting alignment parameters: reduce control_cell_size, lower cross_correlation_threshold_alignment, "
            "or provide a proper stable_area polygon."
        )
    
    if np.all(moved_image2_matrix == 0):
        raise ValueError(
            "Alignment produced an image with all zero values. "
            "This may indicate that the transformation matrix is invalid or the control points are insufficient. "
            "Try adjusting alignment parameters: reduce control_cell_size, lower cross_correlation_threshold_alignment, "
            "or provide a proper stable_area polygon."
        )
    
    # Check if the aligned image has reasonable variance (not constant values)
    if np.var(moved_image2_matrix) < 1e-10:
        raise ValueError(
            "Alignment produced an image with nearly constant values (very low variance). "
            "This may indicate that the transformation matrix is invalid or the control points are insufficient. "
            "Try adjusting alignment parameters: reduce control_cell_size, lower cross_correlation_threshold_alignment, "
            "or provide a proper stable_area polygon."
        )


    if return_alignment_transformation_matrix:
        return [image1_matrix, moved_image2_matrix, tracked_control_pixels_valid, sampling_transformation_matrix]


    return [image1_matrix, moved_image2_matrix, tracked_control_pixels_valid]
