import logging

import geopandas as gpd
import numpy as np
import scipy
import sklearn
import rasterio

from ..CreateGeometries.HandleGeometries import grid_points_on_polygon_by_distance
from .TrackMovement import move_indices_from_transformation_matrix
from .TrackMovement import track_movement_lsm
from ..Parameters.AlignmentParameters import AlignmentParameters
from .ImageInterpolator import ImageInterpolator


def align_images_lsm_scarce(image1_matrix, image2_matrix, image_transform, reference_area: gpd.GeoDataFrame,
                            alignment_parameters: AlignmentParameters):
    """
    Aligns two georeferenced images opened in rasterio by matching them in the area given by the reference area.
    Takes only those image sections into account that have a cross-correlation higher than the specified threshold
    (default: 0.95). It moves the second image to match the first image, i.e. after applying this transform one can
    assume the second image to have the same transform as the first one.
    Parameters
    ----------
    image1_matrix :
        A raster image matrix to be aligned with the second image.
    image2_matrix :
        A raster image matrix to be aligned with the first image. The alignment takes place via the creation of an image
        transform for the second matrix. The transform of the first image has to be supplied. Thus, the first image is
        assumed to be correctly georeferenced.
    image_transform :
        An object of the class Affine as provided by the rasterio package specifying the transform for the first image.
    reference_area : gpd.GeoDataFrame
        A single-element GeoDataFrame, containing a polygon for specifying the reference area used for the alignment.
        This is the area, where no movement is suspected.
    alignment_parameters: AlignmentParameters
        The alignment parameters used for alignment, e.g. control_search_size_px
    Returns
    ----------
    [image1_matrix, new_matrix2]: The two matrices representing the raster image as numpy arrays. As the two matrices
    are aligned, they possess the same transformation. You can therefore assume that image_transform (the transform for
    the first image) corresponds to both images.
    """

    if len(reference_area) == 0:
        raise ValueError("No polygon provided in the reference area GeoDataFrame. Please provide a GeoDataFrame with "
                         "exactly one element.")

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
                                                task_label="Tracking points for alignment"
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

    print("Used " + str(len(tracked_control_pixels_valid)) + " pixels for alignment.")
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
        
        logging.warning(f"Filtered out {np.sum(~valid_mask)} points with NaN values from alignment data.")
    
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

    indices = np.array(np.meshgrid(np.arange(0, image1_matrix.shape[-2]), np.arange(0, image1_matrix.shape[-1]))
                       ).T.reshape(-1, 2).T
    moved_indices = move_indices_from_transformation_matrix(sampling_transformation_matrix, indices)
    image2_matrix_spline = ImageInterpolator(image2_matrix)
    print("Resampling the second image matrix with transformation matrix\n" + str(sampling_transformation_matrix) +
          "\nThis may take some time.")
    moved_image2_matrix = image2_matrix_spline.ev(moved_indices[0, :], moved_indices[1, :]).reshape(
        image1_matrix.shape)

    return [image1_matrix, moved_image2_matrix, tracked_control_pixels_valid]
