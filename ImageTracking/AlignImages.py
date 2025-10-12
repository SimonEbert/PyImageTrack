import geopandas as gpd
import numpy as np
import scipy
import sklearn

from ImageTracking.TrackMovement import track_movement_lsm
from ImageTracking.TrackMovement import move_indices_from_transformation_matrix
from Parameters.AlignmentParameters import AlignmentParameters
from Plots.MakePlots import plot_movement_of_points
from Plots.MakePlots import plot_distribution_of_point_movement
from CreateGeometries.HandleGeometries import grid_points_on_polygon_by_distance



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
        An object of the class Affine as provided by the rasterio package. The two images are assumed to be aligned
        (for example as a result of align_images) and therefore have the same transform.
    reference_area : gpd.GeoDataFrame
        A single-element GeoDataFrame, containing a polygon for specifying the reference area used for the alignment.
        This is the area, where no movement is suspected.
    alignment_parameters: AlignmentParameters
        The alignment parameters used for alignment, e.g. control_search_size_px
    Returns
    ----------
    [image1_matrix, new_matrix2]: The two matrices representing the raster image as numpy arrays. As the two matrices
    are aligned, they possess the same transformation. You can therefore assume that
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
        tracked_control_pixels_valid = tracked_control_pixels_valid[tracked_control_pixels_valid["movement_distance_pixels"] <= maximal_alignment_movement]
    if len(tracked_control_pixels_valid) == 0:
        raise ValueError("Was not able to track any points with a cross-correlation higher than the cross-correlation "
                         "threshold. Cross-correlation values were " + str(
            list(tracked_control_pixels["correlation_coefficient"])) + "\n(None-values may signify problems during tracking).")

    print("Used " + str(len(tracked_control_pixels_valid)) + " pixels for alignment.")
    tracked_control_pixels_valid["new_row"] = (tracked_control_pixels_valid["row"]
                                         + tracked_control_pixels_valid["movement_row_direction"])
    tracked_control_pixels_valid["new_column"] = (tracked_control_pixels_valid["column"]
                                            + tracked_control_pixels_valid["movement_column_direction"])

    linear_model_input = np.column_stack([tracked_control_pixels_valid["row"], tracked_control_pixels_valid["column"]])
    linear_model_output = np.column_stack([tracked_control_pixels_valid["new_row"],tracked_control_pixels_valid["new_column"]])
    transformation_linear_model = sklearn.linear_model.LinearRegression()
    transformation_linear_model.fit(linear_model_input, linear_model_output)


    residuals = transformation_linear_model.predict(linear_model_input) - linear_model_output
    tracked_control_pixels_valid["residuals_row"] = residuals[:,0]
    tracked_control_pixels_valid["residuals_column"] = residuals[:,1]

    # fig, ax = plt.subplots()
    # ax.grid(True, which='both')
    # ax.axhline(y=0, color='k')
    # ax.axvline(x=0, color='k')
    # # ax.set_xlim((-1,1))
    # # ax.set_ylim((-1,1))
    # plt.scatter(residuals[:, 0], residuals[:, 1])
    # plt.title("Model_score:" + str(transformation_linear_model.score(linear_model_input, linear_model_output))
    #           + "\nResidual correlation" + str(np.corrcoef(residuals[:, 0], residuals[:, 1])[0, 1]))
    # plt.show()

    # if np.abs(np.corrcoef(residuals[:,0], residuals[:,1])[0,1]) > 0.7:
    #     print("Skipping this image pair due to poor alignment (correlation between row and column residuals: " + str(np.corrcoef(residuals[:,0], residuals[:,1])[0,1]) + ")")
    #     raise ValueError("Valid alignment was not possible.")


    sampling_transformation_matrix = np.array([[transformation_linear_model.coef_[0,0],transformation_linear_model.coef_[0,1],transformation_linear_model.intercept_[0]],
                                              [transformation_linear_model.coef_[1,0],transformation_linear_model.coef_[1,1],transformation_linear_model.intercept_[1]]])



    indices = np.array(np.meshgrid(np.arange(0, image1_matrix.shape[0]), np.arange(0, image1_matrix.shape[1]))
                       ).T.reshape(-1, 2).T
    moved_indices = move_indices_from_transformation_matrix(sampling_transformation_matrix, indices)
    image2_matrix_spline = scipy.interpolate.RectBivariateSpline(np.arange(0, image2_matrix.shape[0]),
                                                                 np.arange(0, image2_matrix.shape[1]),
                                                                 image2_matrix)
    print("Resampling the second image matrix with transformation matrix\n" + str(sampling_transformation_matrix) +
          "\nThis may take some time.")
    moved_image2_matrix = image2_matrix_spline.ev(moved_indices[0, :], moved_indices[1, :]).reshape(
        image1_matrix.shape)


    return [image1_matrix, moved_image2_matrix, tracked_control_pixels_valid]

