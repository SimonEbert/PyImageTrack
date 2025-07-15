import geopandas as gpd
import numpy as np
import scipy
import sklearn
import matplotlib.pyplot as plt

from ImageTracking.TrackMovement import track_movement_lsm
from CreateGeometries.HandleGeometries import grid_points_on_polygon_by_number_of_points
from ImageTracking.TrackMovement import move_indices_from_transformation_matrix
from Plots.MakePlots import plot_movement_of_points
from Plots.MakePlots import plot_distribution_of_point_movement
from CreateGeometries.HandleGeometries import georeference_tracked_points


def align_images_lsm_scarce(image1_matrix, image2_matrix, image_transform, reference_area: gpd.GeoDataFrame,
                            number_of_control_points: int, cell_size: int = 50, tracking_area_size: int = 60,
                            cross_correlation_threshold: float = 0.8,
                            maximal_alignment_movement: float = None):
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
    number_of_control_points: int
        An approximation of how many points should be created on the reference_area polygon to track possible camera
        position differences. For details see grid_points_on_polygon.
    cell_size: int = 50
        The size of image sections in pixels which are compared during the tracking. See track_movement for details.
    tracking_area_size: int = 60
        The size of the image sections in pixels which are used as a search area during the tracking. Must be greater
        than the parameter cell_size. See track_movement for details.
    cross_correlation_threshold: float = 0.8
        Threshold for which points will be used for aligning the image. Only cells that match with a correlation
        coefficient higher than this value will be considered.
    maximal_alignment_movement: float = None
        Gives the maximal movement in pixels (!) allowed for a single point to be taken into consideration for the
        alignment. If None (the default) no filter is applied.
    Returns
    ----------
    [image1_matrix, new_matrix2]: The two matrices representing the raster image as numpy arrays. As the two matrices
    are aligned, they possess the same transformation. You can therefore assume that
    """

    if len(reference_area) == 0:
        raise ValueError("No polygon provided in the reference area GeoDataFrame. Please provide a GeoDataFrame with "
                         "exactly one element.")
    reference_area_point_grid = grid_points_on_polygon_by_number_of_points(reference_area,
                                                                           number_of_points=number_of_control_points)
    tracked_control_pixels = track_movement_lsm(image1_matrix, image2_matrix, image_transform,
                                                points_to_be_tracked=reference_area_point_grid,
                                                movement_cell_size=cell_size,
                                                movement_tracking_area_size=tracking_area_size,
                                                cross_correlation_threshold=cross_correlation_threshold,
                                                save_columns=["movement_row_direction",
                                                              "movement_column_direction",
                                                              "movement_distance_pixels",
                                                              "movement_bearing_pixels",
                                                              "correlation_coefficient"]
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

    model_row = sklearn.linear_model.LinearRegression()
    model_row.fit(
        np.column_stack([tracked_control_pixels_valid["row"], tracked_control_pixels_valid["column"]]),
                        tracked_control_pixels_valid["new_row"]
    )

    model_column = sklearn.linear_model.LinearRegression()
    model_column.fit(
        np.column_stack([tracked_control_pixels_valid["row"], tracked_control_pixels_valid["column"]]),
                        tracked_control_pixels_valid["new_column"]
    )



    sampling_transformation_matrix = np.array([[model_row.coef_[0],model_row.coef_[1],model_row.intercept_],
                                               [model_column.coef_[0],model_column.coef_[1],model_column.intercept_]])

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

    residuals_row = model_row.predict(np.column_stack([tracked_control_pixels_valid["row"], tracked_control_pixels_valid["column"]])) - tracked_control_pixels_valid["new_row"]
    residuals_column = model_column.predict(np.column_stack([tracked_control_pixels_valid["row"], tracked_control_pixels_valid["column"]])) - tracked_control_pixels_valid["new_column"]

    # fig, ax = plt.subplots()
    # ax.grid(True, which='both')
    #
    # ax.axhline(y=0, color='k')
    # ax.axvline(x=0, color='k')
    # # ax.set_xlim((-1,1))
    # # ax.set_ylim((-1,1))
    # ax.scatter(residuals_row, residuals_column)
    # plt.title("Mean of the residual movement: "
    #       + str(np.mean(np.linalg.norm(np.column_stack((residuals_row, residuals_column)), axis=1)))
    #       + "\nCorrelation coefficient: " + str(np.corrcoef(residuals_row, residuals_column)[0,1]))
    # plt.show()

    if np.abs(np.corrcoef(residuals_row, residuals_column)[0,1]) > 0.6:
        print("Skipping this image pair due to poor alignment (correlation between row and column residuals: " + str(np.corrcoef(residuals_row, residuals_column)[0,1]))
        raise ValueError("Valid alignment was not possible.")




    return [image1_matrix, moved_image2_matrix, tracked_control_pixels_valid]

