import geopandas as gpd
import numpy as np
import scipy
import sklearn

from ImageTracking.TrackMovement import track_movement_lsm
from CreateGeometries.HandleGeometries import grid_points_on_polygon_by_number_of_points
from ImageTracking.TrackMovement import move_indices_from_transformation_matrix


def align_images_lsm_scarce(image1_matrix, image2_matrix, image_transform, reference_area: gpd.GeoDataFrame,
                            number_of_control_points: int, cell_size: int = 50, tracking_area_size: int = 60,
                            cross_correlation_threshold: float = 0.95
                            ):
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
    cross_correlation_threshold: float = 0.95
        Threshold for which points will be used for aligning the image. Only cells that match with a correlation
        coefficient higher than this value will be considered.
    Returns
    ----------
    [image1_matrix, new_matrix2]: The two matrices representing the raster image as numpy arrays. As the two matrices
    are aligned, they possess the same transformation. You can therefore assume that
    """

    reference_area_point_grid = grid_points_on_polygon_by_number_of_points(reference_area,
                                                                           number_of_points=number_of_control_points)

    tracked_control_pixels = track_movement_lsm(image1_matrix, image2_matrix, image_transform,
                                                points_to_be_tracked=reference_area_point_grid,
                                                movement_cell_size=cell_size,
                                                movement_tracking_area_size=tracking_area_size,
                                                save_columns=["movement_row_direction",
                                                              "movement_column_direction",
                                                              "movement_distance_pixels",
                                                              "correlation_coefficient"]
                                                )
    # tracked_control_pixels = tracked_control_pixels[
    #     tracked_control_pixels["correlation_coefficient"] > cross_correlation_threshold]
    tracked_control_pixels = tracked_control_pixels[tracked_control_pixels["movement_row_direction"].notna()]

    tracked_control_pixels["new_row"] = (tracked_control_pixels["row"]
                                         + tracked_control_pixels["movement_row_direction"])
    tracked_control_pixels["new_column"] = (tracked_control_pixels["column"]
                                            + tracked_control_pixels["movement_column_direction"])


    model_row = sklearn.linear_model.LinearRegression().fit(
        np.column_stack([tracked_control_pixels["row"], tracked_control_pixels["column"]]),
                        tracked_control_pixels["new_row"]
    )

    model_column = sklearn.linear_model.LinearRegression().fit(
        np.column_stack([tracked_control_pixels["row"], tracked_control_pixels["column"]]),
                        tracked_control_pixels["new_column"]
    )

    transformation_matrix = np.array([[model_row.coef_[0],model_row.coef_[1],model_row.intercept_],
                                     [model_column.coef_[0],model_column.coef_[1],model_column.intercept_]])

    indices = np.array(np.meshgrid(np.arange(0, image1_matrix.shape[0]), np.arange(0, image1_matrix.shape[1]))
                       ).T.reshape(-1, 2).T
    moved_indices = move_indices_from_transformation_matrix(transformation_matrix, indices)

    image2_matrix_spline = scipy.interpolate.RectBivariateSpline(np.arange(0, image2_matrix.shape[0]),
                                                                 np.arange(0, image2_matrix.shape[1]),
                                                                 image2_matrix)
    print("Resampling the second image matrix with transformation matrix\n" + str(transformation_matrix) +
          "\nThis may take some time.")
    moved_image2_matrix = image2_matrix_spline.ev(moved_indices[0, :], moved_indices[1, :]).reshape(
        image1_matrix.shape)
    return [image1_matrix, moved_image2_matrix]

