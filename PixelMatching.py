import geopandas as gpd
import pandas as pd
import rasterio.mask
import rasterio.plot
import numpy as np
import scipy.interpolate
import shapely
import scipy.optimize as opt
from CreateGeometries.HandleGeometries import grid_points_on_polygon
from CreateGeometries.HandleGeometries import get_raster_indices_from_points
from rasterio import features
from CreateGeometries.HandleGeometries import get_overlapping_area


def get_submatrix_symmetric(central_index, shape, matrix):
    """Makes even given shapes one number smaller to ascertain there is one unique central index"""
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
    moved_cell_matrix = move_cell_rotation_approach(coefficients=coefficients,
                                                    tracked_cell_matrix_shape=tracked_cell_matrix.shape,
                                                    interpolator_search_cell=interpolator_search_cell,
                                                    indices=indices)
    return np.sum((tracked_cell_matrix - moved_cell_matrix) ** 2)


def track_cell_lsm(tracked_cell_matrix: np.ndarray, search_cell_matrix: np.ndarray, initial_shift_values=None,
                   return_full_coefficients: bool = False):
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
                                       method="Powell")

        # retrieve optimal transformation coefficients
        t1, t2, t3, t4, b1, b2 = solution_global.x
        # What to do if the optimization did not work
        if not solution_global.success:
            print("did not converge")
            return [np.nan, np.nan]
        transformation_matrix = np.array([[t1, t2], [t3, t4]])

        if np.abs(np.linalg.det(transformation_matrix) - 1) >= 0.2:
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


def get_tracked_pixels_square(tracked_pixels, central_pixel_coordinates):
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


def retrack_wrong_matching_pixels(tracked_pixels, track_matrix, search_matrix, cell_size: int, tracking_area_size: int,
                                  fallback_on_cross_correlation: bool = False):

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
                                                             movement_column_direction_neighbours_mean])

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


def track_movement(image1_matrix, image2_matrix, image_transform, tracking_area: gpd.GeoDataFrame,
                   number_of_tracked_points: int, cell_size: int = 40,
                   tracking_area_size: int = 50, tracking_method: str = "lsm", remove_outliers: bool = True,
                   retry_matching: bool = True):
    """Creates roughly number_of_tracked_points points on the tracking area and  tracks these points between image1 and
    image2. The size of the cell, which is tracked for each point as well as the search area can be adjusted. Takes two aligned matrices image1 and image 2 and the respective transform"""

    # get grid of tracked points
    points = grid_points_on_polygon(polygon=tracking_area, number_of_points=number_of_tracked_points)

    # get the matrix indices for every point
    rows, cols = get_raster_indices_from_points(points, image_transform)
    tracked_points_pixels = np.array([rows, cols]).transpose()

    # initialize dataframe for the resulting tracked points
    tracked_pixels = pd.DataFrame()

    # Loop over all points
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
            match = track_cell_lsm(track_cell1, search_area2)
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

    # calculate the movement distance in pixels from the movement along the axes for the whole results dataframe
    tracked_pixels.insert(4, "movement_distance_pixels",
                          np.linalg.norm(tracked_pixels.loc[:, ["movement_row_direction", "movement_column_direction"]],
                                         axis=1))

    # Perform postprocessing, such as outlier removal and second matching attempt
    if remove_outliers:
        tracked_pixels = remove_outlying_tracked_pixels(tracked_pixels, from_rotation=True, from_velocity=True)
    if retry_matching & (tracking_method == "lsm"):
        tracked_pixels = retrack_wrong_matching_pixels(tracked_pixels, image1_matrix, image2_matrix, cell_size,
                                                       tracking_area_size, fallback_on_cross_correlation=False)
        if remove_outliers:
            tracked_pixels = remove_outlying_tracked_pixels(tracked_pixels, from_rotation=True, from_velocity=True)

    return tracked_pixels


def align_images(image1, image2, reference_area: gpd.GeoDataFrame, number_of_control_points: int, cell_size: int = 40,
                 tracking_area_size: int = 60, select_bands=None, image_alignment_via_lsm: bool = False):
    """
    Aligns two georeferenced images opened in rasterio by matching them in the area given by the reference area.

    Parameters
    ----------
    image1 :
        A raster image opened in the rasterio package to be aligned with the second image.
    image2 :
        A raster image opened in the rasterio package to be aligned with the first image. The alignment takes
        place via an adjustment of the transform of the second raster image, which means that the first image is
        assumed to be correctly georeferenced.
    reference_area : gpd.GeoDataFrame
        A single-element GeoDataFrame, containing a polygon for specifying the reference area used for the alignment.
        This is the area, where no movement is suspected.

    """

    # crop the images to the same extent
    [image1_matrix, image_transform], [image2_matrix, _] = get_overlapping_area(image1, image2)

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
