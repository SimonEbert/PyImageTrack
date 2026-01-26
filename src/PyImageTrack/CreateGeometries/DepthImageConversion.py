import geopandas as gpd
import pandas as pd
import numpy as np


def calculate_3d_position_from_depth_image(points: np.ndarray, depth_image: np.ndarray,
                                           camera_intrinsics_matrix: np.ndarray,
                                           camera_to_3d_coordinates_transform: np.ndarray = None) -> np.ndarray:
    """
    Calculates 3d displacement vectors given tracking results on an undistorted image and corresponding depth images.
    The depth images are assumed to be given in the exact same coordinate system as the images on which the tracking
    happened (i.e. for non-georeferenced photos undistorted images of the same size as the tracking images)
    Parameters
    ----------
    points: np.ndarray
        An nx2 np.array containing the 2d point positions in the image (given as image coordinates) in the format
        (row, column)
    depth_image: np.ndarray
        The depth image corresponding to the image. Assumed to be an undistorted image in
        the same (pixel) coordinate system as 'points'
    camera_intrinsic_matrix: np.ndarray
        The intrinsic matrix of the camera. Assumed to have the format [[f_x, s, c_x],\n
                                                                        [0, f_y, c_y],\n
                                                                        [0, 0, 1]]
        This matrix is used (if applicable together with 'camera_to_3d_coordinates_transform') to transform points
        in the image coordinate system to 3d points.
    camera_to_3d_coordinates_transform: np.ndarray = None
        A 4x4 homogeneous transformation matrix used to transform points from the camera coordinate system to an
        arbitrary 3d coordinate system. The matrix is assumed to have the format [[R, t]\n
                                                                                  [0, 1]]

    Returns
    -------
    points_transformed: np.ndarray
        An nx3 np.array containing the transformed points in the format (x, y, z). The coordinates correspond to the
        camera coordinate system if camera_to_3d_coordinates_transform is None, i.e. the X- and Y- coordinates align
        with the width (=columns) and height (=rows) of the image and the Z-coordinate is given by the optical axis of
        the camera. Otherwise the 'x', 'y' and 'z' coordinates belong to the coordinate system corresponding to the
        specified 'camera_to_3d_coordinates_transform'.
    """



    # Assume depth image coordinate system corresponds exactly to camera coordinate system
    point_depths = depth_image[points[:, 0].astype(int), points[:, 1].astype(int)]
    points = np.hstack((points, np.ones((len(points),1))))
    points = np.linalg.inv(camera_intrinsics_matrix) @ points.transpose()
    points = points.transpose()
    Z = point_depths
    X = points[:, 1] * Z
    # change sign, s.t. positive y-axis means upwards
    Y = -points[:, 0] * Z
    points_xyz = np.stack([X, Y, Z, np.ones_like(Z)], axis=1).transpose()
    if camera_to_3d_coordinates_transform is not None:
        points_transformed = camera_to_3d_coordinates_transform @ points_xyz
    else:
        points_transformed = points_xyz
    points_transformed = points_transformed[0:3]
    return points_transformed.transpose()


def calculate_displacement_from_depth_images(tracked_points: pd.DataFrame, depth_image_time1: np.ndarray,
                                             depth_image_time2: np.ndarray, camera_intrinsics_matrix: np.ndarray,
                                             years_between_observations: float,
                                             camera_to_3d_coordinates_transform: np.ndarray = None,) -> gpd.GeoDataFrame:
    """
    Calculates 3d displacement vectors given tracking results on an undistorted image and corresponding depth images.
    The depth images are assumed to be given in the exact same coordinate system as the images on which the tracking
    happened (i.e. for non-georeferenced photos undistorted images of the same size as the tracking images)
    Parameters
    ----------
    tracked_points: pd.DataFrame
        The DataFrame containing tracked points with columns "row", "column", "movement_row_direction" and
        "movement_column_direction" as returned by e.g. track_movement_lsm.
    depth_image_time1: np.ndarray
        The depth image corresponding to the first time point of the tracking. Assumed to be an undistorted image in
        the same (pixel) coordinate system as the images on which the tracking was performed
    depth_image_time2: np.ndarray
        The depth image corresponding to the second time point of the tracking. As depth_image_time1 it is assumed
        to be an undistorted image in the same coordinate system as the images on which the tracking was performed.
        It is possible to give the same depth image for these two variables. However, the resulting
        "3d-displacement-column" will then only include the 2d displacement in the plane perpendicular to the
        optical axis of the camera. It will, however, be given normalized to the units of the depth image, i.e.
        perspective effects stemming from varying distances to the camera across the image will be removed.
    camera_intrinsic_matrix: np.ndarray
        The intrinsic matrix of the camera. Assumed to have the format [[f_x, s, c_x],\n
                                                                        [0, f_y, c_y],\n
                                                                        [0, 0, 1]]
        This matrix is used (if applicable together with 'camera_to_3d_coordinates_transform') to transform points
        in the image coordinate system to 3d points.
    years_between_observations: float
        The difference in time (given in years) between the two observations. Used to normalize displacements to
        full years and thereby obtaining comparable velocities
    camera_to_3d_coordinates_transform: np.ndarray = None
        In some cases it might be useful to get the position of tracked points in a specific 3d coordinate system
        (e.g. when georeferencing tracked points afterwards using their 3d positions). In this case there is the
        possibility to specify the respective transformation between the camera coordinate system and a different 3d
        coordinate system such that the "x", "y" and "z" entries of the resulting gpd.GeoDataFrame are given in
        transformed coordinates (in the new coordinate system independent of the image coordinate system). As for
        the 'row' and 'column' data, the 'x', 'y' and 'z' values correspond to the position at the first time point.
        This matrix is assumed to be a 4x4 homogeneous matrix in the format [[R, t]\n
                                                                             [0, 1]]

    Returns
    -------
    georeferenced_tracked_pixels: gpd.GeoDataFrame
        A GeoDataFrame that is 'pseudo-georeferenced': It contains a column '3d_displacement_distance' that gives
        the full displacement between the 2 image time points as well as a column '3d_displacement_distance_per_year'
        that gives a comparable velocity for each point. The units of these columns are determined by the unit of
        the depth image (e.g. m). The DataFrame also contains the 'x', 'y' and 'z' positions corresponding to the
        used 3d coordinate system. If camera_to_3d_coordinates_transform is None, this coordinate system is such
        that the X- and Y- coordinates align with the width (=columns) and height (=rows) of the image and the
        Z-coordinate is given by the optical axis of the camera. Otherwise the 'x', 'y' and 'z' coordinates belong
        to the coordinate system corresponding to the specified 'camera_to_3d_coordinates_transform'.
    """

    points1 = np.array([tracked_points["row"].values,
                        tracked_points["column"].values], dtype=np.float32).transpose()
    points2 = np.array([tracked_points["row"].values + tracked_points["movement_row_direction"].values,
                        tracked_points["column"].values + tracked_points["movement_column_direction"].values],
                       dtype=np.float32).transpose()
    points_3d1 = calculate_3d_position_from_depth_image(points1, depth_image_time1, camera_intrinsics_matrix,
                                                        camera_to_3d_coordinates_transform)
    points_3d2 = calculate_3d_position_from_depth_image(points2, depth_image_time2, camera_intrinsics_matrix,
                                                        camera_to_3d_coordinates_transform)

    points_3d_displacement = points_3d2 - points_3d1
    tracked_points["3d_displacement_distance"] = np.linalg.norm(points_3d_displacement, axis=1)
    rows = tracked_points["row"]
    columns = tracked_points["column"]
    georeferenced_tracked_pixels = gpd.GeoDataFrame(tracked_points, geometry=gpd.points_from_xy(x=columns, y=-rows))
    georeferenced_tracked_pixels["x"] = points_3d1[:,0]
    georeferenced_tracked_pixels["y"] = points_3d1[:,1]
    georeferenced_tracked_pixels["z"] = points_3d1[:,2]

    georeferenced_tracked_pixels["valid"] = True
    georeferenced_tracked_pixels.loc[
        np.isnan(georeferenced_tracked_pixels["3d_displacement_distance"]),
        "valid"] = False
    georeferenced_tracked_pixels["3d_displacement_distance_per_year"] \
        = georeferenced_tracked_pixels["3d_displacement_distance"] / years_between_observations

    return georeferenced_tracked_pixels

