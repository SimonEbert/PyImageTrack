import geopandas as gpd
import numpy as np

from ..ConsoleOutput import get_console
from ..CreateGeometries.HandleGeometries import circular_std_deg
from ..CreateGeometries.HandleGeometries import georeference_tracked_points
from ..ImageTracking import TrackMovement
from ..Parameters.FilterParameters import FilterParameters
from ..Parameters.TrackingParameters import TrackingParameters


def circular_median_deg(angles: np.ndarray) -> float:
    """
    Calculate the circular median of angles in degrees.
    
    The circular median is the angle that minimizes the sum of circular distances
    to all other angles. This is different from the regular median, which doesn't
    account for the circular nature of angles (e.g., 350° and 10° are close).
    
    Parameters
    ----------
    angles : np.ndarray
        Array of angles in degrees. NaN values are ignored.
    
    Returns
    -------
    float
        The circular median in degrees, or NaN if all values are NaN.
    
    Examples
    --------
    >>> circular_median_deg([350, 10, 20])
    0.0  # The median is at 0°, not 10° as regular median would give
    """
    # Remove NaN values
    valid_angles = angles[~np.isnan(angles)]
    if len(valid_angles) == 0:
        return np.nan
    
    # Convert to radians
    radians = np.deg2rad(valid_angles)
    
    # Calculate circular median using vector approach
    # The circular median is the angle of the resultant vector
    sin_sum = np.sum(np.sin(radians))
    cos_sum = np.sum(np.cos(radians))
    
    median_rad = np.arctan2(sin_sum, cos_sum)
    
    # Convert back to degrees and normalize to [0, 360)
    median_deg = np.rad2deg(median_rad) % 360
    
    return median_deg


def mad(values: np.ndarray) -> float:
    """
    Calculate the Median Absolute Deviation (MAD) of values.
    
    MAD is a robust measure of variability that is less sensitive to outliers
    than standard deviation. It is calculated as the median of absolute
    deviations from the median.
    
    Parameters
    ----------
    values : np.ndarray
        Array of values (NaN values are ignored).
    
    Returns
    -------
    float
        The MAD value, or NaN if all values are NaN.
    """
    valid_values = values[~np.isnan(values)]
    if len(valid_values) == 0:
        return np.nan
    
    median_val = np.nanmedian(valid_values)
    mad_val = np.nanmedian(np.abs(valid_values - median_val))
    return mad_val


def calculate_lod_points(image1_matrix: np.ndarray, image2_matrix: np.ndarray, image_transform,
                         points_for_lod_calculation: gpd.GeoDataFrame,
                         tracking_parameters: TrackingParameters, crs, years_between_observations) -> gpd.GeoDataFrame:
    """

    Parameters
    ----------
    image1_matrix
    image2_matrix
    image_transform
    points_for_lod_calculation
    tracking_parameters
    crs
    years_between_observations

    Returns
    -------
    tracked_points: gpd.GeoDataFrame
        The random points which can be used for calculating the LoD.
    """
    points = points_for_lod_calculation
    tracked_points = TrackMovement.track_movement_lsm(
        image1_matrix=image1_matrix, image2_matrix=image2_matrix, image_transform=image_transform,
        points_to_be_tracked=points, tracking_parameters=tracking_parameters, alignment_tracking=False,
        save_columns=["movement_row_direction",
                      "movement_column_direction",
                      "movement_distance_pixels",
                      "movement_bearing_pixels",
                      ],
        task_label="Tracking points for LoD"
    )
    tracked_control_pixels_valid = tracked_points[tracked_points["movement_row_direction"].notna()]

    if len(tracked_control_pixels_valid) == 0:
        # Check if correlation_coefficient column exists before accessing it
        if "correlation_coefficient" in tracked_points.columns:
            cc_values = str(list(tracked_points["correlation_coefficient"]))
        else:
            cc_values = "N/A (column not available)"
        raise ValueError("Was not able to track any points with a cross-correlation higher than the cross-correlation "
                         "threshold. Cross-correlation values were " + cc_values +
                         " (None-values may signify problems during tracking).")

    console = get_console()
    console.info(f"Used {len(tracked_control_pixels_valid)} pixels for LoD calculation.")

    tracked_points = georeference_tracked_points(tracked_control_pixels_valid, image_transform, crs=crs,
                                                 years_between_observations=years_between_observations)

    return tracked_points


def _ensure_bool_col(df, col):
    """
    Ensures a boolean column exists in the DataFrame and returns it as a numpy array.
    
    If the column doesn't exist, it's created with False values. The returned
    array is guaranteed to be of boolean dtype.
    
    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to check/modify.
    col : str
        The column name to ensure exists.
    
    Returns
    -------
    np.ndarray
        A boolean numpy array of the column values.
    """
    if col not in df.columns:
        df[col] = False
    return df[col].astype(bool).to_numpy()


def filter_lod_points(tracking_results: gpd.GeoDataFrame, level_of_detection: float, displacement_column_name: str) -> gpd.GeoDataFrame:
    """
    Sets the movement distance of all points that fall below the calculated level of detection to 0 and their
        movement bearing to NaN. Returns the respective changed GeoDataFrame.
    Parameters
    ----------
    tracking_results: The GeoDataFrame as obtained from an image tracking
    level_of_detection: The value to filter for. Yearly movement rates below this value will be set to 0 and the
    corresponding movement bearing to NaN.
    displacement_column_name: The column name of the displacement column ('movement_distance_per_year' for georeferenced
    images and '3d_displacement_distance_per_year' for non-georeferenced images, for which 3d displacements have been
    calculated.
    Returns
    -------
    tracking_results: GeoDataFrame
        The changed GeoDataFrame
    """

    tracking_results["is_below_LoD"] = False
    tracking_results.loc[tracking_results[displacement_column_name] < level_of_detection, "is_below_LoD"] = True
    tracking_results.loc[tracking_results["is_below_LoD"], "valid"] = False
    return tracking_results

def prepare_tracking_results_for_filtering(tracking_results) -> gpd.GeoDataFrame:

    # --- safety: normalize index and ensure required columns exist ---
    tracking_results = tracking_results.reset_index(drop=True)
    if "valid" not in tracking_results.columns:
        tracking_results["valid"] = True


    available_outlier_columns = list(
        {"is_bearing_difference_outlier", "is_bearing_standard_deviation_outlier",
         "is_movement_rate_difference_outlier", "is_movement_rate_standard_deviation_outlier"}
        & set(tracking_results.columns))

    if available_outlier_columns:
        is_outlier = (
                _ensure_bool_col(tracking_results, "is_bearing_difference_outlier")
                | _ensure_bool_col(tracking_results, "is_bearing_standard_deviation_outlier")
                | _ensure_bool_col(tracking_results, "is_movement_rate_difference_outlier")
                | _ensure_bool_col(tracking_results, "is_movement_rate_standard_deviation_outlier")
        )
        tracking_results["is_outlier"] = is_outlier

    else:
        tracking_results["is_outlier"] = False
    return tracking_results


def filter_outliers_movement_bearing_difference(tracking_results: gpd.GeoDataFrame,
                                                filter_parameters: FilterParameters) -> gpd.GeoDataFrame:
    """
    Filters rotation outliers from the tracking results dataframe. All points that divert more than the given threshold
    (in degrees) from the average movement direction of surrounding points will be removed. The distance up to which
    surrounding points are being considered for the calculation of the average movement direction can be specified (in
    the unit of the crs of the GeoDataFrame tracking_results). Note that in the calculation of the average direction all
    points (also those that are being removed as outliers) are being taken into account. It is therefore advisable to
    use an inclusion distance that is not too small.
    Parameters
    ----------
    tracking_results: gpd.GeoDataFrame
        A GeoDataFrame as obtained from an image tracking.
    filter_parameters: FilterParameters
        An instance of FilterParameters containing the parameters used to filter the results. If the parameters that are
        relevant for this sort of filtering are set to None, no filtering is performed. The value of irrelevant filter
        parameters is ignored.
    Returns
    -------
    tracking_results: GeoDataFrame
        The changed GeoDataFrame
    """
    rotation_threshold = filter_parameters.difference_movement_bearing_threshold
    inclusion_distance = filter_parameters.difference_movement_bearing_moving_window_size
    # check if one of the filter parameters is None and perform no filtering in this case
    if rotation_threshold is None or inclusion_distance is None:
        return tracking_results

    if "is_bearing_difference_outlier" not in tracking_results.columns:
        tracking_results["is_bearing_difference_outlier"] = False
    
    tracking_results_prepared = prepare_tracking_results_for_filtering(tracking_results)

    for i in list(tracking_results.index.values):
        list_is_within_current_point = tracking_results.dwithin(tracking_results.geometry[i],
                                                                     inclusion_distance)
        if not any(list_is_within_current_point):
            continue
        surrounding_points = tracking_results.loc[list_is_within_current_point, :]
        average_movement_bearing = circular_median_deg(surrounding_points["movement_bearing_pixels"].values)

        difference = abs(average_movement_bearing - tracking_results_prepared.loc[i, "movement_bearing_pixels"]) % 360
        angular_difference = min(difference, 360 - difference)
        if angular_difference > rotation_threshold:
            tracking_results_prepared.loc[i, "is_bearing_difference_outlier"] = True
            tracking_results_prepared.loc[i, "is_outlier"] = True
            tracking_results_prepared.loc[i, "valid"] = False
    return tracking_results_prepared


def filter_outliers_movement_bearing_standard_deviation(tracking_results: gpd.GeoDataFrame,
                                                        filter_parameters: FilterParameters) -> gpd.GeoDataFrame:
    """
    Filters rotation outliers from the tracking results dataframe. All points that have neighbouring points such that
    the standard deviation of the movement bearing exceeds the given threshold (specified in filter_parameters), will be
    removed. The distance up to which surrounding points are being considered for the calculation of the average
    movement direction can be specified (in the unit of the crs of the GeoDataFrame tracking_results). Note that in the
    calculation of the average direction all points (also those that are being removed as outliers) are being taken into
    account. It is therefore advisable to use a moving window size that is not too small.
    Parameters
    ----------
    tracking_results: gpd.GeoDataFrame
        A GeoDataFrame as obtained from an image tracking.
    filter_parameters: FilterParameters
        An instance of FilterParameters containing the parameters used to filter the results. If the parameters that are
        relevant for this sort of filtering are set to None, no filtering is performed. The value of irrelevant filter
        parameters is ignored.
    Returns
    -------
    tracking_results: GeoDataFrame
        The changed GeoDataFrame
    """


    standard_deviation_threshold = filter_parameters.standard_deviation_movement_bearing_threshold
    inclusion_distance = filter_parameters.standard_deviation_movement_bearing_moving_window_size
    # check if one of the filter parameters is None and perform no filtering in this case
    if standard_deviation_threshold is None or inclusion_distance is None:
        return tracking_results

    tracking_results_prepared = prepare_tracking_results_for_filtering(tracking_results)

    tracking_results_non_outliers = tracking_results_prepared.loc[~tracking_results_prepared["is_outlier"]].copy()
    tracking_results_non_outliers.reset_index(drop=True, inplace=True)

    if "is_bearing_standard_deviation_outlier" not in tracking_results.columns:
        tracking_results["is_bearing_standard_deviation_outlier"] = False
    for i in list(tracking_results.index.values):
        list_is_within_current_point = tracking_results_non_outliers.dwithin(tracking_results.geometry[i],
                                                                             inclusion_distance)
        if not any(list_is_within_current_point):
            continue
        surrounding_points = tracking_results_non_outliers.loc[list_is_within_current_point, :]
        movement_bearings = surrounding_points["movement_bearing_pixels"]
        valid_movement_bearings = movement_bearings[~np.isnan(movement_bearings)]
        standard_deviation = circular_std_deg(valid_movement_bearings)
        if standard_deviation > standard_deviation_threshold:
            tracking_results_prepared.loc[i, "is_bearing_standard_deviation_outlier"] = True
            tracking_results_prepared.loc[i, "is_outlier"] = True
            tracking_results_prepared.loc[i, "valid"] = False
    return tracking_results_prepared


def filter_outliers_movement_rate_difference(tracking_results: gpd.GeoDataFrame,
                                             filter_parameters: FilterParameters, displacement_column_name: str) -> gpd.GeoDataFrame:
    """
    Filters movement rate outliers from the tracking results dataframe. All points that have neighbouring points whose
     average movement rate deviates more than the given threshold (specified in filter_parameters), will be removed. The
    distance up to which surrounding points are being considered for the calculation of the average movement rate can be
    specified (in the unit of the crs of the GeoDataFrame tracking_results). Note that in the calculation of the average
    movement rate all points (also those that are being removed as outliers) are being taken into account. It is
    therefore advisable to use a moving window size that is not too small.
    Parameters
    ----------
    tracking_results: gpd.GeoDataFrame
        A GeoDataFrame as obtained from an image tracking.
    filter_parameters: FilterParameters
        An instance of FilterParameters containing the parameters used to filter the results. If the parameters that are
        relevant for this sort of filtering are set to None, no filtering is performed. The value of irrelevant filter
        parameters is ignored.
    displacement_column_name: str
        The column name of the displacement column ('movement_distance_per_year' for georeferenced
        images and '3d_displacement_distance_per_year' for non-georeferenced images, for which 3d displacements have been
        calculated).
    Returns
    -------
    tracking_results: GeoDataFrame
        The changed GeoDataFrame
    """

    movement_rate_threshold = filter_parameters.difference_movement_rate_threshold
    inclusion_distance = filter_parameters.difference_movement_rate_moving_window_size
    # check if one of the filter parameters is None and perform no filtering in this case
    if movement_rate_threshold is None or inclusion_distance is None:
        return tracking_results

    tracking_results_prepared = prepare_tracking_results_for_filtering(tracking_results)

    tracking_results_non_outliers = tracking_results_prepared.loc[~tracking_results_prepared["is_outlier"]].copy()
    tracking_results_non_outliers.reset_index(drop=True, inplace=True)

    if "is_movement_rate_difference_outlier" not in tracking_results.columns:
        tracking_results["is_movement_rate_difference_outlier"] = False
    for i in list(tracking_results.index.values):
        list_is_within_current_point = tracking_results_non_outliers.dwithin(tracking_results.geometry[i],
                                                                             inclusion_distance)
        if not any(list_is_within_current_point):
            continue
        surrounding_points = tracking_results_non_outliers.loc[list_is_within_current_point, :]
        average_movement_rate = np.nanmedian(surrounding_points[displacement_column_name])

        if np.abs(average_movement_rate - tracking_results_prepared.loc[
            i, displacement_column_name]) > movement_rate_threshold:
            tracking_results_prepared.loc[i, "is_movement_rate_difference_outlier"] = True
            tracking_results_prepared.loc[i, "is_outlier"] = True
            tracking_results_prepared.loc[i, "valid"] = False
    return tracking_results_prepared


def filter_outliers_movement_rate_standard_deviation(tracking_results: gpd.GeoDataFrame,
                                                     filter_parameters: FilterParameters,
                                                     displacement_column_name: str) -> gpd.GeoDataFrame:
    """
    Filters movement rate outliers from the tracking results dataframe using a Z-score approach. All points that deviate
    more than the specified number of standard deviations (threshold) from the mean of neighbouring points will be removed.
    The distance up to which surrounding points are being considered for the calculation of the mean and standard deviation
    can be specified (in the unit of the crs of the GeoDataFrame tracking_results). Note that in the calculation of the
    mean and standard deviation all points (also those that are being removed as outliers) are being taken into account.
    It is therefore advisable to use a moving window size that is not too small.
    
    Parameters
    ----------
    tracking_results: gpd.GeoDataFrame
        A GeoDataFrame as obtained from an image tracking.
    filter_parameters: FilterParameters
        An instance of FilterParameters containing the parameters used to filter the results. The threshold parameter
        represents the number of standard deviations (Z-score) above which a point is considered an outlier. If the
        parameters that are relevant for this sort of filtering are set to None, no filtering is performed. The value of
        irrelevant filter parameters is ignored.
    displacement_column_name: str
        The column name of the displacement column ('movement_distance_per_year' for georeferenced
        images and '3d_displacement_distance_per_year' for non-georeferenced images, for which 3d displacements have been
        calculated).
    
    Returns
    -------
    tracking_results: GeoDataFrame
        The changed GeoDataFrame
    """

    movement_rate_threshold = filter_parameters.standard_deviation_movement_rate_threshold
    inclusion_distance = filter_parameters.standard_deviation_movement_rate_moving_window_size
    # check if one of the filter parameters is None and perform no filtering in this case
    if movement_rate_threshold is None or inclusion_distance is None:
        return tracking_results

    tracking_results_prepared = prepare_tracking_results_for_filtering(tracking_results)

    tracking_results_non_outliers = tracking_results_prepared.loc[~tracking_results_prepared["is_outlier"]].copy()
    tracking_results_non_outliers.reset_index(drop=True, inplace=True)

    if "is_movement_rate_standard_deviation_outlier" not in tracking_results.columns:
        tracking_results["is_movement_rate_standard_deviation_outlier"] = False
    outlier_count = 0
    for i in list(tracking_results.index.values):
        list_is_within_current_point = tracking_results_non_outliers.dwithin(tracking_results.geometry[i],
                                                                             inclusion_distance)
        if not any(list_is_within_current_point):
            continue
        surrounding_points = tracking_results_non_outliers.loc[list_is_within_current_point, :]
        mean_movement_rate = np.nanmean(surrounding_points[displacement_column_name])
        std_movement_rate = np.nanstd(surrounding_points[displacement_column_name])
        
        # Avoid division by zero - if std is 0, no point can be an outlier
        if std_movement_rate > 0:
            z_score = np.abs(tracking_results_prepared.loc[i, displacement_column_name] - mean_movement_rate) / std_movement_rate
            if z_score > movement_rate_threshold:
                tracking_results_prepared.loc[i, "is_movement_rate_standard_deviation_outlier"] = True
                tracking_results_prepared.loc[i, "is_outlier"] = True
            tracking_results_prepared.loc[i, "valid"] = False
    return tracking_results_prepared


def filter_outliers_depth_change_fraction(tracking_results: gpd.GeoDataFrame,filter_parameters: FilterParameters,
                                          displacement_column_name) -> gpd.GeoDataFrame:
    if displacement_column_name != "3d_displacement_distance_per_year":
        raise ValueError("Trying to filter based on the fraction of depth change compared to 3d displacement, but "
                         "no 3d displacement column is available in the tracking results.")

    maximal_fraction = getattr(filter_parameters, "maximal_fraction_depth_change_of_3d_displacement", None)
    if maximal_fraction is None:
        return tracking_results

    tracking_results_prepared = prepare_tracking_results_for_filtering(tracking_results)

    tracking_results_prepared["is_depth_fraction_outlier"] = (
            maximal_fraction <
            tracking_results_prepared["depth_change"] / tracking_results_prepared["3d_displacement_distance_per_year"])
    tracking_results_prepared["is_outlier"] = (tracking_results_prepared["is_outlier"]
                                               | tracking_results_prepared["is_depth_fraction_outlier"])
    tracking_results_prepared["valid"] = (tracking_results_prepared["valid"]
                                          & ~tracking_results_prepared["is_depth_fraction_outlier"])
    return tracking_results_prepared


def filter_outliers_movement_rate_mad(tracking_results: gpd.GeoDataFrame,
                                       filter_parameters: FilterParameters,
                                       displacement_column_name: str) -> gpd.GeoDataFrame:
    """
    Filters movement rate outliers from the tracking results dataframe using a modified Z-score approach. All points that deviate
    more than the specified number of MADs (threshold) from the median of neighbouring points will be removed.
    The distance up to which surrounding points are being considered for the calculation of the median and MAD
    can be specified (in the unit of the crs of the GeoDataFrame tracking_results). Note that in the calculation of the
    median and MAD all points (also those that are being removed as outliers) are being taken into account.
    It is therefore advisable to use a moving window size that is not too small.
    
    Parameters
    ----------
    tracking_results: gpd.GeoDataFrame
        A GeoDataFrame as obtained from an image tracking.
    filter_parameters: FilterParameters
        An instance of FilterParameters containing the parameters used to filter the results. The threshold parameter
        represents the number of MADs (modified Z-score) above which a point is considered an outlier. If the
        parameters that are relevant for this sort of filtering are set to None, no filtering is performed. The value of
        irrelevant filter parameters is ignored.
    displacement_column_name: str
        The column name of the displacement column ('movement_distance_per_year' for georeferenced
        images and '3d_displacement_distance_per_year' for non-georeferenced images, for which 3d
        displacements have been calculated).
    
    Returns
    -------
    tracking_results: GeoDataFrame
        The changed GeoDataFrame
    """
    # --- safety: normalize index and ensure required columns exist ---
    tracking_results = tracking_results.reset_index(drop=True)
    if "valid" not in tracking_results.columns:
        tracking_results["valid"] = True

    movement_rate_threshold = filter_parameters.standard_deviation_movement_rate_threshold
    inclusion_distance = filter_parameters.standard_deviation_movement_rate_moving_window_size
    # check if one of the filter parameters is None and perform no filtering in this case
    if movement_rate_threshold is None or inclusion_distance is None:
        return tracking_results

    available_outlier_columns = list(
        {"is_bearing_difference_outlier", "is_bearing_standard_deviation_outlier",
         "is_movement_rate_difference_outlier", "is_movement_rate_standard_deviation_outlier"}
        & set(tracking_results.columns))

    if available_outlier_columns:
        is_outlier = (
                _ensure_bool_col(tracking_results, "is_bearing_difference_outlier")
                | _ensure_bool_col(tracking_results, "is_bearing_standard_deviation_outlier")
                | _ensure_bool_col(tracking_results, "is_movement_rate_difference_outlier")
                | _ensure_bool_col(tracking_results, "is_movement_rate_standard_deviation_outlier")
        )
        tracking_results["is_outlier"] = is_outlier

    else:
        tracking_results["is_outlier"] = False

    tracking_results_non_outliers = tracking_results.loc[~tracking_results["is_outlier"]].copy()
    tracking_results_non_outliers.reset_index(drop=True, inplace=True)

    if "is_movement_rate_standard_deviation_outlier" not in tracking_results.columns:
        tracking_results["is_movement_rate_standard_deviation_outlier"] = False
    outlier_count = 0
    for i in list(tracking_results.index.values):
        list_is_within_current_point = tracking_results_non_outliers.dwithin(tracking_results.geometry[i],
                                                                             inclusion_distance)
        if not any(list_is_within_current_point):
            continue
        surrounding_points = tracking_results_non_outliers.loc[list_is_within_current_point, :]
        median_movement_rate = np.nanmedian(surrounding_points[displacement_column_name])
        mad_movement_rate = mad(surrounding_points[displacement_column_name])
        
        # Avoid division by zero - if MAD is 0, no point can be an outlier
        if mad_movement_rate > 0:
            modified_z_score = np.abs(tracking_results.loc[i, displacement_column_name] - median_movement_rate) / mad_movement_rate
            if modified_z_score > movement_rate_threshold:
                tracking_results.loc[i, "is_movement_rate_standard_deviation_outlier"] = True
                tracking_results.loc[i, "valid"] = False
    return tracking_results


# OLD: Function using mean and standard deviation (kept for reference)
# def filter_outliers_movement_rate_standard_deviation(tracking_results: gpd.GeoDataFrame,
#                                                      filter_parameters: FilterParameters,
#                                                      displacement_column_name: str) -> gpd.GeoDataFrame:
#     """
#     Filters movement rate outliers from the tracking results dataframe using a Z-score approach. All points that deviate
#     more than the specified number of standard deviations (threshold) from the mean of neighbouring points will be removed.
#     The distance up to which surrounding points are being considered for the calculation of the mean and standard deviation
#     can be specified (in the unit of the crs of the GeoDataFrame tracking_results). Note that in the calculation of the
#     mean and standard deviation all points (also those that are being removed as outliers) are being taken into account.
#     It is therefore advisable to use a moving window size that is not too small.
#
#     Parameters
#     ----------
#     tracking_results: gpd.GeoDataFrame
#         A GeoDataFrame as obtained from an image tracking.
#     filter_parameters: FilterParameters
#         An instance of FilterParameters containing the parameters used to filter the results. The threshold parameter
#         represents the number of standard deviations (Z-score) above which a point is considered an outlier. If the
#         parameters that are relevant for this sort of filtering are set to None, no filtering is performed. The value of
#         irrelevant filter parameters is ignored.
#     displacement_column_name: str
#         The column name of the displacement column ('movement_distance_per_year' for georeferenced
#         images and '3d_displacement_distance_per_year' for non-georeferenced images, for which 3d
#         displacements have been calculated).
#
#     Returns
#     -------
#     tracking_results: GeoDataFrame
#         The changed GeoDataFrame
#     """
#     # --- safety: normalize index and ensure required columns exist ---
#     tracking_results = tracking_results.reset_index(drop=True)
#     if "valid" not in tracking_results.columns:
#         tracking_results["valid"] = True
#
#     movement_rate_threshold = filter_parameters.standard_deviation_movement_rate_threshold
#     inclusion_distance = filter_parameters.standard_deviation_movement_rate_moving_window_size
#     # check if one of the filter parameters is None and perform no filtering in this case
#     if movement_rate_threshold is None or inclusion_distance is None:
#         return tracking_results
#
#     available_outlier_columns = list(
#         {"is_bearing_difference_outlier", "is_bearing_standard_deviation_outlier",
#          "is_movement_rate_difference_outlier", "is_movement_rate_standard_deviation_outlier"}
#         & set(tracking_results.columns))
#
#     if available_outlier_columns:
#         is_outlier = (
#                 _ensure_bool_col(tracking_results, "is_bearing_difference_outlier")
#                 | _ensure_bool_col(tracking_results, "is_bearing_standard_deviation_outlier")
#                 | _ensure_bool_col(tracking_results, "is_movement_rate_difference_outlier")
#                 | _ensure_bool_col(tracking_results, "is_movement_rate_standard_deviation_outlier")
#         )
#         tracking_results["is_outlier"] = is_outlier
#
#     else:
#         tracking_results["is_outlier"] = False
#
#     tracking_results_non_outliers = tracking_results.loc[~tracking_results["is_outlier"]].copy()
#     tracking_results_non_outliers.reset_index(drop=True, inplace=True)
#
#     if "is_movement_rate_standard_deviation_outlier" not in tracking_results.columns:
#         tracking_results["is_movement_rate_standard_deviation_outlier"] = False
#     outlier_count = 0
#     for i in list(tracking_results.index.values):
#         list_is_within_current_point = tracking_results_non_outliers.dwithin(tracking_results.geometry[i],
#                                                                              inclusion_distance)
#         if not any(list_is_within_current_point):
#             continue
#         surrounding_points = tracking_results_non_outliers.loc[list_is_within_current_point, :]
#         mean_movement_rate = np.nanmean(surrounding_points[displacement_column_name])
#         std_movement_rate = np.nanstd(surrounding_points[displacement_column_name])
#
#         # Avoid division by zero - if std is 0, no point can be an outlier
#         if std_movement_rate > 0:
#             z_score = np.abs(tracking_results.loc[i, displacement_column_name] - mean_movement_rate) / std_movement_rate
#             if z_score > movement_rate_threshold:
#                 tracking_results.loc[i, "is_movement_rate_standard_deviation_outlier"] = True
#                 tracking_results.loc[i, "valid"] = False
#     return tracking_results


def filter_outliers_full(tracking_results: gpd.GeoDataFrame, filter_parameters: FilterParameters,
                         displacement_column_name: str) -> gpd.GeoDataFrame:
    """
    Apply all outlier filters independently on the unfiltered dataset and combine masks at the end.

    Steps (conceptually parallel):
    1. Create an untouched copy of the input GeoDataFrame.
    2. Run each filter on its own copy of that base to extract only the respective outlier mask.
    3. OR the masks and write both the individual masks and the aggregated validity flag back to the base copy.

    Parameters
    ----------
    tracking_results : gpd.GeoDataFrame
        A GeoDataFrame as obtained from an image tracking.
    filter_parameters : FilterParameters
        An instance of FilterParameters containing the parameters used to filter the results.
    displacement_column_name : str
        The column name of the displacement column ('movement_distance_per_year' for georeferenced
        images and '3d_displacement_distance_per_year' for non-georeferenced images, for which 3d
        displacements have been calculated).
    
    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame with individual outlier flags and an aggregated `valid` column based on the OR-combined mask.
    """
    base_df = tracking_results.reset_index(drop=True).copy()
    if "valid" not in base_df.columns:
        base_df["valid"] = True

    def _mask_from(df: gpd.GeoDataFrame, col: str) -> np.ndarray:
        if col not in df.columns:
            return np.zeros(len(df), dtype=bool)
        return np.where(df[col].isna(), False, df[col]).astype(bool)

    bd_df = filter_outliers_movement_bearing_difference(base_df.copy(), filter_parameters)
    bsd_df = filter_outliers_movement_bearing_standard_deviation(base_df.copy(), filter_parameters)
    md_df = filter_outliers_movement_rate_difference(base_df.copy(), filter_parameters, displacement_column_name)
    msd_df = filter_outliers_movement_rate_mad(base_df.copy(), filter_parameters, displacement_column_name)

    mask_bd = _mask_from(bd_df, "is_bearing_difference_outlier")
    mask_bsd = _mask_from(bsd_df, "is_bearing_standard_deviation_outlier")
    mask_md = _mask_from(md_df, "is_movement_rate_difference_outlier")
    mask_msd = _mask_from(msd_df, "is_movement_rate_standard_deviation_outlier")

    combined_outlier_mask = mask_bd | mask_bsd | mask_md | mask_msd

    base_df["is_bearing_difference_outlier"] = mask_bd
    base_df["is_bearing_standard_deviation_outlier"] = mask_bsd
    base_df["is_movement_rate_difference_outlier"] = mask_md
    base_df["is_movement_rate_standard_deviation_outlier"] = mask_msd

    base_df.loc[combined_outlier_mask, "valid"] = False

    return base_df
