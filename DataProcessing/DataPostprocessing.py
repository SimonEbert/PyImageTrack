import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
import operator

from ImageTracking import TrackMovement
from CreateGeometries.HandleGeometries import georeference_tracked_points
from CreateGeometries.HandleGeometries import random_points_on_polygon_by_number
from Parameters.TrackingParameters import TrackingParameters
from Plots.MakePlots import plot_movement_of_points
from Plots.MakePlots import plot_raster_and_geometry
from CreateGeometries.HandleGeometries import circular_std_deg
from Parameters.FilterParameters import FilterParameters


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
        image1_matrix=image1_matrix, image2_matrix=image2_matrix,image_transform=image_transform,
        points_to_be_tracked=points, movement_cell_size=tracking_parameters.movement_cell_size,
        movement_tracking_area_size=tracking_parameters.movement_tracking_area_size,
        cross_correlation_threshold=tracking_parameters.cross_correlation_threshold_movement,
        save_columns=["movement_row_direction",
                      "movement_column_direction",
                      "movement_distance_pixels",
                      "movement_bearing_pixels"]
    )
    tracked_control_pixels_valid = tracked_points[tracked_points["movement_row_direction"].notna()]

    if len(tracked_control_pixels_valid) == 0:
        raise ValueError("Was not able to track any points with a cross-correlation higher than the cross-correlation "
                         "threshold. Cross-correlation values were " + str(
            list(tracked_points["correlation_coefficient"])) + " (None-values may signify problems during tracking).")

    print("Used " + str(len(tracked_control_pixels_valid)) + " pixels for LoD calculation.")

    tracked_points = georeference_tracked_points(tracked_control_pixels_valid, image_transform, crs=crs,
                                                 years_between_observations=years_between_observations)

    return tracked_points


def filter_lod_points(tracking_results: gpd.GeoDataFrame, level_of_detection: float) -> gpd.GeoDataFrame:
    """
    Sets the movement distance of all points that fall below the calculated level of detection to 0 and their
        movement bearing to NaN. Returns the respective changed GeoDataFrame.
    Parameters
    ----------
    tracking_results: The GeoDataFrame as obtained from an image tracking
    level_of_detection: The value to filter for. Yearly movement rates below this value will be set to 0 and the
    corresponding movement bearing to NaN.
    Returns
    -------
    tracking_results: GeoDataFrame
        The changed GeoDataFrame
    """
    tracking_results["is_below_LoD"] = False
    tracking_results.loc[tracking_results["movement_distance_per_year"] < level_of_detection, "is_below_LoD"] = True
    tracking_results.loc[tracking_results["is_below_LoD"], "valid"] = False
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

    available_outlier_columns = list(
        set(["is_bearing_difference_outlier", "is_bearing_standard_deviation_outlier",
             "is_movement_rate_difference_outlier", "is_movement_rate_standard_deviation_outlier"])
        & set(tracking_results.columns))

    if available_outlier_columns:
        tracking_results.loc[:, "is_outlier"] = reduce(operator.or_,
                                                       (tracking_results[outlier_column] for outlier_column in
                                                        available_outlier_columns))
    else:
        tracking_results.loc[:, "is_outlier"] = False

    tracking_results_non_outliers = tracking_results.loc[~tracking_results["is_outlier"]]

    if "is_bearing_difference_outlier" not in tracking_results_non_outliers.columns:
        tracking_results["is_bearing_difference_outlier"] = False
    for i in list(tracking_results.index.values):
        list_is_within_current_point = tracking_results_non_outliers.dwithin(tracking_results.geometry[i], inclusion_distance)
        if not any(list_is_within_current_point):
            continue
        surrounding_points = tracking_results_non_outliers.loc[list_is_within_current_point, :]
        average_movement_bearing = np.nanmedian(surrounding_points["movement_bearing_pixels"])

        difference = abs(average_movement_bearing - tracking_results.loc[i, "movement_bearing_pixels"]) % 360
        angular_difference = min(difference, 360 - difference)
        if angular_difference > rotation_threshold:
            tracking_results.loc[i,"is_bearing_difference_outlier"] = True
            tracking_results.loc[i, "valid"] = False
    return tracking_results

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

    available_outlier_columns = list(
        set(["is_bearing_difference_outlier", "is_bearing_standard_deviation_outlier",
             "is_movement_rate_difference_outlier", "is_movement_rate_standard_deviation_outlier"])
        & set(tracking_results.columns))

    if available_outlier_columns:
        tracking_results.loc[:, "is_outlier"] = reduce(operator.or_,
                                                    (tracking_results[outlier_column] for outlier_column in
                                                     available_outlier_columns))
    else:
        tracking_results.loc[:, "is_outlier"] = False

    tracking_results_non_outliers = tracking_results.loc[~tracking_results["is_outlier"]]

    if "is_bearing_standard_deviation_outlier" not in tracking_results_non_outliers.columns:
        tracking_results["is_bearing_standard_deviation_outlier"] = False
    for i in list(tracking_results.index.values):
        list_is_within_current_point = tracking_results_non_outliers.dwithin(tracking_results.geometry[i], inclusion_distance)
        if not any(list_is_within_current_point):
            continue
        surrounding_points = tracking_results_non_outliers.loc[list_is_within_current_point,:]
        movement_bearings = surrounding_points["movement_bearing_pixels"]
        valid_movement_bearings = movement_bearings[~np.isnan(movement_bearings)]
        standard_deviation = circular_std_deg(valid_movement_bearings)
        if standard_deviation > standard_deviation_threshold:
            tracking_results.loc[i,"is_bearing_standard_deviation_outlier"] = True
            tracking_results.loc[i, "valid"] = False
    return tracking_results


def filter_outliers_movement_rate_difference(tracking_results: gpd.GeoDataFrame,
                                                filter_parameters: FilterParameters) -> gpd.GeoDataFrame:
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

    available_outlier_columns = list(
        set(["is_bearing_difference_outlier", "is_bearing_standard_deviation_outlier",
             "is_movement_rate_difference_outlier", "is_movement_rate_standard_deviation_outlier"])
        & set(tracking_results.columns))

    if available_outlier_columns:
        tracking_results.loc[:, "is_outlier"] = reduce(operator.or_,
                                                       (tracking_results[outlier_column] for outlier_column in
                                                        available_outlier_columns))
    else:
        tracking_results.loc[:, "is_outlier"] = False

    tracking_results_non_outliers = tracking_results.loc[~tracking_results["is_outlier"]]

    if "is_movement_rate_difference_outlier" not in tracking_results_non_outliers.columns:
        tracking_results["is_movement_rate_difference_outlier"] = False
    for i in list(tracking_results.index.values):
        list_is_within_current_point = tracking_results_non_outliers.dwithin(tracking_results.geometry[i], inclusion_distance)
        if not any(list_is_within_current_point):
            continue
        surrounding_points = tracking_results_non_outliers.loc[list_is_within_current_point,:]
        average_movement_rate = np.nanmedian(surrounding_points["movement_distance_per_year"])

        if np.abs(average_movement_rate - tracking_results.loc[i,"movement_distance_per_year"]) > movement_rate_threshold:
            tracking_results.loc[i,"is_movement_rate_difference_outlier"] = True
            tracking_results.loc[i, "valid"] = False
    return tracking_results

def filter_outliers_movement_rate_standard_deviation(tracking_results: gpd.GeoDataFrame,
                                                filter_parameters: FilterParameters) -> gpd.GeoDataFrame:
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

    available_outlier_columns = list(
        set(["is_bearing_difference_outlier", "is_bearing_standard_deviation_outlier",
             "is_movement_rate_difference_outlier", "is_movement_rate_standard_deviation_outlier"])
        & set(tracking_results.columns))

    if available_outlier_columns:
        tracking_results.loc[:, "is_outlier"] = reduce(operator.or_,
                                                       (tracking_results[outlier_column] for outlier_column in
                                                        available_outlier_columns))
    else:
        tracking_results.loc[:, "is_outlier"] = False

    tracking_results_non_outliers = tracking_results.loc[~tracking_results["is_outlier"]]

    if "is_movement_rate_standard_deviation_outlier" not in tracking_results_non_outliers.columns:
        tracking_results["is_movement_rate_standard_deviation_outlier"] = False
    for i in list(tracking_results.index.values):
        list_is_within_current_point = tracking_results_non_outliers.dwithin(tracking_results.geometry[i], inclusion_distance)
        if not any(list_is_within_current_point):
            continue
        surrounding_points = tracking_results_non_outliers.loc[list_is_within_current_point,:]
        standard_deviation_movement_rate = np.nanstd(surrounding_points["movement_distance_per_year"])
        if (np.abs(standard_deviation_movement_rate - tracking_results.loc[i,"movement_distance_per_year"]) >
                movement_rate_threshold):
            tracking_results.loc[i,"is_movement_rate_standard_deviation_outlier"] = True
            tracking_results.loc[i, "valid"] = False
    return tracking_results


def filter_outliers_full(tracking_results: gpd.GeoDataFrame, filter_parameters: FilterParameters) -> gpd.GeoDataFrame:



    filtered_tracking_results = filter_outliers_movement_bearing_difference(tracking_results, filter_parameters)
    filtered_tracking_results = filter_outliers_movement_bearing_standard_deviation(
        filtered_tracking_results, filter_parameters)
    filtered_tracking_results = filter_outliers_movement_rate_difference(
        filtered_tracking_results, filter_parameters)
    filtered_tracking_results = filter_outliers_movement_rate_standard_deviation(
        filtered_tracking_results, filter_parameters)
    return filtered_tracking_results

