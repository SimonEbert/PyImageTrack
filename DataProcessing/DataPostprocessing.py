import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt

from ImageTracking import TrackMovement
from CreateGeometries.HandleGeometries import georeference_tracked_points
from CreateGeometries.HandleGeometries import random_points_on_polygon_by_number
from dataloader.TrackingParameters import TrackingParameters
from Plots.MakePlots import plot_movement_of_points
from Plots.MakePlots import plot_raster_and_geometry


def calculate_lod(image1_matrix: np.ndarray, image2_matrix: np.ndarray, image_transform,
                  reference_area: gpd.GeoDataFrame, number_of_reference_points,
                  tracking_parameters: TrackingParameters, crs, years_between_observations) -> gpd.GeoDataFrame:
    """

    Parameters
    ----------
    image1_matrix
    image2_matrix
    image_transform
    reference_area
    number_of_reference_points
    tracking_parameters
    crs
    years_between_observations

    Returns
    -------
    tracked_points: gpd.GeoDataFrame
        The random points which can be used for calculating the LoD.
    """
    points = random_points_on_polygon_by_number(reference_area, number_of_points=number_of_reference_points)
    tracked_points = TrackMovement.track_movement_lsm(
        image1_matrix=image1_matrix, image2_matrix=image2_matrix,image_transform=image_transform,
        points_to_be_tracked=points, movement_cell_size=tracking_parameters.movement_cell_size,
        movement_tracking_area_size=tracking_parameters.movement_tracking_area_size,
        save_columns=["movement_row_direction",
                      "movement_column_direction",
                      "movement_distance_pixels",
                      "movement_bearing_pixels",
                      "correlation_coefficient"]
    )
    tracked_points = tracked_points[
        tracked_points["correlation_coefficient"] > tracking_parameters.cross_correlation_threshold]
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
    return tracking_results


def filter_rotation_outliers(tracking_results: gpd.GeoDataFrame, rotation_threshold: float, inclusion_distance: float)\
        -> gpd.GeoDataFrame:
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
    rotation_threshold: float
        The threshold up to which deviations from the average are considered non-outliers (given in degrees).
    inclusion_distance: float
        The distance up to which points are being taken into account for the average movement direction calculation (
        given in terms of the unit of the crs)
    Returns
    -------
    tracking_results: GeoDataFrame
        The changed GeoDataFrame
    """
    tracking_results["is_rotation_outlier"] = False
    for i in range(len(tracking_results)):
        surrounding_points = tracking_results.loc[tracking_results.dwithin(tracking_results.geometry[i], inclusion_distance),:]
        average_movement_bearing = np.nanmean(surrounding_points["movement_bearing_pixels"])
        # ToDo: Calculate angular difference correctly
        # angular_difference = np.minimum(())
        if np.abs(average_movement_bearing - tracking_results.loc[i,"movement_bearing_pixels"]) > rotation_threshold:
            tracking_results.loc[i,"is_rotation_outlier"] = True
    return tracking_results


def filter_velocity_outliers(tracking_results: gpd.GeoDataFrame, velocity_threshold: float, inclusion_distance: float)\
        -> gpd.GeoDataFrame:
    """
    Filters velocity outliers from the tracking results dataframe. All points that divert more than the given threshold
    (given in the unit of "movement_distance_per_year") from the average velocity of surrounding points will be removed.
    The distance up to which surrounding points are being considered for the calculation of the average velocity can be
    specified (in the unit of the crs of the GeoDataFrame tracking_results). Note that in the calculation of the average
    velocity all points (also those that are being removed as outliers) are being taken into account. It is therefore
    advisable to use an inclusion distance that is not too small.
    Parameters
    ----------
    tracking_results: gpd.GeoDataFrame
        A GeoDataFrame as obtained from an image tracking.
    velocity_threshold: float
        The threshold up to which deviations from the average are considered non-outliers (given in the unit of
        "movement_distance_per_year").
    inclusion_distance: float
        The distance up to which points are being taken into account for the average velocity calculation (given in
        terms of the unit of the crs)
    Returns
    -------
    tracking_results: GeoDataFrame
        The changed GeoDataFrame
    """
    tracking_results["is_velocity_outlier"] = False
    for i in range(len(tracking_results)):
        surrounding_points = tracking_results.loc[tracking_results.dwithin(tracking_results.geometry[i], inclusion_distance),:]
        average_velocity = np.nanmean(surrounding_points["movement_distance_per_year"])
        if np.abs(average_velocity - tracking_results.loc[i,"movement_distance_per_year"]) > velocity_threshold:
            tracking_results.loc[i,"is_velocity_outlier"] = True
    tracking_results.loc[tracking_results["is_velocity_outlier"], "movement_distance_per_year"] = 0
    tracking_results.loc[tracking_results["is_velocity_outlier"], "movement_bearing_pixels"] = np.nan
    return tracking_results
