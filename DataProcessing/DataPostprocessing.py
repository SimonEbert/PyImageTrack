import geopandas as gpd
import numpy as np

from ImageTracking import TrackMovement
from CreateGeometries.HandleGeometries import georeference_tracked_points
from CreateGeometries.HandleGeometries import random_points_on_polygon_by_number
from dataloader.TrackingParameters import TrackingParameters


def calculate_lod(image1_matrix: np.ndarray, image2_matrix: np.ndarray, image_transform,
                  reference_area: gpd.GeoDataFrame, number_of_reference_points,
                  tracking_parameters: TrackingParameters, crs, years_between_observations,
                  level_of_detection_quantile: float = 0.5) -> float:

    points = random_points_on_polygon_by_number(reference_area, number_of_points=number_of_reference_points)
    tracked_points = TrackMovement.track_movement_lsm(
        image1_matrix=image1_matrix, image2_matrix=image2_matrix,image_transform=image_transform,
        points_to_be_tracked=points, movement_cell_size=tracking_parameters.movement_cell_size,
        movement_tracking_area_size=tracking_parameters.movement_tracking_area_size,)
    tracked_points = georeference_tracked_points(tracked_points, image_transform, crs=crs,
                                                 years_between_observations=years_between_observations)
    level_of_detection = np.quantile(tracked_points.loc[~tracked_points["movement_distance_per_year"].isna(),
                                                        "movement_distance_per_year"], level_of_detection_quantile)
    return level_of_detection


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

    tracking_results.loc[tracking_results["movement_distance_per_year"] < level_of_detection,
                         ["movement_distance_per_year", "movement_distance",
                          "movement_row_direction", "movement_column_direction",
                          "movement_distance_pixels"]] = 0
    tracking_results.loc[tracking_results["movement_distance_per_year"] < level_of_detection,
                         "movement_bearing_pixels"] = np.nan
    return tracking_results
