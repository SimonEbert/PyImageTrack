
import geopandas as gpd
import numpy as np

def calculate_displacement_from_position_image(tracked_points: gpd.GeoDataFrame, position_image_time1: np.ndarray,
                                               position_image_time2: np.ndarray) -> gpd.GeoDataFrame:

    for i in tracked_points.index:
        point = tracked_points.loc[i]

        three_d_point_position1 = position_image_time1[:,
            np.round(point["row"]).astype(int),
            np.round(point["column"]).astype(int)]

        three_d_point_position2 = position_image_time2[:,
            np.round(point["row"] + point["movement_row_direction"]).astype(int),
            np.round(point["column"] + point["movement_column_direction"]).astype(int)
        ]


        three_d_displacement_distance = three_d_point_position2 - three_d_point_position1


        tracked_points.loc[i,"3d_position1_x"] = three_d_point_position1[0]
        tracked_points.loc[i,"3d_position1_y"] = three_d_point_position1[1]
        tracked_points.loc[i,"3d_position1_z"] = three_d_point_position1[2]

        tracked_points.loc[i,"3d_position2_x"] = three_d_point_position2[0]
        tracked_points.loc[i,"3d_position2_y"] = three_d_point_position2[1]
        tracked_points.loc[i,"3d_position2_z"] = three_d_point_position2[2]
        tracked_points.loc[i,"3d_displacement_distance"] = np.linalg.norm(three_d_displacement_distance)
    rows = tracked_points["row"]
    columns = tracked_points["column"]
    georeferenced_tracked_pixels = gpd.GeoDataFrame(tracked_points, geometry=gpd.points_from_xy(x=columns, y=-rows))

    georeferenced_tracked_pixels["valid"] = True
    georeferenced_tracked_pixels.loc[
        np.isnan(georeferenced_tracked_pixels["3d_displacement_distance"]),
        "valid"] = False

    return georeferenced_tracked_pixels
