import numpy as np
import geopandas as gpd
import rasterio.plot
import matplotlib.pyplot as plt
import pandas as pd
import rasterio.fill


def plot_raster_and_geometry(raster_matrix: np.ndarray, raster_transform, geometry: gpd.GeoDataFrame, alpha=0.6):
    plot_extent = rasterio.plot.plotting_extent(raster_matrix, raster_transform)
    fig, ax = plt.subplots()
    geometry.plot(ax=ax, color="blue", alpha=alpha, markersize=1)
    rasterio.plot.show(raster_matrix, ax=ax, extent=plot_extent, cmap="Greys")
    plt.show()


def plot_movement_of_points(raster_matrix: np.ndarray, raster_transform, point_movement: gpd.GeoDataFrame,
                            masking_polygon: gpd.GeoDataFrame = None, show_figure: bool = True, save_path: str = None):
    fig, ax = plt.subplots()

    if masking_polygon is not None:
        masking_polygon = masking_polygon.to_crs(crs=point_movement.crs)
        point_movement = gpd.overlay(point_movement, masking_polygon, how="intersection")

    point_movement.plot(ax=ax, column="movement_distance_per_year", legend=True, markersize=8, marker="s", alpha=1.0,
    # # #             # vmin=0, vmax=5,
                 )

    rasterio.plot.show(raster_matrix, transform=raster_transform, ax=ax, cmap="Greys")

    # thesis visualization
    # background_rock_glacier = rasterio.open("../Test_Data/temp_rock_glacier_background.tif")
    # rasterio.plot.show(background_rock_glacier, ax=ax, cmap="Greys")
    # #
    # outlier_points = point_movement[(point_movement["movement_row_direction"] == -0.001) | (point_movement["movement_row_direction"] == -0.002) | (point_movement["movement_row_direction"] == -0.003)| (point_movement["movement_row_direction"] == -0.004)| (point_movement["movement_row_direction"] == -0.005)]
    #
    # outlier_points.loc[outlier_points["movement_row_direction"] == -0.001, "movement_row_direction"] = "Cross-correlation yielded no valid result"
    # outlier_points.loc[outlier_points["movement_row_direction"] == -0.002, "movement_row_direction"] = "Optimization did not converge"
    # outlier_points.loc[outlier_points["movement_row_direction"] == -0.003, "movement_row_direction"] = "Unrealistic Transformation determinant"
    # outlier_points.loc[outlier_points["movement_row_direction"] == -0.004, "movement_row_direction"] = "Rotation outlier"
    # outlier_points.loc[outlier_points["movement_row_direction"] == -0.005, "movement_row_direction"] = "Velocity or rotation outlier"


    # outlier_points.plot(categorical=True, column = "movement_row_direction", ax=ax, markersize=8, marker="o", alpha=1.0, legend=True, legend_kwds={"loc": "lower left", "fontsize": "small", "reverse": True}, cmap="plasma")

    # Arrow plotting
    for row in sorted(list(set(point_movement.loc[:, "row"])))[::4]:
        for column in sorted(list(set(point_movement.loc[:, "column"])))[::4]:

            arrow_point = point_movement.loc[(point_movement['row'] == row) & (point_movement['column'] == column)]
            if not arrow_point.empty:
                arrow_point = arrow_point.iloc[0]
                plt.arrow(arrow_point["geometry"].x, arrow_point["geometry"].y,
                          arrow_point["movement_column_direction"] * 1.5 / arrow_point["movement_distance_per_year"],
                          -arrow_point["movement_row_direction"] * 1.5 / arrow_point["movement_distance_per_year"],
                          head_width=10, head_length=10, color="black", alpha=1)
    plt.title("Movement Distance in " + point_movement.crs.axis_info[0].unit_name + " per year")
    plt.title("Reasons for invalid matching of points")
    if show_figure:
        fig.show()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches='tight')

