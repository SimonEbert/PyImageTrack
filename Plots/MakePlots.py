import numpy as np
import geopandas as gpd
import rasterio.plot
import matplotlib.pyplot as plt
import pandas as pd
import rasterio.fill


def plot_raster_and_geometry(raster_matrix: np.ndarray, raster_transform, geometry: gpd.GeoDataFrame, alpha=0.6):
    plot_extent = rasterio.plot.plotting_extent(raster_matrix, raster_transform)
    fig, ax = plt.subplots()
    geometry.plot(ax=ax, color="red", alpha=0.7)
    rasterio.plot.show(raster_matrix, ax=ax, extent=plot_extent)
    plt.show()


def plot_movement_of_points(raster_matrix: np.ndarray, raster_transform, point_movement: pd.DataFrame,
                            masking_polygon: gpd.GeoDataFrame):
    """WARNING: Still hard coded crs in here!!"""
    plot_extent = rasterio.plot.plotting_extent(raster_matrix, raster_transform)
    fig, ax = plt.subplots()
    rasterio.plot.show(raster_matrix, transform=raster_transform, ax=ax)

    [x, y] = rasterio.transform.xy(raster_transform, point_movement.loc[:, "row"], point_movement.loc[:, "column"])
    points = gpd.GeoDataFrame(point_movement.loc[:, ["movement_row_direction", "movement_column_direction"]],
                              geometry=gpd.points_from_xy(x=x, y=y), crs=32632)

    masking_polygon = masking_polygon.to_crs(32632)
    points = gpd.overlay(points, masking_polygon, how="intersection")
    points.insert(3, "movement_distance",
                  np.linalg.norm(points.loc[:, ["movement_row_direction", "movement_column_direction"]], axis=1))
    # points.plot(ax=ax, c=points.loc[:, "movement_distance"], legend=True)
    points.plot(ax=ax, column="movement_distance", legend=True)
    plt.title("Movement Distance of the Rock glacier in Pixels")
    plt.show()


def plot_movement_of_points_interpolated(raster_matrix: np.ndarray, raster_transform, point_movement: pd.DataFrame):
    plot_extent = rasterio.plot.plotting_extent(raster_matrix, raster_transform)
    fig, ax = plt.subplots()
    rasterio.plot.show(raster_matrix, ax=ax)

    points = gpd.GeoDataFrame(point_movement.loc[:, ["movement_row_direction", "movement_column_direction"]],
                              geometry=gpd.points_from_xy(x=point_movement.loc[:, "column"],
                                                          y=point_movement.loc[:, "row"]))
    points.insert(3, "movement_distance",
                  np.linalg.norm(points.loc[:, ["movement_row_direction", "movement_column_direction"]], axis=1))

    movement_direction_points = np.linalg.norm(points.loc[:, ["movement_row_direction", "movement_column_direction"]],
                                               axis=1)
    points_rasterized = np.zeros(raster_matrix.shape)
    for i in np.arange(len(points)):
        row = point_movement.loc[i, "row"]
        col = point_movement.loc[i, "column"]
        points_rasterized[row, col] = movement_direction_points[i]

    points_rasterized = rasterio.fill.fillnodata(image=points_rasterized, mask=points_rasterized)
    rasterio.plot.show(points_rasterized, ax=ax)
    plt.title("Movement Distance of the Rock glacier in Pixels")
    plt.show()
