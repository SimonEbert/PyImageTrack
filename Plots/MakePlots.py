import numpy as np
import geopandas as gpd
import rasterio.plot
import matplotlib.pyplot as plt
import pandas as pd


def plot_raster_and_geometry(raster_matrix: np.ndarray, raster_transform, geometry: gpd.GeoDataFrame, alpha=0.6):
    plot_extent = rasterio.plot.plotting_extent(raster_matrix, raster_transform)
    fig, ax = plt.subplots()
    geometry.plot(ax=ax, color="red", alpha=0.7)
    rasterio.plot.show(raster_matrix, ax=ax, extent=plot_extent)
    plt.show()


def plot_movement_of_points(raster_matrix: np.ndarray, raster_transform, point_movement: pd.DataFrame):
    plot_extent = rasterio.plot.plotting_extent(raster_matrix, raster_transform)
    fig, ax = plt.subplots()
    rasterio.plot.show(raster_matrix, ax=ax)

    points = gpd.GeoDataFrame(point_movement.loc[:, ["movement_row_direction", "movement_column_direction"]],
                              geometry=gpd.points_from_xy(x=point_movement.loc[:, "column"],
                                                          y=point_movement.loc[:, "row"]))
    points.insert(3, "movement_distance",
                  np.linalg.norm(points.loc[:, ["movement_row_direction", "movement_column_direction"]], axis=1))
    points.plot(ax=ax, c=points.loc[:, "movement_distance"], legend=True)
    ax.legend()
    plt.show()
