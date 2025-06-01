import numpy as np
import geopandas as gpd
import rasterio.plot
import matplotlib.pyplot as plt
import scipy.interpolate
import logging


def plot_raster_and_geometry(raster_matrix: np.ndarray, raster_transform, geometry: gpd.GeoDataFrame, alpha=0.6):

    """
    Plots a matrix representing a raster image with given transform and the geometries of a given GeoDataFrame in one
    figure.
    Parameters
    ----------
    raster_matrix: np.ndarray
        The matrix representing the raster image to be plotted.
    raster_transform
        An object of the class Affine as used by the rasterio package, which gives the transform of the raster image to
        the coordinate reference system of the geometry GeoDataFrame.
    geometry: gpd.GeoDataFrame
        The geometry to be plotted.
    alpha=0.6
        The opacity of the plotted geometry (which will be plotted on top of the raster image).
    Returns
    ----------
    None
    """

    plot_extent = rasterio.plot.plotting_extent(raster_matrix, raster_transform)
    fig, ax = plt.subplots()
    geometry.plot(ax=ax, color="blue", alpha=alpha, markersize=1)
    rasterio.plot.show(raster_matrix, ax=ax, extent=plot_extent, cmap="Greys")
    plt.show()


def plot_movement_of_points(raster_matrix: np.ndarray, raster_transform, point_movement: gpd.GeoDataFrame,
                            point_color: str = None, masking_polygon: gpd.GeoDataFrame = None, fig = None, ax = None,
                            save_path: str = None):
    # ToDo: Change size of the single points for a smooth image regardless of point grid resolution
    
    """
    Plots the movement of tracked points as a geometry on top of a given raster image matrix. Velocity is shown via a
    colour scale, while the movement direction is shown with arrows for selected pixels.
    Parameters
    ----------
    raster_matrix: np.ndarray
        The matrix representing the raster image to be plotted.
    raster_transform :
        An object of the class Affine as used by the rasterio package, which gives the transform of the raster image to
        the coordinate reference system of the geometry GeoDataFrame.
    point_movement: gpd.GeoDataFrame
        A GeoDataFrame containing the columns "row", "column" giving the position of the points expressed in matrix
        indices, as well as "movement_column_direction", "movement_row_direction" and "movement_distance_per_year". The
        unit of the movement is taken from the coordinate reference system of this GeoDataFrame.
    point_color: str = None
        Forces all the points to have a single color specified via this string. If None, a colormap is used to denote
        different movement velocities.
    masking_polygon: gpd.GeoDataFrame = None
        A single-element GeoDataFrame to allow masking the plotted points to a certain area. If None, the points will
        not be masked.
    fig = None
        Specifies the figure for plotting multiple results simultaneously.
    ax = None
        Specifies the axes on which to plot the movement of tracked points. If None (the default) the figure is plotted
        onto a new canvas. If fig, ax are not provided, but save_path is, the figure is only saved and not displayed.
    save_path : str = None
        The file location, where the created plot is stored. When no path is given (the default), the figure is not
        saved.
    Returns
    ----------
    None
    """

    show_figure = False
    if ax is None and fig is None:
        fig, ax = plt.subplots(dpi=200)
        if save_path is None:
            show_figure = True

    if masking_polygon is not None:
        masking_polygon = masking_polygon.to_crs(crs=point_movement.crs)
        point_movement = gpd.overlay(point_movement, masking_polygon, how="intersection")

    if point_color is None:
        point_movement.plot(ax=ax, column="movement_distance_per_year", legend=True, markersize=1, marker=".", alpha=1.0,
                            # missing_kwds={'color': 'gray'}
                            # vmin=0, vmax=3.5,
                            )
    else:
        point_movement.plot(ax=ax, color=point_color, markersize=1, marker=".", alpha=1.0)

    ax.ticklabel_format(scilimits= (-3,4))
    if raster_matrix is not None:
        rasterio.plot.show(raster_matrix, transform=raster_transform, ax=ax, cmap="Greys")

    # Arrow plotting
    for row in sorted(list(set(point_movement.loc[:, "row"])))[::8]:
        for column in sorted(list(set(point_movement.loc[:, "column"])))[::8]:

            arrow_point = point_movement.loc[(point_movement['row'] == row) & (point_movement['column'] == column)]
            if not arrow_point.empty:
                arrow_point = arrow_point.iloc[0]
                if arrow_point["movement_distance_per_year"] == 0:
                    continue
                ax.arrow(arrow_point["geometry"].x, arrow_point["geometry"].y,
                         arrow_point["movement_column_direction"] * 1.5 / arrow_point["movement_distance_per_year"],
                         -arrow_point["movement_row_direction"] * 1.5 / arrow_point["movement_distance_per_year"],
                         head_width=10, head_length=10, color="black", alpha=1)
    plt.title("Movement velocity in " + point_movement.crs.axis_info[0].unit_name + " per year")

    if show_figure:
        fig.show()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches='tight')


def plot_movement_of_points_with_lod_mask(raster_matrix: np.ndarray, raster_transform, point_movement: gpd.GeoDataFrame,
                                          save_path: str = None):
    """
    Plots the movement of tracked points as a geometry on top of a given raster image matrix. Velocity is shown via a
    colour scale, while the movement direction is shown with arrows for selected pixels.
    Parameters
    ----------
    raster_matrix: np.ndarray
        The matrix representing the raster image to be plotted.
    raster_transform :
        An object of the class Affine as used by the rasterio package, which gives the transform of the raster image to
        the coordinate reference system of the geometry GeoDataFrame.
    point_movement: gpd.GeoDataFrame
        A GeoDataFrame containing the columns "row", "column" giving the position of the points expressed in matrix
        indices, as well as "movement_column_direction", "movement_row_direction" and "movement_distance_per_year". The
        unit of the movement is taken from the coordinate reference system of this GeoDataFrame.
    save_path : str = None
        The file location, where the created plot is stored. When no path is given (the default), the figure is not
        saved.
    Returns
    ----------
    None
    """
    fig, ax = plt.subplots(dpi=200)
    point_movement_positive = point_movement[point_movement["movement_distance_per_year"] > 0]
    point_movement_zero = point_movement[point_movement["movement_distance_per_year"] == 0]

    plot_movement_of_points(None, raster_transform, point_movement_zero, point_color="gray", fig=fig, ax=ax)
    plot_movement_of_points(raster_matrix, raster_transform, point_movement_positive, fig=fig, ax=ax, save_path=None)

    if save_path is None:
        fig.show()
    else:
        fig.savefig(save_path, bbox_inches='tight')

