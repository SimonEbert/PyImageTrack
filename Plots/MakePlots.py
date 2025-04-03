import numpy as np
import geopandas as gpd
import rasterio.plot
import matplotlib.pyplot as plt
import scipy.interpolate


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
                            masking_polygon: gpd.GeoDataFrame = None, show_figure: bool = True, save_path: str = None):
    
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
    masking_polygon: gpd.GeoDataFrame = None
        A single-element GeoDataFrame to allow masking the plotted points to a certain area. If None, the points will
        not be masked.
    show_figure : bool = True
        If True, the created plot is displayed on the current canvas.
    save_path : str = None
        The file location, where the created plot is stored. When no path is given (the default), the figure is not
        saved.
    Returns
    ----------
    None
    """
    
    fig, ax = plt.subplots(dpi=200)

    if masking_polygon is not None:
        masking_polygon = masking_polygon.to_crs(crs=point_movement.crs)
        point_movement = gpd.overlay(point_movement, masking_polygon, how="intersection")


    point_movement.plot(ax=ax, column="movement_distance_per_year", legend=True, markersize=1, marker=".", alpha=1.0,
                        # missing_kwds={'color': 'gray'}
                        vmin=0, vmax=3.5,
                        )

    rasterio.plot.show(raster_matrix, transform=raster_transform, ax=ax, cmap="Greys")

    # Arrow plotting
    for row in sorted(list(set(point_movement.loc[:, "row"])))[::8]:
        for column in sorted(list(set(point_movement.loc[:, "column"])))[::8]:

            arrow_point = point_movement.loc[(point_movement['row'] == row) & (point_movement['column'] == column)]
            if not arrow_point.empty:
                arrow_point = arrow_point.iloc[0]
                plt.arrow(arrow_point["geometry"].x, arrow_point["geometry"].y,
                          arrow_point["movement_column_direction"] * 9 / arrow_point["movement_distance"],
                          -arrow_point["movement_row_direction"] * 9 / arrow_point["movement_distance"],
                          head_width=10, head_length=10, color="black", alpha=1)
    plt.title("Movement velocity in " + point_movement.crs.axis_info[0].unit_name + " per year")
    # plt.title("Reasons for invalid matching of points")
    if show_figure:
        fig.show()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches='tight')





def plot_movement_of_points_interpolated(raster_matrix: np.ndarray, raster_transform, point_movement: gpd.GeoDataFrame,
                            masking_polygon: gpd.GeoDataFrame = None, show_figure: bool = True, save_path: str = None):

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
    masking_polygon: gpd.GeoDataFrame = None
        A single-element GeoDataFrame to allow masking the plotted points to a certain area. If None, the points will
        not be masked.
    show_figure : bool = True
        If True, the created plot is displayed on the current canvas.
    save_path : str = None
        The file location, where the created plot is stored. When no path is given (the default), the figure is not
        saved.
    Returns
    ----------
    None
    """

    movement_matrix = np.full(raster_matrix.shape, 0)
    rows = point_movement["row"]
    columns = point_movement["column"]

    for row in rows:
        for column in columns:
            movement_matrix = point_movement.loc[(point_movement["row"] == row) & (point_movement["column"] == column),
            "movement_distance_per_year"]

    movement_spline = scipy.interpolate.RectBivariateSpline(rows, columns, movement_matrix)

    x_fine = np.linspace(0, 5000, 1000)
    y_fine = np.linspace(0, 5000, 1000)
    X_fine, Y_fine = np.meshgrid(x_fine, y_fine)
    Z_fine = movement_spline(x_fine, y_fine)  # Evaluating the spline

    # 2D Contour plot
    plt.figure(figsize=(6, 5))
    plt.contourf(X_fine, Y_fine, Z_fine, levels=50, cmap="viridis")
    plt.colorbar(label="Interpolated Values")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("RectBivariateSpline Contour Plot")
    plt.show()


    # geoplot.kdeplot(point_movement[["movement_distance_per_year", "geometry"]], fill=True, ax=ax, cmap="Reds", clip=masking_polygon.geometry, legend=True)

    rasterio.plot.show(raster_matrix, transform=raster_transform, ax=ax, cmap="Greys")

    # Arrow plotting
    # for row in sorted(list(set(point_movement.loc[:, "row"])))[::4]:
    #     for column in sorted(list(set(point_movement.loc[:, "column"])))[::4]:
    #
    #         arrow_point = point_movement.loc[(point_movement['row'] == row) & (point_movement['column'] == column)]
    #         if not arrow_point.empty:
    #             arrow_point = arrow_point.iloc[0]
    #             plt.arrow(arrow_point["geometry"].x, arrow_point["geometry"].y,
    #                       arrow_point["movement_column_direction"] * 3 / arrow_point["movement_distance"],
    #                       -arrow_point["movement_row_direction"] * 3 / arrow_point["movement_distance"],
    #                       head_width=10, head_length=10, color="black", alpha=1)
    # plt.title("Movement velocity in " + point_movement.crs.axis_info[0].unit_name + " per year")
    # plt.title("Reasons for invalid matching of points")
    # if show_figure:
    #     fig.show()
    # if save_path is not None:
    #     fig.savefig(save_path, bbox_inches='tight')




