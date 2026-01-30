import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio.plot


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
                            point_color: str = None, masking_polygon: gpd.GeoDataFrame = None, fig=None, ax=None,
                            save_path: str = None, show_arrows: bool = True, unit_name: str = None):

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
    fig: plt.figure = None
        Specifies the figure for plotting multiple results simultaneously.
    ax: plt.ax = None
        Specifies the axes on which to plot the movement of tracked points. If None (the default) the figure is plotted
        onto a new canvas. If fig, ax are not provided, but save_path is, the figure is only saved and not displayed.
    save_path : str = None
        The file location, where the created plot is stored. When no path is given (the default), the figure is not
        saved.
    show_arrows: bool = True
        Wether to show direction arrows on the resulting image
    unit_name: str = None
        Name of the unit to display in the plot
    Returns
    ----------
    None
    """

    # --- Estimate median point spacing in map units ---
    coords = np.vstack([point_movement.geometry.x, point_movement.geometry.y]).T

    # Sort by x then y for fast nearest-neighbor estimate
    coords_sorted = coords[np.lexsort((coords[:, 1], coords[:, 0]))]
    deltas = np.sqrt(np.sum(np.diff(coords_sorted, axis=0) ** 2, axis=1))
    median_spacing = np.nanmedian(deltas[deltas > 0])

    if not np.isfinite(median_spacing):
        median_spacing = 1.0

    show_figure = False
    if ax is None and fig is None:
        fig, ax = plt.subplots(dpi=200)
        if save_path is None:
            show_figure = True

    if masking_polygon is not None:
        masking_polygon = masking_polygon.to_crs(crs=point_movement.crs)
        point_movement = gpd.overlay(point_movement, masking_polygon, how="intersection")

    point_size = np.clip((median_spacing ** 2) * 0.04, 2, 30)

    if "3d_displacement_distance_per_year" in list(point_movement.columns):
        displacement_column_name = "3d_displacement_distance_per_year"
        legend_title_part = "Point velocity in "
    else:
        displacement_column_name = "movement_distance_per_year"
        legend_title_part = "Point velocity in "

    if unit_name is None:
        unit_name = point_movement.crs.axis_info[0].unit_name if point_movement.crs is not None else "pixel"


    if point_color is None:
        try:
            point_movement.plot(ax=ax, column=displacement_column_name, legend=True, markersize=point_size, marker=".",
                                    alpha=1.0,legend_kwds={"label":legend_title_part + unit_name + " / year"}
                                    )


        except:
            raise ValueError("Could not find columns 'movement_distance_per_year' or '3d_displacement_distance_per_year'."
                             "Provide a dataframe with either one of these columns for movement plotting.")

    else:
        point_movement.plot(ax=ax, color=point_color, markersize=1, marker=".", alpha=1.0,
                            )

    ax.ticklabel_format(scilimits=(-3, 4))
    if raster_matrix is not None:
        rasterio.plot.show(raster_matrix, transform=raster_transform, ax=ax, cmap="Greys")



    # Arrow plotting
    if show_arrows:
        arrow_spacing = median_spacing * 3
        arrow_length = median_spacing * 1.2

        xmin, ymin, xmax, ymax = point_movement.total_bounds

        ix = ((point_movement.geometry.x - xmin) // arrow_spacing).astype(int)
        iy = ((point_movement.geometry.y - ymin) // arrow_spacing).astype(int)

        pm = point_movement.assign(_ix=ix, _iy=iy)

        # One arrow per grid cell
        for (_, _), p in pm.groupby(["_ix", "_iy"]).first().iterrows():

            dx = p["movement_column_direction"]
            dy = -p["movement_row_direction"]

            norm = np.hypot(dx, dy)
            if norm == 0:
                continue

            dx /= norm
            dy /= norm

            ax.arrow(
                p.geometry.x,
                p.geometry.y,
                dx * arrow_length,
                dy * arrow_length,
                width = arrow_length * 0.12,
                head_width=arrow_length * 0.36,
                head_length=arrow_length * 0.25,
                color="black",
                alpha=1,
                length_includes_head=True,
                zorder=5
            )

    if displacement_column_name == "3d_displacement_distance_per_year":
        plt.title("3d-displacement distance per year")
    else:
        plt.title("Movement velocity in " + unit_name + " per year")

    if show_figure:
        fig.show()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches='tight')


def plot_movement_of_points_with_valid_mask(raster_matrix: np.ndarray, raster_transform,
                                            point_movement: gpd.GeoDataFrame,
                                            save_path: str = None,
                                            unit_name: str = None,):
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
    unit_name : str = None
        The unit name to label the colorbar of displacement values
    Returns
    ----------
    None
    """
    fig, ax = plt.subplots(dpi=200)
    point_movement_valid = point_movement[point_movement["valid"]]
    point_movement_invalid = point_movement[~point_movement["valid"]]

    plot_movement_of_points(None, raster_transform, point_movement_invalid, point_color="gray",
                            show_arrows=False, fig=fig, ax=ax)
    plot_movement_of_points(raster_matrix, raster_transform, point_movement_valid, fig=fig, ax=ax, save_path=None,
                            unit_name=unit_name)

    if save_path is None:
        fig.show()
    else:
        fig.savefig(save_path, bbox_inches='tight')  #


def plot_distribution_of_point_movement(moving_points: gpd.GeoDataFrame):
    fig, ax = plt.subplots()
    ax.grid(True, which='both')

    ax.axhline(y=0, color='k')
    ax.axvline(x=0, color='k')
    # ax.set_xlim((-1,1))
    # ax.set_ylim((-1,1))

    ax.scatter(moving_points["movement_row_direction"],
               moving_points["movement_column_direction"])
    plt.show()
