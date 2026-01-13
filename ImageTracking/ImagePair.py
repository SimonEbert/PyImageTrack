import logging
import os

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.plot
from rasterio.crs import CRS
from datetime import datetime, timedelta
import logging
from geocube.api.core import make_geocube
import os
from rasterio.coords import BoundingBox
from geocube.rasterize import rasterize_points_griddata
from shapely.geometry import box
import numpy as np
import pandas as pd
import scipy
import sklearn

# Parameter classes
from ..Parameters.TrackingParameters import TrackingParameters
from ..Parameters.FilterParameters import FilterParameters
from ..Parameters.AlignmentParameters import AlignmentParameters

# Alignment and Tracking functions
from .TrackMovement import track_movement_lsm, move_indices_from_transformation_matrix
from ..CreateGeometries.HandleGeometries import crop_images_to_intersection
from ..CreateGeometries.HandleGeometries import georeference_tracked_points
from ..CreateGeometries.HandleGeometries import grid_points_on_polygon_by_distance
# Geometry Handling
from ..CreateGeometries.HandleGeometries import random_points_on_polygon_by_number
# filter functions
from ..DataProcessing.DataPostprocessing import calculate_lod_points
from ..DataProcessing.DataPostprocessing import filter_lod_points
from ..DataProcessing.DataPostprocessing import filter_outliers_full
# DataPreProcessing
from ..DataProcessing.ImagePreprocessing import equalize_adapthist_images
from .AlignImages import align_images_lsm_scarce
# Alignment and Tracking functions
from .TrackMovement import track_movement_lsm
from ..Parameters.AlignmentParameters import AlignmentParameters
from ..Parameters.FilterParameters import FilterParameters
# Parameter classes
from ..Parameters.TrackingParameters import TrackingParameters
# Plotting
from ..Plots.MakePlots import plot_movement_of_points
from ..Plots.MakePlots import plot_movement_of_points_with_valid_mask
# Date Handling
from ..Utils import parse_date


class ImagePair:
    def __init__(self, parameter_dict: dict = None):
        self.images_aligned = False
        self.valid_alignment_possible = None
        # self.mask_array = None

        # Data
        self.image1_matrix = None
        self.image1_transform = None
        self.image1_observation_date = None
        self.image2_matrix = None
        self.image2_transform = None
        self.image2_observation_date = None
        self.image_bounds = None

        # Optional: store original (true-color) matrices for writing aligned products
        self.image1_matrix_original = None
        self.image2_matrix_original = None
        self.image2_matrix_truecolor = None

        # Parameters
        self.tracking_parameters = TrackingParameters(parameter_dict=parameter_dict)
        self.filter_parameters = None
        self.alignment_parameters = AlignmentParameters(parameter_dict=parameter_dict)

        # Fake georef switches (fed via run_pipeline param_dict)
        self.use_fake_georeferencing = bool(
            parameter_dict.get("use_fake_georeferencing", False)) if parameter_dict else False
        self.fake_crs_epsg = parameter_dict.get("fake_crs_epsg", None) if parameter_dict else None
        self.fake_pixel_size = float(parameter_dict.get("fake_pixel_size", 1.0)) if parameter_dict else 1.0

        # Meta-Data and results
        self.crs = None
        self.tracked_control_points = None
        self.tracking_results = None
        self.level_of_detection = None
        self.level_of_detection_points = None

    def _effective_pixel_size(self) -> float:
        """CRS units per pixel (assumes square pixels)."""
        if self.image1_transform is None:
            return 1.0
        a = float(self.image1_transform.a)
        e = float(self.image1_transform.e)
        return max(abs(a), abs(e))

    def select_image_channels(self, selected_channels: int = None):
        if selected_channels is None:
            selected_channels = [0, 1, 2]
        if len(self.image1_matrix.shape) == 3:
            self.image1_matrix = self.image1_matrix[selected_channels, :, :]
            self.image2_matrix = self.image2_matrix[selected_channels, :, :]

    def load_images_from_file(self, filename_1: str, observation_date_1: str, filename_2: str, observation_date_2: str,
                              selected_channels: int = None, NA_value: float = None):
        """
        Loads two image files from the respective file paths. The order of the provided image paths is expected to
        align with the observation order, that is the first image is assumed to be the earlier observation. The two
        images are cropped to the same geospatial extent, assuming they are given in the same coordinate reference
        system. Optionally image channels can be provided which will be selected during data loading. Selects by default
        the first 3 image channels for tracking.
        Parameters
        ----------
        filename_1: str
            The filename of the first image
        observation_date_1: str
            The observation date of the first image in format %d-%m-%Y
        filename_2: str
            The filename of the second image
        observation_date_2: str
            The observation date of the second image in format %d-%m-%Y
        selected_channels: list[int]
            The image channels to be selected. Defaults to [0,1,2]
        Returns
        -------
        """
        file1 = rasterio.open(filename_1, 'r+')
        file2 = rasterio.open(filename_2, 'r+')

        # Choose path: true georef vs fake georef
        no_crs_either = (file1.crs is None) or (file2.crs is None)
        force_fake = self.use_fake_georeferencing or no_crs_either

        if not force_fake:
            if file1.crs != file2.crs:
                raise ValueError("Got images with crs " + str(file1.crs) + " and " + str(file2.crs) +
                                 " but the two images must have the same crs.")
            self.crs = file1.crs

            # Spatial intersection (true georef)
            poly1 = box(*file1.bounds)
            poly2 = box(*file2.bounds)
            intersection = poly1.intersection(poly2)

            # Keep search windows inside valid area
            px_size = max(-file1.transform[4], file1.transform[0])  # pixel size (>0)
            ext = getattr(self.tracking_parameters, "search_extent_px", None)
            if not ext:
                raise ValueError("TrackingParameters.search_extent_px must be set (tuple posx,negx,posy,negy).")
            eff_search_radius_px = max(ext)

            image_bounds = gpd.GeoDataFrame(
                gpd.GeoDataFrame({'geometry': [intersection]}, crs=self.crs).buffer(-px_size * eff_search_radius_px)
            )
            image_bounds = image_bounds.rename(columns={0: "geometry"})
            image_bounds.set_geometry("geometry", inplace=True)
            self.image_bounds = image_bounds

            ([self.image1_matrix, self.image1_transform],
             [self.image2_matrix, self.image2_transform]) = crop_images_to_intersection(file1, file2)

        else:
            # FAKE georeferencing path (e.g., JPGs)
            arr1 = file1.read()  # (bands, rows, cols)
            arr2 = file2.read()

            def squeeze(arr):
                return arr[0] if arr.shape[0] == 1 else arr

            self.image1_matrix = squeeze(arr1)
            self.image2_matrix = squeeze(arr2)

            # Top-left crop to common size
            h = min(self.image1_matrix.shape[-2], self.image2_matrix.shape[-2])
            w = min(self.image1_matrix.shape[-1], self.image2_matrix.shape[-1])
            self.image1_matrix = self.image1_matrix[..., :h, :w] if self.image1_matrix.ndim == 3 else \
                self.image1_matrix[:h, :w]
            self.image2_matrix = self.image2_matrix[..., :h, :w] if self.image2_matrix.ndim == 3 else \
                self.image2_matrix[:h, :w]

            # Synthetic transform: origin (0,0) upper-left, pixel size = fake_pixel_size
            from affine import Affine
            px = float(self.fake_pixel_size)
            tform = Affine(px, 0, 0, 0, -px, 0)  # x = px*col ; y = -px*row

            self.image1_transform = tform
            self.image2_transform = tform

            # CRS from user config (poly_CRS passed via fake_crs_epsg)
            if self.fake_crs_epsg is None:
                raise ValueError("use_fake_georeferencing=True but no fake_crs_epsg was provided.")
            self.crs = CRS.from_epsg(int(self.fake_crs_epsg))

            # Bounds in that CRS (pixel grid space), then shrink by search radius
            from rasterio.transform import array_bounds
            bounds_poly = box(*array_bounds(h, w, tform))
            ext = getattr(self.tracking_parameters, "search_extent_px", None)
            if not ext:
                raise ValueError("TrackingParameters.search_extent_px must be set (tuple posx,negx,posy,negy).")
            buffer_len = px * max(ext)

            image_bounds = gpd.GeoDataFrame({'geometry': [bounds_poly]}, crs=self.crs).buffer(-buffer_len)
            image_bounds = gpd.GeoDataFrame(geometry=image_bounds, crs=self.crs)
            image_bounds = image_bounds.rename(columns={0: "geometry"})
            image_bounds.set_geometry("geometry", inplace=True)
            self.image_bounds = image_bounds


        self.image1_observation_date = parse_date(observation_date_1)
        self.image2_observation_date = parse_date(observation_date_2)

        if NA_value is not None:
            self.image1_matrix[self.image1_matrix == NA_value] = 0
            self.image2_matrix[self.image2_matrix == NA_value] = 0

        # Store original matrices before any further preprocessing (e.g. channel selection, CLAHE)
        self.image1_matrix_original = self.image1_matrix.copy()
        self.image2_matrix_original = self.image2_matrix.copy()

        self.select_image_channels(selected_channels=selected_channels)

    def load_images_from_matrix_and_transform(self, image1_matrix: np.ndarray, observation_date_1: str,
                                              image2_matrix: np.ndarray, observation_date_2: str, image_transform, crs,
                                              selected_channels: int = None):
        """
        Loads two images from two matrices, which are assumed to have the same image transform. For more details cf
        load_images_from_file
        Parameters
        ----------
        image1_matrix
        observation_date_1
        image2_matrix
        observation_date_2
        image_transform
        crs
        bounds
        selected_channels

        Returns
        -------

        """
        self.image1_matrix = image1_matrix
        self.image1_transform = image_transform
        self.image2_matrix = image2_matrix
        self.image2_transform = image_transform

        # Also store original matrices for potential true-color alignment output
        self.image1_matrix_original = image1_matrix.copy()
        self.image2_matrix_original = image2_matrix.copy()

        self.image1_observation_date = parse_date(observation_date_1)
        self.image2_observation_date = parse_date(observation_date_2)
        self.crs = crs

        bbox = rasterio.transform.array_bounds(image1_matrix.shape[-2], image1_matrix.shape[-1], image_transform)
        poly1 = box(*bbox)

        # Same idea as above: extents-only buffer
        px_size = max(-image_transform[4], image_transform[0])
        ext = getattr(self.tracking_parameters, "search_extent_px", None)
        if not ext:
            raise ValueError("TrackingParameters.search_extent_px must be set (tuple posx,negx,posy,negy).")
        eff_search_radius_px = max(ext)

        image_bounds = gpd.GeoDataFrame(
            gpd.GeoDataFrame({'geometry': [poly1]}, crs=self.crs).buffer(-px_size * eff_search_radius_px)
        )
        # set correct geometry column
        image_bounds = image_bounds.rename(columns={0: "geometry"})
        image_bounds.set_geometry("geometry", inplace=True)
        self.image_bounds = image_bounds

    def align_images(self, reference_area: gpd.GeoDataFrame) -> None:
        """
        Aligns the two images based on matching the given reference area. The number of tracked points created in the
        reference area is determined by the tracking parameters. Assumes the image transform of the first matrix is
        correct so that the two image matrices have the same image transform. Therefore, the values of image2_matrix
        and image2_transform are updated by this function.
        Parameters
        ----------
        reference_area: gpd.GeoDataFrame
            A one-element GeoDataFrame containing the area in which the points are defined to align the two images.
        Returns
        -------
        """
        print("Starting image alignment.")
        if reference_area.crs != self.crs:
            raise ValueError("Got reference area with crs " + str(reference_area.crs) + " and images with crs "
                             + str(self.crs) + ". Reference area and images are supposed to have the same crs.")
        reference_area = gpd.GeoDataFrame(reference_area.intersection(self.image_bounds))
        reference_area.rename(columns={0: 'geometry'}, inplace=True)
        reference_area.set_geometry('geometry', inplace=True)

        [_, new_image2_matrix, tracked_control_points] = (
            align_images_lsm_scarce(image1_matrix=self.image1_matrix,
                                    image2_matrix=self.image2_matrix,
                                    image_transform=self.image1_transform,
                                    reference_area=reference_area,
                                    alignment_parameters=self.alignment_parameters))

        self.valid_alignment_possible = True

        delta_hours = (self.image2_observation_date - self.image1_observation_date).total_seconds() / 3600.0
        years_between_observations = delta_hours / (24.0 * 365.25)

        self.tracked_control_points = georeference_tracked_points(tracked_control_points,
                                                                  self.image1_transform,
                                                                  self.crs,
                                                                  years_between_observations)

        self.image2_matrix = new_image2_matrix
        self.image2_transform = self.image1_transform

        self.images_aligned = True

        # Optionally derive a true-color aligned image (if original data are available)
        try:
            self.compute_truecolor_aligned_from_control_points()
        except Exception as e:
            logging.warning(f"Could not compute true-color aligned image: {e}")

    def compute_truecolor_aligned_from_control_points(self):
        """rebuild a true-color aligned version of image2 using the alignment control points.

        this uses the same least-squares affine model + spline resampling approach
        as in alignimages.align_images_lsm_scarce, but applies it to the original
        (potentially multi-band) second image.
        """
        if self.tracked_control_points is None or len(self.tracked_control_points) == 0:
            raise ValueError("no tracked control points available – cannot compute true-color alignment.")

        if self.image2_matrix_original is None:
            raise ValueError("no original second image stored – cannot compute true-color alignment.")

        df = self.tracked_control_points

        required_cols = ["row", "column", "movement_row_direction", "movement_column_direction"]
        if not all(col in df.columns for col in required_cols):
            raise ValueError("tracked control points are missing required pixel columns for reconstruction.")

        # input: original pixel positions (row, column)
        linear_model_input = np.column_stack([df["row"].values, df["column"].values])

        # output: new positions after shift (row + drow, col + dcol)
        linear_model_output = np.column_stack([
            df["row"].values + df["movement_row_direction"].values,
            df["column"].values + df["movement_column_direction"].values,
        ])

        # fit affine transform (same approach as in alignimages.align_images_lsm_scarce)
        transformation_linear_model = sklearn.linear_model.LinearRegression()
        transformation_linear_model.fit(linear_model_input, linear_model_output)

        sampling_transformation_matrix = np.array([
            [transformation_linear_model.coef_[0, 0],
             transformation_linear_model.coef_[0, 1],
             transformation_linear_model.intercept_[0]],
            [transformation_linear_model.coef_[1, 0],
             transformation_linear_model.coef_[1, 1],
             transformation_linear_model.intercept_[1]],
        ])

        # build output index grid (rows, cols) in the aligned image grid
        # use the shape of self.image1_matrix (reference image)
        if self.image1_matrix is None:
            raise ValueError("image1_matrix is not set – cannot infer output grid size.")

        if self.image1_matrix.ndim == 2:
            out_rows, out_cols = self.image1_matrix.shape
        else:
            # assume (bands, rows, cols)
            out_rows = self.image1_matrix.shape[-2]
            out_cols = self.image1_matrix.shape[-1]

        indices = np.array(
            np.meshgrid(np.arange(0, out_rows), np.arange(0, out_cols))
        ).T.reshape(-1, 2).T

        # move indices according to the affine transform
        moved_indices = move_indices_from_transformation_matrix(sampling_transformation_matrix, indices)

        # warp the original second image (can be single- or multi-band)
        src = self.image2_matrix_original

        if src.ndim == 2:
            src_rows = np.arange(0, src.shape[0])
            src_cols = np.arange(0, src.shape[1])
            spline = scipy.interpolate.RectBivariateSpline(src_rows, src_cols, src.astype(float))
            moved = spline.ev(moved_indices[0, :], moved_indices[1, :]).reshape(out_rows, out_cols)
        else:
            bands, src_rows_n, src_cols_n = src.shape
            src_rows = np.arange(0, src_rows_n)
            src_cols = np.arange(0, src_cols_n)
            moved = np.zeros((bands, out_rows, out_cols), dtype=float)
            for b in range(bands):
                spline = scipy.interpolate.RectBivariateSpline(
                    src_rows,
                    src_cols,
                    src[b, :, :].astype(float),
                )
                moved[b, :, :] = spline.ev(
                    moved_indices[0, :], moved_indices[1, :]
                ).reshape(out_rows, out_cols)

        self.image2_matrix_truecolor = moved


    def track_points(self, tracking_area: gpd.GeoDataFrame) -> gpd.geodataframe:
        """
        Creates a grid of points based on the polygon given in tracking_area. Tracks these points using the specified
        tracking parameters. Georeferences these points and returns the respective GeoDataFrame including a column with
        yearly movement rates for each point.
        Parameters
        ----------
        tracking_area: gpd:GeoDataFrame
            A one-element GeoDataFrame containing the polygon defining the tracking area.

        Returns
        -------
        georeferenced_tracked_points: gpd.GeoDataFrame
        """
        print("Starting point tracking.")
        if tracking_area.crs != self.crs:
            raise ValueError("Got tracking area with crs " + str(tracking_area.crs) + " and images with crs "
                             + str(self.crs) + ". Tracking area and images are supposed to have the same crs.")

        if not self.images_aligned:
            logging.warning("Images have not been aligned. Any resulting velocities are likely invalid.")

        # spacing in px -> recalculate into CRS
        px_size = self._effective_pixel_size()
        dp_px = float(self.tracking_parameters.distance_of_tracked_points_px)
        spacing_crs = dp_px * px_size

        points_to_be_tracked = grid_points_on_polygon_by_distance(
            polygon=tracking_area,
            distance_of_points=spacing_crs,
            distance_px=dp_px,
            pixel_size=px_size,
        )

        tracked_points = track_movement_lsm(self.image1_matrix, self.image2_matrix, self.image1_transform,
                                            points_to_be_tracked=points_to_be_tracked,
                                            tracking_parameters=self.tracking_parameters,
                                            alignment_tracking=False,
                                            task_label="Tracking points for movement tracking")

        # calculate the years between observations from the two given observation dates
        delta_hours = (self.image2_observation_date - self.image1_observation_date).total_seconds() / 3600.0
        years_between_observations = delta_hours / (24.0 * 365.25)

        georeferenced_tracked_points = georeference_tracked_points(tracked_pixels=tracked_points,
                                                                   raster_transform=self.image1_transform,
                                                                   crs=tracking_area.crs,
                                                                   years_between_observations=years_between_observations
                                                                   )

        return georeferenced_tracked_points

    def perform_point_tracking(self, reference_area: gpd.GeoDataFrame, tracking_area: gpd.GeoDataFrame) -> None:
        """
        Performs the necessary tracking steps. This method is designed as a helper method to facilitate tracking and
        writes the results directly to the respective object of the ImagePair class
        Parameters
        ----------
        reference_area: gpd.GeoDataFrame
            Area used for alignment of the two images. Needs to have the same crs as the two image files.
        tracking_area: gpd.GeoDataFrame
            Area used for creating the tracking point grid. Needs to have the same crs as the two image files.

        Returns
        -------
        None
        """
        if not getattr(self, "images_aligned", False):
            self.align_images(reference_area)
            if not self.valid_alignment_possible:
                return
        tracked_points = self.track_points(tracking_area)
        self.tracking_results = tracked_points

    def plot_images(self) -> None:
        """
        Plots the two raster images separately to the current canvas.
        Returns
        -------
        None
        """
        rasterio.plot.show(self.image1_matrix, title="Image 1, Observation date: "
                                                     + self.image1_observation_date.strftime("%d-%m-%Y"))
        rasterio.plot.show(self.image2_matrix, title="Image 2, Observation date: "
                                                     + self.image2_observation_date.strftime("%d-%m-%Y"))

    def plot_tracking_results(self) -> None:
        """
        Plots the first raster image and the movement of points in a single figure
        Returns
        -------
        None
        """
        if self.tracking_results is not None:
            plot_movement_of_points(self.image1_matrix, self.image1_transform, self.tracking_results)
        else:
            logging.warning("No results calculated yet. Plot not provided")

    def plot_tracking_results_with_valid_mask(self) -> None:
        """
        Plots the first raster image and the movement of points in a single figure. Every point that has 0 movement rate
        is shown in gray
        Returns
        -------
        """
        if self.tracking_results is not None:
            plot_movement_of_points_with_valid_mask(self.image1_matrix, self.image1_transform, self.tracking_results)
        else:
            logging.warning("No results calculated yet. Plot not provided")

    def filter_outliers(self, filter_parameters: FilterParameters):
        """
        Filters outliers based on the filter_parameters
        Parameters
        ----------
        filter_parameters: FilterParameters
            The Parameters used for Filtering. If some of the parameters are set to None, the respective filtering
            will not be performed
        Returns
        -------

        """
        if not self.valid_alignment_possible:
            return
        print("Filtering outliers. This may take a moment.")
        self.filter_parameters = filter_parameters
        self.tracking_results = filter_outliers_full(self.tracking_results, filter_parameters)

    def calculate_lod(self, points_for_lod_calculation: gpd.GeoDataFrame,
                      filter_parameters: FilterParameters = None) -> None:
        """
        Calculates the Level of Detection of a matching between two images. For calculating the LoD a specified number
        of points are generated randomly in some reference area, which is assumed to be stable. The level of detection
        is defined as some quantile (default: 0.5) of the movement rates of these points. The quantile can be set in the
        tracking parameters. This method sets the value of the lod as self.level_of_detection. The level of detection is
        given in movement per year.
        Parameters
        ----------
        points_for_lod_calculation: gpd.GeoDataFrame
            The points in the area (no motion assumed) for calculating the level of detection. Since for image alignment an evenly spaced grid is used and here, a random distribution of
            points is advisable, such that it is possible to use the same reference area for both tasks.
        filter_parameters
        Returns
        -------
        None
        """

        # Set used filter parameters if given as a variable, also set them as correct object for ImagePair
        if filter_parameters is None:
            if self.filter_parameters is None:
                return
            else:
                filter_parameters = self.filter_parameters
        else:
            self.filter_parameters = filter_parameters

        points_for_lod_calculation = gpd.GeoDataFrame(
            points_for_lod_calculation.intersection(self.image_bounds.geometry[0]))
        points_for_lod_calculation.rename(columns={0: 'geometry'}, inplace=True)
        points_for_lod_calculation.set_geometry('geometry', inplace=True)

        delta_hours = (self.image2_observation_date - self.image1_observation_date).total_seconds() / 3600.0
        years_between_observations = delta_hours / (24.0 * 365.25)

        # check if a LoD filter parameter is provided, if this is None, don't perform LoD calculation
        if (filter_parameters.level_of_detection_quantile is None
                or filter_parameters.number_of_points_for_level_of_detection is None):
            return

        level_of_detection_quantile = filter_parameters.level_of_detection_quantile

        unfiltered_level_of_detection_points = calculate_lod_points(image1_matrix=self.image1_matrix,
                                                                    image2_matrix=self.image2_matrix,
                                                                    image_transform=self.image1_transform,
                                                                    points_for_lod_calculation=points_for_lod_calculation,
                                                                    tracking_parameters=self.tracking_parameters,
                                                                    crs=self.crs,
                                                                    years_between_observations=years_between_observations)
        self.level_of_detection_points = unfiltered_level_of_detection_points

        self.level_of_detection = np.nanquantile(unfiltered_level_of_detection_points["movement_distance_per_year"],
                                                 level_of_detection_quantile)

        print("Found level of detection with quantile " + str(level_of_detection_quantile) + " as "
              + str(np.round(self.level_of_detection, decimals=5)) + " " + str(
            points_for_lod_calculation.crs.axis_info[0].unit_name) + "/year")

    def filter_lod_points(self) -> None:
        """
        Sets the movement distance of all points that fall below the calculated level of detection to 0 and their
        movement bearing to NaN. Note that this directly affects the dataframe self.tracking_results.
        Returns
        -------

        """

        if not self.valid_alignment_possible:
            return
        self.tracking_results = filter_lod_points(self.tracking_results, self.level_of_detection)

    def full_filter(self, reference_area, filter_parameters: FilterParameters):
        points_for_lod_calculation = random_points_on_polygon_by_number(reference_area,
                                                                        filter_parameters.number_of_points_for_level_of_detection)
        self.filter_outliers(filter_parameters)
        self.calculate_lod(points_for_lod_calculation, filter_parameters)
        self.filter_lod_points()

    def equalize_adapthist_images(self):
        self.image1_matrix = equalize_adapthist_images(self.image1_matrix,
                                                       kernel_size=50)
        self.image2_matrix = equalize_adapthist_images(self.image2_matrix,
                                                       kernel_size=50)

    def save_full_results(self, folder_path: str, save_files: list) -> None:
        """
        Saves tracking outputs (GeoJSON, GeoTIFFs, masks, stats) into `folder_path`.

        Notes / conventions:
        - "valid" points are those marked True in tracking_results["valid"].
        - "LoD-filtered" rasters include ALL points above LoD (i.e., is_below_LoD == False), regardless of outlier flags.
        - "all" rasters include ALL tracked points without any filtering.
        - "outlier-filtered" rasters exclude ONLY outliers (keep LoD-below points unless your 'valid' flag already removes them).
        - New outlier masks write value 1 at locations of the specific outlier reason (0/NaN elsewhere).

        Parameters
        ----------
        folder_path : str
            The folder where all results will be saved.
        save_files : list
            A list of tokens that control what is saved. Supported tokens include:
            - "first_image_matrix", "second_image_matrix"
            - "movement_bearing_valid_tif", "movement_rate_valid_tif"
            - "movement_bearing_outlier_filtered_tif", "movement_rate_outlier_filtered_tif"
            - "movement_bearing_LoD_filtered_tif", "movement_rate_LoD_filtered_tif"
            - "movement_bearing_all_tif", "movement_rate_all_tif"
            - "mask_invalid_tif", "mask_LoD_tif"
            - "mask_outlier_md_tif", "mask_outlier_msd_tif", "mask_outlier_bd_tif", "mask_outlier_bsd_tif"
            - "LoD_points_geojson", "control_points_geojson"
            - "statistical_parameters_txt"
        """
        os.makedirs(folder_path, exist_ok=True)

        # --- Save input images if requested ---
        if "first_image_matrix" in save_files:
            metadata = {
                "driver": "GTiff",
                "count": 1,
                "dtype": self.image1_matrix.dtype,
                "crs": str(self.crs),
                "width": self.image1_matrix.shape[1],
                "height": self.image1_matrix.shape[0],
                "transform": self.image1_transform,
            }
            with rasterio.open(
                    f"{folder_path}/image_{self.image1_observation_date.year}.tif", "w", **metadata
            ) as dst:
                dst.write(self.image1_matrix, 1)

        if "second_image_matrix" in save_files:
            metadata = {
                "driver": "GTiff",
                "count": 1,
                "dtype": self.image2_matrix.dtype,
                "crs": str(self.crs),
                "width": self.image2_matrix.shape[1],
                "height": self.image2_matrix.shape[0],
                "transform": self.image2_transform,
            }
            with rasterio.open(
                    f"{folder_path}/image_{self.image2_observation_date.year}.tif", "w", **metadata
            ) as dst:
                dst.write(self.image2_matrix, 1)

        # --- Always save the full tracking results GeoJSON ---
        self.tracking_results.to_file(
            f"{folder_path}/tracking_results_{self.image1_observation_date.year}_{self.image2_observation_date.year}.geojson",
            driver="GeoJSON",
        )

        # --- Prepare common subsets and guards ---
        tr_all = self.tracking_results
        tr_valid = tr_all.loc[tr_all["valid"]].copy()

        has_lod_col = "is_below_LoD" in tr_all.columns
        tr_above_lod = tr_all.loc[~tr_all["is_below_LoD"]].copy() if has_lod_col else tr_all.copy()

        # Outlier columns may or may not be present; guard accordingly
        has_md = "is_movement_rate_difference_outlier" in tr_all.columns
        has_msd = "is_movement_rate_standard_deviation_outlier" in tr_all.columns
        has_bd = "is_bearing_difference_outlier" in tr_all.columns
        has_bsd = "is_bearing_standard_deviation_outlier" in tr_all.columns

        if any([has_md, has_msd, has_bd, has_bsd]):
            is_outlier = (
                    (tr_all["is_bearing_difference_outlier"] if has_bd else False)
                    | (tr_all["is_bearing_standard_deviation_outlier"] if has_bsd else False)
                    | (tr_all["is_movement_rate_difference_outlier"] if has_md else False)
                    | (tr_all["is_movement_rate_standard_deviation_outlier"] if has_msd else False)
            )
        else:
            # No outlier annotation present
            is_outlier = pd.Series(False, index=tr_all.index)

        tr_without_outliers = tr_all.loc[~is_outlier].copy()

        # --- Helper to make grids safely (avoids errors if empty) ---
        def _make_grid(df, measurements):
            if df.empty:
                # Create an empty grid by falling back to valid subset if available, else skip
                return None
            res_crs = float(self.tracking_parameters.distance_of_tracked_points_px) * self._effective_pixel_size()
            return make_geocube(
                vector_data=df,
                measurements=measurements,
                resolution=res_crs,
            )

        # Grids for various subsets
        meas = ["movement_bearing_pixels", "movement_distance_per_year"]
        grid_valid = _make_grid(tr_valid, meas)
        grid_outlier_filtered = _make_grid(tr_without_outliers, meas)
        grid_lod_filtered = _make_grid(tr_above_lod, meas)  # above LoD, keep outliers
        grid_all = _make_grid(tr_all, meas)  # absolutely all points

        # --- Save requested rasters ---

        # Valid rasters
        if grid_valid is not None:
            if "movement_bearing_valid_tif" in save_files:
                grid_valid["movement_bearing_pixels"].rio.to_raster(
                    f"{folder_path}/movement_bearing_valid_{self.image1_observation_date.year}_{self.image2_observation_date.year}.tif"
                )
            if "movement_rate_valid_tif" in save_files:
                grid_valid["movement_distance_per_year"].rio.to_raster(
                    f"{folder_path}/movement_rate_valid_{self.image1_observation_date.year}_{self.image2_observation_date.year}.tif"
                )

        # Outlier-filtered rasters (exclude outliers, keep everything else)
        if grid_outlier_filtered is not None:
            if "movement_bearing_outlier_filtered_tif" in save_files:
                grid_outlier_filtered["movement_bearing_pixels"].rio.to_raster(
                    f"{folder_path}/movement_bearing_outlier_filtered_{self.image1_observation_date.year}_{self.image2_observation_date.year}.tif"
                )
            if "movement_rate_outlier_filtered_tif" in save_files:
                grid_outlier_filtered["movement_distance_per_year"].rio.to_raster(
                    f"{folder_path}/movement_rate_outlier_filtered_{self.image1_observation_date.year}_{self.image2_observation_date.year}.tif"
                )

        # LoD-filtered rasters (keep all points above LoD, including outliers)
        if grid_lod_filtered is not None:
            if "movement_bearing_LoD_filtered_tif" in save_files:
                grid_lod_filtered["movement_bearing_pixels"].rio.to_raster(
                    f"{folder_path}/movement_bearing_LoD_filtered_{self.image1_observation_date.year}_{self.image2_observation_date.year}.tif"
                )
            if "movement_rate_LoD_filtered_tif" in save_files:
                grid_lod_filtered["movement_distance_per_year"].rio.to_raster(
                    f"{folder_path}/movement_rate_LoD_filtered_{self.image1_observation_date.year}_{self.image2_observation_date.year}.tif"
                )

        # ALL rasters (absolutely all tracked points, no filters)
        if grid_all is not None:
            if "movement_bearing_all_tif" in save_files:
                grid_all["movement_bearing_pixels"].rio.to_raster(
                    f"{folder_path}/movement_bearing_all_{self.image1_observation_date.year}_{self.image2_observation_date.year}.tif"
                )
            if "movement_rate_all_tif" in save_files:
                grid_all["movement_distance_per_year"].rio.to_raster(
                    f"{folder_path}/movement_rate_all_{self.image1_observation_date.year}_{self.image2_observation_date.year}.tif"
                )

        # --- Masks ---

        # invalid mask: marks all non-valid points
        if "mask_invalid_tif" in save_files:
            invalid_mask = tr_all.loc[~tr_all["valid"]].copy()
            invalid_mask["invalid_int"] = 1  # write 1 where invalid (more intuitive and consistent for masks)
            invalid_grid = _make_grid(invalid_mask, ["invalid_int"])
            if invalid_grid is not None:
                invalid_grid["invalid_int"].rio.to_raster(
                    f"{folder_path}/mask_invalid_{self.image1_observation_date.year}_{self.image2_observation_date.year}.tif"
                )

        # reason-specific outlier masks 
        def _write_reason_mask(flag_col: str, token: str, filename_root: str):
            if token in save_files and flag_col in tr_all.columns:
                mask_df = tr_all.loc[tr_all[flag_col]].copy()
                if not mask_df.empty:
                    mask_df["mask_int"] = 1
                    mask_grid = _make_grid(mask_df, ["mask_int"])
                    if mask_grid is not None:
                        mask_grid["mask_int"].rio.to_raster(
                            f"{folder_path}/{filename_root}_{self.image1_observation_date.year}_{self.image2_observation_date.year}.tif"
                        )

        _write_reason_mask("is_movement_rate_difference_outlier", "mask_outlier_md_tif", "mask_outlier_md")
        _write_reason_mask("is_movement_rate_standard_deviation_outlier", "mask_outlier_msd_tif", "mask_outlier_msd")
        _write_reason_mask("is_bearing_difference_outlier", "mask_outlier_bd_tif", "mask_outlier_bd")
        _write_reason_mask("is_bearing_standard_deviation_outlier", "mask_outlier_bsd_tif", "mask_outlier_bsd")

        # LoD points
        if "LoD_points_geojson" in save_files and hasattr(self, "level_of_detection_points"):
            self.level_of_detection_points.to_file(
                f"{folder_path}/LoD_points_{self.image1_observation_date.year}_{self.image2_observation_date.year}.geojson",
                driver="GeoJSON",
            )

        if "control_points_geojson" in save_files and hasattr(self, "tracked_control_points"):
            self.tracked_control_points.to_file(
                f"{folder_path}/control_points_{self.image1_observation_date.year}_{self.image2_observation_date.year}.geojson",
                driver="GeoJSON",
            )

        # --- Statistics text file (robust to missing columns) ---
        if "statistical_parameters_txt" in save_files:
            total_number_of_points = len(tr_all)
            number_of_points_below_lod = int(tr_all["is_below_LoD"].sum()) if has_lod_col else 0
            number_of_outliers = int(is_outlier.sum())
            valid_lod_points = 0
            total_lod_points = 0
            if hasattr(self, "level_of_detection_points") & (self.level_of_detection_points is not None):
                valid_lod_points = int(self.level_of_detection_points["valid"].sum())
                total_lod_points = len(self.level_of_detection_points)

            def _nan_stat(series, fn):
                try:
                    return fn(series)
                except Exception:
                    return np.nan

            with (open(
                    f"{folder_path}/statistical_results_{self.image1_observation_date.year}_{self.image2_observation_date.year}.txt",
                    "w",
            ) as statistics_file):
                lod_str = (
                    f"{self.level_of_detection:.2f}"
                    if getattr(self, "level_of_detection", None) is not None
                    else "NA"
                )
                statistics_file.write(f"Level of Detection: {lod_str}\n")

                statistics_file.write(
                    "Total number of points: " + str(total_number_of_points) + "\n"
                    + "thereof\n"
                    + "\tbelow LoD: " + str(number_of_points_below_lod)
                    + " (" + str(np.round(
                        (number_of_points_below_lod / total_number_of_points * 100) if total_number_of_points else 0,
                        2)) + "%)\n"
                    + "\toutliers: " + str(number_of_outliers)
                    + " (" + str(
                        np.round((number_of_outliers / total_number_of_points * 100) if total_number_of_points else 0,
                                 2)) + "%)\n"
                )
                # Breakdowns if available
                if has_bd:
                    statistics_file.write(
                        "\t\t" + str(
                            int(tr_all["is_bearing_difference_outlier"].sum())) + " bearing difference outliers\n"
                    )
                if has_bsd:
                    statistics_file.write(
                        "\t\t" + str(int(tr_all[
                                             "is_bearing_standard_deviation_outlier"].sum())) + " bearing standard deviation outliers\n"
                    )
                if has_md:
                    statistics_file.write(
                        "\t\t" + str(int(tr_all[
                                             "is_movement_rate_difference_outlier"].sum())) + " movement rate difference outliers\n"
                    )
                if has_msd:
                    statistics_file.write(
                        "\t\t" + str(int(tr_all[
                                             "is_movement_rate_standard_deviation_outlier"].sum())) + " movement rate standard deviation outliers\n"
                    )

                statistics_file.write(
                    "Valid points: " + str(len(tr_valid)) + " ("
                    + str(np.round((len(tr_valid) / total_number_of_points * 100) if total_number_of_points else 0,
                                   2)) + "%)\n"
                )

                # Helper to format stats
                def _fmt(x):
                    return "NA" if x is None or np.isnan(x) else f"{x:.2f}"

                ref_df1 = tr_without_outliers if not tr_without_outliers.empty else tr_all
                statistics_file.write(
                    "Movement rate with points below LoD:\n"
                    + f"\tMean: {_fmt(np.nanmean(ref_df1['movement_distance_per_year']))}\n"
                    + f"\tMedian: {_fmt(np.nanmedian(ref_df1['movement_distance_per_year']))}\n"
                    + f"\tStandard deviation: {_fmt(np.nanstd(ref_df1['movement_distance_per_year']))}\n"
                    + f"\tQ90: {_fmt(np.nanquantile(ref_df1['movement_distance_per_year'], 0.9))}\n"
                    + f"\tQ99: {_fmt(np.nanquantile(ref_df1['movement_distance_per_year'], 0.99))}\n"
                )

                ref_df2 = tr_valid if not tr_valid.empty else tr_above_lod
                statistics_file.write(
                    "Movement rate without points below LoD:\n"
                    + f"\tMean: {_fmt(np.nanmean(ref_df2['movement_distance_per_year']))}\n"
                    + f"\tMedian: {_fmt(np.nanmedian(ref_df2['movement_distance_per_year']))}\n"
                    + f"\tStandard deviation: {_fmt(np.nanstd(ref_df2['movement_distance_per_year']))}\n"
                    + f"\tQ90: {_fmt(np.nanquantile(ref_df2['movement_distance_per_year'], 0.9))}\n"
                    + f"\tQ99: {_fmt(np.nanquantile(ref_df2['movement_distance_per_year'], 0.99))}\n"
                )

                if hasattr(self, "level_of_detection_points") & (self.level_of_detection_points is not None
                ) and len(self.level_of_detection_points) > 0:
                    statistics_file.write(
                        "Movement rate of LoD points:\n"
                        + f"\tMean: {_fmt(np.nanmean(self.level_of_detection_points['movement_distance_per_year']))}\n"
                        + f"\tMedian: {_fmt(np.nanmedian(self.level_of_detection_points['movement_distance_per_year']))}\n"
                        + f"\tStandard deviation: {_fmt(np.nanstd(self.level_of_detection_points['movement_distance_per_year']))}\n"
                        + f"\tQ90: {_fmt(np.nanquantile(self.level_of_detection_points['movement_distance_per_year'], 0.9))}\n"
                        + f"\tQ99: {_fmt(np.nanquantile(self.level_of_detection_points['movement_distance_per_year'], 0.99))}\n"
                        + f"\tUsed points: {valid_lod_points} points\n"
                    )

        # --- Parameter logs ---
        with open(
                f"{folder_path}/parameters_{self.image1_observation_date.year}_{self.image2_observation_date.year}.txt",
                "w",
        ) as text_file:
            text_file.write(self.alignment_parameters.__str__())

        if self.tracking_parameters is not None:
            with open(
                    f"{folder_path}/parameters_{self.image1_observation_date.year}_{self.image2_observation_date.year}.txt",
                    "a",
            ) as text_file:
                text_file.write(self.tracking_parameters.__str__())

        if self.filter_parameters is not None:
            with (open(folder_path + "/parameters_" + str(self.image1_observation_date.year)
                       + "_" + str(self.image2_observation_date.year)
                       + ".txt", "a") as text_file):
                text_file.write(self.filter_parameters.__str__())

        # --- Plots and LoD annotation  ---
        if self.level_of_detection is not None:
            if "mask_LoD_tif" in save_files and has_lod_col:
                lod_mask = tr_all.loc[tr_all["is_below_LoD"]].copy()
                lod_mask["is_below_LoD_int"] = 1  # write 1 where below LoD
                lod_grid = _make_grid(lod_mask, ["is_below_LoD_int"])
                if lod_grid is not None:
                    lod_grid["is_below_LoD_int"].rio.to_raster(
                        f"{folder_path}/mask_LoD_{self.image1_observation_date.year}_{self.image2_observation_date.year}.tif"
                    )

            plot_movement_of_points_with_valid_mask(
                self.image1_matrix,
                self.image1_transform,
                self.tracking_results,
                save_path=f"{folder_path}/tracking_results_{self.image1_observation_date.year}_{self.image2_observation_date.year}.jpg",
            )
        else:
            plot_movement_of_points(
                self.image1_matrix,
                self.image1_transform,
                self.tracking_results,
                save_path=f"{folder_path}/tracking_results_{self.image1_observation_date.year}_{self.image2_observation_date.year}.jpg",
            )

    def load_results(self, file_path, reference_area):
        saved_tracking_results = gpd.read_file(file_path)
        saved_tracking_results = saved_tracking_results.loc[
            :, ["row", "column", "movement_row_direction", "movement_column_direction",
                "movement_distance_pixels", "movement_bearing_pixels", "movement_distance",
                "movement_distance_per_year", "geometry"]]
        saved_tracking_results["valid"] = True
        self.align_images(reference_area)
        self.tracking_results = saved_tracking_results
