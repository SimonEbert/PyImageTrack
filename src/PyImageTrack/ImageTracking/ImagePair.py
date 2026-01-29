import logging
import os

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.plot
from matplotlib import image as mpimg
from rasterio.crs import CRS
from geocube.api.core import make_geocube
from scipy.sparse.csgraph import depth_first_tree
from shapely.geometry import box
import scipy
import sklearn

# Parameter classes
from ..Parameters.TrackingParameters import TrackingParameters
from ..Parameters.FilterParameters import FilterParameters
from ..Parameters.AlignmentParameters import AlignmentParameters

# Alignment and Tracking functions
from .TrackMovement import track_movement_lsm, move_indices_from_transformation_matrix
from ..CreateGeometries.HandleGeometries import (
    crop_images_to_intersection,
    georeference_tracked_points,
    grid_points_on_polygon_by_distance,
    random_points_on_polygon_by_number,
)
from ..CreateGeometries.DepthImageConversion import calculate_displacement_from_depth_images

# filter functions
from ..DataProcessing.DataPostprocessing import (
    filter_lod_points,
    filter_outliers_full,
)
# DataPreProcessing
from ..DataProcessing.ImagePreprocessing import equalize_adapthist_images, undistort_camera_image, undistort_polygon
from .AlignImages import align_images_lsm_scarce
# Plotting
from ..Plots.MakePlots import (
    plot_movement_of_points,
    plot_movement_of_points_with_valid_mask, plot_raster_and_geometry,
)
# Date Handling
from ..Utils import parse_date
# Effective search extent calculation
from ..Utils import make_effective_extents_from_deltas


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
        self.years_between_observations = None
        self.safe_image_bounds_tracking = None
        self.safe_image_bounds_alignment = None

        # Optional: store original (true-color) matrices for writing aligned products
        self.image1_matrix_original = None
        self.image2_matrix_original = None

        # Optional: Store position images corresponding to non-georeferenced images for 3d displacement calculation
        self.depth_image1 = None
        self.depth_image2 = None

        # Parameters
        self.tracking_parameters = TrackingParameters(parameter_dict=parameter_dict)
        self.filter_parameters = None
        self.alignment_parameters = AlignmentParameters(parameter_dict=parameter_dict)
        # Fake georef switches (fed via run_pipeline param_dict)
        self.use_no_georeferencing = bool(
            parameter_dict.get("use_no_georeferencing", False)) if parameter_dict else False
        self.fake_pixel_size = float(parameter_dict.get("fake_pixel_size", 1.0)) if parameter_dict else 1.0
        self.downsample_factor = int(parameter_dict.get("downsample_factor", 1)) if parameter_dict else 1
        if self.downsample_factor < 1:
            self.downsample_factor = 1

        # Meta-Data and results
        self.crs = parameter_dict.get("crs", None)
        self.image_bands = parameter_dict.get("image_bands", None)
        self.coordinate_system_unit_name = parameter_dict.get("unit_name", None)
        self.use_adaptive_tracking_window = parameter_dict.get("use_adaptive_tracking_window", False)
        self.tracked_control_points = None
        self.tracking_results = None
        self.level_of_detection = None
        self.level_of_detection_points = None
        self.convert_to_3d_displacement = parameter_dict.get("convert_to_3d_displacement", False)
        self.undistort_image = parameter_dict.get("undistort_image", False)
        self.camera_intrinsics_matrix = parameter_dict.get("camera_intrinsics_matrix", None)
        self.camera_distortion_coefficients = parameter_dict.get("camera_distortion_coefficients", None)
        self.camera_to_3d_coordinates_transform = parameter_dict.get("camera_to_3d_coordinates_transform", None)
        if self.convert_to_3d_displacement:
            self.displacement_column_name = "3d_displacement_distance_per_year"
        else:
            self.displacement_column_name = "movement_distance_per_year"

    def _effective_pixel_size(self) -> float:
        """CRS units per pixel (assumes square pixels)."""
        if self.image1_transform is None:
            return 1.0
        a = float(self.image1_transform.a)
        e = float(self.image1_transform.e)
        return max(abs(a), abs(e))

    def _downsample_array(self, arr: np.ndarray, factor: int) -> np.ndarray:
        if factor <= 1:
            return arr
        if arr.ndim == 3:
            return arr[:, ::factor, ::factor]
        return arr[::factor, ::factor]

    def _downsample_transform(self, transform, factor: int):
        if factor <= 1:
            return transform
        from affine import Affine
        return transform * Affine.scale(factor, factor)

    def select_image_channels(self, selected_channels: int = None):
        if self.image1_matrix.ndim == 2:
            return
        if self.image1_matrix.ndim != self.image2_matrix.ndim:
            raise ValueError("Got matrices with a different number of dimensions, signifying possibly a single-channel "
                             "vs. multi-channel mismatch between the images.")
        if selected_channels is None:
            selected_channels = [0, 1, 2]
        if len(self.image1_matrix.shape) == 3:
            self.image1_matrix = self.image1_matrix[selected_channels, :, :]
            self.image2_matrix = self.image2_matrix[selected_channels, :, :]

    def load_images_from_file(self, filename_1: str, observation_date_1: str, filename_2: str, observation_date_2: str,NA_value: float = None):
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
        Returns
        -------
        """
        file1 = rasterio.open(filename_1, 'r')
        file2 = rasterio.open(filename_2, 'r')
        # Choose path: true georef vs fake georef
        no_crs_either = (file1.crs is None) or (file2.crs is None)
        force_fake = self.use_no_georeferencing or no_crs_either

        factor = getattr(self, "downsample_factor", 1)
        if factor is None or factor < 1:
            factor = 1

        if not force_fake:
            if file1.crs != file2.crs:
                raise ValueError("Got images with crs " + str(file1.crs) + " and " + str(file2.crs) +
                                 " but the two images must have the same crs.")
            if self.crs != file1.crs:
                raise ValueError(
                    "Specified crs of data in config to be " + str(self.crs) + "but images are given with crs" +
                    str(file1.crs))
            self.coordinate_system_unit_name = file1.crs.axis_info[0].unit_name

            # Spatial intersection (true georef)
            poly1 = box(*file1.bounds)
            poly2 = box(*file2.bounds)
            intersection = poly1.intersection(poly2)

            ([self.image1_matrix, self.image1_transform],
             [self.image2_matrix, self.image2_transform]) = crop_images_to_intersection(file1, file2)

            # Store original matrices before any preprocessing (e.g. undistorting, channel selection, CLAHE)
            self.image1_matrix_original = self.image1_matrix.copy()
            self.image2_matrix_original = self.image2_matrix.copy()

            if factor > 1:
                self.image1_matrix = self._downsample_array(self.image1_matrix, factor)
                self.image2_matrix = self._downsample_array(self.image2_matrix, factor)
                self.image1_transform = self._downsample_transform(self.image1_transform, factor)
                self.image2_transform = self._downsample_transform(self.image2_transform, factor)

            # Keep search windows inside valid area
            px_size = max(-file1.transform[4], file1.transform[0]) * factor  # pixel size (>0)
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

        else:
            # FAKE georeferencing path (e.g., JPGs)
            arr1 = file1.read()  # (bands, rows, cols)
            arr2 = file2.read()

            def squeeze(arr):
                return arr[0] if arr.shape[0] == 1 else arr

            arr1 = squeeze(arr1)
            arr2 = squeeze(arr2)

            # Store original matrices before any preprocessing (e.g. undistorting, channel selection, CLAHE)
            self.image1_matrix_original = arr1.copy()
            self.image2_matrix_original = arr2.copy()

            # Adjust unit name if we are working in the non-georeferenced setting and the name has not been set
            if self.coordinate_system_unit_name is None:
                self.coordinate_system_unit_name = "pixel"

            if self.undistort_image:
                arr1 = undistort_camera_image(arr1, self.camera_intrinsics_matrix, self.camera_distortion_coefficients)
                arr2 = undistort_camera_image(arr2, self.camera_intrinsics_matrix, self.camera_distortion_coefficients)

            self.image1_matrix = arr1
            self.image2_matrix = arr2

            if self.convert_to_3d_displacement:
                basename1 = os.path.splitext(os.path.basename(filename_1))[0]
                basename2 = os.path.splitext(os.path.basename(filename_2))[0]
                depth_image1_path = os.path.join(os.path.dirname(filename_1), "Depth_images", basename1
                                                 + "_depth.tiff")
                depth_image2_path = os.path.join(os.path.dirname(filename_2), "Depth_images", basename2
                                                 + "_depth.tiff")
                depth_image1 = rasterio.open(depth_image1_path, 'r').read()
                depth_image2 = rasterio.open(depth_image2_path, 'r').read()
                depth_image1 = squeeze(depth_image1)
                depth_image2 = squeeze(depth_image2)
                if self.undistort_image:
                    depth_image1 = undistort_camera_image(depth_image1, self.camera_intrinsics_matrix,
                                                          self.camera_distortion_coefficients)
                    depth_image2 = undistort_camera_image(depth_image2, self.camera_intrinsics_matrix,
                                                          self.camera_distortion_coefficients)
                self.depth_image1 = depth_image1
                self.depth_image2 = depth_image2

            # Top-left crop to common size
            h = min(self.image1_matrix.shape[-2], self.image2_matrix.shape[-2])
            w = min(self.image1_matrix.shape[-1], self.image2_matrix.shape[-1])
            self.image1_matrix = self.image1_matrix[..., :h, :w]
            self.image2_matrix = self.image2_matrix[..., :h, :w]
            if factor > 1:
                self.image1_matrix = self._downsample_array(self.image1_matrix, factor)
                self.image2_matrix = self._downsample_array(self.image2_matrix, factor)
                self.depth_image1 = self._downsample_array(self.depth_image1, factor)
                self.depth_image2 = self._downsample_array(self.depth_image2, factor)

            # Synthetic transform: origin (0,0) upper-left, pixel size = fake_pixel_size
            from affine import Affine
            px = float(self.fake_pixel_size) * factor
            tform = Affine(px, 0, 0, 0, -px, 0)  # x = px*col ; y = -px*row

            self.image1_transform = tform
            self.image2_transform = tform

            # Bounds in that CRS (pixel grid space), then shrink by search radius
            from rasterio.transform import array_bounds
            bounds_poly = box(*array_bounds(h, w, tform))

            def make_safe_bounds_from_search_extents(extents):
                if not extents:
                    raise ValueError("Search_extent_px must be set (tuple posx,negx,posy,negy).")
                buffer_len = px * max(extents)

                image_bounds = gpd.GeoDataFrame({'geometry': [bounds_poly]}, crs=self.crs).buffer(-buffer_len)
                image_bounds = gpd.GeoDataFrame(geometry=image_bounds, crs=self.crs)
                image_bounds = image_bounds.rename(columns={0: "geometry"})
                image_bounds.set_geometry("geometry", inplace=True)
                return image_bounds

            self.safe_image_bounds_tracking = make_safe_bounds_from_search_extents(getattr(self.tracking_parameters, "search_extent_px", None))
            self.safe_image_bounds_alignment = make_safe_bounds_from_search_extents(getattr(self.alignment_parameters, "control_search_extent_px", None))

        self.image1_observation_date = parse_date(observation_date_1)
        self.image2_observation_date = parse_date(observation_date_2)#
        # compute years_between (hour-precise)
        delta_hours = (self.image2_observation_date - self.image1_observation_date).total_seconds() / 3600.0
        self.years_between_observations = delta_hours / (24.0 * 365.25)

        if NA_value is not None:
            self.image1_matrix[self.image1_matrix == NA_value] = 0
            self.image2_matrix[self.image2_matrix == NA_value] = 0


        # Compute effective extents if needed
        if ((self.tracking_parameters.search_extent_px is not None) &
                (self.tracking_parameters.search_extent_full_cell is None)):
            self.tracking_parameters.search_extent_full_cell = (
                make_effective_extents_from_deltas(
                    self.tracking_parameters.search_extent_px,
                    self.tracking_parameters.movement_cell_size,
                    years_between=self.years_between_observations if self.use_adaptive_tracking_window else 1.0,
                    cap_per_side=None
                ))
        else:
            raise ValueError("Set exactly one of 'search_extent_px' and 'search_extent_full_cell'.")


        if ((self.alignment_parameters.control_search_extent_px is not None) &
            (self.alignment_parameters.control_search_extent_full_cell is None)):
            self.alignment_parameters.control_search_extent_full_cell = (
                make_effective_extents_from_deltas(
                    self.alignment_parameters.control_search_extent_px,
                    self.alignment_parameters.control_cell_size,
                    years_between=1.0,
                    cap_per_side=None
                )
            )
        else:
            raise ValueError("Set exactly one of 'control_search_extent_px' and 'control_search_extent_full_cell'.")


        # Select image bands
        if self.image_bands is not None:
            self.select_image_channels(selected_channels=self.image_bands)
        elif self.image1_matrix.ndim == 3:
            self.image_bands = self.image1_matrix.shape[0]
        else:
            self.image_bands = None

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
        self.safe_image_bounds_tracking = image_bounds

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
        if reference_area.crs != self.crs:
            raise ValueError("Got reference area with crs " + str(reference_area.crs) + " and images with crs "
                             + str(self.crs) + ". Reference area and images are supposed to have the same crs.")

        if self.undistort_image:
            reference_area = undistort_polygon(reference_area, self.image1_matrix_original.shape[-2:],
                                               self.camera_intrinsics_matrix,
                                               self.camera_distortion_coefficients)

        reference_area = gpd.GeoDataFrame(reference_area.intersection(self.safe_image_bounds_alignment))
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


    def track_points(self, tracking_area: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
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

        if self.undistort_image:
            tracking_area = undistort_polygon(tracking_area, self.image1_matrix_original.shape[-2:],
                                               self.camera_intrinsics_matrix,
                                               self.camera_distortion_coefficients)

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
        image_bounds_values = self.safe_image_bounds_tracking.bounds.iloc[0]
        within_image_mask = ((image_bounds_values.minx <= points_to_be_tracked.geometry.x) &
                            (image_bounds_values.maxx >= points_to_be_tracked.geometry.x) &
                            (image_bounds_values.miny <= points_to_be_tracked.geometry.y) &
                            (image_bounds_values.maxy >= points_to_be_tracked.geometry.y))

        points_to_be_tracked = points_to_be_tracked[within_image_mask]


        tracked_points = track_movement_lsm(self.image1_matrix, self.image2_matrix, self.image1_transform,
                                            points_to_be_tracked=points_to_be_tracked,
                                            tracking_parameters=self.tracking_parameters,
                                            alignment_tracking=False,
                                            task_label="Tracking points for movement tracking")
        # calculate the years between observations from the two given observation dates
        delta_hours = (self.image2_observation_date - self.image1_observation_date).total_seconds() / 3600.0
        years_between_observations = delta_hours / (24.0 * 365.25)

        if self.convert_to_3d_displacement:
            georeferenced_tracked_points = calculate_displacement_from_depth_images(
                tracked_points, depth_image_time1=self.depth_image1,
                depth_image_time2=self.depth_image2, camera_intrinsics_matrix=self.camera_intrinsics_matrix,
                camera_to_3d_coordinates_transform=self.camera_to_3d_coordinates_transform,
                years_between_observations=years_between_observations)
            if self.coordinate_system_unit_name is "pixel":
                self.coordinate_system_unit_name = "meter"
        else:
            georeferenced_tracked_points = georeference_tracked_points(
                tracked_pixels=tracked_points, raster_transform=self.image1_transform, crs=tracking_area.crs,
                years_between_observations=years_between_observations)

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
            plot_movement_of_points(self.image1_matrix, self.image1_transform, self.tracking_results,
                                    unit_name=self.coordinate_system_unit_name)
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
            plot_movement_of_points_with_valid_mask(self.image1_matrix, self.image1_transform, self.tracking_results,
                                                    unit_name=self.coordinate_system_unit_name)
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
        self.tracking_results = filter_outliers_full(self.tracking_results, filter_parameters,
                                                     self.displacement_column_name)

    def track_lod_points(self, points_for_lod_calculation: gpd.GeoDataFrame,
                         years_between_observations) -> gpd.GeoDataFrame:
        """
        Performs tracking on given stable points (for calculating an LoD)
        Parameters
        ----------
        points_for_lod_calculation
        years_between_observations

        Returns
        -------
        tracked_points: gpd.GeoDataFrame
            The tracked points which can be used for calculating the LoD.
        """
        points = points_for_lod_calculation
        tracking_parameters = self.tracking_parameters
        if tracking_parameters.initial_shift_values is not None:
            tracking_parameters.initial_shift_values = [0,0]

        tracked_points = track_movement_lsm(
            image1_matrix=self.image1_matrix, image2_matrix=self.image2_matrix, image_transform=self.image1_transform,
            points_to_be_tracked=points, tracking_parameters=tracking_parameters, alignment_tracking=False,
            save_columns=["movement_row_direction",
                          "movement_column_direction",
                          "movement_distance_pixels",
                          "movement_bearing_pixels",
                          ],
            task_label="Tracking points for LoD"
        )
        tracked_control_pixels_valid = tracked_points[tracked_points["movement_row_direction"].notna()]

        if len(tracked_control_pixels_valid) == 0:
            raise ValueError(
                "Was not able to track any points with a cross-correlation higher than the cross-correlation "
                "threshold. Cross-correlation values were " + str(
                    list(tracked_points[
                             "correlation_coefficient"])) + " (None-values may signify problems during tracking).")

        print("Used " + str(len(tracked_control_pixels_valid)) + " pixels for LoD calculation.")

        if self.convert_to_3d_displacement:
            tracked_points = calculate_displacement_from_depth_images(
                tracked_control_pixels_valid,self.depth_image1,self.depth_image2,
                camera_intrinsics_matrix=self.camera_intrinsics_matrix,
                camera_to_3d_coordinates_transform=self.camera_to_3d_coordinates_transform,
                years_between_observations=years_between_observations)
        else:
            tracked_points = georeference_tracked_points(tracked_control_pixels_valid,
                                                         self.image1_transform,
                                                         crs=self.crs,
                                                         years_between_observations=years_between_observations)

        return tracked_points

    def calculate_lod(self, reference_area: gpd.GeoDataFrame,
                      filter_parameters: FilterParameters = None) -> None:
        """
        Calculates the Level of Detection of a matching between two images. For calculating the LoD a specified number
        of points are generated randomly in some reference area, which is assumed to be stable. The level of detection
        is defined as some quantile of the movement rates of these points. The quantile can be set in the
        tracking parameters. This method sets the value of the lod as self.level_of_detection. The level of detection is
        given in movement per year.
        Parameters
        ----------
        points_for_lod_calculation: gpd.GeoDataFrame
            The area on which the randomly distributed points will be created (assumed to be stable)
        filter_parameters
        Returns
        -------
        None
        """
        if self.undistort_image:
            reference_area = undistort_polygon(reference_area, self.image1_matrix_original.shape[-2:],
                                               self.camera_intrinsics_matrix,
                                               self.camera_distortion_coefficients)

        points_for_lod_calculation = random_points_on_polygon_by_number(
            polygon=reference_area,
            number_of_points=filter_parameters.number_of_points_for_level_of_detection
        )

        # Set used filter parameters if given as a variable, also set them as correct object for ImagePair
        if filter_parameters is None:
            if self.filter_parameters is None:
                return
            else:
                filter_parameters = self.filter_parameters
        else:
            self.filter_parameters = filter_parameters

        points_for_lod_calculation = gpd.GeoDataFrame(
            points_for_lod_calculation.intersection(self.safe_image_bounds_tracking.geometry[0]))
        points_for_lod_calculation.rename(columns={0: 'geometry'}, inplace=True)
        points_for_lod_calculation.set_geometry('geometry', inplace=True)

        delta_hours = (self.image2_observation_date - self.image1_observation_date).total_seconds() / 3600.0
        years_between_observations = delta_hours / (24.0 * 365.25)

        # check if a LoD filter parameter is provided, if this is None, don't perform LoD calculation
        if (filter_parameters.level_of_detection_quantile is None
                or filter_parameters.number_of_points_for_level_of_detection is None):
            return

        level_of_detection_quantile = filter_parameters.level_of_detection_quantile

        unfiltered_level_of_detection_points = self.track_lod_points(
            points_for_lod_calculation=points_for_lod_calculation,
            years_between_observations=years_between_observations)
        self.level_of_detection_points = unfiltered_level_of_detection_points

        self.level_of_detection = np.nanquantile(unfiltered_level_of_detection_points[self.displacement_column_name],
                                                 level_of_detection_quantile)

        if points_for_lod_calculation.crs is not None:
            unit_name = points_for_lod_calculation.crs.axis_info[0].unit_name
        else:
            unit_name = "pixel"
        print("Found level of detection with quantile " + str(level_of_detection_quantile) + " as "
              + str(np.round(self.level_of_detection, decimals=5)) + " " + str(
            unit_name) + "/year")

    def filter_lod_points(self) -> None:
        """
        Sets the movement distance of all points that fall below the calculated level of detection to 0 and their
        movement bearing to NaN. Note that this directly affects the dataframe self.tracking_results.
        Returns
        -------
        """

        if not self.valid_alignment_possible:
            return
        self.tracking_results = filter_lod_points(self.tracking_results, self.level_of_detection,
                                                  self.displacement_column_name)

    def full_filter(self, reference_area, filter_parameters: FilterParameters):

        if self.undistort_image:
            reference_area = undistort_polygon(reference_area, self.image1_matrix_original.shape[-2:],
                                               self.camera_intrinsics_matrix,
                                               self.camera_distortion_coefficients)

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

        # --- Always save the full tracking results GeoJSON ---

        self.tracking_results.to_file(
            f"{folder_path}/tracking_results_{self.image1_observation_date.strftime(format='%Y-%m-%d')}_{self.image2_observation_date.strftime(format='%Y-%m-%d')}.fgb",
            driver="FlatGeobuf"
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
        def _make_raster(df, measurements):
            if df.empty:
                # return no grid
                return None

            bounds = df.geometry.total_bounds
            height = np.abs(bounds[3] - bounds[1])
            width = np.abs(bounds[2] - bounds[0])

            # Calculate resolution and therefore full raster dimensions from distance of tracked points for raster cells
            # as large as possible without burning several points to the same raster pixel
            res = float(self.tracking_parameters.distance_of_tracked_points_px) * self._effective_pixel_size()
            height = int(np.ceil(height / res)) + 1  # + 1 needed for proper centering of points w.r.t. raster grid
            width = int(np.ceil(width / res)) + 1  # + 1 needed for proper centering of points w.r.t. raster grid

            # if df.crs is not None:
            #     transform = rasterio.transform.from_origin(bounds[0], bounds[3], res, res)
            #     crs = df.crs
            # else:
            #     # If no crs is given, assume results in image coordinates and rasterize with identity transform (raster
            #     # will also be given in image coordinates)
            #     transform = rasterio.transform.from_origin(bounds[0],bounds[3],res,res)
            #     crs = None
            transform = rasterio.transform.from_origin(bounds[0] - res / 2, bounds[3] + res / 2, res, res)
            crs = df.crs

            data = {}
            for measurement in measurements:
                shapes = ((geometry, value) for geometry, value in zip(df.geometry, df[measurement]))
                data[measurement] = rasterio.features.rasterize(
                    shapes=shapes,
                    out_shape=(height, width),
                    transform=transform,
                    fill=np.nan,
                    dtype="float32"
                )
            return {
                "raster": data,
                "transform": transform,
                "crs": crs
            }

        def _save_raster_as_tif(path, raster, transform, crs, driver="GTiff"):
            if driver == "JPEG":
                dtype = "uint8"
            else:
                dtype = "float32"
            if raster.ndim == 2:
                raster_to_write = raster[np.newaxis, ...]
                count = 1
            elif raster.ndim == 3:
                raster_to_write = raster
                count = raster.shape[0]
            else:
                raise ValueError("Expecting raster to be 2D or 3D, but supplied raster with shape" + str(raster.shape))
            with rasterio.open(
                    path,
                    "w",
                    driver=driver,
                    height=raster_to_write.shape[-2],
                    width=raster_to_write.shape[-1],
                    transform=transform,
                    crs=crs,
                    count=count,
                    nodata=np.nan,
                    dtype=dtype
            ) as dst:
                dst.write(raster_to_write)

                # --- Save input images if requested ---

        if "first_image_matrix" in save_files:
            _save_raster_as_tif(
                path=f"{folder_path}/image_{self.image1_observation_date.strftime(format='%Y-%m-%d')}.jpeg",
                raster=self.image1_matrix.astype(np.uint8),
                transform=self.image1_transform,
                crs=self.crs,
                driver="JPEG"
            )

        if "second_image_matrix" in save_files:
            _save_raster_as_tif(
                path=f"{folder_path}/image_{self.image2_observation_date.strftime(format='%Y-%m-%d')}.jpeg",
                raster=self.image2_matrix.astype(np.uint8),
                transform=self.image1_transform,
                crs=self.crs,
                driver="JPEG"
            )

        # Grids for various subsets

        meas = ["movement_bearing_pixels", self.displacement_column_name]

        raster_valid = _make_raster(tr_valid, meas)
        raster_outlier_filtered = _make_raster(tr_without_outliers, meas)
        raster_lod_filtered = _make_raster(tr_above_lod, meas)  # above LoD, keep outliers
        raster_all = _make_raster(tr_all, meas)  # absolutely all points

        # --- Save requested rasters ---

        # Valid rasters
        if raster_valid is not None:
            if "movement_bearing_valid_tif" in save_files:
                _save_raster_as_tif(
                    path=f"{folder_path}/movement_bearing_valid_{self.image1_observation_date.strftime(format='%Y-%m-%d')}_{self.image2_observation_date.strftime(format='%Y-%m-%d')}.tif",
                    raster=raster_valid["raster"]["movement_bearing_pixels"],
                    transform=raster_valid["transform"],
                    crs=raster_valid["crs"]
                )

            if "movement_rate_valid_tif" in save_files:
                _save_raster_as_tif(
                    path=f"{folder_path}/movement_rate_valid_{self.image1_observation_date.strftime(format='%Y-%m-%d')}_{self.image2_observation_date.strftime(format='%Y-%m-%d')}.tif",
                    raster=raster_valid["raster"][self.displacement_column_name],
                    transform=raster_valid["transform"],
                    crs=raster_valid["crs"]
                )

        # Outlier-filtered rasters (exclude outliers, keep everything else)
        if raster_outlier_filtered is not None:
            if "movement_bearing_outlier_filtered_tif" in save_files:
                _save_raster_as_tif(
                    path=f"{folder_path}/movement_bearing_outlier_filtered_{self.image1_observation_date.strftime(format='%Y-%m-%d')}_{self.image2_observation_date.strftime(format='%Y-%m-%d')}.tif",
                    raster=raster_outlier_filtered["raster"]["movement_bearing_pixels"],
                    transform=raster_outlier_filtered["transform"],
                    crs=raster_outlier_filtered["crs"]
                )
            if "movement_rate_outlier_filtered_tif" in save_files:
                _save_raster_as_tif(
                    path=f"{folder_path}/movement_rate_outlier_filtered_{self.image1_observation_date.strftime(format='%Y-%m-%d')}_{self.image2_observation_date.strftime(format='%Y-%m-%d')}.tif",
                    raster=raster_outlier_filtered["raster"][self.displacement_column_name],
                    transform=raster_outlier_filtered["transform"],
                    crs=raster_outlier_filtered["crs"]
                )

        # LoD-filtered rasters (keep all points above LoD, including outliers)
        if raster_lod_filtered is not None:
            if "movement_bearing_LoD_filtered_tif" in save_files:
                _save_raster_as_tif(
                    path=f"{folder_path}/movement_bearing_LoD_filtered_{self.image1_observation_date.strftime(format='%Y-%m-%d')}_{self.image2_observation_date.strftime(format='%Y-%m-%d')}.tif",
                    raster=raster_lod_filtered["raster"]["movement_bearing_pixels"],
                    transform=raster_lod_filtered["transform"],
                    crs=raster_lod_filtered["crs"]
                )
            if "movement_rate_LoD_filtered_tif" in save_files:
                _save_raster_as_tif(
                    path=f"{folder_path}/movement_rate_LoD_filtered_{self.image1_observation_date.strftime(format='%Y-%m-%d')}_{self.image2_observation_date.strftime(format='%Y-%m-%d')}.tif",
                    raster=raster_lod_filtered["raster"][self.displacement_column_name],
                    transform=raster_lod_filtered["transform"],
                    crs=raster_lod_filtered["crs"]
                )

        # ALL rasters (absolutely all tracked points, no filters)
        if raster_all is not None:
            if "movement_bearing_all_tif" in save_files:
                _save_raster_as_tif(
                    path=f"{folder_path}/movement_bearing_all_{self.image1_observation_date.strftime(format='%Y-%m-%d')}_{self.image2_observation_date.strftime(format='%Y-%m-%d')}.tif",
                    raster=raster_all["raster"]["movement_bearing_pixels"],
                    transform=raster_all["transform"],
                    crs=raster_all["crs"]
                )

            if "movement_rate_all_tif" in save_files:
                _save_raster_as_tif(
                    path=f"{folder_path}/movement_rate_all_{self.image1_observation_date.strftime(format='%Y-%m-%d')}_{self.image2_observation_date.strftime(format='%Y-%m-%d')}.tif",
                    raster=raster_all["raster"][self.displacement_column_name],
                    transform=raster_all["transform"],
                    crs=raster_all["crs"]
                )

        # --- Masks ---

        # invalid mask: marks all non-valid points
        if "mask_invalid_tif" in save_files:
            invalid_mask = tr_all.loc[~tr_all["valid"]].copy()
            invalid_mask["invalid_int"] = 1  # write 1 where invalid (more intuitive and consistent for masks)
            invalid_raster = _make_raster(invalid_mask, ["invalid_int"])
            if invalid_raster is not None:
                _save_raster_as_tif(
                    path=f"{folder_path}/mask_invalid_{self.image1_observation_date.strftime(format='%Y-%m-%d')}_{self.image2_observation_date.strftime(format='%Y-%m-%d')}.tif",
                    raster=invalid_raster["raster"]["invalid_int"],
                    transform=invalid_raster["transform"],
                    crs=invalid_raster["crs"]
                )

        # reason-specific outlier masks
        def _write_reason_mask(flag_col: str, token: str, filename_root: str):
            if token in save_files and flag_col in tr_all.columns:
                mask_df = tr_all.loc[tr_all[flag_col]].copy()
                if not mask_df.empty:
                    mask_df["mask_int"] = 1
                    mask_grid = _make_raster(mask_df, ["mask_int"])
                    if mask_grid is not None:
                        _save_raster_as_tif(
                            path=f"{folder_path}/{filename_root}_{self.image1_observation_date.strftime(format='%Y-%m-%d')}_{self.image2_observation_date.strftime(format='%Y-%m-%d')}.tif",
                            raster=mask_grid["raster"]["mask_int"],
                            transform=mask_grid["transform"],
                            crs=mask_grid["crs"]
                        )

        _write_reason_mask("is_movement_rate_difference_outlier", "mask_outlier_md_tif", "mask_outlier_md")
        _write_reason_mask("is_movement_rate_standard_deviation_outlier", "mask_outlier_msd_tif", "mask_outlier_msd")
        _write_reason_mask("is_bearing_difference_outlier", "mask_outlier_bd_tif", "mask_outlier_bd")
        _write_reason_mask("is_bearing_standard_deviation_outlier", "mask_outlier_bsd_tif", "mask_outlier_bsd")

        # LoD points
        if "LoD_points_geojson" in save_files and self.level_of_detection_points is not None:
            self.level_of_detection_points.to_file(
                f"{folder_path}/LoD_points_{self.image1_observation_date.strftime(format='%Y-%m-%d')}_{self.image2_observation_date.strftime(format='%Y-%m-%d')}.fgb",
                driver="FlatGeobuf"
            )

        if "control_points_geojson" in save_files and self.tracked_control_points is not None:
            self.tracked_control_points.to_file(
                f"{folder_path}/control_points_{self.image1_observation_date.strftime(format='%Y-%m-%d')}_{self.image2_observation_date.strftime(format='%Y-%m-%d')}.fgb",
                driver="FlatGeobuf"
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
                    f"{folder_path}/statistical_results_{self.image1_observation_date.strftime(format='%Y-%m-%d')}_{self.image2_observation_date.strftime(format='%Y-%m-%d')}.txt",
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
                    + f"\tMean: {_fmt(np.nanmean(ref_df1[self.displacement_column_name]))}\n"
                    + f"\tMedian: {_fmt(np.nanmedian(ref_df1[self.displacement_column_name]))}\n"
                    + f"\tStandard deviation: {_fmt(np.nanstd(ref_df1[self.displacement_column_name]))}\n"
                    + f"\tQ90: {_fmt(np.nanquantile(ref_df1[self.displacement_column_name], 0.9))}\n"
                    + f"\tQ99: {_fmt(np.nanquantile(ref_df1[self.displacement_column_name], 0.99))}\n"
                )

                ref_df2 = tr_valid if not tr_valid.empty else tr_above_lod
                statistics_file.write(
                    "Movement rate without points below LoD:\n"
                    + f"\tMean: {_fmt(np.nanmean(ref_df2[self.displacement_column_name]))}\n"
                    + f"\tMedian: {_fmt(np.nanmedian(ref_df2[self.displacement_column_name]))}\n"
                    + f"\tStandard deviation: {_fmt(np.nanstd(ref_df2[self.displacement_column_name]))}\n"
                    + f"\tQ90: {_fmt(np.nanquantile(ref_df2[self.displacement_column_name], 0.9))}\n"
                    + f"\tQ99: {_fmt(np.nanquantile(ref_df2[self.displacement_column_name], 0.99))}\n"
                )

                if hasattr(self, "level_of_detection_points") & (self.level_of_detection_points is not None
                ) and len(self.level_of_detection_points) > 0:
                    statistics_file.write(
                        "Movement rate of LoD points:\n"
                        + f"\tMean: {_fmt(np.nanmean(self.level_of_detection_points[self.displacement_column_name]))}\n"
                        + f"\tMedian: {_fmt(np.nanmedian(self.level_of_detection_points[self.displacement_column_name]))}\n"
                        + f"\tStandard deviation: {_fmt(np.nanstd(self.level_of_detection_points[self.displacement_column_name]))}\n"
                        + f"\tQ90: {_fmt(np.nanquantile(self.level_of_detection_points[self.displacement_column_name], 0.9))}\n"
                        + f"\tQ99: {_fmt(np.nanquantile(self.level_of_detection_points[self.displacement_column_name], 0.99))}\n"
                        + f"\tUsed points: {valid_lod_points} points\n"
                    )

                distance_series = ref_df2.get("movement_distance")
                statistics_file.write(
                    "Total movement between images:\n"
                    + f"\tMean: {_fmt(_nan_stat(distance_series, np.nanmean))}\n"
                    + f"\tMedian: {_fmt(_nan_stat(distance_series, np.nanmedian))}\n"
                    + f"\tStandard deviation: {_fmt(_nan_stat(distance_series, np.nanstd))}\n"
                    + f"\tQ90: {_fmt(_nan_stat(distance_series, lambda s: np.nanquantile(s, 0.9)))}\n"
                    + f"\tQ99: {_fmt(_nan_stat(distance_series, lambda s: np.nanquantile(s, 0.99)))}\n"
                )

        # --- Parameter logs ---
        with open(
                f"{folder_path}/parameters_{self.image1_observation_date.strftime(format='%Y-%m-%d')}_{self.image2_observation_date.strftime(format='%Y-%m-%d')}.txt",
                "w",
        ) as text_file:
            text_file.write(self.alignment_parameters.__str__())

        if self.tracking_parameters is not None:
            with open(
                    f"{folder_path}/parameters_{self.image1_observation_date.strftime(format='%Y-%m-%d')}_{self.image2_observation_date.strftime(format='%Y-%m-%d')}.txt",
                    "a",
            ) as text_file:
                text_file.write(self.tracking_parameters.__str__())

        if self.filter_parameters is not None:
            with (open(folder_path + "/parameters_" + str(self.image1_observation_date.strftime(format='%Y-%m-%d'))
                       + "_" + str(self.image2_observation_date.strftime(format='%Y-%m-%d'))
                       + ".txt", "a") as text_file):
                text_file.write(self.filter_parameters.__str__())

        # --- Plots and LoD annotation  ---
        if self.level_of_detection is not None:
            if "mask_LoD_tif" in save_files and has_lod_col:
                lod_mask = tr_all.loc[tr_all["is_below_LoD"]].copy()
                lod_mask["is_below_LoD_int"] = 1  # write 1 where below LoD
                lod_raster = _make_raster(lod_mask, ["is_below_LoD_int"])
                if lod_raster is not None:
                    _save_raster_as_tif(
                        path=f"{folder_path}/mask_below_LoD_{self.image1_observation_date.strftime(format='%Y-%m-%d')}_{self.image2_observation_date.strftime(format='%Y-%m-%d')}.tif",
                        raster=lod_raster["raster"]["is_below_LoD_int"],
                        transform=lod_raster["transform"],
                        crs=lod_raster["crs"]
                    )

            plot_movement_of_points_with_valid_mask(
                self.image1_matrix,
                self.image1_transform,
                self.tracking_results,
                save_path=f"{folder_path}/tracking_results_{self.image1_observation_date.strftime(format='%Y-%m-%d')}_{self.image2_observation_date.strftime(format='%Y-%m-%d')}.jpg",
                unit_name=self.coordinate_system_unit_name
            )
        else:
            plot_movement_of_points(
                self.image1_matrix,
                self.image1_transform,
                self.tracking_results,
                save_path=f"{folder_path}/tracking_results_{self.image1_observation_date.strftime(format='%Y-%m-%d')}_{self.image2_observation_date.strftime(format='%Y-%m-%d')}.jpg",
                unit_name=self.coordinate_system_unit_name
            )

