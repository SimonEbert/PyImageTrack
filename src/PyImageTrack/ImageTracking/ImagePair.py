import os
import warnings

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
from ..Utils import make_effective_extents_from_deltas

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
from ..DataProcessing.ImagePreprocessing import (
    equalize_adapthist_images,
    undistort_camera_image,
    convert_float_to_uint,
    harmonize_dtypes,
    harmonize_resolution,
    check_channels_compatible
)
from .AlignImages import align_images_lsm_scarce
# Plotting
from ..Plots.MakePlots import (
    plot_movement_of_points,
    plot_movement_of_points_with_valid_mask, plot_raster_and_geometry,
)
# Date Handling
from ..Utils import parse_date
from ..ConsoleOutput import get_console


class ImagePair:
    """
    Main class for processing and tracking movement between two images.

    Workflow:
    - Load and preprocess images (georeferenced or non-georeferenced/fake georef)
    - Align images (cross-correlation / LSM)
    - Track movement between images
    - Filter outliers and compute level of detection (LoD)
    - Save results (vector/raster products, masks, statistics)

    Supported modes include optional camera undistortion, fake georeferencing with
    synthetic pixel size, CLAHE enhancement, downsampling, and 3D displacement from
    depth images. Output units can be yearly or total.

    Parameters
    ----------
    parameter_dict : dict, optional
        Configuration for alignment, tracking, filtering, output units, georeferencing
        mode (real vs fake), pixel size overrides, downsample factor, CLAHE settings,
        camera calibration, and 3D displacement.

    Attributes
    ----------
    images_aligned : bool
        Whether the two images have been aligned.
    valid_alignment_possible : bool or None
        Whether a valid alignment was possible.
    image1_matrix : np.ndarray
        Matrix of the first (reference) image.
    image1_transform : Affine
        Transform of the first image.
    image1_observation_date : datetime
        Observation date of the first image.
    image2_matrix : np.ndarray
        Matrix of the second (aligned) image.
    image2_transform : Affine
        Transform of the second image.
    image2_observation_date : datetime
        Observation date of the second image.
    image_bounds : gpd.GeoDataFrame
        Bounds of the images as a GeoDataFrame.
    tracked_control_points : gpd.GeoDataFrame
        Control points used for alignment.
    tracking_results : gpd.GeoDataFrame
        Results of movement tracking.
    level_of_detection : float
        Calculated level of detection.
    level_of_detection_points : gpd.GeoDataFrame
        Points used for LoD calculation.
    """

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

        # Optional: Store position images corresponding to non-georeferenced images for 3d displacement calculation
        self.depth_image1 = None
        self.depth_image2 = None

        # Parameters
        self.tracking_parameters = TrackingParameters(parameter_dict=parameter_dict)
        self.filter_parameters = None
        self.alignment_parameters = AlignmentParameters(parameter_dict=parameter_dict)
        # Moving-area ID column name (propagated to outputs)
        self.moving_id_column = parameter_dict.get("moving_id_column", "moving_id") if parameter_dict else "moving_id"
        # Fake georef switches (fed via run_pipeline param_dict)
        self.use_no_georeferencing = bool(
            parameter_dict.get("use_no_georeferencing", False)) if parameter_dict else False
        self.fake_pixel_size = float(parameter_dict.get("fake_pixel_size", 1.0)) if parameter_dict else 1.0
        self.downsample_factor = int(parameter_dict.get("downsample_factor", 1)) if parameter_dict else 1
        if self.downsample_factor < 1:
            self.downsample_factor = 1

        # Meta-Data and results
        self.crs = parameter_dict.get("crs", None)
        self.tracked_control_points = None
        self.tracking_results = None
        self.level_of_detection = None
        self.level_of_detection_points = None
        self.convert_to_3d_displacement = parameter_dict.get("convert_to_3d_displacement", False)
        self.undistort_image = parameter_dict.get("undistort_image", False)
        self.camera_intrinsics_matrix = parameter_dict.get("camera_intrinsics_matrix", None)
        self.camera_distortion_coefficients = parameter_dict.get("camera_distortion_coefficients", None)
        self.camera_to_3d_coordinates_transform = parameter_dict.get("camera_to_3d_coordinates_transform", None)
        # Image enhancement parameters
        self.enhancement_type = parameter_dict.get("enhancement_type", "none")
        self.enhancement_kernel_size = parameter_dict.get("enhancement_kernel_size", 50)
        self.enhancement_clip_limit = parameter_dict.get("enhancement_clip_limit", 0.9)
        # Image bands (delegates to tracking_parameters.image_bands)
        self.image_bands = parameter_dict.get("image_bands")
        # Adaptive tracking window
        self.use_adaptive_tracking_window = parameter_dict.get("use_adaptive_tracking_window", False)
        # Output units mode
        self.output_units_mode = parameter_dict.get("output_units_mode", "per_year")
        if self.convert_to_3d_displacement:
            if self.output_units_mode == "total":
                self.displacement_column_name = "3d_displacement_distance_total"
            else:
                self.displacement_column_name = "3d_displacement_distance_per_year"
        else:
            if self.output_units_mode == "total":
                self.displacement_column_name = "movement_distance_total"
            else:
                self.displacement_column_name = "movement_distance_per_year"

    def _effective_pixel_size(self) -> float:
        """
        Calculate the effective pixel size in CRS units.

        Assumes square pixels and returns the maximum of the absolute values
        of the transform's a and e coefficients.

        Returns
        -------
        float
            Pixel size in CRS units, or 1.0 if no transform is available.
        """
        if self.image1_transform is None:
            return 1.0
        a = float(self.image1_transform.a)
        e = float(self.image1_transform.e)
        return max(abs(a), abs(e))

    def _downsample_array(self, arr: np.ndarray, factor: int) -> np.ndarray:
        """
        Downsample an array by a given factor.

        Parameters
        ----------
        arr : np.ndarray
            Array to downsample (2D or 3D).
        factor : int
            Downsampling factor (must be >= 1).

        Returns
        -------
        np.ndarray
            Downsampled array.
        """
        if factor <= 1:
            return arr
        if arr.ndim == 3:
            return arr[:, ::factor, ::factor]
        return arr[::factor, ::factor]

    def _downsample_transform(self, transform, factor: int):
        """
        Downsample an affine transform by a given factor.

        Parameters
        ----------
        transform : Affine
            Affine transform to downsample.
        factor : int
            Downsampling factor (must be >= 1).

        Returns
        -------
        Affine
            Downsampled transform.
        """
        if factor <= 1:
            return transform
        from affine import Affine
        return transform * Affine.scale(factor, factor)

    def select_image_channels(self, selected_channels=None):
        """
        Select specific image channels from multi-band images.

        Parameters
        ----------
        selected_channels : int | list[int] | tuple[int] | None, optional
            Channel index/indices to select. ``None`` defaults to ``[0, 1, 2]``.
            Only applied when images are 3D (bands, height, width).
        """
        if selected_channels is None:
            selected_channels = [0, 1, 2]
        if len(self.image1_matrix.shape) == 3:
            self.image1_matrix = self.image1_matrix[selected_channels, :, :]
            self.image2_matrix = self.image2_matrix[selected_channels, :, :]

    def load_images_from_file(self, filename_1: str, observation_date_1: str, filename_2: str, observation_date_2: str,
                              selected_channels=None, NA_value: float = None):
        """
        Load two image files, crop/align extents, harmonize dtypes/resolution, and store metadata.

        Behavior depends on georeferencing:
        - Georeferenced: validates CRS match; crops to intersection; checks channel compatibility;
          harmonizes dtype/resolution; optional downsample; sets buffered ``image_bounds``.
        - Fake georeferencing (missing CRS or ``use_no_georeferencing``): reads arrays, optional camera
          undistortion, converts to uint16, optional depth image loading, top-left crops to common size,
          creates synthetic transform, checks channels, harmonizes dtype/resolution, optional downsample;
          sets bounds in pixel space.

        Common steps: parse observation dates, replace ``NA_value`` with 0 if provided, store originals
        (for true-color outputs), and select channels.

        Parameters
        ----------
        filename_1 : str
            Path to the first image (assumed earlier observation).
        observation_date_1 : str
            Observation date of the first image (ISO-style, multiple formats accepted by ``parse_date``).
        filename_2 : str
            Path to the second image.
        observation_date_2 : str
            Observation date of the second image.
        selected_channels : int | list[int] | tuple[int] | None, optional
            Channels to select from 3D inputs. Defaults to ``[0, 1, 2]`` when None.
        NA_value : float, optional
            If provided, pixels equal to this value are set to 0 in both images.

        Returns
        -------
        None
        """
        # Validate file existence before opening
        if not os.path.exists(filename_1):
            raise FileNotFoundError(f"Image file does not exist: {filename_1}")
        if not os.path.exists(filename_2):
            raise FileNotFoundError(f"Image file does not exist: {filename_2}")
        
        # Suppress NotGeoreferencedWarning when opening non-georeferenced images
        # This is expected behavior when use_no_georeferencing = true
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)
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

            # Spatial intersection (true georef)
            poly1 = box(*file1.bounds)
            poly2 = box(*file2.bounds)
            intersection = poly1.intersection(poly2)

            ([self.image1_matrix, self.image1_transform],
             [self.image2_matrix, self.image2_transform]) = crop_images_to_intersection(file1, file2)

            # Check channel compatibility
            check_channels_compatible(self.image1_matrix, self.image2_matrix)

            # Harmonize datatypes
            self.image1_matrix, self.image2_matrix = harmonize_dtypes(self.image1_matrix, self.image2_matrix)

            # Harmonize resolution if necessary
            self.image1_matrix, self.image2_matrix, self.image1_transform, self.image2_transform = harmonize_resolution(
                self.image1_matrix, self.image2_matrix, self.image1_transform, self.image2_transform)

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
            # Store bounds polygon for safe bounds calculation (will be computed after else block)
            bounds_poly = intersection

        else:
            # FAKE georeferencing path (e.g., JPGs)
            arr1 = file1.read()  # (bands, rows, cols)
            arr2 = file2.read()

            def squeeze(arr):
                return arr[0] if arr.shape[0] == 1 else arr

            arr1 = squeeze(arr1)
            arr2 = squeeze(arr2)

            if self.undistort_image:
                arr1 = undistort_camera_image(arr1, self.camera_intrinsics_matrix, self.camera_distortion_coefficients)
                arr2 = undistort_camera_image(arr2, self.camera_intrinsics_matrix, self.camera_distortion_coefficients)

            # Automatically convert float images to uint16 for better alignment
            arr1 = convert_float_to_uint(arr1)
            arr2 = convert_float_to_uint(arr2)

            self.image1_matrix = arr1
            self.image2_matrix = arr2

            if self.convert_to_3d_displacement:
                basename1 = os.path.splitext(os.path.basename(filename_1))[0]
                basename2 = os.path.splitext(os.path.basename(filename_2))[0]
                depth_image1_path = os.path.join(os.path.dirname(filename_1), "Depth_images", basename1
                                                 + "_depth.tiff")
                depth_image2_path = os.path.join(os.path.dirname(filename_2), "Depth_images", basename2
                                                 + "_depth.tiff")
                
                # Validate depth image files exist
                if not os.path.exists(depth_image1_path):
                    raise FileNotFoundError(f"Depth image file does not exist: {depth_image1_path}")
                if not os.path.exists(depth_image2_path):
                    raise FileNotFoundError(f"Depth image file does not exist: {depth_image2_path}")
                
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
            self.image1_matrix = self.image1_matrix[..., :h, :w] if self.image1_matrix.ndim == 3 else \
                self.image1_matrix[:h, :w]
            self.image2_matrix = self.image2_matrix[..., :h, :w] if self.image2_matrix.ndim == 3 else \
                self.image2_matrix[:h, :w]

            # Create synthetic transform first (needed for harmonize_resolution)
            from affine import Affine
            px = float(self.fake_pixel_size)
            tform = Affine(px, 0, 0, 0, -px, 0)  # x = px*col ; y = -px*row
            self.image1_transform = tform
            self.image2_transform = tform

            # Check channel compatibility for non-georeferenced images
            check_channels_compatible(self.image1_matrix, self.image2_matrix)
            
            # Harmonize dtypes (may already be uint16, but verifies)
            self.image1_matrix, self.image2_matrix = harmonize_dtypes(self.image1_matrix, self.image2_matrix)
            
            # Harmonize resolution if necessary (before applying downsample_factor)
            self.image1_matrix, self.image2_matrix, self.image1_transform, self.image2_transform = harmonize_resolution(
                self.image1_matrix, self.image2_matrix, self.image1_transform, self.image2_transform)

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

            # Bounds in that CRS (pixel grid space)
            from rasterio.transform import array_bounds
            bounds_poly = box(*array_bounds(h, w, tform))

        # Common safe bounds calculation for both true and fake georef paths
        def make_safe_bounds_from_buffer(buffer, base_polygon):
            if not buffer:
                raise ValueError("Search_extent_px must be set (tuple posx,negx,posy,negy).")
            # Use appropriate pixel size variable based on which path was taken
            if self.use_no_georeferencing:
                # FAKE georef path - px is defined locally
                buffer_len = px * buffer
            else:
                # TRUE georef path - px_size is defined locally
                buffer_len = px_size * buffer

            safe_bounds = gpd.GeoDataFrame({'geometry': [base_polygon]}, crs=self.crs).buffer(-buffer_len)
            safe_bounds = gpd.GeoDataFrame(geometry=safe_bounds, crs=self.crs)
            safe_bounds = safe_bounds.rename(columns={0: "geometry"})
            safe_bounds.set_geometry("geometry", inplace=True)
            return safe_bounds

        self.safe_image_bounds_tracking = make_safe_bounds_from_buffer(
            max(getattr(self.tracking_parameters, "search_extent_px", None))
            + getattr(self.tracking_parameters, "movement_cell_size", None)/2,
            bounds_poly)

        self.safe_image_bounds_alignment = make_safe_bounds_from_buffer(
            max(getattr(self.alignment_parameters, "control_search_extent_px", None))
            + getattr(self.alignment_parameters, "control_cell_size", None)/2,
            bounds_poly)

        self.image1_observation_date = parse_date(observation_date_1)
        self.image2_observation_date = parse_date(observation_date_2)

        # Calculate years between observations
        delta_hours = (self.image2_observation_date - self.image1_observation_date).total_seconds() / 3600.0
        self.years_between_observations = delta_hours / (24.0 * 365.25)

        if NA_value is not None:
            self.image1_matrix[self.image1_matrix == NA_value] = 0
            self.image2_matrix[self.image2_matrix == NA_value] = 0


        # Compute effective buffer if needed
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

    def load_images_from_matrix_and_transform(self, image1_matrix: np.ndarray, observation_date_1: str,
                                              image2_matrix: np.ndarray, observation_date_2: str, image_transform, crs,
                                              selected_channels=None):
        """
        Load two images from matrices with a common transform.

        Stores the matrices, dates, CRS/transform, copies originals, builds bounds,
        and optionally selects channels. No file I/O is performed.

        Parameters
        ----------
        image1_matrix : np.ndarray
            Matrix of the first image.
        observation_date_1 : str
            Observation date of the first image. Multiple ISO-like formats accepted by ``parse_date``.
        image2_matrix : np.ndarray
            Matrix of the second image.
        observation_date_2 : str
            Observation date of the second image.
        image_transform : Affine
            Common transform for both images.
        crs : any
            Coordinate reference system.
        selected_channels : int | list[int] | tuple[int] | None, optional
            Channels to select from 3D inputs. Defaults to no selection when ``None``.

        Returns
        -------
        None
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
        
        # Also compute safe_image_bounds_alignment for align_images() method
        alignment_ext = getattr(self.alignment_parameters, "control_search_extent_px", None)
        if not alignment_ext:
            raise ValueError("AlignmentParameters.control_search_extent_px must be set (tuple posx,negx,posy,negy).")
        alignment_cell_size = getattr(self.alignment_parameters, "control_cell_size", None)
        if alignment_cell_size is None:
            raise ValueError("AlignmentParameters.control_cell_size must be set.")
        
        alignment_buffer = px_size * max(alignment_ext) + px_size * alignment_cell_size / 2
        safe_image_bounds_alignment = gpd.GeoDataFrame(
            gpd.GeoDataFrame({'geometry': [poly1]}, crs=self.crs).buffer(-alignment_buffer)
        )
        safe_image_bounds_alignment = safe_image_bounds_alignment.rename(columns={0: "geometry"})
        safe_image_bounds_alignment.set_geometry("geometry", inplace=True)
        self.safe_image_bounds_alignment = safe_image_bounds_alignment
    def align_images(self, reference_area: gpd.GeoDataFrame, polygon_inside: gpd.GeoDataFrame = None) -> None:
        """
        Align images by tracking control points in a stable reference area.

        Re-checks channel compatibility, harmonizes dtype/resolution, validates CRSs,
        and intersects the reference area with ``image_bounds``. If ``reference_area`` is
        ``None``, uses ``image_bounds`` minus ``polygon_inside`` (must be provided).
        Updates ``image2_matrix`` and ``image2_transform`` and stores georeferenced
        control points.

        Parameters
        ----------
        reference_area : gpd.GeoDataFrame or None
            Stable area used for alignment. If None, computed from ``image_bounds`` and
            ``polygon_inside``.
        polygon_inside : gpd.GeoDataFrame, optional
            Moving area polygon, required when ``reference_area`` is None.

        Returns
        -------
        None
        """
        console = get_console()

        # Harmonization checks - performed during alignment, not during load, to avoid
        # unnecessary processing when alignment is loaded from cache
        
        # Check channel compatibility
        check_channels_compatible(self.image1_matrix, self.image2_matrix)
        
        # Harmonize datatypes
        self.image1_matrix, self.image2_matrix = harmonize_dtypes(self.image1_matrix, self.image2_matrix)
        
        # Harmonize resolution if necessary
        self.image1_matrix, self.image2_matrix, self.image1_transform, self.image2_transform = harmonize_resolution(
            self.image1_matrix, self.image2_matrix, self.image1_transform, self.image2_transform)
        
        # Validate reference_area if provided
        if reference_area is not None:
            if len(reference_area) == 0:
                raise ValueError("Reference area GeoDataFrame is empty.")
            if 'geometry' not in reference_area.columns:
                raise ValueError("Reference area GeoDataFrame must contain a 'geometry' column.")
        
        # Handle fallback mode when reference_area is None
        if reference_area is None:
            if polygon_inside is None:
                raise ValueError(
                    "polygon_inside must be provided when reference_area is None."
                )
            reference_area = gpd.GeoDataFrame(
                geometry=self.image_bounds.difference(polygon_inside),
                crs=self.crs
            )
            reference_area = reference_area.rename(columns={0: 'geometry'})
            reference_area.set_geometry('geometry', inplace=True)
            console.warning(
                "Using image_bounds minus moving_area as stable area. "
                "This may result in slightly lower alignment quality. "
                "Consider increasing number_of_control_points to compensate."
            )
        
        if reference_area.crs != self.crs:
            raise ValueError("Got reference area with crs " + str(reference_area.crs) + " and images with crs "
                             + str(self.crs) + ". Reference area and images are supposed to have the same crs.")

        if self.undistort_image:
            reference_area = undistort_polygon(reference_area, self.image1_matrix_original.shape[-2:],
                                               self.camera_intrinsics_matrix,
                                               self.camera_distortion_coefficients)


        reference_area_safe_bounds = gpd.GeoDataFrame(reference_area.intersection(self.safe_image_bounds_alignment))
        reference_area_safe_bounds.rename(columns={0: 'geometry'}, inplace=True)
        reference_area_safe_bounds.set_geometry('geometry', inplace=True)
        
        # Check if reference_area is empty after intersection
        if len(reference_area) == 0 or reference_area.geometry.iloc[0].is_empty:
            raise ValueError(
                "Reference area is empty after intersection with image bounds. "
                "This may happen if the moving area covers the entire image."
            )

        if self.depth_image1 is not None:
            if self.depth_image2 is None:
                raise ValueError("Got depth image for time point 1, but not for time point 2.")

            [_, new_image2_matrix, tracked_control_points, alignment_transformation_matrix] = (
                align_images_lsm_scarce(image1_matrix=self.image1_matrix,
                                        image2_matrix=self.image2_matrix,
                                        image_transform=self.image1_transform,
                                        reference_area=reference_area_safe_bounds,
                                        alignment_parameters=self.alignment_parameters,
                                        return_alignment_transformation_matrix=True))
            self.depth_image2 = move_image_matrix_from_transformation(self.depth_image2, alignment_transformation_matrix,
                                                  target_shape=self.depth_image1.shape[-2:])
        else:
            [_, new_image2_matrix, tracked_control_points] = (
                align_images_lsm_scarce(image1_matrix=self.image1_matrix,
                                        image2_matrix=self.image2_matrix,
                                        image_transform=self.image1_transform,
                                        reference_area=reference_area_safe_bounds,
                                        alignment_parameters=self.alignment_parameters))

        self.valid_alignment_possible = True

        delta_hours = (self.image2_observation_date - self.image1_observation_date).total_seconds() / 3600.0
        years_between_observations = delta_hours / (24.0 * 365.25)

        self.tracked_control_points = georeference_tracked_points(tracked_control_points,
                                                                  self.image1_transform,
                                                                  self.crs,
                                                                  years_between_observations,
                                                                  self.output_units_mode)

        self.image2_matrix = new_image2_matrix
        self.image2_transform = self.image1_transform

        self.images_aligned = True

        # Optionally derive a true-color aligned image (if original data are available)
        self.compute_truecolor_aligned_from_control_points()

    def compute_truecolor_aligned_from_control_points(self):
        """
        Rebuild a true-color aligned version of image2 using alignment control points.

        This method uses the same least-squares affine model and spline resampling
        approach as in align_images_lsm_scarce, but applies it to the original
        (potentially multi-band) second image.

        Raises
        ------
        ValueError
            If no tracked control points are available or no original second image is stored.
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
        # Validate tracking_area
        if len(tracking_area) == 0:
            raise ValueError("Tracking area GeoDataFrame is empty.")
        if 'geometry' not in tracking_area.columns:
            raise ValueError("Tracking area GeoDataFrame must contain a 'geometry' column.")
        
        console = get_console()
        if tracking_area.crs != self.crs:
            raise ValueError("Got tracking area with crs " + str(tracking_area.crs) + " and images with crs "
                             + str(self.crs) + ". Tracking area and images are supposed to have the same crs.")

        if not self.images_aligned:
            console = get_console()
            console.warning("Images have not been aligned. Any resulting velocities are likely invalid.")

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
        image_bounds_values = self.image_bounds.bounds.iloc[0]
        within_image_mask = ((image_bounds_values.minx <= points_to_be_tracked.geometry.x) &
                            (image_bounds_values.maxx >= points_to_be_tracked.geometry.x) &
                            (image_bounds_values.miny <= points_to_be_tracked.geometry.y) &
                            (image_bounds_values.maxy >= points_to_be_tracked.geometry.y))

        points_to_be_tracked = points_to_be_tracked[within_image_mask]


        tracked_points = track_movement_lsm(self.image1_matrix, self.image2_matrix, self.image1_transform,
                                            points_to_be_tracked=points_to_be_tracked,
                                            tracking_parameters=self.tracking_parameters,
                                            alignment_tracking=False,
                                            task_label="[~] Tracking points for movement tracking")
        # calculate the years between observations from the two given observation dates
        delta_hours = (self.image2_observation_date - self.image1_observation_date).total_seconds() / 3600.0
        years_between_observations = delta_hours / (24.0 * 365.25)

        if self.convert_to_3d_displacement:
            georeferenced_tracked_points = calculate_displacement_from_depth_images(
                tracked_points, depth_image_time1=self.depth_image1,
                depth_image_time2=self.depth_image2, camera_intrinsics_matrix=self.camera_intrinsics_matrix,
                camera_to_3d_coordinates_transform=self.camera_to_3d_coordinates_transform,
                years_between_observations=years_between_observations,
                output_unit_mode=self.output_units_mode)
        else:
            georeferenced_tracked_points = georeference_tracked_points(
                tracked_pixels=tracked_points, raster_transform=self.image1_transform, crs=tracking_area.crs,
                years_between_observations=years_between_observations,
                output_unit_mode=self.output_units_mode)

        return georeferenced_tracked_points

    def perform_point_tracking(self, reference_area: gpd.GeoDataFrame, tracking_area: gpd.GeoDataFrame) -> None:
        """
        Perform alignment (if needed) and movement tracking in one call.

        Runs ``align_images`` if not yet aligned, then ``track_points`` on the
        supplied tracking area and stores results in ``tracking_results``.

        Parameters
        ----------
        reference_area : gpd.GeoDataFrame
            Area used for alignment (must share CRS with the images).
        tracking_area : gpd.GeoDataFrame
            Area used to generate the tracking point grid (must share CRS).

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
        Plot the two raster images separately.

        Displays both images on the current matplotlib canvas with their
        observation dates as titles.

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
        Plot the first raster image with movement vectors.

        Displays the first image with tracked points and their movement
        vectors overlaid.

        Returns
        -------
        None
        """
        plot_movement_of_points(self.image1_matrix, self.image1_transform, self.tracking_results)

    def plot_tracking_results_with_valid_mask(self) -> None:
        """
        Plot the first raster image with movement vectors and validity mask.

        Displays the first image with tracked points and their movement vectors.
        Points with zero movement rate (below LoD) are shown in gray.

        Returns
        -------
        None
        """
        plot_movement_of_points_with_valid_mask(self.image1_matrix, self.image1_transform, self.tracking_results)

    def filter_outliers(self, filter_parameters: FilterParameters):
        """
        Filter outliers from tracking results.

        Runs all outlier filters independently on the original tracking_results snapshot
        and combines masks at the end (no sequential dependency between filters).

        Parameters
        ----------
        filter_parameters : FilterParameters
            Parameters for filtering. Includes settings for:
            - Level of detection calculation
            - Movement bearing outlier filtering
            - Movement rate outlier filtering

        Returns
        -------
        None
        """
        if not self.valid_alignment_possible:
            return
        console = get_console()
        console.processing("Filtering outliers. This may take a moment.")
        self.filter_parameters = filter_parameters
        self.tracking_results = filter_outliers_full(self.tracking_results, filter_parameters,
                                                     self.displacement_column_name)


    def track_lod_points(self, points_for_lod_calculation: gpd.GeoDataFrame,
                         years_between_observations) -> gpd.GeoDataFrame:
        """
        Track movement on stable points for level of detection calculation.

        Performs tracking on points in an area assumed to be stable (no real
        movement) to calculate the level of detection.

        Parameters
        ----------
        points_for_lod_calculation : gpd.GeoDataFrame
            Points in the stable area for calculating the level of detection.
        years_between_observations : float
            Time span in years between the two observations.

        Returns
        -------
        tracked_points : gpd.GeoDataFrame
            The tracked points which can be used for calculating the LoD.
        """
        points = points_for_lod_calculation
        tracked_points = track_movement_lsm(
            image1_matrix=self.image1_matrix, image2_matrix=self.image2_matrix, image_transform=self.image1_transform,
            points_to_be_tracked=points, tracking_parameters=self.tracking_parameters, alignment_tracking=False,
            save_columns=["movement_row_direction",
                          "movement_column_direction",
                          "movement_distance_pixels",
                          "movement_bearing_pixels",
                          "correlation_coefficient",
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

        console = get_console()
        console.info(f"Used {len(tracked_control_pixels_valid)} pixels for LoD calculation.")

        if self.convert_to_3d_displacement:
            tracked_points = calculate_displacement_from_depth_images(
                tracked_control_pixels_valid,self.depth_image1,self.depth_image2,
                camera_intrinsics_matrix=self.camera_intrinsics_matrix,
                camera_to_3d_coordinates_transform=self.camera_to_3d_coordinates_transform,
                years_between_observations=years_between_observations,
                output_unit_mode=self.output_units_mode)
        else:
            tracked_points = georeference_tracked_points(tracked_control_pixels_valid,
                                                         self.image1_transform,
                                                         crs=self.crs,
                                                         years_between_observations=years_between_observations,
                                                         output_unit_mode=self.output_units_mode)

        return tracked_points

    def calculate_lod(self, points_for_lod_calculation: gpd.GeoDataFrame,
                      filter_parameters: FilterParameters = None) -> None:
        """
        Calculate the Level of Detection (LoD) for the image pair.

        The LoD is calculated by tracking movement in a stable area (assumed
        to have no real movement) and computing a quantile of the observed
        movement rates. This represents the minimum detectable movement
        given the image quality and processing parameters.

        Parameters
        ----------
        points_for_lod_calculation : gpd.GeoDataFrame
            Points in the stable area for calculating the level of detection.
            A random distribution is recommended to avoid bias from the
            evenly-spaced grid used for alignment.
        filter_parameters : FilterParameters, optional
            Filter parameters containing the quantile for LoD calculation.
            If None, uses the filter_parameters already set on the object.

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
        console = get_console()
        console.success(f"Found level of detection with quantile {level_of_detection_quantile} as {np.round(self.level_of_detection, decimals=5)} {unit_name}/year")

    def filter_lod_points(self) -> None:
        """
        Filter points below the level of detection.

        Sets the movement distance of all points that fall below the calculated
        level of detection to 0 and their movement bearing to NaN. This
        directly modifies the tracking_results dataframe.

        Returns
        -------
        None
        """

        if not self.valid_alignment_possible:
            return
        self.tracking_results = filter_lod_points(self.tracking_results, self.level_of_detection,
                                                  self.displacement_column_name)

    def full_filter(self, reference_area, filter_parameters: FilterParameters):
        """
        Perform complete filtering workflow.

        This method:
        1. Filters outliers from tracking results
        2. Calculates the level of detection
        3. Filters points below the level of detection

        Parameters
        ----------
        reference_area : gpd.GeoDataFrame
            Reference area for generating LoD calculation points.
        filter_parameters : FilterParameters
            Parameters for filtering and LoD calculation.

        Returns
        -------
        None
        """
        points_for_lod_calculation = random_points_on_polygon_by_number(reference_area,
                                                                        filter_parameters.number_of_points_for_level_of_detection)
        self.filter_outliers(filter_parameters)
        self.calculate_lod(points_for_lod_calculation, filter_parameters)
        self.filter_lod_points()

    def equalize_adapthist_images(self):
        """
        Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) in-place.

        Uses ``enhancement_kernel_size`` and ``enhancement_clip_limit`` on
        ``image1_matrix`` and ``image2_matrix``. Mutates the stored matrices.

        Returns
        -------
        None
        """
        self.image1_matrix = equalize_adapthist_images(self.image1_matrix,
                                                       kernel_size=self.enhancement_kernel_size,
                                                       clip_limit=self.enhancement_clip_limit)
        self.image2_matrix = equalize_adapthist_images(self.image2_matrix,
                                                       kernel_size=self.enhancement_kernel_size,
                                                       clip_limit=self.enhancement_clip_limit)

    def save_full_results(self, folder_path: str, save_files: list) -> None:
        """
        Save tracking outputs (vectors, rasters, masks, statistics) into ``folder_path``.

        Notes / conventions:
        - "valid" points are those marked True in ``tracking_results['valid']``.
        - "LoD-filtered" rasters keep all points above LoD (``is_below_LoD == False``), regardless of outlier flags.
        - "all" rasters include every tracked point without filtering.
        - "outlier-filtered" rasters exclude only outliers; LoD-below points remain unless excluded by ``valid``.
        - Outlier masks write value 1 where the specific outlier reason is true (0/NaN elsewhere).
        - Vector outputs use FlatGeobuf; rasters use GTiff (masks/metrics) or JPEG (raw images).
        - Statistics text is written when ``"statistical_parameters_txt"`` is requested.

        Parameters
        ----------
        folder_path : str
            Target folder for all outputs.
        save_files : list
            Tokens controlling what to save. Supported tokens include:
            - "first_image_matrix", "second_image_matrix"
            - "movement_bearing_valid_tif", "movement_rate_valid_tif"
            - "movement_bearing_outlier_filtered_tif", "movement_rate_outlier_filtered_tif"
            - "movement_bearing_LoD_filtered_tif", "movement_rate_LoD_filtered_tif"
            - "movement_bearing_all_tif", "movement_rate_all_tif"
            - "mask_invalid_tif", "mask_LoD_tif"
            - "mask_outlier_md_tif", "mask_outlier_msd_tif", "mask_outlier_bd_tif", "mask_outlier_bsd_tif"
            - "LoD_points_geojson", "control_points_geojson"
            - "statistical_parameters_txt"

        Returns
        -------
        None
        """
        os.makedirs(folder_path, exist_ok=True)

        # --- Always save the full tracking results GeoJSON ---

        self.tracking_results.to_file(
            f"{folder_path}/tracking_results_{self.image1_observation_date.strftime(format='%Y-%m-%d')}_{self.image2_observation_date.strftime(format='%Y-%m-%d')}.fgb",
            driver="FlatGeobuf"
        )

        # --- Prepare common subsets and guards ---
        tr_all = self.tracking_results
        moving_id_col = self.moving_id_column if self.moving_id_column in tr_all.columns else None
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

            with rasterio.open(
                    path,
                    "w",
                    driver=driver,
                    height=raster.shape[0],
                    width=raster.shape[1],
                    transform=transform,
                    crs=crs,
                    count=1,
                    nodata=np.nan,
                    dtype=dtype
            ) as dst:
                dst.write(raster, 1)

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

            # Determine unit label for statistics
            unit_label = "per year" if "per_year" in self.displacement_column_name else "total"
            
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
                if getattr(self.filter_parameters,"maximal_fraction_depth_change_of_3d_displacement"):
                    statistics_file.write(
                        "\t\t" + str(int(tr_all["is_depth_fraction_outlier"].sum())) + " depth fraction outliers\n"
                    )

                statistics_file.write(
                    "Valid points: " + str(len(tr_valid)) + " ("
                    + str(np.round((len(tr_valid) / total_number_of_points * 100) if total_number_of_points else 0,
                                   2)) + "%)\n"
                )

                # Helper to format stats
                def _fmt(x):
                    return "NA" if x is None or np.isnan(x) else f"{x:.2f}"

                # Per-polygon breakdown if available
                if moving_id_col is not None:
                    statistics_file.write(f"\nPer polygon statistics (by '{moving_id_col}'):\n")

                    def _pct(num, denom):
                        return f"{(num / denom * 100):.2f}%" if denom else "0.00%"

                    for id_val, df_id_all in tr_all.groupby(moving_id_col):
                        df_id_no_out = df_id_all
                        if moving_id_col in tr_without_outliers.columns:
                            df_id_no_out = tr_without_outliers[tr_without_outliers[moving_id_col] == id_val]
                        df_id_valid = df_id_all
                        if moving_id_col in tr_valid.columns:
                            df_id_valid = tr_valid[tr_valid[moving_id_col] == id_val]
                        df_id_above_lod = df_id_all
                        if moving_id_col in tr_above_lod.columns:
                            df_id_above_lod = tr_above_lod[tr_above_lod[moving_id_col] == id_val]

                        n_total = len(df_id_all)
                        n_valid = len(df_id_valid)
                        n_below_lod = int(df_id_all["is_below_LoD"].sum()) if has_lod_col and "is_below_LoD" in df_id_all.columns else 0
                        n_outliers = int(is_outlier.loc[df_id_all.index].sum()) if not is_outlier.empty else 0

                        statistics_file.write(
                            f"  polygon={id_val} | total: {n_total} | valid: {n_valid} ({_pct(n_valid, n_total)})"
                            + (f" | below LoD: {n_below_lod} ({_pct(n_below_lod, n_total)})" if has_lod_col else "")
                            + f" | outliers: {n_outliers} ({_pct(n_outliers, n_total)})\n"
                        )

                        def _stat_block(df_ref, title):
                            statistics_file.write(
                                f"    {title}:\n"
                                + f"      Mean: {_fmt(np.nanmean(df_ref[self.displacement_column_name]))}\n"
                                + f"      Median: {_fmt(np.nanmedian(df_ref[self.displacement_column_name]))}\n"
                                + f"      Standard deviation: {_fmt(np.nanstd(df_ref[self.displacement_column_name]))}\n"
                                + f"      Q90: {_fmt(np.nanquantile(df_ref[self.displacement_column_name], 0.9))}\n"
                                + f"      Q99: {_fmt(np.nanquantile(df_ref[self.displacement_column_name], 0.99))}\n"
                            )

                        # Movement including points below LoD (outliers removed when available)
                        df_poly_including = df_id_no_out if not df_id_no_out.empty else df_id_all
                        _stat_block(df_poly_including, f"Movement ({unit_label}) including points below LoD")

                        # Movement excluding points below LoD
                        df_poly_excluding = df_id_valid if not df_id_valid.empty else df_id_above_lod
                        _stat_block(df_poly_excluding, f"Movement ({unit_label}) excluding points below LoD")

                        # Movement of LoD points for this polygon (if available)
                        df_poly_lod = None
                        if hasattr(self, "level_of_detection_points") and self.level_of_detection_points is not None:
                            if moving_id_col in self.level_of_detection_points.columns:
                                df_poly_lod = self.level_of_detection_points[
                                    self.level_of_detection_points[moving_id_col] == id_val
                                ]
                        if df_poly_lod is not None and len(df_poly_lod) > 0:
                            used_lod_points = int(df_poly_lod["valid"].sum()) if "valid" in df_poly_lod.columns else len(df_poly_lod)
                            statistics_file.write(
                                f"    Movement ({unit_label}) of LoD points:\n"
                                + f"      Mean: {_fmt(np.nanmean(df_poly_lod[self.displacement_column_name]))}\n"
                                + f"      Median: {_fmt(np.nanmedian(df_poly_lod[self.displacement_column_name]))}\n"
                                + f"      Standard deviation: {_fmt(np.nanstd(df_poly_lod[self.displacement_column_name]))}\n"
                                + f"      Q90: {_fmt(np.nanquantile(df_poly_lod[self.displacement_column_name], 0.9))}\n"
                                + f"      Q99: {_fmt(np.nanquantile(df_poly_lod[self.displacement_column_name], 0.99))}\n"
                                + f"      Used points: {used_lod_points} points\n"
                            )

                        # Total movement between images
                        distance_series_id = None
                        if not df_id_valid.empty and "movement_distance" in df_id_valid.columns:
                            distance_series_id = df_id_valid["movement_distance"]
                        elif "movement_distance" in df_id_all.columns:
                            distance_series_id = df_id_all["movement_distance"]

                        statistics_file.write(
                            "    Total movement between images:\n"
                            + f"      Mean: {_fmt(_nan_stat(distance_series_id, np.nanmean))}\n"
                            + f"      Median: {_fmt(_nan_stat(distance_series_id, np.nanmedian))}\n"
                            + f"      Standard deviation: {_fmt(_nan_stat(distance_series_id, np.nanstd))}\n"
                            + f"      Q90: {_fmt(_nan_stat(distance_series_id, lambda s: np.nanquantile(s, 0.9)))}\n"
                            + f"      Q99: {_fmt(_nan_stat(distance_series_id, lambda s: np.nanquantile(s, 0.99)))}\n"
                        )

                        statistics_file.write("\n")

                statistics_file.write("Overall statistics (all polygons combined):\n")

                ref_df1 = tr_without_outliers if not tr_without_outliers.empty else tr_all
                statistics_file.write(
                    f"  Movement ({unit_label}) including points below LoD:\n"
                    + f"    Mean: {_fmt(np.nanmean(ref_df1[self.displacement_column_name]))}\n"
                    + f"    Median: {_fmt(np.nanmedian(ref_df1[self.displacement_column_name]))}\n"
                    + f"    Standard deviation: {_fmt(np.nanstd(ref_df1[self.displacement_column_name]))}\n"
                    + f"    Q90: {_fmt(np.nanquantile(ref_df1[self.displacement_column_name], 0.9))}\n"
                    + f"    Q99: {_fmt(np.nanquantile(ref_df1[self.displacement_column_name], 0.99))}\n"
                )

                ref_df2 = tr_valid if not tr_valid.empty else tr_above_lod
                statistics_file.write(
                    f"  Movement ({unit_label}) excluding points below LoD:\n"
                    + f"    Mean: {_fmt(np.nanmean(ref_df2[self.displacement_column_name]))}\n"
                    + f"    Median: {_fmt(np.nanmedian(ref_df2[self.displacement_column_name]))}\n"
                    + f"    Standard deviation: {_fmt(np.nanstd(ref_df2[self.displacement_column_name]))}\n"
                    + f"    Q90: {_fmt(np.nanquantile(ref_df2[self.displacement_column_name], 0.9))}\n"
                    + f"    Q99: {_fmt(np.nanquantile(ref_df2[self.displacement_column_name], 0.99))}\n"
                )

                if hasattr(self, "level_of_detection_points") & (self.level_of_detection_points is not None
                ) and len(self.level_of_detection_points) > 0:
                    statistics_file.write(
                        f"  Movement ({unit_label}) of LoD points:\n"
                        + f"    Mean: {_fmt(np.nanmean(self.level_of_detection_points[self.displacement_column_name]))}\n"
                        + f"    Median: {_fmt(np.nanmedian(self.level_of_detection_points[self.displacement_column_name]))}\n"
                        + f"    Standard deviation: {_fmt(np.nanstd(self.level_of_detection_points[self.displacement_column_name]))}\n"
                        + f"    Q90: {_fmt(np.nanquantile(self.level_of_detection_points[self.displacement_column_name], 0.9))}\n"
                        + f"    Q99: {_fmt(np.nanquantile(self.level_of_detection_points[self.displacement_column_name], 0.99))}\n"
                        + f"    Used points: {valid_lod_points} points\n"
                    )

                distance_series = ref_df2.get("movement_distance")
                statistics_file.write(
                    "  Total movement between images:\n"
                    + f"    Mean: {_fmt(_nan_stat(distance_series, np.nanmean))}\n"
                    + f"    Median: {_fmt(_nan_stat(distance_series, np.nanmedian))}\n"
                    + f"    Standard deviation: {_fmt(_nan_stat(distance_series, np.nanstd))}\n"
                    + f"    Q90: {_fmt(_nan_stat(distance_series, lambda s: np.nanquantile(s, 0.9)))}\n"
                    + f"    Q99: {_fmt(_nan_stat(distance_series, lambda s: np.nanquantile(s, 0.99)))}\n"
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
            )
        else:
            plot_movement_of_points(
                self.image1_matrix,
                self.image1_transform,
                self.tracking_results,
                save_path=f"{folder_path}/tracking_results_{self.image1_observation_date.strftime(format='%Y-%m-%d')}_{self.image2_observation_date.strftime(format='%Y-%m-%d')}.jpg",
            )

    def load_results(self, file_path, reference_area):
        """
        Load previously saved tracking results from a file.

        Parameters
        ----------
        file_path : str
            Path to the file containing saved tracking results.
        reference_area : gpd.GeoDataFrame
            Reference area for alignment (required for proper coordinate handling).

        Returns
        -------
        None
        """
        saved_tracking_results = gpd.read_file(file_path)
        saved_tracking_results = saved_tracking_results.loc[
            :, ["row", "column", "movement_row_direction", "movement_column_direction",
                "movement_distance_pixels", "movement_bearing_pixels", "movement_distance",
                self.displacement_column_name, "geometry"]]
        saved_tracking_results["valid"] = True
        self.align_images(reference_area)
        self.tracking_results = saved_tracking_results
