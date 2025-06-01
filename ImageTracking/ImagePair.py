import geopandas as gpd
import rasterio
import rasterio.plot
from datetime import datetime
import logging
import numpy as np

from ImageTracking.TrackMovement import track_movement_lsm
from dataloader import TrackingParameters
from CreateGeometries.HandleGeometries import crop_images_to_intersection
from ImageTracking.AlignImages import align_images_lsm_scarce
from CreateGeometries.HandleGeometries import grid_points_on_polygon_by_distance
from CreateGeometries.HandleGeometries import georeference_tracked_points
from dataloader.TrackingParameters import TrackingParameters
from Plots.MakePlots import plot_movement_of_points
from Plots.MakePlots import plot_movement_of_points_with_lod_mask
from DataProcessing.DataPostprocessing import calculate_lod
from DataProcessing.DataPostprocessing import filter_lod_points



class ImagePair:
    def __init__(self, parameter_dict: dict = None):
        self.images_aligned = False
        # self.mask_array = None

        self.image1_matrix = None
        self.image1_transform = None
        self.image1_observation_date = None
        self.image2_matrix = None
        self.image2_transform = None
        self.image2_observation_date = None

        self.tracking_parameters = TrackingParameters(parameter_dict=parameter_dict)

        self.crs = None
        self.tracking_results = None
        self.level_of_detection = None

    def select_image_channels(self, selected_channels: int = None):
        if selected_channels is None:
            selected_channels = [0, 1, 2]
        if len(self.image1_matrix.shape) == 3:
            self.image1_matrix = self.image1_matrix[selected_channels, :, :]
            self.image2_matrix = self.image2_matrix[selected_channels, :, :]

    def load_images_from_file(self, filename_1: str, observation_date_1: str, filename_2: str, observation_date_2: str,
                              selected_channels: int = None):
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
        file1 = rasterio.open(filename_1)
        file2 = rasterio.open(filename_2)
        if file1.crs != file2.crs:
            raise ValueError("Got images with crs " + str(file1.crs) + " and " + str(file2.crs) +
                             "but the two images must  have the same crs.")
        self.crs = file1.crs
        ([self.image1_matrix, self.image1_transform],
         [self.image2_matrix, self.image2_transform]) = crop_images_to_intersection(file1, file2)
        self.image1_observation_date = datetime.strptime(observation_date_1, "%d-%m-%Y").date()
        self.image2_observation_date = datetime.strptime(observation_date_2, "%d-%m-%Y").date()

        # self.mask_array = (self.image1_matrix[3, :, :] != 255) & (self.image2_matrix[3, :, :] != 255)
        self.select_image_channels(selected_channels=selected_channels)

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

        if self.tracking_parameters.image_alignment_via_lsm:
            [_, new_image2_matrix] = (
                align_images_lsm_scarce(image1_matrix=self.image1_matrix, image2_matrix=self.image2_matrix,
                                        image_transform=self.image1_transform, reference_area=reference_area,
                                        number_of_control_points=
                                        self.tracking_parameters.image_alignment_number_of_control_points))

            self.image2_matrix = new_image2_matrix
            self.image2_transform = self.image1_transform

            # if self.tracking_parameters.use_4th_channel_as_data_mask:
            #     self.image1_matrix[self.mask_array] = 0
            #     self.image2_matrix[self.mask_array] = 0

            self.images_aligned = True

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
        points_to_be_tracked = grid_points_on_polygon_by_distance(
            polygon=tracking_area,
            distance_of_points=self.tracking_parameters.distance_of_tracked_points)
        tracked_points = track_movement_lsm(self.image1_matrix, self.image2_matrix, self.image1_transform,
                                            points_to_be_tracked=points_to_be_tracked,
                                            movement_cell_size=self.tracking_parameters.movement_cell_size,
                                            movement_tracking_area_size=
                                            self.tracking_parameters.movement_tracking_area_size,
                                            )
        # calculate the years between observations from the two given observation dates
        years_between_observations = (self.image2_observation_date - self.image1_observation_date).days // 365.25
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
        self.align_images(reference_area)
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

    def plot_tracking_results_lod_mask(self) -> None:
        """
        Plots the first raster image and the movement of points in a single figure. Every point that has 0 movement rate
        is shown in gray
        Returns
        -------
        """
        if self.tracking_results is not None:
            plot_movement_of_points_with_lod_mask(self.image1_matrix, self.image1_transform, self.tracking_results)
        else:
            logging.warning("No results calculated yet. Plot not provided")

    def calculate_lod(self, number_of_points: int, reference_area: gpd.GeoDataFrame) -> None:
        """
        Calculates the Level of Detection of a matching between two images. For calculating the LoD a specified number
        of points are generated randomly in some reference area, which is assumed to be stable. The level of detection
        is defined as some quantile (default: 0.5) of the movement rates of these points. The quantile can be set in the
        tracking parameters. This method sets the value of the lod as self.level_of_detection. The level of detection is
        given in movement per year.
        Parameters
        ----------
        number_of_points: int
            Defines the number of points used to calculate the level of detection.
        reference_area: gpd.GeoDataFrame
            The stable area (no motion assumed) on which the points for calculating the level of detection are created
            randomly. Since for image alignment an evenly spaced grid is used and here, we have a random distribution of
            points, it is possible to use the same reference area for both tasks.
        Returns
        -------
        None
        """
        print("Starting level of detection calculation.")
        years_between_observations = (self.image2_observation_date - self.image1_observation_date).days // 365.25

        # check if a custom level of detection value was provided with tracking parameters, if not set it to 0.5
        if self.tracking_parameters.level_of_detection_quantile is None:
            level_of_detection_quantile = 0.5
            self.tracking_parameters.level_of_detection_quantile = 0.5
        else:
            level_of_detection_quantile = self.tracking_parameters.level_of_detection_quantile
        self.level_of_detection = calculate_lod(image1_matrix=self.image1_matrix, image2_matrix=self.image2_matrix,
                                                image_transform=self.image1_transform, reference_area=reference_area,
                                                number_of_reference_points=number_of_points,
                                                tracking_parameters=self.tracking_parameters,
                                                crs=self.crs, years_between_observations=years_between_observations,
                                                level_of_detection_quantile=level_of_detection_quantile)
        print("Found level of detection with quantile " + str(level_of_detection_quantile) + " as "
              + str(self.level_of_detection))


    def filter_lod_points(self) -> None:
        """
        Sets the movement distance of all points that fall below the calculated level of detection to 0 and their
        movement bearing to NaN. Note that this directly affects the dataframe self.tracking_results.
        Returns
        -------
        """
        self.tracking_results = filter_lod_points(self.tracking_results, self.level_of_detection)
