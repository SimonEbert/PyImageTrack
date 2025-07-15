import geopandas as gpd
import rasterio
import rasterio.plot
from rasterio.crs import CRS
from datetime import datetime
import logging
from geocube.api.core import make_geocube
import os
from rasterio.coords import BoundingBox
from geocube.rasterize import rasterize_points_griddata
from shapely.geometry import box
import numpy as np

# Parameter classes
from Parameters.TrackingParameters import TrackingParameters
from Parameters.FilterParameters import FilterParameters
# Alignment and Tracking functions
from ImageTracking.TrackMovement import track_movement_lsm
from CreateGeometries.HandleGeometries import crop_images_to_intersection
from ImageTracking.AlignImages import align_images_lsm_scarce
from CreateGeometries.HandleGeometries import grid_points_on_polygon_by_distance
from CreateGeometries.HandleGeometries import georeference_tracked_points
# Plotting
from Plots.MakePlots import plot_movement_of_points
from Plots.MakePlots import plot_movement_of_points_with_valid_mask
from Plots.MakePlots import plot_raster_and_geometry
# DataPreProcessing
from DataProcessing.ImagePreprocessing import equalize_adapthist_images
# filter functions
from DataProcessing.DataPostprocessing import calculate_lod_points
from DataProcessing.DataPostprocessing import filter_lod_points
from DataProcessing.DataPostprocessing import filter_outliers_full
# Geometry Handling
from CreateGeometries.HandleGeometries import random_points_on_polygon_by_number



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

        # Parameters
        self.tracking_parameters = TrackingParameters(parameter_dict=parameter_dict)
        self.filter_parameters = None

        # Meta-Data and results
        self.crs = None
        self.tracked_control_points = None
        self.tracking_results = None
        self.level_of_detection = None
        self.level_of_detection_points = None

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
        file1 = rasterio.open(filename_1, 'r+')
        file2 = rasterio.open(filename_2, 'r+')

        if file1.crs != file2.crs:
            raise ValueError("Got images with crs " + str(file1.crs) + " and " + str(file2.crs) +
                             "but the two images must  have the same crs.")
        self.crs = file1.crs
        # set the valid data box for the intersection of the two images
        bbox1 = file1.bounds
        bbox2 = file2.bounds

        poly1 = box(*bbox1)
        poly2 = box(*bbox2)
        intersection = poly1.intersection(poly2)
        # leave a buffer to the boundary so that every search_cell is contained in the valid data area
        image_bounds = gpd.GeoDataFrame(gpd.GeoDataFrame({'geometry': [intersection]}, crs=self.crs).buffer(
            -max(-file1.transform[4],file1.transform[0])*self.tracking_parameters.movement_tracking_area_size))
        # set correct geometry column
        image_bounds = image_bounds.rename(columns={0: "geometry"})
        image_bounds.set_geometry("geometry", inplace=True)

        self.image_bounds = image_bounds

        ([self.image1_matrix, self.image1_transform],
         [self.image2_matrix, self.image2_transform]) = crop_images_to_intersection(file1, file2)
        self.image1_observation_date = datetime.strptime(observation_date_1, "%d-%m-%Y").date()
        self.image2_observation_date = datetime.strptime(observation_date_2, "%d-%m-%Y").date()

        self.select_image_channels(selected_channels=selected_channels)

    def load_images_from_matrix_and_transform(self, image1_matrix: np.ndarray, observation_date_1: str,
                                              image2_matrix: np.ndarray, observation_date_2: str, image_transform, crs,
                                              selected_channels:int = None):
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
        self.image1_observation_date = datetime.strptime(observation_date_1, "%d-%m-%Y").date()
        self.image2_observation_date = datetime.strptime(observation_date_2, "%d-%m-%Y").date()
        self.crs=crs


        bbox = rasterio.transform.array_bounds(image1_matrix.shape[-2], image1_matrix.shape[-1], image_transform)
        poly1 = box(*bbox)
        image_bounds = gpd.GeoDataFrame(gpd.GeoDataFrame({'geometry': [poly1]}, crs=self.crs).buffer(
            -max(-image_transform[4], image_transform[0]) * self.tracking_parameters.movement_tracking_area_size))
        # set correct geometry column
        image_bounds = image_bounds.rename(columns={0: "geometry"})
        image_bounds.set_geometry("geometry", inplace=True)
        self.image_bounds = image_bounds
        # self.select_image_channels(selected_channels=selected_channels)




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

        try:
            [_, new_image2_matrix, tracked_control_points] = (
                align_images_lsm_scarce(image1_matrix=self.image1_matrix, image2_matrix=self.image2_matrix,
                                       image_transform=self.image1_transform, reference_area=reference_area,
                                        number_of_control_points=
                                        self.tracking_parameters.image_alignment_number_of_control_points,
                                        cell_size=self.tracking_parameters.image_alignment_control_cell_size,
                                        tracking_area_size=self.tracking_parameters.image_alignment_control_tracking_area_size,
                                        cross_correlation_threshold=
                                        self.tracking_parameters.cross_correlation_threshold_alignment,
                                        maximal_alignment_movement=self.tracking_parameters.maximal_alignment_movement))
            self.valid_alignment_possible = True
        except:
            self.valid_alignment_possible = False
            return


        years_between_observations = (self.image2_observation_date - self.image1_observation_date).days / 365.25
        self.tracked_control_points = georeference_tracked_points(tracked_control_points, self.image1_transform,
                                                                  self.crs, years_between_observations)

        self.image2_matrix = new_image2_matrix
        self.image2_transform = self.image1_transform

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
                                            cross_correlation_threshold=self.tracking_parameters.cross_correlation_threshold_movement
                                            )
        # calculate the years between observations from the two given observation dates
        years_between_observations = (self.image2_observation_date - self.image1_observation_date).days / 365.25
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

    def calculate_lod(self, points_for_lod_calculation: gpd.GeoDataFrame, filter_parameters: FilterParameters = None) -> None:
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

        points_for_lod_calculation = gpd.GeoDataFrame(points_for_lod_calculation.intersection(self.image_bounds.geometry[0]))
        points_for_lod_calculation.rename(columns={0: 'geometry'}, inplace=True)
        points_for_lod_calculation.set_geometry('geometry', inplace=True)

        years_between_observations = (self.image2_observation_date - self.image1_observation_date).days / 365.25
        # check if a LoD filter parameter is provided, if this is None, don't perform LoD calculation
        if (filter_parameters.level_of_detection_quantile is None
            or filter_parameters.number_of_points_for_level_of_detection is None):
            return

        level_of_detection_quantile = filter_parameters.level_of_detection_quantile

        unfiltered_level_of_detection_points = calculate_lod_points(image1_matrix=self.image1_matrix, image2_matrix=self.image2_matrix,
                                                              image_transform=self.image1_transform,
                                                              points_for_lod_calculation=points_for_lod_calculation,
                                                              tracking_parameters=self.tracking_parameters,
                                                              crs=self.crs, years_between_observations=years_between_observations)
        self.level_of_detection_points = unfiltered_level_of_detection_points

        self.level_of_detection = np.nanquantile(unfiltered_level_of_detection_points["movement_distance_per_year"],
                                         level_of_detection_quantile)

        print("Found level of detection with quantile " + str(level_of_detection_quantile) + " as "
              + str(self.level_of_detection))

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
        points_for_lod_calculation = random_points_on_polygon_by_number(reference_area, filter_parameters.number_of_points_for_level_of_detection)
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
        Saves the full results of tracking as a geosjon and the movement bearing and rate as geotiffs in the provided
        folder. If a level of detection is present, additionally the respective LoD mask is saved. A visualization of
        the tracking results is saved as a jpg file to the same folder.
        Parameters
        ----------
        folder_path: str
            The folder where all results will be saved.
        save_files: list
            A list, giving the files that should be saved. Possible options are: "movement_bearing_valid_tif", "movement_bearing_full_tif",
              "movement_rate_valid_tif", "movement_rate_full_tif",
              "movement_rate_with_lod_points_tif", "movement_bearing_with_lod_points_tif",
              "statistical_parameters_csv", "LoD_points_geojson", "control_points_geojson", "first_image_matrix", "second_image_matrix".
              The tracking parameters and the full tracking results (as geojson) will always be saved to prevent loss of
              data.
        Returns
        -------
        """


        if "first_image_matrix" in save_files:
            metadata = {
                'driver': 'GTiff',
                'count': 1,  # Number of bands
                'dtype': self.image1_matrix.dtype,  # Adjust if necessary
                'crs': str(self.crs),  # Define the Coordinate Reference System (CRS)
                'width': self.image1_matrix.shape[1],  # Number of columns (x)
                'height': self.image1_matrix.shape[0],  # Number of rows (y)
                'transform': self.image1_transform,  # Affine transform for georeferencing
            }

            with rasterio.open(folder_path + "/image_" + str(self.image1_observation_date.year) + ".tif", 'w', **metadata) as dst:
                dst.write(self.image1_matrix, 1)

        if "second_image_matrix" in save_files:
            metadata = {
                'driver': 'GTiff',
                'count': 1,  # Number of bands
                'dtype': self.image2_matrix.dtype,  # Adjust if necessary
                'crs': str(self.crs),  # Define the Coordinate Reference System (CRS)
                'width': self.image2_matrix.shape[1],  # Number of columns (x)
                'height': self.image2_matrix.shape[0],  # Number of rows (y)
                'transform': self.image2_transform,  # Affine transform for georeferencing
            }

        if not self.valid_alignment_possible:
            return
        os.makedirs(folder_path, exist_ok=True)

        self.tracking_results.to_file(folder_path + "/tracking_results_" + str(self.image1_observation_date.year) + "_"
                                      + str(self.image2_observation_date.year) + ".geojson", driver="GeoJSON")
        tracking_results_valid = self.tracking_results.loc[self.tracking_results["valid"], :]
        results_grid_valid = make_geocube(vector_data=tracking_results_valid,
                                          measurements=["movement_bearing_pixels", "movement_distance_per_year"],
                                          resolution = self.tracking_parameters.distance_of_tracked_points)

        if "is_outlier" in tracking_results_valid.columns:
            is_outlier = (self.tracking_results["is_bearing_difference_outlier"]
                | self.tracking_results["is_bearing_standard_deviation_outlier"]
                | self.tracking_results["is_movement_rate_difference_outlier"]
                | self.tracking_results["is_movement_rate_standard_deviation_outlier"]
            )
            tracking_results_without_outliers = self.tracking_results.loc[~is_outlier]
            results_grid_filtered = make_geocube(vector_data=tracking_results_without_outliers,
                                             measurements=["movement_bearing_pixels", "movement_distance_per_year"],
                                             resolution = self.tracking_parameters.distance_of_tracked_points,
                                                 rasterize_function=rasterize_points_griddata)

        if "movement_bearing_valid_tif" in save_files:
            results_grid_valid["movement_bearing_pixels"].rio.to_raster(folder_path + "/movement_bearing_valid_"
                                                                  + str(self.image1_observation_date.year)
                                                                  + "_" + str(self.image2_observation_date.year)
                                                                  + ".tif")
        if "movement_rate_valid_tif" in save_files:
            results_grid_valid["movement_distance_per_year"].rio.to_raster(folder_path + "/movement_rate_valid_"
                                                                     + str(self.image1_observation_date.year)
                                                                     + "_" + str(self.image2_observation_date.year)
                                                                     + ".tif")

        if "movement_bearing_outlier_filtered_tif" in save_files:
            results_grid_filtered["movement_bearing_pixels"].rio.to_raster(
                folder_path +"/movement_bearing_outlier_filtered_" + str(self.image1_observation_date.year)
                                                                   + "_" + str(self.image2_observation_date.year)
                                                                   + ".tif")

        if "movement_rate_outlier_filtered_tif" in save_files:
            results_grid_filtered["movement_bearing_pixels"].rio.to_raster(
                folder_path + "/movement_bearing_outlier_filtered_" + str(self.image1_observation_date.year)
                + "_" + str(self.image2_observation_date.year)
                + ".tif")

        if "invalid_mask_tif" in save_files:
            invalid_mask = self.tracking_results.loc[~self.tracking_results["valid"]]
            invalid_mask = invalid_mask.copy()
            invalid_mask.loc[:, "valid_int"] = invalid_mask["valid"].astype(int)
            invalid_mask_grid = make_geocube(vector_data=invalid_mask,
                                         measurements=["valid_int"],
                                         resolution=self.tracking_parameters.distance_of_tracked_points)
            invalid_mask_grid["valid_int"].rio.to_raster(folder_path + "/Invalid_mask"
                                                        + str(self.image1_observation_date.year)
                                                        + "_" + str(self.image2_observation_date.year)
                                                        + ".tif")

        if "LoD_points_geojson" in save_files:
            self.level_of_detection_points.to_file(
                folder_path + "/LoD_points_" + str(self.image1_observation_date.year) + "_"
                + str(self.image2_observation_date.year) + ".geojson", driver="GeoJSON")


        if "control_points_geojson" in save_files:
            self.tracked_control_points.to_file(
                folder_path + "/control_points_" + str(self.image1_observation_date.year) + "_"
                + str(self.image2_observation_date.year) + ".geojson", driver="GeoJSON")


        if "statistical_parameters_txt" in save_files:
            total_number_of_points = len(self.tracking_results)
            number_of_points_below_lod = len(self.tracking_results[self.tracking_results["is_below_LoD"]])
            number_of_outliers = len(self.tracking_results[is_outlier])
            number_of_valid_lod_points = len(self.level_of_detection_points[self.level_of_detection_points["valid"]])
            total_number_of_lod_points = len(self.level_of_detection_points)
            with open(folder_path + "/statistical_results_" + str(self.image1_observation_date.year)
                                                                   + "_" + str(self.image2_observation_date.year)
                                                                   + ".txt", "w") as statistics_file:
                statistics_file.write("Total number of points: " + str(total_number_of_points) + "\n" +
                           "thereof\n\tbelow LoD: " + str(number_of_points_below_lod) + " (" + str(np.round(number_of_points_below_lod
                                / total_number_of_points * 100, decimals=2)) + "%)\n" +
                           "\toutliers: " + str(number_of_outliers) + "(" + str(np.round(number_of_outliers / total_number_of_points * 100, decimals=2))
                           + "%)\n"
                           + "\tthereof\n\t\t" + str(len(self.tracking_results[self.tracking_results["is_bearing_difference_outlier"]])) + " bearing difference outliers\n" +
                           "\t\t" + str(len(self.tracking_results[self.tracking_results["is_bearing_standard_deviation_outlier"]])) + " bearing standard deviation outliers\n" +
                           "\t\t" + str(len(self.tracking_results[self.tracking_results["is_movement_rate_difference_outlier"]])) + " movement rate difference outliers\n" +
                           "\t\t" + str(len(self.tracking_results[self.tracking_results["is_movement_rate_standard_deviation_outlier"]])) + " movement rate standard deviation outliers\n" +
                           "Valid points: " + str(len(tracking_results_valid)) + "(" + str(np.round(len(tracking_results_valid) / total_number_of_points * 100, decimals=2)) + "%)\n"
                           + "Movement rate with points below LoD:\n" +
                           "\tMean: " + str(np.nanmean(tracking_results_without_outliers["movement_distance_per_year"])) + "\n" +
                           "\tMedian: " + str(np.nanmedian(tracking_results_without_outliers["movement_distance_per_year"])) + "\n" +
                           "\tStandard deviation: " + str(np.nanstd(tracking_results_without_outliers["movement_distance_per_year"])) + "\n" +
                           "\tQ90: " + str(np.nanquantile(tracking_results_without_outliers["movement_distance_per_year"],0.9)) + "\n" +
                           "\tQ99: " + str(np.nanquantile(tracking_results_without_outliers["movement_distance_per_year"],0.99)) + "\n" +
                           "Movement rate without points below LoD:\n" +
                           "\tMean: " + str(np.nanmean(tracking_results_valid["movement_distance_per_year"])) + "\n" +
                           "\tMedian: " + str(np.nanmedian(tracking_results_valid["movement_distance_per_year"])) + "\n" +
                           "\tStandard deviation: " + str(np.nanstd(tracking_results_valid["movement_distance_per_year"])) + "\n" +
                           "\tQ90: " + str(np.nanquantile(tracking_results_valid["movement_distance_per_year"],0.9)) + "\n" +
                           "\tQ99: " + str(np.nanquantile(tracking_results_valid["movement_distance_per_year"],0.99)) + "\n" +
                           "Movement rate of LoD points:\n" +
                           "\tMean: " + str(np.nanmean(self.level_of_detection_points["movement_distance_per_year"])) + "\n" +
                           "\tMedian: " + str(np.nanmedian(self.level_of_detection_points["movement_distance_per_year"])) + "\n" +
                           "\tStandard deviation: " + str(np.nanstd(self.level_of_detection_points["movement_distance_per_year"])) + "\n" +
                           "\tQ90: " + str(np.nanquantile(self.level_of_detection_points["movement_distance_per_year"], 0.9)) + "\n" +
                           "\tQ99: " + str(np.nanquantile(self.level_of_detection_points["movement_distance_per_year"], 0.99)) + "\n" +
                           "\tUsed points: " + str(number_of_valid_lod_points) + " points\n"
                           )

        with (open(folder_path + "/parameters_" + str(self.image1_observation_date.year)
                                                                   + "_" + str(self.image2_observation_date.year)
                                                                   + ".txt", "w")
              as text_file):
            text_file.write(self.tracking_parameters.__str__())

        if self.filter_parameters is not None:
            with (open(folder_path + "/parameters_" + str(self.image1_observation_date.year)
                                                                   + "_" + str(self.image2_observation_date.year)
                                                                   + ".txt", "a") as text_file):
                text_file.write(self.filter_parameters.__str__())


        if self.level_of_detection is not None:
            if "lod_mask_tif" in save_files:
                lod_mask = self.tracking_results.loc[self.tracking_results["is_below_LoD"]]
                lod_mask = lod_mask.copy()
                lod_mask.loc[:, "is_below_LoD_int"] = lod_mask["is_below_LoD"].astype(int)
                lod_mask_grid = make_geocube(vector_data=lod_mask,
                                            measurements=["is_below_LoD_int"],
                                            resolution=self.tracking_parameters.distance_of_tracked_points)
                lod_mask_grid["is_below_LoD_int"].rio.to_raster(folder_path + "/LoD_mask_"
                                                                          + str(self.image1_observation_date.year)
                                                                          + "_" + str(self.image2_observation_date.year)
                                                                          + ".tif")

            plot_movement_of_points_with_valid_mask(self.image1_matrix, self.image1_transform, self.tracking_results,
                                                    save_path=folder_path + "/tracking_results_" +
                                                  str(self.image1_observation_date.year) + "_" +
                                                  str(self.image2_observation_date.year) + ".jpg")
            with (open(folder_path + "/parameters_" + str(self.image1_observation_date.year)
                                                                   + "_" + str(self.image2_observation_date.year)
                                                                   + ".txt", "a")
                  as text_file):
                text_file.write("Level of Detection: " + str(self.level_of_detection) + "\n")
        else:
            plot_movement_of_points(self.image1_matrix, self.image1_transform, self.tracking_results,
                                    save_path=folder_path + "/tracking_results_" +
                                              str(self.image1_observation_date.year) + "_" +
                                              str(self.image2_observation_date.year) + ".jpg")


    def load_results(self, file_path, reference_area):
        saved_tracking_results = gpd.read_file(file_path)
        saved_tracking_results = saved_tracking_results.loc[:, ["row", "column", "movement_row_direction", "movement_column_direction",
                        "movement_distance_pixels", "movement_bearing_pixels", "movement_distance", "movement_distance_per_year", "geometry"]]
        saved_tracking_results["valid"] = True
        self.align_images(reference_area)
        self.tracking_results = saved_tracking_results