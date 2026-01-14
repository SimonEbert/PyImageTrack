from functools import reduce

import geopandas as gpd
import numpy as np

from src import FilterParameters
from src import ImagePair
from src import TrackingParameters
from src import equalize_adapthist_images
from src import random_points_on_polygon_by_number


class ImageBatch:

    def __init__(self, parameter_dict: dict = None):

        # Image data
        self.list_of_image_pairs = list()
        self.data_bounds = None


        # Parameters
        self.tracking_parameter_dict = parameter_dict
        self.tracking_parameters = TrackingParameters(parameter_dict=parameter_dict)
        self.filter_parameters = None
        self.crs = None

        self.lod_points = None

    def load_images_from_file_list(self, list_of_image_files: list[str], list_of_observation_dates: list,
                                   pixels_per_metre: float, maximal_assumed_movement_rate: float):
        if len(list_of_image_files) != len(list_of_observation_dates):
            raise ValueError('Number of image files does not match the number of observation dates.')

        for i in range(len(list_of_image_files) - 1):

            file1 = list_of_image_files[i]
            file2 = list_of_image_files[i + 1]
            observation_date1 = list_of_observation_dates[i]
            observation_date2 = list_of_observation_dates[i + 1]
            self.list_of_image_pairs.append(ImagePair(self.tracking_parameter_dict))

            observation_time_difference = list_of_observation_dates[i + 1] - list_of_observation_dates[i]

            delta_hours = observation_time_difference.total_seconds() / 3600.0
            years_between_observations = delta_hours / (24.0 * 365.25)

            # Derive a symmetric search extent (posx=negx=posy=negy) from your movement budget [px]
            movement_budget_px = np.ceil(
                2 * maximal_assumed_movement_rate * years_between_observations * pixels_per_metre
                + self.tracking_parameters.movement_cell_size
            )
            half_extent = int(max(1, np.floor(movement_budget_px / 2)))
            self.list_of_image_pairs[-1].tracking_parameters.search_extent_px = (half_extent, half_extent, half_extent, half_extent)

            observation_date1 = observation_date1.strftime("%d-%m-%Y")
            observation_date2 = observation_date2.strftime("%d-%m-%Y")
            self.list_of_image_pairs[-1].load_images_from_file(filename_1=file1, observation_date_1=observation_date1,
                                                               filename_2=file2, observation_date_2=observation_date2,
                                                               selected_channels=0)

            # clip matrix values to 0, ..., 255
            self.list_of_image_pairs[-1].image1_matrix[self.list_of_image_pairs[-1].image1_matrix > 255] = 0
            self.list_of_image_pairs[-1].image2_matrix[self.list_of_image_pairs[-1].image2_matrix > 255] = 0

        self.crs = self.list_of_image_pairs[0].crs
        # print(self.list_of_image_pairs[0].image1_observation_date, self.list_of_image_pairs[0].image2_observation_date)
        # print(self.list_of_image_pairs[1].image1_observation_date, self.list_of_image_pairs[1].image2_observation_date)
        # print(self.list_of_image_pairs[2].image1_observation_date, self.list_of_image_pairs[2].image2_observation_date)
        # print(self.list_of_image_pairs[3].image1_observation_date, self.list_of_image_pairs[3].image2_observation_date)


        list_of_image_bounds = [image_pair.image_bounds for image_pair in self.list_of_image_pairs]
        image_bounds_geometries = [bounds.geometry.iloc[0] for bounds in list_of_image_bounds]
        intersected_image_bounds_geometry = reduce(lambda a, b: a.intersection(b), image_bounds_geometries)
        self.data_bounds = gpd.GeoDataFrame(geometry=[intersected_image_bounds_geometry], crs=self.crs)


    # def align_images(self, reference_area: gpd.GeoDataFrame):
    #     for image_pair in self.list_of_image_pairs:
    #         image_pair.align_images(reference_area)
    #
    # def track_points(self, tracking_area: gpd.GeoDataFrame):
    #     for image_pair in self.list_of_image_pairs:
    #         print("Starting tracking for " + str(image_pair.image1_observation_date))
    #
    #         image_pair.track_points(tracking_area)
    #         image_pair.plot_tracking_results()

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
        for image_pair in self.list_of_image_pairs:
            print("Starting tracking for " + str(image_pair.image1_observation_date))
            image_pair.perform_point_tracking(reference_area=reference_area, tracking_area=tracking_area)

    def equalize_adapthist_images(self):
        for image_pair in self.list_of_image_pairs:
            image_pair.image1_matrix = equalize_adapthist_images(image_pair.image1_matrix,
                                                           kernel_size=50)
            image_pair.image2_matrix = equalize_adapthist_images(image_pair.image2_matrix,
                                                           kernel_size=50)


    def calculate_and_filter_lod(self, filter_parameters: FilterParameters, reference_area: gpd.GeoDataFrame = None):
        if self.filter_parameters is None:
            self.filter_parameters = filter_parameters
        if self.lod_points is None and reference_area is not None:
            lod_points = random_points_on_polygon_by_number(reference_area, self.filter_parameters.number_of_points_for_level_of_detection)
            lod_points = lod_points.intersection(self.data_bounds.geometry[0], align=False)
            self.lod_points = lod_points
        if self.lod_points is None:
            raise ValueError("Was not able to define LoD points.")

        for image_pair in self.list_of_image_pairs:
            image_pair.calculate_lod(self.lod_points, filter_parameters=filter_parameters)
            image_pair.filter_lod_points()

    def filter_outliers(self, filter_parameters: FilterParameters):
        for image_pair in self.list_of_image_pairs:
            image_pair.filter_outliers(filter_parameters)

    def save_full_results(self, folder_path: str, save_files: list):
        tracking_results_full = None
        for image_pair in self.list_of_image_pairs:
            image_pair.save_full_results(folder_path=folder_path + "/" + str(image_pair.image1_observation_date.year) + "_" + str(image_pair.image2_observation_date.year), save_files=save_files)

            if not image_pair.valid_alignment_possible:
                continue
            tracking_results = image_pair.tracking_results
            tracking_results = tracking_results.add_suffix("_" + str(image_pair.image1_observation_date.year) + "_" + str(image_pair.image2_observation_date.year))
            tracking_results = tracking_results.rename(columns={("geometry_" + str(image_pair.image1_observation_date.year) + "_" + str(image_pair.image2_observation_date.year)): "geometry"})
            tracking_results.set_geometry("geometry", inplace=True)

            if tracking_results_full is None:
                tracking_results_full = tracking_results
            else:
                tracking_results_full = gpd.sjoin(tracking_results_full, tracking_results,
                                              predicate="intersects")
                tracking_results_full = tracking_results_full.drop(['index_right'], axis=1)
        valid_cols = [column_name for column_name in tracking_results_full.columns if 'valid' in column_name]
        valid_tracking_results_full = tracking_results_full[tracking_results_full[valid_cols].all(axis=1)]
        valid_tracking_results_full.to_file(folder_path + "/full_time_series_results.geojson", driver="GeoJSON")






