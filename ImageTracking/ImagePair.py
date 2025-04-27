
import rasterio
import shapely
import numpy as np

from dataloader import TrackingParameters
from CreateGeometries.HandleGeometries import crop_images_to_intersection
from PixelMatching import align_images

class ImagePair:
    def __init__(self, parameter_dict: dict = None):
        self.images_aligned = False
        self.image1_matrix = None
        self.image1_transform = None
        self.image1_observation_date = None
        self.image2_matrix = None
        self.image2_transform = None
        self.image2_observation_date = None

        self.tracking_parameters = TrackingParameters.TrackingParameters(parameter_dict=parameter_dict)

    def load_images_from_file(self, filename_1: str, filename_2: str):
        """
        Loads two image files from the respective file paths. The order of the provided image paths is expected to
        align with the observation order, that is the first image is assumed to be the earlier observation. The two
        images are cropped to the same geospatial extent, assuming they are given in the same coordinate reference
        system.
        Parameters
        ----------
        filename_1: str
            The filename of the first image
        filename_2: str
            The filename of the second image
        Returns
        -------

        """
        file1 = rasterio.open(filename_1)
        file2 = rasterio.open(filename_2)
        if file1.crs != file2.crs:
            raise ValueError("Got images with crs " + str(file1.crs) + " and " + str(file2.crs) +
                             "but the two images must  have the same crs.")
        ([self.image1_matrix, self.image1_transform],
         [self.image2_matrix, self.image2_transform]) = crop_images_to_intersection(file1, file2)

    def select_image_channels(self, selected_channels: list = None):
        if selected_channels is None:
            selected_channels = [0, 1, 2]
        if len(self.image1_matrix.shape) == 3:
            self.image1_matrix = self.image1_matrix[selected_channels, :, :]
            self.image2_matrix = self.image2_matrix[selected_channels, :, :]


    def align_images(self, reference_area):








