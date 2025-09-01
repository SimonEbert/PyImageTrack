import numpy as np
import rasterio

from ImageTracking.ImagePair import ImagePair
from ImageTracking.TrackMovement import track_cell_lsm
from CreateGeometries.HandleGeometries import get_submatrix_symmetric


# Set parameters
number_of_control_points = 200
image_bands = 0
control_tracking_area_size = 320
control_cell_size = 300
distance_of_tracked_points = 5
movement_tracking_area_size = 60
movement_cell_size = 20
cross_correlation_threshold_alignment = 0.75
cross_correlation_threshold_movement = 0.5




Test_Image_Pair = ImagePair(
    parameter_dict={"image_alignment_number_of_control_points": number_of_control_points,
                    "used_image_bands": image_bands,
                    "image_alignment_control_tracking_area_size": control_tracking_area_size,
                    "image_alignment_control_cell_size": control_cell_size,
                    "distance_of_tracked_points": distance_of_tracked_points,
                    "movement_tracking_area_size": movement_tracking_area_size,
                    "movement_cell_size": movement_cell_size,
                    "cross_correlation_threshold_alignment": cross_correlation_threshold_alignment,
                    "cross_correlation_threshold_movement": cross_correlation_threshold_movement})

Test_Image_Pair.load_images_from_file(filename_1="../Lisa_Kaunertal/Testdaten_Alignment/2022-07_HS_10.tif",
                                            observation_date_1="01-07-2022",
                                            filename_2="../Lisa_Kaunertal/Testdaten_Alignment/2023-07_HS_10.tif",
                                            observation_date_2="01-07-2023",
                                            selected_channels=0, NA_value=-9999)

print(Test_Image_Pair.image1_matrix.shape, Test_Image_Pair.image2_matrix.shape)

image1_submatrix = get_submatrix_symmetric([1500,5500], [500,500], Test_Image_Pair.image1_matrix)
image2_submatrix = get_submatrix_symmetric([1499,5498], [500,500], Test_Image_Pair.image2_matrix)
rasterio.plot.show(image1_submatrix)
rasterio.plot.show(image2_submatrix)

metadata = {
                'driver': 'GTiff',
                'count': 1,  # Number of bands
                'dtype': Test_Image_Pair.image1_matrix.dtype,  # Adjust if necessary
                'crs': str(Test_Image_Pair.crs),  # Define the Coordinate Reference System (CRS)
                'width': Test_Image_Pair.image1_matrix.shape[1],  # Number of columns (x)
                'height': Test_Image_Pair.image1_matrix.shape[0],  # Number of rows (y)
                'transform': Test_Image_Pair.image1_transform,  # Affine transform for georeferencing
            }

with rasterio.open("/media/simon/Swap/Dokumente/Studium/14.Semester/HiWi_Arbeit_PyImageTrack/Lisa_Kaunertal" + "/image_" + str(Test_Image_Pair.image1_observation_date.year) + ".tif", 'w', **metadata) as dst:
                dst.write(Test_Image_Pair.image1_matrix, 1)

metadata = {
                'driver': 'GTiff',
                'count': 1,  # Number of bands
                'dtype': Test_Image_Pair.image2_matrix.dtype,  # Adjust if necessary
                'crs': str(Test_Image_Pair.crs),  # Define the Coordinate Reference System (CRS)
                'width': Test_Image_Pair.image2_matrix.shape[1],  # Number of columns (x)
                'height': Test_Image_Pair.image2_matrix.shape[0],  # Number of rows (y)
                'transform': Test_Image_Pair.image2_transform,  # Affine transform for georeferencing
            }

with rasterio.open("/media/simon/Swap/Dokumente/Studium/14.Semester/HiWi_Arbeit_PyImageTrack/Lisa_Kaunertal" + "/image_" + str(Test_Image_Pair.image2_observation_date.year) + ".tif", 'w', **metadata) as dst:
                dst.write(Test_Image_Pair.image2_matrix, 1)
