import geopandas as gpd
import rasterio

from ImageTracking.ImagePair import ImagePair


# Set parameters
alignment_via_lsm = True
number_of_control_points = 2000
image_bands = 0
control_tracking_area_size = 60
control_cell_size = 40
tracking_method = "lsm"
distance_of_tracked_points = 5
movement_tracking_area_size = 60
movement_cell_size = 20
level_of_detection_quantile = 0.9
use_4th_channel_as_data_mask = True

Kaiserberg_pair_19_21 = ImagePair(
    parameter_dict={"image_alignment_via_lsm": alignment_via_lsm,
                    "image_alignment_number_of_control_points": number_of_control_points,
                    "used_image_bands": image_bands,
                    "image_alignment_control_tracking_area_size": control_tracking_area_size,
                    "image_alignment_control_cell_size": control_cell_size,
                    "distance_of_tracked_points": distance_of_tracked_points,
                    "movement_tracking_area_size": movement_tracking_area_size,
                    "movement_cell_size": movement_cell_size,
                    "level_of_detection_quantile": level_of_detection_quantile,
                    "use_4th_channel_as_data_mask": use_4th_channel_as_data_mask})

Kaiserberg_pair_19_21.load_images_from_file(filename_1="../Test_Data/Orthophotos_Kaiserberg_historic/1953_ortho_1m_RG_rend_bw.tif",
                                            observation_date_1="02-09-1953",
                                            filename_2="../Test_Data/Orthophotos_Kaiserberg_historic/1970_ortho_1m_RG_rend_bw.tif",
                                            observation_date_2="29-09-1970",
                                            selected_channels=0)


polygon_outside_RG = gpd.read_file("../Test_Data/Orthophotos_Kaiserberg_historic/Area_outside_rock_glacier_adjusted.shp")
polygon_outside_RG = polygon_outside_RG.to_crs(crs=31254)

rock_glacier_polygon = gpd.read_file("../Test_Data/Orthophotos_Kaiserberg_historic/Area_inside_rock_glacier.shp")
rock_glacier_polygon = rock_glacier_polygon.to_crs(crs=31254)

Kaiserberg_pair_19_21.perform_point_tracking(reference_area=polygon_outside_RG, tracking_area=rock_glacier_polygon)

Kaiserberg_pair_19_21.calculate_lod(1000, polygon_outside_RG)
Kaiserberg_pair_19_21.filter_lod_points()

Kaiserberg_pair_19_21.plot_tracking_results_lod_mask()

Kaiserberg_pair_19_21.save_full_results("../Test_results")
