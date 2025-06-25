import matplotlib.pyplot as plt
import requests
import imageio.v3
import numpy as np


def download_wms_data(minx: float, miny: float, maxx: float, maxy: float, width_image_pixels: float,
                      height_image_pixels: float, epsg_code, layer_name,
                      wms_url: str = "https://gis.tirol.gv.at/arcgis/services/Service_Public/orthofoto/MapServer/WMSServer"):
    # Define the parameters for the WMS GetMap request
    params = {
        'SERVICE': 'WMS',
        'VERSION': '1.3.0',
        'REQUEST': 'GetMap',
        'LAYERS': layer_name,
        'STYLES': '',  # Empty since no specific style is specified
        'FORMAT': 'image/tiff',
        'TRANSPARENT': 'FALSE',
        'CRS': 'EPSG:' + str(epsg_code),  # Coordinate reference system
        'BBOX': str(miny) + "," + str(minx) + "," + str(maxy) + "," + str(maxx),  # Bounding box
        'WIDTH': int(width_image_pixels),  # Image width
        'HEIGHT': int(height_image_pixels),  # Image height
    }

    # Send the GET request to the WMS server
    response = requests.get(wms_url, params=params)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        print("Tile downloaded successfully.")
    else:
        print(f"Failed to download the tile. HTTP Status code: {response.status_code}")

    # Read image data and store to np array
    image_data = np.array(imageio.v3.imread(response.content))

    # Swap axis according to rasterio's convention
    image_data = image_data.transpose(2, 0, 1)

    return image_data


import matplotlib.pyplot as plt
import geopandas as gpd
import rasterio.plot
import shapely
import skimage
from datetime import datetime
from ImageTracking.ImagePair import ImagePair
from Parameters.FilterParameters import FilterParameters


# improve figure quality
plt.rcParams['figure.dpi'] = 300

pixels_per_metre = 2
epsg_code = 31254


# Set parameters
number_of_control_points = 2000
image_bands = 0
control_tracking_area_size = 60
control_cell_size = 40
distance_of_tracked_points = 5
movement_cell_size = 30
cross_correlation_threshold_alignment = 0.85
cross_correlation_threshold_movement = 0.6


# === SAVE OPTIONS ===
# tracking_results.geojson will always be saved, since it contains the full information.
save_files = ["movement_bearing_valid_tif", "movement_bearing_outlier_filtered_tif",
              "movement_rate_valid_tif", "movement_rate_outlier_filtered_tif", "invalid_mask_tif", "lod_mask_tif",
              "statistical_parameters_txt"]


# === FILTER PARAMETERS ===
# FILTERING OPTIONS: For preventing the use of a specific filter, set the respective values to None
# Filters points whose movement bearing deviates more than the given threshold from the movement bearing of surrounding
# points
filter_parameters = FilterParameters({
    "level_of_detection_quantile": 0.9,
    "number_of_points_for_level_of_detection": 1000,
    # Filters points, whose movement bearings deviate more than the given threshold from the movementt rate of surrounding points
    "difference_movement_bearing_threshold": 360, # in degrees
    "difference_movement_bearing_moving_window_size": 40, # in units of the used crs
    # Filters points, where the standard deviation of movement bearing of neighbouring points exceeds the given threshold
    "standard_deviation_movement_bearing_threshold": 60, # in degrees,
    "standard_deviation_movement_bearing_moving_window_size": 20, # in units of the used crs
    # Filters points whose movement rates deviate more than the given threshold from the movement rate of surrounding points
    "difference_movement_rate_threshold": 2, # in units of the used crs / year
    "difference_movement_rate_moving_window_size": 50, # in units of the used crs
    # Filters points whose standard deviation of movement rates of neighbouring points exceeds the given threshold
    "standard_deviation_movement_rate_threshold": 2, # in units of the used crs / year
    "standard_deviation_movement_rate_moving_window_size": 30 # in units of the used crs
})



#
# available_layer_names = ["Image_1999_2004", "Image_2004_2009", "Image_2009_2012", "Image_2013_2015", "Image_2019_2021", "Image_Aktuell_RGB"]
# flight_years = [2003, 2009, 2010, 2015, 2020, 2023]
# flight_years_str = ["05-09-2003", "09-09-2009", "31-07-2010", "03-08-2015", "08-09-2020", "06-09-2023"]

# Possible layer names: ["Image_1949_1954", "Image_1970_1982", "Image_1999_2004", "Image_2004_2009", "Image_2009_2012", "Image_2013_2015", "Image_2019_2021", "Image_Aktuell_RGB"]
available_layer_names = ["Image_1949_1954", "Image_1970_1982"]
flight_years = [1953, 1971]
flight_years_str = ["02-09-1953", "19-08-1971"]

# is already in EPSG:31254, otherwise transform!!
rock_glacier_inventory_tyrol = gpd.read_file("../Rock_glacier_analysis_Tyrol/rock_glacier_inventory_tyrol/rock_glacier_polygons_tyrol.shp")

rock_glacier_id = 201
rock_glacier_id = 443
# for rock_glacier_id in rock_glacier_inventory_tyrol.index:




for layer_name_picture1 in available_layer_names[:-1]:
    index_first_layer = available_layer_names.index(layer_name_picture1)
    layer_name_picture2 = available_layer_names[index_first_layer + 1]

    years_between_observations = flight_years[index_first_layer + 1] - flight_years[index_first_layer]

    movement_tracking_area_size = 2*2*years_between_observations*pixels_per_metre+movement_cell_size


    # print("Rock Glacier ID: " + str(rock_glacier_id))

    single_rock_glacier = rock_glacier_inventory_tyrol.iloc[[rock_glacier_id, ]]

    # change index of the single feature dataframe to 0
    single_rock_glacier.set_index(np.arange(1), inplace=True)

    minx = single_rock_glacier.bounds.loc[0, 'minx'] - 100
    miny = single_rock_glacier.bounds.loc[0, 'miny'] - 100
    maxx = single_rock_glacier.bounds.loc[0, 'maxx'] + 100
    maxy = single_rock_glacier.bounds.loc[0, 'maxy'] + 100

    extent_corners = gpd.GeoDataFrame(["minx_miny", "maxx_miny", "minx_maxy", "maxx_maxy"],
                                      columns=["names"],
                                      geometry=[shapely.geometry.Point(minx, miny),
                                                shapely.geometry.Point(maxx, miny),
                                                shapely.geometry.Point(minx, maxy),
                                                shapely.geometry.Point(maxx, maxy)],
                                      crs=single_rock_glacier.crs)

    extent_area = gpd.GeoDataFrame(geometry=[shapely.geometry.Polygon(
        ((minx + 50, miny + 50), (maxx - 50, miny + 50), (maxx - 50, maxy - 50), (minx + 50, maxy - 50)))],
                                   crs=single_rock_glacier.crs)

    reference_area = gpd.GeoDataFrame(geometry=extent_area.difference(single_rock_glacier, align=False))


    width_image_crs_unit = extent_corners.iloc[0].geometry.distance(extent_corners.iloc[1].geometry)
    height_image_crs_unit = extent_corners.iloc[0].geometry.distance(extent_corners.iloc[2].geometry)

    width_image_pixels = np.ceil(width_image_crs_unit * pixels_per_metre)
    height_image_pixels = np.ceil(height_image_crs_unit * pixels_per_metre)

    if (width_image_pixels > 6000 or height_image_pixels > 6000):
        print("Skipping since images with more than 6000 pixels edge length cannot be downloaded.")
        continue

    # download raster data for the first image
    raster_image1 = download_wms_data(minx=minx, miny=miny, maxx=maxx, maxy=maxy, width_image_pixels=width_image_pixels,
                                      height_image_pixels=height_image_pixels, epsg_code=epsg_code,
                                      layer_name=layer_name_picture1)


    if np.all(raster_image1 == 255):
        print("Skipping due to missing data")
        continue

    # download raster data for the second image
    raster_image2 = download_wms_data(minx=minx, miny=miny, maxx=maxx, maxy=maxy, width_image_pixels=width_image_pixels,
                                      height_image_pixels=height_image_pixels, epsg_code=epsg_code,
                                      layer_name=layer_name_picture2)



    #  define the four corners as ground control points and
    ground_control_points = [rasterio.transform.GroundControlPoint(row=0, col=0,
                                                                   x=minx, y=maxy),
                             rasterio.transform.GroundControlPoint(row=height_image_pixels - 1, col=0,
                                                                   x=minx, y=miny),
                             rasterio.transform.GroundControlPoint(row=0, col=width_image_pixels - 1,
                                                                   x=maxx, y=maxy),
                             rasterio.transform.GroundControlPoint(row=height_image_pixels - 1, col=width_image_pixels - 1,
                                                                   x=maxx, y=miny)]

    raster_transform_image1 = rasterio.transform.from_gcps(ground_control_points)

    raster_image1 = raster_image1[0]
    raster_image2 = raster_image2[0]

    image_pair = ImagePair(
        parameter_dict={"image_alignment_number_of_control_points": number_of_control_points,
                        "used_image_bands": image_bands,
                        "image_alignment_control_tracking_area_size": control_tracking_area_size,
                        "image_alignment_control_cell_size": control_cell_size,
                        "distance_of_tracked_points": distance_of_tracked_points,
                        "movement_tracking_area_size": movement_tracking_area_size,
                        "movement_cell_size": movement_cell_size,
                        "cross_correlation_threshold_alignment": cross_correlation_threshold_alignment,
                        "cross_correlation_threshold_movement": cross_correlation_threshold_movement})

    image_pair.load_images_from_matrix_and_transform(image1_matrix=raster_image1, observation_date_1=flight_years_str[index_first_layer],
                                                     image2_matrix=raster_image2, observation_date_2=flight_years_str[index_first_layer + 1],
                                                     image_transform=raster_transform_image1, crs=reference_area.crs)


    image_pair.equalize_adapthist_images()
    print(image_pair.tracking_parameters.movement_tracking_area_size)

    image_pair.perform_point_tracking(reference_area=reference_area, tracking_area=single_rock_glacier)

    image_pair.full_filter(reference_area=reference_area, filter_parameters=filter_parameters)

    image_pair.plot_tracking_results_with_valid_mask()

    image_pair.save_full_results("../Test_results_Tirol/" + str(rock_glacier_id) + "/full_results", save_files=save_files)





    # [image1_matrix, image2_matrix, image_transform] = (
    #     AlignImages.align_images_lsm_scarce(raster_image1_equalized, raster_image2_equalized, raster_transform_image1,
    #                                         reference_area=reference_area,
    #                                         number_of_control_points=number_of_control_points,
    #                                         tracking_area_size=control_tracking_area_size,
    #                                         cell_size=control_cell_size))
    #
    #
    # tracked_pixels = PixelMatching.track_movement(image1_matrix, image2_matrix, image_transform, single_rock_glacier,
    #                                               distance_of_tracked_points=distance_of_tracked_points,
    #                                               tracking_area_size=movement_tracking_area_size,
    #                                               cell_size=movement_cell_size,
    #                                               tracking_method=tracking_method,
    #                                               remove_outliers=remove_outliers,
    #                                               retry_matching=retry_matching)
    #
    # tracked_pixels = georeference_tracked_points(tracked_pixels, image_transform, crs=single_rock_glacier.crs,
    #                                              years_between_observations=years_between_observations)
    #
    # # tracked_pixels = gpd.read_file("../../Output_results/20250319171023/tracking_results.geojson")
    #
    #
    #
    # # save parameter dictionary
    # parameter_dict = {"layer_image1": layer_name_picture1,
    #                   "layer_image2": layer_name_picture2,
    #                   "alignment_via_lsm": alignment_via_lsm,
    #                   "number_of_control_points": number_of_control_points,
    #                   "image_bands": image_bands,
    #                   "control_tracking_area_size": control_tracking_area_size,
    #                   "control_cell_size": control_cell_size,
    #                   "tracking_method": tracking_method,
    #                   # "number_of_tracked_points": number_of_tracked_points,
    #                   "distance_of_tracked_points": distance_of_tracked_points,
    #                   "movement_tracking_area_size": movement_tracking_area_size,
    #                   "movement_cell_size": movement_cell_size,
    #                   "remove_outliers": remove_outliers,
    #                   "retry_matching": retry_matching,
    #                   "years_between_observations": years_between_observations,
    #                   "computation_time": "NA"
    #                   }
    #
    #
    # # tracked_pixels, level_of_detection = remove_points_below_LoD(image1_matrix,image2_matrix,image_transform,reference_area,tracked_pixels)
    #
    # # write results with the parameter dictionary to the specified directory
    # HandleFiles.write_results(tracked_pixels, parameter_dict, folder_path=output_folder_path)
    #
    # # plot and save the movement visualization
    # plot_movement_of_points(raster_image1, image_transform, tracked_pixels,
    #                         save_path=output_folder_path + "/point_movement_" + datetime.now().strftime(
    #                             "%Y_%m_%d_%H_%M_%S") + ".png")
    #
    # # plot without saving
    # # plot_movement_of_points(image1_matrix, image_transform, tracked_pixels)