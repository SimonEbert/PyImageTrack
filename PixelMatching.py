import geopandas as gpd
import rasterio.mask
import rasterio.plot
import numpy as np
import shapely
import numpy.linalg as la


def get_buffer_around_point(raster_image, center_point: gpd.GeoDataFrame, buffer_radius: int = 30):
    center_point = center_point.to_crs(raster_image.crs)

    if len(center_point) != 1:
        raise ValueError("There is not exactly one point to be tracked.")

    masking_shape_matching_area = center_point.buffer(distance=buffer_radius)
    matching_area, matching_area_transform = rasterio.mask.mask(dataset=raster_image,
                                                                shapes=masking_shape_matching_area, crop=True)

    return matching_area, matching_area_transform


def find_matching_area(tracked_image, search_image, tracked_point: gpd.GeoDataFrame, matching_radius: int = 5, search_radius: int = 10):
    global optimal_match_point
    tracked_point = tracked_point.to_crs(tracked_image.crs)

    if len(tracked_point) != 1:
        raise ValueError("There is not exactly one point to be tracked.")

    if tracked_image.crs != search_image.crs:
        raise ValueError("The two raster images are not based on the same crs")

    #if ~tracked_point.touches(raster_image):
        #raise ValueError("Tracked point and raster image do not intersect.")

    matching_area, matching_area_transform = get_buffer_around_point(raster_image=tracked_image,
                                                                     center_point=tracked_point,
                                                                     buffer_radius=matching_radius)

    search_area, search_area_transform = get_buffer_around_point(raster_image=search_image,
                                                                 center_point=tracked_point,
                                                                 buffer_radius=search_radius)

    rasterio.plot.show(matching_area)
    height = search_area.shape[1]
    width = search_area.shape[2]
    cols, rows = np.meshgrid(range(width), range(height))
    x, y = rasterio.transform.xy(search_area_transform, rows, cols)
    longitudes = np.array(x[0])
    latitudes = np.array(y[0])
    best_loss = np.inf
    for lon in longitudes:
        for lat in latitudes:
            center_point = gpd.GeoDataFrame(index=[0], crs=tracked_image.crs, geometry=[shapely.Point(lon, lat)])
            compare_area, compare_area_transform = get_buffer_around_point(raster_image=search_image,
                                                                           center_point=center_point,
                                                                           buffer_radius=matching_radius)
            loss = la.norm(matching_area[0]-compare_area[0], ord=1)
            if loss < best_loss:
                best_loss = loss
                optimal_match_point = [lon, lat]
    return optimal_match_point

