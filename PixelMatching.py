import geopandas as gpd
import rasterio.mask
import rasterio.plot

def find_matching_area(raster_image, tracked_point: gpd.GeoDataFrame, matching_radius: int=5, search_radius: int=30):
    tracked_point = tracked_point.to_crs(raster_image.crs)

    if len(tracked_point)!=1:
        raise ValueError("There is not exactly one point to be tracked.")

    #if ~tracked_point.touches(raster_image):
        #raise ValueError("Tracked point and raster image do not intersect.")

    masking_shape_matching_area = tracked_point.buffer(distance=matching_radius)
    matching_area, matching_area_transform = rasterio.mask.mask(dataset=raster_image, shapes=masking_shape_matching_area, crop=True)
    rasterio.plot.show(matching_area)
    print(matching_area)