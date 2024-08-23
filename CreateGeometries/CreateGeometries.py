import geopandas as gpd
import shapely
import numpy as np
import math


def grid_points_on_polygon(polygon: gpd.GeoDataFrame, number_of_points: int = 10):
    """Creates a grid of roughly a given number of evenly spaced points on the given polygon. The exact number of points
    can vary depending on the shape of the polygon."""
    minlongitude, minlatitude, maxlongitude, maxlatitude = polygon.bounds.iloc[0]

    length_latitude = maxlatitude-minlatitude
    length_longitude = maxlongitude-minlongitude
    enclosing_rectangle = shapely.Polygon((
        (minlongitude, minlatitude),
        (maxlongitude, minlatitude),
        (maxlongitude, maxlatitude),
        (minlongitude, maxlatitude),
        (minlongitude, minlatitude)
    ))

    enclosing_rectangle = gpd.GeoDataFrame(index=[0], crs=polygon.crs, geometry=[enclosing_rectangle])

    area_ratio=(enclosing_rectangle.area/polygon.area).iloc[0]
    number_of_latitude_points = np.sqrt(length_latitude/length_longitude*number_of_points)
    number_of_longitude_points = (length_longitude/length_latitude*number_of_latitude_points)
    number_of_latitude_points *= np.sqrt(area_ratio)
    number_of_longitude_points *= np.sqrt(area_ratio)
    number_of_latitude_points = math.ceil(number_of_latitude_points)
    number_of_longitude_points = math.ceil(number_of_longitude_points)

    points = []
    for lat in np.arange(minlatitude, maxlatitude, length_latitude/number_of_latitude_points):
        for lon in np.arange(minlongitude, maxlongitude, length_longitude/number_of_longitude_points):
            points.append(shapely.Point(lon, lat))

    points = gpd.GeoDataFrame(crs=polygon.crs, geometry=points)


    points = points[points.intersects(polygon.loc[0, "geometry"])]
    print("Created ", len(points), " points on the polygon.")
    return points