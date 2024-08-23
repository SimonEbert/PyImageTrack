import rasterio
import rasterio.plot

def read_two_image_files(path1: str, path2: str):
    """Reads two .tiff files from the given paths and returns them as rasterio objects"""
    file1 = rasterio.open(path1)
    file2 = rasterio.open(path2)
    return file1, file2
