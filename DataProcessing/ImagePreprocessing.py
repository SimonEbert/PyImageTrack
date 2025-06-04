import skimage

def equalize_adapthist_images(image_matrix, kernel_size):
    equalized_image = skimage.exposure.equalize_adapthist(image=image_matrix.astype(int), kernel_size=kernel_size, clip_limit=0.9)
    return equalized_image

# image1_matrix = skimage.exposure.equalize_adapthist(image=image1_matrix.astype(int),
# kernel_size=movement_tracking_area_size, clip_limit=0.9)
# image2_matrix = skimage.exposure.equalize_adapthist(image=image2_matrix.astype(int),
# kernel_size=movement_tracking_area_size, clip_limit=0.9)
# rasterio.plot.show(image1_matrix)
# rasterio.plot.show(image2_matrix)