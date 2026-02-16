import skimage
import numpy as np

def equalize_adapthist_images(image_matrix, kernel_size):
    equalized_image = skimage.exposure.equalize_adapthist(image=image_matrix.astype(int), kernel_size=kernel_size,
                                                          clip_limit=0.9)
    return equalized_image

# image1_matrix = skimage.exposure.equalize_adapthist(image=image1_matrix.astype(int),
# kernel_size=movement_tracking_area_size, clip_limit=0.9)
# image2_matrix = skimage.exposure.equalize_adapthist(image=image2_matrix.astype(int),
# kernel_size=movement_tracking_area_size, clip_limit=0.9)
# rasterio.plot.show(image1_matrix)
# rasterio.plot.show(image2_matrix)



def undistort_camera_image(image_matrix: np.ndarray, camera_intrinsic_matrix, distortion_coefficients: np.ndarray)\
        -> np.ndarray:
    """
    Undistorts camera image given as a np.array by employing opencv under the hood and returns the undistorted image
    cropped to a rectangular shape containing only valid pixels.
    WARNING: Since we are working in image coordinates, this function should not be seen as a transformation from image
    coordinates to camera coordinates. The coordinates that are intrinsically given by the returned np.array have to be
    transformed using the camera intrinsic matrix at a later point if such a transformation is desired. The undistortion
    step is considered a preprocessing step here, such that later transformations don't have to consider inverting the
    distortion and such that tracking is done on an undistorted image.
    Parameters
    ----------
    image_matrix: np.ndarray
        The array representing the distorted image.
    camera_intrinsic_matrix: np.ndarray
        The intrinsic matrix of the camera. Assumed to have the format [[f_x, s, c_x],\n
                                                                        [0, f_y, c_y],\n
                                                                        [0, 0, 1]]
    distortion_coefficients: np.ndarray
        Distortion coefficients of the camera as a one-dimensional np.array. The format is as required by opencv, i.e.
        in most cases either a 2-element array, containing the two radial distortion coefficients or a 4-element array
        containing first the two radial and then the two tangential distortion coefficients.
    Returns
    -------
    image_matrix_undistorted: np.ndarray
        The image matrix corresponding to the undistorted image. The image is cropped to a rectangular shape, where all
        pixels are valid (no invalid areas at the boundary). Therefore, the image shape is reduced depending on the
        severity of the distortion compared with the original array. The image matrix should have the same format as the
        original array (e.g. HxWxC for rasterio-read data).
    """
    import cv2

    assert camera_intrinsic_matrix.shape == (3, 3), "The camera intrinsic matrix must be of shape (3,3)."
    assert ((camera_intrinsic_matrix[2, :] == np.array([0,0,1])).all()
            & (camera_intrinsic_matrix[1,0] == 0)), ("The camera intrinsic matrix must be of the form\n"
                                                    "[[f_x, s, c_x],\n"
                                                    "[0, f_y, c_y],\n"
                                                    "[0, 0, 1]]")

    change_format = (image_matrix.shape[0] == 3)
    if change_format:
        image_matrix = np.transpose(image_matrix, axes=(1, 2, 0))

    im_width = image_matrix.shape[1]
    im_height = image_matrix.shape[0]
    newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_intrinsic_matrix, distortion_coefficients, (im_width, im_height), 1, (im_width, im_height))

    image_matrix_undistorted = cv2.undistort(src=image_matrix,
                                             cameraMatrix=camera_intrinsic_matrix,
                                             distCoeffs=distortion_coefficients,
                                             newCameraMatrix=newCameraMatrix
                                             )

    # crop the image
    x, y, w, h = roi
    image_matrix_undistorted = image_matrix_undistorted[y:y + h, x:x + w]
    if change_format:
        image_matrix_undistorted = np.transpose(image_matrix_undistorted, axes=(2, 0, 1))
    return image_matrix_undistorted
