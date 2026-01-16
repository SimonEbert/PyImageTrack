import skimage
import numpy as np
import cv2

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
    Undistorts camera image by employing opencv under the hood
    """
    assert camera_intrinsic_matrix.shape == (3, 3), "The camera intrinsic matrix must be of shape (3,3)."
    assert ((camera_intrinsic_matrix[2, :] == np.array([0,0,1])).all()
            & (camera_intrinsic_matrix[1,0] == 0)), ("The camera intrinsic matrix must be of the form\n"
                                                    "[[f_x, s, c_x],\n"
                                                    "[0, f_y, c_y],\n"
                                                    "[0, 0, 1]]")
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
    image_matrix_undistorted = np.transpose(image_matrix_undistorted, axes=(2, 0, 1))
    return image_matrix_undistorted
