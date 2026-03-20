import skimage
import numpy as np
import scipy.ndimage
import geopandas as gpd
from shapely.geometry import Polygon

from ..ConsoleOutput import get_console


def equalize_adapthist_images(image_matrix, kernel_size, clip_limit):
    """
    Applies adaptive histogram equalization to enhance image contrast.
    
    This function uses Contrast Limited Adaptive Histogram Equalization (CLAHE)
    to improve local contrast while limiting noise amplification.
    
    Parameters
    ----------
    image_matrix : np.ndarray
        The input image matrix to be enhanced.
    kernel_size : tuple or int
        Defines the shape of contextual regions used for local contrast
        enhancement. If tuple, should be (height, width). If int, a square
        kernel of that size is used.
    clip_limit : float
        Contrast limiting threshold. Higher values allow more contrast
        enhancement but may increase noise. Typical values are between 0.0
        and 1.0.
    
    Returns
    -------
    np.ndarray
        The contrast-enhanced image matrix.
    """
    equalized_image = skimage.exposure.equalize_adapthist(image=image_matrix.astype(np.uint16), kernel_size=kernel_size,
                                                          clip_limit=clip_limit)
    return equalized_image


def get_optimal_camera_model(image_matrix_shape: tuple, camera_intrinsic_matrix: np.ndarray,
                             distortion_coefficients: np.ndarray):
    import cv2

    assert len(image_matrix_shape) == 2, "Image matrix is assumed to be two-dimensional height x width."

    assert camera_intrinsic_matrix.shape == (3, 3), "The camera intrinsic matrix must be of shape (3,3)."
    assert ((camera_intrinsic_matrix[2, :] == np.array([0, 0, 1])).all()
            & (camera_intrinsic_matrix[1, 0] == 0)), ("The camera intrinsic matrix must be of the form\n"
                                                      "[[f_x, s, c_x],\n"
                                                      "[0, f_y, c_y],\n"
                                                      "[0, 0, 1]]")


    im_width = image_matrix_shape[1]
    im_height = image_matrix_shape[0]
    newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_intrinsic_matrix, distortion_coefficients, (im_width, im_height), 1, (im_width, im_height))
    return newCameraMatrix, roi


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
        The intrinsic matrix of the camera. Assumed to have the format [[f_x, s, c_x],\
                                                                        [0, f_y, c_y],\
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

    assert camera_intrinsic_matrix.shape == (3, 3), "The camera intrinsic matrix must be of shape (3,3)."
    assert ((camera_intrinsic_matrix[2, :] == np.array([0,0,1])).all()
            & (camera_intrinsic_matrix[1,0] == 0)), ("The camera intrinsic matrix must be of the form\n"
                                                    "[[f_x, s, c_x],\n"
                                                    "[0, f_y, c_y],\n"
                                                    "[0, 0, 1]]")
    import cv2
    change_format = (image_matrix.shape[0] == 3)
    if change_format:
        image_matrix = np.transpose(image_matrix, axes=(1, 2, 0))

    image_matrix_shape = image_matrix.shape[0:2]

    newCameraMatrix, roi = get_optimal_camera_model(image_matrix_shape,
                                                    camera_intrinsic_matrix,
                                                    distortion_coefficients,)

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


def convert_float_to_uint(image_matrix: np.ndarray) -> np.ndarray:
    """
    Convert any input image to ``uint16`` (wrapper around :func:`convert_to_uint16`).

    Historically intended for float inputs, this now delegates to
    :func:`convert_to_uint16` which supports a wide range of dtypes (ints, floats,
    bool) and handles NaNs by replacing them with 0 before scaling. Floats and
    signed integers are scaled into the full ``uint16`` range; unsigned integers
    are scaled/clamped as appropriate; booleans map to {0, 65535}.

    Parameters
    ----------
    image_matrix : np.ndarray
        Input image matrix (2D or 3D) of any dtype supported by
        :func:`convert_to_uint16`.

    Returns
    -------
    np.ndarray
        ``uint16`` image matrix.
    """
    return convert_to_uint16(image_matrix)


def convert_to_uint16(image_matrix: np.ndarray) -> np.ndarray:
    """
    Converts an image matrix to uint16 for optimal alignment results.
    Supports various input dtypes: uint8, uint16, uint32, int8, int16, int32, int64, float32, float64, bool.
    
    Parameters
    ----------
    image_matrix : np.ndarray
        The image matrix to convert. Can be single-band (2D) or multi-band (3D).
    
    Returns
    -------
    np.ndarray
        The converted image matrix as uint16.
    """
    console = get_console()
    
    # If already uint16, return as-is
    if image_matrix.dtype == np.uint16:
        return image_matrix
    
    # Handle NaN values by replacing them with 0
    nan_count = 0
    if np.issubdtype(image_matrix.dtype, np.floating):
        nan_mask = np.isnan(image_matrix)
        nan_count = np.sum(nan_mask)
        if nan_count > 0:
            image_matrix = np.where(nan_mask, 0, image_matrix)
    
    data_min = np.min(image_matrix)
    data_max = np.max(image_matrix)
    
    # Conversion strategies based on dtype
    orig_dtype = str(image_matrix.dtype)
    
    if image_matrix.dtype == np.uint8:
        # Scale up: uint8 range (0-255) -> uint16 range (0-65535)
        result = image_matrix.astype(np.uint16) * 256
        return result
        
    elif image_matrix.dtype == np.uint32:
        # Scale down: clamp values > 65535 to 65535
        result = np.clip(image_matrix, 0, 65535).astype(np.uint16)
        if data_max > 65535:
            console.warning(f"Clamped {orig_dtype} values > 65535 to uint16 range. Some precision lost.")
        return result
        
    elif np.issubdtype(image_matrix.dtype, np.signedinteger):
        # SIGNED integers: shift and scale to uint16
        if data_max > data_min:
            result = ((image_matrix.astype(np.float64) - data_min) / (data_max - data_min) * 65535).astype(np.uint16)
        else:
            result = np.zeros_like(image_matrix, dtype=np.uint16)
        return result
        
    elif np.issubdtype(image_matrix.dtype, np.floating):
        # FLOAT inputs: scale to uint16 range
        if data_max > data_min:
            result = ((image_matrix - data_min) / (data_max - data_min) * 65535).astype(np.uint16)
        else:
            result = np.zeros_like(image_matrix, dtype=np.uint16)
        if nan_count > 0:
            console.success(f"Converted {orig_dtype} to uint16 (replaced {nan_count} NaN values with 0).")
        else:
            console.success(f"Converted {orig_dtype} to uint16.")
        return result
        
    elif image_matrix.dtype == np.bool_:
        # Boolean: 0/1 -> 0/65535
        result = (image_matrix.astype(np.uint16) * 65535)
        return result
        
    else:
        # Unknown dtype: attempt direct conversion with clamping
        console.warning(f"Unexpected dtype {orig_dtype}. Attempting direct conversion to uint16.")
        result = np.clip(image_matrix, 0, 65535).astype(np.uint16)
    
    return result


def harmonize_dtypes(image1_matrix: np.ndarray, image2_matrix: np.ndarray):
    """
    Ensures both images have the same dtype by converting to uint16 if needed.
    
    Parameters
    ----------
    image1_matrix : np.ndarray
        First image matrix.
    image2_matrix : np.ndarray
        Second image matrix.
    
    Returns
    -------
    tuple
        (image1_matrix, image2_matrix) with matching dtypes.
    """
    console = get_console()
    
    if image1_matrix.dtype == image2_matrix.dtype:
        # Already matching - nothing to do
        return image1_matrix, image2_matrix
    
    dt1 = str(image1_matrix.dtype)
    dt2 = str(image2_matrix.dtype)
    
    console.warning(f"Datatype mismatch: {dt1} vs {dt2} - harmonizing to uint16")
    
    # Convert both to uint16
    image1_converted = convert_to_uint16(image1_matrix)
    image2_converted = convert_to_uint16(image2_matrix)
    
    return image1_converted, image2_converted


def harmonize_resolution(image1_matrix: np.ndarray, image2_matrix: np.ndarray,
                         image1_transform, image2_transform):
    """
    Match resolutions by downsampling to the smaller common shape.

    The function computes target height/width as the minimum of both images and
    uses cubic interpolation to downsample either image that exceeds the target
    in any dimension. Affine transforms are scaled accordingly. If a transform
    is ``None``, a fallback pixel size of 1.0 is assumed for logging.

    Parameters
    ----------
    image1_matrix : np.ndarray
        First image matrix.
    image2_matrix : np.ndarray
        Second image matrix.
    image1_transform : rasterio.transform.Affine
        Transform of the first image (or ``None``).
    image2_transform : rasterio.transform.Affine
        Transform of the second image (or ``None``).

    Returns
    -------
    tuple
        (image1_matrix, image2_matrix, image1_transform, image2_transform) with matching resolution.
    """
    console = get_console()
    
    # Check if shapes already match
    if image1_matrix.shape == image2_matrix.shape:
        return image1_matrix, image2_matrix, image1_transform, image2_transform
    
    # Calculate pixel sizes in meters from transforms
    if image1_transform is not None:
        px_size1 = max(abs(image1_transform.a), abs(image1_transform.e))
    else:
        px_size1 = 1.0  # fallback
    
    if image2_transform is not None:
        px_size2 = max(abs(image2_transform.a), abs(image2_transform.e))
    else:
        px_size2 = 1.0  # fallback
    
    # Only warn when resolutions differ meaningfully
    if not np.isclose(px_size1, px_size2, rtol=1e-6, atol=1e-6):
        console.warning(f"Resolution mismatch: {px_size1:.3f}m/px vs {px_size2:.3f}m/px")
    
    # Determine which image has higher resolution
    h1, w1 = image1_matrix.shape[-2], image1_matrix.shape[-1]
    h2, w2 = image2_matrix.shape[-2], image2_matrix.shape[-1]
    
    # Use the smaller dimensions as the target
    target_h = min(h1, h2)
    target_w = min(w1, w2)
    
    # Downsampling factors
    factor1_h = h1 / target_h
    factor1_w = w1 / target_w
    factor2_h = h2 / target_h
    factor2_w = w2 / target_w
    
    from affine import Affine
    
    # Convert image1 if needed
    if factor1_h > 1.0 or factor1_w > 1.0:
        zoom_factors = (1.0 / factor1_h, 1.0 / factor1_w)
        
        if len(image1_matrix.shape) == 3:
            # 3D case (bands, height, width)
            bands, height, width = image1_matrix.shape
            image1_down = np.zeros((bands, target_h, target_w), dtype=image1_matrix.dtype)
            for b in range(bands):
                image1_down[b] = scipy.ndimage.zoom(image1_matrix[b], zoom_factors, order=3, mode='constant', cval=0)
            image1_matrix = image1_down
        else:
            # 2D case
            image1_matrix = scipy.ndimage.zoom(image1_matrix, zoom_factors, order=3, mode='constant', cval=0)
        
        image1_transform = image1_transform * Affine.scale(factor1_w, factor1_h)
    
    # Convert image2 if needed
    if factor2_h > 1.0 or factor2_w > 1.0:
        zoom_factors = (1.0 / factor2_h, 1.0 / factor2_w)
        
        if len(image2_matrix.shape) == 3:
            # 3D case (bands, height, width)
            bands, height, width = image2_matrix.shape
            image2_down = np.zeros((bands, target_h, target_w), dtype=image2_matrix.dtype)
            for b in range(bands):
                image2_down[b] = scipy.ndimage.zoom(image2_matrix[b], zoom_factors, order=3, mode='constant', cval=0)
            image2_matrix = image2_down
        else:
            # 2D case
            image2_matrix = scipy.ndimage.zoom(image2_matrix, zoom_factors, order=3, mode='constant', cval=0)
        
        image2_transform = image2_transform * Affine.scale(factor2_w, factor2_h)
        new_px_size = px_size2 * max(factor2_h, factor2_w)
        console.success(f"Downsampled image2 by {max(factor2_h, factor2_w):.2f}x to {new_px_size:.3f}m/px")
    
    return image1_matrix, image2_matrix, image1_transform, image2_transform


def check_channels_compatible(image1_matrix: np.ndarray, image2_matrix: np.ndarray) -> None:
    """
    Checks if the two images have compatible channel configurations.
    Raises ValueError if incompatible.
    
    Parameters
    ----------
    image1_matrix : np.ndarray
        First image matrix.
    image2_matrix : np.ndarray
        Second image matrix.
    
    Raises
    ------
    ValueError
        If the images have incompatible channel configurations (different dimensions
        or different number of bands).
    """
    ndim1 = image1_matrix.ndim
    ndim2 = image2_matrix.ndim
    
    # Both 2D: OK
    if ndim1 == 2 and ndim2 == 2:
        return
    
    # Both 3D: check number of bands
    if ndim1 == 3 and ndim2 == 3:
        bands1 = image1_matrix.shape[0]
        bands2 = image2_matrix.shape[0]
        if bands1 != bands2:
            raise ValueError(
                f"Incompatible number of channels between images. "
                f"Image 1 has {bands1} channel(s) (shape: {image1_matrix.shape}), "
                f"Image 2 has {bands2} channel(s) (shape: {image2_matrix.shape}). "
                f"Cannot process images with different channel configurations."
            )
        return
    
    # Mixed 2D and 3D: error
    raise ValueError(
        f"Incompatible channel dimensions between images. "
        f"Image 1 has shape {image1_matrix.shape} ({ndim1}D), "
        f"Image 2 has shape {image2_matrix.shape} ({ndim2}D). "
        f"Cannot process images with different channel configurations. "
        f"Both images must have the same dimensionality (both 2D or both 3D)."
    )



def undistort_polygon(polygon: gpd.GeoDataFrame, image_shape: tuple, camera_intrinsic_matrix: np.ndarray,
                      distortion_coefficients: np.ndarray,
                      ):
    import cv2

    polygon_geom = polygon.geometry.iloc[0]
    point_coordinates = np.array(polygon_geom.exterior.coords)
    point_coordinates = point_coordinates.astype(np.float32).reshape(-1,1,2)
    point_coordinates[..., 1] = -point_coordinates[..., 1]


    newCameraMatrix, roi = get_optimal_camera_model(image_shape,camera_intrinsic_matrix,distortion_coefficients)

    undistorted_point_coordinates = cv2.undistortPoints(point_coordinates, camera_intrinsic_matrix,
                                                        distortion_coefficients,
                                                        P=newCameraMatrix)

    undistorted_xy = undistorted_point_coordinates.reshape(-1, 2)
    undistorted_xy[:, 0] -= roi[0]
    undistorted_xy[:, 1] -= roi[1]
    undistorted_xy[..., 1] = -undistorted_xy[...,1]
    undistorted_point_coordinates = Polygon(undistorted_xy)
    undistorted_polygon = gpd.GeoDataFrame(geometry=[undistorted_point_coordinates], crs=polygon.crs).make_valid()
    return undistorted_polygon

