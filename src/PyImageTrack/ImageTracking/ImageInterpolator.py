import numpy as np
from scipy.ndimage import map_coordinates, spline_filter1d

# Step used for the forward-difference derivatives of the spline (in pixels). The spline is evaluated in float64, so
# truncation error (~step/2 * second derivative) and round-off error (~1e-16 * value / step) are both far below the
# precision relevant for tracking.
_DERIVATIVE_STEP = 1e-4

# Number of edge-replicated pixels padded around the image before computing the spline coefficients. The influence of
# the boundary condition on the coefficients decays by a factor of ~0.27 per pixel, so 12 pixels (the value scipy.ndimage
# itself uses for mode="nearest") reduces it to the order of 1e-7.
_PADDING = 12


class ImageInterpolator:
    """
    Cubic B-spline interpolation of a 2D image or of a stack of independent 2D image channels (first axis).

    The spline coefficients are computed once with scipy.ndimage.spline_filter1d. Evaluation uses
    scipy.ndimage.map_coordinates with prefilter=False. Like scipy.interpolate.RectBivariateSpline (with s=0), this is
    an interpolating cubic spline that reproduces the image values at the pixel centers, and coordinates outside the
    image are clamped to the image domain. The two agree to about 1e-4 (for values of order 1e2-1e3) in the interior of
    the image and differ slightly near the borders, because of different boundary conditions (here: edge replication
    instead of not-a-knot).

    Coordinates are given as (row, column), i.e. along the last two axes of the image.
    """

    def __init__(self, image_matrix):
        # Remove all singleton dimensions to handle cases like (1, 1, H, W) -> (H, W)
        image_matrix = np.squeeze(image_matrix)

        # Ensure the result is 2D or 3D
        if image_matrix.ndim not in (2, 3):
            raise ValueError(f"Expected 2D or 3D array after squeezing, got shape {image_matrix.shape}")

        self.shape = image_matrix.shape

        # float64 is required: single precision coefficients would limit the sub-pixel accuracy of the tracking.
        padded = np.pad(np.asarray(image_matrix, dtype=np.float64),
                        [(0, 0)] * (image_matrix.ndim - 2) + [(_PADDING, _PADDING)] * 2, mode="edge")
        padded = spline_filter1d(padded, order=3, axis=-2, mode="mirror")
        self._coefficients = spline_filter1d(padded, order=3, axis=-1, mode="mirror")

        # Valid coordinate range. The upper bound leaves room for the forward-difference step of the derivatives.
        self._lower = np.array([[0.], [0.]])
        self._upper = np.array([[self.shape[-2] - 1 - _DERIVATIVE_STEP], [self.shape[-1] - 1 - _DERIVATIVE_STEP]])

    def _prepare_coordinates(self, yq, xq):
        """Stacks the coordinates to shape (2, n) and clamps them to the image domain."""
        coordinates = np.vstack([np.ravel(yq), np.ravel(xq)]).astype(np.float64)
        return np.clip(coordinates, self._lower, self._upper)

    def _sample(self, coordinates):
        """Evaluates the spline at coordinates of shape (2, n). Returns shape (n,) or (channels, n)."""
        shifted = coordinates + _PADDING
        if len(self.shape) == 2:
            return map_coordinates(self._coefficients, shifted, order=3, mode="nearest", prefilter=False)
        return np.stack([map_coordinates(channel, shifted, order=3, mode="nearest", prefilter=False)
                         for channel in self._coefficients], axis=0)

    def ev(self, yq, xq, dx=0, dy=0, shape=None):
        """
        Evaluates the spline (dx = dy = 0) or its first derivative along the rows (dx = 1) or the columns (dy = 1) at
        the points (yq, xq). Only first derivatives are supported. For repeated evaluation of the value and both
        derivatives at the same points, use ev_all, which is considerably faster.
        """
        if dx not in (0, 1) or dy not in (0, 1) or (dx and dy):
            raise ValueError("Only the value or a single first derivative (dx=1 or dy=1) are supported.")
        coordinates = self._prepare_coordinates(yq, xq)
        value = self._sample(coordinates)
        if dx or dy:
            step = np.array([[_DERIVATIVE_STEP], [0.]]) if dx else np.array([[0.], [_DERIVATIVE_STEP]])
            result = (self._sample(coordinates + step) - value) / _DERIVATIVE_STEP
        else:
            result = value
        if shape is not None:
            result = result.reshape(shape)
        return result

    def ev_all(self, yq, xq):
        """
        Evaluates the spline and its first derivatives along the rows and along the columns at the points (yq, xq)
        in one single call to scipy.ndimage.map_coordinates.

        Returns
        -------
        value, derivative_rows, derivative_columns:
            Each of shape (n,) for a single-band image or (channels, n) for a multi-band image.
        """
        coordinates = self._prepare_coordinates(yq, xq)
        n = coordinates.shape[1]
        batched = np.concatenate([coordinates,
                                  coordinates + np.array([[_DERIVATIVE_STEP], [0.]]),
                                  coordinates + np.array([[0.], [_DERIVATIVE_STEP]])], axis=1)
        sampled = self._sample(batched)
        value = sampled[..., :n]
        derivative_rows = (sampled[..., n:2 * n] - value) / _DERIVATIVE_STEP
        derivative_columns = (sampled[..., 2 * n:] - value) / _DERIVATIVE_STEP
        return value, derivative_rows, derivative_columns
