import numpy as np
import pytest
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import gaussian_filter

from PyImageTrack.ImageTracking.ImageInterpolator import ImageInterpolator


@pytest.fixture
def smooth_image():
    rng = np.random.default_rng(0)
    return gaussian_filter(rng.normal(size=(60, 70)), 2.0) * 100 + 500


@pytest.fixture
def interior_points():
    rng = np.random.default_rng(1)
    # keep a margin to the border, where the boundary treatment of the two spline implementations differs
    return rng.uniform(10, 49, size=500), rng.uniform(10, 59, size=500)


def test_values_match_rect_bivariate_spline(smooth_image, interior_points):
    rows, cols = interior_points
    reference = RectBivariateSpline(np.arange(60), np.arange(70), smooth_image)
    interpolator = ImageInterpolator(smooth_image)
    np.testing.assert_allclose(interpolator.ev(rows, cols), reference.ev(rows, cols), atol=1e-4)


def test_derivatives_match_rect_bivariate_spline(smooth_image, interior_points):
    rows, cols = interior_points
    reference = RectBivariateSpline(np.arange(60), np.arange(70), smooth_image)
    interpolator = ImageInterpolator(smooth_image)
    np.testing.assert_allclose(interpolator.ev(rows, cols, dx=1), reference.ev(rows, cols, dx=1), atol=1e-2)
    np.testing.assert_allclose(interpolator.ev(rows, cols, dy=1), reference.ev(rows, cols, dy=1), atol=1e-2)


def test_ev_all_is_consistent_with_ev(smooth_image, interior_points):
    rows, cols = interior_points
    interpolator = ImageInterpolator(smooth_image)
    value, d_rows, d_cols = interpolator.ev_all(rows, cols)
    np.testing.assert_allclose(value, interpolator.ev(rows, cols))
    np.testing.assert_allclose(d_rows, interpolator.ev(rows, cols, dx=1), atol=1e-8)
    np.testing.assert_allclose(d_cols, interpolator.ev(rows, cols, dy=1), atol=1e-8)


def test_interpolates_exactly_at_grid_nodes(smooth_image):
    interpolator = ImageInterpolator(smooth_image)
    rows, cols = np.meshgrid(np.arange(60), np.arange(70), indexing="ij")
    result = interpolator.ev(rows.ravel(), cols.ravel()).reshape(60, 70)
    # not bit-exact: the boundary condition leaves ~1e-4 residuals and the clamp keeps 1e-4 px away from the last pixel
    np.testing.assert_allclose(result, smooth_image, atol=2e-3)


def test_multiband_matches_per_band(smooth_image, interior_points):
    rows, cols = interior_points
    stack = np.stack([smooth_image, smooth_image * 0.5 + 3.0, smooth_image[::-1]])
    interpolator = ImageInterpolator(stack)
    value, d_rows, d_cols = interpolator.ev_all(rows, cols)
    assert value.shape == (3, rows.size)
    for band in range(3):
        single = ImageInterpolator(stack[band])
        np.testing.assert_allclose(value[band], single.ev(rows, cols))
        np.testing.assert_allclose(d_rows[band], single.ev(rows, cols, dx=1))
        np.testing.assert_allclose(d_cols[band], single.ev(rows, cols, dy=1))


def test_singleton_dimensions_are_squeezed(smooth_image, interior_points):
    rows, cols = interior_points
    plain = ImageInterpolator(smooth_image).ev(rows, cols)
    wrapped = ImageInterpolator(smooth_image[None, None]).ev(rows, cols)
    np.testing.assert_allclose(wrapped, plain)


def test_integer_and_float32_input_use_double_precision(smooth_image, interior_points):
    rows, cols = interior_points
    as_float32 = smooth_image.astype(np.float32)
    expected = RectBivariateSpline(np.arange(60), np.arange(70), as_float32.astype(float)).ev(rows, cols)
    np.testing.assert_allclose(ImageInterpolator(as_float32).ev(rows, cols), expected, atol=1e-4)


def test_shape_argument_reshapes_output(smooth_image):
    rows, cols = np.meshgrid(np.arange(5, 15), np.arange(5, 12), indexing="ij")
    result = ImageInterpolator(smooth_image).ev(rows.ravel(), cols.ravel(), shape=(10, 7))
    assert result.shape == (10, 7)


def test_coordinates_outside_the_image_are_clamped(smooth_image):
    # same behaviour as scipy.interpolate.RectBivariateSpline, which the interpolator replaces
    interpolator = ImageInterpolator(smooth_image)
    np.testing.assert_allclose(interpolator.ev(np.array([-5.0]), np.array([10.0])),
                               interpolator.ev(np.array([0.0]), np.array([10.0])))
    np.testing.assert_allclose(interpolator.ev(np.array([30.0]), np.array([90.0])),
                               interpolator.ev(np.array([30.0]), np.array([69.0])), atol=1e-2)
    # derivatives at clamped coordinates are the derivatives at the border, not zero
    d_outside = interpolator.ev(np.array([70.0]), np.array([10.0]), dx=1)
    d_edge = interpolator.ev(np.array([59.0]), np.array([10.0]), dx=1)
    assert abs(d_edge[0]) > 0.1
    np.testing.assert_allclose(d_outside, d_edge, atol=1e-2)


def test_unsupported_derivative_orders_raise(smooth_image):
    interpolator = ImageInterpolator(smooth_image)
    with pytest.raises(ValueError):
        interpolator.ev(np.array([10.0]), np.array([10.0]), dx=2)
    with pytest.raises(ValueError):
        interpolator.ev(np.array([10.0]), np.array([10.0]), dx=1, dy=1)


def test_invalid_dimensionality_raises():
    with pytest.raises(ValueError):
        ImageInterpolator(np.zeros(10))
