import numpy as np
import pytest

from PyImageTrack.DataProcessing.ImageEnhancement import (
    ImageProcessingState,
    apply_image_enhancement_state,
    normalize_image_enhancement,
)


def _config(steps, **tables):
    config = {"image_enhancement": {"image_enhancement_steps": steps}}
    config.update({f"image_enhancement_{name}": [parameters] for name, parameters in tables.items()})
    return config


def test_normalize_image_enhancement_preserves_order_and_parameters():
    config = _config(
        ["clahe", "clahe"],
        clahe={"kernel_size": 50, "clip_limit": 0.9},
    )
    # One table is valid for one occurrence only.
    with pytest.raises(ValueError):
        normalize_image_enhancement(config)

    config["image_enhancement_clahe"].append({"kernel_size": 25, "clip_limit": 0.5})
    steps = normalize_image_enhancement(config)
    assert [step["type"] for step in steps] == ["clahe", "clahe"]
    assert steps[1]["parameters"]["kernel_size"] == 25


def test_unknown_or_incomplete_configuration_has_one_error():
    with pytest.raises(ValueError, match="Invalid image enhancement configuration"):
        normalize_image_enhancement({"image_enhancement": {"image_enhancement_steps": ["gaussian"]}})

    with pytest.raises(ValueError, match="Invalid image enhancement configuration"):
        normalize_image_enhancement(_config(["clahe"], clahe={"kernel_size": 50}))


def test_clahe_state_preserves_band_layout_and_selected_bands():
    image = np.arange(48, dtype=np.uint16).reshape(3, 4, 4)
    steps = normalize_image_enhancement(
        _config(["clahe"], clahe={"kernel_size": 2, "clip_limit": 0.9, "bands": [1]})
    )
    state = apply_image_enhancement_state(image, steps)

    assert isinstance(state, ImageProcessingState)
    assert state.image.shape == image.shape
    assert state.valid_mask.shape == image.shape[1:]
    np.testing.assert_array_equal(state.image[0], image[0])
    np.testing.assert_array_equal(state.image[2], image[2])


def test_mask_contract_uses_spatial_mask_and_preserves_invalid_pixels():
    image = np.arange(48, dtype=np.float32).reshape(3, 4, 4)
    image[:, 1, 2] = np.nan
    steps = normalize_image_enhancement(
        _config(["clahe"], clahe={"kernel_size": 2, "clip_limit": 0.9})
    )

    state = apply_image_enhancement_state(image, steps)

    assert state.valid_mask.shape == (4, 4)
    assert not state.valid_mask[1, 2]
    assert np.isnan(state.image[:, 1, 2]).all()
    assert state.image.dtype == np.float32


def test_percentile_stretch_defaults_and_clips_values():
    image = np.array([[0.0, 1.0, 2.0, 3.0, 100.0]], dtype=np.float64)
    steps = normalize_image_enhancement(_config(["percentile_stretch"]))
    assert steps[0]["parameters"] == {"lower_percentile": 1.0, "upper_percentile": 99.0}
    result = apply_image_enhancement_state(image, steps).image
    assert result.dtype == np.float32
    assert result[0, 0] == 0.0
    assert result[0, -1] == 1.0
    assert np.all(np.diff(result[0]) >= 0)


def test_percentile_stretch_processes_each_band_and_preserves_nan():
    image = np.array([
        [[0, 1], [2, 3]],
        [[10, 20], [30, 40]],
        [[100, 100], [100, 100]],
    ], dtype=np.float32)
    image[0, 0, 0] = np.nan
    steps = normalize_image_enhancement(
        _config(["percentile_stretch"], percentile_stretch={"lower_percentile": 25, "upper_percentile": 75})
    )
    result = apply_image_enhancement_state(image, steps).image
    assert result.shape == image.shape
    assert np.isnan(result[0, 0, 0])
    np.testing.assert_allclose(result[1], [[10, 0], [0.5, 1]])
    assert result[2, 0, 0] == 100
    assert np.all(result[2, 0, 1:] == 0)
    assert np.all(result[2, 1] == 0)


@pytest.mark.parametrize("parameters", [
    {"lower_percentile": -1, "upper_percentile": 99},
    {"lower_percentile": 50, "upper_percentile": 50},
    {"lower_percentile": 1, "upper_percentile": 101},
])
def test_percentile_stretch_rejects_invalid_percentiles(parameters):
    with pytest.raises(ValueError, match="lower_percentile"):
        normalize_image_enhancement(_config(["percentile_stretch"], percentile_stretch=parameters))


def test_adaptive_gamma_matches_wang_equations():
    from PyImageTrack.DataProcessing.ImageEnhancement import apply_image_enhancement_state

    image = np.array([[0.0, 2.0, 4.0]], dtype=np.float32)
    config = {"image_enhancement": {"steps": ["adaptive_gamma"]}}
    step = normalize_image_enhancement(config)
    mean = 2.0
    maximum = 4.0
    gamma = 1.0 / (2.0 ** (mean / maximum))
    expected = maximum * (image / maximum) ** gamma
    result = apply_image_enhancement_state(image, step).image
    np.testing.assert_allclose(result, expected)


def test_adaptive_gamma_processes_channels_and_images_independently():
    from PyImageTrack.DataProcessing.ImageEnhancement import apply_image_enhancement_state

    image = np.array([
        [[0, 2], [4, 4]],
        [[0, 10], [20, 20]],
    ], dtype=np.uint16)
    steps = normalize_image_enhancement({"image_enhancement": {
        "steps": ["adaptive_gamma"], "adaptive_gamma": True,
    }})
    result = apply_image_enhancement_state(image, steps).image
    assert result.shape == image.shape
    assert np.issubdtype(result.dtype, np.floating)
    assert result[0, 0, 1] != result[1, 0, 1]
    np.testing.assert_allclose(result[:, 1, 1], image[:, 1, 1])


def test_adaptive_gamma_excludes_masked_and_invalid_pixels():
    from PyImageTrack.DataProcessing.ImageEnhancement import apply_image_enhancement_state

    image = np.array([[1.0, 3.0, np.nan, 100.0]])
    mask = np.array([[True, True, False, False]])
    steps = normalize_image_enhancement({"image_enhancement": {
        "steps": ["adaptive_gamma"], "adaptive_gamma": True,
    }})
    result = apply_image_enhancement_state(image, steps, valid_mask=mask).image
    gamma = 1.0 / (2.0 ** (2.0 / 3.0))
    np.testing.assert_allclose(result[0, :2], 3.0 * (image[0, :2] / 3.0) ** gamma)
    assert np.isnan(result[0, 2])
    assert result[0, 3] == 100.0


def test_adaptive_gamma_rejects_negative_valid_intensities():
    from PyImageTrack.DataProcessing.ImageEnhancement import apply_image_enhancement_state

    steps = normalize_image_enhancement({"image_enhancement": {
        "steps": ["adaptive_gamma"], "adaptive_gamma": True,
    }})
    with pytest.raises(ValueError, match="non-negative"):
        apply_image_enhancement_state(np.array([[-1.0, 2.0]]), steps)


def test_multi_scale_retinex_supports_multiband_masks_and_signed_values():
    image = np.array([
        [[-2.0, 0.0, 2.0], [1.0, 3.0, 5.0]],
        [[4.0, 4.0, 4.0], [4.0, np.nan, 4.0]],
    ])
    config = {"image_enhancement": {
        "steps": ["multi_scale_retinex"],
        "multi_scale_retinex": [[1.0, 2.0]],
    }}
    steps = normalize_image_enhancement(config)
    result = apply_image_enhancement_state(image, steps).image
    assert result.shape == image.shape
    assert result.dtype == np.float64
    assert np.isfinite(result[0]).all()
    assert np.isnan(result[1, 1, 1])
    np.testing.assert_allclose(result[1, 0, :], 0.0)


@pytest.mark.parametrize("sigmas", [[], [0], [-1], [float("nan")], ["bad"]])
def test_multi_scale_retinex_rejects_invalid_sigmas(sigmas):
    config = {"image_enhancement": {"steps": ["multi_scale_retinex"],
                                     "multi_scale_retinex": sigmas}}
    with pytest.raises(ValueError, match="gaussian_sigmas"):
        normalize_image_enhancement(config)


def test_difference_of_gaussians_uses_independent_scales_and_preserves_signed_output():
    from scipy.ndimage import gaussian_filter
    from PyImageTrack.DataProcessing.ImageEnhancement import apply_image_enhancement_state

    image = np.zeros((9, 9), dtype=np.float32)
    image[4, 4] = 10.0
    config = {"image_enhancement": {
        "steps": ["difference_of_gaussians"],
        "difference_of_gaussians": [1.0, 1.6],
    }}
    steps = normalize_image_enhancement(config)
    result = apply_image_enhancement_state(image, steps).image
    expected = gaussian_filter(image, 1.6, mode="reflect") - gaussian_filter(
        image, 1.0, mode="reflect"
    )
    np.testing.assert_allclose(result, expected)
    assert np.min(result) < 0
    assert result.dtype == np.float64


def test_unsharp_masking_matches_gaussian_reference_and_processes_bands_independently():
    from scipy.ndimage import gaussian_filter
    from PyImageTrack.DataProcessing.ImageEnhancement import (
        apply_image_enhancement_state,
        normalize_image_enhancement,
    )

    image = np.stack([np.arange(25, dtype=np.int16).reshape(5, 5), -np.ones((5, 5))])
    config = {"image_enhancement": {"steps": ["unsharp_masking"], "unsharp_masking": [1.0, 1.5]}}
    result = apply_image_enhancement_state(image, normalize_image_enhancement(config)).image
    expected = image.astype(np.float64) + 1.5 * (
        image.astype(np.float64) - np.stack([gaussian_filter(image[0], 1.0, mode="reflect"), image[1]])
    )
    np.testing.assert_allclose(result, expected)
    assert result.dtype == np.float64


def test_unsharp_masking_excludes_masked_extreme_values_and_validates_parameters():
    from PyImageTrack.DataProcessing.ImageEnhancement import (
        apply_image_enhancement_state,
        normalize_image_enhancement,
    )

    image = np.zeros((5, 5), dtype=float)
    image[2, 2] = 1e100
    valid_mask = np.ones_like(image, dtype=bool)
    valid_mask[2, 2] = False
    config = {"image_enhancement": {"steps": ["unsharp_masking"], "unsharp_masking": [1.0, 1.0]}}
    result = apply_image_enhancement_state(image, normalize_image_enhancement(config), valid_mask).image
    assert np.all(result[valid_mask] == 0)
    for parameters in ([0, 1], [1, -1], [np.inf, 1], [1, np.nan]):
        with pytest.raises(ValueError):
            normalize_image_enhancement({"image_enhancement": {"steps": ["unsharp_masking"], "unsharp_masking": parameters}})


def test_difference_of_gaussians_is_bandwise_and_mask_aware():
    from PyImageTrack.DataProcessing.ImageEnhancement import apply_image_enhancement_state

    image = np.ones((2, 7, 7), dtype=np.float32)
    image[1] *= 4
    valid_mask = np.ones((7, 7), dtype=bool)
    valid_mask[3, 3] = False
    image[:, 3, 3] = 1e9
    config = {"image_enhancement": {
        "steps": ["difference_of_gaussians"],
        "difference_of_gaussians": [1.0, 1.6],
    }}
    result = apply_image_enhancement_state(
        image, normalize_image_enhancement(config), valid_mask=valid_mask
    ).image
    np.testing.assert_allclose(result[:, valid_mask], 0.0, atol=1e-12)
    np.testing.assert_array_equal(result[:, 3, 3], image[:, 3, 3])


@pytest.mark.parametrize("sigmas", [[0, 1.6], [-1, 1.6], [1, 1], [1, np.inf]])
def test_difference_of_gaussians_rejects_invalid_sigmas(sigmas):
    config = {"image_enhancement": {
        "steps": ["difference_of_gaussians"],
        "difference_of_gaussians": sigmas,
    }}
    with pytest.raises(ValueError, match="Difference of Gaussians"):
        normalize_image_enhancement(config)


def test_naof_is_configured_without_parameters_and_preserves_shape():
    image = np.arange(75, dtype=np.uint16).reshape(3, 5, 5)
    config = {"image_enhancement": {"steps": ["naof"], "naof": []}}
    steps = normalize_image_enhancement(config)
    result = apply_image_enhancement_state(image, steps).image
    assert result.shape == image.shape
    assert result.dtype == np.float64
    assert np.isfinite(result).all()
    assert np.isfinite(result).all()


def test_naof_is_bandwise_signed_and_mask_aware():
    image = np.zeros((2, 7, 7), dtype=np.float32)
    image[0, 3, 3] = 10.0
    image[1, 3, 3] = -10.0
    valid_mask = np.ones((7, 7), dtype=bool)
    valid_mask[3, 3] = False
    image[:, 3, 3] = np.nan
    steps = normalize_image_enhancement({"image_enhancement": {"steps": ["naof"]}})
    result = apply_image_enhancement_state(image, steps, valid_mask=valid_mask).image
    assert result.shape == image.shape
    assert np.isnan(result[:, 3, 3]).all()
    neighborhood = result[:, 2:5, 2:5].copy()
    neighborhood[:, 1, 1] = 0.0
    assert np.all(neighborhood == 0.0)


def test_naof_constant_band_has_zero_response():
    image = np.full((9, 9), -4.5, dtype=np.float64)
    steps = normalize_image_enhancement({"image_enhancement": {"steps": ["naof"]}})
    result = apply_image_enhancement_state(image, steps).image
    np.testing.assert_allclose(result, 0.0)
