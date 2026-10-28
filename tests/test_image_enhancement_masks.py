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
