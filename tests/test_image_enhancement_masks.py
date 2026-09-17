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
