"""Configuration and execution helpers for image enhancement steps."""

from collections import defaultdict
from dataclasses import dataclass

import numpy as np


# The category is an implementation detail. Users select method names only.
MASK_METHODS = set()
FILTER_METHODS = {"clahe"}
SUPPORTED_METHODS = MASK_METHODS | FILTER_METHODS


@dataclass
class ImageProcessingState:
    """Image data and the cumulative spatial validity mask."""

    image: np.ndarray
    valid_mask: np.ndarray


def _initial_processing_state(image_matrix):
    image = np.asarray(image_matrix)
    if image.ndim not in (2, 3):
        raise ValueError("Image enhancement supports only 2D or 3D image arrays.")
    return ImageProcessingState(
        image=image,
        valid_mask=np.ones(image.shape[-2:], dtype=bool),
    )


def normalize_image_enhancement(config=None, enabled=True):
    """Normalize the ordered image-enhancement configuration.

    ``config`` may be the complete TOML configuration or the
    ``image_enhancement`` section. The new format uses
    ``image_enhancement_steps`` plus one array-of-tables configuration for
    each method, for example ``[[image_enhancement_clahe]]``.
    """
    if not enabled:
        return []
    if config is None:
        return []
    if not isinstance(config, dict):
        _invalid_enhancement_config("unknown")

    section = config.get("image_enhancement", config)
    if not isinstance(section, dict):
        _invalid_enhancement_config("unknown")
    steps = section.get("image_enhancement_steps", [])
    if not isinstance(steps, list):
        _invalid_enhancement_config("unknown")

    normalized = []
    occurrences = defaultdict(int)
    for method_name in steps:
        if not isinstance(method_name, str):
            _invalid_enhancement_config("unknown")
        method_name = method_name.lower()
        if method_name not in SUPPORTED_METHODS:
            _invalid_enhancement_config(method_name)
        parameters = _method_parameters(config, section, method_name, occurrences[method_name])
        occurrences[method_name] += 1
        category = "mask" if method_name in MASK_METHODS else "filter"
        normalized.append({
            "category": category,
            "type": method_name,
            "parameters": parameters,
        })
    return normalized


def _method_parameters(config, section, method_name, occurrence):
    """Read one method's parameter table."""
    keys = (f"image_enhancement_{method_name}", method_name)
    table = next((config[key] for key in keys if key in config), None)
    if table is None:
        table = next((section[key] for key in keys if key in section), None)
    if table is None:
        return {}
    if not isinstance(table, list):
        _invalid_enhancement_config(method_name)
    if occurrence >= len(table):
        _invalid_enhancement_config(method_name)
    parameters = table[occurrence]
    if not isinstance(parameters, dict):
        _invalid_enhancement_config(method_name)
    parameters = dict(parameters)
    if method_name == "clahe" and not {"kernel_size", "clip_limit"}.issubset(parameters):
        _invalid_enhancement_config(method_name)
    return parameters


def _invalid_enhancement_config(method_name):
    """Raise the single public error used for invalid enhancement config."""
    message = (
        f"Invalid image enhancement configuration for '{method_name}': "
        "method or parameters are unknown or incomplete."
    )
    raise ValueError(message)


def apply_image_enhancement(image_matrix, image_enhancement):
    """Apply normalized enhancement steps and return the processed image."""
    state = _initial_processing_state(image_matrix)
    for step in image_enhancement or []:
        state = apply_image_enhancement_step(state, step)
    return state.image


def apply_image_enhancement_step(state, enhancement_step):
    """Dispatch one normalized enhancement step on a processing state."""
    if not isinstance(state, ImageProcessingState):
        raise TypeError("Image enhancement steps require an ImageProcessingState.")
    if not isinstance(enhancement_step, dict):
        _invalid_enhancement_config("unknown")
    category = enhancement_step.get("category")
    method_name = enhancement_step.get("type")
    if category not in {"mask", "filter"}:
        _invalid_enhancement_config(method_name or "unknown")
    if method_name not in SUPPORTED_METHODS:
        _invalid_enhancement_config(method_name or "unknown")
    if method_name == "clahe":
        parameters = enhancement_step.get("parameters", {})
        state.image = _apply_clahe(
            state.image,
            valid_mask=state.valid_mask,
            kernel_size=parameters["kernel_size"],
            clip_limit=parameters["clip_limit"],
        )
        return state
    _invalid_enhancement_config(method_name)


def _apply_clahe(image_matrix, valid_mask, kernel_size, clip_limit):
    """Apply Contrast Limited Adaptive Histogram Equalization."""
    import skimage

    if image_matrix.ndim == 2:
        result = skimage.exposure.equalize_adapthist(
            image=np.asarray(image_matrix).astype(np.uint16),
            kernel_size=kernel_size,
            clip_limit=clip_limit,
        )
        result[~valid_mask] = image_matrix[~valid_mask]
        return result

    result = np.array(image_matrix, dtype=float, copy=True)
    for band_index in range(image_matrix.shape[0]):
        band = skimage.exposure.equalize_adapthist(
            image=np.asarray(image_matrix[band_index]).astype(np.uint16),
            kernel_size=kernel_size,
            clip_limit=clip_limit,
        )
        result[band_index] = np.where(valid_mask, band, image_matrix[band_index])
    return result
