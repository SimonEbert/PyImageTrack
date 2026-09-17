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
    finite = np.isfinite(image)
    valid_mask = finite if image.ndim == 2 else np.all(finite, axis=0)
    return ImageProcessingState(
        image=image,
        valid_mask=valid_mask,
    )


def normalize_image_enhancement(config=None, enabled=True):
    """Normalize the ordered image-enhancement configuration.

    ``config`` is the complete TOML configuration. The new format uses
    ``image_enhancement_steps`` plus one array-of-tables configuration for
    each method, for example ``[[image_enhancement_clahe]]``.
    """
    if not enabled:
        return []
    if config is None:
        return []
    if not isinstance(config, dict):
        _invalid_enhancement_config("unknown")

    section = config.get("image_enhancement")
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
        parameters = _method_parameters(config, method_name, occurrences[method_name])
        occurrences[method_name] += 1
        category = "mask" if method_name in MASK_METHODS else "filter"
        normalized.append({
            "category": category,
            "type": method_name,
            "parameters": parameters,
        })
    return normalized


def _method_parameters(config, method_name, occurrence):
    """Read one method's parameter table."""
    table = config.get(f"image_enhancement_{method_name}")
    if table is None:
        _invalid_enhancement_config(method_name)
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


def apply_image_enhancement_state(image_matrix, image_enhancement):
    """Apply normalized steps and return the complete processing state."""
    state = _initial_processing_state(image_matrix)
    for step in image_enhancement or []:
        state = apply_image_enhancement_step(state, step)
    return state


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
            bands=parameters.get("bands"),
        )
        return state
    _invalid_enhancement_config(method_name)


def _apply_clahe(image_matrix, valid_mask, kernel_size, clip_limit, bands=None):
    """Apply Contrast Limited Adaptive Histogram Equalization."""
    import skimage

    if image_matrix.ndim == 2:
        if bands not in (None, [], [0]):
            _invalid_enhancement_config("clahe")
        valid_values = image_matrix[valid_mask]
        fill_value = np.nanmedian(valid_values) if valid_values.size else 0.0
        calculation_image = np.where(valid_mask, image_matrix, fill_value)
        result = skimage.exposure.equalize_adapthist(
            image=np.asarray(calculation_image).astype(np.uint16),
            kernel_size=kernel_size,
            clip_limit=clip_limit,
        )
        result[~valid_mask] = image_matrix[~valid_mask]
        return result

    band_indices = _selected_band_indices(image_matrix, bands)
    if any(band < 0 or band >= image_matrix.shape[0] for band in band_indices):
        _invalid_enhancement_config("clahe")

    result = np.array(image_matrix, dtype=float, copy=True)
    for band_index in band_indices:
        band_image = image_matrix[band_index]
        band_valid = np.isfinite(band_image) & valid_mask
        valid_values = band_image[band_valid]
        fill_value = np.nanmedian(valid_values) if valid_values.size else 0.0
        calculation_band = np.where(band_valid, band_image, fill_value)
        band = skimage.exposure.equalize_adapthist(
            image=np.asarray(calculation_band).astype(np.uint16),
            kernel_size=kernel_size,
            clip_limit=clip_limit,
        )
        result[band_index] = np.where(valid_mask, band, image_matrix[band_index])
    return result


def _selected_band_indices(image_matrix, bands):
    """Return validated zero-based band indices for a bands-first image."""
    try:
        indices = list(range(image_matrix.shape[0])) if bands is None else [int(band) for band in bands]
    except (TypeError, ValueError):
        _invalid_enhancement_config("clahe")
    if any(index < 0 or index >= image_matrix.shape[0] for index in indices):
        _invalid_enhancement_config("clahe")
    return indices
