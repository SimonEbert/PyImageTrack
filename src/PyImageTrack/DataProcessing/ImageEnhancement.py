"""Configuration and execution helpers for image enhancement steps."""

from collections import defaultdict


# The category is an implementation detail. Users select method names only.
MASK_METHODS = set()
FILTER_METHODS = {"clahe"}
SUPPORTED_METHODS = MASK_METHODS | FILTER_METHODS


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
    """Apply normalized enhancement steps in their configured order."""
    enhanced_image = image_matrix
    for step in image_enhancement or []:
        enhanced_image = apply_image_enhancement_step(enhanced_image, step)
    return enhanced_image


def apply_image_enhancement_step(image_matrix, enhancement_step):
    """Dispatch one normalized enhancement step."""
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
        return _apply_clahe(
            image_matrix,
            kernel_size=parameters["kernel_size"],
            clip_limit=parameters["clip_limit"],
        )
    _invalid_enhancement_config(method_name)


def _apply_clahe(image_matrix, kernel_size, clip_limit):
    """Apply Contrast Limited Adaptive Histogram Equalization."""
    import numpy as np
    import skimage

    return skimage.exposure.equalize_adapthist(
        image=np.asarray(image_matrix).astype(np.uint16),
        kernel_size=kernel_size,
        clip_limit=clip_limit,
    )
