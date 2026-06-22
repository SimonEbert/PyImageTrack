import numpy as np
import skimage


def normalize_image_enhancement(image_enhancement_config=None, enabled=True):
    """
    Normalize image enhancement configuration to ordered steps.

    The TOML configuration must use a ``steps`` list and method-specific
    parameter lists, for example ``steps = ["clahe"]`` and
    ``clahe = [50, 0.9]``.

    Parameters
    ----------
    image_enhancement_config : dict, optional
        Image enhancement configuration from the TOML file.
    enabled : bool, optional
        If False, return an empty image enhancement sequence.

    Returns
    -------
    list[dict]
        Ordered enhancement step dictionaries.
    """
    if not enabled:
        return []

    if image_enhancement_config is None:
        return []

    if not isinstance(image_enhancement_config, dict):
        raise TypeError("Image enhancement configuration must be a dictionary.")

    if "steps" not in image_enhancement_config:
        if image_enhancement_config:
            raise ValueError("Image enhancement configuration must define a steps list.")
        return []

    steps = image_enhancement_config["steps"]
    if not isinstance(steps, list):
        raise TypeError("image_enhancement.steps must be a list of method names.")

    normalized_steps = [
        _normalize_step_from_method_name(step, image_enhancement_config)
        for step in steps
    ]
    return [step for step in normalized_steps if step.get("type") != "none"]


def apply_image_enhancement(image_matrix, image_enhancement):
    """
    Apply an ordered sequence of image enhancement steps to an image matrix.

    Parameters
    ----------
    image_matrix : np.ndarray
        Input image matrix.
    image_enhancement : list[dict]
        Ordered enhancement step dictionaries.

    Returns
    -------
    np.ndarray
        Enhanced image matrix.
    """
    enhanced_image = image_matrix
    for step in image_enhancement or []:
        enhanced_image = apply_image_enhancement_step(enhanced_image, step)
    return enhanced_image


def apply_image_enhancement_step(image_matrix, enhancement_step):
    """
    Apply one image enhancement step.

    Parameters
    ----------
    image_matrix : np.ndarray
        Input image matrix.
    enhancement_step : dict
        Enhancement step dictionary with at least a ``type`` key.

    Returns
    -------
    np.ndarray
        Enhanced image matrix.
    """
    step = _normalize_step(enhancement_step)
    enhancement_type = step.get("type", "none")

    if enhancement_type == "none":
        return image_matrix
    if enhancement_type == "clahe":
        return _apply_clahe(
            image_matrix,
            kernel_size=step.get("kernel_size", 50),
            clip_limit=step.get("clip_limit", 0.9),
        )

    raise ValueError(f"Unsupported image enhancement type: {enhancement_type}")


def _normalize_step(step):
    """
    Normalize a single image enhancement step dictionary.

    Parameters
    ----------
    step : dict
        Enhancement step configuration.

    Returns
    -------
    dict
        Normalized copy of the enhancement step.
    """
    if step is None:
        return {"type": "none"}
    if not isinstance(step, dict):
        raise TypeError("Each image enhancement step must be a dictionary.")

    normalized_step = dict(step)
    enhancement_type = normalized_step.get("type", "none")
    normalized_step["type"] = str(enhancement_type).lower() if enhancement_type else "none"
    return normalized_step


def _normalize_step_from_method_name(method_name, image_enhancement_config):
    """
    Normalize one configured image enhancement method.

    Parameters
    ----------
    method_name : str
        Enhancement method name from ``image_enhancement.steps``.
    image_enhancement_config : dict
        Full image enhancement configuration.

    Returns
    -------
    dict
        Normalized enhancement step.
    """
    if not isinstance(method_name, str):
        raise TypeError("Each image enhancement step must be a method name string.")

    method_name = method_name.lower()
    if method_name == "none":
        return {"type": "none"}
    if method_name == "clahe":
        return _normalize_clahe_parameters(image_enhancement_config.get("clahe"))

    raise ValueError(f"Unsupported image enhancement type: {method_name}")


def _normalize_clahe_parameters(parameters):
    """
    Normalize CLAHE parameter list.

    Parameters
    ----------
    parameters : list
        CLAHE parameters in the order ``[kernel_size, clip_limit]``.

    Returns
    -------
    dict
        Normalized CLAHE enhancement step.
    """
    if parameters is None:
        raise ValueError("image_enhancement.clahe must define [kernel_size, clip_limit].")
    if not isinstance(parameters, list) or len(parameters) != 2:
        raise TypeError("image_enhancement.clahe must be [kernel_size, clip_limit].")

    return {
        "type": "clahe",
        "kernel_size": parameters[0],
        "clip_limit": parameters[1],
    }


def _apply_clahe(image_matrix, kernel_size, clip_limit):
    """
    Apply Contrast Limited Adaptive Histogram Equalization.

    Parameters
    ----------
    image_matrix : np.ndarray
        Input image matrix.
    kernel_size : tuple or int
        Size of the contextual regions used for local contrast enhancement.
    clip_limit : float
        Contrast limiting threshold.

    Returns
    -------
    np.ndarray
        Contrast-enhanced image matrix.
    """
    return skimage.exposure.equalize_adapthist(
        image=image_matrix.astype(np.uint16),
        kernel_size=kernel_size,
        clip_limit=clip_limit,
    )
