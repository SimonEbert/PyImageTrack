"""Configuration and execution helpers for image enhancement steps."""

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.warp import Resampling, reproject


# The category is an implementation detail. Users select method names only.
MASK_METHODS = {"mask"}
FILTER_METHODS = {"clahe", "percentile_stretch"}
SUPPORTED_METHODS = MASK_METHODS | FILTER_METHODS


@dataclass
class ImageProcessingState:
    """Image data and the cumulative spatial validity mask."""

    image: np.ndarray
    valid_mask: np.ndarray


def _initial_processing_state(image_matrix, valid_mask=None):
    image = np.asarray(image_matrix)
    if image.ndim not in (2, 3):
        raise ValueError("Image enhancement supports only 2D or 3D image arrays.")
    finite = np.isfinite(image)
    finite_mask = finite if image.ndim == 2 else np.all(finite, axis=0)
    if valid_mask is None:
        valid_mask = finite_mask
    else:
        if valid_mask.shape != image.shape[-2:]:
            raise ValueError("Image enhancement valid mask must match the image grid.")
        valid_mask = np.asarray(valid_mask, dtype=bool) & finite_mask
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
    steps = section.get("image_enhancement_steps", section.get("steps", []))
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
        if method_name == "mask":
            parameters = _mask_parameters(section)
        else:
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
    table = config.get(f"image_enhancement_{method_name}")
    if table is None and method_name == "clahe":
        compact = section.get("clahe")
        if isinstance(compact, list) and len(compact) >= 2:
            return {
                "kernel_size": compact[0],
                "clip_limit": compact[1],
                **({"bands": compact[2]} if len(compact) > 2 else {}),
            }
    if table is None and method_name == "percentile_stretch":
        compact = section.get("percentile_stretch")
        if compact is None:
            return {"lower_percentile": 1.0, "upper_percentile": 99.0}
        if isinstance(compact, list) and len(compact) == 2:
            parameters = {
                "lower_percentile": compact[0],
                "upper_percentile": compact[1],
            }
            _validate_percentile_parameters(parameters)
            return parameters
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
    if method_name == "percentile_stretch":
        parameters.setdefault("lower_percentile", 1.0)
        parameters.setdefault("upper_percentile", 99.0)
        _validate_percentile_parameters(parameters)
    return parameters


def _validate_percentile_parameters(parameters):
    """Validate percentile-stretch bounds."""
    try:
        lower = float(parameters["lower_percentile"])
        upper = float(parameters["upper_percentile"])
    except (KeyError, TypeError, ValueError):
        raise ValueError(
            "Percentile stretch requires numeric lower_percentile and upper_percentile values."
        ) from None
    if not 0 <= lower < upper <= 100:
        raise ValueError(
            "Percentile stretch requires 0 <= lower_percentile < upper_percentile <= 100."
        )
    parameters["lower_percentile"] = lower
    parameters["upper_percentile"] = upper


def mask_for_grid(mask_path, shape, transform, crs, mode="include"):
    """Return a boolean mask where True denotes an allowed image pixel."""
    if mode not in {"include", "exclude"}:
        raise ValueError("Image mask mode must be either 'include' or 'exclude'.")
    path = Path(mask_path)
    if not path.exists():
        raise FileNotFoundError(f"Image mask file does not exist: {path}")

    if path.suffix.lower() in {".shp", ".gpkg", ".geojson", ".json"}:
        geometries = gpd.read_file(path)
        if geometries.empty:
            raise ValueError(f"Image mask file contains no geometries: {path}")
        if geometries.crs is not None and crs is not None:
            geometries = geometries.to_crs(crs)
        elif geometries.crs != crs:
            raise ValueError("Vector image masks and images must have compatible CRS values.")
        result = rasterize(
            ((geometry, 1) for geometry in geometries.geometry if geometry is not None),
            out_shape=shape, transform=transform, fill=0, dtype="uint8",
        ).astype(bool)
        return result if mode == "include" else ~result

    with rasterio.open(path) as source:
        if source.crs is None or crs is None:
            if source.shape != shape or source.transform != transform:
                raise ValueError(
                    "A raster image mask without CRS must have the same shape and transform as the image."
                )
            result = source.read(1) != 0
        else:
            destination = np.zeros(shape, dtype=np.uint8)
            reproject(
                source=source.read(1), destination=destination,
                src_transform=source.transform, src_crs=source.crs,
                dst_transform=transform, dst_crs=crs,
                resampling=Resampling.nearest, src_nodata=0, dst_nodata=0,
            )
            result = destination.astype(bool)
    return result if mode == "include" else ~result


def _mask_parameters(section):
    """Normalize the compact mask configuration."""
    value = section.get("mask")
    if not isinstance(value, list) or len(value) != 3:
        _invalid_enhancement_config("mask")
    mask_types, nodata_values, custom_mode = value
    if not isinstance(mask_types, list) or not isinstance(nodata_values, list):
        _invalid_enhancement_config("mask")
    if not set(mask_types).issubset({"nodata", "custom"}):
        _invalid_enhancement_config("mask")
    if custom_mode not in {"include", "exclude", "none"}:
        _invalid_enhancement_config("mask")
    return {
        "types": mask_types,
        "nodata_values": nodata_values,
        "custom_mode": custom_mode,
    }


def _invalid_enhancement_config(method_name):
    """Raise the single public error used for invalid enhancement config."""
    message = (
        f"Invalid image enhancement configuration for '{method_name}': "
        "method or parameters are unknown or incomplete."
    )
    raise ValueError(message)


def apply_image_enhancement_state(image_matrix, image_enhancement, valid_mask=None):
    """Apply normalized steps and return the complete processing state."""
    state = _initial_processing_state(image_matrix, valid_mask=valid_mask)
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
    if method_name == "percentile_stretch":
        parameters = enhancement_step.get("parameters", {})
        _validate_percentile_parameters(parameters)
        state.image = _apply_percentile_stretch(
            state.image, state.valid_mask,
            parameters["lower_percentile"], parameters["upper_percentile"],
        )
        return state
    if method_name == "mask":
        return state
    _invalid_enhancement_config(method_name)


def _apply_clahe(image_matrix, valid_mask, kernel_size, clip_limit, bands=None):
    """Apply Contrast Limited Adaptive Histogram Equalization."""
    import skimage

    if image_matrix.ndim == 2:
        if bands not in (None, [], [0]):
            _invalid_enhancement_config("clahe")
        return _apply_clahe_band(image_matrix, valid_mask, kernel_size, clip_limit)

    band_indices = _selected_band_indices(image_matrix, bands)
    if any(band < 0 or band >= image_matrix.shape[0] for band in band_indices):
        _invalid_enhancement_config("clahe")

    result = np.array(image_matrix, dtype=np.float32, copy=True)
    for band_index in band_indices:
        band_image = image_matrix[band_index]
        band_valid = np.isfinite(band_image) & valid_mask
        result[band_index] = _apply_clahe_band(
            band_image, band_valid, kernel_size, clip_limit
        )
    return result


def _apply_percentile_stretch(image_matrix, valid_mask, lower_percentile, upper_percentile):
    """Stretch each image band independently to the unit interval."""
    from skimage.exposure import rescale_intensity

    image = np.asarray(image_matrix)
    result = image.astype(np.float32, copy=True)
    bands = [image] if image.ndim == 2 else image
    for index, band in enumerate(bands):
        valid = np.isfinite(band) & valid_mask
        values = band[valid]
        if values.size == 0:
            continue
        low, high = np.percentile(values, [lower_percentile, upper_percentile])
        if high == low:
            stretched = np.zeros(band.shape, dtype=np.float32)
        else:
            calculation = np.where(valid, band, low)
            stretched = rescale_intensity(
                calculation, in_range=(low, high), out_range=(0.0, 1.0)
            ).astype(np.float32, copy=False)
        if image.ndim == 2:
            result[valid] = stretched[valid]
        else:
            result[index][valid] = stretched[valid]
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


def _apply_clahe_band(image, valid_mask, kernel_size, clip_limit):
    """Apply CLAHE to one float image while preserving its valid value range."""
    import skimage

    result = np.asarray(image, dtype=np.float32).copy()
    valid_values = result[valid_mask]
    if valid_values.size == 0:
        return result

    minimum = np.min(valid_values)
    maximum = np.max(valid_values)
    if maximum <= minimum:
        return result

    fill_value = np.median(valid_values)
    calculation_image = np.where(valid_mask, result, fill_value)
    normalized = (calculation_image - minimum) / (maximum - minimum)
    enhanced = skimage.exposure.equalize_adapthist(
        image=normalized.astype(np.float32),
        kernel_size=kernel_size,
        clip_limit=clip_limit,
    )
    result[valid_mask] = (
        enhanced[valid_mask] * (maximum - minimum) + minimum
    ).astype(np.float32)
    return result
