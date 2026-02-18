# PyImageTrack Documentation Review Report

**Date**: 2026-02-18  
**Reviewer**: Architect Mode  
**Documentation Version**: 2026-01-26 (from pyimagetrack_documentation.md)

---

## Executive Summary

This review analyzes the PyImageTrack documentation for accuracy, completeness, and structural organization. The documentation is generally well-structured and covers the main functionality, but there are several areas that need attention including duplicate content, missing function documentation, and outdated parameter descriptions.

**Overall Assessment**: The documentation is functional but requires updates to reflect the current codebase state.

---

## Critical Issues

### 1. Duplicate Content (Lines 50-55 and 96-101)

**Location**: [`pyimagetrack_documentation.md`](docs/pyimagetrack_documentation.md:50-55) and [`pyimagetrack_documentation.md`](docs/pyimagetrack_documentation.md:96-101)

**Issue**: The "Downsampling" section appears twice with identical content.

**Recommendation**: Remove the duplicate section (lines 96-101).

---

## Missing Documentation

### 2. LoD Cache Functions

**Location**: [`Cache.py`](src/PyImageTrack/Cache.py:111-171)

**Issue**: The functions `lod_cache_paths()`, `save_lod_cache()`, and `load_lod_cache()` exist in the code but are not documented, even though they are imported in [`run_pipeline.py`](src/PyImageTrack/run_pipeline.py:39-40).

**Recommendation**: Add documentation for these functions in the "Module: Cache.py" section.

### 3. Undistortion Function

**Location**: [`ImagePreprocessing.py`](src/PyImageTrack/DataProcessing/ImagePreprocessing.py:18-77)

**Issue**: The `undistort_camera_image()` function exists and is referenced in the config options (`[no_georef] undistort_image`), but is not documented.

**Recommendation**: Add documentation for this function in the "Module: DataProcessing/ImagePreprocessing.py" section.

### 4. ImagePair Methods

**Location**: [`ImagePair.py`](src/PyImageTrack/ImageTracking/ImagePair.py)

**Issue**: Several important methods are not documented:
- `track_lod_points()` (lines 616-665)
- `compute_truecolor_aligned_from_control_points()` (lines 390-473)
- `load_results()` (lines 1190-1199)
- `select_image_channels()` (lines 126-131)
- `load_images_from_matrix_and_transform()` (lines 290-340)

**Recommendation**: Add documentation for these methods in the "Module: ImageTracking/ImagePair.py" section.

### 5. Internal Helper Functions

**Location**: [`run_pipeline.py`](src/PyImageTrack/run_pipeline.py) and [`Utils.py`](src/PyImageTrack/Utils.py)

**Issue**: Several internal helper functions are not documented:
- `_resolve_config_path()` (run_pipeline.py:44-52)
- `_resolve_path()` (run_pipeline.py:85-91)
- `_crs_label()` (run_pipeline.py:94-95)
- `_normalize_crs()` (run_pipeline.py:98-101)
- `_resolve_common_crs()` (run_pipeline.py:104-134)
- `_recompute_lod_from_points()` (run_pipeline.py:159-170)
- `_successive_pairs()` (Utils.py:72-73)
- `_extract_lead()` (Utils.py:249-255)
- `_resolve_csv_token_to_id()` (Utils.py:257-265)

**Recommendation**: Consider whether these internal functions need documentation. If they are part of the public API, document them. If they are purely internal, add a note that they are internal helpers.

### 6. Tracking Functions

**Location**: [`TrackMovement.py`](src/PyImageTrack/ImageTracking/TrackMovement.py)

**Issue**: Several tracking functions are not documented:
- `track_cell_cc()` (lines 20-123)
- `track_cell_lsm()` (lines 150-323)
- `track_cell_lsm_parallelized()` (lines 325-394)
- `move_indices_from_transformation_matrix()` (lines 125-148)

**Recommendation**: Add documentation for these functions in the "Module: ImageTracking/TrackMovement.py" section.

### 7. TrackingResults Class

**Location**: [`TrackingResults.py`](src/PyImageTrack/ImageTracking/TrackingResults.py:4-23)

**Issue**: The `TrackingResults` class is referenced in the documentation but the class itself is not documented.

**Recommendation**: Add documentation for the `TrackingResults` class.

### 8. Filter Functions

**Location**: [`DataPostprocessing.py`](src/PyImageTrack/DataProcessing/DataPostprocessing.py)

**Issue**: The filter functions are listed but not fully documented:
- `filter_outliers_movement_bearing_difference()` (lines 92-161)
- `filter_outliers_movement_bearing_standard_deviation()` (lines 163-231)
- `filter_outliers_movement_rate_difference()` (lines 233-305)
- `filter_outliers_movement_rate_standard_deviation()` (lines 307-379)
- `filter_outliers_full()` (lines 381-391)

**Recommendation**: Add full documentation for these functions including all parameters.

### 9. Geometry Functions

**Location**: [`HandleGeometries.py`](src/PyImageTrack/CreateGeometries/HandleGeometries.py)

**Issue**: Several geometry functions are not documented:
- `get_submatrix_symmetric()` (lines 10-42)
- `grid_points_on_polygon_by_distance()` (lines 44-92)
- `random_points_on_polygon_by_number()` (lines 94-109)
- `get_raster_indices_from_points()` (lines 111-135)
- `crop_images_to_intersection()` (lines 137-168)
- `georeference_tracked_points()` (lines 170-214)
- `circular_std_deg()` (lines 216-231)
- `get_submatrix_rect_from_extents()` (lines 233-??)

**Recommendation**: Add documentation for these functions in the "Module: CreateGeometries/HandleGeometries.py" section.

### 10. Plotting Functions

**Location**: [`MakePlots.py`](src/PyImageTrack/Plots/MakePlots.py)

**Issue**: The plotting functions are documented but may have missing parameters:
- `plot_raster_and_geometry()` (lines 7-32)
- `plot_movement_of_points()` (lines 34-173)
- `plot_movement_of_points_with_valid_mask()` (lines 175-211)
- `plot_distribution_of_point_movement()` (lines 213-??)

**Recommendation**: Verify that all parameters are documented for these functions.

---

## Parameter Inconsistencies

### 11. Missing `clip_limit` Parameter

**Location**: [`ImagePreprocessing.py`](src/PyImageTrack/DataProcessing/ImagePreprocessing.py:4-7) vs [`pyimagetrack_documentation.md`](docs/pyimagetrack_documentation.md:1144-1153)

**Issue**: The `equalize_adapthist_images()` function has a `clip_limit` parameter that is not documented.

**Recommendation**: Add the `clip_limit` parameter to the function documentation.

### 12. Missing `displacement_column_name` Parameter

**Location**: [`DataPostprocessing.py`](src/PyImageTrack/DataProcessing/DataPostprocessing.py:68-89) vs [`pyimagetrack_documentation.md`](docs/pyimagetrack_documentation.md:1187-1196)

**Issue**: The `filter_lod_points()` function has a `displacement_column_name` parameter that is not documented.

**Recommendation**: Add the `displacement_column_name` parameter to the function documentation.

### 13. Missing `displacement_column_name` Parameter in filter_outliers_full

**Location**: [`DataPostprocessing.py`](src/PyImageTrack/DataProcessing/DataPostprocessing.py:381-391) vs [`pyimagetrack_documentation.md`](docs/pyimagetrack_documentation.md:1210-1211)

**Issue**: The `filter_outliers_full()` function has a `displacement_column_name` parameter that is not documented.

**Recommendation**: Add the `displacement_column_name` parameter to the function documentation.

---

## Documentation Structure Issues

### 14. Module Organization

**Issue**: The documentation is organized by module, but some modules are missing or incomplete:
- `ImageTracking/AlignImages.py` - Only `align_images_lsm_scarce()` is documented
- `ImageTracking/TrackingResults.py` - Class is not documented
- `ImageTracking/TrackMovement.py` - Only `track_movement_lsm()` is documented
- `CreateGeometries/HandleGeometries.py` - Only some functions are documented
- `CreateGeometries/DepthImageConversion.py` - Functions are documented but may need updates
- `DataProcessing/ImagePreprocessing.py` - Only `equalize_adapthist_images()` is documented
- `DataProcessing/DataPostprocessing.py` - Only some functions are documented
- `Plots/MakePlots.py` - Functions are documented but may need updates

**Recommendation**: Complete the documentation for all modules and functions.

### 15. Parameter Classes

**Issue**: The parameter classes (`AlignmentParameters`, `TrackingParameters`, `FilterParameters`) are documented but the documentation could be more detailed about the fields and their meanings.

**Recommendation**: Add more detailed documentation for the parameter classes, including:
- Description of each field
- Valid ranges or values
- Default values
- How they affect the tracking process

---

## Configuration Documentation

### 16. Config File Examples

**Location**: [`pyimagetrack_documentation.md`](docs/pyimagetrack_documentation.md:30-48)

**Issue**: The documentation mentions config templates (`configs/drone_hs.toml`, `configs/smoketest_drone_hs.toml`, `configs/timelapse_fake_ortho.toml`) but only `configs/example_config.toml` exists in the repository.

**Recommendation**: Update the documentation to reflect the actual config files that exist, or create the missing config files.

### 17. Adaptive Tracking Window

**Location**: [`pyimagetrack_documentation.md`](docs/pyimagetrack_documentation.md:46) and [`example_config.toml`](configs/example_config.toml:56-60)

**Issue**: The `[adaptive_tracking_window]` section is mentioned in the documentation but the parameters are not fully explained.

**Recommendation**: Add more detailed documentation for the adaptive tracking window feature, including:
- What it does
- When to use it
- How the parameters affect the tracking
- Examples of typical values

---

## Date and Version Information

### 18. Documentation Date

**Location**: [`pyimagetrack_documentation.md`](docs/pyimagetrack_documentation.md:2)

**Issue**: The documentation shows "2026-01-26" but we are currently in 2026-02-18, so the documentation is about 3 weeks old.

**Recommendation**: Update the documentation date to reflect the current state.

### 19. Version Information

**Location**: [`pyproject.toml`](pyproject.toml:7)

**Issue**: The package version is "0.2" but there is no version information in the documentation.

**Recommendation**: Add version information to the documentation header.

---

## Installation Guide Review

### 20. Python Version

**Location**: [`absolute_beginner_installation.md`](docs/absolute_beginner_installation.md:45) and [`pyproject.toml`](pyproject.toml:17)

**Issue**: The installation guide mentions "Python 3.11 (64-bit)" but the `pyproject.toml` specifies `requires-python = ">=3.8"`.

**Recommendation**: Update the installation guide to reflect the minimum Python version (3.8) or update the `pyproject.toml` to require 3.11 if that's the intended minimum.

### 21. Windows PowerShell Execution Policy

**Location**: [`absolute_beginner_installation.md`](docs/absolute_beginner_installation.md:66-69)

**Issue**: The instruction to set the execution policy is good, but it could be more specific about when this is needed.

**Recommendation**: Add a note that this is only needed if the activation script is blocked.

---

## README Review

### 22. Feature List

**Location**: [`README.md`](README.md:6-18)

**Issue**: The feature list is comprehensive but could be more detailed about the 3D displacement feature.

**Recommendation**: Add more information about the 3D displacement feature, including:
- When to use it
- What depth images are required
- How to configure it

### 23. Performance Note

**Location**: [`README.md`](README.md:57)

**Issue**: The note about performance being better on Linux than Windows is helpful but could be more specific.

**Recommendation**: Add more details about the performance difference and why it exists.

---

## Recommendations Summary

### High Priority

1. **Remove duplicate content** (Issue #1)
2. **Add missing LoD cache functions documentation** (Issue #2)
3. **Add undistortion function documentation** (Issue #3)
4. **Update documentation date** (Issue #18)
5. **Fix Python version inconsistency** (Issue #20)

### Medium Priority

6. **Add missing ImagePair methods documentation** (Issue #4)
7. **Add missing tracking functions documentation** (Issue #6)
8. **Add TrackingResults class documentation** (Issue #7)
9. **Add missing filter functions documentation** (Issue #8)
10. **Add missing geometry functions documentation** (Issue #9)
11. **Fix parameter inconsistencies** (Issues #11-13)
12. **Complete module documentation** (Issue #14)
13. **Update config file examples** (Issue #16)
14. **Add adaptive tracking window documentation** (Issue #17)

### Low Priority

15. **Add internal helper function documentation** (Issue #5)
16. **Enhance parameter classes documentation** (Issue #15)
17. **Add version information** (Issue #19)
18. **Improve installation guide** (Issue #21)
19. **Enhance feature list** (Issue #22)
20. **Add performance details** (Issue #23)

---

## Conclusion

The PyImageTrack documentation is generally well-structured and covers the main functionality of the package. However, there are several areas that need attention:

1. **Duplicate content** should be removed
2. **Missing function documentation** should be added, especially for functions that are part of the public API
3. **Parameter inconsistencies** should be fixed
4. **Module documentation** should be completed
5. **Configuration documentation** should be updated to reflect the actual state of the codebase

The documentation would benefit from a systematic review to ensure that all public functions and classes are documented with accurate parameter descriptions and return values.

---

## Next Steps

1. Review this report and prioritize the issues
2. Create a plan to address the high-priority issues
3. Update the documentation incrementally, starting with the most critical issues
4. Consider implementing automated documentation generation tools (e.g., Sphinx) to help keep the documentation in sync with the code
5. Establish a process for updating the documentation when code changes are made