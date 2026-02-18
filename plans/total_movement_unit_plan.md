# Plan: Add Total Movement Unit Option

## Overview
Add a configuration option to choose between "movement per year" and "total movement" (raw displacement without dividing by years_between) for all output files and internal calculations.

## Motivation
For one-time events like landslides, the unit "m/yr" is not meaningful. Using total movement makes results easier to interpret.

## Design Decisions

### 1. Configuration Option
Add a new section `[output_units]` in the config file with a `mode` option:
```toml
[output_units]
# Options: "per_year" or "total"
mode = "total"
```

**Note**: This is a required configuration option. No default value is provided.

### 2. Scope of Changes
The option affects:
- **Output TIF files**: `movement_rate_*.tif` files will show either per-year or total displacement
- **Internal calculations**: All filtering, LoD, and statistical calculations use the same units
- **Column names**: The displacement column name changes based on mode:
  - `per_year`: `movement_distance_per_year` (existing)
  - `total`: `movement_distance_total` (new)

### 3. Filter Threshold Interpretation
When using `total` mode, filter thresholds are interpreted as raw displacement values:
- `difference_movement_rate_threshold`: raw displacement threshold (e.g., 5 meters)
- `standard_deviation_movement_rate_threshold`: raw displacement threshold
- `level_of_detection_quantile`: calculated from raw displacement values

### 4. Cache Key Integration
The `output_units.mode` should be included in the cache key at the same level as the image enhancement code (to avoid adding another directory level). This prevents mixing cached results from different unit modes.

## Implementation Steps

### Step 1: Update Configuration Schema
**File**: `configs/example_config.toml`
- Add new `[output_units]` section with `mode` option
- No default value - user must explicitly specify

### Step 2: Read Config Option in Pipeline
**File**: `src/PyImageTrack/run_pipeline.py`
- Read `output_units.mode` from config (required - raise error if missing)
- Validate mode is either `"per_year"` or `"total"`
- Pass the mode to `ImagePair` via parameter_dict
- Include mode in cache key at enhancement level

### Step 3: Modify Georeferencing Function
**File**: `src/PyImageTrack/CreateGeometries/HandleGeometries.py`
- Modify `georeference_tracked_points()` function
- Add parameter `output_unit_mode: str`
- When mode is `"total"`, set `movement_distance_total = movement_distance` (no division)
- When mode is `"per_year"`, keep existing behavior: `movement_distance_per_year = movement_distance / years_between`

### Step 4: Update ImagePair Class
**File**: `src/PyImageTrack/ImageTracking/ImagePair.py`
- Add `output_unit_mode` attribute to `ImagePair`
- Update `displacement_column_name` logic:
  - If mode is `"per_year"`: use `"movement_distance_per_year"` (existing)
  - If mode is `"total"`: use `"movement_distance_total"` (new)
- Pass mode to `georeference_tracked_points()` calls
- For non-georeferenced images with 3D displacement, create `3d_displacement_distance_total` column when mode is `"total"`

### Step 5: Update Filtering Functions
**File**: `src/PyImageTrack/DataProcessing/DataPostprocessing.py`
- Update `filter_lod_points()` to accept `displacement_column_name` parameter
- Update `filter_outliers_movement_rate_difference()` to use correct column name
- Update `filter_outliers_movement_rate_standard_deviation()` to use correct column name
- Ensure all filtering functions work with either column name

### Step 6: Update LoD Calculation
**File**: `src/PyImageTrack/run_pipeline.py`
- Update `_recompute_lod_from_points()` to use correct column name based on mode
- Update LoD calculation to use the appropriate displacement column

### Step 7: Update Plotting Functions
**File**: `src/PyImageTrack/Plots/MakePlots.py`
- Update plotting functions to detect and use the correct displacement column name
- Support both `movement_distance_per_year` and `movement_distance_total`

### Step 8: Update Statistical Parameters Output
**File**: `src/PyImageTrack/ImageTracking/ImagePair.py`
- Update `statistical_parameters_txt` output to reflect the correct units
- Add unit label to output (e.g., "m/year" vs "m")

### Step 9: Update Documentation
**File**: `docs/pyimagetrack_documentation.md`
- Document the new `[output_units]` configuration option
- Explain the difference between `per_year` and `total` modes
- Provide examples and note about filter threshold interpretation

## Files to Modify

1. `configs/example_config.toml` - Add new config section
2. `src/PyImageTrack/run_pipeline.py` - Read and pass config option
3. `src/PyImageTrack/CreateGeometries/HandleGeometries.py` - Modify georeferencing
4. `src/PyImageTrack/ImageTracking/ImagePair.py` - Update column name logic
5. `src/PyImageTrack/DataProcessing/DataPostprocessing.py` - Update filtering functions
6. `src/PyImageTrack/Plots/MakePlots.py` - Update plotting functions
7. `docs/pyimagetrack_documentation.md` - Update documentation

## Configuration Requirement

The `[output_units]` section with `mode` is **required**. Users must explicitly specify either `"per_year"` or `"total"`. No backward compatibility is maintained.

## Testing Considerations

1. Test with `per_year` mode - verify per-year calculations work correctly
2. Test with `total` mode - verify:
   - Output TIF files show raw displacement values
   - Filtering works correctly with total displacement
   - LoD is calculated from total displacement
   - Statistical parameters reflect total displacement
3. Test with different `years_between` values to verify total mode is independent of time span
4. Test cache invalidation when switching between modes

## Edge Cases

1. **Invalid mode value**: If user provides a value other than `"per_year"` or `"total"`, raise a clear error message.

2. **Cached results**: The cache key should include the `output_units.mode` at the same level as the image enhancement code (to avoid adding another directory level). This prevents mixing cached results from different unit modes.

3. **3D displacement for non-georeferenced images**: Apply the same logic to create `3d_displacement_distance_total` column when mode is `"total"`.