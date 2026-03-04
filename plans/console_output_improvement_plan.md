# Console Output Improvement Plan

## Current State Analysis

The current console output is functional but lacks clear structure and context. Users cannot easily identify which processing step corresponds to which configuration section.

### Current Output Example:
```
Image pairs to process (custom): 1

Processed image pair: 20040107 (2004-01-07) → 20040114 (2004-01-14)
   File 1: /home/lisa/projects/pyimagetrack/input/glaciares_daniel_2004/20040107.tif
   File 2: /home/lisa/projects/pyimagetrack/input/glaciares_daniel_2004/20040114.tif
Converting image from float32 to uint16 for alignment.
WARNING:root:Found 3947 NaN values in float image. Replacing with 0.
Converting image from float32 to uint16 for alignment.
WARNING:root:Found 3948 NaN values in float image. Replacing with 0.
Starting image alignment.
Created 2017 points on the polygon with distance 142.2 metre.
Tracking points for alignment: 100%|███████████████████████████████████████████| 2017/2017 points[00:00, 184.36points/s]
Used 1654 pixels for alignment.
Resampling the second image matrix with transformation matrix
[[ 9.99233851e-01 -3.75648964e-04  2.86831429e-01]
 [-1.83164458e-04  1.00043619e+00  1.99671001e-01]]
This may take some time.
[CACHE] Alignment saved to:   /home/lisa/projects/pyimagetrack/output/glaciares_daniel_2004/20040107_20040114/E_none_U_per_year/A_AS2_2_2_2_CP2000_CC50_CCa0.3  (pair 20040107->20040114)
Starting point tracking.
Created 8888 points on the polygon with distance 30.0 metre (1.0 px).
Tracking points for movement tracking: 100%|███████████████████████████████████| 8888/8888 points[00:00, 396.65points/s]
[CACHE] Tracking saved to:   /home/lisa/projects/pyimagetrack/output/glaciares_daniel_2004/20040107_20040114/E_none_U_.../T_TS2_2_3_1_IB0_DP1_MC10_CC0.5  (pair 20040107->20040114)

Summary:
Successfully processed: 1 pairs
   - 20040107 → 20040114

Skipped: 0 pairs
```

## Proposed Improvements

### 1. Create a Centralized Output Utility Module

**File:** `src/PyImageTrack/ConsoleOutput.py`

A new module to handle all console output with consistent formatting:
- Step labels with config section references
- Colored output for different message types (info, warning, error, success)
- Progress bar integration
- Timing information
- Parameter summaries

### 2. Improved Output Structure

The new output will follow this structure:

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                    PyImageTrack - Image Tracking Pipeline                     ║
╚══════════════════════════════════════════════════════════════════════════════╝

[CONFIG] Loading configuration from: configs/glaciares_daniel_2004.toml
[CONFIG] Image pairs to process (custom): 1

═══════════════════════════════════════════════════════════════════════════════
Processing Image Pair: 20040107 (2004-01-07) → 20040114 (2004-01-14)
═══════════════════════════════════════════════════════════════════════════════
  File 1: /home/lisa/projects/pyimagetrack/input/glaciares_daniel_2004/20040107.tif
  File 2: /home/lisa/projects/pyimagetrack/input/glaciares_daniel_2004/20040114.tif
  Time between observations: 7.0 days (0.019 years)

─────────────────────────────────────────────────────────────────────────────────
[PREPROCESSING] [flags.do_image_enhancement] Converting images for alignment
─────────────────────────────────────────────────────────────────────────────────
  ✓ Converting image 1: float32 → uint16
  ⚠ Found 3947 NaN values in float image. Replacing with 0.
  ✓ Converting image 2: float32 → uint16
  ⚠ Found 3948 NaN values in float image. Replacing with 0.

─────────────────────────────────────────────────────────────────────────────────
[ALIGNMENT] [flags.do_alignment] Co-registering image pairs
─────────────────────────────────────────────────────────────────────────────────
  Parameters:
    • Number of control points: 2000
    • Control cell size: 50 px
    • Search extent: [2, 2, 2, 2] px
    • Cross-correlation threshold: 0.3
    • Maximal alignment movement: 3 px

  ✓ Created 2017 points on the polygon (distance: 142.2 m)
  ✓ Tracking points for alignment: 2017/2017 points [00:00, 184.36 points/s]
  ✓ Used 1654 pixels for alignment (82.0% of points passed threshold)
  ✓ Resampling second image with transformation matrix:
      [[ 9.99233851e-01 -3.75648964e-04  2.86831429e-01]
       [-1.83164458e-04  1.00043619e+00  1.99671001e-01]]
  ✓ Alignment completed in 0.5s

  [CACHE] Alignment saved to:
    /home/lisa/projects/pyimagetrack/output/.../A_AS2_2_2_2_CP2000_CC50_CCa0.3

─────────────────────────────────────────────────────────────────────────────────
[TRACKING] [flags.do_tracking] Detecting movement between images
─────────────────────────────────────────────────────────────────────────────────
  Parameters:
    • Image bands: 0
    • Distance of tracked points: 1 px
    • Movement cell size: 10 px
    • Search extent: [2, 2, 3, 1] px
    • Cross-correlation threshold: 0.5
    • Adaptive tracking window: enabled (scaled by 0.019 years)

  ✓ Created 8888 points on the polygon (distance: 30.0 m, 1.0 px)
  ✓ Tracking points for movement: 8888/8888 points [00:00, 396.65 points/s]
  ✓ Tracking completed in 0.3s

  [CACHE] Tracking saved to:
    /home/lisa/projects/pyimagetrack/output/.../T_TS2_2_3_1_IB0_DP1_MC10_CC0.5

─────────────────────────────────────────────────────────────────────────────────
[FILTERING] [flags.do_filtering] Removing outliers from tracking results
─────────────────────────────────────────────────────────────────────────────────
  ℹ Filtering is disabled (skipping this step)

─────────────────────────────────────────────────────────────────────────────────
[PLOTTING] [flags.do_plotting] Generating diagnostic plots
─────────────────────────────────────────────────────────────────────────────────
  ℹ Plotting is disabled (skipping this step)

─────────────────────────────────────────────────────────────────────────────────
[OUTPUT] [save.files] Saving results
─────────────────────────────────────────────────────────────────────────────────
  ℹ No output files configured (skipping this step)

═══════════════════════════════════════════════════════════════════════════════
Summary
═══════════════════════════════════════════════════════════════════════════════
  ✓ Successfully processed: 1 pair
    • 20040107 → 20040114

  ℹ Skipped: 0 pairs

  Total processing time: 1.2s
```

### 3. Key Improvements

1. **Clear Section Headers**: Each processing step is clearly labeled with:
   - Step name (e.g., [ALIGNMENT])
   - Config section reference (e.g., [flags.do_alignment])
   - Visual separators (lines)

2. **Parameter Summaries**: Before each major step, display the relevant parameters from the config

3. **Status Indicators**: Use symbols to indicate status:
   - ✓ Success
   - ⚠ Warning
   - ✗ Error
   - ℹ Info

4. **Timing Information**: Add timing for each major step and total processing time

5. **Better Progress Context**: Show percentage of points that passed thresholds

6. **Consistent Formatting**: Use consistent indentation and spacing

7. **Color Support**: Optional color output for terminals that support it

### 4. Additional Informative Outputs

Add the following new outputs:

1. **Configuration Summary**: Display key config values at startup
2. **Time Between Observations**: Show the time span being analyzed
3. **Point Statistics**: Show how many points passed/failed thresholds
4. **Processing Time per Step**: Time each major operation
5. **Memory Usage**: Optional memory statistics
6. **Cache Status**: Clear indication when cache is used vs. computed
7. **Output File List**: List of files being saved

### 5. Files to Modify

1. **Create**: `src/PyImageTrack/ConsoleOutput.py` - New utility module
2. **Modify**: `src/PyImageTrack/run_pipeline.py` - Main pipeline orchestration
3. **Modify**: `src/PyImageTrack/DataProcessing/ImagePreprocessing.py` - Preprocessing output
4. **Modify**: `src/PyImageTrack/ImageTracking/AlignImages.py` - Alignment output
5. **Modify**: `src/PyImageTrack/ImageTracking/ImagePair.py` - ImagePair output
6. **Modify**: `src/PyImageTrack/CreateGeometries/HandleGeometries.py` - Geometry output
7. **Modify**: `src/PyImageTrack/ImageTracking/TrackMovement.py` - Tracking output

### 6. Implementation Approach

1. Create the `ConsoleOutput` utility class with methods for:
   - Section headers with config references
   - Parameter summaries
   - Status messages with icons
   - Progress tracking
   - Timing measurements

2. Replace existing `print()` statements with calls to the new utility

3. Add timing decorators/context managers for measuring step durations

4. Maintain backward compatibility - the new output should be an improvement, not a breaking change

### 7. Optional Features

1. **Verbose Mode**: Add a `--verbose` flag for more detailed output
2. **Quiet Mode**: Add a `--quiet` flag for minimal output
3. **JSON Output**: Optional JSON logging for machine parsing
4. **Log File**: Write output to a log file in addition to console
5. **Progress Bars**: Keep existing tqdm progress bars but integrate them better

## Config Section Mapping

| Step | Config Section | Current Output | New Output |
|------|---------------|----------------|------------|
| Pairing | [pairing] | "Image pairs to process (custom): 1" | "[CONFIG] Image pairs to process (custom): 1" |
| Preprocessing | [flags.do_image_enhancement] | "Converting image from float32..." | "[PREPROCESSING] [flags.do_image_enhancement] Converting..." |
| Alignment | [flags.do_alignment] | "Starting image alignment." | "[ALIGNMENT] [flags.do_alignment] Co-registering..." |
| Tracking | [flags.do_tracking] | "Starting point tracking." | "[TRACKING] [flags.do_tracking] Detecting movement..." |
| Filtering | [flags.do_filtering] | (no output when disabled) | "[FILTERING] [flags.do_filtering] Removing outliers..." |
| Plotting | [flags.do_plotting] | (no output when disabled) | "[PLOTTING] [flags.do_plotting] Generating plots..." |
| Output | [save.files] | (no output) | "[OUTPUT] [save.files] Saving results..." |