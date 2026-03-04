# Console Output Review

This document lists all console outputs in PyImageTrack (excluding obsolete code) for review.

## Files with Console Outputs

### 1. ConsoleOutput.py
**Status:** ✅ Clean and up-to-date
- Module docstring updated
- Class docstring updated
- All methods properly documented

### 2. run_pipeline.py

| Line | Current Output | Type | Status |
|------|----------------|------|--------|
| 175 | `Found level of detection with quantile...` | print | ⚠️ Needs review |
| 211 | `[OK] Loaded configuration from...` | console.config_loaded | ✅ Good |
| 344 | `[i] Image pairs to process...` | console.info | ✅ Good |
| 355-356 | `Warning: Could not load stable area file...` | print | ⚠️ Needs review |
| 417 | `Processing Image Pair...` | console.section_header | ✅ Good |
| 418-419 | `[i] File 1: ...` / `[i] File 2: ...` | console.info | ✅ Good |
| 426-430 | `[i] Time between observations...` | console.info | ✅ Good |
| 556 | `[i] Cache loaded.` | console.cache_info | ✅ Good |
| 559 | `ALIGNMENT` header | console.section_header | ✅ Good |
| 560-565 | `[i] Parameters:` | console.parameter_summary | ✅ Good |
| 573-574 | `[X] Alignment failed...` | console.error | ✅ Good |
| 590 | `[i] Cache saved.` | console.cache_info | ✅ Good |
| 609 | `[!] CRS not compatible...` | console.warning | ✅ Good |
| 611 | `[i] Cache loaded.` | console.cache_info | ✅ Good |
| 615 | `TRACKING` header | console.section_header | ✅ Good |
| 617-625 | `[i] Parameters:` | console.parameter_summary | ✅ Good |
| 629 | `Tracking` timer | console.timer | ✅ Good |
| 643 | `[i] Cache saved.` | console.cache_info | ✅ Good |
| 645 | `[i] Tracking is disabled...` | console.info | ✅ Good |
| 662 | `[!] CRS not compatible...` | console.warning | ✅ Good |
| 664 | `[i] Cache loaded.` | console.cache_info | ✅ Good |
| 695 | `[i] Cache saved.` | console.cache_info | ✅ Good |
| 700 | `PLOTTING` header | console.section_header | ✅ Good |
| 702 | `Plotting` timer | console.timer | ✅ Good |
| 704-705 | `[i] Plotting is disabled...` | console.info | ✅ Good |
| 719 | `OUTPUT` header | console.section_header | ✅ Good |
| 721 | `[i] Saving files:` | console.file_list | ✅ Good |
| 723 | `[i] No output files configured...` | console.info | ✅ Good |
| 726-731 | `[i] Filtering/Plotting/Output disabled...` | console.info | ✅ Good |
| 741 | Summary | console.print_summary | ✅ Good |

### 3. Utils.py

| Line | Current Output | Type | Status |
|------|----------------|------|--------|
| 165-174 | `[WARN] Only year+month detected...` | print | ⚠️ Needs review |
| 188-197 | `[WARN] Only year detected...` | print | ⚠️ Needs review |
| 353 | `[WARN] Skipping empty pair row...` | print | ⚠️ Needs review |
| 360 | `[WARN] Skipping pair...` | print | ⚠️ Needs review |

### 4. DataProcessing/ImagePreprocessing.py

| Line | Current Output | Type | Status |
|------|----------------|------|--------|
| 103 | `[OK] Converted image...` | console.success | ✅ Good |
| 108 | `[!] Found NaN values...` | console.warning | ✅ Good |

### 5. DataProcessing/DataPostprocessing.py

| Line | Current Output | Type | Status |
|------|----------------|------|--------|
| 54 | `Used X pixels for LoD calculation.` | print | ⚠️ Needs review |

### 6. CreateGeometries/HandleGeometries.py

| Line | Current Output | Type | Status |
|------|----------------|------|--------|
| 83-91 | `[OK] Created X points...` | console.success | ✅ Good |

### 7. ImageTracking/AlignImages.py

| Line | Current Output | Type | Status |
|------|----------------|------|--------|
| 76 | `[~] Tracking points for alignment` | task_label | ✅ Good |
| 93 | `[OK] Used X pixels for alignment...` | console.success | ✅ Good |
| 114 | `logging.warning` - Filtered out NaN points | logging | ⚠️ Needs review |
| 135 | `[i] Resampling second image...` | console.info | ✅ Good |
| 136-142 | Transformation matrix display | console.info | ✅ Good |
| 143 | `[i] This may take some time...` | console.info | ✅ Good |

### 8. ImageTracking/ImagePair.py

| Line | Current Output | Type | Status |
|------|----------------|------|--------|
| 387-388 | `logging.warning` - FALLBACK stable area | logging | ⚠️ Needs review |
| 434 | `logging.warning` - Could not compute true-color | logging | ⚠️ Needs review |
| 541 | `logging.warning` - Images not aligned | logging | ⚠️ Needs review |
| 566 | `[~] Tracking points for movement tracking` | task_label | ✅ Good |
| 631 | `logging.warning` - No results calculated (plot) | logging | ⚠️ Needs review |
| 643 | `logging.warning` - No results calculated (plot) | logging | ⚠️ Needs review |
| 659 | `Filtering outliers. This may take a moment.` | print | ⚠️ Needs review |
| 699 | `Used X pixels for LoD calculation.` | print | ⚠️ Needs review |
| 772-773 | `Found level of detection...` | print | ⚠️ Needs review |

### 9. ImageTracking/TrackMovement.py

| Line | Current Output | Type | Status |
|------|----------------|------|--------|
| 99 | `logging.info` - No matching with positive correlation | logging | ℹ️ Debug info |
| 196 | `logging.info` - Cross-correlation did not provide result | logging | ℹ️ Debug info |
| 200 | `logging.info` - Going with default shift values | logging | ℹ️ Debug info |
| 250 | `logging.info` - NaN values detected in LSM | logging | ℹ️ Debug info |
| 280 | `logging.info` - Did not converge after 50 iterations | logging | ℹ️ Debug info |

## Summary

### ✅ Already Updated (Good)
- ConsoleOutput.py module
- All console.success/info/warning/error calls in run_pipeline.py
- All console calls in ImagePreprocessing.py
- All console calls in HandleGeometries.py
- All console calls in AlignImages.py
- Task labels for tracking progress

### ⚠️ Needs Review/Update

**run_pipeline.py:**
1. Line 175: `print("Found level of detection with quantile...")` - Should use console
2. Lines 355-356: `print("Warning: Could not load stable area file...")` - Should use console.warning

**Utils.py:**
1. Lines 165-174: `print("[WARN] Only year+month detected...")` - Should use console.warning
2. Lines 188-197: `print("[WARN] Only year detected...")` - Should use console.warning
3. Line 353: `print("[WARN] Skipping empty pair row...")` - Should use console.warning
4. Line 360: `print("[WARN] Skipping pair...")` - Should use console.warning

**DataPostprocessing.py:**
1. Line 54: `print("Used X pixels for LoD calculation.")` - Should use console.success

**ImagePair.py:**
1. Line 387-388: `logging.warning` - FALLBACK stable area - Should use console.warning
2. Line 434: `logging.warning` - Could not compute true-color - Should use console.warning
3. Line 541: `logging.warning` - Images not aligned - Should use console.warning
4. Lines 631, 643: `logging.warning` - No results calculated - Should use console.warning
5. Line 659: `print("Filtering outliers. This may take a moment.")` - Should use console.processing
6. Line 699: `print("Used X pixels for LoD calculation.")` - Should use console.success
7. Lines 772-773: `print("Found level of detection...")` - Should use console.success

**AlignImages.py:**
1. Line 114: `logging.warning` - Filtered out NaN points - Should use console.warning

**TrackMovement.py:**
1. Lines 99, 196, 200, 250, 280: `logging.info` - Debug messages - Keep as logging (verbose only)

### ℹ️ Debug Info (Keep as logging)
- TrackMovement.py: All logging.info messages are debug-level information about tracking internals