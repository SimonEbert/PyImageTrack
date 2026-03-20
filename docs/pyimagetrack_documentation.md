# PyImageTrack Documentation

Version: 0.2
Date: 2026-02-18

## Overview
PyImageTrack provides alignment and feature tracking for georeferenced imagery, with optional filtering and output visualization.
The main entry point is the CLI command `pyimagetrack-run`, configured via TOML files in `configs/`.

## Running The Pipeline
```
pyimagetrack-run --config configs/example_config.toml
```
Config paths are resolved relative to the repo root.
The CLI invokes `PyImageTrack.run_pipeline:main` under the hood.

## Command-Line Interface Options

The `pyimagetrack-run` command provides several options for controlling output verbosity and logging.

### Basic Options

- `--config` (required): Path to TOML configuration file

### Output Control

- `--verbose`: Enable verbose output with detailed information about processing steps
- `--quiet`: Enable quiet mode with minimal output (only essential messages)
- `--no-color`: Disable colored terminal output

### Logging Options

- `--log-file PATH`: Specify a custom log file path. If not provided, logs are written to `pyimagetrack.log` in the output folder.
- `--log-level LEVEL`: Set the logging level for file output. Available levels: DEBUG, INFO, WARNING, ERROR, CRITICAL. Default: INFO.
- `--log-max-bytes BYTES`: Maximum size of log file before rotation. Default: 10MB.
- `--log-backup-count COUNT`: Number of backup log files to keep. Default: 5.

### Verbose Mode

Enable verbose output with `--verbose` to see detailed information about:
- Parameter values being used
- Cache operations (loading/saving)
- File paths and pair identifiers
- Timing information for each processing step

### Quiet Mode

Enable quiet mode with `--quiet` to minimize console output. In quiet mode:
- Only essential messages are displayed (success, warning, error)
- Section headers and informational messages are suppressed
- All messages are still written to the log file

### Color Output

Use `--no-color` to disable ANSI color codes in terminal output. This is useful for:
- Non-terminal environments (e.g., CI/CD pipelines)
- Log files that should be plain text
- Terminals that don't support color

### Example Usage

```bash
# Verbose run with debug logging
pyimagetrack-run --config configs/example_config.toml --verbose --log-level DEBUG

# Quiet run with custom log file
pyimagetrack-run --config configs/example_config.toml --quiet --log-file /path/to/custom.log

# Disable colors for CI/CD
pyimagetrack-run --config configs/example_config.toml --no-color
```

## Batch Processing

PyImageTrack supports batch processing of multiple configurations with automatic filtering based on identifiers extracted from filenames. This is useful when processing multiple rock glaciers or similar datasets.

### Command-Line Interface

The `pyimagetrack-batch` command processes multiple configurations using two CSV tables:

```bash
pyimagetrack-batch --table-a table_a.csv --config-col config_name --class-col-a class \
                  --table-b table_b.csv --identifier-col identifier --class-col-b class
```

### Table A (Config Mapping)

Table A maps configuration files to classes. Each row specifies which configuration file should be used for a given class.

| config_name | class |
|-------------|-------|
| config_large.toml | large |
| config_medium.toml | medium |
| config_small.toml | small |

### Table B (Identifier Classification)

Table B maps identifiers to classes. Each row specifies which class an identifier belongs to.

| identifier | class |
|------------|-------|
| 9109 | large |
| 9110 | medium |
| 9111 | small |

### How Batch Processing Works

1. The batch processor reads both tables
2. For each configuration in Table A:
   - Gets the class associated with the configuration
   - Filters Table B to find all identifiers with that class
   - Filters identifiers to only those present in the input folder
   - Processes each identifier separately using the configuration
   - Replaces wildcards in shapefile names with the identifier

### Identifier Extraction

Identifiers are extracted from filenames using the pattern `id<identifier>` where:
- `id` is a literal prefix
- `<identifier>` is an alphanumeric sequence (letters and numbers)
- The identifier ends at a separator (`-`, `_`, `.`) or end of filename

Examples:
- `2008_id9109_az315.tif` → identifier: `9109`
- `2008_id9110_az315.tif` → identifier: `9110`
- `2008_id9111_az315.tif` → identifier: `9111`

### Wildcard Support in Shapefiles

Shapefile names in configuration files can include a wildcard (`*`) that gets replaced with the identifier during processing:

```toml
[polygons]
stable_area_filename = "stable_area_*.shp"
moving_area_filename = "moving_area_*.shp"
```

When processing identifier `9109`, these become:
- `stable_area_9109.shp`
- `moving_area_9109.shp`

**Stable / Moving Areas:**
- Stable-area shapefiles can include one or multiple singlepart polygons; they are merged into one reference area for alignment.
- Moving-area shapefiles can include one or multiple polygons. Provide an ID column to report stats per polygon; missing IDs are filled with the row index.
- Use `polygons.moving_id_column` to choose the ID column name (default: `moving_id`).

### Batch Processing Options

- `--table-a PATH`: Path to Table A CSV file (config mapping)
- `--config-col NAME`: Column name in Table A containing configuration file names (default: config_name)
- `--class-col-a NAME`: Column name in Table A containing class names (default: class)
- `--table-b PATH`: Path to Table B CSV file (identifier classification)
- `--identifier-col NAME`: Column name in Table B containing identifiers (default: identifier)
- `--class-col-b NAME`: Column name in Table B containing class names (default: class)
- `--verbose`: Enable verbose output
- `--quiet`: Enable quiet mode
- `--no-color`: Disable colored output
- `--log-file PATH`: Path to log file
- `--log-level LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- `--log-max-bytes BYTES`: Maximum log file size before rotation (default: 10MB)
- `--log-backup-count COUNT`: Number of backup log files (default: 5)

### Important Notes

- In single config mode (`pyimagetrack-run`), identifiers are completely ignored - the system works as before
- Only images with the same identifier can be processed together in batch mode
- Files without identifiers are skipped in batch mode
- The batch processor automatically looks for config files in the `configs/` folder if not found at the specified path
- Only identifiers that actually exist in the input folder are processed

## Entry point for Python Scripts
The package provides the possibility of accessing the tracking routine from another Python script. This is done using
the `run_from_config` function that can be imported by
```
from PyImageTrack import run_from_config
```
IMPORTANT: When calling the tracking routine from within another Python Script or Project, the process must always be
guarded in the following way
```
if __name__ == "__main__":
    run_from_config(path_to_config)
```
This is due to the tracking routine making use of the multiprocessing package for parallelization and without this
safety measure, the tracking routine will be called several times with identical parameters.

### Programmatic Usage with Output Control

When calling `run_from_config()` from Python scripts, you can pass the same options as the CLI:

```python
from PyImageTrack import run_from_config

if __name__ == "__main__":
    run_from_config(
        config_path="configs/example_config.toml",
        verbose=True,
        quiet=False,
        use_colors=True,
        log_file="my_tracking.log",
        log_level="DEBUG",
        log_max_bytes=10 * 1024 * 1024,
        log_backup_count=5
    )
```

For batch processing with identifier filtering:

```python
from PyImageTrack import run_from_config

if __name__ == "__main__":
    run_from_config(
        config_path="configs/example_config.toml",
        identifier="9109",  # Only process files with id9109
        verbose=True,
        quiet=False,
        use_colors=True,
        log_file="my_tracking.log",
        log_level="DEBUG",
        log_max_bytes=10 * 1024 * 1024,
        log_backup_count=5
    )
```

### Parameters

- `config_path` (str): Path to TOML configuration file
- `verbose` (bool): Enable verbose output. Default: False
- `quiet` (bool): Enable quiet mode. Default: False
- `use_colors` (bool): Use ANSI colors in terminal output. Default: True
- `log_file` (str): Path to log file. Default: None (uses `pyimagetrack.log` in output folder)
- `log_level` (str): Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL). Default: INFO
- `log_max_bytes` (int): Maximum log file size before rotation in bytes. Default: 10MB
- `log_backup_count` (int): Number of backup log files to keep. Default: 5
- `identifier` (str, optional): Identifier for batch processing. Only images with matching identifiers are processed. Default: None (single mode)

## Configuration Files (TOML)
Configs are TOML files and share the same structure. Use `configs/example_config.toml` as a template.

### Key Sections
- `[paths]`: input/output folders, optional CSVs for dates/pairs.
- `[polygons]`: stable/moving area shapefiles and CRS.
- `[pairing]`: pairing mode (`all`, `first_to_all`, `successive`, `custom`).
- `[no_georef]`: enable for non-ortho images (e.g., JPGs) and configure fake georeferencing and depth-image options.
- `[downsampling]`: optional downsampling for fast smoke tests.
- `[flags]`: alignment/tracking/filtering toggles.
- `[image_enhancement]`: optional image enhancement (CLAHE) configuration.
- `[cache]`: caching and recompute flags.
- `[output]`: optional outputs such as true-color alignment and interactive plot display.
- `[output_units]`: **required** - specifies whether movement values are normalized per year or reported as total displacement.
- `[adaptive_tracking_window]`: when enabled, search extent scales by time span between images. The `search_extent_px` parameter then represents the expected movement per year, and the actual search window is calculated as `search_extent_px * years_between_observations`.
- `[alignment]`, `[tracking]`, `[filter]`: algorithm parameters.
- `[save]`: list of output files to write.

### Stable Area Fallback Mode
The `[polygons]` section supports an optional fallback mode for defining the stable area. If `stable_area_filename` is set to `"none"` or the specified file doesn't exist, the system will automatically use `image_bounds minus moving_area` as the stable area. This assumes all areas outside the moving area are stable. The resulting stable area is treated like the merged reference area described above.

**Note**: This fallback mode may result in slightly lower alignment quality compared to using a properly defined stable area polygon. To compensate, consider increasing the `number_of_control_points` parameter in the `[alignment]` section.

### [no_georef] options and depth-image settings
If you enable fake/no-georeferencing via `[no_georef]`, additional options control how non-georeferenced images are handled and how optional 3D displacement calculation from depth images is performed.

Example TOML snippet:
```toml
[no_georef]
use_no_georeferencing = true
fake_pixel_size = 1                 # CRS units per pixel (e.g., meters per pixel)
convert_to_3d_displacement = true   # If true, compute 3D displacements using depth images
undistort_image = true              # If true, undistort both RGB and depth images before tracking

# Camera intrinsics: 3x3 matrix in the following format
camera_intrinsics_matrix = [
  [fx, s, cx],
  [0.0, fy, cy],
  [0.0, 0.0, 1.0]
]

# Distortion coefficients: 2 or 4 elements as required by OpenCV (radial +/- tangential)
camera_distortion_coefficients = [k1, k2]  # or [k1,k2,p1,p2]

# Optional 4x4 homogeneous transform mapping camera coords -> target 3D coords
# Can be used to transform computed 3d image coordinates from the depth image to an arbitrary 3d coordinate system
# given the respective homogeneous transform in the following format
camera_to_3d_coordinates_transform = [
  [r11, r12, r13, t1],
  [r21, r22, r23, t2],
  [r31, r32, r33, t3],
  [0.0,  0.0,  0.0,  1.0]
]
```
Notes and requirements:
- `convert_to_3d_displacement`: when true, the pipeline will look for per-image depth rasters and compute 3D displacements.
- `fake_pixel_size` gives the pixel size used when working without a CRS (units per pixel).
- `camera_intrinsics_matrix` and `camera_distortion_coefficients` are required if `undistort_image = true` or when computing image→camera coordinate transforms.
- `camera_to_3d_coordinates_transform` must be a 4×4 homogeneous matrix in standard row-major layout: [[R (3×3), t (3×1)], [0 0 0, 1]]. The pipeline applies this matrix directly (no internal transpose) when transforming points.
- Arrays in TOML are parsed into lists and converted to numpy arrays by the pipeline — use numeric literals (no strings).

### [output_units] - Movement Units
The `[output_units]` section is **required** and specifies how movement values are calculated and reported.

**Important Notes:**
- When using `"total"` mode, all filter thresholds in the `[filter]` section are interpreted as raw displacement values (e.g., meters) rather than per-year rates.
- The output TIF files (`L*_movement-rate_*.tif`) will show either per-year or total displacement depending on the mode.
- The statistical parameters output file will indicate the units used (e.g., "Movement (per year)" vs "Movement (total)").
- The cache key includes the output units mode to prevent mixing cached results from different modes.

## Module: run_pipeline.py
### _load_config(path: str) -> dict
Loads a TOML config. Relative paths are resolved against the repository root.

#### Parameters

`path` : str
    Path to a TOML configuration file.

#### Returns

config : dict
    Parsed configuration dictionary.

### _get(cfg: dict, section: str, key: str, default=None)
Returns a config value or a default if missing.

#### Parameters
`cfg` : dict
    Parsed config.

``section`` : str
    Section name.

``key`` : str
    Key within the section.

``default`` : any
    Fallback value.

#### Returns

``value`` : any
    The config value or `default`.

### _require(cfg: dict, section: str, key: str)
Returns a required config value or raises KeyError.

#### Parameters

``cfg`` : dict
    Parsed config.

``section`` : str
    Section name.

``key`` : str
    Key within the section.

#### Returns

``value`` : any
    The required config value.

### _as_optional_value(value)
Normalizes `""`, `"none"`, and `"null"` to `None`.

#### Parameters
``value`` : any
    Input value.

#### Returns
``value_or_none``
    The normalized value.

### make_effective_extents_from_deltas(deltas, cell_size, years_between=1.0, cap_per_side=None)
Converts per-year delta extents into effective search extents by adding half-cell padding and scaling by time span.

#### Parameters
``deltas`` : tuple
    (posx, negx, posy, negy) extra pixels per year beyond half the template.

``cell_size`` : int
    Size of the tracked cell (movement or control cell size).

``years_between`` : float
    Time span in years between two images.

``cap_per_side`` : int or None
    Optional clamp for each side (to keep windows bounded).

#### Returns
``extents`` : tuple
    (posx, negx, posy, negy) effective extents in pixels.

### run_from_config(config_path: str, verbose: bool = False, quiet: bool = False, use_colors: bool = True, log_file: str = None, log_level: str = 'INFO', log_max_bytes: int = 10 * 1024 * 1024, log_backup_count: int = 5, identifier: Optional[str] = None)
Runs the full PyImageTrack pipeline from a configuration file.

This function orchestrates the complete processing workflow:
- Collects image pairs from input folder/CSV
- Loads polygons
- Builds per-pair directories and codes
- Loads images (with optional downsampling)
- Aligns and tracks with cache support
- Filters, displays interactive plots (optional), saves outputs and summary statistics

#### Parameters
``config_path`` : str
    Path to TOML configuration file.

``verbose`` : bool
    Enable verbose output. Default is False.

``quiet`` : bool
    Enable quiet mode. Default is False.

``use_colors`` : bool
    Use ANSI colors in terminal output. Default is True.

``log_file`` : str or None
    Path to log file. Default is None (uses `pyimagetrack.log` in output folder).

``log_level`` : str
    Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL). Default is 'INFO'.

``log_max_bytes`` : int
    Maximum log file size before rotation in bytes. Default is 10MB.

``log_backup_count`` : int
    Number of backup log files. Default is 5.

``identifier`` : str or None
    Optional identifier for batch processing. When provided, only images with matching identifiers are processed.
    Identifiers are extracted from filenames using the pattern `id<identifier>`.
    When an identifier is provided, the output folder includes an identifier subfolder and wildcards in shapefile names are replaced with the identifier.

#### Returns
``None``

#### Notes
- When `identifier` is None (default), the pipeline works in single mode and identifiers are completely ignored.
- When `identifier` is provided, the pipeline processes only images with matching identifiers.
- See `extract_identifier()` documentation for identifier extraction pattern.
- See `collect_pairs()` documentation for how identifiers affect image pairing.

### main()
Orchestrates the full pipeline:
- Collects image pairs from input folder/CSV
- Loads polygons
- Builds per-pair directories and codes
- Loads images (with optional downsampling)
- Aligns and tracks with cache support
- Filters, displays interactive plots (optional), saves outputs and summary statistics

## Module: Utils.py
### parse_date(s: str) -> datetime
Parses ISO-standard date strings with flexible separators.

Only accepts dates starting with year (YY or YYYY). Supports separators: '-', '_', or none.
Missing or invalid parts default to first standard (month=01, day=01, hour=00, minute=00, second=00).
No rounding - exact values are used.

#### Supported Formats
- Year only: 2024, 24
- Year-Month: 2024-09, 2024_09, 202409, 24-09, 24_09, 2409
- Year-Month-Day: 2024-09-01, 2024_09_01, 20240901, 24-09-01, 24_09_01, 240901
- With time: 2024-09-01-14-30-45, 2024_09_01_14_30_45, 20240901143045

#### Parameters
``s`` : str
    Input string (filename or date string).

#### Returns
``dt`` : datetime
    Parsed datetime with exact values (no rounding).

#### Note
If a part is present but invalid (e.g., month=13, day=47, hour=99), it is
ignored and the default value is used. This allows filenames like "2008_9109"
to be parsed as year-only (2008-01-01), or "2024_09_47" to be parsed as
year-month (2024-09-01).

### extract_identifier(filename: str) -> Optional[str]
Extracts an identifier from a filename using the pattern `id<identifier>`.

The identifier is extracted from filenames that contain the pattern `id<identifier>` where:
- `id` is a literal prefix
- `<identifier>` is an alphanumeric sequence (letters and numbers)
- The identifier ends at a separator (`-`, `_`, `.`) or end of filename

#### Parameters
``filename`` : str
    Input filename.

#### Returns
``identifier`` : str or None
    The extracted identifier, or None if no identifier is found.

#### Examples
- `2008_id9109_az315.tif` → `9109`
- `2008_id9110_az315.tif` → `9110`
- `2008_id9111_az315.tif` → `9111`
- `2008_9109_az315.tif` → `None` (no `id` prefix)

### collect_pairs(input_folder, date_csv_path=None, pairs_csv_path=None, pairing_mode="all", extensions=None, identifier=None)
Builds image pairs from filenames and returns:
- year_pairs: list of (id1, id2)
- id_to_file: id -> file path
- id_to_date: id -> date string

#### Parameters
``input_folder`` : str
    Folder containing images. Filenames must contain date tokens in ISO format (year-first).

``date_csv_path`` : str or None
    Optional CSV with `date` column for resolving dates from filenames. Dates in CSV use the same format as filename dates.

``pairs_csv_path`` : str or None
    Optional CSV specifying custom pairs (date_earlier/date_later).

``pairing_mode`` : str
    all | successive | first_to_all | custom.

``extensions`` : tuple or None
    Allowed extensions; defaults to (".tif", ".tiff").

``identifier`` : str or None
    Optional identifier to filter images. Only images with matching identifiers will be paired.
    Identifiers are extracted from filenames using the pattern `id<identifier>`.

#### Returns
``year_pairs`` : list[tuple]
    List of (id1, id2) pairs.

``id_to_file`` : dict
    Mapping from id to file path.

``id_to_date`` : dict
    Mapping from id to date string.

#### Notes
- Date tokens in filenames are extracted using the same `parse_date()` function as CSV dates.
- All dates must start with year (YY or YYYY) and use ISO-standard format.
- See `parse_date()` documentation for supported date formats.
- When `identifier` is provided, only images with matching identifiers are paired.
- See `extract_identifier()` documentation for identifier extraction pattern.

### ensure_dir(path: str)
Creates a directory if missing.

#### Parameters
``path`` : str
    Directory path.

#### Returns
None

### float_compact(x)
Formats a float into a compact string without trailing zeros.

#### Parameters
``x`` : float or any
    Value to format.

#### Returns
``s`` : str
    Compact string.

### _get(obj, name, default="NA")
Returns `obj.name` or `obj[name]` with a default fallback.

#### Parameters
``obj`` : object or dict
    Source object.

``name`` : str
    Attribute or key.

``default`` : any
    Default if missing.

#### Returns
``value`` : any
    Attribute/key value or default.

### abbr_alignment(ap)
Builds a short alignment code for output folders.

#### Parameters
``ap`` : object or dict
    Alignment parameters.

#### Returns
``code`` : str
    Folder code.

### abbr_tracking(tp)
Builds a short tracking code for output folders.

#### Parameters
``tp`` : object or dict
    Tracking parameters.

#### Returns
``code`` : str
    Folder code.

### abbr_filter(fp)
Builds a short filter code for output folders.

#### Parameters
``fp`` : FilterParameters
    Filter parameters.

#### Returns
``code`` : str
    Folder code.

### abbr_enhancement(ep)
Builds a short enhancement code for output folders.

#### Parameters
``ep`` : object or dict
    Enhancement parameters.

#### Returns
``code`` : str
    Folder code (e.g., "E_none", "E_clahe_K50_C0.9").

## Image Enhancement Configuration

The pipeline supports optional image enhancement before alignment and tracking. Enhancement is applied to both images and affects both subsequent operations.

### Configuration Section

`[image_enhancement]` section in the TOML config:

```toml
[flags]
do_image_enhancement = true

[image_enhancement]
# Enhancement type: "clahe", "none", or future types
type = "clahe"
# CLAHE-specific parameters (common defaults: kernel_size = 50, clip_limit = 0.9)
kernel_size = 50
clip_limit = 0.9
```

### Enhancement Types

#### CLAHE (Contrast Limited Adaptive Histogram Equalization)
- `type = "clahe"`
- `kernel_size`: Size of the grid for histogram equalization
- `clip_limit`: Contrast limiting threshold

#### None (no enhancement)
- `type = "none"` or omit the section

### Output Path Structure

The output path structure includes codes for each processing step:

```
output_folder / year1_year2 / enhancement_code / align_code / track_code / filter_code
```

### Path Codes

Each processing step generates a code that appears in the output path:

- **Enhancement**: `E_none` (disabled) or `E_clahe_K50_C0.9` (with parameters)
- **Alignment**: `A_none` (disabled) or `A_CP2000_CC5_CCa0.8` (with parameters)
- **Tracking**: Always uses tracking parameters (e.g., `T_IB3_DP3_MC20_CC0.5`)
- **Filtering**: `F_none` (disabled) or `F_LoDq0.5_N1000_...` (with parameters)

### Special "none" Codes

When certain processing steps are disabled and not loaded from cache, the output path uses special "none" codes:

- **Enhancement**: If `do_image_enhancement = false`, the path uses `E_none`
- **Alignment**: If `do_alignment = false`, the path uses `A_none`
- **Filtering**: If `do_filtering = false`, the path uses `F_none`
- **Tracking**: No "none" code is used for tracking, as tracking is required for meaningful output


## Module: Cache.py
### _sha256(path: str) -> str
Computes a SHA-256 hash for a file.

#### Parameters
``path`` : str
    File path.

#### Returns
``hex_digest`` : str
    Hash string.

### alignment_cache_paths(align_dir, year1, year2)
Returns paths for aligned raster, control points, and metadata JSON.

#### Parameters
``align_dir`` : str
    Alignment folder.

``year1``, ``year2`` : str
    Pair identifiers.

#### Returns
``aligned_tif``, ``control_pts``, ``meta_json`` : tuple
    Output paths.

### lod_cache_paths(track_dir, year1, year2)
Returns paths for LoD points GeoJSON and metadata JSON.

#### Parameters
``track_dir`` : str
    Tracking folder.

``year1``, ``year2`` : str
    Pair identifiers.

#### Returns
``lod_geojson``, ``meta_json`` : tuple
    Output paths.

### save_lod_cache(image_pair, track_dir, year1, year2, filenames, dates, version="v1")
Writes LoD points GeoJSON and metadata JSON. The metadata includes the calculated level of detection value.

#### Parameters
``image_pair`` : ImagePair
    Pair object with LoD points and level_of_detection value.

``track_dir`` : str
    Tracking folder.

``year1``, ``year2`` : str
    Pair identifiers.

``filenames`` : dict
    Mapping id -> path.

``dates`` : dict
    Mapping id -> date.

``version`` : str
    Metadata version.

#### Returns
``None``

### load_lod_cache(image_pair, track_dir, year1, year2) -> bool
Loads LoD points and the level of detection value from cache into an ImagePair.

#### Parameters
``image_pair`` : ImagePair
    Target object.

``track_dir`` : str
    Tracking folder.

``year1``, ``year2`` : str
    Pair identifiers.

#### Returns
``success`` : bool
    True if cache was loaded. The level_of_detection value is restored from the cache metadata.

### save_alignment_cache(image_pair, align_dir, year1, year2, align_params, filenames, dates, version="v1", save_truecolor_aligned=False)
Writes aligned raster (and optional true-color raster), control points, and metadata JSON.

#### Parameters
``image_pair`` : ImagePair
    Pair object with aligned image data.

``align_dir`` : str
    Alignment folder.

``year1``, ``year2`` : str
    Pair identifiers.

``align_params`` : dict
    Alignment parameters for metadata.

``filenames`` : dict
    Mapping id -> path.

``dates`` : dict
    Mapping id -> date.

``version`` : str
    Metadata version.

``save_truecolor_aligned`` : bool
    If true, writes a true-color aligned image if available.

#### Returns
``None``

### load_alignment_cache(image_pair, align_dir, year1, year2) -> bool
Loads aligned raster and control points into an ImagePair.

#### Parameters
``image_pair`` : ImagePair
    Target object.

``align_dir`` : str
    Alignment folder.

``year1``, ``year2`` : str
    Pair identifiers.

#### Returns
``success`` : bool
    True if cache was loaded.

### tracking_cache_paths(track_dir, year1, year2)
Returns paths for raw tracking GeoJSON and metadata JSON.

#### Parameters
``track_dir`` : str
    Tracking folder.

``year1``, ``year2`` : str
    Pair identifiers.

#### Returns
``raw_geojson``, ``meta_json`` : tuple
    Output paths.

### save_tracking_cache(image_pair, track_dir, year1, year2, track_params, filenames, dates, version="v1")
Writes raw tracking GeoJSON and metadata JSON.

#### Parameters
``image_pair`` : ImagePair
    Pair object with tracking results.

``track_dir`` : str
    Tracking folder.

``year1``, ``year2`` : str
    Pair identifiers.

``track_params`` : dict
    Tracking parameters for metadata.

``filenames`` : dict
    Mapping id -> path.

``dates`` : dict
    Mapping id -> date.

``version`` : str
    Metadata version.

#### Returns
``None``

### load_tracking_cache(image_pair, track_dir, year1, year2) -> bool
Loads tracking GeoJSON into an ImagePair.

#### Parameters
``image_pair`` : ImagePair
    Target object.

``track_dir`` : str
    Tracking folder.

``year1``, ``year2`` : str
    Pair identifiers.

#### Returns
``success`` : bool
    True if cache was loaded.

## Module: Parameters
### AlignmentParameters
Container for alignment parameters.

#### Parameters
``parameter_dict`` : dict
    Source dict of parameters.

#### Fields

- ``number_of_control_points``
- ``control_search_extent_px``
- ``control_search_extent_deltas``
- ``control_cell_size``
- ``cross_correlation_threshold_alignment``
- ``maximal_alignment_movement``

#### Methods
__str__()

Human-readable summary.

to_dict()

Keys expected by ImagePair(parameter_dict=...).

### TrackingParameters
Container for tracking parameters.

#### Parameters
``parameter_dict`` : dict
    Source dict of parameters.

#### Fields

- ``image_bands``
- ``distance_of_tracked_points_px``
- ``search_extent_px``
- ``search_extent_deltas``
- ``movement_cell_size``
- ``cross_correlation_threshold_movement``

#### Methods

__str__()

Human-readable summary.

to_dict()

Keys expected by ImagePair(parameter_dict=...).

### FilterParameters
Container for filtering parameters.

#### Parameters
``parameter_dict`` : dict
    Source dict of parameters.

#### Fields

- ``level_of_detection_quantile``
- ``number_of_points_for_level_of_detection``
- ``difference_movement_bearing_threshold``
- ``difference_movement_bearing_moving_window_size``
- ``standard_deviation_movement_bearing_threshold``
- ``standard_deviation_movement_bearing_moving_window_size``
- ``difference_movement_rate_threshold``
- ``difference_movement_rate_moving_window_size``
- ``standard_deviation_movement_rate_threshold``
- ``standard_deviation_movement_rate_moving_window_size``

#### Methods

__str__()

Human-readable summary.

## Module: CreateGeometries/HandleGeometries.py
### get_submatrix_symmetric(central_index, shape, matrix)
Extracts a symmetric submatrix centered at `central_index`.

#### Parameters
``central_index`` : list or array
    [row, col] for the center pixel.

``shape`` : list or tuple
    [height, width] of the submatrix. Even sizes are reduced by 1.

``matrix`` : np.ndarray
    2D or 3D (channels-first) array.

#### Returns
``submatrix`` : np.ndarray
    Extracted submatrix.

### grid_points_on_polygon_by_distance(polygon, distance_of_points=10, distance_px=None, pixel_size=None)
Creates an evenly spaced grid of points inside a polygon at a given spacing.

#### Parameters
``polygon`` : gpd.GeoDataFrame
    Single-polygon GeoDataFrame.

``distance_of_points`` : float
    Desired spacing in CRS units.

``distance_px`` : float or None
    Optional pixel spacing for logging.

``pixel_size`` : float or None
    Pixel size in CRS units (for logging).

#### Returns
``points`` : gpd.GeoDataFrame
    Grid points inside the polygon.

### random_points_on_polygon_by_number(polygon, number_of_points)
Creates randomly distributed points inside a polygon.

#### Parameters
``polygon`` : gpd.GeoDataFrame
    Single-polygon GeoDataFrame.

``number_of_points`` : int
    Number of points to generate.

#### Returns
``points`` : gpd.GeoDataFrame
    Random points inside the polygon.

### get_raster_indices_from_points(points, raster_matrix_transform)
Converts point coordinates to raster row/column indices.

#### Parameters
``points`` : gpd.GeoDataFrame
    Points in CRS coordinates.

``raster_matrix_transform`` : Affine
    Raster transform.

#### Returns
``rows``, ``cols`` : list
    Row/column indices for points.

### crop_images_to_intersection(file1, file2)
Crops two rasters to their spatial intersection and returns arrays + transforms.

#### Parameters
``file1``, ``file2``
    Rasterio-opened datasets.

#### Returns
[``array_file1``, ``array_file1_transform``], [``array_file2``, ``array_file2_transform``]
    Cropped arrays and transforms.

### georeference_tracked_points(tracked_pixels, raster_transform, crs, years_between_observations=1)
Converts tracked pixel offsets into georeferenced movement vectors and yearly rates.

#### Parameters
``tracked_pixels`` : pd.DataFrame
    Must include row/column and movement direction fields.

``raster_transform`` : Affine
    Raster transform.

``crs`` : any
    CRS identifier.

``years_between_observations`` : float
    Time span between images.

#### Returns
``georeferenced_tracked_pixels`` : gpd.GeoDataFrame
    GeoDataFrame with movement distance and movement per year.

### circular_std_deg(angles_deg)
Computes circular standard deviation (degrees).

#### Parameters
``angles_deg`` : array-like
    Angles in degrees.

#### Returns
``std_deg`` : float
    Circular standard deviation.

### grid_points_on_polygon_by_distance(polygon, distance_of_points=10, distance_px=None, pixel_size=None)
Creates an evenly spaced grid of points inside a polygon at a given spacing.

#### Parameters
``polygon`` : gpd.GeoDataFrame
    Single-polygon GeoDataFrame.

``distance_of_points`` : float
    Desired spacing in CRS units.

``distance_px`` : float or None
    Optional pixel spacing for logging.

``pixel_size`` : float or None
    Pixel size in CRS units (for logging).

#### Returns
``points`` : gpd.GeoDataFrame
    Grid points inside the polygon.

### random_points_on_polygon_by_number(polygon, number_of_points)
Creates randomly distributed points inside a polygon.

#### Parameters
``polygon`` : gpd.GeoDataFrame
    Single-polygon GeoDataFrame.

``number_of_points`` : int
    Number of points to generate.

#### Returns
``points`` : gpd.GeoDataFrame
    Random points inside the polygon.

### get_raster_indices_from_points(points, raster_matrix_transform)
Converts point coordinates to raster row/column indices.

#### Parameters
``points`` : gpd.GeoDataFrame
    Points in CRS coordinates.

``raster_matrix_transform`` : Affine
    Raster transform.

#### Returns
``rows``, ``cols`` : list
    Row/column indices for points.

### crop_images_to_intersection(file1, file2)
Crops two rasters to their spatial intersection and returns arrays + transforms.

#### Parameters
``file1``, ``file2``
    Rasterio-opened datasets.

#### Returns
[``array_file1``, ``array_file1_transform``], [``array_file2``, ``array_file2_transform``]
    Cropped arrays and transforms.

### georeference_tracked_points(tracked_pixels, raster_transform, crs, years_between_observations=1)
Converts tracked pixel offsets into georeferenced movement vectors and yearly rates.

#### Parameters
``tracked_pixels`` : pd.DataFrame
    Must include row/column and movement direction fields.

``raster_transform`` : Affine
    Raster transform.

``crs`` : any
    CRS identifier.

``years_between_observations`` : float
    Time span between images.

#### Returns
``georeferenced_tracked_pixels`` : gpd.GeoDataFrame
    GeoDataFrame with movement distance and movement per year.

### circular_std_deg(angles_deg)
Computes circular standard deviation (degrees).

#### Parameters
``angles_deg`` : array-like
    Angles in degrees.

#### Returns
``std_deg`` : float
    Circular standard deviation.

### get_submatrix_rect_from_extents(central_index, extents, matrix)
Extracts an asymmetric rectangular window given extents (posx, negx, posy, negy).

#### Parameters
``central_index`` : list or array
    [row, col] center.

``extents`` : tuple
    (posx, negx, posy, negy) in pixels.

``matrix`` : np.ndarray
    2D or 3D array.

#### Returns
``submatrix`` : np.ndarray
    Extracted window.

``center_in_submatrix`` : tuple
    (row, col) of original center inside the submatrix.


## Module: CreateGeometries/DepthImageConversion.py

This module provides utilities to convert pixel coordinates and depth rasters into 3D positions and to compute 3D
displacements from tracked points and depth images.
### calculate_3d_position_from_depth_image(points, depth_image, camera_intrinsics_matrix, camera_to_3d_coordinates_transform=None)
Transform 2D image pixel coordinates with corresponding depth values into 3D coordinates.

#### Parameters
`points`: numpy.ndarray, shape (n, 2)
    An array of image pixel coordinates in the format (row, column).

`depth_image`: numpy.ndarray, shape (H, W)
    Single-band depth raster giving Z (distance along the camera optical axis) per pixel in consistent units (e.g.,
meters). Indexing uses [row, column].

`camera_intrinsics_matrix`: numpy.ndarray, shape (3, 3)
    Intrinsic camera matrix in row-major form: [[f_x, s, c_x], [0, f_y, c_y], [0, 0, 1 ]]

`camera_to_3d_coordinates_transform`: numpy.ndarray, shape (4, 4), optional
    Optional homogeneous transform mapping camera coordinates to a desired 3D coordinate system. Expected layout (row-major): [[R (3×3), t (3×1)], [0 0 0, 1 ]]

Returns

`points_transformed`: numpy.ndarray, shape (n, 4)
An n×3 array containing corresponding 3D coordinates: columns are [x, y, z]. Coordinate sign conventions:
        X aligns with image columns (increasing to the right).
        Y is set so that positive Y points upwards (computed as -row-direction × Z).
        Z is along the camera optical axis (distance from the camera).


### calculate_displacement_from_depth_images(tracked_points, depth_image_time1, depth_image_time2, camera_intrinsics_matrix, years_between_observations, camera_to_3d_coordinates_transform=None)
 Compute 3D displacements and annualized velocities for tracked points using two depth images.

#### Parameters

 `tracked_points`: pd.DataFrame with at least the following columns:

- "row", "column": pixel coordinates of the point at time1
- "movement_row_direction", "movement_column_direction": float offsets from time1 to time2 in pixel units (as returned by
tracking functions such as track_movement_lsm)

`depth_image_time1`: numpy.ndarray, shape (H, W)
    Single-band depth raster giving Z (distance along the camera optical axis) per pixel in consistent units (e.g.,
meters) for the first time point of tracking.

`depth_image_time2`: numpy.ndarray, shape (H, W)
    Single-band depth raster giving Z (distance along the camera optical axis) per pixel in consistent units (e.g.,
meters) for the second time point of tracking.

`camera_intrinsics_matrix`: numpy.ndarray, shape (3, 3)
    Intrinsic camera matrix in row-major form: [[f_x, s, c_x], [0, f_y, c_y], [0, 0, 1 ]]

`camera_to_3d_coordinates_transform`: numpy.ndarray, shape (4, 4), optional
    Optional homogeneous transform mapping camera coordinates to a desired 3D coordinate system. Expected layout (row-major): [[R (3×3), t (3×1)], [0 0 0, 1 ]]


#### Returns

`georeferenced_tracked_pixels`: geopandas.GeoDataFrame with the following important columns:
- "3d_displacement_distance": Euclidean length of the 3D displacement (units of depth image, e.g., meters)
- "3d_displacement_distance_per_year": the above value divided by years_between_observations
- "x","y","z": 3D coordinates (from time1) in the coordinate system given by camera_to_3d_coordinates_transform
if provided, otherwise in camera coordinates
- "valid": boolean; points with NaN displacement_distance are marked invalid
- geometry: created as gpd.points_from_xy(x=column, y=-row). Note the negative sign for y so that geometries follow the convention of y increasing upwards.

### Note

    Depth images should encode distances along the optical axis (Z). Use consistent units (e.g., meters).
    Missing or invalid depths should be encoded as NaN. The function marks points with NaN displacements as invalid in the returned GeoDataFrame.
    Depth values ≤ 0 are not explicitly handled by the function and may lead to unexpected results; clean or mask invalid depths prior to invocation.


## Module: ImageTracking/TrackingResults.py
### class TrackingResults
Represents the result of tracking a single point/cell.

#### __init__(movement_rows, movement_cols, tracking_method, transformation_matrix=None, cross_correlation_coefficient=None, tracking_success=False)

#### Parameters
``movement_rows`` : float
    Movement in row direction.

``movement_cols`` : float
    Movement in column direction.

``tracking_method`` : str
    Tracking method used (e.g., "lsm", "cc").

``transformation_matrix`` : np.ndarray or None
    Affine transformation matrix (LSM only).

``cross_correlation_coefficient`` : float or None
    Cross-correlation coefficient.

``tracking_success`` : bool
    Whether tracking was successful.

#### Fields
``movement_rows`` : float
    Movement in row direction.

``movement_cols`` : float
    Movement in column direction.

``tracking_method`` : str
    Tracking method used.

``transformation_matrix`` : np.ndarray or None
    Affine transformation matrix (LSM only).

``cross_correlation_coefficient`` : float or None
    Cross-correlation coefficient.

``tracking_success`` : bool
    Whether tracking was successful.

## Module: ImageTracking/TrackMovement.py
### track_cell_cc(tracked_cell_matrix, search_cell_matrix, search_center=None)
Cross-correlation tracking for a single cell inside a search window.

#### Parameters
``tracked_cell_matrix`` : np.ndarray
    Template cell from image1.

``search_cell_matrix`` : np.ndarray
    Search window from image2.

``search_center`` : list or None
    Optional logical center of the search window (for asymmetric extents).

#### Returns
``tracking_results`` : TrackingResults
    Movement in rows/cols and correlation coefficient.

### move_indices_from_transformation_matrix(transformation_matrix, indices)
Applies an affine transform (2x3) to a set of row/column indices.

#### Parameters
``transformation_matrix`` : np.ndarray
    2x3 affine transform.

``indices`` : np.ndarray
    2xn array of indices.

#### Returns
``moved_indices`` : np.ndarray
    2xn array after transform.

### track_cell_lsm(tracked_cell_matrix, search_cell_matrix, initial_shift_values=None, search_center=None)
Least-squares tracking for a single cell with optional initial shift.

#### Parameters
``tracked_cell_matrix`` : np.ndarray
    Template cell.

``search_cell_matrix`` : np.ndarray
    Search window.

``initial_shift_values`` : np.ndarray or None
    Initial movement estimates.

``search_center`` : list or None
    Optional logical center of the search window (for asymmetric extents).

#### Returns
``tracking_results`` : TrackingResults
    Shift estimates and correlation coefficient; invalid results return NaNs.

### track_cell_lsm_parallelized(central_index, shm1_name, shm2_name, shape1, shape2, dtype, tracked_cell_size, control_search_extents=None, search_extents=None)
Multiprocessing helper that tracks a single cell using shared memory for the full image matrices and parameters.

#### Parameters
``central_index`` : np.ndarray
    Central index to track.

``shm1_name`` : str
    Shared memory name for image1.

``shm2_name`` : str
    Shared memory name for image2.

``shape1`` : tuple
    Shape of image1.

``shape2`` : tuple
    Shape of image2.

``dtype`` : type
    Data type of images.

``tracked_cell_size`` : int
    Size of the tracked cell.

``control_search_extents`` : tuple or None
    Control search extents (for alignment tracking).

``search_extents`` : tuple or None
    Search extents (for movement tracking).

#### Returns
``tracking_results`` : TrackingResults
    Result for the given index.

### track_movement_lsm(image1_matrix, image2_matrix, image_transform, points_to_be_tracked, tracking_parameters=None, alignment_parameters=None, alignment_tracking=False, save_columns=None, task_label="Tracking points")
Tracks a set of points using the least-squares approach.

#### Parameters
``image1_matrix`` : np.ndarray
    First observation (2D or 3D).

``image2_matrix`` : np.ndarray
    Second observation (same shape as image1_matrix).

``image_transform`` : Affine
    Shared transform for both matrices.

``points_to_be_tracked`` : gpd.GeoDataFrame
    Point locations to track.

``tracking_parameters`` : TrackingParameters
    Parameters for movement tracking.

``alignment_parameters`` : AlignmentParameters
    Parameters for alignment tracking.

``alignment_tracking`` : bool
    If True, uses alignment parameters and control-search extents.

``save_columns`` : list[str] or None
    Columns to include in output. Defaults to movement/bearing fields.

``task_label`` : str
    Progress label for the tqdm bar.

#### Returns
``tracked_pixels`` : pd.DataFrame
    Movement vectors and correlation coefficients (filtered by threshold).

## Module: ImageTracking/AlignImages.py
### align_images_lsm_scarce(image1_matrix, image2_matrix, image_transform, reference_area, alignment_parameters)
Aligns image2 to image1 using least-squares tracking on control points within a reference area.

#### Parameters
``image1_matrix`` : np.ndarray
    Reference image.

``image2_matrix`` : np.ndarray
    Image to be aligned.

``image_transform`` : Affine
    Transform for image1.

``reference_area`` : gpd.GeoDataFrame
    Stable area polygon.

``alignment_parameters`` : AlignmentParameters
    Alignment parameters.

#### Returns
``image1_matrix`` : np.ndarray
``moved_image2_matrix`` : np.ndarray
``tracked_control_points`` : pd.DataFrame
    Control points and tracking info.

## Module: ImageTracking/ImagePair.py
### class ImagePair
Encapsulates a pair of images and the full alignment/tracking/filtering workflow.

#### __init__(parameter_dict: dict = None)
Parameters are read from `parameter_dict`. Common keys:
- ``use_no_georeferencing``
- ``fake_pixel_size``
- ``downsample_factor``
- ``convert_to_3d_displacement``       # when true, compute 3D displacements using depth rasters
- ``undistort_image``                 # if true, undistort both image and depth rasters using camera intrinsics
- ``camera_intrinsics_matrix``        # 3x3 matrix (if undistortion or 3D conversion is enabled)
- ``camera_distortion_coefficients``  # 2- or 4-element array (OpenCV format)
- ``camera_to_3d_coordinates_transform``  # optional 4x4 homogeneous transform for output coordinates
- ``enhancement_type``                # image enhancement type ("clahe", "none")
- ``enhancement_kernel_size``         # kernel size for CLAHE
- ``enhancement_clip_limit``          # clip limit for CLAHE

#### _effective_pixel_size() -> float
Returns CRS units per pixel (assumes square pixels).

#### _downsample_array(arr, factor) -> np.ndarray
Decimates a 2D or 3D array by integer `factor`.

#### _downsample_transform(transform, factor)
Scales an affine transform by `factor` for downsampling.

#### select_image_channels(selected_channels=None)
Selects bands for tracking. Default uses first three channels.

#### Parameters
``selected_channels`` : list[int] or int or None
    Channels to use for tracking. If None, uses [0, 1, 2].

#### Returns
``None``

#### load_images_from_file(filename_1, observation_date_1, filename_2, observation_date_2, selected_channels=None, NA_value=None)
Loads and crops images to the intersection, handles fake georeferencing and downsampling, and sets bounds. When operating with fake georeferencing and `convert_to_3d_displacement` enabled, the function will attempt to read corresponding depth rasters using the naming convention described above.

#### Parameters
``filename_1``, ``filename_2`` : str
    File paths.

``observation_date_1``, ``observation_date_2`` : str
    Dates for the observations. Supported formats (ISO standard, year-first):
    Year only: 2024, 24
    Year-Month: 2024-09, 2024_09, 202409, 24-09, 24_09, 2409
    Year-Month-Day: 2024-09-01, 2024_09_01, 20240901, 24-09-01, 24_09_01, 240901
    With time: 2024-09-01-14-30-45, 2024_09_01_14_30_45, 20240901143045
    Separators: '-', '_', or none. Missing or invalid parts default to: month=01, day=01, hour=00, minute=00, second=00.
    Invalid parts (e.g., month=13, day=47) are ignored and replaced with defaults.

``selected_channels`` : list[int] or int or None
    Channels to use for tracking.

``NA_value`` : float or None
    If provided, set that value to 0 in both images.

#### Returns
``None``

#### load_images_from_matrix_and_transform(image1_matrix, observation_date_1, image2_matrix, observation_date_2, image_transform, crs, selected_channels=None)
Loads pre-supplied matrices with a shared transform and CRS.

#### Parameters
``image1_matrix``, ``image2_matrix`` : np.ndarray
    Image matrices.

``observation_date_1``, ``observation_date_2`` : str
    Observation dates. Supported formats (ISO standard, year-first):
    Year only: 2024, 24
    Year-Month: 2024-09, 2024_09, 202409, 24-09, 24_09, 2409
    Year-Month-Day: 2024-09-01, 2024_09_01, 20240901, 24-09-01, 24_09_01, 240901
    With time: 2024-09-01-14-30-45, 2024_09_01_14_30_45, 20240901143045
    Separators: '-', '_', or none. Missing or invalid parts default to: month=01, day=01, hour=00, minute=00, second=00.
    Invalid parts (e.g., month=13, day=47) are ignored and replaced with defaults.

``image_transform`` : Affine
    Shared transform for both matrices.

``crs`` : any
    Coordinate reference system.

``selected_channels`` : list[int] or int or None
    Channels to use for tracking.

#### Returns
``None``

#### align_images(reference_area)
Aligns the two images based on a reference area; updates `image2_matrix` and `image2_transform`.

#### Parameters
``reference_area`` : gpd.GeoDataFrame
    Stable area polygon.

#### Returns
``None``

#### compute_truecolor_aligned_from_control_points()
Builds a true-color aligned version of image2 using alignment control points.

#### Returns
``None``
    Sets `self.image2_matrix_truecolor` with the true-color aligned image.

#### track_points(tracking_area)
Creates a grid of points within the tracking area and tracks movement.

#### Parameters
``tracking_area`` : gpd.GeoDataFrame
    Moving area polygon.

#### Returns
``georeferenced_tracked_points`` : gpd.GeoDataFrame
    Points with movement rates and bearings.

#### perform_point_tracking(reference_area, tracking_area)
Aligns (if needed) and tracks points, storing results in `self.tracking_results`.

#### Parameters
``reference_area`` : gpd.GeoDataFrame
``tracking_area`` : gpd.GeoDataFrame

#### Returns
``None``

#### plot_images()
Plots the two raster images.

#### plot_tracking_results()
Plots movement vectors on the first image.

#### plot_tracking_results_with_valid_mask()
Plots valid vs invalid points on the first image. This method is called when `display_plots = true` in the `[output]` section of the config file to show interactive plots during processing.

#### filter_outliers(filter_parameters)
Applies outlier filtering using FilterParameters.

#### Parameters
``filter_parameters`` : FilterParameters

#### Returns
``None``

#### track_lod_points(points_for_lod_calculation, years_between_observations)
Tracks random points in a stable area for LoD calculation.

#### Parameters
``points_for_lod_calculation`` : gpd.GeoDataFrame
    Points for LoD calculation.

``years_between_observations`` : float
    Time span between observations in years.

#### Returns
``tracked_points`` : gpd.GeoDataFrame
    Tracked points for LoD.

#### calculate_lod(points_for_lod_calculation, filter_parameters=None)
Computes the level of detection (LoD) from random points in a stable area.

#### Parameters
``points_for_lod_calculation`` : gpd.GeoDataFrame
    Points for LoD calculation.

``filter_parameters`` : FilterParameters or None
    Filter parameters for LoD calculation.

#### Returns
``None``
    Sets `self.level_of_detection` with the computed LoD value.

#### filter_lod_points()
Marks points below LoD as invalid.

#### Returns
``None``

#### full_filter(reference_area, filter_parameters)
Runs outlier filtering, LoD calculation, and LoD filtering.

#### Parameters
``reference_area`` : gpd.GeoDataFrame
``filter_parameters`` : FilterParameters

#### Returns
``None``

#### equalize_adapthist_images()
Applies CLAHE to both images.

#### Returns
``None``

#### save_full_results(folder_path: str, save_files: list) -> None
Writes tracking outputs (GeoJSON, GeoTIFFs, masks, stats) into `folder_path`.

#### Parameters
``folder_path`` : str
    Output directory.

``save_files`` : list
    Tokens controlling which outputs are written. Supported tokens include:

- "L0_movement-bearing_raw_tif", "L0_movement-rate_raw_tif"
- "L1_movement-bearing_above-LoD_tif", "L1_movement-rate_above-LoD_tif"
- "L2_movement-bearing_above-LoD_filtered_tif", "L2_movement-rate_above-LoD_filtered_tif"
- "tracking_results_fgb", "tracking_results_figure_jpg", "statistical_results_txt"
- "mask_invalid_tif", "mask_LoD_tif"
- "mask_outlier_md_tif", "mask_outlier_msd_tif", "mask_outlier_bd_tif", "mask_outlier_bsd_tif"
- "parameters_txt", "first_image_matrix_jpg", "second_image_matrix_jpg"
- "LoD_points_fgb", "alignment_points_fgb"

#### Returns
``None``

#### load_results(file_path, reference_area)
Loads saved tracking results and aligns images to a reference area.

#### Parameters
``file_path`` : str
    Path to the saved tracking results GeoJSON file.

``reference_area`` : gpd.GeoDataFrame
    Reference area polygon for alignment.

#### Returns
``None``
    Loads tracking results into `self.tracking_results` and aligns images if needed.

## Module: DataProcessing/ImagePreprocessing.py
### undistort_camera_image(image_matrix, camera_intrinsic_matrix, distortion_coefficients)
Undistorts a camera image using OpenCV and returns the undistorted image cropped to a rectangular shape containing only valid pixels.

#### Parameters
``image_matrix`` : np.ndarray
    The array representing the distorted image.

``camera_intrinsic_matrix`` : np.ndarray
    The intrinsic matrix of the camera. Format: [[f_x, s, c_x], [0, f_y, c_y], [0, 0, 1]].

``distortion_coefficients`` : np.ndarray
    Distortion coefficients of the camera as a one-dimensional array. Format: [k1, k2] (radial) or [k1, k2, p1, p2] (radial + tangential).

#### Returns
``image_matrix_undistorted`` : np.ndarray
    The undistorted image matrix, cropped to a rectangular shape where all pixels are valid.
### equalize_adapthist_images(image_matrix, kernel_size, clip_limit)
Applies CLAHE (adaptive histogram equalization) using scikit-image.

#### Parameters
``image_matrix`` : np.ndarray
    Input image matrix.

``kernel_size`` : int
    Size of the grid for histogram equalization.

``clip_limit`` : float
    Contrast limiting threshold.

#### Returns
``equalized_image`` : np.ndarray
    Equalized image.

## Module: DataProcessing/DataPostprocessing.py
### calculate_lod_points(image1_matrix, image2_matrix, image_transform, points_for_lod_calculation,tracking_parameters, crs, years_between_observations)
Tracks random points to estimate LoD (level of detection).

#### Parameters
``image1_matrix``, image2_matrix : np.ndarray

``image_transform`` : Affine

``points_for_lod_calculation`` : gpd.GeoDataFrame

``tracking_parameters`` : TrackingParameters

``crs`` : any

``years_between_observations`` : float

#### Returns
``tracked_points`` : gpd.GeoDataFrame
    Tracked points for LoD.

### _ensure_bool_col(df, col)
Ensures a boolean column exists (internal helper).

#### Parameters
``df`` : pd.DataFrame

``col`` : str

#### Returns
``col_values`` : np.ndarray

### filter_lod_points(tracking_results, level_of_detection, displacement_column_name)
Marks points below the LoD as invalid.

#### Parameters
``tracking_results`` : gpd.GeoDataFrame
    Tracking results GeoDataFrame.

``level_of_detection`` : float
    Level of detection threshold.

``displacement_column_name`` : str
    Name of the displacement column ('movement_distance_per_year' for georeferenced images, '3d_displacement_distance_per_year' for non-georeferenced images with 3D displacements).

#### Returns
``tracking_results`` : gpd.GeoDataFrame
    Updated tracking results with points below LoD marked as invalid.

### filter_outliers_movement_bearing_difference(tracking_results, filter_parameters)
Removes outliers based on bearing difference vs local neighborhood.

#### Parameters
``tracking_results`` : gpd.GeoDataFrame
    Tracking results GeoDataFrame.

``filter_parameters`` : FilterParameters
    Filter parameters containing threshold and window size.

#### Returns
``tracking_results`` : gpd.GeoDataFrame
    Updated tracking results with outliers marked as invalid.

### filter_outliers_movement_bearing_standard_deviation(tracking_results, filter_parameters)
Removes outliers based on bearing standard deviation in a neighborhood.

#### Parameters
``tracking_results`` : gpd.GeoDataFrame
    Tracking results GeoDataFrame.

``filter_parameters`` : FilterParameters
    Filter parameters containing threshold and window size.

#### Returns
``tracking_results`` : gpd.GeoDataFrame
    Updated tracking results with outliers marked as invalid.

### filter_outliers_movement_rate_difference(tracking_results, filter_parameters, displacement_column_name)
Removes outliers based on movement rate difference vs local neighborhood.

#### Parameters
``tracking_results`` : gpd.GeoDataFrame
    Tracking results GeoDataFrame.

``filter_parameters`` : FilterParameters
    Filter parameters containing threshold and window size.

``displacement_column_name`` : str
    Name of the displacement column ('movement_distance_per_year' for georeferenced images, '3d_displacement_distance_per_year' for non-georeferenced images with 3D displacements).

#### Returns
``tracking_results`` : gpd.GeoDataFrame
    Updated tracking results with outliers marked as invalid.

### filter_outliers_movement_rate_mad(tracking_results, filter_parameters, displacement_column_name)
Removes outliers based on a modified Z-score approach using the median and Median Absolute Deviation (MAD) of movement rates in a neighborhood. A point is marked as an outlier if its modified Z-score (absolute deviation from the median divided by MAD) exceeds the threshold.

#### Parameters
``tracking_results`` : gpd.GeoDataFrame
    Tracking results GeoDataFrame.

``filter_parameters`` : FilterParameters
    Filter parameters where the threshold represents the number of MADs (modified Z-score) above which a point is considered an outlier. Common values are 2 or 3.

``displacement_column_name`` : str
    Name of the displacement column ('movement_distance_per_year' for georeferenced images, '3d_displacement_distance_per_year' for non-georeferenced images with 3D displacements).

#### Returns
``tracking_results`` : gpd.GeoDataFrame
    Updated tracking results with outliers marked as invalid.

#### Notes
- The threshold parameter represents the modified Z-score, not a raw movement rate value.
- For example, with threshold=2, points more than 2 MADs from the neighborhood median are filtered.
- If the neighborhood MAD is 0, no points are filtered (to avoid division by zero).
- MAD is more robust to outliers than standard deviation, making this filter less sensitive to extreme values.

### filter_outliers_movement_rate_standard_deviation(tracking_results, filter_parameters, displacement_column_name)
**DEPRECATED**: This function has been replaced by `filter_outliers_movement_rate_mad` for improved robustness to outliers. The old implementation using mean and standard deviation is kept commented out in the source code for reference.

Removes outliers based on a Z-score approach using the mean and standard deviation of movement rates in a neighborhood. A point is marked as an outlier if its Z-score (absolute deviation from the mean divided by the standard deviation) exceeds the threshold.

#### Parameters
``tracking_results`` : gpd.GeoDataFrame
    Tracking results GeoDataFrame.

``filter_parameters`` : FilterParameters
    Filter parameters where the threshold represents the number of standard deviations (Z-score) above which a point is considered an outlier. Common values are 2 or 3.

``displacement_column_name`` : str
    Name of the displacement column ('movement_distance_per_year' for georeferenced images, '3d_displacement_distance_per_year' for non-georeferenced images with 3D displacements).

#### Returns
``tracking_results`` : gpd.GeoDataFrame
    Updated tracking results with outliers marked as invalid.

#### Notes
- The threshold parameter represents the Z-score, not a raw movement rate value.
- For example, with threshold=2, points more than 2 standard deviations from the neighborhood mean are filtered.
- If the neighborhood standard deviation is 0, no points are filtered (to avoid division by zero).
- This filter is more sensitive to outliers than the MAD-based filter.

### filter_outliers_full(tracking_results, filter_parameters, displacement_column_name)
Applies all outlier filters in sequence.

#### Parameters
``tracking_results`` : gpd.GeoDataFrame
    Tracking results GeoDataFrame.

``filter_parameters`` : FilterParameters
    Filter parameters.

``displacement_column_name`` : str
    Name of the displacement column ('movement_distance_per_year' for georeferenced images, '3d_displacement_distance_per_year' for non-georeferenced images with 3D displacements).

#### Returns
``tracking_results`` : gpd.GeoDataFrame
    Updated tracking results with all outlier filters applied.

#### Note on Filtering Order
The pipeline applies filters in the following order:
1. **Level of Detection (LoD) filtering**: Points with movement below the detection threshold are marked as invalid first. This removes noise before statistical calculations.
2. **Outlier filtering**: Statistical filters (bearing difference, bearing standard deviation, rate difference, rate standard deviation) are then applied to the remaining points above the LoD threshold.

## Module: Plots/MakePlots.py
### plot_raster_and_geometry(raster_matrix, raster_transform, geometry, alpha=0.6)
Plots a raster with a geometry overlay.

#### Parameters
``raster_matrix`` : np.ndarray

``raster_transform`` : Affine

``geometry`` : gpd.GeoDataFrame

``alpha`` : float

#### Returns
``None``

### plot_movement_of_points(raster_matrix, raster_transform, point_movement, point_color=None, masking_polygon=None,
fig=None, ax=None, save_path=None, show_arrows=True)
Plots tracked point movement with optional masking and saving.

#### Parameters
``raster_matrix`` : np.ndarray

``raster_transform`` : Affine

``point_movement`` : gpd.GeoDataFrame

``point_color`` : str or None

``masking_polygon`` : gpd.GeoDataFrame or None

``fig``, ``ax`` : matplotlib objects or None

``save_path`` : str or None

``show_arrows`` : bool

#### Returns
``None``

### plot_movement_of_points_with_valid_mask(raster_matrix, raster_transform, point_movement, save_path=None)
Plots valid vs invalid points in different colors. When `save_path` is provided, saves the plot to a file. When `save_path` is None, displays the plot interactively (requires a graphical display).

#### Parameters
``raster_matrix`` : np.ndarray

``raster_transform`` : Affine

``point_movement`` : gpd.GeoDataFrame

``save_path`` : str or None
    If provided, saves the plot to this path. If None, displays the plot interactively.

#### Returns
``None``

### plot_distribution_of_point_movement(moving_points)
Scatter plot of movement row vs column directions.

#### Parameters
``moving_points`` : gpd.GeoDataFrame

#### Returns
``None``

## Package __init__.py
The top-level `src/PyImageTrack/__init__.py` is a namespace package wrapper. It optionally appends a
`PyImageTrack_scripts` subdirectory to `__path__` to support legacy imports.
