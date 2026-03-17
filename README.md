# PyImageTrack

PyImageTrack is a Python library implementing feature tracking approaches based on the normalized cross-correlation and
least-squares matching for usage on rock glaciers.

## Features
- Image alignment based on a reference area
- Creation of a grid of track points
- Feature tracking using the normalized cross-correlation or least-squares matching methods with symmetric and
  asymmetric search windows
- Visualization of movement data
- Calculation of the Level of Detection of a performed tracking
- Removing outliers based on movement bearing and movement rate in the surrounding area
- Tracking on non-georeferenced images and giving the results in pixels
	--> For this the respective shapefiles must be in image coordinates and have no valid CRS. This can be achieved by
deleting the "filename.prj" file from the folder where the "filename.shp" file is stored
- Optional 3D displacement from depth images when working with non-georeferenced photos
- Full documentation: `docs/pyimagetrack_documentation.md`
- Absolute beginner installation + quickstart + input file layout: `docs/absolute_beginner_installation.md`
- Config templates: `configs/`

## Quick start (CLI)

Follow the steps for your platform.

### Linux / macOS
1) Create and activate a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
2) Install the package in editable mode from the repo root (installs all dependencies):
   ```bash
   pip install -e .
   ```
3) Run the pipeline with a config:
   ```bash
   pyimagetrack-run --config configs/your_config.toml
   ```

### Windows (PowerShell)
1) Create and activate a virtual environment:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\activate
   ```
2) Install the package in editable mode from the repo root (installs all dependencies):
   ```powershell
   pip install -e .
   ```
3) Run the pipeline with a config:
   ```powershell
   pyimagetrack-run --config configs/your_config.toml
   ```

Notes:
- Performance of PyImageTrack is better on Linux than Windows.
- Input filenames must contain a date token (e.g., `YYYY-MM-DD`, `YYYYMMDD`, or `YYYYMMDD-HHMMSS`). The date token can appear anywhere in the filename (e.g., `HS_2008_09_01.tif` will extract `2008_09_01` as the date).

## Batch Processing

For processing multiple configurations with automatic filtering based on identifiers, use the batch processing command:

```bash
pyimagetrack-batch \
  --table-a path/to/table_a.csv \
  --config-col config_name \
  --class-col-a class \
  --table-b path/to/table_b.csv \
  --identifier-col identifier \
  --class-col-b class
```

### Table A (Config Mapping)

Maps configuration files to classes.

**Required columns:**
- `config_name`: Path to the TOML configuration file
- `class`: Class name for grouping

**Example:**
```csv
config_name,class
configs/example_large.toml,large
configs/example_small.toml,medium
```

### Table B (Identifier Classification)

Maps identifiers to classes.

**Required columns:**
- `identifier`: Identifier extracted from filenames (pattern: `id<identifier>`)
- `class`: Class name for grouping

**Example:**
```csv
identifier,class
9109,large
9110,large
9111,small
```

### Filename Pattern for Batch Processing

**Important:** For batch processing to work, your input image filenames must contain an identifier in the format `id<identifier>`, followed by a date token. Examples:

- `id9109_2024-06-09.tif` → identifier: `9109`, date: `2024-06-09`
- `id9110_HS_2025-07-01.tif` → identifier: `9110`, date: `2025-07-01`
- `id9111_20240901.tif` → identifier: `9111`, date: `20240901`

The identifier can appear anywhere in the filename, but must be preceded by `id`. The date token is extracted as described in the "CSV File Formats" section below.

### How Batch Processing Works

1. Reads Table A and Table B
2. For each configuration in Table A:
   - Gets the class associated with the config
   - Filters Table B to find all identifiers with that class
   - Filters identifiers to only those present in the input folder
   - Processes each identifier separately using the configuration
   - Replaces wildcards (`*`) in shapefile names with the identifier

### Wildcard Support in Shapefiles

Configuration files support wildcard patterns for shapefile names. The `*` wildcard is replaced with the actual identifier during batch processing.

**Example configuration:**
```toml
[polygons]
stable_area_filename = "stable_area_*.shp"
moving_area_filename = "moving_area_*.shp"
```

For identifier `9109`, this will load `stable_area_9109.shp` and `moving_area_9109.shp`.

**Note:** Wildcard patterns are only valid when processing in batch mode with an identifier specified.

**Stable / Moving Areas:**
- Stable-area shapefiles can include one or multiple singlepart polygons; they are merged into one reference area for alignment.
- Moving-area shapefiles can include one or multiple polygons. Provide an ID column to report stats per polygon; missing IDs are filled with the row index.
- Use `polygons.moving_id_column` to choose the ID column name (default: `moving_id`).

## CSV File Formats

### image_dates.csv

This CSV file maps abbreviated date tokens from image names to full dates. When provided, the dates from this file **override** the dates extracted from filenames. This is useful when the filename contains an abbreviated date (e.g., `2024`) but you need to specify the full date (e.g., `2024-06-15`). If this file is not provided or set to `"none"`, dates are extracted directly from the filenames.

**Required columns:**
- `file` or `year` or `file/year`: The filename (without extension) or the date token as it appears in the image filename
- `date`: The date in any supported format (see "Supported Date Token Formats" below). Missing parts default to standard values (month=01, day=01, hour=00, minute=00, second=00).

**Example:**
```csv
file;date
HS_2024-06-09;2024-06-15
HS_2025-07-08;2025-07-08
```

Or using date tokens:
```csv
year;date
2018;2018-01-01
2019;2019-06-15
2020;2020-12-31
2024;2024-06        # Will be parsed as 2024-06-01
```

**Note:** The delimiter can be either comma (`,`) or semicolon (`;`).

### image_pairs.csv

This CSV file defines custom image pairs for tracking. It is required when `pairing.mode` is set to `"custom"`.

**Required columns:**
- `date_earlier`: The date token of the earlier image
- `date_later`: The date token of the later image

**Example:**
```csv
date_earlier;date_later
2024_07;2025_06
2024_07;2024_09
2024_09;2025_06
```

**Important notes:**
- The delimiter can be either comma (`,`) or semicolon (`;`)
- The date tokens must match the leading numeric tokens in your image filenames
- The system extracts the leading numeric token from filenames (e.g., `2024-06-09.tif` → `2024-06-09`)
- You can use abbreviated tokens if they uniquely identify the images (e.g., `24` for `2024-06-09.tif` if there's only one image from 2024)

### Supported Date Token Formats

Date tokens in filenames and CSV files are parsed flexibly:

| Format | Examples |
|--------|-----------|
| Year only | `2024`, `24` |
| Year-Month | `2024-09`, `2024_09`, `202409`, `24-09`, `24_09`, `2409` |
| Year-Month-Day | `2024-09-01`, `2024_09_01`, `20240901`, `24-09-01`, `24_09_01`, `240901` |
| With time | `2024-09-01-14-30-45`, `2024_09_01_14_30_45`, `20240901143045` |

**Separators:** `-`, `_`, or none

**Default values:** Missing parts default to: month=01, day=01, hour=00, minute=00, second=00

## Command-Line Options

The `pyimagetrack-run` command supports the following options:

- `--config` (required): Path to TOML configuration file
- `--verbose`: Enable verbose output with detailed information about processing steps
- `--quiet`: Enable quiet mode with minimal output (only essential messages)

### Examples

Run with verbose output:
```bash
pyimagetrack-run --config configs/your_config.toml --verbose
```

Run in quiet mode (minimal output):
```bash
pyimagetrack-run --config configs/your_config.toml --quiet
```

## Project layout

- Code lives under `src/` (src layout). This avoids accidental imports from the repo root
  and keeps project files (docs/configs) clearly separated from Python package code.

## Acknowledgment
The code in this respository is written by Lisa Rehn and Simon Ebert and maintained by Simon Ebert and Lisa Rehn. Its first version is
based on the master thesis "Comparison and Python Implementation of Different Image Tracking Approaches Using the Example
of the Kaiserberg Rock Glacier" by Simon Ebert.


## License
This project is licensed under the Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0) License.
