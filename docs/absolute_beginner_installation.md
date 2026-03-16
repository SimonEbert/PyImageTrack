# Absolute Beginner Installation (CLI)

This guide explains how to install PyImageTrack from scratch.
It assumes no prior Python or Git experience.

---

## Linux (Ubuntu / Debian-based)

Install Python and Git:
```bash
sudo apt update
sudo apt upgrade # optional, confirm with "y"
sudo apt install -y python3 python3-venv python3-pip git
```

Download the repository:
```bash
cd ~
git clone https://github.com/lisasophie/PyImageTrack.git
cd PyImageTrack
```

Create and activate a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the package in editable mode (installs all dependencies):
```bash
pip install -e .
```

Run the pipeline with a config:
```bash
pyimagetrack-run --config configs/your_config.toml
```

---

## Windows (PowerShell)

Install Python:
- Download Python 3.11 (64-bit) from https://www.python.org/downloads/windows/. Make sure it is really the correct version of Python and not the latest release!
- During installation, check **"Add python.exe to PATH"**
- After successful installation, check **"Disable path length limit"**

Install Git:
- Download from https://git-scm.com/download/win
- Default installer settings are sufficient

Download the repository:
```powershell
cd "$env:USERPROFILE\Documents"
git clone https://github.com/lisasophie/PyImageTrack.git
cd PyImageTrack
```

Create and activate a virtual environment:
```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

If script execution is blocked, run once:
```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Install the package in editable mode (installs all dependencies):
```powershell
pip install -e .
```

Run the pipeline with a config:
```powershell
pyimagetrack-run --config configs\your_config.toml
```

---

## Quickstart (Every Run)

After installation, these are the steps you repeat each time you want to run PyImageTrack.

Linux:
```bash
cd ~/PyImageTrack
source .venv/bin/activate
pyimagetrack-run --config configs/your_config.toml
```

Windows (PowerShell):
```powershell
cd $env:USERPROFILE\Documents\PyImageTrack
.\.venv\Scripts\activate
pyimagetrack-run --config configs\your_config.toml
```

---

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

---

## Input Files: Where They Go

Your config file (e.g. `configs/your_config.toml`) controls where input files are read from.
Check and update these entries:

- `paths.input_folder`: folder that contains the input images
- `paths.date_csv_path`: CSV file with image dates (or `"none"` if not used)
- `paths.pairs_csv_path`: CSV file with image pairs (or `"none"` if not used)
- `polygons.moving_area_filename` and `polygons.stable_area_filename`: shapefiles with moving / stable area
- Stable-area shapefiles can contain one or multiple singlepart polygons; they are merged into one reference area for alignment.
- Moving-area shapefiles can contain one or multiple polygons. Provide an ID column to report stats per polygon; missing IDs are filled with the row index. Use `polygons.moving_id_column` to choose the ID column name (default: `moving_id`).
- **Optional fallback**: If `stable_area_filename` is set to `"none"` or the file doesn't exist, the system will automatically use `image_bounds minus moving_area` as the stable area. This assumes all areas outside the moving area are stable. Alignment quality may be slightly lower but can be improved by increasing `number_of_control_points`.

Example (from `configs/example_config.toml`):
```toml
[paths]
input_folder = "../input/hillshades"
date_csv_path = "../input/hillshades/image_dates.csv"
pairs_csv_path = "../input/hillshades/image_pairs.csv"

[polygons]
stable_area_filename = "stable_area.shp"
moving_area_filename = "moving_area.shp"
```

---

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
