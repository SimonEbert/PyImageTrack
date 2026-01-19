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
- Download Python 3.11 (64-bit) from https://www.python.org/downloads/windows/ # make sure it is really the correct version of Python and not the latest release!
- During installation, check **"Add python.exe to PATH"**

Install Git:
- Download from https://git-scm.com/download/win
- Default installer settings are sufficient

Download the repository:
```powershell
cd %USERPROFILE%\Documents
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

## Input Files: Where They Go

Your config file (e.g. `configs/your_config.toml`) controls where input files are read from.
Check and update these entries:

- `paths.input_folder`: folder that contains the input images
- `paths.date_csv_path`: CSV file with image dates (or `"none"` if not used)
- `paths.pairs_csv_path`: CSV file with image pairs (or `"none"` if not used)
- `polygons.inside_filename` and `polygons.outside_filename`: these are just filenames, so the shapefiles must live inside `paths.input_folder`

Example (from `configs/example_config.toml`):
```toml
[paths]
input_folder = "../input/hillshades"
date_csv_path = "../input/hillshades/image_dates.csv"
pairs_csv_path = "../input/hillshades/image_pairs.csv"

[polygons]
outside_filename = "stable_area.shp"
inside_filename = "moving_area.shp"
```
