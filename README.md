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
	--> For this the respective shapefiles must be in image coordinates and have no valid CRS. This can be achieved by deleting the "filename.prj" file from the folder where the "filename.shp" file is stored

- Full documentation: `docs/pyimagetrack_documentation.md`
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
   .\.venv\Scripts\Activate.ps1
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
- The package uses `pyproject.toml` (modern packaging standard) at the repo root.
- Editable install means code changes in `src/PyImageTrack/` take effect immediately.
- Use `pip install .` (no `-e`) if you want a fixed install from this repo.
- Dependencies are installed automatically when you use `pip install -e .` or `pip install .`.
- Use `[downsampling]` in your config to speed up smoke tests (`downsample_factor = 4`).

## Project layout

- Code lives under `src/` (src layout). This avoids accidental imports from the repo root
  and keeps project files (docs/configs) clearly separated from Python package code.

## Acknowledgment
The code in this respository is written and maintained by Lisa Rehn and Simon Ebert. Its first version is
based on the master thesis "Comparison and Python Implementation of Different Image Tracking Approaches Using the Example
of the Kaiserberg Rock Glacier" by Simon Ebert.


## License
This project is licensed under the Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0) License.
