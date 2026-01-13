# PyImageTrack

This repository contains the PyImageTrack package and pipeline scripts.

- Package docs and usage: `src/PyImageTrack/README.md`
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
2) Install the package in editable mode from the repo root:
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
2) Install the package in editable mode from the repo root:
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

## Project layout

- Code lives under `src/` (src layout). This avoids accidental imports from the repo root
  and keeps project files (docs/configs) clearly separated from Python package code.
