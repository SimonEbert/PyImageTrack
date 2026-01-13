# PyImageTrack
PyImageTrack is a Python library implementing feature tracking approaches based on the normalized cross-correlation and least-squares matching for usage on rock glaciers.
## Features
- Image alignment based on a reference area
- Creation of a grid of track points
- Feature tracking using the normalized cross-correlation or least-squares matching methods with symmetric and asymmetric search windows
- Visualization of movement data
- Calculation of the Level of Detection of a performed tracking
- Removing outliers based on movement bearing and movement rate in the surrounding area


## Acknowledgment
The code in this respository is written by Lisa Rehn and Simon Ebert and maintained by Lisa Rehn. Its first version is based on the master thesis "Comparison and Python Implementation of Different Image Tracking Approaches Using the Example of the Kaiserberg Rock Glacier" by Simon Ebert.

## Usage
Pipeline runs are configured via TOML files in `configs/`.
Run the pipeline via the CLI: `pyimagetrack-run --config configs/your_config.toml`.

```

Notes:
- If you use Python < 3.11, install `tomli` (see `src/PyImageTrack/requirements.txt`).
- Use `[downsampling]` in your config to speed up smoke tests (`downsample_factor = 4`).


## License
This project is licensed under the Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0) License.
