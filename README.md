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
## Installation
To install PyImageTrack, follow these steps:
1. Clone the repository: `git clone https://github.com/SimonEbert/PyImageTrack.git`
2. Navigate to the project directory: `cd PyImageTrack`
3. Install the required dependencies: `pip install -r requirements.txt`

## Usage
Pipeline runs are configured via TOML files in `configs/`.

Example:
```
/home/lisa/projects/pyimagetrack/.venv/bin/python -m PyImageTrack.run_pipeline --config configs/drone_HS.toml
```

Notes:


## License
This project is licensed under the Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0) License.
