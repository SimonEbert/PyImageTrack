# PyImageTrack
PyImageTrack is a Python library implementing feature tracking approaches based on the normalized cross-correlation and least-squares matching for usage on rock glaciers. A documentation of implemented functions can be seen in "documentation.pdf".

## Features

- Image alignment based on a reference area
- Creation of a grid of track points
- Feature tracking using the normalized cross-correlation or least-squares
- removal of outliers based on adjacent tracked points
- Visualization of movement data
- Support for multi-channel images

## Acknowledgment
The code in this repository corresponds to the master thesis "Comparison and Python Implementation of Different Image Tracking Approaches	Using the Example of the Kaiserberg Rock Glacier" by Simon Ebert from October 2024 to February 2025 at the University of Freiburg.

## Installation
To install PyImageTrack, follow these steps:
1. Clone the repository: `git clone https://github.com/SimonEbert/PyImageTrack.git`
2. Navigate to the project directory: `cd PyImageTrack`
3. Install the required dependencies: `pip install -r requirements.txt`

## License
This project is licensed under the Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0) License.
