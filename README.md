# PyImageTrack
PyImageTrack is a Python library implementing feature tracking approaches based on the normalized cross-correlation and least-squares matching for usage on rock glaciers. A possible example how the library can be used on a very basic level is shown in "example.py" for single ImagePair matchings and "example2.py" for multiple ImagePairs. Of course, it is possible to also access the underlying functions directly.
If you encounter any bugs or would like features added, please contact me under my e-mail-address "simon.ebert.kn@gmx.de".

## Features

- Image alignment based on a reference area
- Creation of a grid of track points
- Feature tracking using the normalized cross-correlation or least-squares
- Visualization of movement data
- Calculation of the Level of Detection of a performed tracking
- Removing outliers based on movement bearing and movement rate in the surrounding area

### Planned

- Visualization adjustments
- Multichannel image support
- Pre- and post-processing of the data

## Acknowledgment
The code in this repository is based on the master thesis "Comparison and Python Implementation of Different Image Tracking Approaches Using the Example of the Kaiserberg Rock Glacier" by Simon Ebert from October 2024 to February 2025 at the University of Freiburg. It contains Code written by Simon Ebert and Lisa Rehn.

## Installation
To install PyImageTrack, follow these steps:
1. Clone the repository: `git clone https://github.com/SimonEbert/PyImageTrack.git`
2. Navigate to the project directory: `cd PyImageTrack`
3. Install the required dependencies: `pip install -r requirements.txt`

## License
This project is licensed under the Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0) License.
