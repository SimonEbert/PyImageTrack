# PyImageTrack Documentation
2026-01-26

## Overview
PyImageTrack provides alignment and feature tracking for georeferenced imagery, with optional filtering and plotting. The main entry point is the CLI command `pyimagetrack-run` configured via TOML files in `configs/`.

This document documents the pipeline entry points, configuration options (including the new depth-image / 3D displacement functionality), and key modules and functions of the project.

## Running the Pipeline
```
pyimagetrack-run --config configs/drone_hs.toml
```
Config paths are resolved relative to the repo root. The CLI invokes `PyImageTrack.run_pipeline:main` under the hood.

## Entry point for Python Scripts
The package provides access to the tracking routine from Python using `run_from_config`:
```
from PyImageTrack import run_from_config
```
IMPORTANT: When calling the tracking routine from within another Python script or project, guard the call with:
```
if __name__ == "__main__":
    run_from_config(path_to_config)
```
This is required because the pipeline uses multiprocessing; without this guard the routine may be executed multiple times.

## Configuration Files (TOML)
Configs are TOML files and share a common structure. Templates are available under `configs/`:
- `configs/drone_hs.toml`
- `configs/smoketest_drone_hs.toml`
- `configs/timelapse_fake_ortho.toml`

### Key Sections
- `[paths]`: input/output folders, optional CSVs for dates/pairs.
- `[polygons]`: stable/moving area shapefiles and CRS.
- `[pairing]`: pairing mode (`all`, `first_to_all`, `successive`, `custom`).
- `[no_georef]`: enable for non-ortho images (e.g., JPGs) and configure fake georeferencing and depth-image options.
- `[downsampling]`: optional downsampling for fast smoke tests.
- `[flags]`: alignment/tracking/filtering/plotting toggles.
- `[cache]`: caching and recompute flags.
- `[output]`: optional outputs such as true-color alignment.
- `[adaptive_tracking_window]`: search extent scales by time span when enabled.
- `[alignment]`, `[tracking]`, `[filter]`: algorithm parameters.
- `[save]`: list of output files to write.

### Downsampling
```
[downsampling]
downsample_factor = 4
```
Set `downsample_factor = 1` to keep full resolution. If `downsample_factor > 1`, the pipeline will decimate image arrays by an integer factor.

### [no_georef] options and depth-image settings
If you enable fake/no-georeferencing via `[no_georef]`, additional options control how non-georeferenced images are handled and how optional 3D displacement calculation from depth images is performed.

Example TOML snippet:
```toml
[no_georef]
use_no_georeferencing = true
fake_pixel_size = 0.0025            # CRS units per pixel (e.g., meters per pixel)
convert_to_3d_displacement = true   # If true, compute 3D displacements using depth images
undistort_image = true              # If true, undistort both RGB and depth images before tracking

# Camera intrinsics: 3x3 matrix in row-major form
camera_intrinsics_matrix = [
  [fx, s, cx],
  [0.0, fy, cy],
  [0.0, 0.0, 1.0]
]

# Distortion coefficients: 2 or 4 elements as required by OpenCV (radial +/- tangential)
camera_distortion_coefficients = [k1, k2]  # or [k1,k2,p1,p2]

# Optional 4x4 homogeneous transform mapping camera coords -> target 3D coords
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

Important: Downsampling and depth rasters
If you use `[downsampling]` with `downsample_factor > 1`, the pipeline downsamples the image matrices for tracking. Depth rasters must either:
- be prepared at the same (downsampled) resolution prior to running the pipeline, or
- the pipeline must be configured to downsample depth rasters in the same way (recommended).

A mismatch between tracked-image resolution and depth image resolution will lead to incorrect 3D positions and displacements (wrong indexing into the depth raster). We recommend downsampling depth images using the same integer factor and method used for the images to preserve correct pixel alignment.

## Module: run_pipeline.py
### _load_config(path: str) -> dict
Loads a TOML config. Relative paths are resolved against the repository root.

Parameters
----------
path : str
    Path to a TOML configuration file.

Returns
-------
config : dict
    Parsed configuration dictionary.

### _get(cfg: dict, section: str, key: str, default=None)
Returns a config value or a default if missing.

Parameters
----------
cfg : dict
    Parsed config.
section : str
    Section name.
key : str
    Key within the section.
default : any
    Fallback value.

Returns
-------
value : any
    The config value or `default`.

### _require(cfg: dict, section: str, key: str)
Returns a required config value or raises KeyError.

Parameters
----------
cfg : dict
    Parsed config.
section : str
    Section name.
key : str
    Key within the section.

Returns
-------
value : any
    The required config value.

### _as_optional_value(value)
Normalizes `""`, `"none"`, and `"null"` to `None`.

Parameters
----------
value : any
    Input value.

Returns
-------
value_or_none
    The normalized value.

### make_effective_extents_from_deltas(deltas, cell_size, years_between=1.0, cap_per_side=None)
Converts per-year delta extents into effective search extents by adding half-cell padding and scaling by time span.

Parameters
----------
deltas : tuple
    (posx, negx, posy, negy) extra pixels per year beyond half the template.
cell_size : int
    Size of the tracked cell (movement or control cell size).
years_between : float
    Time span in years between two images.
cap_per_side : int or None
    Optional clamp for each side (to keep windows bounded).

Returns
-------
extents : tuple
    (posx, negx, posy, negy) effective extents in pixels.

### main()
Orchestrates the full pipeline:
- Collects image pairs from input folder/CSV
- Loads polygons
- Builds per-pair directories and codes
- Loads images (with optional downsampling)
- Aligns and tracks with cache support
- Filters, plots, saves outputs and summary statistics

## Module: CreateGeometries/DepthImageConversion.py
This module provides utilities to convert pixel coordinates and depth rasters into 3D positions and to compute
3D displacements from tracked points and depth images.

calculate_3d_position_from_depth_image(points, depth_image, camera_intrinsics_matrix, camera_to_3d_coordinates_transform=None)
--------------------------------------------------------------------------------------------------------
Purpose
- Transform 2D image pixel coordinates with corresponding depth values into 3D coordinates.

Parameters (summary)
- points: numpy.ndarray, shape (n, 2)
  An array of image pixel coordinates in the format (row, column). IMPORTANT: row first, column second.
- depth_image: numpy.ndarray, shape (H, W)
  Single-band depth raster giving Z (distance along the camera optical axis) per pixel in consistent units (e.g., meters).
  Indexing uses [row, column].
- camera_intrinsics_matrix: numpy.ndarray, shape (3, 3)
  Intrinsic camera matrix in row-major form:
    [[f_x, s, c_x],
     [0,   f_y, c_y],
     [0,   0,   1  ]]
- camera_to_3d_coordinates_transform: numpy.ndarray, shape (4, 4), optional
  Optional homogeneous transform mapping camera coordinates to a desired 3D coordinate system. Expected layout (row-major):
    [[R (3×3), t (3×1)],
     [0 0 0,   1       ]]
  The pipeline applies this matrix directly (no internal transpose) when transforming points.

Returns
- numpy.ndarray, shape (n, 4)
  An n×4 array containing homogeneous 3D coordinates: columns are [x, y, z, 1]. Use columns 0..2 for the 3D position.
  Coordinate sign conventions:
  - X aligns with image columns (increasing to the right).
  - Y is set so that positive Y points upwards (computed as -row-direction × Z).
  - Z is along the camera optical axis (distance from the camera).

Behavioral notes
- Pixel coordinates must be integers (the function casts to int when indexing). Sub-pixel alignment is not performed here.
- Missing or invalid depth values (NaN) will propagate into the output and should be handled by callers.

calculate_displacement_from_depth_images(tracked_points, depth_image_time1, depth_image_time2, camera_intrinsics_matrix, years_between_observations, camera_to_3d_coordinates_transform=None)
------------------------------------------------------------------------------------------------------------------------------------
Purpose
- Compute 3D displacements and annualized velocities for tracked points using two depth images.

Required tracked_points columns
- "row", "column": pixel coordinates of the point at time1
- "movement_row_direction", "movement_column_direction": integer or float offsets from time1 to time2 in pixel units (as returned by tracking functions such as track_movement_lsm)

Key behavior
- The function constructs the pixel coordinates at time1 and time2, extracts depth values for both positions, converts both to 3D coordinates and computes the displacement vector.
- If the same depth image is supplied for time1 and time2, the resulting 3D displacement will include only the plane-perpendicular displacement (perspective effects removed) and will be normalized to the depth units.
- The parameter `years_between_observations` is used to compute a yearly displacement rate.

Returns
- geopandas.GeoDataFrame with the following important columns:
  - "3d_displacement_distance": Euclidean length of the 3D displacement (units of depth image, e.g., meters)
  - "3d_displacement_distance_per_year": the above value divided by years_between_observations
  - "x","y","z": 3D coordinates (from time1) in the coordinate system given by camera_to_3d_coordinates_transform if provided, otherwise in camera coordinates
  - "valid": boolean; points with NaN displacement_distance are marked invalid
- geometry: created as gpd.points_from_xy(x=column, y=-row). Note the negative sign for y so that geometries follow the convention of y increasing upwards.

Depth image units and invalid values
- Depth images should encode distances along the optical axis (Z). Use consistent units (e.g., meters).
- Missing or invalid depths should be encoded as NaN. The function marks points with NaN displacements as invalid in the returned GeoDataFrame.
- Depth values ≤ 0 are not explicitly handled by the function and may lead to unexpected results; clean or mask invalid depths prior to invocation.

Example usage
```python
from PyImageTrack.CreateGeometries.DepthImageConversion import calculate_displacement_from_depth_images

gdf = calculate_displacement_from_depth_images(
    tracked_points=tracked_pd_df,
    depth_image_time1=depth_arr1,
    depth_image_time2=depth_arr2,
    camera_intrinsics_matrix=camera_intrinsics,         # numpy array 3x3
    years_between_observations=1.0,
    camera_to_3d_coordinates_transform=camera2world4x4  # optional numpy array 4x4
)
# access results:
gdf[["x","y","z","3d_displacement_distance_per_year","valid"]].head()
```

Depth images and naming convention
When `convert_to_3d_displacement` is enabled the pipeline expects depth images to be stored next to each original
image in a "Depth_images" subfolder, with basename appended by "_depth.tiff". Example:
- Image: /path/to/foo.jpg
- Depth raster: /path/to/Depth_images/foo_depth.tiff

Depth rasters must be single-band arrays with depth values along the camera optical axis (Z) in a consistent unit
e.g., meters). Depth arrays are read using rasterio and treated as having the same image coordinate system as the
RGB/gray images. If `undistort_image = true`, the pipeline will undistort depth rasters using the same intrinsics and
distortion coefficients as for the RGB images (be aware of interpolation/artifacts that may be introduced by undistortion).
Ensure depth rasters are prepared so that pixel indices (row, column) correspond directly to the tracking image pixels.

## Module: ImageTracking/ImagePair.py
### class ImagePair
Encapsulates a pair of images and the full alignment/tracking/filtering workflow.

#### __init__(parameter_dict: dict = None)
Parameters are read from `parameter_dict`. Common keys:
- use_no_georeferencing
- fake_pixel_size
- downsample_factor
- convert_to_3d_displacement       # when true, compute 3D displacements using depth rasters
- undistort_image                 # if true, undistort both image and depth rasters using camera intrinsics
- camera_intrinsics_matrix        # 3x3 matrix (if undistortion or 3D conversion is enabled)
- camera_distortion_coefficients  # 2- or 4-element array (OpenCV format)
- camera_to_3d_coordinates_transform  # optional 4x4 homogeneous transform for output coordinates

#### _effective_pixel_size() -> float
Returns CRS units per pixel (assumes square pixels).

#### _downsample_array(arr, factor) -> np.ndarray
Decimates a 2D or 3D array by integer `factor`.

#### _downsample_transform(transform, factor)
Scales an affine transform by `factor` for downsampling.

#### load_images_from_file(filename_1, observation_date_1, filename_2, observation_date_2, selected_channels=None, NA_value=None)
Loads and crops images to the intersection, handles fake georeferencing and downsampling, and sets bounds. When operating with fake georeferencing and `convert_to_3d_displacement` enabled, the function will attempt to read corresponding depth rasters using the naming convention described above.

Parameters
----------
filename_1, filename_2 : str
    File paths.
observation_date_1, observation_date_2 : str
    Dates for the observations.
selected_channels : list[int] or int or None
    Channels to use for tracking.
NA_value : float or None
    If provided, set that value to 0 in both images.

Returns
-------
None

#### Other ImagePair methods
- select_image_channels(selected_channels=None)
- load_images_from_matrix_and_transform(...)
- align_images(reference_area)
- compute_truecolor_aligned_from_control_points()
- track_points(tracking_area)
- perform_point_tracking(reference_area, tracking_area)
- plot_images()
- plot_tracking_results()
- plot_tracking_results_with_valid_mask()
- filter_outliers(filter_parameters)
- calculate_lod(points_for_lod_calculation, filter_parameters=None)
- filter_lod_points()
- full_filter(reference_area, filter_parameters)
- save_full_results(folder_path: str, save_files: list) -> None

## Module: ImageTracking/TrackingResults.py
### class TrackingResults
Represents the result of tracking a single point/cell.

Parameters
----------
movement_rows : float
movement_cols : float
tracking_method : str
transformation_matrix : numpy.ndarray or None
cross_correlation_coefficient : float or None
tracking_success : bool

Fields
------
movement_rows, movement_cols
tracking_method
transformation_matrix (LSM only)
cross_correlation_coefficient
tracking_success

## Module: DataProcessing/ImagePreprocessing.py
### undistort_camera_image(image_matrix, camera_intrinsic_matrix, distortion_coefficients) -> numpy.ndarray
Undistorts a camera image using OpenCV and returns the undistorted, cropped image. When `undistort_image = true` this is applied to both RGB/gray images and depth rasters.

Parameters
----------
image_matrix: numpy.ndarray
    The array representing the distorted image.
camera_intrinsic_matrix: numpy.ndarray
    The 3x3 intrinsic matrix in row-major form.
distortion_coefficients: numpy.ndarray
    Distortion coefficients as a 1D array (2 or 4 elements).

Returns
-------
image_matrix_undistorted: numpy.ndarray
    The undistorted and cropped image array.

## Module: CreateGeometries/HandleGeometries.py
(Functions such as get_submatrix_symmetric, grid_points_on_polygon_by_distance, get_raster_indices_from_points and others are documented in-line in source files. Consult source for details.)

## Appendix: Examples
### Simple depth-based 3D displacement computing (Python snippet)
```python
from PyImageTrack import run_from_config
# or run the CLI: pyimagetrack-run --config configs/drone_hs.toml

# If you prefer to call functionality directly, prepare a TOML config as shown above and call:
run_from_config("configs/drone_hs.toml")
```

End of file