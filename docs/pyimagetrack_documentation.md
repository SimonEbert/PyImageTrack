# PyImageTrack Documentation
2026-01-13

## Overview
PyImageTrack provides alignment and feature tracking for georeferenced imagery, with optional filtering and plotting.
The main entry point is the CLI command `pyimagetrack-run`, configured via TOML files in `configs/`.

## Running The Pipeline
```
pyimagetrack-run --config configs/drone_hs.toml
```
Config paths are resolved relative to the repo root.
The CLI invokes `PyImageTrack.run_pipeline:main` under the hood.

## Entry point for Python Scripts
The package provides the possibility of accessing the tracking routine from another Python script. This is done using 
the `run_from_config` function that can be imported by
```
from PyImageTrack import run_from_config
```
IMPORTANT: When calling the tracking routine from within another Python Script or Project, the process must always be 
guarded in the following way
```
if __name__ == "__main__":
    run_from_config(path_to_config)
```
This is due to the tracking routine making use of the multiprocessing package for parallelization and without this
safety measure, the tracking routine will be called several times with identical parameters.

## Configuration Files (TOML)
Configs are TOML files and share the same structure. Use these as templates:
- `configs/drone_hs.toml`
- `configs/smoketest_drone_hs.toml`
- `configs/timelapse_fake_ortho.toml`

### Key Sections
- `[paths]`: input/output folders, optional CSVs for dates/pairs.
- `[polygons]`: stable/moving area shapefiles and CRS.
- `[pairing]`: pairing mode (`all`, `first_to_all`, `successive`, `custom`).
- `[no_georef]`: enable for non-ortho images (e.g., JPGs).
- `[downsampling]`: optional downsampling for fast smoke tests.
- `[flags]`: alignment/tracking/filtering/plotting toggles.
- `[cache]`: caching and recompute flags.
- `[output]`: optional outputs such as true-color alignment.
- `[adaptive_tracking_window]`: search extent scales by time span when enabled.
- `[alignment]`, `[tracking]`, `[filter]`: algorithm parameters.
- `[save]`: list of output files to write.


### Config: [no_georef] options

If you enable fake/no-georeferencing via `[no_georef]`, additional options control how non-georeferenced images are
handled and how optional 3D displacement calculation from depth images is performed. First, the usage of the different
options is described and then an example config file part is given:

- `convert_to_3d_displacement`: When true, the pipeline will look for per-image depth rasters and compute 3D displacements.
The depth image corresponding to an image `image_filename` on which tracking is performed is assumed to be called
`<image_filename>_depth.tiff` and located in a subfolder `Depth_images` of the folder, where the tracking images are
stored.
- `fake_pixel_size` gives the pixel size used when working without a CRS (units per pixel).
- `camera_intrinsics_matrix` and `camera_distortion_coefficients` are only required if undistort_image = true. In this
case the positions will be correctly transformed into the camera coordinate system.
- `camera_to_3d_coordinates_transform` must be a 4×4 homogeneous matrix. The expected format is: [[R (3×3), t (3×1)],
[0 0 0, 1 ]] where R is rotation and t is translation. (See CreateGeometries/DepthImageConversion.py for details and a
note about current implementation behavior.)

Arrays in TOML are parsed into lists and converted to numpy arrays by the pipeline — use numeric literals (no strings).


#### Depth images

When convert_to_3d_displacement is enabled the pipeline expects depth images to be stored next to each original image in a "Depth_images" subfolder, with basename appended by "_depth.tiff". Example:

    Image: /path/to/foo.jpg
    Depth raster: /path/to/Depth_images/foo_depth.tiff

Depth rasters must be single-band arrays with depth values along the camera optical axis (Z) in a consistent unit (e.g.,
meters). Depth arrays are treated as having the same image coordinate system as the tracking images. If
`undistort_image = true`, the pipeline will undistort depth rasters using the same intrinsics and distortion
coefficients as for the tracking images automatically. Therefore, no preprocessing step is needed, except ensuring that
tracking image pixels and depth_image_pixels correspond to the *exact* same locations.

Example TOML snippet:
```toml
[no_georef]
use_no_georeferencing = true
fake_pixel_size = 1                     # CRS units per pixel (e.g., meters per pixel)
# If true, compute 3D displacements using depth images. In this case, the folder that contains the tracking images
# should contain a subfolder named "Depth_images", which itself should contain a depth image file for every image file,
# named <image_file_name>_depth.tiff
convert_to_3d_displacement = true      
# If true, undistort both RGB and depth images before tracking. In this case the camera_intrinsics_matrix and the
# camera_distortion_coefficients need to be specified
undistort_image = true                  

# Camera intrinsics: 3x3 matrix in the following format
camera_intrinsics_matrix = [
  [fx, s, cx],
  [0.0, fy, cy],
  [0.0, 0.0, 1.0]
]

# Distortion coefficients: 2 or 4 elements as required by OpenCV (radial +/- tangential)
camera_distortion_coefficients = [k1, k2]  # or [k1,k2,p1,p2]

# Optional 4x4 homogeneous transform mapping camera coords -> target 3D coords
# Can be used to transform computed 3d image coordinates from the depth image to an arbitrary 3d coordinate system
# given the respective homogeneous transform in the following format
camera_to_3d_coordinates_transform = [
  [r11, r12, r13, t1],
  [r21, r22, r23, t2],
  [r31, r32, r33, t3],
  [0.0,  0.0,  0.0,  1.0]
]
```


### Downsampling
```
[downsampling]
downsample_factor = 4
```
Set `downsample_factor = 1` to keep full resolution.

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
Normalizes "", "none", and "null" to `None`.

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

## Module: Utils.py
### _round_to_nearest_hour(dt: datetime) -> datetime
Rounds a datetime to the nearest hour (>=30 min rounds up).

Parameters
----------
dt : datetime
    Input datetime.

Returns
-------
dt_rounded : datetime
    Rounded datetime.

### parse_date(s: str) -> datetime
Parses flexible date/time strings from filenames or plain text. Supports:
- YYYYMMDD-HHMMSS, YYYYMMDD, YYYY-MM-DD
- YYYY-MM-DD HH:MM[:SS] and DD-MM-YYYY variants

Parameters
----------
s : str
    Input string (filename or date string).

Returns
-------
dt : datetime
    Parsed datetime.

### collect_pairs(input_folder, date_csv_path=None, pairs_csv_path=None, pairing_mode="all", extensions=None)
Builds image pairs and returns:
- year_pairs: list of (id1, id2)
- id_to_file: id -> file path
- id_to_date: id -> date string
- id_hastime_from_filename: id -> bool

Parameters
----------
input_folder : str
    Folder containing images.
date_csv_path : str or None
    Optional CSV with `year`/`date` for ambiguous filenames (YYYY or YYYY-MM).
pairs_csv_path : str or None
    Optional CSV specifying custom pairs (date_earlier/date_later).
pairing_mode : str
    all | successive | first_to_all | custom.
extensions : tuple or None
    Allowed extensions; defaults to (".tif", ".tiff").

Returns
-------
year_pairs : list[tuple]
    List of (id1, id2) pairs.
id_to_file : dict
    Mapping from id to file path.
id_to_date : dict
    Mapping from id to date string.
id_hastime_from_filename : dict
    Mapping from id to bool (time was encoded in filename).

### ensure_dir(path: str)
Creates a directory if missing.

Parameters
----------
path : str
    Directory path.

Returns
-------
None

### float_compact(x)
Formats a float into a compact string without trailing zeros.

Parameters
----------
x : float or any
    Value to format.

Returns
-------
s : str
    Compact string.

### _get(obj, name, default="NA")
Returns `obj.name` or `obj[name]` with a default fallback.

Parameters
----------
obj : object or dict
    Source object.
name : str
    Attribute or key.
default : any
    Default if missing.

Returns
-------
value : any
    Attribute/key value or default.

### abbr_alignment(ap)
Builds a short alignment code for output folders.

Parameters
----------
ap : object or dict
    Alignment parameters.

Returns
-------
code : str
    Folder code.

### abbr_tracking(tp)
Builds a short tracking code for output folders.

Parameters
----------
tp : object or dict
    Tracking parameters.

Returns
-------
code : str
    Folder code.

### abbr_filter(fp)
Builds a short filter code for output folders.

Parameters
----------
fp : FilterParameters
    Filter parameters.

Returns
-------
code : str
    Folder code.

## Module: Cache.py
### _sha256(path: str) -> str
Computes a SHA-256 hash for a file.

Parameters
----------
path : str
    File path.

Returns
-------
hex_digest : str
    Hash string.

### alignment_cache_paths(align_dir, year1, year2)
Returns paths for aligned raster, control points, and metadata JSON.

Parameters
----------
align_dir : str
    Alignment folder.
year1, year2 : str
    Pair identifiers.

Returns
-------
aligned_tif, control_pts, meta_json : tuple
    Output paths.

### save_alignment_cache(image_pair, align_dir, year1, year2, align_params, filenames, dates, version="v1", save_truecolor_aligned=False)
Writes aligned raster (and optional true-color raster), control points, and metadata JSON.

Parameters
----------
image_pair : ImagePair
    Pair object with aligned image data.
align_dir : str
    Alignment folder.
year1, year2 : str
    Pair identifiers.
align_params : dict
    Alignment parameters for metadata.
filenames : dict
    Mapping id -> path.
dates : dict
    Mapping id -> date.
version : str
    Metadata version.
save_truecolor_aligned : bool
    If true, writes a true-color aligned image if available.

Returns
-------
None

### load_alignment_cache(image_pair, align_dir, year1, year2) -> bool
Loads aligned raster and control points into an ImagePair.

Parameters
----------
image_pair : ImagePair
    Target object.
align_dir : str
    Alignment folder.
year1, year2 : str
    Pair identifiers.

Returns
-------
success : bool
    True if cache was loaded.

### tracking_cache_paths(track_dir, year1, year2)
Returns paths for raw tracking GeoJSON and metadata JSON.

Parameters
----------
track_dir : str
    Tracking folder.
year1, year2 : str
    Pair identifiers.

Returns
-------
raw_geojson, meta_json : tuple
    Output paths.

### save_tracking_cache(image_pair, track_dir, year1, year2, track_params, filenames, dates, version="v1")
Writes raw tracking GeoJSON and metadata JSON.

Parameters
----------
image_pair : ImagePair
    Pair object with tracking results.
track_dir : str
    Tracking folder.
year1, year2 : str
    Pair identifiers.
track_params : dict
    Tracking parameters for metadata.
filenames : dict
    Mapping id -> path.
dates : dict
    Mapping id -> date.
version : str
    Metadata version.

Returns
-------
None

### load_tracking_cache(image_pair, track_dir, year1, year2) -> bool
Loads tracking GeoJSON into an ImagePair.

Parameters
----------
image_pair : ImagePair
    Target object.
track_dir : str
    Tracking folder.
year1, year2 : str
    Pair identifiers.

Returns
-------
success : bool
    True if cache was loaded.

## Module: Parameters
### AlignmentParameters
Container for alignment parameters.

Parameters
----------
parameter_dict : dict
    Source dict of parameters.

Fields
------
number_of_control_points
control_search_extent_px
control_search_extent_deltas
control_cell_size
cross_correlation_threshold_alignment
maximal_alignment_movement

Methods
-------
__str__()
    Human-readable summary.
to_dict()
    Keys expected by ImagePair(parameter_dict=...).

### TrackingParameters
Container for tracking parameters.

Parameters
----------
parameter_dict : dict
    Source dict of parameters.

Fields
------
image_bands
distance_of_tracked_points_px
search_extent_px
search_extent_deltas
movement_cell_size
cross_correlation_threshold_movement

Methods
-------
__str__()
    Human-readable summary.
to_dict()
    Keys expected by ImagePair(parameter_dict=...).

### FilterParameters
Container for filtering parameters.

Parameters
----------
parameter_dict : dict
    Source dict of parameters.

Fields
------
level_of_detection_quantile
number_of_points_for_level_of_detection
difference_movement_bearing_threshold
difference_movement_bearing_moving_window_size
standard_deviation_movement_bearing_threshold
standard_deviation_movement_bearing_moving_window_size
difference_movement_rate_threshold
difference_movement_rate_moving_window_size
standard_deviation_movement_rate_threshold
standard_deviation_movement_rate_moving_window_size

Methods
-------
__str__()
    Human-readable summary.

## Module: CreateGeometries/HandleGeometries.py
### get_submatrix_symmetric(central_index, shape, matrix)
Extracts a symmetric submatrix centered at `central_index`.

Parameters
----------
central_index : list or array
    [row, col] for the center pixel.
shape : list or tuple
    [height, width] of the submatrix. Even sizes are reduced by 1.
matrix : np.ndarray
    2D or 3D (channels-first) array.

Returns
-------
submatrix : np.ndarray
    Extracted submatrix.

### grid_points_on_polygon_by_distance(polygon, distance_of_points=10, distance_px=None, pixel_size=None)
Creates an evenly spaced grid of points inside a polygon at a given spacing.

Parameters
----------
polygon : gpd.GeoDataFrame
    Single-polygon GeoDataFrame.
distance_of_points : float
    Desired spacing in CRS units.
distance_px : float or None
    Optional pixel spacing for logging.
pixel_size : float or None
    Pixel size in CRS units (for logging).

Returns
-------
points : gpd.GeoDataFrame
    Grid points inside the polygon.

### random_points_on_polygon_by_number(polygon, number_of_points)
Creates randomly distributed points inside a polygon.

Parameters
----------
polygon : gpd.GeoDataFrame
    Single-polygon GeoDataFrame.
number_of_points : int
    Number of points to generate.

Returns
-------
points : gpd.GeoDataFrame
    Random points inside the polygon.

### get_raster_indices_from_points(points, raster_matrix_transform)
Converts point coordinates to raster row/column indices.

Parameters
----------
points : gpd.GeoDataFrame
    Points in CRS coordinates.
raster_matrix_transform : Affine
    Raster transform.

Returns
-------
rows, cols : list
    Row/column indices for points.

### crop_images_to_intersection(file1, file2)
Crops two rasters to their spatial intersection and returns arrays + transforms.

Parameters
----------
file1, file2
    Rasterio-opened datasets.

Returns
-------
[array_file1, array_file1_transform], [array_file2, array_file2_transform]
    Cropped arrays and transforms.

### georeference_tracked_points(tracked_pixels, raster_transform, crs, years_between_observations=1)
Converts tracked pixel offsets into georeferenced movement vectors and yearly rates.

Parameters
----------
tracked_pixels : pd.DataFrame
    Must include row/column and movement direction fields.
raster_transform : Affine
    Raster transform.
crs : any
    CRS identifier.
years_between_observations : float
    Time span between images.

Returns
-------
georeferenced_tracked_pixels : gpd.GeoDataFrame
    GeoDataFrame with movement distance and movement per year.

### circular_std_deg(angles_deg)
Computes circular standard deviation (degrees).

Parameters
----------
angles_deg : array-like
    Angles in degrees.

Returns
-------
std_deg : float
    Circular standard deviation.

### get_submatrix_rect_from_extents(central_index, extents, matrix)
Extracts an asymmetric rectangular window given extents (posx, negx, posy, negy).

Parameters
----------
central_index : list or array
    [row, col] center.
extents : tuple
    (posx, negx, posy, negy) in pixels.
matrix : np.ndarray
    2D or 3D array.

Returns
-------
submatrix : np.ndarray
    Extracted window.
center_in_submatrix : tuple
    (row, col) of original center inside the submatrix.


## Module: CreateGeometries/DepthImageConversion.py

This module provides utilities to convert pixel coordinates and depth rasters into 3D positions and to compute 3D
displacements from tracked points and depth images.
### calculate_3d_position_from_depth_image(points, depth_image, camera_intrinsics_matrix, camera_to_3d_coordinates_transform=None)
Transform 2D image pixel coordinates with corresponding depth values into 3D coordinates.

#### Parameters
`points`: numpy.ndarray, shape (n, 2)
    An array of image pixel coordinates in the format (row, column).

`depth_image`: numpy.ndarray, shape (H, W)
    Single-band depth raster giving Z (distance along the camera optical axis) per pixel in consistent units (e.g.,
meters). Indexing uses [row, column].

`camera_intrinsics_matrix`: numpy.ndarray, shape (3, 3)
    Intrinsic camera matrix in row-major form: [[f_x, s, c_x], [0, f_y, c_y], [0, 0, 1 ]]

`camera_to_3d_coordinates_transform`: numpy.ndarray, shape (4, 4), optional
    Optional homogeneous transform mapping camera coordinates to a desired 3D coordinate system. Expected layout (row-major): [[R (3×3), t (3×1)], [0 0 0, 1 ]]

Returns

`points_transformed`: numpy.ndarray, shape (n, 4)
An n×3 array containing corresponding 3D coordinates: columns are [x, y, z]. Coordinate sign conventions:
        X aligns with image columns (increasing to the right).
        Y is set so that positive Y points upwards (computed as -row-direction × Z).
        Z is along the camera optical axis (distance from the camera).


### calculate_displacement_from_depth_images(tracked_points, depth_image_time1, depth_image_time2, camera_intrinsics_matrix, years_between_observations, camera_to_3d_coordinates_transform=None)
 Compute 3D displacements and annualized velocities for tracked points using two depth images.

#### Parameters

 `tracked_points`: pd.DataFrame with at least the following columns:

- "row", "column": pixel coordinates of the point at time1
- "movement_row_direction", "movement_column_direction": float offsets from time1 to time2 in pixel units (as returned by
tracking functions such as track_movement_lsm)

`depth_image_time1`: numpy.ndarray, shape (H, W)
    Single-band depth raster giving Z (distance along the camera optical axis) per pixel in consistent units (e.g.,
meters) for the first time point of tracking.

`depth_image_time2`: numpy.ndarray, shape (H, W)
    Single-band depth raster giving Z (distance along the camera optical axis) per pixel in consistent units (e.g.,
meters) for the second time point of tracking.

`camera_intrinsics_matrix`: numpy.ndarray, shape (3, 3)
    Intrinsic camera matrix in row-major form: [[f_x, s, c_x], [0, f_y, c_y], [0, 0, 1 ]]

`camera_to_3d_coordinates_transform`: numpy.ndarray, shape (4, 4), optional
    Optional homogeneous transform mapping camera coordinates to a desired 3D coordinate system. Expected layout (row-major): [[R (3×3), t (3×1)], [0 0 0, 1 ]]


#### Returns

`georeferenced_tracked_pixels`: geopandas.GeoDataFrame with the following important columns:
- "3d_displacement_distance": Euclidean length of the 3D displacement (units of depth image, e.g., meters)
- "3d_displacement_distance_per_year": the above value divided by years_between_observations
- "x","y","z": 3D coordinates (from time1) in the coordinate system given by camera_to_3d_coordinates_transform
if provided, otherwise in camera coordinates
- "valid": boolean; points with NaN displacement_distance are marked invalid
- geometry: created as gpd.points_from_xy(x=column, y=-row). Note the negative sign for y so that geometries follow the convention of y increasing upwards.

### Note

    Depth images should encode distances along the optical axis (Z). Use consistent units (e.g., meters).
    Missing or invalid depths should be encoded as NaN. The function marks points with NaN displacements as invalid in the returned GeoDataFrame.
    Depth values ≤ 0 are not explicitly handled by the function and may lead to unexpected results; clean or mask invalid depths prior to invocation.


## Module: ImageTracking/TrackingResults.py
### class TrackingResults
Represents the result of tracking a single point/cell.

Parameters
----------
movement_rows : float
movement_cols : float
tracking_method : str
transformation_matrix : np.ndarray or None
cross_correlation_coefficient : float or None
tracking_success : bool

Fields
------
movement_rows, movement_cols
tracking_method
transformation_matrix (LSM only)
cross_correlation_coefficient
tracking_success

## Module: ImageTracking/TrackMovement.py
### track_cell_cc(tracked_cell_matrix, search_cell_matrix, search_center=None)
Cross-correlation tracking for a single cell inside a search window.

Parameters
----------
tracked_cell_matrix : np.ndarray
    Template cell from image1.
search_cell_matrix : np.ndarray
    Search window from image2.
search_center : list or None
    Optional logical center of the search window (for asymmetric extents).

Returns
-------
tracking_results : TrackingResults
    Movement in rows/cols and correlation coefficient.

### move_indices_from_transformation_matrix(transformation_matrix, indices)
Applies an affine transform (2x3) to a set of row/column indices.

Parameters
----------
transformation_matrix : np.ndarray
    2x3 affine transform.
indices : np.ndarray
    2xn array of indices.

Returns
-------
moved_indices : np.ndarray
    2xn array after transform.

### track_cell_lsm(tracked_cell_matrix, search_cell_matrix, initial_shift_values=None, search_center=None)
Least-squares tracking for a single cell with optional initial shift.

Parameters
----------
tracked_cell_matrix : np.ndarray
    Template cell.
search_cell_matrix : np.ndarray
    Search window.
initial_shift_values : np.ndarray or None
    Initial movement estimates.
search_center : list or None
    Optional logical center of the search window (for asymmetric extents).

Returns
-------
tracking_results : TrackingResults
    Shift estimates and correlation coefficient; invalid results return NaNs.

### track_cell_lsm_parallelized(central_index)
Multiprocessing helper that tracks a single cell using shared globals.

Parameters
----------
central_index : np.ndarray
    Central index to track.

Returns
-------
tracking_results : TrackingResults
    Result for the given index.

### track_movement_lsm(image1_matrix, image2_matrix, image_transform, points_to_be_tracked, tracking_parameters=None,
alignment_parameters=None, alignment_tracking=False, save_columns=None, task_label="Tracking points")
Tracks a set of points using the least-squares approach.

Parameters
----------
image1_matrix : np.ndarray
    First observation (2D or 3D).
image2_matrix : np.ndarray
    Second observation (same shape as image1_matrix).
image_transform : Affine
    Shared transform for both matrices.
points_to_be_tracked : gpd.GeoDataFrame
    Point locations to track.
tracking_parameters : TrackingParameters
    Parameters for movement tracking.
alignment_parameters : AlignmentParameters
    Parameters for alignment tracking.
alignment_tracking : bool
    If True, uses alignment parameters and control-search extents.
save_columns : list[str] or None
    Columns to include in output. Defaults to movement/bearing fields.
task_label : str
    Progress label for the tqdm bar.

Returns
-------
tracked_pixels : pd.DataFrame
    Movement vectors and correlation coefficients (filtered by threshold).

## Module: ImageTracking/AlignImages.py
### align_images_lsm_scarce(image1_matrix, image2_matrix, image_transform, reference_area, alignment_parameters)
Aligns image2 to image1 using least-squares tracking on control points within a reference area.

Parameters
----------
image1_matrix : np.ndarray
    Reference image.
image2_matrix : np.ndarray
    Image to be aligned.
image_transform : Affine
    Transform for image1.
reference_area : gpd.GeoDataFrame
    Stable area polygon.
alignment_parameters : AlignmentParameters
    Alignment parameters.

Returns
-------
image1_matrix : np.ndarray
moved_image2_matrix : np.ndarray
tracked_control_points : pd.DataFrame
    Control points and tracking info.

## Module: ImageTracking/ImagePair.py
### class ImagePair
Encapsulates a pair of images and the full alignment/tracking/filtering workflow.

#### __init__(parameter_dict: dict = None)
Parameters are read from `parameter_dict`. Common keys:
- use_no_georeferencing
- fake_pixel_size
- downsample_factor
- convert_to_3d_displacement # when true, compute 3D displacements using depth rasters
- undistort_image # if true, undistort both image and depth rasters using camera intrinsics
- camera_intrinsics_matrix # 3x3 matrix (if undistortion or 3D conversion is enabled)
- camera_distortion_coefficients # 2- or 4-element array (OpenCV format)
- camera_to_3d_coordinates_transform # optional 4x4 homogeneous transform for output coordinates

#### _effective_pixel_size() -> float
Returns CRS units per pixel (assumes square pixels).

#### _downsample_array(arr, factor) -> np.ndarray
Decimates a 2D or 3D array by integer `factor`.

#### _downsample_transform(transform, factor)
Scales an affine transform by `factor` for downsampling.

#### select_image_channels(selected_channels=None)
Selects bands for tracking. Default uses first three channels.

#### load_images_from_file(filename_1, observation_date_1, filename_2, observation_date_2, selected_channels=None, NA_value=None)
Loads and crops images to the intersection, handles fake georeferencing and downsampling, and sets bounds.

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

#### load_images_from_matrix_and_transform(image1_matrix, observation_date_1, image2_matrix, observation_date_2, image_transform, crs, selected_channels=None)
Loads pre-supplied matrices with a shared transform and CRS.

Parameters
----------
image1_matrix, image2_matrix : np.ndarray
observation_date_1, observation_date_2 : str
image_transform : Affine
crs : any
selected_channels : list[int] or int or None

Returns
-------
None

#### align_images(reference_area)
Aligns the two images based on a reference area; updates `image2_matrix` and `image2_transform`.

Parameters
----------
reference_area : gpd.GeoDataFrame
    Stable area polygon.

Returns
-------
None

#### compute_truecolor_aligned_from_control_points()
Builds a true-color aligned version of image2 using alignment control points.

Returns
-------
None

#### track_points(tracking_area)
Creates a grid of points within the tracking area and tracks movement.

Parameters
----------
tracking_area : gpd.GeoDataFrame
    Moving area polygon.

Returns
-------
georeferenced_tracked_points : gpd.GeoDataFrame
    Points with movement rates and bearings.

#### perform_point_tracking(reference_area, tracking_area)
Aligns (if needed) and tracks points, storing results in `self.tracking_results`.

Parameters
----------
reference_area : gpd.GeoDataFrame
tracking_area : gpd.GeoDataFrame

Returns
-------
None

#### plot_images()
Plots the two raster images.

#### plot_tracking_results()
Plots movement vectors on the first image.

#### plot_tracking_results_with_valid_mask()
Plots valid vs invalid points on the first image.

#### filter_outliers(filter_parameters)
Applies outlier filtering using FilterParameters.

Parameters
----------
filter_parameters : FilterParameters

Returns
-------
None

#### calculate_lod(points_for_lod_calculation, filter_parameters=None)
Computes the level of detection (LoD) from random points in a stable area.

Parameters
----------
points_for_lod_calculation : gpd.GeoDataFrame
filter_parameters : FilterParameters or None

Returns
-------
None

#### filter_lod_points()
Marks points below LoD as invalid.

Returns
-------
None

#### full_filter(reference_area, filter_parameters)
Runs outlier filtering, LoD calculation, and LoD filtering.

Parameters
----------
reference_area : gpd.GeoDataFrame
filter_parameters : FilterParameters

Returns
-------
None

#### equalize_adapthist_images()
Applies CLAHE to both images.

Returns
-------
None

#### save_full_results(folder_path: str, save_files: list) -> None
Writes tracking outputs (GeoJSON, GeoTIFFs, masks, stats) into `folder_path`.

Parameters
----------
folder_path : str
    Output directory.
save_files : list
    Tokens controlling which outputs are written. Supported tokens include:
    - "first_image_matrix", "second_image_matrix"
    - "movement_bearing_valid_tif", "movement_rate_valid_tif"
    - "movement_bearing_outlier_filtered_tif", "movement_rate_outlier_filtered_tif"
    - "movement_bearing_LoD_filtered_tif", "movement_rate_LoD_filtered_tif"
    - "movement_bearing_all_tif", "movement_rate_all_tif"
    - "mask_invalid_tif", "mask_LoD_tif"
    - "mask_outlier_md_tif", "mask_outlier_msd_tif", "mask_outlier_bd_tif", "mask_outlier_bsd_tif"
    - "LoD_points_geojson", "control_points_geojson"
    - "statistical_parameters_txt"

Returns
-------
None

#### load_results(file_path, reference_area)
Loads saved tracking results and aligns images to a reference area.

Parameters
----------
file_path : str
reference_area : gpd.GeoDataFrame

Returns
-------
None

## Module: DataProcessing/ImagePreprocessing.py
### equalize_adapthist_images(image_matrix, kernel_size)
Applies CLAHE (adaptive histogram equalization) using scikit-image.

Parameters
----------
image_matrix : np.ndarray
kernel_size : int

Returns
-------
equalized_image : np.ndarray

## Module: DataProcessing/DataPostprocessing.py
### calculate_lod_points(image1_matrix, image2_matrix, image_transform, points_for_lod_calculation,
tracking_parameters, crs, years_between_observations)
Tracks random points to estimate LoD (level of detection).

Parameters
----------
image1_matrix, image2_matrix : np.ndarray
image_transform : Affine
points_for_lod_calculation : gpd.GeoDataFrame
tracking_parameters : TrackingParameters
crs : any
years_between_observations : float

Returns
-------
tracked_points : gpd.GeoDataFrame
    Tracked points for LoD.

### _ensure_bool_col(df, col)
Ensures a boolean column exists (internal helper).

Parameters
----------
df : pd.DataFrame
col : str

Returns
-------
col_values : np.ndarray

### filter_lod_points(tracking_results, level_of_detection)
Marks points below the LoD as invalid.

Parameters
----------
tracking_results : gpd.GeoDataFrame
level_of_detection : float

Returns
-------
tracking_results : gpd.GeoDataFrame

### filter_outliers_movement_bearing_difference(tracking_results, filter_parameters)
Removes outliers based on bearing difference vs local neighborhood.

### filter_outliers_movement_bearing_standard_deviation(tracking_results, filter_parameters)
Removes outliers based on bearing standard deviation in a neighborhood.

### filter_outliers_movement_rate_difference(tracking_results, filter_parameters)
Removes outliers based on movement rate difference vs local neighborhood.

### filter_outliers_movement_rate_standard_deviation(tracking_results, filter_parameters)
Removes outliers based on movement rate standard deviation in a neighborhood.

### filter_outliers_full(tracking_results, filter_parameters)
Applies all outlier filters in sequence.

## Module: Plots/MakePlots.py
### plot_raster_and_geometry(raster_matrix, raster_transform, geometry, alpha=0.6)
Plots a raster with a geometry overlay.

Parameters
----------
raster_matrix : np.ndarray
raster_transform : Affine
geometry : gpd.GeoDataFrame
alpha : float

Returns
-------
None

### plot_movement_of_points(raster_matrix, raster_transform, point_movement, point_color=None, masking_polygon=None,
fig=None, ax=None, save_path=None, show_arrows=True)
Plots tracked point movement with optional masking and saving.

Parameters
----------
raster_matrix : np.ndarray
raster_transform : Affine
point_movement : gpd.GeoDataFrame
point_color : str or None
masking_polygon : gpd.GeoDataFrame or None
fig, ax : matplotlib objects or None
save_path : str or None
show_arrows : bool

Returns
-------
None

### plot_movement_of_points_with_valid_mask(raster_matrix, raster_transform, point_movement, save_path=None)
Plots valid vs invalid points in different colors.

Parameters
----------
raster_matrix : np.ndarray
raster_transform : Affine
point_movement : gpd.GeoDataFrame
save_path : str or None

Returns
-------
None

### plot_distribution_of_point_movement(moving_points)
Scatter plot of movement row vs column directions.

Parameters
----------
moving_points : gpd.GeoDataFrame

Returns
-------
None

## Package __init__.py
The top-level `src/PyImageTrack/__init__.py` is a namespace package wrapper. It optionally appends a
`PyImageTrack_scripts` subdirectory to `__path__` to support legacy imports.
