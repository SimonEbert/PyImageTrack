# Example workflows

PyImageTrack includes two example workflows for getting started. The examples use the same structure as the regular project data:

```text
configs/<example>.toml
input/<example>/
output/<example>/
```

The configuration file defines the input and output folders and the processing parameters. Input files belong in the matching folder under `input/`. Generated results are written to the matching folder under `output/`.

## Example 00: Two images

This is the smallest complete workflow. It compares two images and calculates movement between them.

```text
configs/00_example_two_images.toml
input/00_example_two_images/
output/00_example_two_images/
```

Expected input files:

```text
image_01.tif
image_02.tif
moving_area.shp
moving_area.shx
moving_area.dbf
moving_area.prj
stable_area.shp
stable_area.shx
stable_area.dbf
stable_area.prj
```

This example is intended to explain the basic PyImageTrack workflow without an image-date table or a custom image-pair table.

## Example 01: Image series

This example uses several images from different points in time. Consecutive images are compared as a series.

```text
configs/01_example_image_series.toml
input/01_example_image_series/
output/01_example_image_series/
```

Expected input files:

```text
image_01.tif
image_02.tif
image_03.tif
image_dates.csv
moving_area.shp
moving_area.shx
moving_area.dbf
moving_area.prj
stable_area.shp
stable_area.shx
stable_area.dbf
stable_area.prj
```

This example is intended to explain processing of a time series and the use of `image_dates.csv`. The exact CSV format must match the format documented in the main user documentation.

## Example 02: Non-georeferenced images

This advanced example is prepared for image sequences captured without georeferencing, such as camera-based time-lapse images.

```text
configs/02_example_non_georeferenced.toml
input/02_example_non_georeferenced/
output/02_example_non_georeferenced/
```

Expected input files include JPEG images, `image_pairs.csv`, and the moving-area and stable-area shapefile components. The configuration will use the non-georeferenced workflow and will require camera parameters.

Depth images and 3D displacement conversion are intentionally not included yet.

## Current status

The files in these example folders are placeholders. They are not runnable example datasets until valid images, shapefiles, CSV content, and TOML configuration values have been added.

Advanced workflows, such as custom image pairs, image enhancement, non-georeferenced time-lapse images, depth images, and batch processing, can be added later.
