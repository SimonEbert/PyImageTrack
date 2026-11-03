# Example 02: Non-georeferenced images

This folder contains the input data for a non-georeferenced PyImageTrack workflow. It is intended for image sequences captured with a camera rather than georeferenced raster data.

## Required files

```text
image_01.jpg
image_02.jpg
image_pairs.csv
moving_area.shp
moving_area.shx
moving_area.dbf
moving_area.prj
stable_area.shp
stable_area.shx
stable_area.dbf
stable_area.prj
```

The four files with the same base name form one shapefile. The moving-area shapefile defines the area in which movement is expected. The stable-area shapefile defines the reference area used for image alignment.

`image_pairs.csv` defines which images are compared. Since the images are not georeferenced, the configuration must explicitly enable the non-georeferenced workflow and provide the required camera parameters.

The matching configuration file is:

```text
configs/02_example_non_georeferenced.toml
```

The input and output paths in that file must point to this folder and to:

```text
output/02_example_non_georeferenced/
```

Depth images and 3D displacement conversion are not part of this initial example. They can be added as a later extension.

The files currently present are empty placeholders and must be replaced with valid data before running PyImageTrack.
