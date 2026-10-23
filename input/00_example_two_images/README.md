# Example 00: Two images

This folder contains the input data for the simplest PyImageTrack workflow: comparing two images.

## Required files

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

The four files with the same base name form one shapefile. Do not provide only the `.shp` file; the accompanying files are required by the geospatial data format.

The moving-area shapefile defines the area in which movement is expected. The stable-area shapefile defines the reference area used for image alignment.

The matching configuration file is:

```text
configs/00_example_two_images.toml
```

The input and output paths in that file must point to this folder and to:

```text
output/00_example_two_images/
```

The files currently present are empty placeholders and must be replaced with valid data before running PyImageTrack.
