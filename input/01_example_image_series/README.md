# Example 01: Image series

This folder contains the input data for a PyImageTrack time-series workflow. Several images are compared over time, usually using consecutive image pairs.

## Required files

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

The four files with the same base name form one shapefile. The moving-area shapefile defines the area in which movement is expected. The stable-area shapefile defines the reference area used for image alignment.

`image_dates.csv` provides the dates associated with the images. The image identifiers in this file must match the image filenames or the date tokens extracted from them.

The matching configuration file is:

```text
configs/01_example_image_series.toml
```

The input and output paths in that file must point to this folder and to:

```text
output/01_example_image_series/
```

The files currently present are empty placeholders and must be replaced with valid data before running PyImageTrack.
