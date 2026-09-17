from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import geopandas as gpd
from affine import Affine
from shapely.geometry import Point

from PyImageTrack.DataProcessing.ImagePreprocessing import (
    check_channels_compatible,
    harmonize_dtypes,
    harmonize_resolution,
)
from PyImageTrack.Utils import collect_pairs
from PyImageTrack.run_pipeline import _get, _load_config, _require, _resolve_path
from PyImageTrack.Cache import load_alignment_cache, save_alignment_cache
from PyImageTrack.DataProcessing.DataPostprocessing import (
    filter_lod_points,
    prepare_tracking_results_for_filtering,
)


def test_collect_pairs_builds_successive_pairs_from_real_files(tmp_path: Path):
    first = tmp_path / "2008_08_17_id9109.tif"
    second = tmp_path / "2023_08_17_id9109.tif"
    unrelated = tmp_path / "notes.txt"
    first.touch()
    second.touch()
    unrelated.touch()

    pairs, files, dates, has_filename_time, identifiers = collect_pairs(
        str(tmp_path), pairing_mode="successive", identifier="9109"
    )

    assert pairs == [("2008-08-17_9109", "2023-08-17_9109")]
    assert set(files) == {"2008-08-17_9109", "2023-08-17_9109"}
    assert dates["2008-08-17_9109"] == "2008-08-17"
    assert all(has_filename_time.values())
    assert all(value == "9109" for value in identifiers.values())


def test_collect_pairs_uses_csv_dates_for_files_without_date_tokens(tmp_path: Path):
    (tmp_path / "survey_a.tif").touch()
    (tmp_path / "survey_b.tif").touch()
    dates_csv = tmp_path / "image_dates.csv"
    dates_csv.write_text(
        "dataset,date\nsurvey_a,2019-04-05\nsurvey_b,2020-04-05\n",
        encoding="utf-8",
    )

    pairs, files, dates, has_filename_time = collect_pairs(
        str(tmp_path), date_csv_path=str(dates_csv), pairing_mode="successive"
    )

    assert pairs == [("survey_a", "survey_b")]
    assert set(files) == {"survey_a", "survey_b"}
    assert dates == {"survey_a": "2019-04-05", "survey_b": "2020-04-05"}
    assert has_filename_time == {"survey_a": False, "survey_b": False}


def test_config_loading_and_path_resolution_are_composable(tmp_path: Path):
    config_path = tmp_path / "config.toml"
    config_path.write_text('[input]\nfolder = "images"\nrequired = "yes"\n', encoding="utf-8")

    config = _load_config(config_path)
    assert _get(config, "input", "missing", "fallback") == "fallback"
    assert _require(config, "input", "required") == "yes"
    assert Path(_resolve_path(_get(config, "input", "folder"), tmp_path)) == tmp_path / "images"

    with pytest.raises(KeyError):
        _require(config, "input", "missing")


def test_preprocessing_components_form_a_compatible_pipeline():
    image1 = np.arange(16, dtype=np.uint8).reshape(4, 4)
    image2 = np.arange(9, dtype=np.uint16).reshape(3, 3)

    image1, image2 = harmonize_dtypes(image1, image2)
    check_channels_compatible(image1, image2)
    image1, image2, transform1, transform2 = harmonize_resolution(
        image1, image2, Affine.identity(), Affine.identity()
    )

    assert image1.shape == image2.shape == (3, 3)
    assert transform1.a == pytest.approx(4 / 3)
    assert transform1.e == pytest.approx(4 / 3)
    assert transform2 == Affine.identity()


def test_postprocessing_marks_below_lod_points_and_preserves_geometry():
    results = gpd.GeoDataFrame(
        {"movement_distance_per_year": [0.5, 2.0], "valid": [True, True]},
        geometry=[Point(0, 0), Point(1, 1)],
        crs="EPSG:32632",
    )

    filtered = filter_lod_points(results, 1.0, "movement_distance_per_year")
    assert filtered["is_below_LoD"].tolist() == [True, False]
    assert filtered["valid"].tolist() == [False, True]
    assert filtered.geometry.tolist() == results.geometry.tolist()


def test_prepare_tracking_results_adds_consistent_filter_columns():
    results = gpd.GeoDataFrame(
        {"valid": [True, True], "is_movement_rate_difference_outlier": [True, False]},
        geometry=[Point(0, 0), Point(1, 1)],
    )

    prepared = prepare_tracking_results_for_filtering(results)
    assert prepared.index.tolist() == [0, 1]
    assert prepared["is_outlier"].tolist() == [True, False]


def test_alignment_cache_roundtrip(tmp_path: Path):
    source1 = tmp_path / "first.tif"
    source2 = tmp_path / "second.tif"
    source1.write_bytes(b"first")
    source2.write_bytes(b"second")
    image = np.arange(9, dtype=np.uint16).reshape(3, 3)
    transform = Affine(2, 0, 100, 0, -2, 200)
    original = SimpleNamespace(
        image2_matrix=image,
        image2_transform=transform,
        image2_matrix_truecolor=None,
        depth_image2=None,
        tracked_control_points=None,
        crs=None,
    )
    save_alignment_cache(
        original,
        str(tmp_path / "cache"),
        "first",
        "second",
        {"method": "test"},
        {"first": str(source1), "second": str(source2)},
        {"first": "2000-01-01", "second": "2001-01-01"},
    )

    loaded = SimpleNamespace(
        image1_matrix=image,
        image1_transform=transform,
        depth_image1=None,
        fake_pixel_size=1.0,
        tracking_parameters=SimpleNamespace(search_extent_px=(1, 1, 1, 1), movement_cell_size=1),
        alignment_parameters=SimpleNamespace(control_search_extent_px=(1, 1, 1, 1), control_cell_size=1),
        crs=None,
    )
    assert load_alignment_cache(loaded, str(tmp_path / "cache"), "first", "second")
    np.testing.assert_array_equal(loaded.image2_matrix, image)
    assert loaded.images_aligned is True
