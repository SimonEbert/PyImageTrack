from datetime import datetime

from pathlib import Path

import numpy as np
import pytest

from PyImageTrack.DataProcessing.ImagePreprocessing import convert_to_uint16
from PyImageTrack.DataProcessing.ImagePreprocessing import check_channels_compatible, harmonize_dtypes
from PyImageTrack.Cache import alignment_cache_paths, lod_cache_paths, tracking_cache_paths
from PyImageTrack.Utils import (
    extract_identifier,
    make_effective_extents_from_deltas,
    normalize_image_shape,
    find_files_recursive_with_duplicates,
    processing_hash,
    hash_directory_name,
    write_hash_metadata,
    extract_datetime_from_token
)


def test_processing_hash_is_canonical_and_step_specific(tmp_path):
    first = processing_hash("tracking", {"b": 2, "a": [1, 2]})
    second = processing_hash("tracking", {"a": [1, 2], "b": 2})
    changed = processing_hash("tracking", {"a": [1, 3], "b": 2})
    assert first == second
    assert first != changed
    assert hash_directory_name("tracking", first) == f"t_{first[:12]}"

    directory = tmp_path / "tracking"
    directory.mkdir()
    write_hash_metadata(str(directory), "tracking", {"a": 1}, processing_hash("tracking", {"a": 1}))
    write_hash_metadata(str(directory), "tracking", {"a": 1}, processing_hash("tracking", {"a": 1}))
    with pytest.raises(ValueError, match="does not match"):
        write_hash_metadata(str(directory), "tracking", {"a": 2}, processing_hash("tracking", {"a": 2}))


@pytest.mark.parametrize(
    ("filename", "token", "identifier", "expected_date"),
    [
        ("2008_08_17_id9109_az315.tif", "2008-08-17", "9109", datetime(2008, 8, 17)),
        ("HS_2008_08_17_id9109_az315.tif", "2008-08-17", "9109", datetime(2008, 8, 17)),
        ("2008_08_17_IDつ12345_az315.tif", "2008-08-17", None, datetime(2008, 8, 17)),
        ("2008_08_17_1430_id9109.tif", "2008-08-17", "9109", datetime(2008, 8, 17)),
        ("200803171430_id9109.tif", "200803171430", "9109", datetime(2008, 3, 17, 14, 30)),
        ("2008_03_id9109.tif", "2008-03", "9109", datetime(2008, 3, 1)),
        ("24_08_17_id9109.tif", "24-08-17", "9109", datetime(2024, 8, 17)),
        ("2008_08_17.tif", "2008-08-17", None, datetime(2008, 8, 17)),
    ],
)
def test_extract_and_parse_date(filename, token, identifier, expected_date):
    # extracted = extract_date_token(filename)
    #
    # assert extracted == token
    assert extract_identifier(filename) == identifier
    assert extract_datetime_from_token(filename) == expected_date


@pytest.mark.parametrize(
    "filename",
        ["no_date.tif", "HS_nodate.tif"],
)
def test_extract_date_token_returns_none_without_date(filename):
    assert extract_datetime_from_token(filename) is None


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (np.zeros((1, 2, 3)), (2, 3)),
        (np.zeros((2, 3)), (2, 3)),
        (np.zeros((3, 2, 3)), (3, 2, 3)),
    ],
)
def test_normalize_image_shape(value, expected):
    assert normalize_image_shape(value).shape == expected


def test_make_effective_extents_scales_and_caps_each_side():
    assert make_effective_extents_from_deltas(
        (1.4, 2.6, 0, 10), cell_size=8, years_between=2, cap_per_side=12
    ) == (7, 9, 4, 12)


def test_convert_to_uint16_scales_uint8_and_handles_nan():
    np.testing.assert_array_equal(
        convert_to_uint16(np.array([0, 255], dtype=np.uint8)),
        np.array([0, 65280], dtype=np.uint16),
    )
    result = convert_to_uint16(np.array([np.nan, 2.0], dtype=np.float32))
    assert result.dtype == np.uint16
    assert result[0] == 0
    assert result[1] == 65535


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("24", datetime(2024, 1, 1)),
            ("2024-09-03", datetime(2024, 9, 3)),
        ("2024-13-47", datetime(2024, 1, 1)),
        ("202409", datetime(2024, 9, 1)),
    ],
)
def test_parse_date_supported_and_invalid_parts(value, expected):
    assert extract_datetime_from_token(value) == expected


def test_parse_date_rejects_input_without_year():
    with pytest.raises(ValueError, match="Invalid date format"):
        extract_datetime_from_token("not-a-date")


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("idABC123.tif", "ABC123"),
        ("image_id9109_az315.tif", "9109"),
        ("image_ID9109.tif", None),
        ("imageidentifier9109.tif", None),
    ],
)
def test_extract_identifier_boundaries(filename, expected):
    assert extract_identifier(filename) == expected


def test_find_files_recursive_ignores_depth_images_and_rejects_duplicates(tmp_path: Path):
    (tmp_path / "nested").mkdir()
    (tmp_path / "Depth_images").mkdir()
    (tmp_path / "image.tif").touch()
    (tmp_path / "nested" / "other.TIFF").touch()
    (tmp_path / "Depth_images" / "ignored.tif").touch()

    files = find_files_recursive_with_duplicates(str(tmp_path), (".tif", ".tiff"))
    assert set(files) == {"image.tif", "other.TIFF"}

    (tmp_path / "nested" / "image.tif").touch()
    with pytest.raises(ValueError, match="duplicate filenames"):
        find_files_recursive_with_duplicates(str(tmp_path), (".tif", ".tiff"))


def test_harmonize_dtypes_converts_only_when_needed():
    first = np.array([0, 255], dtype=np.uint8)
    second = np.array([1, 2], dtype=np.uint16)
    converted_first, converted_second = harmonize_dtypes(first, second)
    assert converted_first.dtype == converted_second.dtype == np.uint16
    np.testing.assert_array_equal(converted_second, second)

    same_first, same_second = harmonize_dtypes(second, second.copy())
    assert same_first is second
    assert same_second.dtype == np.uint16


def test_check_channels_compatible_accepts_matching_and_rejects_mismatched_images():
    check_channels_compatible(np.zeros((3, 4)), np.ones((3, 4)))
    check_channels_compatible(np.zeros((2, 3, 4)), np.ones((2, 3, 4)))
    with pytest.raises(ValueError):
        check_channels_compatible(np.zeros((2, 3, 4)), np.ones((3, 3, 4)))


def test_cache_paths_include_both_image_ids(tmp_path: Path):
    directory = str(tmp_path)
    assert alignment_cache_paths(directory, "a", "b")[2].endswith("alignment_control_points_a_b.fgb")
    assert tracking_cache_paths(directory, "a", "b")[0].endswith("tracking_raw_a_b.fgb")
    assert lod_cache_paths(directory, "a", "b")[0].endswith("lod_points_a_b.fgb")
