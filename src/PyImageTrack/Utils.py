# PyImageTrack/Utils.py
"""
Utility functions for PyImageTrack.

This module provides helper functions for:
- Date parsing and formatting
- Image file collection and pairing
- Parameter abbreviation for cache keys
- Directory management
"""
import itertools
import os
import re
from datetime import datetime, timedelta
from typing import Optional
import numpy as np
from pathlib import Path

import pandas as pd
from pyproj import CRS as PyprojCRS

from .ConsoleOutput import get_console

def _round_to_nearest_hour(dt: datetime) -> datetime:
    """Round to nearest hour (>=30 min rounds up)."""
    if dt.minute > 30 or (dt.minute == 30 and (dt.second > 0 or dt.microsecond > 0)):
        dt = dt.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    else:
        dt = dt.replace(minute=0, second=0, microsecond=0)
    return dt


def normalize_image_shape(arr: np.ndarray) -> np.ndarray:
    """Normalize image shape: ``(1,H,W)->(H,W)``, keep ``(H,W)`` and ``(C,H,W), C>1``."""
    if arr.ndim == 3 and arr.shape[0] == 1:
        return arr[0]
    return arr

def make_effective_extents_from_deltas(deltas, cell_size, years_between=1.0, cap_per_side=None):
    """
    Convert delta-per-year extents (posx,negx,posy,negy) into effective absolute extents
    by adding half the template size per side and scaling deltas by years_between.

    deltas: (dx+, dx-, dy+, dy-) meaning *extra* pixels beyond half the template per year.
    cell_size: movement_cell_size or control_cell_size
    years_between: time span in years between the two images
    cap_per_side: optional int to clamp each side (to keep windows bounded)

    Returns (posx, negx, posy, negy) as ints >= half.
    """
    half = int(cell_size) // 2
    def one(v):
        eff = half + int(round(float(v) * float(years_between)))
        if cap_per_side is not None:
            eff = min(int(cap_per_side), eff)
        return eff #max(half, eff)

    px, nx, py, ny = deltas
    return (one(px), one(nx), one(py), one(ny))
#
# def _validate_date_token(token: str, year_part: str) -> bool:
#     """
#     Validate if a date token is complete and parsable.
#
#     Returns True if all segments in the token are valid (no ignored invalid parts).
#     Compact format: expects 4, 8, or 10+ digits (YYYYMMDD, YYYYMMDDHHMM, YYYYMMDDHHMMSS)
#     Separated format: expects complete date/time pairs (YYYY-MM-DD-HH-MM)
#
#     Parameters
#     ----------
#     token : str
#         The token to validate.
#     year_part : str
#         The year that was extracted first.
#
#     Returns
#     -------
#     bool
#         True if token is complete and valid, False if parts are incomplete or invalid.
#     """
#
#     # Extract year from token
#     year_match = re.match(r'^(\d{2,4})', token)
#     if not year_match:
#         return False
#
#     year_len = len(year_match.group(1))
#     remaining = token[year_len:]
#
#     # Determine format by checking for hyphens
#     has_hyphens = '-' in remaining
#
#     try:
#         if len(year_part) == 2:
#             year = 2000 + int(year_part)
#         else:
#             year = int(year_part)
#
#         if has_hyphens:
#             # Separated format - strip leading hyphen that was already counted by has_hyphens check
#             remaining = remaining.lstrip('-')
#             parts = remaining.split('-')
#
#             if len(parts) >= 1 and parts[0]:
#                 m = int(parts[0])
#                 if not (1 <= m <= 12):
#                     return False
#
#             if len(parts) >= 2 and parts[1]:
#                 d = int(parts[1])
#                 if not (1 <= d <= 31):
#                     return False
#
#             # If we have hour, we MUST have minute too (time must be complete)
#             # Only check for hour if the part is non-empty (not just from trailing hyphen)
#             if len(parts) >= 3 and parts[2] and parts[2].strip():
#                 h = int(parts[2])
#                 if not (0 <= h <= 23):
#                     return False
#                 # Hour without minute or with non-digit minute is incomplete
#                 if len(parts) < 4 or not parts[3] or not parts[3].isdigit():
#                     return False
#                 m = int(parts[3])
#                 if not (0 <= m <= 59):
#                     return False
#         else:
#             # Compact format
#             remaining_digits = remaining.replace('-', '')
#             digit_count = len(remaining_digits)
#
#             # For progressive extraction, we need to accept intermediate states
#             # 0 = year only (valid)
#             # 2 = month only (potentially valid)
#             # 4 = month + day (valid)
#             # 6 = month + day + hour (potentially valid - could add minute)
#             # 8 = month + day + hour + minute (valid)
#             # 10 = month + day + hour + minute + second (valid)
#
#             # Check that we have even-length digits (no partial segments)
#             if digit_count % 2 != 0:
#                 return False  # Odd length means incomplete digit pair
#
#             # Validate only the segments we have so far - don't require completeness
#             if digit_count >= 2:
#                 m = int(remaining_digits[0:2])
#                 if not (1 <= m <= 12):
#                     return False
#
#             if digit_count >= 4:
#                 d = int(remaining_digits[2:4])
#                 if not (1 <= d <= 31):
#                     return False
#
#             if digit_count >= 6:
#                 h = int(remaining_digits[4:6])
#                 if not (0 <= h <= 23):
#                     return False
#
#             if digit_count >= 8:
#                 m = int(remaining_digits[6:8])
#                 if not (0 <= m <= 59):
#                     return False
#
#         return True
#
#     except (ValueError, IndexError):
#         return False

import dateutil.parser
def extract_datetime_from_token(s: str) -> tuple[datetime, str]:
    """Gets a filename (part) and extracts the found datetime token as a datetime object and the respective string given by the user."""
    token_without_id = re.sub(r'id\d+','',s)
    token_without_id = token_without_id.replace(':', '')
    token_without_id = token_without_id.replace('T', '')
    token_without_id = token_without_id.replace('_', '')
    token_without_id = token_without_id.replace('-', '')
    datetime_observation, unused_string_parts = dateutil.parser.parse(token_without_id,yearfirst=True,fuzzy_with_tokens=True,dayfirst=False)
    # Obtain string that was used for parsing in the original cleaned format
    datetime_string_user_clean = re.sub(f"^{re.escape(unused_string_parts[0])}|{re.escape(unused_string_parts[-1])}$", "", token_without_id)
    datetime_string_user = re.search(r"(?:id\d+|[:T_-])*".join(map(re.escape, datetime_string_user_clean)), s).group()

    return datetime_observation, datetime_string_user

def extract_identifier(filename: str) -> Optional[str]:
    """
    Extract identifier from filename using pattern 'id<identifier>'.
    
    The identifier is extracted from the pattern 'id<identifier>' where
    identifier consists of alphanumeric characters and ends at the next
    separator ('-', '_', '.', or end of string).
    
    Parameters
    ----------
    filename : str
        Filename to extract identifier from.
    
    Returns
    -------
    Optional[str]
        The extracted identifier, or None if no identifier pattern is found.
    
    Examples
    --------
    >>> extract_identifier("HS_2008_id9109_az315.tif")
    '9109'
    >>> extract_identifier("image_idAB12_test.tif")
    'AB12'
    >>> extract_identifier("test_id12345.tif")
    '12345'
    >>> extract_identifier("idABC123.tif")
    'ABC123'
    >>> extract_identifier("no_identifier.tif")
    None
    """
    # Match pattern: id followed by alphanumeric characters until separator or end
    # The 'id' must be preceded by a separator or start of string
    # Separators: '-', '_', '.', or end of string
    match = re.search(r'(?:^|[-_.])id([A-Za-z0-9]+)(?:[-_.]|$)', filename)
    if match:
        return match.group(1)
    return None


def _successive_pairs(sorted_years):
    """
    Generate successive pairs from a sorted list.

    Parameters
    ----------
    sorted_years : list
        A sorted list of identifiers.

    Returns
    -------
    list
        A list of tuples representing successive pairs: [(a,b), (b,c), ...].
    """
    return [(sorted_years[i], sorted_years[i + 1]) for i in range(len(sorted_years) - 1)]


def find_files_recursive_with_duplicates(base_folder: str, extensions: tuple):
    """
    Recursively find all files with specified extensions and check for duplicates.
    Exludes files that are stored in a folder called "Depth_images" since these are used for 3d displacement calculation
    Parameters
    ----------
    base_folder : str
        The base folder to search in.
    extensions : tuple
        Tuple of file extensions to filter by (e.g., (".tif", ".tiff")).
    
    Returns
    -------
    dict
        A dictionary mapping filenames to their full paths.
    
    Raises
    ------
    ValueError
        If duplicate filenames are found in different subfolders.
    """
    console = get_console()
    
    # Dictionary to track seen filenames and their paths
    filename_to_paths = {}
    # Dictionary to return: filename -> (full path)
    result = {}
    
    # Recursively walk through all subdirectories
    for root, dirs, files in os.walk(base_folder):
        # Check if folder is called 'Depth_images' --> Then it will be excluded
        if Path(root).parts[-1] == "Depth_images":
            continue
        for filename in files:
            if Path(filename).parent.name == "Depth_images":
                continue
            # Check if file has one of the allowed extensions
            if filename.lower().endswith(extensions):
                full_path = os.path.join(root, filename)
                
                if filename in filename_to_paths:
                    # Duplicate found - add to list of paths for this filename
                    filename_to_paths[filename].append(full_path)
                else:
                    filename_to_paths[filename] = [full_path]
    
    # Check for duplicates and build result
    duplicates_found = []
    for filename, paths in filename_to_paths.items():
        if len(paths) > 1:
            duplicates_found.append((filename, paths))
        else:
            # Only one file, add to result
            result[filename] = paths[0]
    
    # If duplicates found, raise error with detailed information
    if duplicates_found:
        error_msg = "Duplicate filenames found in input folder subdirectories:\n"
        for filename, paths in duplicates_found:
            error_msg += f"\n  {filename} found at:\n"
            for path in paths:
                error_msg += f"    - {path}\n"
        console.error(error_msg)
        raise ValueError(
            f"Found {len(duplicates_found)} file(s) with duplicate filenames. "
            "Please ensure each file has a unique name within the input folder and its subdirectories. "
            f"See console output for details."
        )
    
    console.info_verbose(f"Found {len(result)} files with extensions {extensions} in {base_folder}")
    return result


def collect_pairs(input_folder: str,
                  date_csv_path: Optional[str] = None,
                  pairs_csv_path: Optional[str] = None,
                  pairing_mode: str = "all",
                  extensions: Optional[tuple] = None,
                  identifier: Optional[str] = None):
    """
    Build pairs and return:
      - datetime_pairs: list of (id1, id2)
      - id_to_file: id -> tif path
      - id_to_date: id -> datetime
      - id_hastime_from_filename: id -> bool (True if time came from filename)
      - id_to_identifier: id -> identifier (only if identifier parameter is provided)
    
    Date extraction from filenames:
      - Extracts leading numeric token (YY or YYYY followed by optional parts)
      - Passes token directly to parse_date() for parsing
      - CSV dates handled identically to filename dates
    
    ID Generation:
      - When an identifier is present in the filename, IDs are generated as "date_token_identifier"
        to ensure uniqueness when multiple files share the same date token (e.g., year).
      - When no identifier is present, IDs use only the date token.
      - This prevents collisions when processing files with the same date but different identifiers.
    
    Parameters
    ----------
    input_folder : str
        Path to folder containing image files.
    date_csv_path : Optional[str], optional
        Path to CSV file mapping file IDs to dates. Default is None.
    pairs_csv_path : Optional[str], optional
        Path to CSV file defining custom image pairs. Default is None.
    pairing_mode : str, optional
        Pairing strategy: "all", "successive", "first_to_all", or "custom". Default is "all".
    extensions : Optional[tuple], optional
        Allowed file extensions. Default is None (uses .tif, .tiff).
    identifier : Optional[str], optional
        If provided, only process files matching this identifier. Default is None.
    
    Returns
    -------
    tuple
        (datetime_pairs, id_to_file, id_to_date, id_hastime_from_filename, id_to_identifier)
        where id_to_identifier is only included if identifier parameter is not None.
    """
    console = get_console()
    
    # 1) Try to read image_dates.csv if provided and exists
    csv_year_to_date: dict[str, str] = {}
    if date_csv_path is not None and os.path.exists(date_csv_path):
        try:
            date_df = pd.read_csv(date_csv_path)
            date_df.columns = date_df.columns.str.strip()
            # Check for either "file" or "year" column (or "file/year")
            if "file" in date_df.columns:
                id_col = "file"
            elif "year" in date_df.columns:
                id_col = "year"
            elif "file/year" in date_df.columns:
                id_col = "file/year"
            elif "datetime" in date_df.columns:
                id_col = "datetime"
            elif "file/datetime" in date_df.columns:
                id_col = "file/datetime"
            else:
                raise ValueError("image_dates.csv must contain a 'file', 'year', or 'file/year' column.")
            if "date" not in date_df.columns:
                raise ValueError("image_dates.csv must contain a 'date' column.")
            date_df[id_col] = date_df[id_col].astype(str)
            csv_year_to_date = dict(zip(date_df[id_col], date_df["date"]))
        except Exception as e:
            raise ValueError(f"Failed to read image_dates.csv at {date_csv_path}: {e}")
    elif date_csv_path is not None:
        pass

    # 2) Collect all files with allowed extensions (recursive search with duplicate detection)
    if extensions is None:
        extensions = (".tif", ".tiff")
    exts = tuple(e.lower() for e in extensions)

    # Use recursive search to find all image files in subdirectories
    filename_to_path = find_files_recursive_with_duplicates(input_folder, exts)
    img_files = list(filename_to_path.keys())

    id_to_file = {}
    id_to_date = {}
    id_to_date_user_string = {}
    id_hastime_from_filename = {}

    for f in img_files:
        # Parse the date token
        if True:#try:
            observation_datetime, observation_datetime_user_string = extract_datetime_from_token(f)
        else:#except ValueError:
            # Skip files with invalid date tokens
            console.warning(f"Could not extract datetime from token: {f}")
            continue
        if observation_datetime is None:
            continue

        # Use the full path from recursive search
        path = filename_to_path[f]


        # Extract identifier from filename to make ID unique
        file_identifier = extract_identifier(f)
        
        # Use exact datetime as ID, but make it unique by appending identifier if present
        if file_identifier:
            id_ = f"{observation_datetime.strftime("%Y%m%dT%H%M%S%f")}_{file_identifier}"
        else:
            id_ = observation_datetime.strftime("%Y%m%dT%H%M%S%f")


        id_to_file[id_] = path
        
        # Check if this file has a date override in the CSV file
        # First try to match by full filename (without extension), then by token
        csv_date_str = None
        if f in csv_year_to_date:
            csv_date_str = csv_year_to_date[f]
        elif id_ in csv_year_to_date:
            csv_date_str = csv_year_to_date[id_]
        
        if csv_date_str is not None:
            # Use the date from CSV, parsing it with the same logic as filename dates
            try:
                observation_datetime, observation_datetime_user_string = extract_datetime_from_token(csv_date_str)
                id_hastime_from_filename[id_] = False  # Date came from CSV, not filename
            except ValueError:
                # If CSV date is invalid, fall back to filename date
                id_hastime_from_filename[id_] = True
        else:
            id_hastime_from_filename[id_] = True

        id_to_date[id_] = observation_datetime
        id_to_date_user_string[id_] = observation_datetime_user_string

    # 3) Order by actual time
    items = [(k, id_to_date[k]) for k in id_to_file.keys() if k in id_to_date]
    items.sort(key=lambda t: t[1])
    ordered_ids = [k for k, _ in items]

    # 3.5) Filter by identifier if provided
    id_to_identifier = {}
    if identifier is not None:
        # Build mapping of file IDs to their identifiers
        for id_ in list(id_to_file.keys()):
            # Extract identifier from the ID (format: "date_token_identifier" or "date_token")
            # The identifier is the part after the last underscore if it exists
            parts = id_.rsplit('_', 1)
            if len(parts) == 2 and parts[1].isalnum():
                file_identifier = parts[1]
            else:
                # No identifier in ID, extract from filename
                file_identifier = extract_identifier(os.path.basename(id_to_file[id_]))
            
            id_to_identifier[id_] = file_identifier
            # Filter out files that don't match the specified identifier
            if file_identifier != identifier:
                id_to_file.pop(id_, None)
                id_to_date.pop(id_, None)
                id_hastime_from_filename.pop(id_, None)
        # Re-sort after filtering
        items = [(k, extract_datetime_from_token(id_to_date[k])) for k in id_to_file.keys() if k in id_to_date]
        items.sort(key=lambda t: t[1])
        ordered_ids = [k for k, _ in items]
        console.info_verbose(f"Files remaining after identifier filtering: {len(ordered_ids)}")

    # 4) Build pairs
    if pairing_mode == "all":
        # every id with every later id
        datetime_pairs = list(itertools.combinations(ordered_ids, 2))

    elif pairing_mode == "successive":
        # only neighbours: (t1,t2), (t2,t3), ...
        def _successive_pairs(ids):
            return [(ids[i], ids[i + 1]) for i in range(len(ids) - 1)]
        datetime_pairs = _successive_pairs(ordered_ids)

    elif pairing_mode == "first_to_all":
        # always use the first id as anchor: (first, second), (first, third), ...
        if len(ordered_ids) < 2:
            datetime_pairs = []
        else:
            anchor = ordered_ids[0]
            datetime_pairs = [(anchor, other) for other in ordered_ids[1:]]

    elif pairing_mode == "custom":
        # --- read CSV with auto delimiter (',' or ';') ---
        if pairs_csv_path is None:
            console.error("Pairing mode is set to 'custom', but no image_pairs.csv file was provided.")
            console.error("To fix this, you have two options:")
            console.error("  1. Provide a valid path to an image_pairs.csv file in your config:")
            console.error("     pairs_csv_path = '/path/to/image_pairs.csv'")
            console.error("     The CSV must contain columns 'date_earlier' and 'date_later'.")
            console.error("  2. Change the pairing mode to an automatic option:")
            console.error("     - 'all': Pair each image with every other image")
            console.error("     - 'successive': Pair consecutive images (1-2, 2-3, 3-4, ...)")
            console.error("     - 'first_to_all': Pair the first image with all subsequent images")
            raise ValueError("Pairing mode is 'custom' but no image_pairs.csv file was provided.")
        if not os.path.exists(pairs_csv_path):
            raise FileNotFoundError(f"image_pairs.csv file does not exist: {pairs_csv_path}")
        if not os.access(pairs_csv_path, os.R_OK):
            raise PermissionError(f"image_pairs.csv file is not readable: {pairs_csv_path}")
        try:
            pairs_df = pd.read_csv(pairs_csv_path, sep=None, engine="python", encoding="utf-8-sig")
        except Exception as e:
            raise ValueError(f"Failed to read image_pairs.csv file: {e}") from e
        pairs_df.columns = (
            pairs_df.columns
            .str.replace("\ufeff", "", regex=False)
            .str.strip()
            .str.lower()
        )
        if not {"date_earlier", "date_later"}.issubset(pairs_df.columns):
            raise ValueError(
                "image_pairs.csv must contain columns 'date_earlier' and 'date_later'."
            )

        left_col, right_col = "date_earlier", "date_later"

        # Map CSV token -> ID used in id_to_file
        def _resolve_csv_token_to_id(raw: str) -> str:
            lead = extract_datetime_from_token(raw)
            if lead is None:
                raise ValueError(f"Unrecognized pair token: {raw!r}")
            
            # Try exact match first
            if lead in id_to_file:
                return lead


            raise KeyError(
                f"No file ID matching token '{raw}' found in input folder. "
                f"Make sure a file with that date prefix exists."
            )

        # Build pairs using the resolver
        pairs = []
        for _, row in pairs_df.iterrows():
            left_raw = str(row[left_col]).strip()
            right_raw = str(row[right_col]).strip()
            if not left_raw or not right_raw or left_raw.lower() == "nan" or right_raw.lower() == "nan":
                console.warning(f"Skipping empty pair row: {row.to_dict()}")
                continue
            try:
                left_id = _resolve_csv_token_to_id(left_raw)
                right_id = _resolve_csv_token_to_id(right_raw)
                pairs.append((left_id, right_id))
            except Exception as e:
                console.error(f"Skipping pair ({left_raw!r}, {right_raw!r}): {e}")

        datetime_pairs = pairs

    # Return with or without id_to_identifier based on whether identifier was provided
    if identifier is not None:
        return datetime_pairs, id_to_file, id_to_date, id_to_date_user_string, id_hastime_from_filename, id_to_identifier
    else:
        return datetime_pairs, id_to_file, id_to_date, id_to_date_user_string, id_hastime_from_filename


def ensure_dir(path: str):
    """
    Create a directory if it doesn't exist.

    Parameters
    ----------
    path : str
        Path to the directory to create.
    """
    os.makedirs(path, exist_ok=True)


def float_compact(x):
    """
    Convert a float to a compact string without trailing zeros.

    Parameters
    ----------
    x : float or any
        Value to convert. If not a float, returns str(x).

    Returns
    -------
    str
        Compact string representation without trailing zeros.
    """
    if isinstance(x, float):
        s = f"{x:.3f}".rstrip("0").rstrip(".")
        return s or "0"
    return str(x)


def _get(obj, name, default="NA"):
    """
    Get an attribute or dict key from an object.

    Supports both dictionary and object access patterns.

    Parameters
    ----------
    obj : dict or object
        Object to get the value from.
    name : str
        Name of the attribute or key.
    default : any, optional
        Default value to return if the attribute/key is not found. Default is "NA".

    Returns
    -------
    any
        The value of the attribute/key, or the default if not found.
    """
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    # object: use getattr
    return getattr(obj, name, default)


def abbr_alignment(ap):
    """
    Generate a short code for alignment parameters.

    This code is used for creating cache directory names that uniquely
    identify the alignment configuration.

    Parameters
    ----------
    ap : AlignmentParameters or dict
        Alignment parameters object or dictionary.

    Returns
    -------
    str
        Short code string starting with "A_".
    """
    parts = []
    image_bands = _get(ap, 'image_bands', None)
    if image_bands is not None:
        parts.append(f"IB{image_bands}".replace(" ",""))

    # control extents (posx,negx,posy,negy) if provided
    ext = _get(ap, "control_search_extent_px", None)
    if ext:
        try:
            parts.append(f"AS{int(ext[0])}_{int(ext[1])}_{int(ext[2])}_{int(ext[3])}")
        except Exception:
            parts.append(f"AS{ext}")  # fallback

    parts += [
        f"CP{_get(ap, 'number_of_control_points')}",
        f"CC{_get(ap, 'control_cell_size')}",
        f"CCa{float_compact(_get(ap, 'cross_correlation_threshold_alignment'))}",
    ]
    # drop empty/None/NA fragments
    parts = [p for p in parts if p not in (None, "", "NA")]
    return "A_" + "_".join(parts)

def _part(prefix: str, value: str | None) -> str | None:
    if value is None:
        return None

    return f"{prefix}{str(value)}"


def abbr_tracking(tp):
    """
    Generate a short code for tracking parameters.

    This code is used for creating cache directory names that uniquely
    identify the tracking configuration.

    Parameters
    ----------
    tp : TrackingParameters or dict
        Tracking parameters object or dictionary.

    Returns
    -------
    str
        Short code string starting with "T_".
    """
    parts = []
    # movement extents (posx,negx,posy,negy)
    ext = _get(tp, "search_extent_px", None)
    if ext:
        try:
            parts.append(f"TS{int(ext[0])}_{int(ext[1])}_{int(ext[2])}_{int(ext[3])}")
        except Exception:
            parts.append(f"TS{ext}")  # fallback

    parts += [
        f"DP{_get(tp, 'distance_of_tracked_points_px')}",
        f"MC{_get(tp, 'movement_cell_size')}",
        f"CC{float_compact(_get(tp, 'cross_correlation_threshold_movement'))}",
        _part("MPnb", _get(tp, 'nb_initial_estimate_peaks', None)),
        _part("MPth", _get(tp, 'correlation_threshold_initial_estimates', None)),
        _part("MPd", _get(tp, 'min_distance_initial_estimates', None)),    ]
    if _get(tp, "use_adaptive_tracking_window", False):
        parts += [
            f"adaptive"
        ]

    parts = [p for p in parts if p not in (None, "", "NA")]
    return "T_" + "_".join(parts)


def abbr_filter(fp) -> str:
    """
    Generate a short code for filter parameters.

    This code is used for creating cache directory names that uniquely
    identify the filtering configuration.

    Parameters
    ----------
    fp : FilterParameters
        Filter parameters object.

    Returns
    -------
    str
        Short code string starting with "F_".
    """
    fc = float_compact
    parts = [
        f"LoDq{fc(fp.level_of_detection_quantile)}",
        f"N{fp.number_of_points_for_level_of_detection}",
        f"dB{fc(fp.difference_movement_bearing_threshold)}",
        f"dBw{fc(fp.difference_movement_bearing_moving_window_size)}",
        f"sdB{fc(fp.standard_deviation_movement_bearing_threshold)}",
        f"sdBw{fc(fp.standard_deviation_movement_bearing_moving_window_size)}",
        f"dR{fc(fp.difference_movement_rate_threshold)}",
        f"dRw{fc(fp.difference_movement_rate_moving_window_size)}",
        f"sdR{fc(fp.standard_deviation_movement_rate_threshold)}",
        f"sdRw{fc(fp.standard_deviation_movement_rate_moving_window_size)}",
    ]
    return "F_" + "_".join(parts)


def abbr_output_units(mode: str) -> str:
    """
    Generate a short code for the output units mode.

    Parameters
    ----------
    mode : str
        Output units mode ("per_year" or "total").

    Returns
    -------
    str
        Short code string starting with "U_".
    """
    return "U_" + mode


def abbr_enhancement(ep) -> str:
    """
    Generate a short code for image enhancement parameters.

    This code is used for creating cache directory names that uniquely
    identify the enhancement configuration.

    Parameters
    ----------
    ep : dict or object
        Enhancement parameters object or dictionary.

    Returns
    -------
    str
        Short code string starting with "E_".
    """
    fc = float_compact
    enhancement_type = _get(ep, "type", "none")
    
    if enhancement_type == "none" or not enhancement_type:
        return "E_none"
    
    parts = []
    
    if enhancement_type == "clahe":
        kernel_size = _get(ep, "kernel_size", 50)
        clip_limit = _get(ep, "clip_limit", 0.9)
        parts.append(f"clahe")
        parts.append(f"K{kernel_size}")
        parts.append(f"C{fc(clip_limit)}")
    elif enhancement_type == "gamma":
        gamma = _get(ep, "gamma", 1.0)
        parts.append(f"gamma")
        parts.append(f"G{fc(gamma)}")
    elif enhancement_type == "histeq":
        parts.append(f"histeq")
    elif enhancement_type == "denoise":
        denoise_type = _get(ep, "denoise_type", "gaussian")
        strength = _get(ep, "strength", 1.0)
        parts.append(f"denoise")
        parts.append(f"D{denoise_type}")
        parts.append(f"S{fc(strength)}")
    else:
        # Unknown enhancement type - just use the type name
        parts.append(enhancement_type)
    
    # Drop empty/None/NA fragments
    parts = [p for p in parts if p not in (None, "", "NA")]
    return "E_" + "_".join(parts)

def _recompute_lod_from_points(image_pair, filter_params) -> bool:
    points = getattr(image_pair, "level_of_detection_points", None)
    if points is None or len(points) == 0:
        return False
    quantile = filter_params.level_of_detection_quantile
    if quantile is None or "movement_distance_per_year" not in points.columns:
        return False
    image_pair.level_of_detection = np.nanquantile(points["movement_distance_per_year"], quantile)
    if points.crs is not None:
        crs_obj = PyprojCRS.from_user_input(points.crs)
        unit_name = crs_obj.axis_info[0].unit_name if crs_obj.is_projected else "meter"
    else:
        unit_name = "pixel"
    print("Found level of detection with quantile " + str(quantile) + " as "
          + str(np.round(image_pair.level_of_detection, decimals=5)) + " " + str(unit_name) + "/year")
    return True



