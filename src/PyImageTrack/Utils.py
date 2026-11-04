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
import hashlib
import json
import os
import re
from datetime import datetime, timedelta
from typing import Optional
import numpy as np
from pathlib import Path
import dateutil.parser


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

def extract_datetime_from_token(s: str) -> tuple[datetime | None, str]:
    """Gets a filename (part) and extracts the found datetime token as a datetime object and the respective string given by the user."""
    console = get_console()

    token_without_id = re.sub(r'id\d+','',s)

    token_without_id = token_without_id.replace(':', '-')
    token_without_id = token_without_id.replace('T', '-')
    token_without_id = token_without_id.replace('_', '-')

    default_datetime = datetime(2000,1,1,0,0,0,0)
    DATE_RE = re.compile(r"""
    (?<!\d)
    (?P<year>(?:19|20)\d{2})
    (?:
      (?P<sep>[-./]?)
      (?P<month>0[1-9]|1[0-2])
      (?:
        (?P=sep)
        (?P<day>0[1-9]|[12]\d|3[01])
        (?:
          [T_/ .-]?
          (?P<hour>[01]\d|2[0-3])
          (?:
            (?P<tsep>[-:._/ ]?)
            (?P<minute>[0-5]\d)
            (?:
              (?P=tsep)
              (?P<second>[0-5]\d)
              (?:[.,](?P<frac>\d+))?
            )?
          )?
        )?
      )?
    )?
    (?!\d)
    """, re.VERBOSE)

    YY_RE = re.compile(r"""
    (?<!\d)
    (?P<yy>\d{2})
    (?P<sep>[-./])
    (?P<month>0[1-9]|1[0-2])
    (?:
      (?P=sep)
      (?P<day>0[1-9]|[12]\d|3[01])
      (?:
          [T_/ .-]?
          (?P<hour>[01]\d|2[0-3])
          (?:
            (?P<tsep>[-:._/ ]?)
            (?P<minute>[0-5]\d)
            (?:
              (?P=tsep)
              (?P<second>[0-5]\d)
              (?:[.,](?P<frac>\d+))?
            )?
          )?
        )?
    )?
    (?!\d)
    """, re.VERBOSE)


    def _from_groups(g):
        """Build a datetime from DATE_RE groups, applying the defaults."""
        frac = (g["frac"] or "0")[:6].ljust(6, "0")  # microseconds
        return datetime(
            int(g["year"]), int(g["month"] or 1), int(g["day"] or 1),
            int(g["hour"] or 0), int(g["minute"] or 0), int(g["second"] or 0), int(frac),
        )

    def _parse_four_digit(m):
        """Try dateutil first; fall back to the regex groups if it fails or disagrees."""
        return _from_groups(m.groupdict())

    def _parse_two_digit(m):
        yy = int(m["yy"])
        year = 2000 + yy
        return datetime(year, int(m["month"]), int(m["day"] or 1))

    datetime_guess_four_digit = DATE_RE.search(token_without_id)
    datetime_guess_two_digit = YY_RE.search(token_without_id)  # fallback only if no 4-digit-year date exists

    try:
        if datetime_guess_four_digit:
            parsed_datetime = _parse_four_digit(datetime_guess_four_digit)
            return parsed_datetime, datetime_guess_four_digit.group()
        if datetime_guess_two_digit:
            return _parse_two_digit(datetime_guess_two_digit), datetime_guess_two_digit.group()
        if (not datetime_guess_four_digit) and (not datetime_guess_two_digit):
            console.warning("Was not able to extract datetime from string " + s)
            return None, s
    except ValueError:  # impossible dates such as 2008-02-31
        if datetime_guess_four_digit:
            console.warning("Was not able to parse datetime from extracted string " + datetime_guess_four_digit.group())
            return None, datetime_guess_four_digit.group()
        elif datetime_guess_two_digit:
            console.warning("Was not able to parse datetime from extracted string " + datetime_guess_two_digit.group())
            return None, datetime_guess_two_digit.group()
        else:
            console.warning("An unexpected error occurred while trying to parse datetime from string " + token_without_id)
    return None, token_without_id


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


def _match_csv_image_key(filename_without_ext: str, filename: str, csv_keys) -> Optional[str]:
    """
    Resolve an image_dates.csv key for a filename.

    Exact filename matches are preferred. If no exact match exists, allow a CSV
    key to match the beginning of the filename stem when the next character is a
    separator. This supports dataset IDs like "1953002" in filenames such as
    "1953002_HS.tif" without treating "19530020.tif" as the same dataset.
    """
    for candidate in (filename_without_ext, filename):
        if candidate in csv_keys:
            return os.path.splitext(candidate)[0] if candidate == filename else candidate

    prefix_matches = []
    for key in csv_keys:
        if not filename_without_ext.startswith(key):
            continue
        if len(filename_without_ext) == len(key) or filename_without_ext[len(key)] in ("_", "-", ".", " "):
            prefix_matches.append(key)

    if not prefix_matches:
        return None

    prefix_matches.sort(key=len, reverse=True)
    if len(prefix_matches) > 1 and len(prefix_matches[0]) == len(prefix_matches[1]):
        return None
    return prefix_matches[0]


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
    
    Date extraction:
      - If image_dates.csv contains the filename without extension or image ID,
        its date is used even when the filename has no date token.
      - Otherwise, extracts a numeric token (YY or YYYY followed by optional parts)
        from the filename and passes it directly to parse_date().
      - CSV dates are parsed identically to filename dates.
    
    ID Generation:
      - When image_dates.csv matches a filename without extension or image ID,
        that CSV key becomes the image ID.
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
            date_df = pd.read_csv(date_csv_path, sep=None, engine="python", encoding="utf-8-sig")
            date_df.columns = (
                date_df.columns
                .str.replace("\ufeff", "", regex=False)
                .str.strip()
                .str.lower()
            )
            # Check for either "dataset", "file", or legacy "year" columns.
            if "dataset" in date_df.columns:
                id_col = "dataset"
            elif "file" in date_df.columns:
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
                raise ValueError("image_dates.csv must contain a 'dataset', 'file', 'year', or 'file/year' column.")
            if "date" not in date_df.columns:
                raise ValueError("image_dates.csv must contain a 'date' column.")
            date_df[id_col] = date_df[id_col].astype(str).str.strip()
            date_df["date"] = date_df["date"].astype(str).str.strip()
            csv_year_to_date = {
                key: value
                for key, value in zip(date_df[id_col], date_df["date"])
                if key and key.lower() != "nan" and value and value.lower() != "nan"
            }
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
    id_to_identifier = {}

    for f in img_files:
        filename_without_ext = os.path.splitext(f)[0]
        path = filename_to_path[f]
        file_identifier = extract_identifier(f)
        csv_date_str = None
        csv_id = _match_csv_image_key(filename_without_ext, f, csv_year_to_date.keys())
        if csv_id is not None:
            csv_date_str = csv_year_to_date[csv_id]
        use_datetime_from_filename = csv_date_str is None
        if not use_datetime_from_filename:
            try:
                observation_datetime, observation_datetime_user_string = extract_datetime_from_token(csv_date_str)
                if observation_datetime is None:
                    console.warning("Skipping csv parsing for image file " + f + " and attempting to read date from filename.")
                    raise ValueError
                id_ = csv_id
                id_to_file[id_] = path
                id_to_identifier[id_] = file_identifier
                id_hastime_from_filename[id_] = False
                id_to_date[id_] = observation_datetime
                id_to_date_user_string[id_] = observation_datetime_user_string
            except ValueError:
                # Fall back to filename parsing below if the CSV date is invalid.
                use_datetime_from_filename = True

        if use_datetime_from_filename:
            # Use the date from filename
            observation_datetime, observation_datetime_user_string = extract_datetime_from_token(f)
            if observation_datetime is None:
                console.warning("Skipping file " + f + " since no datetime parsing for this was possible.")
                continue
            # Use the original token as ID, but make it unique by appending identifier if present
            if file_identifier:
                id_ = f"{observation_datetime.strftime("%Y%m%dT%H%M%S%f")}_{file_identifier}" if observation_datetime is not None else observation_datetime_user_string
            else:
                id_ = observation_datetime.strftime(
                    "%Y%m%dT%H%M%S%f") if observation_datetime is not None else observation_datetime_user_string

            id_hastime_from_filename[id_] = True
            id_to_file[id_] = path
            id_to_identifier[id_] = file_identifier
            id_to_date[id_] = observation_datetime
            id_to_date_user_string[id_] = observation_datetime_user_string

    # 3) Order by actual time
    items = [(k, id_to_date[k]) for k in id_to_file.keys() if k in id_to_date]
    items.sort(key=lambda t: t[1])
    ordered_ids = [k for k, _ in items]

    # 3.5) Filter by identifier if provided
    if identifier is not None:
        # Build mapping of file IDs to their identifiers
        for id_ in list(id_to_file.keys()):
            file_identifier = id_to_identifier.get(id_) or extract_identifier(os.path.basename(id_to_file[id_]))
            id_to_identifier[id_] = file_identifier
            # Filter out files that don't match the specified identifier
            if file_identifier != identifier:
                id_to_file.pop(id_, None)
                id_to_date.pop(id_, None)
                id_hastime_from_filename.pop(id_, None)
        # Re-sort after filtering
        items = [(k, id_to_date[k]) for k in id_to_file.keys() if k in id_to_date]
        items.sort(key=lambda t: t[1])
        ordered_ids = [k for k, _ in items]
        console.info_verbose(f"Files remaining after identifier filtering: {len(ordered_ids)}")
    # 4) Build pairs
    if pairing_mode == "all":
        # every id with every later id
        datetime_pairs = list(itertools.combinations(ordered_ids, 2))

    elif pairing_mode == "successive":
        # only neighbours: (t1,t2), (t2,t3), ...
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
            console.error("     The CSV must contain columns 'dataset_earlier' and 'dataset_later'")
            console.error("     or the legacy columns 'date_earlier' and 'date_later'.")
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
        if {"dataset_earlier", "dataset_later"}.issubset(pairs_df.columns):
            left_col, right_col = "dataset_earlier", "dataset_later"
        elif {"date_earlier", "date_later"}.issubset(pairs_df.columns):
            left_col, right_col = "date_earlier", "date_later"
        else:
            raise ValueError(
                "image_pairs.csv must contain columns 'dataset_earlier' and 'dataset_later' "
                "or legacy columns 'date_earlier' and 'date_later'."
            )

        # Map CSV token -> ID used in id_to_file
        def _resolve_csv_token_to_id(raw: str) -> str:
            raw_s = str(raw).strip()

            if raw_s in id_to_file:
                return raw_s
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


HASH_VERSION = "1"


def canonical_json(value) -> str:
    """Serialize configuration values deterministically for hashing and metadata."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def processing_hash(step: str, parameters: dict, logic_version: str = "1") -> str:
    """Return a stable full SHA-256 hash for one processing step."""
    payload = {"hash_version": HASH_VERSION, "step": step,
               "logic_version": logic_version, "parameters": parameters}
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def file_hash(path: str) -> str | None:
    """Return a SHA-256 file hash, or ``None`` when no file is available."""
    if not path or not Path(path).is_file():
        return None
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_directory_name(prefix: str, digest: str) -> str:
    """Create the compact directory name while retaining the processing-step prefix."""
    short_prefixes = {
        "enhancement": "e",
        "alignment": "a",
        "tracking": "t",
        "filter": "f",
    }
    return f"{short_prefixes.get(prefix, prefix)}_{digest[:12]}"


def write_hash_metadata(directory: str, step: str, parameters: dict,
                        digest: str, logic_version: str = "1") -> None:
    """Create or validate the human-readable metadata for a hashed directory."""
    metadata_path = Path(directory) / "config.json"
    expected = {
        "hash": digest,
        "hash_version": HASH_VERSION,
        "step": step,
        "logic_version": logic_version,
        "parameters": parameters,
    }
    expected = json.loads(canonical_json(expected))
    if metadata_path.exists():
        try:
            with metadata_path.open(encoding="utf-8") as handle:
                actual = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid cache metadata: {metadata_path}") from exc
        if actual != expected:
            raise ValueError(f"Cache metadata does not match expected configuration: {metadata_path}")
        return
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(expected, handle, indent=2, sort_keys=True, default=str)


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
        f"MPR{float_compact(_get(tp, 'minimum_valid_tracking_point_ratio', 0.0))}",
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
    image_enhancement = _get(ep, "steps", ep)
    if image_enhancement:
        step_codes = [_abbr_enhancement_step(step) for step in image_enhancement]
        canonical = json.dumps(
            image_enhancement,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        config_hash = hashlib.sha256(canonical).hexdigest()[:8].upper()
        return "E_" + "_".join(step_codes) + "_H" + config_hash

    return "E_none"


def _abbr_enhancement_step(ep) -> str:
    """
    Generate a short code fragment for one image enhancement step.

    Parameters
    ----------
    ep : dict or object
        Enhancement step parameters.
    Returns a short method code without parameters.
    """
    enhancement_type = _get(ep, "type", "none")
    method_codes = {
        "clahe": "CL",
        "percentile_stretch": "PS",
        "adaptive_gamma": "AG",
        "histogram_matching": "HM",
        "local_contrast_normalization": "LCN",
        "multi_scale_retinex": "MSR",
        "difference_of_gaussians": "DOG",
        "unsharp_masking": "USM",
        "naof": "NAOF",
        "gradient_magnitude": "GM",
        "snowmask": "SM",
        "nodatamask": "ND",
        "gaussian": "G",
    }
    if enhancement_type not in method_codes:
        raise ValueError(f"Unsupported image enhancement type: {enhancement_type}")
    return method_codes[enhancement_type]

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



