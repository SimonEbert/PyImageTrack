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
from datetime import datetime
from typing import Optional

import pandas as pd

from .ConsoleOutput import get_console

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
        return max(half, eff)
    px, nx, py, ny = deltas
    return (one(px), one(nx), one(py), one(ny))

def _validate_date_token(token: str, year_part: str) -> bool:
    """
    Validate if a date token is complete and parsable.
    
    Returns True if all segments in the token are valid (no ignored invalid parts).
    Compact format: expects 4, 8, or 10+ digits (YYYYMMDD, YYYYMMDDHHMM, YYYYMMDDHHMMSS)
    Separated format: expects complete date/time pairs (YYYY-MM-DD-HH-MM)
    
    Parameters
    ----------
    token : str
        The token to validate.
    year_part : str
        The year that was extracted first.
    
    Returns
    -------
    bool
        True if token is complete and valid, False if parts are incomplete or invalid.
    """
    from datetime import datetime
    
    # Extract year from token
    year_match = re.match(r'^(\d{2,4})', token)
    if not year_match:
        return False
    
    year_len = len(year_match.group(1))
    remaining = token[year_len:]
    
    # Determine format by checking for hyphens
    has_hyphens = '-' in remaining
    
    try:
        if len(year_part) == 2:
            year = 2000 + int(year_part)
        else:
            year = int(year_part)
        
        if has_hyphens:
            # Separated format - strip leading hyphen that was already counted by has_hyphens check
            remaining = remaining.lstrip('-')
            parts = remaining.split('-')
            
            if len(parts) >= 1 and parts[0]:
                m = int(parts[0])
                if not (1 <= m <= 12):
                    return False
                
            if len(parts) >= 2 and parts[1]:
                d = int(parts[1])
                if not (1 <= d <= 31):
                    return False
                
            # If we have hour, we MUST have minute too (time must be complete)
            # Only check for hour if the part is non-empty (not just from trailing hyphen)
            if len(parts) >= 3 and parts[2] and parts[2].strip():
                h = int(parts[2])
                if not (0 <= h <= 23):
                    return False
                # Hour without minute or with non-digit minute is incomplete
                if len(parts) < 4 or not parts[3] or not parts[3].isdigit():
                    return False
                m = int(parts[3])
                if not (0 <= m <= 59):
                    return False
        else:
            # Compact format
            remaining_digits = remaining.replace('-', '')
            digit_count = len(remaining_digits)
            
            # For progressive extraction, we need to accept intermediate states
            # 0 = year only (valid)
            # 2 = month only (potentially valid)
            # 4 = month + day (valid)
            # 6 = month + day + hour (potentially valid - could add minute)
            # 8 = month + day + hour + minute (valid)
            # 10 = month + day + hour + minute + second (valid)
            
            # Check that we have even-length digits (no partial segments)
            if digit_count % 2 != 0:
                return False  # Odd length means incomplete digit pair
            
            # Validate only the segments we have so far - don't require completeness
            if digit_count >= 2:
                m = int(remaining_digits[0:2])
                if not (1 <= m <= 12):
                    return False
            
            if digit_count >= 4:
                d = int(remaining_digits[2:4])
                if not (1 <= d <= 31):
                    return False
            
            if digit_count >= 6:
                h = int(remaining_digits[4:6])
                if not (0 <= h <= 23):
                    return False
            
            if digit_count >= 8:
                m = int(remaining_digits[6:8])
                if not (0 <= m <= 59):
                    return False
        
        return True
        
    except (ValueError, IndexError):
        return False


def extract_date_token(s: str) -> Optional[str]:
    """
    Extract the complete date token from a string.
    
    Normalizes the string by replacing underscores with hyphens and extracts
    the date token starting with a year (YY or YYYY) from anywhere in the string.
    The token includes the year and any following date/time parts with separators.
    
    This is used in conjunction with parse_date() to ensure consistent token
    extraction across the codebase. The extraction stops when an invalid part
    is encountered (e.g., a letter or a value that doesn't make sense for dates).
    
    Parameters
    ----------
    s : str
        String to extract date token from (e.g., filename).
    
    Returns
    -------
    Optional[str]
        The extracted date token, or None if no valid token is found.
    
    Examples
    --------
    >>> extract_date_token("2023-09-01-1504")
    '2023-09-01-1504'
    >>> extract_date_token("HS_2023_09_01_1504_xyz.tif")
    '2023-09-01-1504'
    >>> extract_date_token("image_20230317_1504.tif")
    '202303171504'
    >>> extract_date_token("no_date.tif")
    None
    """
    # Shortcut: if s looks like a complete date token already, return it
    # This handles the case where parse_date is called on already-extracted tokens
    # Completeness check: should have year, optionally month/day
    # For separated format: YYYY-MM or YYYY-MM-DD or YYYY-MM-DD-HH-MM
    # For compact format: YYYY, YYYYMMDD, YYYYMMDDHHMM, etc.
    s_norm = s.replace('_', '-')
    full_match = re.match(r'^(\d{2,4})(?:-\d{2}(?:-\d{2}(?:-\d{2}(?:-\d{2})?)?)?)?$', s_norm)
    if full_match:
        # Check if it's more than just year
        matched_str = full_match.group(0)
        year_only = full_match.group(1)
        if len(matched_str) > len(year_only):
            # Already looks like a valid complete token
                return s_norm
    
    # Store original to check for separators before normalization
    original = s
    
    # Normalize: replace underscores with hyphens for consistent parsing
    normalized = s.replace('_', '-')
    
    # Extract date token starting with year (YY or YYYY) anywhere in the string
    # The token may include separators and continues while the pattern makes sense for dates
    match = re.search(r'(\d{2,4})', normalized)
    if not match:
        return None
    
    year_part = match.group(1)
    token = year_part
    # FIX: Use the actual match position, not just the year length
    year_pos = match.start()
    remaining = normalized[year_pos + len(year_part):]
    
    # Check if original has separator (hyphen or underscore) immediately after the year
    # This determines whether we use separated format (2023-03-16)
    # or compact format (20230316)
    year_idx = original.find(year_part)
    has_separator = False
    if year_idx >= 0 and year_idx + len(year_part) < len(original):
        next_char = original[year_idx + len(year_part)]
        has_separator = next_char in ('-', '_')
    
    # Progressive extraction: build token incrementally and validate completeness
    # Stop when adding more parts makes the date incomplete or invalid
    
    # Match all characters that could be part of the date
    all_match = re.match(r'^([0-9\\-]*)(?=[^0-9\\-]|$)', remaining)
    if all_match:
        all_additional = all_match.group(1)
        all_additional = all_additional.lstrip('-').rstrip('-')
        
        if all_additional:
            # FIX: Base format decision on the actual extracted content, not the original string
            # If all_additional contains hyphens, treat as separated format
            # If all_additional is all digits, treat as compact format
            extracted_has_hyphen = '-' in all_additional
            
            if extracted_has_hyphen:
                # Separated mode: split by hyphens and add each segment as a whole
                segments = all_additional.split('-')
                validated_token = token
                
                for seg in segments:
                    # FIX: Check for identifier pattern before proceeding
                    if seg.startswith('id'):
                        # This is an identifier segment, stop extraction
                        break
                    
                    if not seg:
                        continue
                    
                    # Only accept numeric segments at least 2 digits long
                    if not seg.isdigit() or len(seg) < 2:
                        break
                    
                    # FIX: Handle compact time segments (4 digits = HHMM)
                    # If segment is 4 digits and we already have month/day, split it into hour/minute
                    current_token_parts = len(validated_token.split('-'))
                    if len(seg) == 4 and current_token_parts >= 3:
                        # This is likely a compact time segment, try splitting it
                        hour_part = seg[0:2]
                        minute_part = seg[2:4]
                        
                        # First add just the hour
                        hour_token = validated_token + '-' + hour_part
                        if _validate_date_token(hour_token, year_part):
                            validated_token = hour_token
                            # Then add the minute
                            minute_token = validated_token + '-' + minute_part
                            if _validate_date_token(minute_token, year_part):
                                validated_token = minute_token
                            # If validation fails, stop
                            break
                        # If hour validation fails, stop
                        break
                    
                    # Try adding this segment with a hyphen separator
                    test_token = validated_token + '-' + seg
                    
                    # FIX: Don't validate partial tokens - only validate when we have a chance of being complete
                    # For a date token to be valid after adding a segment, the segment should be:
                    # - 2 digits (month or day or hour/minute)
                    # - And make the token more complete
                    valid = _validate_date_token(test_token, year_part)
                    
                    if valid:
                        validated_token = test_token
                    else:
                        # If adding this segment makes it invalid, we're done
                        # But also check if adding more might help (e.g., we added partial day)
                        # For simplicity, stop here - the token is no longer valid
                        break
                
                token = validated_token
            else:
                # Compact mode: all_additional should be all digits
                if not all_additional.isdigit():
                    # Not pure digits - not a valid compact format
                    token = token
                else:
                    # Compact mode: add characters in 2-character pairs (month, day, etc.)
                    validated_token = token
                    pos = 0
                    
                    while pos + 2 <= len(all_additional):
                        seg = all_additional[pos:pos+2]
                        
                        # Try adding this 2-character segment directly
                        test_token = validated_token + seg
                        
                        # Check if the test token's remaining length is valid
                        remaining_after_add = len(test_token) - len(year_part)
                        
                        # Compact format should have even-length remainder after year
                        # (2 for month, 4 for month+day, 6 for month+day+hour, etc.)
                        if remaining_after_add % 2 != 0:
                            # Odd length = incomplete, stop
                            break
                        
                        if _validate_date_token(test_token, year_part):
                            validated_token = test_token
                            pos += 2
                        else:
                            # Invalid, stop
                            break
                    
                    token = validated_token
    
    return token


def parse_date(s: str) -> datetime:
    """
    Parse ISO-standard date strings with flexible separators.
    
    Only accepts dates starting with year (YY or YYYY).
    Supports separators: '-', '_', or none.
    Missing or invalid parts default to first standard (month=01, day=01, etc.).
    No rounding - exact values are used.
    
    Parameters
    ----------
    s : str
        Date string to parse.
    
    Returns
    -------
    datetime
        Parsed datetime object.
    
    Raises
    ------
    ValueError
        If the date format is not recognized or invalid.
    
    Supported Formats
    ------------------
    Year only:           2024, 24
    Year-Month:          2024-09, 2024_09, 202409, 24-09, 24_09, 2409
    Year-Month-Day:      2024-09-01, 2024_09_01, 20240901, 24-09-01, 24_09_01, 240901
    With time:           2024-09-01-14-30-45, 2024_09_01_14_30_45, 20240901143045
    
    Separators: '-', '_', or none
    Missing or invalid parts default to: month=01, day=01, hour=00, minute=00, second=00
    No rounding - exact values are used
    
    Note
    ----
    If a part is present but invalid (e.g., month=13, day=47, hour=99), it is
    ignored and the default value is used. This allows filenames like
    "2008_9109" to be parsed as year-only (2008-01-01), or "2024_09_47" to
    be parsed as year-month (2024-09-01).
    """
    console = get_console()
    
    # Pre-extract and normalize the date token using our helper
    token = extract_date_token(s)
    if token is None:
        console.error(f"Invalid date format: {s!r}")
        console.error("")
        console.error("Supported formats (ISO standard, year-first):")
        console.error("  Year only:           2024, 24")
        console.error("  Year-Month:          2024-09, 2024_09, 202409, 24-09, 24_09, 2409")
        console.error("  Year-Month-Day:      2024-09-01, 2024_09_01, 20240901, 24-09-01, 24_09_01, 240901")
        console.error("  With time:           2024-09-01-14-30-45, 2024_09_01_14_30_45, 20240901143045")
        console.error("")
        console.error("Separators: '-', '_', or none")
        console.error("Missing or invalid parts default to: month=01, day=01, hour=00, minute=00, second=00")
        console.error("No rounding - exact values are used")
        raise ValueError(f"Invalid date format: {s!r}")
    
    # Extract year from the token (first 2-4 digits)
    year_match = re.match(r'^(\d{2,4})', token)
    year_part = year_match.group(1)
    
    # Determine year (2-digit -> 2000s, 4-digit -> as-is)
    if len(year_part) == 2:
        year = 2000 + int(year_part)
    elif len(year_part) == 4:
        year = int(year_part)
    else:
        console.error(f"Invalid year format: {year_part!r}")
        raise ValueError(f"Invalid year format: {year_part!r}")
    
    # Default values
    month = 1
    day = 1
    hour = 0
    minute = 0
    second = 0
    
    # Get the remaining part after the year
    remaining = token[len(year_part):]
    
    # Check if there are separators
    if '-' in remaining:
        # Strip leading hyphens that might be present from the year separator
        remaining = remaining.lstrip('-')
        # Parse with separators (e.g., 2016-03-19 or 2016-03-19-14-30-45)
        parts = remaining.split('-')
        
        # Extract and validate values - ignore invalid parts
        if len(parts) >= 1 and parts[0]:
            try:
                m = int(parts[0])
                if 1 <= m <= 12:
                    month = m
            except ValueError:
                pass  # Keep default month
        
        if len(parts) >= 2 and parts[1]:
            try:
                d = int(parts[1])
                if 1 <= d <= 31:
                    day = d
            except ValueError:
                pass  # Keep default day
        
        if len(parts) >= 3 and parts[2]:
            try:
                h = int(parts[2])
                if 0 <= h <= 23:
                    hour = h
            except ValueError:
                pass  # Keep default hour
        
        if len(parts) >= 4 and parts[3]:
            try:
                m = int(parts[3])
                if 0 <= m <= 59:
                    minute = m
            except ValueError:
                pass  # Keep default minute
        
        if len(parts) >= 5 and parts[4]:
            try:
                s = int(parts[4])
                if 0 <= s <= 59:
                    second = s
            except ValueError:
                pass  # Keep default second
    else:
        # Parse without separators (e.g., 20160319 or 20160319143045)
        # Extract month (2 digits after year)
        if len(remaining) >= 2:
            try:
                m = int(remaining[0:2])
                if 1 <= m <= 12:
                    month = m
            except ValueError:
                pass  # Keep default month
        
        # Extract day (next 2 digits)
        if len(remaining) >= 4:
            try:
                d = int(remaining[2:4])
                if 1 <= d <= 31:
                    day = d
            except ValueError:
                pass  # Keep default day
        
        # Extract hour (next 2 digits)
        if len(remaining) >= 6:
            try:
                h = int(remaining[4:6])
                if 0 <= h <= 23:
                    hour = h
            except ValueError:
                pass  # Keep default hour
        
        # Extract minute (next 2 digits)
        if len(remaining) >= 8:
            try:
                m = int(remaining[6:8])
                if 0 <= m <= 59:
                    minute = m
            except ValueError:
                pass  # Keep default minute
        
        # Extract second (next 2 digits)
        if len(remaining) >= 10:
            try:
                s = int(remaining[8:10])
                if 0 <= s <= 59:
                    second = s
            except ValueError:
                pass  # Keep default second
    
    return datetime(year, month, day, hour, minute, second)


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
        for filename in files:
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
      - year_pairs: list of (id1, id2)
      - id_to_file: id -> tif path
      - id_to_date: id -> date string ("YYYY-MM-DD" or "YYYY-MM-DD HH:MM:SS")
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
        (year_pairs, id_to_file, id_to_date, id_hastime_from_filename, id_to_identifier)
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
    id_hastime_from_filename = {}

    for f in img_files:
        # Extract date token using our helper function
        lead = extract_date_token(f)
        if lead is None:
            continue
        
        # Use the full path from recursive search
        path = filename_to_path[f]
        
        # Parse the date token
        try:
            dt = parse_date(lead)
        except ValueError:
            # Skip files with invalid date tokens
            continue
        
        # Extract identifier from filename to make ID unique
        file_identifier = extract_identifier(f)
        
        # Use the original token as ID, but make it unique by appending identifier if present
        if file_identifier:
            id_ = f"{lead}_{file_identifier}"
        else:
            id_ = lead
        
        id_to_file[id_] = path
        
        # Check if this file has a date override in the CSV file
        # First try to match by full filename (without extension), then by token
        filename_without_ext = os.path.splitext(f)[0]
        csv_date_str = None
        if filename_without_ext in csv_year_to_date:
            csv_date_str = csv_year_to_date[filename_without_ext]
        elif id_ in csv_year_to_date:
            csv_date_str = csv_year_to_date[id_]
        
        if csv_date_str is not None:
            # Use the date from CSV, parsing it with the same logic as filename dates
            try:
                csv_dt = parse_date(csv_date_str)
                dt = csv_dt  # Override with CSV date
                id_hastime_from_filename[id_] = False  # Date came from CSV, not filename
            except ValueError:
                # If CSV date is invalid, fall back to filename date
                id_hastime_from_filename[id_] = True
        else:
            id_hastime_from_filename[id_] = True
        
        # Format date string (with time if present, without if not)
        if dt.hour != 0 or dt.minute != 0 or dt.second != 0:
            id_to_date[id_] = dt.strftime("%Y-%m-%d %H:%M:%S")
        else:
            id_to_date[id_] = dt.strftime("%Y-%m-%d")

    # 3) Order by actual time
    items = [(k, parse_date(id_to_date[k])) for k in id_to_file.keys() if k in id_to_date]
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
        items = [(k, parse_date(id_to_date[k])) for k in id_to_file.keys() if k in id_to_date]
        items.sort(key=lambda t: t[1])
        ordered_ids = [k for k, _ in items]
        console.info_verbose(f"Files remaining after identifier filtering: {len(ordered_ids)}")

    # 4) Build pairs
    if pairing_mode == "all":
        # every id with every later id
        year_pairs = list(itertools.combinations(ordered_ids, 2))

    elif pairing_mode == "successive":
        # only neighbours: (t1,t2), (t2,t3), ...
        def _successive_pairs(ids):
            return [(ids[i], ids[i + 1]) for i in range(len(ids) - 1)]
        year_pairs = _successive_pairs(ordered_ids)

    elif pairing_mode == "first_to_all":
        # always use the first id as anchor: (first, second), (first, third), ...
        if len(ordered_ids) < 2:
            year_pairs = []
        else:
            anchor = ordered_ids[0]
            year_pairs = [(anchor, other) for other in ordered_ids[1:]]

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
            lead = extract_date_token(raw)
            if lead is None:
                raise ValueError(f"Unrecognized pair token: {raw!r}")
            
            # Try exact match first
            if lead in id_to_file:
                return lead
            
            # Try prefix match (for cases where filename has additional suffixes)
            candidates = [k for k in id_to_file.keys() if k.startswith(lead)]
            if candidates:
                return sorted(candidates)[0]
            
            raise KeyError(
                f"No file ID matching token '{lead}' found in input folder. "
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

        year_pairs = pairs

    # Return with or without id_to_identifier based on whether identifier was provided
    if identifier is not None:
        return year_pairs, id_to_file, id_to_date, id_hastime_from_filename, id_to_identifier
    else:
        return year_pairs, id_to_file, id_to_date, id_hastime_from_filename


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
        f"IB{_get(tp, 'image_bands')}",
        f"DP{_get(tp, 'distance_of_tracked_points_px')}",
        f"MC{_get(tp, 'movement_cell_size')}",
        f"CC{float_compact(_get(tp, 'cross_correlation_threshold_movement'))}",
        _part("MPnb", _get(tp, 'nb_initial_estimate_peaks', None)),
        _part("MPth", _get(tp, 'correlation_threshold_initial_estimates', None)),
        _part("MPd", _get(tp, 'min_distance_initial_estimates', None)),
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
    if mode == "per_year":
        return "U_per_year"
    elif mode == "total":
        return "U_total"
    else:
        return f"U_{mode}"


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
