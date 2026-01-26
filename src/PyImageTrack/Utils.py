# PyImageTrack/Utils.py
import itertools
import os
import re
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd


def _round_to_nearest_hour(dt: datetime) -> datetime:
    """Round to nearest hour (>=30 min rounds up)."""
    if dt.minute > 30 or (dt.minute == 30 and (dt.second > 0 or dt.microsecond > 0)):
        dt = dt.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    else:
        dt = dt.replace(minute=0, second=0, microsecond=0)
    return dt


def parse_date(s: str) -> datetime:
    """
    Parse flexible date/time strings. Behavior:
      - Filenames with compact tokens:
        * YYYYMMDD-HHMMSS... -> time kept and rounded to nearest hour
        * YYYYMMDD           -> 00:00:00
        * YYYY-MM-DD         -> 00:00:00
      - Plain strings:
        * 'YYYY-MM-DD HH:MM:SS' or 'YYYY-MM-DD HH:MM' -> rounded to nearest hour
        * 'YYYY-MM-DD' -> 00:00:00
    IMPORTANT: 'YYYY-MM' and 'YYYY' are NOT parsed here for filenames;
               those should be resolved via image_dates.csv and then parsed.
    """
    name = os.path.basename(str(s))

    # YYYYMMDD-HHMMSS...
    m = re.match(r'^(\d{4})(\d{2})(\d{2})[-_](\d{2})(\d{2})(\d{2})', name)
    if m:
        y, mo, d, H, M, S = map(int, m.groups())
        return _round_to_nearest_hour(datetime(y, mo, d, H, M, S))

    # YYYYMMDD (no time)
    m = re.match(r'^(\d{4})(\d{2})(\d{2})', name)
    if m:
        y, mo, d = map(int, m.groups())
        return datetime(y, mo, d, 0, 0, 0)

    # YYYY-MM-DD at start
    m = re.match(r'^(\d{4})[-_](\d{2})[-_](\d{2})', name)
    if m:
        y, mo, d = map(int, m.groups())
        return datetime(y, mo, d, 0, 0, 0)

    # Plain strings (with/without time)
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M",
                "%d-%m-%Y %H:%M:%S", "%d-%m-%Y %H:%M"):
        try:
            dt = datetime.strptime(s, fmt)
            return _round_to_nearest_hour(dt)
        except ValueError:
            pass

    for fmt in ("%Y-%m-%d", "%d-%m-%Y"):
        try:
            dt = datetime.strptime(s, fmt)
            return dt.replace(minute=0, second=0, microsecond=0)
        except ValueError:
            pass

    raise ValueError(f"Unsupported date format: {s!r}")


def _successive_pairs(sorted_years):
    return [(sorted_years[i], sorted_years[i + 1]) for i in range(len(sorted_years) - 1)]


def collect_pairs(input_folder: str,
                  date_csv_path: Optional[str] = None,
                  pairs_csv_path: Optional[str] = None,
                  pairing_mode: str = "all",
                  extensions: Optional[tuple] = None):
    """
    Build pairs and return:
      - year_pairs: list of (id1, id2)
      - id_to_file: id -> tif path
      - id_to_date: id -> date string ("YYYY-MM-DD" or "YYYY-MM-DD HH:00:00")
      - id_hastime_from_filename: id -> bool (True if hour came from filename)

    Behavior:
      - If date_csv_path is None or the file is missing, files whose leading token is only 'YYYY' or 'YYYY-MM'
        will be SKIPPED with a warning. Files with a full day (YYYYMMDD, YYYY-MM-DD, or YYYYMMDD-HHMMSS...)
        will still be used.
    """
    # 1) Try to read image_dates.csv if provided and exists
    csv_year_to_date: dict[str, str] = {}
    if date_csv_path is not None and os.path.exists(date_csv_path):
        date_df = pd.read_csv(date_csv_path)
        date_df.columns = date_df.columns.str.strip()
        if not {"year", "date"}.issubset(date_df.columns):
            raise ValueError("image_dates.csv must contain 'year' and 'date' columns.")
        date_df["year"] = date_df["year"].astype(str)
        csv_year_to_date = dict(zip(date_df["year"], date_df["date"]))
    elif date_csv_path is not None:
        pass

    # 2) Collect all files with allowed extensions
    # Default (None) = keep legacy behavior: only TIF/TIFF
    if extensions is None:
        extensions = (".tif", ".tiff")
    exts = tuple(e.lower() for e in extensions)

    img_files = [f for f in os.listdir(input_folder) if f.lower().endswith(exts)]

    id_to_file = {}
    id_to_date = {}
    id_hastime_from_filename = {}

    for f in img_files:
        token_match = re.match(r'^([0-9][0-9\-_]*)', f)
        if not token_match:
            continue
        lead = token_match.group(1)
        path = os.path.join(input_folder, f)
        lead = lead.rstrip("-_")  # '1953_' -> '1953', '1953-09_' -> '1953-09'

        # Case: YYYYMMDD-HHMMSS...
        if re.match(r'^\d{8}[-_]\d{6}', lead):
            dt = parse_date(lead)  # rounded to nearest hour
            id_ = lead
            id_to_file[id_] = path
            id_to_date[id_] = dt.strftime("%Y-%m-%d %H:00:00")
            id_hastime_from_filename[id_] = True

        # Case: DD-MM-YYYY at start (e.g., 02-09-1953_*.tif)
        elif re.match(r'^\d{2}-\d{2}-\d{4}', lead):
            dt = parse_date(lead)
            id_ = dt.strftime("%Y-%m-%d")
            id_to_file[id_] = path
            id_to_date[id_] = dt.strftime("%Y-%m-%d")
            id_hastime_from_filename[id_] = False

        # Case: YYYY-MM-DD...
        elif re.match(r'^\d{4}-\d{2}-\d{2}', lead):
            dt = parse_date(lead)
            id_ = re.match(r'^(\d{4}-\d{2}-\d{2})', lead).group(1)
            id_to_file[id_] = path
            id_to_date[id_] = dt.strftime("%Y-%m-%d")  # no hour printed later
            id_hastime_from_filename[id_] = False

        # Case: YYYYMMDD (no time)
        elif re.match(r'^\d{8}', lead):
            dt = parse_date(lead)
            id_ = re.match(r'^(\d{8})', lead).group(1)
            id_to_file[id_] = path
            id_to_date[id_] = dt.strftime("%Y-%m-%d")  # no hour printed later
            id_hastime_from_filename[id_] = False

        # Case: YYYY-MM -> use CSV if available, else assume first day of month
        elif re.match(r'^\d{4}-\d{2}$', lead):
            ym = re.match(r'^(\d{4}-\d{2})', lead).group(1)
            if csv_year_to_date and ym in csv_year_to_date:
                date_str = csv_year_to_date[ym]
            else:
                date_str = f"{ym}-01"
                if not csv_year_to_date:
                    print(
                        f"[WARN] Only year+month detected in filename '{f}'. "
                        f"image_dates.csv not found at: {date_csv_path}. "
                        f"Assuming date '{date_str}'."
                    )
                else:
                    print(
                        f"[WARN] Only year+month detected in filename '{f}'. "
                        f"No CSV entry for '{ym}'. Assuming date '{date_str}'."
                    )
            id_ = ym
            id_to_file[id_] = path
            id_to_date[id_] = date_str
            id_hastime_from_filename[id_] = False

        # Case: YYYY -> use CSV if available, else assume Jan 1st
        elif re.match(r'^\d{4}$', lead):
            y = lead[:4]
            if csv_year_to_date and y in csv_year_to_date:
                date_str = csv_year_to_date[y]
            else:
                date_str = f"{y}-01-01"
                if not csv_year_to_date:
                    print(
                        f"[WARN] Only year detected in filename '{f}'. "
                        f"image_dates.csv not found at: {date_csv_path}. "
                        f"Assuming date '{date_str}'."
                    )
                else:
                    print(
                        f"[WARN] Only year detected in filename '{f}'. "
                        f"No CSV entry for '{y}'. Assuming date '{date_str}'."
                    )
            id_ = y
            id_to_file[id_] = path
            id_to_date[id_] = date_str
            id_hastime_from_filename[id_] = False

        else:
            continue

    # 3) Order by actual time
    items = [(k, parse_date(id_to_date[k])) for k in id_to_file.keys() if k in id_to_date]
    items.sort(key=lambda t: t[1])
    ordered_ids = [k for k, _ in items]

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
        pairs_df = pd.read_csv(pairs_csv_path, sep=None, engine="python", encoding="utf-8-sig")
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

        # Helper: extract a leading numeric date token similar to filename handling
        token_re = re.compile(r"^([0-9][0-9\-_]*)")

        def _extract_lead(any_str: str) -> str | None:
            if not isinstance(any_str, str):
                any_str = str(any_str)
            any_str = any_str.strip()
            m = token_re.match(any_str)
            return m.group(1) if m else None

        # Map CSV token -> ID used in id_to_file; relies on already scanned files (id_to_file)
        def _resolve_csv_token_to_id(raw: str) -> str:
            lead = _extract_lead(raw)
            if not lead:
                raise ValueError(f"Unrecognized pair token: {raw!r}")

            # 1) YYYYMMDD-HHMMSS...
            if re.match(r"^\d{8}[-_]\d{6}", lead):
                key = re.match(r"^(\d{8}[-_]\d{6})", lead).group(1)
                if key in id_to_file:
                    return key
                # fallback: try exact match by startswith for very rare cases
                cand = [k for k in id_to_file if k.startswith(key)]
                if cand:
                    return cand[0]
                raise KeyError(f"No file ID matching time token '{key}' found in input folder.")

            # 1b) DD-MM-YYYY (normalize to YYYY-MM-DD and resolve)
            if re.match(r"^\d{2}-\d{2}-\d{4}", lead):
                key = parse_date(lead).strftime("%Y-%m-%d")
                if key in id_to_file:
                    return key
                raise KeyError(
                    f"No file ID for date '{lead}' (normalized to '{key}') found. "
                    f"Make sure a file with that day exists or rename the file prefix."
                )

            # 2) YYYY-MM-DD...
            if re.match(r"^\d{4}-\d{2}-\d{2}", lead):
                key = re.match(r"^(\d{4}-\d{2}-\d{2})", lead).group(1)
                if key in id_to_file:
                    return key
                raise KeyError(f"No file ID for date '{key}' found. Make sure a file with that day exists.")

            # 3) YYYYMMDD
            if re.match(r"^\d{8}$", lead):
                key = lead  # prefer exact YYYYMMDD if present
                if key in id_to_file:
                    return key
                cand = [k for k in id_to_file if str(k).startswith(key)]
                if cand:
                    return sorted(cand)[0]
                raise KeyError(f"No file ID for date '{key}' (YYYYMMDD) found.")

            # 4) YYYY-MM (needs image_dates.csv-driven key)
            if re.match(r"^\d{4}-\d{2}$", lead):
                key = lead  # our id_to_file uses 'YYYY-MM' as key when resolved via CSV
                if key in id_to_file:
                    return key
                raise KeyError(
                    f"No ID for '{key}'. Either provide image_dates.csv entry for this month and ensure a file exists, "
                    f"or switch to a day-resolved token present in filenames."
                )

            # 5) YYYYMM (6 digits) -> treat as YYYY-MM; requires CSV-backed entry
            if re.match(r"^\d{6}$", lead):
                key = f"{lead[:4]}-{lead[4:6]}"
                if key in id_to_file:
                    return key
                raise KeyError(
                    f"No ID for month token '{lead}' (mapped to '{key}'). "
                    f"Provide image_dates.csv and ensure a corresponding file is used."
                )

            # 6) YYYY (map via image_dates.csv if available; otherwise accept year-only ID)
            if re.match(r"^\d{4}$", lead):
                # a) if there is a year-ID (file name 'YYYY')
                if lead in id_to_file:
                    return lead
                # b) otherwise map image_dates.csv
                if not csv_year_to_date:
                    raise KeyError(f"CSV token '{lead}' is a year, but no image_dates.csv was provided.")
                if lead not in csv_year_to_date:
                    raise KeyError(f"Year '{lead}' not found in image_dates.csv.")
                mapped = str(csv_year_to_date[lead]).strip()  # e.g. '02-09-1953'
                try:
                    norm = parse_date(mapped).strftime("%Y-%m-%d")  # '1953-09-02'
                except Exception:
                    norm = mapped
                if norm in id_to_file:
                    return norm
                if mapped in id_to_file:
                    return mapped
                raise KeyError(
                    f"Year '{lead}' mapped to '{mapped}' (normalized '{norm}'), "
                    f"but no input file ID matches it."
                )

            # Otherwise: unsupported
            raise ValueError(f"Unsupported token in image_pairs.csv: {raw!r}")

        # Build pairs using the resolver
        pairs = []
        for _, row in pairs_df.iterrows():
            left_raw = str(row[left_col]).strip()
            right_raw = str(row[right_col]).strip()
            if not left_raw or not right_raw or left_raw.lower() == "nan" or right_raw.lower() == "nan":
                print(f"[WARN] Skipping empty pair row: {row.to_dict()}")
                continue
            try:
                left_id = _resolve_csv_token_to_id(left_raw)
                right_id = _resolve_csv_token_to_id(right_raw)
                pairs.append((left_id, right_id))
            except Exception as e:
                print(f"[WARN] Skipping pair ({left_raw!r}, {right_raw!r}): {e}")

        year_pairs = pairs

    return year_pairs, id_to_file, id_to_date, id_hastime_from_filename


def ensure_dir(path: str):
    """Create directory if missing."""
    os.makedirs(path, exist_ok=True)


def float_compact(x):
    """Compact float to short string without trailing zeros."""
    if isinstance(x, float):
        s = f"{x:.3f}".rstrip("0").rstrip(".")
        return s or "0"
    return str(x)


def _get(obj, name, default="NA"):
    """Return attribute or dict key `name` from `obj`, supporting both objects and dicts."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    # object: use getattr
    return getattr(obj, name, default)


def abbr_alignment(ap):
    """Short code for alignment parameters; supports objects or dicts."""
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


def abbr_tracking(tp):
    """Short code for tracking parameters; supports objects or dicts."""
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
    ]
    parts = [p for p in parts if p not in (None, "", "NA")]
    return "T_" + "_".join(parts)


def abbr_filter(fp) -> str:
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
