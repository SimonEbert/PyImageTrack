#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyImageTrack Batch Processor

This module provides batch processing functionality for PyImageTrack, allowing
multiple configurations to be processed with automatic filtering based on
identifiers extracted from filenames.

The batch processor reads two CSV tables:
- Table A: Maps configuration files to classes
- Table B: Maps identifiers to classes

For each configuration in Table A, the processor:
1. Finds all identifiers in Table B with the same class
2. Filters identifiers to only those present in the input folder
3. Processes each identifier separately using the configuration
4. Replaces wildcards in shapefile names with the identifier
"""

import os
import sys
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import pandas as pd
import tomllib

from .ConsoleOutput import get_console, ConsoleOutput
from .run_pipeline import run_from_config
from .Utils import extract_identifier


def read_table_a(path: str, config_col: str, class_col: str) -> pd.DataFrame:
    """
    Read Table A (config mapping) from CSV file.

    Parameters
    ----------
    path : str
        Path to the CSV file.
    config_col : str
        Name of the column containing configuration file names.
    class_col : str
        Name of the column containing class names.

    Returns
    -------
    pd.DataFrame
        DataFrame with the config mapping data.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If required columns are missing.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Table A file does not exist: {path}")
    
    try:
        df = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
    except Exception as e:
        raise ValueError(f"Failed to read Table A file: {e}") from e
    
    # Clean column names
    df.columns = df.columns.str.strip()
    
    # Validate required columns
    if config_col not in df.columns:
        raise ValueError(f"Column '{config_col}' not found in Table A. Available columns: {list(df.columns)}")
    if class_col not in df.columns:
        raise ValueError(f"Column '{class_col}' not found in Table A. Available columns: {list(df.columns)}")
    
    return df[[config_col, class_col]]


def read_table_b(path: str, identifier_col: str, class_col: str) -> pd.DataFrame:
    """
    Read Table B (identifier classification) from CSV file.

    Parameters
    ----------
    path : str
        Path to the CSV file.
    identifier_col : str
        Name of the column containing identifiers.
    class_col : str
        Name of the column containing class names.

    Returns
    -------
    pd.DataFrame
        DataFrame with the identifier classification data.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If required columns are missing.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Table B file does not exist: {path}")
    
    try:
        df = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
    except Exception as e:
        raise ValueError(f"Failed to read Table B file: {e}") from e
    
    # Clean column names
    df.columns = df.columns.str.strip()
    
    # Validate required columns
    if identifier_col not in df.columns:
        raise ValueError(f"Column '{identifier_col}' not found in Table B. Available columns: {list(df.columns)}")
    if class_col not in df.columns:
        raise ValueError(f"Column '{class_col}' not found in Table B. Available columns: {list(df.columns)}")
    
    return df[[identifier_col, class_col]]


def filter_identifiers_by_class(table_b: pd.DataFrame, class_name: str,
                                 identifier_col: str, class_col: str) -> List[str]:
    """
    Filter Table B to get all identifiers for a given class.

    Parameters
    ----------
    table_b : pd.DataFrame
        Table B DataFrame.
    class_name : str
        The class to filter by.
    identifier_col : str
        Name of the identifier column.
    class_col : str
        Name of the class column.

    Returns
    -------
    List[str]
        List of identifiers for the given class.
    """
    # Filter by class
    filtered = table_b[table_b[class_col] == class_name]
    
    # Get unique identifiers
    identifiers = filtered[identifier_col].dropna().unique().tolist()
    
    # Convert to strings and strip whitespace
    identifiers = [str(id_).strip() for id_ in identifiers if str(id_).strip()]
    
    return identifiers


def get_input_folder_from_config(config_path: str) -> str:
    """
    Get the input folder path from a configuration file.

    Parameters
    ----------
    config_path : str
        Path to the TOML configuration file.

    Returns
    -------
    str
        The input folder path.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist.
    KeyError
        If the input_folder is not specified in the config.
    """
    config_path_obj = Path(config_path).expanduser()
    if not config_path_obj.is_absolute():
        # Try configs/ folder if relative
        if not config_path_obj.exists():
            configs_path = Path("configs") / config_path_obj
            if configs_path.exists():
                config_path_obj = configs_path
    
    if not config_path_obj.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with config_path_obj.open("rb") as f:
        cfg = tomllib.load(f)
    
    if "paths" not in cfg or "input_folder" not in cfg["paths"]:
        raise KeyError(f"Config file {config_path} does not contain 'paths.input_folder'")
    
    return cfg["paths"]["input_folder"]


def get_identifiers_in_folder(input_folder: str) -> set:
    """
    Get all identifiers present in the input folder and its subfolders.

    Parameters
    ----------
    input_folder : str
        Path to the input folder.

    Returns
    -------
    set
        Set of identifiers found in the folder and its subfolders.
    """
    identifiers = set()
    
    if not os.path.isdir(input_folder):
        return identifiers
    
    # Recursively search all subdirectories
    for root, dirs, files in os.walk(input_folder):
        for filename in files:
            identifier = extract_identifier(filename)
            if identifier:
                identifiers.add(identifier)
    
    return identifiers


def run_batch(table_a_path: str, config_col: str, class_col_a: str,
              table_b_path: str, identifier_col: str, class_col_b: str,
              verbose: bool = False, quiet: bool = False, use_colors: bool = True,
              log_file: str = None, log_level: str = 'INFO',
              log_max_bytes: int = 10 * 1024 * 1024, log_backup_count: int = 5):
    """
    Run batch processing of multiple configurations with identifier filtering.

    This function:
    1. Reads Table A (config mapping) and Table B (identifier classification)
    2. For each config in Table A:
       - Gets the class associated with the config
       - Filters Table B to find all identifiers with that class
       - For each identifier, runs the pipeline with that identifier
    3. Replaces wildcards in shapefile names with the identifier

    Parameters
    ----------
    table_a_path : str
        Path to Table A CSV file (config mapping).
    config_col : str
        Column name in Table A containing configuration file names.
    class_col_a : str
        Column name in Table A containing class names.
    table_b_path : str
        Path to Table B CSV file (identifier classification).
    identifier_col : str
        Column name in Table B containing identifiers.
    class_col_b : str
        Column name in Table B containing class names.
    verbose : bool, optional
        Enable verbose output. Default is False.
    quiet : bool, optional
        Enable quiet mode. Default is False.
    use_colors : bool, optional
        Use ANSI colors in output. Default is True.
    log_file : str, optional
        Path to log file. Default is None.
    log_level : str, optional
        Logging level. Default is 'INFO'.
    log_max_bytes : int, optional
        Maximum log file size in bytes. Default is 10MB.
    log_backup_count : int, optional
        Number of backup log files. Default is 5.

    Returns
    -------
    Tuple[int, int, List[Tuple[str, str]]]
        (total_configs_processed, total_identifiers_processed, errors)
        where errors is a list of (config, identifier) tuples that failed.
    """
    # Initialize console
    console = ConsoleOutput(
        verbose=verbose,
        quiet=quiet,
        use_colors=use_colors,
        log_file=None,  # Log file will be set per config
        log_level=log_level,
        log_max_bytes=log_max_bytes,
        log_backup_count=log_backup_count
    )
    
    console.show_batch_banner(mode='start')
    console.info("Starting PyImageTrack batch processing")
    
    # Read tables
    console.info(f"Reading Table A from: {table_a_path}")
    table_a = read_table_a(table_a_path, config_col, class_col_a)
    console.info(f"  Found {len(table_a)} config entries")
    
    console.info(f"Reading Table B from: {table_b_path}")
    table_b = read_table_b(table_b_path, identifier_col, class_col_b)
    console.info(f"  Found {len(table_b)} identifier entries")
    
    # Process each config
    total_configs_processed = 0
    total_identifiers_processed = 0
    errors = []
    
    for idx, row in table_a.iterrows():
        config_name = str(row[config_col]).strip()
        class_name = str(row[class_col_a]).strip()
        
        # Skip empty rows
        if not config_name or config_name.lower() == 'nan':
            console.warning(f"Skipping row {idx + 1}: empty config name")
            continue
        if not class_name or class_name.lower() == 'nan':
            console.warning(f"Skipping row {idx + 1}: empty class name")
            continue
        
        # Resolve config path: try as-is first, then try configs/ folder
        config_path = config_name
        if not os.path.isabs(config_path):
            # Try relative path from current directory
            if not os.path.exists(config_path):
                # Try configs/ folder
                configs_path = os.path.join("configs", config_name)
                if os.path.exists(configs_path):
                    config_path = configs_path
                    console.info_verbose(f"Using config from configs/ folder: {config_path}")
                else:
                    console.warning(f"Config file not found: {config_name} (also tried configs/{config_name})")
                    continue
        
        console.section_header("CONFIG", f"Processing configuration: {config_path}", f"(class: {class_name})", level=1)
        
        # Get identifiers for this class
        identifiers = filter_identifiers_by_class(table_b, class_name, identifier_col, class_col_b)
        
        if not identifiers:
            console.warning(f"No identifiers found for class '{class_name}'. Skipping config '{config_name}'.")
            continue
        
        console.info(f"Found {len(identifiers)} identifiers for class '{class_name}'")
        console.info_verbose(f"  Identifiers: {', '.join(identifiers)}")
        
        # Get input folder from config and filter identifiers by what's actually present
        try:
            input_folder = get_input_folder_from_config(config_path)
            console.info_verbose(f"Input folder from config: {input_folder}")
            
            identifiers_in_folder = get_identifiers_in_folder(input_folder)
            console.info_verbose(f"Identifiers found in input folder: {', '.join(identifiers_in_folder)}")
            
            # Filter to only identifiers that exist in the input folder
            identifiers = [id_ for id_ in identifiers if id_ in identifiers_in_folder]
            
            if not identifiers:
                console.warning(f"No identifiers from class '{class_name}' found in input folder '{input_folder}'. Skipping config '{config_name}'.")
                continue
            
            console.info(f"Processing {len(identifiers)} identifiers that exist in input folder: {', '.join(identifiers)}")
        except Exception as e:
            console.warning(f"Could not filter identifiers by input folder: {e}. Processing all {len(identifiers)} identifiers.")
        
        # Process each identifier
        for identifier in identifiers:
            console.section_header("IDENTIFIER", f"Processing identifier: {identifier}", level=2)
            
            try:
                # Run pipeline with this identifier
                run_from_config(
                    config_path=config_path,
                    verbose=verbose,
                    quiet=quiet,
                    use_colors=use_colors,
                    log_file=log_file,
                    log_level=log_level,
                    log_max_bytes=log_max_bytes,
                    log_backup_count=log_backup_count,
                    identifier=identifier
                )
                total_identifiers_processed += 1
                console.success(f"Successfully processed identifier '{identifier}' with config '{config_name}'")
            except Exception as e:
                console.error(f"Failed to process identifier '{identifier}' with config '{config_name}': {e}")
                errors.append((config_name, identifier))
        
        total_configs_processed += 1
        
        # Show "Next Process" banner if there are more configs to process
        if idx < len(table_a) - 1:
            console.show_batch_banner(mode='next')
    
    # Print summary
    console.section_header("SUMMARY", "Batch processing complete", level=1)
    console.info(f"Total configs processed: {total_configs_processed}")
    console.info(f"Total identifiers processed: {total_identifiers_processed}")
    console.info(f"Total errors: {len(errors)}")
    
    if errors:
        console.warning("Errors occurred:")
        for config_name, identifier in errors:
            console.warning(f"  - Config: {config_name}, Identifier: {identifier}")
    
    return total_configs_processed, total_identifiers_processed, errors


def main(argv=None):
    """
    Command-line interface entry point for batch processing.

    Parameters
    ----------
    argv : list, optional
        Command-line arguments. If None, uses sys.argv.
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="PyImageTrack Batch Processor: Process multiple configs with identifier filtering"
    )
    parser.add_argument("--table-a", required=True, 
                       help="Path to Table A CSV file (config mapping)")
    parser.add_argument("--config-col", required=True, default="config_name",
                       help="Column name in Table A containing configuration file names (default: config_name)")
    parser.add_argument("--class-col-a", required=True, default="class",
                       help="Column name in Table A containing class names (default: class)")
    parser.add_argument("--table-b", required=True,
                       help="Path to Table B CSV file (identifier classification)")
    parser.add_argument("--identifier-col", required=True, default="identifier",
                       help="Column name in Table B containing identifiers (default: identifier)")
    parser.add_argument("--class-col-b", required=True, default="class",
                       help="Column name in Table B containing class names (default: class)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--quiet", action="store_true", help="Enable quiet mode (minimal output)")
    parser.add_argument("--no-color", action="store_true", help="Disable colored output")
    parser.add_argument("--log-file", type=str, default=None,
                       help="Path to log file (default: pyimagetrack.log in output folder)")
    parser.add_argument("--log-level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                       help="Logging level for file output (default: INFO)")
    parser.add_argument("--log-max-bytes", type=int, default=10 * 1024 * 1024,
                       help="Maximum log file size before rotation (default: 10MB)")
    parser.add_argument("--log-backup-count", type=int, default=5,
                       help="Number of backup log files to keep (default: 5)")
    
    args = parser.parse_args(argv)
    
    try:
        run_batch(
            table_a_path=args.table_a,
            config_col=args.config_col,
            class_col_a=args.class_col_a,
            table_b_path=args.table_b,
            identifier_col=args.identifier_col,
            class_col_b=args.class_col_b,
            verbose=args.verbose,
            quiet=args.quiet,
            use_colors=not args.no_color,
            log_file=args.log_file,
            log_level=args.log_level,
            log_max_bytes=args.log_max_bytes,
            log_backup_count=args.log_backup_count
        )
    except Exception as e:
        import traceback
        sys.stderr.write(f"\nERROR: {e}\n")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
