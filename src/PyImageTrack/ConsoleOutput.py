#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Console Output Utility Module

Provides consistent, formatted console output for PyImageTrack with:
- Section headers with step names
- Status indicators ([OK], [!], [X], [i], [~])
- Timing information
- Parameter summaries
- Log file support (plain text, no ANSI codes)
- Verbose/quiet modes
- Configurable log levels
- Log rotation support
"""

import logging
import logging.handlers
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List


class ConsoleOutput:
    """
    Centralized console output utility for PyImageTrack.
    
    Features:
    - Consistent formatting with section headers
    - Status indicators ([OK], [!], [X], [i], [~])
    - Timing measurements
    - Parameter summaries
    - Log file support
    - Verbose/quiet modes
    """
    
    # ANSI color codes for terminal output
    COLORS = {
        'reset': '\033[0m',
        'bold': '\033[1m',
        'dim': '\033[2m',
        'red': '\033[31m',
        'green': '\033[32m',
        'yellow': '\033[33m',
        'blue': '\033[34m',
        'magenta': '\033[35m',
        'cyan': '\033[36m',
        'white': '\033[37m',
    }
    
    # Status icons (classic ASCII)
    ICONS = {
        'success': '[OK]',
        'warning': '[!]',
        'error': '[X]',
        'info': '[i]',
        'processing': '[~]',
    }
    
    def __init__(self,
                 verbose: bool = False,
                 quiet: bool = False,
                 use_colors: bool = True,
                 log_file: Optional[Path] = None,
                 log_level: str = 'INFO',
                 log_max_bytes: int = 10 * 1024 * 1024,
                 log_backup_count: int = 5):
        """
        Initialize ConsoleOutput.
        
        Parameters
        ----------
        verbose : bool
            Enable verbose output (more detailed information)
        quiet : bool
            Enable quiet mode (minimal output)
        use_colors : bool
            Use ANSI colors in output (disable for non-terminal output)
        log_file : Path or None
            Path to log file. If None, no log file is written.
        log_level : str
            Logging level for file output (DEBUG, INFO, WARNING, ERROR, CRITICAL).
            Independent of verbose/quiet settings.
        log_max_bytes : int
            Maximum size of log file before rotation (default: 10MB).
        log_backup_count : int
            Number of backup log files to keep (default: 5).
        """
        self.verbose = verbose
        self.quiet = quiet
        self.use_colors = use_colors and sys.stdout.isatty()
        self.log_file = log_file
        self.log_level = log_level.upper()
        self.log_max_bytes = log_max_bytes
        self.log_backup_count = log_backup_count
        
        # Setup logging for log file
        self._setup_logging()
        
        # Timing context stack
        self._timing_stack: List[Dict[str, Any]] = []
        
        # Track if banner has been shown
        self._banner_shown = False
    
    def _setup_logging(self):
        """Setup logging for log file output with rotation support."""
        self.logger = logging.getLogger('PyImageTrack')
        self.logger.setLevel(logging.DEBUG)
        
        # Remove existing handlers
        self.logger.handlers.clear()
        
        if self.log_file:
            # Ensure log directory exists
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Rotating file handler with configurable max size and backup count
            file_handler = logging.handlers.RotatingFileHandler(
                self.log_file,
                mode='a',
                maxBytes=self.log_max_bytes,
                backupCount=self.log_backup_count,
                encoding='utf-8'
            )
            
            # Set the log level for file output (independent of console)
            file_handler.setLevel(getattr(logging, self.log_level, logging.INFO))
            
            # Plain text formatter (no ANSI codes)
            file_formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
    
    def _colorize(self, text: str, color: str) -> str:
        """Apply color to text if colors are enabled."""
        if not self.use_colors:
            return text
        return f"{self.COLORS.get(color, '')}{text}{self.COLORS['reset']}"
    
    def _log(self, message: str, level: str = 'info'):
        """
        Write plain text message to log file if configured.
        
        This method logs the original message WITHOUT any ANSI color codes,
        ensuring log files remain clean and parseable.
        """
        if self.log_file:
            log_level = getattr(logging, level.upper(), logging.INFO)
            self.logger.log(log_level, message)
    
    def print(self, message: str = '', color: str = None, level: str = 'info'):
        """
        Print message to console and log file.
        
        Parameters
        ----------
        message : str
            Message to print
        color : str
            Color to apply (from COLORS dict)
        level : str
            Logging level (debug, info, warning, error)
        """
        # Log the original message BEFORE colorization (plain text)
        # This happens regardless of quiet mode - logs should always be written
        self._log(message, level)
        
        # Apply color for console output only
        if color:
            message = self._colorize(message, color)
        
        print(message)
    
    def show_banner(self):
        """Display PyImageTrack banner."""
        if self._banner_shown:
            return
        
        banner = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    PyImageTrack - Image Tracking Pipeline                    ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
        if not self.quiet:
            self.print(banner.strip(), color='cyan')
        else:
            # Still log to file in quiet mode
            self._log(banner.strip(), 'info')
        self._banner_shown = True
    
    def show_batch_banner(self, mode: str = 'start'):
        """
        Display batch processing banner.
        
        Parameters
        ----------
        mode : str
            Either 'start' or 'next' to determine the banner text.
        """
        if mode == 'start':
            banner_text = "PyImageTrack - Image Tracking Pipeline - Batch Mode: Start"
        else:
            banner_text = "PyImageTrack - Image Tracking Pipeline - Batch Mode: Next Process"
        
        # Calculate padding to center the text
        banner_width = 80
        text_length = len(banner_text)
        padding = (banner_width - text_length - 2) // 2  # -2 for the spaces on each side
        
        banner = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║{' ' * padding}{banner_text}{' ' * (banner_width - padding - text_length - 2)}║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
        # Log to file
        self._log(banner.strip(), 'info')
        
        # Print to console (not in quiet mode)
        if not self.quiet:
            # Use magenta color for batch mode banners to distinguish from normal banner
            self.print(banner.strip(), color='magenta')
    
    def section_header(self,
                      step_name: str,
                      config_ref: str,
                      description: str = None,
                      level: int = 1):
        """
        Print a section header with config reference.
        
        Parameters
        ----------
        step_name : str
            Name of the processing step (e.g., "ALIGNMENT")
        config_ref : str
            Config section reference (e.g., "[flags.do_alignment]")
        description : str
            Optional description of the step
        level : int
            Header level (1 = main, 2 = sub-section)
        """
        if self.quiet:
            # Still log to file, but don't print to console
            self._log(f"{step_name} {config_ref} {description or ''}", 'info')
            return
        
        if level == 1:
            line = '═' * 80
            self.print()
            self.print(line, color='cyan')
            self.print(f"{step_name} {config_ref} {description or ''}", color='bold')
            self.print(line, color='cyan')
        else:
            line = '─' * 80
            self.print()
            self.print(line, color='cyan')
            self.print(f"{step_name} {config_ref} {description or ''}", color='bold')
            self.print(line, color='cyan')
    
    def status(self,
              status_type: str,
              message: str,
              indent: int = 2):
        """
        Print a status message with icon.
        
        Parameters
        ----------
        status_type : str
            Type of status: 'success', 'warning', 'error', 'info', 'processing'
        message : str
            Status message
        indent : int
            Number of spaces to indent
        """
        # In quiet mode, only show success, warning, error, and processing messages
        if self.quiet and status_type not in ('success', 'warning', 'error', 'processing'):
            # Still log to file, but don't print to console
            icon = self.ICONS.get(status_type, '[*]')
            prefix = ' ' * indent
            level_map = {
                'success': 'info',
                'warning': 'warning',
                'error': 'error',
                'info': 'info',
                'processing': 'info'
            }
            level = level_map.get(status_type, 'info')
            self._log(f"{prefix}{icon} {message}", level)
            return
        
        icon = self.ICONS.get(status_type, '[*]')
        prefix = ' ' * indent
        
        # Map status_type to log level
        level_map = {
            'success': 'info',
            'warning': 'warning',
            'error': 'error',
            'info': 'info',
            'processing': 'info'
        }
        level = level_map.get(status_type, 'info')
        
        if status_type == 'success':
            self.print(f"{prefix}{icon} {message}", color='green', level=level)
        elif status_type == 'warning':
            self.print(f"{prefix}{icon} {message}", color='yellow', level=level)
        elif status_type == 'error':
            self.print(f"{prefix}{icon} {message}", color='red', level=level)
        elif status_type == 'info':
            self.print(f"{prefix}{icon} {message}", color='white', level=level)
        else:
            self.print(f"{prefix}{icon} {message}", level=level)
    
    def info(self, message: str, indent: int = 0):
        """Print an info message."""
        self.status('info', message, indent)
    
    def success(self, message: str, indent: int = 0):
        """Print a success message."""
        self.status('success', message, indent)
    
    def warning(self, message: str, indent: int = 0):
        """Print a warning message."""
        self.status('warning', message, indent)
    
    def error(self, message: str, indent: int = 0):
        """Print an error message."""
        self.status('error', message, indent)
    
    def processing(self, message: str, indent: int = 0):
        """Print a processing message."""
        self.status('processing', message, indent)
    
    def parameter_summary(self, params: Dict[str, Any], indent: int = 2):
        """
        Print a parameter summary (verbose only).
        
        Parameters
        ----------
        params : dict
            Dictionary of parameter names and values
        indent : int
            Number of spaces to indent
        """
        if self.quiet or not self.verbose:
            return
        
        self.print("[i] Parameters:", color='white')
        prefix = ' ' * indent
        for key, value in params.items():
            self.print(f"{prefix}  • {key}: {value}", color='dim')
    
    def config_info(self, key: str, value: Any, indent: int = 0):
        """Print a configuration value."""
        if self.quiet:
            return
        prefix = ' ' * indent
        self.print(f"{prefix}{key}: {value}", color='white')
    
    def config_loaded(self, key: str, value: Any, indent: int = 0):
        """Print a configuration loaded message with success status."""
        if self.quiet:
            return
        prefix = ' ' * indent
        self.print(f"{prefix}[OK] {key}: {value}", color='green')
    
    def cache_info(self, action: str, path: str, pair_id: str = None, cache_type: str = None):
        """
        Print cache information.
        
        Parameters
        ----------
        action : str
            Action performed (e.g., "saved", "loaded")
        path : str
            Path to cache
        pair_id : str
            Optional pair identifier
        cache_type : str
            Type of cache (e.g., "alignment", "tracking", "LoD")
        """
        # Show success message in all modes (normal, verbose, quiet)
        cache_name = f"{cache_type} " if cache_type else ""
        if action == "saved":
            self.success(f"Saved {cache_name}cache.")
        elif action == "loaded":
            self.success(f"Loaded {cache_name}cache.")
        else:
            self.success(f"Cache {cache_name}{action}.")
        
        # Only show path and pair details in verbose mode
        if not self.verbose:
            return
        
        prefix = '  '
        self.print(f"{prefix}Path: {path}", color='dim')
        if pair_id:
            self.print(f"{prefix}Pair: {pair_id}", color='dim')
    
    def file_list(self, label: str, files: list, indent: int = 2):
        """
        Print a list of files as bullet points (verbose only).
        
        Parameters
        ----------
        label : str
            Label for the file list
        files : list
            List of file names
        indent : int
            Number of spaces to indent
        """
        if self.quiet or not self.verbose:
            return
        
        prefix = ' ' * indent
        if label:
            self.print(f"{prefix}[i] {label}:", color='white')
        for f in sorted(files, key=str.lower):
            self.print(f"{prefix}  • {f}", color='dim')
    
    def info_verbose(self, message: str, indent: int = 0):
        """Print an info message only in verbose mode."""
        if self.quiet or not self.verbose:
            return
        self.status('info', message, indent)
    
    @contextmanager
    def timer(self, step_name: str, verbose: bool = None):
        """
        Context manager for timing a code block.
        
        Parameters
        ----------
        step_name : str
            Name of the step being timed
        verbose : bool
            Override verbose setting. If None, uses instance verbose.
        """
        if verbose is None:
            verbose = self.verbose
        
        start_time = time.time()
        self._timing_stack.append({'name': step_name, 'start': start_time})
        
        try:
            yield
        finally:
            elapsed = time.time() - start_time
            self._timing_stack.pop()
            
            # Don't print timer completion message - it will be shown in Summary
    
    def get_elapsed(self, step_name: str = None) -> float:
        """
        Get elapsed time for a step.
        
        Parameters
        ----------
        step_name : str
            Name of step to get time for. If None, returns total time.
        
        Returns
        -------
        float
            Elapsed time in seconds
        """
        if step_name:
            for item in reversed(self._timing_stack):
                if item['name'] == step_name:
                    return time.time() - item['start']
        elif self._timing_stack:
            return time.time() - self._timing_stack[0]['start']
        return 0.0
    
    def print_summary(self,
                     successes: List,
                     skipped: List,
                     total_elapsed: float = None):
        """
        Print processing summary.
        
        Parameters
        ----------
        successes : list
            List of successfully processed pairs. Each item is a tuple:
            (date_token_1, date_token_2, identifier) where identifier can be None.
        skipped : list
            List of skipped pairs with reasons. Each item is a tuple:
            (year1, year2, reason) where year1/year2 are full IDs.
        total_elapsed : float, optional
            Total elapsed time in seconds. If provided, will be shown in summary.
        """
        if self.quiet:
            return
        
        line = '═' * 80
        self.print()
        self.print(line, color='cyan')
        self.print("SUMMARY", color='bold')
        self.print(line, color='cyan')
        
        self.success(f"Successfully processed: {len(successes)} pair{'s' if len(successes) != 1 else ''}")
        for s in successes:
            # s is (date_token_1, date_token_2, identifier)
            if s[2] is not None:
                self.print(f"  • {s[0]} -> {s[1]}; id: {s[2]}")
            else:
                self.print(f"  • {s[0]} -> {s[1]}")
        
        if skipped:
            self.info(f"Skipped: {len(skipped)} pair{'s' if len(skipped) != 1 else ''}")
            for s in skipped:
                # Extract date tokens from full IDs for skipped items
                date_token_1 = s[0].split('_')[0] if '_' in s[0] else s[0]
                date_token_2 = s[1].split('_')[0] if '_' in s[1] else s[1]
                identifier = s[0].split('_')[1] if '_' in s[0] else None
                if identifier is not None:
                    self.print(f"  • {date_token_1} -> {date_token_2}; id: {identifier} | Reason: {s[2]}")
                else:
                    self.print(f"  • {date_token_1} -> {date_token_2} | Reason: {s[2]}")
        
        # Show total processing time in verbose mode
        if total_elapsed is not None and self.verbose:
            self.success(f"Total processing completed in {total_elapsed:.2f}s")
    
    @staticmethod
    def format_duration(delta_hours: float) -> str:
        """
        Format a duration in hours to a human-readable string.
        
        Shows the first reached unit and its next smaller unit (if non-zero).
        The parenthetical value shows the total duration in that unit.
        Examples: "90 sec (1 min)", "80 days (1920 h)", "7 days (168 h)"
        
        Parameters
        ----------
        delta_hours : float
            Duration in hours
            
        Returns
        -------
        str
            Formatted duration string
        """
        # Convert to seconds for easier calculations
        total_seconds = delta_hours * 3600
        
        # Define unit thresholds in seconds
        MINUTE = 60
        HOUR = 60 * MINUTE
        DAY = 24 * HOUR
        YEAR = 365.25 * DAY
        
        # Determine the primary unit and format accordingly
        if total_seconds < MINUTE:
            # Less than 1 minute: show seconds only
            return f"{total_seconds:.0f} sec"
        elif total_seconds < HOUR:
            # Less than 1 hour: show minutes (and total seconds)
            minutes = int(total_seconds // MINUTE)
            # Show total seconds
            seconds = round(total_seconds)
            return f"{minutes} min ({seconds} sec)"
        elif total_seconds < DAY:
            # Less than 1 day: show hours (and total minutes)
            hours = int(total_seconds // HOUR)
            # Show total minutes, not remainder minutes
            minutes = round(total_seconds / MINUTE)
            return f"{hours} h ({minutes} min)"
        elif total_seconds < YEAR:
            # Less than 1 year: show days (and total hours)
            days = int(total_seconds // DAY)
            # Show total hours, not remainder hours
            hours = round(total_seconds / HOUR)
            return f"{days} days ({hours} h)"
        else:
            # 1 year or more: show years (and total days)
            years = total_seconds / YEAR
            # Show total days, not remainder days
            days = round(total_seconds / DAY)
            return f"{years:.1f} years ({days} days)"


# Global instance for backward compatibility
_global_console: Optional[ConsoleOutput] = None


def get_console(verbose: bool = False,
                quiet: bool = False,
                use_colors: bool = True,
                log_file: Optional[Path] = None,
                log_level: str = 'INFO',
                log_max_bytes: int = 10 * 1024 * 1024,
                log_backup_count: int = 5) -> ConsoleOutput:
    """
    Get or create the global ConsoleOutput instance.
    
    Parameters
    ----------
    verbose : bool
        Enable verbose output
    quiet : bool
        Enable quiet mode
    use_colors : bool
        Use ANSI colors
    log_file : Path or None
        Path to log file
    log_level : str
        Logging level for file output (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    log_max_bytes : int
        Maximum size of log file before rotation (default: 10MB)
    log_backup_count : int
        Number of backup log files to keep (default: 5)
    
    Returns
    -------
    ConsoleOutput
        The global ConsoleOutput instance
    """
    global _global_console
    
    if _global_console is None:
        _global_console = ConsoleOutput(
            verbose=verbose,
            quiet=quiet,
            use_colors=use_colors,
            log_file=log_file,
            log_level=log_level,
            log_max_bytes=log_max_bytes,
            log_backup_count=log_backup_count
        )
    
    return _global_console


def reset_console():
    """Reset the global ConsoleOutput instance."""
    global _global_console
    _global_console = None