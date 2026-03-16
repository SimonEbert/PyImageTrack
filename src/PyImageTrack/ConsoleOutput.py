#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Console Output Utility Module

Provides consistent, formatted console output for PyImageTrack with:
- Section headers with step names
- Status indicators ([OK], [!], [X], [i], [~])
- Timing information
- Parameter summaries
- Log file support
- Verbose/quiet modes
"""

import logging
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
                 log_file: Optional[Path] = None):
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
        """
        self.verbose = verbose
        self.quiet = quiet
        self.use_colors = use_colors and sys.stdout.isatty()
        self.log_file = log_file
        
        # Setup logging for log file
        self._setup_logging()
        
        # Timing context stack
        self._timing_stack: List[Dict[str, Any]] = []
        
        # Track if banner has been shown
        self._banner_shown = False
    
    def _setup_logging(self):
        """Setup logging for log file output."""
        self.logger = logging.getLogger('PyImageTrack')
        self.logger.setLevel(logging.DEBUG)
        
        # Remove existing handlers
        self.logger.handlers.clear()
        
        if self.log_file:
            # Ensure log directory exists
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            
            # File handler
            file_handler = logging.FileHandler(self.log_file, mode='w', encoding='utf-8')
            file_handler.setLevel(logging.DEBUG)
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
        """Write message to log file if configured."""
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
        if self.quiet and level not in ('error', 'warning'):
            return
        
        if color:
            message = self._colorize(message, color)
        
        print(message)
        self._log(message, level)
    
    def show_banner(self):
        """Display PyImageTrack banner."""
        if self._banner_shown:
            return
        
        banner = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    PyImageTrack - Image Tracking Pipeline                    ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
        self.print(banner.strip(), color='cyan')
        self._banner_shown = True
    
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
            return
        
        icon = self.ICONS.get(status_type, '[*]')
        prefix = ' ' * indent
        
        if status_type == 'success':
            self.print(f"{prefix}{icon} {message}", color='green')
        elif status_type == 'warning':
            self.print(f"{prefix}{icon} {message}", color='yellow')
        elif status_type == 'error':
            self.print(f"{prefix}{icon} {message}", color='red')
        elif status_type == 'info':
            self.print(f"{prefix}{icon} {message}", color='white')
        else:
            self.print(f"{prefix}{icon} {message}")
    
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
            Type of cache (e.g., "alignment", "tracking", "lod") for specific messages
        """
        # Show success message in all modes (normal, verbose, quiet)
        if action == "saved":
            if cache_type:
                self.success(f"Saved {cache_type} cache.")
            else:
                self.success("Saved cache.")
        elif action == "loaded":
            if cache_type:
                self.success(f"Loaded {cache_type} cache.")
            else:
                self.success("Loaded cache.")
        else:
            if cache_type:
                self.success(f"Cache {cache_type} {action}.")
            else:
                self.success(f"Cache {action}.")
        
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
        for f in sorted(files):
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
            List of successfully processed pairs
        skipped : list
            List of skipped pairs with reasons
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
            self.print(f"  • {s[0]} → {s[1]}")
        
        if skipped:
            self.info(f"Skipped: {len(skipped)} pair{'s' if len(skipped) != 1 else ''}")
            for s in skipped:
                self.print(f"  • {s[0]} → {s[1]} | Reason: {s[2]}")
        
        # Show total processing time in verbose mode
        if total_elapsed is not None and self.verbose:
            self.success(f"Total processing completed in {total_elapsed:.2f}s")


# Global instance for backward compatibility
_global_console: Optional[ConsoleOutput] = None


def get_console(verbose: bool = False, 
                quiet: bool = False,
                use_colors: bool = True,
                log_file: Optional[Path] = None) -> ConsoleOutput:
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
            log_file=log_file
        )
    
    return _global_console


def reset_console():
    """Reset the global ConsoleOutput instance."""
    global _global_console
    _global_console = None