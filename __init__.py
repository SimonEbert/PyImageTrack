"""
Wrapper package for the project.
The actual Python package lives in the subdirectory
`PyImageTrack_scripts`.  By adding that subdirectory to ``__path__``,
an import such as

    from PyImageTrack.ImageTracking.ImagePair import ImagePair

works correctly.  This turns ``PyImageTrack`` into a *namespace package*.
"""
from pathlib import Path

# Directory in which this file resides
_this_dir = Path(__file__).resolve().parent

# Subdirectory that contains the real project package
_inner_pkg = _this_dir / "PyImageTrack_scripts"

# If it exists, add it to the package's search‑path attribute
if _inner_pkg.is_dir():
    __path__.append(str(_inner_pkg))