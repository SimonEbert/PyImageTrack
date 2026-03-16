"""PyImageTrack package."""

# Lazy import to prevent RuntimeWarning when running as module after package import
# See: https://docs.python.org/3/library/runpy.html#notes
def __getattr__(name: str):
    if name == "run_from_config":
        from .run_pipeline import run_from_config
        return run_from_config
    if name == "run_batch":
        from .batch_processor import run_batch
        return run_batch
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
