# PyImageTrack/Parameters/TrackingParameters.py
from numbers import Number


class TrackingParameters:
    """
    Container for tracking-related parameters (no alignment fields).
    
    This class stores and validates all parameters used for point tracking
    between aligned images. Tracking is the process of finding the displacement
    of individual points between two observations.
    
    Attributes
    ----------
    image_bands : int or list of int
        Image band(s) to use for tracking. Can be a single band index (0-based)
        or a list of band indices. If None, all bands are used.
    distance_of_tracked_points_px : float
        Distance between tracked points in pixels. Points are arranged in a
        grid pattern across the tracking area.
    search_extent_px : tuple of int
        Search window extents in pixels for movement tracking, specified as
        (pos_x, neg_x, pos_y, neg_y) where each value is the number of pixels
        to search in that direction from the center.
    search_extent_deltas : tuple of int
        User-specified search extents in CRS units (e.g., meters). Falls back
        to search_extent_px if not specified.
    movement_cell_size : int
        Size of the tracking cell (template window) in pixels for movement tracking.
    cross_correlation_threshold_movement : float
        Minimum cross-correlation coefficient (0-1) for accepting a tracked point
        as valid.
    minimum_valid_tracking_point_ratio : float
        Minimum fraction of generated points that must pass the correlation
        threshold.
    """
    
    """
    tracked_cell_size_cc
    tracked_cell_size_lsm
    initial_estimate_mode: str
        "count" or "threshold".
        If "count" (the default) LSM is performed for the n (default 1) initial values with the highest cross-correlation
        values. The number of guesses is determined by "nb_initial_estimate_peaks". By default only a single estimate is
        used, corresponding to the normally used LSM implementations (EMT, ...)
        If "threshold", LSM is performed for all peaks, where the cross-correlation exceeds the given threshold. Note
        that depending on the data structure (e.g. low-contrast images) this can lead to an enormous additional amount of
        computation and should therefore only be considered in very distinct cases.
    nb_initial_estimate_peaks: int = 1
        The number of peaks that are considered if the initial_estimate_mode is "count".
    correlation_threshold_initial_estimates
        The correlation threshold that must be exceeded in order for the LSM to be performed if the initial_estimate_mode
        is "threshold"."""

    def __init__(self, parameter_dict: dict | None = None):
        """
        Initialize TrackingParameters from a dictionary.
        
        Parameters
        ----------
        parameter_dict : dict
            Dictionary containing tracking parameters.
        """
        if parameter_dict is not None:
            self.distance_of_tracked_points_px = parameter_dict.get("distance_of_tracked_points_px")
            self.search_extent_px = parameter_dict.get("search_extent_px")  # e.g., (60, 20, 40, 10)
            self.search_extent_deltas = parameter_dict.get("search_extent_deltas", self.search_extent_px)
            self.search_extent_full_cell = None  # Will be computed during tracking
            self.movement_cell_size = parameter_dict.get("movement_cell_size")
            self.minimum_valid_tracking_point_ratio = parameter_dict.get("minimum_valid_tracking_point_ratio", 0.0)
            self.initial_shift_values = parameter_dict.get("initial_shift_values")
            self.initial_estimate_mode = parameter_dict.get("initial_estimate_mode", "count")
            self.nb_initial_estimate_peaks = parameter_dict.get("nb_initial_estimate_peaks", 1)
            self.correlation_threshold_initial_estimates = parameter_dict.get("correlation_threshold_initial_estimates", None)
            self.min_distance_initial_estimates = parameter_dict.get("min_distance_initial_estimates", 1)
            self.multi_scale_initialization_downsample_factors = parameter_dict.get("multi_scale_initialization_downsample_factors", [1])
            self.use_adaptive_tracking_window = parameter_dict.get("use_adaptive_tracking_window", False)

            # Validate parameters
            self._validate()
    
    def _validate(self):
        """
        Validate tracking parameters.
        
        Raises
        ------
        ValueError
            If any parameter has an invalid value.
        """
        if self.movement_cell_size is not None and self.movement_cell_size <= 0:
            raise ValueError(f"movement_cell_size must be positive, got {self.movement_cell_size}")
        
        if self.distance_of_tracked_points_px is not None and self.distance_of_tracked_points_px <= 0:
            raise ValueError(f"distance_of_tracked_points_px must be positive, got {self.distance_of_tracked_points_px}")


        if not 0 <= self.minimum_valid_tracking_point_ratio <= 1:
            raise ValueError(
                "minimum_valid_tracking_point_ratio must be between 0 and 1, "
                f"got {self.minimum_valid_tracking_point_ratio}"
            )
        
        if self.search_extent_px is not None:
            if not isinstance(self.search_extent_px, (tuple, list)) or len(self.search_extent_px) != 4:
                raise ValueError(f"search_extent_px must be a tuple of 4 values (posx,negx,posy,negy), got {self.search_extent_px}")
            if any(v < 0 for v in self.search_extent_px):
                raise ValueError(f"search_extent_px values must be non-negative, got {self.search_extent_px}")
        if self.multi_scale_initialization_downsample_factors is not None:
            if isinstance(self.multi_scale_initialization_downsample_factors, Number):
                self.multi_scale_initialization_downsample_factors = [self.multi_scale_initialization_downsample_factors]

            if not all(isinstance(v, Number) for v in self.multi_scale_initialization_downsample_factors):
                raise ValueError(f"multi_scale_initialization_downsample_factors must be numeric, got {self.multi_scale_initialization_downsample_factors}")



        

    def __str__(self) -> str:
        """
        Return string representation of tracking parameters.
        
        Returns
        -------
        str
            Formatted string showing all tracking parameters.
        """
        return (
            "TrackingParameters:\n"
            f"\tmovement cell size: {self.movement_cell_size}\n"
            f"\tminimum valid tracking point ratio: {self.minimum_valid_tracking_point_ratio} "
            f"({self.minimum_valid_tracking_point_ratio:.1%})\n"
            f"\tsearch (user deltas, posx,negx,posy,negy): {self.search_extent_deltas}\n"
            f"\tsearch extent px (posx,negx,posy,negy): {self.search_extent_px}\n"
            f"\tsearch extent full cell: {self.search_extent_full_cell}\n"
            f"\tdistance of tracked points (pixel): {self.distance_of_tracked_points_px}\n"
            f"\tinitial shift values: {self.initial_shift_values}\n"
            f"\tinitial estimate mode: {self.initial_estimate_mode}\n"
            f"\tnumber of initial estimates: {self.nb_initial_estimate_peaks}\n"
            f"\tcorrelation threshold initial estimates: {self.correlation_threshold_initial_estimates}\n"
            f"\tminimum distance initial estimates: {self.min_distance_initial_estimates}\n"
            f"\tmulti scale initialization downsample factors: {self.multi_scale_initialization_downsample_factors}\n"
        )

    def to_dict(self) -> dict:
        """
        Convert parameters to dictionary format.
        
        Returns
        -------
        dict
            Dictionary with tracking parameters.
        """
        return {
            "movement_cell_size": self.movement_cell_size,
            "minimum_valid_tracking_point_ratio": self.minimum_valid_tracking_point_ratio,
            "distance_of_tracked_points_px": self.distance_of_tracked_points_px,
            "search_extent_px": self.search_extent_px,
            "search_extent_deltas": self.search_extent_deltas,
            "search_extent_full_cell": self.search_extent_full_cell,
            "initial_shift_values": self.initial_shift_values,
            "initial_estimate_mode": self.initial_estimate_mode,
            "nb_initial_estimate_peaks": self.nb_initial_estimate_peaks,
            "correlation_threshold_initial_estimates": self.correlation_threshold_initial_estimates,
            "min_distance_initial_estimates": self.min_distance_initial_estimates,
            "multi_scale_initialization_downsample_factors": self.multi_scale_initialization_downsample_factors

        }

    @classmethod
    def from_dict(cls, d: dict) -> "TrackingParameters":
        return cls(d)