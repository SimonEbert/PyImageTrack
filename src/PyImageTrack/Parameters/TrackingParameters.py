# PyImageTrack/Parameters/TrackingParameters.py
class TrackingParameters:
    """Container for tracking-related parameters (no alignment fields)."""

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

    def __init__(self, parameter_dict: dict):
        self.image_bands = parameter_dict.get("image_bands")
        self.distance_of_tracked_points_px = parameter_dict.get("distance_of_tracked_points_px")
        self.search_extent_px = parameter_dict.get("search_extent_px")  # e.g., (60, 20, 40, 10)
        self.search_extent_deltas = parameter_dict.get("search_extent_deltas", self.search_extent_px)
        self.search_extent_full_cell = parameter_dict.get("search_extent_full_cell")
        self.movement_cell_size = parameter_dict.get("movement_cell_size")
        self.cross_correlation_threshold_movement = parameter_dict.get("cross_correlation_threshold_movement")
        self.initial_shift_values = parameter_dict.get("initial_shift_values")
        self.initial_estimate_mode = parameter_dict.get("initial_estimate_mode", "count")
        self.nb_initial_estimate_peaks = parameter_dict.get("nb_initial_estimate_peaks", 1)
        self.correlation_threshold_initial_estimates = parameter_dict.get("correlation_threshold_initial_estimates", None)
        self.min_distance_initial_estimates = parameter_dict.get("min_distance_initial_estimates", 1)

    def __str__(self):
        return (
            "TrackingParameters:\n"
            f"\timage bands: {self.image_bands}\n"
            f"\tmovement cell size: {self.movement_cell_size}\n"
            f"\tCC threshold (movement): {self.cross_correlation_threshold_movement}\n"
            f"\tsearch (user deltas, posx,negx,posy,negy): {self.search_extent_deltas}\n"
            f"\tsearch extent px (posx,negx,posy,negy): {self.search_extent_px}\n"
            f"\tsearch extent full cell: {self.search_extent_full_cell}\n"
            f"\tdistance of tracked points (pixel): {self.distance_of_tracked_points_px}\n"
            f"\tinitial shift values: {self.initial_shift_values}\n"
            f"\tinitial estimate mode: {self.initial_estimate_mode}\n"
            f"\tnumber of initial estimates: {self.nb_initial_estimate_peaks}\n"
            f"\tcorrelation threshold initial estimates: {self.correlation_threshold_initial_estimates}\n"
            f"\tminimum distance initial estimates: {self.min_distance_initial_estimates}\n"
        )

    def to_dict(self) -> dict:
        return {
            "used_image_bands": self.image_bands,
            "movement_cell_size": self.movement_cell_size,
            "cross_correlation_threshold_movement": self.cross_correlation_threshold_movement,
            "distance_of_tracked_points_px": self.distance_of_tracked_points_px,
            "search_extent_px": self.search_extent_px,
            "search_extent_deltas": self.search_extent_deltas,
            "search_extent_full_cell": self.search_extent_full_cell,
            "initial_shift_values": self.initial_shift_values,
            "initial_estimate_mode": self.initial_estimate_mode,
            "nb_initial_estimate_peaks": self.nb_initial_estimate_peaks,
            "correlation_threshold_initial_estimates": self.correlation_threshold_initial_estimates,
            "min_distance_initial_estimates": self.min_distance_initial_estimates
        }
