class FilterParameters:
    """
    Container for filtering-related parameters.

    This class stores parameters used for filtering tracking results,
    including level of detection calculation and various outlier
    filtering methods.

    Parameters
    ----------
    parameter_dict : dict
        Dictionary containing filter parameters. Expected keys include:
        - level_of_detection_quantile: Quantile for LoD calculation (0-1)
        - number_of_points_for_level_of_detection: Number of points for LoD
        - difference_movement_bearing_threshold: Threshold for bearing difference outliers (degrees)
        - difference_movement_bearing_moving_window_size: Window size for bearing difference
        - standard_deviation_movement_bearing_threshold: Threshold for bearing std dev outliers (degrees)
        - standard_deviation_movement_bearing_moving_window_size: Window size for bearing std dev
        - difference_movement_rate_threshold: Threshold for rate difference outliers
        - difference_movement_rate_moving_window_size: Window size for rate difference
        - standard_deviation_movement_rate_threshold: Z-score threshold for rate std dev outliers (number of standard deviations from mean)
        - standard_deviation_movement_rate_moving_window_size: Window size for rate std dev

    Attributes
    ----------
    level_of_detection_quantile : float
        Quantile for level of detection calculation.
    number_of_points_for_level_of_detection : int
        Number of random points to use for LoD calculation.
    difference_movement_bearing_threshold : float
        Threshold for filtering bearing difference outliers (degrees).
    difference_movement_bearing_moving_window_size : float
        Distance window for bearing difference filtering.
    standard_deviation_movement_bearing_threshold : float
        Threshold for filtering bearing standard deviation outliers (degrees).
    standard_deviation_movement_bearing_moving_window_size : float
        Distance window for bearing standard deviation filtering.
    difference_movement_rate_threshold : float
        Threshold for filtering movement rate difference outliers.
    difference_movement_rate_moving_window_size : float
        Distance window for movement rate difference filtering.
    standard_deviation_movement_rate_threshold : float
        Modified Z-score threshold for filtering movement rate outliers using median and MAD. Points with a modified Z-score (absolute deviation from median divided by MAD) greater than this value are filtered. Common values are 2 or 3.
    standard_deviation_movement_rate_moving_window_size : float
        Distance window for movement rate standard deviation filtering.
    """

    def __init__(self, parameter_dict):
        # Level of Detection
        self.level_of_detection_quantile = parameter_dict.get("level_of_detection_quantile")
        self.number_of_points_for_level_of_detection = parameter_dict.get("number_of_points_for_level_of_detection")

        # Outlier filtering
        self.difference_movement_bearing_threshold = parameter_dict.get("difference_movement_bearing_threshold")
        self.difference_movement_bearing_moving_window_size = parameter_dict.get(
            "difference_movement_bearing_moving_window_size")

        self.standard_deviation_movement_bearing_threshold = parameter_dict.get(
            "standard_deviation_movement_bearing_threshold")
        self.standard_deviation_movement_bearing_moving_window_size = parameter_dict.get(
            "standard_deviation_movement_bearing_moving_window_size")

        self.difference_movement_rate_threshold = parameter_dict.get("difference_movement_rate_threshold")
        self.difference_movement_rate_moving_window_size = parameter_dict.get(
            "difference_movement_rate_moving_window_size")

        self.standard_deviation_movement_rate_threshold = parameter_dict.get(
            "standard_deviation_movement_rate_threshold")
        self.standard_deviation_movement_rate_moving_window_size = parameter_dict.get(
            "standard_deviation_movement_rate_moving_window_size")
        self.maximal_fraction_depth_change_of_3d_displacement = parameter_dict.get(
            "maximal_fraction_depth_change_of_3d_displacement")

        # Validate parameters
        self._validate()
    
    def _validate(self):
        """Validate filter parameters."""
        if self.level_of_detection_quantile is not None:
            if not 0 < self.level_of_detection_quantile < 1:
                raise ValueError(f"level_of_detection_quantile must be between 0 and 1 (exclusive), got {self.level_of_detection_quantile}")
        
        if self.number_of_points_for_level_of_detection is not None:
            if self.number_of_points_for_level_of_detection <= 0:
                raise ValueError(f"number_of_points_for_level_of_detection must be positive, got {self.number_of_points_for_level_of_detection}")
        if self.maximal_fraction_depth_change_of_3d_displacement is not None:
            if self.maximal_fraction_depth_change_of_3d_displacement <= 0:
                raise ValueError(f"maximal_fraction_depth_change_of_3d_displacement must be positive, got {self.maximal_fraction_depth_change_of_3d_displacement}")
        
        # Validate threshold values (must be non-negative)
        for name in ["difference_movement_bearing_threshold", "standard_deviation_movement_bearing_threshold",
                     "difference_movement_rate_threshold", "standard_deviation_movement_rate_threshold"]:
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} must be non-negative, got {value}")
        
        # Validate moving window sizes (must be positive)
        for name in ["difference_movement_bearing_moving_window_size", "standard_deviation_movement_bearing_moving_window_size",
                     "difference_movement_rate_moving_window_size", "standard_deviation_movement_rate_moving_window_size"]:
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")

    def __str__(self):
        return (f'FilterParameters:\n'
                f'\tlevel of detection quantile: {self.level_of_detection_quantile}\n'
                f'\tnumber of points for level of detection: {self.number_of_points_for_level_of_detection}\n'
                f'\tdifference movement bearing threshold: {self.difference_movement_bearing_threshold}\n'
                f'\tdifference movement bearing moving window size: {self.difference_movement_bearing_moving_window_size}\n'
                f'\tstandard deviation movement bearing threshold: {self.standard_deviation_movement_bearing_threshold}\n'
                f'\tstandard deviation movement bearing window size: {self.standard_deviation_movement_bearing_moving_window_size}\n'
                f'\tdifference movement rate threshold: {self.difference_movement_rate_threshold}\n'
                f'\tdifference movement rate moving window size: {self.difference_movement_rate_moving_window_size}\n'
                f'\tstandard deviation movement rate threshold: {self.standard_deviation_movement_rate_threshold}\n'
                f'\tstandard deviation movement rate moving window size: {self.standard_deviation_movement_rate_moving_window_size}\n'
                f'\tmaximal_fraction_depth_change_of_3d_displacement: {self.maximal_fraction_depth_change_of_3d_displacement}\n')
