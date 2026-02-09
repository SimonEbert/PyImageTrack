class FilterParameters:
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
            "maximal_fraction_depth_change_of_3d_displacement"
        )

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
