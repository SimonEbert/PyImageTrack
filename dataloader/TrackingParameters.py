

class TrackingParameters:
    def __init__(self, parameter_dict):
        self.image_alignment_via_lsm = parameter_dict.get('image_alignment_via_lsm')
        self.image_alignment_number_of_control_points = parameter_dict.get('image_alignment_number_of_control_points')
        self.image_alignment_control_tracking_area_size = parameter_dict.get(
            'image_alignment_control_tracking_area_size')
        self.image_alignment_control_cell_size = parameter_dict.get('image_alignment_control_cell_size')
        self.used_image_bands = parameter_dict.get('used_image_bands')
        self.tracking_method = parameter_dict.get('tracking_method')
        self.distance_of_tracked_points = parameter_dict.get('distance_of_tracked_points')
        self.number_of_tracked_points = parameter_dict.get('number_of_tracked_points')
        self.movement_tracking_area_size = parameter_dict.get('movement_tracking_area_size')
        self.movement_cell_size = parameter_dict.get('movement_cell_size')
        self.remove_outliers = parameter_dict.get('remove_outliers')
        self.retry_matching = parameter_dict.get('retry_matching')
        self.transformation_determinant_threshold = parameter_dict.get('transformation_determinant_threshold')
        self.cross_correlation_threshold = parameter_dict.get('cross_correlation_threshold')
        self.level_of_detection_quantile = parameter_dict.get('level_of_detection_quantile')
        self.use_4th_channel_as_data_mask = parameter_dict.get('use_4th_channel_as_data_mask')


    def __str__(self):
        return (f'TrackingParameters:\n'
                f'\timage alignment via lsm: {self.image_alignment_via_lsm}\n'
                f'\timage alignment number of control_points: {self.image_alignment_number_of_control_points}\n'
                f'\timage alignment control tracking area size: {self.image_alignment_control_tracking_area_size}\n'
                f'\timage alignment control cell size: {self.image_alignment_control_cell_size}\n'
                f'\tused image bands: {self.used_image_bands}\n'
                f'\ttracking method: {self.tracking_method}\n'
                f'\tdistance of tracked points: {self.distance_of_tracked_points}\n'
                f'\tnumber of tracked points: {self.number_of_tracked_points}\n'
                f'\tmovement tracking area size: {self.movement_tracking_area_size}\n'
                f'\tmovement cell size: {self.movement_cell_size}\n'
                f'\tremove outliers: {self.remove_outliers}\n'
                f'\tretry matching: {self.retry_matching}\n'
                f'\ttransformation determinant threshold: {self.transformation_determinant_threshold}\n'
                f'\tlevel of detection quantile: {self.level_of_detection_quantile}\n'
                f'\tuse 4th channel as data_mask: {self.use_4th_channel_as_data_mask}\n')
