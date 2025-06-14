

class TrackingParameters:
    def __init__(self, parameter_dict):
        self.image_alignment_number_of_control_points = parameter_dict.get('image_alignment_number_of_control_points')
        self.image_alignment_control_tracking_area_size = parameter_dict.get(
            'image_alignment_control_tracking_area_size')
        self.image_alignment_control_cell_size = parameter_dict.get('image_alignment_control_cell_size')
        self.used_image_bands = parameter_dict.get('used_image_bands')
        self.distance_of_tracked_points = parameter_dict.get('distance_of_tracked_points')
        self.number_of_tracked_points = parameter_dict.get('number_of_tracked_points')
        self.movement_tracking_area_size = parameter_dict.get('movement_tracking_area_size')
        self.movement_cell_size = parameter_dict.get('movement_cell_size')
        self.cross_correlation_threshold = parameter_dict.get('cross_correlation_threshold')


    def __str__(self):
        return (f'TrackingParameters:\n'
                f'\timage alignment number of control_points: {self.image_alignment_number_of_control_points}\n'
                f'\timage alignment control tracking area size: {self.image_alignment_control_tracking_area_size}\n'
                f'\timage alignment control cell size: {self.image_alignment_control_cell_size}\n'
                f'\tused image bands: {self.used_image_bands}\n'
                f'\tdistance of tracked points: {self.distance_of_tracked_points}\n'
                f'\tnumber of tracked points: {self.number_of_tracked_points}\n'
                f'\tmovement tracking area size: {self.movement_tracking_area_size}\n'
                f'\tmovement cell size: {self.movement_cell_size}\n'
                f'\tcross correlation threshold: {self.cross_correlation_threshold}\n')
