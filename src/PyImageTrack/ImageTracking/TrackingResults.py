import numpy as np


class TrackingResults:
    def __init__(self, movement_rows, movement_cols, tracking_method, transformation_matrix: np.ndarray = None,
                 cross_correlation_coefficient: float = None, tracking_success: bool = False
                 ):
        self.movement_rows = movement_rows
        self.movement_cols = movement_cols
        self.tracking_method = tracking_method
        if tracking_method == "lsm":
            self.transformation_matrix = transformation_matrix
        else:
            self.transformation_matrix = None
        self.cross_correlation_coefficient = cross_correlation_coefficient
        self.tracking_success = tracking_success

    def __str__(self):
        return (f'Tracking Results:\n'
                f'\tmovement rows: {self.movement_rows}\n'
                f'\tmovement columns: {self.movement_cols}\n'
                f'\tCorrelation coefficient: {self.cross_correlation_coefficient}\n'
                f'\ttracking success: {self.tracking_success}')
