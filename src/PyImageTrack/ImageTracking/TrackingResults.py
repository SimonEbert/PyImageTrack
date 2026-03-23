import numpy as np


class TrackingResults:
    """
    Container for tracking results from image movement detection.

    This class stores the results of tracking a single point or image section
    between two images, including the detected movement in row and column
    directions, the tracking method used, and quality metrics.

    Parameters
    ----------
    movement_rows : float
        Detected movement in the row (vertical) direction in pixels.
    movement_cols : float
        Detected movement in the column (horizontal) direction in pixels.
    tracking_method : str
        The tracking method used. Supported values: "cross-correlation", "lsm".
    transformation_matrix : np.ndarray, optional
        The affine transformation matrix for least-squares method. Only used
        when tracking_method is "lsm". Default is None.
    cross_correlation_coefficient : float, optional
        The cross-correlation coefficient indicating tracking quality.
        Higher values (closer to 1.0) indicate better matches. Default is None.
    tracking_success : bool, optional
        Whether the tracking was successful. Default is False.

    Attributes
    ----------
    movement_rows : float
        Detected movement in the row (vertical) direction in pixels.
    movement_cols : float
        Detected movement in the column (horizontal) direction in pixels.
    tracking_method : str
        The tracking method used ("cross-correlation" or "lsm").
    transformation_matrix : np.ndarray or None
        The affine transformation matrix (only for "lsm" method).
    cross_correlation_coefficient : float or None
        The cross-correlation coefficient indicating tracking quality.
    tracking_success : bool
        Whether the tracking was successful.
    """

    def __init__(self, movement_rows, movement_cols, tracking_method, transformation_matrix: np.ndarray = None,
                 cross_correlation_coefficient: float = None, tracking_success: bool = False,rmse = None
                 ):
        self.movement_rows = movement_rows
        self.movement_cols = movement_cols
        self.tracking_method = tracking_method
        if (tracking_method == "least-squares") | (tracking_method == "lsm"):
            self.transformation_matrix = transformation_matrix
            self.rmse = rmse
        else:
            self.transformation_matrix = None
            self.rmse = None
        self.cross_correlation_coefficient = cross_correlation_coefficient
        self.tracking_success = tracking_success

    def __str__(self):
        """
        Return a string representation of the tracking results.

        Returns
        -------
        str
            Formatted string showing movement values, correlation coefficient,
            and tracking success status.
        """
        return (f'Tracking Results:\n'
                f'\tmovement rows: {self.movement_rows}\n'
                f'\tmovement columns: {self.movement_cols}\n'
                f'\tCorrelation coefficient: {self.cross_correlation_coefficient}\n'
                f'\ttracking success: {self.tracking_success}\n'
                f'\ttransformation matrix: {self.transformation_matrix}\n'
                f'\tRMSE: {self.rmse}\n')
