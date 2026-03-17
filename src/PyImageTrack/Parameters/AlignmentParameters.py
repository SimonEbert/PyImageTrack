# PyImageTrack/Parameters/AlignmentParameters.py
class AlignmentParameters:
    """
    Container for alignment-related parameters.
    
    This class stores and validates all parameters used for image alignment,
    which is the process of geometrically transforming one image to match
    another using control points in a stable reference area.
    
    Attributes
    ----------
    number_of_control_points : int
        Number of control points to use for alignment. Points are distributed
        across the reference area using a grid pattern.
    control_search_extent_px : tuple of int
        Search window extents in pixels for alignment tracking, specified as
        (pos_x, neg_x, pos_y, neg_y) where each value is the number of pixels
        to search in that direction from the center.
    control_search_extent_deltas : tuple of int
        User-specified search extents in CRS units (e.g., meters). Falls back
        to control_search_extent_px if not specified.
    control_cell_size : int
        Size of the tracking cell (template window) in pixels for alignment.
    cross_correlation_threshold_alignment : float
        Minimum cross-correlation coefficient (0-1) for accepting a control point
        as valid for alignment.
    maximal_alignment_movement : float or None
        Maximum allowed movement in pixels for control points. Points with
        movement exceeding this value are excluded from alignment.
    """
    
    def __init__(self, parameter_dict: dict):
        """
        Initialize AlignmentParameters from a dictionary.
        
        Parameters
        ----------
        parameter_dict : dict
            Dictionary containing alignment parameters. Supports both legacy
            keys (e.g., "image_alignment_number_of_control_points") and
            current keys (e.g., "number_of_control_points").
        """
        self.number_of_control_points = parameter_dict.get("number_of_control_points") or parameter_dict.get(
            "image_alignment_number_of_control_points")
        self.control_search_extent_px = parameter_dict.get("control_search_extent_px")
        self.control_search_extent_deltas = parameter_dict.get("control_search_extent_deltas",
                                                               self.control_search_extent_px)
        self.control_cell_size = parameter_dict.get("control_cell_size") or parameter_dict.get(
            "image_alignment_control_cell_size")
        self.cross_correlation_threshold_alignment = parameter_dict.get("cross_correlation_threshold_alignment")
        self.maximal_alignment_movement = parameter_dict.get("maximal_alignment_movement")
        self.control_search_extent_full_cell = None  # Will be computed during alignment
        self.image_bands = parameter_dict.get("image_bands")
        
        # Validate parameters
        self._validate()
    
    def _validate(self):
        """
        Validate alignment parameters.
        
        Raises
        ------
        ValueError
            If any parameter has an invalid value.
        """
        if self.number_of_control_points is not None and self.number_of_control_points <= 0:
            raise ValueError(f"number_of_control_points must be positive, got {self.number_of_control_points}")
        
        if self.control_cell_size is not None and self.control_cell_size <= 0:
            raise ValueError(f"control_cell_size must be positive, got {self.control_cell_size}")
        
        if self.cross_correlation_threshold_alignment is not None:
            if not 0 <= self.cross_correlation_threshold_alignment <= 1:
                raise ValueError(f"cross_correlation_threshold_alignment must be between 0 and 1, got {self.cross_correlation_threshold_alignment}")
        
        if self.maximal_alignment_movement is not None and self.maximal_alignment_movement < 0:
            raise ValueError(f"maximal_alignment_movement must be non-negative, got {self.maximal_alignment_movement}")
        
        if self.control_search_extent_px is not None:
            if not isinstance(self.control_search_extent_px, (tuple, list)) or len(self.control_search_extent_px) != 4:
                raise ValueError(f"control_search_extent_px must be a tuple of 4 values (posx,negx,posy,negy), got {self.control_search_extent_px}")
            if any(v < 0 for v in self.control_search_extent_px):
                raise ValueError(f"control_search_extent_px values must be non-negative, got {self.control_search_extent_px}")

    def __str__(self) -> str:
        """
        Return string representation of alignment parameters.
        
        Returns
        -------
        str
            Formatted string showing all alignment parameters.
        """
        return (
            "AlignmentParameters:\n"
            f"\tcontrol points: {self.number_of_control_points}\n"
            f"\tcontrol search extent px (posx,negx,posy,negy): {self.control_search_extent_px}\n"
            f"\tcontrol search (user deltas, posx,negx,posy,negy): {self.control_search_extent_deltas}\n"
            f"\tcell size: {self.control_cell_size}\n"
            f"\tCC threshold (alignment): {self.cross_correlation_threshold_alignment}\n"
            f"\tmax movement (px): {self.maximal_alignment_movement}\n"
            f"\timage bands: {self.image_bands}\n"
        )

    def to_dict(self) -> dict:
        """
        Convert parameters to dictionary format.
        
        Returns
        -------
        dict
            Dictionary with keys expected by ImagePair(parameter_dict=...).
        """
        return {
            "image_alignment_number_of_control_points": self.number_of_control_points,
            "image_alignment_control_cell_size": self.control_cell_size,
            "cross_correlation_threshold_alignment": self.cross_correlation_threshold_alignment,
            "maximal_alignment_movement": self.maximal_alignment_movement,
            "control_search_extent_px": self.control_search_extent_px,
            "control_search_extent_full_cell": self.control_search_extent_full_cell,
            "image_bands": self.image_bands
        }
