# PyImageTrack/Parameters/TrackingParameters.py
class TrackingParameters:
    """Container for tracking-related parameters (no alignment fields)."""

    def __init__(self, parameter_dict: dict):
        self.image_bands = parameter_dict.get("image_bands")
        self.distance_of_tracked_points_px = parameter_dict.get("distance_of_tracked_points_px")
        self.search_extent_px = parameter_dict.get("search_extent_px")  # e.g., (60, 20, 40, 10)
        self.search_extent_deltas = parameter_dict.get("search_extent_deltas", self.search_extent_px)
        self.movement_cell_size = parameter_dict.get("movement_cell_size")
        self.cross_correlation_threshold_movement = parameter_dict.get("cross_correlation_threshold_movement")
        self.initial_shift_values = parameter_dict.get("initial_shift_values")

    def __str__(self):
        return (
            "TrackingParameters:\n"
            f"\timage bands: {self.image_bands}\n"
            f"\tmovement cell size: {self.movement_cell_size}\n"
            f"\tCC threshold (movement): {self.cross_correlation_threshold_movement}\n"
            f"\tsearch (user deltas, posx,negx,posy,negy): {self.search_extent_deltas}\n"
            f"\tsearch extent px (posx,negx,posy,negy): {self.search_extent_px}\n"
            f"\tDPx{self.distance_of_tracked_points_px}\n"
        )

    def to_dict(self) -> dict:
        return {
            "used_image_bands": self.image_bands,
            "movement_cell_size": self.movement_cell_size,
            "cross_correlation_threshold_movement": self.cross_correlation_threshold_movement,
            "distance_of_tracked_points_px": self.distance_of_tracked_points_px,
            "search_extent_px": self.search_extent_px,
            "search_extent_deltas": self.search_extent_deltas,
            "initial_shift_values": self.initial_shift_values,
        }
