# PyImageTrack/Parameters/AlignmentParameters.py
class AlignmentParameters:
    """Container for alignment-related parameters."""

    def __init__(self, parameter_dict: dict):
        self.number_of_control_points = parameter_dict.get("number_of_control_points") or parameter_dict.get(
            "image_alignment_number_of_control_points")
        self.control_search_extent_px = parameter_dict.get("control_search_extent_px")
        self.control_search_extent_deltas = parameter_dict.get("control_search_extent_deltas",
                                                               self.control_search_extent_px)
        self.control_cell_size = parameter_dict.get("control_cell_size") or parameter_dict.get(
            "image_alignment_control_cell_size")
        self.cross_correlation_threshold_alignment = parameter_dict.get("cross_correlation_threshold_alignment")
        self.maximal_alignment_movement = parameter_dict.get("maximal_alignment_movement")

    def __str__(self):
        return (
            "AlignmentParameters:\n"
            f"\tcontrol points: {self.number_of_control_points}\n"
            f"\tcontrol search extent px (posx,negx,posy,negy): {self.control_search_extent_px}\n"
            f"\tcontrol search (user deltas, posx,negx,posy,negy): {self.control_search_extent_deltas}\n"
            f"\tcell size: {self.control_cell_size}\n"
            f"\tCC threshold (alignment): {self.cross_correlation_threshold_alignment}\n"
            f"\tmax movement (px): {self.maximal_alignment_movement}\n"
        )

    def to_dict(self) -> dict:
        """Return keys expected by ImagePair(parameter_dict=...)."""
        return {
            "image_alignment_number_of_control_points": self.number_of_control_points,
            "image_alignment_control_cell_size": self.control_cell_size,
            "cross_correlation_threshold_alignment": self.cross_correlation_threshold_alignment,
            "maximal_alignment_movement": self.maximal_alignment_movement,
        }
