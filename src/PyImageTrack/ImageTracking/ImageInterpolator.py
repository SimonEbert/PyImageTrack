import numpy as np
from scipy.interpolate import RectBivariateSpline

class ImageInterpolator:
    def __init__(self, image_matrix):
        # Remove all singleton dimensions to handle cases like (1, 1, H, W) -> (H, W)
        image_matrix = np.squeeze(image_matrix)
        
        # Ensure the result is 2D or 3D
        if image_matrix.ndim not in (2, 3):
            raise ValueError(f"Expected 2D or 3D array after squeezing, got shape {image_matrix.shape}")
        
        self.shape = image_matrix.shape
        
        if len(self.shape) == 2:
            # Single-band image: (height, width)
            self.splines = RectBivariateSpline(np.arange(image_matrix.shape[-2]),
                                               np.arange(image_matrix.shape[-1]),
                                               image_matrix)
        else:
            # Multi-band image: (bands, height, width)
            self.splines = [
                RectBivariateSpline(np.arange(image_matrix.shape[-2]),
                                    np.arange(image_matrix.shape[-1]),
                                    image_matrix[c])
                for c in range(image_matrix.shape[0])
            ]
    def ev(self, yq, xq, dx=0, dy=0, shape=None):
        if len(self.shape) == 2:
            # Single-band: output should match yq shape
            result = self.splines.ev(yq, xq, dx, dy)
        else:
            # Multi-band: stack results and handle reshaping
            out = np.stack(
                [s.ev(yq, xq, dx=dx, dy=dy) for s in self.splines],
                axis=0
            )
            result = out
        
        # Explicitly reshape if target shape provided
        if shape is not None:
            result = result.reshape(shape)
        return result