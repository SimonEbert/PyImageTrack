import numpy as np
from scipy.interpolate import RectBivariateSpline

class ImageInterpolator:
    def __init__(self, image_matrix):
        self.shape = image_matrix.shape
        if len(self.shape) == 2:
            self.splines = RectBivariateSpline(np.arange(image_matrix.shape[-2]),
                                               np.arange(image_matrix.shape[-1]),
                                               image_matrix)
        else:
            self.splines = [
                RectBivariateSpline(np.arange(image_matrix.shape[-2]),
                                    np.arange(image_matrix.shape[-1]),
                                    image_matrix[c])
                for c in range(image_matrix.shape[0])
            ]
    def ev(self, yq, xq, dx=0, dy=0, shape=None):
        if len(self.shape) == 2:
            return self.splines.ev(yq, xq, dx, dy)
        else:
            out = np.stack(
                [s.ev(yq, xq, dx=dx, dy=dy) for s in self.splines],
                axis=0
            )
            return out.reshape(shape) if shape is not None else out