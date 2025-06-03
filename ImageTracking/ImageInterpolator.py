import numpy as np
import scipy

class ImageInterpolator:

    def __init__(self, image_matrix: np.ndarray):
        self.image_matrix = image_matrix

        self.spline_list = []
        # check if the first dimension signifies channels (if we have a 3-dimensional array)
        if len(image_matrix.shape) == 3:
            self.image_channels = image_matrix.shape[0]
            for i in range(image_matrix.shape[0]):
                self.spline_list.append(scipy.interpolate.RectBivariateSpline(np.arange(0, image_matrix.shape[-2]),
                                                                              np.arange(0, image_matrix.shape[-1]),
                                                                              image_matrix[i, :, :]))
        else:
            self.spline_list.append(scipy.interpolate.RectBivariateSpline(np.arange(0, image_matrix.shape[-2]),
                                                                          np.arange(0, image_matrix.shape[-1]),
                                                                          image_matrix))
            self.image_channels = 1


    def ev(self, xi, yi, dx=0, dy=0):
        evaluated_matrix = np.empty(self.image_matrix.shape, dtype=float)
        for i in range(self.image_channels):
            print(xi.shape, yi.shape)
            print(self.spline_list[i].ev(xi, yi, dx, dy).shape)

            evaluated_matrix[i, :, :] = self.spline_list[i].ev(xi, yi, dx, dy).reshape(self.image_matrix.shape[-2:])
        return evaluated_matrix

