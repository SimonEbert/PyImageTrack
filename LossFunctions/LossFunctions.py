from PixelMatching import get_submatrix_symmetric
import numpy as np

def square_error_around_pixel(matrix1, matrix2, consideration_area_size=19):
    max_row = matrix1.shape[0]
    max_column = matrix1.shape[1]
    error = 0
    for row in np.arange(0, max_row, int(np.floor(consideration_area_size/2))):
        for column in np.arange(0, max_column, int(np.floor(consideration_area_size/2))):
            submatrix1 = get_submatrix_symmetric([row, column],
                                                 [consideration_area_size, consideration_area_size],
                                                 matrix1)
            submatrix2 = get_submatrix_symmetric([row, column],
                                                 [consideration_area_size, consideration_area_size],
                                                 matrix2)
            central_index_adjustment = matrix2[row, column] - matrix1[row, column]
            submatrix2 -= central_index_adjustment
            error += np.sum(pow(np.subtract(submatrix1, submatrix2), 2))
    return error

