from PixelMatching import get_submatrix_symmetric
import numpy as np
import torch

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
            submatrix2_adjusted = torch.subtract(submatrix2, torch.tensor(central_index_adjustment))
            error = error + torch.sum(torch.pow(torch.subtract(submatrix1, submatrix2_adjusted), 2))
    return error

def square_error_at_pixel(matrix1, matrix2):
    return torch.sum(torch.abs(torch.pow(torch.subtract(matrix1, matrix2), 2)))
