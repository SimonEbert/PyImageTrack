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


def square_error_around_pixel_vectorized_subtract(matrix1, matrix2, pixel_consideration=2):
    """Considers only diagonal pixels at the moment"""
    for i in np.arange(1, pixel_consideration+1):
        matrix1_3d = matrix1.unsqueeze(2).repeat(1, 1, pixel_consideration*4+1)
        matrix2_3d = matrix2.unsqueeze(2).repeat(1, 1, pixel_consideration*4+1)
        matrix1_3d[:, :, (i-1)*4+1] = torch.roll(matrix1_3d[:, :, 0], i, dims=0)
        matrix1_3d[:, :, (i-1)*4+2] = torch.roll(matrix1_3d[:, :, 0], -i, dims=0)
        matrix1_3d[:, :, (i-1)*4+3] = torch.roll(matrix1_3d[:, :, 0], i, dims=1)
        matrix1_3d[:, :, (i-1)*4+4] = torch.roll(matrix1_3d[:, :, 0], -i, dims=1)
        matrix2_3d[:, :, (i-1)*4+1] = torch.roll(matrix2_3d[:, :, 0], i, dims=0)
        matrix2_3d[:, :, (i-1)*4+2] = torch.roll(matrix2_3d[:, :, 0], -i, dims=0)
        matrix2_3d[:, :, (i-1)*4+3] = torch.roll(matrix2_3d[:, :, 0], i, dims=1)
        matrix2_3d[:, :, (i-1)*4+4] = torch.roll(matrix2_3d[:, :, 0], -i, dims=1)



    # error = torch.sum(torch.nn.functional.conv1d(matrix1_3d, matrix2_3d_flipped))
    error = torch.sum(torch.pow(torch.subtract(matrix1_3d, matrix2_3d), 2))
    # error = 1/torch.sum((torch.nn.functional.cosine_similarity(matrix1_3d, matrix2_3d, dim=2)+1))
    return error


def square_error_around_pixel_vectorized_convolution(matrix1, matrix2):
    """Only matching 3x3"""
    matrix1_3d_search = matrix1.unsqueeze(2).repeat(1, 1, 3).unsqueeze(3).repeat(1, 1, 1, 3)
    matrix2_3d_match = matrix2.unsqueeze(2).repeat(1, 1, 3).unsqueeze(3).repeat(1, 1, 1, 3)
    for i in np.arange(0,3):
        for j in np.arange(0,3):
            matrix1_3d_search[0::3, 0::3, i, j] = torch.roll(matrix1_3d_search[0::3, 0::3, i, j], i-1, j-1)
            matrix2_3d_match[0::3, 0::3, i, j] = torch.roll(matrix2_3d_match[0::3, 0::3, i, j], i-1, j-1)
    print("Finished unsqueezing and rolling")
    error_tensor = torch.pow(torch.subtract(matrix1_3d_search, matrix2_3d_match), 2)
    error = error_tensor.sum()
    return error
    # conv_matrix = torch.einsum("ijkl, ijkl->ij", matrix1_3d_search, matrix2_3d_match)

    #    print(torch.nn.functional.conv2d(input=matrix1_3d_search[batch:batch+10000, :, :, :], weight=matrix2_3d_match[batch:batch+10000, :, :, :], stride=1).shape)
    # error = -torch.sum(conv_matrix)
    # return error

def square_error_at_pixel(matrix1, matrix2):
    return torch.sum(torch.pow(torch.subtract(matrix1, matrix2), 2))


def square_error_at_pixel(matrix1, matrix2):
    return torch.sum(torch.pow(torch.subtract(matrix1, matrix2), 2))
