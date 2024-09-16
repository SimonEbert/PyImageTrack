import numpy as np
import torch
from torch import optim
import torch.nn as nn
import rasterio.plot
from LossFunctions.LossFunctions import square_error_around_pixel

class ShiftingLayer(nn.Module):
    def __init__(self, size_in, row_length):
        super().__init__()
        self.size_in = size_in
        weights_row, weights_column = torch.tensor(float(size_in)), torch.tensor(float(size_in))
        self.weights_row = nn.Parameter(weights_row)
        self.weights_column = nn.Parameter(weights_column)
        self.row_length = row_length

    def forward(self, x):
        shifted_vector = np.zeros(len(x))
        for i in np.arange(0, len(x)):
            shifted_vector[i+self.weights_row[i]+(self.row_length*self.weights_column[i])] = x[i]
        return shifted_vector


class ShiftPixelsNN(nn.Module):
    def __init__(self, input_shape):
        super(ShiftPixelsNN, self).__init__()
        self.size = input_shape[0]*input_shape[1]
        self.model = nn.Sequential(nn.Flatten(),
                                   ShiftingLayer(self.size, input_shape[0]),
                                   nn.Unflatten(self.size, input_shape))

    def forward(self, x):
        return self.model(x)


def shift_matrix_by(matrix, movement_matrix):
    max_rows = matrix.shape[0]
    max_columns = matrix.shape[1]
    shifted_matrix = np.zeros(matrix.shape)
    for row in np.arange(0, max_rows):
        for column in np.arange(0, max_columns):
            shifted_matrix[row + movement_matrix[row, column, 0],
                           column + movement_matrix[row, column, 1]] = matrix[row, column]
    return shifted_matrix

def get_pixel_movements_optimizer(matrix1: np.ndarray, matrix2: np.ndarray):
    if matrix1.shape != matrix2.shape:
        print("Matrices have not the same shape. Skipping.")
        return
    overlap_matrix = np.multiply(matrix1, matrix2).flatten()
    np.put(a=overlap_matrix, ind=np.where(overlap_matrix!=0), v=1)
    overlap_matrix = overlap_matrix.reshape(matrix1.shape)
    matrix1 = np.multiply(overlap_matrix, matrix1)
    matrix2 = np.multiply(overlap_matrix, matrix2)
    model = ShiftPixelsNN(matrix1.shape)
    optimizer = optim.Adam(model.parameters())
    loss = square_error_around_pixel(matrix1, matrix2)

    for epoch in np.arange(0,100):
        model.train()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    row_weights = model.layer[1].row_weights
    column_weights = model.layer[1].column_weights
    row_weights, column_weights = row_weights.reshape(matrix1.shape), column_weights.reshape(matrix1.shape)
    movement_matrix = [row_weights, column_weights]
    return movement_matrix



