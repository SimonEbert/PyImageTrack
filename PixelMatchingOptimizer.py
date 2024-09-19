import numpy as np
import torch
from mpmath import sincpi
from torch import optim
import torch.nn as nn
import rasterio.plot
from LossFunctions.LossFunctions import square_error_around_pixel
from LossFunctions.LossFunctions import square_error_at_pixel

class ShiftingLayer(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        self.input_shape = input_shape
        weights_row = torch.zeros(size=input_shape)
        weights_column = torch.zeros(size=input_shape)
        self.weights_row = nn.Parameter(weights_row)
        self.weights_column = nn.Parameter(weights_column)

    def forward(self, x):
        shifted_matrix = torch.zeros(self.input_shape)
        for row in np.arange(0, self.input_shape[0]):
            print("forwarding row ", row)
            for column in np.arange(0, self.input_shape[1]):
                shifted_matrix[row + self.weights_row[row, column].int(),
                column + self.weights_column[row, column].int()] = x[row, column]
        return shifted_matrix

class ShiftPixelsNN(nn.Module):
    def __init__(self, input_shape):
        super(ShiftPixelsNN, self).__init__()
        self.layers = nn.Sequential(ShiftingLayer(input_shape))

    def forward(self, x):
        return self.layers(x)



class ShiftingLayerVector(nn.Module):
    def __init__(self, input_length, row_length):
        super().__init__()
        self.input_length = input_length
        self.row_length = row_length
        weights_row = torch.zeros(size=[input_length])
        weights_column = torch.zeros(size=[input_length])
        self.weights_row = nn.Parameter(weights_row)
        self.weights_column = nn.Parameter(weights_column)



    def forward(self, x):
        shifted_vector = torch.zeros(self.input_length, device=x.device, dtype = x.dtype)
        indices = (self.weights_row+(self.row_length*self.weights_column)).int()
        shifted_vector[indices] = x
        # for i in np.arange(0, self.input_length):
        #     if i % 10000 == 0:
        #         print("forwarding vector index ", i)
        #     shifted_vector[i+self.weights_row[i].long()+(self.row_length*self.weights_column[i].long())] = x[i]
        return shifted_vector


class ShiftPixelsNNVectorized(nn.Module):
    def __init__(self, input_shape):
        super(ShiftPixelsNNVectorized, self).__init__()
        self.layers = nn.Sequential(nn.Flatten(start_dim=0, end_dim=1),
                                    ShiftingLayerVector(input_shape[0]*input_shape[1], input_shape[0]),
                                    nn.Unflatten(-1, input_shape))

    def forward(self, x):
        return self.layers(x)

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
    matrix1 = np.float64(np.multiply(overlap_matrix, matrix1))
    matrix2 = np.float64(np.multiply(overlap_matrix, matrix2))
    matrix1 = matrix1[~np.all(matrix1 == 0, axis=1)]
    matrix2 = matrix2[~np.all(matrix2 == 0, axis=1)]

    # test_vector = matrix1.flatten()
    # test_vector2 = matrix2.flatten()
    # shifted_vector = np.zeros(len(test_vector))
    # for i in np.arange(0, len(test_vector)):
    #     if i % 100000 == 0:
    #         print("forwarding vector index ", i)
    #     shifted_index = int(i + test_vector2[i] + (15 * test_vector2[i]))
    #     if shifted_index >= len(test_vector):
    #         continue
    #     shifted_vector[shifted_index] = test_vector[i]
    # print(shifted_vector)

    model = ShiftPixelsNNVectorized(matrix1.shape)
    optimizer = optim.Adam(model.parameters())
    input_matrix = torch.tensor(matrix1)

    for epoch in np.arange(0,10):
        print("Epoch: ", epoch)
        model.train()
        optimizer.zero_grad()
        shifted_matrix = model(input_matrix)
        loss = square_error_at_pixel(shifted_matrix,
                                         torch.tensor(matrix2, requires_grad=True))
        loss.backward()
        optimizer.step()
    row_weights = model.layers[1].weights_row.detach().numpy()
    column_weights = model.layers[1].weights_column.detach().numpy()
    row_weights, column_weights = row_weights.reshape(matrix1.shape), column_weights.reshape(matrix1.shape)
    movement_matrix = [row_weights, column_weights]
    return row_weights, column_weights



