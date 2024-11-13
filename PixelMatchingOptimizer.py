import numpy as np
import torch
from mpmath import sincpi
from torch import optim
import torch.nn as nn
import rasterio.plot
from LossFunctions.LossFunctions import square_error_around_pixel
from LossFunctions.LossFunctions import square_error_at_pixel
from LossFunctions.LossFunctions import square_error_around_pixel_vectorized_convolution




# lines from the main file for the NN approach

# row_weights, column_weights = get_pixel_movements_optimizer(trial_area1_matrix, trial_area2_matrix)
# print(row_weights)
# fig, ax = plt.subplots()
# plot1 = rasterio.plot.show(row_weights, title="Row weights", ax=ax)
# im = plot1.get_images()[0]
# fig.colorbar(im, ax=ax)
# plt.show()
# fig, ax = plt.subplots()
# plot2 = rasterio.plot.show(column_weights, title="Column weights", ax=ax)
# im = plot2.get_images()[0]
# fig.colorbar(im, ax=ax)
# plt.show()
# fig, ax = plt.subplots()
# plot3 = rasterio.plot.show(np.sqrt(row_weights**2+column_weights**2), title="Movement", ax=ax)
# im = plot3.get_images()[0]
# fig.colorbar(im, ax=ax)
# plt.show()
# print(datetime.now()-start_time)
#
#

class ShiftingLayer(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        self.input_rows = input_shape[0]
        self.input_cols = input_shape[1]
        weights_row = torch.zeros(size=input_shape)
        weights_column = torch.zeros(size=input_shape)
        self.weights_row = nn.Parameter(weights_row)
        self.weights_column = nn.Parameter(weights_column)

    def forward(self, x):
        row_indices = torch.arange(self.input_rows).unsqueeze(1).repeat(1, self.input_cols)
        col_indices = torch.arange(self.input_cols).unsqueeze(0).repeat(self.input_rows, 1)
        shifted_row_indices = torch.remainder(torch.add(row_indices, -self.weights_row), self.input_rows)
        shifted_col_indices = torch.remainder(torch.add(col_indices, -self.weights_column), self.input_cols)
        shifted_matrix = x.gather(0, shifted_row_indices.to(torch.int64))
        shifted_matrix = shifted_matrix.gather(1, shifted_col_indices.to(torch.int64))
        return shifted_matrix


class ShiftingLayerFlowField(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        downscaled_rows = np.round(input_shape[0] / 3).astype(int)
        downscaled_cols = np.round(input_shape[1] / 3).astype(int)
        self.flow_field = torch.zeros((downscaled_rows, downscaled_cols, 2), dtype=torch.double)
        self.flow_field = nn.Parameter(self.flow_field)

    def forward(self, x):
        unsqueezed_input = x.unsqueeze(0).unsqueeze(1)
        unsqueezed_flow_field = self.flow_field.unsqueeze(0)
        unsqueezed_flow_field = unsqueezed_flow_field.transpose(dim0=0, dim1=3).transpose(dim0=1, dim1=3)
        print(unsqueezed_flow_field.shape)
        flow_field_interpolated = torch.nn.functional.interpolate(unsqueezed_flow_field, scale_factor=3)
        flow_field_interpolated = flow_field_interpolated.transpose(dim0=0, dim1=3).transpose(dim0=0, dim1=2).transpose(dim0=0, dim1=1)
        shifted_matrix = torch.nn.functional.grid_sample(unsqueezed_input, flow_field_interpolated, align_corners=False)
        return shifted_matrix[0, 0, :, :]



class ShiftPixelsNN(nn.Module):
    def __init__(self, input_shape):
        super(ShiftPixelsNN, self).__init__()
        self.layers = nn.Sequential(ShiftingLayerFlowField(input_shape))

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
    np.put(a=overlap_matrix, ind=np.where(overlap_matrix != 0), v=1)
    overlap_matrix = overlap_matrix.reshape(matrix1.shape)
    matrix1 = np.float64(np.multiply(overlap_matrix, matrix1))
    matrix2 = np.float64(np.multiply(overlap_matrix, matrix2))
    matrix1 = matrix1[~np.all(matrix1 == 0, axis=1)]
    matrix2 = matrix2[~np.all(matrix2 == 0, axis=1)]
    indices_matrix1_0 = np.where(matrix1 == 0)
    indices_matrix2_0 = np.where(matrix2 == 0)
    matrix2[indices_matrix1_0] = 0
    matrix1[indices_matrix2_0] = 0
    matrix1, matrix2 = matrix1[:, :-1], matrix2[:, :-1]

    # matrix1 = np.zeros((4,4))
    # for i in np.arange(0,4):
    #     matrix1[i,0] = i+1
    # matrix2 = np.zeros((4,4))
    # for i in np.arange(0,4):
    #     matrix2[i,1] = i+1

    model = ShiftPixelsNN(matrix1.shape)
    optimizer = optim.Adam(model.parameters())
    input_matrix = torch.tensor(matrix1, dtype=torch.float64, requires_grad=True)


    comparison_matrix = torch.tensor(matrix2, requires_grad=True)
    for epoch in np.arange(0, 10):
        print("Epoch: ", epoch)
        model.train()
        optimizer.zero_grad()
        shifted_matrix = model(input_matrix)
        print("Finished forward propagation")
        loss = square_error_around_pixel_vectorized_convolution(shifted_matrix,
                                                    comparison_matrix)
        print("Calculated loss: ", loss)
        print("Finished loss calculation")
        loss.backward()
        print("Finished backward propagation")
        optimizer.step()
    rasterio.plot.show(matrix1, title="Matrix 1")
    rasterio.plot.show(model(input_matrix).detach().numpy(), title="Matrix shifted")
    rasterio.plot.show(matrix2)
    row_weights, column_weights = torch.squeeze(model.layers[0].flow_field[:, :, 0]), torch.squeeze(model.layers[0].flow_field[:, :, 1])
    row_weights, column_weights = row_weights.detach().numpy(), column_weights.detach().numpy()
    print("Minimale und maximale Gewichte: ", np.min(row_weights), np.max(row_weights))
    row_weights = 1/2*matrix1.shape[0] * row_weights # + 1/2 * matrix1.shape[0]
    column_weights = 1/2*matrix2.shape[1] * column_weights # + 1/2 * matrix2.shape[1]
    print("Minimale und maximale Gewichte: ", np.min(row_weights), np.max(row_weights))
    return row_weights, column_weights





"""Flattened NN"""


class ShiftingLayerVector(nn.Module):
    def __init__(self, input_length, row_length):
        super().__init__()
        self.input_length = input_length
        self.row_length = row_length
        weights_row = torch.zeros(size=[input_length])
        weights_column = torch.zeros(size=[input_length])
        weights_column[0:input_length] = torch.ones(size=[input_length])
        self.weights_row = nn.Parameter(weights_row)
        self.weights_column = nn.Parameter(weights_column)

    def forward(self, x):
        shifted_vector = torch.zeros(self.input_length+10, device=x.device, dtype=x.dtype)
        indices_shift = (torch.add(self.weights_column, torch.multiply(self.row_length, self.weights_row))).int()
        shifted_vector[torch.add(torch.arange(0, len(x)), indices_shift)] = x
        shifted_vector = shifted_vector[0:len(x)]
        return shifted_vector


class ShiftPixelsNNVectorized(nn.Module):
    def __init__(self, input_shape):
        super(ShiftPixelsNNVectorized, self).__init__()
        self.layers = nn.Sequential(nn.Flatten(start_dim=0, end_dim=1),
                                    ShiftingLayerVector(input_shape[0]*input_shape[1], input_shape[0]-1),
                                    nn.Unflatten(-1, input_shape))

    def forward(self, x):
        return self.layers(x)

