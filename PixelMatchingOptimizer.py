import numpy as np
import pandas as pd
import torch
from mpmath import sincpi
from torch import optim
import torch.nn as nn
import rasterio.plot
from torch.nn import AvgPool2d

from LossFunctions.LossFunctions import square_error_around_pixel
from LossFunctions.LossFunctions import square_error_at_pixel
from LossFunctions.LossFunctions import square_error_around_pixel_vectorized_convolution
import scipy

class MovePixelsIsolated(nn.Module):

    def __init__(self, matrix_shape):
        super(MovePixelsIsolated, self).__init__()
        self.matrix_shape = matrix_shape
        self.movements = nn.Parameter(torch.zeros((matrix_shape[-2]*matrix_shape[-1], 2)))



    def forward(self, x):
        x = nn.Flatten(0, -2)(x)
        x = x + self.movements
        x = nn.Unflatten(0, (self.matrix_shape[-2], self.matrix_shape[-1]))(x)
        x.unsqueeze_(0)
        return x


class IndexShiftNN(nn.Module):
    def __init__(self, num_hidden_layers: int = 3, width: int = 32):
        super(IndexShiftNN, self).__init__()
        self.num_hidden_layers = num_hidden_layers
        self.width = width
        self.model = nn.Sequential()

        self.model.add_module('input', nn.Linear(in_features=2, out_features=width))
        # nn.init.eye_(self.model[0].weight)
        # nn.init.constant_(self.model[0].bias, 0),
        self.model.add_module('relu', nn.ReLU())

        for i in range(0, num_hidden_layers):
            self.model.add_module(f'hidden{i + 1}', nn.Linear(width, width))
            # nn.init.eye_(self.model[-1].weight)
            # nn.init.constant_(self.model[-1].bias, 0)
            self.model.add_module(f'relu{i + 1}', nn.ReLU())

        self.model.add_module('output', nn.Linear(in_features=width, out_features=2))
        # nn.init.eye_(self.model[-1].weight)
        # nn.init.constant_(self.model[-1].bias, 0),
        self.model.append(nn.ReLU())



    def forward(self, x):
        # x = 1/2*x+1/2
        # x = nn.Flatten()(x)
        x = self.model(x)
        # x = 2*x-1
        return x




class ShiftingLayer(nn.Module):

    def __init__(self, cell_size, tracked_matrix_shape):
        super().__init__()
        self.padding_length = 10
        assert self.padding_length < cell_size, "Padding length is smaller than cell size. Not yet implemented"
        self.cell_size = cell_size
        self.cell_size_padded = self.cell_size + 2*self.padding_length
        self.matrix_shape_padded = [2*self.padding_length+tracked_matrix_shape[0] + self.cell_size - (tracked_matrix_shape[0] % self.cell_size),
                                    2 * self.padding_length + tracked_matrix_shape[1] + self.cell_size - (tracked_matrix_shape[1] % self.cell_size)]
        # self.number_of_cells_dim0 = np.ceil(tracked_matrix_shape[0] / self.cell_size).astype(int)
        # self.number_of_cells_dim1 = np.ceil(tracked_matrix_shape[1] / self.cell_size).astype(int)
        self.number_of_cells_dim0 = np.ceil((tracked_matrix_shape[0]) / self.cell_size).astype(int)
        self.number_of_cells_dim1 = np.ceil((tracked_matrix_shape[1]) / self.cell_size).astype(int)


        self.transformation_matrices = nn.Parameter(torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]).unsqueeze(0).repeat((self.number_of_cells_dim0*self.number_of_cells_dim1, 1, 1)))

    def forward(self, matrix):
        original_matrix_shape = matrix.shape
        matrix = torch.cat((torch.zeros(self.padding_length, matrix.shape[1]), matrix, torch.zeros(self.cell_size - (matrix.shape[0] % self.cell_size) + self.padding_length, matrix.shape[1])), 0)
        matrix = torch.cat((torch.zeros(matrix.shape[0], self.padding_length), matrix, torch.zeros(matrix.shape[0], self.cell_size - (matrix.shape[1] % self.cell_size) + self.padding_length)), 1)
        #matrix_segmented = matrix.unfold(0, self.cell_size, self.cell_size).unfold(1, self.cell_size, self.cell_size)
        matrix = matrix.unsqueeze(0).unsqueeze(1)

        matrix_segmented = torch.nn.functional.unfold(matrix, kernel_size=self.cell_size_padded, stride=self.cell_size, padding=self.padding_length)
        # matrix_segmented = matrix_segmented.reshape(1, self.cell_size, self.cell_size, self.number_of_cells_dim0*self.number_of_cells_dim1)


        matrix_segmented = matrix_segmented.reshape(1, self.cell_size_padded, self.cell_size_padded, self.number_of_cells_dim0 * self.number_of_cells_dim1)

        matrix_segmented = matrix_segmented.permute(3, 0, 1, 2)

        # moved_indices_tensor = torch.nn.functional.affine_grid(self.transformation_matrices, size=torch.Size([self.number_of_cells_dim0*self.number_of_cells_dim1, 1, self.cell_size_padded, self.cell_size_padded]), align_corners=False)
        moved_indices_tensor = torch.nn.functional.affine_grid(self.transformation_matrices, size=torch.Size([matrix_segmented.shape[0], 1, self.cell_size_padded, self.cell_size_padded]), align_corners=False)
        moved_matrix = torch.nn.functional.grid_sample(matrix_segmented, moved_indices_tensor, mode='bicubic', align_corners=False, padding_mode='zeros')
        moved_matrix = moved_matrix.permute(1, 2, 3, 0)
        moved_matrix = moved_matrix[:, self.padding_length:self.cell_size_padded-self.padding_length, self.padding_length:self.cell_size_padded-self.padding_length, :]
        moved_matrix = moved_matrix.view(1, self.cell_size, self.cell_size, self.number_of_cells_dim0, self.number_of_cells_dim1)
        moved_matrix = moved_matrix.permute(0, 3, 4, 1, 2)
        moved_matrix = moved_matrix.permute(0, 1, 3, 2, 4).reshape(1, self.cell_size*self.number_of_cells_dim0, self.cell_size*self.number_of_cells_dim1)

        # rasterio.plot.show(moved_matrix[0, :, :].detach().numpy())

        #
        # moved_matrix = moved_matrix.reshape(1, self.cell_size*self.cell_size, self.number_of_cells_dim0*self.number_of_cells_dim1)
        # moved_matrix = torch.nn.functional.fold(moved_matrix, output_size=(matrix.shape[2], matrix.shape[3]), kernel_size=self.cell_size, stride=self.cell_size)
        moved_matrix = moved_matrix.squeeze()
        moved_matrix = moved_matrix[0:original_matrix_shape[0], 0:original_matrix_shape[1]]
        return moved_matrix


class ShiftingLayerIndices(nn.Module):
    """Moves indices between start and end values (for rows and columns) via an affine transformation matrix and a shift vector """
    # def __init__(self, start_index_row, end_index_row, start_index_col, end_index_col):
    #     super().__init__()
    #     self.transformation_matrix = nn.Parameter(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
    #     # print(np.array(np.meshgrid(np.arange(start_index_row, end_index_row),
    #     #                         np.arange(start_index_col, end_index_col))).T.reshape(-1, 2).T)
    #     self.indices = torch.tensor(np.array(np.meshgrid(np.arange(start_index_row, end_index_row),
    #                             np.arange(start_index_col, end_index_col))).T.reshape(-1, 2).T, dtype=torch.float)
    #     self.shift_vector = nn.Parameter(torch.tensor([[0.0], [0.0]]).expand(2, self.indices.shape[1]))
    #
    # def forward(self, test):
    #     moved_indices = torch.matmul(self.transformation_matrix, self.indices) + self.shift_vector
    #     return moved_indices.int()


    def __init__(self, cell_size, tracked_matrix_shape):
        super().__init__()
        self.cell_size = cell_size
        self.number_of_cells_dim0 = np.ceil(tracked_matrix_shape[0] / self.cell_size).astype(int)
        self.number_of_cells_dim1 = np.ceil(tracked_matrix_shape[1] / self.cell_size).astype(int)
        # self.transformation_matrices = nn.ParameterList()
        # self.shift_vectors = nn.ParameterList()
        # # initialize transformation matrx and shift vector
        # transformation_matrix = torch.eye(2)
        # shift_vector = torch.zeros(2)
        # # make one NN parameter for each
        # for i in np.arange(0, self.number_of_cells_dim0*self.number_of_cells_dim1):
        #         self.transformation_matrices.append(nn.Parameter(transformation_matrix))
        #         self.shift_vectors.append(nn.Parameter(shift_vector))
        self.transformation_matrices = nn.Parameter(torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]).unsqueeze(0).repeat((self.number_of_cells_dim0*self.number_of_cells_dim1, 1, 1)))

    def forward(self, index_tensor):
        # moved_indices_tensor = torch.zeros(index_tensor.shape, requires_grad=True)
        # moved_indices_tensor = moved_indices_tensor.clone()
        # for i in np.arange(0, self.number_of_cells_dim0):
        #     for j in np.arange(0, self.number_of_cells_dim1):
        #         # extract the cell part that should be moved (index not out of bounds of index tensor
        #
        #         indices_dim0 = slice(torch.tensor(i*self.cell_size).int(), torch.minimum(torch.tensor(i*self.cell_size+self.cell_size), torch.tensor(index_tensor.shape[0])).int())
        #         indices_dim1 = slice(torch.tensor(j*self.cell_size).int(), torch.minimum(torch.tensor(j*self.cell_size+self.cell_size), torch.tensor(index_tensor.shape[1])).int())
        #         subtensor = index_tensor[indices_dim0, indices_dim1, :]
        #         subtensor_vectorized = subtensor.flatten(start_dim=0, end_dim=1)
        #         subtensor_vectorized_moved = torch.matmul(subtensor_vectorized.float(), self.transformation_matrices[int(i*self.number_of_cells_dim0 + j)])+ self.shift_vectors[int(i*self.number_of_cells_dim0 + j)].repeat(subtensor_vectorized.shape[0], 1)
        #
        #         subtensor_moved = subtensor_vectorized_moved.unflatten(
        #             0,
        #             (torch.minimum(torch.tensor(self.cell_size),
        #                            torch.abs(torch.tensor(i*self.cell_size - index_tensor.shape[0]))).int(),
        #                 torch.minimum(torch.tensor(self.cell_size),
        #                               torch.abs(torch.tensor(j*self.cell_size - index_tensor.shape[1]))).int()))
        #
        #         moved_indices_tensor[indices_dim0, indices_dim1, :] = subtensor_moved

        moved_indices_tensor = torch.nn.functional.affine_grid(self.transformation_matrices, size=torch.Size([self.number_of_cells_dim0*self.number_of_cells_dim1, 1, self.cell_size, self.cell_size]))

        print(moved_indices_tensor)

        return moved_indices_tensor




def lsm_loss(moved_matrix, search_matrix):

    return torch.sum((moved_matrix-search_matrix)**2)


def lsm_loss_geometric_difference_constraint(moved_matrix, search_matrix, moved_index_tensor, original_index_tensor):
    lasso_error = torch.linalg.vector_norm(moved_index_tensor - original_index_tensor, ord=1)
    matrix_error = lsm_loss(moved_matrix, search_matrix)
    moved_index_tensor = moved_index_tensor.unsqueeze(-1).expand(-1, -1, -1, -1, 5)
    moved_index_tensor = moved_index_tensor.clone()
    moved_index_tensor[:, 0:-1, :, :, 1] = moved_index_tensor[:, 1:, :, :, 0]
    moved_index_tensor[:, 2:, :, :, 2] = moved_index_tensor[:, 1:-1, :, :, 0]
    moved_index_tensor[:, :, 0:-1, :, 3] = moved_index_tensor[:, :, 1:, :, 0]
    moved_index_tensor[:, :, 2:, :, 4] = moved_index_tensor[:, :, 1:-1, :, 0]
    similar_movement_loss = torch.sum(moved_index_tensor[:, :, :, :, 0] - moved_index_tensor[:, :, :, :, 1])**2
    similar_movement_loss += torch.sum(moved_index_tensor[:, :, :, :, 0] - moved_index_tensor[:, :, :, :, 2])**2
    similar_movement_loss += torch.sum(moved_index_tensor[:, :, :, :, 0] - moved_index_tensor[:, :, :, :, 3])**2
    similar_movement_loss += torch.sum(moved_index_tensor[:, :, :, :, 0] - moved_index_tensor[:, :, :, :, 4])**2
    print(lasso_error*1e6, matrix_error, similar_movement_loss*1e2)
    return torch.mul(lasso_error, 1e+9) + matrix_error + torch.mul(similar_movement_loss, 1e+2)


def create_index_tensor(size):
    """First two dimensions are the image height and width. The last dimension is for storing the coordinate tuple at the given position"""
    indices = np.array(np.meshgrid(np.arange(0, size[0]),
                          np.arange(0, size[1]))).T.reshape(size[0], size[1], 2)
    indices = torch.tensor(indices, dtype=torch.float)
    indices = indices.unsqueeze(0)
    print(indices.shape)
    indices = torch.stack(
        (2 / (size[-1]-1) * indices[:, :, :, 1] - 1, 2 / (size[-2]-1) * indices[:, :, :, 0] - 1), dim=3).float()
    return indices


def evaluate_at_indices(matrix, indices, normalized_indices=True):
    """given a mxn matrix and indices shaped mxnx2, returns """
    # indices_vector = indices.flatten(start_dim=0, end_dim=1)
    # indices = torch.subtract(
    #     torch.multiply(indices,
    #                    torch.tensor([2/matrix.shape[0], 2/matrix.shape[1]]).unsqueeze(0).unsqueeze(1).repeat(indices.shape[0], indices.shape[1], 1)),
    #     torch.tensor([1, 1]).unsqueeze(0).unsqueeze(1).repeat(indices.shape[0], indices.shape[1], 1))
    # indices_dim0 = torch.maximum(torch.minimum(indices_vector[:, 0], torch.tensor(matrix.shape[0]-1, dtype=torch.float, requires_grad=True)), torch.tensor(0, dtype=torch.float, requires_grad=True))
    # indices_dim1 = torch.maximum(torch.minimum(indices_vector[:, 1], torch.tensor(matrix.shape[1]-1, dtype=torch.float, requires_grad=True)), torch.tensor(0, dtype=torch.float, requires_grad=True))
    # Not differentiable
    # evaluated_matrix = matrix[torch.round(indices_dim0).int(), torch.round(indices_dim1).int()].unflatten(0, matrix.shape)
    matrix_extended = matrix.unsqueeze(0).unsqueeze(1).float()
    if not normalized_indices:
        indices_extended = torch.stack((2/matrix.shape[-1]*indices[:, :, :, 1]-1, 2/matrix.shape[-2]*indices[:, :, :, 0]-1), dim=3).float()
    else:
        indices_extended = indices

    evaluated_matrix = torch.nn.functional.grid_sample(input=matrix_extended, grid=indices_extended, mode='bilinear')
    return evaluated_matrix


def get_pixel_movement_from_model(model, matrix_size):
    track_matrix = torch.arange(matrix_size[0]*matrix_size[1])
    track_matrix = track_matrix.reshape(matrix_size[0], matrix_size[1])
    # track_matrix[0:250, :] = 0
    # track_matrix[:, 0:1250] = 0
    # track_matrix[400:, :] = 0
    # track_matrix[:, 1400:] = 0
    moved_matrix = model(track_matrix)
    tracked_pixels = pd.DataFrame()
    rasterio.plot.show(track_matrix.detach().numpy())
    rasterio.plot.show(moved_matrix.detach().numpy())
    # print(track_matrix[1250:1400, 2250:2400])
    # print(moved_matrix[1250:1400, 2250:2400])
    for i in range(0, moved_matrix.shape[0], 50):
        if i % 100 == 0:
            print(i)
        for j in range(0, moved_matrix.shape[1], 50):
            value = np.round(moved_matrix[i, j].detach().numpy())
            [origin_row, origin_col] = np.where(track_matrix == value)
            #     round(value / matrix_size[1]))
            # origin_col = round(value % matrix_size[1])
            try:
                tracked_pixels = pd.concat([tracked_pixels, pd.DataFrame({"row": origin_row,
                                                                         "column": origin_col,
                                                                         "movement_row_direction": i-origin_row,
                                                                         "movement_column_direction": j-origin_col},
                                                                         index=[len(tracked_pixels)])])
            except:
                continue
    return tracked_pixels


def move_pixels_global(tracked_matrix, search_matrix):
    blockwise_model = False
    # THE FOLLOWING IS THE APPROACH WHERE THE NEURAL NETWORK ONLY APPLIES ON INDICES
    if not blockwise_model:
        print(tracked_matrix.shape)
        tracked_matrix[np.where(search_matrix == 0)] = 0
        search_matrix[np.where(tracked_matrix == 0)] = 0
        rasterio.plot.show(tracked_matrix, title="Original Image")
        tracked_matrix_torch = torch.tensor(tracked_matrix)
        search_matrix_torch = torch.tensor(search_matrix)
        index_tensor = create_index_tensor(tracked_matrix.shape)
        print(index_tensor.shape)
        # model = IndexShiftNN(num_hidden_layers=3, width=32)

        model = MovePixelsIsolated(matrix_shape=tracked_matrix.shape)

        optimizer = optim.Adam(model.parameters())

        optimizer.zero_grad()

        for epoch in range(25):
            model.train()
            print("Epoch: ", epoch)
            moved_indices_tensor = model(index_tensor)
            moved_matrix = evaluate_at_indices(tracked_matrix_torch, moved_indices_tensor)
            # loss = lsm_loss(moved_matrix, search_matrix_torch)
            loss = lsm_loss_geometric_difference_constraint(moved_matrix, search_matrix_torch, moved_indices_tensor, index_tensor)
            print(loss)
            # moved_matrix[torch.where(torch.tensor(search_matrix_torch == 0))] = 0
            if epoch % 5 == 0:
                rasterio.plot.show(moved_matrix.detach().numpy(), title=("Training epoch", epoch))
            # loss = lsm_loss(moved_matrix, search_matrix_torch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        model.eval()
        moved_indices_tensor = model(index_tensor)



        rasterio.plot.show(tracked_matrix, title="Unmoved_tracked matrix")
        rasterio.plot.show(evaluate_at_indices(tracked_matrix_torch, moved_indices_tensor).detach().numpy(), title="Moved_tracked_matrix")
        rasterio.plot.show(search_matrix, title="Search matrix")

        # moved_indices_tensor[:, :, :, 0] = 1 / 2 * (tracked_matrix.shape[-2]) * moved_indices_tensor[:, :, :,
        #                                                                         0] + 1 / 2 * tracked_matrix.shape[-2]
        # moved_indices_tensor[:, :, :, 1] = 1 / 2 * (tracked_matrix.shape[-1]) * moved_indices_tensor[:, :, :,
        #                                                                         1] + 1 / 2 * tracked_matrix.shape[-1]
        model_parameters = list(model.parameters())[-1]

        model_parameters = nn.Unflatten(0, (tracked_matrix.shape[-2], tracked_matrix.shape[-1]))(model_parameters)
        model_parameters = model_parameters.detach().numpy()

        model_parameters[:, :, 0] = 1 / 2 * (tracked_matrix.shape[-2]) * model_parameters[:, :, 0] + 1 / 2 * tracked_matrix.shape[-2]
        model_parameters[:, :, 1] = 1 / 2 * (tracked_matrix.shape[-1]) * model_parameters[:, :, 1] + 1 / 2 * tracked_matrix.shape[-1]

        rasterio.plot.show(model_parameters[:, :, 0], title="Row_movement")
        rasterio.plot.show(model_parameters[:, :, 1], title="Col_movement")
        print(np.sum(model_parameters[:, :, 0] == 0))
        print(np.min(model_parameters[:, :, 1]))



    ## THE FOLLOWING IS THE ORIGINAL APPROACH FOR THE BLOCKWISE MODEL
    if blockwise_model:
        index_tensor = create_index_tensor(tracked_matrix.shape)
        # shifted_matrix = evaluate_at_indices(torch.from_numpy(tracked_matrix), index_tensor)
        model = nn.Sequential(
            # ShiftingLayer(1000, tracked_matrix.shape),
            # ShiftingLayer(50, tracked_matrix.shape),

            # ShiftingLayer(13, tracked_matrix.shape),
            ShiftingLayer(40, tracked_matrix.shape),
            # ShiftingLayer(47, tracked_matrix.shape),

        )
        tracked_matrix[np.where(search_matrix == 0)] = 0
        search_matrix[np.where(tracked_matrix == 0)] = 0
        # rasterio.plot.show(evaluate_at_indices(torch.from_numpy(tracked_matrix), moved_indices_tensor).detach().numpy())
        tracked_matrix_torch = torch.tensor(tracked_matrix, dtype=torch.float, requires_grad=True)
        search_matrix_torch = torch.tensor(search_matrix, dtype=torch.float, requires_grad=True)
        # moved_indices_tensor = model(index_tensor)
        moved_matrix = model(tracked_matrix_torch)
        model.train()
        optimizer = optim.Adam(model.parameters(), lr=0.1)

        # rasterio.plot.show(evaluate_at_indices(tracked_matrix_torch, index_tensor).detach().numpy())


        for epoch in range(20):
            print("Epoch: ", epoch)
            optimizer.zero_grad()
            moved_matrix = model(tracked_matrix_torch)

            # loss = lsm_loss(evaluate_at_indices(tracked_matrix_torch, moved_indices_tensor), search_matrix_torch)
            moved_matrix[torch.where(torch.tensor(search_matrix_torch == 0))] = 0
            if epoch % 10 == 0:
                rasterio.plot.show(moved_matrix.detach().numpy())
            loss = lsm_loss(moved_matrix, search_matrix_torch)
            print(loss)
            loss.backward()
            # for param in model.parameters():
            #     print(param.requires_grad)
            optimizer.step()


        model.eval()
        moved_matrix = model(tracked_matrix_torch)

        rasterio.plot.show(tracked_matrix, title="Unmoved_tracked matrix")
        rasterio.plot.show(moved_matrix.detach().numpy(), title="Moved_tracked_matrix")
        rasterio.plot.show(search_matrix, title="Search matrix")

        tracked_pixels = get_pixel_movement_from_model(model, search_matrix.shape)
    return tracked_pixels

