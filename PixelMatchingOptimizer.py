import numpy as np
import torch
import rasterio.plot

def optimize_NN(model):



def build_NN_model(matrix1, matrix2):
    vector1_torch = torch.tensor(matrix1)
    vector2_torch = torch.tensor(matrix2)




def get_pixel_movements_optimizer(matrix1: np.ndarray, matrix2: np.ndarray):
    if matrix1.shape != matrix2.shape:
        print("Matrices have not the same shape. Skipping.")
        return
    overlap_matrix = np.multiply(matrix1, matrix2).flatten()
    np.put(a=overlap_matrix, ind=np.where(overlap_matrix!=0), v=1)
    overlap_matrix = overlap_matrix.reshape(matrix1.shape)
    matrix1 = np.multiply(overlap_matrix, matrix1)
    matrix2 = np.multiply(overlap_matrix, matrix2)
    rasterio.plot.show(matrix1)
    rasterio.plot.show(matrix2)


