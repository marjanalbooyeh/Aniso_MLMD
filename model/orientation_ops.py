
import torch
def rot_matrix_principal_axis_dot_product(R1, R2):

def rot_matrix_principal_axis_dot_product(R1, R2):
    features = []
    for i in range(3):
        for j in range(3):
            features.append(torch.dot(R1[:, i], R2[:, j]))
    torch.tensor(features)

def rot_matrix_cross_product(R1, R2):
    features = []
    for i in range(3):
        for j in range(3):
            features.append(torch.cross(R1[:, i], R2[:, j]))
    torch.tensor(features)