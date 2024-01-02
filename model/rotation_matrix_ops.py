import torch


def principal_axis_dot_product(R1, R2):
    # R1: particle 1 orientation rotation matrix (Bx 3 x 3)
    # R2: particle 2 orientation rotation matrix (Bx 3 x 3)

    principal_dot_prod = torch.zeros((R1.shape[0], 3))
    for i in range(3):
        principal_dot_prod[:, i] = torch.einsum('ij, ij-> i', R1[:, :, i],
                                                R2[:, :, i])
    return principal_dot_prod


def cross_axis_dot_product(R1, R2):
    # R1: particle 1 orientation rotation matrix (Bx 3 x 3)
    # R2: particle 2 orientation rotation matrix (Bx 3 x 3)
    principal_axis_dot_prod = torch.zeros(R1.shape[0], 6)
    counter = 0
    for i in range(3):
        for j in range(3):
            if i != j:
                principal_axis_dot_prod[:, counter] = torch.einsum(
                    'ij, ij -> i', R1[:, :, i],
                    R2[:, :, j])
                counter += 1
    return principal_axis_dot_prod

def dot_product(R1, R2):
    # R1: particle 1 orientation rotation matrix (Bx 3 x 3)
    # R2: particle 2 orientation rotation matrix (Bx 3 x 3)
    dot_product = torch.zeros(R1.shape[0], 9)
    counter = 0
    for i in range(3):
        for j in range(3):
            dot_product[:, counter] = torch.einsum('ij, ij -> i', R1[:, :, i],
                                                R2[:, :, j])
            counter += 1
    return dot_product


def cross_product(R1, R2):
    features = torch.zeros(R1.shape[0], 9, 3)
    counter = 0
    for i in range(3):
        for j in range(3):
            features[:, counter, :] = torch.cross(R1[:, :, i], R2[:, :, j])
            counter += 1
    return features.reshape(R1.shape[0], -1)