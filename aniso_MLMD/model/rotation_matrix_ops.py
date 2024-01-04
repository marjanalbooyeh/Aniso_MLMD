import torch
import pytorch3d.transforms as p3d

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
    cross_prod = torch.zeros(R1.shape[0], 9, 3)
    cross_prod_norms = torch.zeros(R1.shape[0], 9)
    counter = 0
    for i in range(3):
        for j in range(3):
            cross = torch.cross(R1[:, :, i], R2[:, :, j])
            cross_prod[:, counter, :] = cross
            cross_prod_norms[:, counter] = torch.norm(cross, dim=1)
            counter += 1
    return cross_prod, cross_prod_norms

def relative_orientation(R1, R2):
    # R1: particle 1 orientation rotation matrix (Bx 3 x 3)
    # R2: particle 2 orientation rotation matrix (Bx 3 x 3)
    return torch.matmul(R1, R2.transpose(1, 2))

def rel_pos_orientation(dr, R1, R2):
    rel_pos_alignment = torch.zeros(R1.shape[0], 6)
    rel_pos_project = torch.zeros(R1.shape[0], 6, 3)
    for i in range(3):
        R1_dot_p = torch.einsum('ij, ij -> i', dr, R1[:, :, i])
        R2_dot_p = torch.einsum('ij, ij -> i', dr, R2[:, :, i])
        rel_pos_alignment[:, i] = R1_dot_p
        rel_pos_alignment[:, i + 3] = R2_dot_p
        rel_pos_project[:, i, :] = R1_dot_p.unsqueeze(-1) * R1[:, :, i]
        rel_pos_project[:, i + 3, :] = R2_dot_p.unsqueeze(-1) * R2[:, :, i]

    return rel_pos_alignment, rel_pos_project

def RBF_rel_pos(dr, R1, R2):
    rbf = torch.zeros(R1.shape[0], 6)
    for i in range(3):
        rbf[:, i] = torch.exp(-torch.norm(torch.cross(dr, R1[:, :, i]), dim=1))
        rbf[:, i + 3] = torch.exp(-torch.norm(torch.cross(dr, R2[:, :, i]), dim=1))
    return rbf

def rot_matrix_to_angle(R):
    # R: rotation matrix (Bx 3 x 3)
    # returns: angle in radians (Bx 1)
    return p3d.matrix_to_euler_angles(R, convention="XYZ")