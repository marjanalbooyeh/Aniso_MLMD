import torch
# import pytorch3d.transforms as p3d


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
        rbf[:, i + 3] = torch.exp(
            -torch.norm(torch.cross(dr, R2[:, :, i]), dim=1))
    return rbf


def rot_matrix_to_angle(R):
    # R: rotation matrix (Bx 3 x 3)
    # returns: angle in radians (Bx 1)
    # return p3d.matrix_to_euler_angles(R, convention="XYZ")

    return NotImplementedError

###########################

def dot_product_NN(particle_orient_R, neighbors_orient_R):
    # particle_orient_R: particle orientation rotation matrix, repeated over
    # neighbors dimension (Bx Nx 3 x 3)
    # neighbors_orient_R: neighbors orientation rotation matrix (Bx N x 3 x 3)
    # output: dot product of particle_orient_R columns with each of the N
    # neighbors columns (B, N, 3, 3)
    dot_prod = torch.einsum('ijhk, ijhl -> ijkl', particle_orient_R,
                            neighbors_orient_R)
    return dot_prod


def element_product_NN(particle_orient_R, neighbors_orient_R):
    # particle_orient_R: particle orientation rotation matrix, repeated over
    # neighbors dimension (Bx Nx 3 x 3)
    # neighbors_orient_R: neighbors orientation rotation matrix (Bx N x 3 x 3)
    # output: element-wise product of particle_orient_R columns with each of the N
    # neighbors columns (B, N, 3,  3, 3)
    cross_prod = torch.einsum('ijhk, ijhl -> ijklh', particle_orient_R,
                              neighbors_orient_R)
    return cross_prod


def cross_product_principal_axis(particle_orient_R, neighbors_orient_R):
    # particle_orient_R: particle orientation rotation matrix, repeated over
    # neighbors dimension (Bx Nx 3 x 3)
    # neighbors_orient_R: neighbors orientation rotation matrix (Bx N x 3 x 3)
    # output: cross product of particle_orient_R columns with corresponding
    # column of neighbors_orient_R (B, N, 3, 3)
    cross_prod = torch.cross(torch.transpose(particle_orient_R, -1, -2),
                             torch.transpose(neighbors_orient_R, -1, -2),
                             dim=-1)
    return cross_prod


def relative_orientation_NN(particle_orient_R, neighbors_orient_R):
    # particle_orient_R: particle orientation rotation matrix, repeated over
    # neighbors dimension (Bx Nx 3 x 3)
    # neighbors_orient_R: neighbors orientation rotation matrix (Bx N x 3 x 3)
    # output: relative orientation of particle_orient_R with each of the N
    # neighbors orientations (B, N, 3, 3)

    rel_orient = torch.matmul(particle_orient_R,
                              torch.transpose(neighbors_orient_R, -1, -2))
    return rel_orient


def RBF_dr_NN(dr, particle_orient_R):
    # dr: relative position vector between particle and neighbors (Bx Nx 3)
    # particle_orient_R: could be particle orientation rotation matrix,
    # repeated over neighbors dimension or the neighbors
    # orientation rotation matrix (Bx N x 3 x 3)
    # output: RBF of dr with each of the columns in  neighbors orientations
    # (B, N, 3)

    B = dr.shape[0]
    N = dr.shape[1]

    dr_broadcast = dr.unsqueeze(2).broadcast_to((B, N, 3, 3))
    cross = torch.cross(dr_broadcast,
                        torch.transpose(particle_orient_R, -1, -2), dim=-1)
    norm_sq = torch.pow(torch.norm(cross, dim=-1), 2)
    rbf = torch.exp(-norm_sq)

    return rbf
