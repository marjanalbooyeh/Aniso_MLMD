import torch
import pytorch3d.transforms as p3d


def neighbors_distance_vector(positions, neighbor_list):
    """
    Calculate the distance vector between particles and their neighbors
    Parameters
    ----------
    positions: particle positions (B, N, 3)
    neighbor_list: list of neighbors for each particle (B, N, N_neighbors)

    Returns
    -------
    dr: distance vector between particles and their neighbors (B, N, N_neighbors, 3)
    """
    B = positions.shape[0]
    N = positions.shape[1]
    N_neighbors = neighbor_list.shape[-1]

    NN_repeated = neighbor_list.unsqueeze(-1).expand((B, N, N_neighbors, 3))
    positions_repeated = positions[:, :, None, :].expand(
        (-1, -1, N_neighbors, -1))
    neighbor_positions = torch.gather(positions_repeated, dim=1,
                                      index=NN_repeated)
    dr = positions_repeated - neighbor_positions  # (B, N, N_neighbors, 3)

    return dr


def adjust_periodic_boundary(dr, box_len):
    half_box_len = box_len / 2
    dr = torch.where(dr > half_box_len, dr - box_len, dr)
    dr = torch.where(dr < -half_box_len, dr + box_len, dr)
    return dr


def orientation_dot_product(particle_orientations, neighbors_orientations):
    """
    Calculate the dot product of the principal axis of the particle orientation
    with the principal axis of the neighbors orientations.

    Parameters
    ----------
    particle_orientations: particle orientation rotation matrix repeated along
     the neighbor axis (B, N, N_neighbors, 3, 3)
    neighbors_orientations: neighbors orientation rotation matrix
     (B, N, N_neighbors, 3, 3)

    Returns
    -------
    dot product (B, N, N_neighbors, 3, 3)

    """
    dot_prod = torch.einsum('ijkhl, ijkhm -> ijklm', particle_orientations,
                            neighbors_orientations)  # (B, N, N_neighbors, 3, 3)

    return dot_prod


def orientation_element_product(particle_orientations, neighbors_orientations):
    """
    Calculate the element wise product of the particle orientation columns with
    the neighbors orientations columns.

    Parameters
    ----------
    particle_orientations: particle orientation rotation matrix repeated along
     the neighbor axis (B, N, N_neighbors, 3, 3)
    neighbors_orientations: neighbors orientation rotation matrix
     (B, N, N_neighbors, 3, 3)

    Returns
    -------
    element wise product (B, N, N_neighbors, 3, 3, 3)

    """
    element_prod = torch.einsum('ijkhl, ijkhm -> ijklmh', particle_orientations,
                                neighbors_orientations)
    return element_prod


def orientation_principal_cross_product(particle_orientations,
                                        neighbors_orientations):
    """
    Calculate the cross product of the principal axis of the particle orientation
    with the principal axis of the neighbors orientations.

    Parameters
    ----------
    particle_orientations: particle orientation rotation matrix repeated along
     the neighbor axis (B, N, N_neighbors, 3, 3)
    neighbors_orientations: neighbors orientation rotation matrix
     (B, N, N_neighbors, 3, 3)

    Returns
    -------
    cross product (B, N, N_neighbors, 3, 3)

    """
    cross_prod = torch.cross(torch.transpose(particle_orientations, -1, -2),
                             torch.transpose(neighbors_orientations, -1, -2),
                             dim=-1)

    return cross_prod


def relative_orientation(particle_orientations, neighbors_orientations):
    """
    Calculate the relative orientation between the particle and its neighbors.

    Parameters
    ----------
    particle_orientations: particle orientation rotation matrix repeated along
     the neighbor axis (B, N, N_neighbors, 3, 3)
    neighbors_orientations: neighbors orientation rotation matrix
     (B, N, N_neighbors, 3, 3)

    Returns
    -------
    relative orientation (B, N, N_neighbors, 3, 3)

    """
    relative_orientation = torch.matmul(particle_orientations,
                                        neighbors_orientations.transpose(-1, -2))
    return relative_orientation


def RBF_dr_orientation(dr, orientations):
    """
    Calculate the RBF of the distance vector and the orientation vector.

    Parameters
    ----------
    dr: distance vector between particles and their neighbors (B, N, N_neighbors, 3)
    orientations: particle orientation rotation matrix repeated along
     the neighbor axis (B, N, N_neighbors, 3, 3)

    Returns
    -------
    RBF (B, N, N_neighbors, 3)

    """
    dr_broadcast = dr.unsqueeze(-1).expand(-1, -1, -1, 3, 3)
    cross = torch.cross(dr_broadcast,
                        torch.transpose(orientations, -1, -2), dim=-1)
    norm_sq = torch.pow(torch.norm(cross, dim=-1), 2)
    rbf = torch.exp(-norm_sq)
    return rbf


def rot_matrix_to_euler_angle(orientations):
    """
    Convert the rotation matrix of the relative orientations to Euler angles in
    radians.


    Parameters
    ----------
    orientations: relative orientation rotation matrix (B, N, N_neighbors, 3, 3)

    Returns
    -------
    Euler angles (B, N, N_neighbors, 3)

    """

    return p3d.matrix_to_euler_angles(orientations, 'XYZ')