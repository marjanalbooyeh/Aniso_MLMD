import torch
import torch.nn as nn

def feature_vector(dr, particle_orientation, neighbors_orientations):
    """
    Calculate the feature vector for particle and its neighbors.

    Parameters
    ----------
    dr: distance vector between particles and their neighbors
     (B, N_neighbors, 3, 1)
    particle_orientation: particle orientation rotation matrix (B, 1, 3, 3)
    neighbors_orientations : neighbors orientation rotation matrix (B, N_neighbors, 3, 3)

    Returns
    -------
    feature_vector (B, N_neighbors, 15, 1)
    """
    v1 = dr_particle_orientation_product(dr, particle_orientation)  # (B, N_neighbors, 3, 1)
    v2 = dr_neighbors_orientation_product(dr, neighbors_orientations)  # (B, N_neighbors, 3, 1)
    v3 = particle_neighbor_orientation_product(particle_orientation, neighbors_orientations)  # (B, N_neighbors, 9, 1)
    return torch.cat((v1, v2, v3), dim=2)

def dr_particle_orientation_product(dr, particle_orientations):
    """
    Calculate the product of the distance vector between a particles and its
    neighbors with the particle orientation.

    Parameters
    ----------
    dr: distance vector between particles and their neighbors
     (B, N_neighbors, 3, 1)
    particle_orientations: particle orientation rotation matrix repeated along
     the neighbor axis (B, 1, 3, 3)

    Returns
    -------
    product (B, N_neighbors, 3, 1)
    """
    Nb = dr.shape[1]
    prod = torch.matmul(torch.transpose(particle_orientations, dim0=-2, dim1=-1),
                            dr).reshape(-1,Nb,3,1)
    return prod

def dr_neighbors_orientation_product(dr, neighbors_orientations):
    """
    Calculate the product of the distance vector between a particles and its
    neighbors with its neighbors orientation.

    Parameters
    ----------
    dr: distance vector between particles and their neighbors
     (B, N_neighbors, 3, 1)
    neighbors_orientations: neighbors orientation rotation matrix
     (B, N_neighbors, 3, 3)

    Returns
    -------
    product (B, N_neighbors, 3, 1)
    """

    Nb = dr.shape[1]
    prod = torch.matmul(torch.transpose(neighbors_orientations, dim0=-2, dim1=-1),
                 dr).reshape(-1,Nb, 3, 1)
    return prod

def particle_neighbor_orientation_product(particle_orientations, neighbors_orientations):
    """
    Calculate the product of the particle orientation columns with
    the neighbors orientations columns.

    Parameters
    ----------
    particle_orientations: particle orientation rotation matrix repeated along
     the neighbor axis (B, 1, 3,  3)
    neighbors_orientations: neighbors orientation rotation matrix
     (B, N_neighbors, 3, 3)

    Returns
    -------
    product (B, N_neighbors, 9, 1)
    """
    Nb = neighbors_orientations.shape[1]
    prod = torch.matmul(torch.transpose(particle_orientations, dim0=-2, dim1=-1),
                          neighbors_orientations).reshape(-1,Nb,9,1)
    return prod

def get_act_fn(act_fn):
    act = getattr(nn, act_fn)
    return act()