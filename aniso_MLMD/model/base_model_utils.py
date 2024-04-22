import torch
import torch.nn as nn
import aniso_MLMD.model.neighbor_ops as neighbor_ops




def init_net_weights(m):
    # todo: add option to initialize uniformly for weights and biases
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def get_act_fn(act_fn):
    act = getattr(nn, act_fn)
    return act()


def orientation_feature_vector_v1(position,
                                  orientation_R,
                                  neighbor_list,
                                  box_len,
                                  device):
    """

    Parameters
    ----------
    position: particle positions (B, N, 3)
    orientation_R: particle orientation rotation matrix (B, N, 3, 3)
    neighbor_list: list of neighbors for each particle (B, N * N_neighbors, 2)

    Returns
    -------

    """
    batch_size = position.shape[0]
    N_particles = position.shape[1]

    # change tuple based neighbor list to (B, N, neighbor_idx)
    neighbor_list = neighbor_list.reshape(batch_size, N_particles, -1,
                                          neighbor_list.shape[-1])[:, :, :,
                    1].to(device)  # (B, N, N_neighbors)
    N_neighbors = neighbor_list.shape[-1]
    dr = neighbor_ops.neighbors_distance_vector(position,
                                                neighbor_list)  # (B, N, N_neighbors, 3)
    dr = neighbor_ops.adjust_periodic_boundary(dr, box_len)

    R = torch.norm(dr, dim=-1, keepdim=True)  # (B, N, N_neighbors, 1)

    inv_R = 1. / R  # (B, N, N_neighbors, 1)

    ################ orientation related features ################

    # repeat the neighbors idx to match the shape of orientation_R. This
    # is necessary to gather each particle's neighbors' orientation
    NN_expanded = neighbor_list[:, :, :, None, None].expand(
        (-1, -1, -1, 3, 3))  # (B, N, N_neighbors, 3, 3)
    # repeart the orientation_R to match the shape of neighbor_list
    orientation_R_expanded = orientation_R[:, :, None, :, :].expand(
        (-1, -1, N_neighbors, -1, -1))  # (B, N, N_neighbors, 3, 3)
    # get all neighbors' orientation
    neighbors_orient_R = torch.gather(orientation_R_expanded, dim=1,
                                      index=NN_expanded)  # (B, N, N_neighbors, 3, 3)

    # dot product: (B, N, N_neighbors, 3, 3)
    orient_dot_prod = neighbor_ops.orientation_dot_product(
        orientation_R_expanded,
        neighbors_orient_R)

    # element product: (B, N, N_neighbors, 3, 3, 3)
    orient_element_prod = neighbor_ops.orientation_element_product(
        orientation_R_expanded,
        neighbors_orient_R)
    # element product norm: (B, N, N_neighbors, 3, 3)
    element_prod_norm = torch.norm(orient_element_prod,
                                   dim=-1)

    # principal cross product: (B, N, N_neighbors, 3, 3)
    orient_cross_prod = neighbor_ops.orientation_principal_cross_product(
        orientation_R_expanded,
        neighbors_orient_R)
    # cross product norm: (B, N, N_neighbors, 3)
    cross_prod_norm = torch.norm(orient_cross_prod,
                                 dim=-1)

    # relative orientation: (B, N, N_neighbors, 3, 3)
    rel_orient = neighbor_ops.relative_orientation(orientation_R_expanded,
                                                   neighbors_orient_R)

    ################ RBF features ################

    # RBF for particles:(B, N, N_neighbors, 3)
    rbf_particle = neighbor_ops.RBF_dr_orientation(dr,
                                                   orientation_R_expanded)
    # RBF for neighbors: (B, N, N_neighbors, 3)
    rbf_neighbors = neighbor_ops.RBF_dr_orientation(dr, neighbors_orient_R)

    # euler angle (B, N, N_neighbors, 3)
    angle = neighbor_ops.rot_matrix_to_euler_angle(rel_orient)

    # concatenate all features (B, N, N_neighbors, 80)
    features = torch.cat((R,
                          dr / R,
                          inv_R,
                          orient_dot_prod.flatten(start_dim=-2),
                          orient_element_prod.flatten(start_dim=-3),
                          element_prod_norm.flatten(start_dim=-2),
                          orient_cross_prod.flatten(start_dim=-2),
                          cross_prod_norm,
                          rel_orient.flatten(start_dim=-2),
                          rbf_particle,
                          rbf_neighbors,
                          # angle
                          ),
                         dim=-1)

    return features.to(device), R


def orientation_feature_vector_v2(position,
                                  orientation_R,
                                  neighbor_list,
                                  box_len,
                                  device):
    batch_size = position.shape[0]
    N_particles = position.shape[1]

    # change tuple based neighbor list to (B, N, neighbor_idx)
    neighbor_list = neighbor_list.reshape(batch_size, N_particles, -1,
                                          neighbor_list.shape[-1])[:, :, :,
                    1].to(device)  # (B, N, N_neighbors)
    N_neighbors = neighbor_list.shape[-1]
    dr = neighbor_ops.neighbors_distance_vector(position,
                                                neighbor_list)  # (B, N, N_neighbors, 3)
    dr = neighbor_ops.adjust_periodic_boundary(dr, box_len)

    R = torch.norm(dr, dim=-1, keepdim=True)  # (B, N, N_neighbors, 1)

    ################ particle orientation and relative distance features ################
    dr_orient_dot = neighbor_ops.dr_particle_orientation_dot_product(dr,
                                                                     orientation_R)  # (B, N, N_neighbors, 3)

    ################ neighbors orientation and relative distance features ################
    # repeat the neighbors idx to match the shape of orientation_R. This
    # is necessary to gather each particle's neighbors' orientation
    NN_expanded = neighbor_list[:, :, :, None, None].expand(
        (-1, -1, -1, 3, 3))  # (B, N, N_neighbors, 3, 3)
    # repeart the orientation_R to match the shape of neighbor_list
    orientation_R_expanded = orientation_R[:, :, None, :, :].expand(
        (-1, -1, N_neighbors, -1, -1))  # (B, N, N_neighbors, 3, 3)
    # get all neighbors' orientation
    neighbors_orient_R = torch.gather(orientation_R_expanded, dim=1,
                                      index=NN_expanded)  # (B, N, N_neighbors, 3, 3)

    dr_Nb_orient_dot = neighbor_ops.dr_neighbor_orientation_dot_product(dr,
                                                                        neighbors_orient_R)  # (B, N, N_neighbors, 3)

    ################### Relative orientation features ###################
    # dot product: (B, N, N_neighbors, 3, 3)
    orient_dot_prod = neighbor_ops.orientation_dot_product(
        orientation_R_expanded,
        neighbors_orient_R)

    # concatenate all features (B, N, N_neighbors, 19)
    features = torch.cat((R,
                          dr/R,
                          dr_orient_dot,
                          dr_Nb_orient_dot,
                          orient_dot_prod.flatten(start_dim=-2),
                          ),
                         dim=-1)
    return features.to(device), R


def pool_neighbors(neighbor_features, pool_type):
    # neighbor_features: (B, N, N_neighbors, hidden_dim)
    if pool_type == 'mean':
        return neighbor_features.mean(dim=2)
    elif pool_type == 'max':
        return neighbor_features.max(dim=2)[0]
    elif pool_type == 'sum':
        return neighbor_features.sum(dim=2)
    else:
        raise ValueError('Invalid neighbor pooling method')


def pool_particles(pooled_features, pool_type):
    # pooled_features: (B, N, neighbor_hidden_dim)
    if pool_type == 'mean':
        return pooled_features.mean(dim=1)
    elif pool_type == 'max':
        return pooled_features.max(dim=1)[0]
    elif pool_type == 'sum':
        return pooled_features.sum(dim=1)
    else:
        raise ValueError('Invalid neighbor pooling method')



