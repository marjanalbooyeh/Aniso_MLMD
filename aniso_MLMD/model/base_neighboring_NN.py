import torch
import torch.nn as nn
from aniso_MLMD.model.rotation_matrix_ops import (dot_product_NN,
                                                  element_product_NN,
                                                  cross_product_principal_axis,
                                                  relative_orientation_NN,
                                                  RBF_dr_NN,
                                                  rot_matrix_to_angle)
from aniso_MLMD.utils import adjust_periodic_boundary

from .deep_set_layer import DTanh


class BaseNeighborNN(nn.Module):
    """Base class for neural networks trained on pair particles."""

    def __init__(self, in_dim, hidden_dim, n_layers, box_len, N_neighbors,
                 act_fn="ReLU",
                 dropout=0.3, batch_norm=True, device=None, pool='max1'
                 ):
        super(BaseNeighborNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.energy_out_dim = 1
        self.n_layers = n_layers
        self.box_len = box_len
        self.N_neighbors = N_neighbors
        self.act_fn = act_fn
        self.dropout = dropout
        self.device = device
        self.in_dim = in_dim
        self.batch_norm = batch_norm
        self.pool = pool

        self.energy_net = DTanh(d_dim=self.hidden_dim, x_dim=self.in_dim,
                                pool=self.pool)

        # initialize weights and biases
        # self.energy_net.apply(self.init_net_weights)

    def init_net_weights(self, m):
        # todo: add option to initialize uniformly for weights and biases
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def _get_act_fn(self):
        act = getattr(nn, self.act_fn)
        return act()

    def _prep_features_rot_matrix(self, position, orientation_R, neighbor_list):
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

        neighbor_list = neighbor_list.reshape(batch_size, N_particles, -1,
                                              neighbor_list.shape[-1])[:, :, :, 1] # (B, N, N_neighbors)

        dr = (particle_pos[:, None, :] - neighbors_pos)
        dr = adjust_periodic_boundary(dr, self.box_len)
        R = torch.norm(dr, dim=-1, keepdim=True)
        NN_sorted_indices = torch.argsort(R, dim=-2, descending=False)[:,
                            :self.N_neighbors, :]
        sorted_indeces_3D = NN_sorted_indices.broadcast_to(
            (NN_sorted_indices.shape[0], NN_sorted_indices.shape[1], 3))
        sorted_indeces_4D = sorted_indeces_3D.unsqueeze(-1).broadcast_to(
            (sorted_indeces_3D.shape[0], sorted_indeces_3D.shape[1], 3, 3))

        NN_R = torch.gather(R, dim=1, index=NN_sorted_indices)  # (B, N, 1)
        NN_dr = torch.gather(dr, dim=1, index=sorted_indeces_3D)  # (B, N, 3)
        NN_dr = NN_dr / NN_R
        inv_r = 1. / NN_R  # (B, N, 1)

        NN_neighbor_orient_R = torch.gather(neighbors_R, dim=1,
                                            index=sorted_indeces_4D)

        BR_particle_R = particle_R.unsqueeze(1).broadcast_to(
            (batch_size, self.N_neighbors, 3, 3))

        # orientation related features

        orient_dot_prod = dot_product_NN(BR_particle_R,
                                         NN_neighbor_orient_R)  # (B, N, 3, 3)
        orient_element_prod = element_product_NN(BR_particle_R,
                                                 NN_neighbor_orient_R)  # (B, N, 3, 3, 3)
        element_prod_norm = torch.norm(orient_element_prod,
                                       dim=-1)  # (B, N, 3, 3)

        orient_cross_prod = cross_product_principal_axis(BR_particle_R,
                                                         NN_neighbor_orient_R)  # (B, N, 3, 3)
        cross_prod_norm = torch.norm(orient_cross_prod, dim=-1)  # (B, N, 3)
        rel_orient = relative_orientation_NN(BR_particle_R,
                                             NN_neighbor_orient_R)  # (B, N, 3, 3)
        rbf_particle = RBF_dr_NN(NN_dr, BR_particle_R)  # (B, N, 3)
        rbf_neighbor = RBF_dr_NN(NN_dr, NN_neighbor_orient_R)  # (B, N, 3)
        angle = rot_matrix_to_angle(rel_orient)  # (B, N, 3)

        features = torch.cat((NN_R, inv_r, NN_dr,
                              orient_dot_prod.flatten(start_dim=-2),
                              orient_element_prod.flatten(start_dim=-3),
                              element_prod_norm.flatten(start_dim=-2),
                              orient_cross_prod.flatten(start_dim=-2),
                              cross_prod_norm,
                              rel_orient.flatten(start_dim=-2),
                              rbf_particle,
                              rbf_neighbor,
                              angle
                              ),
                             dim=-1)  # (B, N, 80)

        return features.to(self.device)

    def forward(self, position, orientation_R):
        """

        :param particle_pos: Particle's position, shape (batch_size, 3)
        :param neighbors_pos: Neighbors' positions, shape (batch_size, n_neighbors, 3)
        :param particle_R: Particle's orientation, shape (batch_size, 3, 3)
        :param neighbors_R: Neighbors' orientations, shape (batch_size, n_neighbors, 3, 3)

        """

        features = self._prep_features_rot_matrix(position, orientation_R)

        energy = self.energy_net(features)
        return energy
