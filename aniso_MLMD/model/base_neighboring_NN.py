import torch
import torch.nn as nn
import aniso_MLMD.model.neighbor_ops as neighbor_ops

from aniso_MLMD.model.deep_set_layer import DTanh


class BaseNeighborNN(nn.Module):
    """Base class for neural networks trained on pair particles."""

    def __init__(self, in_dim, neighbor_hidden_dim, particle_hidden_dim,
                 n_layers, box_len,
                 act_fn="ReLU",
                 dropout=0.3, batch_norm=True, device=None,
                 neighbor_pool="mean", particle_pool='max1',
                 prior_energy=True,
                 prior_energy_sigma=1.0, prior_energy_n=12
                 ):
        super(BaseNeighborNN, self).__init__()
        self.neighbor_hidden_dim = neighbor_hidden_dim
        self.particle_hidden_dim = particle_hidden_dim
        self.energy_out_dim = 1
        self.n_layers = n_layers
        self.box_len = box_len
        self.act_fn = act_fn
        self.dropout = dropout
        self.device = device
        self.in_dim = in_dim
        self.batch_norm = batch_norm
        self.particle_pool = particle_pool
        self.neighbor_pool = neighbor_pool
        self.prior_energy = prior_energy
        self.prior_energy_sigma = prior_energy_sigma
        self.prior_energy_n = prior_energy_n

        self.neighbors_net = self._neighbors_net().to(self.device)

        self.energy_net = DTanh(d_dim=self.particle_hidden_dim, x_dim=self.neighbor_hidden_dim,
                                pool=self.particle_pool).to(self.device)

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

        # change tuple based neighbor list to (B, N, neighbor_idx)
        neighbor_list = neighbor_list.reshape(batch_size, N_particles, -1,
                                              neighbor_list.shape[-1])[:, :, :,
                        1]  # (B, N, N_neighbors)
        N_neighbors = neighbor_list.shape[-1]
        dr = neighbor_ops.neighbors_distance_vector(position,
                                                    neighbor_list)  # (B, N, N_neighbors, 3)
        dr = neighbor_ops.adjust_periodic_boundary(dr, self.box_len)

        self.R = torch.norm(dr, dim=-1, keepdim=True)  # (B, N, N_neighbors, 1)

        inv_R = 1. / self.R  # (B, N, N_neighbors, 1)

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
        features = torch.cat((self.R,
                              dr / self.R,
                              inv_R,
                              orient_dot_prod.flatten(start_dim=-2),
                              orient_element_prod.flatten(start_dim=-3),
                              element_prod_norm.flatten(start_dim=-2),
                              orient_cross_prod.flatten(start_dim=-2),
                              cross_prod_norm,
                              rel_orient.flatten(start_dim=-2),
                              rbf_particle,
                              rbf_neighbors,
                              angle
                              ),
                             dim=-1)

        return features.to(self.device)

    def _neighbors_net(self):
        layers = [nn.Linear(self.in_dim, self.neighbor_hidden_dim),
                  self._get_act_fn()]
        for i in range(self.n_layers - 1):
            layers.append(
                nn.Linear(self.neighbor_hidden_dim, self.neighbor_hidden_dim))
            if self.batch_norm:
                layers.append(nn.BatchNorm1d(self.neighbor_hidden_dim))
            layers.append(self._get_act_fn())
            layers.append(nn.Dropout(p=self.dropout))
        layers.append(nn.Linear(self.neighbor_hidden_dim, self.neighbor_hidden_dim))
        return nn.Sequential(*layers)

    def _pool_neighbors(self, neighbor_features):
        # neighbor_features: (B, N, N_neighbors, hidden_dim)
        if self.neighbor_pool == 'mean':
            return neighbor_features.mean(dim=2)
        elif self.neighbor_pool == 'max':
            return neighbor_features.max(dim=2)[0]
        elif self.neighbor_pool == 'sum':
            return neighbor_features.sum(dim=2)
        else:
            raise ValueError('Invalid neighbor pooling method')

    def _calculate_prior_energy(self):
        U_0 = torch.pow(self.prior_energy_sigma / self.R, self.prior_energy_n)
        return U_0.sum(dim=2).sum(dim=1)

    def forward(self, position, orientation_R, neighbor_list):
        # position: particle positions (B, N, 3)
        # orientation_R: particle orientation rotation matrix (B, N, 3, 3)
        # neighbor_list: list of neighbors for each particle
        # (B, N * N_neighbors, 2)

        # features: (B, N, N_neighbors, 80)
        features = self._prep_features_rot_matrix(position, orientation_R,
                                                  neighbor_list)
        neighbor_features = self.neighbors_net(
            features)  # (B, N, N_neighbors, neighbor_hidden_dim)
        # pool over the neighbors dimension
        pooled_features = self._pool_neighbors(
            neighbor_features)  # (B, N, neighbor_hidden_dim)
        # deep set layer for particle pooling
        energy = self.energy_net(pooled_features)  # (B, 1)
        if self.prior_energy:
            U_0 = self._calculate_prior_energy()
            energy = energy + U_0.to(self.device)

        return energy
