import torch.nn as nn
import aniso_MLMD.model.base_model_utils as base_model_utils
from aniso_MLMD.model.deep_set_layer import DTanh


class EnergyPredictor_v1(nn.Module):
    def __init__(self, in_dim,
                 neighbor_hidden_dim,
                 particle_hidden_dim,
                 n_layers,
                 box_len,
                 act_fn="ReLU",
                 dropout=0.3,
                 batch_norm=True,
                 device=None,
                 neighbor_pool="mean",
                 particle_pool='max1',
                 prior_energy=True,
                 prior_energy_sigma=1.0, prior_energy_n=12
                 ):
        super(EnergyPredictor_v1, self).__init__()
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

        self.energy_net = DTanh(d_dim=self.particle_hidden_dim,
                                x_dim=self.neighbor_hidden_dim,
                                pool=self.particle_pool,
                                dropout=self.dropout).to(self.device)

    def _neighbors_net(self):
        layers = [nn.Linear(self.in_dim, self.neighbor_hidden_dim),
                  base_model_utils.get_act_fn(self.act_fn)]
        for i in range(self.n_layers - 1):
            layers.append(
                nn.Linear(self.neighbor_hidden_dim, self.neighbor_hidden_dim))
            if self.batch_norm:
                layers.append(nn.BatchNorm1d(self.neighbor_hidden_dim))
            layers.append(base_model_utils.get_act_fn(self.act_fn))
            layers.append(nn.Dropout(p=self.dropout))
        layers.append(
            nn.Linear(self.neighbor_hidden_dim, self.neighbor_hidden_dim))
        return nn.Sequential(*layers)

    def _calculate_prior_energy(self, R):
        U_0 = torch.pow(self.prior_energy_sigma / R, self.prior_energy_n)
        return U_0.sum(dim=2).sum(dim=1)

    def forward(self, position, orientation_R, neighbor_list):
        # position: particle positions (B, N, 3)
        # orientation_R: particle orientation rotation matrix (B, N, 3, 3)
        # neighbor_list: list of neighbors for each particle
        # (B, N * N_neighbors, 2)

        # features: (B, N, N_neighbors, in_dim)
        features, R = base_model_utils.orientation_feature_vector_v2(position,
                                                                     orientation_R,
                                                                     neighbor_list,
                                                                     self.box_len,
                                                                     self.device)
        neighbor_features = self.neighbors_net(
            features)  # (B, N, N_neighbors, neighbor_hidden_dim)
        # pool over the neighbors dimension
        pooled_features = base_model_utils.pool_neighbors(
            neighbor_features,
            self.neighbor_pool)  # (B, N, neighbor_hidden_dim)
        # deep set layer for particle pooling
        energy = self.energy_net(pooled_features)  # (B, 1)
        if self.prior_energy:
            U_0 = self._calculate_prior_energy(R)
            energy = energy + U_0.to(self.device)

        return energy
