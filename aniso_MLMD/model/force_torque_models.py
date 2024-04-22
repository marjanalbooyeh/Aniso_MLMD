import torch.nn as nn
import torch
import aniso_MLMD.model.base_model_utils as base_model_utils


class ForTorPredictor_v1(nn.Module):
    def __init__(self, in_dim, neighbor_hidden_dim,
                 n_layers, box_len,
                 act_fn="ReLU",
                 dropout=0.3, batch_norm=True, device=None,
                 neighbor_pool="mean", prior_force=False, prior_force_sigma=1.0,
                 prior_force_n=12):
        super(ForTorPredictor_v1, self).__init__()
        self.neighbor_hidden_dim = neighbor_hidden_dim
        self.energy_out_dim = 1
        self.n_layers = n_layers
        self.box_len = box_len
        self.act_fn = act_fn
        self.dropout = dropout
        self.device = device
        self.in_dim = in_dim
        self.batch_norm = batch_norm
        self.neighbor_pool = neighbor_pool
        self.prior_force = prior_force
        self.prior_force_sigma = prior_force_sigma
        self.prior_force_n = prior_force_n

        self.neighbors_net = self._neighbors_net().to(self.device)

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
        layers.append(nn.Linear(self.neighbor_hidden_dim, 3))
        return nn.Sequential(*layers)

    def _calculate_prior_force(self, R):
        F_0 = (-1) * (self.prior_force_n / R) * torch.pow(
            self.prior_force_sigma / R, self.prior_force_n)
        return F_0.sum(dim=2)

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
        prediction = base_model_utils.pool_neighbors(
            neighbor_features,
            self.neighbor_pool)  # (B, N, neighbor_hidden_dim)
        if self.prior_force:
            F_0 = self._calculate_prior_force(R)

            prediction = prediction + F_0.to(self.device)

        return prediction
