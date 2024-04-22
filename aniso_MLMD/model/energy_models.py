import torch
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


class EnergyPredictor_v2(nn.Module):
    def __init__(self, in_dim,
                 out_dim,
                 neighbors_net_config,
                 prior_net_config,
                 energy_net_config,
                 dropout=0.3,
                 batch_norm=True,
                 device=None,
                 ):
        super(EnergyPredictor_v2, self).__init__()

        self.neighbor_hidden_dim = neighbors_net_config['hidden_dim']
        self.neighbor_pool = neighbors_net_config['pool']
        self.neighbors_n_layers = neighbors_net_config['n_layers']
        self.neighbors_act_fn = neighbors_net_config['act_fn']

        self.prior_hidden_dim = prior_net_config['hidden_dim']
        self.prior_n_layers = prior_net_config['n_layers']
        self.prior_act_fn = prior_net_config['act_fn']

        self.energy_hidden_dim = energy_net_config['hidden_dim']
        self.energy_n_layers = energy_net_config['n_layers']
        self.energy_act_fn = energy_net_config['act_fn']

        self.dropout = dropout
        self.device = device
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.batch_norm = batch_norm

        self.neighbors_net = self._MLP_net(in_dim=self.in_dim,
                                           h_dim=self.neighbor_hidden_dim,
                                           out_dim=self.out_dim,
                                           n_layers=self.neighbors_n_layers,
                                           act_fn=self.neighbors_act_fn,
                                           dropout=self.dropout).to(
            self.device)

        self.prior_net = self._MLP_net(in_dim=self.in_dim,
                                       h_dim=self.prior_hidden_dim,
                                       out_dim=self.out_dim,
                                       n_layers=self.prior_n_layers,
                                       act_fn=self.prior_act_fn,
                                       dropout=self.dropout).to(self.device)

        self.energy_net = self._MLP_net(in_dim=self.out_dim,
                                        h_dim=self.energy_hidden_dim,
                                        out_dim=1,
                                        n_layers=self.energy_n_layers,
                                        act_fn=self.energy_act_fn,
                                        dropout=self.dropout).to(self.device)

        # self.prior_energy_factor_1 =torch.nn.Parameter(torch.rand(1, 3), requires_grad=True)
        # self.prior_energy_factor_2 =torch.nn.Parameter(torch.rand(1, 3), requires_grad=True)

    def _MLP_net(self, in_dim, h_dim, out_dim,
                 n_layers, act_fn, dropout):
        layers = [nn.Linear(in_dim, h_dim),
                  base_model_utils.get_act_fn(act_fn)]
        for i in range(n_layers - 1):
            layers.append(
                nn.Linear(h_dim, h_dim))
            # if self.batch_norm:
            #     layers.append(nn.BatchNorm1d(self.neighbor_hidden_dim))
            layers.append(base_model_utils.get_act_fn(act_fn))
            layers.append(nn.Dropout(p=dropout))
        layers.append(
            nn.Linear(h_dim, out_dim))
        return nn.Sequential(*layers)

    def forward(self, position, orientation_R, neighbor_list, box_size):
        # position: particle positions (B, N, 3)
        # orientation_R: particle orientation rotation matrix (B, N, 3, 3)
        # neighbor_list: list of neighbors for each particle
        # (B, N * N_neighbors, 2)
        # box_size: (B , 1)
        epsilon = 1e-8
        ##################### Neighbors NET #####################
        # features: (B, N, N_neighbors, in_dim-1)
        features, R, dr = base_model_utils.orientation_feature_vector_v2(
            position,
            orientation_R,
            neighbor_list,
            box_size,
            self.device)

        d = torch.cat((R, features), dim=-1)

        neighbor_features = self.neighbors_net(
            d)  # (B, N, N_neighbors,out_dim)
        # pool over the neighbors dimension
        particle_features = base_model_utils.pool_neighbors(
            neighbor_features,
            self.neighbor_pool)  # (B, N, out_dim)

        ##################### Prior Energy NET #####################

        U_0 = self.prior_net(d)  # (B, N, N_neighbors, out_dim)
        pooled_U_0 = base_model_utils.pool_neighbors(
            U_0,
            self.neighbor_pool)  # (B, N, out_dim)
        # pooled_U_0 = (epsilon + (self.prior_energy_factor_1) *
        #        (pooled_U_0[:, :, None, :] ** (epsilon + self.prior_energy_factor_2)))
        # pooled_U_0 = pooled_U_0.squeeze(2)
        ##################### Energy NET #####################
        energy_feature = (particle_features) + (pooled_U_0)
        predicted_energy = self.energy_net(energy_feature)  # (B, N, 1)

        ################## Calculate Force ##################
        neighbors_force = - torch.autograd.grad(predicted_energy.sum(dim=1).sum(),
                                                dr,
                                                create_graph=True)[0].to(
            self.device)  # (B, N, N_neighbors, 3)

        predicted_force = torch.sum(neighbors_force, dim=2)


        ################## Calculate Torque ##################
        torque_grad = - torch.autograd.grad(predicted_energy.sum(dim=1).sum(),
                                            orientation_R,
                                            create_graph=True)[0].to(
            self.device) # (B, N, 3, 3)

        tq_x = torch.cross(torque_grad[:, :, :, 0], orientation_R[:, :, :, 0])
        tq_y = torch.cross(torque_grad[:, :, :, 1], orientation_R[:, :, :, 1])
        tq_z = torch.cross(torque_grad[:, :, :, 2], orientation_R[:, :, :, 2])
        predicted_torque = (tq_x + tq_y + tq_z).to(self.device)

        return predicted_force, predicted_torque, predicted_energy
