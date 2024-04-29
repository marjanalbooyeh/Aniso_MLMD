import torch
import torch.nn as nn

from aniso_per_particle.model import feature_vector, get_act_fn



class ParticleEnergyPredictor(nn.Module):
    def __init__(self, in_dim,
                 out_dim,
                 neighbors_net_config,
                 prior_net_config,
                 energy_net_config,
                 dropout=0.3,
                 batch_norm=True,
                 device=None,
                 ):
        super(ParticleEnergyPredictor, self).__init__()

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
                  get_act_fn(act_fn)]
        for i in range(n_layers - 1):
            layers.append(
                nn.Linear(h_dim, h_dim))
            # if self.batch_norm:
            #     layers.append(nn.BatchNorm1d(self.neighbor_hidden_dim))
            layers.append(get_act_fn(act_fn))
            layers.append(nn.Dropout(p=dropout))
        layers.append(
            nn.Linear(h_dim, out_dim))
        return nn.Sequential(*layers)

    def forward(self, dr, orientation, n_orientation):
        # dr: (B, Nb, 3) # distance vector between particle and its neighbors
        # orientation: (B, 3, 3) # particle's orientation
        # n_orientation: (B, Nb,  3, 3) # neighbors' orientation
        B = dr.shape[0]
        Nb = dr.shape[1]
        ##########################################
        # features: (B, N_neighbors, 15)
        dr = dr.reshape(B, Nb, 3, 1)
        R = torch.norm(dr, dim=2, keepdim=True)# (B, N_neighbors,1, 1)
        dr_norm = dr / R
        orientation = orientation.reshape(B, 1, 3, 3)
        features = feature_vector(dr_norm, orientation, n_orientation).squeeze(-1) # (B, N_neighbors, 15, 1)

        #############Neighbor Net ##############
        neighbor_features = self.neighbors_net(features)  # (B, N, N_neighbors,out_dim)
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
