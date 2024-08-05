import torch
import torch.nn as nn
from aniso_per_particle.model import get_act_fn
class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, act_fn, batch_norm=False, dropout=0.0, bn_dim=None):
        super(ResidualBlock, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.batch_norm = nn.BatchNorm1d(out_dim if bn_dim is None else bn_dim) if batch_norm else None
        self.act_fn = get_act_fn(act_fn)
        self.dropout = nn.Dropout(p=dropout)
        self.same_dim = (in_dim == out_dim)
        if not self.same_dim:
            self.projection = nn.Linear(in_dim, out_dim)
        else:
            self.projection = None

    def forward(self, x):
        identity = x
        out = self.linear(x)
        if self.batch_norm:
            out = self.batch_norm(out)
        out = self.act_fn(out)
        out = self.dropout(out)
        if not self.same_dim:
            identity = self.projection(identity)
        out += identity
        return out
class EnergyPredictor_Residual(nn.Module):
    def __init__(self, in_dim,
                 energy_net_config,
                 dropout=0.3,
                 batch_norm=False,
                 device=None,
                 initial_weights=None,
                 nonlinearity="relu",
                 bn_dim=25
                 ):
        super(EnergyPredictor_Residual, self).__init__()

        self.energy_hidden_dim = energy_net_config['hidden_dim']
        self.energy_n_layers = energy_net_config['n_layers']
        self.energy_act_fn = energy_net_config['act_fn']


        self.dropout = dropout
        self.device = device
        self.in_dim = in_dim
        self.batch_norm = batch_norm
        self.bn_dim = bn_dim
        self.nonlinearity = nonlinearity
        self.initial_weights = initial_weights

        self.energy_net = self._MLP_net(in_dim=self.in_dim,
                                        h_dim=self.energy_hidden_dim,
                                        out_dim=1,
                                        n_layers=self.energy_n_layers,
                                        act_fn=self.energy_act_fn).to(self.device)


        if initial_weights:
            self.energy_net.apply(self.weights_init)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            if self.initial_weights == "xavier_normal":
                nn.init.xavier_normal_(m.weight.data, gain=nn.init.calculate_gain(self.nonlinearity))
            elif self.initial_weights == "xavier_uniform":
                nn.init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain(self.nonlinearity))
            elif self.initial_weights == "kaiming_normal":
                nn.init.kaiming_normal_(m.weight, nonlinearity=self.nonlinearity)
            elif self.initial_weights == "kaiming_uniform":
                nn.init.kaiming_uniform_(m.weight, nonlinearity=self.nonlinearity)
            if m.bias is not None:
                m.bias.data.fill_(0.01)
            # nn.init.xavier_normal_(m.bias.data)

    def _MLP_net(self, in_dim, h_dim, out_dim, n_layers, act_fn):

        layers = [ResidualBlock(in_dim, h_dim[0], act_fn, self.batch_norm, self.dropout, self.bn_dim)]

        for i in range(1, n_layers):
            layers.append(ResidualBlock(h_dim[i - 1], h_dim[i], act_fn, self.batch_norm, self.dropout, self.bn_dim))

        layers.append(nn.Linear(h_dim[-1], out_dim))

        return nn.Sequential(*layers)

    def forward(self, dr, orientation, n_orientation):
        # dr: (B, Nb, 3) # distance vector between particle and its neighbors
        # orientation: (B, 3, 3) # particle's orientation
        # n_orientation: (B, Nb,  3, 3) # neighbors' orientation
        B = dr.shape[0]
        Nb = dr.shape[1]
        epsilon = 1e-8
        R_cut = 4.8
        ##########################################
        # features: (B, N_neighbors, 15)
        dr = dr + epsilon
        R = torch.norm(dr, dim=-1, keepdim=True)  # (B, N_neighbors,1)
        dr_norm = dr / R

        ################## Prep features ##################
        dr_orient_dot = torch.einsum('ijk, ikh -> ijh', dr_norm,
                                     orientation).to(
            self.device)  # (B, N_neighbors, 3)
        dr_n_orient_dot = torch.einsum('ijk, ijkh -> ijh', dr_norm,
                                       n_orientation).to(
            self.device)  # (B, N_neighbors, 3)
        orient_n_orient_dot = torch.einsum('ihl, ijhm -> ijlm',
                                           orientation,
                                     n_orientation).to(
            self.device)  # (B, N_neighbors, 3, 3)

        features = torch.cat(
            [dr_norm, dr_orient_dot, dr_n_orient_dot,
             orient_n_orient_dot.reshape(B, Nb, 9)],
            dim=-1)  # (B, N_neighbors, 18)
        # all_features = (torch.cat([R, 1 / R,
        #                            features], dim=-1).to(
        #     self.device))  # (B, N_neighbors, 20)
        fcR = (torch.where(R > R_cut,
                           0.0,
                           (0.5 * torch.cos(torch.pi * R / R_cut) + 0.5)).
               reshape(B, Nb, 1))

        ############# Energy Net ##############
        energy = self.energy_net(features)  # (B, Nb, 1)
        predicted_energy = torch.sum(energy, dim=1)

        ################## Calculate Force ##################
        neighbors_force = torch.autograd.grad(predicted_energy.sum(),
                                              dr,
                                              create_graph=True)[0].to(
            self.device)
        predicted_force = torch.sum(neighbors_force, dim=1).reshape(B, 3)
        ################## Calculate Torque ##################
        torque_grad = torch.autograd.grad(predicted_energy.sum(),
                                          orientation,
                                          create_graph=True)[0].to(self.device)
        tq_x = torch.cross(torque_grad[:, :, 0].reshape(-1, 3), orientation[:,  :, 0].reshape(-1, 3))
        tq_y = torch.cross(torque_grad[:, :, 1].reshape(-1, 3), orientation[:, :, 1].reshape(-1, 3))
        tq_z = torch.cross(torque_grad[:, :, 2].reshape(-1, 3), orientation[:,  :, 2].reshape(-1, 3))

        predicted_torque = (tq_x + tq_y + tq_z).to(self.device)
        return predicted_force, predicted_torque, predicted_energy
