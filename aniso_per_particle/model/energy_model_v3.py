import torch
import torch.nn as nn
from aniso_per_particle.model import get_act_fn
class EnergyPredictor_V3_Huang(nn.Module):
    def __init__(self, in_dim,
                 prior_net_config,
                 energy_net_config,
                 dropout=0.3,
                 batch_norm=False,
                 device=None,
                 initial_weights=None,
                 gain=1.5,
                 ):
        super(EnergyPredictor_V3_Huang, self).__init__()

        self.prior_hidden_dim = prior_net_config['hidden_dim']
        self.prior_n_layers = prior_net_config['n_layers']
        self.prior_act_fn = prior_net_config['act_fn']

        self.energy_hidden_dim = energy_net_config['hidden_dim']
        self.energy_n_layers = energy_net_config['n_layers']
        self.energy_act_fn = energy_net_config['act_fn']


        self.dropout = dropout
        self.device = device
        self.in_dim = in_dim
        self.batch_norm = batch_norm
        self.gain = gain

        self.energy_net = self._MLP_net(in_dim=240,
                                        h_dim=self.energy_hidden_dim,
                                        out_dim=1,
                                        n_layers=self.energy_n_layers,
                                        act_fn=self.energy_act_fn,).to(self.device)
        self.prior_net = self._MLP_net(in_dim=self.in_dim,
                                       h_dim=self.prior_hidden_dim,
                                       out_dim=1,
                                       n_layers=self.prior_n_layers,
                                       act_fn=self.prior_act_fn).to(self.device)
        self.prior_energy_factor_1 = torch.nn.Parameter(torch.rand(3, 1), requires_grad=True)
        self.prior_energy_factor_2 = torch.nn.Parameter(torch.rand(3, 1), requires_grad=True)
        nn.init.uniform_(self.prior_energy_factor_1)
        nn.init.uniform_(self.prior_energy_factor_2)

        if initial_weights:
            self.energy_net.apply(self.weights_init)
            self.prior_net.apply(self.weights_init)
            nn.init.uniform_(self.prior_energy_factor_1)
            nn.init.uniform_(self.prior_energy_factor_2)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.uniform_(m.weight.data)
            nn.init.uniform_(m.bias.data)

    def _MLP_net(self, in_dim, h_dim, out_dim,
                 n_layers, act_fn):

        layers = [nn.Linear(in_dim, h_dim[0]),
                  get_act_fn(act_fn)]
        for i in range(n_layers):
            layers.append(
                nn.Linear(h_dim[i], h_dim[i + 1]))
            # if self.batch_norm:
            #     layers.append(nn.BatchNorm1d(bn_dim))
            layers.append(get_act_fn(act_fn))
            layers.append(nn.Dropout(p=self.dropout))
        layers.append(
            nn.Linear(h_dim[-1], out_dim))
        return nn.Sequential(*layers)

    def forward(self, dr, orientation, n_orientation):
        # dr: (B, Nb, 3) # distance vector between particle and its neighbors
        # orientation: (B, 3, 3) # particle's orientation
        # n_orientation: (B, Nb,  3, 3) # neighbors' orientation
        B = dr.shape[0]
        Nb = dr.shape[1]
        epsilon = 1e-8
        R_cut = 4.8
        eta = torch.tensor([[2.], [1.]]).to(self.device)
        eta_size = eta.shape[0]
        eta = eta.reshape(-1, 1, 1, 1, 1, 1, eta_size)
        zeta = torch.tensor([[2.], [4.0], [8.0], [16.0], [32.], [64.]]).to(self.device)
        R_s = torch.linspace(1, 4.8, 10).to(self.device)

        ##########################################
        # features: (B, N_neighbors, 15)
        dr = dr + epsilon
        R = torch.norm(dr, dim=-1, keepdim=True)  # (B, N_neighbors,1)
        dr_norm = dr / R

        new_Nb = 20
        closest_idx = torch.argsort(R, dim=1)[:, :new_Nb, :]
        new_R = R[:, closest_idx, :].reshape(B, new_Nb, 1)
        new_dr_norm = dr_norm[:, closest_idx, :].reshape(B, new_Nb, 3)
        new_n_orientation = n_orientation[:, closest_idx, :, :].reshape(B, new_Nb, 3, 3)
        ################## Prep features ##################
        dr_orient_dot = torch.einsum('ijk, ikh -> ijh', new_dr_norm,
                                     orientation).to(
            self.device)  # (B, N_neighbors, 3)
        dr_n_orient_dot = torch.einsum('ijk, ijkh -> ijh', new_dr_norm,
                                       new_n_orientation).to(
            self.device)  # (B, N_neighbors, 3)
        orient_n_orient_dot = torch.einsum('ihl, ijhm -> ijlm',
                                           orientation,
                                     new_n_orientation).to(
            self.device)  # (B, N_neighbors, 3, 3)

        features = torch.cat(
            [dr_orient_dot, dr_n_orient_dot,
             orient_n_orient_dot.reshape(B, new_Nb, 9)],
            dim=-1)  # (B, N_neighbors, 15)

        fcR = (torch.where(new_R > R_cut,
                           0.0,
                           (0.5 * torch.cos(torch.pi * new_R / R_cut) + 0.5)).
               reshape(B, new_Nb, 1))

        ############# Prior Net ##############
        tanh = nn.Tanh()
        Ecin = (features.squeeze(-1)).reshape(B, new_Nb, 15)
        encoder0 = tanh(self.prior_net(Ecin))
        ECodeout = ((encoder0 ** 2) + epsilon).reshape(B, new_Nb, 1, 1, 1)

        prior_out = new_R.reshape(B, new_Nb, 1, 1, 1) - ECodeout
        prior_out = (self.prior_energy_factor_1 ** 2 + epsilon) * prior_out
        prior_out = prior_out ** (-1 * (self.prior_energy_factor_2 ** 2 + epsilon))
        prior_out = torch.sum(prior_out, dim=3)

        ############# Sym fun ##############
        # fcR = torch.where(R > R_cut,
        #                   0.0,
        #                   (0.5 * torch.cos(torch.pi * R / R_cut) + 0.5)).reshape(B, Nb, 1, 1, 1, 1, 1)
        #
        # R_reduced = (R - R_s).reshape(B, Nb, 1, 1,R_s.shape[0] , 1, 1)
        #
        # pre_factor = torch.exp(-eta * R_reduced ** 2)
        #
        # pre_factor = pre_factor.reshape(B, Nb, 1, 1, R_s.shape[0], 1, eta_size)
        #
        # ang = (features.squeeze(-1) ** 2).reshape(B, Nb, self.in_dim, 1, 1, 1)
        # pos_factor_1 = (2**(1-zeta)*(1-ang)**zeta).reshape(B, Nb, self.in_dim, zeta.shape[0], 1, 1, 1)
        # pos_factor_2 = (2**(1-zeta)*(1+ang)**zeta).reshape(B, Nb, self.in_dim, zeta.shape[0], 1, 1, 1)
        #
        # fca1 = pre_factor * fcR * pos_factor_1
        # fca2 = pre_factor * fcR * pos_factor_2
        # fca = torch.cat([fca1, fca2], dim=5)
        # fcr = torch.sum(fca, dim=2)
        # GAR = fcr.reshape(B, Nb, R_s.shape[0] * 2 * zeta.shape[0] * 1 * eta_size)
        # DES = torch.sum(GAR, dim=1).reshape(-1, R_s.shape[0] * 2 * zeta.shape[0] * 1 * eta_size)
        #
        # ############# Energy Net ##############
        # leaky_relu_1 = nn.LeakyReLU(negative_slope=1.0)
        # leaky_relu_2 = nn.LeakyReLU(negative_slope=1.0)
        # sym_encode = leaky_relu_1(self.energy_net(DES))
        # predicted_energy = leaky_relu_2(sym_encode
        #                                 + torch.sum(prior_out.reshape(B, Nb, 1), dim=1))
        predicted_energy = torch.sum(prior_out.reshape(B, new_Nb, 1), dim=1) *fcR

        ################## Calculate Force ##################
        neighbors_force = torch.autograd.grad(predicted_energy.sum(),
                                                new_dr_norm,
                                                create_graph=True)[0].to(self.device)  # (B, N, 3)

        predicted_force = torch.sum(neighbors_force, dim=1).reshape(B, 3)


        ################## Calculate Torque ##################
        torque_grad = torch.autograd.grad(predicted_energy.sum(),
                                            orientation,
                                            create_graph=True)[0].to(self.device) #(B, N, 3, 3)

        tq_x = torch.cross(torque_grad[:, :, 0].reshape(-1, 3), orientation[:, :, 0].reshape(-1, 3))
        tq_y = torch.cross(torque_grad[:, :, 1].reshape(-1, 3), orientation[:, :, 1].reshape(-1, 3))
        tq_z = torch.cross(torque_grad[:, :, 2].reshape(-1, 3), orientation[:, :, 2].reshape(-1, 3))
        predicted_torque = (tq_x + tq_y + tq_z).to(self.device)

        return predicted_force, predicted_torque, predicted_energy
